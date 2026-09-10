"""Chain-BC dataset builder for exp09 W4a Phase B.

Reads the W4a-Phase-A `dataset_v2` JSONL shards (schema `exp09-chain-bc/v2`,
an internal dataset path, 262,259 (state, action) samples) and builds a flat,
pre-encoded numpy cache for supervised behavior cloning:

  X.npy               float32 [N, obs_size]  -- v4 one-turn-context state encoding
  y.npy               int64   [N]            -- ACTION_CATALOG index of the human action
  game_id.npy         <U64    [N]             -- source game id (cluster-split key)
  player_skill.npy    float32 [N]             -- per-user max NewRank; RANK_SENTINEL
                                                  (1500.0) where the player never had one
  has_rank.npy        bool    [N]             -- True iff player_skill is a REAL rank
                                                  (False = the 1500.0 sentinel was used)
  action_kind.npy     <U32    [N]             -- compiler's action_kind string (reporting)
  synthetic.npy       bool    [N]             -- PLAN decision 3/4 synthetic flag
  choice_declined.npy bool    [N]             -- turn_meta.choice_declined (PLAN decision 6)
  retention.npy       <U8     [N]             -- "strict" | "tolerant" (bonus slice key)
  turn.npy            int32   [N]             -- 1-based turn number (bonus slice key)
  manifest.json                              -- build params + coverage/NaN checks

Every sample's `action` dict is mapped to its `ACTION_CATALOG` index via the
EXACT key format `sap_ppo.train.env` uses to build `ACTION_INDEX_BY_KEY`
(imported directly from there, never reimplemented, so the two can never
silently drift).

Per PLAN.md Rev 4 / STATUS.md corrected framing: dataset_v2 is the
W2-replay-player-VALIDATED (state, action) stream -- this loader does not
judge or filter samples by the one-turn-agent's `api.legal_actions`; every
retained (ok_strict/ok) turn's ops are legitimate BC labels, reorder included.
Training is UNMASKED end to end; masking belongs only at eval decode time
(a different module), never here.

Usage (PYTHONPATH=python, from the checkout root, venv python):

  # Step 1 -- coverage check only (no state encoding; run this FIRST):
  python -m sap_ppo.train.chain_bc_dataset --check-coverage

  # Full cache build (parallel across the 14 source shards):
  python -m sap_ppo.train.chain_bc_dataset --out an internal dataset path --procs 14

  # Smoke slice (exact record count, cut only at a game boundary -- a game's
  # rows are always contiguous within one shard since serialize.py stripes
  # whole games per worker):
  python -m sap_ppo.train.chain_bc_dataset --out an internal dataset path \\
      --max-samples 25000 --procs 2
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .env import ACTION_INDEX_BY_KEY, _action_key
from .observation import OBSERVATION_MODE_V4, build_state_encoder, encoder_observation_spec

DEFAULT_DATASET_DIR = Path("data/prepared/bc")
DEFAULT_CACHE_DIR = Path("data/prepared/bc-cache")
EXPECTED_SCHEMA_VERSION = "exp09-chain-bc/v2"
CACHE_FORMAT_VERSION = "exp09-w4a-bc-cache/v1"

# Normalization divisor only (matches train_bc_warmstart.py / eval_tempo_planner.py
# convention) -- turns beyond this simply clamp to 1.0 in the encoder; never
# dropped, never an error.
DEFAULT_MAX_TURN = 15

# Phase-0 decision (Ruihan, 2026-07-13): players who never have a NewRank on any
# cached game get this sentinel skill value (has_rank=False marks it as such).
RANK_SENTINEL = 1500.0

DEFAULT_VAL_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 42

WEIGHT_MODE_RANK = "rank"
WEIGHT_MODE_FLAT = "flat"
WEIGHT_MODE_CHOICES = (WEIGHT_MODE_RANK, WEIGHT_MODE_FLAT)
RANK_WEIGHT_MIN_P = 0.05
RANK_WEIGHT_MAX_P = 0.95
RANK_WEIGHT_BASE = 0.5


# ---------------------------------------------------------------------------
# Shard discovery + JSONL streaming
# ---------------------------------------------------------------------------

def _read_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest_not_found:{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = str(manifest.get("schema_version") or "")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        raise ValueError(
            f"unexpected_dataset_schema_version:{schema_version}:expected={EXPECTED_SCHEMA_VERSION}"
        )
    return manifest


def discover_shards(dataset_dir: Path) -> list[Path]:
    manifest = _read_manifest(dataset_dir)
    names = manifest.get("shard_files")
    if not isinstance(names, list) or not names:
        raise ValueError(f"dataset_manifest_missing_shard_files:{dataset_dir}")
    shard_paths = [dataset_dir / str(name) for name in names]
    missing = [str(p) for p in shard_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing_dataset_shards:{missing}")
    return shard_paths


def iter_shard_records(shard_path: Path) -> Iterator[dict[str, Any]]:
    with shard_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_records_upto(shard_paths: list[Path], max_samples: int | None) -> Iterator[dict[str, Any]]:
    """Stream records across shards in file order. If `max_samples` is set,
    stop once that many records have been yielded AND the game_id has just
    changed -- i.e. finish the current game fully before stopping, never
    truncating one mid-stream. serialize.py stripes whole games per worker,
    so a game's rows are always contiguous within a single shard file; this
    only needs to watch for the boundary, not reorder anything."""
    if max_samples is None:
        for shard_path in shard_paths:
            yield from iter_shard_records(shard_path)
        return
    count = 0
    pending_stop = False
    last_game: Any = object()  # sentinel that matches no real game_id
    for shard_path in shard_paths:
        for rec in iter_shard_records(shard_path):
            gid = rec.get("game_id")
            if pending_stop and gid != last_game:
                return
            yield rec
            count += 1
            last_game = gid
            if count >= int(max_samples):
                pending_stop = True


def map_action_index(action: Any) -> int | None:
    """Map a sample's engine-level action dict to its ACTION_CATALOG index.

    Uses `sap_ppo.train.env`'s OWN key format (imported, not reimplemented) so
    this can never silently drift from the catalog it is keyed against.
    """
    if not isinstance(action, dict):
        return None
    return ACTION_INDEX_BY_KEY.get(_action_key(action))


# ---------------------------------------------------------------------------
# Step 1: action-index coverage check (report-only, no state encoding)
# ---------------------------------------------------------------------------

def check_action_coverage(dataset_dir: Path, *, max_examples_per_kind: int = 3) -> dict[str, Any]:
    shard_paths = discover_shards(dataset_dir)
    total = 0
    misses = 0
    kind_counts: Counter[str] = Counter()
    miss_by_kind: Counter[str] = Counter()
    examples_by_kind: dict[str, list[dict[str, Any]]] = {}

    t0 = time.time()
    for shard_path in shard_paths:
        for rec in iter_shard_records(shard_path):
            total += 1
            kind = str(rec.get("action_kind") or "?")
            kind_counts[kind] += 1
            idx = map_action_index(rec.get("action"))
            if idx is None:
                misses += 1
                miss_by_kind[kind] += 1
                bucket = examples_by_kind.setdefault(kind, [])
                if len(bucket) < max_examples_per_kind:
                    bucket.append(
                        {
                            "game_id": rec.get("game_id"),
                            "turn": rec.get("turn"),
                            "decoded_i": rec.get("decoded_i"),
                            "action": rec.get("action"),
                        }
                    )
    elapsed = time.time() - t0
    return {
        "dataset_dir": str(dataset_dir),
        "n_shards": len(shard_paths),
        "total_samples": total,
        "hits": total - misses,
        "misses": misses,
        "kind_counts": dict(sorted(kind_counts.items(), key=lambda kv: -kv[1])),
        "miss_by_kind": dict(sorted(miss_by_kind.items(), key=lambda kv: -kv[1])),
        "miss_examples_by_kind": examples_by_kind,
        "elapsed_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# Step 2: parallel encode cache build
# ---------------------------------------------------------------------------

def _encode_records(records: Iterator[dict[str, Any]], encoder: Any) -> dict[str, Any]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    game_ids: list[str] = []
    skills: list[float] = []
    has_ranks: list[bool] = []
    kinds: list[str] = []
    synthetics: list[bool] = []
    declineds: list[bool] = []
    retentions: list[str] = []
    turns: list[int] = []
    miss_examples: list[dict[str, Any]] = []
    n_miss = 0

    for rec in records:
        action = rec.get("action")
        idx = map_action_index(action)
        if idx is None:
            # Defense-in-depth only: step 1's coverage check must already have
            # confirmed 0-miss over the full dataset before this ever runs.
            # Never silently drop into a mismatched X/y row -- count it, keep a
            # few examples, and skip the row entirely (never invent an index).
            n_miss += 1
            if len(miss_examples) < 3:
                miss_examples.append(
                    {
                        "game_id": rec.get("game_id"),
                        "turn": rec.get("turn"),
                        "decoded_i": rec.get("decoded_i"),
                        "action": action,
                    }
                )
            continue
        state = rec.get("state") or {}
        vec = encoder.encode(state)
        xs.append(vec)
        ys.append(int(idx))
        game_ids.append(str(rec.get("game_id") or ""))
        skill_raw = rec.get("player_skill")
        has_rank = skill_raw is not None
        skills.append(float(skill_raw) if has_rank else float(RANK_SENTINEL))
        has_ranks.append(bool(has_rank))
        kinds.append(str(rec.get("action_kind") or ""))
        synthetics.append(bool(rec.get("synthetic")))
        turn_meta = rec.get("turn_meta") or {}
        declineds.append(bool(turn_meta.get("choice_declined")))
        retentions.append(str(rec.get("retention") or ""))
        turns.append(int(rec.get("turn") or 0))

    n = len(ys)
    obs_size = int(encoder.size)
    X = (
        np.asarray(xs, dtype=np.float32).reshape(n, obs_size)
        if n
        else np.zeros((0, obs_size), dtype=np.float32)
    )
    return {
        "n": n,
        "n_miss": n_miss,
        "miss_examples": miss_examples,
        "X": X,
        "y": np.asarray(ys, dtype=np.int64),
        "game_id": np.asarray(game_ids, dtype="<U64"),
        "player_skill": np.asarray(skills, dtype=np.float32),
        "has_rank": np.asarray(has_ranks, dtype=np.bool_),
        "action_kind": np.asarray(kinds, dtype="<U32"),
        "synthetic": np.asarray(synthetics, dtype=np.bool_),
        "choice_declined": np.asarray(declineds, dtype=np.bool_),
        "retention": np.asarray(retentions, dtype="<U8"),
        "turn": np.asarray(turns, dtype=np.int32),
    }


def _encode_shard_worker(args: tuple[Path, str, int]) -> dict[str, Any]:
    shard_path, observation_mode, max_turn = args
    encoder = build_state_encoder(observation_mode=observation_mode, max_turn=int(max_turn))
    result = _encode_records(iter_shard_records(shard_path), encoder)
    result["shard"] = shard_path.name
    return result


_ARRAY_NAMES = (
    "X",
    "y",
    "game_id",
    "player_skill",
    "has_rank",
    "action_kind",
    "synthetic",
    "choice_declined",
    "retention",
    "turn",
)


def build_cache(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_CACHE_DIR,
    observation_mode: str = OBSERVATION_MODE_V4,
    max_turn: int = DEFAULT_MAX_TURN,
    procs: int = 14,
    max_samples: int | None = None,
) -> dict[str, Any]:
    shard_paths = discover_shards(dataset_dir)

    t0 = time.time()
    if max_samples is not None:
        # Smoke slice: exact record count, single process is plenty fast at
        # this scale (a small fraction of the full 262,259-row build).
        encoder = build_state_encoder(observation_mode=observation_mode, max_turn=int(max_turn))
        results = [_encode_records(_iter_records_upto(shard_paths, int(max_samples)), encoder)]
    else:
        n_procs = max(1, min(int(procs), len(shard_paths)))
        work = [(p, observation_mode, int(max_turn)) for p in shard_paths]
        if n_procs == 1:
            results = [_encode_shard_worker(w) for w in work]
        else:
            with mp.get_context("spawn").Pool(n_procs) as pool:
                results = pool.map(_encode_shard_worker, work)
    elapsed = time.time() - t0

    total_miss = sum(r["n_miss"] for r in results)
    if total_miss:
        miss_examples: list[dict[str, Any]] = []
        for r in results:
            miss_examples.extend(r["miss_examples"])
        raise RuntimeError(
            f"action_index_coverage_regression:{total_miss}_misses_during_encode:"
            f"examples={json.dumps(miss_examples[:5])}"
        )

    n_total = sum(r["n"] for r in results)
    encoder_probe = build_state_encoder(observation_mode=observation_mode, max_turn=max_turn)
    obs_size = int(encoder_probe.size)

    arrays: dict[str, np.ndarray] = {}
    dtypes = {
        "X": np.float32,
        "y": np.int64,
        "game_id": "<U64",
        "player_skill": np.float32,
        "has_rank": np.bool_,
        "action_kind": "<U32",
        "synthetic": np.bool_,
        "choice_declined": np.bool_,
        "retention": "<U8",
        "turn": np.int32,
    }
    for name in _ARRAY_NAMES:
        if n_total:
            arrays[name] = np.concatenate([r[name] for r in results], axis=0)
        else:
            shape = (0, obs_size) if name == "X" else (0,)
            arrays[name] = np.zeros(shape, dtype=dtypes[name])

    nan_count = int(np.isnan(arrays["X"]).sum())
    inf_count = int(np.isinf(arrays["X"]).sum())
    if nan_count or inf_count:
        raise RuntimeError(f"encoded_observation_has_nan_or_inf:nan={nan_count}:inf={inf_count}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _ARRAY_NAMES:
        np.save(out_dir / f"{name}.npy", arrays[name])

    encoder_spec = encoder_observation_spec(encoder_probe)
    n_games = int(len(set(arrays["game_id"].tolist())))
    action_kind_counts = Counter(arrays["action_kind"].tolist())
    manifest_out = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source_dataset_dir": str(dataset_dir),
        "source_schema_version": EXPECTED_SCHEMA_VERSION,
        "shard_files": [p.name for p in shard_paths],
        "max_samples_requested": max_samples,
        "n_samples": int(n_total),
        "n_games": n_games,
        **encoder_spec,
        "max_turn": int(max_turn),
        "rank_sentinel": float(RANK_SENTINEL),
        "n_with_rank": int(arrays["has_rank"].sum()),
        "n_without_rank": int((~arrays["has_rank"]).sum()),
        "action_kind_counts": {k: int(v) for k, v in sorted(action_kind_counts.items(), key=lambda kv: -kv[1])},
        "nan_count": nan_count,
        "inf_count": inf_count,
        "build_seconds": round(elapsed, 1),
        "n_procs": (1 if max_samples is not None else max(1, min(int(procs), len(shard_paths)))),
        "fields": {
            "X.npy": f"float32 [N,{obs_size}] -- {observation_mode} state encoding",
            "y.npy": "int64 [N] -- ACTION_CATALOG index of the human action",
            "game_id.npy": "<U64 [N] -- source game id (cluster-split key)",
            "player_skill.npy": f"float32 [N] -- per-user max NewRank; {RANK_SENTINEL} sentinel if never ranked",
            "has_rank.npy": "bool [N] -- True iff player_skill is a real rank, not the sentinel",
            "action_kind.npy": "<U32 [N] -- compiler action_kind string (reporting/slicing only)",
            "synthetic.npy": "bool [N] -- PLAN decision 3/4 synthetic flag",
            "choice_declined.npy": "bool [N] -- turn_meta.choice_declined (PLAN decision 6)",
            "retention.npy": "<U8 [N] -- 'strict' | 'tolerant'",
            "turn.npy": "int32 [N] -- 1-based turn number",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_out, indent=1), encoding="utf-8")
    return manifest_out


# ---------------------------------------------------------------------------
# Cache loader (consumed by train_chain_bc.py)
# ---------------------------------------------------------------------------

@dataclass
class ChainBcCache:
    X: np.ndarray
    y: np.ndarray
    game_id: np.ndarray
    player_skill: np.ndarray
    has_rank: np.ndarray
    action_kind: np.ndarray
    synthetic: np.ndarray
    choice_declined: np.ndarray
    retention: np.ndarray
    turn: np.ndarray
    manifest: dict[str, Any]


def load_cache(cache_dir: Path) -> ChainBcCache:
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    def _load(name: str) -> np.ndarray:
        return np.load(cache_dir / f"{name}.npy", allow_pickle=False)

    return ChainBcCache(
        X=_load("X"),
        y=_load("y"),
        game_id=_load("game_id"),
        player_skill=_load("player_skill"),
        has_rank=_load("has_rank"),
        action_kind=_load("action_kind"),
        synthetic=_load("synthetic"),
        choice_declined=_load("choice_declined"),
        retention=_load("retention"),
        turn=_load("turn"),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Step 2b (exp09 W2 distillation): teacher-row dataset loading.
#
# `tools/gen_distill_dataset.py` emits rows in this SAME dataset_v2 sample
# shape (schema_version left at EXPECTED_SCHEMA_VERSION -- "REUSE the
# dataset_v2 sample schema... with minimal glue", PLAN.md W2 work item 1),
# plus five new fields layered on top: `source` ("teacher_v1"),
# `teacher_margin`, `teacher_score`, `myopic_score`, `candidates_n` (see that
# module's docstring for exactly what each means). This loader is a
# deliberately SEPARATE, additive code path from `_encode_records`/
# `build_cache`/`ChainBcCache` above -- it does not touch any of them (zero
# risk to the human BC pipeline those already serve) -- because a one-off
# distillation training run has no need for `build_cache`'s persisted,
# shard-parallel numpy cache (built once, reused across many human-BC
# training runs): `train_chain_bc.py --extra-dataset` calls this directly at
# train start, encoding straight from the raw JSONL row file(s) given (NOT a
# `dataset_dir`-with-manifest.json the way `discover_shards`/`_read_manifest`
# require -- "the teacher rows file(s)", PLAN.md W2 work item 3a).
# Reuses `build_state_encoder`/`map_action_index`/`iter_shard_records`
# UNCHANGED so a teacher row's state encodes through the EXACT SAME v4
# pipeline as every human row (verified directly by
# `gen_distill_dataset.py`'s own encoder-smoke gate: a sample row's
# `encoder.encode(state)` must match `encoder.size` -- 1993 for v4/turn<=15).
# ---------------------------------------------------------------------------

TEACHER_SOURCE = "teacher_v1"
HUMAN_SOURCE = "human"


@dataclass
class ExtraDatasetCache:
    """Encoded teacher-distillation rows -- same array shapes as
    `ChainBcCache`'s core X/y/game_id, plus the four new per-row weighting/
    diagnostic fields `train_chain_bc.py`'s mix-mode logic reads."""

    X: np.ndarray
    y: np.ndarray
    game_id: np.ndarray
    source: np.ndarray  # <U16, e.g. "teacher_v1"
    teacher_margin: np.ndarray  # float32 -- mix-mode weighting input
    teacher_score: np.ndarray  # float32 -- diagnostic only
    myopic_score: np.ndarray  # float32 -- diagnostic only
    candidates_n: np.ndarray  # int32 -- diagnostic only
    n_miss: int
    miss_examples: list[dict[str, Any]]
    paths: list[str]


def encode_extra_dataset_rows(
    paths: list[str | Path],
    *,
    observation_mode: str = OBSERVATION_MODE_V4,
    max_turn: int = DEFAULT_MAX_TURN,
) -> ExtraDatasetCache:
    """Encode one or more teacher-distillation JSONL row files directly (no
    manifest.json / `discover_shards` required -- see section docstring
    above). `map_action_index` misses are counted (`n_miss`/`miss_examples`)
    rather than silently dropped without a trace, but -- unlike
    `build_cache`'s hard-fail contract for the human dataset -- are NOT
    raised here: `merge_distill_dataset.py`'s gate is the place a nonzero
    miss count must already have been caught (its own catalog-coverage
    check runs over every row before training ever starts); a caller that
    still sees a nonzero `n_miss` here after that gate passed has a real,
    reportable bug, so `train_chain_bc.py` treats a nonzero `n_miss` from
    THIS function as a hard error at train time (belt-and-suspenders, not
    a duplicate gate -- see its own `--extra-dataset` handling).
    """
    path_objs = [Path(p) for p in paths]
    encoder = build_state_encoder(observation_mode=observation_mode, max_turn=int(max_turn))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    game_ids: list[str] = []
    sources: list[str] = []
    margins: list[float] = []
    tscores: list[float] = []
    mscores: list[float] = []
    cands: list[int] = []
    n_miss = 0
    miss_examples: list[dict[str, Any]] = []

    for path in path_objs:
        for rec in iter_shard_records(path):
            idx = map_action_index(rec.get("action"))
            if idx is None:
                n_miss += 1
                if len(miss_examples) < 5:
                    miss_examples.append(
                        {
                            "path": str(path),
                            "game_id": rec.get("game_id"),
                            "turn": rec.get("turn"),
                            "action": rec.get("action"),
                        }
                    )
                continue
            state = rec.get("state") or {}
            xs.append(encoder.encode(state))
            ys.append(int(idx))
            game_ids.append(str(rec.get("game_id") or ""))
            sources.append(str(rec.get("source") or HUMAN_SOURCE))
            margins.append(float(rec.get("teacher_margin") or 0.0))
            teacher_score = rec.get("teacher_score")
            tscores.append(float(teacher_score) if teacher_score is not None else 0.0)
            myopic_score = rec.get("myopic_score")
            mscores.append(float(myopic_score) if myopic_score is not None else 0.0)
            cands.append(int(rec.get("candidates_n") or 0))

    n = len(ys)
    obs_size = int(encoder.size)
    X = np.asarray(xs, dtype=np.float32).reshape(n, obs_size) if n else np.zeros((0, obs_size), dtype=np.float32)
    return ExtraDatasetCache(
        X=X,
        y=np.asarray(ys, dtype=np.int64),
        game_id=np.asarray(game_ids, dtype="<U64"),
        source=np.asarray(sources, dtype="<U16"),
        teacher_margin=np.asarray(margins, dtype=np.float32),
        teacher_score=np.asarray(tscores, dtype=np.float32),
        myopic_score=np.asarray(mscores, dtype=np.float32),
        candidates_n=np.asarray(cands, dtype=np.int32),
        n_miss=n_miss,
        miss_examples=miss_examples,
        paths=[str(p) for p in path_objs],
    )


# ---------------------------------------------------------------------------
# Step 2b: exp05c eval-manifest holdout exclusion (MUST run before the split)
# ---------------------------------------------------------------------------

def exclude_games(cache: ChainBcCache, excluded_game_ids: set[str]) -> tuple[ChainBcCache, dict[str, Any]]:
    """Drop every sample whose `game_id` is in `excluded_game_ids`, returning a
    NEW `ChainBcCache` (the input is never mutated) plus a small report.

    This is how the exp05c eval manifest's games get carved out of the BC
    training population entirely -- they are the EXTERNAL held-out eval set,
    so they must land in neither the train split nor the monitoring-val split
    `split_by_game` produces. Callers MUST call this (when holding out at
    all) BEFORE `split_by_game`, never after -- filtering after the split
    would leave the excluded games' *other* samples from the same game
    scattered across train/val instead of removed altogether, and would not
    change `split_by_game`'s own game-level shuffle/counts.

    Bug context (exp09 W4a, this session): the exp05c manifest keys games by
    `participation_id`; this cache's `game_id` is the raw replay's outer `id`
    field (a DIFFERENT uuid on the same record -- see
    `replay_compile.serialize.FULL_CACHE` / `tools/build_manifest_holdout_gameids.py`
    for the id<->participation_id join). Before this fix nobody excluded the
    manifest's games from BC training at all: all 163 manifest games were
    present in dataset_v2/this cache, and most landed in rank_v1/flat_v1's
    TRAIN split -- those checkpoints are not held out w.r.t. the eval
    manifest. `excluded_game_ids` must be `game_id`-space ids (this cache's
    own join key), not participation_ids.
    """
    game_list = cache.game_id.tolist()
    excluded = {str(g) for g in excluded_game_ids}
    games_present = set(game_list)
    games_excluded_here = games_present & excluded
    keep_mask = np.array([g not in excluded for g in game_list], dtype=bool)
    filtered = ChainBcCache(
        X=cache.X[keep_mask],
        y=cache.y[keep_mask],
        game_id=cache.game_id[keep_mask],
        player_skill=cache.player_skill[keep_mask],
        has_rank=cache.has_rank[keep_mask],
        action_kind=cache.action_kind[keep_mask],
        synthetic=cache.synthetic[keep_mask],
        choice_declined=cache.choice_declined[keep_mask],
        retention=cache.retention[keep_mask],
        turn=cache.turn[keep_mask],
        manifest=cache.manifest,
    )
    report = {
        "excluded_game_ids_requested": len(excluded),
        "excluded_game_ids_found_in_cache": len(games_excluded_here),
        "excluded_game_ids_not_in_cache": len(excluded - games_present),
        "samples_before": int(len(game_list)),
        "samples_after": int(filtered.game_id.shape[0]),
        "samples_excluded": int(len(game_list) - int(filtered.game_id.shape[0])),
        "games_before": int(len(games_present)),
        "games_after": int(len(set(filtered.game_id.tolist()))),
    }
    return filtered, report


# ---------------------------------------------------------------------------
# Step 3a: train/val cluster split BY GAME
# ---------------------------------------------------------------------------

def split_by_game(
    game_id: np.ndarray,
    *,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Cluster split: hold out ~val_fraction of GAMES for validation, never
    splitting one game's samples across train/val. Fixed seed."""
    game_list = game_id.tolist()
    unique_games = sorted(set(game_list))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique_games))
    shuffled = [unique_games[i] for i in order]
    n_val_games = max(1, round(float(val_fraction) * len(shuffled))) if shuffled else 0
    val_games = set(shuffled[:n_val_games])
    val_mask = np.array([g in val_games for g in game_list], dtype=bool)
    train_mask = ~val_mask
    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "n_games_train": len(shuffled) - n_val_games,
        "n_games_val": n_val_games,
        "n_samples_train": int(train_mask.sum()),
        "n_samples_val": int(val_mask.sum()),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
    }


# ---------------------------------------------------------------------------
# Step 3b: sample weight function (rank / flat)
# ---------------------------------------------------------------------------

def _empirical_upper_cdf(sorted_values: np.ndarray, query: np.ndarray, n: int) -> np.ndarray:
    """mean(sorted_values <= query), vectorized; ties share the upper bound
    (side='right' = count of entries <= query, inclusive)."""
    counts_le = np.searchsorted(sorted_values, query, side="right")
    return counts_le.astype(np.float64) / float(max(1, n))


def _rank_weight_components(s: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    n = int(s.shape[0])
    sorted_s = np.sort(s)
    p = np.clip(_empirical_upper_cdf(sorted_s, s, n), RANK_WEIGHT_MIN_P, RANK_WEIGHT_MAX_P)
    w_raw = RANK_WEIGHT_BASE + p
    raw_sum = float(w_raw.sum())
    norm = (float(n) / raw_sum) if raw_sum > 0 else 1.0
    return w_raw * norm, norm, sorted_s


def compute_train_weights(skill_train: np.ndarray, *, mode: str) -> np.ndarray:
    """Per-sample weights over the TRAIN split (Ruihan's exact spec):

    rank: s = player_skill (already 1500.0-sentineled for no-rank players by
      the cache builder). p = empirical upper-CDF percentile of s within the
      train split (ties share the upper bound). Clamp p in [0.05, 0.95].
      w = 0.5 + p. Renormalize so mean(w) == 1.0 exactly.
    flat: w = 1.0 for everyone.

    Validation weights are NOT computed here -- callers must use 1.0 for val
    (unweighted val metrics), per spec.
    """
    mode_norm = str(mode or WEIGHT_MODE_RANK).strip().lower()
    n = int(np.asarray(skill_train).shape[0])
    if mode_norm == WEIGHT_MODE_FLAT:
        return np.ones(n, dtype=np.float64)
    if mode_norm != WEIGHT_MODE_RANK:
        raise ValueError(f"unknown_weight_mode:{mode}:allowed={WEIGHT_MODE_CHOICES}")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    w, _norm, _sorted = _rank_weight_components(np.asarray(skill_train, dtype=np.float64))
    return w


def rank_weight_report(skill_train: np.ndarray, query_skills: list[float]) -> dict[str, Any]:
    """Diagnostic report for rank-mode weights: realized min/median/max plus
    the weight AT specific query skill values (not necessarily present in the
    data), all on the same normalized scale as the real per-sample weights."""
    s = np.asarray(skill_train, dtype=np.float64)
    n = int(s.shape[0])
    if n == 0:
        return {"n_train": 0, "mean": None, "min": None, "median": None, "max": None, "weight_at_skill": {}}
    w, norm, sorted_s = _rank_weight_components(s)
    weight_at: dict[str, float] = {}
    for q in query_skills:
        p_q = float(
            np.clip(
                _empirical_upper_cdf(sorted_s, np.asarray([float(q)]), n)[0],
                RANK_WEIGHT_MIN_P,
                RANK_WEIGHT_MAX_P,
            )
        )
        weight_at[str(q)] = float((RANK_WEIGHT_BASE + p_q) * norm)
    return {
        "n_train": n,
        "mean": float(w.mean()),
        "min": float(w.min()),
        "median": float(np.median(w)),
        "max": float(w.max()),
        "weight_at_skill": weight_at,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--check-coverage",
        action="store_true",
        help="Report action-index coverage only over the full dataset; do not encode.",
    )
    ap.add_argument("--observation-mode", type=str, default=OBSERVATION_MODE_V4)
    ap.add_argument("--max-turn", type=int, default=DEFAULT_MAX_TURN)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke slice: exact record cap, cut only at a game boundary. Omit for the full build.",
    )
    args = ap.parse_args(argv)

    if args.check_coverage:
        report = check_action_coverage(args.dataset_dir)
        print(json.dumps(report, indent=1))
        return 0 if report["misses"] == 0 else 1

    manifest = build_cache(
        dataset_dir=args.dataset_dir,
        out_dir=args.out,
        observation_mode=args.observation_mode,
        max_turn=args.max_turn,
        procs=args.procs,
        max_samples=args.max_samples,
    )
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
