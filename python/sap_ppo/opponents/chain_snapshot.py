"""Chain-snapshot-backed opponent source (exp09 W5 P0, neutral shared module).

`ChainSnapshotSource` was originally defined inline in
`tools/eval_versus_fullgame.py` (the W4c full-game versus eval driver). It is
extracted here, UNCHANGED in eval-facing behavior, so the PPO training
runtime (`train/runtime.py::build_opponent_provider`, `chain_snapshot` mode)
can share the exact same opponent-sampling code the eval frame uses instead
of a different data source (`TrainingEnv.__init__` used to default to
`ReplayDBOpponentProvider`, a live-Postgres lookup against a database that no
longer runs -- see internal design notes "W5 P0" and
`tools/frame_parity_harness.py`'s "opponent" dimension, which is the
regression gate for this fix).

Deliberately imports from NEITHER `..train` NOR `..tools`: `..train.env`
already gets imported by `tools/bc_recommender.py`, so a neutral module that
both `train/runtime.py` and `tools/eval_versus_fullgame.py` import from must
not close that into a cycle. Only stdlib + nothing else is imported here.

## Held-out pid split (exp09 W5 P0)

The production snapshot (`data/opponents/chain_snapshot_v45_all.json`) ships
a built-in `splits` dict (`{"train": [...], "val": [...], "test": [...]}`,
verified: 5504/688/688 of 6880 games, seed 42, mutually disjoint -- built by
`tools/build_chain_snapshot.py::_build_splits`/`build_chain_snapshot`). This
module's `split=` constructor arg restricts `by_pid`/`by_turn`/`long_pids` to
one named split, so:
- TRAINING can be pointed at `split="train"` (see `train/runtime.py`'s new
  `chain_snapshot` mode, default `--opponent-split train`).
- The held-out/eval reference can be pointed at a FIXED `split="test"`.
These are disjoint by construction (see `_assert_splits_disjoint`, run
whenever a caller actually SELECTS a split -- exp09 W5 P0 Fix E, cross-model
review finding 6 -- see `_assert_splits_disjoint`'s own docstring for why a
`split=None` caller skips it) -- training can never draw an opponent the
held-out reference also uses to score it.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterator, NamedTuple

_CHAIN_SNAPSHOT_RELPATH = "data/opponents/chain_snapshot_v45_all.json"


def _default_chain_snapshot() -> str:
    """Resolve the default snapshot without hard-coding a build-box path.

    `SAP_CHAIN_SNAPSHOT` wins if set; otherwise the path is taken relative to
    the repository root, which is where `data/` lives (a symlink to the data
    disk here, a real directory in a fresh clone).
    """
    override = os.environ.get("SAP_CHAIN_SNAPSHOT", "").strip()
    if override:
        return override
    return str(Path(__file__).resolve().parents[3] / _CHAIN_SNAPSHOT_RELPATH)


DEFAULT_CHAIN_SNAPSHOT = _default_chain_snapshot()
DEFAULT_SEED = 0
DEFAULT_LONG_MIN = 10
CHAIN_SNAPSHOT_VERSION = "chain_snapshot_v1"
# exp09 W5.3: `tools/build_chain_snapshot.py`'s new local-source mode emits
# `chain_snapshot_v2` (turtle-only, identity-and-rank-carrying games, a
# builder-computed `splits_meta` sibling of `splits` -- see that module's
# docstring). `CHAIN_SNAPSHOT_VERSION` (above) is kept importable UNCHANGED
# (other modules import it by name, e.g. `frame_parity_harness.py`'s test
# payloads), it just no longer describes the FULL set of accepted versions
# on its own -- `CHAIN_SNAPSHOT_VERSIONS` is that set.
CHAIN_SNAPSHOT_VERSIONS: tuple[str, ...] = (CHAIN_SNAPSHOT_VERSION, "chain_snapshot_v2")

# Held-out split constants. Seed/ratios match `build_chain_snapshot.py`'s own
# defaults (`--seed 42 --val-ratio 0.1 --test-ratio 0.1`) so a
# deterministically-DERIVED fallback (only used if a snapshot's own
# `splits` is absent/malformed) lands on the same shape as a "real" one.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1


def _derive_splits(pids: list[str], *, seed: int, val_ratio: float, test_ratio: float) -> dict[str, list[str]]:
    """Deterministically assign `pids` to train/val/test (fallback path only).

    Same algorithm as `tools/build_chain_snapshot.py::_build_splits`
    (sorted-unique -> seeded shuffle -> ratio slice), reimplemented here
    rather than imported: that module lives under `tools/` and pulls in
    psycopg/replay-db build plumbing this neutral module must not depend on
    (see module docstring). Only exercised when a snapshot file's own
    `splits` key is missing or malformed -- production snapshots carry a
    real one (verified directly, see module docstring).
    """
    unique = sorted({str(p).strip() for p in pids if str(p).strip()})
    rng = random.Random(int(seed))
    rng.shuffle(unique)

    total = len(unique)
    n_test = int(round(total * float(max(0.0, min(0.45, test_ratio)))))
    n_val = int(round(total * float(max(0.0, min(0.45, val_ratio)))))
    n_test = max(0, min(total, n_test))
    n_val = max(0, min(total - n_test, n_val))
    n_train = max(0, total - n_val - n_test)

    return {
        "train": unique[:n_train],
        "val": unique[n_train : n_train + n_val],
        "test": unique[n_train + n_val :],
    }


def _assert_splits_disjoint(splits: dict[str, list[str]]) -> None:
    """Raise ValueError if any two named splits share a participation_id.

    Called by `ChainSnapshotSource.__init__` whenever a caller actually
    SELECTS a split (`split=` is not None) -- exp09 W5 P0 Fix E (cross-model
    review finding 6): a `split=None` (unsplit) caller never relies on the
    train/eval boundary holding at all (the W4c eval driver, this repo's
    own frame-parity harness "opponent" dimension), so an old/regenerated
    snapshot's UNUSED, overlapping `splits` metadata must not block
    construction for it -- only a caller that IS relying on the boundary
    (training pointed at `split="train"`, the held-out reference pointed at
    `split="test"`) needs this to fire, so a corrupted/regenerated snapshot
    can never silently let THAT caller's training run and its held-out
    reference share opponents. Cheap either way: the pid lists are tiny,
    already resident in memory from the `json.loads` that just happened.
    See module docstring, "Held-out pid split".
    """
    names = sorted(splits.keys())
    sets = {name: {str(p).strip() for p in (splits.get(name) or []) if str(p).strip()} for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                example = sorted(overlap)[:3]
                raise ValueError(
                    f"chain_snapshot_splits_not_disjoint:{a}&{b}:overlap_count={len(overlap)}:e.g.={example}"
                )


class _StoreRow(NamedTuple):
    """One entry's byte range in the blob, plus the three game-level tags.

    A NamedTuple and not a dict on purpose: about 31,000 of these stay resident
    per process, and the whole point of the store is that what stays resident
    is small.
    """

    offset: int
    length: int
    replay_id: Any
    opponent_pack: Any
    opponent_rank: Any


class ChainSnapshotSource:
    """Local chain-snapshot backed opponent source (the replay DB is dead).

    Loads the chain snapshot file ONCE (`chain_snapshot_v1`: `{version,
    games: [{participation_id, replay_id, pack, opponent_pack, turns:
    [{turn, battle, parsed_state, ...}]}], splits}`, ~652MB, ~16s to
    `json.loads`) and builds three in-memory indexes so per-turn sampling
    during a rollout is O(1)/O(k):

    - `by_pid[pid][turn] -> entry`: exact (participation_id, turn) lookup,
      used to follow one real game's board chain across a whole versus game
      (the "one opponent building over the game" semantic -- NOT resampling
      a fresh unrelated opponent every turn).
    - `by_turn[turn] -> [(pid, entry), ...]`: all entries at a given turn,
      used for the random fallback when the followed game runs out of
      turns (`end_turn.py` calls this automatically and appends
      `end_turn_versus_chain_fallback_random` to `engine_notes` when it
      does -- see `resolve_end_turn_with_sampled_battle`).
    - `long_pids`: participation_ids of games with >= `long_min` indexed
      turns. TELEMETRY ONLY since exp09 W5.3 (Ruihan, 2026-07-18): both the
      initial followed-pid choice and the random fallback USED to prefer
      long chains, but chain length is survivor bias (a long chain = an
      opponent who survived = a systematically stronger player), which is
      exactly the wrong filter for a low-rank curriculum pool. Selection is
      now uniform over the filtered pool; the reported fallback rate is the
      honest cost.

    Each stored entry is tagged with `_replay_id`/`_opponent_pack` from its
    parent game (the raw per-turn entry itself only carries
    `turn`/`battle`/`parsed_state`; `replay_id`/`opponent_pack` live at the
    game level) so `sample_for_pid`/`sample_random` can report them without
    a second lookup.

    CRITICAL semantic (locked by the project owner): `parsed_state` is
    returned AS-IS, never flipping player/opponent. Downstream,
    `resolve_end_turn_with_sampled_battle` overwrites `config["playerPets"]`
    with the caller's own current board (`team_to_pet_configs(battle_state
    ["team"])`) and keeps `parsed_state["opponentPets"]` untouched as the
    board the caller fights -- that IS the opponent the followed real player
    faced that turn. This is the exact same live-training opponent-sampling
    envelope as `sap_ppo.opponents.replay_db.sample_opponent_team_for_pid`
    (same `ok`/`error`/`battle`/`parsed_state`/`participation_id`/
    `replay_id`/`side`/`side_pack`/`source`/`build_model`/`team` key set),
    so `end_turn.py` consumes a snapshot-sourced payload identically to a
    live-DB one -- and this class's own `.sample()` (below) makes it a
    drop-in `train/opponents.py::OpponentProvider` too.

    `split` (exp09 W5 P0, default None = unrestricted, all pids): restricts
    `by_pid`/`by_turn`/`long_pids` to one of the snapshot's named splits
    (`"train"`/`"val"`/`"test"`, see module docstring). `None` preserves the
    original, pre-split behavior byte-for-byte (used by the existing W4c
    eval driver, which predates the split concept and is not changed here).

    `opponent_pack` (exp09 W5 Rev 6, default None = all packs): keeps only
    games whose game-level `opponent_pack` equals the given pack name
    (e.g. "Turtle" -- the mixed production snapshot is only ~40% Turtle
    opponents, see PLAN.md "W5 continuation (Rev 6)"). Filtering happens at
    index time, BEFORE the split restriction, so `by_pid`/`by_turn`/
    `long_pids` and the split intersection all see only that pack's games.
    `None` preserves the mixed-pool behavior byte-for-byte.

    `opponent_rank_min` / `opponent_rank_max` (exp09 W5.3, default None =
    unbounded): keep only games whose game-level `opponent_rank` (an int
    ladder rank, present on `chain_snapshot_v2` games built by
    `tools/build_chain_snapshot.py`'s local-source mode -- see that module's
    docstring; ABSENT on every `chain_snapshot_v1` game) falls in
    `[opponent_rank_min, opponent_rank_max]`, INCLUSIVE on both ends, either
    bound independently optional. This is the "low-rank curriculum pool"
    knob (PLAN.md "W5.3 curriculum data rebuild": e.g. `opponent_rank_max=
    1500` trains against ladder-baseline-and-below Turtle opponents only).
    Filtering happens at index time, immediately AFTER the `opponent_pack`
    filter above and BEFORE the split restriction, so every downstream
    index (`by_pid`/`by_turn`/`long_pids`) and the split intersection all
    see only rank-qualifying games -- same ordering discipline as
    `opponent_pack`, for the same reason (a caller restricting itself to a
    split is relying on `long_pids`/`by_pid` already reflecting every prior
    filter).

    CRITICAL semantic (mirrors the funnel-exclusion discipline
    `tools/build_chain_snapshot.py` uses when it BUILDS the rank field, see
    that module's `opponent_rank_min`/`_max` runtime-filter note): whenever
    EITHER bound is set, a game whose `opponent_rank` is not a plain `int`
    (missing, `None`, or -- pathologically -- a `bool`, which Python's
    `isinstance(x, int)` would otherwise accept) is EXCLUDED, never silently
    treated as in-bounds. Unknown skill must not leak into a rank-bounded
    training pool -- an opponent this code cannot rank is exactly the kind
    of opponent a low-rank curriculum run must not accidentally include. A
    direct, load-bearing consequence: pointing a rank-bounded construction
    at a `chain_snapshot_v1` file (whose games carry no `opponent_rank` key
    at all) excludes EVERY game, so `long_pids` ends up empty and this
    class's existing `no_long_games_in_snapshot` guard (below) fires --
    loudly, at construction time, with no special-casing needed for that
    combination. `None`/`None` (the default) preserves rank-unfiltered
    behavior byte-for-byte, including for `chain_snapshot_v1` files (which
    have no rank field to filter on in the first place).

    Rank-filtered entries are counted (`self.rank_filtered_out`,
    `self.rank_unranked_excluded`) for tests/telemetry, and every stored
    entry is additionally tagged `_opponent_rank` (alongside the existing
    `_replay_id`/`_opponent_pack` tags) so `sample_for_pid`/`sample_random`
    can report the followed opponent's rank (`side_rank`) without a second
    lookup, the same pattern `_opponent_pack`/`side_pack` already use.
    """

    def __init__(
        self,
        snapshot_path: str | Path,
        *,
        long_min: int = DEFAULT_LONG_MIN,
        seed: int = DEFAULT_SEED,
        split: str | None = None,
        split_seed: int = DEFAULT_SPLIT_SEED,
        val_ratio: float = DEFAULT_VAL_RATIO,
        test_ratio: float = DEFAULT_TEST_RATIO,
        opponent_pack: str | None = None,
        opponent_rank_min: int | None = None,
        opponent_rank_max: int | None = None,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.long_min = int(long_min)
        self.seed = int(seed)
        self.opponent_pack = str(opponent_pack).strip() if opponent_pack else None
        self.opponent_rank_min = int(opponent_rank_min) if opponent_rank_min is not None else None
        self.opponent_rank_max = int(opponent_rank_max) if opponent_rank_max is not None else None
        self._rank_bounds_active = self.opponent_rank_min is not None or self.opponent_rank_max is not None
        self.rank_filtered_out = 0
        self.rank_unranked_excluded = 0

        # WHERE THE ENTRIES LIVE. A sibling `<snapshot>.store/` directory holds
        # every usable entry's `battle` and `parsed_state` in one blob plus an
        # index, so N worker processes share ONE copy through the page cache
        # instead of each parsing the same 250 MiB into 1.22 GiB of private
        # Python objects. Absent, this loads the JSON exactly as before.
        #
        # The store is bound to the snapshot by digest and refuses to pair
        # with another one, so "the store is stale" cannot be a silent state.
        # `storage_mode` is public because a run that reports its numbers
        # should be able to say which way it read its opponents.
        self.store = None
        self.storage_mode = "json"
        store_dir = Path(str(self.snapshot_path) + ".store")
        if store_dir.is_dir():
            from .chain_snapshot_store import ChainSnapshotStore, sha256_file

            self.store = ChainSnapshotStore(
                store_dir, expect_source_sha256=sha256_file(self.snapshot_path)
            )
            self.storage_mode = "store"

        if self.store is not None:
            payload = {
                "version": self.store.index.get("version"),
                "metadata": self.store.index.get("metadata"),
                "splits": self.store.index.get("splits"),
                "splits_meta": self.store.index.get("splits_meta"),
            }
            games = self.store.games
        else:
            # A gzipped snapshot is 20x smaller, which is the difference between a
            # pool that can ship in a repository and one that cannot.
            if self.snapshot_path.suffix == ".gz":
                import gzip

                with gzip.open(self.snapshot_path, "rt", encoding="utf-8") as fh:
                    payload = json.load(fh)
            else:
                payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            games = payload.get("games") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or str(payload.get("version", "")) not in CHAIN_SNAPSHOT_VERSIONS:
            raise ValueError(
                f"unsupported_chain_snapshot_version:{self.snapshot_path}:"
                f"{payload.get('version') if isinstance(payload, dict) else type(payload)}"
            )
        self.snapshot_version = str(payload.get("version", ""))
        if not isinstance(games, list) or not games:
            raise ValueError(f"chain_snapshot_missing_games:{self.snapshot_path}")

        by_pid: dict[str, dict[int, dict[str, Any]]] = {}
        by_turn: dict[int, list[tuple[str, dict[str, Any]]]] = {}
        long_pids: list[str] = []

        for game in self._iter_source_games(games):
            if game is None:
                continue
            # exp09 W5 Rev 6: turtle-only (or any single-pack) opponent pool.
            if self.opponent_pack is not None and str(game["opponent_pack"] or "").strip() != self.opponent_pack:
                continue
            # exp09 W5.3: low-rank curriculum pool. Runs AFTER the
            # opponent_pack filter and BEFORE pid/turns validation, so an
            # unranked or out-of-band game never reaches long_pids/by_pid/
            # by_turn at all when a bound is set.
            if self._rank_bounds_active:
                rank = game["opponent_rank"]
                if not isinstance(rank, int) or isinstance(rank, bool):
                    self.rank_unranked_excluded += 1
                    continue
                if self.opponent_rank_min is not None and rank < self.opponent_rank_min:
                    self.rank_filtered_out += 1
                    continue
                if self.opponent_rank_max is not None and rank > self.opponent_rank_max:
                    self.rank_filtered_out += 1
                    continue
            pid = str(game["participation_id"] or "").strip()
            if not pid or not game["n_raw_turns"]:
                continue

            per_turn_map: dict[int, Any] = {}
            for turn_i, value in game["entries"]:
                per_turn_map[turn_i] = value
                by_turn.setdefault(turn_i, []).append((pid, value))

            if not per_turn_map:
                continue
            by_pid[pid] = per_turn_map
            # "long" is measured by VALID indexed turns (`per_turn_map`), not
            # the raw `turns` length -- a game with many raw turns but few
            # usable ones is not actually a long, switch-free opponent, and
            # preferring it would raise the fallback rate.
            if len(per_turn_map) >= self.long_min:
                long_pids.append(pid)

        if not by_pid:
            # exp09 W5.3 (Ruihan, 2026-07-18): the guard used to require LONG
            # games because selection preferred them; selection is now
            # length-blind (see initial_pid_for_game / sample_random), so the
            # construction-time requirement is simply "any usable game".
            raise ValueError(
                f"no_games_in_snapshot:{self.snapshot_path}:"
                f"games={len(games)}:opponent_pack={self.opponent_pack}:"
                f"rank_bounds=({self.opponent_rank_min},{self.opponent_rank_max})"
            )

        # --- exp09 W5 P0: held-out pid split -------------------------------
        # Resolve the split assignment from the snapshot's own built-in
        # `splits` (verified present + well-formed on the production
        # chain_snapshot_v45_all.json: train=5504/val=688/test=688 of 6880,
        # seed=42, mutually disjoint), falling back to a deterministically
        # derived one (same shape/algorithm, see `_derive_splits`) only if
        # the file's own `splits` is absent or malformed -- logged loudly
        # because that should not happen for a real snapshot.
        splits_raw = payload.get("splits")
        if isinstance(splits_raw, dict) and all(isinstance(splits_raw.get(name), list) for name in SPLIT_NAMES):
            self.splits: dict[str, list[str]] = {
                name: [str(p).strip() for p in splits_raw[name] if str(p).strip()] for name in SPLIT_NAMES
            }
            self.splits_source = "snapshot_builtin"
        else:
            if self.snapshot_version != CHAIN_SNAPSHOT_VERSION:
                # codex W5.3 review P2: a chain_snapshot_v2 file ADVERTISES
                # opponent-user-grouped, match-pruned, leakage-free splits
                # (that is half the point of v2 -- see
                # tools/build_chain_snapshot.py::_build_local_splits). Falling
                # back to the pid-shuffle derivation here would silently
                # REPLACE that contract with plain pid splits (recurring
                # opponent users straddling train/test) while looking
                # perfectly healthy downstream. A v2 payload without valid
                # built-in splits is therefore a hard error; only v1 (which
                # never promised grouped splits) keeps the legacy fallback.
                raise ValueError(
                    f"chain_snapshot_v2_missing_builtin_splits:{self.snapshot_path}:"
                    "a v2 snapshot must carry its builder-computed 'splits' "
                    "(pid-shuffle derivation would break the leakage contract)"
                )
            print(
                f"WARNING: {self.snapshot_path} has no valid built-in 'splits' key "
                f"(a {CHAIN_SNAPSHOT_VERSION} snapshot should carry one -- see "
                "tools/build_chain_snapshot.py); deterministically deriving one instead "
                f"from sorted long_pids (seed={split_seed}, val_ratio={val_ratio}, "
                f"test_ratio={test_ratio}). This should not happen for a production "
                "snapshot -- treat any split-dependent result built against a "
                "fallback-derived split with extra caution.",
                flush=True,
            )
            self.splits = _derive_splits(
                sorted(long_pids), seed=int(split_seed), val_ratio=float(val_ratio), test_ratio=float(test_ratio)
            )
            self.splits_source = f"derived_fallback_seed={split_seed}"

        self.split = str(split).strip().lower() if split is not None else None
        if self.split is not None:
            # exp09 W5 P0 Fix E (cross-model review finding 6): disjointness
            # is validated only when a split is actually SELECTED. A caller
            # restricting itself to one split is RELYING on the train/eval
            # boundary holding, so this always runs for that caller (cheap:
            # the pid lists are tiny, already resident in memory). A
            # `split=None` (unsplit) caller -- the W4c eval driver, this
            # harness's own "opponent" dimension -- never relies on the
            # boundary at all, so an old/regenerated snapshot's UNUSED,
            # overlapping `splits` metadata must not block it (previously
            # unconditional here, which did exactly that).
            _assert_splits_disjoint(self.splits)
            if self.split not in self.splits:
                raise ValueError(f"unknown_chain_snapshot_split:{self.split}:valid={sorted(self.splits.keys())}")
            include_pids = set(self.splits[self.split])
            by_pid = {pid: turn_map for pid, turn_map in by_pid.items() if pid in include_pids}
            by_turn = {
                turn_i: [(pid, entry) for pid, entry in rows if pid in include_pids]
                for turn_i, rows in by_turn.items()
            }
            by_turn = {turn_i: rows for turn_i, rows in by_turn.items() if rows}
            long_pids = [pid for pid in long_pids if pid in include_pids]
            if not by_pid:
                raise ValueError(
                    f"no_games_in_snapshot_split:{self.snapshot_path}:split={self.split}:"
                    f"pids_in_split={len(include_pids)}"
                )

        self.by_pid = by_pid
        self.by_turn = by_turn
        # Telemetry only since exp09 W5.3 (see initial_pid_for_game): how
        # many chains could serve a full game without switching. NOT used
        # for selection anywhere.
        self.long_pids = long_pids
        # Deterministic, sorted universe for the length-blind initial pick.
        self.all_pids = sorted(by_pid)
        self._random_rng = random.Random(self.seed)

    @property
    def max_turn_with_candidates(self) -> int | None:
        """Deepest turn this pool can serve an opponent at, None if empty.

        exp13 F1 (2026-08-07): `sample_random` / `sample_random_with_rng`
        return `no_snapshot_opponent_for_turn:N` for every turn past this
        one, so this number is the pool's real depth AS AN OPPONENT SUPPLY.
        It is a property of the FILTERED pool -- split, opponent pack and
        rank bound are all applied above before `by_turn` is stored -- not of
        the snapshot file, which is why it is derived on read rather than
        recorded once somewhere.

        A caller running a game against this pool must not let it reach a
        turn beyond this number. `eval_versus_fullgame.effective_arena_max_turn`
        is the single place that clamp is computed; nothing hard-codes a
        depth, because the two measured values (21 on the v45 val pool, 25 on
        the v45 train pool, both measured 2026-08-07) move the moment the
        snapshot, split or rank bound moves.
        """
        return max(self.by_turn) if self.by_turn else None

    def initial_pid_for_game(self, game_index: int) -> str:
        """Deterministic (by seed + game_index) initial followed pid, uniform
        over the WHOLE filtered pool.

        exp09 W5.3 (Ruihan, 2026-07-18): this used to draw from LONG games
        only (>= long_min indexed turns). Chain length is survivor bias --
        a long chain means that opponent SURVIVED long in their own game,
        i.e. systematically the stronger player -- which is exactly the
        wrong filter for a low-rank curriculum pool (the weakest opponents
        die fast and were being excluded from ever starting a game). All
        length-based selection is removed; `long_min`/`long_pids` remain as
        REPORTED telemetry only. The cost is a higher fallback rate (a short
        starting chain runs out sooner and the game switches to another
        chain mid-way), which every eval reports.
        """
        rng = random.Random(f"snapshot_opponent_source_init_pid:{self.seed}:{int(game_index)}")
        return str(rng.choice(self.all_pids))


    def _iter_source_games(self, games: list[Any]) -> Iterator[dict[str, Any] | None]:
        """One normalised record per game, in the snapshot's own order.

        Yields `None` for a game the loader has always skipped outright, so the
        caller's `continue` is the same statement it was, and so a malformed
        game still occupies its position -- position is what
        `sample_random_with_rng` selects by.

        `entries` carries, in the game's own turn order, either the tagged dict
        the JSON path has always built or a `_StoreRow` naming a byte range in
        the blob. Everything downstream treats it as opaque and hands it back
        to `_entry_to_payload`.
        """
        if self.store is not None:
            for game in games:
                if game.get("malformed"):
                    yield None
                    continue
                yield {
                    "participation_id": game.get("participation_id"),
                    "opponent_pack": game.get("opponent_pack"),
                    "opponent_rank": game.get("opponent_rank"),
                    "n_raw_turns": int(game.get("n_raw_turns") or 0),
                    "entries": [
                        (
                            int(turn_i),
                            _StoreRow(
                                int(offset),
                                int(length),
                                game.get("replay_id"),
                                game.get("opponent_pack"),
                                game.get("opponent_rank"),
                            ),
                        )
                        for turn_i, offset, length in game.get("turns") or []
                    ],
                }
            return

        for game in games:
            if not isinstance(game, dict):
                yield None
                continue
            replay_id = game.get("replay_id")
            opponent_pack = game.get("opponent_pack")
            opponent_rank = game.get("opponent_rank")
            turns = game.get("turns")
            entries: list[tuple[int, Any]] = []
            if isinstance(turns, list):
                for entry in turns:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        turn_i = int(entry.get("turn"))
                    except (TypeError, ValueError):
                        continue
                    # Only index entries a downstream battle can actually use:
                    # BOTH the raw `battle` payload AND the pre-parsed
                    # `parsed_state` must be present dicts. A malformed entry
                    # would otherwise be returned as `ok=True` with a null
                    # payload and hard-FAIL that turn instead of the chain
                    # cleanly falling back to another game's board.
                    if not isinstance(entry.get("battle"), dict) or not isinstance(
                        entry.get("parsed_state"), dict
                    ):
                        continue
                    # Shallow copy + tag: nested payloads are shared (not
                    # deep-copied) between by_pid and by_turn -- safe because
                    # every consumer deep-copies before mutating.
                    tagged = dict(entry)
                    tagged["_replay_id"] = replay_id
                    tagged["_opponent_pack"] = opponent_pack
                    tagged["_opponent_rank"] = opponent_rank
                    entries.append((turn_i, tagged))
            yield {
                "participation_id": game.get("participation_id"),
                "opponent_pack": opponent_pack,
                "opponent_rank": opponent_rank,
                "n_raw_turns": len(turns) if isinstance(turns, list) else 0,
                "entries": entries,
            }

    def _entry_to_payload(self, pid: str, entry: Any) -> dict[str, Any]:
        """The one formatter, for either storage.

        Exactly five fields are ever read out of an entry, which is why the
        store carries only these: the two payload dicts, and three game-level
        tags. A store row resolves its two dicts from the blob on demand; a
        JSON entry has them already.
        """
        if isinstance(entry, _StoreRow):
            blob = self.store.read_entry(entry.offset, entry.length)
            battle = blob.get("battle")
            parsed_state = blob.get("parsed_state")
            replay_id = entry.replay_id
            side_pack = entry.opponent_pack
            side_rank = entry.opponent_rank
        else:
            battle = entry.get("battle")
            parsed_state = entry.get("parsed_state")
            replay_id = entry.get("_replay_id")
            side_pack = entry.get("_opponent_pack")
            side_rank = entry.get("_opponent_rank")
        return {
            "ok": True,
            "error": None,
            "battle": battle,
            "parsed_state": parsed_state,
            "participation_id": pid,
            "replay_id": replay_id,
            "created_at": None,
            "side": "",
            "source": "chain_snapshot",
            "build_model": None,
            "side_pack": side_pack or "Turtle",
            # exp09 W5.3: the followed opponent's game-level ladder rank (int)
            # or None -- absent entirely on chain_snapshot_v1 entries.
            "side_rank": side_rank,
            "team": [],
        }

    def sample_for_pid(self, pid: str, turn: int) -> dict[str, Any]:
        """Exact (pid, turn) lookup. Missing -> `ok:False` (triggers end_turn.py's fallback)."""
        turn_i = int(turn)
        entry = self.by_pid.get(str(pid), {}).get(turn_i)
        if entry is None:
            return {
                "ok": False,
                "error": f"pid_turn_not_in_snapshot:{pid}:turn={turn_i}",
                "team": [],
                "participation_id": pid,
                "source": "chain_snapshot",
            }
        return self._entry_to_payload(str(pid), entry)

    def sample_random(self, turn: int) -> dict[str, Any]:
        """Fallback sampler: a random OTHER game's board at this turn, preferring long games."""
        return self.sample_random_with_rng(turn, self._random_rng)

    def sample_random_with_rng(self, turn: int, rng: random.Random) -> dict[str, Any]:
        """Same selection as `sample_random`, but drawing from the CALLER-
        supplied `rng` instead of this instance's own shared `self._random_rng`.

        exp09 W1 (teacher-ceiling rollout scorer): `search_recommender.py`'s
        rollout-scoring continuations (`--search-scoring rollout`) need a
        random-fallback draw with the EXACT same distribution `sample_random`
        uses, but must never advance `self._random_rng` itself -- that stream
        also serves the real game this `ChainSnapshotSource` instance backs,
        and a throwaway rollout simulation consuming draws from it would make
        the real game's own later fallback draws depend on how many rollouts
        happened to run first (order-dependent, breaks reproducibility, and
        is exactly the hidden cross-talk the rollout design explicitly rules
        out -- "must not mutate the outer game's opponent-chain iterator
        state"). `sample_random` is unchanged behavior: it now delegates here
        with `self._random_rng`, so every existing caller draws from the
        identical stream in the identical order as before this method
        existed.
        """
        turn_i = int(turn)
        rows = self.by_turn.get(turn_i, [])
        if not rows:
            return {
                "ok": False,
                "error": f"no_snapshot_opponent_for_turn:{turn_i}",
                "team": [],
                "participation_id": None,
                "source": "chain_snapshot",
            }
        # exp09 W5.3 (Ruihan, 2026-07-18): no long-game preference here
        # either -- same survivor bias as initial_pid_for_game. Note a short
        # game can only ever be a candidate at turns it actually HAS: at the
        # common "followed chain ran out at turn T" fallback, every candidate
        # holding turn T is by construction at least T turns long.
        pid, entry = rng.choice(rows)
        return self._entry_to_payload(pid, entry)

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        """`train/opponents.py::OpponentProvider` adapter.

        Delegates to `sample_for_pid` when a pid is forced, else
        `sample_random` -- the same forced-pid-or-random convention
        `ReplayDBOpponentProvider.sample` already uses (`train/opponents.py`),
        so this class is a drop-in `OpponentProvider` for `TrainingEnv` /
        `train/runtime.py::build_opponent_provider(mode="chain_snapshot")`.
        """
        pid = str(forced_pid or "").strip()
        if pid:
            return self.sample_for_pid(pid, int(turn))
        return self.sample_random(int(turn))
