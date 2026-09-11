"""Deliberately imports from NEITHER `..train` NOR `..tools`: `..train.env`
already gets imported by `tools/bc_recommender.py`, so a neutral module that
both `train/runtime.py` and `tools/eval_versus_fullgame.py` import from must
not close that into a cycle. Only stdlib + nothing else is imported here."""

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
    """Raise ValueError if any two named splits share a participation_id."""
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
    lookup, the same pattern `_opponent_pack`/`side_pack` already use."""

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

            if self.opponent_pack is not None and str(game["opponent_pack"] or "").strip() != self.opponent_pack:
                continue


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


            raise ValueError(
                f"no_games_in_snapshot:{self.snapshot_path}:"
                f"games={len(games)}:opponent_pack={self.opponent_pack}:"
                f"rank_bounds=({self.opponent_rank_min},{self.opponent_rank_max})"
            )


        splits_raw = payload.get("splits")
        if isinstance(splits_raw, dict) and all(isinstance(splits_raw.get(name), list) for name in SPLIT_NAMES):
            self.splits: dict[str, list[str]] = {
                name: [str(p).strip() for p in splits_raw[name] if str(p).strip()] for name in SPLIT_NAMES
            }
            self.splits_source = "snapshot_builtin"
        else:
            if self.snapshot_version != CHAIN_SNAPSHOT_VERSION:


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


        self.long_pids = long_pids
        # Deterministic, sorted universe for the length-blind initial pick.
        self.all_pids = sorted(by_pid)
        self._random_rng = random.Random(self.seed)

    @property
    def max_turn_with_candidates(self) -> int | None:
        """Deepest turn this pool can serve an opponent at, None if empty."""
        return max(self.by_turn) if self.by_turn else None

    def initial_pid_for_game(self, game_index: int) -> str:
        """Deterministic (by seed + game_index) initial followed pid, uniform
        over the WHOLE filtered pool."""
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
        supplied `rng` instead of this instance's own shared `self._random_rng`."""
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
