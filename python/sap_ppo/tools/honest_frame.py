"""THE FIX IS NOT UNSEEDING (A1 ruling 1). Play keeps its seeded stream P
exactly as before, so We8 reproducibility, `--game-index-start` sharding,
CRN and the galleries all survive. What changes is that every IMAGINED
application of actions runs on a CLONE whose `meta.seed` is overridden with
an independent derived stream S:

    S = hash(engine_seed, turn, segment_index, sample_r)

`engine_seed` is the GAME's seed (`play_one_game`'s own
`engine_seed_rng.randrange`, i.e. a pure function of `(--seed, game_index)`),
NOT the state's current chained `meta.seed` -- keying on the latter would be
keying on P's position, which is exactly the coupling being removed. The
consequences of that choice:

- S is a pure function of (seed, game_index, turn, segment_index, sample_r),
  so an honest-frame run is as reproducible and as shardable as a
  determinized one; `--game-index-start` still splits an N-game stage
  across processes with no frame change.
- S never coincides with P's own next draw, so imagination cannot see the
  roll play is about to make. That is the whole property, and it is what
  the foresight-elimination test pins.
- `sample_r` is the CRN axis of A1 ruling 3: the k imagined completions of
  a stochastic prefix are keyed on `(decision, r)` with the CANDIDATE INDEX
  DELIBERATELY ABSENT, so sibling candidates that reach the same chance node
  face the SAME resampled outcome for the same r, and only the move under
  evaluation differs. `PROPOSAL_SAMPLE_R` (0) is reserved for the proposal
  stream (the state the driver hands the recommender), so no scored sample
  ever shares a stream with the chain generator that proposed it."""

from __future__ import annotations

import copy
import hashlib
from typing import Any


TURN_MODE_WHOLE_DETERMINIZED = "whole-determinized"
TURN_MODE_SEGMENTED_HONEST = "segmented-honest"
TURN_MODES: tuple[str, ...] = (TURN_MODE_WHOLE_DETERMINIZED, TURN_MODE_SEGMENTED_HONEST)
DEFAULT_TURN_MODE = TURN_MODE_WHOLE_DETERMINIZED


DEFAULT_STOCHASTIC_SAMPLES = 3

# `sample_r` reserved for the PROPOSAL stream -- the clone the driver hands
# the recommender each segment, on which candidate chains are generated and
# their deterministic prefixes are identified. Scored samples start at 1, so
# a candidate is never evaluated on the same imagined roll that proposed it.
PROPOSAL_SAMPLE_R = 0


MAX_SEGMENTS_PER_TURN = 64

IMAGINATION_SALT = "exp13_a1_imagination"
IMAGINATION_SEED_BITS = 63

# Which input the imagination key was actually derived from, recorded in the
# search diagnostics so a report can never quietly claim the honest frame
# while running off a fallback (see `search_recommender._imagination_seed`).
# "driver_context" is the only value a driver-run arm should ever show.
KEY_SOURCE_DRIVER_CONTEXT = "driver_context"
KEY_SOURCE_STATE_SEED = "state_seed"


def normalize_turn_mode(turn_mode: str | None) -> str:
    mode = str(turn_mode or DEFAULT_TURN_MODE).strip().lower()
    if mode not in TURN_MODES:
        raise ValueError(f"unknown_turn_mode:{turn_mode}:valid={list(TURN_MODES)}")
    return mode


def is_honest(turn_mode: str | None) -> bool:
    return normalize_turn_mode(turn_mode) == TURN_MODE_SEGMENTED_HONEST


def imagination_seed(*, engine_seed: int, turn: int, segment_index: int, sample_r: int) -> int:
    """The A1 stream-S key: `hash(engine_seed, turn, segment_index, sample_r)`.

    Stable across processes and python runs (sha256, not `hash()`), so an
    honest-frame shard reproduces the unsharded run exactly -- see the module
    docstring for why every component is in the key and why P's chained
    `meta.seed` is not.
    """
    key = (
        f"{IMAGINATION_SALT}:{int(engine_seed)}:{int(turn)}:"
        f"{int(segment_index)}:{int(sample_r)}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - IMAGINATION_SEED_BITS)


def imagined_clone(state: dict[str, Any], *, seed: int) -> dict[str, Any]:
    """A deep copy of `state` whose engine RNG runs off stream S instead of P.

    `seed_known` is forced True alongside the override: with it False,
    `engine._rng_from_state` would draw OS entropy instead, which is honest
    but throws away reproducibility (A1 ruling 1 chose stream separation
    over unseeding precisely to keep it). Nothing else on the state is
    touched, so the imagined board differs from the real one in the RNG
    stream and in nothing else.
    """
    work = copy.deepcopy(state)
    meta = work.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        work["meta"] = meta
    meta["seed_known"] = True
    meta["seed"] = int(seed)
    return work


def read_engine_seed(state: dict[str, Any]) -> int:
    """`meta.seed` off a state, defensively (0 when absent/unparseable).

    Called ONCE per game by the driver, on the turn-1 state, to capture the
    game's own engine seed before play chains it forward -- that captured
    value is the `engine_seed` component of every key above.
    """
    meta = state.get("meta") if isinstance(state, dict) else None
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("seed", 0) or 0)
    except (TypeError, ValueError):
        return 0
