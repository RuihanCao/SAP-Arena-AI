"""Turn-1 "opening source" for full-game versus training/eval.

THE BUG THIS FIXES: every full-game driver (`tools/eval_versus_fullgame.py`,
`train/env.py::TrainingEnv`, `train/eval_ppo.py`) used to deep-copy the SAME
turn-1 fixture (`fixtures/parity_cases/sample_case.json`, shop hardcoded to
pet-horse/pet-beaver/pet-mosquito/food-honey) for EVERY game. The engine only
rolls the shop on an explicit ROLL action or a turn-2+ advance
(`engine.py::apply_action`'s ROLL branch / `resolve_end_turn_post_battle`),
never at turn-1 construction, so every game's OPENING was identical (measured
via `tools/glassbox/diversity_panel.py`: 1 unique agent turn-1 board across a
300-game eval run).

THE FIX: draw each game's turn-1 opening (real shop presented + gold 10 +
empty team + turn 1 -- i.e. BEFORE any purchase, so the agent makes its own
decisions) from a fresh, per-`game_index`-seeded engine roll
(`engine._roll_shop_slots`, the EXACT SAME mechanism a real turn-2+ advance /
an explicit ROLL action already uses -- not reimplemented here, just invoked
once more at turn-1 construction). The opening shop is NOT special: it is
just another random roll, exactly like every other turn's shop. Deterministic
per `game_index` (so eval stays reproducible run to run, and repeated calls
for the same index are stable); independent across indices (so parallel
games never collide on the same draw).

TRAINING DECORRELATION: parallel `--num-envs` training workers each count
episodes from their OWN local 0, 1, 2, ... (`train_ppo.py::_make_env(rank)`,
one `SubprocVecEnv` subprocess per rank) -- wiring `state_for_game` straight
off that bare counter would make every worker draw the IDENTICAL opening
sequence (rank 0 episode 5 and rank 3 episode 5 would get the SAME turn-1
shop). `training_opening_index(env_seed, episode_index)` fixes this for the
TRAINING path only (see `TrainingEnv`'s `opening_env_seed` constructor arg):
hash each worker's own `env_seed` (train_ppo.py's existing per-rank
`args.seed + rank*9973` namespacing, reused verbatim, not reinvented)
together with the episode index into a distinct per-worker index sequence,
fed to `state_for_game` in place of the bare episode index. EVAL's own
direct `state_for_game(game_index)` calls (`eval_versus_fullgame.py`,
single-process, no rank) are completely untouched -- nothing on the eval
path ever calls this function."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import Any, Callable

from ..catalog import load_turtle_catalog, tier_for_turn
from ..engine import _rebuild_shop_for_roll, _roll_shop_slots, _sort_shop_by_tier

DEFAULT_OPENING_SEED_NAMESPACE = "opening_source_roll_v1"


def _empty_team_slots() -> list[dict[str, Any]]:
    """5 empty team slots, matching `fixtures/parity_cases/sample_case.json`'s
    own empty-slot shape exactly (no `sell_value`/`equipment_name` -- that is
    a DIFFERENT helper's convention elsewhere in this repo, not what the
    fixture `TrainingEnv`/`eval_versus_fullgame.py` have always shipped
    actually carries)."""
    return [
        {
            "slot_index": i,
            "pet_id": None,
            "attack": 0,
            "health": 0,
            "level": 1,
            "exp": 0,
            "equipment_id": None,
            "status_effects": [],
        }
        for i in range(5)
    ]


def _base_state_skeleton() -> dict[str, Any]:
    """Turn-1, gold-10, empty-team, empty-shop skeleton that `state_for_game`
    rolls a fresh shop into. `meta.oracle` (the sapai-main parity snapshot
    the hand-authored fixture also carries) is deliberately NOT included: it
    is read only by `api.py`'s dedicated engine-vs-sapai parity checker
    (`fixtures/parity_cases/` tooling / `python/tests/test_replay_player.py`
    -style tests), never by `TrainingEnv`/`eval_versus_fullgame.py`'s actual
    gameplay path -- so a state built here without it plays identically;
    only that one, unrelated parity tool would need a different fixture, and
    it keeps using the original one directly (unrelated to this module)."""
    return {
        "pack": "Turtle",
        "turn": 1,
        "gold": 10,
        "lives": 6,
        "trophies": 0,
        "team": _empty_team_slots(),
        "shop": [],
        "meta": {
            "game_mode": "versus",
            "versus": {"opponent_lives": 6},
            "seed_known": False,
            "source": "opening_source",
            "version": "v1",
        },
    }


def _rolled_shop_slots(seed: int) -> list[dict[str, Any]]:
    """A fresh tier-1 shop via the engine's OWN roll mechanism
    (`engine._roll_shop_slots`, the exact function a real turn-2+ advance /
    an explicit ROLL action uses -- not reimplemented here), seeded
    deterministically. THE mechanism `state_for_game` uses for every
    `game_index` -- there is no other path in this module."""
    state = _base_state_skeleton()
    state["meta"]["seed_known"] = True
    state["meta"]["seed"] = int(seed)
    notes: list[str] = []
    _rebuild_shop_for_roll(state, notes)  # builds the empty tier-1 (3 pet + 1 food) skeleton
    cat = load_turtle_catalog()
    tier = tier_for_turn(int(state["turn"]))
    rng = random.Random(int(seed))
    _roll_shop_slots(state, cat, rng, tier)
    _sort_shop_by_tier(state, cat)
    return state["shop"]


def _seed_for_index(namespace: str, game_index: int) -> int:
    """Deterministic per-(namespace, game_index) seed, same SHA1-digest
    construction `tools/eval_tempo_planner.py::_stable_shop_seed` already
    uses for its own hash-based shop seeding -- reused here for consistency
    rather than inventing a second convention."""
    digest = hashlib.sha1(f"{namespace}:{int(game_index)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _mark_seed(state: dict[str, Any], seed: int) -> None:
    """Set `meta.seed_known=True` + `meta.seed` on a state whose shop came
    from `_rolled_shop_slots`. Every real caller (`TrainingEnv.reset()`,
    `eval_versus_fullgame.py::_new_game_state`) overwrites both fields again
    before either is ever consulted for a real decision, so this has no
    gameplay effect -- it only keeps the state `state_for_game` hands back
    internally consistent on its own terms."""
    state["meta"]["seed_known"] = True
    state["meta"]["seed"] = int(seed)


def training_opening_index(env_seed: int, episode_index: int) -> int:
    """Stable hash of (`env_seed`, `episode_index`) -> an index to feed
    `VariedOpeningSource.state_for_game`, so parallel TRAINING envs with
    different `env_seed` (one per `--num-envs` worker) draw DECORRELATED
    per-episode opening sequences instead of sharing the bare
    `episode_index` every worker would otherwise count identically. See
    module docstring's "TRAINING DECORRELATION" section. Same SHA1-digest
    hash construction `_seed_for_index` already uses (reused for
    consistency, not reinvented), over a namespace unique to this use so the
    two hash streams never collide. Public (no leading underscore -- unlike
    this module's other private helpers) because `train/env.py` imports it
    directly.

    Unbounded: `state_for_game` has no pool to stay inside anymore -- every
    `game_index`, however large, just feeds a fresh seeded roll -- so unlike
    the pre-correction version of this function (which hashed into
    `[0, real_pool_size * factor)` to keep a share of training draws inside
    the real-opening pool), there is no span/modulus here: the raw 8-byte
    hash IS the index.
    """
    digest = hashlib.sha1(
        f"training_opening_index_v1:{int(env_seed)}:{int(episode_index)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class VariedOpeningSource:
    """Yields a turn-1 versus `initial_state` per `game_index`: a fresh,
    per-`game_index`-seeded engine roll (see module docstring). The SAME
    `game_index` always yields the SAME state (deterministic, reproducible
    across repeated calls -- `state_for_game` never consults any mutable
    state); different `game_index` values yield independently rolled shops.
    Construct via `build_varied_opening_source()`, not directly."""

    def __init__(self, *, seed_namespace: str = DEFAULT_OPENING_SEED_NAMESPACE) -> None:
        self._seed_namespace = str(seed_namespace)

    def provenance_for_game(self, game_index: int) -> str:
        return "seeded_roll"

    def state_for_game(self, game_index: int) -> dict[str, Any]:
        idx = int(game_index)
        seed = _seed_for_index(self._seed_namespace, idx)
        state = _base_state_skeleton()
        state["shop"] = _rolled_shop_slots(seed)
        _mark_seed(state, seed)
        state["meta"]["opening_source"] = {"kind": "seeded_roll", "game_index": idx}
        return state


class FixedOpeningSource:
    """Wraps a single fixture state so `state_for_game(i)` returns a fresh
    deep copy of the SAME state for every `i` -- byte-for-byte the OLD
    single-fixture behavior (minus this module entirely), kept behind an
    explicit flag at each call site for parity/debug (e.g.
    `eval_versus_fullgame.py --opening-mode fixed`). Duck-type compatible
    with `VariedOpeningSource` (same two public methods) so callers never
    need to branch on which one they were given."""

    def __init__(self, initial_state: dict[str, Any]) -> None:
        self._initial_state = copy.deepcopy(initial_state)

    def provenance_for_game(self, game_index: int) -> str:
        return "fixed_single_fixture"

    def state_for_game(self, game_index: int) -> dict[str, Any]:
        return copy.deepcopy(self._initial_state)


def fixed_opening_source(initial_state: dict[str, Any]) -> FixedOpeningSource:
    return FixedOpeningSource(initial_state)


def build_varied_opening_source(
    *,
    seed_namespace: str = DEFAULT_OPENING_SEED_NAMESPACE,
    log: Callable[[str], None] | None = None,
) -> VariedOpeningSource:
    """Build the opening source every `--opening-mode varied` call site uses
    (`train/runtime.py::build_training_env`, `tools/eval_versus_fullgame.py::
    main`, `tools/gen_distill_dataset.py::main`): every `game_index` gets a
    fresh, deterministically-seeded engine roll (see module docstring).
    Takes no file paths and cannot fail -- there is no manifest or raw
    replay cache to be missing anymore, so (unlike the pre-correction
    version of this function) there is no separate portability-fallback
    wrapper needed; this IS the only builder, safe to call on any host."""
    logger = log if log is not None else (lambda msg: print(msg, flush=True))
    logger(f"opening_source_built:mode=varied:kind=seeded_roll:seed_namespace={seed_namespace}")
    return VariedOpeningSource(seed_namespace=seed_namespace)
