"""Shared runtime helpers for Step 3 PPO scripts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

from ..opponents.chain_snapshot import DEFAULT_CHAIN_SNAPSHOT, ChainSnapshotSource
from .env import TrainingEnv, load_initial_state_from_fixture
from .opening_source import build_varied_opening_source
from .opponents import (
    CurriculumOpponentProvider,
    OpponentProvider,
    RandomOpponentProvider,
    ReplayDBOpponentProvider,
    ReplaySnapshotProvider,
    SelfPlayPoolProvider,
)

# Default held-out split for TRAINING rollouts (exp09 W5 P0): the eval/
# held-out reference is meant to score against a FIXED, disjoint "test"
# split (see `sap_ppo.opponents.chain_snapshot` module docstring) -- so
# training's own default must not silently draw from the same pool.
DEFAULT_TRAIN_OPPONENT_SPLIT = "train"

# exp09 W5 P0 Fix C (codex finding 3): the periodic/standalone EVAL
# reference must default to a FIXED, held-out split, distinct from
# `DEFAULT_TRAIN_OPPONENT_SPLIT` above, so a plain launch scores itself
# against opponents training never sees, without the operator having to
# remember to pass a separate flag. Used by `train_ppo.py`'s own built-in
# periodic eval (a SEPARATE split from its training-rollout split) and by
# `eval_ppo.py`'s single `--opponent-split` default.
DEFAULT_EVAL_OPPONENT_SPLIT = "test"

# exp09 W5 P0 Fix A (codex finding 1): episode-length turn-cap defaults.
# Named constants -- previously bare literals inline in `train_ppo.py`'s
# argparse handling (`if args.max_turn is None: args.max_turn = 30 if ...
# else 15`) -- given ONE importable home here so every W5 entrypoint
# (`train_ppo.py`, `eval_ppo.py`, `init_ppo_model.py`,
# `train_bc_warmstart.py`) and `tools/frame_parity_harness.py`'s
# `encoder_horizon` regression dimension share the SAME resolution instead
# of each re-deriving (or hardcoding) "30 for chain_snapshot, else 15".
DEFAULT_MAX_TURN_STANDARD = 15
DEFAULT_MAX_TURN_CHAIN_SNAPSHOT = 30

# exp09 W5 P0 Fix A (codex finding 1): the V4 encoder's OWN turn-
# normalization horizon for chain-PPO is FIXED at 15 -- decoupled from the
# episode-length cap above (30) -- see `resolve_encoder_max_turn`'s
# docstring for why these must not be the same knob.
CHAIN_SNAPSHOT_ENCODER_MAX_TURN = 15


def resolve_max_turn(*, opponent_mode: str, max_turn_arg: int | None) -> int:
    """Resolve the EPISODE-LENGTH turn cap (`TrainingEnv`'s own `max_turn`,
    i.e. what `TrainingEnv._is_done`'s turn check compares against) for a
    training/eval launch.

    Extracted out of what used to be inline argparse handling in
    `train_ppo.py`'s `main()` (a bare `if args.max_turn is None: args.max_turn
    = 30 if ... else 15`) so this resolution has ONE importable home --
    `tools/frame_parity_harness.py`'s `encoder_horizon` dimension needs to
    replicate what a REAL chain-PPO launch resolves, and importing this
    function keeps that check honest (reflecting whatever this file actually
    does) instead of duplicating "30" as a second, driftable literal in the
    harness.

    `--opponent-mode chain_snapshot` (exp09 W5 versus-mode training) defaults
    to `DEFAULT_MAX_TURN_CHAIN_SNAPSHOT` (30) to match the eval frame's
    episode-length allowance (`tools/eval_versus_fullgame.py`'s
    `DEFAULT_MAX_TURN`) so a training episode is not cut short relative to
    eval; every other mode keeps the legacy one-turn/arena default of
    `DEFAULT_MAX_TURN_STANDARD` (15). An explicit `max_turn_arg` (a real
    `--max-turn` CLI value) always wins over either default.
    """
    if max_turn_arg is not None:
        return int(max_turn_arg)
    return DEFAULT_MAX_TURN_CHAIN_SNAPSHOT if str(opponent_mode) == "chain_snapshot" else DEFAULT_MAX_TURN_STANDARD


def resolve_encoder_max_turn(*, opponent_mode: str, episode_max_turn: int) -> int:
    """Resolve the V4 ENCODER's turn-normalization horizon (`_norm(turn,
    self.max_turn)` / `_turns_to_next_tier`, `train/observation.py`) for a
    training/eval launch -- DECOUPLED from the episode-length cap above
    (exp09 W5 P0 Fix A, cross-model review finding 1).

    Before this fix, every entrypoint fed the SAME resolved `max_turn` to
    both the episode-length cap AND the encoder's `max_turn=` -- for
    `--opponent-mode chain_snapshot` that meant the encoder was built at
    max_turn=30. But the eval frame's `BcRecommender` (`tools/
    bc_recommender.py`, `DEFAULT_MAX_TURN=15`) -- and, more importantly, the
    frozen `flat_v2` warm-start / KL anchor checkpoint chain-PPO is built
    from -- were both trained/built with max_turn=15. A turn-30 encoder
    normalizes `_norm(turn, 30)` / `_norm(turns_to_next, 30)` to HALF the
    value a turn-15 encoder would for the identical raw turn number, so
    every state the chain-PPO policy sees would be silently off-
    distribution relative to what its own warm-start/anchor was trained on
    -- see `tools/frame_parity_harness.py`'s `encoder_horizon` dimension,
    the regression gate for this fix.

    For `--opponent-mode chain_snapshot` this is therefore a FIXED
    `CHAIN_SNAPSHOT_ENCODER_MAX_TURN` (15), regardless of `episode_max_turn`
    (still resolved to 30 by `resolve_max_turn`, unaffected -- the episode
    is allowed to run longer than the encoder's own horizon constant; turns
    past 15 simply clamp their turn-normalized features at 1.0, exactly as
    `BcRecommender`'s frame already does for any post-turn-15 state it ever
    encodes). Every other mode is unaffected: `episode_max_turn` is
    returned as-is, the same value the encoder always used before this fix.
    """
    if str(opponent_mode) == "chain_snapshot":
        return CHAIN_SNAPSHOT_ENCODER_MAX_TURN
    return int(episode_max_turn)


def build_opponent_provider(
    *,
    mode: str,
    seed: int,
    snapshot_path: Path | None = None,
    chain_snapshot_path: Path | None = None,
    opponent_split: str | None = None,
    chain_opponent_pack: str | None = None,
    chain_opponent_rank_min: int | None = None,
    chain_opponent_rank_max: int | None = None,
    self_play_path: Path | None = None,
    database_url: str | None = None,
    curriculum_stage_a_episodes: int = 10_000,
    curriculum_stage_b_episodes: int = 30_000,
) -> OpponentProvider:
    mode_norm = str(mode or "snapshot").strip().lower()
    if mode_norm == "db":
        return ReplayDBOpponentProvider(database_url=database_url)
    if mode_norm == "snapshot":
        if snapshot_path is None:
            raise ValueError("snapshot_mode_requires_snapshot_path")
        return ReplaySnapshotProvider(snapshot_path=Path(snapshot_path), seed=int(seed))
    if mode_norm == "chain_snapshot":
        # exp09 W5 P0: the SAME chain-snapshot opponent source the W4c eval
        # frame uses (`tools/eval_versus_fullgame.py`), instead of the
        # `snapshot` mode's DIFFERENT, schema-incompatible
        # `replay_snapshot_v1.json` (see `frame_parity_harness.py`'s
        # "opponent" finding). `ChainSnapshotSource.sample()` satisfies
        # `OpponentProvider` directly -- no adapter needed.
        path = Path(chain_snapshot_path) if chain_snapshot_path is not None else Path(DEFAULT_CHAIN_SNAPSHOT)
        return ChainSnapshotSource(
            path,
            seed=int(seed),
            split=opponent_split,
            opponent_pack=chain_opponent_pack,
            # exp09 W5.3: low-rank curriculum pool pass-through (see
            # `ChainSnapshotSource`'s own docstring for the exclusion rule
            # applied when either bound is set).
            opponent_rank_min=chain_opponent_rank_min,
            opponent_rank_max=chain_opponent_rank_max,
        )
    if mode_norm == "random":
        return RandomOpponentProvider(seed=int(seed))
    if mode_norm == "self_play":
        if self_play_path is None:
            raise ValueError("self_play_mode_requires_self_play_path")
        return SelfPlayPoolProvider(pool_path=Path(self_play_path), seed=int(seed))
    if mode_norm == "curriculum":
        providers: dict[str, OpponentProvider] = {
            "random": RandomOpponentProvider(seed=int(seed)),
        }
        if snapshot_path is not None:
            providers["snapshot"] = ReplaySnapshotProvider(snapshot_path=Path(snapshot_path), seed=int(seed))
        if self_play_path is not None:
            providers["self_play"] = SelfPlayPoolProvider(pool_path=Path(self_play_path), seed=int(seed))

        if "snapshot" not in providers and "self_play" not in providers:
            # Fallback curriculum degenerates to random if no offline pools provided.
            providers["snapshot"] = RandomOpponentProvider(seed=int(seed) + 1)

        stages = [
            {
                "name": "stage-a-random",
                "until_episode": int(curriculum_stage_a_episodes),
                "weights": {"random": 1.0},
            },
            {
                "name": "stage-b-random-snapshot",
                "until_episode": int(curriculum_stage_b_episodes),
                "weights": {"random": 0.3, "snapshot": 0.7},
            },
            {
                "name": "stage-c-self-play",
                "weights": {"snapshot": 0.2, "self_play": 0.8},
            },
        ]
        return CurriculumOpponentProvider(providers=providers, stages=stages, seed=int(seed))
    raise ValueError(f"unknown_opponent_mode:{mode}")


def build_training_env(
    *,
    fixture_path: Path,
    opponent_provider: OpponentProvider,
    max_turn: int | None = None,
    one_turn_mode: bool = False,
    opening_mode: str = "varied",
    opening_log: Callable[[str], None] | None = None,
    opening_env_seed: int | None = None,
) -> TrainingEnv:
    """Build a `TrainingEnv` from a fixture path + opponent provider.

    `opening_mode` (full-game frame fix, train/opening_source.py; default
    "varied" -- this is what makes full-game TRAINING (`train_ppo.py`, which
    calls this same helper and does not override the default) and eval
    (`eval_ppo.py`) both start each versus, non-one-turn episode from a
    fresh, seeded engine roll instead of always the same hardcoded fixture):
      - "varied": build a `VariedOpeningSource` (a per-`game_index`-seeded
        engine roll, no file/pool dependency -- see opening_source.py's
        module docstring) and wire it into the returned `TrainingEnv`. Only
        actually built when the fixture's own `meta.game_mode` is "versus"
        and `one_turn_mode` is False -- the exact condition
        `TrainingEnv.reset()` gates the substitution on, so building it for
        an arena/one-turn caller would be a wasted allocation for a source
        that would never be consulted.
      - "fixed": the OLD behavior -- every episode re-copies the one fixture
        state, byte-for-byte, kept for parity/debug. Implemented as passing
        no `opening_source` at all (`TrainingEnv`'s own default), not a
        `FixedOpeningSource` wrapper -- both are behaviorally identical for
        this call site, but this avoids paying for the wrapper object.
    `opening_env_seed` (review finding 1, default None): forwarded straight
    to `TrainingEnv(opening_env_seed=...)` -- see that class's docstring.
    `train_ppo.py`'s parallel-rollout `_make_env(rank)` factory is the one
    real caller that passes a non-None value (its per-rank `env_seed`); every
    other call site (eval_ppo.py, train_ppo.py's own periodic-eval env,
    train_ppo.py's throwaway `reference_core_env`) leaves this at the default
    None, keeping the bare-`episode_index` path unchanged.
    """
    initial_state = load_initial_state_from_fixture(Path(fixture_path))

    mode_norm = str(opening_mode or "varied").strip().lower()
    if mode_norm not in ("varied", "fixed"):
        raise ValueError(f"unknown_opening_mode:{opening_mode}:allowed=varied,fixed")

    opening_source = None
    game_mode = str(initial_state.get("meta", {}).get("game_mode") or "arena").strip().lower()
    if mode_norm == "varied" and game_mode == "versus" and not one_turn_mode:
        opening_source = build_varied_opening_source(log=opening_log)

    return TrainingEnv(
        copy.deepcopy(initial_state),
        opponent_provider=opponent_provider,
        max_turn=max_turn,
        one_turn_mode=bool(one_turn_mode),
        opening_source=opening_source,
        opening_env_seed=opening_env_seed,
    )
