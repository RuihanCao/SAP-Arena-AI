"""Policy evaluation against recorded opponent boards.

Game rules and opponent sampling are separate settings. Arena sampling draws a
board at each turn from the selected pool; chain sampling follows a recorded
opponent chain. Segmented-honest execution searches with independent simulation
seeds, commits actions against the actual state, and re-searches at structural
chance nodes. The public evaluation entry point selects its published defaults."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


from ..api import set_skip_imagined_validation, step as engine_step
from ..end_turn import resolve_end_turn_with_sampled_battle
from ..opponents.chain_snapshot import (
    CHAIN_SNAPSHOT_VERSION,
    DEFAULT_CHAIN_SNAPSHOT,
    DEFAULT_LONG_MIN,
    DEFAULT_SEED,
    SPLIT_NAMES,
    ChainSnapshotSource,
)
from ..oracles.sap_calc_battle_oracle import battle_worker_stats
from ..train.env import TrainingEnv, _set_last_opponent_team, load_initial_state_from_fixture
from ..versus_lives import (
    ARENA_MAX_LIVES,
    ARENA_START_LIVES,
    GAME_MODE_ARENA,
    GAME_MODE_VERSUS,
    MAX_TROPHIES,
    VERSUS_START_LIVES,
)
from ..train.opening_source import (
    FixedOpeningSource,
    VariedOpeningSource,
    build_varied_opening_source,
    fixed_opening_source,
)
from ..visited_guard import state_signature
from .honest_frame import (
    DEFAULT_STOCHASTIC_SAMPLES,
    DEFAULT_TURN_MODE,
    MAX_SEGMENTS_PER_TURN,
    PROPOSAL_SAMPLE_R,
    TURN_MODE_SEGMENTED_HONEST,
    TURN_MODES,
    imagination_seed,
    imagined_clone,
    is_honest,
    normalize_turn_mode,
    read_engine_seed,
)
from .cat_trigger_audit import empty_counts as _cat_audit_empty, sum_counts as _cat_audit_sum
from .segmented_turn import bc_decide, run_segmented_turn
from .bc_recommender import DECODE_MODES as BC_DECODE_MODES
from .bc_recommender import DEFAULT_DECODE_MODE as BC_DEFAULT_DECODE_MODE
from .bc_recommender import DEFAULT_SAMPLE_SEED as BC_DEFAULT_SAMPLE_SEED
from .bc_recommender import DEFAULT_SAMPLE_TEMPERATURE as BC_DEFAULT_SAMPLE_TEMPERATURE
from .bc_recommender import BcRecommender
from .cluster_ci import cluster_ci
from .search_recommender import COMPLETION_AGG_MAX as SEARCH_COMPLETION_AGG_MAX
from .search_recommender import COMPLETION_AGGREGATES as SEARCH_COMPLETION_AGGREGATES
from .search_recommender import COMPLETION_BC_GREEDY as SEARCH_COMPLETION_BC_GREEDY
from .search_recommender import COMPLETION_POLICIES as SEARCH_COMPLETION_POLICIES
from .search_recommender import DEFAULT_KSIM as SEARCH_DEFAULT_KSIM
from .search_recommender import DEFAULT_N_CANDIDATES as SEARCH_DEFAULT_N_CANDIDATES
from .search_recommender import DEFAULT_ROLLOUT_KSIM as SEARCH_DEFAULT_ROLLOUT_KSIM
from .search_recommender import DEFAULT_ROLLOUT_OPPONENT_MODE as SEARCH_DEFAULT_ROLLOUT_OPPONENT_MODE
from .search_recommender import DEFAULT_ROLLOUT_REPEATS as SEARCH_DEFAULT_ROLLOUT_REPEATS
from .search_recommender import DEFAULT_ROLLOUT_SHORTLIST as SEARCH_DEFAULT_ROLLOUT_SHORTLIST
from .search_recommender import DEFAULT_SCORING as SEARCH_DEFAULT_SCORING
from .search_recommender import ROLLOUT_OPPONENT_MODES as SEARCH_ROLLOUT_OPPONENT_MODES
from .search_recommender import SCORING_MODES as SEARCH_SCORING_MODES
from .search_recommender import SCORING_ROLLOUT as SEARCH_SCORING_ROLLOUT
from .search_recommender import SCORING_VGAME as SEARCH_SCORING_VGAME
from .search_recommender import SearchRecommender

from .._artifact_defaults import BC_CHECKPOINT as DEFAULT_BC_CHECKPOINT  # noqa: E402
DEFAULT_MAX_TURN = 30


DEFAULT_TEACHER_ROLLOUTS = 8


VGAME_DEFAULT_BLEND = 0.0
VGAME_DEFAULT_PESSIMISM = 0.0

# Turn-1 versus fixture (verified: turn=1, lives=6, meta.game_mode="versus",
# meta.versus.opponent_lives=6 -- see fixtures/parity_cases/sample_case.json).
# Relative to repo root, matching this repo's established convention of
# repo-root-relative default paths for fixtures/manifests (e.g.
# eval_tempo_planner.py's `--case-manifest` default).
FIXTURE_PATH = Path("fixtures/parity_cases/sample_case.json")


END_REASON_PLAYER_LIVES_0 = "player_lives_0"
END_REASON_OPPONENT_LIVES_0 = "opponent_lives_0"
END_REASON_TURN_CAP = "turn_cap"
END_REASON_DECODE_FAILED = "decode_failed"
END_REASON_END_TURN_FAILED = "end_turn_failed"
END_REASON_CHAIN_REPLAY_DIVERGED = "chain_replay_diverged"


END_REASON_TROPHIES_10 = "trophies_10"


END_REASON_OPPONENT_POOL_EXHAUSTED = "opponent_pool_exhausted"


COMPLETED_END_REASONS = frozenset(
    {
        END_REASON_PLAYER_LIVES_0,
        END_REASON_OPPONENT_LIVES_0,
        END_REASON_TURN_CAP,
        END_REASON_TROPHIES_10,
        END_REASON_OPPONENT_POOL_EXHAUSTED,
    }
)

# The `resolve_end_turn_with_sampled_battle` error suffix that means "the
# snapshot pool holds no candidate at this turn" (`chain_snapshot.py`'s
# `sample_random_with_rng`). Matched as a substring rather than a prefix
# because `end_turn.py` wraps it in one of two `opponent_sampling_failed`
# prefixes and `play_out_game` wraps THAT in `end_turn_failed:`.
POOL_EXHAUSTED_ERROR_TOKEN = "no_snapshot_opponent_for_turn:"


OPPONENT_MODE_CHAIN = "chain"
OPPONENT_MODE_ARENA = "arena"
OPPONENT_MODES: tuple[str, ...] = (OPPONENT_MODE_CHAIN, OPPONENT_MODE_ARENA)

# What a run's `num_fallbacks`/`fallback_turns`/`fallback.rate_games` MEAN
# under each ruler. Emitted next to those counters (per game and in the
# aggregate) because under `arena` they are structurally 0 -- there is no
# followed chain that could run out -- and a bare 0.000 would otherwise read
# as "this ruler happened to need no fallbacks", a very different claim.
FALLBACK_SEMANTICS_CHAIN = "followed_chain_exhausted_random_resample"
FALLBACK_SEMANTICS_ARENA = "n/a_arena"


def _fallback_semantics(opponent_mode: str) -> str:
    return FALLBACK_SEMANTICS_ARENA if opponent_mode == OPPONENT_MODE_ARENA else FALLBACK_SEMANTICS_CHAIN


GAME_RULES_VERSUS = "versus"
GAME_RULES_ARENA = "arena"
GAME_RULES: tuple[str, ...] = (GAME_RULES_VERSUS, GAME_RULES_ARENA)


ARENA_RACE_TROPHIES_MAPPED = "trophies_mapped"
ARENA_RACE_CONST6 = "const6"
ARENA_RACE_CONVENTIONS: tuple[str, ...] = (ARENA_RACE_TROPHIES_MAPPED, ARENA_RACE_CONST6)
DEFAULT_ARENA_RACE_CONVENTION = ARENA_RACE_TROPHIES_MAPPED

# What a report's trophy block MEANS under each ruler, emitted next to it for
# the same reason `FALLBACK_SEMANTICS_*` is: under versus rules there is no
# trophy race at all, and the block is None rather than a pile of zeros that
# would read as "this policy earned no trophies".
TROPHY_SEMANTICS_ARENA = "arena_10_trophies_completes_the_run"
TROPHY_SEMANTICS_VERSUS = "n/a_versus"


def _normalize_game_rules(game_rules: str | None) -> str:
    rules = str(game_rules or GAME_RULES_VERSUS).strip().lower()
    if rules not in GAME_RULES:
        raise ValueError(f"unknown_game_rules:{game_rules}:valid={list(GAME_RULES)}")
    return rules


def _game_rules_refusal(
    *,
    game_rules: str | None,
    opponent_mode: str | None,
    scoring: str | None,
) -> str | None:
    """The two combinations that would silently produce wrong numbers rather
    than fail -- returned as a message, or None when the combination is fine.

    `scoring` is the leaf-scoring mode of the recommender that will play
    (`SearchRecommender.scoring`; None for anything without one)."""
    if _normalize_game_rules(game_rules) != GAME_RULES_ARENA:
        return None
    if str(opponent_mode or OPPONENT_MODE_CHAIN).strip().lower() != OPPONENT_MODE_ARENA:
        return (
            "--game-rules arena requires --opponent-mode arena (game_rules=/opponent_mode= "
            "on the API): end_turn.py only follows a chain's forced pid under versus "
            "rules, so an arena-rules game cannot follow one -- it would silently become "
            "a random-opponent run labelled 'chain'."
        )
    if str(scoring or "").strip().lower() == SEARCH_SCORING_ROLLOUT:
        return (
            "--game-rules arena does not support --search-scoring rollout yet: "
            "SearchRecommender's rollout continuations resolve under VERSUS rules and score "
            "with _versus_win(lives, opp_lives), which has no meaning when there is no "
            "opponent life bar -- every continuation would read as an instant win."
        )
    return None


def _turn_mode_refusal(*, turn_mode: str | None, scoring: str | None) -> str | None:
    """A1 section 3 defines expectation scoring -- score a prefix that ends at a
    chance node as the mean of k imagined completions -- for the LEARNED
    LEAF only. A myopic or rollout leaf under this frame would still rank
    candidates by whatever the single proposal-stream roll happened to
    produce, and since each candidate's roll is no longer shared with play,
    the argmax would systematically pick the candidate that got lucky: a
    number that OVERSTATES the value of rolling, produced silently.

    So the combination is refused, exactly like the two `_game_rules_refusal`
    cases. `--recommender bc` (no candidate ranking at all, therefore no
    lucky-roll selection) and `--recommender llm` pass: `scoring` is None for
    anything without a leaf.

    Same two faces / one rule structure as `_game_rules_refusal`:
    `_check_turn_mode_compatibility` (CLI, `SystemExit`) and
    `_assert_game_rules_supported` (API, `ValueError`)."""
    if not is_honest(turn_mode):
        return None
    leaf = str(scoring or "").strip().lower()
    if leaf and leaf != SEARCH_SCORING_VGAME:
        return (
            f"--turn-mode {TURN_MODE_SEGMENTED_HONEST} does not support --search-scoring "
            f"{leaf}: expectation scoring over the k "
            f"imagined completions of a stochastic prefix is implemented for the "
            f"{SEARCH_SCORING_VGAME} leaf only. A {leaf} leaf under this frame would rank "
            "candidates on one imagined roll each and systematically pick whichever "
            "candidate rolled well, overstating the value of rolling."
        )
    return None


def _turn_mode_agreement_refusal(
    *, recommender_turn_mode: str | None, driver_turn_mode: str | None
) -> str | None:
    """A1's frame is not one setting, it is two halves that only mean anything
    together. The driver owns segmentation (commit op by op, stop at the first
    realized `stochastic_reason`, re-search) and owns the imagined clone it
    hands over each segment; the recommender owns prefix dedup and expectation
    scoring over the k completions. Building a `SearchRecommender` with one
    `turn_mode` and running the driver with the other yields a run that is
    NEITHER frame, and until this check existed it did so silently:

    - determinized recommender under an honest driver: candidates are ranked
      as WHOLE turns, one imagined roll each -- the pick-the-lucky-roll bias
      A1 exists to remove -- while the driver commits only the prefix, so the
      post-boundary ops the winner was chosen FOR are thrown away and never
      re-searched. The report still says `turn_mode: segmented-honest`.
    - honest recommender under a determinized driver: `chain_preview` comes
      back TRUNCATED to the prefix (`_prefix_chain`), and the determinized
      driver commits one chain and resolves the turn -- so the agent stops
      playing mid-turn on every turn that crosses a chance node, with no
      re-search and no report field saying so.

    Refused at the API face, next to the rules and leaf checks, for the same
    reason they are: `run_versus_eval`/`play_one_game` are reusable entry
    points and a direct caller must not be able to assemble this by hand.
    Anything without a `turn_mode` attribute (a plain `BcRecommender`, an
    `LlmRecommender`) is unaffected -- it has no imagination of its own to
    disagree with, which is exactly why `--recommender bc` is honest under the
    honest driver on segmentation alone.

    `main()` passes `args.turn_mode` to both halves, so the CLI cannot build a
    mismatch and there is deliberately no second CLI face here."""
    if recommender_turn_mode is None:
        return None
    built = normalize_turn_mode(recommender_turn_mode)
    driving = normalize_turn_mode(driver_turn_mode)
    if built == driving:
        return None
    return (
        f"turn_mode_mismatch:recommender={built}:driver={driving}: the recommender was CONSTRUCTED for one frame and the driver "
        "is RUNNING the other. Segmentation lives in the driver and expectation "
        "scoring lives in the recommender, so a mismatched pair is neither frame -- "
        "pass the same turn_mode to both."
    )


def _assert_game_rules_supported(
    recommender: Any,
    *,
    game_rules: str | None,
    opponent_mode: str | None,
    turn_mode: str | None = DEFAULT_TURN_MODE,
) -> None:
    """API face of `_game_rules_refusal` + `_turn_mode_refusal` +
    `_turn_mode_agreement_refusal` -- raises `ValueError` (this module's
    established library-level error, as in `unknown_opponent_mode`).

    The scoring mode and the frame are read off the recommender the same
    duck-typed way `play_one_game` reads `set_decision_context`: a plain
    `BcRecommender` has neither `.scoring` nor `.turn_mode`, and is unaffected
    by both checks."""
    scoring = getattr(recommender, "scoring", None)
    message = _game_rules_refusal(
        game_rules=game_rules,
        opponent_mode=opponent_mode,
        scoring=scoring,
    )
    if message is None:
        message = _turn_mode_refusal(turn_mode=turn_mode, scoring=scoring)
    if message is None:
        message = _turn_mode_agreement_refusal(
            recommender_turn_mode=getattr(recommender, "turn_mode", None),
            driver_turn_mode=turn_mode,
        )
    if message is not None:
        raise ValueError(message)


def _normalize_arena_race_convention(convention: str | None) -> str:
    name = str(convention or DEFAULT_ARENA_RACE_CONVENTION).strip().lower()
    if name not in ARENA_RACE_CONVENTIONS:
        raise ValueError(
            f"unknown_arena_race_convention:{convention}:valid={list(ARENA_RACE_CONVENTIONS)}"
        )
    return name


def _arena_race_opp_lives(trophies: int, convention: str) -> int:
    """Pure arithmetic over the trophy count, so the convention is one
    expression in one place -- the driver, the recorders and (through the
    state field below) the learned leaf all read the same number."""
    if convention == ARENA_RACE_CONST6:
        return VERSUS_START_LIVES
    # `trophies` is capped at MAX_TROPHIES by the lives helper, so this is
    # already non-negative; the max() is a guard, not a rule.
    return max(0, min(VERSUS_START_LIVES, MAX_TROPHIES - int(trophies)))


def _apply_arena_race_context(state: dict[str, Any], *, convention: str) -> int:
    """Write this turn's D2 race convention onto `meta.versus.opponent_lives`
    and return it.

    That field is where `SearchRecommender._race_scalars` (and therefore the
    V bypass) reads `opp_lives` from, and where this driver reads
    `pre_turn_opp_lives` for the afterstate/teacher recorders -- so writing
    the convention THERE is what makes "the number V scored with" and "the
    number the dataset stores" the same object, instead of two definitions
    that can drift.

    Under arena rules the field is a RACE FEATURE, never a life total:
    `end_turn.py`'s arena branch neither reads nor writes it (only its
    `mode == "versus"` branches touch `meta.versus`), so nothing downstream
    can mistake it for one.
    """
    trophies = int(state.get("trophies", 0) or 0)
    opp_lives = _arena_race_opp_lives(trophies, convention)
    meta = state.setdefault("meta", {})
    versus_meta = meta.setdefault("versus", {})
    versus_meta["opponent_lives"] = int(opp_lives)
    return int(opp_lives)


def _is_completed_end_reason(end_reason: str | None) -> bool:
    """True iff `end_reason` is a genuine completed-game verdict (see
    `COMPLETED_END_REASONS`), not an infrastructure failure."""
    return str(end_reason) in COMPLETED_END_REASONS


def _decode_failure_reason(rec: dict[str, Any] | None, stop_reason: str | None) -> str:
    """WHY the recommender refused to act, as the suffix of `decode_failed:*`.

    `error` is read FIRST and `diagnostics.stop_reason` is the fallback.
    `BcRecommender` sets `error` to exactly the stop reason on every failure,
    so the BC path yields that anticipated form either way; but
    `SearchRecommender`'s hard-failure shape carries
    `search_greedy_call_failed:<exc>` in `error` and an EMPTY diagnostics
    block, so reading the stop reason first would throw away the only thing
    that says what happened. Collapsed to one line and truncated, because
    this string ends up in `end_reason_distribution` keys."""
    for value in ((rec or {}).get("error"), stop_reason):
        text = " ".join(str(value or "").split())
        if text:
            return text[:120]
    return "unknown"


def _is_pool_exhausted_end_reason(end_reason: str | None) -> bool:
    """True iff this `end_turn_failed:*` reason is "the arena pool holds no
    candidate at that turn" rather than a real infrastructure failure."""
    return POOL_EXHAUSTED_ERROR_TOKEN in str(end_reason or "")


def effective_arena_max_turn(
    max_turn: int,
    arena_source: "ChainSnapshotSource | None",
    *,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
) -> int:
    """The turn cap a game against `arena_source` can actually be played to.

    Only `opponent_mode="arena"` is clamped. Under chain mode the opponent
    comes from the FOLLOWED chain and running out is the normal, designed
    event that the random fallback exists to absorb, so a pool depth there is
    not a turn cap. `arena_source is None` (the chain-mode default) and an
    empty pool both return `max_turn` unchanged."""
    if opponent_mode != OPPONENT_MODE_ARENA or arena_source is None:
        return int(max_turn)
    depth = arena_source.max_turn_with_candidates
    if depth is None:
        return int(max_turn)
    return min(int(max_turn), int(depth))


def _versus_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("meta")
    return meta.get("versus", {}) if isinstance(meta, dict) and isinstance(meta.get("versus"), dict) else {}


def _assert_fixture_shape(state: dict[str, Any]) -> None:
    """Verify the turn-1 versus fixture actually matches what this driver requires."""
    versus = _versus_meta(state)
    meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
    problems = []
    if int(state.get("turn", -1)) != 1:
        problems.append(f"turn={state.get('turn')}")
    if int(state.get("lives", -1)) != 6:
        problems.append(f"lives={state.get('lives')}")
    if str(meta.get("game_mode", "")).strip().lower() != "versus":
        problems.append(f"game_mode={meta.get('game_mode')}")
    if int(versus.get("opponent_lives", -1)) != 6:
        problems.append(f"opponent_lives={versus.get('opponent_lives')}")
    if problems:
        raise ValueError(f"fixture_shape_mismatch:{FIXTURE_PATH}:{','.join(problems)}")


def _new_game_state(
    fixture_initial_state: dict[str, Any],
    followed_pid: str,
    *,
    engine_seed: int,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
) -> dict[str, Any]:
    """Deep-copy the fixture and wire up this game's followed opponent + versus life state.

    Also sets `meta.seed_known=True` + a concrete `meta.seed`. The fixture ships
    `seed_known=False`, which makes `engine._rng_from_state` seed every RNG draw
    (ROLL's shop regen, food-target selection, ...) from OS entropy instead of
    from the state. That is harmless for a single one-shot decode, but this
    driver's per-turn loop calls `engine_step` TWICE over the same chain of
    non-END_TURN ops: once implicitly inside `BcRecommender.recommend()`'s own
    internal decode walk, and once explicitly here to replay `chain_preview`
    and reconstruct the pre-END_TURN board (see module docstring). With
    `seed_known=False` those two walks draw independent OS entropy at any
    ROLL, so they can land on different shops -- and a later shop-position
    action in the SAME chain (BUY_PET/BUY_COMBINE/BUY_FOOD/FREEZE/UNFREEZE
    after the ROLL) then replays against the wrong shop and can turn illegal.
    Verified directly: an un-seeded game hit exactly this
    (`chain_replay_diverged:BUY_COMBINE:illegal_on_replay` on turn 3, chain
    `FREEZE, BUY_COMBINE, BUY_FOOD, BUY_COMBINE, ROLL, FREEZE, FREEZE`); the
    identical game with `seed_known=True` replayed all 9 turns (every one
    ROLL-then-shop-op) with zero divergence. `eval_tempo_planner.py` hits the
    same requirement for its own chain-replay (`_simulate_state_after_chain`)
    and already sets `seed_known=True` + a concrete seed for exactly this
    reason (see its `_run_one_case`/`_run_one_start_case` `base_state`
    construction) -- mirrored here rather than reinvented.
    `engine._reseed_meta` chains the seed forward deterministically turn to
    turn once `seed_known` is True, so setting it once at game start covers
    the whole game."""
    state = copy.deepcopy(fixture_initial_state)
    state["turn"] = 1
    state["lives"] = 6
    meta = state.setdefault("meta", {})
    meta["game_mode"] = "versus"
    meta["seed_known"] = True
    meta["seed"] = int(engine_seed)
    versus = meta.setdefault("versus", {})
    versus["opponent_lives"] = 6
    versus["current_opponent_participation_id"] = followed_pid
    if _normalize_game_rules(game_rules) == GAME_RULES_ARENA:
        state["lives"] = ARENA_START_LIVES
        state["trophies"] = 0
        meta["game_mode"] = GAME_MODE_ARENA
        # WHICH arena. `game_mode` alone is ambiguous: the RL training env's
        # episodes are also `game_mode="arena"` and legitimately start on 6
        # lives with a 7-trophy target (`train/env.py::_is_done`), while THIS
        # ruler starts on 5 with a 10-trophy target. `schemas/state_v1.json`
        # needs to tell them apart to keep its turn-1 start-of-game invariant
        # exact (turn 1 -> exactly 5 lives here, exactly 6 everywhere else)
        # rather than weakened to "5 or 6". Without it every arena game died
        # on turn 1 with `decode_failed:legal_mask_failed` -- `legal_actions`
        # validates the state. Written under arena rules ONLY, so a versus
        # state is byte-identical to before this flag existed.
        meta["game_rules"] = GAME_RULES_ARENA
        _apply_arena_race_context(
            state, convention=_normalize_arena_race_convention(arena_race_convention)
        )
    return state


def _search_width_counts(turn_entry: dict[str, Any], key: str) -> list[int]:
    """Under the determinized frame a turn IS one decision, and this returns the
    single pre-A1 value (or nothing, for a recommender that reports no
    counts) -- byte-identical."""
    segments = turn_entry.get("segments")
    if segments is None:
        if turn_entry.get("search_used") and turn_entry.get(key) is not None:
            return [int(turn_entry[key])]
        return []
    return [
        int(segment[key])
        for segment in segments
        if segment.get("search_used") and segment.get(key) is not None
    ]


def _turn_record(
    *,
    turn: int,
    chain_types: list[str],
    stop_reason: str,
    outcome: str | None,
    opponent_pets: Any,
    lives: int,
    opp_lives: int,
    detail: dict[str, Any] | None = None,
    search_used: bool = False,
    search_n_generated: int | None = None,
    search_n_dedup: int | None = None,
    search_diagnostics: dict[str, Any] | None = None,
    segments: list[dict[str, Any]] | None = None,
    segments_capped: bool = False,
    cat_trigger_audit: dict[str, int] | None = None,
) -> dict[str, Any]:
    record = {
        "turn": int(turn),
        "chain_types": list(chain_types),
        "stop_reason": str(stop_reason),
        "outcome": outcome,
        "opponent_pets": opponent_pets,
        "lives": int(lives),
        "opp_lives": int(opp_lives),
        # None unless `play_one_game(..., capture_detail=True)` (We2 render
        # path); always present (rather than an absent key) so every
        # `per_turn` record has the same schema regardless of mode.
        "detail": detail,


        "search_used": bool(search_used),


        "search_n_generated": (int(search_n_generated) if search_n_generated is not None else None),
        "search_n_dedup": (int(search_n_dedup) if search_n_dedup is not None else None),


        "search_diagnostics": search_diagnostics,


        "cat_trigger_audit": dict(cat_trigger_audit or _cat_audit_empty()),
    }


    if segments is not None:
        record["segments"] = segments
        record["n_segments"] = len(segments)
        record["segments_capped"] = bool(segments_capped)
    return record


def _build_turn_detail(
    *,
    state_before: dict[str, Any] | None,
    board_pre_battle: dict[str, Any] | None,
    parsed_state: dict[str, Any] | None,
    battle: dict[str, Any] | None,
    state_after: dict[str, Any] | None,
    full: bool,
) -> dict[str, Any]:
    """Per-turn board/battle snapshot. `full` is the `--render-games` form (We2).

    Every argument is already computed by the verified per-turn loop in
    `play_one_game` -- this function does not compute anything new, it only
    deep-copies for safekeeping across the rest of the game's turns:
    - `state_before`: the engine state at the TOP of the turn, before
      `bc.recommend` was even called (for the "Start of turn" render row).
    - `board_pre_battle`: `resolved["battle_state"]`, the EXACT board
      `resolve_end_turn_with_sampled_battle` fed the battle oracle as
      `config["playerPets"]` -- not the locally re-derived `board` var,
      which can differ from it by whatever `resolve_end_turn_pre_battle`
      normalizes internally (see that function's own pre-battle step).
    - `parsed_state`: `resolved["parsed_state"]` as-is (never flipped, see
      `ChainSnapshotSource`'s docstring) -- `["opponentPets"]` is the
      duel opponent's board the BC actually fought this turn.
    - `battle`: `resolved["battle"]`, the oracle payload (`outcome` +
      `calculator_link` already computed against the real fought boards).
    - `state_after`: the engine state once this turn's battle + life
      bookkeeping + turn-3 recovery are all applied (i.e. `state` right
      after the caller's `state = resolved["transition"]["state_after"]`)."""
    return {
        "state_before": (
            copy.deepcopy(state_before) if full and isinstance(state_before, dict) else None
        ),
        # The one field the light form carries: the EXACT board that was
        # fought, on every row, whatever mode the run is in.
        "board_pre_battle": copy.deepcopy(board_pre_battle) if isinstance(board_pre_battle, dict) else None,
        "parsed_state": (
            copy.deepcopy(parsed_state) if full and isinstance(parsed_state, dict) else None
        ),
        "battle": copy.deepcopy(battle) if full and isinstance(battle, dict) else None,
        "state_after": (
            copy.deepcopy(state_after) if full and isinstance(state_after, dict) else None
        ),
    }


def _teacher_self_check(record: dict[str, Any], *, played_signature: str) -> dict[str, Any]:
    """- `chosen_score_matches`: the chosen candidate's own recorded
      `teacher_score` IS the decision-level `chosen_teacher_score` the driver
      acted on, bit-exact (catches an index misalignment between the deduped
      candidate list and the shortlist).
    - `argmax_matches`: recomputing the argmax over the rolled-out candidates
      with the same `(rollout score, myopic score)` key `_search_rollout`
      uses lands on `chosen_index`.
    - `group_complete`: one entry per scored candidate, indices 0..n-1 once
      each, exactly one `chosen`, and the shortlist fully rolled out.
    - `chosen_board_matches`: the chosen candidate's afterstate signature is
      the board the driver actually replayed and is about to fight."""
    cands = record.get("candidates") or []
    n = int(record.get("n_candidates") or 0)
    chosen_index = record.get("chosen_index")
    rolled = [c for c in cands if c.get("rolled_out")]

    indices_ok = sorted(int(c.get("index", -1)) for c in cands) == list(range(len(cands)))
    group_complete = bool(
        len(cands) == n
        and n > 0
        and indices_ok
        and sum(1 for c in cands if c.get("chosen")) == 1
        and len(rolled) == int(record.get("shortlist_size") or -1)
    )

    chosen = None
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(cands):
        chosen = cands[chosen_index]
    chosen_score_matches = bool(
        chosen is not None
        and chosen.get("teacher_score") is not None
        and float(chosen["teacher_score"]) == float(record.get("chosen_teacher_score"))
        and bool(chosen.get("chosen"))
    )

    argmax_matches = False
    if rolled:
        best = max(rolled, key=lambda c: (float(c["teacher_score"]), float(c["myopic_score"])))
        argmax_matches = int(best["index"]) == chosen_index

    chosen_board_matches = bool(
        chosen is not None and str(chosen.get("signature") or "") == str(played_signature)
    )

    return {
        "chosen_score_matches": chosen_score_matches,
        "argmax_matches": argmax_matches,
        "group_complete": group_complete,
        "chosen_board_matches": chosen_board_matches,
    }


def _versus_win(lives: int, opp_lives: int) -> bool:
    """Versus win rule, shared so it is spelled out in exactly one place:
    the driver's `play_out_game` tail AND `search_recommender.py`'s rollout
    scorer (which needs the identical rule for a continuation that ends the
    same turn its shortlisted candidate's own battle resolves)."""
    return bool(int(opp_lives) <= 0 and int(lives) > 0)


def _arena_win(lives: int, trophies: int) -> bool:
    """Arena win."""
    return bool(int(trophies) >= MAX_TROPHIES and int(lives) > 0)


def _resolve_versus_turn(
    board: dict[str, Any],
    *,
    sample_for_pid_fn: Callable[[str, int], dict[str, Any]],
    sample_random_fn: Callable[[int], dict[str, Any]],
    parse_cache: dict[str, Any] | None = None,
    simulation_count: int = 1,
    battle_logs_enabled: bool = False,
    game_mode: str = GAME_MODE_VERSUS,
    max_lives: int | None = None,
) -> dict[str, Any]:
    """Resolve END_TURN on an ALREADY-DECODED-AND-APPLIED `board` (this
    turn's shop-phase actions are already committed) via
    `end_turn.py::resolve_end_turn_with_sampled_battle` -- the single source
    of truth for versus battle resolution + lives/turn-3-recovery
    bookkeeping. This helper only adds the "read the result into a
    state_after that carries `last_opponent_team` forward, plus whether a
    random fallback fired" glue `play_out_game`'s per-turn loop needs, ONCE,
    so it is identical regardless of which caller invokes it: the real
    per-game loop (`play_out_game`, `simulation_count=1`, its default), or
    `search_recommender.py`'s rollout scorer resolving a shortlisted
    candidate's OWN turn (`simulation_count=--rollout-ksim`; every
    subsequent turn of that same continuation calls `play_out_game`, which
    reverts to the default).

    Returns:
    - `{"ok": False, "end_reason": "end_turn_failed:<error>", "resolved": <envelope>}`
    - `{"ok": True, "state_after": dict, "outcome": str, "parsed_state":
      dict | None, "battle": dict, "battle_state": dict (the board actually
      fought, == `board`), "fallback_used": bool, "resolved": <envelope>}`"""
    resolve_kwargs: dict[str, Any] = {
        "game_mode": game_mode,
        "sample_for_pid_fn": sample_for_pid_fn,
        "sample_random_fn": sample_random_fn,
        "parse_cache": parse_cache,
        "simulation_count": simulation_count,
    }
    if battle_logs_enabled:
        resolve_kwargs["battle_logs_enabled"] = True
    if max_lives is not None:
        resolve_kwargs["max_lives"] = int(max_lives)
    resolved = resolve_end_turn_with_sampled_battle(board, **resolve_kwargs)
    if not resolved.get("ok"):
        return {
            "ok": False,
            "end_reason": f"{END_REASON_END_TURN_FAILED}:{resolved.get('error')}",
            "resolved": resolved,
        }

    battle = resolved.get("battle") if isinstance(resolved.get("battle"), dict) else {}
    outcome = str(battle.get("outcome", "unknown"))
    parsed_state = resolved.get("parsed_state")

    state_after = resolved["transition"]["state_after"]
    if isinstance(parsed_state, dict):
        _set_last_opponent_team(state_after, parsed_state.get("opponentPets"))

    engine_notes = resolved["transition"].get("engine_notes") or []
    fallback_used = "end_turn_versus_chain_fallback_random" in engine_notes

    return {
        "ok": True,
        "state_after": state_after,
        "outcome": outcome,
        "parsed_state": parsed_state,
        "battle": battle,
        "battle_state": resolved.get("battle_state"),
        "fallback_used": fallback_used,
        "resolved": resolved,
    }


def play_out_game(
    state: dict[str, Any],
    bc: BcRecommender | SearchRecommender,
    *,
    initial_pid: str,
    sample_for_pid_fn: Callable[[str, int], dict[str, Any]],
    sample_random_fn: Callable[[int], dict[str, Any]],
    max_turn: int = DEFAULT_MAX_TURN,
    parse_cache: dict[str, Any] | None = None,
    capture_detail: bool = False,
    on_turn: Callable[[dict[str, Any]], None] | None = None,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    afterstate_sink: Callable[[dict[str, Any]], None] | None = None,
    teacher_sink: Callable[[dict[str, Any]], None] | None = None,
    segment_sink: Callable[[dict[str, Any]], None] | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> dict[str, Any]:
    """Play `state` to a lives verdict (or the turn cap), one turn at a
    time: decode (`bc.recommend`) -> replay non-END_TURN ops through the
    engine -> resolve END_TURN (`_resolve_versus_turn`, `simulation_count=1`,
    the driver's normal single-draw mechanics) -> bookkeeping -> repeat.

    `sample_for_pid_fn`/`sample_random_fn` (plain callables, not an
    `opp_source` object) so a caller can substitute an ISOLATED fallback
    sampler without mutating a shared `ChainSnapshotSource`'s own
    `_random_rng` -- required by the rollout scorer, whose throwaway
    continuations must never perturb the real game's own later (for-real)
    draws from that same shared source (see
    `chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`).

    `initial_pid`: the fallback value for `final_followed_pid` if
    `state`'s own `meta.versus.current_opponent_participation_id` is ever
    absent (defensive only -- `_new_game_state` always sets it, so this is
    never hit via `play_one_game`; the rollout scorer passes whatever pid
    `state` is already following when the continuation starts).

    `capture_detail` (We2, default False): see `play_one_game`'s docstring
    -- unchanged meaning, just threaded through as a parameter now.

    1. the battle resolves with `game_mode="arena"` and the 5-life heal cap,
       so `versus_lives.py`'s existing arena branch does the bookkeeping
       (win -> trophy, loss -> life) instead of the versus one;
    2. the D2 race convention is (re)written onto the state at the TOP of
       every turn, before `bc.recommend` sees it and before this loop reads
       `pre_turn_opp_lives` off it, so the leaf, the recorders and the report
       all carry one number with one definition;
    3. the terminal rule is arena's: 10 trophies -> win, 0 lives -> loss,
       `max_turn` -> non-win. `TrainingEnv._is_done` is deliberately NOT
       consulted under arena rules -- its arena branch ends the episode at
       SEVEN trophies (the RL env's own frame, `train/env.py`), which is not
       this ruler.

    Returns the SAME keys `play_one_game` returns minus the game-identity
    fields (`game_index`/`followed_pid`/`initial_followed_pid`, which only
    that function's caller knows): `final_followed_pid`, `win`,
    `player_lives`, `opponent_lives`, `turns_survived`, `end_reason`,
    `num_fallbacks`, `fallback_turns`, `search_used_turns`, `stop_reasons`,
    `per_turn`, plus We8's `opponent_mode` / `fallback_semantics` and Wb's
    `game_rules` / `arena_race_convention` / `trophies`."""

    from .exp13_w1_records import SegmentRecorder

    rules = _normalize_game_rules(game_rules)
    race_convention = _normalize_arena_race_convention(arena_race_convention)
    is_arena = rules == GAME_RULES_ARENA


    engine_game_mode = GAME_MODE_ARENA if is_arena else GAME_MODE_VERSUS
    heal_cap = ARENA_MAX_LIVES if is_arena else None
    per_turn: list[dict[str, Any]] = []
    stop_reason_counts: Counter[str] = Counter()
    fallback_turns: list[int] = []
    turns_completed = 0
    end_reason: str | None = None


    pool_exhausted_detail: dict[str, Any] | None = None


    wins_so_far = 0


    set_race_context = getattr(bc, "set_race_context", None)
    if not callable(set_race_context):
        set_race_context = None


    turn_mode = normalize_turn_mode(turn_mode)
    honest = is_honest(turn_mode)
    game_engine_seed = read_engine_seed(state)
    set_imagination_context = getattr(bc, "set_imagination_context", None)
    if not callable(set_imagination_context):
        set_imagination_context = None

    while True:


        if opponent_mode == OPPONENT_MODE_ARENA:
            live_versus_meta = _versus_meta(state)
            live_versus_meta.pop("current_opponent_participation_id", None)


        if is_arena:
            race_wins = int(state.get("trophies", 0) or 0)
            _apply_arena_race_context(state, convention=race_convention)
        else:
            race_wins = wins_so_far

        # `capture_detail`-only: the board as of the TOP of this turn, before
        # `bc.recommend` acts on it -- the "Start of turn" render row. Cheap
        # to skip entirely in the default (fast, N~=300-game) path.
        state_before_snapshot = copy.deepcopy(state) if capture_detail else None


        hook_state_before = copy.deepcopy(state) if on_turn is not None else None

        pre_turn_lives = int(state.get("lives", 0))
        pre_turn_opp_lives = int(_versus_meta(state).get("opponent_lives", 0))

        def _count_stop_reason(reason: str) -> None:
            stop_reason_counts[reason] += 1


        step_records: list[dict[str, Any]] | None = [] if on_turn is not None else None


        segment_recorder = (
            SegmentRecorder(segment_sink, engine_seed=game_engine_seed, honest=honest)
            if segment_sink is not None
            else None
        )
        decide = None
        if segment_recorder is not None:
            segment_recorder.begin_turn(state)
            decide = segment_recorder.wrap_decide(bc_decide(bc))

        turn_out = run_segmented_turn(
            state,
            bc=bc,
            honest=honest,
            game_engine_seed=game_engine_seed,
            race_wins=race_wins,
            set_race_context=set_race_context,
            set_imagination_context=set_imagination_context,
            decide=decide,
            step_records=step_records,
            stop_reason_sink=_count_stop_reason,
            on_committed_op=(
                segment_recorder.on_committed_op
                if segment_recorder is not None
                else None
            ),
        )
        board = turn_out.board
        rec = turn_out.rec
        chain_preview = turn_out.chain_preview
        committed_ops = turn_out.committed_ops
        chain_types = turn_out.chain_types
        stop_reason = turn_out.stop_reason
        replay_diverged_at = turn_out.replay_diverged_at
        decode_failed = turn_out.decode_failed
        segments = turn_out.segments
        segments_capped = turn_out.segments_capped
        if segment_recorder is not None:
            # Before the `decode_failed` / `replay_diverged_at`
            # branches below, both of which break out of the turn
            # loop -- a failed turn's segments are recorded too.
            segment_recorder.end_turn(turn_out)

        if decode_failed:
            # The REASON travels with the failure. See `_decode_failure_reason`.
            end_reason = (
                f"{END_REASON_DECODE_FAILED}:{_decode_failure_reason(rec, stop_reason)}"
            )
            per_turn.append(
                _turn_record(
                    turn=state.get("turn", 0),
                    # Empty under the determinized frame (nothing was decoded
                    # at all); under the honest frame it carries whatever
                    # earlier segments of this turn already committed.
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        board_pre_battle=None,
                        parsed_state=None,
                        battle=None,
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": state.get("turn", 0),
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": None,
                    }
                )
            break

        if replay_diverged_at is not None:
            end_reason = f"{END_REASON_CHAIN_REPLAY_DIVERGED}:{replay_diverged_at}"
            per_turn.append(
                _turn_record(
                    turn=state.get("turn", 0),
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        board_pre_battle=board,
                        parsed_state=None,
                        battle=None,
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": state.get("turn", 0),
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": step_records or [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": board,
                    }
                )
            break

        turn_played = int(board.get("turn", state.get("turn", 1)))


        if afterstate_sink is not None:
            afterstate_sink(
                {
                    "turn": turn_played,
                    "stop_reason": stop_reason,
                    "lives": pre_turn_lives,
                    "opp_lives": pre_turn_opp_lives,
                    "wins": race_wins,
                    "state": copy.deepcopy(board),
                }
            )


        if teacher_sink is not None:
            teacher_record = rec.get("teacher_record")
            if isinstance(teacher_record, dict):
                played_signature = state_signature(board)
                teacher_sink(
                    {
                        **teacher_record,
                        "turn": turn_played,
                        "race": {
                            "turn": turn_played,
                            "lives": pre_turn_lives,
                            "opp_lives": pre_turn_opp_lives,
                            "wins": race_wins,
                        },
                        "stop_reason": stop_reason,
                        "played_signature": played_signature,


                        "played_chain": copy.deepcopy(
                            committed_ops if honest else chain_preview
                        ),


                        "start_state": copy.deepcopy(state),
                        "self_check": _teacher_self_check(
                            teacher_record, played_signature=played_signature
                        ),
                    }
                )

        turn_resolution = _resolve_versus_turn(
            board,
            sample_for_pid_fn=sample_for_pid_fn,
            sample_random_fn=sample_random_fn,
            parse_cache=parse_cache,
            simulation_count=1,
            game_mode=engine_game_mode,
            max_lives=heal_cap,
        )

        if not turn_resolution["ok"]:
            end_reason = turn_resolution["end_reason"]


            pool_exhausted = is_arena and _is_pool_exhausted_end_reason(end_reason)
            if pool_exhausted:
                pool_exhausted_detail = {
                    "turn": turn_played,
                    "raw_end_reason": str(end_reason),
                }
                end_reason = END_REASON_OPPONENT_POOL_EXHAUSTED
            resolved = turn_resolution["resolved"]
            per_turn.append(
                _turn_record(
                    turn=turn_played,
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        # Best-effort: `resolved` failed at some point
                        # after `battle_state` was computed, so it may or
                        # may not carry `parsed_state`/`battle` depending
                        # on which stage failed (see
                        # `resolve_end_turn_with_sampled_battle`'s own
                        # per-branch envelopes) -- fall back to the
                        # locally replayed `board` if `battle_state`
                        # itself is absent.
                        board_pre_battle=(
                            resolved.get("battle_state")
                            if isinstance(resolved.get("battle_state"), dict)
                            else board
                        ),
                        parsed_state=resolved.get("parsed_state"),
                        battle=resolved.get("battle"),
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": turn_played,
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": step_records or [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": (
                            resolved.get("battle_state") if isinstance(resolved.get("battle_state"), dict) else board
                        ),
                    }
                )
            break

        state = turn_resolution["state_after"]
        if turn_resolution["fallback_used"]:
            fallback_turns.append(turn_played)

        turns_completed += 1


        if turn_resolution["outcome"] == "win":
            wins_so_far += 1
        parsed_state = turn_resolution["parsed_state"]
        per_turn.append(
            _turn_record(
                turn=turn_played,
                chain_types=chain_types,
                stop_reason=stop_reason,
                outcome=turn_resolution["outcome"],
                opponent_pets=(parsed_state.get("opponentPets") if isinstance(parsed_state, dict) else None),
                lives=int(state.get("lives", 0)),
                opp_lives=int(_versus_meta(state).get("opponent_lives", 0)),
                detail=_build_turn_detail(
                    state_before=state_before_snapshot,
                    board_pre_battle=turn_resolution["battle_state"],
                    parsed_state=parsed_state,
                    battle=turn_resolution["battle"],
                    # `state` was already reassigned to
                    # `turn_resolution["state_after"]` above, so it IS
                    # this turn's true state_after.
                    state_after=state,
                    full=capture_detail,
                ),
                search_used=bool(rec.get("search_used", False)),
                search_n_generated=rec.get("search_n_generated"),
                search_n_dedup=rec.get("search_n_dedup"),
                search_diagnostics=rec.get("search_diagnostics"),
                segments=segments,
                segments_capped=segments_capped,
                cat_trigger_audit=turn_out.cat_trigger_audit,
            )
        )
        if on_turn is not None:
            on_turn(
                {
                    "turn": turn_played,
                    "state_before": hook_state_before,
                    "rec": rec,
                    "step_records": step_records or [],
                    "ok": True,
                    "end_reason": None,
                    "state_after": state,
                    "board_pre_battle": turn_resolution["battle_state"],
                }
            )


        if is_arena:
            lives_done = bool(
                int(state.get("lives", 0)) <= 0
                or int(state.get("trophies", 0) or 0) >= MAX_TROPHIES
            )
        else:
            lives_done = bool(TrainingEnv._is_done(state, None))
        if lives_done or turn_played >= max_turn:
            break

    lives_final = int(state.get("lives", 0))
    opp_lives_final = int(_versus_meta(state).get("opponent_lives", 0))
    trophies_final = int(state.get("trophies", 0) or 0) if is_arena else None

    if end_reason is None:
        if is_arena:
            # A turn awards a trophy or costs a life, never both, so these
            # two branches cannot both be true; lives is checked first to
            # mirror the versus ordering below.
            if lives_final <= 0:
                end_reason = END_REASON_PLAYER_LIVES_0
            elif int(trophies_final or 0) >= MAX_TROPHIES:
                end_reason = END_REASON_TROPHIES_10
            else:
                end_reason = END_REASON_TURN_CAP
        elif lives_final <= 0:
            end_reason = END_REASON_PLAYER_LIVES_0
        elif opp_lives_final <= 0:
            end_reason = END_REASON_OPPONENT_LIVES_0
        else:
            end_reason = END_REASON_TURN_CAP

    win = (
        _arena_win(lives_final, int(trophies_final or 0))
        if is_arena
        else _versus_win(lives_final, opp_lives_final)
    )

    # FIX 4: `final_followed_pid` records which pid the versus chain was
    # actually on at game end -- it differs from `initial_pid` exactly when
    # a fallback switched the followed game
    # (`resolve_end_turn_with_sampled_battle` rewrites
    # `meta.versus.current_opponent_participation_id` on each resolved turn),
    # so the two together make the fallback's effect on opponent identity
    # visible without cross-referencing `fallback_turns`.
    final_followed_pid = str(_versus_meta(state).get("current_opponent_participation_id") or initial_pid)


    if afterstate_sink is not None:
        afterstate_sink(
            {
                "final": True,
                "win": win,
                "end_reason": end_reason,
                "turns_survived": turns_completed,


                **({"game_rules": rules} if is_arena else {}),
            }
        )


    if teacher_sink is not None:
        teacher_sink(
            {
                "final": True,
                "win": win,
                "end_reason": end_reason,
                "turns_survived": turns_completed,


                **({"game_rules": rules} if is_arena else {}),
            }
        )

    return {
        "final_followed_pid": final_followed_pid,
        "win": win,
        "player_lives": lives_final,


        "opponent_lives": opp_lives_final,
        "trophies": trophies_final,
        "game_rules": rules,
        "arena_race_convention": (race_convention if is_arena else None),
        "turns_survived": turns_completed,
        "end_reason": end_reason,


        **(
            {"pool_exhausted": pool_exhausted_detail}
            if pool_exhausted_detail is not None
            else {}
        ),
        "num_fallbacks": len(fallback_turns),
        "fallback_turns": fallback_turns,


        "opponent_mode": opponent_mode,
        "fallback_semantics": _fallback_semantics(opponent_mode),


        "search_used_turns": sum(1 for t in per_turn if t.get("search_used")),


        "search_dedup_by_turn": [
            v for t in per_turn for v in _search_width_counts(t, "search_n_dedup")
        ],
        "search_generated_by_turn": [
            v for t in per_turn for v in _search_width_counts(t, "search_n_generated")
        ],


        **(
            {
                "turn_mode": turn_mode,
                "segments_by_turn": [int(t.get("n_segments") or 0) for t in per_turn],
                "n_segments_total": sum(int(t.get("n_segments") or 0) for t in per_turn),
                # How many SEARCHES this game paid for: one per segment that
                # actually searched. Under the determinized frame this would
                # equal `search_used_turns` by construction; the honest frame
                # is where the two come apart, and that gap IS the frame's
                # compute cost (A1 ruling 6(ii)/(iii) budget both off it).
                "n_searched_segments": sum(
                    1
                    for t in per_turn
                    for s in (t.get("segments") or [])
                    if s.get("search_used")
                ),


                "boundary_reason_counts": dict(
                    sorted(
                        Counter(
                            str(s["boundary_reason"])
                            for t in per_turn
                            for s in (t.get("segments") or [])
                            if s.get("boundary_reason")
                        ).items()
                    )
                ),
            }
            if honest
            else {}
        ),
        "stop_reasons": dict(stop_reason_counts),


        "cat_trigger_audit": _cat_audit_sum(t.get("cat_trigger_audit") for t in per_turn),
        "per_turn": per_turn,
    }


def play_one_game(
    bc: BcRecommender | SearchRecommender,
    opp_source: ChainSnapshotSource,
    opening: VariedOpeningSource | FixedOpeningSource | dict[str, Any],
    *,
    game_index: int,
    max_turn: int = DEFAULT_MAX_TURN,
    seed: int = DEFAULT_SEED,
    parse_cache: dict[str, Any] | None = None,
    capture_detail: bool = False,
    on_turn: Callable[[dict[str, Any]], None] | None = None,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    arena_source: ChainSnapshotSource | None = None,
    afterstate_sink: Callable[[dict[str, Any]], None] | None = None,
    teacher_sink: Callable[[dict[str, Any]], None] | None = None,
    segment_sink: Callable[[dict[str, Any]], None] | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> dict[str, Any]:
    """Play one whole game with the BC policy to a lives/trophy verdict.

    `opening` (full-game frame fix, `train/opening_source.py`): either an
    opening-source object (`.state_for_game(game_index)` -- a VARIED,
    per-game turn-1 state; `main()`'s new default) or, for full backward
    compatibility with existing direct callers that still pass a raw fixture
    dict (`tools/gen_distill_dataset.py`, some tests -- neither of which this
    fix touches), a plain `dict`, silently wrapped as a `FixedOpeningSource`
    so every game gets that SAME state -- byte-for-byte the old
    single-fixture behavior those callers already depend on.

    `capture_detail` (We2, default False): when True, every `per_turn` entry
    also carries a `detail` sub-dict (`_build_turn_detail`) with the boards
    and battle payload needed to render a gallery image for that turn. This
    changes ONLY what gets recorded, never the decode/replay/battle-
    resolution above -- it never reads its own output, so turning it on
    cannot change which action gets chosen or which battle gets fought.
    It is NOT, however, a "replay this game later" switch: the battle
    oracle this turn's END_TURN calls into (`resolve_end_turn_with_sampled_battle`
    -> `run_battle_oracle_with_config`) is an unseeded Monte Carlo draw
    (verified: the identical board config produced different W/L outcomes
    across repeated calls), so re-running `game_index` a second time -- with
    or without `capture_detail` -- is NOT guaranteed to reproduce the same
    game. Callers that need to render a SPECIFIC game must set
    `capture_detail=True` on the ONE pass that plays it (see
    `render_selected_games`'s docstring), not capture it after the fact."""


    _assert_game_rules_supported(
        bc, game_rules=game_rules, opponent_mode=opponent_mode, turn_mode=turn_mode
    )

    opening_source = opening if hasattr(opening, "state_for_game") else fixed_opening_source(opening)
    fixture_initial_state = opening_source.state_for_game(game_index)

    mode = str(opponent_mode or OPPONENT_MODE_CHAIN).strip().lower()
    if mode not in OPPONENT_MODES:
        raise ValueError(f"unknown_opponent_mode:{opponent_mode}:valid={list(OPPONENT_MODES)}")


    set_context = getattr(bc, "set_decision_context", None)
    if callable(set_context):
        set_context(game_index=int(game_index))

    followed_pid = opp_source.initial_pid_for_game(game_index)
    # Namespaced separately from `initial_pid_for_game`'s own RNG stream so
    # the two draws (which pid to follow vs. which engine seed to start
    # from) vary independently across games. See `_new_game_state` for why
    # an explicit engine seed is required at all.
    engine_seed_rng = random.Random(f"eval_versus_fullgame_engine_seed:{seed}:{game_index}")
    engine_seed = engine_seed_rng.randrange(0, 2**31)
    state = _new_game_state(
        fixture_initial_state,
        followed_pid,
        engine_seed=engine_seed,
        game_rules=game_rules,
        arena_race_convention=arena_race_convention,
    )


    if mode == OPPONENT_MODE_ARENA:
        if arena_source is None:
            raise ValueError("arena_opponent_mode_requires_arena_source")
        arena_rng = random.Random(f"exp12_arena_opponent:{seed}:{game_index}")

        def _arena_sample_random(turn: int) -> dict[str, Any]:
            return arena_source.sample_random_with_rng(int(turn), arena_rng)

        sample_random_fn: Callable[[int], dict[str, Any]] = _arena_sample_random
    else:
        chain_fallback_rng = random.Random(f"exp12_chain_fallback:{seed}:{game_index}")

        def _chain_sample_random(turn: int) -> dict[str, Any]:
            return opp_source.sample_random_with_rng(int(turn), chain_fallback_rng)

        sample_random_fn = _chain_sample_random


    play_max_turn = effective_arena_max_turn(max_turn, arena_source, opponent_mode=mode)

    outcome = play_out_game(
        state,
        bc,
        initial_pid=followed_pid,
        sample_for_pid_fn=opp_source.sample_for_pid,
        sample_random_fn=sample_random_fn,
        max_turn=play_max_turn,
        parse_cache=parse_cache,
        capture_detail=capture_detail,
        on_turn=on_turn,
        opponent_mode=mode,
        afterstate_sink=afterstate_sink,
        teacher_sink=teacher_sink,
        segment_sink=segment_sink,
        game_rules=game_rules,
        arena_race_convention=arena_race_convention,


        turn_mode=turn_mode,
    )

    return {
        "game_index": int(game_index),
        "followed_pid": followed_pid,
        "initial_followed_pid": followed_pid,


        "initial_pid_unused": mode == OPPONENT_MODE_ARENA,


        "opening_source_kind": opening_source.provenance_for_game(game_index),
        **outcome,
    }


INCOMPLETE_FRACTION_WARN_THRESHOLD = 0.02


def _search_width_aggregate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A width arm is only real if the proposer produces distinct end boards at
    that width: `--search-candidates 72` that dedups to 9 boards a turn is a
    72-candidate arm in name and a 9-candidate arm in fact, and the width
    curve would then be measuring nothing. So every report carries the
    per-turn dedup distribution (mean / median / min / max / deciles) next
    to the requested width, and `dedup_ratio` = mean distinct / requested.

    Returns None when no turn reported counts (plain `--recommender bc`, or
    a JSONL captured before this telemetry existed) rather than a block of
    zeros that would read as "the proposer produced nothing"."""
    dedup: list[int] = []
    generated: list[int] = []
    for r in results:
        dedup.extend(int(v) for v in (r.get("search_dedup_by_turn") or []))
        generated.extend(int(v) for v in (r.get("search_generated_by_turn") or []))
    if not dedup:
        return None
    ordered = sorted(dedup)
    requested = sorted(set(generated))
    mean_dedup = statistics.mean(ordered)
    return {
        "n_searched_turns": len(ordered),
        # Normally one value (the arm's width); a list keeps a hand-merged
        # multi-arm JSONL honest instead of silently quoting one of them.
        "requested_candidates": requested,
        "dedup_mean": mean_dedup,
        "dedup_median": statistics.median(ordered),
        "dedup_min": ordered[0],
        "dedup_max": ordered[-1],
        "dedup_deciles": [ordered[min(len(ordered) - 1, (len(ordered) * d) // 10)] for d in range(1, 10)],
        "dedup_ratio": (mean_dedup / float(requested[0]) if len(requested) == 1 and requested[0] > 0 else None),
        "dedup_histogram": dict(sorted(Counter(ordered).items())),
    }


def _honest_frame_aggregate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Returns None -- so the report gains NO key at all -- whenever no game
    carries a `turn_mode`, which is every determinized run and every report
    re-derived from a pre-A1 JSONL. That is what keeps the byte-identity pin
    green; a block of zeros would instead read as "this run segmented
    nothing", a different claim from "this run was not on that frame"."""
    tagged = [r for r in results if r.get("turn_mode")]
    if not tagged:
        return None
    if len(tagged) != len(results):
        raise ValueError(
            f"honest_frame_aggregate_mixed_frames:tagged={len(tagged)}:total={len(results)} -- "
            f"{len(results) - len(tagged)} game rows carry no turn_mode (i.e. they were "
            "played on the whole-determinized frame) while others do. Aggregate each "
            "frame's rows separately; the two arms are different rulers, not one run."
        )
    modes = sorted({str(r["turn_mode"]) for r in tagged})
    segments = [int(v) for r in results for v in (r.get("segments_by_turn") or [])]
    searched = [int(r.get("n_searched_segments") or 0) for r in results]
    boundary: Counter[str] = Counter()
    for r in results:
        for reason, count in (r.get("boundary_reason_counts") or {}).items():
            boundary[str(reason)] += int(count)
    return {
        "turn_mode": modes[0] if len(modes) == 1 else "mixed:" + ",".join(modes),
        "n_turns": len(segments),
        "n_segments": sum(segments),
        "mean_segments_per_turn": (statistics.mean(segments) if segments else None),
        "segments_histogram": {str(k): int(v) for k, v in sorted(Counter(segments).items())},
        "n_searched_segments": sum(searched),
        "mean_searches_per_game": (statistics.mean(searched) if searched else None),
        "boundary_reason_counts": dict(sorted(boundary.items())),
    }


def _compute_aggregate(
    results: list[dict[str, Any]],
    *,
    arena_pool_size_by_turn: dict[int, int] | None = None,
) -> dict[str, Any]:
    """FIX 1 (metric integrity): every game is classified `completed`
    (`end_reason` in `COMPLETED_END_REASONS`) or `incomplete` (an
    infrastructure failure: decode/end_turn/chain_replay). The POLICY-OUTCOME
    metrics -- winrate + its CI, avg player/opponent lives, avg turns
    survived -- are computed over COMPLETED games ONLY, because an incomplete
    game carries `win=False` and truncated lives/turns that are artifacts of
    the harness giving up, not of the policy losing; including them would
    bias every one of these down. `n_completed`/`n_incomplete`/
    `incomplete_by_reason`/`incomplete_fraction` are reported alongside so
    the drop is explicit. The DIAGNOSTIC distributions (end_reason across ALL
    games, stop_reason, chain_length, fallback) stay over ALL games on
    purpose -- the whole point of `end_reason_distribution` is to SHOW the
    infrastructure failures, and fallback/chain/stop stats describe the
    opponent source and decoder regardless of how the game ended.

    Note `completion_rate` is about the ARENA RUN being completed (10
    trophies); `n_completed`/`incomplete_*` above are about the HARNESS
    carrying a game to any verdict at all. Different questions, both kept."""
    n = len(results)
    modes = sorted({str(r.get("opponent_mode") or OPPONENT_MODE_CHAIN) for r in results})
    if len(modes) == 1:
        opponent_mode: str | None = modes[0]
    elif not modes:
        opponent_mode = None
    else:
        opponent_mode = "mixed:" + ",".join(modes)
    rules_seen = sorted({str(r.get("game_rules") or GAME_RULES_VERSUS) for r in results})
    if len(rules_seen) == 1:
        game_rules: str | None = rules_seen[0]
    elif not rules_seen:
        game_rules = None
    else:
        game_rules = "mixed:" + ",".join(rules_seen)
    conventions = sorted(
        {str(r["arena_race_convention"]) for r in results if r.get("arena_race_convention")}
    )
    ruler_block = {
        "opponent_mode": opponent_mode,


        "game_rules": game_rules,
        "arena_race_convention": (
            conventions[0]
            if len(conventions) == 1
            else ("mixed:" + ",".join(conventions) if conventions else None)
        ),
        "trophy_semantics": (
            TROPHY_SEMANTICS_ARENA if game_rules == GAME_RULES_ARENA else TROPHY_SEMANTICS_VERSUS
        ),
        "fallback_semantics": (_fallback_semantics(opponent_mode) if len(modes) == 1 else None),
        # {turn: n_candidate_games} of the ARENA pool -- None under the
        # chain ruler (no arena pool is built) so a chain report never
        # implies one existed.
        "arena_pool_size_by_turn": (
            {str(int(t)): int(v) for t, v in sorted(arena_pool_size_by_turn.items())}
            if isinstance(arena_pool_size_by_turn, dict)
            else None
        ),
    }
    if n == 0:
        return {"n_games": 0, "n_completed": 0, "n_incomplete": 0, **ruler_block}

    completed = [r for r in results if _is_completed_end_reason(r["end_reason"])]
    incomplete = [r for r in results if not _is_completed_end_reason(r["end_reason"])]
    n_completed = len(completed)
    n_incomplete = len(incomplete)
    incomplete_by_reason = dict(sorted(Counter(r["end_reason"] for r in incomplete).items()))

    if completed:
        win_pairs = [(int(r["game_index"]), 1.0 if r["win"] else 0.0) for r in completed]
        win_mean, win_lo, win_hi, win_n_clusters = cluster_ci(win_pairs)
        wins = sum(1 for r in completed if r["win"])
        winrate = {
            "mean": win_mean,
            "lo95": win_lo,
            "hi95": win_hi,
            "n_games": win_n_clusters,
            "n_wins": wins,
        }
        avg_turns_survived = statistics.mean(r["turns_survived"] for r in completed)
        avg_player_lives = statistics.mean(r["player_lives"] for r in completed)


        avg_opponent_lives = (
            None
            if game_rules == GAME_RULES_ARENA
            else statistics.mean(r["opponent_lives"] for r in completed)
        )
    else:
        winrate = None
        avg_turns_survived = avg_player_lives = avg_opponent_lives = None


    mean_trophies: float | None = None
    mean_trophies_ci: dict[str, Any] | None = None
    trophies_histogram: dict[str, int] | None = None
    completion_rate: float | None = None
    if game_rules == GAME_RULES_ARENA and completed:
        trophy_values = [int(r.get("trophies") or 0) for r in completed]
        t_mean, t_lo, t_hi, t_n = cluster_ci(
            [(int(r["game_index"]), float(int(r.get("trophies") or 0))) for r in completed]
        )
        mean_trophies = t_mean
        mean_trophies_ci = {"mean": t_mean, "lo95": t_lo, "hi95": t_hi, "n_games": t_n}
        trophies_histogram = {str(k): int(v) for k, v in sorted(Counter(trophy_values).items())}
        completion_rate = (
            sum(1 for r in completed if r["end_reason"] == END_REASON_TROPHIES_10) / len(completed)
        )

    # Diagnostic distributions -- over ALL games (see docstring).
    games_with_fallback = sum(1 for r in results if r["num_fallbacks"] > 0)
    total_fallback_turns = sum(r["num_fallbacks"] for r in results)
    all_fallback_turns = sorted(t for r in results for t in r["fallback_turns"])
    end_reason_counts: Counter[str] = Counter(r["end_reason"] for r in results)


    games_with_search_used = sum(1 for r in results if r["search_used_turns"] > 0)
    total_search_used_turns = sum(r["search_used_turns"] for r in results)
    search_width = _search_width_aggregate(results)

    stop_reason_totals: Counter[str] = Counter()
    chain_lengths: list[int] = []
    for r in results:
        for turn_entry in r["per_turn"]:
            stop_reason_totals[turn_entry["stop_reason"]] += 1
            chain_lengths.append(len(turn_entry["chain_types"]))
    sorted_lengths = sorted(chain_lengths)

    report: dict[str, Any] = {
        **ruler_block,
        "n_games": n,
        "n_completed": n_completed,
        "n_incomplete": n_incomplete,
        "incomplete_fraction": n_incomplete / n,
        "incomplete_by_reason": incomplete_by_reason,
        # winrate + these three averages are over COMPLETED games only.
        "winrate": winrate,
        "avg_turns_survived": avg_turns_survived,
        "avg_player_lives": avg_player_lives,
        "avg_opponent_lives": avg_opponent_lives,


        "mean_trophies": mean_trophies,
        "mean_trophies_ci": mean_trophies_ci,
        "trophies_histogram": trophies_histogram,
        "completion_rate": completion_rate,
        # fallback + the distributions below are over ALL games.
        "fallback": {
            "rate_games": games_with_fallback / n,
            "n_games_with_fallback": games_with_fallback,
            "total_fallback_turns": total_fallback_turns,
            "fallback_turn_numbers": all_fallback_turns,
        },
        "search_used": {
            "rate_games": games_with_search_used / n,
            "n_games_with_search_used": games_with_search_used,
            "total_search_used_turns": total_search_used_turns,
        },


        "search_width": search_width,


        "cat_trigger_audit": _cat_audit_sum(r.get("cat_trigger_audit") for r in results),
        "end_reason_distribution": dict(sorted(end_reason_counts.items())),
        "stop_reason_distribution": dict(sorted(stop_reason_totals.items())),
        "chain_length": (
            {
                "min": sorted_lengths[0],
                "median": statistics.median(sorted_lengths),
                "max": sorted_lengths[-1],
            }
            if sorted_lengths
            else None
        ),
    }


    honest_block = _honest_frame_aggregate(results)
    if honest_block is not None:
        report["turn_mode"] = honest_block.pop("turn_mode")
        report["segments"] = honest_block
    return report


def _print_aggregate(
    results: list[dict[str, Any]],
    *,
    arena_pool_size_by_turn: dict[int, int] | None = None,
) -> None:
    print("=== aggregate ===", flush=True)
    agg = _compute_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)
    print(
        f"game_rules: {agg['game_rules']}  arena_race_convention: {agg['arena_race_convention']}  "
        f"opponent_mode: {agg['opponent_mode']}  fallback_semantics: {agg['fallback_semantics']}",
        flush=True,
    )
    if agg["arena_pool_size_by_turn"]:
        sizes = agg["arena_pool_size_by_turn"]
        head = {t: sizes[t] for t in list(sizes)[:8]}
        print(f"arena_pool_size_by_turn (first 8 turns): {head}", flush=True)
    if agg["n_games"] == 0:
        print("no_games_played", flush=True)
        return

    n = agg["n_games"]
    n_completed = agg["n_completed"]
    n_incomplete = agg["n_incomplete"]
    fb = agg["fallback"]

    print(f"games_total: {n}  completed: {n_completed}  incomplete: {n_incomplete}", flush=True)
    if n_incomplete:
        print(f"incomplete_by_reason: {agg['incomplete_by_reason']}", flush=True)
    if agg["incomplete_fraction"] > INCOMPLETE_FRACTION_WARN_THRESHOLD:
        print(
            f"WARNING: incomplete_fraction={agg['incomplete_fraction']:.3f} "
            f"(> {INCOMPLETE_FRACTION_WARN_THRESHOLD:.2f}) -- the eval infra dropped "
            f"{n_incomplete}/{n} games before a lives verdict; policy metrics below are "
            f"over the {n_completed} COMPLETED games only, so treat them with care and "
            f"investigate the incomplete_by_reason breakdown.",
            flush=True,
        )

    if agg["winrate"] is None:
        print("winrate: n/a (no completed games)", flush=True)
    else:
        w = agg["winrate"]
        print(
            f"winrate (completed only): {w['mean']:.3f} ({w['n_wins']}/{n_completed})  "
            f"95% cluster-CI (by game): [{w['lo95']:.3f}, {w['hi95']:.3f}]",
            flush=True,
        )
        print(f"avg_turns_survived (completed): {agg['avg_turns_survived']:.2f}", flush=True)
        print(f"avg_player_lives (completed): {agg['avg_player_lives']:.2f}", flush=True)
        if agg["avg_opponent_lives"] is not None:
            print(f"avg_opponent_lives (completed): {agg['avg_opponent_lives']:.2f}", flush=True)


    if agg.get("mean_trophies") is not None:
        ci = agg["mean_trophies_ci"]
        print(
            f"mean_trophies (completed): {agg['mean_trophies']:.3f}  "
            f"95% cluster-CI (by game): [{ci['lo95']:.3f}, {ci['hi95']:.3f}]",
            flush=True,
        )
        print(
            f"completion_rate (reached {MAX_TROPHIES} trophies, completed): "
            f"{agg['completion_rate']:.3f}",
            flush=True,
        )
        print(f"trophies_histogram (completed): {agg['trophies_histogram']}", flush=True)

    print(
        f"fallback_rate_games (all): {fb['rate_games']:.3f} ({fb['n_games_with_fallback']}/{n})  "
        f"fallback_turns_total: {fb['total_fallback_turns']}",
        flush=True,
    )
    su = agg["search_used"]
    print(
        f"search_used_rate_games (all): {su['rate_games']:.3f} ({su['n_games_with_search_used']}/{n})  "
        f"search_used_turns_total: {su['total_search_used_turns']}",
        flush=True,
    )
    sw = agg.get("search_width")
    if sw:
        ratio = "n/a" if sw["dedup_ratio"] is None else f"{sw['dedup_ratio']:.2f}"
        print(
            f"search_width: requested={sw['requested_candidates']} "
            f"dedup_mean={sw['dedup_mean']:.2f} dedup_median={sw['dedup_median']:.1f} "
            f"dedup_min={sw['dedup_min']} dedup_max={sw['dedup_max']} "
            f"dedup_ratio={ratio} searched_turns={sw['n_searched_turns']}",
            flush=True,
        )


    seg = agg.get("segments")
    if seg:
        print(
            f"turn_mode: {agg['turn_mode']}  "
            f"mean_segments_per_turn: {seg['mean_segments_per_turn']:.3f} "
            f"({seg['n_segments']}/{seg['n_turns']} turns)  "
            f"mean_searches_per_game: {seg['mean_searches_per_game']:.2f}  "
            f"segments_histogram: {seg['segments_histogram']}",
            flush=True,
        )
        print(f"boundary_reason_counts: {seg['boundary_reason_counts']}", flush=True)
    print(f"end_reason_distribution (all): {agg['end_reason_distribution']}", flush=True)
    if agg["chain_length"] is not None:
        cl = agg["chain_length"]
        print(f"chain_length_min_median_max (all): {cl['min']}/{cl['median']:.1f}/{cl['max']}", flush=True)
    print(f"stop_reason_distribution (all): {agg['stop_reason_distribution']}", flush=True)


class TeacherRecordWriter:
    """Gzip JSONL sink for `--teacher-record-out` that closes ONE COMPLETE
    GZIP MEMBER PER GAME.

    Framing per GAME fixes the contract at the granularity the data actually
    has: every game that FINISHED is a self-contained, trailer-terminated
    member, python's gzip reads concatenated members transparently, and only
    the game in flight can be short. The per-row `flush()` is kept ON TOP of
    that, so the in-flight game's completed rows still reach the disk and are
    salvageable by `iter_rows`'s truncation path.

    `write` opens the current member lazily, so a game that emits no rows
    costs no member at all."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None
        self.members_written = 0

    def write(self, text: str) -> None:
        if self._fh is None:
            self._fh = gzip.open(self.path, "at", encoding="utf-8")
        self._fh.write(text)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def end_game(self) -> None:
        """Close the current member, writing its gzip trailer. Idempotent, so
        the caller's `finally` can fire on a game that wrote nothing."""
        fh, self._fh = self._fh, None
        if fh is None:
            return
        fh.close()
        self.members_written += 1

    def close(self) -> None:
        self.end_game()


def run_versus_eval(
    bc: BcRecommender | SearchRecommender,
    opp_source: ChainSnapshotSource,
    opening: VariedOpeningSource | FixedOpeningSource | dict[str, Any],
    *,
    num_games: int,
    max_turn: int = DEFAULT_MAX_TURN,
    seed: int = DEFAULT_SEED,
    parse_cache: dict[str, Any] | None = None,
    out_fh: Any = None,
    log: bool = True,
    capture_detail: bool = False,
    game_index_start: int = 0,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    arena_source: ChainSnapshotSource | None = None,
    afterstate_out_fh: Any = None,
    teacher_writer: TeacherRecordWriter | None = None,
    segment_writer: TeacherRecordWriter | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> list[dict[str, Any]]:
    """Play `num_games` full games and return their result dicts.

    Factored out of `main` (We2). Behavior for the plain N-game path is
    unchanged from We1's inline loop: same per-game progress line, same
    optional `--out` JSONL write.

    `opening` (full-game frame fix): passed straight through to
    `play_one_game` on every iteration -- see that function's docstring for
    the accepted types (an opening-source object, the new default from
    `main()`, or a raw dict for backward compatibility with existing direct
    callers).

    `parse_cache` (FIX 2, default None): passed straight through to every
    `play_one_game` -> `resolve_end_turn_with_sampled_battle`. It is left
    None here and by `main`, and this loop does NOT lazily create a shared
    `{}` for it, because the chain snapshot ships EVERY row pre-parsed
    (`sampled["parsed_state"]` is always a dict), so `end_turn.py` never
    READS the cache -- yet it still deep-copies each parsed_state INTO a
    non-None cache on every turn, which over N~=300 games x ~10 turns is
    thousands of retained parsed-state copies (unbounded growth, pure waste).
    None makes that write a no-op. (A caller with a NON-preparsed source can
    still pass its own dict.)"""


    _assert_game_rules_supported(
        bc, game_rules=game_rules, opponent_mode=opponent_mode, turn_mode=turn_mode
    )

    results: list[dict[str, Any]] = []
    start_index = int(game_index_start)
    for game_index in range(start_index, start_index + int(num_games)):
        afterstate_sink = None
        if afterstate_out_fh is not None:

            def afterstate_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:
                payload = {"game_index": int(_game_index), **row}
                afterstate_out_fh.write(json.dumps(payload, sort_keys=True) + "\n")
                afterstate_out_fh.flush()

        teacher_sink = None
        if teacher_writer is not None:

            def teacher_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:
                payload = {"game_index": int(_game_index), **row}
                teacher_writer.write(json.dumps(payload, sort_keys=True) + "\n")
                teacher_writer.flush()

        segment_sink = None
        if segment_writer is not None:

            def segment_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:

                from .exp13_w1_records import build_segment_record

                payload = build_segment_record(row, game_id=int(_game_index))
                segment_writer.write(json.dumps(payload, sort_keys=True) + "\n")
                segment_writer.flush()

        t_game = time.monotonic()
        try:
            result = play_one_game(
                bc,
                opp_source,
                opening,
                game_index=game_index,
                max_turn=max_turn,
                seed=seed,
                parse_cache=parse_cache,
                capture_detail=capture_detail,
                opponent_mode=opponent_mode,
                arena_source=arena_source,
                afterstate_sink=afterstate_sink,
                teacher_sink=teacher_sink,
                segment_sink=segment_sink,
                game_rules=game_rules,
                arena_race_convention=arena_race_convention,
                turn_mode=turn_mode,
            )
        finally:
            # Close THIS game's gzip member (writing its trailer) whether the
            # game finished or blew up -- see `TeacherRecordWriter`.
            if teacher_writer is not None:
                teacher_writer.end_game()
            if segment_writer is not None:
                segment_writer.end_game()
        elapsed = time.monotonic() - t_game
        if log:
            print(
                f"game_done:idx={game_index}:pid={result['followed_pid']}:win={result['win']}:"
                f"turns_survived={result['turns_survived']}:end_reason={result['end_reason']}:"
                f"player_lives={result['player_lives']}:opponent_lives={result['opponent_lives']}:"


                f"trophies={result['trophies']}:"
                f"num_fallbacks={result['num_fallbacks']}:elapsed={elapsed:.1f}s",
                flush=True,
            )
        results.append(result)
        if out_fh is not None:
            out_fh.write(json.dumps(result, sort_keys=True) + "\n")
            out_fh.flush()
    return results


def select_representative_games(results: list[dict[str, Any]], k: int) -> list[tuple[str, int]]:
    """If ANY game won, always include the first win (a 5-game smoke went 0/5,
    so a win is the rarer, more informative case -- see the task background).
    Remaining slots are filled from the games that did NOT win, spread
    across `turns_survived` (worst, then best, then median of what is left)
    so the gallery shows the RANGE of how this checkpoint's games actually
    go, not three near-duplicates. Returns `(reason, game_index)` pairs in
    the order they should be rendered."""
    n = len(results)
    if n == 0 or k <= 0:
        return []

    chosen: list[tuple[str, int]] = []
    used: set[int] = set()

    win_idx = next((i for i, r in enumerate(results) if r["win"]), None)
    if win_idx is not None:
        chosen.append(("win", win_idx))
        used.add(win_idx)

    labels = ["worst", "best", "median"]
    slot = 0
    while len(chosen) < k:
        pool = sorted((i for i in range(n) if i not in used), key=lambda i: (results[i]["turns_survived"], i))
        if not pool:
            break
        if slot == 0:
            pick = pool[0]
        elif slot == 1:
            pick = pool[-1]
        else:
            pick = pool[len(pool) // 2]
        chosen.append((labels[min(slot, len(labels) - 1)], pick))
        used.add(pick)
        slot += 1

    return chosen[:k]


def _replay_order_helpers() -> tuple[Callable[[Any], list[Any]], Callable[[Any], list[Any]]]:
    """Lazy import of `eval_tempo_planner`'s pet-row reorientation helpers."""
    from .eval_tempo_planner import _replay_order_from_parsed_pets, _replay_order_from_state_team

    return _replay_order_from_state_team, _replay_order_from_parsed_pets


def _pretty_pet_name(slot: dict[str, Any]) -> str | None:
    """A readable pet name for a sidecar board row: the explicit `pet_name`
    if the engine slot carries one, else the `pet_id` de-prefixed
    (`pet-fairy-armadillo` -> `Fairy Armadillo`) so the sidecar reads in
    display names, not raw ids (the PNG already shows the sprite)."""
    name = slot.get("pet_name") or slot.get("pet_name_id")
    if isinstance(name, str) and name.strip():
        return name.strip()
    pet_id = slot.get("pet_id")
    if isinstance(pet_id, str) and pet_id.strip():
        base = pet_id.strip()
        if base.startswith("pet-"):
            base = base[len("pet-"):]
        return base.replace("-", " ").replace("_", " ").title() or pet_id.strip()
    return None


def _engine_board_summary(team: Any) -> list[dict[str, Any]]:
    """Compact per-pet summary of an engine-format board (`state.team` slots)
    for the sidecar JSON. Front-to-back, empty slots dropped."""
    out: list[dict[str, Any]] = []
    for slot in team or []:
        if not isinstance(slot, dict):
            continue
        name = _pretty_pet_name(slot)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "attack": slot.get("attack"),
                "health": slot.get("health"),
                "level": slot.get("level"),
                "equipment": slot.get("equipment_id"),
            }
        )
    return out


def _calc_opponent_summary(opponent_pets: Any) -> list[dict[str, Any]]:
    """Compact per-pet summary of a calculator-format opponent board
    (`parsed_state.opponentPets`) for the sidecar JSON."""
    out: list[dict[str, Any]] = []
    for pet in opponent_pets or []:
        if not isinstance(pet, dict):
            continue
        name = pet.get("name")
        if not name:
            continue
        equip = pet.get("equipment")
        equip_name = equip.get("name") if isinstance(equip, dict) else equip
        out.append(
            {
                "name": str(name),
                "attack": pet.get("attack"),
                "health": pet.get("health"),
                "equipment": equip_name,
            }
        )
    return out


def _render_heart_lives(outcomes: list[str], *, max_lives: int = 6) -> list[int]:
    """The exact per-row heart value the whole-game PNG shows, replicated in
    Python for the sidecar cross-check.

    This mirrors `SAP-Replay-Bot/lib/render.js`'s `renderReplayImage` life
    model EXACTLY (read directly before relying on it): `currentLives` starts
    at `maxLives`; at row index 2 (turn 3) it regains 1 if below max; the
    value is DRAWN (lives going INTO that turn's battle); THEN a LOSS
    decrements it by 1 (WIN/DRAW leave it). That model is identical to this
    driver's own versus life bookkeeping (`end_turn.py`: -1 life per loss,
    draws free, turn-3 +1 recovery), which is WHY feeding the real per-turn
    outcome sequence to the renderer reproduces the true lives race with no
    explicit lives field -- verified on 3 real games (the drawn sequence ends
    at each game's captured final `player_lives`)."""
    lives = int(max_lives)
    drawn: list[int] = []
    for i, outcome in enumerate(outcomes):
        if i == 2 and lives < max_lives:
            lives += 1
        drawn.append(lives)
        if str(outcome).strip().lower() == "loss":
            lives -= 1
    return drawn


def _full_game_rows_and_sidecar(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (calc_rows, sidecar_turns) for ONE whole-game image."""
    replay_order_state_team, replay_order_parsed_pets = _replay_order_helpers()

    rows: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    for entry in result.get("per_turn", []):
        detail = entry.get("detail") or {}
        board_pre_battle = detail.get("board_pre_battle")
        parsed_state = detail.get("parsed_state")
        battle = detail.get("battle") if isinstance(detail.get("battle"), dict) else {}
        state_before = detail.get("state_before")
        # A whole-game row needs the scored board + the fought opponent. Every
        # turn of a COMPLETED game has both; a turn that ended the game on an
        # infrastructure failure (no battle) is skipped so it doesn't inject a
        # blank row that would desync render.js's row-index turn numbering.
        if not isinstance(board_pre_battle, dict) or not isinstance(parsed_state, dict):
            continue

        outcome = str(entry.get("outcome") or battle.get("outcome") or "unknown")
        rows.append(
            {
                "turn": int(entry.get("turn", len(rows) + 1)),
                "turnLabel": "",
                "outcome": outcome,
                "opponentName": "",
                "playerPets": replay_order_state_team(board_pre_battle.get("team")),
                "opponentPets": replay_order_parsed_pets(parsed_state.get("opponentPets")),
            }
        )
        sidecar.append(
            {
                "turn": int(entry.get("turn", len(sidecar) + 1)),
                "outcome": outcome,
                "player_lives_after_battle": entry.get("lives"),
                "opponent_lives_after_battle": entry.get("opp_lives"),
                "chain_types": entry.get("chain_types"),
                "stop_reason": entry.get("stop_reason"),
                "calc_link": battle.get("calculator_link"),
                "start_board": _engine_board_summary(
                    state_before.get("team") if isinstance(state_before, dict) else None
                ),
                "end_board": _engine_board_summary(board_pre_battle.get("team")),
                "opponent_board": _calc_opponent_summary(parsed_state.get("opponentPets")),
            }
        )

    heart_lives = _render_heart_lives([r["outcome"] for r in rows])
    for turn_entry, hl in zip(sidecar, heart_lives):
        # The value the heart actually shows on this row (lives going INTO the
        # battle, render.js convention) -- distinct from
        # `player_lives_after_battle` (post-battle), both kept for the cross-check.
        turn_entry["heart_lives_shown"] = hl
    return rows, sidecar


def render_selected_games(
    results: list[dict[str, Any]],
    *,
    k: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Outputs (in `<out_dir>/images/`): `full_game_<pid8>.png` (whole-game
    image, `pid8` = initial followed pid) + `full_game_<pid8>.json` (per-turn
    chain/calc-link/lives sidecar)."""
    from ..opponents import render_replay_image_from_calc_rows

    chosen = select_representative_games(results, k)
    print(
        f"render_games_selected: "
        f"{[(label, idx, results[idx]['followed_pid']) for label, idx in chosen]}",
        flush=True,
    )
    if not chosen:
        return {"k_requested": k, "games": []}

    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    games_report: list[dict[str, Any]] = []
    for label, game_index in chosen:
        detailed = results[game_index]
        pid8 = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(detailed["followed_pid"])[:8]
        )
        rows, sidecar_turns = _full_game_rows_and_sidecar(detailed)


        stem = f"full_game_{int(game_index):03d}_{pid8}"
        png_path = images_dir / f"{stem}.png"
        json_path = images_dir / f"{stem}.json"
        render_error: str | None = None

        if not rows:
            render_error = "no_scored_turns_to_render"
        else:


            rendered = render_replay_image_from_calc_rows(
                rows,
                max_lives=6,
                player_name=None,
                header_opponent_name=None,
                include_odds=False,
            )
            if not bool(rendered.get("ok", False)):
                render_error = str(rendered.get("error") or "render_failed")
            else:
                image = rendered.get("image")
                if not isinstance(image, (bytes, bytearray)):
                    render_error = "render_output_empty"
                else:
                    png_path.write_bytes(bytes(image))

        sidecar = {
            "reason": label,
            "game_index": game_index,
            "followed_pid": detailed["followed_pid"],
            "initial_followed_pid": detailed.get("initial_followed_pid", detailed["followed_pid"]),
            "final_followed_pid": detailed.get("final_followed_pid"),
            "win": detailed["win"],
            "player_lives": detailed["player_lives"],
            "opponent_lives": detailed["opponent_lives"],
            "turns_survived": detailed["turns_survived"],
            "end_reason": detailed["end_reason"],
            "num_fallbacks": detailed.get("num_fallbacks"),
            "fallback_turns": detailed.get("fallback_turns"),
            "n_rows_rendered": len(rows),
            "chain_params_captured": False,
            "chain_note": (
                "chain_types is the action-TYPE sequence only; this run did not "
                "capture per-op shop_index/team_index params. start_board -> "
                "end_board shows what the chain achieved."
            ),
            "image": str(png_path) if render_error is None else None,
            "render_error": render_error,
            "turns": sidecar_turns,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

        if render_error is not None:
            print(f"render_full_game_error:game_index={game_index}:pid={pid8}:error={render_error}", flush=True)
        else:
            print(
                f"render_full_game_ok:reason={label}:game_index={game_index}:pid={pid8}:"
                f"rows={len(rows)}:png={png_path}",
                flush=True,
            )

        games_report.append(
            {
                "reason": label,
                "game_index": game_index,
                "followed_pid": detailed["followed_pid"],
                "win": detailed["win"],
                "player_lives": detailed["player_lives"],
                "opponent_lives": detailed["opponent_lives"],
                "turns_survived": detailed["turns_survived"],
                "end_reason": detailed["end_reason"],
                "image": str(png_path) if render_error is None else None,
                "sidecar": str(json_path),
                "n_rows_rendered": len(rows),
                "heart_lives_shown": [t["heart_lives_shown"] for t in sidecar_turns],
                "render_error": render_error,
            }
        )

    return {"k_requested": k, "games": games_report}


def load_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load per-game result dicts from a `--out` JSONL (one game per line),
    ordered by `game_index`. Lets `--render-from-jsonl` re-render an ALREADY
    captured run's galleries without replaying anything (the only faithful
    way -- see `render_selected_games`)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("game_index", 0)))
    return rows


def build_mock_end_turn_client() -> MockLlmClient:
    """Build mock end turn client."""
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from ..llm_agent.client import (
        LlmResponse,
        LlmUsage,
        MockLlmClient,
        ToolCallRequest,
    )


    def _always_end_turn(call_index: int, **_kwargs: Any) -> LlmResponse:
        return LlmResponse(
            text=None,
            tool_calls=[
                ToolCallRequest(
                    id=f"mock-end-turn-{call_index}", name="submit_chain", input={"actions": [{"type": "END_TURN"}]}
                )
            ],
            stop_reason="tool_calls",
            usage=LlmUsage(input_tokens=0, output_tokens=0),
            model="mock-llm-always-end-turn",
            raw={},
        )

    return MockLlmClient(_always_end_turn, model="mock-llm-always-end-turn")


def _resolve_llm_transcript_path(args: argparse.Namespace) -> Path | None:
    """`--llm-transcript` default: next to `--out` (`<out>.transcript.jsonl`)
    whenever `--out` is given; transcripting stays off (`None`) if neither
    is set. An explicit `--llm-transcript` always wins over the derived
    default."""
    if args.llm_transcript is not None:
        return Path(args.llm_transcript)
    if args.out is not None:
        return Path(str(args.out) + ".transcript.jsonl")
    return None


def _build_llm_agent_config(args: argparse.Namespace, *, transcript_path: Path | None) -> LlmAgentConfig:
    """Build llm agent config."""
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from .llm_recommender import LlmAgentConfig

    config = LlmAgentConfig.from_json_file(args.llm_config) if args.llm_config is not None else LlmAgentConfig()
    if args.llm_model is not None:
        config.model = str(args.llm_model)
    if args.llm_fallback is not None:
        config.fallback_mode = str(args.llm_fallback)
    if args.llm_cache_dir is not None:
        config.llm_cache_dir = str(args.llm_cache_dir)
    if args.llm_cache_mode is not None:
        config.llm_cache_mode = str(args.llm_cache_mode)
    if transcript_path is not None:
        config.transcript_path = str(transcript_path)
    if args.llm_provider is not None:
        config.provider = str(args.llm_provider)
    return config


def _is_llm_recommender(recommender: object) -> bool:
    """True for `LlmRecommender`, identified by capability rather than class.

    `llm_recommender` is imported lazily inside `_build_llm_recommender` so that
    importing this module does not pull the LLM stack, which is what the public
    release needs. The report block below runs on EVERY evaluation, so it cannot
    name the class: doing so raised `NameError` on every run. `cost_totals` and
    `fallback_reason_counts` are defined only on `LlmRecommender`.
    """
    return hasattr(recommender, "cost_totals") and hasattr(recommender, "fallback_reason_counts")

def _build_llm_recommender(config: LlmAgentConfig, *, bc: BcRecommender | None) -> LlmRecommender:
    """`--recommender llm`'s construction, matching `LlmRecommender`'s actual
    constructor (part A, `tools/llm_recommender.py`): a fresh
    `GameMemoryStore` (a real, whole-games full-game eval keeps cross-turn
    scratchpad memory ON, unlike the one-shot `eval_tempo_planner.py` harness
    -- see that file's own `memory=None` wiring), `fallback=bc.recommend`
    only when a `BcRecommender` was actually constructed (`--llm-fallback
    bc`), and an explicit `MockLlmClient` when `--llm-provider mock` asks for
    the zero-cost escape hatch (`LlmRecommender` cannot build a mock client
    from `config` alone -- `provider="mock"` requires an explicit `client=`,
    see that class's own `_build_client`)."""
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from ..llm_agent.memory import GameMemoryStore
    from ..llm_agent.transcript import TranscriptWriter
    from .llm_recommender import PROVIDER_MOCK, LlmRecommender

    transcript_path = Path(config.transcript_path) if config.transcript_path else None
    client = build_mock_end_turn_client() if config.provider == PROVIDER_MOCK else None
    recommender = LlmRecommender(
        config,
        client=client,
        memory=GameMemoryStore(),
        fallback=(bc.recommend if bc is not None else None),
        transcript=(TranscriptWriter(transcript_path) if transcript_path is not None else None),
    )
    return recommender


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a policy against recorded opponents using selectable game rules and search settings."
        )
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=Path(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--chain-snapshot", type=Path, default=Path(DEFAULT_CHAIN_SNAPSHOT))
    parser.add_argument(
        "--recommender", type=str, default="bc", choices=("bc", "search", "llm"),
        help="Policy backend: BC decoding, value-guided search, or an optional external LLM backend.",
    )
    parser.add_argument(
        "--search-candidates", type=int, default=SEARCH_DEFAULT_N_CANDIDATES,
        help="Number of root proposals before deduplication. Reports also record the distinct candidate count.",
    )
    parser.add_argument(
        "--search-ksim", type=int, default=SEARCH_DEFAULT_KSIM,
        help="--recommender search only: SearchRecommender's ksim (oracle "
        "simulation_count per candidate).",
    )
    parser.add_argument(
        "--search-scoring", type=str, default=SEARCH_DEFAULT_SCORING, choices=list(SEARCH_SCORING_MODES),
        help="Candidate scoring: myopic battle score, game rollouts, or a learned value model.",
    )
    parser.add_argument(
        "--rollout-shortlist", type=int, default=SEARCH_DEFAULT_ROLLOUT_SHORTLIST,
        help="--search-scoring rollout only: how many of the myopic-ranked deduped "
        "candidates get a full rest-of-game rollout.",
    )
    parser.add_argument(
        "--rollout-repeats", type=int, default=SEARCH_DEFAULT_ROLLOUT_REPEATS,
        help="--search-scoring rollout only: how many times each shortlisted candidate's "
        "rest-of-game is simulated (battle draws are stochastic; the score is the mean "
        "over these repeats).",
    )
    parser.add_argument(
        "--rollout-ksim", type=int, default=SEARCH_DEFAULT_ROLLOUT_KSIM,
        help="--search-scoring rollout only: simulation_count used to resolve a "
        "shortlisted candidate's OWN turn (a steadier signal for the turn actually being "
        "decided). Every SUBSEQUENT simulated turn in that same continuation reverts to "
        "the driver's normal single-draw mechanics (simulation_count=1) -- this flag "
        "does not affect them.",
    )
    parser.add_argument(
        "--rollout-opponent-mode", type=str, default=SEARCH_DEFAULT_ROLLOUT_OPPONENT_MODE,
        choices=list(SEARCH_ROLLOUT_OPPONENT_MODES),
        help="Rollout opponents: the followed chain, a random pool chain, or a chain that reached this turn.",
    )
    parser.add_argument(
        "--vgame-heads", type=Path, action="append", default=None,
        help="Value checkpoint; repeat for an ensemble. Required for vgame scoring.",
    )
    parser.add_argument(
        "--vgame-extractor", type=Path, default=None,
        help="BC feature extractor expected by the value checkpoint; its hash is checked on load.",
    )
    parser.add_argument(
        "--vgame-blend", type=float, default=VGAME_DEFAULT_BLEND,
        help="Leaf score: (1-blend)*V + blend*myopic - pessimism*ensemble_std. Zero avoids myopic oracle calls.",
    )
    parser.add_argument(
        "--vgame-pessimism", type=float, default=VGAME_DEFAULT_PESSIMISM,
        help="Penalty on ensemble standard deviation. A nonzero penalty requires multiple value heads.",
    )
    parser.add_argument(
        "--num-games", type=int, default=None,
        help="number of versus games to play; required UNLESS --render-from-jsonl is given "
        "(that mode only re-renders an existing run's galleries and plays nothing)",
    )
    parser.add_argument(
        "--opponent-mode", type=str, default=OPPONENT_MODE_CHAIN, choices=list(OPPONENT_MODES),
        help="chain follows one replay chain; arena samples a new opponent at the same turn from the pool. Independent of life rules.",
    )
    parser.add_argument(
        "--game-rules", type=str, default=GAME_RULES_VERSUS, choices=list(GAME_RULES),
        help="versus uses opposing life bars; arena ends at 10 trophies or zero lives and requires arena opponent sampling.",
    )
    parser.add_argument(
        "--arena-race-convention", type=str, default=DEFAULT_ARENA_RACE_CONVENTION,
        choices=list(ARENA_RACE_CONVENTIONS),
        help="Value input for opponent lives in Arena: trophies_mapped uses min(6, 10-trophies); const6 uses 6. Must match training.",
    )
    parser.add_argument(
        "--turn-mode", type=str, default=DEFAULT_TURN_MODE, choices=list(TURN_MODES),
        help="whole-determinized commits a full chain. segmented-honest uses independent simulation seeds and re-searches after structural chance outcomes.",
    )
    parser.add_argument(
        "--capture-candidate-chains", action="store_true",
        help="Record generated and scored candidate chains in diagnostics. Increases report size.",
    )
    parser.add_argument(
        "--completion-policy", default=SEARCH_COMPLETION_BC_GREEDY,
        choices=list(SEARCH_COMPLETION_POLICIES),
        help="Chance-node continuation policy: one greedy BC chain, or multiple BC continuations scored by V.",
    )
    parser.add_argument(
        "--completion-width", type=int, default=1,
        help="Continuation proposals per sampled outcome; includes one greedy proposal. Use 1 for bc_greedy and at least 2 for v_search.",
    )
    parser.add_argument(
        "--completion-aggregate", default=SEARCH_COMPLETION_AGG_MAX,
        choices=list(SEARCH_COMPLETION_AGGREGATES),
        help="How to aggregate continuation scores within one outcome: max selects a continuation; mean is a comparison option.",
    )
    parser.add_argument(
        "--mc-rerank-k", type=int, default=0,
        help="Re-rank the top K value-ranked groups using trophy rollouts. Zero disables this extra stage.",
    )
    parser.add_argument(
        "--stochastic-samples", type=int, default=DEFAULT_STOCHASTIC_SAMPLES,
        help="Number of chance outcomes averaged for each prefix. Candidates share sampling keys to reduce comparison noise.",
    )
    parser.add_argument(
        "--skip-imagined-validation", action="store_true",
        help="Skip schema checks only inside simulation. Committed game actions remain validated.",
    )
    parser.add_argument(
        "--arena-pool-rank-max", type=int, default=None,
        help="Inclusive rank cap for Arena opponents; unranked games are excluded when a cap is set.",
    )
    parser.add_argument(
        "--arena-pool-split", type=str, default=None, choices=list(SPLIT_NAMES),
        help="Arena opponent split: train, val or test. Distinct from the chain-pool split.",
    )
    parser.add_argument(
        "--game-index-start", type=int, default=0,
        help="First game index for deterministic sharding; each game's seeds are derived from its index.",
    )
    parser.add_argument("--max-turn", type=int, default=DEFAULT_MAX_TURN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--long-min", type=int, default=DEFAULT_LONG_MIN)
    parser.add_argument(
        "--opening-mode", type=str, default="varied", choices=("varied", "fixed"),
        help="Full-game frame fix (train/opening_source.py). \"varied\" (default) starts "
        "each game from a fresh, per-game_index-seeded engine roll (gold 10, empty team, "
        "before any purchase; engine._roll_shop_slots, the SAME roll mechanism a real "
        "turn-2+ advance uses), instead of always the same hardcoded FIXTURE_PATH shop. "
        "\"fixed\" restores the OLD single-fixture-every-game behavior (byte-for-byte), "
        "kept for parity/debug.",
    )
    parser.add_argument(
        "--opponent-pack", type=str, default=None,
        help="Restrict chain opponents to this game-level pack.",
    )
    parser.add_argument(
        "--opponent-rank-min", type=int, default=None,
        help="Inclusive minimum game-level rank for chain opponents; excludes unranked games.",
    )
    parser.add_argument(
        "--opponent-rank-max", type=int, default=None,
        help="Inclusive maximum game-level rank for chain opponents; excludes unranked games.",
    )
    parser.add_argument(
        "--opponent-split", type=str, default=None, choices=list(SPLIT_NAMES),
        help="Split used for chain opponents; Arena opponents have a separate split option.",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional output path for per-game JSONL")
    parser.add_argument(
        "--afterstate-out", type=Path, default=None,
        help="optional path, APPEND-opened, to stream one JSONL row per played turn's pre-battle afterstate (the exact state handed to _resolve_versus_turn, meta.versus intact) plus driver-true race scalars (lives/opp_lives at the END_TURN decision, pre-battle cumulative wins -- Vic semantics, RESULTS_W1 finding 2), plus one final {win, end_reason, turns_survived} row per game. Written + flushed per row, never buffered across games (unlike --render-dir/capture_detail) -- see module docstring 'We9 addition'. Off by default; every existing run is byte-for-byte unaffected.",
    )
    parser.add_argument(
        "--teacher-record-out", type=Path, default=None,
        help="Write per-candidate rollout targets for value training.",
    )
    parser.add_argument(
        "--segment-record-out", type=Path, default=None,
        help="optional GZIP JSONL stream with one row per real honest-frame segment decision. Keeps the real start and committed result separate from imagined prefix-completion afterstates, records all raw candidate chains and deduplicated groups, and adds play-independent S12/S16 label selection. Requires search+vgame+segmented-honest.",
    )
    parser.add_argument(
        "--teacher-rollouts", type=int, default=DEFAULT_TEACHER_ROLLOUTS,
        help="rollout repeats per candidate AT LABEL TIME, replacing --rollout-repeats for the recorded run. Separate knob because the recorded score is a regression TARGET (its standard error bounds what a distilled V can reproduce), not just a move choice, so it wants a bigger budget than the deployable preset's 4 without editing that preset.",
    )
    parser.add_argument(
        "--rollout-crn", action=argparse.BooleanOptionalAction, default=None,
        help="common random numbers across the SIBLING candidates of one decision. The r-th rollout of every candidate at a decision then shares the same future (random-fallback opponent draw, pool_random/retrieval pid draw, and the continuation's engine/shop seed), keyed on (seed, game_index, turn, repeat) with the candidate index deliberately absent; different repeats/turns/games stay independent. The battle oracle's own Monte-Carlo draw is NOT controllable (the JS simulator takes no seed) and remains the residual noise source. Default: ON when --teacher-record-out is set, OFF otherwise (so no existing measurement frame moves).",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="We2: torch.set_num_threads() pin, default 1. Not primarily a performance "
        "knob (it also happens to be faster here, not slower, for this small a model): "
        "BcRecommender's forward pass (bc_recommender.py::_masked_action_probs) is NOT "
        "bit-reproducible across separate calls under torch's default multi-threaded "
        "matmul (verified: two in-process decodes of the same state diverged at 32 "
        "threads, matched bit-for-bit at 1) -- this only matters when a near-tied "
        "top-2 action probability flips order between calls, rare per decode step but "
        "not rare over a whole game. NOTE this alone does NOT make a game reproducible "
        "end to end -- the battle oracle is a separate, unseeded source of "
        "non-determinism this flag cannot fix (see `render_selected_games`'s "
        "docstring); it only removes the decode side's contribution. Mirrors "
        "`train_chain_bc.py --torch-threads`'s existing pin for the same reason, at "
        "training time.",
    )
    parser.add_argument(
        "--report-json", type=Path, default=None,
        help="We2: optional path to write the aggregate metrics report (winrate + CI, "
        "lives, fallback, end/stop-reason distributions) as JSON",
    )
    parser.add_argument(
        "--render-games", type=int, default=3,
        help="We2: number of representative games to render (only takes effect if "
        "--render-dir is also given)",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=None,
        help="We2: base output dir for rendered galleries "
        "(one whole-game image per game at <dir>/images/full_game_<pid8>.png + a "
        "<dir>/images/full_game_<pid8>.json sidecar, summary at <dir>/render_report.json); "
        "rendering is SKIPPED entirely unless this is set",
    )
    parser.add_argument(
        "--render-from-jsonl", type=Path, default=None,
        help="We2: re-render galleries from an EXISTING run's per-game JSONL (a prior "
        "--out file that was written with detail capture on) WITHOUT replaying any game "
        "-- the only faithful way to regenerate a specific run's galleries, since the "
        "battle oracle is non-deterministic (see render_selected_games). Requires "
        "--render-dir; skips the eval, the BC checkpoint, and the chain snapshot load.",
    )
    parser.add_argument(
        "--decode-mode", type=str, default=BC_DEFAULT_DECODE_MODE, choices=list(BC_DECODE_MODES),
        help="ranked scans legal actions in BC probability order; sample draws from the masked distribution and retries visited-state collisions.",
    )
    parser.add_argument(
        "--sample-temperature", type=float, default=BC_DEFAULT_SAMPLE_TEMPERATURE,
        help="--decode-mode sample only: softmax temperature applied to the masked action "
        "distribution before sampling (1.0 = distribution as predicted, unchanged).",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=BC_DEFAULT_SAMPLE_SEED,
        help="--decode-mode sample only: base seed for the per-(game, turn, step) "
        "deterministic draw (see bc_recommender.py::_sample_rng). Independent of --seed, "
        "which only controls opponent/engine draws.",
    )
    parser.add_argument(
        "--llm-config", type=Path, default=None,
        help="path to a JSON file matching LlmAgentConfig's fields (loaded via LlmAgentConfig.from_json_file). Omit to use LlmAgentConfig()'s own defaults plus whatever --llm-* overrides below are given.",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="overrides LlmAgentConfig.model (default: from --llm-config, else LlmAgentConfig's own default deepseek-v4-flash).",
    )
    parser.add_argument(
        "--llm-fallback", type=str, default=None, choices=("end_turn", "bc"),
        help="overrides LlmAgentConfig.fallback_mode (default when neither this flag nor --llm-config set one: LlmAgentConfig's own default, \"end_turn\"). \"end_turn\" -- every fallback ladder rung (illegal-chain retries exhausted, turn budget exhausted, provider error) submits a bare END_TURN. \"bc\" -- fall back to the BcRecommender this driver also constructs for that purpose (this is the ONE case where --recommender llm still loads --bc-checkpoint/imports torch).",
    )
    parser.add_argument(
        "--llm-cache-dir", type=Path, default=None,
        help="overrides LlmAgentConfig.llm_cache_dir (the RecordedReplayClient cache directory; required whenever --llm-cache-mode is not \"off\").",
    )
    parser.add_argument(
        "--llm-cache-mode", type=str, default=None, choices=("off", "record", "replay", "record_missing"),
        help="overrides LlmAgentConfig.llm_cache_mode (default when neither this flag nor --llm-config set one: LlmAgentConfig's own default, \"off\") -- see llm_agent/client.py::RecordedReplayClient for the 3 non-off modes (record writes, replay reads-only-ever and hard-raises on a miss, record_missing reads-if-cached else writes).",
    )
    parser.add_argument(
        "--llm-transcript", type=Path, default=None,
        help="per-turn JSONL transcript path (llm_agent/transcript.py::TranscriptWriter). Default: <out>.transcript.jsonl next to --out when --out is given; transcripting stays off if neither is set.",
    )
    parser.add_argument(
        "--llm-provider", type=str, default=None, choices=("openai_compatible", "mock"),
        # Hidden CLI escape hatch for zero-cost MockLLM E2E smokes (see
        # build_mock_end_turn_client()) -- intentionally undocumented in --help.
        help=argparse.SUPPRESS,
    )
    return parser


def _run_render_only(args: argparse.Namespace) -> None:
    """`--render-from-jsonl` mode: re-render galleries from an existing run's
    captured JSONL, replaying NOTHING (no eval, no BC, no snapshot load)."""
    if args.render_dir is None:
        raise SystemExit("--render-from-jsonl requires --render-dir")
    print(f"render_from_jsonl:{args.render_from_jsonl}", flush=True)
    results = load_results_jsonl(args.render_from_jsonl)
    print(f"loaded_results:n_games={len(results)}", flush=True)
    render_report = render_selected_games(results, k=int(args.render_games), out_dir=args.render_dir)
    args.render_dir.mkdir(parents=True, exist_ok=True)
    render_report_path = args.render_dir / "render_report.json"
    render_report_path.write_text(json.dumps(render_report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"render_report_written:{render_report_path}", flush=True)


def _build_arena_pool(
    args: argparse.Namespace,
) -> tuple[ChainSnapshotSource | None, dict[int, int] | None]:
    """The arena ruler's own opponent pool (`--opponent-mode arena`), or
    (None, None) under chain mode.

    Factored out of `main()` so both the flag wiring and the two
    only-applies-to-arena refusals are reachable from a test without loading
    a checkpoint. Raises `SystemExit` for a misapplied flag, matching the
    rest of this module's flag rejections."""
    if args.opponent_mode != OPPONENT_MODE_ARENA:
        for flag, value in (
            ("--arena-pool-rank-max", args.arena_pool_rank_max),
            ("--arena-pool-split", args.arena_pool_split),
        ):
            if value is not None:
                raise SystemExit(f"{flag} only applies to --opponent-mode arena")
        return None, None

    print(
        f"loading_arena_pool:{args.chain_snapshot}:"
        f"splits={args.arena_pool_split or 'all'}:"
        f"opponent_pack={args.opponent_pack}:arena_pool_rank_max={args.arena_pool_rank_max}",
        flush=True,
    )
    t0 = time.monotonic()
    arena_source = ChainSnapshotSource(
        args.chain_snapshot,
        long_min=int(args.long_min),
        seed=int(args.seed),
        split=args.arena_pool_split,
        opponent_pack=args.opponent_pack,
        opponent_rank_min=None,
        opponent_rank_max=args.arena_pool_rank_max,
    )
    arena_pool_size_by_turn = {int(t): len(rows) for t, rows in arena_source.by_turn.items()}


    pool_depth = arena_source.max_turn_with_candidates
    effective = effective_arena_max_turn(
        int(args.max_turn), arena_source, opponent_mode=OPPONENT_MODE_ARENA
    )
    print(
        f"arena_pool_loaded:games={len(arena_source.by_pid)}:"
        f"split={arena_source.split}:splits_source={arena_source.splits_source}:"
        f"turn1_candidates={len(arena_source.by_turn.get(1, []))}:"
        f"max_turn_with_candidates={pool_depth if pool_depth is not None else 0}:"
        f"configured_max_turn={int(args.max_turn)}:"
        f"effective_max_turn={effective}:"
        f"turn_cap_bound_by={'pool_depth' if effective < int(args.max_turn) else 'configured'}:"
        f"elapsed={time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return arena_source, arena_pool_size_by_turn


def _arena_pool_report_fields(
    args: argparse.Namespace,
    arena_source: ChainSnapshotSource | None,
) -> dict[str, Any]:
    """The `arena_pool_*` block of `--report-json`.

    Every key is present regardless of `--opponent-mode` (None under chain,
    where no arena pool exists), matching the metadata block's own
    "diffable field-by-field" convention."""
    effective = (
        None
        if arena_source is None
        else effective_arena_max_turn(
            int(args.max_turn), arena_source, opponent_mode=OPPONENT_MODE_ARENA
        )
    )
    return {
        "arena_pool_rank_max": args.arena_pool_rank_max,
        "arena_pool_splits": (None if arena_source is None else (arena_source.split or "all")),
        "arena_pool_splits_source": (
            None if arena_source is None or arena_source.split is None else arena_source.splits_source
        ),
        "arena_pool_opponent_pack": (args.opponent_pack if arena_source is not None else None),
        "arena_pool_games": (len(arena_source.by_pid) if arena_source is not None else None),
        "arena_pool_snapshot_version": (
            arena_source.snapshot_version if arena_source is not None else None
        ),
        "arena_pool_max_turn": (
            None if arena_source is None else arena_source.max_turn_with_candidates
        ),
        "arena_configured_max_turn": (None if arena_source is None else int(args.max_turn)),
        "arena_effective_max_turn": effective,
    }


def _check_game_rules_compatibility(args: argparse.Namespace) -> None:
    """Check game rules compatibility."""
    message = _game_rules_refusal(
        game_rules=getattr(args, "game_rules", GAME_RULES_VERSUS),
        opponent_mode=getattr(args, "opponent_mode", OPPONENT_MODE_CHAIN),
        scoring=(
            getattr(args, "search_scoring", None)
            if str(getattr(args, "recommender", "bc")) == "search"
            else None
        ),
    )
    if message is not None:
        raise SystemExit(message)


def _check_turn_mode_compatibility(args: argparse.Namespace) -> None:
    """`--stochastic-samples` is validated here too: it is a k, and a k below 1
    would silently score every stochastic prefix on zero completions."""
    if int(getattr(args, "stochastic_samples", DEFAULT_STOCHASTIC_SAMPLES)) < 1:
        raise SystemExit(
            f"--stochastic-samples must be >= 1 (got {args.stochastic_samples})"
        )
    message = _turn_mode_refusal(
        turn_mode=getattr(args, "turn_mode", DEFAULT_TURN_MODE),
        scoring=(
            getattr(args, "search_scoring", None)
            if str(getattr(args, "recommender", "bc")) == "search"
            else None
        ),
    )
    if message is not None:
        raise SystemExit(message)


SEGMENT_RECORD_RECOMMENDERS: tuple[str, ...] = ("search", "bc")


def segment_record_flag_problem(
    *, recommender: str, search_scoring: str, turn_mode: str
) -> str | None:
    """Why this flag combination may not write segment records, or None.

    WHAT IS STILL REQUIRED, AND WHY EACH ONE IS.

    `--turn-mode segmented-honest` binds for EVERY recommender. The rows are
    per-SEGMENT: `SegmentRecorder` opens one per `decide` call, records the
    imagined clone the recommender was handed and the `imagination_seed` that
    clone ran on, and chains `decision_start_real` to `committed_result_real`
    across the segment boundary. Under the determinized frame there is one
    whole-turn decision, no imagined clone and no boundary, so a row would be
    a differently-shaped object wearing the same schema version."""
    if recommender not in SEGMENT_RECORD_RECOMMENDERS:
        allowed = " ".join(SEGMENT_RECORD_RECOMMENDERS)
        return (
            f"--segment-record-out requires --recommender one of [{allowed}], "
            f"got {recommender!r}"
        )
    if turn_mode != TURN_MODE_SEGMENTED_HONEST:
        return (
            "--segment-record-out requires --turn-mode segmented-honest, got "
            f"{turn_mode!r}"
        )
    if recommender == "search" and search_scoring != SEARCH_SCORING_VGAME:
        return (
            "--segment-record-out with --recommender search requires "
            f"--search-scoring vgame, got {search_scoring!r}"
        )
    return None


def main() -> None:
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from .llm_recommender import FALLBACK_MODE_BC

    args = _build_arg_parser().parse_args()

    # Render-only re-run of an existing capture (no eval; see the flag's help).
    if args.render_from_jsonl is not None:
        _run_render_only(args)
        return

    if args.num_games is None:
        raise SystemExit("--num-games is required unless --render-from-jsonl is given")


    _check_game_rules_compatibility(args)
    _check_turn_mode_compatibility(args)


    if args.skip_imagined_validation:
        set_skip_imagined_validation(True)
        print("skip_imagined_validation:on", flush=True)

    fixture_state = load_initial_state_from_fixture(FIXTURE_PATH)
    _assert_fixture_shape(fixture_state)

    # Full-game frame fix (train/opening_source.py): "varied" (default)
    # builds a `VariedOpeningSource` (a per-game_index-seeded engine roll,
    # no file/pool dependency); "fixed" keeps the untouched single-fixture
    # `fixture_state` loaded above, wrapped so `play_one_game` sees the
    # same `.state_for_game()` interface either way.
    opening_mode = str(args.opening_mode)
    if opening_mode == "varied":
        opening: VariedOpeningSource | FixedOpeningSource = build_varied_opening_source()
    else:
        opening = fixed_opening_source(fixture_state)
        print(f"opening_source_built:mode=fixed:fixture={FIXTURE_PATH}", flush=True)


    early_llm_config: LlmAgentConfig | None = None
    if args.recommender == "llm":
        early_llm_config = _build_llm_agent_config(args, transcript_path=_resolve_llm_transcript_path(args))


    needs_bc = args.recommender in ("bc", "search") or (
        args.recommender == "llm" and early_llm_config is not None and early_llm_config.fallback_mode == FALLBACK_MODE_BC
    )

    bc: BcRecommender | None = None
    if needs_bc:
        # Pin BEFORE constructing BcRecommender (see --torch-threads' own help
        # string for why this is a correctness pin, not a speed knob).
        import torch

        torch.set_num_threads(int(args.torch_threads))

        print(f"loading_bc_checkpoint:{args.bc_checkpoint}", flush=True)
        print(
            f"decode_mode={args.decode_mode}:sample_temperature={args.sample_temperature}:"
            f"sample_seed={args.sample_seed}",
            flush=True,
        )
        t0 = time.monotonic()
        bc = BcRecommender(
            args.bc_checkpoint,
            decode_mode=args.decode_mode,
            sample_temperature=args.sample_temperature,
            sample_seed=args.sample_seed,
        )
        print(f"bc_checkpoint_loaded:elapsed={time.monotonic() - t0:.1f}s", flush=True)


    print(
        f"loading_chain_snapshot:{args.chain_snapshot}:opponent_split={args.opponent_split}:"
        f"opponent_pack={args.opponent_pack}:opponent_rank_min={args.opponent_rank_min}:"
        f"opponent_rank_max={args.opponent_rank_max}",
        flush=True,
    )
    t0 = time.monotonic()
    opp_source = ChainSnapshotSource(
        args.chain_snapshot,
        long_min=int(args.long_min),
        seed=int(args.seed),
        split=args.opponent_split,
        opponent_pack=args.opponent_pack,
        opponent_rank_min=args.opponent_rank_min,
        opponent_rank_max=args.opponent_rank_max,
    )
    print(
        f"chain_snapshot_loaded:games={len(opp_source.by_pid)}:"
        f"long_pids={len(opp_source.long_pids)}:split={opp_source.split}:"
        f"splits_source={opp_source.splits_source}:elapsed={time.monotonic() - t0:.1f}s",
        flush=True,
    )


    arena_source, arena_pool_size_by_turn = _build_arena_pool(args)


    teacher_record_on = args.teacher_record_out is not None
    if teacher_record_on and (
        args.recommender != "search" or args.search_scoring != SEARCH_SCORING_ROLLOUT
    ):
        raise SystemExit(
            "--teacher-record-out requires --recommender search --search-scoring rollout"
        )
    segment_record_on = args.segment_record_out is not None
    if segment_record_on:
        problem = segment_record_flag_problem(
            recommender=args.recommender,
            search_scoring=args.search_scoring,
            turn_mode=args.turn_mode,
        )
        if problem is not None:
            raise SystemExit(problem)
    rollout_crn = bool(teacher_record_on if args.rollout_crn is None else args.rollout_crn)
    rollout_repeats = int(args.teacher_rollouts if teacher_record_on else args.rollout_repeats)


    vgame_scorer = None
    if args.search_scoring == SEARCH_SCORING_VGAME:
        from .vgame_scorer import VGameLeafScorer

        if args.recommender != "search":
            raise SystemExit("--search-scoring vgame requires --recommender search")
        if not args.vgame_heads:
            raise SystemExit("--search-scoring vgame requires --vgame-heads")
        vgame_scorer = VGameLeafScorer.from_checkpoints(
            args.vgame_heads,
            args.vgame_extractor or args.bc_checkpoint,
            blend=float(args.vgame_blend),
            pessimism=float(args.vgame_pessimism),
        )
        print(f"vgame_leaf_loaded:{json.dumps(vgame_scorer.describe(), sort_keys=True)}", flush=True)
    elif args.vgame_heads:
        raise SystemExit("--vgame-heads requires --search-scoring vgame")

    recommender: BcRecommender | SearchRecommender | LlmRecommender
    llm_config: LlmAgentConfig | None = None
    if args.recommender == "search":
        recommender = SearchRecommender(
            bc,
            n_candidates=int(args.search_candidates),
            ksim=int(args.search_ksim),
            seed=int(args.seed),
            scoring=args.search_scoring,
            rollout_shortlist=int(args.rollout_shortlist),
            rollout_repeats=rollout_repeats,
            rollout_ksim=int(args.rollout_ksim),


            rollout_opponent_mode=args.rollout_opponent_mode,


            opp_source=opp_source,
            max_turn=int(args.max_turn),

            rollout_crn=rollout_crn,
            capture_teacher_record=teacher_record_on,

            vgame_scorer=vgame_scorer,


            turn_mode=args.turn_mode,
            stochastic_samples=int(args.stochastic_samples),


            game_rules=str(args.game_rules),
            arena_race_convention=str(args.arena_race_convention),
            mc_rerank_k=int(args.mc_rerank_k),


            completion_policy=str(args.completion_policy),
            completion_width=int(args.completion_width),
            completion_aggregate=str(args.completion_aggregate),

            capture_candidate_chains=bool(
                args.capture_candidate_chains or segment_record_on
            ),
        )
        print(
            f"search_recommender_wrapped:n_candidates={args.search_candidates}:"
            f"ksim={args.search_ksim}:scoring={args.search_scoring}:"
            f"rollout_shortlist={args.rollout_shortlist}:rollout_repeats={rollout_repeats}:"
            f"rollout_ksim={args.rollout_ksim}:rollout_opponent_mode={args.rollout_opponent_mode}:"
            f"rollout_crn={rollout_crn}:teacher_record={teacher_record_on}:"
            f"vgame_leaf={'yes' if vgame_scorer is not None else 'no'}:"
            f"turn_mode={args.turn_mode}:stochastic_samples={args.stochastic_samples}:"
            f"completion_policy={args.completion_policy}:"
            f"completion_width={args.completion_width}:"
            f"completion_aggregate={args.completion_aggregate}:"
            f"game_rules={args.game_rules}:mc_rerank_k={args.mc_rerank_k}",
            flush=True,
        )
        if teacher_record_on and int(args.rollout_shortlist) < int(args.search_candidates):
            # Not an error (a narrower shortlist is a legitimate, cheaper
            # recording), but it means most candidates carry NO teacher score
            # and the decision's within-group contrast is only as wide as the
            # shortlist -- which is the whole point of route a, so say so.
            print(
                f"teacher_record_warning:shortlist={args.rollout_shortlist}"
                f"<candidates={args.search_candidates}:"
                "only shortlisted candidates will carry a teacher score",
                flush=True,
            )
    elif args.recommender == "llm":


        assert early_llm_config is not None
        llm_config = early_llm_config
        recommender = _build_llm_recommender(llm_config, bc=bc)
        print(
            f"llm_recommender_built:provider={llm_config.provider}:model={llm_config.model}:"
            f"fallback_mode={llm_config.fallback_mode}:llm_cache_mode={llm_config.llm_cache_mode}:"
            f"llm_cache_dir={llm_config.llm_cache_dir}:transcript_path={llm_config.transcript_path}",
            flush=True,
        )
    else:
        # needs_bc is True for plain "bc", so this is never None here.
        recommender = bc

    want_render = args.render_dir is not None and int(args.render_games) > 0

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.out.open("w", encoding="utf-8") if args.out else None


    if args.afterstate_out is not None:
        args.afterstate_out.parent.mkdir(parents=True, exist_ok=True)
    afterstate_out_fh = args.afterstate_out.open("a", encoding="utf-8") if args.afterstate_out else None


    if args.teacher_record_out is not None:
        args.teacher_record_out.parent.mkdir(parents=True, exist_ok=True)
    teacher_writer = (
        TeacherRecordWriter(args.teacher_record_out) if args.teacher_record_out else None
    )
    if args.segment_record_out is not None:
        args.segment_record_out.parent.mkdir(parents=True, exist_ok=True)
    segment_writer = (
        TeacherRecordWriter(args.segment_record_out) if args.segment_record_out else None
    )
    try:
        results = run_versus_eval(
            recommender,
            opp_source,
            opening,
            num_games=int(args.num_games),
            max_turn=int(args.max_turn),
            seed=int(args.seed),
            # FIX 2: no parse_cache -- the snapshot source is fully
            # pre-parsed, so a cache is never read, only written into
            # (unbounded growth over N games). None makes that write a no-op.
            parse_cache=None,
            out_fh=out_fh,
            # Only pay for full per-turn board capture when a gallery was
            # actually requested -- see `render_selected_games`'s docstring
            # for why rendering reads off THIS pass instead of re-running.
            capture_detail=want_render,
            game_index_start=int(args.game_index_start),
            opponent_mode=args.opponent_mode,
            arena_source=arena_source,
            afterstate_out_fh=afterstate_out_fh,
            teacher_writer=teacher_writer,
            segment_writer=segment_writer,
            game_rules=args.game_rules,
            arena_race_convention=args.arena_race_convention,
            turn_mode=args.turn_mode,
        )
    finally:
        if out_fh is not None:
            out_fh.close()
        if afterstate_out_fh is not None:
            afterstate_out_fh.close()
        if teacher_writer is not None:
            teacher_writer.close()
        if segment_writer is not None:
            segment_writer.close()

    _print_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)


    print(f"battle_worker_stats:{json.dumps(battle_worker_stats(), sort_keys=True)}", flush=True)

    if args.report_json is not None:
        report = _compute_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)


        report["metadata"] = {
            "opponent_split": args.opponent_split,
            "opponent_pack": args.opponent_pack,
            "opponent_rank_min": args.opponent_rank_min,
            "opponent_rank_max": args.opponent_rank_max,
            "chain_snapshot": str(args.chain_snapshot),


            "chain_pool_games": len(opp_source.by_pid),
            "bc_checkpoint": str(args.bc_checkpoint),
            "num_games": int(args.num_games),
            "seed": int(args.seed),
            "long_min": int(args.long_min),

            "recommender": args.recommender,
            "search_candidates": int(args.search_candidates),
            "search_ksim": int(args.search_ksim),

            "decode_mode": args.decode_mode,
            "sample_temperature": float(args.sample_temperature),
            "sample_seed": int(args.sample_seed),

            "search_scoring": args.search_scoring,
            "rollout_shortlist": int(args.rollout_shortlist),
            # The EFFECTIVE repeats this run actually used: --teacher-rollouts
            # replaces --rollout-repeats whenever --teacher-record-out is set,
            # so echoing the raw flag here would misdescribe a recorded run.
            "rollout_repeats": int(rollout_repeats),
            "rollout_repeats_flag": int(args.rollout_repeats),
            "rollout_ksim": int(args.rollout_ksim),


            "rollout_crn": bool(rollout_crn),
            "teacher_record_out": (
                str(args.teacher_record_out) if args.teacher_record_out is not None else None
            ),
            "segment_record_out": (
                str(args.segment_record_out) if args.segment_record_out is not None else None
            ),
            "teacher_rollouts": int(args.teacher_rollouts),


            "llm_provider": (llm_config.provider if llm_config is not None else None),
            "llm_model": (llm_config.model if llm_config is not None else None),
            "llm_config_path": (str(args.llm_config) if args.llm_config is not None else None),
            "llm_fallback_mode": (llm_config.fallback_mode if llm_config is not None else None),
            "llm_cache_mode": (llm_config.llm_cache_mode if llm_config is not None else None),
            "llm_cache_dir": (llm_config.llm_cache_dir if llm_config is not None else None),
            "llm_transcript_path": (llm_config.transcript_path if llm_config is not None else None),
            "llm_cost_totals": (recommender.cost_totals() if _is_llm_recommender(recommender) else None),
            "llm_fallback_reason_counts": (
                recommender.fallback_reason_counts() if _is_llm_recommender(recommender) else None
            ),

            "rollout_opponent_mode": args.rollout_opponent_mode,


            "vgame": (vgame_scorer.describe() if vgame_scorer is not None else None),
            # Full-game frame fix (train/opening_source.py)
            "opening_mode": opening_mode,


            "opponent_mode": args.opponent_mode,


            "game_rules": args.game_rules,
            "arena_race_convention": (
                args.arena_race_convention if args.game_rules == GAME_RULES_ARENA else None
            ),


            "turn_mode": args.turn_mode,
            "stochastic_samples": int(args.stochastic_samples),
            "mc_rerank_k": int(args.mc_rerank_k),


            "completion_policy": str(args.completion_policy),
            "completion_width": int(args.completion_width),
            "completion_aggregate": str(args.completion_aggregate),


            "capture_candidate_chains": bool(
                args.capture_candidate_chains or segment_record_on
            ),


            "skip_imagined_validation": bool(args.skip_imagined_validation),
            **_arena_pool_report_fields(args, arena_source),


            "search_rollout_opponent_pool": (
                "chain_pool"
                if (args.recommender == "search" and args.search_scoring == "rollout")
                else None
            ),


            "fallback_rng": "per_game",
            "game_index_start": int(args.game_index_start),


            "battle_worker": battle_worker_stats(),
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report_json_written:{args.report_json}", flush=True)

    if want_render:
        render_report = render_selected_games(results, k=int(args.render_games), out_dir=args.render_dir)
        args.render_dir.mkdir(parents=True, exist_ok=True)
        render_report_path = args.render_dir / "render_report.json"
        render_report_path.write_text(json.dumps(render_report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"render_report_written:{render_report_path}", flush=True)


if __name__ == "__main__":
    main()
