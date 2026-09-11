"""Search over behavior-cloned action chains.

With segmented-honest value scoring, candidates are grouped by their prefix up
to the first structural chance node or end of turn. Chance outcomes are sampled
on independent simulation streams. Each outcome is completed by BC, optionally
choosing the best of several continuations with V; outcome values are averaged
to rank prefixes. The caller executes the selected prefix and re-searches after
observing the real outcome.

The module also supports myopic battle scoring and rollout scoring. All search
walks use copied states; deterministic seed namespaces preserve reproducibility."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from typing import Any, Callable

import numpy as np


from ..api import imagined_step as engine_step
from ..oracles.sap_calc_battle_oracle import build_simulation_config
from ..train.env import ACTION_CATALOG, TrainingEnv
from ..visited_guard import set_training_rolls_this_turn, state_signature
from . import honest_frame
from .bc_recommender import DECODE_MODE_RANKED as _BC_DECODE_MODE_RANKED
from .bc_recommender import DEFAULT_MAX_CHAIN_STEPS as _BC_DEFAULT_MAX_CHAIN_STEPS
from .bc_recommender import legal_mask

# Constructor defaults, exported so callers (the eval driver's argparse
# defaults) share one source of truth instead of a second copy of "12"/"16".
DEFAULT_N_CANDIDATES = 12
DEFAULT_KSIM = 16


DEFAULT_ANYTIME_CHUNK = 1
# Two group scores within this of each other are a TIE for the anytime
# path's argmax -- see `_search_anytime` for the batch-shape float noise
# it exists to absorb.
SCORE_TIE_EPS = 1e-6
DEFAULT_SEED = 0


SCORING_MYOPIC = "myopic"
SCORING_ROLLOUT = "rollout"


SCORING_VGAME = "vgame"
SCORING_MODES: tuple[str, ...] = (SCORING_MYOPIC, SCORING_ROLLOUT, SCORING_VGAME)
DEFAULT_SCORING = SCORING_MYOPIC
DEFAULT_ROLLOUT_SHORTLIST = 4
DEFAULT_ROLLOUT_REPEATS = 4

#: The frame the ROLLOUT continuation resolves under. Plain strings rather than
#: imports from `eval_versus_fullgame`, which this module may only import lazily
#: (see the lazy import inside `_rollout_score_candidate`); they are checked
#: against that module's own constants at the point of use.
SEARCH_GAME_RULES_VERSUS = "versus"
SEARCH_GAME_RULES_ARENA = "arena"
SEARCH_GAME_RULES = (SEARCH_GAME_RULES_VERSUS, SEARCH_GAME_RULES_ARENA)
DEFAULT_SEARCH_GAME_RULES = SEARCH_GAME_RULES_VERSUS
#: Mirrors `eval_versus_fullgame.DEFAULT_ARENA_RACE_CONVENTION`.
DEFAULT_SEARCH_ARENA_RACE_CONVENTION = "trophies_mapped"


DEFAULT_MC_RERANK_K = 0
DEFAULT_ROLLOUT_KSIM = 16
# Matches `eval_versus_fullgame.DEFAULT_MAX_TURN`; not imported from there
# (module-level import would cycle -- that module imports THIS one at its
# own top level) -- every real caller (`eval_versus_fullgame.py::main`)
# passes its own `--max-turn` explicitly, so this constant only matters for
# a caller/test that omits it.
DEFAULT_ROLLOUT_MAX_TURN = 30


ROLLOUT_OPPONENT_TRUE = "true"
ROLLOUT_OPPONENT_POOL_RANDOM = "pool_random"
ROLLOUT_OPPONENT_RETRIEVAL = "retrieval"
ROLLOUT_OPPONENT_MODES: tuple[str, ...] = (
    ROLLOUT_OPPONENT_TRUE,
    ROLLOUT_OPPONENT_POOL_RANDOM,
    ROLLOUT_OPPONENT_RETRIEVAL,
)
DEFAULT_ROLLOUT_OPPONENT_MODE = ROLLOUT_OPPONENT_TRUE


DEFAULT_ROLLOUT_CRN = False
DEFAULT_CAPTURE_TEACHER_RECORD = False
# Salts, exported so the checker tool can re-derive a recorded key instead of
# hard-coding a second copy of the format string.
CRN_FALLBACK_SALT = "exp12_wa_crn_fallback"
CRN_OPPONENT_SALT = "exp12_wa_crn_opponent"
CRN_ENGINE_SALT = "exp12_wa_crn_engine"
# Width of the CRN continuation seed, in bits -- see `_crn_engine_seed`.
# Exported so the test that pins "the range the engine accepts" and this
# derivation share one number instead of two copies of it.
CRN_ENGINE_SEED_BITS = 63
TEACHER_RECORD_SCHEMA = "exp12-teacher-record/v1"


RESAMPLABLE_EMPTY_STOPS: frozenset[str] = frozenset(
    {"sampled_cycle", "sampled_action_illegal", "sampled_action_error"}
)


UNFINISHED_WALK_STOPS: frozenset[str] = frozenset(
    {
        "cap_reached",
        "legal_mask_failed",
        "no_legal_actions",
        "sampled_no_action_left",
        "sampled_action_error",
        "sampled_action_illegal",
        "sampled_cycle",
        "structural_boundary",
    }
)
MAX_SAMPLE_ATTEMPTS = 8


MAX_EXTRA_SAMPLE_LEVELS = 4096


COMPLETION_TAIL_STOP = "stop"
#: Let the walk stop, then hand the tail to the greedy decoder -- what index 0
#: already does. Closes comparability; costs diversity, because every
#: alternative then ends the same way.
COMPLETION_TAIL_GREEDY = "finish_greedy"


COMPLETION_TAIL_RESAMPLE = "resample"
COMPLETION_TAILS = (
    COMPLETION_TAIL_STOP,
    COMPLETION_TAIL_GREEDY,
    COMPLETION_TAIL_RESAMPLE,
)

COMPLETION_BC_GREEDY = "bc_greedy"
COMPLETION_V_SEARCH = "v_search"
COMPLETION_POLICIES = (COMPLETION_BC_GREEDY, COMPLETION_V_SEARCH)


COMPLETION_AGG_MAX = "max"
COMPLETION_AGG_MEAN = "mean"
COMPLETION_AGGREGATES = (COMPLETION_AGG_MAX, COMPLETION_AGG_MEAN)


def completion_agg(values: list[float], aggregate: str = COMPLETION_AGG_MAX) -> float:
    """One imagined sample's inner completions -> one number.

    A single value returns unchanged, which is what makes `bc_greedy` (inner
    width 1) bit-identical to every run before deepening existed."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if aggregate == COMPLETION_AGG_MAX:
        return float(max(values))
    return sum(values) / float(len(values))

def _read_last_opponent_team(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """`state["meta"]["versus"]["last_opponent_team"]`, deep-copied, or None
    if absent/wrong-typed/empty (turn 1, before any battle has resolved --
    see `train/env.py::_set_last_opponent_team`, this field's only writer).
    Never raises: every level of the path is defensively type-checked.
    """
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return None
    versus = meta.get("versus")
    if not isinstance(versus, dict):
        return None
    team = versus.get("last_opponent_team")
    if not isinstance(team, list) or not team:
        return None
    return copy.deepcopy(team)


def _read_current_opponent_pid(state: dict[str, Any]) -> str | None:
    """Read current opponent pid."""
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return None
    versus = meta.get("versus")
    if not isinstance(versus, dict):
        return None
    pid = str(versus.get("current_opponent_participation_id") or "").strip()
    return pid or None


def _apply_chain(state: dict[str, Any], chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-apply `chain` (a list of action dicts, as found in a recommend-
    result's `chain_preview`) op-by-op from a FRESH `copy.deepcopy` of
    `state`, via `api.step` -- the identical replay procedure
    `tools/eval_versus_fullgame.py::play_one_game` itself uses on the
    winning chain (that function's per-turn loop: `ops = [op for op in
    chain_preview if ... != "END_TURN"]` then `engine_step(board, op)` in a
    loop). Reusing the SAME procedure here, rather than trusting whatever
    intermediate state a candidate generator tracked internally, guarantees
    the board this module scores a candidate on is bit-identical to the
    board the driver will actually reach if that candidate wins -- if the
    two diverged, search could pick a candidate based on a board the real
    replay never lands on.

    END_TURN is never applied (it does not change the shop-phase board;
    `play_one_game` skips it the same way). Stops at the first
    illegal/failing step -- the board reached so far still counts as this
    candidate's end board (a partially-applied chain is still a valid, if
    suboptimal, candidate rather than a hard failure).
    """
    work = copy.deepcopy(state)
    for action in chain:
        if not isinstance(action, dict):
            break
        if str(action.get("type") or "").strip().upper() == "END_TURN":
            break
        try:
            trans = engine_step(work, action)
        except Exception:
            break
        if not trans.get("legal"):
            break
        next_state = trans.get("state_after")
        if not isinstance(next_state, dict):
            break
        work = next_state
    return work


def _annotate_skip(greedy_result: dict[str, Any]) -> dict[str, Any]:
    """The greedy candidate's result, unchanged, with `search_*` keys added
    to record that NO search happened (turn 1 / no opponent yet, or every
    oracle call failed). Deliberately distinct from the `recommend()`-level
    exception fallback: this is a designed, expected outcome (not an
    error), so `search_error` is never set here.
    """
    result = dict(greedy_result)
    result.update(
        {
            "search_used": False,
            "search_n_candidates": 0,


            "search_n_generated": 0,
            "search_n_dedup": 0,
            "search_scores": [],
            "search_chosen_index": 0,
            "search_greedy_score": None,


            "search_candidate_chains": None,
        }
    )
    return result


def _searched_verdict(chosen_chain: list[dict[str, Any]]) -> dict[str, Any]:
    """The four keys a COMPLETED prefix search owns on its own result.

    An empty winning prefix is a CHOICE, not a failure. Its dedup key is
    `("det", signature(start board))` -- the same key an END_TURN-first
    candidate walks to -- and the board it was scored on is the decision's own
    start board, so what won is "end the turn now with what is on the board".
    Which of the two chains the group carries is an accident of which candidate
    index reached that key first, so this says it the way the other one would:
    the driver filters END_TURN out of the ops it commits, so the turn resolves
    with nothing committed either way, and `recommended_action` stops being
    `None` for the consumers that read it (`play_web`'s `/api/infer/recommend`
    would otherwise answer `ok` with no action in it).

    `diagnostics` stays the winning candidate's on purpose: it is decode
    telemetry about that chain, and `run_segmented_turn`'s stop-reason sink --
    hence every `stop_reasons` histogram already on disk -- reads `stop_reason`
    off it."""
    chain = chosen_chain if chosen_chain else [{"type": "END_TURN"}]
    return {
        "ok": True,
        "error": None,
        "chain_preview": chain,
        "recommended_action": copy.deepcopy(chain[0]),
    }


class SearchRecommender:
    """Searchrecommender."""

    def __init__(
        self,
        bc_recommender: Any,
        *,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        ksim: int = DEFAULT_KSIM,
        seed: int = DEFAULT_SEED,
        battle_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_chain_steps: int | None = None,
        scoring: str = DEFAULT_SCORING,
        rollout_shortlist: int = DEFAULT_ROLLOUT_SHORTLIST,
        rollout_repeats: int = DEFAULT_ROLLOUT_REPEATS,
        rollout_ksim: int = DEFAULT_ROLLOUT_KSIM,
        rollout_opponent_mode: str = DEFAULT_ROLLOUT_OPPONENT_MODE,
        opp_source: Any | None = None,
        max_turn: int = DEFAULT_ROLLOUT_MAX_TURN,
        capture_candidate_chains: bool = False,
        rollout_crn: bool = DEFAULT_ROLLOUT_CRN,
        capture_teacher_record: bool = DEFAULT_CAPTURE_TEACHER_RECORD,
        vgame_scorer: Any | None = None,
        turn_mode: str = honest_frame.DEFAULT_TURN_MODE,
        stochastic_samples: int = honest_frame.DEFAULT_STOCHASTIC_SAMPLES,
        completion_policy: str = COMPLETION_BC_GREEDY,
        completion_width: int = 1,
        completion_tail: str = COMPLETION_TAIL_RESAMPLE,
        completion_aggregate: str = COMPLETION_AGG_MAX,
        game_rules: str = DEFAULT_SEARCH_GAME_RULES,
        arena_race_convention: str = DEFAULT_SEARCH_ARENA_RACE_CONVENTION,
        mc_rerank_k: int = DEFAULT_MC_RERANK_K,
    ) -> None:
        # The SAME loaded model -- this class never loads a checkpoint of
        # its own, only ever calls into the one it was handed (see module
        # docstring's "candidate diversity" section for exactly how).
        self.bc = bc_recommender


        if game_rules not in SEARCH_GAME_RULES:
            raise ValueError(
                f"search_recommender_bad_game_rules:{game_rules!r}:"
                f"expected one of {SEARCH_GAME_RULES!r}"
            )
        self.game_rules = str(game_rules)
        self.arena_race_convention = str(arena_race_convention)
        self.mc_rerank_k = int(mc_rerank_k)
        if self.mc_rerank_k < 0:
            raise ValueError(
                f"search_recommender_bad_mc_rerank_k:{mc_rerank_k!r}:must_be_at_least_0"
            )
        self.n_candidates = int(n_candidates)
        self.ksim = int(ksim)
        self.seed = int(seed)
        # None => lazy-import `run_battle_oracle_with_config` on first use
        # (see `_resolve_battle_fn`); tests inject a stub here instead.
        self._battle_fn = battle_fn
        self.max_chain_steps = (
            int(max_chain_steps)
            if max_chain_steps is not None
            else int(getattr(bc_recommender, "max_chain_steps", _BC_DEFAULT_MAX_CHAIN_STEPS))
        )
        # Bumped once per `recommend()` call (once per turn during a game)
        # so each turn's candidate RNGs draw fresh entropy instead of
        # repeating the same per-candidate sequence turn after turn -- see
        # `_rng_for_candidate`.
        self._recommend_call_count = 0


        if scoring not in SCORING_MODES:
            raise ValueError(f"search_recommender_bad_scoring:{scoring!r}:expected_one_of={SCORING_MODES}")
        self.scoring = str(scoring)
        self.rollout_shortlist = int(rollout_shortlist)
        self.rollout_repeats = int(rollout_repeats)
        self.rollout_ksim = int(rollout_ksim)


        if rollout_opponent_mode not in ROLLOUT_OPPONENT_MODES:
            raise ValueError(
                f"search_recommender_bad_rollout_opponent_mode:{rollout_opponent_mode!r}:"
                f"expected_one_of={ROLLOUT_OPPONENT_MODES}"
            )
        self.rollout_opponent_mode = str(rollout_opponent_mode)
        self.max_turn = int(max_turn)
        if self.scoring == SCORING_ROLLOUT and opp_source is None:
            # Rollout scoring must sample the followed chain's actual
            # subsequent boards (module docstring's "actual-chain-only, no
            # future-averaging" section) -- unlike myopic scoring, which
            # only ever reads `state["meta"]["versus"]["last_opponent_team"]`
            # and needs no opponent source at all, so this is required ONLY
            # for this mode, not a blanket constructor requirement.
            raise ValueError("search_recommender_scoring_rollout_requires_opp_source")
        self.opp_source = opp_source

        self.capture_candidate_chains = bool(capture_candidate_chains)


        self.rollout_crn = bool(rollout_crn)
        self.capture_teacher_record = bool(capture_teacher_record)
        # Set by `set_decision_context` (the driver, once per game) and by
        # `recommend` (the turn, once per decision). Both None for a caller
        # that never sets them -- see `_crn_decision_id`.
        self._crn_game_index: int | None = None
        self._decision_turn: int | None = None


        self.vgame_scorer = vgame_scorer
        if self.scoring == SCORING_VGAME and vgame_scorer is None:
            raise ValueError("search_recommender_scoring_vgame_requires_vgame_scorer")
        self._race_wins: int | None = None


        self.turn_mode = honest_frame.normalize_turn_mode(turn_mode)
        self.honest = honest_frame.is_honest(self.turn_mode)
        self.stochastic_samples = int(stochastic_samples)
        if self.stochastic_samples < 1:
            raise ValueError(
                f"search_recommender_bad_stochastic_samples:{stochastic_samples!r}:must_be_at_least_1"
            )
        if self.mc_rerank_k and self.scoring != SCORING_VGAME:


            raise ValueError(
                f"search_recommender_mc_rerank_requires_vgame_scoring:{self.scoring!r}:"
                f"mc_rerank_k={self.mc_rerank_k} needs scoring={SCORING_VGAME!r} so the "
                f"shortlist is taken in V order"
            )
        if self.honest and self.scoring != SCORING_VGAME:
            # LOUD rather than silently biased: a myopic/rollout leaf under
            # the honest frame ranks candidates on the ONE proposal-stream
            # roll, i.e. it picks whichever candidate got lucky. Expectation
            # scoring (A1 ruling 3) is implemented for the V leaf only.
            raise ValueError(
                f"search_recommender_honest_frame_requires_vgame_scoring:{self.scoring!r}:"
                f"expectation scoring at the root is "
                f"implemented for scoring={SCORING_VGAME!r} only; a {self.scoring!r} leaf "
                f"under turn_mode={honest_frame.TURN_MODE_SEGMENTED_HONEST!r} would rank candidates on a "
                f"single imagined roll and systematically prefer the lucky one"
            )

        if completion_policy not in COMPLETION_POLICIES:
            raise ValueError(
                f"search_recommender_bad_completion_policy:{completion_policy!r}:"
                f"expected_one_of={COMPLETION_POLICIES}"
            )
        if completion_aggregate not in COMPLETION_AGGREGATES:
            raise ValueError(
                f"search_recommender_bad_completion_aggregate:{completion_aggregate!r}:"
                f"expected_one_of={COMPLETION_AGGREGATES}"
            )
        if completion_tail not in COMPLETION_TAILS:
            raise ValueError(
                f"search_recommender_bad_completion_tail:{completion_tail!r}:"
                f"expected_one_of={COMPLETION_TAILS}"
            )
        self.completion_tail = str(completion_tail)
        self.completion_policy = str(completion_policy)
        self.completion_aggregate = str(completion_aggregate)
        self.completion_width = int(completion_width)
        if self.completion_width < 1:
            raise ValueError(
                f"search_recommender_bad_completion_width:{completion_width!r}:must_be_at_least_1"
            )
        if self.completion_policy == COMPLETION_V_SEARCH:


            if self.completion_width < 2:
                raise ValueError(
                    f"search_recommender_v_search_needs_width_ge_2:{self.completion_width}:"
                    "width 1 proposes only the greedy completion, so the arm would "
                    "measure the bc_greedy policy under the v_search name"
                )
            if not self.honest:
                raise ValueError(
                    f"search_recommender_v_search_requires_honest_frame:{self.turn_mode!r}:"
                    "there is no chance node to search over under the determinized frame"
                )
            if self.scoring != SCORING_VGAME:
                raise ValueError(
                    f"search_recommender_v_search_requires_vgame_scoring:{self.scoring!r}"
                )
        elif self.completion_width != 1:
            raise ValueError(
                f"search_recommender_bc_greedy_width_must_be_1:{self.completion_width}:"
                "the greedy policy emits exactly one completion per imagined sample"
            )
        # Counted per decision, reported on the result, and gated upstream:
        # how often the inner search's pick differs from the greedy one.
        self._completion_divergent = 0
        self._completion_decided = 0
        # How many inner boards a sample ACTUALLY got, against how many the
        # config asked for. See `_imagined_completions` for why a configured
        # width is not evidence that the width happened.
        self._completion_boards = 0
        self._completion_samples = 0
        self._completion_dropped = 0
        # Direct evidence for the completion-policy contract. A returned board
        # is classified at the point where the walk's stop reason and tail
        # policy are still available, before V can score it.
        self._completion_terminal_boards = 0
        self._completion_unfinished_boards = 0


        self._completion_second_chance = 0


        self._completion_rescued = 0

        # Set by `set_imagination_context` (the driver, once per SEGMENT).
        # None => `_imagination_seed` falls back to the handed state's own
        # seed and says so in the diagnostics -- see that method.
        self._imagination_engine_seed: int | None = None
        self._imagination_segment_index: int = 0
        # The seed of the state the current `recommend()` was handed, read
        # defensively there; only ever used as the fallback key base above.
        self._state_seed: int = 0

    def set_decision_context(self, *, game_index: int | None, turn: int | None = None) -> None:
        """Called once per game by `eval_versus_fullgame.py::play_one_game`,
        duck-typed, so a `BcRecommender` or any other recommender that does
        not define it is simply not called. Never raises."""
        self._crn_game_index = None if game_index is None else int(game_index)
        if turn is not None:
            self._decision_turn = int(turn)

    def _crn_decision_id(self) -> str:
        """Stable id of the DECISION (never of a candidate) that every CRN
        key below is scoped to: `(game_index, turn)` when the driver has
        supplied both, else this process's recommend counter -- which is
        also constant across one decision's candidates and distinct across
        decisions, so the CRN property survives, only cross-process
        reproducibility does not.
        """
        if self._crn_game_index is not None and self._decision_turn is not None:
            return f"g{int(self._crn_game_index)}:t{int(self._decision_turn)}"
        return f"call{int(self._recommend_call_count)}"

    def _crn_fallback_key(self, *, candidate_index: int, repeat_index: int) -> str:
        """Seed string for a repeat's isolated random-fallback sampler.
        Under CRN the candidate index is ABSENT (that is the whole point);
        with CRN off this returns the pre-A0 key byte-for-byte."""
        if self.rollout_crn:
            return f"{CRN_FALLBACK_SALT}:{self.seed}:{self._crn_decision_id()}:{repeat_index}"
        return (
            f"w1_rollout_fallback:{self.seed}:{self._recommend_call_count}:"
            f"{candidate_index}:{repeat_index}"
        )

    def _crn_opponent_key(self, *, candidate_index: int, repeat_index: int) -> str:
        """Seed string for a repeat's `pool_random`/`retrieval` pid draw --
        same CRN rule as `_crn_fallback_key`."""
        if self.rollout_crn:
            return (
                f"{CRN_OPPONENT_SALT}:{self.rollout_opponent_mode}:{self.seed}:"
                f"{self._crn_decision_id()}:{repeat_index}"
            )
        return (
            f"w2_opponent_mode:{self.rollout_opponent_mode}:{self.seed}:"
            f"{self._recommend_call_count}:{candidate_index}:{repeat_index}"
        )

    def _crn_engine_seed(self, repeat_index: int) -> int:
        """The engine (shop) seed every sibling candidate's repeat-`r`
        continuation is started from under CRN. sha256 rather than
        `hash()` so it is stable across processes and python runs."""
        key = f"{CRN_ENGINE_SALT}:{self.seed}:{self._crn_decision_id()}:{repeat_index}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") >> (64 - CRN_ENGINE_SEED_BITS)

    def _resolve_battle_fn(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if self._battle_fn is None:
            from ..oracles.sap_calc_battle_oracle import run_battle_oracle_with_config

            self._battle_fn = run_battle_oracle_with_config
        return self._battle_fn

    def _rng_for_completion(self, sample_r: int, index: int) -> np.random.Generator:
        """RNG for inner completion `index` of imagined sample `sample_r`.

        Keyed on the IMAGINATION seed, not on the play seed and not on the
        group, for two reasons that are both load-bearing. Imagination,
        because a completion proposal drawn off the play stream would be
        peeking at the shop the agent has not rolled yet. Not the group,
        because the outer frame deliberately keys `(decision, r)` without the
        candidate index so siblings reaching the same chance node face the
        same resampled shop (CRN); drawing the inner proposals per group
        would put an independent noise draw back under the argmax that CRN
        exists to remove.
        """
        seed_seq = np.random.SeedSequence(
            [
                int(self.seed),
                int(self._recommend_call_count),
                int(self._imagination_seed(sample_r)) & 0xFFFFFFFF,
                int(index),
            ]
        )
        return np.random.default_rng(seed_seq)

    def _rng_for_candidate(self, index: int) -> np.random.Generator:
        """Deterministic-given-(seed, turn, candidate) RNG stream: distinct
        across candidates within one turn AND across turns within one game
        (via `_recommend_call_count`), so repeated turns in a long game do
        not all sample the exact same noise pattern.
        """
        seed_seq = np.random.SeedSequence([int(self.seed), int(self._recommend_call_count), int(index)])
        return np.random.default_rng(seed_seq)

    def _call_bc_recommend(
        self,
        state: dict[str, Any],
        *,
        deterministic: bool,
        force_ranked_decode: bool = False,
    ) -> dict[str, Any]:
        """Call `self.bc.recommend(state)` with `.deterministic` temporarily
        set to `deterministic`, restoring the ORIGINAL value in a `finally`
        (matches the spec this class was built from: candidate 0 always
        uses `deterministic=True`; `_generate_candidate`'s fallback path
        uses `deterministic=False`). A no-op toggle if `self.bc` has no
        `deterministic` attribute at all (defensive only)."""
        has_det = hasattr(self.bc, "deterministic")
        original_det = self.bc.deterministic if has_det else None
        has_decode = force_ranked_decode and hasattr(self.bc, "decode_mode")
        original_decode = self.bc.decode_mode if has_decode else None
        try:
            if has_det:
                self.bc.deterministic = bool(deterministic)
            if has_decode:
                self.bc.decode_mode = _BC_DECODE_MODE_RANKED
            return self.bc.recommend(state)
        finally:
            if has_det:
                self.bc.deterministic = original_det
            if has_decode:
                self.bc.decode_mode = original_decode

    def _sample_candidate(
        self,
        state: dict[str, Any],
        rng: np.random.Generator,
        *,
        completion_mode: bool = False,
        for_completion: bool = False,
    ) -> dict[str, Any]:
        """One sampled candidate: `_sample_walk`, drawn again while it is empty.

        So draw again. Each attempt is an independent walk from the SAME masked
        distribution, so what changes is how often this class of candidate
        occurs, not the distribution a returned chain is drawn from: the
        candidate is the sampler's own distribution conditioned on committing
        at least one op. It costs one extra decode on ~1 candidate in 5,000.

        Only DRAW rejections are retried (`RESAMPLABLE_EMPTY_STOPS`). A walk
        that had nothing to draw (`no_legal_actions`, `legal_mask_failed`)
        would draw the same nothing, and one that is still empty after
        `MAX_SAMPLE_ATTEMPTS` has had its chances; both come back `ok=False`
        exactly as before. An empty chain therefore stays REACHABLE, which is
        why `_searched_verdict` owns the search's verdict rather than trusting
        that this method no longer produces one.

        `diagnostics.sample_attempts` is how many walks this candidate cost, so
        a resample can never be invisible."""
        attempts = 0
        while True:
            attempts += 1
            chain, stop_reason = self._sample_walk(
                state, rng, completion_mode=completion_mode,
                for_completion=for_completion,
            )
            if (
                chain
                or attempts >= MAX_SAMPLE_ATTEMPTS
                or stop_reason not in RESAMPLABLE_EMPTY_STOPS
            ):
                break

        ok = bool(chain)
        return {
            "ok": ok,
            "error": None if ok else stop_reason,
            "recommended_action": copy.deepcopy(chain[0]) if chain else None,
            "chain_preview": chain,
            "wdl_probs": None,
            "diagnostics": {
                "stop_reason": stop_reason,
                "chain_length": len(chain),
                "sampled": True,
                "sample_attempts": attempts,
            },
        }

    def _sample_walk(
        self,
        state: dict[str, Any],
        rng: np.random.Generator,
        *,
        completion_mode: bool = False,
        for_completion: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """ONE stochastically-sampled walk from `state`: the same
        per-step machinery `BcRecommender.recommend` uses (`legal_mask`,
        `self.bc.encoder`/`self.bc.model` via `_masked_action_probs`, the
        within-turn `state_signature` anti-cycle set) but DRAWS each step
        from the masked categorical distribution (`rng.choice`) instead of
        always taking the arg-max -- see the module docstring's "candidate
        diversity" section.

        Unlike that method's guarded scan (which, on a would-be revisit,
        tries the NEXT-best legal action instead of stopping), a sampled
        walk that would revisit an already-seen board, or draws an action
        that turns out illegal to apply, simply STOPS there -- there is no
        well-defined "next best" for a single random draw the way there is
        for a full probability ranking. Stopping after at least one committed
        op still leaves a valid candidate -- `_apply_chain`'s own "the board
        reached so far still counts" contract applies equally here -- but
        stopping on the FIRST draw leaves nothing, which is what
        `_sample_candidate` draws again for.

        - the structural break's own justification is "everything past it is
          provably discarded", true of an outer candidate and FALSE of a
          completion, whose tail is exactly the board that gets scored;
        - the cycle stop gives up because a single draw has no well-defined
          "next best". A completion can simply mask that action and draw again
          from what is left, which keeps it a sample instead of a stump."""
        work = copy.deepcopy(state)
        set_training_rolls_this_turn(work, 0)  # same observation-parity reset BcRecommender.recommend uses
        chain: list[dict[str, Any]] = []
        visited: set[str] = {state_signature(work)}
        stop_reason = "cap_reached"

        for _ in range(self.max_chain_steps):
            try:
                mask = legal_mask(work)
            except Exception:
                stop_reason = "legal_mask_failed"
                break
            legal_idx = np.flatnonzero(mask)
            if legal_idx.size == 0:
                stop_reason = "no_legal_actions"
                break

            obs = self.bc.encoder.encode(work)
            probs = np.asarray(self.bc._masked_action_probs(obs, mask), dtype=np.float64)

            # `banned` only ever grows in completion mode, so with it empty
            # `pool is legal_idx` and every draw below is the identical call.
            banned: set[int] = set()
            step = None
            while True:
                pool = legal_idx
                if banned:
                    pool = legal_idx[~np.isin(legal_idx, np.fromiter(banned, dtype=legal_idx.dtype))]
                    if pool.size == 0:
                        step = ("stop", "sampled_no_action_left")
                        break
                legal_probs = probs[pool]
                total = float(legal_probs.sum())
                legal_probs = (legal_probs / total) if total > 0.0 else np.full(pool.size, 1.0 / pool.size)
                chosen = int(rng.choice(pool, p=legal_probs))
                action = ACTION_CATALOG[chosen]

                if str(action.get("type") or "").strip().upper() == "END_TURN":
                    step = ("end_turn", action)
                    break

                try:
                    trans = engine_step(work, action)
                except Exception:
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_action_error")
                    break
                next_state = trans.get("state_after")
                if not trans.get("legal") or not isinstance(next_state, dict):
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_action_illegal")
                    break
                sig = state_signature(next_state)
                if sig in visited:
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_cycle")
                    break
                step = ("commit", action, trans, next_state, sig)
                break

            if step[0] == "stop":
                stop_reason = step[1]
                break
            if step[0] == "end_turn":
                chain.append(copy.deepcopy(step[1]))
                stop_reason = "end_turn_chosen"
                break

            _, action, trans, next_state, sig = step
            chain.append(copy.deepcopy(action))
            work = next_state
            visited.add(sig)


            if for_completion and trans.get("stochastic_structural"):
                self._completion_second_chance += 1
            # The structural break belongs to the PROPOSAL frame only: see the
            # docstring for why its justification is false for a completion.
            if self.honest and not completion_mode and trans.get("stochastic_structural"):
                stop_reason = "structural_boundary"
                break
        else:
            stop_reason = "cap_reached"

        return chain, stop_reason

    def _generate_candidate(self, state: dict[str, Any], index: int) -> dict[str, Any]:
        """Candidate `index` (>= 1; candidate 0 is always the greedy call
        made directly in `recommend`, never routed through here). Prefers
        this class's own sampling decode (genuine diversity) when `self.bc`
        exposes the lower-level `.model`/`.encoder` surface a real
        `BcRecommender` has; falls back to toggling `.deterministic` and
        calling `.recommend()` again for minimal duck-typed stand-ins (this
        module's own unit tests) that do not expose it -- see the module
        docstring's "candidate diversity" section for why both paths exist.
        The internal path's `except Exception: pass` is deliberate (mirrors
        `bc_recommender.py`'s own "try the next-best/fallback path" style,
        e.g. `_choose_next_action`'s `except Exception: continue`): if
        `self.bc` claims the richer surface but it turns out unusable for
        any reason, this candidate is not lost, it is generated the
        guaranteed-safe way instead.
        """
        if hasattr(self.bc, "model") and hasattr(self.bc, "encoder"):
            try:
                return self._sample_candidate(state, self._rng_for_candidate(index))
            except Exception:
                pass
        return self._call_bc_recommend(state, deterministic=False)

    def _score_end_board(self, end_board: dict[str, Any], opponent_team: list[dict[str, Any]]) -> float:
        """Score end board."""
        config = build_simulation_config(
            end_board, opponent_team=copy.deepcopy(opponent_team), simulation_count=self.ksim
        )
        response = self._resolve_battle_fn()(config)
        if not isinstance(response, dict):
            raise RuntimeError(f"battle_fn_returned_non_dict:{type(response).__name__}")
        if "ok" in response and not response.get("ok"):
            raise RuntimeError(f"battle_fn_reported_failure:{response.get('error')}")
        payload = response.get("result")
        if not isinstance(payload, dict):
            payload = response  # tolerate a flat stub with no ok/result envelope
        player_wins = float(payload.get("playerWins", 0))
        opponent_wins = float(payload.get("opponentWins", 0))
        return (player_wins - opponent_wins) / float(self.ksim)

    def _search(self, state: dict[str, Any], greedy_result: dict[str, Any]) -> dict[str, Any]:
        opponent_team = _read_last_opponent_team(state)
        if not opponent_team:
            # Turn 1 (or any turn before a battle has ever resolved):
            # nothing to score candidates against yet. Skip generating the
            # other n_candidates-1 candidates entirely, not just the oracle
            # calls -- without an opponent there is nothing for them to be
            # scored against (see module docstring).
            return _annotate_skip(greedy_result)

        candidates = [greedy_result]
        for i in range(1, self.n_candidates):
            candidates.append(self._generate_candidate(state, i))


        if self.honest:
            return self._search_vgame_honest(
                state=state,
                greedy_result=greedy_result,
                candidates=candidates,
                opponent_team=opponent_team,
            )

        # De-duplicate by end-board signature (`_apply_chain`, the same
        # replay procedure the driver itself uses on the winning chain): a
        # board reached by more than one candidate costs one oracle call.
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []  # signatures, first-seen (== oracle-call) order
        sig_by_index: list[str] = []
        for i, cand in enumerate(candidates):
            end_board = _apply_chain(state, cand.get("chain_preview") or [])
            sig = state_signature(end_board)
            sig_by_index.append(sig)
            if sig not in groups:
                groups[sig] = {"end_board": end_board, "first_index": i, "score": None, "scored": False}
                order.append(sig)


        if self.scoring == SCORING_VGAME and not self._vgame_needs_myopic():
            for sig in order:
                groups[sig]["score"] = None
                groups[sig]["scored"] = True
            scored_sigs = list(order)
        else:
            for sig in order:
                group = groups[sig]
                try:
                    group["score"] = self._score_end_board(group["end_board"], opponent_team)
                    group["scored"] = True
                except Exception:
                    pass  # this group is excluded from the argmax below; others still count

            scored_sigs = [sig for sig in order if groups[sig]["scored"]]
            if not scored_sigs:
                # Every oracle call failed (e.g. the calculator CLI/worker is
                # down) -- nothing usable was searched, so behave exactly like
                # the no-opponent case above rather than picking a candidate on
                # no evidence. Applies to BOTH scoring modes: without even one
                # scored candidate there is nothing to shortlist for rollout
                # either.
                return _annotate_skip(greedy_result)

        greedy_group = groups[sig_by_index[0]]
        greedy_score = greedy_group["score"] if greedy_group["scored"] else None
        # None both when the greedy candidate's own oracle call failed and
        # (vgame, blend 0) when no oracle call was made at all.


        candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains or self.capture_teacher_record:
            candidate_chains = [
                copy.deepcopy(candidates[groups[sig]["first_index"]].get("chain_preview") or [])
                for sig in scored_sigs
            ]


        if self.scoring == SCORING_VGAME:
            return self._search_vgame(
                state=state,
                greedy_result=greedy_result,
                candidates=candidates,
                groups=groups,
                scored_sigs=scored_sigs,
                greedy_score=greedy_score,
                candidate_chains=(candidate_chains if self.capture_candidate_chains else None),
                n_dedup=len(order),
            )


        if self.scoring == SCORING_ROLLOUT:
            return self._search_rollout(
                candidates=candidates,
                groups=groups,
                scored_sigs=scored_sigs,
                greedy_score=greedy_score,
                candidate_chains=(candidate_chains if self.capture_candidate_chains else None),
                teacher_chains=candidate_chains,
                n_dedup=len(order),
            )

        best_sig = max(scored_sigs, key=lambda sig: groups[sig]["score"])
        winner = candidates[groups[best_sig]["first_index"]]

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_n_candidates": len(scored_sigs),


                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(len(order)),
                "search_scores": [float(groups[sig]["score"]) for sig in scored_sigs],
                "search_chosen_index": scored_sigs.index(best_sig),
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),

                "search_candidate_chains": candidate_chains,
            }
        )
        return result

    def set_race_context(self, *, wins: int | None) -> None:
        """A vgame search that was never told `wins` REFUSES to score (the
        search is skipped and `search_error` says why) rather than guessing
        0, because a wrong bypass value is a silent train/serve skew and a
        skipped search is a loud one. Never raises."""
        self._race_wins = None if wins is None else int(wins)

    def _vgame_needs_myopic(self) -> bool:
        """Whether this vgame leaf reads the stage-1 myopic scores (i.e.
        `blend != 0`). Defensive: a scorer that does not expose the property
        is treated as needing them, so the oracle calls are only ever
        skipped on an explicit False."""
        return bool(getattr(self.vgame_scorer, "needs_myopic", True))

    def _race_scalars(self, state: dict[str, Any]) -> dict[str, int]:
        """The DECISION's driver-true race block for the V bypass.

        `lives` and `opponent_lives` are read off the state search was
        handed, which is exactly where `eval_versus_fullgame.py` reads its
        own `pre_turn_lives` / `pre_turn_opp_lives` from, so the two agree by
        construction. `wins` comes from `set_race_context`. Raises if `wins`
        was never supplied -- the caller turns that into a skipped search.
        """
        if self._race_wins is None:
            raise RuntimeError("vgame_race_wins_unset")
        meta = state.get("meta") if isinstance(state, dict) else None
        versus = (meta or {}).get("versus") if isinstance(meta, dict) else None
        return {
            "turn": int(state.get("turn", 0) or 0),
            "lives": int(state.get("lives", 0) or 0),
            "opponent_lives": int((versus or {}).get("opponent_lives", 0) or 0),
            "wins": int(self._race_wins),
        }


    def set_imagination_context(
        self, *, engine_seed: int | None, segment_index: int = 0
    ) -> None:
        """`engine_seed` is the GAME's own engine seed (`play_one_game`'s
        `engine_seed_rng` draw, a pure function of `(--seed, game_index)`),
        never the state's chained `meta.seed` -- keying on the latter would
        key on the play stream's position, which is the coupling A1 removes.
        `segment_index` counts stochastic boundaries already crossed THIS
        turn, so the re-search after a real ROLL imagines off a different
        stream than the search that chose to roll.

        Called once per SEGMENT by `eval_versus_fullgame.py::play_out_game`,
        duck-typed exactly like `set_decision_context`/`set_race_context`, so
        a plain `BcRecommender` is simply never called. Never raises."""
        self._imagination_engine_seed = None if engine_seed is None else int(engine_seed)
        self._imagination_segment_index = int(segment_index)

    def _imagination_seed(self, sample_r: int) -> int:
        """`S(engine_seed, turn, segment_index, sample_r)` for this decision."""
        if self._imagination_engine_seed is not None:
            engine_seed = int(self._imagination_engine_seed)
            segment_index = int(self._imagination_segment_index)
        else:
            engine_seed = int(self._state_seed)
            segment_index = 0
        return honest_frame.imagination_seed(
            engine_seed=engine_seed,
            turn=int(self._decision_turn or 0),
            segment_index=segment_index,
            sample_r=int(sample_r),
        )

    def _imagination_key_source(self) -> str:
        return (
            honest_frame.KEY_SOURCE_DRIVER_CONTEXT
            if self._imagination_engine_seed is not None
            else honest_frame.KEY_SOURCE_STATE_SEED
        )

    def _prefix_walk(
        self, state: dict[str, Any], chain: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The chain's COMMITTABLE PREFIX and where it stopped (A2.2).

        Replays `chain` op by op from a fresh copy of `state` through
        `api.step` -- the same procedure `_apply_chain` and the driver use --
        and stops at the first op whose transition reports a STRUCTURAL
        stochastic resolution (`stochastic_structural`), INCLUDING that op.
        Detection is by that field alone, never by an action-type or pet
        whitelist, so ROLL, the level-up reward slot, a random summon and any
        future randomness that writes `legal_actions`'s read-set are all
        boundaries the day they exist. The driver's commit loop filters on the
        SAME boolean, so the search cut and the execution cut are identical by
        construction.

        A2.2 RETIRES THE DETERMINISTIC-PREFIX INVARIANT, explicitly. Under A1
        this walk stopped at every `stochastic_reason`, so the prefix consumed
        no randomness and the board it reached was a PREDICTION of the board
        the driver would commit. Under A2 a prefix may contain a NON-structural
        resolution (a stat buff on random friends, a random-target food),
        resolved once on the proposal stream, so:
        - `board` and `pre_board` are ESTIMATES. They differ from the real
          committed board in which pets got buffed, never in what is legal,
          because `legal_actions` never reads attack or health.
        - the remaining chain still cannot become illegal through that path,
          so `chain_replay_diverged` stays reachable only through a structural
          resolution, which this walk still cuts.
        - `_imagined_completion` is mechanically unaffected: it overrides
          `meta.seed` regardless of what `pre_board` carries, and `pre_board`
          is still computed once and shared by all k samples.

        Returns `{"ops", "boundary", "board", "pre_board", "stop"}`:
        - `ops`: the prefix, END_TURN excluded (as everywhere else here).
        - `boundary`: the `stochastic_reason` of the STRUCTURAL op the walk
          stopped at, or None when the prefix ran to the end of the chain
          without one. Non-structural resolutions passed on the way are not
          reported here: they are not chance nodes for scoring (A2.3) and not
          cut points for execution.
        - `board`: the board the prefix reaches, on the PROPOSAL stream.
        - `pre_board`: the board just BEFORE the structural op (None when
          there is none).
        - `stop`: why the walk ended (`end_turn`/`stochastic_boundary`/
          `chain_end`, or one of the defensive `illegal_on_replay`/
          `step_error`/`no_state_after`/`malformed_action`).
        """
        work = copy.deepcopy(state)
        ops: list[dict[str, Any]] = []
        boundary: str | None = None
        pre_board: dict[str, Any] | None = None
        stop = "chain_end"
        for action in chain:
            if not isinstance(action, dict):
                stop = "malformed_action"
                break
            if str(action.get("type") or "").strip().upper() == "END_TURN":
                stop = "end_turn"
                break
            try:
                trans = engine_step(work, action)
            except Exception:
                stop = "step_error"
                break
            if not trans.get("legal"):
                stop = "illegal_on_replay"
                break
            next_state = trans.get("state_after")
            if not isinstance(next_state, dict):
                stop = "no_state_after"
                break
            reason = trans.get("stochastic_reason")
            ops.append(copy.deepcopy(action))
            if trans.get("stochastic_structural"):
                boundary = str(reason or "structural")
                pre_board = work
                work = next_state
                stop = "stochastic_boundary"
                break
            work = next_state
        return {
            "ops": ops,
            "boundary": boundary,
            "board": work,
            "pre_board": pre_board,
            "stop": stop,
        }

    @staticmethod
    def _prefix_group_key(walk: dict[str, Any]) -> tuple[str, ...]:
        """The dedup key: one entry per DISTINCT committable prefix.

        A deterministic prefix is keyed by the END BOARD it reaches, exactly
        as the determinized path keys its candidates, so two different op
        orders landing on the same board still cost one V forward.

        A stochastic prefix is keyed by `(board just before the chance node,
        the stochastic op itself)` rather than by the whole op sequence: two
        candidates that reach the same board by different orderings and then
        roll are the SAME decision, they face the same distribution, and
        collapsing them is what makes the k-sample cost scale with distinct
        chance nodes instead of with candidate width. The post-op board is
        deliberately NOT in the key -- it is a draw, not a decision.
        """
        if walk["boundary"] is None:
            return ("det", state_signature(walk["board"]))
        pre = walk["pre_board"] if isinstance(walk["pre_board"], dict) else walk["board"]
        last_op = walk["ops"][-1] if walk["ops"] else {}
        return (
            "stoch",
            state_signature(pre),
            json.dumps(last_op, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _prefix_chain(walk: dict[str, Any]) -> list[dict[str, Any]]:
        """What `chain_preview` becomes for a winning candidate: the prefix.

        The driver commits op by op and stops at the boundary anyway, so
        truncating here is belt-and-braces -- but it makes the contract exact
        ("what search chose IS what gets committed") for every other consumer
        of a recommend result too, and it means a post-boundary op decoded
        against an imagined shop can never be handed to anyone. END_TURN is
        re-appended only when the walk consumed the whole chain and that
        chain really did end on END_TURN, i.e. when the policy's own choice
        to stop is what ended the prefix.
        """
        chain = [copy.deepcopy(op) for op in walk["ops"]]
        if walk["stop"] == "end_turn":
            chain.append({"type": "END_TURN"})
        return chain

    def _reset_completion_counters(self) -> None:
        """Zero the per-search completion instruments.

        They are reported on every segment record, so they have to describe
        THAT search. They were only ever zeroed in `__init__`, which made
        `completion_dropped` a running total for the life of the process and
        `completion_boards_mean` a lifetime average: one drop in segment 3 and
        every later segment carried it. The first reading happened to be
        `(8.0, 0)`, and a zero total looks identical either way, so nothing
        said so. Called at the top of both search entry points.
        """
        self._completion_boards = 0
        self._completion_samples = 0
        self._completion_dropped = 0
        self._completion_terminal_boards = 0
        self._completion_unfinished_boards = 0
        self._completion_second_chance = 0


        self._completion_rescued = 0

    def _completion_instruments(self) -> dict[str, Any]:
        """The completion instruments as reported, defined once so the two
        search paths cannot drift into meaning different things by them."""
        return {
            "search_completion_samples": int(self._completion_samples),
            "search_completion_boards": int(self._completion_boards),
            "search_completion_boards_mean": (
                float(self._completion_boards) / float(self._completion_samples)
                if self._completion_samples
                else None
            ),
            "search_completion_terminal_boards": int(
                self._completion_terminal_boards
            ),
            "search_completion_unfinished_boards": int(
                self._completion_unfinished_boards
            ),
            "search_completion_terminal_rate": (
                float(self._completion_terminal_boards) / float(self._completion_boards)
                if self._completion_boards
                else None
            ),
            "search_completion_dropped": int(self._completion_dropped),
            "search_completion_second_chance_nodes": int(self._completion_second_chance),
            "search_completion_rescued": int(self._completion_rescued),
        }

    def _imagined_completions(self, walk: dict[str, Any], sample_r: int) -> list[dict[str, Any]]:
        """The boards one imagined sample contributes.

        One of the k imagined completions of a stochastic prefix.

        Re-applies the ONE stochastic op on a clone of `pre_board` seeded
        with `S(..., sample_r)` (everything earlier in the prefix was
        deterministic, so replaying it would land on the identical board),
        then lets the wrapped recommender greedily decode the REST of the
        turn from the resampled board and applies that chain through
        `_apply_chain`. The returned end board is what V scores.

        The completion runs on the same `S(..., sample_r)` stream, so a
        second roll inside the completion chains off it rather than off the
        play stream -- imagination stays imagination all the way down."""


        greedy_board = self._imagined_completion(walk, sample_r)
        if self.completion_policy == COMPLETION_BC_GREEDY:
            return [greedy_board]
        # Recomputing the resampled board costs one clone plus one engine step
        # and is a pure function of `(walk, sample_r)`, so it lands on the
        # identical board the greedy completion started from. That is cheap
        # next to the width-N decodes below, and it is the price of leaving
        # the singular method's contract untouched.
        after = self._resample_after(walk, sample_r)


        boards = [greedy_board]
        terminal_boards = 1
        unfinished_boards = 0
        for j in range(1, self.completion_width):
            try:
                cand = self._sample_completion(after, sample_r, j)
                if self.completion_tail == COMPLETION_TAIL_RESAMPLE:
                    chain = (cand or {}).get("chain_preview") or []


                    reason = str(((cand or {}).get("diagnostics") or {}).get("stop_reason") or "")
                    if not chain or reason in UNFINISHED_WALK_STOPS:
                        # RESCUED, not dropped: dropping would return fewer than
                        # `completion_width` boards for this sample, and that
                        # count is a contract elsewhere. The greedy finisher is
                        # the same decoder index 0 already is, so the board is
                        # an end-of-turn board either way and only this rare
                        # alternative loses its sampled tail.
                        self._completion_rescued += 1
                        boards.append(
                            self._finish_completion_greedily(_apply_chain(after, chain))
                        )
                        terminal_boards += 1
                        continue
                board = _apply_chain(after, cand.get("chain_preview") or [])
                if self.completion_tail == COMPLETION_TAIL_GREEDY:
                    board = self._finish_completion_greedily(board)
            except Exception:


                self._completion_dropped += 1
                continue
            boards.append(board)
            if self.completion_tail == COMPLETION_TAIL_STOP:
                reason = str(
                    ((cand or {}).get("diagnostics") or {}).get("stop_reason") or ""
                )
                if not ((cand or {}).get("chain_preview") or []) or reason in UNFINISHED_WALK_STOPS:
                    unfinished_boards += 1
                else:
                    terminal_boards += 1
            else:
                terminal_boards += 1
        if terminal_boards + unfinished_boards != len(boards):
            raise AssertionError(
                "search_recommender_completion_terminal_accounting_mismatch:"
                f"terminal={terminal_boards}:unfinished={unfinished_boards}:"
                f"boards={len(boards)}"
            )
        self._completion_boards += len(boards)
        self._completion_samples += 1
        self._completion_terminal_boards += terminal_boards
        self._completion_unfinished_boards += unfinished_boards
        return boards

    def _finish_completion_greedily(self, board: dict[str, Any]) -> dict[str, Any]:
        """Play the rest of the turn out greedily from wherever a SAMPLED
        completion stopped, so every completion this method returns is an
        end-of-turn board.

        `_sample_walk` stops early for two different reasons and BOTH are
        wrong HERE while being right where they came from:

        - `structural_boundary` is justified in the docstring by "everything
          past it is provably discarded", which is true of an outer CANDIDATE
          (the search truncates it at that op anyway) and false of a
          completion, whose whole purpose is to reach a scorable end board.
        - `sampled_cycle` stops because a single random draw has no
          well-defined "next best". The greedy scan does have one, so handing
          the tail to it is also what resolves that case.

        Deterministic given `board`, and `board` is a pure function of
        `(walk, sample_r, j)`, so CRN and reproducibility are unchanged.
        Index 0 does not come through here: it already runs to END_TURN."""
        finish = self._call_bc_recommend(
            board, deterministic=True, force_ranked_decode=True
        )
        return _apply_chain(board, finish.get("chain_preview") or [])

    def _sample_completion(
        self, after: dict[str, Any], sample_r: int, index: int
    ) -> dict[str, Any]:
        """One alternative completion of the turn from the resampled board.

        Same two-path shape as `_generate_candidate` for the same reason (a
        duck-typed stand-in without `.model`/`.encoder` still has to work),
        but the RNG comes from `_rng_for_completion`, which is keyed inside
        the imagination.
        """
        if hasattr(self.bc, "model") and hasattr(self.bc, "encoder"):
            try:
                return self._sample_candidate(
                    after,
                    self._rng_for_completion(sample_r, index),
                    completion_mode=(self.completion_tail == COMPLETION_TAIL_RESAMPLE),
                    for_completion=True,
                )
            except Exception:
                pass
        return self._call_bc_recommend(after, deterministic=False)

    def _resample_after(self, walk: dict[str, Any], sample_r: int) -> dict[str, Any]:
        """The board the ONE stochastic op lands on for imagined sample `r`.

        Everything earlier in the prefix was deterministic, so replaying it
        would reach the identical board; only the last op is re-applied, on a
        clone reseeded to `S(..., sample_r)`.
        """
        pre = walk["pre_board"] if isinstance(walk["pre_board"], dict) else walk["board"]
        work = honest_frame.imagined_clone(pre, seed=self._imagination_seed(sample_r))
        trans = engine_step(work, walk["ops"][-1])
        if not trans.get("legal"):
            raise RuntimeError("honest_resample_illegal")
        after = trans.get("state_after")
        if not isinstance(after, dict):
            raise RuntimeError("honest_resample_no_state_after")
        return after

    def _completion_agg(self, values: list[float]) -> float:
        """This recommender's configured inner aggregation.

        A thin bind of the module-level `completion_agg` to
        `self.completion_aggregate`. The arithmetic lives at module level
        because the Bellman labeller has to apply the SAME aggregation to the
        SAME boards; see `completion_agg`.
        """
        return completion_agg(values, self.completion_aggregate)

    @staticmethod
    def _count_completion_decisions(
        leaf_scores: list[float],
        sub_spans: list[list[tuple[int, int, int]]],
    ) -> tuple[int, int]:
        """`(decided, divergent)` over the imagined samples that had a CHOICE.

        A sample with fewer than two boards had nothing to decide and is
        counted in NEITHER total, so `bc_greedy` reports 0 of 0 rather than a
        vacuous 100% agreement -- a ratio computed off this pair therefore has
        no denominator to divide by rather than a misleading one."""
        decided = 0
        divergent = 0
        for subs in sub_spans:
            for (a, b, _r) in subs:
                if b - a < 2:
                    continue
                decided += 1
                window = leaf_scores[a:b]
                if max(range(len(window)), key=lambda i: window[i]) != 0:
                    divergent += 1
        return decided, divergent

    def _imagined_completion(self, walk: dict[str, Any], sample_r: int) -> dict[str, Any]:
        """The greedy imagined completion of one stochastic sample."""
        after = self._resample_after(walk, sample_r)
        completion = self._call_bc_recommend(after, deterministic=True, force_ranked_decode=True)
        return _apply_chain(after, completion.get("chain_preview") or [])

    def _honest_myopic_scores(
        self, boards: list[dict[str, Any]], opponent_team: list[dict[str, Any]]
    ) -> list[float | None]:
        """Stage-1 myopic scores for the honest leaf, or all-None at blend 0.

        Same skip rule as the determinized vgame path: at `blend == 0`
        nothing reads them, so the oracle is not called at all. At a non-zero
        blend every SCORED board needs one, which under this frame means one
        per imagined completion -- k times the determinized cost. That is the
        real price of blending under expectation scoring and it is paid
        honestly rather than approximated.
        """
        if not self._vgame_needs_myopic():
            return [None] * len(boards)
        scores: list[float | None] = []
        for board in boards:
            try:
                scores.append(self._score_end_board(board, opponent_team))
            except Exception:
                scores.append(None)
        return scores

    def _search_vgame_honest(
        self,
        *,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        candidates: list[dict[str, Any]],
        opponent_team: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """1. Every candidate chain is reduced to its deterministic prefix
           (`_prefix_walk`) and the prefixes are deduped (`_prefix_group_key`)
           -- the committable unit is what gets ranked.
        2. Each group is turned into the boards V will score: ONE end board
           for a deterministic-to-END_TURN prefix (identical to the
           determinized path), or k imagined completions for a prefix that
           ends at a chance node (`_imagined_completion`).
        3. All of those boards go through ONE batched V forward, and a
           group's score is the MEAN over its own boards. Argmax wins, ties
           break by mean myopic when a blend made one, else by first-seen
           order, i.e. the greedy chain.

        NEVER-RAISE, same contract as `_search_vgame`: any failure degrades
        this turn to the greedy chain with `search_used=False` plus a
        `search_error`, so a broken leaf costs strength and is counted, never
        a crashed game. That degradation is also the FAIL-CLOSED route for a
        prefix whose completions all failed: it is dropped rather than scored
        on the proposal stream's own board -- see the drop site in step 2."""


        self._reset_completion_counters()
        try:
            groups: dict[tuple[str, ...], dict[str, Any]] = {}
            order: list[tuple[str, ...]] = []
            for i, cand in enumerate(candidates):
                walk = self._prefix_walk(state, cand.get("chain_preview") or [])
                key = self._prefix_group_key(walk)
                if key in groups:
                    continue
                groups[key] = {"first_index": i, "walk": walk}
                order.append(key)

            boards: list[dict[str, Any]] = []
            spans: list[tuple[int, int]] = []


            sub_spans: list[list[tuple[int, int, int]]] = []
            kept: list[tuple[str, ...]] = []
            dropped: list[tuple[str, ...]] = []
            for key in order:
                walk = groups[key]["walk"]
                start = len(boards)
                subs: list[tuple[int, int, int]] = []
                if walk["boundary"] is None:
                    boards.append(walk["board"])
                    subs.append((start, len(boards), 0))
                else:
                    for sample_r in range(1, self.stochastic_samples + 1):
                        sub_start = len(boards)
                        try:
                            got = self._imagined_completions(walk, sample_r)
                            if not got:
                                raise RuntimeError("honest_resample_no_completion")
                            boards.extend(got)
                            subs.append((sub_start, len(boards), int(sample_r)))
                        except Exception:
                            # Defensive only: the resampled op's legality is a
                            # function of the PRE-op board, which is identical
                            # across samples, so this cannot fire for any
                            # engine randomness that exists today. Losing one
                            # sample must not lose the candidate -- the mean
                            # is simply taken over the samples that survived.
                            continue
                    if len(boards) == start:


                        dropped.append(key)
                        continue
                kept.append(key)
                spans.append((start, len(boards)))
                sub_spans.append(subs)

            if not kept or kept[0] != order[0]:
                raise RuntimeError(
                    f"honest_prefix_completions_failed:dropped={len(dropped)}/{len(order)}"
                )
            order = kept

            myopic_scores = self._honest_myopic_scores(boards, opponent_team)
            race = self._race_scalars(state)
            scored = self.vgame_scorer.score_boards(
                boards, myopic_scores=myopic_scores, **race
            )
            leaf_scores = [float(x) for x in scored["score"]]
            if len(leaf_scores) != len(boards):
                raise RuntimeError(
                    f"vgame_scorer_length_mismatch:{len(leaf_scores)}!={len(boards)}"
                )
            v_scores = [float(x) for x in scored["v"]]
            ensemble_std = [float(x) for x in scored["ensemble_std"]]
        except Exception as exc:
            result = _annotate_skip(greedy_result)
            result["search_scoring"] = SCORING_VGAME
            result["search_turn_mode"] = self.turn_mode
            result["search_error"] = f"honest_vgame_scoring_failed:{exc}"
            return result

        def _mean(values: list[float]) -> float:
            return sum(values) / float(len(values)) if values else 0.0

        _agg = self._completion_agg

        decided_here, divergent_here = self._count_completion_decisions(
            leaf_scores, sub_spans
        )
        self._completion_decided += decided_here
        self._completion_divergent += divergent_here

        group_scores = [
            _mean([_agg(leaf_scores[a:b]) for (a, b, _r) in subs]) for subs in sub_spans
        ]
        group_myopic: list[float | None] = []
        for (a, b) in spans:
            present = [float(m) for m in myopic_scores[a:b] if m is not None]
            group_myopic.append(_mean(present) if present else None)


        rerank_block: dict[str, Any] | None = None
        if self.mc_rerank_k > 0 and group_scores:
            v_ranked = sorted(
                range(len(order)),
                key=lambda i: (-float(group_scores[i]), int(groups[order[i]]["first_index"])),
            )
            shortlist = v_ranked[: int(self.mc_rerank_k)]
            budget = max(1, int(self.rollout_repeats))
            rerank_scores: dict[int, float] = {}
            rerank_rows: list[dict[str, Any]] = []
            for slot, group_i in enumerate(shortlist):
                start, stop = spans[group_i]
                group_boards = list(range(start, stop))
                if not group_boards:
                    continue
                # Round-robin split of the budget across this group's boards.
                per_board = [budget // len(group_boards)] * len(group_boards)
                for extra in range(budget % len(group_boards)):
                    per_board[extra] += 1
                totals: list[float] = []
                spent = 0
                for board_i, reps in zip(group_boards, per_board):
                    if reps <= 0:
                        continue
                    roll = self._rollout_score_candidate(
                        boards[board_i], candidate_index=group_i, repeats=reps
                    )
                    totals.append(float(roll["score"]) * reps)
                    spent += reps
                if not spent:
                    continue
                mc_score = sum(totals) / float(spent)
                rerank_scores[group_i] = mc_score
                rerank_rows.append({
                    "v_rank": slot,
                    "group_index": int(group_i),
                    "v_score": float(group_scores[group_i]),
                    "mc_score": float(mc_score),
                    "rollouts_spent": int(spent),
                    "n_boards": len(group_boards),
                    "stochastic": groups[order[group_i]]["walk"]["boundary"] is not None,
                })
            if rerank_scores:
                # The shortlist is re-ordered among ITSELF; everything below it
                # keeps the order V gave it. A group outside the shortlist can
                # never overtake one inside it, which is what makes this a
                # rerank of V's top-k rather than a second full ranking.
                reordered = sorted(
                    rerank_scores,
                    key=lambda i: (-rerank_scores[i], int(groups[order[i]]["first_index"])),
                )
                new_order_index = {g: p for p, g in enumerate(reordered)}
                bumped = max(group_scores) + 1.0
                for position, group_i in enumerate(reordered):
                    # Rewrite the score so the single argmax downstream picks
                    # the rerank winner, preserving V's ordering below.
                    group_scores[group_i] = bumped + float(len(reordered) - position)
                rerank_block = {
                    "mc_rerank_k": int(self.mc_rerank_k),
                    "mc_rerank_m": int(budget),
                    "shortlist_taken_by": "v_score",
                    "scored_on": (
                        "trophies" if self.game_rules == SEARCH_GAME_RULES_ARENA
                        else "versus_outcome"
                    ),
                    "game_rules": self.game_rules,
                    "shortlist_group_indices": [int(g) for g in shortlist],
                    "v_order_before": [int(g) for g in v_ranked[: int(self.mc_rerank_k)]],
                    "mc_order_after": [int(g) for g in reordered],
                    "changed_the_pick": bool(
                        reordered and v_ranked and reordered[0] != v_ranked[0]
                    ),
                    "rows": rerank_rows,
                    "total_rollouts": int(sum(r["rollouts_spent"] for r in rerank_rows)),
                }

        candidate_groups: list[dict[str, Any]] | None = None
        if self.capture_candidate_chains:
            ranked = sorted(
                range(len(order)),
                key=lambda i: (-float(group_scores[i]), int(groups[order[i]]["first_index"])),
            )
            rank_by_group = {group_i: rank for rank, group_i in enumerate(ranked)}
            candidate_groups = []
            for group_i, key in enumerate(order):
                walk = groups[key]["walk"]
                start, stop = spans[group_i]
                # The sample a board belongs to comes from `sub_spans`, not
                # from its position: under `v_search` one imagined sample
                # contributes several boards, so inferring `sample_r` from the
                # offset silently relabels every board after the first.
                completions: list[dict[str, Any]] = []
                for (a, b, sample_r) in sub_spans[group_i]:
                    for inner_index, board_i in enumerate(range(a, b)):
                        completions.append(
                            {
                                "completion_r": int(sample_r),
                                "completion_inner_index": int(inner_index),
                                "imagination_seed": (
                                    self._imagination_seed(sample_r) if sample_r > 0 else None
                                ),
                                "candidate_imagined_afterstate": copy.deepcopy(boards[board_i]),
                                "leaf_score": float(leaf_scores[board_i]),
                                "v0_score": float(v_scores[board_i]),
                                "ensemble_std": float(ensemble_std[board_i]),
                            }
                        )
                candidate_groups.append(
                    {
                        "group_index": int(group_i),
                        "raw_candidate_index": int(groups[key]["first_index"]),
                        "dedup_prefix_key": list(key),
                        "committed_ops": self._prefix_chain(walk),
                        "boundary_reason": walk["boundary"],
                        "prefix_stop": walk["stop"],
                        "completion_count": len(completions),
                        "v0_group_score": float(group_scores[group_i]),
                        "v0_rank": int(rank_by_group[group_i]),
                        "completions": completions,
                    }
                )

        # `max` returns the FIRST maximal element, and candidate 0 (the
        # greedy chain) is always group 0, so a full tie keeps plain BC --
        # the same "search can never do worse than greedy" property the
        # determinized path has.
        best_i = max(
            range(len(order)),
            key=lambda i: (group_scores[i], group_myopic[i] if group_myopic[i] is not None else 0.0),
        )
        best_key = order[best_i]
        best_walk = groups[best_key]["walk"]
        winner = candidates[groups[best_key]["first_index"]]
        chosen_chain = self._prefix_chain(best_walk)

        candidate_chains: list[list[dict[str, Any]]] | None = None
        decoded_candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains:
            candidate_chains = [self._prefix_chain(groups[key]["walk"]) for key in order]
            # A2.4's required telemetry: what the DECODER produced, before
            # `_prefix_walk` truncated it. `candidate_chains` above is one
            # entry per surviving GROUP and is already truncated, so it cannot
            # measure the decode saving; this is one entry per generated
            # candidate, so per-candidate chain lengths pin the size of the
            # saving instead of it being asserted.
            decoded_candidate_chains = [
                copy.deepcopy(cand.get("chain_preview") or []) for cand in candidates
            ]

        result = dict(winner)
        result.update(
            {
                **_searched_verdict(chosen_chain),
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_turn_mode": self.turn_mode,
                "search_n_candidates": len(order),
                "search_n_generated": int(self.n_candidates),
                # Under this frame "dedup" counts DISTINCT COMMITTABLE
                # PREFIXES, which is the width that is actually being bought
                # once the turn is segmented -- not distinct whole-turn end
                # boards, a unit this frame never commits.
                "search_n_dedup": len(order),


                "search_completion_policy": self.completion_policy,
                "search_completion_width": int(self.completion_width),
                "search_completion_aggregate": self.completion_aggregate,
                "search_completion_decided": int(decided_here),
                "search_completion_divergent": int(divergent_here),
                **self._completion_instruments(),
                "search_scores": group_scores,
                "search_chosen_index": best_i,
                "search_candidate_groups": candidate_groups,
                # Parallel to the determinized vgame path: this is the GREEDY
                # candidate's stage-1 myopic score, None at blend 0 where no
                # oracle call is made. Its leaf score is in the diagnostics.
                "search_greedy_score": group_myopic[0] if group_myopic else None,
                "search_candidate_chains": candidate_chains,
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "turn_mode": self.turn_mode,
                    "stochastic_samples": int(self.stochastic_samples),


                    "mc_rerank": rerank_block,
                    "completion_boards_mean": (
                        float(self._completion_boards) / float(self._completion_samples)
                        if self._completion_samples
                        else None
                    ),
                    "completion_dropped": int(self._completion_dropped),
                    "imagination_key_source": self._imagination_key_source(),
                    "imagination_segment_index": int(self._imagination_segment_index),
                    "imagination_sample_seeds": [
                        self._imagination_seed(r)
                        for r in range(1, self.stochastic_samples + 1)
                    ],
                    "prefix_boundaries": [groups[key]["walk"]["boundary"] for key in order],
                    "prefix_stops": [groups[key]["walk"]["stop"] for key in order],
                    "prefix_lengths": [len(groups[key]["walk"]["ops"]) for key in order],


                    "prefix_n_samples": [len(subs) for subs in sub_spans],
                    "prefix_n_boards": [b - a for (a, b) in spans],


                    "prefix_n_dropped": len(dropped),
                    "prefix_sample_scores": [leaf_scores[a:b] for (a, b) in spans],
                    "decoded_candidate_chains": decoded_candidate_chains,
                    "group_scores": group_scores,
                    "greedy_leaf_score": (group_scores[0] if group_scores else None),
                    "v_scores": v_scores,
                    "leaf_scores": leaf_scores,
                    "ensemble_std": ensemble_std,
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in myopic_scores
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                },
            }
        )
        return result

    def _search_vgame(
        self,
        *,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        candidates: list[dict[str, Any]],
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        greedy_score: float | None,
        candidate_chains: list[list[dict[str, Any]]] | None,
        n_dedup: int,
    ) -> dict[str, Any]:
        """Stage 1 is shared with every other mode, so the candidate set this
        scores is bit-identical to the set a rollout run at the same width
        would have scored. The score itself is
        `(1-blend)*V + blend*myopic - pessimism*ensemble_std`, computed by
        `tools/vgame_scorer.py` (which owns the encoder, the bypass block and
        the checkpoint pins).

        NEVER-RAISE: any scorer failure -- a bad board, an encoder error, an
        unset `wins` -- degrades to the greedy candidate with the search
        marked unused, exactly as a turn-1 skip does, plus `search_error` so
        the failure is visible in the report instead of silent. A game never
        dies because the leaf did."""
        boards = [groups[sig]["end_board"] for sig in scored_sigs]
        myopic_scores = [groups[sig]["score"] for sig in scored_sigs]
        try:
            race = self._race_scalars(state)
            scored = self.vgame_scorer.score_boards(boards, myopic_scores=myopic_scores, **race)
            values = [float(x) for x in scored["score"]]
            if len(values) != len(scored_sigs):
                raise RuntimeError(
                    f"vgame_scorer_length_mismatch:{len(values)}!={len(scored_sigs)}")
        except Exception as exc:
            result = _annotate_skip(greedy_result)
            result["search_scoring"] = SCORING_VGAME
            # Deliberately distinct from a turn-1 skip: that one is a design
            # outcome, this one is a degraded turn and must be countable.
            result["search_error"] = f"vgame_scoring_failed:{exc}"
            return result

        # Argmax on the leaf score; an EXACT tie breaks by the stage-1 myopic
        # score when one exists (the rollout path's own convention), else by
        # first-seen order, which is candidate 0 = the greedy chain.
        best_i = max(
            range(len(values)),
            key=lambda i: (values[i], (myopic_scores[i] if myopic_scores[i] is not None else 0.0)),
        )
        best_sig = scored_sigs[best_i]
        winner = candidates[groups[best_sig]["first_index"]]

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_n_candidates": len(scored_sigs),
                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(n_dedup),
                "search_scores": values,
                "search_chosen_index": best_i,
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),
                "search_candidate_chains": candidate_chains,
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "v_scores": [float(x) for x in scored["v"]],
                    "leaf_scores": values,
                    "ensemble_std": [float(x) for x in scored["ensemble_std"]],
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in myopic_scores
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                },
            }
        )
        return result

    def _search_rollout(
        self,
        *,
        candidates: list[dict[str, Any]],
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        greedy_score: float | None,
        candidate_chains: list[list[dict[str, Any]]] | None = None,
        teacher_chains: list[list[dict[str, Any]]] | None = None,
        n_dedup: int | None = None,
    ) -> dict[str, Any]:
        """Stage 2 of `scoring="rollout"` (module docstring's ROLLOUT
        section): shortlist the top `self.rollout_shortlist` deduped
        candidates by their stage-1 myopic score, simulate each one's
        rest-of-game `self.rollout_repeats` times
        (`_rollout_score_candidate`), and pick the argmax rollout score --
        an EXACT tie breaks by the myopic score (both wrapped in one
        descending-sorted tuple key, so a plain `max()` does the right
        thing). `candidates`/`groups`/`scored_sigs`/`greedy_score` are the
        SAME stage-1 outputs the myopic path uses, passed in rather than
        recomputed.
        """
        myopic_ranked = sorted(scored_sigs, key=lambda sig: groups[sig]["score"], reverse=True)
        shortlist_sigs = myopic_ranked[: max(1, int(self.rollout_shortlist))]

        rollout_by_sig: dict[str, dict[str, Any]] = {}
        for sig in shortlist_sigs:
            candidate_index = groups[sig]["first_index"]
            rollout_by_sig[sig] = self._rollout_score_candidate(
                groups[sig]["end_board"], candidate_index=candidate_index
            )

        best_myopic_sig = max(scored_sigs, key=lambda sig: groups[sig]["score"])
        best_sig = max(shortlist_sigs, key=lambda sig: (rollout_by_sig[sig]["score"], groups[sig]["score"]))
        winner = candidates[groups[best_sig]["first_index"]]

        teacher_record = (
            self._build_teacher_record(
                groups=groups,
                scored_sigs=scored_sigs,
                shortlist_sigs=shortlist_sigs,
                rollout_by_sig=rollout_by_sig,
                best_sig=best_sig,
                best_myopic_sig=best_myopic_sig,
                teacher_chains=teacher_chains,
                n_dedup=n_dedup,
            )
            if self.capture_teacher_record
            else None
        )

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_scoring": SCORING_ROLLOUT,
                "search_n_candidates": len(scored_sigs),


                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(len(scored_sigs) if n_dedup is None else n_dedup),
                "search_scores": [float(groups[sig]["score"]) for sig in scored_sigs],


                "search_chosen_index": scored_sigs.index(best_myopic_sig),
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),
                "search_diagnostics": {
                    "scoring": SCORING_ROLLOUT,
                    "myopic_scores": [float(groups[sig]["score"]) for sig in scored_sigs],
                    "myopic_chosen_index": scored_sigs.index(best_myopic_sig),
                    "shortlist_indices": [scored_sigs.index(sig) for sig in shortlist_sigs],
                    "rollout_scores": [rollout_by_sig[sig] for sig in shortlist_sigs],
                    "rollout_chosen_index": shortlist_sigs.index(best_sig),
                    "rollout_shortlist": int(self.rollout_shortlist),
                    "rollout_repeats": int(self.rollout_repeats),
                    "rollout_ksim": int(self.rollout_ksim),


                    "rollout_opponent_mode": self.rollout_opponent_mode,


                    "candidate_chains": candidate_chains,
                },


                "teacher_record": teacher_record,
            }
        )
        return result

    def _build_teacher_record(
        self,
        *,
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        shortlist_sigs: list[str],
        rollout_by_sig: dict[str, dict[str, Any]],
        best_sig: str,
        best_myopic_sig: str,
        teacher_chains: list[list[dict[str, Any]]] | None,
        n_dedup: int | None,
    ) -> dict[str, Any]:
        """Build teacher record."""
        candidates_out: list[dict[str, Any]] = []
        for i, sig in enumerate(scored_sigs):
            roll = rollout_by_sig.get(sig)
            chain = None
            if teacher_chains is not None and i < len(teacher_chains):
                chain = copy.deepcopy(teacher_chains[i])
            candidates_out.append(
                {
                    "index": i,
                    "signature": sig,
                    "myopic_score": float(groups[sig]["score"]),
                    "rolled_out": roll is not None,
                    "teacher_score": (float(roll["score"]) if roll is not None else None),
                    "mean_outcome": (float(roll["mean_outcome"]) if roll is not None else None),
                    "mean_lives_diff": (float(roll["mean_lives_diff"]) if roll is not None else None),
                    "n_repeats": (int(roll["n_repeats"]) if roll is not None else 0),
                    "repeat_outcomes": (list(roll["repeat_outcomes"]) if roll is not None else []),
                    "repeat_lives_diffs": (list(roll.get("repeat_lives_diffs") or []) if roll is not None else []),
                    "repeat_crn_keys": (list(roll.get("repeat_crn_keys") or []) if roll is not None else []),
                    "repeat_engine_seeds": (list(roll.get("repeat_engine_seeds") or []) if roll is not None else []),
                    "repeat_chosen_opponent_pids": (
                        list(roll.get("repeat_chosen_opponent_pids") or []) if roll is not None else []
                    ),
                    "chosen": sig == best_sig,
                    "chain": chain,


                    "state": copy.deepcopy(groups[sig]["end_board"]),
                }
            )
        return {
            "schema": TEACHER_RECORD_SCHEMA,
            "n_generated": int(self.n_candidates),
            "n_dedup": int(len(scored_sigs) if n_dedup is None else n_dedup),
            "n_candidates": len(scored_sigs),
            "shortlist_size": len(shortlist_sigs),
            "rollout_repeats": int(self.rollout_repeats),
            "rollout_ksim": int(self.rollout_ksim),
            "rollout_opponent_mode": self.rollout_opponent_mode,
            "chosen_index": scored_sigs.index(best_sig),
            "chosen_signature": best_sig,
            "chosen_teacher_score": float(rollout_by_sig[best_sig]["score"]),
            "myopic_chosen_index": scored_sigs.index(best_myopic_sig),
            "candidates": candidates_out,
            "crn": {
                "enabled": bool(self.rollout_crn),
                "base_seed": int(self.seed),
                "game_index": self._crn_game_index,
                "turn": self._decision_turn,
                "decision_id": self._crn_decision_id(),
            },
        }

    def _choose_rollout_opponent_pid(
        self, *, true_pid: str | None, current_turn: int, rng: random.Random
    ) -> dict[str, Any]:
        """`"pool_random"`: uniform draw over `self.opp_source.all_pids` --
        already exactly the eval frame's own opponent pool (e.g.
        val/Turtle/rank<=1500, whatever `opp_source` was constructed with)
        -- EXCLUDING `true_pid`, so the draw can never silently coincide
        with the privileged `"true"` arm (the correctness proof this
        module's caller requires: the arms must actually use different
        opponents).

        `"retrieval"`: same pool, additionally restricted to pids whose
        indexed chain length (`len(self.opp_source.by_pid[pid])`, the same
        "chain length" measure `ChainSnapshotSource.long_min` already uses)
        is >= `current_turn` -- a "this candidate opponent's own game had
        actually reached this turn" plausibility filter. rank/pack are NOT
        re-checked here: they are pool-wide invariants already enforced by
        how `self.opp_source` itself was constructed (every pid in
        `all_pids` already satisfies them), so re-checking per draw would
        be redundant, not additionally correct.

        Degrades gracefully rather than raising: if the length filter
        leaves zero candidates, falls back to the unrestricted
        (`"pool_random"`-style) population; if THAT is also empty (a
        single-pid pool with `true_pid` excluded), gives up and returns
        `pid=None` -- the caller then leaves `board_copy` untouched for
        just that one repeat (equivalent to `"true"` for that draw only,
        never a hard failure).

        Returns `{"pid": str | None, "fallback_tier": str}`."""
        all_pids = self.opp_source.all_pids
        pool = [p for p in all_pids if p != true_pid] or list(all_pids)
        if not pool:
            return {"pid": None, "fallback_tier": "empty_pool"}

        if self.rollout_opponent_mode == ROLLOUT_OPPONENT_RETRIEVAL:
            eligible = [p for p in pool if len(self.opp_source.by_pid.get(p, {})) >= int(current_turn)]
            if eligible:
                return {"pid": str(rng.choice(eligible)), "fallback_tier": "retrieval_primary"}
            # Too few (or zero) same-tier-and-long-enough candidates at this
            # turn -- relax the length constraint rather than fail the
            # repeat outright (rank/pack still hold, see docstring).
            return {"pid": str(rng.choice(pool)), "fallback_tier": "retrieval_relaxed_length"}

        return {"pid": str(rng.choice(pool)), "fallback_tier": "pool_random"}

    def _rollout_score_candidate(
        self, end_board: dict[str, Any], *, candidate_index: int,
        repeats: int | None = None,
    ) -> dict[str, Any]:
        """Simulate the REST OF THE GAME `self.rollout_repeats` times for one
        shortlisted candidate's `end_board` (this turn's shop-phase actions
        already applied, END_TURN not yet resolved). See module docstring's
        ROLLOUT section for the full algorithm; this is stage 2's per-
        candidate inner loop.

        Each repeat runs on a FRESH `copy.deepcopy(end_board)` and an
        ISOLATED fallback sampler (`random.Random`, never
        `self.opp_source`'s own shared `_random_rng` -- see
        `chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`).
        This turn's own battle uses `simulation_count=self.rollout_ksim`;
        every subsequent turn (`eval_versus_fullgame.py::play_out_game`)
        reverts to that driver's default (1).

        Never raises: a repeat whose current-turn battle resolution itself
        fails (e.g. both the followed chain AND the isolated random
        fallback come up empty for this turn -- `end_turn_failed`) scores
        0.5 (treated as a no-result), the same bucket a genuine turn-cap
        gets, rather than aborting the whole candidate."""
        # Lazy import: avoids a module-level cycle (eval_versus_fullgame.py
        # imports THIS module at its own top level already) -- mirrors this
        # repo's own precedent for the identical reason
        # (`eval_versus_fullgame.py::_replay_order_helpers`'s lazy
        # cross-tools-module import).
        from . import eval_versus_fullgame as _evf

        true_pid = _read_current_opponent_pid(end_board)
        current_turn = int(end_board.get("turn", 1))

        outcomes: list[float] = []
        lives_diffs: list[float] = []
        # Arena's own estimand. Under versus this list stays empty and nothing
        # reads it, so the versus score below is unchanged.
        trophies_seen: list[float] = []
        is_arena = self.game_rules == SEARCH_GAME_RULES_ARENA
        repeat_chosen_opponent_pids: list[str | None] = []
        repeat_fallback_tiers: list[str] = []
        repeat_opponent_pid_traces: list[list[dict[str, Any]]] = []


        repeat_crn_keys: list[str] = []
        repeat_engine_seeds: list[int | None] = []
        n_repeats = max(1, int(self.rollout_repeats if repeats is None else repeats))
        for repeat_index in range(n_repeats):
            fallback_key = self._crn_fallback_key(
                candidate_index=candidate_index, repeat_index=repeat_index
            )
            repeat_crn_keys.append(fallback_key)
            fallback_rng = random.Random(fallback_key)
            sample_random_fn = (
                lambda turn, _rng=fallback_rng: self.opp_source.sample_random_with_rng(turn, _rng)
            )

            board_copy = copy.deepcopy(end_board)


            engine_seed: int | None = None
            if self.rollout_crn:
                copy_meta = board_copy.setdefault("meta", {})
                if isinstance(copy_meta, dict) and copy_meta.get("seed_known"):
                    engine_seed = self._crn_engine_seed(repeat_index)
                    copy_meta["seed"] = int(engine_seed)
            repeat_engine_seeds.append(engine_seed)


            chosen_pid: str | None = None
            fallback_tier = "true_mode_passthrough"
            if self.rollout_opponent_mode != ROLLOUT_OPPONENT_TRUE:
                opponent_rng = random.Random(
                    self._crn_opponent_key(
                        candidate_index=candidate_index, repeat_index=repeat_index
                    )
                )
                choice = self._choose_rollout_opponent_pid(
                    true_pid=true_pid, current_turn=current_turn, rng=opponent_rng
                )
                chosen_pid = choice["pid"]
                fallback_tier = choice["fallback_tier"]
                if chosen_pid:
                    versus_meta = board_copy.setdefault("meta", {}).setdefault("versus", {})
                    versus_meta["current_opponent_participation_id"] = chosen_pid
            repeat_chosen_opponent_pids.append(chosen_pid)
            repeat_fallback_tiers.append(fallback_tier)


            opponent_pid_trace: list[dict[str, Any]] = []

            def _record_pid_used(turn: Any, state_after: dict[str, Any] | None) -> None:
                pid_used = _read_current_opponent_pid(state_after) if isinstance(state_after, dict) else None
                opponent_pid_trace.append({"turn": int(turn) if turn is not None else None, "opponent_pid_used": pid_used})

            if is_arena:
                # What `play_out_game` does at the top of every arena turn:
                # arena has no engine life race, so the opponent life bar is a
                # pinned convention derived from the trophy count. Without it
                # `_versus_win(lives, opp_lives)` below reads a missing bar as
                # an instant win, which is exactly why the driver refuses an
                # arena rollout today.
                _evf._apply_arena_race_context(
                    board_copy, convention=self.arena_race_convention
                )
            this_turn = _evf._resolve_versus_turn(
                board_copy,
                sample_for_pid_fn=self.opp_source.sample_for_pid,
                sample_random_fn=sample_random_fn,
                parse_cache=None,
                simulation_count=self.rollout_ksim,
                game_mode=(_evf.GAME_MODE_ARENA if is_arena else _evf.GAME_MODE_VERSUS),
                max_lives=(_evf.ARENA_MAX_LIVES if is_arena else None),
            )
            if not this_turn["ok"]:
                outcomes.append(0.5)
                lives_diffs.append(0.0)
                repeat_opponent_pid_traces.append(opponent_pid_trace)
                continue

            state_after = this_turn["state_after"]
            _record_pid_used(current_turn, state_after)
            lives = int(state_after.get("lives", 0))
            opp_lives = int(_evf._versus_meta(state_after).get("opponent_lives", 0))
            if bool(TrainingEnv._is_done(state_after, None)):
                win = _evf._versus_win(lives, opp_lives)
                outcomes.append(1.0 if win else (0.0 if lives <= 0 else 0.5))
                lives_diffs.append(float(lives - opp_lives))
                if is_arena:
                    trophies_seen.append(float(state_after.get("trophies", 0) or 0))
                repeat_opponent_pid_traces.append(opponent_pid_trace)
                continue

            current_pid = str(_evf._versus_meta(state_after).get("current_opponent_participation_id") or "")
            remainder = _evf.play_out_game(
                state_after,
                self.bc,  # plain wrapped recommender -- NEVER self (no recursive search)
                initial_pid=current_pid,
                sample_for_pid_fn=self.opp_source.sample_for_pid,
                sample_random_fn=sample_random_fn,
                max_turn=self.max_turn,
                parse_cache=None,
                capture_detail=False,
                # The frame. `play_out_game` has taken all four of these all
                # along; this class simply never carried them, so the
                # continuation resolved under the versus default no matter
                # what frame the arm was actually being judged on.
                game_rules=self.game_rules,
                arena_race_convention=self.arena_race_convention,
                opponent_mode=(_evf.OPPONENT_MODE_ARENA if is_arena
                               else _evf.OPPONENT_MODE_CHAIN),
                turn_mode=self.turn_mode,


                on_turn=lambda payload: _record_pid_used(payload.get("turn"), payload.get("state_after")),
            )
            win = bool(remainder["win"])
            loss = remainder["end_reason"] == _evf.END_REASON_PLAYER_LIVES_0
            outcomes.append(1.0 if win else (0.0 if loss else 0.5))
            if is_arena:
                trophies_seen.append(float(remainder.get("trophies", 0) or 0))
            lives_diffs.append(float(remainder["player_lives"] - remainder["opponent_lives"]))
            repeat_opponent_pid_traces.append(opponent_pid_trace)

        mean_outcome = sum(outcomes) / len(outcomes) if outcomes else 0.5
        mean_lives_diff = sum(lives_diffs) / len(lives_diffs) if lives_diffs else 0.0
        # ARENA IS SCORED ON TROPHIES, because that is what the arena
        # judgement's estimand is. Scoring an arena rollout on the versus
        # win/loss would rank candidates on a quantity no arena verdict reads,
        # and under arena a "win" means the 10-trophy completion, which almost
        # no game reaches -- so nearly every candidate would tie at 0.5.
        mean_trophies = (
            sum(trophies_seen) / len(trophies_seen) if trophies_seen else 0.0
        )
        score = (
            mean_trophies if is_arena
            else mean_outcome + 0.01 * mean_lives_diff
        )
        return {
            "score": score,
            "mean_trophies": mean_trophies if is_arena else None,
            "repeat_trophies": list(trophies_seen) if is_arena else None,
            "scored_on": "trophies" if is_arena else "versus_outcome",
            "mean_outcome": mean_outcome,
            "mean_lives_diff": mean_lives_diff,
            "n_repeats": len(outcomes),
            "repeat_outcomes": outcomes,
            "rollout_crn": bool(self.rollout_crn),
            "crn_decision_id": self._crn_decision_id(),


            **(
                {
                    "repeat_lives_diffs": lives_diffs,
                    "repeat_crn_keys": repeat_crn_keys,
                    "repeat_engine_seeds": repeat_engine_seeds,
                }
                if self.capture_teacher_record
                else {}
            ),


            "rollout_opponent_mode": self.rollout_opponent_mode,
            "true_followed_pid": true_pid,
            "repeat_chosen_opponent_pids": repeat_chosen_opponent_pids,
            "repeat_fallback_tiers": repeat_fallback_tiers,
            "repeat_opponent_pid_traces": repeat_opponent_pid_traces,
        }


    def set_candidate_stream(self, key: int) -> None:
        """`_rng_for_candidate` keys the per-candidate sampling RNG on
        `SeedSequence([seed, _recommend_call_count, index])`, and that counter
        is a PROCESS counter: it counts how many `recommend()` calls this
        object has served since it was constructed. That is the right default
        for an eval driver, where "don't repeat the same noise every turn" is
        all it has to buy -- but it makes a decision depend on how many
        decisions came before it in the same process.

        So the duel worker calls this once per SEGMENT with a key derived
        from `(engine_seed, turn, segment_index)` -- the same tuple A1's
        imagination stream is keyed on, for the same reason -- and the next
        `recommend`/`recommend_anytime` call runs on exactly `key`. Nothing
        else calls it, so the driver's counter behaviour is untouched."""
        self._recommend_call_count = int(key) - 1

    def recommend_anytime(
        self,
        state: dict[str, Any],
        *,
        chunk_size: int = DEFAULT_ANYTIME_CHUNK,
        should_stop: Callable[[], bool] | None = None,
        max_candidates: int | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
        next_chunk_size: Callable[[dict[str, Any]], int] | None = None,
        extra_samples_until_stop: bool = False,
        on_level: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """`recommend()` for an interactive clock: score in CHUNKS, keep the
        best so far, and be interruptible between chunks.

        WHY. A duel turn runs on a fixed wall clock (`turn_budget_s`, 105 s
        by default) split into per-segment deadlines, so the search stops on
        TIME rather than at a preset width, and the caller needs a best-so-far
        answer at whatever moment that is.

        There is no interruption point INSIDE a chunk and this method does not
        pretend otherwise: `should_stop` is consulted at chunk boundaries only.
        `chunk_size` ships at 1, so that boundary is every single candidate.

        WHAT IS AND IS NOT NEW HERE. Every piece of A1 semantics is the same
        object the driver's path uses: `_prefix_walk` for the committable
        prefix, `_prefix_group_key` for the dedup, `_imagined_completion`
        for the k CRN-keyed resamples, `_honest_myopic_scores`,
        `vgame_scorer.score_boards`, `_prefix_chain`. What this method owns
        is the BOOKKEEPING that `_search_vgame_honest` does in one pass and
        this one does incrementally -- grouping, spans, argmax.

        The one deliberate difference in the RESULT: `search_n_generated` is
        how many candidates this call actually generated, not the configured
        width, because under a stop those differ and the UI has to show the
        human the width the AI really got."""
        if not (self.honest and self.scoring == SCORING_VGAME):
            raise ValueError(
                f"search_recommender_anytime_requires_honest_vgame:"
                f"turn_mode={self.turn_mode!r}:scoring={self.scoring!r}"
            )
        chunk_size = max(1, int(chunk_size))

        # Identical preamble to `recommend()` -- same counters, same
        # defensive reads, so a chunked decision keys its CRN streams
        # exactly the way an unchunked one at the same call index would.
        self._recommend_call_count += 1
        turn_value = state.get("turn") if isinstance(state, dict) else None
        try:
            self._decision_turn = int(turn_value) if turn_value is not None else None
        except (TypeError, ValueError):
            self._decision_turn = None
        self._state_seed = honest_frame.read_engine_seed(state) if isinstance(state, dict) else 0

        try:
            greedy_result = self._call_bc_recommend(
                state, deterministic=True, force_ranked_decode=True
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"search_greedy_call_failed:{exc}",
                "recommended_action": None,
                "chain_preview": [],
                "wdl_probs": None,
                "diagnostics": {},
                "search_used": False,
                "search_n_candidates": 0,
                "search_scores": [],
                "search_chosen_index": 0,
                "search_greedy_score": None,
                "search_error": str(exc),
            }

        try:
            return self._search_anytime(
                state=state,
                greedy_result=greedy_result,
                chunk_size=chunk_size,
                should_stop=should_stop,
                max_candidates=max_candidates,
                on_chunk=on_chunk,
                next_chunk_size=next_chunk_size,
                extra_samples_until_stop=extra_samples_until_stop,
                on_level=on_level,
            )
        except Exception as exc:
            result = dict(greedy_result)
            result.update(
                {
                    "search_used": False,
                    "search_n_candidates": 0,
                    "search_scores": [],
                    "search_chosen_index": 0,
                    "search_greedy_score": None,
                    "search_error": str(exc),
                }
            )
            return result

    def _anytime_group_boards(
        self, walk: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
        """The boards V scores for ONE prefix group, and how they split by
        imagined sample.

        Returns `(boards, subs)`. `boards` is 1 board if the prefix is
        deterministic, else the surviving imagined samples' completions
        concatenated. `subs` carries one `(start, stop, sample_r)` per
        SURVIVING sample, as offsets into `boards`."""
        if walk["boundary"] is None:
            return [walk["board"]], [(0, 1, 0)]


        boards: list[dict[str, Any]] = []
        subs: list[tuple[int, int, int]] = []
        for sample_r in range(1, self.stochastic_samples + 1):
            got, span = self._anytime_sample_boards(walk, sample_r, len(boards))
            if span is None:
                continue
            boards.extend(got)
            subs.append(span)
        return boards, subs

    def _anytime_sample_boards(
        self, walk: dict[str, Any], sample_r: int, offset: int
    ) -> tuple[list[dict[str, Any]], tuple[int, int, int] | None]:
        """ONE imagined sample's inner boards, and its span at `offset`.

        `(boards, None)` means every completion of this sample failed. Losing
        ONE sample must not lose the candidate; losing them all is what the
        caller's drop handles."""
        try:
            got = self._imagined_completions(walk, sample_r)
            if not got:
                raise RuntimeError("honest_resample_no_completion")
        except Exception:
            return [], None
        return got, (int(offset), int(offset) + len(got), int(sample_r))

    def _search_anytime(
        self,
        *,
        extra_samples_until_stop: bool = False,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        chunk_size: int,
        should_stop: Callable[[], bool] | None,
        max_candidates: int | None,
        on_chunk: Callable[[dict[str, Any]], None] | None,
        next_chunk_size: Callable[[dict[str, Any]], int] | None,
        on_level: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        opponent_team = _read_last_opponent_team(state)
        if not opponent_team:
            # Turn 1 (or any turn before a battle has ever resolved): the
            # same designed skip `_search` takes, for the same reason.
            return _annotate_skip(greedy_result)

        width = int(self.n_candidates if max_candidates is None else max_candidates)
        width = max(1, width)
        race = self._race_scalars(state)
        self._reset_completion_counters()

        groups: dict[tuple[str, ...], dict[str, Any]] = {}
        order: list[tuple[str, ...]] = []
        group_scores: list[float] = []
        group_myopic: list[float | None] = []
        group_sample_scores: list[list[float]] = []
        # SAMPLES and BOARDS are two different counts once the inner width can
        # exceed 1, so both are carried rather than one being inferred from
        # the other -- the same pair `_search_vgame_honest` reports.
        group_n_samples: list[int] = []
        group_n_boards: list[int] = []


        group_aggs: list[list[float]] = []
        group_keys_in_order: list[tuple[str, ...]] = []
        v_scores: list[float] = []
        ensemble_std: list[float] = []
        all_myopic: list[float | None] = []
        n_dropped = 0
        generated = 0
        chunks = 0
        stopped = False
        chunk_sizes: list[int] = []
        chunk_timings: list[dict[str, Any]] = []
        timing_totals = {
            "candidate_s": 0.0,
            "outcomes_s": 0.0,
            "myopic_s": 0.0,
            "value_s": 0.0,
            "total_s": 0.0,
        }
        leaves_evaluated = 0
        decided_here = 0
        divergent_here = 0

        while generated < width:
            requested_size = chunk_size
            if next_chunk_size is not None:
                requested_size = int(
                    next_chunk_size(
                        {
                            "chunks": int(chunks),
                            "n_generated": int(generated),
                            "n_dedup": int(len(order)),
                            "width_requested": int(width),
                        }
                    )
                )
            size = min(max(1, int(requested_size)), width - generated)
            chunk_started = time.perf_counter()
            chunk_indices = list(range(generated, generated + size))
            generated += size
            chunk_sizes.append(int(size))

            # Same candidates, same order, as the one-pass path: index 0 IS
            # the greedy result, every other index is a pure function of
            # `(state, index)`.
            candidate_started = time.perf_counter()
            new_keys: list[tuple[str, ...]] = []
            for index in chunk_indices:
                cand = greedy_result if index == 0 else self._generate_candidate(state, index)
                walk = self._prefix_walk(state, cand.get("chain_preview") or [])
                key = self._prefix_group_key(walk)
                if key in groups:
                    continue
                groups[key] = {"first_index": index, "walk": walk, "candidate": cand}
                new_keys.append(key)
            candidate_s = time.perf_counter() - candidate_started

            outcomes_started = time.perf_counter()
            boards: list[dict[str, Any]] = []
            spans: list[tuple[int, int]] = []
            # One entry per kept group, each a list of (start, stop, sample_r)
            # into THIS chunk's `boards`. Chunk-local, exactly like `boards`
            # and `spans`, because the V forward below is per chunk.
            sub_spans: list[list[tuple[int, int, int]]] = []
            kept: list[tuple[str, ...]] = []
            for key in new_keys:
                start = len(boards)
                group_boards, group_subs = self._anytime_group_boards(
                    groups[key]["walk"]
                )
                if not group_boards:
                    n_dropped += 1
                    if not order and not kept:
                        # The greedy chain's own group is the tie-break
                        # anchor; losing it degrades the whole decision,
                        # exactly as the one-pass path does.
                        raise RuntimeError(
                            f"honest_prefix_completions_failed:dropped={n_dropped}/{len(new_keys)}"
                        )
                    # Left in `groups` (so a later candidate with the same
                    # prefix is still deduped away, exactly as one-pass does)
                    # but never added to `order`, so it cannot be chosen.
                    continue
                boards.extend(group_boards)
                spans.append((start, len(boards)))
                sub_spans.append(
                    [(start + a, start + b, r) for (a, b, r) in group_subs]
                )
                kept.append(key)
            outcomes_s = time.perf_counter() - outcomes_started

            myopic_s = 0.0
            value_s = 0.0
            if boards:
                myopic_started = time.perf_counter()
                myopic = self._honest_myopic_scores(boards, opponent_team)
                myopic_s = time.perf_counter() - myopic_started
                value_started = time.perf_counter()
                scored = self.vgame_scorer.score_boards(boards, myopic_scores=myopic, **race)
                value_s = time.perf_counter() - value_started
                leaf = [float(x) for x in scored["score"]]
                if len(leaf) != len(boards):
                    raise RuntimeError(
                        f"vgame_scorer_length_mismatch:{len(leaf)}!={len(boards)}"
                    )
                v_scores.extend(float(x) for x in scored["v"])
                ensemble_std.extend(float(x) for x in scored["ensemble_std"])
                all_myopic.extend(myopic)
                chunk_decided, chunk_divergent = self._count_completion_decisions(
                    leaf, sub_spans
                )
                decided_here += chunk_decided
                divergent_here += chunk_divergent
                for key, (a, b), subs in zip(kept, spans, sub_spans):
                    order.append(key)
                    samples = leaf[a:b]
                    # TWO layers, the same shape `_search_vgame_honest` uses:
                    # mean over the imagined SAMPLES of the aggregate over that
                    # sample's inner completions. Under `bc_greedy` every
                    # sub-span is one board and `_completion_agg` returns it
                    # unchanged, so this reduces -- value by value, in the same
                    # order -- to the flat mean it has always been.
                    agged = [self._completion_agg(leaf[x:y]) for (x, y, _r) in subs]
                    group_scores.append(
                        sum(agged) / float(len(agged)) if agged else 0.0
                    )
                    present = [float(m) for m in myopic[a:b] if m is not None]
                    group_myopic.append(sum(present) / float(len(present)) if present else None)
                    group_sample_scores.append(samples)
                    group_n_samples.append(len(subs))
                    group_n_boards.append(b - a)
                    group_aggs.append(list(agged))
                    group_keys_in_order.append(key)
            leaves_evaluated += len(boards)

            chunks += 1
            total_s = time.perf_counter() - chunk_started
            chunk_info = {
                "chunk": int(chunks),
                "size": int(size),
                "n_generated": int(generated),
                "n_dedup": int(len(order)),
                "new_prefixes": int(len(kept)),
                "leaves": int(len(boards)),
                "candidate_s": float(candidate_s),
                "outcomes_s": float(outcomes_s),
                "myopic_s": float(myopic_s),
                "value_s": float(value_s),
                "total_s": float(total_s),
            }
            chunk_timings.append(chunk_info)
            for key in timing_totals:
                timing_totals[key] += float(chunk_info[key])
            if on_chunk is not None:
                on_chunk(copy.deepcopy(chunk_info))
            if should_stop is not None and should_stop():
                stopped = True
                break


        random_groups = [
            gi for gi, key in enumerate(order)
            if groups[key]["walk"].get("boundary") is not None
        ]
        extra_levels = 0


        extra_level_seconds: list[float] = []
        if extra_samples_until_stop and not stopped and order:
            next_r = int(self.stochastic_samples)
            # A hard ceiling as well as the clock. The clock is the normal stop
            # and this never fires in production, but the loop is otherwise
            # bounded only by `should_stop` and by a level failing, and a
            # mutation run that removed the completeness guard turned it into
            # an infinite loop rather than a failing test. An unbounded loop in
            # the inference path is worth two lines to close.
            max_extra_levels = MAX_EXTRA_SAMPLE_LEVELS
            while extra_levels < max_extra_levels:
                if should_stop is None or should_stop():
                    if should_stop is not None:
                        stopped = True
                    break
                next_r += 1
                level_started = time.perf_counter()
                level: list[tuple[int, list[dict[str, Any]]]] = []
                complete = True
                for gi, key in enumerate(group_keys_in_order):
                    walk = groups[key]["walk"]
                    if walk.get("boundary") is None:
                        # A deterministic prefix is one board with no sample to
                        # add. Not a failure: it does not take part.
                        continue
                    got, span = self._anytime_sample_boards(walk, next_r, 0)
                    if span is None:
                        complete = False
                        break
                    level.append((gi, got))
                # BOTH conditions. `complete` was set and never read in the
                # first draft of this loop, so a level whose second group
                # failed still committed its first -- the exact half level the
                # comment above forbids, and nothing would have said so.
                if not complete or not level:
                    break
                level_boards: list[dict[str, Any]] = []
                level_spans: list[tuple[int, int]] = []
                for _gi, got in level:
                    level_spans.append((len(level_boards), len(level_boards) + len(got)))
                    level_boards.extend(got)
                level_myopic = self._honest_myopic_scores(level_boards, opponent_team)
                scored_level = self.vgame_scorer.score_boards(
                    level_boards, myopic_scores=level_myopic, **race
                )
                level_leaf = [float(x) for x in scored_level["score"]]
                if len(level_leaf) != len(level_boards):
                    raise RuntimeError(
                        f"vgame_scorer_length_mismatch:{len(level_leaf)}!={len(level_boards)}"
                    )
                v_scores.extend(float(x) for x in scored_level["v"])
                ensemble_std.extend(float(x) for x in scored_level["ensemble_std"])
                all_myopic.extend(level_myopic)
                lvl_decided, lvl_divergent = self._count_completion_decisions(
                    level_leaf, [[(a, b, next_r)] for (a, b) in level_spans]
                )
                decided_here += lvl_decided
                divergent_here += lvl_divergent
                for (gi, got), (a, b) in zip(level, level_spans):
                    group_aggs[gi].append(self._completion_agg(level_leaf[a:b]))
                    group_scores[gi] = sum(group_aggs[gi]) / float(len(group_aggs[gi]))
                    group_n_samples[gi] += 1
                    group_n_boards[gi] += len(got)
                leaves_evaluated += len(level_boards)
                extra_levels += 1
                extra_level_seconds.append(time.perf_counter() - level_started)
                if on_level is not None:
                    # AFTER the level is committed, never during: a half level
                    # does not enter the ranking, so it must not enter the
                    # readout either. Reporting a level in progress would show
                    # a k the search is not actually using.
                    on_level({
                        "extra_levels": int(extra_levels),
                        "realised_stochastic_samples": int(self.stochastic_samples) + int(extra_levels),
                        "level_seconds": extra_level_seconds[-1],
                        "leaves_evaluated": int(leaves_evaluated),
                    })

        self._completion_decided += decided_here
        self._completion_divergent += divergent_here

        if not order:
            # Nothing survived (only reachable if every group in every chunk
            # dropped, which the greedy-anchor guard above already refuses).
            return _annotate_skip(greedy_result)

        # Same ranking as the one-pass path -- `(group score, group myopic)`,
        # first maximum wins, so a full tie keeps the greedy chain (group 0)
        # and search can never do worse than greedy. The one difference is
        # the EPSILON, and it is not cosmetic: the one-pass path scores every
        # board in ONE batched V forward while this one scores a batch per
        # chunk, and torch's CPU reduction is not bit-identical across batch
        # shapes. Measured on a real duel state, two prefixes that tie to all
        # 17 digits in one pass came out 2.2e-8 apart when chunked, which
        # silently flipped the tie-break away from the earlier candidate.
        # 1e-6 is ~50x that noise and ~1000x below any V difference that
        # means anything, so this restores the first-maximum property
        # without ever overriding a real preference.
        best_i = 0
        for i in range(1, len(order)):
            gap = group_scores[i] - group_scores[best_i]
            if gap > SCORE_TIE_EPS:
                best_i = i
            elif abs(gap) <= SCORE_TIE_EPS:
                mine = group_myopic[i] if group_myopic[i] is not None else 0.0
                best = group_myopic[best_i] if group_myopic[best_i] is not None else 0.0
                if mine > best + SCORE_TIE_EPS:
                    best_i = i
        best_key = order[best_i]
        best_walk = groups[best_key]["walk"]
        winner = groups[best_key]["candidate"]
        chosen_chain = self._prefix_chain(best_walk)

        candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains:
            candidate_chains = [self._prefix_chain(groups[key]["walk"]) for key in order]

        result = dict(winner)
        result.update(
            {
                **_searched_verdict(chosen_chain),
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_turn_mode": self.turn_mode,
                "search_n_candidates": len(order),
                # Deliberately the REAL count, not `self.n_candidates` -- see
                # `recommend_anytime`'s docstring.
                "search_n_generated": int(generated),
                "search_n_dedup": len(order),


                "search_completion_policy": self.completion_policy,
                "search_completion_width": int(self.completion_width),
                "search_completion_aggregate": self.completion_aggregate,
                "search_completion_decided": int(decided_here),
                "search_completion_divergent": int(divergent_here),


                "search_realised_stochastic_samples": (
                    int(self.stochastic_samples) + int(extra_levels)
                    if random_groups else None
                ),
                "search_extra_sample_levels": int(extra_levels) if random_groups else None,
                "search_extra_level_seconds": [round(float(x), 4) for x in extra_level_seconds],
                # Level synchrony is only meaningful ACROSS the loop; a group
                # can already arrive unequal, because a sample that raises is
                # skipped with `continue` in the baseline pass. Pre-existing and
                # not introduced here (the other gears simply never deepen), so
                # it is COUNTED rather than rewritten around: the ranking core
                # is not the place to fix a defect whose measured incidence is
                # still zero. Deterministic groups are excluded on purpose --
                # they legitimately hold exactly one board.
                "search_group_samples_unequal": bool(
                    len({group_n_samples[i] for i in random_groups}) > 1
                ) if random_groups else False,
                **self._completion_instruments(),
                "search_scores": group_scores,
                "search_chosen_index": best_i,
                "search_greedy_score": group_myopic[0] if group_myopic else None,
                "search_candidate_chains": candidate_chains,
                "search_anytime": {
                    "chunk_size": int(chunk_size),
                    "chunk_sizes": chunk_sizes,
                    "chunks": int(chunks),
                    "width_requested": int(width),
                    "width_searched": int(generated),
                    "unique_prefixes": int(len(order)),
                    "leaves_evaluated": int(leaves_evaluated),
                    "stage_seconds": timing_totals,
                    "chunk_timings": chunk_timings,
                    "stopped": bool(stopped),
                },
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "turn_mode": self.turn_mode,
                    "stochastic_samples": int(self.stochastic_samples),
                    "realised_stochastic_samples": int(self.stochastic_samples) + int(extra_levels),
                    "extra_sample_levels": int(extra_levels),
                    "extra_level_seconds": [round(float(x), 4) for x in extra_level_seconds],
                    "completion_second_chance_nodes": int(self._completion_second_chance),
                    "completion_boards_mean": (
                        float(self._completion_boards) / float(self._completion_samples)
                        if self._completion_samples
                        else None
                    ),
                    "completion_dropped": int(self._completion_dropped),
                    "imagination_key_source": self._imagination_key_source(),
                    "imagination_segment_index": int(self._imagination_segment_index),
                    "imagination_sample_seeds": [
                        self._imagination_seed(r)
                        for r in range(1, self.stochastic_samples + 1)
                    ],
                    "prefix_boundaries": [groups[key]["walk"]["boundary"] for key in order],
                    "prefix_stops": [groups[key]["walk"]["stop"] for key in order],
                    "prefix_lengths": [len(groups[key]["walk"]["ops"]) for key in order],
                    # SAMPLES, not boards. Under `v_search` one sample
                    # carries up to `completion_width` boards, so the length of
                    # a group's score list stopped being its sample count.
                    "prefix_n_samples": list(group_n_samples),
                    "prefix_n_boards": list(group_n_boards),
                    "prefix_n_dropped": int(n_dropped),
                    "prefix_sample_scores": group_sample_scores,
                    "group_scores": group_scores,
                    "greedy_leaf_score": (group_scores[0] if group_scores else None),
                    "v_scores": v_scores,
                    "leaf_scores": [s for samples in group_sample_scores for s in samples],
                    "ensemble_std": ensemble_std,
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in all_myopic
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                    "anytime": {
                        "chunk_size": int(chunk_size),
                        "chunk_sizes": chunk_sizes,
                        "chunks": int(chunks),
                        "width_requested": int(width),
                        "width_searched": int(generated),
                        "unique_prefixes": int(len(order)),
                        "leaves_evaluated": int(leaves_evaluated),
                        "stage_seconds": timing_totals,
                        "chunk_timings": chunk_timings,
                        "stopped": bool(stopped),
                    },
                },
            }
        )
        return result

    def recommend(self, state: dict[str, Any]) -> dict[str, Any]:
        """API-compatible with `BcRecommender.recommend` -- see module
        docstring for the full algorithm. Never raises: any failure
        degrades to the greedy result (or, if even the greedy call itself
        fails, to a minimal `ok=False` shape mirroring `BcRecommender`'s
        own hard-failure contract) rather than propagating.
        """
        self._recommend_call_count += 1


        turn_value = state.get("turn") if isinstance(state, dict) else None
        try:
            self._decision_turn = int(turn_value) if turn_value is not None else None
        except (TypeError, ValueError):
            self._decision_turn = None


        self._state_seed = honest_frame.read_engine_seed(state) if isinstance(state, dict) else 0
        try:
            # A2.5: candidate 0 is the greedy ANCHOR -- the tie-break that
            # makes "search can never do worse than greedy" true, the
            # `_annotate_skip` fallback, and the thing a plain-BC arm is
            # comparable to. `.deterministic` does not reach the decoder
            # (see `_call_bc_recommend`), so under `--decode-mode sample` it
            # came back sampled and all three properties were lost.
            greedy_result = self._call_bc_recommend(
                state, deterministic=True, force_ranked_decode=True
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"search_greedy_call_failed:{exc}",
                "recommended_action": None,
                "chain_preview": [],
                "wdl_probs": None,
                "diagnostics": {},
                "search_used": False,
                "search_n_candidates": 0,
                "search_scores": [],
                "search_chosen_index": 0,
                "search_greedy_score": None,
                "search_error": str(exc),
            }

        try:
            return self._search(state, greedy_result)
        except Exception as exc:
            result = dict(greedy_result)
            result.update(
                {
                    "search_used": False,
                    "search_n_candidates": 0,
                    "search_scores": [],
                    "search_chosen_index": 0,
                    "search_greedy_score": None,
                    "search_error": str(exc),
                }
            )
            return result
