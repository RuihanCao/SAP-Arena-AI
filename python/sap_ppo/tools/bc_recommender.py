"""Loads a trained MaskablePPO BC checkpoint (see `train/train_chain_bc.py`) and
the matching observation encoder (`train/observation.py`), and decodes a full
one-turn action chain from an engine state by repeated deterministic
decoding, masked at every step to ENGINE-TRUE legality.

Visited-state anti-cycle guard (added this session): ~40% of chains used to
hit the 25-step cap without ever emitting END_TURN, via degenerate cycles
(FREEZE(i)<->UNFREEZE(i) toggling; reorder thrash) -- a plain per-step
argmax has no memory of where it has already been, so it can walk in a
circle forever (bounded only by the step cap) and gets scored on a
thrashed partial board instead of a real stopping point. The decode loop
below fixes this by tracking every within-turn state it has already
visited (`_state_signature`) and, at each step, scanning LEGAL actions in
descending predicted-probability order for the first one that either is
END_TURN or leads to a genuinely new (unvisited) state -- skipping any
candidate that would revisit an already-seen state or that fails to apply.
Because the within-turn state space is finite and no state is ever
revisited, this guarantees the walk terminates without relying on the
step cap; the cap (`DEFAULT_MAX_CHAIN_STEPS`) remains only as a backstop
against an unusually long but genuine chain (see its docstring below).

Contract (matches `_recommend_next_action_with_timeout`'s return shape, so
`eval_tempo_planner.run_eval_cases` can swap recommenders with no other
change to its scoring/dump/image pipeline -- see that module's
`--recommender bc` wiring):

    {ok: bool, error: str | None, recommended_action: dict | None,
     chain_preview: list[dict], wdl_probs: None, diagnostics: dict}

`diagnostics["stop_reason"]` is one of:
  - "end_turn_chosen": the scan reached END_TURN and took it (the policy's
    own choice to stop -- whether END_TURN was the single top-ranked legal
    action, or was reached after skipping higher-ranked candidates that
    would have cycled).
  - "end_turn_forced_no_progress": defensive-only fallback -- every legal
    action was exhausted (cycled or failed to apply) without the scan ever
    encountering END_TURN itself. Not expected to fire in practice (the
    engine appends END_TURN unconditionally to every state's legal action
    set), but the loop must still terminate cleanly if it ever does.
  - "cap_reached": `DEFAULT_MAX_CHAIN_STEPS` genuine (non-cycling) progress
    steps were taken without the policy ever choosing to stop.
A small set of pre-existing, unrelated hard-failure reasons
("legal_mask_failed", "no_legal_actions") remain possible and are NOT part
of this enum; they guard against a mapping/schema bug, not against cycling,
and were not touched by this change.

Optional measurement aid (env-var gated, off by default, added this
session purely to support eval-time reporting without touching the eval
harness): if the `SAP_BC_DECODE_LOG` environment variable is set to a file
path, every `recommend()` call appends one JSON line
`{turn, stop_reason, chain_length, chain_kinds}` to it. This has zero effect
unless that variable is set."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


from ..api import imagined_legal_actions as engine_legal_actions


from ..api import imagined_step as engine_step
from ..train.env import (
    ACTION_CATALOG,
    ACTION_INDEX_BY_KEY,
    ACTION_INDEX_BY_TUPLE_KEY,
    _action_index_key,
    _action_key,
)
from ..train.observation import OBSERVATION_MODE_V4, build_state_encoder
from ..visited_guard import set_training_rolls_this_turn as _set_training_rolls_this_turn
from ..visited_guard import state_signature as _state_signature

# Matches chain_bc_dataset.py / train_bc_warmstart.py's turn-normalization
# divisor convention (turns beyond this clamp to 1.0 in the encoder; never an
# error, never dropped).
DEFAULT_MAX_TURN = 15

# With the visited-state anti-cycle guard below, decode termination is
# already GUARANTEED (the within-turn state space is finite and no state is
# ever revisited), so this cap is no longer the primary defense against a
# non-terminating decode -- it is now a pure backstop against an unusually
# long but genuine chain (e.g. many distinct buys/sells/rolls on a
# big-gold turn), kept exactly as before.
DEFAULT_MAX_CHAIN_STEPS = 25

# Environment variable gating the optional per-decode diagnostics sink (see
# module docstring). Unset (the default) => zero side effects.
_DECODE_LOG_ENV_VAR = "SAP_BC_DECODE_LOG"


DECODE_MODE_RANKED = "ranked"
DECODE_MODE_SAMPLE = "sample"
DECODE_MODES: tuple[str, ...] = (DECODE_MODE_RANKED, DECODE_MODE_SAMPLE)
DEFAULT_DECODE_MODE = DECODE_MODE_RANKED
DEFAULT_SAMPLE_TEMPERATURE = 1.0
DEFAULT_SAMPLE_SEED = 0

# Canonical END_TURN catalog entry + index, resolved once at import time via
# the same catalog/key machinery `legal_mask` uses -- never hand-rolled --
# so the forced-fallback path (see `_choose_next_action`) always appends the
# exact action the rest of the pipeline (engine, dataset, catalog) agrees is
# END_TURN. Fails fast at import if the catalog ever stopped containing it.
_END_TURN_ACTION_INDEX = ACTION_INDEX_BY_KEY[_action_key({"type": "END_TURN"})]
_END_TURN_ACTION = ACTION_CATALOG[_END_TURN_ACTION_INDEX]


def legal_mask(state: dict[str, Any]) -> np.ndarray:
    """Engine-true legal-action mask over the 309-entry ACTION_CATALOG.

    Built from `api.legal_actions` (the real game rules), NOT
    `TrainingEnv.legal_actions` / `.legal_action_mask()` -- that class's
    legality ALSO bakes in RL anti-spam heuristics (END_TURN requires
    gold==0, SELL requires >2 pets, a tactical-action lock after
    REORDER/FREEZE/UNFREEZE) which are the one-turn PPO agent's own training
    scaffolding, not game rules (see this module's docstring)."""
    mask = np.zeros(len(ACTION_CATALOG), dtype=bool)
    unmapped: list[dict[str, Any]] = []
    for action in engine_legal_actions(state):
        idx = ACTION_INDEX_BY_TUPLE_KEY.get(_action_index_key(action))
        if idx is None:
            unmapped.append(action)
            continue
        mask[idx] = True
    if unmapped:
        raise RuntimeError(
            f"legal_mask_unmapped_actions:count={len(unmapped)}:examples={unmapped[:3]}"
        )
    return mask


def _choose_next_action(
    work: dict[str, Any],
    mask: np.ndarray,
    probs: np.ndarray,
    visited: set[str],
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Scan legal actions in descending predicted-probability order.

    Returns `(kind, action, next_state)`:
      - `("end_turn", action, None)`: END_TURN reached in the scan -- the
        policy's own choice to stop. This can happen either because
        END_TURN was the single highest-probability legal action, or
        because higher-ranked candidates were tried first and skipped for
        cycling into an already-visited state (see below); either way,
        nothing before it in the scan produced genuine progress.
      - `("progress", action, next_state)`: the highest-ranked legal action
        whose resulting state is NOT in `visited` -- genuine forward
        progress. `action` was appended and `next_state` becomes the new
        `work` for the following step.
      - `("forced_end_turn", action, None)`: defensive-only fallback,
        reached iff the scan exhausted every legal action (each one either
        cycling back into `visited` or raising on `api.step`) WITHOUT ever
        encountering END_TURN. Never observed with the current engine --
        `engine.legal_actions` appends `{"type": "END_TURN"}` unconditionally
        to every state, so it is always in `mask` and the scan always
        reaches it eventually if nothing else was chosen first -- but the
        loop must still terminate cleanly if that ever stopped holding.
    """
    legal_idx = np.flatnonzero(mask)
    order = legal_idx[np.argsort(-probs[legal_idx], kind="stable")]
    for idx in order:
        action = ACTION_CATALOG[int(idx)]
        action_type = str(action.get("type") or "").strip().upper()
        if action_type == "END_TURN":
            return "end_turn", action, None
        try:
            trans = engine_step(work, action)
        except Exception:
            continue  # would raise -- try the next-best legal candidate
        if not trans.get("legal"):
            continue  # defense-in-depth only; the mask should prevent this
        next_state = trans.get("state_after")
        if not isinstance(next_state, dict):
            continue
        if _state_signature(next_state) in visited:
            continue  # would revisit an already-seen within-turn state
        return "progress", action, next_state
    return "forced_end_turn", _END_TURN_ACTION, None


def _sample_rng(sample_seed: int, game_seed: int, turn: int, step: int) -> np.random.Generator:
    """Deterministic per-(game, turn, step) RNG for `decode_mode="sample"`.

    `game_seed` is `state["meta"]["seed"]` at decode time -- already a
    deterministic function of (a caller's own eval seed, game_index) via
    e.g. `eval_versus_fullgame.py::_new_game_state`'s `engine_seed_rng`, so
    seeding on `(sample_seed, game_seed, turn, step)` is equivalent to
    seeding on `(sample_seed, game_index, turn, step)` without this module
    ever needing to know the game index itself -- `recommend()` stays a
    pure function of `(self, state)`. `numpy.random.default_rng` accepts a
    sequence of integers and runs them through `SeedSequence`, the
    documented, robust way to combine several integer keys into one
    independent, reproducible stream (no ad-hoc hashing of our own).
    """
    return np.random.default_rng(
        [
            int(sample_seed) & 0xFFFFFFFF,
            int(game_seed) & 0xFFFFFFFF,
            int(turn) & 0xFFFFFFFF,
            int(step) & 0xFFFFFFFF,
        ]
    )


def _choose_next_action_sampled(
    work: dict[str, Any],
    mask: np.ndarray,
    probs: np.ndarray,
    visited: set[str],
    *,
    rng: np.random.Generator,
    temperature: float,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Same contract (`(kind, action, next_state)`, same three `kind` values,
    same visited-state anti-cycle guard, same END_TURN/forced-fallback
    semantics -- see that function's docstring) but instead of a fixed
    probability-descending scan, repeatedly SAMPLES without replacement from
    the masked, optionally temperature-scaled distribution over LEGAL
    actions: draw one candidate; if it is END_TURN, take it immediately (END
    TURN is always acceptable -- there is no "next state" for it to revisit);
    otherwise try to apply it, and if that fails OR would revisit an
    already-seen state, remove it from the draw pool and re-sample among
    what remains. Exhausting every legal action without ever drawing
    END_TURN returns `forced_end_turn`, identically to the ranked scan (see
    its docstring for why this is defensive-only and not expected to fire).

    `temperature` rescales the distribution BEFORE sampling
    (`softmax(log(p) / temperature)`); `1.0` samples the predicted
    distribution unchanged. `rng` is a fully-seeded `numpy.random.Generator`
    (see `_sample_rng`) so the whole draw sequence for one decode step is
    reproducible independent of call order or thread count."""
    legal_idx = np.flatnonzero(mask)
    weights = np.clip(probs[legal_idx].astype(np.float64), 1e-12, None)
    if float(temperature) != 1.0:
        log_weights = np.log(weights) / float(temperature)
        log_weights -= log_weights.max()  # numerical stability only
        weights = np.exp(log_weights)
    remaining = np.ones(legal_idx.shape[0], dtype=bool)

    while remaining.any():
        candidate_positions = np.flatnonzero(remaining)
        candidate_weights = weights[candidate_positions]
        candidate_weights = candidate_weights / candidate_weights.sum()
        drawn_position = int(rng.choice(candidate_positions, p=candidate_weights))
        idx = int(legal_idx[drawn_position])
        action = ACTION_CATALOG[idx]
        action_type = str(action.get("type") or "").strip().upper()
        if action_type == "END_TURN":
            return "end_turn", action, None
        try:
            trans = engine_step(work, action)
        except Exception:
            remaining[drawn_position] = False
            continue  # would raise -- resample among the remaining candidates
        if not trans.get("legal"):
            remaining[drawn_position] = False
            continue  # defense-in-depth only; the mask should prevent this
        next_state = trans.get("state_after")
        if not isinstance(next_state, dict):
            remaining[drawn_position] = False
            continue
        if _state_signature(next_state) in visited:
            remaining[drawn_position] = False
            continue  # would revisit an already-seen within-turn state
        return "progress", action, next_state
    return "forced_end_turn", _END_TURN_ACTION, None


class BcRecommender:
    """Loaded BC policy + encoder; autoregressively decodes one full turn."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        observation_mode: str = OBSERVATION_MODE_V4,
        max_turn: int = DEFAULT_MAX_TURN,
        deterministic: bool = True,
        max_chain_steps: int = DEFAULT_MAX_CHAIN_STEPS,
        decode_mode: str = DEFAULT_DECODE_MODE,
        sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
    ) -> None:
        from sb3_contrib import MaskablePPO

        self.checkpoint_path = str(checkpoint_path)
        self.model = MaskablePPO.load(self.checkpoint_path)
        self.model.policy.set_training_mode(False)
        self.encoder = build_state_encoder(observation_mode=observation_mode, max_turn=int(max_turn))
        # NOTE: no longer consulted by `recommend()` -- the guarded scan
        # below is an inherently deterministic probability-ranked walk (see
        # module docstring), not a sample-vs-argmax choice. Kept only so an
        # existing/future caller passing this kwarg doesn't break.
        self.deterministic = bool(deterministic)
        self.max_chain_steps = int(max_chain_steps)


        if decode_mode not in DECODE_MODES:
            raise ValueError(
                f"bc_recommender_bad_decode_mode:{decode_mode!r}:expected_one_of={DECODE_MODES}"
            )
        if decode_mode == DECODE_MODE_SAMPLE and not (float(sample_temperature) > 0.0):
            raise ValueError(
                f"bc_recommender_bad_sample_temperature:{sample_temperature!r}:must_be_positive"
            )
        self.decode_mode = str(decode_mode)
        self.sample_temperature = float(sample_temperature)
        self.sample_seed = int(sample_seed)


        model_obs_shape = tuple(int(x) for x in self.model.observation_space.shape)
        model_obs_size = int(np.prod(model_obs_shape)) if model_obs_shape else 0
        if model_obs_size != int(self.encoder.size):
            raise RuntimeError(
                f"bc_recommender_obs_size_mismatch:checkpoint={model_obs_size}:"
                f"encoder={self.encoder.size}:observation_mode={observation_mode}:"
                f"checkpoint_path={self.checkpoint_path}"
            )
        model_n_actions = int(getattr(self.model.action_space, "n", -1))
        if model_n_actions != len(ACTION_CATALOG):
            raise RuntimeError(
                f"bc_recommender_action_size_mismatch:checkpoint={model_n_actions}:"
                f"catalog={len(ACTION_CATALOG)}:checkpoint_path={self.checkpoint_path}"
            )

    def _masked_action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Full probability vector (all 309 entries) over the action catalog.

        Mirrors what `MaskablePPO.predict()` does internally (`obs_to_tensor`
        -> `policy.get_distribution(..., action_masks=mask)`) but returns the
        whole post-masking distribution instead of collapsing it to a single
        argmax/sample, so `_choose_next_action` can rank LEGAL candidates by
        probability and fall through to the next-best one when the top
        choice would cycle. Illegal entries come back near-zero, not exactly
        zero (sb3-contrib sets a large-negative logit, not -inf), but that
        never matters here: `_choose_next_action` only ever indexes into
        `np.flatnonzero(mask)`, never raw rank order over all 309.

        Works identically whether the checkpoint's policy class is the
        stock `MaskableActorCriticPolicy` or this repo's
        `TypeBalancedMaskablePolicy` (both checkpoints in play, rank_v2 and
        flat_v2, use the latter) -- both expose the same
        `get_distribution(obs, action_masks)` -> `MaskableDistribution`
        contract; the type-balancing prior only changes the logits that
        contract is built from, not its shape.
        """
        import torch as th

        policy = self.model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with th.no_grad():
            distribution = policy.get_distribution(obs_tensor, action_masks=mask)
        probs = distribution.distribution.probs
        return probs.detach().cpu().numpy().reshape(-1)

    def recommend(self, state: dict[str, Any]) -> dict[str, Any]:
        """Autoregressively decode one turn's action chain from `state`.

        Loop: encode state -> legal_mask(state) -> rank LEGAL actions by
        predicted probability -> take the first that is either END_TURN or
        leads to a genuinely new (unvisited) state, skipping any that would
        revisit one already seen this turn or that fails to apply ->
        repeat, until END_TURN is chosen or the step cap is hit. See the
        module docstring for the full anti-cycle-guard rationale and the
        `stop_reason` enum. The SAME `state["meta"]["versus"]` opponent-
        context block the caller attached before calling this stays on the
        state across the whole walk: `api.step`/`engine.apply_action` deep-
        copies the incoming state and mutates only the fields the action
        touches, so unrelated `meta` keys (including this one) survive every
        step automatically -- verified directly against `engine.py` before
        relying on it here."""
        work = copy.deepcopy(state)
        _set_training_rolls_this_turn(work, 0)
        chain: list[dict[str, Any]] = []
        encountered_errors: list[str] = []
        fallbacks_used: list[str] = []
        visited: set[str] = {_state_signature(work)}
        stop_reason = "cap_reached"


        game_seed = 0
        if self.decode_mode == DECODE_MODE_SAMPLE:
            meta = work.get("meta")
            if isinstance(meta, dict):
                try:
                    game_seed = int(meta.get("seed", 0) or 0)
                except (TypeError, ValueError):
                    game_seed = 0

        for step_index in range(self.max_chain_steps):
            try:
                mask = legal_mask(work)
            except Exception as exc:
                encountered_errors.append(f"legal_mask_failed:{type(exc).__name__}:{exc}")
                stop_reason = "legal_mask_failed"
                break
            if not bool(mask.any()):
                encountered_errors.append("no_legal_actions")
                stop_reason = "no_legal_actions"
                break

            obs = self.encoder.encode(work)
            probs = self._masked_action_probs(obs, mask)
            if self.decode_mode == DECODE_MODE_SAMPLE:
                rng = _sample_rng(self.sample_seed, game_seed, int(work.get("turn") or 0), step_index)
                kind, action, next_state = _choose_next_action_sampled(
                    work, mask, probs, visited, rng=rng, temperature=self.sample_temperature
                )
            else:
                kind, action, next_state = _choose_next_action(work, mask, probs, visited)

            if kind == "end_turn":
                chain.append(copy.deepcopy(action))
                stop_reason = "end_turn_chosen"
                break
            if kind == "forced_end_turn":
                encountered_errors.append("no_progressable_action_end_turn_not_legal")
                fallbacks_used.append("forced_end_turn")
                chain.append(copy.deepcopy(action))
                stop_reason = "end_turn_forced_no_progress"
                break


            chain.append(copy.deepcopy(action))
            work = next_state
            visited.add(_state_signature(work))
        else:
            stop_reason = "cap_reached"

        ok = bool(chain) and stop_reason in (
            "end_turn_chosen",
            "end_turn_forced_no_progress",
            "cap_reached",
        )

        _maybe_log_decode(work, stop_reason=stop_reason, chain=chain)

        return {
            "ok": ok,
            "error": (None if ok else stop_reason),
            "recommended_action": (copy.deepcopy(chain[0]) if chain else None),
            "chain_preview": chain,
            "wdl_probs": None,
            "diagnostics": {
                "stop_reason": stop_reason,
                "chain_length": len(chain),
                "encountered_errors": encountered_errors,
                "fallbacks_used": fallbacks_used,
            },
        }


def _maybe_log_decode(work: dict[str, Any], *, stop_reason: str, chain: list[dict[str, Any]]) -> None:
    """Best-effort append to `SAP_BC_DECODE_LOG` if set; see module docstring.

    Never raises: a measurement sink must not be able to break a decode.
    """
    log_path = os.environ.get(_DECODE_LOG_ENV_VAR, "").strip()
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "turn": work.get("turn"),
                        "stop_reason": stop_reason,
                        "chain_length": len(chain),
                        "chain_kinds": [str(a.get("type") or "") for a in chain],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    except OSError:
        pass
