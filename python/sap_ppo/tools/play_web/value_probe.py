"""Read-only board valuations, optionally after a greedy BC completion.

The response distinguishes a raw score from a calibrated trophy estimate and
includes the value target and continuation policy. A probe never runs search.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ...api import legal_actions, step as engine_step
from ..honest_frame import imagined_clone, imagination_seed
from .agent import build_readonly_models


VALUE_KIND_PROBABILITY = "probability"
VALUE_KIND_TEACHER = "squashed_teacher_score"
VALUE_KIND_TROPHIES = "bounded_trophies"


VALUE_KIND_LEAFMAP = "leafmap_trophies"


from ..._artifact_defaults import TROPHY_CURVE_REL_PATH as CURVE_REL_PATH  # noqa: E402
CURVE_SHA256 = "6edec500ed7ff8bacf68cb8902257b7538816853642da74d55bd12bf8517f3de"


VALUE_PROBE_SEGMENT_INDEX = -1

END_TURN = "END_TURN"

MEANING_END_OF_TURN = "end_of_turn"
MEANING_BC_COMPLETED = "bc_completed"

# THE LABEL IS PART OF THE FEATURE. Keyed by the `completed` flag so the wording
# and the flag are one decision, not two that can drift apart.
MEANINGS: dict[str, str] = {
    MEANING_END_OF_TURN: (
        "This board, valued as it stands at the end of the turn. "
        "Trophies from here if a BC-level player continues -- not if you "
        "continue, not if the agent continues, and not a win probability."
    ),
    MEANING_BC_COMPLETED: (
        "NOT the value of this action. It means: if you take this action and "
        "then let BC finish the turn for you, the resulting board is worth "
        "this much. Trophies from here if a BC-level player continues -- not "
        "if you continue, not if the agent continues, and not a win "
        "probability."
    ),
}

# What `search.ran` says on every response, so the one property the design makes
# this endpoint's reason for existing is on the wire rather than in a comment.
NO_SEARCH_NOTE = (
    "no search: one BC greedy completion per completed board, then one batched "
     "V forward over all of them. This is not a search score."

)


class ValueProbeError(RuntimeError):
    """A value cannot be produced. Always reported, never defaulted."""


def apply_actions(
    state: dict[str, Any],
    actions: Sequence[dict[str, Any]],
    *,
    engine_step_fn: Callable[..., dict[str, Any]] = engine_step,
) -> dict[str, Any]:
    """Replay `actions` op-by-op from a fresh deepcopy of `state`.

    The same procedure `search_recommender._apply_chain` uses, and for the same
    reason: the board scored here has to be bit-identical to the board the turn
    actually reached, not to whatever intermediate state some caller tracked.
    `END_TURN` is never applied (it does not change the shop-phase board), and a
    step that fails stops the replay -- the board reached so far is still the
    board that turn was on.
    """
    work = copy.deepcopy(state)
    for action in actions:
        if not isinstance(action, dict):
            break
        if str(action.get("type") or "").strip().upper() == END_TURN:
            break
        try:
            trans = engine_step_fn(work, action)
        except Exception:
            break
        if not trans.get("legal"):
            break
        after = trans.get("state_after")
        if not isinstance(after, dict):
            break
        work = after
    return work


def meaning_for(completed: bool) -> tuple[str, str]:
    """`(meaning_key, meaning)` for a board that was / was not completed.

    The single place the flag becomes wording. Callers must not build the pair
    themselves -- that is exactly the drift the design's acceptance 2 forbids.
    """
    key = MEANING_BC_COMPLETED if bool(completed) else MEANING_END_OF_TURN
    return key, MEANINGS[key]


def load_pinned_curve(repo_root: Path) -> dict[str, Any]:
    """Load pinned curve."""

    from ..w1_recalibration_curve import load_curve_document

    path = Path(repo_root) / CURVE_REL_PATH
    return load_curve_document(path, CURVE_SHA256)


@dataclass(frozen=True)
class BoardRequest:
    """One board to value, and what it took to get there."""

    key: str
    board: dict[str, Any]
    completed: bool
    action: dict[str, Any] | None = None
    action_index: int | None = None
    completion: dict[str, Any] | None = None
    error: str | None = None


class ValueProbe:
    """One encode plus one batched V forward. No search, ever.

    Construct with `from_agent`; the endpoint only calls `evaluate`.
    """

    def __init__(
        self,
        *,
        bc: Any,
        scorer: Any,
        model_identity: dict[str, Any],
        curve_document: dict[str, Any] | None,
        engine_step_fn: Callable[..., dict[str, Any]] = engine_step,
    ) -> None:
        self.bc = bc
        self.scorer = scorer
        self.model_identity = dict(model_identity)
        self.curve_document = curve_document
        self._engine_step = engine_step_fn
        self.value_kind = str(getattr(scorer, "value_kind", "") or "")
        self._curve_fn = None
        if isinstance(curve_document, dict):
            # Only legacy teacher-score heads use this optional calibration.
            # Native and artifact-mapped trophy heads need neither its module
            # nor its data file in the public runtime.
            from ..w1_recalibration_curve import curve_callable

            self._curve_fn = curve_callable(curve_document)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_agent(cls, agent: Any, *, share_models: bool = False) -> "ValueProbe":
        """Build from a `play_web.agent.AgentHandle`.

        Reads `agent.describe()` for identity and NOT `agent.search`: the object
        this returns has no way to reach a search even by accident.

        By default it loads its OWN (BC, V) pair from the agent's pinned
        checkpoints to avoid races with the search worker's decoder settings.
        The identity below records the models used to produce the value.

        `share_models=True` exists for tests that supply a scripted agent and
        have nothing to load.
        """
        scorer = agent.vgame_scorer
        bc = agent.bc
        if not share_models:
            try:
                bc, scorer = build_readonly_models(agent.config, repo_root=agent.repo_root)
            except Exception as exc:
                raise ValueProbeError(
                    f"value_probe_readonly_models_failed:{type(exc).__name__}:{exc}"
                ) from exc
        description = agent.describe()
        identity = {
            "agent_id": description.get("agent_id"),
            "agent_name": description.get("agent_name"),
            "model_revision": description.get("model_revision"),
            "search_revision": description.get("search_revision"),
            "baseline_id": description.get("baseline_id"),
            "bc_checkpoint": description.get("bc_checkpoint"),
            "vgame_heads": description.get("vgame_heads"),
            "vgame_extractor": description.get("vgame_extractor"),
            "value_kind": str(getattr(scorer, "value_kind", "") or ""),
            "git_sha": description.get("git_sha"),
        }
        curve = None
        if str(getattr(scorer, "value_kind", "")) == VALUE_KIND_TEACHER:
            curve = load_pinned_curve(agent.repo_root)
        return cls(
            bc=bc,
            scorer=scorer,
            model_identity=identity,
            curve_document=curve,
        )

    # -- the trophy scale --------------------------------------------------
    def trophy_scale(self) -> dict[str, Any]:
        """How this head's raw output becomes trophies, and on whose authority.

        Three heads, three answers, and an unrecognised one is an error rather
        than a default -- the same rule `vgame_scorer` applies for the same
        reason: all three scales are plausible small numbers an order of
        magnitude apart.
        """
        from ...versus_lives import MAX_TROPHIES

        trophy_range = [0.0, float(MAX_TROPHIES)]

        if self.value_kind == VALUE_KIND_TEACHER:
            document = self.curve_document or {}
            fit = document.get("fit") or {}
            return {
                "source": "recalibration_curve",
                "raw_scale": VALUE_KIND_TEACHER,
                "curve": {
                    "path": CURVE_REL_PATH,
                    "sha256": document.get("_sha256"),
                    "kind": document.get("kind"),
                    "schema_version": document.get("schema_version"),
                    "fitted_on_split": fit.get("split"),
                    "fit_source": fit.get("source"),
                    "n_fit_points": fit.get("n_points"),
                    "n_fit_games": fit.get("n_games"),
                    "x_range": list(fit.get("x_range") or []),
                    "y_range": list(fit.get("y_range") or []),
                },
                "units": "trophies",
                "range": trophy_range,
            }
        if self.value_kind == VALUE_KIND_TROPHIES:
            return {
                "source": "head_native",
                "raw_scale": VALUE_KIND_TROPHIES,
                "curve": None,
                "units": "trophies",
                "range": trophy_range,
                "note": (
                    "this head is natively expected trophies on [0, 10]; the "
                    "recalibration curve is NOT applied to it"
                ),
            }
        if self.value_kind == VALUE_KIND_LEAFMAP:
            return {
                "source": "artifact_leaf_map",
                "raw_scale": VALUE_KIND_LEAFMAP,
                "curve": None,
                "units": "trophies",
                "range": trophy_range,
                "note": (
                    "this head outputs a squashed teacher score and a fixed "
                    "monotone map carried in the artifact lifts it onto trophies, "
                    "so the value that reaches this panel is already trophies. "
                    "The recalibration curve is NOT applied on top: that would "
                    "map a mapped value twice. The map is reported by the "
                    "scorer's own describe(), not re-derived here"
                ),
            }
        if self.value_kind == VALUE_KIND_PROBABILITY:
            return {
                "source": None,
                "raw_scale": VALUE_KIND_PROBABILITY,
                "curve": None,
                "units": None,
                "range": None,
                "note": (
                    "this head emits a win probability, which the trophy "
                    "recalibration is not defined on; the raw output is "
                    "reported and no trophy number is invented for it"
                ),
            }
        raise ValueProbeError(f"value_probe_unknown_value_kind:{self.value_kind!r}")

    def _to_trophies(self, raw: float) -> dict[str, Any]:
        """One raw head output -> the number a human reads, plus its caveats."""
        if self.value_kind in (VALUE_KIND_TROPHIES, VALUE_KIND_LEAFMAP):
            # Already trophies in both cases -- natively for one, via the
            # artifact's frozen map for the other. Neither takes the curve.
            return {"trophies": float(raw), "clamped": False, "clamp_side": None}
        if self.value_kind == VALUE_KIND_PROBABILITY:
            return {"trophies": None, "clamped": False, "clamp_side": None}
        if self._curve_fn is None:
            raise ValueProbeError("value_probe_curve_not_loaded")
        fit = (self.curve_document or {}).get("fit") or {}
        x_range = list(fit.get("x_range") or [])
        clamp_side: str | None = None
        if len(x_range) == 2:
            if float(raw) < float(x_range[0]):
                clamp_side = "below"
            elif float(raw) > float(x_range[1]):
                clamp_side = "above"
        return {
            "trophies": float(self._curve_fn(float(raw))),
            # The interpolator is clamped outside its knots, so an out-of-range
            # board pins to an endpoint and looks like a confident answer. A
            # human board is exactly where that happens (DESIGN section 6), so
            # it is reported instead of being invisible.
            "clamped": clamp_side is not None,
            "clamp_side": clamp_side,
        }

    # -- the boards --------------------------------------------------------
    def _probe_seed(self, state: dict[str, Any], index: int) -> int:
        meta = state.get("meta") if isinstance(state, dict) else None
        try:
            engine_seed = int((meta or {}).get("seed") or 0)
        except (TypeError, ValueError):
            engine_seed = 0
        try:
            turn = int(state.get("turn") or 0)
        except (TypeError, ValueError):
            turn = 0
        return imagination_seed(
            engine_seed=engine_seed,
            turn=turn,
            segment_index=VALUE_PROBE_SEGMENT_INDEX,
            sample_r=int(index),
        )

    def _complete_greedily(self, state: dict[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Let BC finish the turn from `state`, greedily, on stream S.

        Returns `(end_board, completion_telemetry)`. The chain is applied with
        the same op-by-op engine replay the search uses on a candidate, so the
        board scored here is the board the turn would actually reach.
        """
        work = imagined_clone(state, seed=self._probe_seed(state, index))
        result = self.bc.recommend(work)
        chain = list(result.get("chain_preview") or [])
        board = apply_actions(work, chain, engine_step_fn=self._engine_step)
        telemetry = {
            "ok": bool(result.get("ok")),
            "stop_reason": (result.get("diagnostics") or {}).get("stop_reason"),
            "chain_length": len(chain),
            "action_types": [
                str(a.get("type"))
                for a in chain
                if isinstance(a, dict) and str(a.get("type") or "").upper() != END_TURN
            ],
        }
        return board, telemetry

    def _board_requests(
        self,
        *,
        state: dict[str, Any],
        complete: bool,
        actions: Sequence[dict[str, Any]] | None,
    ) -> list[BoardRequest]:
        requests: list[BoardRequest] = []
        if complete:
            board, completion = self._complete_greedily(state, 0)
            requests.append(
                BoardRequest(key="board", board=board, completed=True, completion=completion)
            )
        else:
            requests.append(
                BoardRequest(key="board", board=copy.deepcopy(state), completed=False)
            )
        for i, action in enumerate(actions or []):
            requests.append(self._action_request(state, action, i))
        return requests

    def _action_request(
        self, state: dict[str, Any], action: dict[str, Any], index: int
    ) -> BoardRequest:
        key = f"action:{index}"
        kind = str((action or {}).get("type") or "").strip().upper()
        if kind == END_TURN:
            # Ending the turn IS the end-of-turn board. Nothing is completed,
            # so this row is level B and carries level B's wording, which is
            # the whole reason the wording is derived from the flag.
            return BoardRequest(
                key=key,
                board=copy.deepcopy(state),
                completed=False,
                action=copy.deepcopy(action),
                action_index=index,
            )
        # `index + 1` keeps the plain board's own stream (`sample_r=0`) for the
        # plain board alone, so a per-action request never reuses it.
        work = imagined_clone(state, seed=self._probe_seed(state, index + 1))
        try:
            trans = self._engine_step(work, action)
        except Exception as exc:
            return BoardRequest(
                key=key,
                board={},
                completed=False,
                action=copy.deepcopy(action),
                action_index=index,
                error=f"action_step_raised:{type(exc).__name__}",
            )
        if not trans.get("legal") or not isinstance(trans.get("state_after"), dict):
            return BoardRequest(
                key=key,
                board={},
                completed=False,
                action=copy.deepcopy(action),
                action_index=index,
                error="action_illegal",
            )
        board, completion = self._complete_greedily(trans["state_after"], index + 1)
        return BoardRequest(
            key=key,
            board=board,
            completed=True,
            action=copy.deepcopy(action),
            action_index=index,
            completion=completion,
        )

    # -- the one call the endpoint makes -----------------------------------
    def evaluate(
        self,
        *,
        state: dict[str, Any],
        race: dict[str, int],
        complete: bool = False,
        actions: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Value `state` (and optionally each of `actions`) in one V forward.

        `race` is the DECISION's driver-true bypass block -- the same four
        scalars `search_recommender._race_scalars` builds, supplied by the
        caller for the same reason: `wins` is not readable off a board, and a
        scorer that guesses it is a silent train/serve skew.
        """
        started = time.time()
        for field in ("turn", "lives", "opponent_lives", "wins"):
            if race.get(field) is None:
                raise ValueProbeError(f"value_probe_race_missing:{field}")
        scale = self.trophy_scale()
        requests = self._board_requests(state=state, complete=complete, actions=actions)
        scorable = [r for r in requests if r.error is None]
        if not scorable:
            raise ValueProbeError("value_probe_no_scorable_boards")
        scored = self.scorer.score_boards(
            [r.board for r in scorable],
            turn=int(race["turn"]),
            lives=int(race["lives"]),
            opponent_lives=int(race["opponent_lives"]),
            wins=int(race["wins"]),
        )
        raws = [float(v) for v in scored["v"]]
        leaves = [float(v) for v in scored["score"]]
        if len(raws) != len(scorable):
            raise ValueProbeError(
                f"value_probe_length_mismatch:{len(raws)}!={len(scorable)}"
            )

        by_key: dict[str, dict[str, Any]] = {}
        for request, raw, leaf in zip(scorable, raws, leaves):
            by_key[request.key] = self._row(request, raw, leaf, scale)
        for request in requests:
            if request.error is not None:
                by_key[request.key] = {
                    "ok": False,
                    "error": request.error,
                    "action": request.action,
                    "action_index": request.action_index,
                }

        board_row = by_key["board"]
        action_rows = [
            by_key[f"action:{i}"] for i in range(len(actions or [])) if f"action:{i}" in by_key
        ]
        return {
            "ok": True,
            "error": None,
            "value": board_row,
            "actions": action_rows,
            "provenance": {
                "model": dict(self.model_identity),
                "scale": scale,
                "race": {k: int(v) for k, v in race.items()},
                "search": {"ran": False, "note": NO_SEARCH_NOTE},
                "imagination": {
                    "key": "sha256(engine_seed, turn, segment_index, sample_r)",
                    "segment_index": VALUE_PROBE_SEGMENT_INDEX,
                    "note": (
                        "completions run on an independent simulation stream at a "
                        "segment index no driver can produce, so a probe never "
                        "shares a stream with a segment the agent is planning"
                    ),
                },
            },
            "n_boards_scored": int(scored.get("n_boards") or len(scorable)),
            "elapsed_ms": int((time.time() - started) * 1000),
        }

    def _row(
        self,
        request: BoardRequest,
        raw: float,
        leaf: float,
        scale: dict[str, Any],
    ) -> dict[str, Any]:
        meaning_key, meaning = meaning_for(request.completed)
        mapped = self._to_trophies(raw)
        row: dict[str, Any] = {
            "ok": True,
            "error": None,
            "raw": float(raw),
            "raw_scale": scale.get("raw_scale"),
            "leaf_score": float(leaf),
            "trophies": mapped["trophies"],
            "trophy_source": scale.get("source"),
            "clamped": bool(mapped["clamped"]),
            "clamp_side": mapped["clamp_side"],
            "completed": bool(request.completed),
            "meaning_key": meaning_key,
            "meaning": meaning,
        }
        if request.completion is not None:
            row["completion"] = request.completion
        if request.action is not None:
            row["action"] = request.action
            row["action_index"] = request.action_index
        return row


def state_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The engine's own legal-action list for a board, for level C.

    Thin on purpose: the endpoint must enumerate exactly what the human can
    press, and that is `api.legal_actions`, the same list the page renders from.
    """
    return list(legal_actions(state))
