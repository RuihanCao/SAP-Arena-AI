"""A2.1 defines a turn under the honest frame as a loop: search from the real
board, commit the chosen chain op by op against that board, stop at the first
op whose engine transition reports `stochastic_structural=True`, observe the
real outcome, then re-search from the new state on a fresh imagination stream.
`stochastic_reason` remains the human-readable diagnostic, but non-structural
randomness no longer creates a boundary. The cut remains engine-owned and
never uses an action-type whitelist.

Everything else -- the imagined clone per segment, the hook calls, the
boundary read, the cap, the per-segment record -- is the driver's own code,
moved."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from ..api import step as engine_step
from .cat_trigger_audit import CatTriggerAudit, empty_counts
from .honest_frame import (
    MAX_SEGMENTS_PER_TURN,
    PROPOSAL_SAMPLE_R,
    imagination_seed,
    imagined_clone,
)


@dataclass
class SegmentedTurn:
    """Everything `play_out_game`'s turn body read off the loop's locals.

    Field for field the same names the loop used, so the call site reads the
    way it did when the loop was inline.
    """

    board: dict[str, Any]
    rec: dict[str, Any] = field(default_factory=dict)
    chain_preview: list[dict[str, Any]] = field(default_factory=list)
    committed_ops: list[dict[str, Any]] = field(default_factory=list)
    chain_types: list[str] = field(default_factory=list)
    stop_reason: str = "unknown"
    replay_diverged_at: str | None = None
    decode_failed: bool = False
    segments: list[dict[str, Any]] | None = None
    segments_capped: bool = False


    cat_trigger_audit: dict[str, int] = field(default_factory=empty_counts)

    @property
    def n_segments(self) -> int:
        return len(self.segments) if self.segments is not None else 1


def _shop_item_id(board: dict[str, Any], shop_index: Any) -> str | None:
    """The item id in a shop slot, off the REAL board an op was committed to."""
    try:
        wanted = int(shop_index)
    except (TypeError, ValueError):
        return None
    for slot in board.get("shop") or []:
        if not isinstance(slot, dict):
            continue
        try:
            if int(slot.get("shop_index", -1)) != wanted:
                continue
        except (TypeError, ValueError):
            continue
        item_id = slot.get("item_id")
        return None if item_id is None else str(item_id)
    return None


def recorded_op(board: dict[str, Any], op: dict[str, Any]) -> dict[str, Any]:
    """A committed op as it goes into the RECORD, never as it goes to the engine."""
    recorded = copy.deepcopy(op)
    if str(op.get("type", "")).strip().upper() == "BUY_FOOD":
        item_id = _shop_item_id(board, op.get("shop_index"))
        if item_id is not None:
            recorded["food_item_id"] = item_id
    return recorded


def bc_decide(bc: Any) -> Callable[[dict[str, Any], int], dict[str, Any]]:
    """This loop's DEFAULT per-segment decision, as a value."""

    def _decide(decision_state: dict[str, Any], _segment_index: int) -> dict[str, Any]:
        return bc.recommend(decision_state)

    return _decide


def run_segmented_turn(
    state: dict[str, Any],
    *,
    bc: Any,
    honest: bool,
    game_engine_seed: int,
    race_wins: int | None,
    set_race_context: Callable[..., None] | None = None,
    set_imagination_context: Callable[..., None] | None = None,
    decide: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    step_records: list[dict[str, Any]] | None = None,
    stop_reason_sink: Callable[[str], None] | None = None,
    on_committed_op: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
    commit_step: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = engine_step,
    max_segments: int = MAX_SEGMENTS_PER_TURN,
    start_segment_index: int = 0,
) -> SegmentedTurn:
    """One whole turn's shop phase: the A1 segment loop, run to END_TURN.

    `honest=False` runs the body EXACTLY ONCE and is line-for-line the
    pre-A1 turn body (decode -> replay every non-END_TURN op -> fall
    through), which is why the determinized frame stays byte-identical.

    `state` is the REAL board on play's stream P and is never mutated: the
    loop reassigns `board` to each transition's `state_after`, and the only
    thing the recommender is ever handed under the honest frame is an
    imagined clone on stream S."""
    decide_fn = decide if decide is not None else bc_decide(bc)


    audit = CatTriggerAudit()

    out = SegmentedTurn(board=state, segments=[] if honest else None)
    segments = out.segments
    board = state
    rec: dict[str, Any] = {}
    chain_preview: list[dict[str, Any]] = []
    committed_ops: list[dict[str, Any]] = []
    chain_types: list[str] = []
    stop_reason = "unknown"
    replay_diverged_at: str | None = None
    decode_failed = False
    segments_capped = False
    segment_index = int(start_segment_index)

    while True:
        if honest:
            # A1 ruling 1: the recommender NEVER sees the play stream's
            # seed. Everything it walks through the engine -- its own
            # decode, search's candidate sampling, its prefix replay and
            # its resample completions -- inherits stream S from this
            # clone. The real `board` is untouched and stays on P.
            segment_seed: int | None = imagination_seed(
                engine_seed=game_engine_seed,
                turn=int(board.get("turn", 0) or 0),
                segment_index=segment_index,
                sample_r=PROPOSAL_SAMPLE_R,
            )
            decision_state = imagined_clone(board, seed=int(segment_seed))
            if set_imagination_context is not None:
                set_imagination_context(
                    engine_seed=game_engine_seed, segment_index=segment_index
                )
        else:
            segment_seed = None
            decision_state = board


        if set_race_context is not None:
            set_race_context(wins=race_wins)

        rec = decide_fn(decision_state, segment_index)
        stop_reason = str(rec.get("diagnostics", {}).get("stop_reason") or "unknown")
        if stop_reason_sink is not None:
            stop_reason_sink(stop_reason)

        if not rec.get("ok"):
            decode_failed = True
            break

        chain_preview = rec.get("chain_preview") or []
        segment_chain_types = [
            str(op.get("type", "")).strip().upper() for op in chain_preview
        ]
        ops = [
            op
            for op in chain_preview
            if str(op.get("type", "")).strip().upper() != "END_TURN"
        ]

        committed_types: list[str] = []
        committed_ops_recorded: list[dict[str, Any]] = []
        boundary_reason: str | None = None
        for op in ops:
            step_before = copy.deepcopy(board) if step_records is not None else None
            try:
                tr = commit_step(board, op)
            except Exception as exc:  # defensive only, see module docstring
                replay_diverged_at = f"{op.get('type')}:{type(exc).__name__}"
                break
            if not tr.get("legal"):
                replay_diverged_at = f"{op.get('type')}:illegal_on_replay"
                break
            next_board = tr.get("state_after")
            if not isinstance(next_board, dict):
                replay_diverged_at = f"{op.get('type')}:no_state_after"
                break
            if step_records is not None:
                step_records.append({"state_before": step_before, "action": copy.deepcopy(op)})
            # Both of these read `board` while it is still the PRE-op board,
            # which is the state whose shop names the food and whose counters
            # the old Cat rule would have read.
            committed_ops_recorded.append(recorded_op(board, op))
            audit.on_committed_op(board, op)
            board = next_board
            committed_ops.append(op)
            committed_types.append(str(op.get("type", "")).strip().upper())
            if on_committed_op is not None:
                on_committed_op(segment_index, op, tr)
            # A2.1: cut only when randomness changes the future legal-action
            # read-set. The engine owns that structural classification; the
            # reason remains diagnostic and must never become a second cut
            # predicate here.
            if honest and tr.get("stochastic_structural"):
                boundary_reason = str(tr.get("stochastic_reason") or "structural")
                break

        if honest:
            chain_types.extend(committed_types)
            if (
                replay_diverged_at is None
                and boundary_reason is None
                and segment_chain_types[-1:] == ["END_TURN"]
            ):
                chain_types.append("END_TURN")
            segments.append(
                {
                    "index": segment_index,
                    "stop_reason": stop_reason,
                    # A1's audit surface: what this segment PROPOSED
                    # (the full chain, including anything past the
                    # boundary that was decoded against an imagined shop
                    # and therefore never committed)...
                    "chain_preview": copy.deepcopy(chain_preview),
                    "chain_types": segment_chain_types,
                    # ...next to what actually reached the real board.
                    # The two differ exactly when a boundary fired.
                    "committed_types": committed_types,


                    "committed_ops": committed_ops_recorded,
                    "n_committed": len(committed_types),
                    "boundary_reason": boundary_reason,
                    "replay_diverged_at": replay_diverged_at,
                    "imagination_seed": segment_seed,
                    "search_used": bool(rec.get("search_used", False)),
                    "search_n_generated": rec.get("search_n_generated"),
                    "search_n_dedup": rec.get("search_n_dedup"),
                    "search_chosen_index": rec.get("search_chosen_index"),
                    "search_error": rec.get("search_error"),


                    "search_completion_policy": rec.get("search_completion_policy"),
                    "search_completion_width": rec.get("search_completion_width"),
                    "search_completion_aggregate": rec.get("search_completion_aggregate"),
                    "search_completion_decided": rec.get("search_completion_decided"),
                    "search_completion_divergent": rec.get("search_completion_divergent"),
                    "search_completion_samples": rec.get("search_completion_samples"),
                    "search_completion_boards": rec.get("search_completion_boards"),
                    "search_completion_boards_mean": rec.get(
                        "search_completion_boards_mean"
                    ),
                    "search_completion_terminal_boards": rec.get(
                        "search_completion_terminal_boards"
                    ),
                    "search_completion_unfinished_boards": rec.get(
                        "search_completion_unfinished_boards"
                    ),
                    "search_completion_terminal_rate": rec.get(
                        "search_completion_terminal_rate"
                    ),
                    "search_completion_dropped": rec.get(
                        "search_completion_dropped"
                    ),
                    "search_completion_second_chance_nodes": rec.get(
                        "search_completion_second_chance_nodes"
                    ),
                    "search_completion_rescued": rec.get(
                        "search_completion_rescued"
                    ),
                }
            )
        else:
            chain_types = segment_chain_types

        if replay_diverged_at is not None or not honest or boundary_reason is None:
            break
        segment_index += 1
        if segment_index >= max_segments:


            segments_capped = True
            break

    out.board = board
    out.rec = rec
    out.chain_preview = chain_preview
    out.committed_ops = committed_ops
    out.chain_types = chain_types
    out.stop_reason = stop_reason
    out.replay_diverged_at = replay_diverged_at
    out.decode_failed = decode_failed
    out.cat_trigger_audit = audit.counts
    out.segments_capped = segments_capped
    return out
