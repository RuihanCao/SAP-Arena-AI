"""Per-segment exp13 W1 records with play-independent label selection.

`SegmentRecorder` is the collection side: it hangs off `SegmentedTurn`'s
`decide` / `on_committed_op` hooks and emits one raw row per real segment.
`build_segment_record` is the schema side: it turns one raw row into the
persisted record, adding the label-group selection. They are kept apart so
the row a run emits can be rebuilt from a different collector -- and so the
schema tests do not need a game to run.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import TYPE_CHECKING, Any, Callable

from sap_ppo.api import legal_actions

if TYPE_CHECKING:  # `segmented_turn` never imports this module, so no cycle
    from sap_ppo.tools.segmented_turn import SegmentedTurn


SCHEMA_VERSION = "exp13_w1_segment_v1"

# WHY A ROW CAN BE IN THE DATASET AND OUT OF THE LABEL POOL.
#
# `PLAN_W1.md` §Pinned frame scores an agent terminal failure (`decode_failed:*`
# and friends) as 0 trophies and non-completion, kept in the intention-to-treat
# denominator, never dropped. It says nothing about what that game's RECORDED
# rows are worth, and the two questions are genuinely separate: the game is a
# real sample of the policy's outcome distribution and a broken sample of its
# decision distribution, because its last decision is the one the driver could
# not carry out.
#
# Ruihan's ruling (2026-08-08): exclude the WHOLE failed game from the label
# pool, not just the offending row. One game in 180 is a negligible cost, and
# salvaging a prefix would mean arguing which of its rows are still valid.
#
# The rows stay in `records.jsonl.gz` -- they are provenance, the coverage gate
# still requires every nominal index to appear, and the census still counts
# them -- and carry this key instead. Exactly like the turn-1 rows whose
# `candidate_groups == []`: excluded by an explicit, counted rule, so
# "excluded on purpose" and "the recorder lost rows" can never look alike.
LABEL_POOL_EXCLUDED_KEY = "label_pool_excluded"


def mark_label_pool_excluded(record: dict[str, Any], reason: str) -> dict[str, Any]:
    """Stamp `record` as out of the label pool, in place, and return it."""
    text = " ".join(str(reason or "").split())
    if not text:
        raise ValueError("label_pool_exclusion_requires_a_reason")
    record[LABEL_POOL_EXCLUDED_KEY] = text
    return record


def is_label_pool_excluded(record: dict[str, Any]) -> bool:
    """True iff this row was stamped out of the label pool.

    ONE definition, read by the merge's census, by the label runner's own
    skip rule and by the tests, so a row can never be counted as excluded in
    one place and labelled in another.
    """
    return bool(record.get(LABEL_POOL_EXCLUDED_KEY))


def in_label_pool(record: dict[str, Any]) -> bool:
    """True iff this row is labellable: it has candidate groups AND was not
    excluded. The label pool's own definition, and the denominator every
    per-decision cost in `w1_readouts` divides by."""
    return bool(record.get("candidate_groups")) and not is_label_pool_excluded(record)


def stable_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_seed(
    engine_seed: int, game_id: int, turn: int, segment_id: int
) -> int:
    identity = [
        int(engine_seed),
        int(game_id),
        int(turn),
        int(segment_id),
        "label_selection",
    ]
    return int(stable_digest(identity)[:16], 16)


def select_label_groups(
    groups: list[dict[str, Any]],
    *,
    engine_seed: int,
    game_id: int,
    turn: int,
    segment_id: int,
) -> dict[str, Any]:
    """Select nested S12/S16 groups without touching any live RNG stream."""
    by_rank = sorted(
        range(len(groups)),
        key=lambda i: (int(groups[i]["v0_rank"]), int(groups[i]["group_index"])),
    )
    top = by_rank[:8]
    remaining = by_rank[8:]
    seed = _selection_seed(engine_seed, game_id, turn, segment_id)
    permutation = list(remaining)
    random.Random(seed).shuffle(permutation)
    s12 = top + permutation[:4]
    s16 = top + permutation[:8]
    if not set(s12).issubset(s16):
        raise AssertionError("S12 must be an exact subset of S16")
    return {
        "rng_key": [
            int(engine_seed),
            int(game_id),
            int(turn),
            int(segment_id),
            "label_selection",
        ],
        "rng_seed": seed,
        "top_group_indices": top,
        "exploration_permutation": permutation,
        "s12_group_indices": s12,
        "s16_group_indices": s16,
        "s12_exploration_shortfall": max(0, 4 - len(permutation)),
        "s16_exploration_shortfall": max(0, 8 - len(permutation)),
    }


def build_segment_record(payload: dict[str, Any], *, game_id: int) -> dict[str, Any]:
    decision_start = payload["decision_start_real"]
    rec = payload.get("recommendation") or {}
    groups = copy.deepcopy(rec.get("search_candidate_groups") or [])
    turn = int(payload["turn"])
    segment_id = int(payload["segment_id"])
    engine_seed = int(payload["engine_seed"])
    selection = select_label_groups(
        groups,
        engine_seed=engine_seed,
        game_id=int(game_id),
        turn=turn,
        segment_id=segment_id,
    )
    s12 = set(selection["s12_group_indices"])
    s16 = set(selection["s16_group_indices"])
    top = set(selection["top_group_indices"])
    for group in groups:
        group_index = int(group["group_index"])
        sources: list[str] = []
        if group_index in top:
            sources.append("v0_top8")
        if group_index in set(selection["exploration_permutation"][:8]):
            sources.append("exploration_permutation")
        group["selection_source"] = sources
        group["selected_s12"] = group_index in s12
        group["selected_s16"] = group_index in s16

    chosen_index = rec.get("search_chosen_index")
    chosen_key = None
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(groups):
        chosen_key = groups[chosen_index].get("dedup_prefix_key")
    action_list = legal_actions(decision_start)
    next_identity = payload.get("next_real_segment_identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "record_kind": "segment_decision",
        "game_id": int(game_id),
        "turn": turn,
        "segment_id": segment_id,
        "decision_seed_identity": {
            "engine_seed": engine_seed,
            "imagination_seed": payload.get("imagination_seed"),
        },
        "proposer_call_identity": stable_digest(
            [engine_seed, int(game_id), turn, segment_id, "proposer"]
        ),
        "decision_start_real": copy.deepcopy(decision_start),
        "decision_start_real_digest": stable_digest(decision_start),
        "decision_state_imagined": copy.deepcopy(payload["decision_state_imagined"]),
        "legal_actions_digest": stable_digest(action_list),
        "raw_candidate_chains": copy.deepcopy(
            (rec.get("search_diagnostics") or {}).get("decoded_candidate_chains") or []
        ),
        "candidate_groups": groups,
        "label_selection": selection,
        "chosen_prefix_group_index": chosen_index,
        "chosen_prefix_key": chosen_key,
        "committed_ops": copy.deepcopy(payload.get("committed_ops") or []),
        "committed_result_real": copy.deepcopy(payload["committed_result_real"]),
        "committed_result_real_digest": stable_digest(payload["committed_result_real"]),
        "stochastic_reason": payload.get("stochastic_reason"),
        "structural_cut": bool(payload.get("structural_cut")),
        "next_real_segment_identity": copy.deepcopy(next_identity),
        "search": {
            "used": bool(rec.get("search_used")),
            "n_generated": rec.get("search_n_generated"),
            "n_dedup": rec.get("search_n_dedup"),
            "chosen_index": chosen_index,
            "scoring": rec.get("search_scoring"),
            "turn_mode": rec.get("search_turn_mode"),
        },
    }


def validate_segment_record(record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version")
    if LABEL_POOL_EXCLUDED_KEY in record and not (
        isinstance(record[LABEL_POOL_EXCLUDED_KEY], str)
        and record[LABEL_POOL_EXCLUDED_KEY].strip()
    ):
        # Optional, but a present-and-empty stamp is a row that reads as
        # excluded to `is_label_pool_excluded`'s truthiness test in one
        # version of the check and as included in another.
        problems.append(LABEL_POOL_EXCLUDED_KEY)
    decision = record.get("decision_start_real")
    if stable_digest(decision) != record.get("decision_start_real_digest"):
        problems.append("decision_start_real_digest")
    if stable_digest(legal_actions(decision)) != record.get("legal_actions_digest"):
        problems.append("legal_actions_digest")
    committed = record.get("committed_result_real")
    if stable_digest(committed) != record.get("committed_result_real_digest"):
        problems.append("committed_result_real_digest")
    groups = record.get("candidate_groups") or []
    indices = [int(group.get("group_index", -1)) for group in groups]
    if indices != list(range(len(groups))):
        problems.append("candidate_group_indices")
    selection = record.get("label_selection") or {}
    s12 = set(selection.get("s12_group_indices") or [])
    s16 = set(selection.get("s16_group_indices") or [])
    if not s12.issubset(s16):
        problems.append("s12_not_subset_s16")
    if len(s12) != len(set(s12)) or len(s16) != len(set(s16)):
        problems.append("selection_duplicates")
    expected_selection = select_label_groups(
        groups,
        engine_seed=int(record["decision_seed_identity"]["engine_seed"]),
        game_id=int(record["game_id"]),
        turn=int(record["turn"]),
        segment_id=int(record["segment_id"]),
    )
    if selection != expected_selection:
        problems.append("label_selection_recompute")
    for group in groups:
        completions = group.get("completions") or []
        if int(group.get("completion_count", -1)) != len(completions):
            problems.append("completion_count")
            break
        if any(not isinstance(row.get("candidate_imagined_afterstate"), dict) for row in completions):
            problems.append("candidate_imagined_afterstate")
            break
    return problems


def action_identity_projection(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "game_id": int(record["game_id"]),
            "turn": int(record["turn"]),
            "segment_id": int(record["segment_id"]),
            "chosen_prefix_key": copy.deepcopy(record.get("chosen_prefix_key")),
            "committed_ops": copy.deepcopy(record.get("committed_ops") or []),
            "committed_result_real_digest": record.get("committed_result_real_digest"),
            "structural_cut": bool(record.get("structural_cut")),
            "stochastic_reason": record.get("stochastic_reason"),
        }
        for record in records
        if record.get("record_kind") == "segment_decision"
    ]


def compare_action_identity(
    control_records: list[dict[str, Any]], treatment_records: list[dict[str, Any]]
) -> list[str]:
    control = action_identity_projection(control_records)
    treatment = action_identity_projection(treatment_records)
    if len(control) != len(treatment):
        return [f"segment_count:{len(control)}!={len(treatment)}"]
    return [
        f"segment_identity:{index}"
        for index, (left, right) in enumerate(zip(control, treatment))
        if left != right
    ]


class SegmentRecorder:
    """The per-segment W1 rows, rebuilt from `SegmentedTurn`'s OWN hooks.

    These rows used to be emitted by a copy of the segment loop that lived
    inside `eval_versus_fullgame.play_out_game`. exp16 W3 moved that loop
    into `tools/segmented_turn.py` so the duel and the driver cannot drift,
    which turned a driver-local recorder into a hook on dead code: it raises
    nothing, it just never records. This class is the re-attachment, and it
    uses only the shared loop's two public hooks.

    - `decide` is called exactly once per segment, with the imagined clone
      the recommender is actually handed. That is what OPENS a row: it is
      the only place both the imagined decision state and the full
      recommendation (candidate groups included) are visible.
    - `on_committed_op` is called for every op that reached the REAL board,
      with the engine transition. `transition["state_after"]` is the object
      the loop assigns to `board`, so a segment's LAST one is both that
      segment's real result and the next segment's real start. Chaining
      them is what keeps `decision_start_real` and `committed_result_real`
      real -- the two fields A1 section 5 exists to separate from imagined
      completions -- without needing a third hook.

    Nothing here re-derives a decision the loop already made. The boundary
    reason and any replay divergence are read back off
    `SegmentedTurn.segments` in `end_turn`, so the recorder cannot disagree
    with the cut that actually happened; and the one thing it does read for
    itself, the imagined clone's stream seed, is asserted against the loop's
    own record of it.

    The rows are the raw payloads `build_segment_record` consumes; this
    class deliberately does no schema work, so the two stay separable.
    """

    def __init__(
        self,
        sink: Callable[[dict[str, Any]], None],
        *,
        engine_seed: int,
        honest: bool,
    ) -> None:
        self._sink = sink
        self._engine_seed = int(engine_seed)
        self._honest = bool(honest)
        self._real_board: dict[str, Any] | None = None
        self._pending: list[dict[str, Any]] = []

    def begin_turn(self, state: dict[str, Any]) -> None:
        """Seed the real-board chain with the turn's starting board.

        Segment 0's `decision_start_real` is this; every later segment's is
        the previous segment's last committed transition.
        """
        self._real_board = copy.deepcopy(state)
        self._pending = []

    def wrap_decide(
        self, decide: Callable[[dict[str, Any], int], dict[str, Any]]
    ) -> Callable[[dict[str, Any], int], dict[str, Any]]:
        """Wrap the loop's decision so each call opens a row.

        Pass `segmented_turn.bc_decide(bc)` -- the loop's own default -- so
        what is recorded is what an unrecorded run would have decided.
        """

        def _decide(decision_state: dict[str, Any], segment_index: int) -> dict[str, Any]:
            self._open_segment(int(segment_index), decision_state)
            rec = decide(decision_state, segment_index)
            self._pending[-1]["recommendation"] = copy.deepcopy(rec)
            return rec

        return _decide

    def on_committed_op(
        self, segment_index: int, op: dict[str, Any], transition: dict[str, Any]
    ) -> None:
        pending = self._pending[int(segment_index)]
        pending["committed_ops"].append(copy.deepcopy(op))
        state_after = transition.get("state_after")
        if isinstance(state_after, dict):
            # One copy per committed op, then shared by reference: the loop
            # never mutates a board it has moved past, and the row builder
            # copies again on its way into the schema.
            self._real_board = copy.deepcopy(state_after)
            pending["committed_result_real"] = self._real_board

    def end_turn(self, turn_out: "SegmentedTurn") -> None:
        """Emit this turn's rows, closed against what the loop reported.

        Called right after `run_segmented_turn` returns and BEFORE the
        driver's decode-failed / replay-diverged branches, both of which
        `break` the turn loop.
        """
        by_index = {
            int(segment["index"]): segment for segment in (turn_out.segments or [])
        }
        for pending in self._pending:
            segment_id = int(pending["segment_id"])
            segment = by_index.get(segment_id)
            # Absent exactly when the loop broke before appending, i.e. a
            # decode failure. Then there was no commit and no boundary.
            boundary_reason = segment.get("boundary_reason") if segment else None
            replay_diverged_at = segment.get("replay_diverged_at") if segment else None
            if segment is not None:
                loop_seed = segment.get("imagination_seed")
                if loop_seed != pending["imagination_seed"]:
                    raise AssertionError(
                        "segment recorder is watching a different imagination "
                        f"stream than the loop ran: {pending['imagination_seed']!r} "
                        f"!= {loop_seed!r} at segment {segment_id}"
                    )
            next_identity = None
            if replay_diverged_at is None and boundary_reason is not None:
                next_identity = {
                    "turn": int(pending["committed_result_real"].get("turn", 0) or 0),
                    "segment_id": segment_id + 1,
                }
            self._sink(
                {
                    "turn": int(pending["decision_start_real"].get("turn", 0) or 0),
                    "segment_id": segment_id,
                    "engine_seed": self._engine_seed,
                    "imagination_seed": pending["imagination_seed"],
                    "decision_start_real": pending["decision_start_real"],
                    "decision_state_imagined": pending["decision_state_imagined"],
                    "recommendation": pending["recommendation"],
                    "committed_ops": pending["committed_ops"],
                    "committed_result_real": pending["committed_result_real"],
                    "stochastic_reason": boundary_reason,
                    "structural_cut": boundary_reason is not None,
                    "next_real_segment_identity": next_identity,
                }
            )
        self._pending = []

    def _open_segment(self, segment_index: int, decision_state: dict[str, Any]) -> None:
        if self._real_board is None:
            raise RuntimeError("SegmentRecorder.begin_turn was not called for this turn")
        if segment_index != len(self._pending):
            raise RuntimeError(
                "segment recorder saw segments out of order: expected "
                f"{len(self._pending)}, got {segment_index}"
            )
        self._pending.append(
            {
                "segment_id": segment_index,
                "decision_start_real": self._real_board,
                "decision_state_imagined": copy.deepcopy(decision_state),
                "recommendation": {},
                "committed_ops": [],
                # Overwritten by every committed op; a segment that commits
                # nothing correctly ends where it started.
                "committed_result_real": self._real_board,
                "imagination_seed": _imagined_stream_seed(decision_state)
                if self._honest
                else None,
            }
        )


def _imagined_stream_seed(decision_state: dict[str, Any]) -> int | None:
    """The stream-S seed off an imagined clone (`honest_frame.imagined_clone`).

    Read from the clone rather than recomputed, so this cannot drift from
    the seed the recommender was actually run on. `end_turn` still checks it
    against the seed the loop recorded for that segment.
    """
    meta = decision_state.get("meta")
    if not isinstance(meta, dict):
        return None
    try:
        return int(meta["seed"])
    except (KeyError, TypeError, ValueError):
        return None
