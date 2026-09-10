"""Public API surface for Step 1 foundation."""

from __future__ import annotations

import copy
import json
from collections import Counter
from typing import Any

from .constants import expected_shop_counts
from .engine import apply_action, legal_actions as engine_legal_actions
from .oracles.sap_calc_battle_oracle import run_battle_oracle
from .schema import validate_action_schema, validate_state_schema, validate_transition_schema


# exp13 W0b' (the schema-validation perf lever). MEASURED on the arm C
# profile shape (an internal dataset path, width 72, vgame leaf):
# `validate_state_schema` + `validate_transition_schema` are 70.6% of profiled
# wall time and 98.9% of `step` itself (559us + 123us + 1046us of a 1746us
# call; the engine plus the two deepcopies are the remaining 154us). The
# validators are ALREADY compiled once and reused (`schema._validators` is
# `lru_cache`d -- verified, one miss for the whole process), so there is no
# recompilation left to cache away; the cost is the jsonschema walk itself,
# and the only structural saving left inside `step` (drop the redundant
# re-walk of `state_before`/`action`, which this function has just validated)
# measured 1.36x, short of the 2x bar W0b' set for keeping validation on
# everywhere.
#
# So: an OPT-IN, process-level skip for IMAGINED walks only. Off by default,
# so no run that does not ask for it moves. "Imagined" means here what it
# means in `tools/honest_frame.py`: an engine walk whose boards are only ever
# LOOKED AT (the BC decode that proposes a chain, search candidate replay,
# prefix walks, the k resampled completions). The COMMITTED path --
# `eval_versus_fullgame.py`'s own op-by-op replay against the real state --
# calls `step` directly and always validates, so everything that actually
# enters a game's history is still schema-checked.
#
# exp13 W1b readout 0 (2026-08-07) re-measured this on the real frame and the
# 70.6% above did not survive: it is 24.1% of play wall clock. The split per
# validator then said something the merged number hid. 23.6 of those 24.1
# points are `validate_state_schema` reached through `legal_actions`, and 0.4
# is every validator on the real committed path put together. The flag could
# not reach the first, because `imagined_step` was the only imagined FACE: an
# imagined walk's `step` skipped validation while the `legal_actions` call
# that chose the action for that same walk did not.
#
# That is a hole in this flag's stated scope, not a reason to widen the scope.
# The paragraph above already names "the BC decode that proposes a chain" as
# imagined, and `bc_recommender.legal_mask` IS that decode reading the rules.
# So the repair is a second imagined face, `imagined_legal_actions`, and the
# boundary is unchanged and still structural:
#
#   - `step` (validating) is what the committed op-by-op replay calls, so
#     every board that enters a game's history is schema-checked as its
#     `state_before` and its transition is checked on the way out.
#   - `legal_actions` (validating) is what the exp13 segment recorder calls
#     for its `legal_actions_digest`, so the real decision boards it pins are
#     schema-checked as well.
#   - `imagined_step` and `imagined_legal_actions` are the two opt-in faces,
#     and neither is reachable from either of the two above.
#
# What the skip removes is therefore a RE-walk: the same real board `step` is
# about to validate, validated once more by a proposer that is only reading
# it. Each of those four statements is pinned as a test in
# `python/tests/test_exp13_imagined_validation.py` rather than left standing
# as this comment.
_SKIP_IMAGINED_VALIDATION = False


def set_skip_imagined_validation(enabled: bool) -> bool:
    """Turn the imagined-walk validation skip on/off; returns the PREVIOUS
    value so a caller (a test, a tool) can restore it in a `finally`."""
    global _SKIP_IMAGINED_VALIDATION
    previous = _SKIP_IMAGINED_VALIDATION
    _SKIP_IMAGINED_VALIDATION = bool(enabled)
    return previous


def skip_imagined_validation() -> bool:
    return _SKIP_IMAGINED_VALIDATION


def validate_state(state: dict[str, Any]) -> None:
    validate_state_schema(state)


def legal_actions(state: dict[str, Any], *, validate: bool = True) -> list[dict[str, Any]]:
    """The engine-true legal action set for `state`, schema-validated in.

    `validate=False` skips the schema check and NOTHING else -- the engine call
    and the returned list are identical either way. Callers should not pass it
    directly; `imagined_legal_actions` is the supported face, so that WHICH
    walks may skip stays one decision in one place, exactly as it is for `step`.
    """
    if validate:
        validate_state_schema(state)
    return engine_legal_actions(state)


def step(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    validate: bool = True,
    copy_state_before: bool = True,
) -> dict[str, Any]:
    """One engine transition, schema-validated in and out.

    `validate=False` skips the three schema checks and NOTHING else -- the
    engine call, the deepcopies and the returned transition are identical
    either way. Callers should not pass it directly; `imagined_step` is the
    supported face, so that WHICH walks may skip stays one decision in one
    place.

    `copy_state_before=False` puts the CALLER'S OWN state object into the
    transition as `state_before` instead of a deep copy of it. The CONTENT is
    identical either way, and that is not a new assumption: this copy is taken
    AFTER `apply_action` has already run, so an engine that mutated the state
    it was handed would have been corrupting today's `state_before` all along
    (`engine.apply_action` in fact deep-copies before it touches anything).
    What the alias costs is that the transition and the caller now share one
    object, so a consumer that WRITES to `state_before` writes to the caller's
    board. Same rule as `validate`: callers should not pass this directly;
    `imagined_step` is the supported face, so which walks share stays one
    decision in one place. On the labelling path this copy is pure waste:
    that path commits nothing, so its step census is 899,373 imagined steps
    out of 899,373, and no imagined consumer reads `state_before` at all.
    Priced at 60.10 s, 5.53% of play wall clock by `RESULTS_speedup.md`'s
    2026-08-08 section 2.2, and measured at 6.43% by the section that
    follows it.
    """
    if validate:
        validate_state_schema(state)
        validate_action_schema(action)
    outcome = apply_action(state, action)
    transition = {
        "state_before": copy.deepcopy(state) if copy_state_before else state,
        "action": copy.deepcopy(action),
        "state_after": outcome.state_after,
        "deterministic": outcome.deterministic,
        "stochastic_reason": outcome.stochastic_reason,
        # exp13 Amendments A2.1 and A4: the honest frame's cut criterion.
        # See `engine.StepOutcome`.
        "stochastic_structural": outcome.stochastic_structural,
        "legal": outcome.legal,
        "engine_notes": outcome.notes,
    }
    if validate:
        validate_transition_schema(transition)
    return transition


def imagined_step(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """`step` for a walk whose boards are only ever looked at, never committed.

    Identical to `step` unless `set_skip_imagined_validation(True)` was called
    (default: it was not), in which case the schema checks are skipped. See
    `_SKIP_IMAGINED_VALIDATION` above for the measurement that motivates it
    and for what the committed path still guarantees.

    It also does not deep-copy `state_before`: on a walk that is only ever
    looked at, that copy is a board nobody reads, and it was 5.53% of
    labelling play wall clock. Unlike the schema skip this is NOT behind the
    switch, because it is not a fidelity/cost trade -- the content of
    `state_before` is the same either way. It is a property of what this face
    MEANS (these boards are read and thrown away), so it holds whenever this
    face is taken. `step`, the committed face, still copies.
    """
    return step(
        state,
        action,
        validate=not _SKIP_IMAGINED_VALIDATION,
        copy_state_before=False,
    )


def imagined_legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    """`legal_actions` for a walk whose boards are only ever looked at.

    The read-only twin of `imagined_step`, and the face `bc_recommender`'s
    legal mask takes. Identical to `legal_actions` unless
    `set_skip_imagined_validation(True)` was called (default: it was not).

    Skipping here does not reduce what is guaranteed about the real game. A
    real board reaches this function only as the state a proposal is reading,
    and the same board is validated by `step` when an op is actually applied
    to it, and by `legal_actions` when the exp13 recorder digests it. See the
    boundary paragraph next to `_SKIP_IMAGINED_VALIDATION`.
    """
    return legal_actions(state, validate=not _SKIP_IMAGINED_VALIDATION)


def _compare_state(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    scalar_keys = ["turn", "gold", "lives", "trophies"]
    for key in scalar_keys:
        if lhs.get(key) != rhs.get(key):
            mismatches.append({"tag": "state_field_mismatch", "path": key, "lhs": lhs.get(key), "rhs": rhs.get(key)})

    for i in range(5):
        lslot = lhs["team"][i]
        rslot = rhs["team"][i]
        for key in ["pet_id", "attack", "health", "level", "exp", "equipment_id", "status_effects"]:
            if lslot.get(key) != rslot.get(key):
                mismatches.append(
                    {
                        "tag": "state_field_mismatch",
                        "path": f"team[{i}].{key}",
                        "lhs": lslot.get(key),
                        "rhs": rslot.get(key),
                    }
                )

    lshop = {(s["shop_index"], s["slot_type"]): s for s in lhs.get("shop", [])}
    rshop = {(s["shop_index"], s["slot_type"]): s for s in rhs.get("shop", [])}
    for key in sorted(set(lshop) | set(rshop)):
        if key not in lshop or key not in rshop:
            mismatches.append({"tag": "state_field_mismatch", "path": f"shop[{key}]", "lhs": lshop.get(key), "rhs": rshop.get(key)})
            continue
        for field in ["item_id", "cost", "frozen"]:
            if lshop[key].get(field) != rshop[key].get(field):
                mismatches.append(
                    {
                        "tag": "state_field_mismatch",
                        "path": f"shop[{key}].{field}",
                        "lhs": lshop[key].get(field),
                        "rhs": rshop[key].get(field),
                    }
                )
    return mismatches


def _action_set(actions: list[dict[str, Any]]) -> set[str]:
    return {json.dumps(a, sort_keys=True, separators=(",", ":")) for a in actions}


def _shop_slot_type_counts(shop: list[dict[str, Any]]) -> tuple[int, int]:
    pets = sum(1 for s in shop if s.get("slot_type") == "pet")
    foods = sum(1 for s in shop if s.get("slot_type") == "food")
    return pets, foods


def _shop_indices_contiguous(shop: list[dict[str, Any]]) -> bool:
    indices = [int(s.get("shop_index", -1)) for s in shop]
    return len(indices) == len(set(indices)) and sorted(indices) == list(range(len(shop)))


def _frozen_signature_counts(shop: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for slot in shop:
        if slot.get("frozen"):
            counter[(str(slot.get("slot_type")), str(slot.get("item_id")))] += 1
    return counter


def _frozen_signature_list(counter: Counter[tuple[str, str]]) -> list[dict[str, Any]]:
    out = []
    for (slot_type, item_id), count in sorted(counter.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.append({"slot_type": slot_type, "item_id": item_id, "count": int(count)})
    return out


def _compare_roll_invariants(
    state_before: dict[str, Any],
    engine_after: dict[str, Any],
    oracle_after: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []

    def add(path: str, lhs: Any, rhs: Any | None = None, expected: Any | None = None) -> None:
        mismatch: dict[str, Any] = {
            "tag": "roll_invariant_mismatch",
            "path": path,
            "lhs": lhs,
        }
        if rhs is not None:
            mismatch["rhs"] = rhs
        if expected is not None:
            mismatch["expected"] = expected
        mismatches.append(mismatch)

    before_gold = int(state_before.get("gold", 0))
    engine_gold_delta = int(engine_after.get("gold", 0)) - before_gold
    oracle_gold_delta = int(oracle_after.get("gold", 0)) - before_gold

    if engine_gold_delta != -1:
        add("roll.gold_delta", engine_gold_delta, expected=-1)
    if engine_gold_delta != oracle_gold_delta:
        add("roll.gold_delta", engine_gold_delta, rhs=oracle_gold_delta)

    for scalar in ["turn", "lives", "trophies"]:
        if engine_after.get(scalar) != state_before.get(scalar):
            add(f"roll.{scalar}_unchanged", engine_after.get(scalar), expected=state_before.get(scalar))
        if engine_after.get(scalar) != oracle_after.get(scalar):
            add(f"roll.{scalar}_parity", engine_after.get(scalar), rhs=oracle_after.get(scalar))

    expected_pets, expected_foods = expected_shop_counts(int(state_before.get("turn", 1)))
    expected_total = expected_pets + expected_foods

    engine_shop = engine_after.get("shop", [])
    oracle_shop = oracle_after.get("shop", [])
    ep, ef = _shop_slot_type_counts(engine_shop)
    op, of = _shop_slot_type_counts(oracle_shop)

    if len(engine_shop) != expected_total:
        add("roll.shop.count", len(engine_shop), expected=expected_total)
    if len(engine_shop) != len(oracle_shop):
        add("roll.shop.count", len(engine_shop), rhs=len(oracle_shop))

    if ep != expected_pets:
        add("roll.shop.pet_count", ep, expected=expected_pets)
    if ef != expected_foods:
        add("roll.shop.food_count", ef, expected=expected_foods)

    if ep != op:
        add("roll.shop.pet_count", ep, rhs=op)
    if ef != of:
        add("roll.shop.food_count", ef, rhs=of)

    engine_contiguous = _shop_indices_contiguous(engine_shop)
    oracle_contiguous = _shop_indices_contiguous(oracle_shop)
    if not engine_contiguous:
        add("roll.shop.index_contiguity", engine_contiguous, expected=True)
    if engine_contiguous != oracle_contiguous:
        add("roll.shop.index_contiguity", engine_contiguous, rhs=oracle_contiguous)

    frozen_before = _frozen_signature_counts(state_before.get("shop", []))
    frozen_engine = _frozen_signature_counts(engine_shop)
    frozen_oracle = _frozen_signature_counts(oracle_shop)

    for signature, needed in frozen_before.items():
        if frozen_engine[signature] < needed:
            add(
                "roll.shop.frozen_preservation",
                {
                    "slot_type": signature[0],
                    "item_id": signature[1],
                    "needed": int(needed),
                    "found": int(frozen_engine[signature]),
                },
            )

    if frozen_engine != frozen_oracle:
        add(
            "roll.shop.frozen_preservation",
            _frozen_signature_list(frozen_engine),
            rhs=_frozen_signature_list(frozen_oracle),
        )

    for slot in engine_shop:
        cost = int(slot.get("cost", 0))
        if cost < 0 or cost > 3:
            add("roll.shop.cost_bounds", {"shop_index": slot.get("shop_index"), "cost": cost}, expected="0..3")
        if not slot.get("frozen") and cost != 3:
            add("roll.shop.cost_nonfrozen", {"shop_index": slot.get("shop_index"), "cost": cost}, expected=3)

    return mismatches


def run_chain(state: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    validate_state_schema(state)
    if not isinstance(actions, list) or not actions:
        return {
            "ok": False,
            "error": "actions_must_be_nonempty",
            "transitions": [],
            "final_state": copy.deepcopy(state),
        }

    for i, action in enumerate(actions):
        validate_action_schema(action)
        if i < len(actions) - 1 and action["type"] in {"ROLL", "END_TURN"}:
            return {
                "ok": False,
                "error": f"chain_constraint_violation:terminal_action_before_end@index={i}",
                "transitions": [],
                "final_state": copy.deepcopy(state),
            }
    if actions[-1]["type"] not in {"ROLL", "END_TURN"}:
        return {
            "ok": False,
            "error": "chain_constraint_violation:last_action_must_be_roll_or_end_turn",
            "transitions": [],
            "final_state": copy.deepcopy(state),
        }

    curr = copy.deepcopy(state)
    transitions = []
    for action in actions:
        trans = step(curr, action)
        transitions.append(trans)
        curr = copy.deepcopy(trans["state_after"])
        if not trans["legal"]:
            return {
                "ok": False,
                "error": "illegal_action_encountered",
                "transitions": transitions,
                "final_state": curr,
            }

    return {"ok": True, "error": None, "transitions": transitions, "final_state": curr}


def oracle_compare(case: dict[str, Any]) -> dict[str, Any]:
    state = case["initial_state"]
    actions = case["actions"]

    engine = run_chain(state, actions)
    result: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "engine": engine,
        "shop_oracle": None,
        "battle_oracle": None,
        "legal_action_set": None,
        "mismatches": [],
    }

    oracle_info = state.get("meta", {}).get("oracle", {})
    player_state = case.get("oracle", {}).get("player_state") or oracle_info.get("player_state")
    if player_state is not None:
        engine_legal = engine_legal_actions(state)
        # Call-time import: the sapai differential is a diagnostic, and sapai
        # is not shipped with the public release.
        from .oracles.sapai_shop_oracle import legal_actions_from_player_state

        oracle_legal = legal_actions_from_player_state(player_state)
        only_engine = sorted(_action_set(engine_legal) - _action_set(oracle_legal))
        only_oracle = sorted(_action_set(oracle_legal) - _action_set(engine_legal))
        result["legal_action_set"] = {
            "engine_count": len(engine_legal),
            "oracle_count": len(oracle_legal),
            "match": not only_engine and not only_oracle,
            "only_engine": only_engine,
            "only_oracle": only_oracle,
        }
        if only_engine or only_oracle:
            result["mismatches"].append(
                {
                    "tag": "legal_action_set_mismatch",
                    "only_engine_count": len(only_engine),
                    "only_oracle_count": len(only_oracle),
                }
            )

        from .oracles.sapai_shop_oracle import run_chain as run_sapai_chain

        shop_oracle = run_sapai_chain(player_state, actions)
        result["shop_oracle"] = shop_oracle

        if engine["transitions"] and shop_oracle["transitions"]:
            min_len = min(len(engine["transitions"]), len(shop_oracle["transitions"]))
            for i in range(min_len):
                e = engine["transitions"][i]
                s = shop_oracle["transitions"][i]

                if e["legal"] != s["legal"]:
                    result["mismatches"].append(
                        {
                            "tag": "action_legality_mismatch",
                            "step": i,
                            "engine_legal": e["legal"],
                            "oracle_legal": s["legal"],
                            "oracle_error": s.get("oracle_error"),
                        }
                    )
                if e["action"]["type"] == "ROLL":
                    roll_mismatches = _compare_roll_invariants(
                        state_before=e["state_before"],
                        engine_after=e["state_after"],
                        oracle_after=s["state_after"],
                    )
                    result["mismatches"].extend([{**m, "step": i} for m in roll_mismatches])
                else:
                    result["mismatches"].extend(
                        [{**m, "step": i} for m in _compare_state(e["state_after"], s["state_after"])]
                    )

    if actions and actions[-1]["type"] == "END_TURN" and engine["transitions"]:
        end_state = engine["transitions"][-1]["state_after"]
        opponent_team = case.get("oracle", {}).get("battle_opponent_team")
        sim_count = int(case.get("oracle", {}).get("battle_simulation_count", 64))
        result["battle_oracle"] = run_battle_oracle(end_state, opponent_team=opponent_team, simulation_count=sim_count)

    tags = [m["tag"] for m in result["mismatches"]]
    result["ok"] = len(result["mismatches"]) == 0 and engine["ok"]
    result["mismatch_tags"] = sorted(set(tags))
    return result
