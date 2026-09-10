"""Replay v2 normalization and completed-game public views.

The archive stores authoritative engine transitions. This module derives a
human-readable view without replaying actions or asking browser JavaScript to
infer state. Legacy v1 games remain readable but are marked explicitly when
transition-level detail was never recorded.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

from ...catalog import load_turtle_catalog, tier_for_turn
from .agent_identity import normalize_ai_version

REPLAY_SCHEMA_VERSION = 2
_TRANSITION_KEYS = (
    "state_before",
    "action",
    "state_after",
    "deterministic",
    "stochastic_reason",
    "stochastic_structural",
    "legal",
    "engine_notes",
)
SUPPORTED_REPLAY_OPS = frozenset(
    {
        "start_turn",
        "roll",
        "freeze",
        "unfreeze",
        "buy_pet",
        "buy_combine",
        "buy_food",
        "sell",
        "merge",
        "ability_stock",
        "reorder",
        "end_turn",
    }
)
EXP08_EVENT_FIELDS = {
    "start_turn": {"op", "shop", "shop_provenance"},
    "buy_pet": {
        "op", "name", "uni", "to_slot", "provenance", "identity_src",
        "gold_delta", "gold_after",
    },
    "roll": {
        "op", "gold_delta", "gold_after", "shop", "shop_provenance",
    },
    "freeze": {"op", "items"},
    "unfreeze": {"op", "items"},
    "end_turn": {"op", "frozen_carried"},
    "merge": {"op", "src", "dst", "gold_delta"},
    "buy_combine": {
        "op", "name", "uni", "onto", "provenance", "identity_src",
        "gold_delta", "gold_after",
    },
    "sell": {
        "op", "name", "uni", "from_slot", "level", "gold_delta",
        "gold_after",
    },
    "ability_stock": {
        "op", "source", "name", "count", "gold_delta",
    },
    "buy_food": {
        "op", "name", "uni", "target", "provenance", "free",
        "identity_src", "gold_delta", "gold_after",
    },
    "reorder": {"op", "board_before", "board_after"},
}


_IDENTITY_SOURCE = "engine-native:item_id+link_id+location"


def is_completed_game(game: dict[str, Any]) -> bool:
    """The single public replay completion predicate."""
    return bool(game.get("done")) and bool(str(game.get("end_reason") or "").strip())


def _name(item_id: Any) -> str:
    if item_id is None:
        return "empty"
    catalog = load_turtle_catalog()
    for section in ("pets", "foods"):
        found = ((catalog.get(section) or {}).get("id_to_name_id") or {}).get(
            str(item_id)
        )
        if found:
            return str(found)
    return str(item_id)


def _identity(
    item_id: Any,
    *,
    link_id: Any = None,
    location: str,
    index: int | None,
) -> dict[str, Any]:
    return {
        "uni": None,
        "canonical_item_id": item_id,
        "link_id": link_id,
        "location": str(location),
        "index": index,
        "identity_source": _IDENTITY_SOURCE,
    }


def _identity_fields(
    item_id: Any,
    *,
    link_id: Any = None,
    location: str,
    index: int | None,
) -> dict[str, Any]:
    identity = _identity(
        item_id,
        link_id=link_id,
        location=location,
        index=index,
    )
    return {
        "uni": None,
        "identity": identity,
        "identity_src": _IDENTITY_SOURCE,
        "identity_source": _IDENTITY_SOURCE,
    }


def board_slots(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return fixed board slots 0 through 4, including empty slots."""
    by_slot = {
        int(slot.get("slot_index", index)): slot
        for index, slot in enumerate((state or {}).get("team") or [])
        if isinstance(slot, dict)
    }
    out: list[dict[str, Any]] = []
    for index in range(5):
        slot = by_slot.get(index) or {}
        item_id = slot.get("pet_id")
        out.append(
            {
                "slot": index,
                "empty": item_id is None,
                "name": _name(item_id),
                "item_id": item_id,
                "level": int(slot.get("level", 1)),
                "exp": int(slot.get("exp", 0)),
                "attack": int(slot.get("attack", 0)),
                "health": int(slot.get("health", 0)),
                "equipment_id": slot.get("equipment_id"),
                "status_effects": copy.deepcopy(slot.get("status_effects") or []),
                **_identity_fields(
                    item_id,
                    location="team",
                    index=index,
                ),
            }
        )
    return out


def _notes(transition: dict[str, Any] | None) -> list[str]:
    raw = (transition or {}).get("engine_notes") or []
    if isinstance(raw, str):
        return [raw]
    return [str(note) for note in raw]


def _ability_specs(transition: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Parse authoritative ability-stock notes emitted by the engine."""
    specs: list[dict[str, Any]] = []
    for note in _notes(transition):
        if not note.startswith("ability_applied:"):
            continue
        source_match = re.match(r"ability_applied:(pet-[^:]+):", note)
        source_id = source_match.group(1) if source_match else "ability"
        count: int | None = None
        item_id: str | None = None

        cow = re.search(r"milk_count=(\d+):item=([^:\s]+)", note)
        pigeon = re.search(r"stocked=(\d+)/([^:\s]+)", note)
        worm = re.search(r"stocked=([^:\s]+):cost=", note)
        if cow:
            count, item_id = int(cow.group(1)), cow.group(2)
        elif pigeon:
            count, item_id = int(pigeon.group(1)), pigeon.group(2)
        elif worm:
            count, item_id = 1, worm.group(1)
        if count is None or item_id is None:
            continue
        specs.append(
            {
                "source_id": source_id,
                "source": _name(source_id),
                "item_id": item_id,
                "name": _name(item_id),
                "count": count,
            }
        )
    return specs


def _shop_key(slot: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(slot.get("shop_index", -1)),
        str(slot.get("item_id") or ""),
        str(slot.get("link_id") or ""),
    )


def _shop_lineage_key(slot: dict[str, Any]) -> tuple[str, str, str]:
    """Identity fields that survive shop compaction and stat mutations."""
    return (
        str(slot.get("slot_type") or ""),
        str(slot.get("item_id") or ""),
        str(slot.get("link_id") or ""),
    )


def _frozen_signature(slot: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(slot.get("slot_type") or ""),
        str(slot.get("item_id") or ""),
        str(slot.get("link_id") or ""),
    )


def _shop_item(
    slot: dict[str, Any],
    *,
    provenance: str,
    injected: bool,
) -> dict[str, Any]:
    index = int(slot.get("shop_index", -1))
    item_id = slot.get("item_id")
    return {
        "shop_index": index,
        "slot_type": str(slot.get("slot_type") or ""),
        "name": _name(item_id),
        "item_id": item_id,
        "cost": int(slot.get("cost", 0)),
        "attack": (
            int(slot.get("attack", 0))
            if str(slot.get("slot_type")) == "pet"
            else None
        ),
        "health": (
            int(slot.get("health", 0))
            if str(slot.get("slot_type")) == "pet"
            else None
        ),
        "frozen": bool(slot.get("frozen")),
        "link_id": slot.get("link_id"),
        "provenance": str(provenance),
        "injected": bool(injected),
        **_identity_fields(
            item_id,
            link_id=slot.get("link_id"),
            location="shop",
            index=index,
        ),
    }


def _initial_provenance(
    state_after: dict[str, Any],
    *,
    source: str,
    state_before: dict[str, Any] | None,
    transition: dict[str, Any] | None,
) -> dict[tuple[int, str, str], str]:
    frozen = Counter(
        _frozen_signature(slot)
        for slot in (state_before or {}).get("shop") or []
        if slot.get("frozen")
    )
    injected = Counter()
    for spec in _ability_specs(transition):
        injected[str(spec["item_id"])] += int(spec["count"])

    result: dict[tuple[int, str, str], str] = {}
    for slot in state_after.get("shop") or []:
        signature = _frozen_signature(slot)
        item_id = str(slot.get("item_id") or "")
        if slot.get("frozen") and frozen[signature] > 0:
            provenance = "frozen_carry"
            frozen[signature] -= 1
        elif injected[item_id] > 0:
            provenance = "ability_injected"
            injected[item_id] -= 1
        elif slot.get("link_id"):
            provenance = "levelup_reward"
        else:
            provenance = source
        result[_shop_key(slot)] = provenance
    return result


def _propagate_provenance(
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    current: dict[tuple[int, str, str], str],
    transition: dict[str, Any],
    *,
    default: str = "engine_shop",
) -> dict[tuple[int, str, str], str]:
    injected = Counter()
    for spec in _ability_specs(transition):
        injected[str(spec["item_id"])] += int(spec["count"])
    action = transition.get("action") or {}
    action_type = str(action.get("type") or "").strip().upper()
    consumed_index = (
        int(action.get("shop_index", -1))
        if action_type in {"BUY_PET", "BUY_COMBINE", "BUY_FOOD"}
        else None
    )
    remaining = [
        slot
        for slot in (state_before.get("shop") or [])
        if consumed_index is None
        or int(slot.get("shop_index", -1)) != consumed_index
    ]
    matched: set[int] = set()
    result: dict[tuple[int, str, str], str] = {}
    for slot in state_after.get("shop") or []:
        key = _shop_key(slot)
        item_id = str(slot.get("item_id") or "")
        lineage = _shop_lineage_key(slot)
        prior_index = next(
            (
                index
                for index, prior in enumerate(remaining)
                if index not in matched and _shop_lineage_key(prior) == lineage
            ),
            None,
        )
        if prior_index is not None:
            matched.add(prior_index)
            provenance = current.get(_shop_key(remaining[prior_index]), default)
        elif injected[item_id] > 0:
            provenance = "ability_injected"
            injected[item_id] -= 1
        elif slot.get("link_id"):
            provenance = "levelup_reward"
        else:
            provenance = default
        result[key] = provenance
    return result


def shop_snapshot(
    state: dict[str, Any] | None,
    *,
    shop_provenance: str,
    provenance: dict[tuple[int, str, str], str] | None = None,
) -> dict[str, Any]:
    slots = []
    for slot in sorted(
        (state or {}).get("shop") or [],
        key=lambda row: int(row.get("shop_index", -1)),
    ):
        item_provenance = (provenance or {}).get(
            _shop_key(slot),
            "levelup_reward" if slot.get("link_id") else shop_provenance,
        )
        slots.append(
            _shop_item(
                slot,
                provenance=item_provenance,
                injected=item_provenance == "ability_injected",
            )
        )
    return {
        "slots": slots,
        "pets": [copy.deepcopy(slot) for slot in slots if slot["slot_type"] == "pet"],
        "foods": [copy.deepcopy(slot) for slot in slots if slot["slot_type"] == "food"],
        "shop_provenance": str(shop_provenance),
    }


def raw_transition(transition: dict[str, Any]) -> dict[str, Any]:
    """Keep only the pinned engine transition contract, without re-execution."""
    return {
        key: copy.deepcopy(transition.get(key))
        for key in _TRANSITION_KEYS
    }


def _shop_slot(state: dict[str, Any], shop_index: Any) -> dict[str, Any]:
    for slot in state.get("shop") or []:
        if int(slot.get("shop_index", -1)) == int(shop_index):
            return slot
    return {}


def _team_slot(state: dict[str, Any], team_index: Any) -> dict[str, Any]:
    for slot in state.get("team") or []:
        if int(slot.get("slot_index", -1)) == int(team_index):
            return slot
    return {}


def _team_item(state: dict[str, Any], team_index: Any) -> dict[str, Any]:
    slot = _team_slot(state, team_index)
    index = int(team_index)
    item_id = slot.get("pet_id")
    return {
        "slot": index,
        "name": _name(item_id),
        "item_id": item_id,
        "level": int(slot.get("level", 1)),
        "exp": int(slot.get("exp", 0)),
        "attack": int(slot.get("attack", 0)),
        "health": int(slot.get("health", 0)),
        "equipment_id": slot.get("equipment_id"),
        "status_effects": copy.deepcopy(slot.get("status_effects") or []),
        **_identity_fields(item_id, location="team", index=index),
    }


def _tier_up_details(transition: dict[str, Any]) -> dict[str, Any]:
    """Attribute archived level-up reward choices to one combine.

    The transition already contains the authoritative before/after shops and
    engine notes. Comparing lineage counts finds only newly inserted linked
    slots, even when older level-up offers remain in the shop. No engine RNG
    is re-run and an incomplete legacy transition is never guessed.
    """
    before = transition.get("state_before") or {}
    after = transition.get("state_after") or {}
    notes = _notes(transition)
    suppressed = any(
        note.startswith("levelup_reward_suppressed:") for note in notes
    )
    # A target card can visibly move from L1 to L2 without earning a reward
    # pair, for example L2 merged onto L1. Only archived reward evidence marks
    # Tier Up choices; level delta alone would invent an offer.
    note_links = []
    for note in notes:
        match = re.search(
            r"(?:levelup_reward_pair_added:\d+:link=|"
            r"levelup_reward_pair_skipped:link=)([^:\s]+)",
            note,
        )
        if match is not None:
            note_links.append(match.group(1))

    remaining = Counter(
        _shop_lineage_key(slot) for slot in (before.get("shop") or [])
    )
    added_linked: list[dict[str, Any]] = []
    for slot in sorted(
        after.get("shop") or [],
        key=lambda row: int(row.get("shop_index", -1)),
    ):
        lineage = _shop_lineage_key(slot)
        if remaining[lineage] > 0:
            remaining[lineage] -= 1
            continue
        if slot.get("link_id"):
            added_linked.append(slot)

    added_links = list(dict.fromkeys(str(slot.get("link_id")) for slot in added_linked))
    link_id = note_links[0] if len(set(note_links)) == 1 else None
    if link_id is None and len(added_links) == 1:
        link_id = added_links[0]
    triggered = not suppressed and bool(note_links or added_linked)
    choices = [
        _shop_item(slot, provenance="levelup_reward", injected=False)
        for slot in added_linked
        if link_id is not None and str(slot.get("link_id")) == str(link_id)
    ]
    return {
        "tier_up_triggered": bool(triggered),
        "tier_up_link_id": link_id if triggered else None,
        "tier_up_choices": choices if triggered else [],
        "tier_up_choices_available": bool(triggered and choices),
        "tier_up_unavailable_reason": (
            "archived_tier_up_choices_unavailable"
            if triggered and not choices
            else None
        ),
    }


def _base_event(
    transition: dict[str, Any],
    transition_index: int,
    op: str,
) -> dict[str, Any]:
    before = transition.get("state_before") or {}
    after = transition.get("state_after") or {}
    return {
        "op": op,
        "transition_index": int(transition_index),
        "raw_action": copy.deepcopy(transition.get("action") or {}),
        "gold_before": int(before.get("gold", 0)),
        "gold_after": int(after.get("gold", 0)),
        "gold_delta": int(after.get("gold", 0)) - int(before.get("gold", 0)),
        "transition": raw_transition(transition),
    }


def _food_target_indices(transition: dict[str, Any]) -> list[int]:
    action = transition.get("action") or {}
    explicit = action.get("team_index")
    if explicit is not None:
        return [int(explicit)]
    item_id = str(
        _shop_slot(
            transition.get("state_before") or {}, action.get("shop_index")
        ).get("item_id")
        or ""
    )
    prefix = f"food_targets:{item_id}:"
    for note in _notes(transition):
        if note.startswith(prefix):
            payload = note[len(prefix):]
            if payload == "none" or not payload:
                return []
            return [int(value) for value in payload.split(",")]
    return []


def _normalized_action(
    transition: dict[str, Any],
    transition_index: int,
    provenance: dict[tuple[int, str, str], str],
    *,
    end_context: dict[str, Any],
) -> list[dict[str, Any]]:
    before = transition.get("state_before") or {}
    after = transition.get("state_after") or {}
    action = transition.get("action") or {}
    action_type = str(action.get("type") or "").strip().upper()
    op = {
        "COMBINE": "merge",
        "BUY_PET": "buy_pet",
        "BUY_COMBINE": "buy_combine",
        "BUY_FOOD": "buy_food",
        "SELL": "sell",
        "FREEZE": "freeze",
        "UNFREEZE": "unfreeze",
        "REORDER": "reorder",
        "ROLL": "roll",
        "END_TURN": "end_turn",
    }.get(action_type, action_type.lower() or "unknown")
    event = _base_event(transition, transition_index, op)

    if action_type == "ROLL":
        next_provenance = _initial_provenance(
            after,
            source="manual_roll",
            state_before=before,
            transition=transition,
        )
        event.update(
            {
                "shop": shop_snapshot(
                    after,
                    shop_provenance="manual_roll",
                    provenance=next_provenance,
                ),
                "shop_provenance": "manual_roll",
            }
        )
    elif action_type in {"FREEZE", "UNFREEZE"}:
        slot = _shop_slot(before, action.get("shop_index"))
        item_provenance = provenance.get(_shop_key(slot), "engine_shop")
        item = _shop_item(
            slot,
            provenance=item_provenance,
            injected=item_provenance == "ability_injected",
        )
        event.update(
            {
                "items": [item],
                "shop_index": int(action.get("shop_index", -1)),
                "provenance": item_provenance,
                "identity_source": _IDENTITY_SOURCE,
                "shop": shop_snapshot(
                    after,
                    shop_provenance="unchanged",
                    provenance=_propagate_provenance(
                        before, after, provenance, transition
                    ),
                ),
            }
        )
    elif action_type in {"BUY_PET", "BUY_COMBINE", "BUY_FOOD"}:
        slot = _shop_slot(before, action.get("shop_index"))
        item_provenance = provenance.get(_shop_key(slot), "engine_shop")
        item = _shop_item(
            slot,
            provenance=item_provenance,
            injected=item_provenance == "ability_injected",
        )
        event.update(
            {
                "name": item["name"],
                "item_id": item["item_id"],
                "uni": None,
                "item": item,
                "from_shop_slot": int(action.get("shop_index", -1)),
                "provenance": item_provenance,
                "identity_src": _IDENTITY_SOURCE,
                "identity_source": _IDENTITY_SOURCE,
            }
        )
        if action_type == "BUY_PET":
            target_index = int(action.get("team_index", -1))
            event["to_slot"] = target_index
            event["pet_after"] = _team_item(after, target_index)
        elif action_type == "BUY_COMBINE":
            target_index = int(action.get("team_index", -1))
            event["onto"] = _team_item(before, target_index)
            event["onto_after"] = _team_item(after, target_index)
            event.update(_tier_up_details(transition))
        else:
            target_indices = _food_target_indices(transition)
            targets = [
                {
                    "slot": target_index,
                    "before": _team_item(before, target_index),
                    "after": _team_item(after, target_index),
                }
                for target_index in target_indices
            ]
            event["target_mode"] = (
                "explicit"
                if action.get("team_index") is not None
                else ("random" if target_indices else "no_target")
            )
            event["target_indices"] = target_indices
            event["targets"] = targets
            event["target"] = targets[0]["before"] if len(targets) == 1 else None
            event["target_after"] = targets[0]["after"] if len(targets) == 1 else None
            event["free"] = int(slot.get("cost", 0)) == 0
    elif action_type == "SELL":
        index = int(action.get("team_index", -1))
        pet = _team_item(before, index)
        event.update(
            {
                "name": pet["name"],
                "item_id": pet["item_id"],
                "uni": None,
                "pet": pet,
                "from_slot": index,
                "level": pet["level"],
                "provenance": "team",
                "identity_src": _IDENTITY_SOURCE,
                "identity_source": _IDENTITY_SOURCE,
            }
        )
    elif action_type == "COMBINE":
        src_index = int(action.get("src_team_index", -1))
        dst_index = int(action.get("dst_team_index", -1))
        event.update(
            {
                "src": _team_item(before, src_index),
                "dst": _team_item(before, dst_index),
                "dst_after": _team_item(after, dst_index),
                "src_slot": src_index,
                "dst_slot": dst_index,
                "provenance": "team",
                "identity_src": _IDENTITY_SOURCE,
                "identity_source": _IDENTITY_SOURCE,
            }
        )
        event.update(_tier_up_details(transition))
    elif action_type == "REORDER":
        event.update(
            {
                "order": copy.deepcopy(action.get("order") or []),
                "board_before": board_slots(before),
                "board_after": board_slots(after),
            }
        )
    elif action_type == "END_TURN":
        next_provenance = _initial_provenance(
            after,
            source="automatic_reroll",
            state_before=before,
            transition=transition,
        )
        frozen = [
            slot
            for slot in shop_snapshot(
                after,
                shop_provenance="automatic_reroll",
                provenance=next_provenance,
            )["slots"]
            if slot["frozen"]
        ]
        event.update(
            {
                "frozen_carried": [slot["name"] for slot in frozen],
                "frozen_items": frozen,
                "next_shop": shop_snapshot(
                    after,
                    shop_provenance="automatic_reroll",
                    provenance=next_provenance,
                ),
                "board_pre_battle": copy.deepcopy(
                    end_context.get("board_pre_battle") or []
                ),
                "battle_result": copy.deepcopy(end_context),
            }
        )

    events = [event]
    for spec in _ability_specs(transition):
        events.append(
            {
                "op": "ability_stock",
                "transition_index": int(transition_index),
                "source": spec["source"],
                "source_id": spec["source_id"],
                "name": spec["name"],
                "item_id": spec["item_id"],
                "uni": None,
                "count": int(spec["count"]),
                "gold_delta": 0,
                "provenance": "ability_injected",
                "injected": True,
                "identity_src": _IDENTITY_SOURCE,
                "identity_source": _IDENTITY_SOURCE,
                "transition": raw_transition(transition),
            }
        )
    return events


def _frozen_retained(before: dict[str, Any], after: dict[str, Any]) -> bool:
    expected = Counter(
        _frozen_signature(slot)
        for slot in before.get("shop") or []
        if slot.get("frozen")
    )
    actual = Counter(
        _frozen_signature(slot)
        for slot in after.get("shop") or []
        if slot.get("frozen")
    )
    return all(actual[key] >= count for key, count in expected.items())


def build_side_replay(
    *,
    turn: int,
    start_transition: dict[str, Any],
    transitions: list[dict[str, Any]],
    board_pre_battle: list[dict[str, Any]],
    end_context: dict[str, Any],
) -> dict[str, Any]:
    """Build one side's exp08-complete view from stored real transitions."""
    start_before = start_transition.get("state_before") or {}
    start_after = start_transition.get("state_after") or {}
    source = "initial_deal" if int(turn) == 1 else "automatic_reroll"
    provenance = _initial_provenance(
        start_after,
        source=source,
        state_before=start_before,
        transition=start_transition,
    )
    start_event = {
        "op": "start_turn",
        "source": source,
        "transition_index": -1,
        "board": board_slots(start_after),
        "gold": int(start_after.get("gold", 0)),
        "shop": shop_snapshot(
            start_after,
            shop_provenance=source,
            provenance=provenance,
        ),
        "shop_provenance": source,
        "transition": raw_transition(start_transition),
    }

    chain: list[dict[str, Any]] = [start_event]
    current_provenance = provenance
    shop_transitions: list[dict[str, Any]] = []
    freeze_ok = _frozen_retained(start_before, start_after)
    for index, transition in enumerate(transitions):
        chain.extend(
            _normalized_action(
                transition,
                index,
                current_provenance,
                end_context=end_context,
            )
        )
        action_type = str(
            ((transition.get("action") or {}).get("type") or "")
        ).strip().upper()
        before = transition.get("state_before") or {}
        after = transition.get("state_after") or {}
        if action_type == "ROLL":
            current_provenance = _initial_provenance(
                after,
                source="manual_roll",
                state_before=before,
                transition=transition,
            )
            freeze_ok = freeze_ok and _frozen_retained(before, after)
        else:
            current_provenance = _propagate_provenance(
                before,
                after,
                current_provenance,
                transition,
                default=(
                    "automatic_reroll"
                    if action_type == "END_TURN"
                    else "engine_shop"
                ),
            )
            if action_type == "END_TURN":
                freeze_ok = freeze_ok and _frozen_retained(before, after)
        if action_type != "END_TURN":
            shop_transitions.append(transition)

    raw_chain = [start_transition, *transitions]
    continuity = all(
        (left.get("state_after") or {}) == (right.get("state_before") or {})
        for left, right in zip(raw_chain, raw_chain[1:])
    )
    legal = all(bool(transition.get("legal")) for transition in raw_chain)
    board_in = board_slots(start_after)
    board_complete = [slot["slot"] for slot in board_in] == list(range(5))

    base = int(start_after.get("gold", 0))
    gained = 0
    spent = 0
    rolls = 0
    left = base
    unattributed: list[dict[str, Any]] = []
    known_gold_ops = {
        "BUY_PET",
        "BUY_COMBINE",
        "BUY_FOOD",
        "SELL",
        "COMBINE",
        "FREEZE",
        "UNFREEZE",
        "REORDER",
        "ROLL",
    }
    for transition in shop_transitions:
        before = transition.get("state_before") or {}
        after = transition.get("state_after") or {}
        delta = int(after.get("gold", 0)) - int(before.get("gold", 0))
        if delta > 0:
            gained += delta
        elif delta < 0:
            spent += -delta
        action_type = str(
            ((transition.get("action") or {}).get("type") or "")
        ).strip().upper()
        rolls += int(action_type == "ROLL")
        left = int(after.get("gold", left))
        if delta and action_type not in known_gold_ops:
            unattributed.append(
                {
                    "gold_delta": delta,
                    "action": copy.deepcopy(transition.get("action") or {}),
                    "engine_notes": copy.deepcopy(transition.get("engine_notes") or []),
                }
            )
    reconciled = base + gained - spent == left and not unattributed
    passed = bool(continuity and legal and board_complete and reconciled and freeze_ok)
    return {
        "version": REPLAY_SCHEMA_VERSION,
        "detail_available": True,
        "turn": int(turn),
        "tier": int(tier_for_turn(int(turn))),
        "gold": {
            "base": base,
            "gained": gained,
            "spent": spent,
            "left": left,
            "rolls": rolls,
            "reconciled": reconciled,
            "unattributed": unattributed,
        },
        "board_in": board_in,
        "chain": chain,
        "self_check": {
            "bucket": "engine_transition_v2",
            "pass": passed,
            "freeze_track_ok": bool(freeze_ok),
            "transition_continuity": bool(continuity),
            "action_legality": bool(legal),
            "board_slots_complete": bool(board_complete),
        },
        "board_pre_battle": copy.deepcopy(board_pre_battle),
    }


def _legacy_side_replay(turn: int, side: dict[str, Any]) -> dict[str, Any]:
    state = side.get("state_before") or {}
    return {
        "version": 1,
        "detail_available": False,
        "unavailable_reason": "legacy_v1_transition_detail_unavailable",
        "turn": int(turn),
        "tier": int(tier_for_turn(int(turn))),
        "gold": {
            "base": int(state.get("gold", 0)),
            "gained": None,
            "spent": None,
            "left": None,
            "rolls": None,
            "reconciled": None,
            "unattributed": [],
        },
        "board_in": board_slots(state),
        "chain": [
            {
                "op": "start_turn",
                "source": "legacy_state_snapshot",
                "detail_available": False,
                "unavailable_reason": "legacy_v1_transition_detail_unavailable",
                "shop": shop_snapshot(
                    state,
                    shop_provenance="legacy_state_snapshot",
                ),
                "shop_provenance": "legacy_state_snapshot",
                "transition": None,
            }
        ],
        "self_check": {
            "bucket": "legacy_v1",
            "pass": None,
            "freeze_track_ok": None,
        },
        "board_pre_battle": copy.deepcopy(side.get("board_pre_battle") or []),
    }


def summarize_public_replay(view: dict[str, Any]) -> dict[str, Any]:
    """Drop the two archive-only blobs the replays page keeps behind a
    collapsed `<details>`, leaving markers that say the detail exists.

    THE COMPLAINT (2026-08-12). `/replays` did not paint at all over the ssh
    tunnel from the US. Nothing was broken: opening the page selects the newest
    game and fetches it whole, and a completed 14-turn duel is 24.8 MB of
    compact JSON. The tunnel measured 125 KB/s, so the first paint needed
    around eight minutes of transfer for a page whose visible content is a few
    hundred kilobytes.

    Two fields are 22 of those 24.8 MB, and NEITHER IS ON SCREEN until it is
    clicked:

      turn["segments"]              the AI's searched chains, rendered inside
                                    "AI search, life, and battle telemetry"
      event["transition"]           the raw engine transition per action,
                                    rendered inside "Raw engine transition"

    Measured on `20260809T094214661410Z-648708037-b0f43aed`, compact then
    gzipped: 24.8 MB / 3.08 MB whole, 8.93 MB / 254 KB without `segments`,
    2.82 MB / 150 KB without both. So the page paints from 150 KB instead of
    3.08 MB, and the detail arrives when someone actually opens it.

    Nothing is lost. `/api/duel/replay_turn` serves one turn with both fields
    intact, which is what the page fetches on toggle, and `detail=full` on
    `/api/duel/replay` still answers with the whole thing.

    This takes ownership of `view` and mutates it, which is safe because every
    caller passes a freshly built `build_public_replay` result. Deliberately
    NOT folded into `build_public_replay`: that function's output is the
    archive's public shape, validated against `schemas/play_web_game_v2.json`,
    and a transport projection has no business changing it.
    """
    view["detail"] = "summary"
    for turn in view.get("turns") or []:
        segments = turn.get("segments")
        if isinstance(segments, list):
            # `n_segments` and the other counts are scalars and stay; it is the
            # segment records themselves that are heavy.
            #
            # Amendment 6 keeps ONE derived list: the searched width of each
            # segment and how it ended. That is the "widths" column the live
            # page has always shown per turn, and Ruihan asked for the same
            # thing to be readable from a replay. It is a handful of small
            # integers against segment records that hold whole candidate
            # chains, so it does not undo what this projection is for.
            turn["segment_widths"] = [
                {
                    "width": int((s or {}).get("width") or 0),
                    "mode": str((s or {}).get("mode") or ""),
                    **({"width_requested": int(s["width_requested"])}
                       if s.get("width_requested") is not None else {}),
                }
                for s in segments
                if isinstance(s, dict)
            ]
            turn["segments"] = None
            turn["segments_available"] = len(segments)
        for chain in _detail_chains(turn):
            for event in chain:
                if isinstance(event, dict) and event.get("transition") is not None:
                    event["transition"] = None
                    event["transition_available"] = True
    return view


def _detail_chains(turn: dict[str, Any]) -> list[list[Any]]:
    """Every list of events in a public turn that can carry a transition.

    There are three: each side's own replay chain, and the copy of the human
    chain that `build_public_replay` lifts to the top of the turn. Missing one
    of them leaves the payload heavy while looking fixed, so they are collected
    structurally here rather than spelled out at each call site.
    """
    chains: list[list[Any]] = []
    for side_name in ("human", "ai"):
        side = turn.get(side_name)
        if not isinstance(side, dict):
            continue
        replay = side.get("replay")
        if isinstance(replay, dict) and isinstance(replay.get("chain"), list):
            chains.append(replay["chain"])
    if isinstance(turn.get("chain"), list):
        chains.append(turn["chain"])
    return chains


def build_public_replay(game: dict[str, Any]) -> dict[str, Any]:
    """Return the completed-only browser view for a raw v1 or v2 archive."""
    if not is_completed_game(game):
        raise ValueError("replay_not_complete")
    view = copy.deepcopy(game)
    view["ai_version"] = normalize_ai_version(view.get("ai_version"))
    game_id = str(view.get("id") or "")
    turns = sorted(
        list(view.get("turns") or []),
        key=lambda row: int(row.get("turn", -1)),
    )
    view["turns"] = turns
    view["game_id"] = game_id
    view["max_turn"] = int((view.get("rules") or {}).get("turn_cap") or 0)
    view["completion"] = {
        "status": "completed",
        "result": {
            "winner": view.get("winner"),
            "reason": view.get("end_reason"),
        },
    }
    last_render_turn = next(
        (
            int(turn.get("turn"))
            for turn in reversed(turns)
            if (turn.get("render") or {}).get("path")
        ),
        None,
    )
    view["full_game_png"] = (
        f"/api/duel/replay_image?id={game_id}&turn={last_render_turn}"
        if last_render_turn is not None
        else None
    )

    legacy = int(view.get("schema_version") or 1) < REPLAY_SCHEMA_VERSION
    for turn in turns:
        number = int(turn.get("turn", 0))
        for side_name in ("human", "ai"):
            side = turn.setdefault(side_name, {})
            if legacy or not isinstance(side.get("replay"), dict):
                side["replay"] = _legacy_side_replay(number, side)
            else:
                for event in side["replay"].get("chain") or []:
                    transition = event.get("transition")
                    if isinstance(transition, dict):
                        event["transition"] = raw_transition(transition)
        human_replay = turn["human"]["replay"]
        for key in ("tier", "gold", "board_in", "chain", "self_check"):
            turn[key] = copy.deepcopy(human_replay.get(key))
    return view
