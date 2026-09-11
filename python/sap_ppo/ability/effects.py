"""Reusable ability effect primitives."""

from __future__ import annotations

from typing import Any

from ..constants import EQUIPMENT_FOOD_IDS, EQUIPMENT_STATUS_BY_FOOD_ID
from .events import (
    AbilityEvent,
    TRIGGER_EATS_FOOD,
    TRIGGER_END_OF_TURN,
    TRIGGER_FRIENDLY_ATE_FOOD,
)

ABILITY_COUNTERS_KEY = "ability_counters"


def _cap_stat(value: int) -> int:
    return max(0, min(50, int(value)))


def ensure_stat_fields(slot: dict[str, Any]) -> None:
    pet_id = slot.get("pet_id")
    if pet_id is None:
        slot.pop("perm_attack", None)
        slot.pop("perm_health", None)
        slot.pop("temp_attack", None)
        slot.pop("temp_health", None)
        return

    observed_attack = int(slot.get("attack", 0))
    observed_health = int(slot.get("health", 0))
    temp_attack = max(0, int(slot.get("temp_attack", 0)))
    temp_health = max(0, int(slot.get("temp_health", 0)))
    perm_attack = int(slot.get("perm_attack", observed_attack - temp_attack))
    perm_health = int(slot.get("perm_health", observed_health - temp_health))
    # Keep canonical fields consistent when legacy paths mutate attack/health directly.
    if perm_attack + temp_attack != observed_attack:
        perm_attack = observed_attack - temp_attack
    if perm_health + temp_health != observed_health:
        perm_health = observed_health - temp_health
    slot["perm_attack"] = _cap_stat(perm_attack)
    slot["perm_health"] = _cap_stat(perm_health)
    slot["temp_attack"] = _cap_stat(temp_attack)
    slot["temp_health"] = _cap_stat(temp_health)
    slot["attack"] = _cap_stat(int(slot["perm_attack"]) + int(slot["temp_attack"]))
    slot["health"] = _cap_stat(int(slot["perm_health"]) + int(slot["temp_health"]))


def add_slot_stats(slot: dict[str, Any], attack: int = 0, health: int = 0, *, temporary: bool = False) -> None:
    if slot.get("pet_id") is None:
        return
    ensure_stat_fields(slot)
    if temporary:
        slot["temp_attack"] = _cap_stat(int(slot.get("temp_attack", 0)) + int(attack))
        slot["temp_health"] = _cap_stat(int(slot.get("temp_health", 0)) + int(health))
    else:
        slot["perm_attack"] = _cap_stat(int(slot.get("perm_attack", 0)) + int(attack))
        slot["perm_health"] = _cap_stat(int(slot.get("perm_health", 0)) + int(health))
    slot["attack"] = _cap_stat(int(slot.get("perm_attack", 0)) + int(slot.get("temp_attack", 0)))
    slot["health"] = _cap_stat(int(slot.get("perm_health", 0)) + int(slot.get("temp_health", 0)))


def clear_temporary_team_stats(state: dict[str, Any]) -> int:
    cleared = 0
    for slot in state.get("team", []):
        if slot.get("pet_id") is None:
            ensure_stat_fields(slot)
            continue
        ensure_stat_fields(slot)
        temp_attack = int(slot.get("temp_attack", 0))
        temp_health = int(slot.get("temp_health", 0))
        if temp_attack != 0 or temp_health != 0:
            cleared += 1
        slot["temp_attack"] = 0
        slot["temp_health"] = 0
        slot["attack"] = _cap_stat(int(slot.get("perm_attack", 0)))
        slot["health"] = _cap_stat(int(slot.get("perm_health", 0)))
    return cleared


def occupied_team_indices(state: dict[str, Any]) -> list[int]:
    return [i for i, slot in enumerate(state.get("team", [])) if slot.get("pet_id") is not None]


def friend_indices(state: dict[str, Any], actor_team_index: int | None) -> list[int]:
    out = occupied_team_indices(state)
    if actor_team_index is not None:
        out = [i for i in out if i != actor_team_index]
    return out


def friend_ahead_indices(state: dict[str, Any], actor_team_index: int | None) -> list[int]:
    if actor_team_index is None:
        return []
    # sapai FriendAhead: lower team index and nearest first.
    occupied = occupied_team_indices(state)
    candidates = [i for i in occupied if i < int(actor_team_index)]
    return sorted(candidates, reverse=True)


def friend_behind_indices(state: dict[str, Any], actor_team_index: int | None) -> list[int]:
    if actor_team_index is None:
        return []
    # Project convention: higher team index means farther left / behind.
    occupied = occupied_team_indices(state)
    candidates = [i for i in occupied if i > int(actor_team_index)]
    return sorted(candidates)


def _known_equipment_statuses() -> set[str]:
    return set(EQUIPMENT_STATUS_BY_FOOD_ID.values())


def has_standard_equipment(slot: dict[str, Any], equipment_id: str) -> bool:
    """Return True for canonical and legacy representations of one standard perk."""
    equip = str(equipment_id)
    if str(slot.get("equipment_id", "")) == equip:
        return True
    status = EQUIPMENT_STATUS_BY_FOOD_ID.get(equip)
    return bool(status and status in {str(x) for x in slot.get("status_effects", [])})


def apply_standard_equipment(slot: dict[str, Any], equipment_id: str) -> None:
    """Atomically replace a pet's one standard equipment perk."""
    equip = str(equipment_id)
    if equip not in EQUIPMENT_FOOD_IDS:
        raise ValueError(f"not a standard equipment food: {equip}")
    effects = [str(x) for x in slot.get("status_effects", [])]
    filtered = [effect for effect in effects if effect not in _known_equipment_statuses()]
    status = EQUIPMENT_STATUS_BY_FOOD_ID.get(equip)
    if status:
        filtered.append(status)
    slot["status_effects"] = sorted(set(filtered))
    slot["equipment_id"] = equip


def effective_actor_for_trigger(
    state: dict[str, Any],
    team_index: int,
    trigger: str,
    fallback_pet_id: str,
    fallback_level: int,
) -> tuple[str, int]:
    """Resolve Parrot copies consistently for ordinary and perk-grant food events."""
    if str(fallback_pet_id) != "pet-parrot" or str(trigger) == TRIGGER_END_OF_TURN:
        return str(fallback_pet_id), int(fallback_level)
    meta = state.setdefault("meta", {})
    store = meta.setdefault("_parrot_copy", {})
    if not isinstance(store, dict):
        store = {}
        meta["_parrot_copy"] = store
    payload = store.get(str(team_index), {})
    if not isinstance(payload, dict):
        return str(fallback_pet_id), int(fallback_level)
    copied_pet_id = str(payload.get("pet_id", ""))
    if copied_pet_id in {"", "pet-parrot"}:
        return str(fallback_pet_id), int(fallback_level)
    try:
        copied_level = int(payload.get("level", fallback_level))
    except (TypeError, ValueError):
        copied_level = int(fallback_level)
    return copied_pet_id, max(1, min(3, copied_level))


def build_food_trigger_event_groups(
    state: dict[str, Any], target_team_index: int, food_item_id: str
) -> tuple[list[AbilityEvent], list[AbilityEvent]]:
    """Build the eats-food then friendly-ate-food groups for one fed pet."""
    target_idx = int(target_team_index)
    team = state.get("team", [])
    if target_idx < 0 or target_idx >= len(team):
        return [], []
    target = team[target_idx]
    if target.get("pet_id") is None:
        return [], []

    item_id = str(food_item_id)
    target_pet_id = str(target.get("pet_id", ""))
    eats_actor_pet_id, eats_actor_level = effective_actor_for_trigger(
        state,
        target_idx,
        TRIGGER_EATS_FOOD,
        target_pet_id,
        int(target.get("level", 1)),
    )
    eats_events = [
        AbilityEvent(
            trigger=TRIGGER_EATS_FOOD,
            actor_pet_id=eats_actor_pet_id,
            actor_level=eats_actor_level,
            actor_team_index=target_idx,
            payload={"food_item_id": item_id},
        )
    ]

    friendly_events: list[AbilityEvent] = []
    for idx, slot in enumerate(team):
        pet_id = slot.get("pet_id")
        if pet_id is None:
            continue
        actor_pet_id, actor_level = effective_actor_for_trigger(
            state,
            idx,
            TRIGGER_FRIENDLY_ATE_FOOD,
            str(pet_id),
            int(slot.get("level", 1)),
        )
        friendly_events.append(
            AbilityEvent(
                trigger=TRIGGER_FRIENDLY_ATE_FOOD,
                actor_pet_id=actor_pet_id,
                actor_level=actor_level,
                actor_team_index=idx,
                payload={
                    "trigger_team_index": target_idx,
                    "trigger_pet_id": target_pet_id,
                    "food_item_id": item_id,
                },
            )
        )
    return eats_events, friendly_events


def grant_standard_perk(
    state: dict[str, Any], target_team_index: int, equipment_id: str, *, ctx: Any
) -> bool:
    """Grant standard equipment and synchronously emit the normal food triggers."""
    target_idx = int(target_team_index)
    team = state.get("team", [])
    if target_idx < 0 or target_idx >= len(team):
        return False
    slot = team[target_idx]
    if slot.get("pet_id") is None:
        return False
    apply_standard_equipment(slot, equipment_id)
    eats_events, friendly_events = build_food_trigger_event_groups(
        state, target_idx, equipment_id
    )
    ctx.emit_groups(eats_events, friendly_events)
    return True


def rightmost_friendly_index(state: dict[str, Any]) -> int | None:
    candidates = occupied_team_indices(state)
    if not candidates:
        return None
    # Project convention: rightmost board pet maps to the lowest team slot index.
    return min(candidates)


def has_level3_friend(state: dict[str, Any], actor_team_index: int | None) -> bool:
    for idx in friend_indices(state, actor_team_index):
        slot = state["team"][idx]
        if int(slot.get("level", 1)) >= 3:
            return True
    return False


def mark_structural(ctx: Any) -> None:
    """Record that this ability resolution wrote a field `legal_actions` reads."""
    ctx.structural_used = True
    if ctx.causality is not None:
        ctx.causality.note_structural()


def mark_random(ctx: Any) -> None:
    """Record a draw whose OUTCOME feeds what this resolution goes on to write."""
    ctx.random_used = True
    if ctx.causality is not None:
        ctx.causality.arm()


def choose_random_indices(ctx: Any, candidates: list[int], n: int) -> list[int]:
    if n <= 0 or not candidates:
        return []
    # Ability targeting should never hit the same friend twice.
    unique_candidates = sorted(set(int(x) for x in candidates))
    if len(unique_candidates) <= n:
        return unique_candidates


    mark_random(ctx)
    return sorted(ctx.rng.sample(unique_candidates, n))


def buff_team_slots(state: dict[str, Any], team_indices: list[int], attack: int = 0, health: int = 0) -> None:
    for idx in team_indices:
        slot = state["team"][idx]
        add_slot_stats(slot, attack=int(attack), health=int(health), temporary=False)


def buff_team_slots_temporary(state: dict[str, Any], team_indices: list[int], attack: int = 0, health: int = 0) -> None:
    for idx in team_indices:
        slot = state["team"][idx]
        add_slot_stats(slot, attack=int(attack), health=int(health), temporary=True)


def buff_shop_pets(state: dict[str, Any], attack: int = 0, health: int = 0) -> int:
    touched = 0
    for slot in state.get("shop", []):
        if slot.get("slot_type") != "pet":
            continue
        if str(slot.get("item_id", "")) in {"", "pet-none"}:
            continue
        slot["attack"] = _cap_stat(int(slot.get("attack", 0)) + int(attack))
        slot["health"] = _cap_stat(int(slot.get("health", 0)) + int(health))
        touched += 1
    return touched


def gain_gold(state: dict[str, Any], amount: int, *, ctx: Any) -> None:
    state["gold"] = max(0, int(state.get("gold", 0)) + int(amount))
    mark_structural(ctx)


def discount_shop_foods(state: dict[str, Any], amount: int, *, ctx: Any) -> int:
    discount = max(0, int(amount))
    touched = 0
    mark_structural(ctx)
    for slot in state.get("shop", []):
        if slot.get("slot_type") != "food":
            continue
        cost = int(slot.get("cost", 3))
        slot["cost"] = max(0, cost - discount)
        touched += 1
    return touched


def _ability_counter_store(state: dict[str, Any]) -> dict[str, dict[str, int]]:
    meta = state.setdefault("meta", {})
    store = meta.setdefault(ABILITY_COUNTERS_KEY, {})
    if not isinstance(store, dict):
        store = {}
        meta[ABILITY_COUNTERS_KEY] = store
    return store


def get_ability_counter(state: dict[str, Any], trigger: str, team_index: int | None) -> int:
    if team_index is None:
        return 0
    store = _ability_counter_store(state)
    by_trigger = store.get(str(trigger), {})
    if not isinstance(by_trigger, dict):
        return 0
    try:
        return max(0, int(by_trigger.get(str(team_index), 0)))
    except (TypeError, ValueError):
        return 0


def increment_ability_counter(state: dict[str, Any], trigger: str, team_index: int | None, amount: int = 1) -> int:
    if team_index is None:
        return 0
    inc = max(0, int(amount))
    store = _ability_counter_store(state)
    by_trigger = store.setdefault(str(trigger), {})
    if not isinstance(by_trigger, dict):
        by_trigger = {}
        store[str(trigger)] = by_trigger
    next_value = get_ability_counter(state, trigger, team_index) + inc
    by_trigger[str(team_index)] = int(next_value)
    return int(next_value)


def decrement_ability_counter(state: dict[str, Any], trigger: str, team_index: int | None, amount: int = 1) -> int:
    if team_index is None:
        return 0
    dec = max(0, int(amount))
    store = _ability_counter_store(state)
    by_trigger = store.setdefault(str(trigger), {})
    if not isinstance(by_trigger, dict):
        by_trigger = {}
        store[str(trigger)] = by_trigger
    curr = get_ability_counter(state, trigger, team_index)
    next_value = max(0, int(curr) - dec)
    by_trigger[str(team_index)] = int(next_value)
    return int(next_value)


def clear_ability_counters_for_team_index(state: dict[str, Any], team_index: int) -> None:
    store = _ability_counter_store(state)
    key = str(team_index)
    for by_trigger in store.values():
        if isinstance(by_trigger, dict):
            by_trigger.pop(key, None)


def remap_ability_counters_for_reorder(state: dict[str, Any], order: list[int]) -> None:
    store = _ability_counter_store(state)
    old_to_new = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(order)}
    for trigger, by_trigger in list(store.items()):
        if not isinstance(by_trigger, dict):
            continue
        remapped: dict[str, int] = {}
        for old_idx_str, value in by_trigger.items():
            try:
                old_idx = int(old_idx_str)
                new_idx = old_to_new[old_idx]
                remapped[str(new_idx)] = int(value)
            except (TypeError, ValueError, KeyError):
                continue
        store[str(trigger)] = remapped


def reset_ability_counters(state: dict[str, Any]) -> None:
    meta = state.setdefault("meta", {})
    meta[ABILITY_COUNTERS_KEY] = {}
