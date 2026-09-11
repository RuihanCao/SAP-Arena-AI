"""Turtle pack ability handlers (incremental Step 2 implementation)."""

from __future__ import annotations

from typing import Any, Callable

from ..catalog import load_turtle_catalog
from ..constants import FOOD_STAT_BUFFS
from .effects import (
    buff_shop_pets,
    buff_team_slots,
    buff_team_slots_temporary,
    choose_random_indices,
    discount_shop_foods,
    friend_ahead_indices,
    friend_behind_indices,
    friend_indices,
    gain_gold,
    get_ability_counter,
    grant_standard_perk,
    has_standard_equipment,
    has_level3_friend,
    increment_ability_counter,
    mark_random,
    mark_structural,
    rightmost_friendly_index,
)
from .events import AbilityEvent, clamp_level

AbilityHandler = Callable[[dict[str, Any], AbilityEvent, Any], None]
PENDING_FOOD_CTX_KEY = "_pending_food_ctx"
PARROT_COPY_META_KEY = "_parrot_copy"


def _level_value(level: int, level_values: tuple[int, int, int]) -> int:
    idx = clamp_level(level) - 1
    return int(level_values[idx])


def _parrot_copy_store(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta = state.setdefault("meta", {})
    store = meta.setdefault(PARROT_COPY_META_KEY, {})
    if not isinstance(store, dict):
        store = {}
        meta[PARROT_COPY_META_KEY] = store
    return store


def _payload_hurt_queue(event: AbilityEvent) -> dict[int, int]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    raw = payload.get("hurt_queue", {})
    return raw if isinstance(raw, dict) else {}


def _remove_status_effect(slot: dict[str, Any], effect_id: str) -> None:
    effect = str(effect_id)
    effects = [str(x) for x in slot.get("status_effects", []) if str(x) != effect]
    slot["status_effects"] = sorted(set(effects))
    if effect == "status-melon-armor" and str(slot.get("equipment_id", "")) == "food-melon":
        slot["equipment_id"] = None
    if effect == "status-coconut-shield" and str(slot.get("equipment_id", "")) == "food-coconut":
        slot["equipment_id"] = None


def _damage_after_mitigation(slot: dict[str, Any], incoming_damage: int) -> int:
    damage = max(0, int(incoming_damage))
    if damage <= 0:
        return 0
    effects = [str(x) for x in slot.get("status_effects", [])]
    if "status-coconut-shield" in effects:
        _remove_status_effect(slot, "status-coconut-shield")
        return 0
    if "status-melon-armor" in effects:
        _remove_status_effect(slot, "status-melon-armor")
        return 0
    if "status-garlic-armor" in effects:


        return min(damage, max(2, damage - 2))
    return damage


def _queue_damage(
    state: dict[str, Any],
    team_indices: list[int],
    damage: int,
    hurt_queue: dict[int, int],
) -> list[int]:
    applied: list[int] = []
    dmg = max(0, int(damage))
    if dmg <= 0:
        return applied
    team = state.get("team", [])
    for idx in team_indices:
        if idx < 0 or idx >= len(team):
            continue
        slot = team[idx]
        if slot.get("pet_id") is None:
            continue
        # Damage-phase mitigation (melon/coconut/garlic) resolves before hurt triggers.
        actual_damage = _damage_after_mitigation(slot, dmg)
        before_health = int(slot.get("health", 0))
        slot["health"] = max(0, before_health - actual_damage)
        if actual_damage > 0:
            hurt_queue[int(idx)] = int(hurt_queue.get(int(idx), 0)) + 1
            applied.append(int(idx))
    return applied


def _summon_queue_from_payload(event: AbilityEvent) -> list[dict[str, Any]]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    raw = payload.get("summon_queue", [])
    return raw if isinstance(raw, list) else []


def _queue_summon(
    event: AbilityEvent,
    ctx: Any,
    *,
    pet_id: str,
    attack: int,
    health: int,
    level: int = 1,
    count: int = 1,
    target_team_index: int | None = None,
    equipment_id: str | None = None,
    status_effects: list[str] | None = None,
    counter_refund: dict[str, int | str] | None = None,
) -> int:
    queue = _summon_queue_from_payload(event)
    if queue is None:
        return 0
    # A2.1: a summon writes team occupancy, `pet_id` and `level`.
    mark_structural(ctx)
    queued = 0
    lvl = clamp_level(level)
    exp = 0 if lvl == 1 else (2 if lvl == 2 else 5)
    for _ in range(max(1, int(count))):
        queue.append(
            {
                "pet_id": str(pet_id),
                "attack": int(attack),
                "health": int(health),
                "level": int(lvl),
                "exp": int(exp),
                "target_team_index": target_team_index,
                "equipment_id": (str(equipment_id) if equipment_id else None),
                "status_effects": [str(x) for x in (status_effects or [])],
                "counter_refund": (dict(counter_refund) if isinstance(counter_refund, dict) else None),
            }
        )
        queued += 1
    return queued


def _partition_shop_pet_food(shop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pets = [dict(s) for s in shop if s.get("slot_type") == "pet"]
    foods = [dict(s) for s in shop if s.get("slot_type") == "food"]
    others = [dict(s) for s in shop if s.get("slot_type") not in {"pet", "food"}]
    return pets + foods + others


def _reindex_shop(shop: list[dict[str, Any]]) -> None:
    for idx, slot in enumerate(shop):
        slot["shop_index"] = int(idx)


def _ability_insert_shop_item(
    state: dict[str, Any],
    *,
    slot_type: str,
    item_id: str,
    cost: int,
    frozen: bool,
    ctx: Any,
) -> bool:
    shop = _partition_shop_pet_food(state.get("shop", []))
    new_slot: dict[str, Any] = {
        "shop_index": -1,
        "slot_type": str(slot_type),
        "item_id": str(item_id),
        "cost": int(cost),
        "frozen": bool(frozen),
        "link_id": None,
    }
    if slot_type == "pet":
        stats = load_turtle_catalog().get("pets", {}).get("base_stats", {}).get(str(item_id), {"attack": 1, "health": 1})
        new_slot["attack"] = int(stats.get("attack", 1))
        new_slot["health"] = int(stats.get("health", 1))

    # If full, evict an existing non-frozen slot first.
    if len(shop) >= 9:
        removable_existing = [i for i, s in enumerate(shop) if not s.get("frozen")]
        if not removable_existing:
            ctx.notes.append("shop_insert_skipped:all_slots_frozen")
            state["shop"] = shop
            _reindex_shop(state["shop"])
            return False
        drop_idx = max(removable_existing) if slot_type == "pet" else min(removable_existing)
        dropped = shop.pop(int(drop_idx))
        ctx.notes.append(f"shop_overflow_drop:{dropped.get('slot_type')}:{dropped.get('item_id')}")

    insert_at = 0 if slot_type == "pet" else len(shop)
    shop.insert(insert_at, new_slot)

    state["shop"] = _partition_shop_pet_food(shop)
    _reindex_shop(state["shop"])
    # A2.1: adding (and possibly evicting) a shop slot writes the slot count and
    # the new slot's `slot_type`/`item_id`/`cost`/`frozen`. Marked HERE, on the
    # only path that writes, and not on entry: the `all_slots_frozen` path above
    # returns having written nothing `legal_actions` reads, and A2.1's rule is
    # write-based. The eviction is followed by this insert, so every real write
    # still reaches this line.
    mark_structural(ctx)
    return True


def _ability_buy_otter(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    target_count = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, target_count)
    # sapai-main behavior: +1 health only, number of targets scales by level.
    buff_team_slots(state, targets, attack=0, health=1)
    ctx.notes.append(f"ability_applied:pet-otter:buy:targets={targets}:buff=0/1")


def _ability_sell_beaver(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, 2)
    buff_team_slots(state, targets, attack=amount, health=0)
    ctx.notes.append(f"ability_applied:pet-beaver:sell:targets={targets}:buff={amount}/0")


def _ability_sell_duck(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    cat = load_turtle_catalog()
    base_stats = cat.get("pets", {}).get("base_stats", {})
    for slot in state.get("shop", []):
        if slot.get("slot_type") != "pet":
            continue
        item_id = str(slot.get("item_id", ""))
        if item_id in {"", "pet-none"}:
            continue
        if "attack" not in slot or "health" not in slot:
            base = base_stats.get(item_id, {"attack": 1, "health": 1})
            slot["attack"] = int(base.get("attack", 1))
            slot["health"] = int(base.get("health", 1))
    touched = buff_shop_pets(state, attack=0, health=amount)
    ctx.notes.append(f"ability_applied:pet-duck:sell:touched={touched}:buff=0/{amount}")


def _ability_sell_pig(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    bonus = _level_value(event.actor_level, (1, 2, 3))
    gain_gold(state, bonus, ctx=ctx)
    ctx.notes.append(f"ability_applied:pet-pig:sell:gold_plus={bonus}")


def _ability_friend_sold_shrimp(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, 1)
    buff_team_slots(state, targets, attack=0, health=amount)
    ctx.notes.append(f"ability_applied:pet-shrimp:friend_sold:targets={targets}:buff=0/{amount}")


def _ability_end_of_turn_snail(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    # Requires battle result context from caller.
    last_result = str(state.get("meta", {}).get("last_battle_result", "")).lower()
    if last_result != "loss":
        ctx.notes.append("ability_skipped:pet-snail:end_of_turn:last_battle_not_loss")
        return
    amount = _level_value(event.actor_level, (1, 2, 3))
    targets = friend_ahead_indices(state, event.actor_team_index)[:3]
    buff_team_slots(state, targets, attack=amount, health=0)
    ctx.notes.append(f"ability_applied:pet-snail:end_of_turn:targets={targets}:buff={amount}/0")


def _ability_levelup_fish(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    ability_level = int(event.payload.get("ability_level", event.actor_level))
    # sapai-main uses the previous level's fish ability on level-up.
    amount = _level_value(ability_level, (1, 2, 2))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, 2)
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-fish:level_up:targets={targets}:buff={amount}/{amount}:ability_level={ability_level}")


def _ability_faint_ant(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, 1)
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-ant:faint:targets={targets}:buff={amount}/{amount}")


def _ability_faint_flamingo(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    targets = friend_behind_indices(state, event.actor_team_index)[:2]
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-flamingo:faint:targets={targets}:buff={amount}/{amount}")


def _ability_faint_hedgehog(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    damage = _level_value(event.actor_level, (2, 4, 6))
    hurt_queue = _payload_hurt_queue(event)
    targets = [idx for idx, slot in enumerate(state.get("team", [])) if slot.get("pet_id") is not None]
    applied = _queue_damage(state, targets, damage, hurt_queue)
    ctx.notes.append(f"ability_applied:pet-hedgehog:faint:targets={applied}:damage={damage}")


def _ability_faint_badger(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    actor_idx = int(actor_idx)
    team = state.get("team", [])
    if actor_idx < 0 or actor_idx >= len(team):
        return
    actor_slot = team[actor_idx]
    attack = int(actor_slot.get("attack", 0))
    mult = _level_value(event.actor_level, (50, 100, 150))
    damage = int((attack * mult) / 100)
    hurt_queue = _payload_hurt_queue(event)
    targets = [i for i in [actor_idx - 1, actor_idx + 1] if 0 <= i < len(team) and team[i].get("pet_id") is not None]
    applied = _queue_damage(state, targets, damage, hurt_queue)
    ctx.notes.append(f"ability_applied:pet-badger:faint:targets={applied}:damage={damage}")


def _ability_faint_mammoth(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (2, 4, 6))
    targets = friend_indices(state, event.actor_team_index)
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-mammoth:faint:targets={targets}:buff={amount}/{amount}")


def _ability_faint_turtle(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    n_targets = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_behind_indices(state, event.actor_team_index)
    targets: list[int] = []
    for idx in candidates:
        slot = state["team"][idx]
        if has_standard_equipment(slot, "food-melon"):
            continue
        if not grant_standard_perk(state, idx, "food-melon", ctx=ctx):
            continue
        targets.append(idx)
        if len(targets) >= n_targets:
            break
    ctx.notes.append(f"ability_applied:pet-turtle:faint:targets={targets}:status=status-melon-armor")


def _ability_after_faint_cricket(_state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    atk = _level_value(event.actor_level, (1, 2, 3))
    hp = _level_value(event.actor_level, (1, 2, 3))
    payload = event.payload if isinstance(event.payload, dict) else {}
    target_idx = payload.get("fainted_team_index")
    queued = _queue_summon(
        event,
        ctx,
        pet_id="pet-zombie-cricket",
        attack=atk,
        health=hp,
        level=event.actor_level,
        count=1,
        target_team_index=(int(target_idx) if target_idx is not None else None),
    )
    ctx.notes.append(f"ability_applied:pet-cricket:after_faint:queued={queued}:summon=pet-zombie-cricket:{atk}/{hp}")


def _ability_after_faint_sheep(_state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    atk = _level_value(event.actor_level, (2, 4, 6))
    hp = _level_value(event.actor_level, (2, 4, 6))
    payload = event.payload if isinstance(event.payload, dict) else {}
    target_idx = payload.get("fainted_team_index")
    queued = _queue_summon(
        event,
        ctx,
        pet_id="pet-ram",
        attack=atk,
        health=hp,
        level=event.actor_level,
        count=2,
        target_team_index=(int(target_idx) if target_idx is not None else None),
    )
    ctx.notes.append(f"ability_applied:pet-sheep:after_faint:queued={queued}:summon=pet-ram:{atk}/{hp}")


def _ability_after_faint_deer(_state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    atk = _level_value(event.actor_level, (5, 10, 15))
    hp = _level_value(event.actor_level, (3, 6, 9))
    payload = event.payload if isinstance(event.payload, dict) else {}
    target_idx = payload.get("fainted_team_index")
    queued = _queue_summon(
        event,
        ctx,
        pet_id="pet-bus",
        attack=atk,
        health=hp,
        level=event.actor_level,
        count=1,
        target_team_index=(int(target_idx) if target_idx is not None else None),
        equipment_id="food-chili",
    )
    ctx.notes.append(f"ability_applied:pet-deer:after_faint:queued={queued}:summon=pet-bus:{atk}/{hp}:with_chili")


def _ability_after_faint_rooster(_state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    payload = event.payload if isinstance(event.payload, dict) else {}
    target_idx = payload.get("fainted_team_index")
    source_attack = int(payload.get("actor_attack", 0))
    chick_attack = max(1, int(source_attack * 0.5))
    count = _level_value(event.actor_level, (1, 2, 3))
    queued = _queue_summon(
        event,
        ctx,
        pet_id="pet-chick",
        attack=chick_attack,
        health=1,
        level=event.actor_level,
        count=count,
        target_team_index=(int(target_idx) if target_idx is not None else None),
    )
    ctx.notes.append(f"ability_applied:pet-rooster:after_faint:queued={queued}:summon=pet-chick:{chick_attack}/1")


def _ability_after_faint_spider(_state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    cat = load_turtle_catalog()
    pool = list(cat.get("pets", {}).get("by_tier", {}).get("3", []))
    if not pool:
        ctx.notes.append("ability_skipped:pet-spider:after_faint:no_tier3_pool")
        return
    chosen = str(ctx.rng.choice(pool))
    # A4: this draw decides WHICH pet gets summoned and the summon below writes
    # team occupancy/`pet_id`/`level`. Spider is the one place in the Turtle
    # pack where ability randomness reaches `legal_actions`'s read-set, so it
    # is the one place ability randomness still cuts.
    mark_random(ctx)
    payload = event.payload if isinstance(event.payload, dict) else {}
    target_idx = payload.get("fainted_team_index")
    queued = _queue_summon(
        event,
        ctx,
        pet_id=chosen,
        attack=_level_value(event.actor_level, (2, 4, 6)),
        health=_level_value(event.actor_level, (2, 4, 6)),
        level=event.actor_level,
        count=1,
        target_team_index=(int(target_idx) if target_idx is not None else None),
    )
    spider_stat = _level_value(event.actor_level, (2, 4, 6))
    ctx.notes.append(
        f"ability_applied:pet-spider:after_faint:queued={queued}:summon={chosen}:{spider_stat}/{spider_stat}:level={clamp_level(event.actor_level)}"
    )


def _ability_after_faint_rat(_state: dict[str, Any], _event: AbilityEvent, ctx: Any) -> None:
    # Shop phase has no enemy team to summon onto.
    ctx.notes.append("ability_skipped:pet-rat:after_faint:no_enemy_team")


def _ability_friend_ahead_faints_ox(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    actor_idx = int(actor_idx)
    payload = event.payload if isinstance(event.payload, dict) else {}
    fainted_idx = payload.get("fainted_team_index")
    if fainted_idx is None:
        return
    try:
        fainted_idx = int(fainted_idx)
    except (TypeError, ValueError):
        return
    if fainted_idx != actor_idx - 1:
        return
    max_triggers = _level_value(event.actor_level, (1, 2, 3))
    current = get_ability_counter(state, "friend_ahead_faints", actor_idx)
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-ox:friend_ahead_faints:max_triggers={max_triggers}")
        return
    increment_ability_counter(state, "friend_ahead_faints", actor_idx, amount=1)
    slot = state["team"][actor_idx]
    grant_standard_perk(state, actor_idx, "food-melon", ctx=ctx)
    buff_team_slots(state, [actor_idx], attack=1, health=0)
    ctx.notes.append(
        f"ability_applied:pet-ox:friend_ahead_faints:self_buff=1/0:status=status-melon-armor:count={current + 1}/{max_triggers}"
    )


def _ability_friend_faints_shark(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    payload = event.payload if isinstance(event.payload, dict) else {}
    fainted_idx = payload.get("fainted_team_index")
    if fainted_idx is not None and int(fainted_idx) == int(actor_idx):
        return
    amount = _level_value(event.actor_level, (2, 4, 6))
    buff_team_slots(state, [int(actor_idx)], attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-shark:friend_faints:self_buff={amount}/{amount}")


def _ability_friend_faints_fly(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    payload = event.payload if isinstance(event.payload, dict) else {}
    fainted_idx = payload.get("fainted_team_index")
    fainted_pet_id = str(payload.get("fainted_pet_id", ""))
    if fainted_idx is not None and int(fainted_idx) == int(actor_idx):
        return
    if fainted_pet_id == "pet-zombie-fly":
        ctx.notes.append("ability_skipped:pet-fly:friend_faints:ignored_zombie_fly")
        return
    max_triggers = 3
    current = get_ability_counter(state, "friend_faints", int(actor_idx))
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-fly:friend_faints:max_triggers={max_triggers}")
        return
    increment_ability_counter(state, "friend_faints", int(actor_idx), amount=1)
    stats = _level_value(event.actor_level, (4, 8, 12))
    queued = _queue_summon(
        event,
        ctx,
        pet_id="pet-zombie-fly",
        attack=stats,
        health=stats,
        level=event.actor_level,
        count=1,
        target_team_index=(int(fainted_idx) if fainted_idx is not None else None),
        counter_refund={"trigger": "friend_faints", "team_index": int(actor_idx), "amount": 1},
    )
    ctx.notes.append(
        f"ability_applied:pet-fly:friend_faints:queued={queued}:summon=pet-zombie-fly:{stats}/{stats}:count={current + 1}/{max_triggers}"
    )


def _ability_hurt_peacock(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    amount = _level_value(event.actor_level, (3, 6, 9))
    buff_team_slots(state, [int(actor_idx)], attack=amount, health=0)
    ctx.notes.append(f"ability_applied:pet-peacock:hurt:self_buff={amount}/0")


def _ability_hurt_camel(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (2, 4, 6))
    targets = friend_behind_indices(state, event.actor_team_index)[:1]
    buff_team_slots(state, targets, attack=amount, health=(amount * 2))
    ctx.notes.append(f"ability_applied:pet-camel:hurt:targets={targets}:buff={amount}/{amount * 2}")


def _ability_hurt_blowfish(_state: dict[str, Any], _event: AbilityEvent, ctx: Any) -> None:
    # Shop phase has no enemy team; this trigger is a no-op here.
    ctx.notes.append("ability_skipped:pet-blowfish:hurt:no_enemy_team")


def _ability_hurt_gorilla(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    actor_idx = int(actor_idx)
    max_triggers = _level_value(event.actor_level, (1, 2, 3))
    current = get_ability_counter(state, "hurt", actor_idx)
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-gorilla:hurt:max_triggers={max_triggers}")
        return
    increment_ability_counter(state, "hurt", actor_idx, amount=1)
    slot = state["team"][actor_idx]
    grant_standard_perk(state, actor_idx, "food-coconut", ctx=ctx)
    ctx.notes.append(
        f"ability_applied:pet-gorilla:hurt:status=status-coconut-shield:count={current + 1}/{max_triggers}"
    )


def _ability_friends_hurt_counter_wolverine(_state: dict[str, Any], _event: AbilityEvent, ctx: Any) -> None:
    # Wolverine targets enemies only; shop phase has no enemy team.
    ctx.notes.append("ability_skipped:pet-wolverine:friends_hurt_counter:no_enemy_team")


def _ability_friendly_ate_food_rabbit(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    max_triggers = 3
    current = get_ability_counter(state, "friendly_ate_food", actor_idx)
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-rabbit:friendly_ate_food:max_triggers={max_triggers}")
        return
    amount = _level_value(event.actor_level, (1, 2, 3))
    trigger_idx = event.payload.get("trigger_team_index")
    if trigger_idx is None:
        return
    try:
        trigger_idx = int(trigger_idx)
    except (TypeError, ValueError):
        return
    if trigger_idx < 0 or trigger_idx >= len(state.get("team", [])):
        return
    if state["team"][trigger_idx].get("pet_id") is None:
        return
    increment_ability_counter(state, "friendly_ate_food", actor_idx, amount=1)
    buff_team_slots(state, [trigger_idx], attack=0, health=amount)
    ctx.notes.append(
        f"ability_applied:pet-rabbit:friendly_ate_food:target={trigger_idx}:buff=0/{amount}:count={current + 1}/{max_triggers}"
    )


def _ability_start_of_turn_swan(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    gain_gold(state, amount, ctx=ctx)
    ctx.notes.append(f"ability_applied:pet-swan:start_of_turn:gold_plus={amount}")


def _ability_start_of_turn_squirrel(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    touched = discount_shop_foods(state, amount=amount, ctx=ctx)
    ctx.notes.append(f"ability_applied:pet-squirrel:start_of_turn:food_discount={amount}:touched={touched}")


def _ability_start_of_turn_giraffe(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    n_targets = _level_value(event.actor_level, (1, 2, 3))
    targets = friend_ahead_indices(state, event.actor_team_index)[:n_targets]
    buff_team_slots(state, targets, attack=1, health=1)
    ctx.notes.append(f"ability_applied:pet-giraffe:start_of_turn:targets={targets}:buff=1/1")


def _ability_start_of_turn_penguin(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    candidates = [
        idx
        for idx in friend_indices(state, event.actor_team_index)
        if int(state["team"][idx].get("level", 1)) >= 2
    ]
    targets = choose_random_indices(ctx, candidates, 2)
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-penguin:start_of_turn:targets={targets}:buff={amount}/{amount}")


def _stock_cow_milk(
    state: dict[str, Any],
    ctx: Any,
    *,
    milk_item: str,
    milk_atk: int,
    milk_hp: int,
    note_prefix: str,
) -> None:
    """Replace the shop's food slots with 2x free (Chocolate) Milk carrying a
    custom stat payload. Shared by the Cow's buy ability and its chocolate
    easter egg; only the milk item/stats and the note prefix differ."""
    # A2.1: the shop's food slots are REMOVED here, before any insert is
    # attempted, so this is structural even when every insert is refused.
    mark_structural(ctx)
    before_shop = list(state.get("shop", []))
    replaced = sum(1 for slot in before_shop if slot.get("slot_type") == "food")
    kept = [dict(slot) for slot in before_shop if slot.get("slot_type") != "food"]
    state["shop"] = _partition_shop_pet_food(kept)
    _reindex_shop(state["shop"])

    stocked = 0
    for _ in range(2):
        if _ability_insert_shop_item(
            state,
            slot_type="food",
            item_id=milk_item,
            cost=0,
            frozen=False,
            ctx=ctx,
        ):
            # Maintain custom Milk stat payload on shop slot.
            for slot in reversed(state.get("shop", [])):
                if slot.get("slot_type") == "food" and slot.get("item_id") == milk_item and int(slot.get("cost", 0)) == 0:
                    slot["attack"] = int(milk_atk)
                    slot["health"] = int(milk_hp)
                    break
            stocked += 1

    ctx.notes.append(
        f"{note_prefix}:food_slots_replaced={replaced}:milk_count={stocked}:item={milk_item}"
    )


def _ability_buy_cow(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    lvl = clamp_level(event.actor_level)
    milk_item = {1: "food-milk", 2: "food-milk2", 3: "food-milk3"}[lvl]
    milk_atk, milk_hp = FOOD_STAT_BUFFS.get(milk_item, (1, 2))
    _stock_cow_milk(
        state,
        ctx,
        milk_item=milk_item,
        milk_atk=milk_atk,
        milk_hp=milk_hp,
        note_prefix="ability_applied:pet-cow:buy",
    )


def _ability_eats_food_cow(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    # Easter egg: feeding a Cow a Chocolate ALSO stocks 2x free Chocolate Milk
    # (attack-only, scaled by the Cow's level), replacing the food shop like the
    # buy ability. The Chocolate's normal +1 exp still applies (the engine emits
    # this eats_food trigger AFTER that exp gain), so event.actor_level already
    # reflects any level-up the Chocolate caused.
    payload = event.payload if isinstance(event.payload, dict) else {}
    if str(payload.get("food_item_id", "")) != "food-chocolate":
        return
    lvl = clamp_level(event.actor_level)
    milk_item = {1: "food-chocolate-milk", 2: "food-chocolate-milk2", 3: "food-chocolate-milk3"}[lvl]
    milk_atk, milk_hp = FOOD_STAT_BUFFS.get(milk_item, (lvl, 0))
    _stock_cow_milk(
        state,
        ctx,
        milk_item=milk_item,
        milk_atk=milk_atk,
        milk_hp=milk_hp,
        note_prefix="ability_applied:pet-cow:eats_chocolate",
    )


def _ability_purchase_food_cat(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    max_triggers = 2
    current = get_ability_counter(state, "purchase_food", actor_idx)
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-cat:purchase_food:max_triggers={max_triggers}")
        return

    meta = state.setdefault("meta", {})
    pending = meta.get(PENDING_FOOD_CTX_KEY)
    if not isinstance(pending, dict):
        return

    mult = _level_value(event.actor_level, (2, 3, 4))


    base_attack = int(pending.get("cat_base_attack", pending.get("attack", pending.get("base_attack", 0))))
    base_health = int(pending.get("cat_base_health", pending.get("health", pending.get("base_health", 0))))
    if base_attack == 0 and base_health == 0:
        ctx.notes.append("ability_skipped:pet-cat:purchase_food:no_stats")
        return
    pending["cat_base_attack"] = base_attack
    pending["cat_base_health"] = base_health
    # Cat stacking is additive on base food effect:
    # each Cat contributes +(mult-1)*base, so two lvl1 Cats => x3 total, not x4/x6.
    pending["attack"] = int(pending.get("attack", 0)) + int(base_attack) * max(0, mult - 1)
    pending["health"] = int(pending.get("health", 0)) + int(base_health) * max(0, mult - 1)
    increment_ability_counter(state, "purchase_food", actor_idx, amount=1)
    ctx.notes.append(
        f"ability_applied:pet-cat:purchase_food:additive_mult={mult}:count={current + 1}/{max_triggers}"
    )


def _ability_eats_food_seal(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    amount = _level_value(event.actor_level, (1, 2, 3))
    candidates = friend_indices(state, event.actor_team_index)
    targets = choose_random_indices(ctx, candidates, 3)
    buff_team_slots(state, targets, attack=amount, health=0)
    ctx.notes.append(f"ability_applied:pet-seal:eats_food:targets={targets}:buff={amount}/0")


def _ability_start_of_turn_worm(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    item_id = {1: "food-apple", 2: "food-apple2", 3: "food-apple3"}[clamp_level(event.actor_level)]
    ok = _ability_insert_shop_item(
        state,
        slot_type="food",
        item_id=item_id,
        cost=2,
        frozen=False,
        ctx=ctx,
    )
    if ok:
        ctx.notes.append(f"ability_applied:pet-worm:start_of_turn:stocked={item_id}:cost=2")
    else:
        ctx.notes.append(f"ability_skipped:pet-worm:start_of_turn:stock_failed:{item_id}")


def _ability_sell_pigeon(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    count = _level_value(event.actor_level, (1, 2, 3))
    stocked = 0
    for _ in range(count):
        if _ability_insert_shop_item(
            state,
            slot_type="food",
            item_id="food-bread-crumbs",
            cost=0,
            frozen=False,
            ctx=ctx,
        ):
            stocked += 1
    ctx.notes.append(f"ability_applied:pet-pigeon:sell:stocked={stocked}/food-bread-crumbs")


def _ability_summoned_scorpion(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    actor_idx = int(actor_idx)
    if actor_idx < 0 or actor_idx >= len(state.get("team", [])):
        return
    slot = state["team"][actor_idx]
    if str(slot.get("pet_id", "")) != "pet-scorpion":
        return
    grant_standard_perk(state, actor_idx, "food-peanut", ctx=ctx)
    ctx.notes.append(f"ability_applied:pet-scorpion:summoned:status=status-peanut:team={actor_idx}")


def _ability_friend_summoned_dog(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    trigger_idx = event.payload.get("trigger_team_index")
    if actor_idx is None or trigger_idx is None:
        return
    try:
        actor_idx = int(actor_idx)
        trigger_idx = int(trigger_idx)
    except (TypeError, ValueError):
        return
    if actor_idx == trigger_idx:
        return
    atk = _level_value(event.actor_level, (2, 4, 6))
    hp = _level_value(event.actor_level, (1, 2, 3))
    buff_team_slots_temporary(state, [actor_idx], attack=atk, health=hp)
    ctx.notes.append(f"ability_applied:pet-dog:friend_summoned:self_temp_buff={atk}/{hp}:trigger={trigger_idx}")


def _ability_friend_summoned_horse(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    trigger_idx = event.payload.get("trigger_team_index")
    if actor_idx is None or trigger_idx is None:
        return
    try:
        actor_idx = int(actor_idx)
        trigger_idx = int(trigger_idx)
    except (TypeError, ValueError):
        return
    if actor_idx == trigger_idx:
        return
    if trigger_idx < 0 or trigger_idx >= len(state.get("team", [])):
        return
    if state["team"][trigger_idx].get("pet_id") is None:
        return
    atk = _level_value(event.actor_level, (1, 2, 3))
    buff_team_slots_temporary(state, [trigger_idx], attack=atk, health=0)
    ctx.notes.append(f"ability_applied:pet-horse:friend_summoned:target={trigger_idx}:temp_buff={atk}/0")


def _ability_friend_summoned_turkey(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    trigger_idx = event.payload.get("trigger_team_index")
    if actor_idx is None or trigger_idx is None:
        return
    try:
        actor_idx = int(actor_idx)
        trigger_idx = int(trigger_idx)
    except (TypeError, ValueError):
        return
    if actor_idx == trigger_idx:
        return
    if trigger_idx < 0 or trigger_idx >= len(state.get("team", [])):
        return
    if state["team"][trigger_idx].get("pet_id") is None:
        return
    atk = _level_value(event.actor_level, (3, 6, 9))
    hp = _level_value(event.actor_level, (1, 2, 3))
    buff_team_slots(state, [trigger_idx], attack=atk, health=hp)
    ctx.notes.append(f"ability_applied:pet-turkey:friend_summoned:target={trigger_idx}:buff={atk}/{hp}")


def _ability_buy_tier1_dragon(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    trigger_tier = int(event.payload.get("trigger_pet_tier", 0))
    if trigger_tier != 1:
        return
    actor_idx = event.actor_team_index
    max_triggers = 4
    current = get_ability_counter(state, "buy_tier1_pet", actor_idx)
    if current >= max_triggers:
        ctx.notes.append(f"ability_skipped:pet-dragon:buy_tier1_pet:max_triggers={max_triggers}")
        return
    amount = _level_value(event.actor_level, (1, 2, 3))
    targets = friend_indices(state, actor_idx)
    increment_ability_counter(state, "buy_tier1_pet", actor_idx, amount=1)
    buff_team_slots(state, targets, attack=amount, health=amount)
    ctx.notes.append(
        f"ability_applied:pet-dragon:buy_tier1_pet:targets={targets}:buff={amount}/{amount}:count={current + 1}/{max_triggers}"
    )


def _ability_end_of_turn_monkey(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    target = rightmost_friendly_index(state)
    if target is None:
        ctx.notes.append("ability_skipped:pet-monkey:end_of_turn:no_friend_target")
        return
    amount = _level_value(event.actor_level, (2, 4, 6))
    buff_team_slots(state, [target], attack=amount, health=amount)
    ctx.notes.append(f"ability_applied:pet-monkey:end_of_turn:target={target}:buff={amount}/{amount}")


def _ability_end_of_turn_bison(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    if not has_level3_friend(state, event.actor_team_index):
        ctx.notes.append("ability_skipped:pet-bison:end_of_turn:requires_level3_friend")
        return
    if get_ability_counter(state, "bison_team_end_of_turn", -1) >= 1:
        ctx.notes.append("ability_skipped:pet-bison:end_of_turn:team_cap_reached")
        return
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    increment_ability_counter(state, "bison_team_end_of_turn", -1, amount=1)
    atk = _level_value(event.actor_level, (2, 4, 6))
    hp = _level_value(event.actor_level, (2, 4, 6))
    buff_team_slots(state, [int(actor_idx)], attack=atk, health=hp)
    ctx.notes.append(f"ability_applied:pet-bison:end_of_turn:self_buff={atk}/{hp}")


def _ability_end_of_turn_parrot(state: dict[str, Any], event: AbilityEvent, ctx: Any) -> None:
    actor_idx = event.actor_team_index
    if actor_idx is None:
        return
    actor_idx = int(actor_idx)
    store = _parrot_copy_store(state)
    ahead = friend_ahead_indices(state, actor_idx)
    if not ahead:
        store.pop(str(actor_idx), None)
        ctx.notes.append("ability_skipped:pet-parrot:end_of_turn:no_friend_ahead")
        return

    source_idx = int(ahead[0])
    source_slot = state["team"][source_idx]
    source_pet_id = source_slot.get("pet_id")
    if source_pet_id is None:
        store.pop(str(actor_idx), None)
        return
    source_pet_id = str(source_pet_id)
    if source_pet_id == "pet-parrot":
        source_override = store.get(str(source_idx), {})
        if isinstance(source_override, dict):
            source_pet_id = str(source_override.get("pet_id", source_pet_id))

    if source_pet_id == "pet-parrot":
        store.pop(str(actor_idx), None)
        ctx.notes.append("ability_skipped:pet-parrot:end_of_turn:parrot_loop")
        return

    store[str(actor_idx)] = {
        "pet_id": source_pet_id,
        "level": clamp_level(event.actor_level),
    }
    ctx.notes.append(
        f"ability_applied:pet-parrot:end_of_turn:copied={source_pet_id}:source={source_idx}:level={clamp_level(event.actor_level)}"
    )


TURTLE_ABILITY_HANDLERS: dict[tuple[str, str], AbilityHandler] = {
    ("pet-ant", "faint"): _ability_faint_ant,
    ("pet-cricket", "after_faint"): _ability_after_faint_cricket,
    ("pet-otter", "buy"): _ability_buy_otter,
    ("pet-flamingo", "faint"): _ability_faint_flamingo,
    ("pet-hedgehog", "faint"): _ability_faint_hedgehog,
    ("pet-rat", "after_faint"): _ability_after_faint_rat,
    ("pet-spider", "after_faint"): _ability_after_faint_spider,
    ("pet-badger", "faint"): _ability_faint_badger,
    ("pet-mammoth", "faint"): _ability_faint_mammoth,
    ("pet-turtle", "faint"): _ability_faint_turtle,
    ("pet-ox", "friend_ahead_faints"): _ability_friend_ahead_faints_ox,
    ("pet-sheep", "after_faint"): _ability_after_faint_sheep,
    ("pet-deer", "after_faint"): _ability_after_faint_deer,
    ("pet-rooster", "after_faint"): _ability_after_faint_rooster,
    ("pet-shark", "friend_faints"): _ability_friend_faints_shark,
    ("pet-fly", "friend_faints"): _ability_friend_faints_fly,
    ("pet-beaver", "sell"): _ability_sell_beaver,
    ("pet-duck", "sell"): _ability_sell_duck,
    ("pet-pig", "sell"): _ability_sell_pig,
    ("pet-pigeon", "sell"): _ability_sell_pigeon,
    ("pet-shrimp", "friend_sold"): _ability_friend_sold_shrimp,
    ("pet-snail", "end_of_turn"): _ability_end_of_turn_snail,
    ("pet-fish", "level_up"): _ability_levelup_fish,
    ("pet-rabbit", "friendly_ate_food"): _ability_friendly_ate_food_rabbit,
    ("pet-giraffe", "start_of_turn"): _ability_start_of_turn_giraffe,
    ("pet-penguin", "start_of_turn"): _ability_start_of_turn_penguin,
    ("pet-swan", "start_of_turn"): _ability_start_of_turn_swan,
    ("pet-squirrel", "start_of_turn"): _ability_start_of_turn_squirrel,
    ("pet-worm", "start_of_turn"): _ability_start_of_turn_worm,
    ("pet-cow", "buy"): _ability_buy_cow,
    ("pet-cow", "eats_food"): _ability_eats_food_cow,
    ("pet-peacock", "hurt"): _ability_hurt_peacock,
    ("pet-camel", "hurt"): _ability_hurt_camel,
    ("pet-blowfish", "hurt"): _ability_hurt_blowfish,
    ("pet-gorilla", "hurt"): _ability_hurt_gorilla,
    ("pet-wolverine", "friends_hurt_counter"): _ability_friends_hurt_counter_wolverine,
    ("pet-seal", "eats_food"): _ability_eats_food_seal,
    ("pet-scorpion", "summoned"): _ability_summoned_scorpion,
    ("pet-dog", "friend_summoned"): _ability_friend_summoned_dog,
    ("pet-horse", "friend_summoned"): _ability_friend_summoned_horse,
    ("pet-turkey", "friend_summoned"): _ability_friend_summoned_turkey,
    ("pet-cat", "purchase_food"): _ability_purchase_food_cat,
    ("pet-dragon", "buy_tier1_pet"): _ability_buy_tier1_dragon,
    ("pet-monkey", "end_of_turn"): _ability_end_of_turn_monkey,
    ("pet-bison", "end_of_turn"): _ability_end_of_turn_bison,
}
