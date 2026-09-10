"""Baseline non-combat shop transition kernel for Step 1."""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass
from typing import Any

from .action_keys import hashable_action_key
from .ability import (
    AbilityEvent,
    AbilityRuntime,
    TRIGGER_AFTER_FAINT,
    TRIGGER_BUY,
    TRIGGER_BUY_TIER1_PET,
    TRIGGER_BUY_FRIEND,
    TRIGGER_END_OF_TURN,
    TRIGGER_EATS_FOOD,
    TRIGGER_FAINT,
    TRIGGER_FRIEND_AHEAD_FAINTS,
    TRIGGER_FRIEND_FAINTS,
    TRIGGER_FRIENDS_HURT_COUNTER,
    TRIGGER_FRIENDLY_ATE_FOOD,
    TRIGGER_FRIEND_SOLD,
    TRIGGER_FRIEND_SUMMONED,
    TRIGGER_HURT,
    TRIGGER_LEVEL_UP,
    TRIGGER_PURCHASE_FOOD,
    TRIGGER_SELL,
    TRIGGER_START_OF_TURN,
    TRIGGER_SUMMONED,
)
from .ability.effects import (
    add_slot_stats,
    apply_standard_equipment,
    build_food_trigger_event_groups,
    clear_temporary_team_stats,
    clear_ability_counters_for_team_index,
    decrement_ability_counter,
    effective_actor_for_trigger,
    ensure_stat_fields,
    get_ability_counter,
    increment_ability_counter,
    remap_ability_counters_for_reorder,
    reset_ability_counters,
)
from .ability.queue import sort_ability_events_for_runtime
from .catalog import load_turtle_catalog, tier_for_turn
from .constants import (
    DEFAULT_SHOP_ITEM_COST,
    EQUIPMENT_FOOD_IDS,
    EQUIPMENT_STATUS_BY_FOOD_ID,
    FOOD_STAT_BUFFS,
    NO_TARGET_FOODS,
    NON_ROLLABLE_FOOD_IDS,
    NON_ROLLABLE_PET_IDS,
    RANDOM_TARGET_FOOD_COUNTS,
    RANDOM_TARGET_FOODS,
    SLOTH_PET_ID,
    SLOTH_ROLL_PROB,
    expected_shop_counts,
    shop_item_cost,
)


@dataclass
class StepOutcome:
    """One engine transition's result.

    `stochastic_structural` (exp13 PLAN Amendments A2.1 and A4, the
    structural-cut rule) is True when this step made a write that RANDOMNESS
    STEERED, into a field `legal_actions` reads. That read-set is exactly shop
    slot `slot_type`/`item_id`/`cost`/`frozen` and the slot count, `gold`, and
    team occupancy/`pet_id`/`level` -- so a resolution that only moved attack,
    health, temporary stats, `sell_value`, `exp`, equipment or status effects,
    or that only chose an ORDERING among events, leaves this False even though
    `deterministic` is False.

    A4 REVISION 2 (2026-08-06, the W1a-3 independent review, findings F1 and
    F2) moved that causal taint from the ability runtime up to the STEP. The
    first cut of A4 raised it inside `AbilityRuntimeContext`, which dies when
    its `AbilityRuntime` does -- and the read-set writes that matter most do
    not happen inside an ability runtime at all. A faint CLEARS a slot from
    `_resolve_shop_hurt_faint_chain`, one or more runtimes after the draw that
    decided who would faint. Two live regressions followed, both reproduced
    against `main`:

    - Ant's faint buffs ONE RANDOM friend through `choose_random_indices`. The
      buff writes health, which `legal_actions` never reads -- but health
      reaching zero is exactly what REMOVES a pet, and removal writes team
      occupancy, which it does read. The dice chose who lived.
    - Two pets fainting in the same tie group, same trigger priority and same
      attack, draw for their order, and the order decides who grants Melon
      before whose damage lands. Again the dice chose who lived.

    So the rule is now: a draw ARMS the step (`StepCausality.arm`), and every
    read-set write from that point on is a cut (`note_structural`). It is
    sticky and it outlives every runtime, which is the only way the taint can
    reach the place the write happens. It over-approximates -- a write the dice
    provably could not have steered still counts once anything has drawn -- and
    that is deliberate: over-cutting wastes compute, under-cutting lets the
    search commit blind through a board it did not predict.

    Writes that happen BEFORE the step first draws are still not cuts, and that
    is what remains of A4 beyond what A2.6 already saves.

    A4 made the two halves CAUSAL rather than merely co-occurring. A2.1 read
    "randomness was consumed AND the read-set was written", each checked over
    the whole step and neither asking whether they were the same event. So a
    sleeping pill that kills a friend and the summon that answers it -- a
    deterministic read-set write the search already planned around -- turned
    into a cut as soon as anything else in that step drew, an event-order tie
    or a stat buff picking its targets. Measured on the post-merge re-pin that
    was 1,218 of 30,015 searched segments across 697 of 1,000 games
    (`RESULTS_W1a2.md` section 9). Stopping there buys nothing: the imagined
    board and the real one agree on exactly the field that moved. The engine
    now reports a cut only when the dice chose part of what got written.

    It is what the honest frame's two cut points filter on
    (`tools/eval_versus_fullgame.py`'s commit loop and
    `tools/search_recommender.py::_prefix_walk`), so the search cut and the
    execution cut stay identical by construction. Detection stays
    ENGINE-REPORTED, never an action-type or pet whitelist (A1 ruling 2): the
    flag is raised by the code that does the writing, so a future random effect
    that grants gold or discounts the shop becomes a cut point the day it lands.
    """

    state_after: dict[str, Any]
    legal: bool
    deterministic: bool
    stochastic_reason: str | None
    notes: list[str]
    stochastic_structural: bool = False


@dataclass
class StepCausality:
    """One step's causal taint, shared by every resolution inside that step.

    exp13 PLAN Amendment A4 as revised after the W1a-3 review (findings F1 and
    F2). See `StepOutcome` for why this had to leave `AbilityRuntimeContext`.

    Three verbs, and every site that draws or writes uses exactly one:

    - `arm()` -- a draw happened whose outcome can steer what this step goes on
      to write. Every draw the engine still makes arms, with one exception.
    - `note_structural()` -- this step wrote a field `legal_actions` reads:
      shop slot `slot_type`/`item_id`/`cost`/`frozen` and the slot count,
      `gold`, team occupancy/`pet_id`/`level`. A cut if already armed.
    - `force_cut()` -- the conservative default for `unsupported_effect`. An
      unmodelled effect could have written anything, for any reason.

    The exception: `_reseed_meta` writes only `meta.seed`, which is not in the
    read-set and no decision can read, so it neither arms nor notes.

    `structural_seen` is telemetry: "did this step write the read-set at all",
    independent of whether the dice steered it.
    """

    armed: bool = False
    structural_seen: bool = False
    cut: bool = False

    def arm(self) -> None:
        self.armed = True

    def note_structural(self) -> None:
        self.structural_seen = True
        if self.armed:
            self.cut = True

    def force_cut(self) -> None:
        self.armed = True
        self.structural_seen = True
        self.cut = True

    def absorb(self, outcome: "StepOutcome") -> None:
        """Merge a sub-outcome (END_TURN's pre- and post-battle halves)."""
        if outcome.stochastic_structural:
            self.cut = True


# The shop-side hurt/faint cascade runaway guard. A module constant rather than
# a local so a test can lower it and drive the `unsupported_effect` branch the
# W1a-3 review found unreachable and unguarded (its mutant M5).
SHOP_FAINT_CHAIN_MAX_STEPS = 256

PENDING_FOOD_CTX_KEY = "_pending_food_ctx"
PARROT_COPY_META_KEY = "_parrot_copy"
SHOP_PET_BONUS_AT_KEY = "_shop_pet_bonus_at"
SHOP_PET_BONUS_HP_KEY = "_shop_pet_bonus_hp"


def _team_slot(state: dict[str, Any], idx: int) -> dict[str, Any]:
    return state["team"][idx]


def _reconcile_team_stat_fields(state: dict[str, Any]) -> None:
    for slot in state.get("team", []):
        ensure_stat_fields(slot)


def _shop_pet_bonus(state: dict[str, Any]) -> tuple[int, int]:
    meta = state.setdefault("meta", {})
    try:
        at = int(meta.get(SHOP_PET_BONUS_AT_KEY, 0))
    except (TypeError, ValueError):
        at = 0
    try:
        hp = int(meta.get(SHOP_PET_BONUS_HP_KEY, 0))
    except (TypeError, ValueError):
        hp = 0
    return max(0, at), max(0, hp)


def _set_shop_pet_bonus(state: dict[str, Any], at: int, hp: int) -> None:
    meta = state.setdefault("meta", {})
    meta[SHOP_PET_BONUS_AT_KEY] = max(0, int(at))
    meta[SHOP_PET_BONUS_HP_KEY] = max(0, int(hp))


def _parrot_copy_store(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    meta = state.setdefault("meta", {})
    store = meta.setdefault(PARROT_COPY_META_KEY, {})
    if not isinstance(store, dict):
        store = {}
        meta[PARROT_COPY_META_KEY] = store
    return store


def _clear_parrot_copy_for_team_index(state: dict[str, Any], team_index: int) -> None:
    store = _parrot_copy_store(state)
    store.pop(str(team_index), None)


def _remap_parrot_copy_for_reorder(state: dict[str, Any], order: list[int]) -> None:
    store = _parrot_copy_store(state)
    old_to_new = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(order)}
    remapped: dict[str, dict[str, Any]] = {}
    for old_idx_str, payload in store.items():
        try:
            old_idx = int(old_idx_str)
            new_idx = old_to_new[old_idx]
        except (TypeError, ValueError, KeyError):
            continue
        if isinstance(payload, dict):
            remapped[str(new_idx)] = dict(payload)
    state.setdefault("meta", {})[PARROT_COPY_META_KEY] = remapped


def _effective_actor_for_trigger(
    state: dict[str, Any],
    team_index: int,
    trigger: str,
    fallback_pet_id: str,
    fallback_level: int,
) -> tuple[str, int]:
    return effective_actor_for_trigger(
        state, team_index, trigger, fallback_pet_id, fallback_level
    )


def _is_empty(slot: dict[str, Any]) -> bool:
    return slot["pet_id"] is None


def _default_sell_value(slot: dict[str, Any]) -> int:
    if slot.get("pet_id") is None:
        try:
            val = int(slot.get("sell_value", 0))
        except (TypeError, ValueError):
            val = 0
        return max(0, val)
    try:
        val = int(slot.get("sell_value", 1))
    except (TypeError, ValueError):
        val = 1
    return max(1, val)


def _first_empty_idx(state: dict[str, Any]) -> int | None:
    for i, slot in enumerate(state["team"]):
        if _is_empty(slot):
            return i
    return None


def _occupied_indices(state: dict[str, Any]) -> list[int]:
    return [i for i, s in enumerate(state["team"]) if not _is_empty(s)]


def _find_shop_slot(state: dict[str, Any], shop_index: int) -> tuple[int, dict[str, Any]] | tuple[None, None]:
    for i, slot in enumerate(state["shop"]):
        if slot["shop_index"] == shop_index:
            return i, slot
    return None, None


def _reindex_shop(shop: list[dict[str, Any]]) -> None:
    for idx, slot in enumerate(shop):
        slot["shop_index"] = idx


# THE SHOP HELPERS BELOW REBUILD `state["shop"]` WITHOUT COPYING ITS SLOTS.
#
# Every one of them is a list comprehension that was already allocating a new
# list, reading slot dicts out of a state THE CURRENT CALL OWNS EXCLUSIVELY:
# `apply_action` deep-copies its argument at the head of the function, and the
# two end-turn resolvers do the same. That ownership copy STAYS -- it is the
# boundary the search depends on, it is 32.6% of the call, and skipping it is
# measured rather than argued: the arm that does mutates the caller's board on
# 518 transitions and leaves 90,472 objects shared between input and output.
# By the time any helper below runs, nothing outside the call can reach the
# slots it rebuilds, so copying them again bought a distinctness nothing reads.
#
# exp13, 2026-08-09: `_enforce_shop_schema_limit` ran 137,043 times over
# 118,937 recorded transitions, met the overflow it guards against 0 times and
# changed the shop 0 times, while it and the `_partition_shop_pet_food` it
# calls accounted for about 10.07 of `apply_action`'s 17.114 top-level deep
# copies per call. Across all fifteen removed sites this is 16.114 of those
# 17.114 copies and 29.80% of the call, priced in `RESULTS_deepcopy_census.md`
# and re-measured on this branch in `RESULTS_apply_action_copies.md`. THE
# GUARD ITSELF IS UNTOUCHED: the overflow branch is still reachable and still
# trims, and "0 of 137,043" is a statement about that corpus, not a proof that
# the branch is dead.
#
# TWO PRECONDITIONS, both asserted rather than inherited, by
# `gate_apply_action_copies_identity.py` over the recorded corpus:
#
#   1. No caller writes through a slot dict it was handed. The one caller that
#      mutates a returned list, `_insert_shop_item`, mutates the LIST (`pop`,
#      `insert`) and never a slot. An ability added later is exactly the thing
#      that would break this, and it would break it into a wrong board rather
#      than a crash, which is why the gate measures it on every run.
#   2. No state entering the engine carries the SAME slot dict at two shop or
#      team positions. An aliased input is the one shape where this removal is
#      not a no-op, because `_reindex_shop` would then write `shop_index`
#      through both positions. The engine cannot create one -- the gate counts
#      0 internal aliases in every returned state, so a chain of calls stays
#      alias-free -- and the gate's constructed `aliased_shop_slot` case is
#      where that boundary is DEMONSTRATED instead of assumed.
#
# `ability/turtle_pack.py` defines an identically named
# `_partition_shop_pet_food` that has used `dict(s)` on the same shop of the
# same state all along.
def _enforce_shop_schema_limit(state: dict[str, Any], notes: list[str]) -> None:
    """Hard-guard shop invariants so schema validation cannot fail on overflow."""
    shop = [s for s in state.get("shop", []) if isinstance(s, dict)]
    shop = _partition_shop_pet_food(shop)

    if len(shop) <= 9:
        state["shop"] = shop
        _reindex_shop(state["shop"])
        return

    notes.append(f"shop_rule_violation:overflow_detected:size={len(shop)}")
    while len(shop) > 9:
        removable = [i for i, slot in enumerate(shop) if not slot.get("frozen")]
        if removable:
            drop_idx = max(removable)
        else:
            # Failsafe: keep state schema-valid even if all slots are frozen.
            drop_idx = len(shop) - 1
            notes.append("shop_rule_violation:overflow_all_frozen_force_drop")
        dropped = shop.pop(int(drop_idx))
        notes.append(f"shop_overflow_trim:{dropped.get('slot_type')}:{dropped.get('item_id')}")

    state["shop"] = _partition_shop_pet_food(shop)
    _reindex_shop(state["shop"])


def _partition_shop_pet_food(shop: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # exp13: the slots are REUSED, not copied; see the note above
    # `_enforce_shop_schema_limit`. Each slot matches exactly one of the three
    # predicates, so the returned list holds every input slot once and none twice.
    pets = [s for s in shop if s.get("slot_type") == "pet"]
    foods = [s for s in shop if s.get("slot_type") == "food"]
    others = [s for s in shop if s.get("slot_type") not in {"pet", "food"}]
    return pets + foods + others


def _expected_shop_counts(turn: int) -> tuple[int, int]:
    return expected_shop_counts(turn)


def _pet_base_stats(cat: dict[str, Any], pet_id: str) -> tuple[int, int]:
    base = cat.get("pets", {}).get("base_stats", {}).get(str(pet_id), {"attack": 1, "health": 1})
    return int(base.get("attack", 1)), int(base.get("health", 1))


def _pet_tier(cat: dict[str, Any], pet_id: str) -> int | None:
    target = str(pet_id)
    by_tier = cat.get("pets", {}).get("by_tier", {})
    for tier_str, pets in by_tier.items():
        if target in pets:
            try:
                return int(tier_str)
            except ValueError:
                return None
    return None


def _food_effect_stats(shop_slot: dict[str, Any]) -> tuple[int, int, int, int]:
    item_id = str(shop_slot.get("item_id", ""))
    base_attack, base_health = FOOD_STAT_BUFFS.get(item_id, (0, 0))
    attack = int(shop_slot.get("attack", base_attack))
    health = int(shop_slot.get("health", base_health))
    return int(base_attack), int(base_health), int(attack), int(health)


def _apply_equipment_to_pet_slot(slot: dict[str, Any], equipment_id: str) -> None:
    apply_standard_equipment(slot, equipment_id)


def _food_target_count(item_id: str) -> int:
    return int(RANDOM_TARGET_FOOD_COUNTS.get(str(item_id), 0))


def _random_food_target_indices(state: dict[str, Any], item_id: str, rng: random.Random) -> tuple[list[int], bool]:
    n = _food_target_count(item_id)
    occupied = _occupied_indices(state)
    if n <= 0 or not occupied:
        return [], False
    n = min(int(n), len(occupied))
    if n == len(occupied):
        return list(occupied), False
    return sorted(rng.sample(list(occupied), n)), True


def _empty_shop_slot(slot_type: str) -> dict[str, Any]:
    if slot_type == "pet":
        return {
            "shop_index": -1,
            "slot_type": "pet",
            "item_id": "pet-none",
            "attack": 0,
            "health": 0,
            "at": 0,
            "hp": 0,
            "cost": int(DEFAULT_SHOP_ITEM_COST),
            "frozen": False,
            "link_id": None,
        }
    return {
        "shop_index": -1,
        "slot_type": "food",
        "item_id": "food-none",
        "cost": int(DEFAULT_SHOP_ITEM_COST),
        "frozen": False,
        "link_id": None,
    }


def _rebuild_shop_for_roll(state: dict[str, Any], notes: list[str]) -> None:
    expected_pets, expected_foods = _expected_shop_counts(int(state["turn"]))

    # exp13: no per-slot copy on these four rescans; see the note above
    # `_enforce_shop_schema_limit`. The four predicates are mutually exclusive, and
    # a slot this rebuild drops is simply absent from the list it assigns.
    frozen_pets = [s for s in state["shop"] if s["slot_type"] == "pet" and s.get("frozen")]
    frozen_foods = [s for s in state["shop"] if s["slot_type"] == "food" and s.get("frozen")]
    nonfrozen_pets = [s for s in state["shop"] if s["slot_type"] == "pet" and not s.get("frozen")]
    nonfrozen_foods = [s for s in state["shop"] if s["slot_type"] == "food" and not s.get("frozen")]

    if len(frozen_pets) > expected_pets:
        notes.append("shop_rule_violation:frozen_pets_exceed_expected")
    if len(frozen_foods) > expected_foods:
        notes.append("shop_rule_violation:frozen_foods_exceed_expected")

    keep_nonfrozen_pets = nonfrozen_pets[: max(0, expected_pets - len(frozen_pets))]
    keep_nonfrozen_foods = nonfrozen_foods[: max(0, expected_foods - len(frozen_foods))]

    pets = frozen_pets + keep_nonfrozen_pets
    foods = frozen_foods + keep_nonfrozen_foods

    while len(pets) < expected_pets:
        pets.append(_empty_shop_slot("pet"))
    while len(foods) < expected_foods:
        foods.append(_empty_shop_slot("food"))

    state["shop"] = pets + foods
    _reindex_shop(state["shop"])


def _items_up_to_tier(by_tier: dict[str, list[str]], max_tier: int) -> list[str]:
    out: list[str] = []
    for tier in range(1, max_tier + 1):
        out.extend(list(by_tier.get(str(tier), [])))
    return out


def _item_tier_map(by_tier: dict[str, list[str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for tier_str, ids in by_tier.items():
        try:
            tier = int(tier_str)
        except ValueError:
            continue
        for item_id in ids:
            out[str(item_id)] = tier
    return out


def _sort_shop_by_tier(state: dict[str, Any], cat: dict[str, Any]) -> None:
    pet_tiers = _item_tier_map(cat.get("pets", {}).get("by_tier", {}))
    food_tiers = _item_tier_map(cat.get("foods", {}).get("by_tier", {}))
    # exp13: no per-slot copy; see the note above `_enforce_shop_schema_limit`.
    # The sorts below reorder these lists of references and READ the slots to key
    # on; they never write through one.
    pets = [s for s in state["shop"] if s.get("slot_type") == "pet"]
    foods = [s for s in state["shop"] if s.get("slot_type") == "food"]

    # Keep frozen slots stable by existing shop index; sort only non-frozen by tier.
    pet_frozen = sorted((s for s in pets if s.get("frozen")), key=lambda s: int(s.get("shop_index", 0)))
    pet_nonfrozen = [s for s in pets if not s.get("frozen")]
    pet_nonfrozen.sort(key=lambda s: (-int(pet_tiers.get(str(s.get("item_id", "")), 0)), str(s.get("item_id", ""))))

    food_frozen = sorted((s for s in foods if s.get("frozen")), key=lambda s: int(s.get("shop_index", 0)))
    food_nonfrozen = [s for s in foods if not s.get("frozen")]
    food_nonfrozen.sort(key=lambda s: (-int(food_tiers.get(str(s.get("item_id", "")), 0)), str(s.get("item_id", ""))))

    # Always keep foods at higher indices than pets.
    state["shop"] = pet_frozen + pet_nonfrozen + food_frozen + food_nonfrozen
    _reindex_shop(state["shop"])


def _roll_shop_slots(state: dict[str, Any], cat: dict[str, Any], rng: random.Random, tier: int) -> None:
    pet_pool = [
        item_id
        for item_id in _items_up_to_tier(cat.get("pets", {}).get("by_tier", {}), tier)
        if str(item_id) not in NON_ROLLABLE_PET_IDS and str(item_id) != SLOTH_PET_ID
    ]
    food_pool = [
        item_id
        for item_id in _items_up_to_tier(cat.get("foods", {}).get("by_tier", {}), tier)
        if str(item_id) not in NON_ROLLABLE_FOOD_IDS
    ]
    bonus_at, bonus_hp = _shop_pet_bonus(state)
    for slot in state["shop"]:
        if slot.get("frozen"):
            if slot.get("slot_type") == "pet":
                pet_id = str(slot.get("item_id", ""))
                if pet_id not in {"", "pet-none"} and ("attack" not in slot or "health" not in slot):
                    atk, hp = _pet_base_stats(cat, pet_id)
                    slot["attack"] = int(_cap_stat(int(atk) + int(bonus_at)))
                    slot["health"] = int(_cap_stat(int(hp) + int(bonus_hp)))
                slot["at"] = int(bonus_at)
                slot["hp"] = int(bonus_hp)
            continue
        if slot["slot_type"] == "pet" and pet_pool:
            slot["item_id"] = rng.choice(pet_pool)
            slot["cost"] = int(shop_item_cost("pet", str(slot["item_id"])))
            slot["link_id"] = None
            atk, hp = _pet_base_stats(cat, str(slot["item_id"]))
            slot["attack"] = int(_cap_stat(int(atk) + int(bonus_at)))
            slot["health"] = int(_cap_stat(int(hp) + int(bonus_hp)))
            slot["at"] = int(bonus_at)
            slot["hp"] = int(bonus_hp)
        elif slot["slot_type"] == "food" and food_pool:
            slot["item_id"] = rng.choice(food_pool)
            slot["cost"] = int(shop_item_cost("food", str(slot["item_id"])))
            slot["link_id"] = None
            slot.pop("attack", None)
            slot.pop("health", None)
            slot.pop("at", None)
            slot.pop("hp", None)

    # Sloth easter egg (exp02 ghidra RandomizeShop): the shop draws once per roll
    # and, only if it lands under SLOTH_ROLL_PROB, turns the first freshly-rolled
    # (non-frozen) pet slot into Sloth. The draw is always consumed so Sloth never
    # appears as a normal uniform pick (it is excluded from pet_pool above).
    if pet_pool and rng.random() < SLOTH_ROLL_PROB:
        for slot in state["shop"]:
            if slot.get("slot_type") == "pet" and not slot.get("frozen"):
                slot["item_id"] = SLOTH_PET_ID
                slot["cost"] = int(shop_item_cost("pet", SLOTH_PET_ID))
                slot["link_id"] = None
                atk, hp = _pet_base_stats(cat, SLOTH_PET_ID)
                slot["attack"] = int(_cap_stat(int(atk) + int(bonus_at)))
                slot["health"] = int(_cap_stat(int(hp) + int(bonus_hp)))
                slot["at"] = int(bonus_at)
                slot["hp"] = int(bonus_hp)
                break


def _rng_from_state(state: dict[str, Any]) -> random.Random:
    seed = state.get("meta", {}).get("seed") if state.get("meta", {}).get("seed_known") else None
    return random.Random(seed)


def _cap_stat(value: int) -> int:
    return max(0, min(50, int(value)))


def _level_from_exp(exp: int) -> int:
    if exp >= 5:
        return 3
    if exp >= 2:
        return 2
    return 1


def _gain_experience(
    slot: dict[str, Any],
    amount: int = 1,
    *,
    stat_amount: int | None = None,
    baseline_level: int | None = None,
) -> bool:
    """Advance ``slot`` by ``amount`` experience; return whether it levelled up.

    Combines need the two halves of an exp gain decoupled: the permanent stat
    bonus is fixed by the LOWER exp of the two pets while the exp itself totals
    both pets plus one, so ``stat_amount`` overrides the +N/+N that normally
    rides along with ``amount``. ``baseline_level`` raises the level the
    returned level-up flag is measured against, which is how a combine reports
    a level-up only when the result outranks BOTH of its inputs.
    """
    ensure_stat_fields(slot)
    gain = max(0, int(amount))
    stat_gain = gain if stat_amount is None else max(0, int(stat_amount))
    if stat_gain > 0:
        slot["perm_attack"] = _cap_stat(int(slot.get("perm_attack", 0)) + stat_gain)
        slot["perm_health"] = _cap_stat(int(slot.get("perm_health", 0)) + stat_gain)
    slot["attack"] = _cap_stat(int(slot.get("perm_attack", 0)) + int(slot.get("temp_attack", 0)))
    slot["health"] = _cap_stat(int(slot.get("perm_health", 0)) + int(slot.get("temp_health", 0)))
    before_exp = max(0, min(5, int(slot.get("exp", 0))))
    before_level = _level_from_exp(before_exp)
    after_exp = max(0, min(5, before_exp + gain))
    after_level = _level_from_exp(after_exp)
    slot["exp"] = after_exp
    slot["level"] = after_level
    if slot.get("pet_id") is not None:
        try:
            current_sell = int(slot.get("sell_value", 1))
        except (TypeError, ValueError):
            current_sell = 1
        level_gain = max(0, int(after_level) - int(before_level))
        next_sell = current_sell + level_gain
        # Keep level-based floor (L1=1, L2=2, L3=3) while allowing effects (e.g. Birthday Cake) to exceed it.
        slot["sell_value"] = _cap_stat(max(int(after_level), int(next_sell), 1))
    trigger_from = before_level if baseline_level is None else max(before_level, int(baseline_level))
    return after_level > trigger_from


def _combine_pet_stats(pet_to_keep: dict[str, Any], pet_to_merge: dict[str, Any]) -> bool:
    ensure_stat_fields(pet_to_keep)
    ensure_stat_fields(pet_to_merge)

    keep_perm_atk = int(pet_to_keep.get("perm_attack", pet_to_keep.get("attack", 0)))
    keep_temp_atk = int(pet_to_keep.get("temp_attack", 0))
    keep_perm_hp = int(pet_to_keep.get("perm_health", pet_to_keep.get("health", 0)))
    keep_temp_hp = int(pet_to_keep.get("temp_health", 0))

    merge_perm_atk = int(pet_to_merge.get("perm_attack", pet_to_merge.get("attack", 0)))
    merge_temp_atk = int(pet_to_merge.get("temp_attack", 0))
    merge_perm_hp = int(pet_to_merge.get("perm_health", pet_to_merge.get("health", 0)))
    merge_temp_hp = int(pet_to_merge.get("temp_health", 0))

    keep_total_atk = _cap_stat(keep_perm_atk + keep_temp_atk)
    keep_total_hp = _cap_stat(keep_perm_hp + keep_temp_hp)
    merge_total_atk = _cap_stat(merge_perm_atk + merge_temp_atk)
    merge_total_hp = _cap_stat(merge_perm_hp + merge_temp_hp)

    total_atk = _cap_stat(max(keep_total_atk, merge_total_atk))
    total_hp = _cap_stat(max(keep_total_hp, merge_total_hp))

    if merge_total_atk > keep_total_atk:
        chosen_perm_atk = merge_perm_atk
    elif merge_total_atk < keep_total_atk:
        chosen_perm_atk = keep_perm_atk
    else:
        chosen_perm_atk = max(keep_perm_atk, merge_perm_atk)
    if merge_total_hp > keep_total_hp:
        chosen_perm_hp = merge_perm_hp
    elif merge_total_hp < keep_total_hp:
        chosen_perm_hp = keep_perm_hp
    else:
        chosen_perm_hp = max(keep_perm_hp, merge_perm_hp)

    chosen_perm_atk = min(_cap_stat(chosen_perm_atk), total_atk)
    chosen_perm_hp = min(_cap_stat(chosen_perm_hp), total_hp)

    pet_to_keep["perm_attack"] = int(chosen_perm_atk)
    pet_to_keep["perm_health"] = int(chosen_perm_hp)
    pet_to_keep["temp_attack"] = int(_cap_stat(total_atk - chosen_perm_atk))
    pet_to_keep["temp_health"] = int(_cap_stat(total_hp - chosen_perm_hp))
    pet_to_keep["attack"] = int(_cap_stat(int(pet_to_keep["perm_attack"]) + int(pet_to_keep["temp_attack"])))
    pet_to_keep["health"] = int(_cap_stat(int(pet_to_keep["perm_health"]) + int(pet_to_keep["temp_health"])))
    merged_status = sorted(set(list(pet_to_keep.get("status_effects", [])) + list(pet_to_merge.get("status_effects", []))))
    pet_to_keep["status_effects"] = merged_status
    # Equipment (Ruihan ruling 2026-08-01): the combine TARGET keeps its own
    # equipment, an unequipped target inherits the merged pet's, and when both
    # carry one the target wins, so the loser's perk status must not ride along
    # in the merged status set.
    #
    # Only equipment_id moves. A perk's status effect is never ADDED here: the
    # replay compiler's entry boards carry equipment_id with an EMPTY status
    # list, and synthesising the status back would silently change damage
    # mitigation on real human turns (measured, exp12 W0 faithfulness probe).
    keep_equipment = pet_to_keep.get("equipment_id") or None
    merge_equipment = pet_to_merge.get("equipment_id") or None
    if keep_equipment is None and merge_equipment is not None:
        pet_to_keep["equipment_id"] = str(merge_equipment)
    elif keep_equipment is not None and merge_equipment is not None:
        loser_status = EQUIPMENT_STATUS_BY_FOOD_ID.get(str(merge_equipment))
        winner_status = EQUIPMENT_STATUS_BY_FOOD_ID.get(str(keep_equipment))
        if loser_status and loser_status != winner_status:
            pet_to_keep["status_effects"] = [s for s in merged_status if s != loser_status]

    # Direction-independent combine (Ruihan ruling 2026-08-01): stats are the
    # max of the two plus 1 + the LOWER exp, which is what "always merge the
    # lower-exp pet into the higher-exp one" evaluates to. Exp still totals
    # keep + merge + 1 (unchanged, already direction-independent), and the
    # level-up flag is measured against the HIGHER of the two input levels so a
    # reverse-direction merge cannot re-fire a threshold that was already
    # crossed.
    keep_exp = max(0, min(5, int(pet_to_keep.get("exp", 0))))
    merge_exp = max(0, min(5, int(pet_to_merge.get("exp", 0))))
    return _gain_experience(
        pet_to_keep,
        amount=(1 + merge_exp),
        stat_amount=(1 + min(keep_exp, merge_exp)),
        baseline_level=max(_level_from_exp(keep_exp), _level_from_exp(merge_exp)),
    )


def _levelup_reward_tier(turn: int) -> int:
    return min(6, tier_for_turn(turn) + 1)


def _next_levelup_link_id(state: dict[str, Any]) -> str:
    meta = state.setdefault("meta", {})
    next_id = int(meta.get("next_levelup_link_id", 1))
    meta["next_levelup_link_id"] = next_id + 1
    return f"levelup-{next_id}"


def _add_levelup_shop_slot(state: dict[str, Any], cat: dict[str, Any], rng: random.Random, notes: list[str]) -> bool:
    tier = _levelup_reward_tier(int(state["turn"]))
    pool = [
        pid
        for pid in dict.fromkeys(cat.get("pets", {}).get("by_tier", {}).get(str(tier), []))
        if str(pid) != SLOTH_PET_ID  # Sloth is never a level-up reward (easter egg only)
    ]
    if not pool:
        notes.append("shop_rule_violation:missing_levelup_pool")
        return False
    link_id = _next_levelup_link_id(state)
    added = 0
    first = rng.choice(pool)
    reward_pets = [first]
    remaining = [pid for pid in pool if str(pid) != str(first)]
    if remaining:
        reward_pets.append(rng.choice(remaining))
    else:
        notes.append("levelup_reward_pair_reduced:single_option_pool")
    for reward_pet in reward_pets:
        if _insert_shop_item(
            state,
            slot_type="pet",
            item_id=reward_pet,
            cost=3,
            frozen=False,
            notes=notes,
            reason_tag=f"levelup_reward_added:tier={tier}",
            cat=cat,
            link_id=link_id,
        ):
            added += 1
    if added:
        notes.append(f"levelup_reward_pair_added:{added}:link={link_id}")
        return True
    notes.append(f"levelup_reward_pair_skipped:link={link_id}")
    return False


def _insert_shop_item(
    state: dict[str, Any],
    slot_type: str,
    item_id: str,
    cost: int,
    frozen: bool,
    notes: list[str],
    reason_tag: str,
    cat: dict[str, Any] | None = None,
    link_id: str | None = None,
) -> bool:
    shop = _partition_shop_pet_food(state["shop"])
    new_slot = {
        "shop_index": -1,
        "slot_type": slot_type,
        "item_id": item_id,
        "cost": int(cost),
        "frozen": bool(frozen),
        "link_id": link_id,
    }
    if slot_type == "pet":
        if cat is None:
            cat = load_turtle_catalog()
        atk, hp = _pet_base_stats(cat, item_id)
        bonus_at, bonus_hp = _shop_pet_bonus(state)
        new_slot["attack"] = int(_cap_stat(int(atk) + int(bonus_at)))
        new_slot["health"] = int(_cap_stat(int(hp) + int(bonus_hp)))
        new_slot["at"] = int(bonus_at)
        new_slot["hp"] = int(bonus_hp)

    # If full, evict an existing non-frozen slot first. This keeps insertion
    # logic stable (no transient len=10 state) and mirrors stack-like behavior.
    if len(shop) >= 9:
        removable_existing = [i for i, s in enumerate(shop) if not s.get("frozen")]
        if not removable_existing:
            notes.append("shop_insert_skipped:all_slots_frozen")
            state["shop"] = shop
            _reindex_shop(state["shop"])
            return False
        drop_idx = max(removable_existing) if slot_type == "pet" else min(removable_existing)
        dropped = shop.pop(int(drop_idx))
        notes.append(f"shop_overflow_drop:{dropped.get('slot_type')}:{dropped.get('item_id')}")

    insert_at = 0 if slot_type == "pet" else len(shop)
    shop.insert(insert_at, new_slot)

    state["shop"] = _partition_shop_pet_food(shop)
    _reindex_shop(state["shop"])
    notes.append(reason_tag)
    return True


def _remove_linked_shop_slots(state: dict[str, Any], link_id: str, notes: list[str]) -> None:
    before = len(state["shop"])
    # exp13: no per-slot copy; see the note above `_enforce_shop_schema_limit`.
    state["shop"] = [s for s in state["shop"] if str(s.get("link_id")) != str(link_id)]
    removed = before - len(state["shop"])
    if removed > 0:
        notes.append(f"linked_shop_slots_removed:{removed}:link={link_id}")
    state["shop"] = _partition_shop_pet_food(state["shop"])
    _reindex_shop(state["shop"])


def _legal_buy_pet(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    if action["team_index"] < 0 or action["team_index"] > 4:
        return False, "team_index_out_of_range"
    idx, shop_slot = _find_shop_slot(state, action["shop_index"])
    if idx is None or shop_slot["slot_type"] != "pet":
        return False, "shop_slot_not_pet"
    if shop_slot["cost"] > state["gold"]:
        return False, "insufficient_gold"
    first = _first_empty_idx(state)
    if first is None:
        return False, "team_full"
    if action["team_index"] != first:
        return False, "unsupported_buy_pet_target"
    return True, None


def _legal_buy_combine(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    if action["team_index"] < 0 or action["team_index"] > 4:
        return False, "team_index_out_of_range"
    idx, shop_slot = _find_shop_slot(state, action["shop_index"])
    if idx is None or shop_slot["slot_type"] != "pet":
        return False, "shop_slot_not_pet"
    if shop_slot["cost"] > state["gold"]:
        return False, "insufficient_gold"
    team_slot = _team_slot(state, action["team_index"])
    if _is_empty(team_slot):
        return False, "combine_target_empty"
    if int(team_slot.get("level", 1)) >= 3:
        return False, "combine_target_level_max"
    if team_slot["pet_id"] != shop_slot["item_id"]:
        return False, "combine_pet_mismatch"
    return True, None


def _legal_buy_food(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    idx, shop_slot = _find_shop_slot(state, action["shop_index"])
    if idx is None or shop_slot["slot_type"] != "food":
        return False, "shop_slot_not_food"
    if shop_slot["cost"] > state["gold"]:
        return False, "insufficient_gold"
    item_id = str(shop_slot.get("item_id", ""))
    team_index = action.get("team_index")
    if item_id in NO_TARGET_FOODS:
        if team_index is not None:
            return False, "food_expects_no_target"
        return True, None
    if item_id in RANDOM_TARGET_FOODS:
        if team_index is not None:
            return False, "food_expects_random_target"
        if len(_occupied_indices(state)) == 0:
            return False, "food_no_possible_target"
        return True, None
    if item_id not in RANDOM_TARGET_FOODS:
        ti = team_index
        if ti is None:
            return False, "food_target_required"
        if ti < 0 or ti > 4:
            return False, "food_target_out_of_range"
        if _is_empty(_team_slot(state, ti)):
            return False, "food_target_empty"
    return True, None


def _legal_sell(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    if action["team_index"] < 0 or action["team_index"] > 4:
        return False, "team_index_out_of_range"
    slot = _team_slot(state, action["team_index"])
    if _is_empty(slot):
        return False, "sell_empty_slot"
    return True, None


def _legal_combine(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    if action["src_team_index"] < 0 or action["src_team_index"] > 4:
        return False, "src_team_index_out_of_range"
    if action["dst_team_index"] < 0 or action["dst_team_index"] > 4:
        return False, "dst_team_index_out_of_range"
    src = _team_slot(state, action["src_team_index"])
    dst = _team_slot(state, action["dst_team_index"])
    if action["src_team_index"] == action["dst_team_index"]:
        return False, "combine_same_slot"
    if _is_empty(src) or _is_empty(dst):
        return False, "combine_empty_slot"
    src_level = int(src.get("level", 1))
    dst_level = int(dst.get("level", 1))
    if src_level >= 3 and dst_level <= 1:
        return False, "combine_src_level3_into_level1_forbidden"
    if int(dst.get("level", 1)) >= 3:
        return False, "combine_target_level_max"
    if src["pet_id"] != dst["pet_id"]:
        return False, "combine_pet_mismatch"
    return True, None


def _legal_reorder(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    order = action["order"]
    if sorted(order) != [0, 1, 2, 3, 4]:
        return False, "invalid_permutation"
    return True, None


def _legal_freeze_unfreeze(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    idx, slot = _find_shop_slot(state, action["shop_index"])
    if idx is None:
        return False, "shop_slot_missing"
    action_type = str(action.get("type", "")).strip().upper()
    frozen = bool(slot.get("frozen"))
    if action_type == "FREEZE" and frozen:
        return False, "freeze_slot_already_frozen"
    if action_type == "UNFREEZE" and not frozen:
        return False, "unfreeze_slot_not_frozen"
    return True, None


def _legal_roll(state: dict[str, Any], _action: dict[str, Any]) -> tuple[bool, str | None]:
    if state["gold"] < 1:
        return False, "roll_requires_gold_gte_1"
    return True, None


def check_legal(state: dict[str, Any], action: dict[str, Any]) -> tuple[bool, str | None]:
    typ = action["type"]
    if typ == "BUY_PET":
        return _legal_buy_pet(state, action)
    if typ == "BUY_COMBINE":
        return _legal_buy_combine(state, action)
    if typ == "BUY_FOOD":
        return _legal_buy_food(state, action)
    if typ == "SELL":
        return _legal_sell(state, action)
    if typ == "COMBINE":
        return _legal_combine(state, action)
    if typ == "REORDER":
        return _legal_reorder(state, action)
    if typ in {"FREEZE", "UNFREEZE"}:
        return _legal_freeze_unfreeze(state, action)
    if typ == "ROLL":
        return _legal_roll(state, action)
    if typ == "END_TURN":
        return True, None
    return False, "unknown_action_type"


def legal_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    first_empty = _first_empty_idx(state)
    occupied = _occupied_indices(state)

    for slot in state["shop"]:
        if slot["slot_type"] == "pet" and first_empty is not None and slot["cost"] <= state["gold"]:
            actions.append({"type": "BUY_PET", "shop_index": slot["shop_index"], "team_index": first_empty})
        if slot["slot_type"] == "pet" and slot["cost"] <= state["gold"]:
            for ti in occupied:
                if state["team"][ti]["pet_id"] == slot["item_id"] and int(state["team"][ti].get("level", 1)) < 3:
                    actions.append({"type": "BUY_COMBINE", "shop_index": slot["shop_index"], "team_index": ti})
        if slot["slot_type"] == "food" and slot["cost"] <= state["gold"]:
            if slot["item_id"] in NO_TARGET_FOODS:
                actions.append({"type": "BUY_FOOD", "shop_index": slot["shop_index"], "team_index": None})
            elif slot["item_id"] in RANDOM_TARGET_FOODS:
                if occupied:
                    actions.append({"type": "BUY_FOOD", "shop_index": slot["shop_index"], "team_index": None})
            else:
                for ti in occupied:
                    actions.append({"type": "BUY_FOOD", "shop_index": slot["shop_index"], "team_index": ti})

    for i in occupied:
        actions.append({"type": "SELL", "team_index": i})

    for i, j in itertools.permutations(occupied, 2):
        src_level = int(state["team"][i].get("level", 1))
        dst_level = int(state["team"][j].get("level", 1))
        if (
            state["team"][i]["pet_id"] == state["team"][j]["pet_id"]
            and dst_level < 3
            and not (src_level >= 3 and dst_level <= 1)
        ):
            actions.append({"type": "COMBINE", "src_team_index": i, "dst_team_index": j})

    # Restrict reorder enumeration to occupied team slots to avoid no-op-heavy action space.
    # Empty slots remain fixed.
    if len(occupied) >= 2:
        identity = [0, 1, 2, 3, 4]
        for perm in itertools.permutations(occupied):
            order = list(identity)
            for slot_idx, src_idx in zip(occupied, perm):
                order[slot_idx] = src_idx
            if order != identity:
                actions.append({"type": "REORDER", "order": order})

    for slot in state["shop"]:
        if bool(slot.get("frozen")):
            actions.append({"type": "UNFREEZE", "shop_index": slot["shop_index"]})
        else:
            actions.append({"type": "FREEZE", "shop_index": slot["shop_index"]})

    if state["gold"] >= 1:
        actions.append({"type": "ROLL"})
    actions.append({"type": "END_TURN"})

    # Deduplicate while PRESERVING ORDER, and order is a contract: an action's
    # position in this list is the action index the recorded chains, the label
    # rows and the BC catalog mask are all built against. A dedup that returned
    # the same actions in a different order would be a silent dataset break,
    # not a speedup, which is why `gate_legal_actions_key_identity.py` compares
    # the returned list byte for byte with order included, over recorded boards.
    #
    # exp13, 2026-08-09: the key used to be
    # `json.dumps(action, sort_keys=True, separators=(",", ":"))` -- one string
    # allocated per enumerated action and never read by anything. Measured on
    # 200 real recorded boards it was 43.9% of this call, this call is 14.95%
    # of labelling play wall clock, and across 22,353 enumerated actions the
    # dedup removed exactly 0 duplicates. So the string was paying for a
    # separation nothing on these boards needed, at the most expensive price
    # available for reaching a `set`.
    #
    # `hashable_action_key` separates exactly what `json.dumps` separates, per
    # value and per type (`action_keys.py` gives the rules and the reason each
    # one is load-bearing). That is what makes this swap a no-op on the
    # returned list by construction rather than a claim resting on the 0.
    # Measured in `RESULTS_legal_actions_key.md`; it is the same defect the
    # 2026-08-08 wave removed from `bc_recommender.legal_mask`, one layer down.
    #
    # The dedup itself STAYS. It removes nothing on the corpus measured, but
    # "nothing on this corpus" is not "nothing by construction" -- BUY_FOOD in
    # particular is enumerated from three branches -- and with a tuple key it
    # is no longer expensive enough to be worth trading a guard for.
    seen = set()
    unique: list[dict[str, Any]] = []
    for action in actions:
        key = hashable_action_key(action)
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return unique


def _reseed_meta(rng: random.Random, state: dict[str, Any]) -> None:
    state.setdefault("meta", {})
    state["meta"]["seed"] = rng.randrange(0, 2**31)


def _merge_stochastic_reason(existing: str | None, incoming: str | None) -> str | None:
    if existing is None:
        return incoming
    if incoming is None or incoming == existing:
        return existing
    return f"{existing}+{incoming}"


def _read_counter_value(by_trigger: dict[str, Any], key: str) -> int:
    raw = by_trigger.get(str(key), 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _set_counter_value(by_trigger: dict[str, Any], key: str, value: int) -> None:
    val = max(0, int(value))
    if val <= 0:
        by_trigger.pop(str(key), None)
        return
    by_trigger[str(key)] = int(val)


def _merge_combine_ability_counters(
    state: dict[str, Any],
    *,
    dst_team_index: int,
    src_team_index: int | None,
    leveled_up: bool,
) -> None:
    store = state.setdefault("meta", {}).get("ability_counters", {})
    if not isinstance(store, dict):
        return
    dst_key = str(int(dst_team_index))
    src_key = str(int(src_team_index)) if src_team_index is not None else None

    for by_trigger in store.values():
        if not isinstance(by_trigger, dict):
            continue
        dst_val = _read_counter_value(by_trigger, dst_key)
        src_val = _read_counter_value(by_trigger, src_key) if src_key is not None else 0
        if leveled_up:
            _set_counter_value(by_trigger, dst_key, 0)
        else:
            _set_counter_value(by_trigger, dst_key, max(dst_val, src_val))
        if src_key is not None:
            by_trigger.pop(src_key, None)


def _sort_ability_events_for_runtime(
    state: dict[str, Any], events: list[AbilityEvent], rng: random.Random
) -> tuple[list[AbilityEvent], bool]:
    return sort_ability_events_for_runtime(state, events, rng)


def _run_ability_events(
    state: dict[str, Any],
    events: list[AbilityEvent],
    notes: list[str],
    *,
    causality: StepCausality,
) -> tuple[bool, str | None]:
    """Run one ability queue -> (deterministic, stochastic_reason).

    The causal half is reported through `causality`, not returned: A4 revision
    2 made the taint per STEP, so there is one accumulator per step and no
    per-call flags to merge. See `StepCausality`.

    The read-set-writing helpers call `causality.note_structural` themselves,
    through `effects.mark_structural` (`effects.gain_gold`,
    `effects.discount_shop_foods`, `turtle_pack._queue_summon`,
    `turtle_pack._ability_insert_shop_item`, `turtle_pack._stock_cow_milk`).
    The drawing helpers arm it themselves through `effects.mark_random`
    (`effects.choose_random_indices`, Spider). Neither names itself, so a
    future random effect that grants gold or summons a pet becomes a cut point
    the day it lands (A1 ruling 2: never a whitelist).

    The EVENT-ORDER tie draw arms too. See A4.2 as revised: the review found it
    steering which of two simultaneously fainting pets acts first, and so which
    slot ends up empty.

    An `unsupported_effect` calls `force_cut` by conservative default: an
    unmodelled effect could have written anything, for any reason.

    The tie draw and the `_reseed_meta` that follows it are UNCHANGED, so the
    board this returns is bit-identical to the pre-amendment engine's; only
    what the step reports is new.
    """
    if not events:
        return True, None

    rng = _rng_from_state(state)
    ordered_events, random_tie_used = _sort_ability_events_for_runtime(state, events, rng)
    if random_tie_used:
        # A4.2 as revised. Armed BEFORE the group resolves, not after: the draw
        # ordered these handlers, so a read-set write made by one of THEM is
        # downstream of it too. That is reachable -- two pets tied on
        # `after_faint` both queue a summon, and the order decides which one
        # lands in the single free slot -- so arming afterwards would leave the
        # same shape of hole the review found in F1 and F2.
        causality.arm()
    runtime = AbilityRuntime(state=state, rng=rng, notes=notes, causality=causality)
    for event in ordered_events:
        runtime.emit(event)
    runtime.run()

    deterministic = True
    reason: str | None = None
    if runtime.random_used or random_tie_used:
        deterministic = False
        reason = "ability_randomness"
        _reseed_meta(rng, state)
    if runtime.unsupported:
        deterministic = False
        reason = _merge_stochastic_reason(reason, "unsupported_effect")
        causality.force_cut()
    if runtime.trace:
        notes.append(f"ability_trace_events={len(runtime.trace)}")
    return deterministic, reason


def _clear_team_slot(state: dict[str, Any], team_index: int) -> None:
    slot = _team_slot(state, int(team_index))
    slot.update(
        pet_id=None,
        attack=0,
        health=0,
        sell_value=0,
        level=1,
        exp=0,
        equipment_id=None,
        status_effects=[],
    )
    clear_ability_counters_for_team_index(state, int(team_index))
    _clear_parrot_copy_for_team_index(state, int(team_index))


def _set_team_slot_to_pet(
    state: dict[str, Any],
    team_index: int,
    pet_id: str,
    attack: int,
    health: int,
    level: int = 1,
    exp: int = 0,
    equipment_id: str | None = None,
) -> None:
    slot = _team_slot(state, int(team_index))
    clamped_level = max(1, min(3, int(level)))
    slot.update(
        pet_id=str(pet_id),
        attack=_cap_stat(int(attack)),
        health=_cap_stat(int(health)),
        sell_value=int(clamped_level),
        level=clamped_level,
        exp=max(0, min(5, int(exp))),
        equipment_id=(str(equipment_id) if equipment_id else None),
        status_effects=[],
    )
    clear_ability_counters_for_team_index(state, int(team_index))
    _clear_parrot_copy_for_team_index(state, int(team_index))


def _emit_summon_events(
    state: dict[str, Any],
    team_index: int,
    pet_id: str,
    level: int,
    notes: list[str],
    *,
    causality: StepCausality,
) -> tuple[bool, str | None]:
    summon_events: list[AbilityEvent] = [
        AbilityEvent(
            trigger=TRIGGER_SUMMONED,
            actor_pet_id=str(pet_id),
            actor_level=max(1, min(3, int(level))),
            actor_team_index=int(team_index),
            payload={"trigger_pet_id": str(pet_id), "trigger_team_index": int(team_index)},
        )
    ]
    for idx, team_slot in enumerate(state["team"]):
        current_pet_id = team_slot.get("pet_id")
        if current_pet_id is None:
            continue
        actor_pet_id, actor_level = _effective_actor_for_trigger(
            state,
            idx,
            TRIGGER_FRIEND_SUMMONED,
            str(current_pet_id),
            int(team_slot.get("level", 1)),
        )
        summon_events.append(
            AbilityEvent(
                trigger=TRIGGER_FRIEND_SUMMONED,
                actor_pet_id=actor_pet_id,
                actor_level=actor_level,
                actor_team_index=idx,
                payload={"trigger_pet_id": str(pet_id), "trigger_team_index": int(team_index)},
            )
        )
    return _run_ability_events(state, summon_events, notes, causality=causality)


def _apply_team_permutation(state: dict[str, Any], order: list[int]) -> None:
    # exp13, 2026-08-09: neither copy is load-bearing. The OUTER one existed so
    # the source list would survive the assignment, which it does anyway -- a list
    # comprehension is fully evaluated before the assignment binds. The INNER one
    # existed so no two entries of the new team could be the same object, and
    # `order` already guarantees that: it is a PERMUTATION, so every index appears
    # exactly once. Both callers build it by rotating a slice of
    # `list(range(len(state["team"])))` (`_push_forward_from_slot`,
    # `_push_backward_from_slot`). That is a proof about the callers rather than a
    # sample -- the recorded corpus reaches this helper 17 and 85 times in 118,937
    # transitions -- so `gate_apply_action_copies_identity.py` asserts the
    # permutation property on every real call AND constructs the summon cases the
    # corpus barely covers. Its `repeated_index` mutation is what a broken `order`
    # would do here.
    old_team = state["team"]
    state["team"] = [old_team[int(i)] for i in order]
    remap_ability_counters_for_reorder(state, list(order))
    _remap_parrot_copy_for_reorder(state, list(order))
    for idx, slot in enumerate(state["team"]):
        slot["slot_index"] = int(idx)


def _find_closest_empty_ahead(state: dict[str, Any], slot: int) -> int | None:
    for idx in range(int(slot) - 1, -1, -1):
        if _is_empty(state["team"][idx]):
            return int(idx)
    return None


def _find_closest_empty_behind(state: dict[str, Any], slot: int) -> int | None:
    for idx in range(int(slot) + 1, len(state["team"])):
        if _is_empty(state["team"][idx]):
            return int(idx)
    return None


def _push_forward_from_slot(state: dict[str, Any], slot: int) -> bool:
    empty_idx = _find_closest_empty_ahead(state, int(slot))
    if empty_idx is None:
        return False
    order = list(range(len(state["team"])))
    for idx in range(int(empty_idx), int(slot)):
        order[idx] = idx + 1
    order[int(slot)] = int(empty_idx)
    _apply_team_permutation(state, order)
    return True


def _push_backward_from_slot(state: dict[str, Any], slot: int) -> bool:
    empty_idx = _find_closest_empty_behind(state, int(slot))
    if empty_idx is None:
        return False
    order = list(range(len(state["team"])))
    for idx in range(int(empty_idx), int(slot), -1):
        order[idx] = idx - 1
    order[int(slot)] = int(empty_idx)
    _apply_team_permutation(state, order)
    return True


def _make_room_for_summon_slot(state: dict[str, Any], slot: int) -> bool:
    slot = int(slot)
    if slot < 0 or slot >= len(state["team"]):
        return False
    if _is_empty(state["team"][slot]):
        return True
    if _push_forward_from_slot(state, slot):
        return True
    return _push_backward_from_slot(state, slot)


def _apply_pending_summons(
    state: dict[str, Any],
    summon_queue: list[dict[str, Any]],
    notes: list[str],
    *,
    causality: StepCausality,
) -> tuple[bool, str | None]:
    deterministic = True
    reason: str | None = None
    if not summon_queue:
        return deterministic, reason

    def _refund_counter_on_fail(req: dict[str, Any]) -> None:
        payload = req.get("counter_refund")
        if not isinstance(payload, dict):
            return
        trigger = str(payload.get("trigger", "")).strip()
        if not trigger:
            return
        try:
            team_index = int(payload.get("team_index"))
        except (TypeError, ValueError):
            return
        try:
            amount = max(0, int(payload.get("amount", 1)))
        except (TypeError, ValueError):
            amount = 1
        new_value = decrement_ability_counter(state, trigger, team_index, amount=amount)
        notes.append(f"ability_counter_refund:{trigger}:team={team_index}:new={new_value}")

    for req in summon_queue:
        if not isinstance(req, dict):
            continue
        pet_id = str(req.get("pet_id", ""))
        if pet_id in {"", "pet-none"}:
            continue
        # A2.1: a real pending summon is an attempt to write team occupancy,
        # `pet_id` and `level`, and it may push friends around
        # (`_make_room_for_summon_slot`) to make space. All three are in
        # `legal_actions`'s read-set, so this whole resolution is structural
        # whether or not the request ends up landing.
        #
        # A4 rev 2: a cut if anything drew earlier in this step. The first
        # cut of A4 argued this site was never causal because the queue's
        # CONTENTS were settled where the summon was queued. That was right
        # about the contents and wrong about the occupancy: which slot this
        # lands in depends on which slots are still empty, and the review
        # reproduced a draw deciding exactly that.
        causality.note_structural()
        try:
            preferred_idx = req.get("target_team_index")
            if preferred_idx is not None:
                preferred_idx = int(preferred_idx)
        except (TypeError, ValueError):
            preferred_idx = None

        spawn_idx: int | None = None
        if preferred_idx is not None:
            if preferred_idx < 0 or preferred_idx >= len(state["team"]):
                notes.append(f"summon_skipped:invalid_target_slot:{pet_id}:{preferred_idx}")
                _refund_counter_on_fail(req)
                continue
            spawn_idx = int(preferred_idx)
            if not _make_room_for_summon_slot(state, spawn_idx):
                notes.append(f"summon_skipped:no_room_at_target:{pet_id}:team={spawn_idx}")
                _refund_counter_on_fail(req)
                continue
        else:
            spawn_idx = _first_empty_idx(state)

        if spawn_idx is None:
            notes.append(f"summon_skipped:team_full:{pet_id}")
            _refund_counter_on_fail(req)
            continue
        if not _is_empty(state["team"][int(spawn_idx)]):
            notes.append(f"summon_skipped:target_not_empty_after_shift:{pet_id}:team={spawn_idx}")
            _refund_counter_on_fail(req)
            continue

        level = max(1, min(3, int(req.get("level", 1))))
        default_exp = 0 if level == 1 else (2 if level == 2 else 5)
        exp = int(req.get("exp", default_exp))
        equipment_id = req.get("equipment_id")
        _set_team_slot_to_pet(
            state,
            team_index=int(spawn_idx),
            pet_id=pet_id,
            attack=int(req.get("attack", 1)),
            health=int(req.get("health", 1)),
            level=level,
            exp=exp,
            equipment_id=(str(equipment_id) if equipment_id else None),
        )
        status_effects = [str(x) for x in req.get("status_effects", [])] if isinstance(req.get("status_effects"), list) else []
        if status_effects:
            slot = state["team"][int(spawn_idx)]
            slot["status_effects"] = sorted(set(status_effects))
        notes.append(
            f"summon_applied:{pet_id}:team={spawn_idx}:atk={int(req.get('attack', 1))}:hp={int(req.get('health', 1))}:lvl={level}"
        )
        ability_deterministic, ability_reason = _emit_summon_events(
            state,
            team_index=int(spawn_idx),
            pet_id=pet_id,
            level=level,
            notes=notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            reason = _merge_stochastic_reason(reason, ability_reason)
    return deterministic, reason


def _resolve_shop_hurt_faint_chain(
    state: dict[str, Any], notes: list[str], *, causality: StepCausality
) -> tuple[bool, str | None]:
    """Resolve shop-side hurt/faint trigger cascades (e.g. sleeping pill).

    A faint CLEARS a team slot and a summon FILLS one, so a chain that actually
    fainted or summoned anything wrote `legal_actions`'s read-set and says so
    through `causality.note_structural`. Whether that is a CUT depends on
    whether anything drew earlier in the step -- see `StepCausality`. The chain
    shares the caller's taint object rather than owning one, which is the whole
    point of A4 revision 2: the draw that decides who faints usually happens in
    a runtime that has been torn down by the time this loop clears the slot.
    """
    deterministic = True
    reason: str | None = None
    hurt_queue: dict[int, int] = {}
    max_steps = SHOP_FAINT_CHAIN_MAX_STEPS
    steps = 0
    # A4 rev 2. A draw that lands INSIDE a live cascade is a cut whether or not
    # the realised outcome went on to remove anybody.
    #
    # The write-based rule alone is outcome-dependent here, and the review's
    # own frequency probe found the case: a random stat buff healed every pet
    # that was about to die, so no slot was cleared after the draw and nothing
    # signalled a chance node -- while the other side of the same draw removed
    # a pet and changed the legal mask. The search's imagined board is one
    # sample of that draw and the real board is another, so they can disagree
    # on team occupancy with no write to point at.
    #
    # This cascade exists to decide who is removed, and removal is the
    # read-set. Any draw inside it can steer that. Outside a cascade the
    # ordinary write-based rule still applies.
    armed_at_entry = bool(causality.armed)

    while steps < max_steps:
        steps += 1

        # Resolve pending hurt events first (matching SAP queue priority model).
        while True:
            pending_hits: list[int] = []
            for idx, count in sorted(hurt_queue.items(), key=lambda row: int(row[0])):
                for _ in range(max(0, int(count))):
                    pending_hits.append(int(idx))
            hurt_queue.clear()
            if not pending_hits:
                break

            hurt_events: list[AbilityEvent] = []
            for idx in pending_hits:
                if idx < 0 or idx >= len(state["team"]):
                    continue
                slot = state["team"][idx]
                pet_id = slot.get("pet_id")
                if pet_id is None:
                    continue
                actor_pet_id, actor_level = _effective_actor_for_trigger(
                    state,
                    idx,
                    TRIGGER_HURT,
                    str(pet_id),
                    int(slot.get("level", 1)),
                )
                hurt_events.append(
                    AbilityEvent(
                        trigger=TRIGGER_HURT,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=idx,
                        payload={"hurt_queue": hurt_queue},
                    )
                )
                for other_idx, other_slot in enumerate(state["team"]):
                    other_pet_id = other_slot.get("pet_id")
                    if other_pet_id is None or int(other_idx) == int(idx):
                        continue
                    counter_actor_pet_id, counter_actor_level = _effective_actor_for_trigger(
                        state,
                        other_idx,
                        TRIGGER_FRIENDS_HURT_COUNTER,
                        str(other_pet_id),
                        int(other_slot.get("level", 1)),
                    )
                    hurt_events.append(
                        AbilityEvent(
                            trigger=TRIGGER_FRIENDS_HURT_COUNTER,
                            actor_pet_id=counter_actor_pet_id,
                            actor_level=counter_actor_level,
                            actor_team_index=other_idx,
                            payload={"hurt_team_index": int(idx)},
                        )
                    )

            ability_deterministic, ability_reason = _run_ability_events(
                state,
                hurt_events,
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                reason = _merge_stochastic_reason(reason, ability_reason)

        faint_indices = [
            idx
            for idx, slot in enumerate(state["team"])
            if slot.get("pet_id") is not None and int(slot.get("health", 0)) <= 0
        ]

        if not faint_indices and not hurt_queue:
            break
        if not faint_indices:
            continue

        # Pre-removal faint event pass (Faint/FriendAheadFaints/FriendFaints).
        pre_events: list[AbilityEvent] = []
        summon_queues: dict[int, list[dict[str, Any]]] = {
            int(idx): [] for idx in faint_indices
        }
        for idx in faint_indices:
            slot = state["team"][idx]
            pet_id = slot.get("pet_id")
            if pet_id is None:
                continue
            pet_id = str(pet_id)
            pet_level = int(slot.get("level", 1))
            pet_attack = int(slot.get("attack", 0))
            summon_queue = summon_queues.setdefault(int(idx), [])

            actor_pet_id, actor_level = _effective_actor_for_trigger(
                state,
                idx,
                TRIGGER_FAINT,
                pet_id,
                pet_level,
            )
            pre_events.append(
                AbilityEvent(
                    trigger=TRIGGER_FAINT,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={
                        "hurt_queue": hurt_queue,
                        "summon_queue": summon_queue,
                        "fainted_team_index": int(idx),
                        "actor_attack": int(pet_attack),
                    },
                )
            )
            for other_idx, other_slot in enumerate(state["team"]):
                other_pet_id = other_slot.get("pet_id")
                if other_pet_id is None:
                    continue
                if int(other_idx) == int(idx):
                    continue
                friend_actor_pet_id, friend_actor_level = _effective_actor_for_trigger(
                    state,
                    other_idx,
                    TRIGGER_FRIEND_FAINTS,
                    str(other_pet_id),
                    int(other_slot.get("level", 1)),
                )
                pre_events.append(
                    AbilityEvent(
                        trigger=TRIGGER_FRIEND_FAINTS,
                        actor_pet_id=friend_actor_pet_id,
                        actor_level=friend_actor_level,
                        actor_team_index=other_idx,
                        payload={
                            "hurt_queue": hurt_queue,
                            "summon_queue": summon_queue,
                            "fainted_team_index": int(idx),
                            "fainted_pet_id": pet_id,
                        },
                    )
                )

            friend_ahead_idx = int(idx) + 1
            if friend_ahead_idx < len(state["team"]):
                ahead_slot = state["team"][friend_ahead_idx]
                ahead_pet_id = ahead_slot.get("pet_id")
                if ahead_pet_id is not None:
                    ahead_actor_pet_id, ahead_actor_level = _effective_actor_for_trigger(
                        state,
                        friend_ahead_idx,
                        TRIGGER_FRIEND_AHEAD_FAINTS,
                        str(ahead_pet_id),
                        int(ahead_slot.get("level", 1)),
                    )
                    pre_events.append(
                        AbilityEvent(
                            trigger=TRIGGER_FRIEND_AHEAD_FAINTS,
                            actor_pet_id=ahead_actor_pet_id,
                            actor_level=ahead_actor_level,
                            actor_team_index=friend_ahead_idx,
                            payload={
                                "hurt_queue": hurt_queue,
                                "summon_queue": summon_queue,
                                "fainted_team_index": int(idx),
                                "fainted_pet_id": pet_id,
                            },
                        )
                    )

        ability_deterministic, ability_reason = _run_ability_events(
            state,
            pre_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            reason = _merge_stochastic_reason(reason, ability_reason)

        # Post-removal pass in slot order (SAP removeDeadPets order), then resolve summons.
        for idx in sorted(int(i) for i in faint_indices):
            slot = state["team"][idx]
            pet_id = slot.get("pet_id")
            if pet_id is None or int(slot.get("health", 0)) > 0:
                continue
            pet_id = str(pet_id)
            pet_level = int(slot.get("level", 1))
            pet_attack = int(slot.get("attack", 0))
            status_effects = [str(x) for x in slot.get("status_effects", [])]
            summon_queue = summon_queues.setdefault(int(idx), [])

            actor_pet_id, actor_level = _effective_actor_for_trigger(
                state,
                idx,
                TRIGGER_AFTER_FAINT,
                pet_id,
                pet_level,
            )
            # A2.1: clearing a slot writes team occupancy and `pet_id`.
            #
            # A4 rev 2, and this is the line the review's F1 turned on. When
            # nothing has drawn this step, reaching zero health is a
            # deterministic consequence of the action just committed (a
            # sleeping pill, a chili) -- the search planned through it and
            # there is nothing to stop for. But health is exactly what a random
            # stat buff writes, so once this step HAS drawn, which slot ends up
            # empty can be the dice's doing. The taint is per-step precisely so
            # it is still alive here, runtimes after the draw that set it.
            _clear_team_slot(state, idx)
            causality.note_structural()

            ability_deterministic, ability_reason = _run_ability_events(
                state,
                [
                    AbilityEvent(
                        trigger=TRIGGER_AFTER_FAINT,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=idx,
                        payload={
                            "hurt_queue": hurt_queue,
                            "summon_queue": summon_queue,
                            "fainted_team_index": int(idx),
                            "fainted_pet_id": pet_id,
                            "actor_attack": int(pet_attack),
                        },
                    )
                ],
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                reason = _merge_stochastic_reason(reason, ability_reason)

            # For same trigger timing, pet-triggered summons resolve before equipment summons.
            if "status-honey-bee" in status_effects:
                summon_queue.append(
                    {
                        "pet_id": "pet-bee",
                        "attack": 1,
                        "health": 1,
                        "level": 1,
                        "exp": 0,
                        "target_team_index": int(idx),
                    },
                )
                notes.append(f"status_trigger:status-honey-bee:summon=pet-bee:team={idx}")
            if "status-extra-life" in status_effects:
                summon_queue.append(
                    {
                        "pet_id": pet_id,
                        "attack": 1,
                        "health": 1,
                        "level": 1,
                        "exp": 0,
                        "target_team_index": int(idx),
                    },
                )
                notes.append(f"status_trigger:status-extra-life:respawn={pet_id}:team={idx}")

            ability_deterministic, ability_reason = _apply_pending_summons(
                state, summon_queue, notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                reason = _merge_stochastic_reason(reason, ability_reason)

    if steps >= max_steps:
        notes.append("unsupported_effect:shop_hurt_faint_loop_limit")
        # Conservative default, as everywhere else `unsupported_effect` is
        # raised: an unmodelled cascade could have written anything.
        causality.force_cut()
        return False, _merge_stochastic_reason(reason, "unsupported_effect")
    if causality.armed and not armed_at_entry:
        causality.cut = True
    return deterministic, reason


def resolve_end_turn_pre_battle(state: dict[str, Any]) -> StepOutcome:
    """Apply end-turn shop effects that must happen before battle."""
    new_state = copy.deepcopy(state)
    notes: list[str] = []
    deterministic = True
    stochastic_reason: str | None = None
    causality = StepCausality()

    _reconcile_team_stat_fields(new_state)

    end_events: list[AbilityEvent] = []
    for idx, slot in enumerate(new_state["team"]):
        pet_id = slot.get("pet_id")
        if pet_id is None:
            continue
        actor_pet_id, actor_level = _effective_actor_for_trigger(
            new_state,
            idx,
            TRIGGER_END_OF_TURN,
            str(pet_id),
            int(slot.get("level", 1)),
        )
        end_events.append(
            AbilityEvent(
                trigger=TRIGGER_END_OF_TURN,
                actor_pet_id=actor_pet_id,
                actor_level=actor_level,
                actor_team_index=idx,
            )
        )
    ability_deterministic, ability_reason = _run_ability_events(
        new_state,
        end_events,
        notes,
        causality=causality,
    )
    if not ability_deterministic:
        deterministic = False
        stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    # Birthday Cake perk: end turn gain +1 sell value.
    cake_buffs = 0
    for slot in new_state.get("team", []):
        if slot.get("pet_id") is None:
            continue
        if str(slot.get("equipment_id", "")) != "food-birthday-cake":
            continue
        slot["sell_value"] = _cap_stat(int(slot.get("sell_value", 1)) + 1)
        cake_buffs += 1
    if cake_buffs > 0:
        notes.append(f"equipment_trigger:food-birthday-cake:end_turn:count={cake_buffs}")

    _reconcile_team_stat_fields(new_state)
    _enforce_shop_schema_limit(new_state, notes)
    return StepOutcome(
        new_state,
        True,
        deterministic,
        stochastic_reason,
        notes,
        bool(causality.cut),
    )


def resolve_end_turn_post_battle(state: dict[str, Any]) -> StepOutcome:
    """Advance from post-battle state into the next shop turn."""
    new_state = copy.deepcopy(state)
    notes: list[str] = []
    deterministic = True
    stochastic_reason: str | None = None
    causality = StepCausality()

    _reconcile_team_stat_fields(new_state)

    # Temporary battle-only stats expire between battle end and next turn start.
    cleared_temp = clear_temporary_team_stats(new_state)
    if cleared_temp > 0:
        notes.append(f"temporary_stats_cleared:{cleared_temp}")

    new_state["turn"] = int(new_state.get("turn", 1)) + 1
    new_state["gold"] = 10
    reset_ability_counters(new_state)

    _rebuild_shop_for_roll(new_state, notes)
    cat = load_turtle_catalog()
    tier = tier_for_turn(int(new_state["turn"]))
    rng = _rng_from_state(new_state)
    _roll_shop_slots(new_state, cat, rng, tier)
    _sort_shop_by_tier(new_state, cat)
    _reseed_meta(rng, new_state)
    deterministic = False
    # A2.1: a roll rebuilds the shop -- `slot_type`, `item_id`, `cost`,
    # `frozen` and the slot count are all read by `legal_actions`.
    # A4: causal by identity -- the draw IS the write.
    causality.arm()
    causality.note_structural()
    stochastic_reason = _merge_stochastic_reason(stochastic_reason, "roll_randomness")

    start_events: list[AbilityEvent] = []
    for idx, slot in enumerate(new_state["team"]):
        pet_id = slot.get("pet_id")
        if pet_id is None:
            continue
        actor_pet_id, actor_level = _effective_actor_for_trigger(
            new_state,
            idx,
            TRIGGER_START_OF_TURN,
            str(pet_id),
            int(slot.get("level", 1)),
        )
        start_events.append(
            AbilityEvent(
                trigger=TRIGGER_START_OF_TURN,
                actor_pet_id=actor_pet_id,
                actor_level=actor_level,
                actor_team_index=idx,
            )
        )
    ability_deterministic, ability_reason = _run_ability_events(
        new_state,
        start_events,
        notes,
        causality=causality,
    )
    if not ability_deterministic:
        deterministic = False
        stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    _reconcile_team_stat_fields(new_state)
    _enforce_shop_schema_limit(new_state, notes)
    return StepOutcome(
        new_state,
        True,
        deterministic,
        stochastic_reason,
        notes,
        bool(causality.cut),
    )


def apply_action(state: dict[str, Any], action: dict[str, Any]) -> StepOutcome:
    new_state = copy.deepcopy(state)
    notes: list[str] = []
    deterministic = True
    stochastic_reason: str | None = None
    causality = StepCausality()

    legal, reason = check_legal(new_state, action)
    if not legal:
        notes.append(f"illegal:{reason}")
        return StepOutcome(new_state, False, True, None, notes, False)

    _reconcile_team_stat_fields(new_state)
    action_type = action["type"]
    if action_type == "BUY_PET":
        shop_pos, shop_slot = _find_shop_slot(new_state, action["shop_index"])
        assert shop_pos is not None and shop_slot is not None
        bought_pet_id = str(shop_slot["item_id"])
        purchased_link_id = shop_slot.get("link_id")
        cat = load_turtle_catalog()
        base_atk, base_hp = _pet_base_stats(cat, bought_pet_id)
        pet_atk = int(shop_slot.get("attack", base_atk))
        pet_hp = int(shop_slot.get("health", base_hp))
        ti = action["team_index"]
        slot = _team_slot(new_state, ti)
        slot.update(
            pet_id=shop_slot["item_id"],
            attack=_cap_stat(pet_atk),
            health=_cap_stat(pet_hp),
            sell_value=1,
            level=1,
            exp=0,
            equipment_id=None,
            status_effects=[],
        )
        clear_ability_counters_for_team_index(new_state, int(ti))
        _clear_parrot_copy_for_team_index(new_state, int(ti))
        new_state["gold"] -= int(shop_slot["cost"])
        del new_state["shop"][shop_pos]
        new_state["shop"] = _partition_shop_pet_food(new_state["shop"])
        _reindex_shop(new_state["shop"])
        if purchased_link_id:
            _remove_linked_shop_slots(new_state, str(purchased_link_id), notes)
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            [
                AbilityEvent(
                    trigger=TRIGGER_BUY,
                    actor_pet_id=bought_pet_id,
                    actor_level=1,
                    actor_team_index=ti,
                )
            ],
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

        buy_friend_events: list[AbilityEvent] = []
        for idx, team_slot in enumerate(new_state["team"]):
            pet_id = team_slot.get("pet_id")
            if pet_id is None:
                continue
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                idx,
                TRIGGER_BUY_FRIEND,
                str(pet_id),
                int(team_slot.get("level", 1)),
            )
            buy_friend_events.append(
                AbilityEvent(
                    trigger=TRIGGER_BUY_FRIEND,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={"trigger_pet_id": bought_pet_id, "trigger_team_index": int(ti)},
                )
            )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            buy_friend_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)
        bought_pet_tier = _pet_tier(cat, bought_pet_id)
        if bought_pet_tier == 1:
            buy_tier1_events: list[AbilityEvent] = []
            for idx, team_slot in enumerate(new_state["team"]):
                pet_id = team_slot.get("pet_id")
                if pet_id is None:
                    continue
                actor_pet_id, actor_level = _effective_actor_for_trigger(
                    new_state,
                    idx,
                    TRIGGER_BUY_TIER1_PET,
                    str(pet_id),
                    int(team_slot.get("level", 1)),
                )
                buy_tier1_events.append(
                    AbilityEvent(
                        trigger=TRIGGER_BUY_TIER1_PET,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=idx,
                        payload={
                            "trigger_pet_id": bought_pet_id,
                            "trigger_team_index": int(ti),
                            "trigger_pet_tier": int(bought_pet_tier),
                        },
                    )
                )
            ability_deterministic, ability_reason = _run_ability_events(
                new_state,
                buy_tier1_events,
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

        # Buying onto board counts as a summon event. Buying to combine does not.
        summon_events: list[AbilityEvent] = [
            AbilityEvent(
                trigger=TRIGGER_SUMMONED,
                actor_pet_id=bought_pet_id,
                actor_level=1,
                actor_team_index=int(ti),
                payload={"trigger_pet_id": bought_pet_id, "trigger_team_index": int(ti)},
            )
        ]
        for idx, team_slot in enumerate(new_state["team"]):
            pet_id = team_slot.get("pet_id")
            if pet_id is None:
                continue
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                idx,
                TRIGGER_FRIEND_SUMMONED,
                str(pet_id),
                int(team_slot.get("level", 1)),
            )
            summon_events.append(
                AbilityEvent(
                    trigger=TRIGGER_FRIEND_SUMMONED,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={"trigger_pet_id": bought_pet_id, "trigger_team_index": int(ti)},
                )
            )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            summon_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    elif action_type == "BUY_COMBINE":
        shop_pos, shop_slot = _find_shop_slot(new_state, action["shop_index"])
        assert shop_pos is not None and shop_slot is not None
        bought_pet_id = str(shop_slot["item_id"])
        purchased_link_id = shop_slot.get("link_id")
        ti = action["team_index"]
        dst = _team_slot(new_state, ti)
        dst_level_before = int(dst.get("level", 1))
        src = {
            "pet_id": shop_slot["item_id"],
            "attack": int(dst["attack"]),
            "health": int(dst["health"]),
            "sell_value": 1,
            "level": 1,
            "exp": 0,
            "equipment_id": None,
            "status_effects": [],
        }
        cat = load_turtle_catalog()
        base_atk, base_hp = _pet_base_stats(cat, bought_pet_id)
        src["attack"] = _cap_stat(int(shop_slot.get("attack", base_atk)))
        src["health"] = _cap_stat(int(shop_slot.get("health", base_hp)))
        new_state["gold"] -= int(shop_slot["cost"])
        del new_state["shop"][shop_pos]
        new_state["shop"] = _partition_shop_pet_food(new_state["shop"])
        _reindex_shop(new_state["shop"])
        if purchased_link_id:
            _remove_linked_shop_slots(new_state, str(purchased_link_id), notes)
        # Buy-combine levels the destination pet up; its own buy ability
        # (e.g. Cow's milk, Otter's buff) must fire at the RESULTING level, not
        # the pre-combine level. Mirror _combine_pet_stats' exp math to predict it.
        combine_result_level = _level_from_exp(
            min(5, max(0, min(5, int(dst.get("exp", 0)))) + 1 + max(0, int(src.get("exp", 0))))
        )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            [
                AbilityEvent(
                    trigger=TRIGGER_BUY,
                    actor_pet_id=bought_pet_id,
                    actor_level=combine_result_level,
                    actor_team_index=int(ti),
                )
            ],
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

        leveled_up = _combine_pet_stats(dst, src)
        _merge_combine_ability_counters(
            new_state,
            dst_team_index=int(ti),
            src_team_index=None,
            leveled_up=bool(leveled_up),
        )
        if leveled_up:
            rng = _rng_from_state(new_state)
            added = _add_levelup_shop_slot(new_state, cat, rng, notes)
            if added:
                _reseed_meta(rng, new_state)
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, "levelup_reward_randomness")
                # A2.1: `_add_levelup_shop_slot` ADDS a shop slot.
                # A4: causal by identity -- the draw picks the slot it adds.
                causality.arm()
                causality.note_structural()
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                int(ti),
                TRIGGER_LEVEL_UP,
                str(dst.get("pet_id", "")),
                int(dst.get("level", 1)),
            )
            ability_deterministic, ability_reason = _run_ability_events(
                new_state,
                [
                    AbilityEvent(
                        trigger=TRIGGER_LEVEL_UP,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=ti,
                        payload={"ability_level": dst_level_before},
                    )
                ],
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

        buy_friend_events: list[AbilityEvent] = []
        for idx, team_slot in enumerate(new_state["team"]):
            pet_id = team_slot.get("pet_id")
            if pet_id is None:
                continue
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                idx,
                TRIGGER_BUY_FRIEND,
                str(pet_id),
                int(team_slot.get("level", 1)),
            )
            buy_friend_events.append(
                AbilityEvent(
                    trigger=TRIGGER_BUY_FRIEND,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={"trigger_pet_id": bought_pet_id, "trigger_team_index": int(ti)},
                )
            )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            buy_friend_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)
        bought_pet_tier = _pet_tier(cat, bought_pet_id)
        if bought_pet_tier == 1:
            buy_tier1_events: list[AbilityEvent] = []
            for idx, team_slot in enumerate(new_state["team"]):
                pet_id = team_slot.get("pet_id")
                if pet_id is None:
                    continue
                actor_pet_id, actor_level = _effective_actor_for_trigger(
                    new_state,
                    idx,
                    TRIGGER_BUY_TIER1_PET,
                    str(pet_id),
                    int(team_slot.get("level", 1)),
                )
                buy_tier1_events.append(
                    AbilityEvent(
                        trigger=TRIGGER_BUY_TIER1_PET,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=idx,
                        payload={
                            "trigger_pet_id": bought_pet_id,
                            "trigger_team_index": int(ti),
                            "trigger_pet_tier": int(bought_pet_tier),
                        },
                    )
                )
            ability_deterministic, ability_reason = _run_ability_events(
                new_state,
                buy_tier1_events,
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    elif action_type == "BUY_FOOD":
        shop_pos, shop_slot = _find_shop_slot(new_state, action["shop_index"])
        assert shop_pos is not None and shop_slot is not None
        item_id = str(shop_slot["item_id"])
        new_state["gold"] -= int(shop_slot["cost"])
        del new_state["shop"][shop_pos]
        new_state["shop"] = _partition_shop_pet_food(new_state["shop"])
        _reindex_shop(new_state["shop"])
        target_indices: list[int]
        rng = _rng_from_state(new_state)
        if item_id in NO_TARGET_FOODS:
            target_indices = []
        elif item_id in RANDOM_TARGET_FOODS:
            target_indices, random_used = _random_food_target_indices(new_state, item_id, rng)
            if random_used:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, "food_random_targeting")
                _reseed_meta(rng, new_state)
                # A4 rev 2: this draw picks WHICH friends the food reaches,
                # so it arms the step like any other draw and every read-set
                # write below it is a cut. The first cut of A4 spelled this as
                # a conjunction at the end of the branch, which the review (F4)
                # correctly called not-a-scope; with a per-step taint the
                # ordinary mechanism covers it and this site is no longer
                # special.
                #
                # No Turtle random-target food can reach a read-set write today
                # -- salad bowl, sushi and pizza are pure stat buffs -- so this
                # is a live hook with no live data behind it. A test injects
                # one rather than leaving the hook merely asserted.
                causality.arm()
        else:
            target_indices = [int(action["team_index"])]

        notes.append(
            f"food_targets:{item_id}:"
            + (",".join(str(int(idx)) for idx in target_indices) or "none")
        )
        base_attack, base_health, buff_attack, buff_health = _food_effect_stats(shop_slot)
        target_payload: int | list[int] | None
        if len(target_indices) == 0:
            target_payload = None
        elif len(target_indices) == 1:
            target_payload = int(target_indices[0])
        else:
            target_payload = [int(x) for x in target_indices]

        # PurchaseFood trigger path (Cat) mutates pending food buff before feeding.
        new_state.setdefault("meta", {})
        new_state["meta"][PENDING_FOOD_CTX_KEY] = {
            "item_id": item_id,
            "base_attack": int(base_attack),
            "base_health": int(base_health),
            "attack": int(buff_attack),
            "health": int(buff_health),
        }
        purchase_food_events: list[AbilityEvent] = []
        for idx, team_slot in enumerate(new_state["team"]):
            pet_id = team_slot.get("pet_id")
            if pet_id is None:
                continue
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                idx,
                TRIGGER_PURCHASE_FOOD,
                str(pet_id),
                int(team_slot.get("level", 1)),
            )
            purchase_food_events.append(
                AbilityEvent(
                    trigger=TRIGGER_PURCHASE_FOOD,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={"food_item_id": item_id, "target_team_index": target_payload},
                )
            )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            purchase_food_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)
        pending_food = new_state.get("meta", {}).pop(PENDING_FOOD_CTX_KEY, None)
        if isinstance(pending_food, dict):
            buff_attack = int(pending_food.get("attack", buff_attack))
            buff_health = int(pending_food.get("health", buff_health))

        food_supported = True
        needs_faint_chain = False

        if item_id == "food-canned-food":
            # The granted amount is `buff_attack`/`buff_health`, i.e. the
            # pending-food context AFTER the PurchaseFood triggers above have
            # had their say.  It is 1/1 bare (from `FOOD_STAT_BUFFS`) and
            # larger when a Cat multiplied it.  A literal 1 here would make
            # canned food the one stats food Cat cannot reach.
            add_at = int(buff_attack)
            add_hp = int(buff_health)
            old_at, old_hp = _shop_pet_bonus(new_state)
            new_at = int(old_at) + add_at
            new_hp = int(old_hp) + add_hp
            # The PERSISTENT bonus carries the boosted amount, so a Cat-boosted
            # canned food keeps paying on every later roll of this game.
            _set_shop_pet_bonus(new_state, new_at, new_hp)
            touched = 0
            for s in new_state.get("shop", []):
                if s.get("slot_type") != "pet":
                    continue
                if str(s.get("item_id", "")) in {"", "pet-none"}:
                    continue
                s["attack"] = _cap_stat(int(s.get("attack", 0)) + add_at)
                s["health"] = _cap_stat(int(s.get("health", 0)) + add_hp)
                s["at"] = int(s.get("at", old_at)) + add_at
                s["hp"] = int(s.get("hp", old_hp)) + add_hp
                touched += 1
            notes.append(
                f"food_effect:food-canned-food:shop_pet_bonus=+{add_at}/+{add_hp}:touched={touched}"
            )
        else:
            levelup_rng: random.Random | None = None
            cat = load_turtle_catalog()
            for target_idx in target_indices:
                slot = _team_slot(new_state, int(target_idx))

                if item_id == "food-sleeping-pill":
                    # sapai behavior: pill makes target immediately faint.
                    slot["health"] = -1000
                    needs_faint_chain = True
                elif item_id == "food-chocolate":
                    # Chocolate grants +1 experience (and therefore +1/+1 via gain_experience).
                    level_before = int(slot.get("level", 1))
                    leveled_up = _gain_experience(slot, amount=1)
                    if leveled_up:
                        clear_ability_counters_for_team_index(new_state, int(target_idx))
                        if levelup_rng is None:
                            levelup_rng = _rng_from_state(new_state)
                        added = _add_levelup_shop_slot(new_state, cat, levelup_rng, notes)
                        if added:
                            _reseed_meta(levelup_rng, new_state)
                            deterministic = False
                            stochastic_reason = _merge_stochastic_reason(
                                stochastic_reason, "levelup_reward_randomness"
                            )
                            # A2.1: `_add_levelup_shop_slot` ADDS a shop slot.
                            # A4: causal by identity -- the draw picks the slot it adds.
                            causality.arm()
                            causality.note_structural()
                        actor_pet_id, actor_level = _effective_actor_for_trigger(
                            new_state,
                            int(target_idx),
                            TRIGGER_LEVEL_UP,
                            str(slot.get("pet_id", "")),
                            int(slot.get("level", 1)),
                        )
                        ability_deterministic, ability_reason = _run_ability_events(
                            new_state,
                            [
                                AbilityEvent(
                                    trigger=TRIGGER_LEVEL_UP,
                                    actor_pet_id=actor_pet_id,
                                    actor_level=actor_level,
                                    actor_team_index=int(target_idx),
                                    payload={"ability_level": level_before},
                                )
                            ],
                            notes,
                            causality=causality,
                        )
                        if not ability_deterministic:
                            deterministic = False
                            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)
                elif item_id in EQUIPMENT_FOOD_IDS:
                    _apply_equipment_to_pet_slot(slot, item_id)
                elif item_id in FOOD_STAT_BUFFS:
                    # Cupcake is temporary in shop phase; all other stat foods are permanent.
                    temporary = item_id in {"food-cupcake"}
                    add_slot_stats(slot, attack=int(buff_attack), health=int(buff_health), temporary=temporary)
                else:
                    food_supported = False

                # Sleeping pill is not treated as "eating food" for shop triggers.
                if item_id != "food-sleeping-pill":
                    eats_food_events, friendly_food_events = build_food_trigger_event_groups(
                        new_state, int(target_idx), item_id
                    )
                    ability_deterministic, ability_reason = _run_ability_events(
                        new_state,
                        eats_food_events,
                        notes,
                        causality=causality,
                    )
                    if not ability_deterministic:
                        deterministic = False
                        stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

                    ability_deterministic, ability_reason = _run_ability_events(
                        new_state,
                        friendly_food_events,
                        notes,
                        causality=causality,
                    )
                    if not ability_deterministic:
                        deterministic = False
                        stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)
                else:
                    notes.append("food_effect:food-sleeping-pill:no_eat_triggers")

                if item_id == "food-sleeping-pill" and slot.get("pet_id") is not None:
                    # Guard against food-trigger buffs (e.g. rabbit) reviving the pill target
                    # before the faint resolver runs.
                    slot["health"] = -1000

        if not food_supported:
            notes.append("unsupported_effect:food_effect_not_simulated")
            deterministic = False
            # Conservative default: an unmodelled effect could have written
            # anything, including the read-set, and for any reason.
            causality.force_cut()
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, "unsupported_effect")

        if needs_faint_chain or any(
            slot.get("pet_id") is not None and int(slot.get("health", 0)) <= 0 for slot in new_state["team"]
        ):
            chain_deterministic, chain_reason = _resolve_shop_hurt_faint_chain(
                new_state, notes,
                causality=causality,
            )
            if not chain_deterministic:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, chain_reason)

    elif action_type == "SELL":
        ti = action["team_index"]
        slot = _team_slot(new_state, ti)
        sold_pet_id = str(slot.get("pet_id", ""))
        sold_level = int(slot.get("level", 1))
        sell_actor_pet_id, sell_actor_level = _effective_actor_for_trigger(
            new_state,
            int(ti),
            TRIGGER_SELL,
            sold_pet_id,
            sold_level,
        )
        sell_gain = _default_sell_value(slot)
        slot.update(
            pet_id=None,
            attack=0,
            health=0,
            sell_value=0,
            level=1,
            exp=0,
            equipment_id=None,
            status_effects=[],
        )
        clear_ability_counters_for_team_index(new_state, int(ti))
        _clear_parrot_copy_for_team_index(new_state, int(ti))
        new_state["gold"] += int(sell_gain)
        sell_events: list[AbilityEvent] = [
            AbilityEvent(
                trigger=TRIGGER_SELL,
                actor_pet_id=sell_actor_pet_id,
                actor_level=sell_actor_level,
                actor_team_index=ti,
            )
        ]
        for idx, team_slot in enumerate(new_state["team"]):
            pet_id = team_slot.get("pet_id")
            if pet_id is None:
                continue
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                idx,
                TRIGGER_FRIEND_SOLD,
                str(pet_id),
                int(team_slot.get("level", 1)),
            )
            sell_events.append(
                AbilityEvent(
                    trigger=TRIGGER_FRIEND_SOLD,
                    actor_pet_id=actor_pet_id,
                    actor_level=actor_level,
                    actor_team_index=idx,
                    payload={
                        "sold_pet_id": sold_pet_id,
                        "sold_team_index": int(ti),
                        "sold_level": sold_level,
                    },
                )
            )
        ability_deterministic, ability_reason = _run_ability_events(
            new_state,
            sell_events,
            notes,
            causality=causality,
        )
        if not ability_deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    elif action_type == "COMBINE":
        src = _team_slot(new_state, action["src_team_index"])
        dst = _team_slot(new_state, action["dst_team_index"])
        src_level_before = int(src.get("level", 1))
        dst_level_before = int(dst.get("level", 1))
        cat = load_turtle_catalog()
        leveled_up = _combine_pet_stats(dst, src)
        _merge_combine_ability_counters(
            new_state,
            dst_team_index=int(action["dst_team_index"]),
            src_team_index=int(action["src_team_index"]),
            leveled_up=bool(leveled_up),
        )
        src.update(
            pet_id=None,
            attack=0,
            health=0,
            sell_value=0,
            level=1,
            exp=0,
            equipment_id=None,
            status_effects=[],
        )
        clear_ability_counters_for_team_index(new_state, int(action["src_team_index"]))
        _clear_parrot_copy_for_team_index(new_state, int(action["src_team_index"]))
        if leveled_up:
            if src_level_before == 2 and dst_level_before == 2:
                notes.append("levelup_reward_suppressed:double_level2_combine")
            else:
                rng = _rng_from_state(new_state)
                added = _add_levelup_shop_slot(new_state, cat, rng, notes)
                if added:
                    _reseed_meta(rng, new_state)
                    deterministic = False
                    stochastic_reason = _merge_stochastic_reason(stochastic_reason, "levelup_reward_randomness")
                    # A2.1: `_add_levelup_shop_slot` ADDS a shop slot.
                    # A4: causal by identity -- the draw picks the slot it adds.
                    causality.arm()
                    causality.note_structural()
            dst_idx = int(action["dst_team_index"])
            actor_pet_id, actor_level = _effective_actor_for_trigger(
                new_state,
                dst_idx,
                TRIGGER_LEVEL_UP,
                str(dst.get("pet_id", "")),
                int(dst.get("level", 1)),
            )
            ability_deterministic, ability_reason = _run_ability_events(
                new_state,
                [
                    AbilityEvent(
                        trigger=TRIGGER_LEVEL_UP,
                        actor_pet_id=actor_pet_id,
                        actor_level=actor_level,
                        actor_team_index=dst_idx,
                        # Canonical direction: the level the result levelled up
                        # FROM is the higher of the two inputs' levels.
                        payload={"ability_level": max(dst_level_before, src_level_before)},
                    )
                ],
                notes,
                causality=causality,
            )
            if not ability_deterministic:
                deterministic = False
                stochastic_reason = _merge_stochastic_reason(stochastic_reason, ability_reason)

    elif action_type == "REORDER":
        # exp13: the same removal as `_apply_team_permutation`, and here the
        # permutation is checked before this branch can run -- `check_legal` has
        # already rejected any REORDER whose `sorted(order) != [0, 1, 2, 3, 4]`
        # (`_legal_reorder`), so every index appears exactly once.
        old = new_state["team"]
        new_state["team"] = [old[i] for i in action["order"]]
        remap_ability_counters_for_reorder(new_state, list(action["order"]))
        _remap_parrot_copy_for_reorder(new_state, list(action["order"]))
        for i, slot in enumerate(new_state["team"]):
            slot["slot_index"] = i

    elif action_type == "FREEZE":
        _pos, slot = _find_shop_slot(new_state, action["shop_index"])
        assert slot is not None
        slot["frozen"] = True

    elif action_type == "UNFREEZE":
        _pos, slot = _find_shop_slot(new_state, action["shop_index"])
        assert slot is not None
        slot["frozen"] = False

    elif action_type == "ROLL":
        _rebuild_shop_for_roll(new_state, notes)
        cat = load_turtle_catalog()
        tier = tier_for_turn(int(new_state["turn"]))
        rng = _rng_from_state(new_state)
        _roll_shop_slots(new_state, cat, rng, tier)
        _sort_shop_by_tier(new_state, cat)
        _reseed_meta(rng, new_state)
        new_state["gold"] = max(0, int(new_state["gold"]) - 1)
        deterministic = False
        # A2.1: the shop is rebuilt, and gold moved.
        # A4: causal by identity -- the draw IS the write.
        causality.arm()
        causality.note_structural()
        stochastic_reason = "roll_randomness"

    elif action_type == "END_TURN":
        pre = resolve_end_turn_pre_battle(new_state)
        new_state = pre.state_after
        notes.extend(pre.notes)
        # A4: `stochastic_structural` on a sub-outcome IS the causal verdict
        # for that half; each half ran with its own `StepCausality`.
        causality.absorb(pre)
        if not pre.deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, pre.stochastic_reason)

        # Standalone engine cannot run battle here yet; continue directly to next shop turn.
        post = resolve_end_turn_post_battle(new_state)
        new_state = post.state_after
        notes.extend(post.notes)
        causality.absorb(post)
        if not post.deterministic:
            deterministic = False
            stochastic_reason = _merge_stochastic_reason(stochastic_reason, post.stochastic_reason)

        notes.append("unsupported_effect:end_turn_battle_not_simulated")
        notes.append("end_turn_no_battle:shop_refresh")
        deterministic = False
        stochastic_reason = _merge_stochastic_reason(stochastic_reason, "end_turn_battle_unknown")

    _reconcile_team_stat_fields(new_state)
    _enforce_shop_schema_limit(new_state, notes)
    return StepOutcome(
        new_state,
        True,
        deterministic,
        stochastic_reason,
        notes,
        bool(causality.cut),
    )
