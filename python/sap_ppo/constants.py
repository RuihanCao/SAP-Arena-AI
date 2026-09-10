"""Project constants and normalized IDs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "schemas"
DATA_DIR = ROOT / "data"

STATE_SCHEMA = SCHEMAS_DIR / "state_v1.json"
ACTION_SCHEMA = SCHEMAS_DIR / "action_v1.json"
TRANSITION_SCHEMA = SCHEMAS_DIR / "transition_v1.json"

ACTION_TYPES = {
    "BUY_PET",
    "BUY_COMBINE",
    "BUY_FOOD",
    "SELL",
    "COMBINE",
    "REORDER",
    "FREEZE",
    "UNFREEZE",
    "ROLL",
    "END_TURN",
}

DEFAULT_SHOP_ITEM_COST = 3

# Item-specific shop cost overrides.
# Anything not listed here uses DEFAULT_SHOP_ITEM_COST.
FOOD_COST_OVERRIDES: dict[str, int] = {
    "food-sleeping-pill": 1,
}

# Minimal baseline food effects for deterministic Step 1 transitions.
FOOD_STAT_BUFFS = {
    "food-apple": (1, 1),
    "food-apple2": (2, 2),
    "food-apple3": (3, 3),
    "food-bread-crumbs": (1, 0),
    "food-salad-bowl": (1, 1),
    "food-sushi": (1, 1),
    "food-pizza": (2, 2),
    "food-milk": (1, 2),
    "food-milk2": (2, 4),
    "food-milk3": (3, 6),
    "food-chocolate-milk": (1, 0),
    "food-chocolate-milk2": (2, 0),
    "food-chocolate-milk3": (3, 0),
    "food-pear": (2, 2),
    "food-cupcake": (3, 3),
    # Canned food grants +1/+1 too, and Ruihan ruled on 2026-08-13 that it
    # counts as a stats food for Cat.  It is priced HERE rather than in a table
    # of its own because this table is what `_food_effect_stats` reads into the
    # pending-food context, and that context is the only channel through which
    # Cat's multiplier can reach a food's value.  Its stats land on the SHOP
    # pets and on the persistent shop-pet bonus rather than on a team pet: the
    # target set is decided by `NO_TARGET_FOODS` below (which gives it no
    # targets at all) and never by membership here, and `engine.py`'s BUY_FOOD
    # path short-circuits into its own branch before the per-target loop that
    # tests `item_id in FOOD_STAT_BUFFS`.
    "food-canned-food": (1, 1),
}

# Foods that select random friends in shop phase.
RANDOM_TARGET_FOOD_COUNTS: dict[str, int] = {
    "food-salad-bowl": 2,
    "food-sushi": 3,
    "food-pizza": 2,
}
RANDOM_TARGET_FOODS = set(RANDOM_TARGET_FOOD_COUNTS.keys())

# Foods that do not target any team pet and instead modify shop/global shop state.
NO_TARGET_FOODS = {
    "food-canned-food",
}

# Generated-only foods should never appear in normal roll pools.
# They are created by abilities/effects (e.g. Cow/Worm/Pigeon), not by rolling.
NON_ROLLABLE_FOOD_IDS = {
    "food-apple2",
    "food-apple3",
    "food-bread-crumbs",
    "food-milk",
    "food-milk2",
    "food-milk3",
    "food-chocolate-milk",
    "food-chocolate-milk2",
    "food-chocolate-milk3",
}

# Token/summoned-only pets should never appear in normal roll pools.
NON_ROLLABLE_PET_IDS = {
    "pet-bus",
    "pet-ram",
    "pet-chick",
    "pet-zombie-cricket",
    "pet-zombie-fly",
    "pet-honey-bee",
}

# Sloth (MinionEnum 71) is the easter-egg pet: it is NOT a uniform member of the
# tier-1 pool. The real game (reverse-engineered in exp02, ghidra RandomizeShop)
# draws once per shop roll and, only if that draw < SLOTH_ROLL_PROB, replaces the
# first freshly-rolled pet slot with Sloth (a vanilla 1/1 with no ability).
# So it must be excluded from every uniform roll/reward pool and injected
# separately at this rate, never picked like a normal tier-1 pet.
SLOTH_PET_ID = "pet-sloth"
SLOTH_ROLL_PROB = 1e-4

# Equipment foods represented as one active equipment per pet.
EQUIPMENT_FOOD_IDS = {
    "food-honey",
    "food-meat-bone",
    "food-garlic",
    "food-chili",
    "food-melon",
    "food-mushroom",
    "food-steak",
    "food-peanut",
    "food-coconut",
    "food-birthday-cake",
}

# Canonical status tags associated with specific equipment foods.
EQUIPMENT_STATUS_BY_FOOD_ID: dict[str, str] = {
    "food-honey": "status-honey-bee",
    "food-meat-bone": "status-bone-attack",
    "food-garlic": "status-garlic-armor",
    "food-chili": "status-splash-attack",
    "food-melon": "status-melon-armor",
    "food-mushroom": "status-extra-life",
    "food-steak": "status-steak-attack",
    "food-peanut": "status-peanut",
    "food-coconut": "status-coconut-shield",
}


def expected_shop_counts(turn: int) -> tuple[int, int]:
    """Return expected (pet_slots, food_slots) by turn for SAP-Arena rules."""
    if turn <= 2:
        return 3, 1
    if turn <= 4:
        return 3, 1
    if turn <= 8:
        return 4, 2
    return 5, 2


def shop_item_cost(slot_type: str, item_id: str) -> int:
    """Return canonical shop cost for a rolled shop item."""
    if str(slot_type) == "food":
        return int(FOOD_COST_OVERRIDES.get(str(item_id), DEFAULT_SHOP_ITEM_COST))
    return int(DEFAULT_SHOP_ITEM_COST)
