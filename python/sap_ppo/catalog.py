"""Catalog loading helpers for Turtle pack data."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .constants import DATA_DIR


GENERATED_FOODS: dict[str, tuple[str, int]] = {
    "food-bread-crumbs": ("BreadCrumbs", 1),
    "food-milk": ("Milk", 1),
    "food-apple2": ("Apple2", 1),
    "food-apple3": ("Apple3", 1),
    "food-milk2": ("Milk2", 1),
    "food-milk3": ("Milk3", 1),
    "food-chocolate-milk": ("ChocolateMilk", 1),
    "food-chocolate-milk2": ("ChocolateMilk2", 1),
    "food-chocolate-milk3": ("ChocolateMilk3", 1),
    "food-peanut": ("Peanut", 7),
    "food-coconut": ("Coconut", 7),
}

GENERATED_PET_NAME_IDS: dict[str, str] = {
    "pet-bus": "Bus",
    "pet-ram": "Ram",
    "pet-chick": "Chick",
    "pet-zombie-cricket": "CricketToken",
    "pet-zombie-fly": "FlyToken",
    "pet-honey-bee": "Bee",
}


GENERATED_PETS_FULL: dict[str, dict[str, Any]] = {
    "pet-sloth": {"name_id": "Sloth", "tier": 1, "attack": 1, "health": 1},
}


def _fallback_catalog() -> dict[str, Any]:
    return {
        "version": "v1",
        "pack": "Turtle",
        "pets": {
            "by_tier": {"1": ["pet-ant", "pet-fish"], "2": ["pet-crab"]},
            "id_to_name_id": {"pet-ant": "Ant", "pet-fish": "Fish", "pet-crab": "Crab"},
            "name_id_to_id": {"Ant": "pet-ant", "Fish": "pet-fish", "Crab": "pet-crab"},
            "base_stats": {
                "pet-ant": {"attack": 2, "health": 1},
                "pet-fish": {"attack": 2, "health": 3},
                "pet-crab": {"attack": 3, "health": 3}
            }
        },
        "foods": {
            "by_tier": {"1": ["food-apple", "food-honey"]},
            "id_to_name_id": {"food-apple": "Apple", "food-honey": "Honey"},
            "name_id_to_id": {"Apple": "food-apple", "Honey": "food-honey"}
        }
    }


def _augment_generated_foods(catalog: dict[str, Any]) -> None:
    foods = catalog.setdefault("foods", {})
    by_tier = foods.setdefault("by_tier", {})
    id_to_name = foods.setdefault("id_to_name_id", {})
    name_to_id = foods.setdefault("name_id_to_id", {})

    for item_id, (name_id, tier) in GENERATED_FOODS.items():
        tier_key = str(int(tier))
        tier_items = by_tier.setdefault(tier_key, [])
        if item_id not in tier_items:
            tier_items.append(item_id)
        if str(item_id) not in id_to_name:
            id_to_name[str(item_id)] = str(name_id)
        if str(name_id) not in name_to_id:
            name_to_id[str(name_id)] = str(item_id)


def _augment_generated_pets(catalog: dict[str, Any]) -> None:
    pets = catalog.setdefault("pets", {})
    id_to_name = pets.setdefault("id_to_name_id", {})
    name_to_id = pets.setdefault("name_id_to_id", {})

    for item_id, name_id in GENERATED_PET_NAME_IDS.items():
        if str(item_id) not in id_to_name:
            id_to_name[str(item_id)] = str(name_id)
        if str(name_id) not in name_to_id:
            name_to_id[str(name_id)] = str(item_id)

    base_stats = pets.setdefault("base_stats", {})
    by_tier = pets.setdefault("by_tier", {})
    for item_id, spec in GENERATED_PETS_FULL.items():
        name_id = str(spec["name_id"])
        if str(item_id) not in id_to_name:
            id_to_name[str(item_id)] = name_id
        if name_id not in name_to_id:
            name_to_id[name_id] = str(item_id)
        if str(item_id) not in base_stats:
            base_stats[str(item_id)] = {"attack": int(spec["attack"]), "health": int(spec["health"])}
        tier_key = str(int(spec["tier"]))
        tier_items = by_tier.setdefault(tier_key, [])
        if str(item_id) not in tier_items:
            tier_items.append(str(item_id))


@lru_cache(maxsize=1)
def load_turtle_catalog() -> dict[str, Any]:
    path = DATA_DIR / "turtle_catalog_v1.json"
    if not path.exists():
        catalog = _fallback_catalog()
        _augment_generated_foods(catalog)
        _augment_generated_pets(catalog)
        return catalog
    with path.open("r", encoding="utf-8") as f:
        catalog = json.load(f)
    _augment_generated_foods(catalog)
    _augment_generated_pets(catalog)
    return catalog


def tier_for_turn(turn: int) -> int:
    if turn <= 2:
        return 1
    if turn <= 4:
        return 2
    if turn <= 6:
        return 3
    if turn <= 8:
        return 4
    if turn <= 10:
        return 5
    return 6
