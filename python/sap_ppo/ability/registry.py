"""Ability registry and trigger declarations."""

from __future__ import annotations

from typing import Any, Callable

from .turtle_pack import TURTLE_ABILITY_HANDLERS

AbilityHandler = Callable[[dict[str, Any], Any, Any], None]

# Pets that carry a handler or a declaration here but are NOT in the Turtle
# catalog (`data/turtle_catalog_v1.json`, the pool the engine actually deals
# from). A handler outside that pool can never fire in a Turtle game, so
# anything built on it -- a test fixture, a probe attribution table, a worked
# example in a report -- is describing a board no game can reach.
#
# Being listed here is not a defect report; it is the annotation that says
# "known, and known to be out of pack". `tests/test_ability_pack_coverage.py`
# fails on any registered pet that is neither in the catalog nor listed here,
# so the next pet that ends up outside the pack is a red test rather than
# something somebody has to notice by diffing two files by eye.
#
# Kept rather than deleted: the handler is correct SAP behaviour, it is unit
# tested (`tests/test_step2_abilities.py`), and deleting it would only mean
# writing it again the day a pack containing the pet is added.
NON_TURTLE_PACK_PETS: dict[str, str] = {
    "pet-shrimp": (
        "Not in the Turtle pack. Its `friend_sold` handler is the ONLY handler "
        "for that trigger in the whole registry, and a SELL step emits exactly "
        "one `sell` event, so no in-catalog SELL step can tie or draw at all. "
    ),
}

# Incremental declaration set: if declared but not implemented, runtime marks explicit unsupported.
DECLARED_TRIGGERS: set[tuple[str, str]] = {
    ("pet-ant", "faint"),
    ("pet-cricket", "after_faint"),
    ("pet-otter", "buy"),
    ("pet-flamingo", "faint"),
    ("pet-hedgehog", "faint"),
    ("pet-rat", "after_faint"),
    ("pet-spider", "after_faint"),
    ("pet-badger", "faint"),
    ("pet-mammoth", "faint"),
    ("pet-turtle", "faint"),
    ("pet-ox", "friend_ahead_faints"),
    ("pet-sheep", "after_faint"),
    ("pet-deer", "after_faint"),
    ("pet-rooster", "after_faint"),
    ("pet-shark", "friend_faints"),
    ("pet-fly", "friend_faints"),
    ("pet-beaver", "sell"),
    ("pet-duck", "sell"),
    ("pet-pig", "sell"),
    ("pet-pigeon", "sell"),
    ("pet-shrimp", "friend_sold"),
    ("pet-snail", "end_of_turn"),
    ("pet-fish", "level_up"),
    ("pet-rabbit", "friendly_ate_food"),
    ("pet-giraffe", "start_of_turn"),
    ("pet-penguin", "start_of_turn"),
    ("pet-swan", "start_of_turn"),
    ("pet-squirrel", "start_of_turn"),
    ("pet-worm", "start_of_turn"),
    ("pet-cow", "buy"),
    ("pet-cow", "eats_food"),
    ("pet-peacock", "hurt"),
    ("pet-camel", "hurt"),
    ("pet-blowfish", "hurt"),
    ("pet-gorilla", "hurt"),
    ("pet-wolverine", "friends_hurt_counter"),
    ("pet-seal", "eats_food"),
    ("pet-scorpion", "summoned"),
    ("pet-cat", "purchase_food"),
    ("pet-dragon", "buy_tier1_pet"),
    ("pet-dog", "friend_summoned"),
    ("pet-horse", "friend_summoned"),
    ("pet-turkey", "friend_summoned"),
    ("pet-monkey", "end_of_turn"),
    ("pet-bison", "end_of_turn"),
}


def resolve_handler(pet_id: str, trigger: str) -> AbilityHandler | None:
    return TURTLE_ABILITY_HANDLERS.get((str(pet_id), str(trigger)))


def is_declared_trigger(pet_id: str, trigger: str) -> bool:
    return (str(pet_id), str(trigger)) in DECLARED_TRIGGERS
