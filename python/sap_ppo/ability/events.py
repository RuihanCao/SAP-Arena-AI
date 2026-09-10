"""Ability event contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
TRIGGER_BUY = "buy"
TRIGGER_SELL = "sell"
TRIGGER_LEVEL_UP = "level_up"
TRIGGER_FRIEND_SOLD = "friend_sold"
TRIGGER_BUY_FRIEND = "buy_friend"
TRIGGER_BUY_TIER1_PET = "buy_tier1_pet"
TRIGGER_SUMMONED = "summoned"
TRIGGER_FRIEND_SUMMONED = "friend_summoned"
TRIGGER_FRIENDLY_ATE_FOOD = "friendly_ate_food"
TRIGGER_START_OF_TURN = "start_of_turn"
TRIGGER_END_OF_TURN = "end_of_turn"
TRIGGER_PURCHASE_FOOD = "purchase_food"
TRIGGER_EATS_FOOD = "eats_food"
TRIGGER_HURT = "hurt"
TRIGGER_FAINT = "faint"
TRIGGER_AFTER_FAINT = "after_faint"
TRIGGER_FRIEND_FAINTS = "friend_faints"
TRIGGER_FRIEND_AHEAD_FAINTS = "friend_ahead_faints"
TRIGGER_FRIENDS_HURT_COUNTER = "friends_hurt_counter"


@dataclass(frozen=True)
class AbilityEvent:
    """Single ability trigger instance."""

    trigger: str
    actor_pet_id: str
    actor_level: int
    actor_team_index: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def clamp_level(level: int) -> int:
    return max(1, min(3, int(level)))
