"""WHY A DIVERGENCE COUNT IS NOT ENOUGH.  The Cat fix stops a stats-free food
from consuming one of Cat's two purchase-food triggers.  Whether that changed
any play is answered by the drift check -- but a drift check that reports zero
divergences is ambiguous: either the fix genuinely changes nothing the agent
does, or the situation it changes never arose in those games.  The two have
opposite consequences for the campaign, and no count of divergences can tell
them apart.  This counter can, because it counts the SITUATION rather than the
outcome, and it counts each of its four conditions separately so a zero at the
joint condition says WHICH condition failed.

THE FOUR CONDITIONS, which are one occurrence and not four counters:

  1. a Cat is on the board when a food is bought,
  2. the food bought grants no stats (the old code burned a trigger here),
  3. a food that DOES grant stats is bought later in the SAME turn (the turn
     is the counter's lifetime: `engine.reset_ability_counters` clears it at
     the turn boundary), and
  4. the counter would already be full at that later purchase under the OLD
     rule while it is not under the new one.

Only when all four hold did the fix actually change the stats a pet received.
`opportunities` counts those; the fields above it count the prefixes of the
same conjunction, so "the trigger never fired" reads directly off the row.

HOW THE COUNTERFACTUAL IS KEPT HONEST.  The old counter is never simulated
independently -- it is derived as `real + preserved`, where `real` is the
engine's own live counter and `preserved` is the number of triggers this turn
that the old rule would have spent and the new one did not.  So the only thing
this module models is the difference between the two rules, and everything
else is read off the state the engine actually produced.  A mid-turn level-up
clears the counters (`clear_ability_counters_for_team_index`), which is
visible here as the real counter dropping, and the counterfactual restarts
with it.

The unit is a (food purchase, Cat) pair, because the trigger budget is
per-Cat: a board with two Cats offers two triggers to spend and two to
preserve."""

from __future__ import annotations

from typing import Any

from ..ability.effects import get_ability_counter
from ..engine import _food_effect_stats

CAT_PET_ID = "pet-cat"
MAX_TRIGGERS = 2
PURCHASE_FOOD_TRIGGER = "purchase_food"

FIELDS: tuple[str, ...] = (
    # (food purchase, Cat on board) pairs seen at all -- the denominator, and
    # the field that says "no Cat ever bought a food" when everything is 0.
    "cat_food_purchases",
    # ... of those, the food granted no stats (condition 2).
    "stats_free_purchases",
    # ... and the old rule would have had a trigger to spend on it, so the fix
    # preserved one (condition 2, with the counter having room).
    "triggers_preserved",
    # ... and a stats food followed in the same turn for the same Cat
    # (condition 3).
    "stats_purchases_after_preserved",
    # ... and the old counter would have been full while the new one is not,
    # so the fix changed what that food granted (condition 4: all four hold).
    "opportunities",
)


def empty_counts() -> dict[str, int]:
    return {name: 0 for name in FIELDS}


def add_counts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, int]:
    left = left or {}
    right = right or {}
    return {name: int(left.get(name, 0)) + int(right.get(name, 0)) for name in FIELDS}


def sum_counts(items: Any) -> dict[str, int]:
    total = empty_counts()
    for item in items:
        total = add_counts(total, item)
    return total


def food_effect_of(board: dict[str, Any], shop_index: Any) -> tuple[int, int] | None:
    """The stats Cat would multiply, or None if the slot is not a food.

    Read through the engine's own `_food_effect_stats`, so what counts as a
    stats food here cannot drift from what the engine prices: base stats from
    `FOOD_STAT_BUFFS`, with the shop slot's own values overriding them (a
    Cow-stocked Milk 2/4).
    """
    try:
        wanted = int(shop_index)
    except (TypeError, ValueError):
        return None
    for slot in board.get("shop") or []:
        if not isinstance(slot, dict):
            continue
        if str(slot.get("slot_type")) != "food":
            continue
        try:
            if int(slot.get("shop_index", -1)) != wanted:
                continue
        except (TypeError, ValueError):
            continue
        _, _, attack, health = _food_effect_stats(slot)
        return int(attack), int(health)
    return None


class CatTriggerAudit:
    """Per-turn accountant, fed the REAL board and the ops that reached it.

    One instance per turn (the counter's lifetime).  `on_committed_op` takes
    the board as it was BEFORE the op, which is the state whose counters and
    shop the old rule would have read.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = empty_counts()
        self._preserved: dict[int, int] = {}
        self._last_real: dict[int, int] = {}

    def on_committed_op(self, board: dict[str, Any], op: dict[str, Any]) -> None:
        if not isinstance(board, dict) or not isinstance(op, dict):
            return
        if str(op.get("type", "")).strip().upper() != "BUY_FOOD":
            return
        team = board.get("team") or []
        cats = [
            index
            for index, slot in enumerate(team)
            if isinstance(slot, dict) and str(slot.get("pet_id") or "") == CAT_PET_ID
        ]
        for index in list(self._preserved):
            if index not in cats:
                # A Cat that left the board (sold, combined, pilled) takes its
                # counterfactual with it rather than leaking onto whoever
                # occupies that slot next.
                self._preserved.pop(index, None)
                self._last_real.pop(index, None)
        if not cats:
            return
        effect = food_effect_of(board, op.get("shop_index"))
        if effect is None:
            return
        stats_free = effect == (0, 0)

        for index in cats:
            real = get_ability_counter(board, PURCHASE_FOOD_TRIGGER, index)
            if real < self._last_real.get(index, 0):
                # The counters were cleared mid-turn (a level-up), so both
                # rules start over and the preserved triggers are spent.
                self._preserved[index] = 0
            self._last_real[index] = real
            preserved = self._preserved.get(index, 0)
            old_counter = real + preserved

            self.counts["cat_food_purchases"] += 1
            if stats_free:
                self.counts["stats_free_purchases"] += 1
                if old_counter < MAX_TRIGGERS:
                    self.counts["triggers_preserved"] += 1
                    self._preserved[index] = preserved + 1
            elif preserved > 0:
                self.counts["stats_purchases_after_preserved"] += 1
                if old_counter >= MAX_TRIGGERS > real:
                    self.counts["opportunities"] += 1
