"""`state_signature` was extracted VERBATIM (function body + docstring
unchanged) from `tools/bc_recommender.py`'s original private
`_state_signature` so BOTH the eval decoder (`tools/bc_recommender.py`) and
the training env (`train/env.py`) share ONE implementation instead of two
copies that could silently drift apart -- porting it exactly is what
preserves the eval baseline (the eval side must not change behavior at all
from this refactor).

Deliberately STDLIB-ONLY and importing neither `train` nor `tools`:
`tools/bc_recommender.py` imports `train.env` (for `ACTION_CATALOG` etc.), so
`train/env.py` importing anything back from `tools/` would be a cycle; living
here, outside both packages, breaks that cycle. Both call sites import
`state_signature` as `from ..visited_guard import state_signature as
_state_signature`, so every other reference to `_state_signature` in either
module's own code/docs stays textually accurate without further edits."""

from __future__ import annotations

import json
from typing import Any


def state_signature(state: dict[str, Any]) -> str:
    """Canonical within-turn state signature for the anti-cycle guard.

    Captures exactly the fields a decode step can change: the OCCUPIED team
    slots (empty slots, `pet_id is None`, carry no information and are
    dropped), the shop, and gold. Two states that are identical in these
    fields are indistinguishable for decoding purposes and must hash equal;
    any difference in them must hash different. Team/shop rows are sorted by
    their own index field before serializing so the signature depends only
    on board content, never on incidental list ordering. Field names match
    `schemas/state_v1.json` exactly (team: slot_index/pet_id/attack/health/
    level/equipment_id; shop: shop_index/item_id/frozen)."""
    team = sorted(
        (
            [
                slot["slot_index"],
                slot["pet_id"],
                slot["attack"],
                slot["health"],
                slot["level"],
                slot["equipment_id"],
            ]
            for slot in state.get("team", [])
            if slot.get("pet_id") is not None
        ),
        key=lambda row: row[0],
    )
    shop = sorted(
        (
            [slot["shop_index"], slot["item_id"], bool(slot.get("frozen"))]
            for slot in state.get("shop", [])
        ),
        key=lambda row: row[0],
    )
    payload = {"team": team, "shop": shop, "gold": state.get("gold")}
    return json.dumps(payload, sort_keys=True)


def get_training_rolls_this_turn(state: dict[str, Any]) -> int:
    """Read `meta.training_rolls_this_turn`, clamped to >= 0."""
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return 0
    try:
        return max(0, int(meta.get("training_rolls_this_turn", 0)))
    except Exception:
        return 0


def set_training_rolls_this_turn(state: dict[str, Any], value: int) -> None:
    """Write `meta.training_rolls_this_turn` on `state` IN PLACE, clamped to
    >= 0. Creates `state["meta"]` if it does not already exist."""
    meta = state.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        state["meta"] = meta
    meta["training_rolls_this_turn"] = max(0, int(value))
