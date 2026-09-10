"""Neutral shared state helpers used by BOTH the training env and the eval
decoder (exp09 W5 P0): the within-turn state-signature (legality parity) and
the `training_rolls_this_turn` reset helpers (observation parity).

`state_signature` was extracted VERBATIM (function body + docstring
unchanged) from `tools/bc_recommender.py`'s original private
`_state_signature` so BOTH the eval decoder (`tools/bc_recommender.py`) and
the training env (`train/env.py`) share ONE implementation instead of two
copies that could silently drift apart -- porting it exactly is what
preserves the eval baseline (the eval side must not change behavior at all
from this refactor).

`get_training_rolls_this_turn` / `set_training_rolls_this_turn` (observation
parity) are the read/write pair for `meta.training_rolls_this_turn` -- the
counter `train/observation.py::StateVectorEncoderV3._rolls_this_turn_norm`
encodes. exp09 W5 P0 DECISION (Ruihan): this counter is deliberately HELD AT
0 everywhere -- `train/env.py::TrainingEnv.step` never increments it and
`tools/bc_recommender.py::BcRecommender.recommend` resets it to 0 per turn --
so the encoded feature is a constant 0.0 on every path (training env, eval
decode, and the counter-BLIND `train/chain_bc_dataset.py` records the
`flat_v2` BC checkpoint was actually trained on: verified, that field is
absent from every dataset_v2 shard -> the encoder read it as 0.0). This is
NOT cosmetic: `flat_v2` is W5's warm-start AND its frozen KL anchor, so
feeding it any nonzero value here would query that anchor out of
distribution (it never saw one). `set_training_rolls_this_turn` therefore
exists ONLY to (re)establish the 0 at turn boundaries; there is intentionally
no increment helper. Reserved for a future counter-AWARE dataset: if BC is
ever retrained on records that populate this field, re-introduce a shared
increment rule here and call it from both `TrainingEnv.step` and
`BcRecommender.recommend` (they are structured to make that a one-line
change at each site).

Deliberately STDLIB-ONLY and importing neither `train` nor `tools`:
`tools/bc_recommender.py` imports `train.env` (for `ACTION_CATALOG` etc.), so
`train/env.py` importing anything back from `tools/` would be a cycle; living
here, outside both packages, breaks that cycle. Both call sites import
`state_signature` as `from ..visited_guard import state_signature as
_state_signature`, so every other reference to `_state_signature` in either
module's own code/docs stays textually accurate without further edits.
"""

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
    level/equipment_id; shop: shop_index/item_id/frozen).

    KNOWN approximation (cross-model review finding 5, exp09 W5 P0 --
    intentionally NOT changed here): omits exp, sell_value, status_effects/
    counters, and the engine's RNG seed, so two states that differ ONLY in
    one of those can still hash equal. This is SHARED by both call sites
    (`train/env.py`'s `TrainingEnv` guard and `tools/bc_recommender.py`'s
    eval-time decode guard both import this exact function), so parity
    between them holds either way -- porting it exactly, warts included,
    was the point of the original extraction. Reserved for a future
    both-sides upgrade + eval rebaseline, not a train-only fix.
    """
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
    """Read `meta.training_rolls_this_turn`, clamped to >= 0.

    0 if `state["meta"]` is absent/not-a-dict, the key is unset, or the
    value cannot be parsed as an int -- never raises. Under the exp09 W5 P0
    held-at-0 decision (see module docstring) this should read 0 on every
    real state; the function stays a defensive clamp used mainly to VERIFY
    that invariant (tests, harness diagnostics).
    """
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return 0
    try:
        return max(0, int(meta.get("training_rolls_this_turn", 0)))
    except Exception:
        return 0


def set_training_rolls_this_turn(state: dict[str, Any], value: int) -> None:
    """Write `meta.training_rolls_this_turn` on `state` IN PLACE, clamped to
    >= 0. Creates `state["meta"]` if it does not already exist.

    exp09 W5 P0: in practice every caller passes 0 -- this exists to
    (re)establish the held-at-0 invariant (see module docstring) at turn
    boundaries and on entry to a decode, NOT to advance a counter. There is
    deliberately no increment helper: the counter must stay 0 to match the
    counter-blind `flat_v2` frame it is a warm-start / KL anchor for.
    """
    meta = state.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        state["meta"] = meta
    meta["training_rolls_this_turn"] = max(0, int(value))
