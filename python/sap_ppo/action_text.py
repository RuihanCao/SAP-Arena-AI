"""Plain-English descriptions of engine actions and boards.

Shared by `tools/decode_gallery_chain.py` (recovering a played game's action
chain) and the play-web replay archive (annotating each AI turn with what it
actually did), so the two never drift into describing the same op two
different ways.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .api import step as engine_step
from .catalog import load_turtle_catalog
from .engine import resolve_end_turn_pre_battle


def _pretty_item(item_id: Any) -> str:
    """`pet-cricket` -> `Cricket`, `food-apple` -> `Apple`; unknown ids pass
    through unchanged so this can never hide what the op actually referenced."""
    if item_id is None:
        return "empty"
    cat = load_turtle_catalog()
    for section in ("pets", "foods"):
        mapping = (cat.get(section) or {}).get("id_to_name_id") or {}
        name = mapping.get(str(item_id))
        if name:
            return str(name)
    return str(item_id)


def _shop_slot(state: dict[str, Any], shop_index: Any) -> dict[str, Any] | None:
    for slot in state.get("shop") or []:
        if int(slot.get("shop_index", -1)) == int(shop_index):
            return slot
    return None


def _describe_shop(state: dict[str, Any], shop_index: Any) -> str:
    slot = _shop_slot(state, shop_index)
    if slot is None:
        return f"shop slot {shop_index} (absent)"
    name = _pretty_item(slot.get("item_id"))
    if str(slot.get("slot_type")) == "pet":
        stats = f" {slot.get('attack')}/{slot.get('health')}"
    else:
        stats = ""
    frozen = " [frozen]" if slot.get("frozen") else ""
    return f"shop slot {shop_index} {name}{stats}{frozen}"


def _describe_team(state: dict[str, Any], team_index: Any) -> str:
    if team_index is None:
        return "the whole team"
    team = state.get("team") or []
    if not (0 <= int(team_index) < len(team)):
        return f"board slot {team_index} (absent)"
    pet = team[int(team_index)]
    if pet.get("pet_id") is None:
        return f"board slot {team_index} (empty)"
    lvl = int(pet.get("level", 1))
    lvl_s = f" L{lvl}" if lvl > 1 else ""
    return (
        f"board slot {team_index} {_pretty_item(pet.get('pet_id'))}"
        f"{lvl_s} {pet.get('attack')}/{pet.get('health')}"
    )


def _describe_op(state: dict[str, Any], op: dict[str, Any]) -> str:
    """One plain-English sentence for `op` applied to `state` (the board as it
    stands immediately BEFORE the op), naming the actual items involved."""
    typ = str(op.get("type", "")).strip().upper()
    if typ == "BUY_PET":
        return f"buy {_describe_shop(state, op['shop_index'])} into {_describe_team(state, op['team_index'])}"
    if typ == "BUY_COMBINE":
        return (
            f"buy {_describe_shop(state, op['shop_index'])} and merge it onto "
            f"{_describe_team(state, op['team_index'])}"
        )
    if typ == "BUY_FOOD":
        return f"buy {_describe_shop(state, op['shop_index'])} and feed it to {_describe_team(state, op['team_index'])}"
    if typ == "COMBINE":
        return (
            f"merge {_describe_team(state, op['src_team_index'])} into "
            f"{_describe_team(state, op['dst_team_index'])}"
        )
    if typ == "SELL":
        return f"sell {_describe_team(state, op['team_index'])}"
    if typ == "FREEZE":
        return f"freeze {_describe_shop(state, op['shop_index'])}"
    if typ == "UNFREEZE":
        return f"unfreeze {_describe_shop(state, op['shop_index'])}"
    if typ == "REORDER":
        order = list(op.get("order") or [])
        moves = ", ".join(f"new slot {k} <- old slot {src}" for k, src in enumerate(order))
        return f"reorder the board ({moves})"
    if typ == "ROLL":
        return "roll the shop (1 gold)"
    if typ == "END_TURN":
        return "end the turn (go to battle)"
    return f"{typ} {json.dumps({k: v for k, v in op.items() if k != 'type'}, sort_keys=True)}"


def annotate_chain(state_before: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Walk `ops` from `state_before` with the real engine, annotating each op
    with the items it actually touched and the gold it cost.

    Returns `{"steps": [...], "end_board": <state after the last non-END_TURN
    op>, "diverged_at": <str|None>}`. `END_TURN` is described but never applied
    (the driver never applies it through `engine_step` either -- see
    `eval_versus_fullgame.py`'s module docstring).
    """
    board = copy.deepcopy(state_before)
    steps: list[dict[str, Any]] = []
    diverged_at: str | None = None
    for i, op in enumerate(ops):
        typ = str(op.get("type", "")).strip().upper()
        gold_before = int(board.get("gold", 0))
        text = _describe_op(board, op)
        if typ == "END_TURN":
            steps.append(
                {
                    "i": i,
                    "op": copy.deepcopy(op),
                    "text": text,
                    "gold_before": gold_before,
                    "gold_after": gold_before,
                }
            )
            continue
        try:
            tr = engine_step(board, op)
        except Exception as exc:  # pragma: no cover - defensive, see docstring
            diverged_at = f"{typ}@{i}:{type(exc).__name__}"
            break
        if not tr.get("legal") or not isinstance(tr.get("state_after"), dict):
            diverged_at = f"{typ}@{i}:illegal_on_replay"
            break
        board = tr["state_after"]
        steps.append(
            {
                "i": i,
                "op": copy.deepcopy(op),
                "text": text,
                "gold_before": gold_before,
                "gold_after": int(board.get("gold", 0)),
            }
        )
    # The board the ORIGINAL run recorded as `detail["board_pre_battle"]` is
    # not this shop-phase board: `resolve_end_turn_with_sampled_battle` runs
    # `resolve_end_turn_pre_battle` first, which fires every end-of-turn
    # ability (Bison's +2/+2 next to a level-3 friend, ...) before handing
    # `playerPets` to the oracle. Re-derive the same way so the cross-check
    # compares like with like -- and so it also covers the end-of-turn
    # triggers, not just the shop chain.
    pre_battle_board: dict[str, Any] | None = None
    if diverged_at is None:
        try:
            pre_battle_board = copy.deepcopy(resolve_end_turn_pre_battle(board).state_after)
        except Exception:  # pragma: no cover - defensive
            pre_battle_board = None
    return {
        "steps": steps,
        "end_board": board,
        "pre_battle_board": pre_battle_board,
        "diverged_at": diverged_at,
    }


def board_summary(state: dict[str, Any] | None) -> list[str]:
    """`['Cricket L2 6/9', '-', ...]`, one entry per board slot."""
    if not isinstance(state, dict):
        return []
    out: list[str] = []
    for pet in state.get("team") or []:
        if pet.get("pet_id") is None:
            out.append("-")
            continue
        lvl = int(pet.get("level", 1))
        lvl_s = f" L{lvl}" if lvl > 1 else ""
        out.append(f"{_pretty_item(pet.get('pet_id'))}{lvl_s} {pet.get('attack')}/{pet.get('health')}")
    return out

