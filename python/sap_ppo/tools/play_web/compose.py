"""exp16 W4: server-side compositions of engine actions.

The engine restricts `BUY_PET` to the FIRST empty team slot
(`engine._legal_buy_pet` -> `unsupported_buy_pet_target`), and
`engine.legal_actions` only enumerates `REORDER` permutations of OCCUPIED
slots, so "buy into slot 4" and "drag my pet onto an empty slab" are both
unreachable from the browser today.

Neither is fixed by widening the action space. `ACTION_CATALOG` is 309
entries and it feeds the BC/PPO encoders and every trained checkpoint, so a
310th entry silently invalidates them all. Both moves are instead COMPOSED
out of actions the engine already has:

    buy into slot t  ==  BUY_PET(shop_index, first_empty f)
                         then REORDER(identity with f and t transposed)

    move src -> dst  ==  REORDER(lift the pet off src, make room at dst the
                         way the engine makes room for a summon, drop it in)
                         -- see `move_pet_order`

Four properties of the engine make that exact rather than approximate. All
four were read off `engine.py` and then measured on 60 random reachable
states x 425 (buyable pet, empty target) pairs before this module existed:

1. `_legal_reorder` accepts ANY permutation of `[0,1,2,3,4]`, empty slots
   included -- it checks `sorted(order) == [0,1,2,3,4]` and nothing else.
   `check_legal` (the path `api.step` takes) calls it directly and is NOT
   filtered through `legal_actions`. Measured: 425/425 such reorders legal,
   0/425 present in `legal_actions`.
2. `REORDER` is `new[k] = old[order[k]]` -- a permutation of the five slots,
   not a swap of neighbours and not a shift. A transposition of f and t
   therefore moves exactly one pet and leaves the other four slots alone.
3. No Turtle-pack buy trigger summons a pet: the buy-side abilities are
   Otter (buff a random friend), Cow (replace the shop's food) and Dragon
   (buff the team on a tier-1 buy); every summon in the pack is a
   faint/battle trigger, which cannot fire in the shop. So after `BUY_PET`
   at f, every other empty slot is still empty and t is still a valid
   destination. Measured: 425/425 buys raised occupancy by exactly {f}.
4. The composition adds nothing of its own. Measured: on all 425 pairs the
   other occupied slots are byte-identical to the plain `BUY_PET(f)`
   reference; the 19 cases where they differ from the PRE-buy state are all
   Otter buffing a friend (`ability_applied:pet-otter:buy:targets=[0]`),
   i.e. the engine's own on-buy ability, which lands the same way with or
   without the reorder.

The group is applied all-or-nothing by `apply_group` below, which both
`play_web/app.py::App` and `play_web/duel.py::DuelSession` call, so the two
surfaces cannot drift on what "atomic" means.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, Callable

from ...api import step

IDENTITY_ORDER: tuple[int, ...] = (0, 1, 2, 3, 4)

COMPOSE_BUY_PET_AT = "buy_pet_at"
COMPOSE_MOVE_PET = "move_pet"
COMPOSE_KINDS = frozenset({COMPOSE_BUY_PET_AT, COMPOSE_MOVE_PET})


def _team_slot(state: dict[str, Any], idx: int) -> dict[str, Any] | None:
    team = state.get("team") or []
    if idx < 0 or idx >= len(team):
        return None
    return team[idx]


def _is_empty(slot: dict[str, Any] | None) -> bool:
    return slot is not None and slot.get("pet_id") is None


def _first_empty_index(state: dict[str, Any]) -> int | None:
    for i, slot in enumerate(state.get("team") or []):
        if slot.get("pet_id") is None:
            return i
    return None


def transposition(a: int, b: int) -> list[int]:
    """The `REORDER` order that swaps slots `a` and `b`, fixing the rest."""
    order = list(IDENTITY_ORDER)
    order[a], order[b] = order[b], order[a]
    return order


# The pet held in hand while the board makes room for it. Negative, so it can
# never be confused with an original slot index in the token list below.
_HOLE = -1


def _closest_empty_ahead(free: frozenset[int], slot: int) -> int | None:
    """`engine._find_closest_empty_ahead`, read off a set of free slabs."""
    for idx in range(int(slot) - 1, -1, -1):
        if idx in free:
            return idx
    return None


def _closest_empty_behind(free: frozenset[int], slot: int, size: int) -> int | None:
    """`engine._find_closest_empty_behind`, read off a set of free slabs."""
    for idx in range(int(slot) + 1, size):
        if idx in free:
            return idx
    return None


def move_pet_order(occupied: Sequence[bool], src: int, dst: int) -> list[int]:
    """The `REORDER` order that lifts slot `src`'s pet and inserts it at `dst`.

    A move is NOT a swap. Dropping a pet on an occupied slab shoves that pet
    (and the run of pets ahead of or behind it) over by one slab, which is
    the rule the engine already owns for landing a summoned pet on a
    requested slot:

        engine._make_room_for_summon_slot(state, slot)
            -> True immediately if the slab is already free
            -> _push_forward_from_slot   if any slab AHEAD of it is free
            -> _push_backward_from_slot  otherwise

    "Forward" is toward the front rank (lower index) and it wins whenever it
    can, and each push moves only the run between the target and the NEAREST
    free slab, not the whole span between src and dst. That rule is copied
    here rather than invented, and `test_play_web_move_insert.py` drives the
    engine's own helpers over every occupancy pattern to assert this function
    agrees with them slab for slab.

    It is copied rather than called because a composition has to come out as
    engine ACTIONS -- `apply_group` only ever goes through `api.step` -- while
    `_make_room_for_summon_slot` mutates a state in place. Nothing is lost by
    doing so: `REORDER` is `new[k] = old[order[k]]`, a full permutation of the
    five slabs with empty ones included, and lift + shift + insert is exactly
    such a permutation, because the shift ends by emptying the slab the lifted
    pet then fills.

    The token list carries WHICH ORIGINAL SLAB ends up where, which is what
    `order` has to say. Lifting the pet leaves a hole that is no original
    slab, so it rides along as `_HOLE`. Exactly one free slab is consumed by
    the insertion; if the hole is not the one consumed it inherits that slab's
    identity, which balances because the board keeps the same number of free
    slabs it started with.

    No ability fires: this is a reposition, and `REORDER` is the engine's
    plain permutation action.
    """
    size = len(occupied)
    tokens: list[int] = list(range(size))
    tokens[src] = _HOLE
    free = frozenset(idx for idx in range(size) if idx == src or not occupied[idx])

    if dst not in free:
        ahead = _closest_empty_ahead(free, dst)
        empty_idx = ahead if ahead is not None else _closest_empty_behind(free, dst, size)
        if empty_idx is None:  # unreachable: `src` is free and is not `dst`
            raise ValueError(f"no free slab to make room at {dst}")
        shift = list(range(size))
        if ahead is not None:
            for idx in range(empty_idx, dst):
                shift[idx] = idx + 1
        else:
            for idx in range(empty_idx, dst, -1):
                shift[idx] = idx - 1
        shift[dst] = empty_idx
        tokens = [tokens[i] for i in shift]

    consumed = tokens[dst]
    tokens[dst] = src
    if _HOLE in tokens:
        tokens[tokens.index(_HOLE)] = consumed
    return tokens


def _coerce_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def compose_buy_pet_at(
    state: dict[str, Any],
    *,
    shop_index: Any,
    team_index: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Actions that buy shop pet `shop_index` into team slot `team_index`.

    Returns `(actions, None)` or `([], code)`. Only the two conditions the
    COMPOSITION itself depends on are rejected here -- the target must be
    empty, and there must be a first empty slot to buy into. Everything else
    (unknown shop slot, food slot, not enough gold) stays the engine's call
    and surfaces from `apply_group` naming the op that failed, so there is
    one source of truth for what a buy costs and when it is legal.

    Buying into the first empty slot is a group of ONE: the engine already
    does that move, and a reorder that transposes a slot with itself is not
    a permutation the caller should have to think about.
    """
    ti = _coerce_index(team_index)
    if ti is None or ti < 0 or ti > 4:
        return [], "team_index_out_of_range"
    si = _coerce_index(shop_index)
    if si is None:
        return [], "shop_index_invalid"

    target = _team_slot(state, ti)
    if target is None:
        return [], "team_index_out_of_range"
    if not _is_empty(target):
        return [], "target_not_empty"

    first = _first_empty_index(state)
    if first is None:
        return [], "team_full"

    buy = {"type": "BUY_PET", "shop_index": si, "team_index": first}
    if ti == first:
        return [buy], None
    return [buy, {"type": "REORDER", "order": transposition(first, ti)}], None


def compose_move_pet(
    state: dict[str, Any],
    *,
    src: Any,
    dst: Any,
) -> tuple[list[dict[str, Any]], str | None]:
    """Actions that move the pet in slot `src` onto slot `dst`.

    `dst` may be free or occupied. An occupied `dst` is an INSERT and not a
    swap -- the pets it displaces shift over one slab, the way the engine
    makes room for a summon; `move_pet_order` carries the derivation.

    Dropping a pet on a SAME-NAME pet is a `COMBINE`, a different move
    entirely, and it stays with `legal_actions`: the browser looks for it
    first and only falls through to here when the drop is a plain
    reposition.
    """
    s = _coerce_index(src)
    if s is None or s < 0 or s > 4:
        return [], "src_team_index_out_of_range"
    d = _coerce_index(dst)
    if d is None or d < 0 or d > 4:
        return [], "dst_team_index_out_of_range"
    if s == d:
        return [], "move_same_slot"

    src_slot = _team_slot(state, s)
    dst_slot = _team_slot(state, d)
    if src_slot is None or dst_slot is None:
        return [], "team_index_out_of_range"
    if _is_empty(src_slot):
        return [], "move_src_empty"

    occupied = [not _is_empty(slot) for slot in (state.get("team") or [])]
    return [{"type": "REORDER", "order": move_pet_order(occupied, s, d)}], None


def compose_actions(
    state: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    """Dispatch a `{"compose": ...}` request payload.

    Returns `(kind, actions, error)`; `kind` is None only when the payload
    names no known composition.
    """
    kind = str(payload.get("compose") or "").strip()
    if kind == COMPOSE_BUY_PET_AT:
        actions, err = compose_buy_pet_at(
            state,
            shop_index=payload.get("shop_index"),
            team_index=payload.get("team_index"),
        )
        return kind, actions, err
    if kind == COMPOSE_MOVE_PET:
        actions, err = compose_move_pet(state, src=payload.get("src"), dst=payload.get("dst"))
        return kind, actions, err
    return None, [], f"unknown_compose:{kind}"


def apply_group(
    state: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    step_fn: Callable[..., dict[str, Any]] | None = None,
    normalize_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[bool, str | None, dict[str, Any], list[dict[str, Any]]]:
    """Apply `actions` to `state` all-or-nothing.

    Returns `(ok, error, state_after, transitions)`. On any illegal or
    rejected op the whole group is dropped: `state_after` is a copy of the
    untouched input state, `transitions` is empty, and `error` names the op
    that failed by index and type. A half-applied buy therefore can never be
    committed, and the caller does not have to unwind anything itself.

    `normalize_fn` is the caller's per-op state fixup (play-web normalizes
    for its game mode after every action); it runs on each committed op so
    the group leaves exactly the state a sequence of single applies would.
    """
    # Resolved per call, not bound as a default: a default argument would
    # freeze `step` at import time, which makes the rollback path impossible
    # to exercise by patching this module.
    apply_one = step if step_fn is None else step_fn

    if not actions:
        return False, "empty_action_group", copy.deepcopy(state), []

    work = copy.deepcopy(state)
    transitions: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        typ = str(action.get("type", "?"))
        try:
            tr = apply_one(work, action)
        except Exception as exc:  # schema validation / malformed action
            return (
                False,
                f"compose_op_rejected:{index}:{typ}:{type(exc).__name__}:{exc}",
                copy.deepcopy(state),
                [],
            )
        if not tr.get("legal"):
            note = tr["engine_notes"][0] if tr.get("engine_notes") else "illegal_action"
            return False, f"compose_op_illegal:{index}:{typ}:{note}", copy.deepcopy(state), []
        after = copy.deepcopy(tr["state_after"])
        if normalize_fn is not None:
            after = normalize_fn(after)
            tr["state_after"] = copy.deepcopy(after)
        work = after
        transitions.append(tr)

    return True, None, work, transitions
