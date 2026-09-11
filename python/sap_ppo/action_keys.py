"""The hashable twin of `json.dumps(action, sort_keys=True, separators=...)`.

A JSON string of an action dict is a perfectly good key -- it is total, it is
canonical, and it separates exactly the values JSON separates. It is also the
most expensive way to reach a `dict` or a `set`, because it allocates a string
per lookup. Two hot paths were paying that and neither ever read the string:

Both want the same function, so it lives here, in a leaf module that imports
nothing from the package: `engine` is below `train.env`, so this could not stay
where it was first written without an import cycle, and a second copy of a key
this subtle is worse than a move.

WHAT "SEPARATES EXACTLY WHAT JSON SEPARATES" BUYS, and why it is not optional
in either caller. `legal_mask` RAISES on an action it cannot map, so a key that
collapsed two distinct actions would silently delete that guard. `legal_actions`
DROPS an action whose key it has already seen, so a key that collapsed two
distinct actions would silently shorten the action list, which is a dataset
contract. Hence every value carries its own TYPE next to it:

  * `1`, `1.0` and `True` are equal (or hash-equal) to each other in Python and
    are three different JSON tokens, so `(int, 1)`, `(float, "1.0")` and
    `(bool, True)` must not be one key;
  * floats key on `repr`, so `0.0` and `-0.0` (equal in Python, different in
    JSON) stay apart and `nan` stays equal to itself;
  * containers carry their type too, so a dict can never collide with the tuple
    of its own items.

The type is the type OBJECT, not a name: it is already in hand as
`value.__class__`, so it costs nothing to carry. `list` and `tuple` share one
tag because `json.dumps` encodes both as JSON arrays. Values `json.dumps`
cannot encode raise here too, as they do there.

The keys are for LOOKUP AND DEDUP ONLY -- never stored, never compared against
anything a previous process wrote. `train/env.py::_action_key` remains the
textual key for everything that does persist one."""

from __future__ import annotations

from typing import Any

NONE_TYPE = type(None)


def action_key_seq(values: Any) -> tuple[Any, ...]:
    """A JSON array's elements. Exact-`int` elements go in raw -- an int can
    never equal a `(type, value)` pair, so no tagged element can collide with
    an untagged one -- and everything else is tagged, which is what keeps
    `[True, 0, 2, 3, 4]` from answering with `[1, 0, 2, 3, 4]`'s index."""
    return tuple(item if item.__class__ is int else action_key_value(item) for item in values)


def action_key_value(value: Any) -> tuple[Any, Any]:
    """One value as a `(type, payload)` pair: hashable, and separating exactly
    what `json.dumps` separates. `hashable_action_key` inlines the three exact
    types that actually occur and delegates everything else here, so a value
    has ONE representation whichever branch reaches it."""
    cls = value.__class__
    if cls is str or cls is int or cls is NONE_TYPE:
        return (cls, value)
    if cls is bool:
        return (bool, value)
    if cls is float:
        # `repr`, not the float: `0.0` and `-0.0` are equal in Python and are
        # different JSON tokens, and `nan` is not even equal to itself.
        return (float, repr(value))
    if cls is list or cls is tuple:
        return (list, action_key_seq(value))
    if cls is dict:
        return (dict, tuple((key, action_key_value(value[key])) for key in sorted(value)))
    # Only SUBCLASSES reach here (an `IntEnum`, a `str` subclass). `json.dumps`
    # encodes them as their base type, so they key as their base type -- if
    # they keyed as themselves the two lookups would disagree.
    if isinstance(value, bool):
        return (bool, bool(value))
    if isinstance(value, int):
        return (int, int(value))
    if isinstance(value, float):
        return (float, repr(float(value)))
    if isinstance(value, str):
        return (str, str(value))
    if isinstance(value, (list, tuple)):
        return (list, action_key_seq(value))
    if isinstance(value, dict):
        return (dict, tuple((key, action_key_value(value[key])) for key in sorted(value)))
    raise TypeError(f"action_key_unsupported_value_type:{type(value).__name__}")


def hashable_action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    """The hashable twin of `json.dumps(action, sort_keys=True, ...)`.

    Every field becomes `(name, type, payload)`, whichever branch produced it,
    so the fast path and the general path cannot represent one value two ways.

    Written as a flat loop rather than the obvious
    `tuple((k, action_key_value(action[k])) for k in sorted(action))` because a
    call per field costs about as much as the `json.dumps` this replaces."""
    items = []
    for key in sorted(action):
        value = action[key]
        cls = value.__class__
        if cls is str or cls is int or cls is NONE_TYPE:
            items.append((key, cls, value))
        elif cls is list:
            items.append((key, list, action_key_seq(value)))  # REORDER's `order`
        else:
            tag, payload = action_key_value(value)
            items.append((key, tag, payload))
    return tuple(items)
