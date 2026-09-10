"""Schema loading and validation helpers."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .constants import ACTION_SCHEMA, STATE_SCHEMA, TRANSITION_SCHEMA


@lru_cache(maxsize=8)
def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _validators() -> tuple[Draft202012Validator, Draft202012Validator, Draft202012Validator]:
    state = _load_schema(STATE_SCHEMA)
    action = _load_schema(ACTION_SCHEMA)
    transition = copy.deepcopy(_load_schema(TRANSITION_SCHEMA))

    # Keep transition schema standalone even when jsonschema local-ref behavior
    # differs between versions.
    defs = transition.setdefault("$defs", {})
    defs["state_v1"] = state
    defs["action_v1"] = action
    props = transition.setdefault("properties", {})
    if "state_before" in props:
        props["state_before"] = {"$ref": "#/$defs/state_v1"}
    if "action" in props:
        props["action"] = {"$ref": "#/$defs/action_v1"}
    if "state_after" in props:
        props["state_after"] = {"$ref": "#/$defs/state_v1"}

    return (
        Draft202012Validator(state),
        Draft202012Validator(action),
        Draft202012Validator(transition),
    )


def validate_state_schema(state: dict[str, Any]) -> None:
    v, _, _ = _validators()
    errs = sorted(v.iter_errors(state), key=lambda e: e.path)
    if errs:
        first = errs[0]
        raise ValueError(f"state schema validation failed at {list(first.path)}: {first.message}")


def validate_action_schema(action: dict[str, Any]) -> None:
    _, v, _ = _validators()
    errs = sorted(v.iter_errors(action), key=lambda e: e.path)
    if errs:
        first = errs[0]
        raise ValueError(f"action schema validation failed at {list(first.path)}: {first.message}")


def validate_transition_schema(transition: dict[str, Any]) -> None:
    _, _, v = _validators()
    errs = sorted(v.iter_errors(transition), key=lambda e: e.path)
    if errs:
        first = errs[0]
        raise ValueError(f"transition schema validation failed at {list(first.path)}: {first.message}")
