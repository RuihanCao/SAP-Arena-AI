"""Agent identity."""

from __future__ import annotations

import copy
from typing import Any


MODEL_REVISION_R0 = "r0"

LEGACY_AGENT_ID = "bc-value-r0-a1"
LEGACY_AGENT_NAME = "BC + V (v1)"
LEGACY_SEARCH_REVISION = "a1"
LEGACY_BASELINE_ID = "search-v1"

A2_AGENT_ID = "bc-value-r0-a2"
A2_AGENT_NAME = "BC + V (v2)"
A2_SEARCH_REVISION = "a2"
A2_BASELINE_ID = "search-v2"


CURRENT_AGENT_ID = "bc-value-r0-a3"
CURRENT_AGENT_NAME = "BC + V (v3)"
CURRENT_SEARCH_REVISION = "a3"


CURRENT_BASELINE_ID = "search-v3"


def normalize_ai_version(value: Any) -> dict[str, Any]:
    """Add stable identity fields without changing archived game evidence."""
    payload = copy.deepcopy(value) if isinstance(value, dict) else {}
    if not payload.get("agent_id"):
        payload.update(
            {
                "agent_id": LEGACY_AGENT_ID,
                "agent_name": LEGACY_AGENT_NAME,
                "model_revision": MODEL_REVISION_R0,
                "search_revision": LEGACY_SEARCH_REVISION,
                "baseline_id": LEGACY_BASELINE_ID,
            }
        )
    else:
        payload.setdefault("agent_name", str(payload.get("agent") or payload["agent_id"]))
        payload.setdefault("model_revision", MODEL_REVISION_R0)
        payload.setdefault("search_revision", "unknown")
        payload.setdefault("baseline_id", "unknown")
    payload.setdefault("agent", payload["agent_name"])
    return payload
