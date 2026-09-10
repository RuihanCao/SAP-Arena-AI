"""Stable two-axis identities for exp16 opponents and legacy replays."""

from __future__ import annotations

import copy
from typing import Any


MODEL_REVISION_R0 = "r0"

LEGACY_AGENT_ID = "arm-c-r0-a1"
LEGACY_AGENT_NAME = "Arm C r0 · Honest A1"
LEGACY_SEARCH_REVISION = "a1"
LEGACY_BASELINE_ID = "exp13-pre-w0f-a1"

A2_AGENT_ID = "arm-c-r0-a2"
A2_AGENT_NAME = "Arm C r0 · Structural A2"
A2_SEARCH_REVISION = "a2"
A2_BASELINE_ID = "exp13-w0f-a2-repin"

# exp13 PLAN Amendments A4 and A2.6 (2026-08-06, its wave W1a-3) changed how
# the shared segment loop cuts a turn and, with A2.6, the engine's random
# stream itself. The weights did not move, so this is a search revision in
# exactly Amendment 3's sense and gets the next letter.
CURRENT_AGENT_ID = "arm-c-r0-a3"
CURRENT_AGENT_NAME = "Arm C r0 · Causal A3"
CURRENT_SEARCH_REVISION = "a3"
# The a3 re-pin landed 2026-08-06 (W1a-3): mean trophies 4.027 [3.892, 4.163],
# ITT over N=1000 on the val-split gate pool, engine semantics f99bbf1.
# It replaces the 4.128 a2 pin, which PLAN.md Amendment A4 retired as a gate
# baseline because it was measured under the uncorrected cut criterion.
# Evidence: internal design notes.
CURRENT_BASELINE_ID = "exp13-w1a3-repin"


def normalize_ai_version(value: Any) -> dict[str, Any]:
    """Add stable identity fields without changing archived game evidence.

    Every archive written before the exp13 A2 alignment used the current r0
    checkpoints with the A1 honest-frame search and had no ``agent_id``.
    Missing identity fields therefore map deterministically to the legacy A1
    identity at read time. The stored transitions, images, and results remain
    untouched.
    """
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
