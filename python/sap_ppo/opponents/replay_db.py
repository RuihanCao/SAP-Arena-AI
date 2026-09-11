"""Stand-in for the live replay database, which this release does not ship.

The return shape is the one `train/opponents.py` uses for a failed sample, so the
caller annotates and surfaces it the same way it surfaces a snapshot miss."""

from __future__ import annotations

from typing import Any

_ERROR = ("no_live_replay_db_in_public_release: this release ships a fixed opponent "
          "snapshot and no database to fall back to")


def sample_opponent_team_for_pid(pid: str, turn: int, **_kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": f"{_ERROR}:pid={pid}:turn={turn}", "team": []}


def sample_random_opponent_team(turn: int, **_kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": f"{_ERROR}:turn={turn}", "team": []}
