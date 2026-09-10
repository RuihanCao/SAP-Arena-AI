"""Stand-in for the live replay database, which this release does not ship.

The private tree can fall back to a Postgres corpus when the opponent snapshot
has no board for a turn or for a followed participation. There is no such
database here, and the real module would pull the corpus-build layer with it, so
these two functions report cleanly that they cannot serve instead of raising
`ModuleNotFoundError` from inside a turn resolution -- which is what happened
before this file existed, and it killed the demo mid-game.

The return shape is the one `train/opponents.py` uses for a failed sample, so the
caller annotates and surfaces it the same way it surfaces a snapshot miss.
"""

from __future__ import annotations

from typing import Any

_ERROR = ("no_live_replay_db_in_public_release: this release ships a fixed opponent "
          "snapshot and no database to fall back to")


def sample_opponent_team_for_pid(pid: str, turn: int, **_kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": f"{_ERROR}:pid={pid}:turn={turn}", "team": []}


def sample_random_opponent_team(turn: int, **_kwargs: Any) -> dict[str, Any]:
    return {"ok": False, "error": f"{_ERROR}:turn={turn}", "team": []}
