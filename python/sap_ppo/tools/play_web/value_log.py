"""## What a row is

One per resolved turn, written as the turn resolves:

- the human's end-of-turn board, valued by `V` (level B: no completion, no
  search), raw and on the trophy scale;
- the AI's end-of-turn board for the SAME turn, valued the same way, which is
  what makes the pair readable -- a lone 4.1 means nothing, `4.1 against 5.6`
  means something;
- who actually won that battle.

## Append-only, and outside the archived game"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "value_log_v1"

ROW_TURN = "turn"
ROW_FINAL = "final"


class ValueLog:
    """The rows for one game, in memory and (optionally) on disk.

    Thread-safe: `end_turn` writes rows while the browser reads them for the
    live table.
    """

    def __init__(self, *, game_id: str | None, seed: int | None, root: Path | None) -> None:
        self.game_id = game_id
        self.seed = seed
        self.root = Path(root) if root is not None else None
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self._lock = threading.RLock()
        self._finalised = False

    # -- writing ----------------------------------------------------------
    def path(self) -> Path | None:
        if self.root is None or not self.game_id:
            return None
        return self.root / "value_log" / f"{self.game_id}.jsonl"

    def _append_to_disk(self, row: dict[str, Any]) -> None:
        """Append one line, flushed and fsynced.

        Best effort by contract: a log that cannot be written must never fail a
        turn the human already played, so the failure is recorded in `error`
        (which the page shows) and the game continues.
        """
        target = self.path()
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as exc:
            self.error = f"value_log_write_failed:{type(exc).__name__}:{exc}"

    def record_turn(
        self,
        *,
        turn: int,
        human: dict[str, Any] | None,
        ai: dict[str, Any] | None,
        human_wins_before: int,
        ai_wins_before: int,
        outcome: str | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """One resolved turn. `human` / `ai` are `ValueProbe` value rows."""
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": ROW_TURN,
            "game_id": self.game_id,
            "seed": self.seed,
            "turn": int(turn),
            "outcome": outcome,
            "human_wins_before": int(human_wins_before),
            "ai_wins_before": int(ai_wins_before),
            "human_predicted_raw": (human or {}).get("raw"),
            "human_predicted_trophies": (human or {}).get("trophies"),
            "human_clamped": bool((human or {}).get("clamped")),
            "ai_predicted_raw": (ai or {}).get("raw"),
            "ai_predicted_trophies": (ai or {}).get("trophies"),
            "ai_clamped": bool((ai or {}).get("clamped")),
            "error": error,
        }
        with self._lock:
            self.rows.append(row)
        self._append_to_disk(row)
        return row

    def finalise(self, *, final_human_wins: int, final_ai_wins: int) -> dict[str, Any] | None:
        """Fill in the realised return-to-go for every turn, once, at game end.

        `realised = final wins - wins before that turn`, on [0, 10], which is
        the quantity the recalibration curve's `y` is denominated in.
        """
        with self._lock:
            if self._finalised:
                return None
            turns = [r for r in self.rows if r.get("kind") == ROW_TURN]
            realised = [
                {
                    "turn": int(r["turn"]),
                    "human_predicted_trophies": r.get("human_predicted_trophies"),
                    "human_realised_trophies": int(final_human_wins) - int(r["human_wins_before"]),
                    "ai_predicted_trophies": r.get("ai_predicted_trophies"),
                    "ai_realised_trophies": int(final_ai_wins) - int(r["ai_wins_before"]),
                }
                for r in turns
            ]
            row = {
                "schema_version": SCHEMA_VERSION,
                "kind": ROW_FINAL,
                "game_id": self.game_id,
                "seed": self.seed,
                "n_turns": len(turns),
                "final_human_wins": int(final_human_wins),
                "final_ai_wins": int(final_ai_wins),
                "realised": realised,
            }
            self.rows.append(row)
            self._finalised = True
        self._append_to_disk(row)
        return row

    # -- reading ----------------------------------------------------------
    def live_rows(self) -> list[dict[str, Any]]:
        """The scalar view the page renders, newest last.

        Every field is a scalar, so `duel_app`'s live-payload shape rule passes
        it through untouched and the table cannot become the next 19 MB
        regression.
        """
        with self._lock:
            rows = list(self.rows)
        return project_rows(rows)

    def describe(self) -> dict[str, Any]:
        target = self.path()
        return {
            "schema_version": SCHEMA_VERSION,
            "game_id": self.game_id,
            "path": str(target) if target is not None else None,
            "n_rows": len([r for r in self.rows if r.get("kind") == ROW_TURN]),
            "finalised": bool(self._finalised),
            "error": self.error,
        }


# -- reading a finished game's log back -----------------------------------
#
# The live table and the replay page render the SAME rows, so they go through
# the same projection. Keeping this at module level rather than duplicating the
# fold is the point: the FINAL row is what turns a prediction into a
# prediction-versus-realised pair, and a second copy of that fold is a second
# chance to forget it.


def project_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn rows, each carrying the realised return-to-go the FINAL row knows.

    `human_realised_trophies` is None while a game is still being played, which
    is honest: nothing realised yet. It fills in for every turn at once when
    `finalise` runs.
    """
    realised_by_turn: dict[int, int] = {}
    for row in rows:
        if row.get("kind") == ROW_FINAL:
            for entry in row.get("realised") or []:
                try:
                    realised_by_turn[int(entry["turn"])] = int(entry["human_realised_trophies"])
                except (KeyError, TypeError, ValueError):
                    continue
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != ROW_TURN:
            continue
        try:
            turn = int(row["turn"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            {
                "turn": turn,
                "outcome": row.get("outcome"),
                "human_predicted_trophies": row.get("human_predicted_trophies"),
                "human_predicted_raw": row.get("human_predicted_raw"),
                "human_clamped": bool(row.get("human_clamped")),
                "ai_predicted_trophies": row.get("ai_predicted_trophies"),
                "ai_clamped": bool(row.get("ai_clamped")),
                "human_realised_trophies": realised_by_turn.get(turn),
                "error": row.get("error"),
            }
        )
    return out


def log_path(root: Path | str, game_id: str) -> Path:
    return Path(root) / "value_log" / f"{game_id}.jsonl"


def read_log(root: Path | str | None, game_id: str) -> dict[str, Any]:
    """The rows for one finished game, read off disk.

    Returns `{"available": bool, "rows": [...], "reason": str|None}` rather than
    raising, because "this game has no readout" is the ORDINARY case and not an
    error: the log only exists for games played after show-value shipped, and
    nothing back-fills it. A page that cannot tell those apart from a broken
    read will either hide a real failure or cry wolf on every old game.

    A truncated trailing line is skipped, not fatal. The file is appended to and
    fsynced per row precisely so a kill mid-game leaves every resolved turn
    readable, and refusing to read the other 12 turns because the 13th was cut
    off would throw away exactly what that design bought.
    """
    if root is None or not game_id:
        return {"available": False, "rows": [], "reason": "value_log_disabled"}
    path = log_path(root, game_id)
    if not path.is_file():
        return {"available": False, "rows": [], "reason": "value_log_absent"}
    rows: list[dict[str, Any]] = []
    skipped = 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
    except OSError as exc:
        return {
            "available": False,
            "rows": [],
            "reason": f"value_log_unreadable:{type(exc).__name__}",
        }
    projected = project_rows(rows)
    if not projected:
        return {"available": False, "rows": [], "reason": "value_log_empty"}
    return {
        "available": True,
        "rows": projected,
        "reason": f"value_log_partial_lines:{skipped}" if skipped else None,
    }
