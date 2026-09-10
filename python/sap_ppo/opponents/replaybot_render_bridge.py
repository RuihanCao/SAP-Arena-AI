"""Bridge helpers for invoking replay-bot image renderer from Python."""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event
from typing import Any

from ..constants import ROOT

REPLAY_BOT_DIR = ROOT.parent / "SAP-Replay-Bot" / "sap-replay-bot"
if not REPLAY_BOT_DIR.exists():
    # e.g. sap_ppo imported from a git worktree outside SAP-Workspace: fall
    # back to the canonical workspace checkout (CLAUDE.md layout).
    _override = os.environ.get("SAP_REPLAY_BOT_DIR", "").strip()
    if _override and Path(_override).exists():
        REPLAY_BOT_DIR = Path(_override)
REPLAY_BOT_RENDER = REPLAY_BOT_DIR / "lib" / "render.js"
REPLAY_BOT_BATTLE = REPLAY_BOT_DIR / "lib" / "battle.js"
REPLAY_BOT_DATA = REPLAY_BOT_DIR / "lib" / "data.js"
REPLAY_BOT_CONFIG = REPLAY_BOT_DIR / "lib" / "config.js"
REPLAY_BOT_RENDER_BRIDGE = ROOT / "tools" / "replaybot_render_bridge.js"
REPLAY_RENDER_TIMEOUT_ENV = "SAP_PPO_REPLAY_RENDER_TIMEOUT_SECONDS"


def _ensure_paths() -> str | None:
    required = [
        REPLAY_BOT_DIR,
        REPLAY_BOT_RENDER,
        REPLAY_BOT_BATTLE,
        REPLAY_BOT_DATA,
        REPLAY_BOT_CONFIG,
        REPLAY_BOT_RENDER_BRIDGE,
    ]
    for path in required:
        if not path.exists():
            return f"missing_path:{path}"
    return None


def _render_timeout_seconds() -> float:
    raw = os.getenv(REPLAY_RENDER_TIMEOUT_ENV)
    if raw is None:
        return 60.0
    try:
        value = float(raw)
    except Exception:
        return 60.0
    return float(max(1.0, value))


def _stop_process_group(proc: subprocess.Popen[str]) -> None:
    """Stop the node bridge and any child it may have spawned."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass
    # The group leader may obey TERM while a descendant ignores it. Probe the
    # group itself and kill any survivor, rather than treating leader exit as
    # proof that the whole renderer tree is gone.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait(timeout=1.0)


def _close_process_pipes(proc: subprocess.Popen[str]) -> None:
    for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        if stream is not None and not stream.closed:
            stream.close()


def _run_render(
    payload: dict[str, Any], *, cancel_event: Event | None = None
) -> dict[str, Any]:
    missing = _ensure_paths()
    if missing is not None:
        return {"ok": False, "error": missing, "image": None}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
        json.dump(payload, fp)
        payload_path = Path(fp.name)

    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            [
                "node",
                str(REPLAY_BOT_RENDER_BRIDGE),
                str(payload_path),
                str(REPLAY_BOT_RENDER),
                str(REPLAY_BOT_BATTLE),
                str(REPLAY_BOT_DATA),
                str(REPLAY_BOT_CONFIG),
            ],
            cwd=str(REPLAY_BOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + _render_timeout_seconds()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _stop_process_group(proc)
                return {"ok": False, "error": "replaybot_render_cancelled", "image": None}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop_process_group(proc)
                return {
                    "ok": False,
                    "error": f"replaybot_render_timeout:{_render_timeout_seconds():.1f}s",
                    "image": None,
                }
            try:
                stdout, stderr = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if proc.returncode != 0:
            err = (stderr or "").strip() or (stdout or "").strip() or "unknown_error"
            return {"ok": False, "error": f"replaybot_render_failed:{err}", "image": None}

        stdout = (stdout or "").strip()
        if not stdout:
            return {"ok": False, "error": "replaybot_render_empty_output", "image": None}

        # replay-bot libs can console.log noise before the bridge writes its
        # base64 (e.g. drawing.js prints "undefined" per pet on the calc-rows
        # path; those letters are valid base64, so a lenient decode silently
        # corrupts the image). The bridge writes the image as the final line;
        # decode strictly so any residual pollution fails loud instead.
        stdout = stdout.splitlines()[-1].strip()

        try:
            image_bytes = base64.b64decode(stdout, validate=True)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"replaybot_render_base64_invalid:{type(exc).__name__}:{exc}",
                "image": None,
            }

        if not image_bytes:
            return {"ok": False, "error": "replaybot_render_empty_image", "image": None}

        return {"ok": True, "error": None, "image": image_bytes}
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"replaybot_render_timeout:{_render_timeout_seconds():.1f}s",
            "image": None,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"ok": False, "error": f"replaybot_render_exception:{type(exc).__name__}:{exc}", "image": None}
    finally:
        if proc is not None:
            _close_process_pipes(proc)
        payload_path.unlink(missing_ok=True)


def render_replay_image_from_raw_battles(
    battles: list[dict[str, Any]],
    *,
    max_lives: int = 6,
    player_name: str | None = None,
    header_opponent_name: str | None = None,
) -> dict[str, Any]:
    """Render replay image from raw replay battle JSON objects using replay-bot."""
    payload = {
        "mode": "battle_json",
        "battles": battles,
        "maxLives": int(max_lives),
        "playerName": player_name,
        "headerOpponentName": header_opponent_name,
    }
    return _run_render(payload)


def render_replay_image_from_calc_rows(
    rows: list[dict[str, Any]],
    *,
    max_lives: int = 6,
    player_name: str | None = None,
    header_opponent_name: str | None = None,
    include_odds: bool = False,
    win_percent_results: list[dict[str, str]] | None = None,
    cancel_event: Event | None = None,
) -> dict[str, Any]:
    """Render replay image from calculator-style row payloads using replay-bot renderer."""
    payload = {
        "mode": "calc_rows",
        "battles": rows,
        "maxLives": int(max_lives),
        "playerName": player_name,
        "headerOpponentName": header_opponent_name,
        "includeOdds": bool(include_odds),
        "winPercentResults": [
            {
                "player": str(row.get("player", "")),
                "opponent": str(row.get("opponent", "")),
                "draw": str(row.get("draw", "")),
            }
            for row in (win_percent_results or [])
            if isinstance(row, dict)
        ],
    }
    return _run_render(payload, cancel_event=cancel_event)
