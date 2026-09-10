"""Bridge helpers for invoking replay-bot parser from Python."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..constants import ROOT

REPLAY_BOT_DIR = ROOT.parent / "SAP-Replay-Bot" / "sap-replay-bot"
REPLAY_BOT_CALCULATOR = REPLAY_BOT_DIR / "lib" / "calculator.js"


def parse_replay_for_calculator_state(
    battle: dict[str, Any] | None,
    build_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse one replay battle into calculator state via replay-bot logic.

    Returns:
    - ok: bool
    - error: str | None
    - state: dict | None
    """
    if not isinstance(battle, dict):
        return {"ok": False, "error": "invalid_battle_json", "state": None}
    if not REPLAY_BOT_CALCULATOR.exists():
        return {
            "ok": False,
            "error": f"replaybot_calculator_missing:{REPLAY_BOT_CALCULATOR}",
            "state": None,
        }

    payload = {"battle": battle, "buildModel": build_model}
    node_script = (
        "const fs=require('fs');"
        "const payload=JSON.parse(fs.readFileSync(process.argv[1],'utf8'));"
        "const mod=require(process.argv[2]);"
        "const out=mod.parseReplayForCalculator(payload.battle,payload.buildModel||null);"
        "process.stdout.write(JSON.stringify(out));"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
        json.dump(payload, fp)
        payload_path = Path(fp.name)

    try:
        proc = subprocess.run(
            ["node", "-e", node_script, str(payload_path), str(REPLAY_BOT_CALCULATOR)],
            cwd=str(REPLAY_BOT_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or (proc.stdout or "").strip() or "unknown_error"
            return {"ok": False, "error": f"replaybot_parse_failed:{err}", "state": None}
        try:
            state = json.loads(proc.stdout)
        except Exception as exc:
            return {
                "ok": False,
                "error": f"replaybot_parse_output_invalid:{type(exc).__name__}:{exc}",
                "state": None,
            }
        if not isinstance(state, dict):
            return {"ok": False, "error": "replaybot_parse_non_object_output", "state": None}
        return {"ok": True, "error": None, "state": state}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"ok": False, "error": f"replaybot_bridge_exception:{type(exc).__name__}:{exc}", "state": None}
    finally:
        payload_path.unlink(missing_ok=True)

