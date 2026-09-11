"""Battle oracle adapter using SAP-Calculator headless CLI."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import atexit
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..catalog import load_turtle_catalog
from ..constants import ROOT as SAP_PPO_ROOT

ROOT = Path(__file__).resolve().parents[3]
SAP_CALC_DIR = ROOT / "third_party" / "SAP-Calculator"
SAP_CALC_CLI = SAP_CALC_DIR / "simulation" / "dist" / "cli.js"
SAP_CALC_INDEX = SAP_CALC_DIR / "simulation" / "dist" / "index.js"
BATTLE_WORKER_ENV = "SAP_PPO_BATTLE_WORKER"
BATTLE_ORACLE_TIMEOUT_ENV = "SAP_PPO_BATTLE_ORACLE_TIMEOUT_SECONDS"


BATTLE_WORKER_MAX_CALLS_ENV = "SAP_PPO_BATTLE_WORKER_MAX_CALLS"
BATTLE_WORKER_MAX_RSS_MB_ENV = "SAP_PPO_BATTLE_WORKER_MAX_RSS_MB"
# 20k calls = 6x below the ~120k call death point measured in finding 16.
DEFAULT_BATTLE_WORKER_MAX_CALLS = 20000
# 600 MB = below the 0.9-1.9 GB range where GC thrash set in, above the
# ~60-90 MB a freshly started worker uses.
DEFAULT_BATTLE_WORKER_MAX_RSS_MB = 600.0
# RSS is read from /proc every this many calls (a ~50 us file read; doing it
# per call would be measurable against a ~5 ms battle).
BATTLE_WORKER_RSS_CHECK_EVERY = 256


SAP_CALC_PIN_SHA = "4d03d89b0c8df346bed9e9f504aefc0494f81155"
_pin_check_done = False


def _warn_if_calculator_drift() -> None:
    """One-time, non-fatal check that the calculator clone is at the pin.

    Silent when the clone (or git) is absent, exp worktrees have no
    third_party clone and the existing cli-missing error already covers that
    case.
    """
    global _pin_check_done
    if _pin_check_done:
        return
    _pin_check_done = True
    if not (SAP_CALC_DIR / ".git").exists():
        return
    try:
        proc = subprocess.run(
            ["git", "-C", str(SAP_CALC_DIR), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        return
    head = (proc.stdout or "").strip()
    if head and head != SAP_CALC_PIN_SHA:
        print(
            f"[sap_calc_battle_oracle] WARNING: {SAP_CALC_DIR} is at "
            f"{head[:12]}, expected pin {SAP_CALC_PIN_SHA[:12]} (pin-v45). "
            "Oracle results may not be comparable with prior experiments; "
            "see docs/dependency-pins.md.",
            file=sys.stderr,
        )

EQUIPMENT_NAME_BY_FOOD_ID: dict[str, str] = {
    "food-honey": "Honey",
    "food-meat-bone": "Meat Bone",
    "food-garlic": "Garlic",
    "food-chili": "Chili",
    "food-melon": "Melon",
    "food-mushroom": "Mushroom",
    "food-steak": "Steak",
    "food-peanut": "Peanut",
    "food-coconut": "Coconut",
    "food-birthday-cake": "Cake",
}

EQUIPMENT_NAME_BY_STATUS: dict[str, str] = {
    "status-honey-bee": "Honey",
    "status-bone-attack": "Meat Bone",
    "status-garlic-armor": "Garlic",
    "status-splash-attack": "Chili",
    "status-melon-armor": "Melon",
    "status-extra-life": "Mushroom",
    "status-steak-attack": "Steak",
    "status-peanut": "Peanut",
    "status-coconut-shield": "Coconut",
}

PACK_NAME_BY_LOWER: dict[str, str] = {
    "turtle": "Turtle",
    "puppy": "Puppy",
    "star": "Star",
    "golden": "Golden",
    "unicorn": "Unicorn",
    "danger": "Danger",
}

# Mirrors sap-library/lib/calculator.js key truncation.
KEY_MAP: dict[str, str] = {
    "playerPack": "pP",
    "opponentPack": "oP",
    "playerToy": "pT",
    "playerToyLevel": "pTL",
    "opponentToy": "oT",
    "opponentToyLevel": "oTL",
    "turn": "t",
    "playerGoldSpent": "pGS",
    "opponentGoldSpent": "oGS",
    "playerRollAmount": "pRA",
    "opponentRollAmount": "oRA",
    "playerSummonedAmount": "pSA",
    "opponentSummonedAmount": "oSA",
    "playerLevel3Sold": "pL3",
    "opponentLevel3Sold": "oL3",
    "playerTransformationAmount": "pTA",
    "opponentTransformationAmount": "oTA",
    "playerPets": "p",
    "opponentPets": "o",
    "angler": "an",
    "allPets": "ap",
    "logFilter": "lf",
    "fontSize": "fs",
    "customPacks": "cp",
    "oldStork": "os",
    "tokenPets": "tp",
    "komodoShuffle": "ks",
    "mana": "m",
    "showAdvanced": "sa",
    "ailmentEquipment": "ae",
    "triggersConsumed": "tc",
    "name": "n",
    "attack": "a",
    "health": "h",
    "exp": "e",
    "equipment": "eq",
    "belugaSwallowedPet": "bSP",
    "timesHurt": "tH",
}


def _default_opponent_team() -> list[dict[str, Any]]:
    return [
        {
            "slot_index": i,
            "pet_id": None,
            "attack": 0,
            "health": 0,
            "sell_value": 0,
            "level": 1,
            "exp": 0,
            "equipment_id": None,
            "status_effects": [],
        }
        for i in range(5)
    ]


def _to_name_id(pet_id: str | None) -> str | None:
    if pet_id is None:
        return None
    catalog = load_turtle_catalog()
    name_id = catalog.get("pets", {}).get("id_to_name_id", {}).get(pet_id)
    if name_id:
        return name_id
    base = pet_id.replace("pet-", "").replace("-", " ").title().replace(" ", "")
    return base


def _name_id_to_human(name_id: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name_id)).strip()


def _normalize_token(text: str | None) -> str:
    raw = str(text or "").strip().lower()
    return "".join(ch for ch in raw if ch.isalnum())


@lru_cache(maxsize=1)
def _equipment_name_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}

    def put(alias: str | None, canonical: str | None) -> None:
        key = _normalize_token(alias)
        value = str(canonical or "").strip()
        if key and value:
            aliases[key] = value

    for food_id, equipment_name in EQUIPMENT_NAME_BY_FOOD_ID.items():
        put(food_id, equipment_name)
        put(equipment_name, equipment_name)

    catalog = load_turtle_catalog()
    id_to_name_id = catalog.get("foods", {}).get("id_to_name_id", {})
    if isinstance(id_to_name_id, dict):
        for food_id, raw_name_id in id_to_name_id.items():
            if not isinstance(food_id, str) or not isinstance(raw_name_id, str):
                continue
            name_id = raw_name_id.strip()
            if not name_id:
                continue
            human = _name_id_to_human(name_id)
            put(food_id, human)
            put(name_id, human)
            put(human, human)

    for status_name, equipment_name in EQUIPMENT_NAME_BY_STATUS.items():
        put(status_name, equipment_name)
        put(equipment_name, equipment_name)

    return aliases


@lru_cache(maxsize=1)
def _name_id_to_display_name_map() -> dict[str, str]:
    out: dict[str, str] = {}
    pets_path = SAP_PPO_ROOT.parent / "SAP-Replay-Bot" / "sap-replay-bot" / "pets.json"
    if not pets_path.exists():
        return out
    try:
        rows = json.loads(pets_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        name_id = str(row.get("NameId") or "").strip()
        display = str(row.get("Name") or "").strip()
        if name_id and display:
            out[name_id] = display
    return out


def _name_id_to_display_name(name_id: str | None) -> str | None:
    if not isinstance(name_id, str) or not name_id.strip():
        return None
    raw = name_id.strip()
    mapping = _name_id_to_display_name_map()
    if raw in mapping:
        return mapping[raw]
    return _name_id_to_human(raw)


def _normalize_pack_name(pack_name: str | None, *, default: str = "Turtle") -> str:
    if not isinstance(pack_name, str):
        return default
    raw = pack_name.strip()
    if not raw:
        return default
    return PACK_NAME_BY_LOWER.get(raw.lower(), raw)


def _to_calculator_pet_name(slot: dict[str, Any]) -> str | None:
    explicit_name = slot.get("pet_name")
    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip()

    explicit_name_id = slot.get("pet_name_id")
    if isinstance(explicit_name_id, str) and explicit_name_id.strip():
        display_name = _name_id_to_display_name(explicit_name_id.strip())
        if display_name:
            return display_name

    pet_id = slot.get("pet_id")
    if isinstance(pet_id, str) and pet_id:
        name_id = _to_name_id(pet_id)
        display_name = _name_id_to_display_name(name_id)
        if display_name:
            return display_name

    if isinstance(explicit_name_id, str) and explicit_name_id.strip():
        return explicit_name_id.strip()

    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip()
    return None


def _to_equipment_name(slot: dict[str, Any]) -> str | None:
    explicit_name = slot.get("equipment_name")
    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip()

    equipment_id = slot.get("equipment_id")
    if isinstance(equipment_id, str) and equipment_id.strip():
        raw = equipment_id.strip()
        mapped = _equipment_name_alias_map().get(_normalize_token(raw))
        if mapped:
            return mapped
        # Accept display-name style equipment values from parsed replay states.
        return raw

    for effect in slot.get("status_effects", []) or []:
        mapped = _equipment_name_alias_map().get(_normalize_token(str(effect)))
        if mapped:
            return mapped
    return None


def _team_to_pet_configs(team: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    out: list[dict[str, Any] | None] = []
    for slot in team:
        pet_name = _to_calculator_pet_name(slot)
        if not pet_name:
            out.append(None)
            continue
        equipment_name = _to_equipment_name(slot)
        pet_cfg: dict[str, Any] = {
            "name": pet_name,
            "attack": int(slot.get("attack", 0)),
            "health": int(slot.get("health", 0)),
            "exp": int(slot.get("exp", 0)),
            "equipment": ({"name": equipment_name} if equipment_name is not None else None),
        }
        mana = slot.get("mana")
        if isinstance(mana, int) and mana != 0:
            pet_cfg["mana"] = int(mana)
        beluga = slot.get("beluga_swallowed_pet")
        if isinstance(beluga, str) and beluga.strip():
            pet_cfg["belugaSwallowedPet"] = beluga.strip()
        times_hurt = slot.get("times_hurt")
        if isinstance(times_hurt, int) and times_hurt != 0:
            pet_cfg["timesHurt"] = int(times_hurt)
        triggers_consumed = slot.get("triggers_consumed")
        if isinstance(triggers_consumed, int) and triggers_consumed != 0:
            pet_cfg["triggersConsumed"] = int(triggers_consumed)
        out.append(
            pet_cfg
        )
    while len(out) < 5:
        out.append(None)
    return out[:5]


def team_to_pet_configs(team: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Public wrapper for converting SAP-Arena team slots to calculator pet configs."""
    return _team_to_pet_configs(team)


def build_simulation_config(
    state: dict[str, Any],
    opponent_team: list[dict[str, Any]] | None = None,
    simulation_count: int = 64,
    *,
    player_pack: str | None = None,
    opponent_pack: str | None = None,
) -> dict[str, Any]:
    opponent_team = opponent_team or _default_opponent_team()
    player_pack_name = _normalize_pack_name(player_pack or str(state.get("pack") or "Turtle"), default="Turtle")
    opponent_pack_name = _normalize_pack_name(opponent_pack or player_pack_name, default="Turtle")
    return {
        "playerPack": player_pack_name,
        "opponentPack": opponent_pack_name,
        "turn": int(state.get("turn", 1)),
        "playerPets": team_to_pet_configs(state.get("team", [])),
        "opponentPets": team_to_pet_configs(opponent_team),
        "simulationCount": int(simulation_count),
        "logsEnabled": False,
    }


def _strip_pet_defaults(pet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not pet or not pet.get("name"):
        return None

    out: dict[str, Any] = {"name": pet.get("name")}
    if int(pet.get("attack", 0)) != 0:
        out["attack"] = int(pet.get("attack", 0))
    if int(pet.get("health", 0)) != 0:
        out["health"] = int(pet.get("health", 0))
    if int(pet.get("exp", 0)) != 0:
        out["exp"] = int(pet.get("exp", 0))
    equipment = pet.get("equipment")
    if equipment:
        out["equipment"] = equipment
    if int(pet.get("mana", 0)) != 0:
        out["mana"] = int(pet.get("mana", 0))
    if pet.get("belugaSwallowedPet") is not None:
        out["belugaSwallowedPet"] = pet.get("belugaSwallowedPet")
    if pet.get("timesHurt"):
        out["timesHurt"] = int(pet.get("timesHurt"))
    if isinstance(pet.get("triggersConsumed"), int) and int(pet.get("triggersConsumed")) != 0:
        out["triggersConsumed"] = int(pet.get("triggersConsumed"))
    return out


def _strip_default_values(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if config.get("playerPack") != "Turtle":
        out["playerPack"] = config.get("playerPack")
    if config.get("opponentPack") != "Turtle":
        out["opponentPack"] = config.get("opponentPack")

    if int(config.get("turn", 11)) != 11:
        out["turn"] = int(config.get("turn", 11))

    for key, default in (
        ("playerGoldSpent", 10),
        ("opponentGoldSpent", 10),
        ("playerRollAmount", 4),
        ("opponentRollAmount", 4),
        ("playerSummonedAmount", 0),
        ("opponentSummonedAmount", 0),
        ("playerLevel3Sold", 0),
        ("opponentLevel3Sold", 0),
        ("playerTransformationAmount", 0),
        ("opponentTransformationAmount", 0),
    ):
        if key in config and int(config.get(key, default)) != int(default):
            out[key] = int(config.get(key, default))

    for key in (
        "playerToy",
        "playerToyLevel",
        "opponentToy",
        "opponentToyLevel",
        "logFilter",
        "customPacks",
        "fontSize",
    ):
        value = config.get(key)
        if value is not None and value != "":
            if key == "fontSize" and int(value) == 13:
                continue
            if key in {"playerToyLevel", "opponentToyLevel"} and int(value) == 1:
                continue
            out[key] = value

    for key in ("angler", "allPets", "oldStork", "tokenPets", "komodoShuffle", "mana", "showAdvanced", "ailmentEquipment"):
        if bool(config.get(key)):
            out[key] = True

    stripped_player = [_strip_pet_defaults(p) for p in list(config.get("playerPets", []))]
    if any(p is not None for p in stripped_player):
        out["playerPets"] = stripped_player

    stripped_opponent = [_strip_pet_defaults(p) for p in list(config.get("opponentPets", []))]
    if any(p is not None for p in stripped_opponent):
        out["opponentPets"] = stripped_opponent

    return out


def _truncate_keys(value: Any) -> Any:
    if isinstance(value, list):
        return [_truncate_keys(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, nested in value.items():
            out[KEY_MAP.get(str(key), str(key))] = _truncate_keys(nested)
        return out
    return value


def generate_calculator_link(config: dict[str, Any]) -> str:
    stripped = _strip_default_values(config)
    truncated = _truncate_keys(stripped)
    packed = json.dumps(truncated, separators=(",", ":"))
    encoded = base64.b64encode(packed.encode("utf-8")).decode("utf-8")
    return f"https://sap-calculator.com/?c={encoded}"


def battle_outcome_from_result(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    try:
        p = int(payload.get("playerWins", 0))
        o = int(payload.get("opponentWins", 0))
        d = int(payload.get("draws", 0))
    except (TypeError, ValueError):
        return "unknown"

    if p > o and p > d:
        return "win"
    if o > p and o > d:
        return "loss"
    if d > p and d > o:
        return "draw"
    if p == 1 and o == 0 and d == 0:
        return "win"
    if o == 1 and p == 0 and d == 0:
        return "loss"
    if d == 1 and p == 0 and o == 0:
        return "draw"
    return "unknown"


def run_battle_oracle(
    state: dict[str, Any],
    opponent_team: list[dict[str, Any]] | None = None,
    simulation_count: int = 64,
    *,
    player_pack: str | None = None,
    opponent_pack: str | None = None,
) -> dict[str, Any]:
    config = build_simulation_config(
        state,
        opponent_team=opponent_team,
        simulation_count=simulation_count,
        player_pack=player_pack,
        opponent_pack=opponent_pack,
    )
    return run_battle_oracle_with_config(config)


def _plain_battle_log_message(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r'<img[^>]*\balt="([^"]+)"[^>]*>', r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def battle_trace_from_result(result: dict[str, Any] | None, *, max_events: int = 80) -> list[dict[str, str]]:
    """Normalize one calculator battle log into a bounded plain-text trace."""
    if not isinstance(result, dict):
        return []
    battles = result.get("battles")
    if not isinstance(battles, list) or not battles or not isinstance(battles[0], dict):
        return []
    logs = battles[0].get("logs")
    if not isinstance(logs, list):
        return []
    out: list[dict[str, str]] = []
    for row in logs[: max(0, int(max_events))]:
        if not isinstance(row, dict):
            continue
        message = _plain_battle_log_message(row.get("rawMessage") or row.get("message"))
        if not message:
            continue
        out.append({"type": str(row.get("type") or "event"), "message": message})
    return out


def run_battle_oracle_trace(
    state: dict[str, Any],
    opponent_team: list[dict[str, Any]],
    *,
    seed: int = 0,
    max_events: int = 80,
) -> dict[str, Any]:
    """Run one independently seeded battle and expose its bounded event log."""
    config = build_simulation_config(state, opponent_team=opponent_team, simulation_count=1)
    config.update(
        {
            "logsEnabled": True,
            "maxLoggedBattles": 1,
            "seed": int(seed),
        }
    )
    out = run_battle_oracle_with_config(config)
    result = out.get("result") if isinstance(out.get("result"), dict) else None
    public = {
        "ok": bool(out.get("ok")),
        "error": out.get("error"),
        "outcome": out.get("outcome"),
        "calculator_link": out.get("calculator_link"),
        "seed": int(seed),
        "trace": battle_trace_from_result(result, max_events=max_events),
    }
    if public["ok"] and not public["trace"]:
        public["ok"] = False
        public["error"] = "battle_trace_missing"
    return public


def _battle_worker_enabled() -> bool:
    raw = os.getenv(BATTLE_WORKER_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _timeout_seconds_from_env(env_name: str, *, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return float(max(1.0, value))


def _battle_oracle_timeout_seconds() -> float:
    return _timeout_seconds_from_env(BATTLE_ORACLE_TIMEOUT_ENV, default=60.0)


def _number_from_env(env_name: str, *, default: float) -> float:
    """Non-negative number from the environment; 0 (or a bad value spelled
    as 0) DISABLES the limit that reads it. Negative inputs clamp to 0.
    """
    raw = os.getenv(env_name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return float(max(0.0, value))


def _battle_worker_max_calls() -> int:
    return int(_number_from_env(BATTLE_WORKER_MAX_CALLS_ENV, default=float(DEFAULT_BATTLE_WORKER_MAX_CALLS)))


def _battle_worker_max_rss_mb() -> float:
    return _number_from_env(BATTLE_WORKER_MAX_RSS_MB_ENV, default=float(DEFAULT_BATTLE_WORKER_MAX_RSS_MB))


def _process_rss_mb(pid: int) -> float | None:
    """Resident set size of `pid` in MB, or None when it cannot be read
    (non-Linux, dead process, permissions). Never raises: an unreadable RSS
    must degrade to "no RSS-based recycling", never to a failed battle.
    """
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


def _readline_with_timeout(stream: Any, *, timeout_sec: float) -> tuple[bool, str | None, str | None]:
    timeout_norm = float(max(1.0, timeout_sec))
    holder: dict[str, str] = {}

    def _reader() -> None:
        try:
            holder["line"] = stream.readline()
        except Exception as exc:
            holder["error"] = f"{type(exc).__name__}:{exc}"

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    thread.join(timeout=timeout_norm)
    if thread.is_alive():
        return False, None, f"battle_worker_timeout:{timeout_norm:.1f}s"
    if "error" in holder:
        return False, None, f"battle_worker_readline_failed:{holder['error']}"
    return True, holder.get("line"), None


class _BattleWorker:
    """Persistent node battle worker with a bounded lifetime.

    1. LEAK: the node process's heap grows with call volume until it GC-
       thrashes and stops answering. Fixed by RECYCLING: the worker is
       killed and transparently restarted once it has served
       `SAP_PPO_BATTLE_WORKER_MAX_CALLS` calls, or once its RSS exceeds
       `SAP_PPO_BATTLE_WORKER_MAX_RSS_MB` (checked every
       `BATTLE_WORKER_RSS_CHECK_EVERY` calls). Either limit set to 0
       disables that check.
    2. SELF-DEADLOCK: `run()` held a non-reentrant `threading.Lock` and, on
       the read timeout that a thrashing worker eventually trips, called
       `stop()`, which tried to take the SAME lock -- so the timeout that
       existed to prevent a hang was itself the hang (the `do_wait` frame
       observed in the stalled shards is CPython's lock acquire). Fixed by
       an `RLock` plus an explicit already-locked `_stop_locked()` used on
       every internal path.

    External behavior is otherwise unchanged: same `run()` return shape,
    same errors, same `stop()`. `stats()` is additive telemetry."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        # Reentrant on purpose -- see the class docstring's point 2.
        self._lock = threading.RLock()
        # Calls served by the CURRENT process, and run-lifetime totals.
        self._calls_on_proc = 0
        self._total_calls = 0
        self._recycles = 0
        self._recycles_by_reason: dict[str, int] = {}
        self._last_rss_mb: float | None = None

    def stats(self) -> dict[str, Any]:
        """Telemetry for run reports: how many oracle calls this process
        made, how often the worker had to be recycled and why, and the last
        RSS sample taken."""
        with self._lock:
            return {
                "total_calls": int(self._total_calls),
                "calls_on_current_worker": int(self._calls_on_proc),
                "recycles": int(self._recycles),
                "recycles_by_reason": dict(sorted(self._recycles_by_reason.items())),
                "last_rss_mb": (round(float(self._last_rss_mb), 1) if self._last_rss_mb is not None else None),
                "max_calls": _battle_worker_max_calls(),
                "max_rss_mb": _battle_worker_max_rss_mb(),
            }

    def _recycle_locked(self, reason: str) -> None:
        """Kill the current worker so the next call starts a fresh one, and
        COUNT it. Caller must hold `self._lock`."""
        rss = self._last_rss_mb
        calls = self._calls_on_proc
        self._stop_locked()
        self._recycles += 1
        self._recycles_by_reason[reason] = self._recycles_by_reason.get(reason, 0) + 1
        print(
            f"[sap_calc_battle_oracle] recycling battle worker: reason={reason} "
            f"calls_on_worker={calls} rss_mb={'n/a' if rss is None else f'{rss:.1f}'} "
            f"total_calls={self._total_calls} recycles={self._recycles}",
            file=sys.stderr,
            flush=True,
        )

    def _maybe_recycle_locked(self) -> None:
        """Enforce the call and RSS budgets BEFORE handing work to the
        current worker. Caller must hold `self._lock`.
        """
        if self._proc is None or self._proc.poll() is not None:
            return
        max_calls = _battle_worker_max_calls()
        if max_calls > 0 and self._calls_on_proc >= max_calls:
            self._recycle_locked("call_budget")
            return
        max_rss_mb = _battle_worker_max_rss_mb()
        if max_rss_mb <= 0 or self._calls_on_proc <= 0:
            return
        if self._calls_on_proc % int(max(1, BATTLE_WORKER_RSS_CHECK_EVERY)) != 0:
            return
        rss_mb = _process_rss_mb(self._proc.pid)
        if rss_mb is None:
            return
        self._last_rss_mb = rss_mb
        if rss_mb >= max_rss_mb:
            self._recycle_locked("rss")

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        node_script = (
            "const readline=require('readline');"
            "const sim=require(process.argv[1]);"
            "const publicResult=(result)=>{"
            "if(Array.isArray(result&&result.battles)){"
            "result.battles=result.battles.map((battle)=>({logs:Array.isArray(battle&&battle.logs)?battle.logs.map((row)=>({"
            "type:row&&row.type,rawMessage:row&&row.rawMessage,message:row&&row.message,bold:!!(row&&row.bold)"
            "})):[]}));"
            "}"
            "return result;"
            "};"
            "const rl=readline.createInterface({input:process.stdin,crlfDelay:Infinity});"
            "rl.on('line',(line)=>{"
            "if(!line){return;}"
            "let msg=null;"
            "try{msg=JSON.parse(line);}catch(e){process.stdout.write(JSON.stringify({ok:false,error:'invalid_json:'+String(e)})+'\\n');return;}"
            "try{"
            "const opts=msg.options||{};"
            "const result=publicResult(sim.runHeadlessSimulation(msg.config||{}, {includeBattles:!!opts.includeBattles, enableLogs:!!opts.enableLogs}));"
            "process.stdout.write(JSON.stringify({ok:true,result:result})+'\\n');"
            "}catch(e){"
            "const err=(e&&e.stack)?e.stack:String(e);"
            "process.stdout.write(JSON.stringify({ok:false,error:'worker_simulation_failed:'+err})+'\\n');"
            "}"
            "});"
            "rl.on('close',()=>process.exit(0));"
        )
        self._proc = subprocess.Popen(
            ["node", "-e", node_script, str(SAP_CALC_INDEX)],
            cwd=str(SAP_CALC_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        with self._lock:


            self._maybe_recycle_locked()
            self._start()
            proc = self._proc
            if proc is None or proc.stdin is None or proc.stdout is None:
                return {"ok": False, "error": "battle_worker_not_ready", "result": None}
            try:
                enable_logs = bool(config.get("logsEnabled"))
                proc.stdin.write(
                    json.dumps(
                        {
                            "config": config,
                            "options": {
                                "includeBattles": enable_logs,
                                "enableLogs": enable_logs,
                            },
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                proc.stdin.flush()
            except Exception as exc:
                self._recycle_locked("write_failed")
                return {"ok": False, "error": f"battle_worker_write_failed:{type(exc).__name__}:{exc}", "result": None}
            # This worker has now been handed a call, whatever comes back.
            self._calls_on_proc += 1
            self._total_calls += 1

            ok_read, line, read_error = _readline_with_timeout(
                proc.stdout,
                timeout_sec=_battle_oracle_timeout_seconds(),
            )
            if not ok_read:
                # A read that timed out and a reader thread that blew up are
                # different operational stories, so they get different
                # reasons -- both are worker REPLACEMENTS and both count.
                self._recycle_locked(
                    "timeout_replacement"
                    if str(read_error or "").startswith("battle_worker_timeout")
                    else "readline_failed"
                )
                return {"ok": False, "error": str(read_error or "battle_worker_timeout"), "result": None}
            if not line:
                err = ""
                if proc.poll() is not None and proc.stderr is not None:
                    try:
                        err = (proc.stderr.read() or "").strip()
                    except Exception:
                        err = ""
                self._recycle_locked("no_output")
                return {
                    "ok": False,
                    "error": f"battle_worker_no_output:{err or 'process_terminated_or_stalled'}",
                    "result": None,
                }
            try:
                payload = json.loads(line)
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"battle_worker_output_invalid:{type(exc).__name__}:{exc}",
                    "result": None,
                }
            if not isinstance(payload, dict):
                return {"ok": False, "error": "battle_worker_output_non_object", "result": None}
            if not payload.get("ok"):
                return {
                    "ok": False,
                    "error": str(payload.get("error") or "battle_worker_simulation_failed"),
                    "result": None,
                }
            return {"ok": True, "error": None, "result": payload.get("result")}

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        """`stop()`'s body, for callers that ALREADY hold `self._lock`.

        Every in-`run()` teardown goes through here: calling the public
        `stop()` from inside the locked region is what deadlocked the
        drivers before the lock became reentrant, and an explicit
        already-locked entry point keeps that mistake unavailable even if
        the lock type changes back.
        """
        proc = self._proc
        self._proc = None
        self._calls_on_proc = 0
        self._last_rss_mb = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    proc.kill()
        except Exception:
            pass


_BATTLE_WORKER = _BattleWorker()
atexit.register(_BATTLE_WORKER.stop)


def battle_worker_stats() -> dict[str, Any]:
    """Battle worker stats."""
    return _BATTLE_WORKER.stats()


def run_battle_oracle_with_config(config: dict[str, Any]) -> dict[str, Any]:
    calculator_link = generate_calculator_link(config)
    _warn_if_calculator_drift()

    if not SAP_CALC_CLI.exists():
        return {
            "ok": False,
            "error": f"sap_calculator_cli_missing:{SAP_CALC_CLI}",
            "result": None,
            "outcome": "unknown",
            "calculator_link": calculator_link,
            "config": config,
        }

    if _battle_worker_enabled() and SAP_CALC_INDEX.exists():
        worker_out = _BATTLE_WORKER.run(config)
        if worker_out.get("ok"):
            payload = worker_out.get("result")
            if isinstance(payload, dict):
                return {
                    "ok": True,
                    "error": None,
                    "result": payload,
                    "outcome": battle_outcome_from_result(payload),
                    "calculator_link": calculator_link,
                    "config": config,
                }
        # Worker failure falls back to one-shot CLI for resilience.

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fp:
        json.dump(config, fp)
        temp_path = Path(fp.name)

    try:
        if bool(config.get("logsEnabled")):
            node_script = (
                "const fs=require('fs');"
                "const sim=require(process.argv[1]);"
                "const config=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));"
                "const result=sim.runHeadlessSimulation(config,{includeBattles:true,enableLogs:true});"
                "if(Array.isArray(result&&result.battles)){"
                "result.battles=result.battles.map((battle)=>({logs:Array.isArray(battle&&battle.logs)?battle.logs.map((row)=>({"
                "type:row&&row.type,rawMessage:row&&row.rawMessage,message:row&&row.message,bold:!!(row&&row.bold)"
                "})):[]}));"
                "}"
                "process.stdout.write(JSON.stringify(result));"
            )
            command = ["node", "-e", node_script, str(SAP_CALC_INDEX), str(temp_path)]
        else:
            command = ["node", str(SAP_CALC_CLI), "--input", str(temp_path)]
        proc = subprocess.run(
            command,
            cwd=str(SAP_CALC_DIR),
            capture_output=True,
            text=True,
            check=False,
            timeout=_battle_oracle_timeout_seconds(),
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "error": f"sap_calculator_failed:{proc.stderr.strip()}",
                "result": None,
                "outcome": "unknown",
                "calculator_link": calculator_link,
                "config": config,
            }
        payload = json.loads(proc.stdout)
        return {
            "ok": True,
            "error": None,
            "result": payload,
            "outcome": battle_outcome_from_result(payload),
            "calculator_link": calculator_link,
            "config": config,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"sap_calculator_timeout:{_battle_oracle_timeout_seconds():.1f}s",
            "result": None,
            "outcome": "unknown",
            "calculator_link": calculator_link,
            "config": config,
        }
    except Exception as exc:  # broad by design for oracle wrappers
        return {
            "ok": False,
            "error": f"exception:{type(exc).__name__}:{exc}",
            "result": None,
            "outcome": "unknown",
            "calculator_link": calculator_link,
            "config": config,
        }
    finally:
        temp_path.unlink(missing_ok=True)
