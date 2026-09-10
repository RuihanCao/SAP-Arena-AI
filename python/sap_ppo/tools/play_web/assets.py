"""Read-only asset paths for the play-web static skin.

The SAP UI font, icons and shop-scene backgrounds all live in the pinned
SAP-Calculator checkout and are served straight from disk with no processing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ASSET_UI_DIR = ROOT / "assets"
SAP_CALC_ART = ROOT.parent / "SAP-Calculator" / "SAP-Calculator" / "src" / "assets" / "art" / "Public" / "Public"
SAP_CALC_ICONS = SAP_CALC_ART / "Icons"
SAP_CALC_ICONS_SPLIT = SAP_CALC_ICONS / "TextMap-resources.assets-31-split"
SAP_CALC_FONTS = SAP_CALC_ART.parents[2] / "fonts"
SAP_CALC_BACKGROUNDS = SAP_CALC_ART / "Background"
SAP_CALC_DATA = SAP_CALC_ART.parents[2] / "data"

ICON_MAP: dict[str, Path] = {
    "fist": SAP_CALC_ICONS / "fist-from-textmap.png",
    "heart": SAP_CALC_ICONS / "heart-from-textmap.png",
    "gold": SAP_CALC_ICONS_SPLIT / "gold.png",
    "l1": SAP_CALC_ICONS_SPLIT / "l1.png",
    "l2": SAP_CALC_ICONS_SPLIT / "l2.png",
    "l3": SAP_CALC_ICONS_SPLIT / "l3.png",
    "levelbar": SAP_CALC_ICONS_SPLIT / "levelbar.png",
    "turn": ASSET_UI_DIR / "Turn.png",
    "roll": SAP_CALC_ICONS_SPLIT / "roll.png",
    "freeze": ASSET_UI_DIR / "Freeze-Unfreeze.png",
    "unfreeze": ASSET_UI_DIR / "Freeze-Unfreeze.png",
    "sell_strip": ASSET_UI_DIR / "Roll-Sell-EndTurn.png",
    "end_turn": ASSET_UI_DIR / "EndTurn.png",
}


@lru_cache(maxsize=2)
def _pack_ability_text(slot_type: str) -> dict[str, list[str]]:
    """Ability copy for the selection card, read straight out of the pinned pack.

    Pets carry one sentence per level, foods one sentence. Keyed by the pack's
    own NameId, which is what the repo catalog maps item ids onto. Read-only:
    nothing here changes an action, a cost or a game rule, and a missing or
    unreadable file simply means the card shows no sentence.
    """
    path = SAP_CALC_DATA / ("pets.json" if slot_type == "pet" else "food.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, list[str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        name_id = str(row.get("NameId") or "").strip()
        if not name_id:
            continue
        if slot_type == "pet":
            abilities = row.get("Abilities")
            if not isinstance(abilities, list):
                continue
            by_level: dict[int, str] = {}
            for entry in abilities:
                if not isinstance(entry, dict):
                    continue
                try:
                    level = int(entry.get("Level", 0))
                except (TypeError, ValueError):
                    continue
                about = str(entry.get("About") or "").strip()
                if level >= 1 and about:
                    by_level[level] = about
            if by_level:
                out[name_id] = [by_level.get(lv, by_level[min(by_level)]) for lv in (1, 2, 3)]
        else:
            about = str(row.get("Ability") or "").strip()
            if about:
                out[name_id] = [about]
    return out


FONT_MAP: dict[str, Path] = {
    "sap-bold": SAP_CALC_FONTS / "LapsusPro-Bold.otf",
    "sap-regular": SAP_CALC_FONTS / "LapsusPro-Regular.otf",
}

# Shop-phase ("Build") scene background. Any Background/*Build.png works; this is the one the
# live turn-1 Arena shop uses, so its terrain bands are exactly where the lanes are anchored
# (path bands at 45.2-56.3% and 71-82.5% of scene height, dark footer band at 90-100%).
DEFAULT_SCENE_BACKGROUND = "FieldBuild"

STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_ALLOWED_NAMES = {
    # the shop page (exp14's skin), served at /sandbox
    "index.html",
    "app.css",
    "app.js",
    # exp16 W5: the menu, the duel page and the replays list
    "landing.html",
    "play.html",
    "replays.html",
    "duel.css",
    "duel.js",
    "replays.js",
    # exp16 Amendment 6: one wording for the AI's search telemetry, loaded by
    # BOTH /play and /replays. Missing from this set it 404s, the page still
    # renders, and only the console says so.
    "search_telemetry.js",
    # every page carries this one: it is what notices a server restart
    "build_guard.js",
}


PLACEHOLDER_DIR = STATIC_DIR / "placeholder"

# Only the chrome that comes from real-game screenshots needs a stand-in. Pet and
# food art is read at runtime from the user's own SAP-Calculator checkout and is
# never redistributed, so it needs nothing here.
PLACEHOLDER_MAP: dict[str, str] = {
    "turn": "turn.svg",
    "freeze": "freeze.svg",
    "unfreeze": "freeze.svg",
    "sell_strip": "sell_strip.svg",
    "end_turn": "end_turn.svg",
}


def icon_or_placeholder(name: str) -> Path | None:
    """Real art if it is on disk, otherwise the drawn stand-in.

    Order matters and is the point: a checkout that has the original assets keeps
    rendering exactly as before, and one that does not still gets a usable
    screen instead of five broken images.
    """
    real = ICON_MAP.get(name)
    if real is not None and real.exists() and real.is_file():
        return real
    fallback = PLACEHOLDER_MAP.get(name)
    if fallback:
        drawn = PLACEHOLDER_DIR / fallback
        if drawn.exists() and drawn.is_file():
            return drawn
    return None


def static_path(name: str) -> Path | None:
    """Resolve a bundled static asset by name, whitelisted so a query string
    can never walk out of static/ onto an arbitrary path."""
    clean = (name or "").strip()
    if clean not in _STATIC_ALLOWED_NAMES:
        return None
    path = STATIC_DIR / clean
    if path.exists() and path.is_file():
        return path
    return None
