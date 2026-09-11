#!/usr/bin/env python3
"""Render saved games with the original, pinned replay-bot Canvas renderer.

    python tools/gallery.py [--records ...] [--out gallery] [--games N]

Accepts bundled `turns` and evaluate.py's `per_turn` JSONL (optionally gzipped).
Rendering never runs an agent or simulates a battle.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "data" / "gallery" / "w3b_games.jsonl.gz"
OUTCOME = {"win": "W", "loss": "L", "draw": "T"}
REPLAY_OUTCOME = {"W": "win", "L": "loss", "T": "draw"}

CSS = """
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 62rem;
       color: #1c1c1c; background: #fbfbfb; }
h1, h2 { font-weight: 600; }
table { border-collapse: collapse; width: 100%; }
td, th { padding: .45rem .5rem; vertical-align: middle; border-bottom: 1px solid #e3e3e3; }
th { text-align: left; font-weight: 600; color: #555; }
.replay { display: block; width: 100%; height: auto; }
.meta { color: #666; }
a { color: #06663d; }
"""


def replay_rows(game: dict) -> list[dict]:
    """Adapt records to the existing calc_rows contract, including board order.

    State/calculator slots run front-to-back; the replay renderer expects
    back-to-front on BOTH sides (the original _replay_order_* helpers).
    """
    def pets(board: list) -> list:
        if len(board) > 5:
            raise ValueError("a board must have at most five slots")
        result = []
        for slot in board:
            if not slot:
                result.append(None)
                continue
            level = int(slot.get("level") or 1)
            result.append({
                "name": slot["name"], "attack": slot["attack"],
                "health": slot["health"], "equipment": slot.get("item"),
                # Legacy example records retain levels, not partial XP.
                # The adapter hides their unknown XP bars instead of inventing XP.
                "level": level, "exp": slot.get("exp"),
                "tempAttack": slot.get("tempAttack", 0),
                "tempHealth": slot.get("tempHealth", 0),
            })
        return list(reversed(result + [None] * (5 - len(result))))

    rows = []
    for turn in game["turns"]:
        if turn["outcome"] not in REPLAY_OUTCOME:
            raise ValueError(f"unknown recorded outcome: {turn['outcome']}")
        lives_after = turn.get("lives")
        if type(lives_after) is not int:
            raise ValueError("recorded Arena lives must be an integer")
        rows.append({
            "turn": turn["turn"], "outcome": REPLAY_OUTCOME[turn["outcome"]],
            "opponentName": "Opponent", "playerPets": pets(turn["player"]),
            "opponentPets": pets(turn["opponent"]),
            # Released Arena uses one life per loss. Do not let the renderer
            # infer lives from its separate six-life/turn-three rule.
            "livesBefore": lives_after + int(turn["outcome"] == "L"),
        })
    return rows


def game_page(game: dict, png_name: str | None) -> str:
    body = (f'<h1>Game {game["game_index"]}</h1>'
            f'<p class="meta">{html.escape(str(game["turns_survived"]))} turns, '
            f'{html.escape(str(game["trophies"]))} trophies, ended: {html.escape(str(game["end_reason"]))}. '
            f'<a href="index.html">All games</a></p>'
            '<p>The agent is on the left; its opponent is on the right.</p>')
    if png_name:
        body += (f'<a href="{png_name}"><img class="replay" src="{png_name}" '
                 f'alt="Recorded battles for game {game["game_index"]}"></a>')
        if any("exp" not in s or s["exp"] is None for t in game["turns"]
               for side in ("player", "opponent") for s in t[side] if s):
            body += '<p class="meta">These example records retain levels, but not partial XP; unknown XP bars are omitted.</p>'
    else:
        body += '<p>No turns were recorded before this game stopped.</p>'
    return body


def page(title: str, body: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>\n")


def compact_game(record: dict, catalog: dict) -> dict:
    """Allowlisted display fields, using the existing release converter's rules.

    Engine attack/health already include temporary bonuses. Calculator opponent
    stats store tempAttack/tempHealth separately; add them exactly once.
    """
    if not isinstance(record, dict) or type(record.get("game_index")) is not int or record["game_index"] < 0:
        raise ValueError("missing or invalid game_index")
    if ("turns" in record) == ("per_turn" in record):
        raise ValueError("expected exactly one schema: compact turns or raw per_turn")
    turns = record.get("turns", record.get("per_turn"))
    if not isinstance(turns, list):
        raise ValueError("turns must be a list")
    if "turns" in record:
        for turn in turns:
            if not isinstance(turn, dict) or not all(k in turn for k in ("turn", "outcome", "lives", "opp_lives", "player", "opponent")):
                raise ValueError("incomplete compact turn")
            if not all(isinstance(turn[k], list) for k in ("player", "opponent")):
                raise ValueError("compact boards must be lists")
        return {key: record[key] for key in ("game_index", "trophies", "turns_survived", "end_reason", "turns")}

    pet_names = catalog["pets"]["id_to_name_id"]
    food_names = catalog["foods"]["id_to_name_id"]
    converted = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("invalid per_turn record")
        board = (turn.get("detail") or {}).get("board_pre_battle") or {}
        if not isinstance(board.get("team"), list) or not isinstance(turn.get("opponent_pets"), list):
            raise ValueError(f"game {record['game_index']} turn {turn.get('turn')}: recorded boards are missing")
        player = []
        for slot in board["team"]:
            if not slot or not slot.get("pet_id"):
                player.append(None)
                continue
            item = slot.get("equipment_id")
            player.append({
                "name": pet_names.get(slot["pet_id"], slot["pet_id"]),
                "attack": slot.get("attack"), "health": slot.get("health"),
                "level": slot.get("level") or (3 if slot.get("exp", 0) >= 5 else 2 if slot.get("exp", 0) >= 2 else 1),
                "exp": slot.get("exp"),
                "tempAttack": slot.get("temp_attack", 0),
                "tempHealth": slot.get("temp_health", 0),
                "item": food_names.get(item, item) if item else None,
            })
        opponent = []
        for slot in turn["opponent_pets"]:
            if not slot:
                opponent.append(None)
                continue
            equipment = slot.get("equipment")
            opponent.append({
                "name": str(slot.get("name") or "?"),
                "attack": (slot.get("attack") or 0) + (slot.get("tempAttack") or 0),
                "health": (slot.get("health") or 0) + (slot.get("tempHealth") or 0),
                "level": 3 if (slot.get("exp") or 0) >= 5 else 2 if (slot.get("exp") or 0) >= 2 else 1,
                "exp": slot.get("exp", 0),
                "tempAttack": slot.get("tempAttack", 0),
                "tempHealth": slot.get("tempHealth", 0),
                "item": equipment.get("name") if isinstance(equipment, dict) else equipment,
            })
        converted.append({"turn": turn["turn"], "lives": turn.get("lives"),
                          "opp_lives": turn.get("opp_lives"), "outcome": OUTCOME.get(turn.get("outcome"), "?"),
                          "player": player, "opponent": opponent})
    return {"game_index": record["game_index"], "trophies": record.get("trophies"),
            "turns_survived": record.get("turns_survived"), "end_reason": record.get("end_reason", "unknown"),
            "turns": converted}


def read_games(records: Path, max_games: int | None = None, catalog: dict | None = None) -> list[dict]:
    if max_games is not None and (type(max_games) is not int or max_games < 0):
        raise ValueError("game count must be nonnegative")
    if max_games == 0:
        return []
    games, indices = [], set()
    opener = gzip.open if records.suffix == ".gz" else open
    with opener(records, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict) and "per_turn" in record and catalog is None:
                # The shipped catalog also maps generated pets/equipment. This
                # is a data-only import, not the engine or a recommender.
                from sap_ppo.catalog import load_turtle_catalog
                from sap_ppo.constants import DATA_DIR
                if not (DATA_DIR / "turtle_catalog_v1.json").is_file():
                    raise FileNotFoundError("raw records need data/turtle_catalog_v1.json")
                catalog = load_turtle_catalog()
            game = compact_game(record, catalog or {})
            if game["game_index"] in indices:
                raise ValueError("duplicate game_index")
            indices.add(game["game_index"])
            games.append(game)
            if max_games is not None and len(games) >= max_games:
                break
    if not games:
        raise ValueError("records hold no games")
    return games


def render_records(records: Path, out: Path, *, max_games: int | None = None,
                   catalog: dict | None = None) -> dict:
    """Render the first N rows in input order, never selecting by outcome."""
    records, out = Path(records), Path(out)
    if max_games == 0:
        return {"status": "disabled", "games": 0}
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing gallery: {out}")
    games = read_games(records, max_games, catalog)
    from sap_ppo.opponents.replaybot_render_bridge import render_replay_image_from_calc_rows
    total_rows = 0
    links, pages, images = [], [], []
    for g in games:
        rows = replay_rows(g)
        total_rows += len(rows)
        png_name = f"game_{g['game_index']:03d}.png" if rows else None
        if rows:
            rendered = render_replay_image_from_calc_rows(rows, max_lives=5)
            png = rendered.get("image")
            if not rendered.get("ok") or not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"game {g['game_index']}: {rendered.get('error') or 'invalid replay PNG'}")
            images.append((png_name, png))
        body = game_page(g, png_name)
        name = f"game_{g['game_index']:03d}.html"
        pages.append((name, page(f"Game {g['game_index']}", body)))
        links.append(f'<tr><td><a href="{name}">game {g["game_index"]}</a></td>'
                     f'<td>{html.escape(str(g["turns_survived"]))}</td><td>{html.escape(str(g["trophies"]))}</td>'
                     f'<td class="meta">{html.escape(str(g["end_reason"]))}</td></tr>')

    index = (f'<h1>Gallery</h1><p>The first {len(games)} games in the supplied records, '
             'in input order, without selecting wins. These are recorded boards and outcomes; '
             'rendering does not run the agent or replay battles. '
             f'Source: <code>{html.escape(records.name)}</code>.</p>'
             '<table><tr><th>game</th><th>turns</th><th>trophies</th><th>ended</th></tr>'
             + "".join(links) + "</table>")
    pages.append(("index.html", page("SAP-Arena gallery", index)))
    out.mkdir(parents=True, exist_ok=False)
    for name, content in pages:
        with (out / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    for name, png in images:
        with (out / name).open("xb") as stream:
            stream.write(png)
    return {"status": "ok", "index": str(out / "index.html"), "games": len(games),
            "turn_rows": total_rows, "images": len(images), "renderer": "sap-replay-bot"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=ROOT / "gallery")
    ap.add_argument("--games", type=int, default=None, help="first N input games; default all, 0 disables")
    args = ap.parse_args()
    if args.games is not None and args.games < 0:
        ap.error("--games must be nonnegative")
    try:
        result = render_records(args.records, args.out, max_games=args.games)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FATAL: gallery rendering failed: {exc}", file=sys.stderr)
        return 2
    if result["status"] == "ok":
        print(f"gallery: {result['games']} games, {result['turn_rows']} turn rows -> {result['index']}")
        print(f"renderer: {result['renderer']}; replay images: {result['images']}")
    else:
        print("gallery: disabled (0 games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
