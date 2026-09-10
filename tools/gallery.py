#!/usr/bin/env python3
"""Render recorded boards without running an agent or simulating battles.

Accepts compact bundled `turns` records and evaluate.py's raw `per_turn` records,
as plain or gzipped JSONL. Pet art comes from the local pinned calculator;
missing pictures fall back to names. Records keep their original configuration.

    python tools/gallery.py [--records ...] [--out gallery] [--games N] [--art DIR]
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "data" / "gallery" / "w3b_games.jsonl.gz"
# The layout the README pins: the calculator checkout sits beside this repository.
DEFAULT_ART = ROOT.parent / "SAP-Calculator" / "SAP-Calculator" / "src" / "assets" / "art" / "Public" / "Public"

TILES = {"art": 0, "by_name": 0, "empty": 0}
OUTCOME = {"win": "W", "loss": "L", "draw": "T"}
ROW_BG = {"W": "#e7f6e9", "L": "#fdeaea", "T": "#f0f0f0"}
ROW_LABEL = {"W": "won", "L": "lost", "T": "drew"}

CSS = """
body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 62rem;
       color: #1c1c1c; background: #fbfbfb; }
h1, h2 { font-weight: 600; }
table { border-collapse: collapse; width: 100%; }
td, th { padding: .45rem .5rem; vertical-align: middle; border-bottom: 1px solid #e3e3e3; }
th { text-align: left; font-weight: 600; color: #555; }
.side { display: flex; gap: .3rem; }
.slot { width: 62px; text-align: center; font-size: 11px; color: #333; }
.slot img { width: 46px; height: 46px; object-fit: contain; display: block; margin: 0 auto; }
.ph { width: 46px; height: 46px; margin: 0 auto; border: 1px dashed #b9b9b9; border-radius: 8px;
      display: flex; align-items: center; justify-content: center; font-size: 9px;
      color: #666; padding: 2px; overflow: hidden; }
.stat { color: #444; }
.item { color: #7a5; }
.empty { opacity: .28; }
.meta { color: #666; }
a { color: #06663d; }
"""


def tile(slot: dict | None, art: Path, out_dir: Path) -> str:
    if not slot:
        TILES["empty"] += 1
        return '<div class="slot empty"><div class="ph">-</div></div>'
    name = str(slot.get("name") or "?")
    png = art / "Pets" / f"{name}.png"
    if not any(c in name for c in ("/", "\\")) and png.is_file():
        # Relative, and computed WITHOUT resolving symlinks: the page must point at
        # the reader's own checkout and carry none of the art itself, and a
        # `resolve()` here would follow a symlinked checkout out of the tree and
        # bake an absolute machine path into a published page.
        src = html.escape(os.path.relpath(png, out_dir).replace(os.sep, "/"))
        TILES["art"] += 1
        pic = f'<img src="{src}" alt="{html.escape(name)}">'
    else:
        TILES["by_name"] += 1
        pic = f'<div class="ph">{html.escape(name)}</div>'
    lvl = slot.get("level") or 1
    item = f'<div class="item">{html.escape(str(slot["item"]))}</div>' if slot.get("item") else ""
    return (f'<div class="slot">{pic}'
            f'<div class="stat">{html.escape(str(slot.get("attack")))}/{html.escape(str(slot.get("health")))}'
            f'{f" L{html.escape(str(lvl))}" if lvl != 1 else ""}</div>{item}</div>')


def game_page(game: dict, art: Path, out_dir: Path) -> tuple[str, int]:
    rows = []
    for t in game["turns"]:
        left = "".join(tile(s, art, out_dir) for s in t["player"])
        right = "".join(tile(s, art, out_dir) for s in t["opponent"])
        bg = ROW_BG.get(t["outcome"], "#fff")
        rows.append(
            f'<tr style="background:{bg}">'
            f'<td>{html.escape(str(t["turn"]))}</td>'
            f'<td><div class="side">{left}</div></td>'
            f'<td>{ROW_LABEL.get(t["outcome"], "?")}</td>'
            f'<td><div class="side">{right}</div></td>'
            f'<td class="meta">{html.escape(str(t["lives"]))} v {html.escape(str(t["opp_lives"]))}</td></tr>')
    body = (f'<h1>Game {game["game_index"]}</h1>'
            f'<p class="meta">{html.escape(str(game["turns_survived"]))} turns, '
            f'{html.escape(str(game["trophies"]))} trophies, ended: {html.escape(str(game["end_reason"]))}. '
            f'<a href="index.html">All games</a></p>'
            '<table><tr><th>turn</th><th>the agent</th><th></th><th>opponent</th>'
            '<th>lives</th></tr>' + "".join(rows) + "</table>"
            + ("" if rows else '<p>No turns were recorded before this game stopped.</p>'))
    return body, len(rows)


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
                   art: Path = DEFAULT_ART, catalog: dict | None = None) -> dict:
    """Render the first N rows in input order, never selecting by outcome."""
    records, out, art = Path(records), Path(out), Path(art)
    if max_games == 0:
        return {"status": "disabled", "games": 0}
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing gallery: {out}")
    games = read_games(records, max_games, catalog)
    TILES.update(art=0, by_name=0, empty=0)
    total_rows = 0
    links, pages = [], []
    for g in games:
        body, n = game_page(g, art, out)
        total_rows += n
        name = f"game_{g['game_index']:03d}.html"
        pages.append((name, page(f"Game {g['game_index']}", body)))
        links.append(f'<tr><td><a href="{name}">game {g["game_index"]}</a></td>'
                     f'<td>{html.escape(str(g["turns_survived"]))}</td><td>{html.escape(str(g["trophies"]))}</td>'
                     f'<td class="meta">{html.escape(str(g["end_reason"]))}</td></tr>')

    have_art = art.is_dir()
    note = ("" if have_art else
            '<p class="meta"><b>No art found</b> at the pinned SAP-Calculator checkout, so '
            'every tile shows the pet name instead of its picture. That is the fallback, not '
            'a failure: no game art is published with this repository.</p>')
    index = (f'<h1>Gallery</h1><p>The first {len(games)} games in the supplied records, '
             'in input order, without selecting wins. These are recorded boards and outcomes; '
             'rendering does not run the agent or replay battles. '
             f'Source: <code>{html.escape(records.name)}</code>.</p>{note}'
             '<table><tr><th>game</th><th>turns</th><th>trophies</th><th>ended</th></tr>'
             + "".join(links) + "</table>")
    pages.append(("index.html", page("SAP-Arena gallery", index)))
    out.mkdir(parents=True, exist_ok=False)
    for name, content in pages:
        with (out / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return {"status": "ok", "index": str(out / "index.html"), "games": len(games),
            "turn_rows": total_rows, "tiles": dict(TILES)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    ap.add_argument("--out", type=Path, default=ROOT / "gallery")
    ap.add_argument("--art", type=Path, default=DEFAULT_ART)
    ap.add_argument("--games", type=int, default=None, help="first N input games; default all, 0 disables")
    args = ap.parse_args()
    if args.games is not None and args.games < 0:
        ap.error("--games must be nonnegative")
    try:
        result = render_records(args.records, args.out, max_games=args.games, art=args.art)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FATAL: gallery rendering failed: {exc}", file=sys.stderr)
        return 2
    if result["status"] == "ok":
        print(f"gallery: {result['games']} games, {result['turn_rows']} turn rows -> {result['index']}")
        tiles = result["tiles"]
        print(f"tiles: art={tiles['art']} by_name={tiles['by_name']} empty={tiles['empty']}")
    else:
        print("gallery: disabled (0 games)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
