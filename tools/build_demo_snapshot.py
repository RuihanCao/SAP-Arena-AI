#!/usr/bin/env python3
"""Build the demo's opponent snapshot from the de-identified pool that ships.

play-web loads a turn-indexed snapshot, which is a different shape from the chain
pool the evaluation harness reads, even though both describe the same games. This
converts one into the other so the demo and the benchmark face the same
opponents, rather than the demo needing a second private artifact.

`team` is left empty on purpose, exactly as `ChainSnapshotSource` leaves it: the
end-of-turn resolver uses the pre-parsed calculator state when it is present and
never looks at that field.

usage: build_demo_snapshot.py <pool.json.gz> <out.json.gz>
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import os

# Works from a source checkout and from an installed package alike.
if os.path.isdir("python"):
    sys.path.insert(0, "python")
from sap_ppo.train.snapshots import save_snapshot  # noqa: E402

POOL, OUT = sys.argv[1], sys.argv[2]

with gzip.open(POOL, "rt", encoding="utf-8") as fh:
    doc = json.load(fh)

by_turn: dict[int, list[dict]] = {}
for game in doc["games"]:
    pid = game.get("participation_id")
    pack = game.get("opponent_pack") or "Turtle"
    for entry in game.get("turns") or []:
        turn = entry.get("turn")
        if turn is None or not isinstance(entry.get("parsed_state"), dict):
            continue
        by_turn.setdefault(int(turn), []).append({
            "ok": True,
            "error": None,
            "battle": entry.get("battle"),
            "parsed_state": entry.get("parsed_state"),
            "participation_id": pid,
            "replay_id": None,
            "created_at": None,
            "side": "",
            "source": "deidentified_pool",
            "build_model": None,
            "side_pack": pack,
            "side_rank": game.get("opponent_rank"),
            "team": [],
        })

n_rows = sum(len(v) for v in by_turn.values())
# save_snapshot writes plain JSON; gzip it afterwards when asked, since the
# loader reads either and the plain form is ~24x larger.
tmp_out = Path(OUT[:-3]) if OUT.endswith(".gz") else Path(OUT)
save_snapshot(
    tmp_out,
    by_turn,
    metadata={
        "source": "arena_val_pool_deidentified",
        "turn_min": min(by_turn) if by_turn else None,
        "turn_max": max(by_turn) if by_turn else None,
        "rows_collected": n_rows,
        "include_parsed_state": True,
        "note": ("Converted from the de-identified opponent pool that ships with "
                 "this repository, so the demo and the benchmark face the same "
                 "opponents."),
    },
)
if OUT.endswith(".gz"):
    import gzip
    import shutil

    with open(tmp_out, "rb") as a, gzip.open(OUT, "wb", compresslevel=9) as b:
        shutil.copyfileobj(a, b)
    tmp_out.unlink()
print(f"turns {min(by_turn)}..{max(by_turn)}, {n_rows} rows -> {OUT}")
