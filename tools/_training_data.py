"""Strict local JSONL input shared by the prepared-data training commands.

Every row has game_id (a nonempty local grouping key of at most 64 characters)
and state (an engine state matching schemas/state_v1.json). Train and validation
files must have disjoint game_ids. No bundled opponent or evaluation pool is
loaded. Plain and gzip JSONL are accepted; training data is never downloaded.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from sap_ppo.schema import validate_state_schema


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_fresh(path: Path) -> None:
    if os.path.lexists(path):
        raise ValueError(f"output already exists; choose a fresh directory: {path}")


def positive(value, name: str) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _bad_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        _bad_constant(value)
    return number


def read_rows(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if not text.strip():
                continue
            row = json.loads(text, parse_constant=_bad_constant, parse_float=_finite_float)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line}: expected an object")
            gid = row.get("game_id")
            if not isinstance(gid, str) or not gid.strip() or len(gid) > 64:
                raise ValueError(f"{path}:{line}: game_id must contain 1..64 characters")
            if not isinstance(row.get("state"), dict):
                raise ValueError(f"{path}:{line}: missing engine state")
            validate_state_schema(row["state"])
            rows.append(row)
    if not rows:
        raise ValueError(f"empty prepared dataset: {path}")
    return rows


def read_train_val(train: Path, val: Path) -> tuple[list[dict], list[dict]]:
    left, right = read_rows(train), read_rows(val)
    overlap = {r["game_id"] for r in left} & {r["game_id"] for r in right}
    if overlap:
        raise ValueError(f"train/val game_id overlap: {len(overlap)} game(s)")
    return left, right


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
