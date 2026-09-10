"""Replay snapshot file utilities for Step 3 training."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "v1"


def _normalize_turn_key(raw: Any) -> int | None:
    try:
        turn = int(raw)
    except (TypeError, ValueError):
        return None
    if turn < 1:
        return None
    return turn


def _normalize_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = copy.deepcopy(row)
        item.setdefault("ok", True)
        item.setdefault("error", None)
        item.setdefault("source", "snapshot")
        out.append(item)
    return out


def normalize_by_turn(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Extract canonical by-turn map from snapshot payload."""
    raw = payload.get("by_turn")
    if not isinstance(raw, dict):
        return {}

    out: dict[int, list[dict[str, Any]]] = {}
    for turn_key, rows in raw.items():
        turn = _normalize_turn_key(turn_key)
        if turn is None:
            continue
        normalized_rows = _normalize_rows(rows)
        if normalized_rows:
            out[turn] = normalized_rows
    return out


def load_snapshot(path: Path) -> dict[int, list[dict[str, Any]]]:
    # A gzipped snapshot is ~24x smaller. That is the difference between one the
    # demo can carry and one it cannot.
    p = Path(path)
    if p.suffix == ".gz":
        import gzip

        with gzip.open(p, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot_payload_not_object")
    version = str(payload.get("version", ""))
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported_snapshot_version:{version}")
    return normalize_by_turn(payload)


def save_snapshot(
    path: Path,
    by_turn: dict[int, list[dict[str, Any]]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "version": SNAPSHOT_VERSION,
        "created_at_unix": int(time.time()),
        "metadata": copy.deepcopy(metadata) if isinstance(metadata, dict) else {},
        "by_turn": {str(int(k)): _normalize_rows(v) for k, v in sorted(by_turn.items(), key=lambda kv: int(kv[0]))},
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

