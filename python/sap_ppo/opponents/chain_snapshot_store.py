"""The chain snapshot as a file the OS can share, instead of objects it cannot.

WHY THIS EXISTS. Every label shard is a separate process that parses the same
250 MiB snapshot into its own Python objects. Measured on this box: a bare
interpreter is 0.010 GiB, adding torch and the package brings it to 0.482, and
parsing the snapshot brings it to 1.705 -- so **1.22 GiB of each shard's 1.95
GiB is a private copy of one identical table**. Sixteen shards therefore spend
about 19.6 GiB storing one table sixteen times, and that, not the core count,
is what caps concurrency: 16 shards sit at ~31 GiB against a 48 GiB ceiling and
keep only ~16 of 32 cores busy.

WHY NOT FORK. Copy-on-write only holds while nobody writes the pages, and
CPython writes to an object's header every time anything merely *reads* it, to
maintain its reference count. `gc.freeze()` removes one source of writes (the
collector's own passes) and does nothing about refcounts, so a forked worker
un-shares the table page by page as it walks it. The sharing an OS gives to
FILE-backed pages has no such leak, because nothing is writing them.

So the table stops being Python objects. It becomes:

  * one blob file, holding each usable (pid, turn) entry's `battle` and
    `parsed_state` as one compact JSON object, concatenated;
  * one index, holding for each game -- IN THE SNAPSHOT'S OWN ORDER -- its
    identifiers and, for each of its usable turns IN ORDER, the byte range of
    that entry in the blob.

At run time a worker keeps the index (a few MB) and mmaps the blob. All
workers on the box then share one copy of the blob through the page cache,
automatically, with no fork and no freeze.

WHAT IS NOT STORED. `_entry_to_payload` reads exactly `battle`, `parsed_state`,
`_replay_id`, `_opponent_pack` and `_opponent_rank`. The last three are
game-level and live in the index. Nothing else in an entry is ever read, so
nothing else is carried."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
from typing import Any, Iterator

STORE_SCHEMA_VERSION = "chain_snapshot_store_v1"

INDEX_NAME = "index.json"
BLOB_NAME = "entries.jsonl"


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _usable_turns(game: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    """The entries the current loader would index, in the game's own order.

    Mirrors `ChainSnapshotSource.__init__` exactly: a turn that does not parse
    as an int is skipped, and an entry is only usable when BOTH `battle` and
    `parsed_state` are dicts -- a half-formed entry would otherwise be handed
    back as `ok=True` with a `None` payload and fail the turn downstream
    instead of falling back to another game's board.
    """
    turns = game.get("turns")
    if not isinstance(turns, list):
        return
    for entry in turns:
        if not isinstance(entry, dict):
            continue
        try:
            turn_i = int(entry.get("turn"))
        except (TypeError, ValueError):
            continue
        if not isinstance(entry.get("battle"), dict):
            continue
        if not isinstance(entry.get("parsed_state"), dict):
            continue
        yield turn_i, entry


def build_store(snapshot_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Convert a chain snapshot JSON into a blob plus an index. Idempotent."""
    snapshot_path = Path(snapshot_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if snapshot_path.suffix == ".gz":
        import gzip

        with gzip.open(snapshot_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError(f"chain_snapshot_missing_games:{snapshot_path}")

    blob_path = out_dir / BLOB_NAME
    index_games: list[dict[str, Any]] = []
    offset = 0
    n_entries = 0

    with open(blob_path, "wb") as blob:
        for game in games:
            if not isinstance(game, dict):
                # Kept as a placeholder so the index's game order is the
                # snapshot's game order including the entries the loader
                # skips; the reader applies the same `isinstance` skip.
                index_games.append({"malformed": True})
                continue
            rows: list[list[int]] = []
            for turn_i, entry in _usable_turns(game):
                raw = json.dumps(
                    {"battle": entry["battle"], "parsed_state": entry["parsed_state"]},
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                blob.write(raw)
                rows.append([turn_i, offset, len(raw)])
                offset += len(raw)
                n_entries += 1
            index_games.append(
                {
                    "participation_id": game.get("participation_id"),
                    "replay_id": game.get("replay_id"),
                    "opponent_pack": game.get("opponent_pack"),
                    "opponent_rank": game.get("opponent_rank"),
                    "n_raw_turns": len(game.get("turns") or [])
                    if isinstance(game.get("turns"), list)
                    else 0,
                    "turns": rows,
                }
            )

    index = {
        "schema_version": STORE_SCHEMA_VERSION,
        "source_path": str(snapshot_path),
        "source_sha256": sha256_file(snapshot_path),
        "blob_name": BLOB_NAME,
        "blob_bytes": offset,
        "n_games": len(index_games),
        "n_entries": n_entries,
        # Everything the loader reads off the top level, carried verbatim so a
        # reader never needs the JSON again.
        "version": payload.get("version"),
        "metadata": payload.get("metadata"),
        "splits": payload.get("splits"),
        "splits_meta": payload.get("splits_meta"),
        "games": index_games,
    }
    (out_dir / INDEX_NAME).write_text(
        json.dumps(index, separators=(",", ":"), ensure_ascii=False), encoding="utf-8"
    )
    return {
        "out_dir": str(out_dir),
        "n_games": len(index_games),
        "n_entries": n_entries,
        "blob_bytes": offset,
        "source_sha256": index["source_sha256"],
    }


class ChainSnapshotStore:
    """Read side: the index in memory, the entries left in the file."""

    def __init__(self, store_dir: str | Path, *, expect_source_sha256: str | None = None):
        self.store_dir = Path(store_dir)
        index_path = self.store_dir / INDEX_NAME
        if not index_path.is_file():
            raise FileNotFoundError(f"chain_snapshot_store_missing_index:{index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        got = str(self.index.get("schema_version") or "")
        if got != STORE_SCHEMA_VERSION:
            raise ValueError(
                f"chain_snapshot_store_schema_mismatch:{index_path}:{got}!={STORE_SCHEMA_VERSION}"
            )
        if expect_source_sha256 is not None:
            have = str(self.index.get("source_sha256") or "")
            if have != expect_source_sha256:
                raise ValueError(
                    "chain_snapshot_store_source_mismatch: this store was built from "
                    f"{have or '<unrecorded>'} but the caller named a snapshot whose "
                    f"digest is {expect_source_sha256}. Rebuild the store, or point at "
                    "the snapshot it belongs to; a store paired with the wrong snapshot "
                    "would serve a different opponent for every draw while every other "
                    "check passed."
                )
        blob_path = self.store_dir / str(self.index.get("blob_name") or BLOB_NAME)
        if not blob_path.is_file():
            raise FileNotFoundError(f"chain_snapshot_store_missing_blob:{blob_path}")
        self.blob_path = blob_path
        self._fh = open(blob_path, "rb")
        size = blob_path.stat().st_size
        declared = int(self.index.get("blob_bytes") or 0)
        if size != declared:
            raise ValueError(
                f"chain_snapshot_store_truncated_blob:{blob_path}:{size}!={declared}"
            )
        # A zero-length file cannot be mapped; a store with no entries is a
        # real (if useless) state and must not raise here.
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if size else None

    @property
    def games(self) -> list[dict[str, Any]]:
        return self.index["games"]

    def read_entry(self, offset: int, length: int) -> dict[str, Any]:
        if self._mm is None:
            raise ValueError("chain_snapshot_store_empty_blob")
        return json.loads(self._mm[offset : offset + length].decode("utf-8"))

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._fh.close()

    def __enter__(self) -> "ChainSnapshotStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
