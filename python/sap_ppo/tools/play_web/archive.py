"""Durable per-turn archives for exp16 human-vs-AI games."""

from __future__ import annotations

import copy
import io
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from .agent_identity import normalize_ai_version
from .replay_view import is_completed_game


DEFAULT_DUEL_ARCHIVE_ROOT = Path(os.environ.get(
    "SAP_PLAY_WEB_ARCHIVE",
    str(Path(__file__).resolve().parents[4] / "data" / "play_web_games"),
))
ARCHIVE_SCHEMA_VERSION = 2
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _fsync_dir(path: Path) -> None:
    """Persist directory entries created or replaced below ``path``."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    dir_fd = os.open(path, flags)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _mkdir_durable(path: Path) -> None:
    """Create a directory tree and fsync every newly exposed parent entry."""
    path = Path(path)
    if path.is_dir():
        return
    if path.parent == path:
        raise FileNotFoundError(path)
    _mkdir_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    _fsync_dir(path.parent)


def _is_safe_id(value: Any) -> bool:
    clean = str(value)
    return bool(_SAFE_ID.fullmatch(clean) and clean not in {".", ".."})


def _atomic_bytes(path: Path, data: bytes) -> None:
    _mkdir_durable(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )


def _validate_png(data: bytes) -> None:
    """Reject missing or corrupt archive images, including on read/recovery."""
    if not data or not bytes(data).startswith(_PNG_SIGNATURE):
        raise ValueError("archive_render_not_png")
    try:
        with Image.open(io.BytesIO(data)) as parsed:
            parsed.verify()
    except Exception as exc:
        raise ValueError("archive_render_not_png") from exc


class DuelArchive:
    """One archive root, safe for the HTTP thread and render workers."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.games_root = self.root / "games"
        self.index_path = self.root / "index.jsonl"
        self._lock = threading.RLock()
        _mkdir_durable(self.games_root)

    def _game_dir(self, game_id: str) -> Path:
        clean = str(game_id)
        if not _is_safe_id(clean):
            raise ValueError("bad_archive_game_id")
        path = self.games_root / clean
        if path.resolve(strict=False).parent != self.games_root.resolve():
            raise ValueError("bad_archive_game_id")
        return path

    def _game_path(self, game_id: str) -> Path:
        return self._game_dir(game_id) / "game.json"

    def _read_game(self, game_id: str) -> dict[str, Any]:
        path = self._game_path(game_id)
        if not path.is_file():
            raise FileNotFoundError(f"archive_game_not_found:{game_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"archive_game_not_object:{game_id}")
        return payload

    def _summary(self, game: dict[str, Any]) -> dict[str, Any]:
        ai = normalize_ai_version(game.get("ai_version"))
        return {
            "id": game["id"],
            "schema_version": int(game.get("schema_version") or 1),
            "created_at": game["created_at"],
            "updated_at": game["updated_at"],
            "turns": len(game.get("turns") or []),
            "done": bool(game.get("done")),
            "winner": game.get("winner"),
            "end_reason": game.get("end_reason"),
            "git_sha": game.get("git_sha"),
            "agent": ai.get("agent_name"),
            "agent_id": ai.get("agent_id"),
            "agent_name": ai.get("agent_name"),
            "model_revision": ai.get("model_revision"),
            "search_revision": ai.get("search_revision"),
            "baseline_id": ai.get("baseline_id"),
            # Amendment 6. The setting is per GAME now, so the list has to be
            # able to tell two games apart without opening both. Absent on
            # everything archived before 2026-08-19, and absent is rendered as
            # "not recorded" rather than as "off".
            "completion_policy": ai.get("completion_policy"),
            "completion_width": ai.get("completion_width"),
            # W11b, same rule: absent on anything archived before the gear was
            # a per-game setting, and the list renders absent as nothing at all
            # rather than as the process default.
            "gear": (game.get("gear_start") or {}).get("gear"),
            "gear_width": (game.get("gear_start") or {}).get("width"),
            "game_seed": (game.get("seeds") or {}).get("game"),
        }

    def _append_index(self, game: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self._summary(game), sort_keys=True) + "\n"
        try:
            with self.index_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            # File fsync does not make creation of index.jsonl durable.
            _fsync_dir(self.root)
        except OSError:
            # game.json is the authoritative atomic record and list_games()
            # rebuilds every summary from games/*/game.json. Losing the
            # already-durable game id merely because this secondary index
            # append failed would create an orphan and duplicate it on retry.
            return

    def begin_game(self, metadata: dict[str, Any]) -> str:
        with self._lock:
            seed = int((metadata.get("seeds") or {}).get("game") or 0)
            rules = metadata.get("rules") or {}
            start_lives = int(rules.get("start_lives") or 0)
            initial_final = copy.deepcopy(metadata.get("final"))
            if not isinstance(initial_final, dict):
                initial_final = {
                    side: {
                        "lives": start_lives,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                    }
                    for side in ("human", "ai")
                }
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            game_id = f"{stamp}-{seed}-{uuid.uuid4().hex[:8]}"
            now = _utc_now()
            game = {
                **copy.deepcopy(metadata),
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "id": game_id,
                "created_at": now,
                "updated_at": now,
                "turns": [],
                "final": initial_final,
                "done": False,
                "winner": None,
                "end_reason": None,
            }
            game_dir = self._game_dir(game_id)
            game_dir.mkdir(parents=True, exist_ok=False)
            # Make the unique games/<id> directory entry durable before
            # record_turn is allowed to report a durable game.json.
            _fsync_dir(self.games_root)
            _atomic_json(game_dir / "game.json", game)
            self._append_index(game)
            return game_id

    def record_turn(
        self,
        game_id: str,
        record: dict[str, Any],
        *,
        render_job: dict[str, Any] | None = None,
        final: dict[str, Any],
        done: bool,
        winner: str | None,
        end_reason: str | None,
    ) -> None:
        with self._lock:
            game = self._read_game(game_id)
            turn = int(record["turn"])
            turns = list(game.get("turns") or [])
            replacement = copy.deepcopy(record)
            if render_job is not None:
                replacement["render_job"] = copy.deepcopy(render_job)
            for i, existing in enumerate(turns):
                if int(existing.get("turn", -1)) == turn:
                    # Render metadata is attached asynchronously after the
                    # committed turn. A later idempotent gap-repair sync must
                    # not replace that richer record with the UI copy, which
                    # intentionally has no archive-only `render` block.
                    if "render" in existing and "render" not in replacement:
                        replacement["render"] = copy.deepcopy(existing["render"])
                    if "render_job" in existing and "render_job" not in replacement:
                        replacement["render_job"] = copy.deepcopy(existing["render_job"])
                    turns[i] = replacement
                    break
            else:
                turns.append(replacement)
            turns.sort(key=lambda row: int(row.get("turn", -1)))
            game.update(
                {
                    "updated_at": _utc_now(),
                    "turns": turns,
                    "final": copy.deepcopy(final),
                    "done": bool(done),
                    "winner": winner,
                    "end_reason": end_reason,
                }
            )
            _atomic_json(self._game_path(game_id), game)
            self._append_index(game)

    def record_game_state(
        self,
        game_id: str,
        *,
        final: dict[str, Any],
        done: bool,
        winner: str | None,
        end_reason: str | None,
    ) -> None:
        """Persist terminal metadata when no battle turn was committed."""
        with self._lock:
            game = self._read_game(game_id)
            game.update(
                {
                    "updated_at": _utc_now(),
                    "final": copy.deepcopy(final),
                    "done": bool(done),
                    "winner": winner,
                    "end_reason": end_reason,
                }
            )
            _atomic_json(self._game_path(game_id), game)
            self._append_index(game)

    def record_render(
        self,
        game_id: str,
        turn: int,
        image: bytes,
        *,
        calculator_link: str | None,
    ) -> None:
        if not image:
            raise ValueError("archive_render_empty")
        _validate_png(bytes(image))
        with self._lock:
            game = self._read_game(game_id)
            name = f"turn-{int(turn):03d}.png"
            _atomic_bytes(self._game_dir(game_id) / name, bytes(image))
            found = False
            for record in game.get("turns") or []:
                if int(record.get("turn", -1)) == int(turn):
                    record["render"] = {
                        "path": name,
                        "bytes": len(image),
                        "calculator_link": calculator_link,
                    }
                    found = True
                    break
            if not found:
                raise ValueError(f"archive_turn_not_found:{turn}")
            game["updated_at"] = _utc_now()
            _atomic_json(self._game_path(game_id), game)
            self._append_index(game)

    def adopt_render_if_present(
        self,
        game_id: str,
        turn: int,
        *,
        calculator_link: str | None,
    ) -> bool:
        """Attach a valid deterministic PNG left before metadata commit.

        record_render commits bytes before game.json metadata. A process death
        in that narrow window must reuse the already-durable W6a bytes rather
        than require replay-bot to be available after restart.
        """
        name = f"turn-{int(turn):03d}.png"
        path = self._game_dir(game_id) / name
        try:
            image = path.read_bytes()
            _validate_png(image)
        except (OSError, ValueError):
            return False
        with self._lock:
            game = self._read_game(game_id)
            for record in game.get("turns") or []:
                if int(record.get("turn", -1)) != int(turn):
                    continue
                existing = record.get("render") or {}
                if existing.get("path"):
                    return True
                record["render"] = {
                    "path": name,
                    "bytes": len(image),
                    "calculator_link": calculator_link,
                }
                game["updated_at"] = _utc_now()
                _atomic_json(self._game_path(game_id), game)
                self._append_index(game)
                return True
        return False

    def load_game(self, game_id: str) -> dict[str, Any]:
        # game.json is replaced atomically, so readers see either the old or
        # new complete document without serializing turn writers behind JSON
        # parsing/copying on the HTTP path.
        return copy.deepcopy(self._read_game(game_id))

    def image_bytes(self, game_id: str, turn: int) -> bytes:
        game = self._read_game(game_id)
        for record in game.get("turns") or []:
            if int(record.get("turn", -1)) == int(turn):
                rel = ((record.get("render") or {}).get("path") or "").strip()
                if not rel or Path(rel).name != rel:
                    break
                path = self._game_dir(game_id) / rel
                if path.is_file():
                    image = path.read_bytes()
                    _validate_png(image)
                    return image
                break
        raise FileNotFoundError(f"archive_render_not_found:{game_id}:{turn}")

    def has_render(self, game_id: str, turn: int) -> bool:
        try:
            self.image_bytes(game_id, turn)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def pending_render_jobs(self) -> list[tuple[str, int, dict[str, Any]]]:
        """Return durable jobs whose PNG is absent or corrupt.

        The job is committed in the same atomic game.json replacement as its
        turn. A fresh DuelApp can therefore reconstruct images acknowledged
        before an unclean process exit without putting rendering on the HTTP
        request path.
        """
        pending: list[tuple[str, int, dict[str, Any]]] = []
        for path in sorted(self.games_root.glob("*/game.json")):
            game_id = path.parent.name
            if not _is_safe_id(game_id):
                continue
            try:
                # Hold the archive lock only for one small atomic JSON read.
                # PNG validation can be comparatively expensive and must not
                # serialize an unrelated HTTP end_turn's record_turn write.
                with self._lock:
                    game = self._read_game(game_id)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if game.get("id") != game_id:
                continue
            for record in game.get("turns") or []:
                try:
                    turn = int(record["turn"])
                except (KeyError, TypeError, ValueError):
                    continue
                job = record.get("render_job")
                if not isinstance(job, dict):
                    continue
                render = record.get("render") or {}
                rel = str(render.get("path") or "").strip()
                valid = False
                if rel and Path(rel).name == rel:
                    try:
                        image = (self._game_dir(game_id) / rel).read_bytes()
                        _validate_png(image)
                    except (OSError, ValueError):
                        pass
                    else:
                        valid = True
                if not valid:
                    pending.append((game_id, turn, copy.deepcopy(job)))
        return pending

    def list_games(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        if self.index_path.is_file():
            for raw in self.index_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and _is_safe_id(row.get("id") or ""):
                    latest[str(row["id"])] = row
        # game.json is the authoritative atomic record. Refresh every summary
        # from disk and discover a game even if power died after its
        # rename/fsync but before the next index append. Atomic replacement
        # makes this scan safe without holding the turn-writer lock; a partial
        # trailing index line is simply ignored and recovered here.
        for path in self.games_root.glob("*/game.json"):
            game_id = path.parent.name
            if not _is_safe_id(game_id):
                continue
            try:
                game = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(game, dict) and game.get("id") == game_id:
                    latest[game_id] = self._summary(game)
            except (OSError, ValueError, KeyError):
                continue
        needle = str(query or "").strip().lower()
        rows = [
            row
            for row in latest.values()
            if is_completed_game(row)
            and (not needle or needle in json.dumps(row, sort_keys=True).lower())
        ]
        rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return copy.deepcopy(rows[: max(1, min(int(limit), 500))])
