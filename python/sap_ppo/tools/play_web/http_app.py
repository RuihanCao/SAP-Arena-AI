"""HTTP route table for the SAP-Arena web simulator."""

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any

from ...api import skip_imagined_validation as _skip_imagined_validation
from ..replay_player import REPLAY_HTML
from .app import SURFACE_AGENT, App, history_token
from .assets import static_path
from .build_identity import build_identity
from .replay_view import summarize_public_replay
from .value_probe import meaning_for


def _json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise for a browser, which is the only reader.

    `indent=2` used to be on this line and it more than doubled every answer
    the server gives -- 24.8 MB of a completed duel replay became 59.0 MB of
    which 34.1 MB was whitespace. Nothing reads these bytes as text: the page
    hands them straight to `JSON.parse`, and a human debugging a route pipes
    curl through `jq`.
    """
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


# Answers at or above this size are gzipped when the client offers it. Below
# it the header and the CPU cost more than the bytes saved.
GZIP_MIN_BYTES = 1024

# Level 6 is zlib's default. Level 9 buys under 2% on this JSON and costs
# several times the CPU; level 1 gives most of the win but leaves ~20% on the
# wire, which matters here because the slow client is an ssh tunnel to another
# continent, not a fast LAN.
GZIP_LEVEL = 6

_COMPRESSIBLE_PREFIXES = (
    "text/",
    "application/json",
    "application/javascript",
    "image/svg+xml",
)


def _compressible(content_type: str) -> bool:
    """PNG, OTF and the other already-compressed types are left alone --
    gzipping them burns CPU to make the body slightly bigger."""
    head = content_type.split(";", 1)[0].strip().lower()
    return head.startswith(_COMPRESSIBLE_PREFIXES)


def _gzip_accepted(header: str | None) -> bool:
    """Whether the client asked for gzip, honouring `q=0`, which means it did
    not. A client that sends no `Accept-Encoding` at all gets bytes as-is."""
    for part in (header or "").split(","):
        token, _, params = part.strip().partition(";")
        if token.strip().lower() != "gzip":
            continue
        for param in params.split(";"):
            key, _, value = param.strip().partition("=")
            if key.strip().lower() == "q":
                try:
                    return float(value) > 0
                except ValueError:
                    return False
        return True
    return False


def _history_cursor(source: dict[str, Any]) -> tuple[int, str] | None:
    """What the client says it already holds: (entry count, its content hash).

    Read out of the query string on a GET and out of the body on a POST, and
    absent on the first request of a session, on a client that does not
    implement it, and on `curl`. All three then get the whole log, which is why
    this is an optimisation rather than a protocol both ends must agree on.
    """
    raw_from = source.get("history_from")
    raw_token = source.get("history_token")
    if isinstance(raw_from, list):  # urllib gives every query param as a list
        raw_from = raw_from[0] if raw_from else None
    if isinstance(raw_token, list):
        raw_token = raw_token[0] if raw_token else None
    try:
        count = int(raw_from)
    except (TypeError, ValueError):
        return None
    if count <= 0 or not isinstance(raw_token, str) or not raw_token:
        return None
    return count, raw_token


def _apply_history_cursor(snapshot: Any, cursor: tuple[int, str] | None) -> None:
    """Replace a snapshot's `history` with the part the client is missing.

    `history` is the only field in a snapshot that grows without bound -- one
    entry per engine action for a whole game -- so a 25-turn game ends up
    re-sending its own log on every shop click. The client keeps the log and is
    told where the piece it just got belongs:

        history_len    how many entries the server has, so the client can
                       check its copy is complete rather than assume it
        history_base   the index this `history` array starts at; 0 means
                       "replace what you have", n means "keep your first n"
        history_token  the content hash of ALL of them, which the client
                       hands back as its cursor next time

    A cursor is honoured only when the server's own first `count` entries hash
    to what the client sent. Anything else -- an undo, a reset, a new game, a
    second tab, a restarted server, a client one release behind -- fails that
    comparison and gets the full log, so the fallback is always the correct
    answer rather than a stale one.
    """
    if not isinstance(snapshot, dict):
        return
    entries = snapshot.get("history")
    if not isinstance(entries, list):
        return
    snapshot["history_len"] = len(entries)
    snapshot["history_token"] = history_token(entries)
    base = 0
    if cursor is not None:
        count, token = cursor
        if count <= len(entries) and history_token(entries, count) == token:
            base = count
    if base:
        snapshot["history"] = entries[base:]
    snapshot["history_base"] = base


def _duel_unavailable() -> dict[str, Any]:
    """The answer every `/api/duel/*` route gives on a server with no duel.

    Shaped like a real answer (`ok`, `error`, `state`) so the page shows the
    reason in its banner rather than throwing on a missing field.
    """
    return {"ok": False, "error": "duel_unavailable", "state": None}


class Handler(BaseHTTPRequestHandler):
    app: App
    # exp16 W5: the duel behind `/play`. `None` on a server started without
    # one, in which case every `/api/duel/*` route answers `duel_unavailable`
    # instead of raising -- `/sandbox` is unaffected either way, which is what
    # keeps exp14's gate runnable on a box with no checkpoints.
    duel_app: Any = None

    # HTTP/1.1, so one TCP connection carries the whole page instead of one per
    # request. The stdlib default is HTTP/1.0, which closes after every reply,
    # and this page is a poller: over Ruihan's tunnel (measured 244 ms RTT and
    # 20% packet loss in internal design notes)
    # a fresh connection per poll means a handshake per poll, and a dropped SYN
    # costs a second before the request has even been sent. The 2026-08-13
    # session saw 82 sockets in TIME_WAIT against `:8766` while one human
    # played. Safe to switch on here because `_send` writes `content-length`
    # for every body, and the only bodyless reply is the 304 in
    # `_send_revalidated`, so no response's end is ambiguous.
    protocol_version = "HTTP/1.1"
    # A kept-alive connection otherwise parks its thread for the life of the
    # tab. `StreamRequestHandler.setup` turns this into a socket timeout, and
    # `handle_one_request` treats the timeout as "close", so an idle browser
    # costs nothing and a live one just reconnects.
    timeout = 65

    # -- what the access log is allowed to claim ---------------------------
    def parse_request(self) -> bool:
        """Start the clock where the REQUEST starts, not where the connection
        became free.

        This was first stamped in `handle_one_request`, which is wrong the
        moment keep-alive is on: that method begins by BLOCKING on the read of
        the next request line, so on the second and later requests of a
        connection the stamp landed while the browser was still thinking, and
        the log then charged the server for the human's idle time. It read
        plausibly and it was nonsense -- a first measurement of a real session
        reported `p50 686 ms` for `status` and a `61 s` maximum on a server
        whose handlers answer in single-digit milliseconds. `parse_request`
        returns once the request line and headers are in, which is the first
        moment this process has any work to do.
        """
        parsed = super().parse_request()
        self._request_started = time.perf_counter()
        self._body_bytes: int | None = None
        self._body_encoded = False
        return parsed

    def log_error(self, format: str, *args: Any) -> None:
        """Drop the one "error" that is this class's `timeout` working.

        `BaseHTTPRequestHandler.handle_one_request` logs `Request timed out`
        when the read of the NEXT request line on a kept-alive connection hits
        `timeout` -- that is, when an idle browser has stopped asking. It is the
        intended way for an idle connection to end, and at six connections per
        tab it would otherwise write a line every 65 s that reads like a fault.
        Only that message is dropped, and only because it is raised before a
        request line exists; everything else still goes to the log.
        """
        if format.startswith("Request timed out"):
            return
        super().log_error(format, *args)

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        """The stdlib line, plus how long the server took and what it wrote.

        The prefix and the quoted request text are byte-identical to the
        default, so anything already reading this log keeps matching; the new
        fields are appended.

        WHAT THE MILLISECONDS MEAN: service time up to the first byte, because
        `send_response` (and so this) runs after the body is built and before it
        is written. That is the honest boundary -- everything after it is the
        network's, not ours. It is also why this instrument cannot see the
        failures it was added for: a request the tunnel drops never reaches this
        method at all, which is exactly how a lost click leaves no trace here
        and why the page had to learn to speak for itself (`app.js`
        `transportFailure`).
        """
        if isinstance(code, HTTPStatus):
            code = code.value
        started = getattr(self, "_request_started", None)
        took = "" if started is None else " %.1fms" % ((time.perf_counter() - started) * 1000.0)
        body_bytes = getattr(self, "_body_bytes", None)
        wrote = "" if body_bytes is None else " %dB%s" % (
            body_bytes,
            " gz" if getattr(self, "_body_encoded", False) else "",
        )
        self.log_message('"%s" %s %s%s%s', self.requestline, str(code), str(size), took, wrote)

    def _agent_surface(self) -> bool:
        return getattr(self.app, "surface", None) == SURFACE_AGENT

    def _page_for(self, path: str) -> str | None:
        """Which HTML page a top-level route serves, or None for 404.

        The agent surface is the shop page and the engine API, full stop. The
        menu, the duel and the duel's replay list are the HUMAN entry points,
        so they are not merely disabled there -- they are not routes. The shop
        page answers at `/` (which is what :8765 has always been) and at
        `/sandbox` (which is what exp14's gate drives).
        """
        if self._agent_surface():
            return {"/": "index.html", "/sandbox": "index.html"}.get(path)
        return {
            "/": "landing.html",
            "/play": "play.html",
            "/sandbox": "index.html",
            "/replays": "replays.html",
        }.get(path)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        cache_control: str = "no-store",
        etag: str | None = None,
    ) -> None:
        compressible = _compressible(content_type)
        encoded = False
        if (
            compressible
            and len(body) >= GZIP_MIN_BYTES
            and _gzip_accepted(self.headers.get("accept-encoding"))
        ):
            packed = gzip.compress(body, GZIP_LEVEL)
            if len(packed) < len(body):
                body, encoded = packed, True
        # Read back by `log_request`, which `send_response` calls on the next
        # line. On the wire is what counts, so this is the post-gzip length.
        self._body_bytes = len(body)
        self._body_encoded = encoded
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", cache_control)
        if compressible:
            # Without this a shared cache could hand a gzipped body to a client
            # that never asked for one.
            self.send_header("vary", "accept-encoding")
        if encoded:
            self.send_header("content-encoding", "gzip")
        if etag is not None:
            self.send_header("etag", etag)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_revalidated(
        self, body: bytes, content_type: str, cache_control: str = "no-cache"
    ) -> None:
        """Send a body the client may keep, but must re-check before reusing.

        The page's own CSS and JS were `no-store`, so every navigation between
        `/`, `/play` and `/replays` re-downloaded 245 KB of unchanged assets --
        two seconds each time over the tunnel. They cannot simply be given a
        long max-age either, because this is the surface we edit and reload all
        day; serving a stale `replays.js` would be worse than serving it twice.
        `no-cache` is the honest middle: the browser still asks every time, and
        an unchanged file comes back as a 304 with no body.
        """
        etag = '"%s"' % hashlib.sha256(body).hexdigest()[:32]
        if self.headers.get("if-none-match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("etag", etag)
            self.send_header("cache-control", cache_control)
            self.end_headers()
            return
        self._send(HTTPStatus.OK, body, content_type, cache_control=cache_control, etag=etag)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _send_state_json(
        self, status: int, payload: dict[str, Any], cursor: tuple[int, str] | None
    ) -> None:
        """Send a response that carries a snapshot, minus the log the client has.

        Every route that answers with a snapshot goes through here, whether the
        snapshot IS the payload (`/api/state`) or sits under `state` (every
        mutation). Keeping the trim at this one boundary is what lets
        `App._snapshot` and `DuelApp.snapshot` go on meaning "the whole
        session, as it is" -- there is exactly one place where a client's claim
        about what it already has can be got wrong.
        """
        _apply_history_cursor(payload, cursor)
        _apply_history_cursor(payload.get("state"), cursor)
        self._send_json(status, payload)

    def _replay_call(self, fn) -> dict[str, Any]:
        """Run a replay-session call, folding expected errors into the JSON
        contract (the player surfaces them as messages, not stack traces)."""
        try:
            return fn()
        except (ValueError, FileNotFoundError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # unexpected: still keep the server alive
            return {"ok": False, "error": f"replay_internal:{type(exc).__name__}:{exc}"}

    def _duel(self) -> Any:
        return getattr(type(self), "duel_app", None)

    def _value_readout(self, duel: Any, game_id: str) -> dict[str, Any]:
        """What `V` thought of both boards at each end of turn, for a replay.

        The rows already exist: show-value writes one per resolved turn beside
        the archive, with both sides valued the same way and the realised
        return-to-go filled in at game end. This route only reads them back, so
        the numbers a replay shows are the ones recorded AS THE GAME WAS PLAYED,
        never a recomputation with today's weights.

        Nothing back-fills. Games archived before show-value shipped have no
        log and say so, which is the honest answer and is why `available` is a
        field rather than an empty list: an empty list reads as "V had nothing
        to say", and the truth is "V was not watching".

        The wording travels WITH the numbers, keyed the way `value_probe` keys
        it, because the replay page is bound by the same acceptance the live
        readout is: a number must not reach the screen without the sentence
        that says what it means. These rows are all end-of-turn boards, so the
        key is `end_of_turn` and comes from `meaning_for`, never typed here.
        """
        meaning_key, meaning = meaning_for(False)
        readout: dict[str, Any] = {"meaning_key": meaning_key, "meaning": meaning}
        try:
            readout.update(duel.archive_value_rows(game_id))
        except Exception as exc:  # a missing readout must never break a replay
            readout.update(
                {
                    "available": False,
                    "rows": [],
                    "reason": f"value_log_failed:{type(exc).__name__}",
                }
            )
        return readout

    def _parse_post(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        data = self.rfile.read(length) if length > 0 else b"{}"
        if not data:
            return {}
        try:
            return json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # exp16 W5 route table:
        #   /          the menu (Play / Replays)
        #   /play      the duel
        #   /sandbox   the free-practice shop page -- exp14's gate's target
        #   /replays   this project's own games (empty until they are archived)
        #   /replay    the archived HUMAN replay player, untouched
        if path in ("/", "/play", "/sandbox", "/replays"):
            name = self._page_for(path)
            if name is None:
                self._send_json(HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": f"route_not_on_agent_surface:{path}"})
                return
            page_path = static_path(name)
            if page_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"page_not_found:{name}"})
                return
            self._send(HTTPStatus.OK, page_path.read_bytes(), "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            name = path[len("/static/"):]
            asset_path = static_path(name)
            if asset_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "static_not_found"})
                return
            mime = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
            self._send_revalidated(asset_path.read_bytes(), mime)
            return

        if path == "/replay":
            self._send(HTTPStatus.OK, REPLAY_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/replay/games":
            q = (params.get("q", [""])[0] or "").strip()
            try:
                limit = int(params.get("limit", ["25"])[0])
            except ValueError:
                limit = 25
            self._send_json(HTTPStatus.OK, self._replay_call(
                lambda: self.app.replay_session.list_games(q, limit)))
            return

        if path == "/api/replay/turn":
            try:
                turn = int(params.get("t", [""])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_or_bad_t"})
                return
            self._send_json(HTTPStatus.OK, self._replay_call(
                lambda: self.app.replay_session.turn_payload(turn)))
            return

        if path == "/api/replay/full_image":
            try:
                body = self.app.replay_session.full_image()
            except (ValueError, FileNotFoundError) as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:  # keep the server alive on render bugs
                self._send_json(HTTPStatus.NOT_FOUND,
                                {"ok": False, "error": f"replay_internal:{type(exc).__name__}:{exc}"})
                return
            self._send(HTTPStatus.OK, body, "image/png", cache_control="no-store")
            return

        # The static half of every snapshot, served once instead of per click.
        # `immutable` is granted only to the content-addressed form, because
        # only that form can honour it: `?v=<current fingerprint>` names bytes
        # that cannot change under the URL, while the bare path means "whatever
        # this server has now" and must be revalidated. Both answer with the
        # same body, so a `curl /api/catalog` still works.
        if path == "/api/catalog":
            document = self.app.catalog_document
            asked = (params.get("v", [""])[0] or "").strip()
            addressed = asked == self.app.catalog_fingerprint
            self._send(
                HTTPStatus.OK,
                _json_bytes(document),
                "application/json; charset=utf-8",
                cache_control=(
                    "public, max-age=86400, immutable" if addressed else "no-cache"
                ),
            )
            return

        if path == "/api/state":
            self._send_state_json(HTTPStatus.OK, self.app._snapshot(), _history_cursor(params))
            return

        if path == "/api/build":
            # "What is this server actually serving?", answerable with one
            # curl and no access to whoever launched it. Deliberately cheap
            # and dependency-free so it stays answerable when the rest of the
            # page is broken.
            self._send_json(HTTPStatus.OK, {
                "ok": True,
                "surface": getattr(self.app, "surface", None),
                "compose_enabled": bool(getattr(self.app, "compose_enabled", True)),
                "duel_enabled": self._duel() is not None,
                "imagined_validation_skipped": bool(_skip_imagined_validation()),
                "build": build_identity(),
            })
            return

        if path == "/api/duel/state":
            duel = self._duel()
            if duel is None:
                self._send_json(HTTPStatus.OK, _duel_unavailable())
                return
            self._send_state_json(HTTPStatus.OK, duel.snapshot(), _history_cursor(params))
            return

        if path == "/api/duel/status":
            duel = self._duel()
            if duel is None:
                self._send_json(HTTPStatus.OK, _duel_unavailable())
                return
            self._send_json(HTTPStatus.OK, duel.status())
            return

        if path == "/api/duel/replays":
            duel = self._duel()
            if duel is None:
                self._send_json(HTTPStatus.OK, {"ok": True, "games": []})
                return
            query = (params.get("q", [""])[0] or "").strip()
            try:
                limit = int(params.get("limit", ["100"])[0])
                games = duel.archive_list(query, limit)
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": f"archive_list_failed:{type(exc).__name__}:{exc}"},
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "games": games})
            return

        if path == "/api/duel/replay":
            duel = self._duel()
            game_id = (params.get("id", [""])[0] or "").strip()
            if duel is None or not game_id:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "archive_game_not_found"})
                return
            # `detail=full` is the old whole-game answer, kept for curl and for
            # anything auditing the archive through this route. The page asks
            # for the default, and fetches the detail one turn at a time.
            full = (params.get("detail", ["summary"])[0] or "").strip() == "full"
            try:
                game = duel.archive_public_game(game_id)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
                return
            if not full:
                game = summarize_public_replay(game)
            game["value_readout"] = self._value_readout(duel, game_id)
            self._send_json(HTTPStatus.OK, {"ok": True, "game": game})
            return

        if path == "/api/duel/replay_turn":
            duel = self._duel()
            game_id = (params.get("id", [""])[0] or "").strip()
            try:
                turn_number = int((params.get("turn", [""])[0] or "").strip())
            except ValueError:
                turn_number = -1
            if duel is None or not game_id or turn_number < 1:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"ok": False, "error": "archive_turn_not_found"}
                )
                return
            try:
                game = duel.archive_public_game(game_id)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
                return
            turn = next(
                (
                    row
                    for row in (game.get("turns") or [])
                    if int(row.get("turn", -1)) == turn_number
                ),
                None,
            )
            if turn is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"ok": False, "error": "archive_turn_not_found"}
                )
                return
            self._send_json(HTTPStatus.OK, {"ok": True, "turn": turn})
            return

        if path == "/api/duel/replay_image":
            duel = self._duel()
            game_id = (params.get("id", [""])[0] or "").strip()
            try:
                turn = int((params.get("turn", [""])[0] or "").strip())
            except ValueError:
                turn = -1
            if duel is None or not game_id or turn < 1:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "archive_render_not_found"})
                return
            try:
                body = duel.archive_image(game_id, turn)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
                return
            self._send(HTTPStatus.OK, body, "image/png", cache_control="public, max-age=31536000, immutable")
            return

        if path == "/api/duel/render":
            duel = self._duel()
            if duel is None:
                self._send_json(HTTPStatus.NOT_FOUND, _duel_unavailable())
                return
            raw_turn = (params.get("turn", [""])[0] or "").strip()
            raw_token = (params.get("v", [""])[0] or "").strip()
            try:
                turn = int(raw_turn) if raw_turn else None
                token = int(raw_token) if raw_token else None
            except ValueError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST, {"ok": False, "error": "bad_render_query"}
                )
                return
            try:
                body = duel.render_image_bytes(turn, token=token)
            except FileNotFoundError as exc:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)}
                )
                return
            self._send(HTTPStatus.OK, body, "image/png", cache_control="no-store")
            return

        if path == "/api/replay_image":
            mode = (params.get("mode", [""])[0] or "").strip()
            if not mode:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_mode"})
                return
            try:
                body = self.app.replay_image_bytes(mode)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._send(HTTPStatus.OK, body, "image/png", cache_control="no-store")
            return

        if path == "/api/image":
            slot_type = (params.get("slot_type", [""])[0] or "").strip()
            item_id = (params.get("item_id", [""])[0] or "").strip()
            if not slot_type or not item_id:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_slot_type_or_item_id"})
                return
            img_path = self.app.image_path(slot_type=slot_type, item_id=item_id)
            if img_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "image_not_found"})
                return
            mime = mimetypes.guess_type(str(img_path))[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, img_path.read_bytes(), mime, cache_control="public, max-age=86400, immutable")
            return

        if path == "/api/icon":
            name = (params.get("name", [""])[0] or "").strip()
            if not name:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "missing_icon_name"})
                return
            icon_path = self.app.icon_path(name)
            if icon_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "icon_not_found"})
                return
            mime = mimetypes.guess_type(str(icon_path))[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, icon_path.read_bytes(), mime, cache_control="public, max-age=86400, immutable")
            return

        if path == "/api/font":
            name = (params.get("name", [""])[0] or "").strip()
            font_path = self.app.font_path(name)
            if font_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "font_not_found"})
                return
            self._send(HTTPStatus.OK, font_path.read_bytes(), "font/otf",
                       cache_control="public, max-age=86400, immutable")
            return

        if path == "/api/background":
            name = (params.get("name", [""])[0] or "").strip()
            bg_path = self.app.background_path(name)
            if bg_path is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "background_not_found"})
                return
            mime = mimetypes.guess_type(str(bg_path))[0] or "image/png"
            self._send(HTTPStatus.OK, bg_path.read_bytes(), mime,
                       cache_control="public, max-age=86400, immutable")
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown_path:{path}"})

    def do_POST(self) -> None:  # noqa: N802
        payload = self._parse_post()
        cursor = _history_cursor(payload)

        if self.path == "/api/reset":
            self._send_state_json(HTTPStatus.OK, {"ok": True, "state": self.app.reset()}, cursor)
            return

        if self.path == "/api/undo":
            ok, err, snapshot = self.app.undo()
            self._send_state_json(
                HTTPStatus.OK, {"ok": ok, "error": err, "state": snapshot}, cursor)
            return

        if self.path == "/api/debug/add_chocolate":
            ok, err, snapshot = self.app.add_debug_chocolate()
            self._send_state_json(
                HTTPStatus.OK, {"ok": ok, "error": err, "state": snapshot}, cursor)
            return

        if self.path == "/api/debug/add_shop_item":
            slot_type = str(payload.get("slot_type", "")).strip()
            item_id = str(payload.get("item_id", "")).strip()
            # No cost in the payload means "whatever the game charges for it";
            # only an explicit cost overrides that.
            cost: int | None
            if payload.get("cost") is None:
                cost = None
            else:
                try:
                    cost = int(payload["cost"])
                except (TypeError, ValueError):
                    cost = None
            ok, err, snapshot = self.app.add_debug_shop_item(slot_type=slot_type, item_id=item_id, cost=cost)
            self._send_state_json(
                HTTPStatus.OK, {"ok": ok, "error": err, "state": snapshot}, cursor)
            return

        if self.path == "/api/replay/load":
            pid = str(payload.get("pid", "")).strip()
            self._send_json(HTTPStatus.OK, self._replay_call(
                lambda: self.app.replay_session.load(pid)))
            return

        if self.path == "/api/apply":
            # `App.apply` takes both shapes: `{"action": ...}` for one engine
            # action and `{"compose": ...}` for a server-side group (exp16 W4).
            out = self.app.apply(payload)
            self._send_state_json(HTTPStatus.OK, out, cursor)
            return

        if self.path == "/api/infer/recommend":
            out = self.app.recommend(payload)
            self._send_json(HTTPStatus.OK, out)
            return

        # exp16 show-value (`DESIGN_show_value.md` section 4). Deliberately
        # narrow, and deliberately answered by the DUEL: the V heads are loaded
        # once, beside the agent that plays, and `app.py::App` has no learned
        # value model at all (its `value_model_loaded` is the exp04 tempo
        # planner's). A server started with `--no-duel` therefore answers
        # `value_unavailable` here, which is the same shape every `/api/duel/*`
        # route answers on such a server rather than a stack trace.
        if self.path == "/api/infer/value":
            duel = self._duel()
            if duel is None:
                self._send_json(
                    HTTPStatus.OK,
                    {"ok": False, "error": "value_unavailable", "value": None},
                )
                return
            self._send_json(HTTPStatus.OK, duel.value(payload))
            return

        # ---------------------------------------------------------- the duel
        if self.path.startswith("/api/duel/"):
            duel = self._duel()
            if duel is None:
                self._send_json(HTTPStatus.OK, _duel_unavailable())
                return
            route = self.path[len("/api/duel/"):]
            if route == "new_game":
                seed = payload.get("seed")
                try:
                    seed_value = int(seed) if seed is not None else None
                except (TypeError, ValueError):
                    seed_value = None
                # Amendment 6. The settings card sends both knobs with the
                # New game. `Number("abc")` reaches us as null, which is
                # indistinguishable from "not sent" once it is a value, so a
                # field that WAS sent and reads as null is refused loudly for
                # the same reason `config` refuses it: saying ok to that
                # silently starts a game on a setting the human did not pick.
                unreadable = [
                    name
                    for name in ("deepen_width", "turn_budget_s", "search_width")
                    if name in payload and payload.get(name) is None
                ]
                if unreadable:
                    self._send_state_json(
                        HTTPStatus.OK,
                        {
                            "ok": False,
                            "error": f"duel_bad_settings:{','.join(unreadable)}",
                            "state": duel.snapshot(),
                        },
                        cursor,
                    )
                    return
                self._send_state_json(
                    HTTPStatus.OK,
                    duel.new_game(
                        seed=seed_value,
                        deepen_width=payload.get("deepen_width"),
                        turn_budget_s=payload.get("turn_budget_s"),
                        gear=payload.get("gear"),
                        search_width=payload.get("search_width"),
                    ),
                    cursor,
                )
                return
            if route == "apply":
                self._send_state_json(HTTPStatus.OK, duel.apply(payload), cursor)
                return
            if route == "undo":
                self._send_state_json(HTTPStatus.OK, duel.undo(), cursor)
                return
            if route == "end_turn":
                self._send_state_json(HTTPStatus.OK, duel.end_turn(), cursor)
                return
            if route == "config":
                gear = payload.get("gear")
                width = payload.get("width")
                turn_budget_s = payload.get("turn_budget_s")
                if "turn_budget_s" in payload and turn_budget_s is None:
                    # The field was sent and could not be read as a number
                    # (the page sends `Number("abc")` as null). Saying
                    # "ok" to that is a silent no-op on a control the human
                    # just used, so it is refused loudly instead.
                    self._send_state_json(
                        HTTPStatus.OK,
                        {
                            "ok": False,
                            "error": "duel_bad_turn_budget_s",
                            "state": duel.snapshot(),
                        },
                        cursor,
                    )
                    return
                self._send_state_json(
                    HTTPStatus.OK,
                    duel.set_config(
                        gear=str(gear) if gear is not None else None,
                        width=width,
                        turn_budget_s=turn_budget_s,
                    ),
                    cursor,
                )
                return
            if route == "reset":
                # The sandbox's Reset means "a fresh shop"; a duel's means "a
                # fresh GAME", and there is nothing in between.
                self._send_state_json(HTTPStatus.OK, duel.new_game(), cursor)
                return
            self._send_json(
                HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown_duel_route:{route}"}
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown_path:{self.path}"})
