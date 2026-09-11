"""WHAT THE BROWSER IS AND IS NOT TOLD. The AI's live board is NEVER in a
snapshot. The only AI boards that leave this module are the pre-battle boards
of turns that have already RESOLVED (`turns[i].ai_board`), which is the same
thing the real game shows you: you see what you just fought, never what your
opponent is holding while you shop.

WHY `end_turn` IS THE ONLY PLACE THE AI IS WAITED ON. The AI starts thinking at
the TOP of the human's turn, on the worker's single background thread, and the
human's shop actions never touch the AI's board -- so nothing here blocks until
the human presses End turn, at which point this layer waits for the AI's own
fixed-clock turn and then collects it. End Turn never changes search policy.

FAILURE IS LOUD. A failed AI turn is recorded with its code, surfaced as
`AI failed turn N: <code>`, and the AI stands still for that turn. Two failures
in a row end the game (`end_reason="ai_failed"`) rather than quietly playing a
weaker opponent, exactly as `duel_smoke.py` does."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ...api import legal_actions, validate_state
from ...action_text import annotate_chain
from ...catalog import load_turtle_catalog
from ...opponents import render_replay_image_from_calc_rows
from ...oracles.sap_calc_battle_oracle import generate_calculator_link
from .agent import AgentConfig, build_agent, build_search, demo_agent_config
from ..search_recommender import COMPLETION_BC_GREEDY, COMPLETION_V_SEARCH
from .public_settings import (
    ROOT_WIDTH, COMPLETION_WIDTH, INITIAL_SAMPLES, DEFAULT_TURN_SECONDS,
    SEARCH_MODES, engine_gear, public_mode, turn_seconds,
)
from .archive import DuelArchive
from .ai_worker import (
    GEARS,
    GEAR_RESAMPLE_CLOCK,
    MAX_SETTABLE_SEARCH_WIDTH,
    AiTurnConfig,
    AiTurnWorker,
)
from .app import (
    catalog_document,
    catalog_url,
    client_history_entries,
    history_precondition_error,
)
from .duel import (
    AI,
    DEFAULT_DUEL_RULES,
    DEFAULT_TURN_CAP,
    HUMAN,
    DuelSession,
    seeded_battle_fn,
)
from .replay_view import build_public_replay, build_side_replay
from .value_log import ValueLog, read_log as read_value_log
from .value_probe import ValueProbe, apply_actions, state_legal_actions


DEFAULT_VALUE_ACTION_CAP = 60


AI_TURN_TIMEOUT_S = 300.0
DEFAULT_SEED_SPACE = 1 << 31
ARCHIVE_RECOVERY_SCAN_INTERVAL_S = 5.0


LIVE_TURN_FIELDS = (
    "turn",
    "ok",
    "error",
    "ai_ok",
    "ai_error",
    "banner",
    "n_segments",
    "searched_segments",
    "deadline_segments",
    "greedy_segments",
    "segments",
    "elapsed_ms",
    "stop_latency_ms",
    "end_wait_ms",
    "end_to_result_ms",
    "gear",
    "finish_turn",
    "chunk_size",
    "adaptive_chunks",
    "turn_budget_s",
    "budget_utilization",
    "deadline_overshoot_ms",
    "effective_unique_prefixes",
    "dedup_ratio",
    "leaves_evaluated",
    "safety_cap_hit",
    "completion_decided",
    "completion_divergent",
    "realised_stochastic_samples",
    "extra_sample_levels",
    "completion_dropped",
    "completion_second_chance_nodes",
    "search_used",
    "search_error",
    "segments_capped",
    "ai_actions",
    "outcome",
    "ai_outcome",
    "lives",
    "wins",
    "life_bonus_applied",
    "ai_board",
    "human_board",
)


MAX_LIVE_SEGMENT_BYTES = 2048
MAX_LIVE_TURN_RECORD_BYTES = 16384
MAX_LIVE_FIELD_BYTES = 4096
MAX_LIVE_SCALAR_BYTES = 512

# The fields a segment reaches the browser with are its scalars -- but the two
# the page actually READS are these, and a guard that quietly dropped them
# would blank the widths column with every test still green (`duel.js`'s
# `renderLog` builds `d90 72` out of exactly this pair). Named here so the
# projection can be checked against the contract rather than against itself.
LIVE_SEGMENT_CONTRACT_FIELDS = ("mode", "width")

# Keys this process has already announced. The per-game set behind
# `duel.payload_guard` is what an operator reads; this only stops the log line
# repeating once per game for the life of a server.
_ANNOUNCED_PAYLOAD_DROPS: set[str] = set()


def _wire_bytes(value: Any) -> int:
    """What `value` costs on the wire, encoded the way responses are encoded
    (`http_app.py::_json_bytes` pretty-prints, so compact would be half)."""
    return len(json.dumps(value, indent=2, default=str).encode("utf-8"))


def _is_live_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _scalar_fits(value: Any) -> bool:
    """A scalar is bounded by construction unless it is a string.

    Checked without encoding anything, because this runs on every field of
    every turn record of every snapshot, and a number cannot be fat.
    """
    return not (isinstance(value, str) and len(value) > MAX_LIVE_SCALAR_BYTES)


def _live_segment_record(segment: dict[str, Any], dropped: set[str]) -> dict[str, Any]:
    """One segment as the browser sees it: its scalars, and nothing that nests.

    A third name list would fail the same way the second one did, so this is a
    shape rule instead: a segment reaches the page as its scalars, and a scalar
    that has grown into a blob (`MAX_LIVE_SCALAR_BYTES`) does not, because
    "scalar" and "small" are not the same claim.

    THE BLAST RADIUS IS WIDER THAN "TELEMETRY", and it is worth knowing which
    way. Measured over 160 real segment rows, the rule also removes
    `committed_stochastic_reasons` (7.3 KB), `committed_types` (4.8 KB) and
    `committed_stochastic_structural` (3.0 KB) -- segment-BOUNDARY evidence
    rather than per-chunk timing. Nothing live reads them today (the page reads
    `mode` and `width`; `duel_smoke.py` reads them in-process; the analyses read
    archives), and 15 KB is not why this rule exists, so they go with the rest
    rather than earning an exception that would put the name list back. If a
    live consumer ever needs them, that is a deliberate change to make, not an
    oversight to notice.

    The FULL segment, per-chunk timings and all, stays in `self.turn_records`
    and in the durable archive -- which is what `benchmark_inference_clock.py`
    and `check_archive.py` read, and neither goes through this function."""
    live: dict[str, Any] = {}
    for key, value in segment.items():
        if not _is_live_scalar(value):
            dropped.add(f"segments.{key}")
            continue
        if not _scalar_fits(value):
            dropped.add(f"segments.{key}")
            continue
        live[key] = value
    return live


def _bounded_live_value(key: str, value: Any, dropped: set[str]) -> tuple[bool, Any]:
    """Every other projected field: the same shape, bounded in size.

    Returns `(keep, projected)`. A field that cannot be made small enough is
    left OUT of the record rather than replaced by a placeholder: the page
    already treats every one of these as optional (`t.lives ? ... : '-'`)."""
    if _is_live_scalar(value):
        if _scalar_fits(value):
            return True, value
        dropped.add(key)
        return False, None
    if not isinstance(value, (list, dict)):
        dropped.add(key)
        return False, None
    if _wire_bytes(value) <= MAX_LIVE_FIELD_BYTES:
        return True, copy.deepcopy(value)
    # Too big whole: keep the parts that are small enough on their own...
    kept: Any
    if isinstance(value, list):
        kept = []
        for item in value:
            keep, projected = _bounded_live_value(key, item, dropped)
            if keep:
                kept.append(projected)
    else:
        kept = {}
        for sub_key, sub_value in value.items():
            keep, projected = _bounded_live_value(f"{key}.{sub_key}", sub_value, dropped)
            if keep:
                kept[sub_key] = projected
    # ...and then measure again, because shrinking the parts is not the same as
    # bounding the whole: ten thousand small dicts in a list is EXACTLY the
    # shape that started all this, and every one of them passes on its own.
    if _wire_bytes(kept) > MAX_LIVE_FIELD_BYTES:
        dropped.add(key)
        return False, None
    return True, kept


def derive_seed(game_seed: int, role: str) -> int:
    """A per-role seed from the game's seed.

    sha256 rather than `hash()` (salted per process) and rather than
    `game_seed + 1` (two games one apart would share a stream), so a game is
    reproducible from the single number the archive stores.
    """
    key = f"exp16_duel_seed:{int(game_seed)}:{role}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % DEFAULT_SEED_SPACE


class DuelApp:
    """One live human-vs-AI game plus the AI worker that plays against it."""

    def __init__(
        self,
        fixture_path: Path,
        *,
        agent_config: AgentConfig | None = None,
        build_search_fn: Any = None,
        ai_config: AiTurnConfig | None = None,
        rules: str = DEFAULT_DUEL_RULES,
        turn_cap: int = DEFAULT_TURN_CAP,
        build_agent_fn: Callable[..., Any] = build_agent,
        clock: Callable[[], float] = time.monotonic,
        render_fn: Callable[..., dict[str, Any]] = render_replay_image_from_calc_rows,
        calculator_link_fn: Callable[[dict[str, Any]], str] = generate_calculator_link,
        archive_root: Path | None = None,
        show_value: bool = True,
        value_action_cap: int = DEFAULT_VALUE_ACTION_CAP,
        build_value_probe_fn: Callable[[Any], Any] = ValueProbe.from_agent,
    ) -> None:
        self.fixture_path = Path(fixture_path)
        payload = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        self.initial_state = payload["initial_state"]
        self.agent_config = agent_config or demo_agent_config()
        # Injectable for the same reason `_build_agent_fn` is: the duel tests
        # are required to run without torch or a checkpoint.
        self._build_search_fn = build_search_fn or build_search
        self.ai_config = ai_config or AiTurnConfig(gear=GEAR_RESAMPLE_CLOCK)
        self.rules = str(rules)
        self.turn_cap = int(turn_cap)
        self._build_agent_fn = build_agent_fn
        self._clock = clock
        self._render_fn = render_fn
        self._calculator_link_fn = calculator_link_fn
        self.archive = DuelArchive(Path(archive_root)) if archive_root is not None else None


        self.show_value = bool(show_value)
        self.value_action_cap = int(value_action_cap)
        self._build_value_probe_fn = build_value_probe_fn
        self.value_probe: Any | None = None
        self.value_probe_error: str | None = None
        self.value_log: ValueLog | None = None


        self._value_lock = threading.Lock()

        self.catalog = load_turtle_catalog()
        # Same pinned pack as `App`'s, so the duel page and the sandbox page
        # ask `/api/catalog` for the same URL and share one browser cache entry.
        self.catalog_document = catalog_document(self.catalog)
        self.catalog_url = catalog_url(self.catalog_document)
        self._lock = threading.RLock()
        self.agent: Any | None = None
        self.worker: AiTurnWorker | None = None
        self.session: DuelSession | None = None
        self.game_seed: int | None = None
        self.seeds: dict[str, int] = {}
        self.turn_records: list[dict[str, Any]] = []
        self.agent_error: str | None = None
        self._gen = 0
        self._game_status_token = 0
        self._ai_failures_in_a_row = 0
        self._forced_end: str | None = None
        self._games_played = 0
        # What the live-payload bounds have refused, and the worst projected
        # record seen, for the game in progress. Surfaced as
        # `duel.payload_guard` so the ceiling is observable in production
        # rather than only inside a test.
        self._payload_dropped: set[str] = set()
        self._payload_over_ceiling = 0
        self._payload_worst_record = 0
        self.archive_game_id: str | None = None
        self.archive_error: str | None = None
        self._archive_ack_pending_turn: int | None = None
        self._archive_state_pending = False
        self._turn_start_states: dict[str, dict[str, Any]] = {}
        # Rendering has two bounded workers and a latest-only pending slot.
        # Two workers let a new game overtake one obsolete in-flight render;
        # coalescing below prevents old full-prefix jobs from growing a queue.
        # The workers never take `_lock`, so neither the node bridge nor link
        # generation can hold up an `/api/duel/end_turn` response.
        self._render_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="duel-render"
        )
        self._archive_render_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="duel-archive-render"
        )
        self._render_lock = threading.RLock()
        self._render_futures: set[Future[Any]] = set()
        self._render_cancel_events: dict[Future[Any], threading.Event] = {}
        self._render_job_payloads: dict[Future[Any], dict[str, Any]] = {}
        self._archive_live_render_keys: set[tuple[str, int]] = set()
        self._archive_render_scheduled: set[tuple[str, int]] = set()
        self._archive_recovery_scan_scheduled = False
        self._archive_recovery_last_scan = 0.0
        self._archive_shutdown = threading.Event()
        self._archive_pending_lock = threading.RLock()
        self._archive_pending_renders: dict[
            tuple[str, int], tuple[bytes, str | None]
        ] = {}
        self._render_game_token = 0
        self._render_states: dict[int, dict[str, Any]] = {}
        # Keyed by (game generation, turn), so an image request already in
        # flight during New game can finish with the OLD bytes rather than a
        # noisy 404 or, worse, the new game's same-numbered turn.
        self._render_images: dict[tuple[int, int], bytes] = {}
        # A previous process may have acknowledged turns whose background PNG
        # had not finished when it died. Their exact render inputs live in the
        # atomic turn records, so recovery remains asynchronous and does not
        # weaken the end_turn latency contract.
        self._request_archive_recovery_scan(force=True)

    # -- lifecycle -------------------------------------------------------
    def shutdown(self) -> None:
        with self._lock:
            if self.worker is not None:
                self.worker.shutdown()
                self.worker = None
        with self._render_lock:
            # A clean stop owns every settled turn already acknowledged to
            # the browser. Queue archive completion before cancelling the
            # latest-only live jobs, then let the dedicated archive worker
            # finish any PNG the live job did not produce.
            for future in tuple(self._render_futures):
                self._queue_archive_fallback_locked(future)
                cancel = self._render_cancel_events.get(future)
                if cancel is not None:
                    cancel.set()
        self._request_archive_recovery_scan(force=True)
        # `shutdown` is the ownership boundary: after it returns no worker may
        # rename another archive file (important for clean server teardown and
        # temporary-root tests). Cancel-aware replay-bot processes exit
        # promptly when the events above are set.
        self._render_executor.shutdown(wait=True, cancel_futures=True)
        # No live producer can add pending bytes beyond this point. Repair a
        # last-turn JSON gap while the archive worker is still open, then put
        # one final scan behind every previously queued archive task. This
        # covers a turn whose live renderer was deliberately not started
        # because its first durable ACK failed.
        with self._lock:
            self._sync_archive_turns()
        try:
            self._archive_render_executor.submit(self._run_archive_recovery_scan)
        except RuntimeError:
            # A repeated shutdown call has no remaining worker ownership.
            pass
        self._archive_render_executor.shutdown(wait=True, cancel_futures=False)
        # Retry already-rendered PNG bytes synchronously at the ownership
        # boundary. This covers a final transient record_render failure after
        # its live Future had already left `_render_futures`.
        self._flush_all_pending_archive_renders()
        self._archive_shutdown.set()

    def _reset_renders(self) -> None:
        """Reset renders."""
        with self._render_lock:
            for future in tuple(self._render_futures):
                self._queue_archive_fallback_locked(future)
                cancel = self._render_cancel_events.get(future)
                if cancel is not None:
                    cancel.set()
                if not future.running():
                    future.cancel()
            self._render_game_token += 1
            self._render_states = {}
            # One previous generation is enough for requests the browser had
            # already issued before it processed the New game response.
            keep_from = int(self._render_game_token) - 1
            self._render_images = {
                key: image
                for key, image in self._render_images.items()
                if int(key[0]) >= keep_from
            }

    def _ensure_agent(self) -> bool:
        """Ensure agent."""
        if self.agent is not None:
            return True
        try:
            self.agent = self._build_agent_fn(self.agent_config)
        except Exception as exc:
            self.agent_error = f"agent_build_failed:{type(exc).__name__}:{exc}"
            return False
        self.agent_error = None
        self.worker = AiTurnWorker(self.agent, self.ai_config, clock=self._clock)
        self._ensure_value_probe()
        return True

    def _ensure_value_probe(self) -> None:
        """Build the value probe beside the agent, and never in front of it.

        A probe that cannot be built (an unpinned curve, a head whose scale this
        code does not recognise) must not stop a game from starting: the duel is
        the product, the readout is an addition to it. The failure is recorded
        and shown on the page instead.
        """
        if not self.show_value or self.value_probe is not None or self.agent is None:
            return
        try:
            self.value_probe = self._build_value_probe_fn(self.agent)
            self.value_probe_error = None
        except Exception as exc:
            self.value_probe = None
            self.value_probe_error = f"value_probe_build_failed:{type(exc).__name__}:{exc}"

    def new_game(
        self,
        *,
        seed: int | None = None,
        deepen_width: int | None = None,
        turn_budget_s: float | None = None,
        gear: str | None = None,
        search_width: int | None = None,
    ) -> dict[str, Any]:
        """Start a fresh duel. Returns the same envelope every mutation does."""
        with self._lock:
            if not self._ensure_agent():
                return {"ok": False, "error": self.agent_error, "state": self.snapshot()}
            assert self.worker is not None

            try:
                pending = self._plan_settings(
                    deepen_width, turn_budget_s, gear, search_width
                )
            except ValueError as exc:
                return {"ok": False, "error": str(exc), "state": self.snapshot()}

            # The old session is the only in-memory owner of its exact turn
            # records. Never replace it until every committed turn is durable;
            # otherwise a transient third write failure followed by New game
            # would make that turn (and any already-rendered pending PNG)
            # unrecoverable. A failed preflight leaves the old game playable
            # and lets the caller retry the mutation.
            if not self._sync_archive_turns():
                return {
                    "ok": False,
                    "error": self.archive_error or "archive_turn_sync_failed",
                    "state": self.snapshot(),
                }
            self._archive_ack_pending_turn = None
            self._reset_renders()
            # Already-produced archive bytes are owned by the dedicated
            # archive worker.  New game must never fsync them while holding
            # the mutation lock; `_sync_archive_turns` requested recovery and
            # `_reset_renders` queued fallbacks for any live producer.
            self.worker.bump_generation()
            # After the bump, and only here: a turn still running in the AI
            # thread has already had its result invalidated, and `_install_
            # settings` swaps whole objects rather than assigning into live
            # ones, so that turn keeps reading the recommender it started on.
            self._install_settings(pending)
            if seed is None:


                seed = int(time.time_ns() % DEFAULT_SEED_SPACE)
            self.game_seed = int(seed)
            self.seeds = {
                "game": int(seed),
                "human": derive_seed(seed, "human"),
                "ai": derive_seed(seed, "ai"),
                "battle": derive_seed(seed, "battle"),
            }
            self.session = DuelSession(
                self.initial_state,
                rules=self.rules,
                turn_cap=self.turn_cap,
                human_seed=self.seeds["human"],
                ai_seed=self.seeds["ai"],
                run_battle_fn=seeded_battle_fn(self.seeds["battle"]),
            )
            self.turn_records = []
            self._archive_state_pending = False
            self._ai_failures_in_a_row = 0
            self._forced_end = None
            self._payload_dropped = set()
            self._payload_over_ceiling = 0
            self._payload_worst_record = 0
            self._games_played += 1
            self.value_log = (
                ValueLog(
                    game_id=None,
                    seed=int(seed),
                    root=self.archive.root if self.archive is not None else None,
                )
                if self.show_value
                else None
            )
            self._submit_ai_turn()


            self._game_status_token += 1
            self._turn_start_states = {
                HUMAN: copy.deepcopy(self.session.human.state),
                AI: copy.deepcopy(self.session.ai.state),
            }
            self._begin_archive()
            return {"ok": True, "error": None, "state": self.snapshot()}

    def _begin_archive(self) -> None:
        self.archive_game_id = None
        self.archive_error = None
        archive = self.archive
        session = self.session
        if archive is None or session is None:
            return
        ai_version = self._agent_description() or {}
        try:
            self.archive_game_id = archive.begin_game(
                {
                    "git_sha": ai_version.get("git_sha"),
                    "ai_version": ai_version,
                    "seeds": copy.deepcopy(self.seeds),
                    "rules": copy.deepcopy(session.snapshot()["rules"]),
                    "final": copy.deepcopy(session.snapshot()["final"]),
                    "gear_start": self.gear_config(),
                }
            )
        except Exception as exc:
            self.archive_error = f"archive_begin_failed:{type(exc).__name__}:{exc}"
        if self.value_log is not None:
            # The log is named after the archived game so the two can be joined
            # later; without an archive it stays in memory and the page still
            # shows it.
            self.value_log.game_id = self.archive_game_id

    def _submit_ai_turn(self) -> None:
        """Set the AI thinking about the turn that is starting now."""
        session = self.session
        worker = self.worker
        if session is None or worker is None or session.done:
            return
        self._gen += 1
        worker.submit(
            copy.deepcopy(session.ai.state),
            gen=self._gen,
            turn=int(session.turn),
            wins=int(session.ai.wins),
            engine_seed=int(self.seeds.get("ai", 0)),
        )

    # -- config ----------------------------------------------------------
    def set_config(
        self,
        *,
        gear: str | None = None,
        width: int | None = None,
        turn_budget_s: float | None = None,
    ) -> dict[str, Any]:
        """Install a validated search configuration for future turns.

        A whole new `AiTurnConfig` is built and swapped in rather than the
        live one mutated: the worker thread may be inside a segment, and a
        half-updated config (new gear, old width) is a state neither gear
        describes.

        The change lands on the NEXT turn, not this one: `AiTurnWorker._play_turn`
        binds its config once, at the top, so a turn already in flight plays out
        (and is RECORDED) under the config it started with. That is deliberate
        -- the per-turn record is what tells the human how hard the AI thought,
        and a turn labelled with a gear it only ran half of would be a wrong
        number rather than a live one.
        """
        with self._lock:
            current = self.ai_config
            try:
                nxt = AiTurnConfig(
                    chunk_size=current.chunk_size,
                    adaptive_chunks=current.adaptive_chunks,
                    adaptive_chunk_sizes=current.adaptive_chunk_sizes,
                    initial_candidate_s=current.initial_candidate_s,
                    chunk_ewma_alpha=current.chunk_ewma_alpha,
                    deadline_safety_s=current.deadline_safety_s,
                    gear=str(gear) if gear is not None else current.gear,
                    width=int(width) if width is not None else current.width,
                    turn_budget_s=(
                        float(turn_budget_s)
                        if turn_budget_s is not None
                        else current.turn_budget_s
                    ),
                    mean_segments=current.mean_segments,
                    full_clock_max_width=current.full_clock_max_width,
                    finish_turn=True,
                    chunk_budget=current.chunk_budget,
                    max_segments=current.max_segments,
                )
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": str(exc), "state": self.snapshot()}
            self.ai_config = nxt
            if self.worker is not None:
                self.worker.config = nxt
            return {"ok": True, "error": None, "state": self.snapshot()}


    def _plan_settings(
        self,
        deepen_width: int | None,
        turn_budget_s: float | None,
        gear: str | None = None,
        search_width: int | None = None,
    ) -> dict[str, Any]:
        """Validate a public preset and prepare its search configuration without installing it."""
        for supplied, expected, label in (
            (deepen_width, COMPLETION_WIDTH, "w"),
            (search_width, ROOT_WIDTH, "root"),
        ):
            if supplied is not None and supplied != expected:
                raise ValueError(f"The demo fixes {label} at {expected}.")
        mode = gear if gear is not None else public_mode(self.ai_config.gear)
        # An obsolete process setting must not become a new game's preset.
        if gear is None and mode not in SEARCH_MODES:
            mode = "grow-k"
        selected_gear = engine_gear(str(mode))
        seconds = turn_seconds(turn_budget_s if turn_budget_s is not None else DEFAULT_TURN_SECONDS)
        pending: dict[str, Any] = {
            "gear": selected_gear,
            "search_width": ROOT_WIDTH,
            "turn_budget_s": seconds,
        }
        current = self.agent_config
        cfg = replace(
            current, width=ROOT_WIDTH, completion_policy=COMPLETION_V_SEARCH,
            completion_width=COMPLETION_WIDTH, stochastic_samples=INITIAL_SAMPLES,
        )
        if cfg != current:
            agent = self.agent
            if agent is None:
                raise ValueError("Agent is not ready.")
            search = self._build_search_fn(
                getattr(agent, "bc", None), getattr(agent, "vgame_scorer", None), cfg,
            )
            pending.update(agent_config=cfg, search=search)
        return pending

    def _install_settings(self, pending: dict[str, Any]) -> None:
        """Install what `_plan_settings` built. Cannot fail; call under the
        lock, after the worker generation has been bumped."""
        if not pending:
            return
        cfg = pending.get("agent_config")
        if cfg is not None:
            self.agent_config = cfg
            agent = self.agent
            if agent is not None:
                # Two plain attribute stores on the handle the worker already
                # holds, so nothing has to be told about the change. The
                # description is NOT cached across configs (see
                # `AgentHandle.describe`), which is what makes the archive
                # record this game's setting rather than the first game's.
                agent.search = pending["search"]
                agent.config = cfg
        budget = pending.get("turn_budget_s")
        gear = pending.get("gear")
        width = pending.get("search_width")
        if budget is not None or gear is not None or width is not None:
            # One `set_config` call, because it builds a whole new
            # `AiTurnConfig` rather than mutating the live one; two calls
            # would put a half-updated pair in force between them.
            self.set_config(gear=gear, width=width, turn_budget_s=budget)

    def gear_config(self) -> dict[str, Any]:
        cfg = self.ai_config
        return {
            "gear": public_mode(cfg.gear),
            "gears": list(SEARCH_MODES),
            "width": int(cfg.width),
            "finish_turn": bool(cfg.finish_turn),
            "chunk_size": int(cfg.chunk_size),
            "adaptive_chunks": bool(cfg.adaptive_chunks),
            "adaptive_chunk_sizes": list(cfg.adaptive_chunk_sizes),
            "turn_budget_s": float(cfg.turn_budget_s),
            "full_clock_max_width": int(cfg.full_clock_max_width),


            "deepen_width": (
                int(self.agent_config.completion_width)
                if str(self.agent_config.completion_policy) == COMPLETION_V_SEARCH
                else 0
            ),
            "completion_policy": str(self.agent_config.completion_policy),
            "completion_width": int(self.agent_config.completion_width),
            "stochastic_samples": int(self.agent_config.stochastic_samples),
        }

    # -- the human's shop phase -----------------------------------------
    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Response shape is `App.apply`'s, so the shop skin's own `applyAction`
        / `applyCompose` drive this without knowing it is a duel."""
        with self._lock:
            session = self.session
            if session is None:
                return self._no_game_envelope()
            # INSIDE the lock. The HTTP server is threaded, so two clicks are
            # two threads: a check made before taking the lock would let both
            # read the same board and both apply, which is the bug.
            stale = history_precondition_error(
                payload, client_history_entries(
                    session.human.history, session.human.history_group_ids)
            )
            if stale is not None:
                return {
                    "ok": False,
                    "error": stale,
                    "transition": None,
                    "state": self.snapshot(),
                }
            if isinstance(payload, dict) and "compose" in payload:
                out = session.apply_human_composition(payload)
                out["state"] = self.snapshot()
                return out
            action, err = self._resolve_action(payload)
            if err is not None:
                return {"ok": False, "error": err, "transition": None, "state": self.snapshot()}
            if str(action.get("type", "")).strip().upper() == "END_TURN":
                # END_TURN moves BOTH sides; it is the duel step, not a shop
                # action, and the page posts it to /api/duel/end_turn.
                return {
                    "ok": False,
                    "error": "use_end_turn_for_duel_step",
                    "transition": None,
                    "state": self.snapshot(),
                }
            out = session.apply_human_action(action)
            out["state"] = self.snapshot()
            return out

    def _resolve_action(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        """`App._resolve_action`'s contract, on the human's side of the duel."""
        session = self.session
        assert session is not None
        if not isinstance(payload, dict):
            return None, "missing_action_or_action_index"
        if "action_index" in payload:
            idx = payload["action_index"]
            if not isinstance(idx, int):
                return None, "action_index_must_be_int"
            actions = legal_actions(session.human.state)
            if idx < 0 or idx >= len(actions):
                return None, f"action_index_out_of_range:{idx}"
            return actions[idx], None
        if "action" not in payload:
            return None, "missing_action_or_action_index"
        action = payload["action"]
        if not isinstance(action, dict):
            return None, "action_must_be_object"
        return action, None

    def undo(self) -> dict[str, Any]:
        """Undo the human's last action -- WITHIN the current turn only."""
        with self._lock:
            session = self.session
            if session is None:
                return self._no_game_envelope()
            out = session.undo_human_action()
            out["state"] = self.snapshot()
            return out


    def _human_race(self, session: DuelSession) -> dict[str, int]:
        """The human decision's driver-true bypass block."""
        return {
            "turn": int(session.human.state.get("turn", 0) or 0),
            "lives": int(session.human.lives),
            "opponent_lives": int(session.ai.lives),
            "wins": int(session.human.wins),
        }

    def _ai_race(self, session: DuelSession) -> dict[str, int]:
        return {
            "turn": int(session.ai.state.get("turn", 0) or 0),
            "lives": int(session.ai.lives),
            "opponent_lives": int(session.human.lives),
            "wins": int(session.ai.wins),
        }

    def value(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """`POST /api/infer/value`: what the agent thinks a board is worth.

        The board defaults to the duel's LIVE HUMAN board, which is the whole
        point of the feature; `state` overrides it for any other board."""
        request = payload if isinstance(payload, dict) else {}
        if not self.show_value:
            return {"ok": False, "error": "value_display_disabled", "value": None}
        with self._lock:
            session = self.session
            if session is None:
                return {"ok": False, "error": "duel_no_game", "value": None}
            if not self._ensure_agent():
                return {"ok": False, "error": self.agent_error, "value": None}
            self._ensure_value_probe()
            probe = self.value_probe
            if probe is None:
                return {
                    "ok": False,
                    "error": self.value_probe_error or "value_probe_unavailable",
                    "value": None,
                }
            supplied = request.get("state")
            if supplied is not None and not isinstance(supplied, dict):
                return {"ok": False, "error": "state_must_be_object", "value": None}
            if supplied is None:
                state = copy.deepcopy(session.human.state)
                race = self._human_race(session)
                board_source = "live_human_board"
            else:
                state = copy.deepcopy(supplied)
                race = self._human_race(session)
                supplied_race = request.get("race")
                if isinstance(supplied_race, dict):
                    for key in ("turn", "lives", "opponent_lives", "wins"):
                        if supplied_race.get(key) is not None:
                            race[key] = int(supplied_race[key])
                board_source = "supplied"
            worker_status = None
            if self.worker is not None:
                try:
                    worker_status = str(self.worker.status().get("status"))
                except Exception:
                    worker_status = None

        complete = bool(request.get("complete"))
        want_actions = bool(request.get("actions"))
        actions: list[dict[str, Any]] = []
        n_legal = 0
        capped = False
        if want_actions:
            try:
                actions = state_legal_actions(state)
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"legal_actions_failed:{type(exc).__name__}:{exc}",
                    "value": None,
                }
            n_legal = len(actions)
            cap = int(request.get("max_actions") or self.value_action_cap)
            if cap > 0 and n_legal > cap:
                actions = actions[:cap]
                capped = True

        # Outside `self._lock`: a value probe is milliseconds, but a level-C
        # batch is one BC completion per action, and holding the mutation lock
        # for that would block the human's own clicks. `_value_lock` still
        # serialises probes against each other.
        with self._value_lock:
            try:
                out = probe.evaluate(
                    state=state, race=race, complete=complete, actions=actions
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "error": f"value_failed:{type(exc).__name__}:{exc}",
                    "value": None,
                }
        out["board_source"] = board_source
        out["actions_capped"] = bool(capped)
        out["n_legal_actions"] = int(n_legal)
        # Which side of the fixed clock this probe landed on. A level-C batch
        # taken while the AI is thinking is GIL time off its own search, so the
        # response says so instead of leaving it to be inferred.
        out["ai_worker_status"] = worker_status
        return out

    def _value_of_end_of_turn_boards(
        self,
        *,
        session: DuelSession,
        human_state_before: dict[str, Any],
        human_actions: list[dict[str, Any]],
        ai_state_before: dict[str, Any],
        ai_actions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        """Both sides' end-of-turn boards for THIS turn, valued the same way."""
        probe = self.value_probe
        if probe is None:
            return None, None, self.value_probe_error or "value_probe_unavailable"
        try:
            human_board = apply_actions(human_state_before, human_actions)
            ai_board = apply_actions(ai_state_before, ai_actions)
            human = probe.evaluate(
                state=human_board,
                race={
                    "turn": int(human_state_before.get("turn", 0) or 0),
                    "lives": int(human_state_before.get("lives", 0) or 0),
                    "opponent_lives": int(ai_state_before.get("lives", 0) or 0),
                    "wins": int(session.human.wins),
                },
            )["value"]
            ai = probe.evaluate(
                state=ai_board,
                race={
                    "turn": int(ai_state_before.get("turn", 0) or 0),
                    "lives": int(ai_state_before.get("lives", 0) or 0),
                    "opponent_lives": int(human_state_before.get("lives", 0) or 0),
                    "wins": int(session.ai.wins),
                },
            )["value"]
            return human, ai, None
        except Exception as exc:
            return None, None, f"value_failed:{type(exc).__name__}:{exc}"

    def _record_value_row(
        self,
        *,
        turn: int,
        record: dict[str, Any],
        human_state_before: dict[str, Any],
        human_actions: list[dict[str, Any]],
        ai_state_before: dict[str, Any],
        ai_actions: list[dict[str, Any]],
        human_wins_before: int,
        ai_wins_before: int,
    ) -> None:
        """Append this turn's predicted-versus-realised row. Never raises.

        A readout must not be able to fail a turn the human already played, so
        every failure below lands in the row's `error` field, which the page
        shows, rather than in the response.
        """
        log = self.value_log
        session = self.session
        if log is None or session is None:
            return
        human, ai, error = self._value_of_end_of_turn_boards(
            session=session,
            human_state_before=human_state_before,
            human_actions=human_actions,
            ai_state_before=ai_state_before,
            ai_actions=ai_actions,
        )
        try:
            log.record_turn(
                turn=int(turn),
                human=human,
                ai=ai,
                human_wins_before=int(human_wins_before),
                ai_wins_before=int(ai_wins_before),
                outcome=record.get("outcome"),
                error=error,
            )
        except Exception:
            return

    def _finalise_value_log(self) -> None:
        log = self.value_log
        session = self.session
        if log is None or session is None:
            return
        if not (session.done or self._forced_end):
            return
        try:
            log.finalise(
                final_human_wins=int(session.human.wins),
                final_ai_wins=int(session.ai.wins),
            )
        except Exception:
            return

    # -- the duel step ---------------------------------------------------
    def end_turn(self) -> dict[str, Any]:
        """Wait for the AI's fixed-clock turn, resolve the battle, advance."""
        with self._lock:
            session = self.session
            worker = self.worker
            if session is None or worker is None:
                return self._no_game_envelope()
            if self._archive_ack_pending_turn is not None:
                pending_turn = int(self._archive_ack_pending_turn)
                archive_ready = self._ensure_archive_game()
                live_key = (
                    self._reserve_archive_live_render(pending_turn)
                    if archive_ready
                    else None
                )
                if not archive_ready or not self._sync_archive_turns():
                    self._release_archive_live_render(live_key)
                    return {
                        "ok": False,
                        "error": self.archive_error or "archive_turn_sync_failed",
                        "turn": pending_turn,
                        "state": self.snapshot(),
                    }
                try:
                    self._schedule_render(pending_turn)
                except Exception:
                    self._release_archive_live_render(live_key)
                    raise
                self._archive_ack_pending_turn = None
                return {
                    "ok": True,
                    "error": None,
                    "turn": pending_turn,
                    "state": self.snapshot(),
                }
            if session.done or self._forced_end:


                return {
                    "ok": False,
                    "error": f"duel_over:{self._forced_end or session.end_reason}",
                    "state": self.snapshot(),
                }

            turn = int(session.turn)
            human_state_before = copy.deepcopy(
                self._turn_start_states.get(HUMAN) or session.human.state
            )
            ai_state_before = copy.deepcopy(
                self._turn_start_states.get(AI) or session.ai.state
            )
            human_history_mark = int(session._human_turn_mark)
            ai_history_mark = len(session.ai.history)
            human_start_transition = copy.deepcopy(
                session.human.turn_start_transition
            )
            ai_start_transition = copy.deepcopy(
                session.ai.turn_start_transition
            )
            human_actions = [
                copy.deepcopy(tr.get("action") or {})
                for tr in session.human.history[human_history_mark:]
            ]


            human_wins_before = int(session.human.wins)
            ai_wins_before = int(session.ai.wins)
            t_end_turn = self._clock()
            worker.note_human_end_turn()
            ai_result = worker.await_turn(self._gen, timeout=AI_TURN_TIMEOUT_S)
            t_ai_ready = self._clock()

            if ai_result.get("ok"):
                self._ai_failures_in_a_row = 0
                session.ai_decide = lambda _s, _a=ai_result["actions"]: copy.deepcopy(_a)
            else:
                self._ai_failures_in_a_row += 1
                session.ai_decide = lambda _s: []

            step = session.end_turn()
            human_transitions = copy.deepcopy(
                session.human.history[human_history_mark:]
            )
            ai_transitions = copy.deepcopy(
                session.ai.history[ai_history_mark:]
            )
            t_result = self._clock()
            worker.bump_generation()

            if not step.get("ok"):
                self._ai_failures_in_a_row += 1
                self._forced_end = f"end_turn_failed:{step.get('error')}"
                self.turn_records.append(
                    self._turn_record(
                        turn=turn,
                        ai_result=ai_result,
                        result=None,
                        error=str(step.get("error")),
                        stop_latency_ms=int((t_ai_ready - t_end_turn) * 1000),
                        end_to_result_ms=int((t_result - t_end_turn) * 1000),
                        human_state_before=human_state_before,
                        ai_state_before=ai_state_before,
                        human_actions=human_actions,
                        human_start_transition=human_start_transition,
                        ai_start_transition=ai_start_transition,
                        human_transitions=human_transitions,
                        ai_transitions=ai_transitions,
                    )
                )
                self._archive_game_state()
                return {"ok": False, "error": step.get("error"), "state": self.snapshot()}

            record = self._turn_record(
                turn=turn,
                ai_result=ai_result,
                result=step["result"],
                error=None,
                stop_latency_ms=int((t_ai_ready - t_end_turn) * 1000),
                end_to_result_ms=int((t_result - t_end_turn) * 1000),
                human_state_before=human_state_before,
                ai_state_before=ai_state_before,
                human_actions=human_actions,
                human_start_transition=human_start_transition,
                ai_start_transition=ai_start_transition,
                human_transitions=human_transitions,
                ai_transitions=ai_transitions,
            )
            self.turn_records.append(record)
            if self._ai_failures_in_a_row >= 2:
                self._forced_end = "ai_failed"
            # After the record, before the archive: the row is a second artifact
            # beside the archive and never inside it, so an archived game is
            # byte-identical with the display on and off.
            self._record_value_row(
                turn=turn,
                record=record,
                human_state_before=human_state_before,
                human_actions=human_actions,
                ai_state_before=ai_state_before,
                ai_actions=copy.deepcopy(ai_result.get("actions") or []),
                human_wins_before=human_wins_before,
                ai_wins_before=ai_wins_before,
            )
            self._finalise_value_log()
            archive_ready = self._ensure_archive_game()
            live_key = (
                self._reserve_archive_live_render(turn)
                if archive_ready
                else None
            )
            archive_synced = archive_ready and self._sync_archive_turns()
            if archive_synced:
                try:
                    # The durable job now exists. Starting the renderer only
                    # after record_turn prevents a fast PNG fsync from taking
                    # the archive lock in front of this HTTP request; the key
                    # reserved above still closes the recovery-scan gap.
                    self._schedule_render(turn)
                except Exception:
                    self._release_archive_live_render(live_key)
                    raise
            else:
                self._release_archive_live_render(live_key)

            self._turn_start_states = {
                HUMAN: copy.deepcopy(session.human.state),
                AI: copy.deepcopy(session.ai.state),
            }

            if not self._forced_end and not session.done:
                self._submit_ai_turn()

            if not archive_synced:
                # The battle is committed in memory, but the API must not
                # acknowledge it as successful until its atomic turn+render
                # job record is durable. A retry first syncs and acknowledges
                # this same turn without playing the next one.
                self._archive_ack_pending_turn = turn
                return {
                    "ok": False,
                    "error": self.archive_error or "archive_turn_sync_failed",
                    "turn": turn,
                    "state": self.snapshot(),
                }
            return {"ok": True, "error": None, "turn": turn, "state": self.snapshot()}

    def _turn_record(
        self,
        *,
        turn: int,
        ai_result: dict[str, Any],
        result: dict[str, Any] | None,
        error: str | None,
        stop_latency_ms: int,
        end_to_result_ms: int,
        human_state_before: dict[str, Any],
        ai_state_before: dict[str, Any],
        human_actions: list[dict[str, Any]],
        human_start_transition: dict[str, Any] | None,
        ai_start_transition: dict[str, Any] | None,
        human_transitions: list[dict[str, Any]],
        ai_transitions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Search counts, deadline minima, and throughput are recorded from the
        worker's own result rather than re-derived from UI timing."""
        ai_ok = bool(ai_result.get("ok"))
        code = ai_result.get("error")
        record: dict[str, Any] = {
            "turn": int(turn),
            "ok": bool(result is not None and error is None),
            "error": error,
            "ai_ok": ai_ok,
            "ai_error": code,
            "banner": None if ai_ok else f"AI failed turn {int(turn)}: {code}",
            "n_segments": int(ai_result.get("n_segments") or 0),
            "searched_segments": int(ai_result.get("searched_segments") or 0),
            "deadline_segments": int(ai_result.get("deadline_segments") or 0),
            "greedy_segments": int(ai_result.get("greedy_segments") or 0),
            "segments": copy.deepcopy(ai_result.get("segments") or []),
            "elapsed_ms": int(ai_result.get("elapsed_ms") or 0),
            "stop_latency_ms": stop_latency_ms,
            "end_wait_ms": stop_latency_ms,
            "end_to_result_ms": end_to_result_ms,
            "gear": ai_result.get("gear") or self.ai_config.gear,
            "finish_turn": bool(ai_result.get("finish_turn", self.ai_config.finish_turn)),
            "chunk_size": int(ai_result.get("chunk_size") or self.ai_config.chunk_size),
            "adaptive_chunks": bool(
                ai_result.get("adaptive_chunks", self.ai_config.adaptive_chunks)
            ),
            "turn_budget_s": float(
                ai_result.get("turn_budget_s") or self.ai_config.turn_budget_s
            ),
            "budget_utilization": ai_result.get("budget_utilization"),
            "deadline_overshoot_ms": int(
                ai_result.get("deadline_overshoot_ms") or 0
            ),
            "effective_unique_prefixes": int(
                ai_result.get("effective_unique_prefixes") or 0
            ),
            "dedup_ratio": ai_result.get("dedup_ratio"),
            "leaves_evaluated": int(ai_result.get("leaves_evaluated") or 0),
            "safety_cap_hit": bool(ai_result.get("safety_cap_hit")),


            "completion_decided": ai_result.get("completion_decided"),
            "completion_divergent": ai_result.get("completion_divergent"),


            "realised_stochastic_samples": ai_result.get("realised_stochastic_samples"),
            "extra_sample_levels": ai_result.get("extra_sample_levels"),
            "completion_dropped": ai_result.get("completion_dropped"),
            "completion_second_chance_nodes": ai_result.get("completion_second_chance_nodes"),
            "search_used": bool(ai_result.get("search_used")),
            "search_error": ai_result.get("search_error"),
            "segments_capped": bool(ai_result.get("segments_capped")),
            "ai_actions": [str(a.get("type")) for a in (ai_result.get("actions") or [])],
        }
        ai_actions = copy.deepcopy(ai_result.get("actions") or [])
        human_annotation = annotate_chain(human_state_before, human_actions)
        ai_annotation = annotate_chain(ai_state_before, ai_actions)
        record.update(
            {
                "human": {
                    "state_before": copy.deepcopy(human_state_before),
                    "actions": copy.deepcopy(human_actions),
                    "action_text": [step["text"] for step in human_annotation["steps"]],
                    "diverged_at": human_annotation["diverged_at"],
                },
                "ai": {
                    "state_before": copy.deepcopy(ai_state_before),
                    "actions": ai_actions,
                    "action_text": [step["text"] for step in ai_annotation["steps"]],
                    "diverged_at": ai_annotation["diverged_at"],
                },
            }
        )
        if result is not None:
            record.update(
                {
                    "outcome": str(result["human"]["outcome"]),
                    "ai_outcome": str(result["ai"]["outcome"]),
                    "life_bonus_applied": bool(result["human"]["life_bonus_applied"]),
                    "lives": {
                        HUMAN: int(result["human"]["lives"]),
                        AI: int(result["ai"]["lives"]),
                    },
                    "battle": copy.deepcopy(result.get("battle") or {}),
                    "timing_ms": copy.deepcopy(result.get("timing_ms") or {}),
                    "wins": {
                        HUMAN: int(self.session.human.wins if self.session else 0),
                        AI: int(self.session.ai.wins if self.session else 0),
                    },
                    # The AI board that actually fought. This is the ONLY AI
                    # board that ever reaches the browser, and only once its
                    # battle has resolved.
                    "ai_board": copy.deepcopy(result["ai"].get("board_pre_battle_team") or []),
                    "human_board": copy.deepcopy(
                        result["human"].get("board_pre_battle_team") or []
                    ),
                }
            )
            record["human"]["board_pre_battle"] = copy.deepcopy(
                result["human"].get("board_pre_battle_team") or []
            )
            record["ai"]["board_pre_battle"] = copy.deepcopy(
                result["ai"].get("board_pre_battle_team") or []
            )
        if not isinstance(human_start_transition, dict) or not isinstance(
            ai_start_transition, dict
        ):
            raise RuntimeError("replay_start_transition_missing")
        human_context = {
            "outcome": record.get("outcome"),
            "lives": (record.get("lives") or {}).get(HUMAN),
            "wins": (record.get("wins") or {}).get(HUMAN),
            "board_pre_battle": (record.get("human") or {}).get(
                "board_pre_battle"
            ),
            "battle": record.get("battle") or {},
            "timing_ms": record.get("timing_ms") or {},
        }
        ai_context = {
            "outcome": record.get("ai_outcome"),
            "lives": (record.get("lives") or {}).get(AI),
            "wins": (record.get("wins") or {}).get(AI),
            "board_pre_battle": (record.get("ai") or {}).get(
                "board_pre_battle"
            ),
            "battle": record.get("battle") or {},
            "timing_ms": record.get("timing_ms") or {},
        }
        record["human"]["replay"] = build_side_replay(
            turn=turn,
            start_transition=human_start_transition,
            transitions=human_transitions,
            board_pre_battle=record["human"].get("board_pre_battle") or [],
            end_context=human_context,
        )
        record["ai"]["replay"] = build_side_replay(
            turn=turn,
            start_transition=ai_start_transition,
            transitions=ai_transitions,
            board_pre_battle=record["ai"].get("board_pre_battle") or [],
            end_context=ai_context,
        )
        return record

    def _archive_game_state(self) -> bool:
        self._archive_state_pending = True
        return self._sync_archive_game_state(attempts=1)

    def _sync_archive_game_state(self, *, attempts: int = 3) -> bool:
        archive = self.archive
        session = self.session
        if archive is None:
            self._archive_state_pending = False
            return True
        if not self._archive_state_pending:
            return True
        if session is None or not self._ensure_archive_game():
            return False
        game_id = self.archive_game_id
        assert game_id is not None
        snapshot = session.snapshot()
        for attempt in range(max(1, int(attempts))):
            try:
                archive.record_game_state(
                    game_id,
                    final=snapshot["final"],
                    done=bool(session.done or self._forced_end),
                    winner=session.winner,
                    end_reason=self._forced_end or session.end_reason,
                )
                break
            except Exception as exc:
                self.archive_error = (
                    f"archive_state_failed:{type(exc).__name__}:{exc}"
                )
                if attempt + 1 < max(1, int(attempts)):
                    time.sleep(0.01 * (attempt + 1))
        else:
            return False
        self._archive_state_pending = False
        if str(self.archive_error or "").startswith("archive_state_failed:"):
            self.archive_error = None
        return True

    def _sync_archive_turns(self) -> bool:
        """Write only missing committed turns, repairing any earlier gap.

        ``record_turn`` replaces and fsyncs the complete game document, so
        replaying the whole in-memory prefix on every request would make a
        long game quadratic in archive writes.  Read the atomic authoritative
        document once, then append only turns that are not durable yet.
        """
        archive = self.archive
        session = self.session
        if archive is None or session is None:
            return True
        if not self.turn_records and not self._archive_state_pending:
            # A zero-turn game is still an authoritative session. If its
            # initial begin_game failed, New game/shutdown must retry creation
            # instead of silently replacing the only in-memory provenance.
            return self._ensure_archive_game()
        if not self._ensure_archive_game():
            return False
        game_id = self.archive_game_id
        assert game_id is not None
        snapshot = session.snapshot()
        try:
            durable_turns = {
                int(row["turn"])
                for row in (archive.load_game(game_id).get("turns") or [])
            }
        except Exception as exc:
            self.archive_error = (
                f"archive_turn_failed:{type(exc).__name__}:{exc}"
            )
            return False
        for record in self.turn_records:
            # A failed DuelSession step did not settle a battle and is an
            # attempt, not a turn. Its terminal reason is persisted by
            # `_archive_game_state` instead.
            if not record.get("ok"):
                continue
            turn = int(record["turn"])
            if turn in durable_turns:
                continue
            render_job = self._build_archive_render_job(turn)
            if render_job is None:
                self.archive_error = (
                    f"archive_turn_failed:missing_render_job:{turn}"
                )
                return False
            for attempt in range(3):
                try:
                    archive.record_turn(
                        game_id,
                        record,
                        render_job=render_job,
                        final=snapshot["final"],
                        done=bool(session.done or self._forced_end),
                        winner=session.winner,
                        end_reason=self._forced_end or session.end_reason,
                    )
                    durable_turns.add(turn)
                    break
                except Exception as exc:
                    self.archive_error = (
                        f"archive_turn_failed:{type(exc).__name__}:{exc}"
                    )
                    if attempt < 2:
                        time.sleep(0.01 * (attempt + 1))
            else:
                return False
        if not self._sync_archive_game_state():
            return False
        if str(self.archive_error or "").startswith("archive_turn_failed:"):
            self.archive_error = None
        # PNG attachment is not part of the durable ACK boundary. Retrying
        # saved bytes here would put validation/fsync back on the HTTP thread.
        # The archive worker performs the retry; clean shutdown retains the
        # final synchronous ownership-boundary flush.
        self._request_archive_recovery_scan(force=True)
        return True

    def _ensure_archive_game(self) -> bool:
        """Create a missing archive before any committed turn is acknowledged."""
        if self.archive is None:
            return True
        if self.session is None:
            return False
        if self.archive_game_id is None:
            self._begin_archive()
        return self.archive_game_id is not None

    def _reserve_archive_live_render(
        self, turn: int
    ) -> tuple[str, int] | None:
        game_id = self.archive_game_id
        if self.archive is None or game_id is None:
            return None
        key = (str(game_id), int(turn))
        with self._render_lock:
            self._archive_live_render_keys.add(key)
        return key

    def _release_archive_live_render(
        self, key: tuple[str, int] | None
    ) -> None:
        if key is None:
            return
        with self._render_lock:
            self._archive_live_render_keys.discard(key)

    def _record_archive_render(
        self,
        game_id: str,
        turn: int,
        image: bytes,
        calculator_link: str | None,
        *,
        token: int | None = None,
    ) -> bool:
        archive = self.archive
        if archive is None:
            return False
        key = (str(game_id), int(turn))
        try:
            archive.record_render(
                key[0], key[1], bytes(image), calculator_link=calculator_link
            )
        except Exception as exc:


            with self._archive_pending_lock:
                self._archive_pending_renders[key] = (bytes(image), calculator_link)
            if key[0] == self.archive_game_id and (
                token is None or int(token) == int(self._render_game_token)
            ):
                self.archive_error = (
                    f"archive_render_failed:{type(exc).__name__}:{exc}"
                )
            return False
        with self._archive_pending_lock:
            self._archive_pending_renders.pop(key, None)
            current_pending = any(
                pending_key[0] == key[0]
                for pending_key in self._archive_pending_renders
            )
        if (
            key[0] == self.archive_game_id
            and not current_pending
            and str(self.archive_error or "").startswith("archive_render_failed:")
        ):
            self.archive_error = None
        return True

    def _flush_pending_archive_renders(self, game_id: str) -> None:
        with self._archive_pending_lock:
            pending = [
                (key, value)
                for key, value in self._archive_pending_renders.items()
                if key[0] == str(game_id)
            ]
        for key, (image, calculator_link) in pending:
            for attempt in range(3):
                if self._record_archive_render(
                    key[0], key[1], image, calculator_link
                ):
                    break
                if attempt < 2:
                    time.sleep(0.01 * (attempt + 1))

    def _flush_all_pending_archive_renders(self) -> None:
        """Retry every saved PNG after all render workers have stopped."""
        with self._archive_pending_lock:
            game_ids = sorted({key[0] for key in self._archive_pending_renders})
        for game_id in game_ids:
            self._flush_pending_archive_renders(game_id)

    # -- live battle rendering -------------------------------------------
    def _build_archive_render_job(self, turn: int) -> dict[str, Any] | None:
        """Snapshot the exact settled prefix needed to recreate one PNG."""
        session = self.session
        if session is None:
            return None
        target = int(turn)
        rows = [
            copy.deepcopy(row)
            for row in session.human.battle_rows
            if int(row.get("turn") or 0) <= target
        ]
        if not rows or int(rows[-1].get("turn") or 0) != target:
            return None
        last = copy.deepcopy(rows[-1])
        return {
            "rows": rows,
            "max_lives": int(session.rules.max_lives or session.rules.start_lives),
            "calc_config": {
                "playerPack": "Turtle",
                "opponentPack": "Turtle",
                "turn": target,
                # battle_rows are reversed for replay-bot's left-to-right
                # image convention. Calculator links use engine slot order.
                "playerPets": list(
                    reversed(copy.deepcopy(last.get("playerPets") or []))
                ),
                "opponentPets": list(
                    reversed(copy.deepcopy(last.get("opponentPets") or []))
                ),
            },
        }

    def _schedule_render(self, turn: int) -> None:
        """Queue a prefix render of every battle settled through `turn`.

        `DuelSession` appends `battle_rows` only after the shared battle is
        committed. Taking the deep copy here therefore excludes both sides'
        live shop state by construction.
        """
        session = self.session
        job = self._build_archive_render_job(int(turn))
        if session is None or not session.human.battle_rows:
            return
        if job is None:


            rows = copy.deepcopy(session.human.battle_rows)
            last = copy.deepcopy(rows[-1])
            job = {
                "rows": rows,
                "max_lives": int(
                    session.rules.max_lives or session.rules.start_lives
                ),
                "calc_config": {
                    "playerPack": "Turtle",
                    "opponentPack": "Turtle",
                    "turn": int(last.get("turn") or turn),
                    "playerPets": list(
                        reversed(copy.deepcopy(last.get("playerPets") or []))
                    ),
                    "opponentPets": list(
                        reversed(copy.deepcopy(last.get("opponentPets") or []))
                    ),
                },
            }
        rows = job["rows"]
        calc_config = job["calc_config"]
        max_lives = int(job["max_lives"])
        with self._render_lock:
            # Only the newest not-yet-started prefix matters. At most two jobs
            # can be running plus this one latest pending job, even if the user
            # ends turns or starts games faster than node can render.
            for future in tuple(self._render_futures):
                self._queue_archive_fallback_locked(future)
                cancel = self._render_cancel_events.get(future)
                if cancel is not None:
                    cancel.set()
                if not future.running():
                    future.cancel()
            token = int(self._render_game_token)
            archive_game_id = self.archive_game_id
            if archive_game_id is not None:
                self._archive_live_render_keys.add(
                    (str(archive_game_id), int(turn))
                )
            self._render_states[int(turn)] = {
                "status": "pending",
                "turn": int(turn),
                "rows": len(rows),
                "calculator_link": None,
                "error": None,
                "url": None,
            }
        cancel_event = threading.Event()
        future = self._render_executor.submit(
            self._run_render,
            token=token,
            turn=int(turn),
            rows=rows,
            max_lives=max_lives,
            calc_config=calc_config,
            cancel_event=cancel_event,
            archive_game_id=archive_game_id,
        )
        with self._render_lock:
            self._render_futures.add(future)
            self._render_cancel_events[future] = cancel_event
            self._render_job_payloads[future] = {
                "turn": int(turn),
                "rows": rows,
                "max_lives": max_lives,
                "calc_config": calc_config,
                "archive_game_id": archive_game_id,
            }
        future.add_done_callback(self._forget_render_future)

    def _queue_archive_fallback_locked(self, source: Future[Any]) -> None:
        archive = self.archive
        job = self._render_job_payloads.get(source)
        if archive is None or not job or not job.get("archive_game_id"):
            return
        key = (str(job["archive_game_id"]), int(job["turn"]))
        # Do not enter the archive lock while holding `_render_lock`:
        # `_run_render` records the archive first and publishes UI state
        # second, so doing so would invert the lock order. The fallback checks
        # whether bytes already exist after it leaves this lock instead.
        if key in self._archive_render_scheduled:
            return
        self._archive_render_scheduled.add(key)
        self._archive_render_executor.submit(
            self._run_archive_fallback, source, copy.deepcopy(job), key
        )

    def _request_archive_recovery_scan(self, *, force: bool = False) -> None:
        """Request a throttled scan without doing archive I/O on the caller."""
        archive = self.archive
        if archive is None or self._archive_shutdown.is_set():
            return
        now = time.monotonic()
        with self._render_lock:
            if self._archive_recovery_scan_scheduled:
                return
            if (
                not force
                and now - self._archive_recovery_last_scan
                < ARCHIVE_RECOVERY_SCAN_INTERVAL_S
            ):
                return
            self._archive_recovery_scan_scheduled = True
            self._archive_recovery_last_scan = now
            try:
                self._archive_render_executor.submit(
                    self._run_archive_recovery_scan
                )
            except RuntimeError:
                self._archive_recovery_scan_scheduled = False

    def _run_archive_recovery_scan(self) -> None:
        """Discover and finish crash-left jobs on the archive worker."""
        try:
            archive = self.archive
            if archive is None or self._archive_shutdown.is_set():
                return
            pending = archive.pending_render_jobs()
            for game_id, turn, job in pending:
                key = (str(game_id), int(turn))
                with self._render_lock:
                    if (
                        key in self._archive_live_render_keys
                        or key in self._archive_render_scheduled
                    ):
                        continue
                    self._archive_render_scheduled.add(key)
                # This method already runs on the single archive worker. Run
                # the job directly so shutdown cannot close the executor
                # between discovery and a nested submit.
                self._run_persisted_archive_job(copy.deepcopy(job), key)
        except Exception as exc:
            self.archive_error = (
                f"archive_recovery_failed:{type(exc).__name__}:{exc}"
            )
        else:
            if str(self.archive_error or "").startswith(
                "archive_recovery_failed:"
            ):
                self.archive_error = None
        finally:
            with self._render_lock:
                self._archive_recovery_scan_scheduled = False

    def _queue_archive_job(
        self, job: dict[str, Any], key: tuple[str, int]
    ) -> None:
        """Deduplicate a standalone durable render retry."""
        if self.archive is None or self._archive_shutdown.is_set():
            return
        with self._render_lock:
            if key in self._archive_render_scheduled:
                return
            self._archive_render_scheduled.add(key)
            try:
                self._archive_render_executor.submit(
                    self._run_persisted_archive_job, copy.deepcopy(job), key
                )
            except RuntimeError:
                # A concurrent shutdown may close the executor between the
                # public read and this submission. The on-disk job remains
                # available for the next process/read to recover.
                self._archive_render_scheduled.discard(key)

    def _render_archive_job(
        self, job: dict[str, Any], key: tuple[str, int]
    ) -> None:
        """Finish one durable job, preferring any already-produced PNG bytes."""
        if self._archive_shutdown.is_set():
            return
        archive = self.archive
        if archive is None:
            return
        self._flush_pending_archive_renders(key[0])
        if archive.has_render(*key):
            return
        calculator_link = self._calculator_link_fn(job["calc_config"])
        if archive.adopt_render_if_present(
            key[0], key[1], calculator_link=calculator_link
        ):
            return
        out = self._render_fn(
            job["rows"],
            max_lives=int(job["max_lives"]),
            player_name="Human",
            header_opponent_name="AI",
            cancel_event=self._archive_shutdown,
        )
        image = out.get("image") if isinstance(out, dict) else None
        if not (
            isinstance(out, dict)
            and out.get("ok")
            and isinstance(image, bytes)
            and image
        ):
            raise RuntimeError(str((out or {}).get("error") or "archive_render_empty"))
        if not self._record_archive_render(
            key[0], key[1], bytes(image), calculator_link
        ):
            raise RuntimeError(self.archive_error or "archive_render_write_failed")

    def _run_persisted_archive_job(
        self, job: dict[str, Any], key: tuple[str, int]
    ) -> None:
        try:
            self._render_archive_job(job, key)
        except Exception as exc:
            if key[0] == self.archive_game_id:
                self.archive_error = (
                    f"archive_render_failed:{type(exc).__name__}:{exc}"
                )
        finally:
            with self._render_lock:
                self._archive_render_scheduled.discard(key)

    def _run_archive_fallback(
        self,
        source: Future[Any],
        job: dict[str, Any],
        key: tuple[str, int],
    ) -> None:
        try:
            try:
                source.result()
            except Exception:
                pass
            self._render_archive_job(job, key)
        except Exception as exc:
            if key[0] == self.archive_game_id:
                self.archive_error = f"archive_render_failed:{type(exc).__name__}:{exc}"
        finally:
            with self._render_lock:
                self._archive_render_scheduled.discard(key)

    def _forget_render_future(self, future: Future[Any]) -> None:
        with self._render_lock:
            job = self._render_job_payloads.get(future)
            if job is not None and job.get("archive_game_id") is not None:
                self._archive_live_render_keys.discard(
                    (str(job["archive_game_id"]), int(job["turn"]))
                )
            self._render_futures.discard(future)
            self._render_cancel_events.pop(future, None)
            self._render_job_payloads.pop(future, None)

    def _run_render(
        self,
        *,
        token: int,
        turn: int,
        rows: list[dict[str, Any]],
        max_lives: int,
        calc_config: dict[str, Any],
        cancel_event: threading.Event,
        archive_game_id: str | None,
    ) -> None:
        # Both helpers may cross the node bridge. Keep both on the dedicated
        # executor so neither can extend `/api/duel/end_turn` latency.
        try:
            calculator_link = self._calculator_link_fn(calc_config)
        except Exception as exc:
            calculator_link = None
            link_error = f"calculator_link_failed:{type(exc).__name__}:{exc}"
        else:
            link_error = None
        if cancel_event.is_set():
            return
        try:
            out = self._render_fn(
                rows,
                max_lives=int(max_lives),
                player_name="Human",
                header_opponent_name="AI",
                cancel_event=cancel_event,
            )
        except Exception as exc:
            out = {
                "ok": False,
                "error": f"render_exception:{type(exc).__name__}:{exc}",
                "image": None,
            }
        image = out.get("image") if isinstance(out, dict) else None
        ok = bool(isinstance(out, dict) and out.get("ok") and isinstance(image, bytes) and image)
        error = None if ok else str(
            (out.get("error") if isinstance(out, dict) else None) or "empty_render"
        )
        archived = False
        if ok and self.archive is not None and archive_game_id is not None:
            archived = self._record_archive_render(
                archive_game_id,
                turn,
                bytes(image),
                calculator_link,
                token=token,
            )
        if self.archive is not None and archive_game_id is not None and (
            not ok or not archived
        ):
            self._queue_archive_job(
                {
                    "rows": rows,
                    "max_lives": int(max_lives),
                    "calc_config": calc_config,
                },
                (str(archive_game_id), int(turn)),
            )
        with self._render_lock:
            # A slow result from the old game is discarded, not relabelled as
            # the new game's turn with the same number.
            if int(token) != int(self._render_game_token):
                return
            state = self._render_states.get(int(turn))
            if state is None:
                return
            if ok:
                self._render_images[(int(token), int(turn))] = bytes(image)
                state.update(
                    {
                        "status": "ready",
                        "calculator_link": calculator_link,
                        "error": link_error,
                        "url": f"/api/duel/render?turn={int(turn)}&v={int(token)}",
                    }
                )
            else:
                state.update({"status": "failed", "error": error, "url": None})

    def render_state(self) -> dict[str, Any]:
        with self._render_lock:
            if not self._render_states:
                return {
                    "generation": int(self._render_game_token),
                    "status": "idle",
                    "turn": None,
                    "rows": 0,
                    "calculator_link": None,
                    "error": None,
                    "url": None,
                }
            state = copy.deepcopy(self._render_states[max(self._render_states)])
            state["generation"] = int(self._render_game_token)
            return state

    def render_image_bytes(
        self, turn: int | None = None, *, token: int | None = None
    ) -> bytes:
        with self._render_lock:
            requested_token = (
                int(token) if token is not None else int(self._render_game_token)
            )
            chosen = int(turn) if turn is not None else (
                max(self._render_states) if self._render_states else -1
            )
            image = self._render_images.get((requested_token, chosen))
            if image is None:
                state = (
                    self._render_states.get(chosen)
                    if requested_token == int(self._render_game_token)
                    else None
                ) or {}
                raise FileNotFoundError(
                    "duel_render_not_ready:"
                    f"{requested_token}:{chosen}:{state.get('status', 'missing')}"
                )
            return bytes(image)

    # -- read-only views -------------------------------------------------
    def _no_game_envelope(self) -> dict[str, Any]:
        return {"ok": False, "error": "no_duel_game", "transition": None, "state": self.snapshot()}

    def archive_list(self, query: str = "", limit: int = 100) -> list[dict[str, Any]]:
        if self.archive is None:
            return []
        self._request_archive_recovery_scan()
        return self.archive.list_games(query, limit)

    def archive_game(self, game_id: str) -> dict[str, Any]:
        if self.archive is None:
            raise FileNotFoundError("duel_archive_disabled")
        self._request_archive_recovery_scan()
        return self.archive.load_game(game_id)
    def archive_public_game(self, game_id: str) -> dict[str, Any]:
        """Completed-only browser view; archive_game remains the raw audit read."""
        if self.archive is None:
            raise FileNotFoundError("duel_archive_disabled")
        self._request_archive_recovery_scan()
        return build_public_replay(self.archive.load_game(game_id))


    def archive_value_rows(self, game_id: str) -> dict[str, Any]:
        """The show-value log for an archived game, or an honest absence.

        Absence is not an error. Only games played after show-value shipped have
        a log, and nothing back-fills one, so `available: false` with a reason
        is the ordinary answer for an older game."""
        if self.archive is None:
            return {"available": False, "rows": [], "reason": "duel_archive_disabled"}
        return read_value_log(self.archive.root, game_id)

    def archive_image(self, game_id: str, turn: int) -> bytes:
        if self.archive is None:
            raise FileNotFoundError("duel_archive_disabled")
        self._request_archive_recovery_scan()
        return self.archive.image_bytes(game_id, turn)

    def status(self) -> dict[str, Any]:
        """Status."""
        game_status_token = int(self._game_status_token)
        worker = self.worker
        session = self.session
        payload: dict[str, Any] = {
            "ok": True,
            "game_generation": game_status_token,
            "active": bool(session is not None),
            "worker": worker.status() if worker is not None else {"status": "idle"},
            "gear": self.gear_config(),
            "agent_error": self.agent_error,
            "render": self.render_state(),
            "archive": {
                "enabled": self.archive is not None,
                "game_id": self.archive_game_id,
                "error": self.archive_error,
            },
        }
        if session is not None:
            payload.update(
                {
                    "turn": int(session.turn),
                    "done": bool(session.done),
                    "winner": session.winner,
                    "end_reason": self._forced_end or session.end_reason,
                    "human_lives": int(session.human.lives),
                    "ai_lives": int(session.ai.lives),
                }
            )
            last = self.turn_records[-1] if self.turn_records else None
            if last is not None:
                payload["last_turn"] = {
                    "turn": last["turn"],
                    "searched_segments": last["searched_segments"],
                    "greedy_segments": last["greedy_segments"],
                    "n_segments": last["n_segments"],
                    "banner": last["banner"],
                }
        return payload

    def _agent_description(self) -> dict[str, Any] | None:
        """Agent description."""
        describe = getattr(self.agent, "describe", None)
        if not callable(describe):
            return None
        try:
            return describe()
        except Exception as exc:
            return {"error": f"describe_failed:{type(exc).__name__}:{exc}"}

    def _live_turn_record(
        self, record: dict[str, Any], dropped: set[str] | None = None
    ) -> dict[str, Any]:
        """Return only fields consumed by the live /play page.

        Full per-action transition histories remain in self.turn_records and
        the durable archive. They are absent here because every mutation
        response carries snapshot() back to the browser.

        Three layers, because the record nests and the whitelist only sees its
        top: the name list here, `_live_segment_record`'s scalars-only rule for
        every segment, and `_bounded_live_value`'s size bound for every OTHER
        field -- so putting something fat under `ai_board` or `wins` is the
        same bug moved one key over, and is refused the same way.

        Segments are never deep-copied whole: the copy of half a megabyte of
        per-chunk timings, on every one of a shop click's snapshots, was a real
        part of the ~1.0 s this used to spend.

        Anything dropped is added to `dropped` so the caller can count it. That
        set is what makes the guard visible in production instead of only in a
        test: see `DuelApp.payload_guard`.
        """
        if dropped is None:
            dropped = set()
        live: dict[str, Any] = {}
        for key in LIVE_TURN_FIELDS:
            if key not in record:
                continue
            value = record[key]
            if key == "segments":
                live[key] = [
                    _live_segment_record(segment, dropped)
                    if isinstance(segment, dict)
                    else copy.deepcopy(segment)
                    for segment in (value or [])
                ]
                continue
            keep, projected = _bounded_live_value(key, value, dropped)
            if keep:
                live[key] = projected
        return live

    def payload_guard(self) -> dict[str, Any]:
        """What the live-payload bounds have actually refused, this game."""
        return {
            "dropped": sorted(self._payload_dropped),
            "over_ceiling": int(self._payload_over_ceiling),
            "max_turn_record_bytes": int(self._payload_worst_record),
            "limit": MAX_LIVE_TURN_RECORD_BYTES,
        }

    def _note_payload_guard(self, dropped: set[str], live_turns: list[dict[str, Any]]) -> None:
        """Count what the projection refused, and shout the first time.

        The newest record is the only one measured per snapshot: every record
        is projected by the same rule, so a regression shows up on the newest
        one within a turn, and measuring all thirty of them on every shop click
        would put back a slice of the cost this whole change removed.
        """
        for key in sorted(dropped):
            self._payload_dropped.add(key)
            if key in _ANNOUNCED_PAYLOAD_DROPS:
                continue
            _ANNOUNCED_PAYLOAD_DROPS.add(key)
            print(
                f"[duel] live payload guard dropped {key!r} from the browser "
                f"snapshot (it stays in the archive); see "
                f"duel_app.py::_live_segment_record",
                flush=True,
            )
        if not live_turns:
            return
        newest = _wire_bytes(live_turns[-1])
        self._payload_worst_record = max(self._payload_worst_record, newest)
        if newest > MAX_LIVE_TURN_RECORD_BYTES:
            self._payload_over_ceiling += 1
            print(
                f"[duel] live payload guard: turn record "
                f"{live_turns[-1].get('turn')} is {newest} bytes, over the "
                f"{MAX_LIVE_TURN_RECORD_BYTES} ceiling",
                flush=True,
            )

    def value_block(self) -> dict[str, Any]:
        """What the page needs to draw the readout, all of it scalar.

        `enabled` is what the page branches on, so the feature switching off
        removes the panel rather than leaving an empty one; `rows` is the
        predicted-versus-realised table, one row per resolved turn.
        """
        log = self.value_log
        return {
            "enabled": bool(self.show_value),
            "error": self.value_probe_error,
            "action_cap": int(self.value_action_cap),
            "log": log.describe() if log is not None else None,
            "rows": log.live_rows() if log is not None else [],
        }

    def duel_block(self) -> dict[str, Any]:
        session = self.session
        dropped: set[str] = set()
        live_turns = [self._live_turn_record(row, dropped) for row in self.turn_records]
        self._note_payload_guard(dropped, live_turns)
        block: dict[str, Any] = {
            "active": bool(session is not None),
            "game_generation": int(self._game_status_token),
            "gear": self.gear_config(),
            "agent_error": self.agent_error,
            "agent": self._agent_description(),
            "seeds": dict(self.seeds),
            "games_played": int(self._games_played),
            "turns": live_turns,
            "payload_guard": self.payload_guard(),
            "value": self.value_block(),
            "render": self.render_state(),
            "archive": {
                "enabled": self.archive is not None,
                "game_id": self.archive_game_id,
                "error": self.archive_error,
            },
        }
        if session is None:
            return block
        done = bool(session.done or self._forced_end)
        block.update(
            {
                "turn": int(session.turn),
                "done": done,
                "winner": session.winner,
                "end_reason": self._forced_end or session.end_reason,
                "rules": {
                    "name": session.rules.name,
                    "start_lives": int(session.rules.start_lives),
                    "turn_cap": int(session.rules.turn_cap),
                },
                HUMAN: {
                    "lives": int(session.human.lives),
                    "wins": int(session.human.wins),
                    "losses": int(session.human.losses),
                    "draws": int(session.human.draws),
                },
                AI: {
                    "lives": int(session.ai.lives),
                    "wins": int(session.ai.wins),
                    "losses": int(session.ai.losses),
                    "draws": int(session.ai.draws),
                },


                "final_human_board": copy.deepcopy(session.human.state.get("team") or []),
            }
        )
        return block

    def snapshot(self) -> dict[str, Any]:
        """`App._snapshot`'s shape for the HUMAN's side, plus the `duel` block."""
        session = self.session
        if session is None:
            return {
                "state": None,
                "catalog_url": self.catalog_url,
                "legal_actions": [],
                "history": [],
                "last_battle": None,
                "game_mode": "versus",
                "opponent_source_mode": "duel",
                "predictor_loaded": False,
                "value_model_loaded": False,
                "duel": self.duel_block(),
            }
        state = session.human.state
        validate_state(state)
        actions = legal_actions(state)
        return {
            "state": state,
            "catalog_url": self.catalog_url,
            "legal_actions": [
                {"index": i, "action": action, "label": json.dumps(action, separators=(",", ":"))}
                for i, action in enumerate(actions)
            ],
            "history": client_history_entries(
                session.human.history, session.human.history_group_ids
            ),
            "last_battle": None,
            "game_mode": "versus",
            "opponent_source_mode": "duel",
            "predictor_loaded": False,
            "value_model_loaded": False,
            "duel": self.duel_block(),
        }
