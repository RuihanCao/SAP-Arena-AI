"""exp16 AI turn execution under one fixed wall-clock budget.

WHAT IT IS FOR. In a duel the AI starts thinking at the TOP OF THE HUMAN'S
TURN and keeps thinking while the human shops. When the human presses
"End turn" the browser waits for the same AI turn that was already running.
Human timing never changes search strength. The only normal stop is the
turn's own deadline, checked between complete inference chunks.

ONE THREAD, ALWAYS. `SearchRecommender` carries mutable per-call state
(`_recommend_call_count`, `_decision_turn`, `_state_seed`, `_race_wins`, the
imagination context), so two overlapping `recommend` calls would interleave
their CRN keys and their race scalars. The executor is therefore
`max_workers=1` and every agent call in this process goes through it.

CHUNKS. There is no interruption point inside a chunk. Far from a deadline
the scheduler batches up to 16 candidate chains to amortize V inference.
Near a deadline it shrinks through 8, 4, 2, and 1 using measured recent
cost. A segment may exceed its allocation by one completed in-flight chunk;
that debt is automatically charged to the remaining global clock.

THE TWO GEARS:

- `measured`: 72 candidates per segment. This is the width exp12's numbers
  were produced at, so "how strong is it" has a published answer.
- `full-clock`: a configurable 105 s per-turn default split across segments,
  each slice = remaining_time / expected_remaining_segments, seeded from
  Structural A2's measured mean of approximately 2.93. Width 4096 is only an
  emergency guard; the deadline is the normal stop.

FAILURE IS LOUD, NEVER QUIETLY WEAKER. Any failure -- decode, a divergence,
an exception -- returns `ok=False` with a code and NO actions, so the caller
plays the "AI stands still this turn" policy and shows
`AI failed turn N: <code>`. The partially committed ops are still reported
(`partial_actions`) for diagnosis but are deliberately not the turn.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..honest_frame import MAX_SEGMENTS_PER_TURN
from ..search_recommender import DEFAULT_ANYTIME_CHUNK
from ..segmented_turn import run_segmented_turn

STATUS_IDLE = "idle"
STATUS_THINKING = "thinking"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

GEAR_MEASURED = "measured"
GEAR_FULL_CLOCK = "full-clock"
#: exp16 W11b. Pin the OUTER width like `measured`, then spend whatever is left
#: of the segment's slice on more imagined samples (k) instead of more
#: candidates. The two existing gears both trade the clock for WIDTH; nothing
#: spent it on resolution. Level-synchronous, so every group always holds the
#: same number of samples -- `search_recommender._search_anytime` owns that and
#: says why.
GEAR_RESAMPLE_CLOCK = "resample-clock"
GEARS = (GEAR_MEASURED, GEAR_FULL_CLOCK, GEAR_RESAMPLE_CLOCK)
#: Gears that run on the turn clock: the segment gets a slice, and the search
#: stops when that slice is spent. `measured` is not one of them because its
#: WIDTH is its budget and `max_candidates` ends the phase on its own.
#:
#: `resample-clock` HAS to be here and was not, which is the whole reason this
#: tuple exists rather than an `== GEAR_FULL_CLOCK` test at each site. Its
#: design is "pin the width, then spend the remainder of the slice on k"; with
#: no slice there is no remainder, nothing ever asked it to stop, and the extra
#: sample loop ran to its 4096 safety valve. Measured on a real game before the
#: fix: 69 seconds of thinking against an 8-second turn budget, with a human
#: already waiting, and zero segments archived.
CLOCKED_GEARS = (GEAR_FULL_CLOCK, GEAR_RESAMPLE_CLOCK)
#: Gears whose per-segment width is NOT pinned by the config, so "the search
#: reached the requested width" means the safety cap bound rather than the
#: setting being honoured.
FLOATING_WIDTH_GEARS = (GEAR_FULL_CLOCK,)
#: Ceiling on a width the SETTINGS CARD may pin, and deliberately the same
#: number as that input's own `max`, so the page and the server cannot disagree.
#: Without it `resample-clock` had no finite wall clock at all: its width is a
#: commitment, so `should_stop()` waits for it, and the card's JS validated only
#: "a whole number, 1 or more". Typing seven digits was enough to occupy the one
#: inference thread indefinitely, and `await_turn`'s timeout returns to the
#: caller without stopping the worker. (codex review, 2026-08-21.)
MAX_SETTABLE_SEARCH_WIDTH = 4096

# Structural A2 measured 2.93 segments per turn. This is only the initial
# allocator prior; every real segment recomputes from the remaining clock.
DEFAULT_MEAN_SEGMENTS = 3.0
# exp16 Wave 4. `mean_segments` is a PRIOR, and the Wave 3 archive shows it
# is wrong on 31% of turns (n_segments 1:1, 2:15, 3:6, 4:10 over 32 turns).
# So a segment never spends the whole remaining clock: it holds back this
# fraction of the turn budget, which keeps the next segment's slice
# positive however many segments actually arrive.
DEFAULT_TAIL_RESERVE_FRAC = 0.10
# Past the clock the fallback is one greedy BC chain, not a bare END_TURN.
# Bound how many of those a single turn may take, so a pathological chain
# cannot walk the turn out to `max_segments` at ~50 ms each.
DEFAULT_MAX_DEADLINE_GREEDY = 4
# Wave 4 review: `turn_budget_s` reached `AiTurnConfig` straight off an
# HTTP payload with no upper bound, so a POST of 100000 bought a 27-hour
# AI turn with no backstop. The UI already advertises 1..600; the server
# now enforces it instead of trusting the input element.
MIN_TURN_BUDGET_S = 0.01
MAX_TURN_BUDGET_S = 600.0
DEFAULT_TURN_BUDGET_S = 105.0
# A SAFETY VALVE, NOT A BUDGET, and 4096 stopped being one.
# `PLAN_A2_FIXED_CLOCK_8766.md` introduced it as "a high safety-only cap,
# initially 4096", with the deadline as the normal stop. `RESULTS_A3.md`
# (2026-08-19) then measured it binding on 16 of 16 duel segments at a median
# 19.8 s inside a 35 s segment budget, which is 43% of the segment clock left
# unused because the valve, not the clock, was deciding how wide the search got.
#
# 32768 restores the intent. The fastest rate that round measured is about 277
# candidates per second, and the longest a single segment can run is the whole
# 105 s turn budget, so the most a segment can physically generate is around
# 29,000. A cap above that is unreachable in normal play, which is what a safety
# valve is for. It still fires on a runaway, and `stop_reason` still records it,
# so if it ever binds again that shows up as a reading rather than as silence.
DEFAULT_FULL_CLOCK_MAX_WIDTH = 32768
DEFAULT_ADAPTIVE_CHUNK_SIZES = (1, 2, 4, 8, 16)
DEFAULT_INITIAL_CANDIDATE_S = 0.125
DEFAULT_CHUNK_EWMA_ALPHA = 0.25
DEFAULT_DEADLINE_SAFETY_S = 0.05
AI_TIMEOUT_GRACE_S = 30.0


class _StaleTurn(Exception):
    """Raised inside the worker thread when the turn it is playing has been
    fenced off by `bump_generation` (a reset or an undo). Unwinds at the next
    segment boundary; the result is discarded.
    """


@dataclass
class AiTurnConfig:
    chunk_size: int = DEFAULT_ANYTIME_CHUNK
    # Adaptive batching is an experiment arm until the accepted equal-clock
    # benchmark proves a throughput gain. Fixed chunk=1 is the safe default.
    adaptive_chunks: bool = False
    adaptive_chunk_sizes: tuple[int, ...] = DEFAULT_ADAPTIVE_CHUNK_SIZES
    initial_candidate_s: float = DEFAULT_INITIAL_CANDIDATE_S
    chunk_ewma_alpha: float = DEFAULT_CHUNK_EWMA_ALPHA
    deadline_safety_s: float = DEFAULT_DEADLINE_SAFETY_S
    gear: str = GEAR_FULL_CLOCK
    width: int = 72
    turn_budget_s: float = DEFAULT_TURN_BUDGET_S
    mean_segments: float = DEFAULT_MEAN_SEGMENTS
    tail_reserve_frac: float = DEFAULT_TAIL_RESERVE_FRAC
    max_deadline_greedy: int = DEFAULT_MAX_DEADLINE_GREEDY
    full_clock_max_width: int = DEFAULT_FULL_CLOCK_MAX_WIDTH
    # Kept in new archive/config payloads as a compatibility assertion. The
    # old False policy no longer exists and is rejected at construction.
    finish_turn: bool = True
    # Deterministic test control: stop after this many complete chunks across
    # the turn. This replaces the wall clock in replay-equivalence tests, so
    # the stopping point is a pure function of the seeds. None means the real
    # fixed turn clock decides.
    chunk_budget: int | None = None
    max_segments: int = MAX_SEGMENTS_PER_TURN

    def __post_init__(self) -> None:
        if self.gear not in GEARS:
            raise ValueError(f"ai_worker_bad_gear:{self.gear!r}:expected_one_of={list(GEARS)}")
        if int(self.chunk_size) < 1:
            raise ValueError(f"ai_worker_bad_chunk_size:{self.chunk_size!r}")
        if not bool(self.finish_turn):
            raise ValueError("ai_worker_human_stop_policy_removed")
        sizes = tuple(int(value) for value in self.adaptive_chunk_sizes)
        if not sizes or sizes[0] != 1 or any(value < 1 for value in sizes):
            raise ValueError(f"ai_worker_bad_adaptive_chunk_sizes:{sizes!r}")
        if tuple(sorted(set(sizes))) != sizes:
            raise ValueError(f"ai_worker_bad_adaptive_chunk_sizes:{sizes!r}")
        if not math.isfinite(float(self.turn_budget_s)) or float(self.turn_budget_s) <= 0.0:
            raise ValueError(f"ai_worker_bad_turn_budget_s:{self.turn_budget_s!r}")
        if not math.isfinite(float(self.mean_segments)) or float(self.mean_segments) <= 0.0:
            raise ValueError(f"ai_worker_bad_mean_segments:{self.mean_segments!r}")
        if (
            not math.isfinite(float(self.initial_candidate_s))
            or float(self.initial_candidate_s) <= 0.0
        ):
            raise ValueError(
                f"ai_worker_bad_initial_candidate_s:{self.initial_candidate_s!r}"
            )
        if (
            not math.isfinite(float(self.chunk_ewma_alpha))
            or not 0.0 <= float(self.chunk_ewma_alpha) <= 1.0
        ):
            raise ValueError(
                f"ai_worker_bad_chunk_ewma_alpha:{self.chunk_ewma_alpha!r}"
            )
        if (
            not math.isfinite(float(self.deadline_safety_s))
            or float(self.deadline_safety_s) < 0.0
        ):
            raise ValueError(
                f"ai_worker_bad_deadline_safety_s:{self.deadline_safety_s!r}"
            )
        budget = float(self.turn_budget_s)
        if (
            not math.isfinite(budget)
            or not MIN_TURN_BUDGET_S <= budget <= MAX_TURN_BUDGET_S
        ):
            raise ValueError(f"ai_worker_bad_turn_budget_s:{self.turn_budget_s!r}")
        if not 0.0 <= float(self.tail_reserve_frac) < 1.0:
            raise ValueError(
                f"ai_worker_bad_tail_reserve_frac:{self.tail_reserve_frac!r}"
            )
        if int(self.max_deadline_greedy) < 0:
            raise ValueError(
                f"ai_worker_bad_max_deadline_greedy:{self.max_deadline_greedy!r}"
            )
        if int(self.full_clock_max_width) < 1:
            raise ValueError(
                f"ai_worker_bad_full_clock_max_width:{self.full_clock_max_width!r}"
            )


@dataclass
class _Progress:
    gen: int = 0
    turn: int = 0
    segment: int = 0
    width: int = 0
    chunks: int = 0
    searched_segments: int = 0
    deadline_segments: int = 0
    greedy_segments: int = 0
    #: The other two knobs, so the badge can show all three while it thinks.
    #: `inner_width` is config and known from the start; `realised_samples`
    #: only moves under `resample-clock` and is None until the first level
    #: commits, because a gear that has added nothing yet has not added zero --
    #: it has not reported.
    inner_width: int | None = None
    realised_samples: int | None = None
    started_at: float | None = None


class AiTurnWorker:
    """Plays the AI's whole turn through exp13's segmented-honest path."""

    def __init__(
        self,
        agent: Any,
        config: AiTurnConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.agent = agent
        self.config = config or AiTurnConfig()
        self._clock = clock
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sap-ai")
        self._lock = threading.Lock()
        self._generation = 0
        self._status = STATUS_IDLE
        self._future: Future | None = None
        self._future_gen: int | None = None
        self._future_config: AiTurnConfig | None = None
        self._result: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._stop_at: float | None = None
        self._progress = _Progress()

    # -- lifecycle -------------------------------------------------------
    def shutdown(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "AiTurnWorker":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.shutdown()

    # -- control ---------------------------------------------------------
    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def bump_generation(self) -> int:
        """The reset/undo fence: everything in flight becomes stale.

        A running future is NOT killed -- there is no interruption point
        inside a search chunk. It unwinds at its next segment boundary
        (`_StaleTurn`) and whatever it produces is discarded, so the worst
        case is that one chunk of work is wasted, never that a stale turn is
        played.
        """
        with self._lock:
            self._generation += 1
            self._status = STATUS_IDLE
            self._result = None
            self._future_config = None
            self._progress = _Progress(gen=self._generation)
            self._stop_at = None
            gen = self._generation
        self._stop.set()
        return gen

    def note_human_end_turn(self) -> None:
        """Record when the browser began waiting without changing search."""
        with self._lock:
            if self._stop_at is None:
                self._stop_at = self._clock()

    def request_stop(self) -> None:
        """Compatibility alias. Human End Turn no longer stops inference."""
        self.note_human_end_turn()

    def submit(
        self,
        state: dict[str, Any],
        *,
        gen: int,
        turn: int,
        wins: int,
        engine_seed: int | None = None,
    ) -> Future:
        """Start thinking about `state` (the AI's board at the top of the
        human's turn). `gen` fences the result: `await_turn` refuses a result
        whose generation is no longer current.

        `engine_seed` is A1 ruling 1's imagination key base: the GAME's own
        engine seed, which in a duel is the side's `DuelSession` seed. It is
        passed rather than read off `meta.seed`, because by turn N the play
        stream has chained that forward and keying imagination on it would
        key it on the play stream's POSITION -- the exact coupling A1
        removes. Falls back to `meta.seed` only when a caller has no seed to
        give (a fixture, a unit test).
        """
        payload = copy.deepcopy(state)
        seed = int(engine_seed) if engine_seed is not None else _engine_seed(payload)
        submitted_at = self._clock()
        cfg = copy.deepcopy(self.config)
        with self._lock:
            self._generation = int(gen)
            self._status = STATUS_THINKING
            self._result = None
            self._stop_at = None
            self._progress = _Progress(
                gen=int(gen), turn=int(turn), started_at=submitted_at
            )
            self._stop.clear()
            future = self._pool.submit(
                self._play_turn,
                payload,
                int(gen),
                int(turn),
                int(wins),
                seed,
                submitted_at,
                cfg,
            )
            self._future = future
            self._future_gen = int(gen)
            self._future_config = cfg
        return future

    def status(self) -> dict[str, Any]:
        with self._lock:
            progress = self._progress
            elapsed_ms = (
                int((self._clock() - progress.started_at) * 1000)
                if progress.started_at is not None
                else 0
            )
            return {
                "status": self._status,
                "gen": progress.gen,
                "turn": progress.turn,
                "segment": progress.segment,
                "width": progress.width,
                "inner_width": progress.inner_width,
                "realised_samples": progress.realised_samples,
                "chunks": progress.chunks,
                "searched_segments": progress.searched_segments,
                "deadline_segments": progress.deadline_segments,
                "greedy_segments": progress.greedy_segments,
                "elapsed_ms": elapsed_ms,
                "human_waiting": self._stop_at is not None,
            }

    def await_turn(self, gen: int, timeout: float | None = None) -> dict[str, Any]:
        """Block for the turn submitted under `gen` and return its result."""
        with self._lock:
            future = self._future
            future_gen = self._future_gen
            cfg = self._future_config
            turn = self._progress.turn
        if future is None or future_gen != int(gen):
            return _failure(gen=int(gen), turn=0, code="ai_no_turn_submitted", cfg=cfg)
        effective_timeout = timeout
        if cfg is not None and cfg.gear in CLOCKED_GEARS:
            fixed_clock_timeout = float(cfg.turn_budget_s) + AI_TIMEOUT_GRACE_S
            effective_timeout = max(float(timeout or 0.0), fixed_clock_timeout)
        try:
            result = future.result(timeout=effective_timeout)
        except TimeoutError:
            return _failure(gen=int(gen), turn=turn, code="ai_timeout", cfg=cfg)
        except Exception as exc:  # the worker body already catches its own
            return _failure(
                gen=int(gen), turn=turn, code=f"ai_worker_crashed:{exc}", cfg=cfg
            )
        if int(self.generation) != int(gen):
            return _failure(gen=int(gen), turn=turn, code="ai_stale_generation", cfg=cfg)
        if int(result.get("gen", -1)) != int(gen):
            return _failure(gen=int(gen), turn=turn, code="ai_stale_generation", cfg=cfg)
        return result

    # -- the turn --------------------------------------------------------
    @staticmethod
    def _segment_width(
        segment_index: int, turn_started: float, cfg: AiTurnConfig
    ) -> int:
        if cfg.gear in (GEAR_MEASURED, GEAR_RESAMPLE_CLOCK):
            return int(cfg.width)
        return int(cfg.full_clock_max_width)

    def _play_turn(
        self,
        state: dict[str, Any],
        gen: int,
        turn: int,
        wins: int,
        engine_seed: int,
        submitted_at: float,
        cfg: AiTurnConfig,
    ) -> dict[str, Any]:
        agent = self.agent
        started = float(submitted_at)
        global_deadline = started + float(cfg.turn_budget_s)
        segment_stats: list[dict[str, Any]] = []
        turn_chunks = {"n": 0}
        deadline_greedy = {"n": 0}
        cost_ewma = {"candidate_s": float(cfg.initial_candidate_s)}
        reasons_by_segment: dict[int, list[str | None]] = {}
        structural_by_segment: dict[int, list[bool]] = {}
        deadline_holder: dict[str, float | None] = {
            "segment": None,
            "segment_budget_s": None,
        }

        def _committed(segment_index: int, _op: dict[str, Any], transition: dict[str, Any]) -> None:
            reason = transition.get("stochastic_reason")
            structural_by_segment.setdefault(segment_index, []).append(
                bool(transition.get("stochastic_structural"))
            )
            reasons_by_segment.setdefault(segment_index, []).append(
                str(reason) if reason else None
            )

        def _decide(decision_state: dict[str, Any], segment_index: int) -> dict[str, Any]:
            if int(self.generation) != int(gen):
                raise _StaleTurn(f"stale:{gen}")
            # Pin candidate sampling to THIS decision (see
            # `SearchRecommender.set_candidate_stream`): without it, replaying
            # a game in the same process draws different candidate chains and
            # diverges at turn 1.
            set_stream = getattr(agent.search, "set_candidate_stream", None)
            if callable(set_stream):
                set_stream(
                    candidate_stream_key(
                        engine_seed=int(engine_seed),
                        turn=int(decision_state.get("turn", turn) or turn),
                        segment_index=int(segment_index),
                    )
                )
            seg_started = self._clock()
            chunk_counter = {"n": 0}
            chunk_exhausted = (
                cfg.chunk_budget is not None
                and turn_chunks["n"] >= int(cfg.chunk_budget)
            )
            clock_exhausted = (
                cfg.gear in CLOCKED_GEARS and seg_started >= global_deadline
            )
            budget_exhausted = clock_exhausted or chunk_exhausted

            with self._lock:
                self._progress.segment = segment_index
                self._progress.chunks = 0
                # Config, so it is known before any work happens. Read off the
                # agent rather than the turn config: the inner width lives on
                # the recommender, not on `AiTurnConfig`.
                self._progress.inner_width = getattr(
                    getattr(agent, "search", None), "completion_width", None
                )
                self._progress.realised_samples = None

            if budget_exhausted:
                # Do not start fresh inference after the global clock. A
                # structural outcome can invalidate the chain selected by the
                # preceding segment, so nothing here may search.
                #
                # exp16 Wave 4: this used to return a bare END_TURN at width 0,
                # which is strictly worse play than the behaviour clone's own
                # move. A turn that ran out of clock stopped playing while
                # holding gold and an open slot, having just paid for a roll.
                # One greedy BC chain costs ~30-66 ms and is bounded, so that
                # is the fallback now. `max_deadline_greedy` bounds how many a
                # turn may take. The chunk-budget path is a deterministic test
                # control and keeps the old END_TURN behaviour, so replay
                # equivalence stays a pure function of the seeds.
                use_greedy = (
                    clock_exhausted
                    and not chunk_exhausted
                    and deadline_greedy["n"] < int(cfg.max_deadline_greedy)
                )
                rec = None
                if use_greedy:
                    with self._lock:
                        self._progress.width = 1
                    try:
                        rec = agent.search._call_bc_recommend(
                            decision_state,
                            deterministic=True,
                            force_ranked_decode=True,
                        )
                    except Exception:
                        rec = None
                    if rec is not None and not rec.get("ok"):
                        rec = None
                if rec is None:
                    use_greedy = False
                    with self._lock:
                        self._progress.width = 0
                    rec = {
                        "ok": True,
                        "error": None,
                        "recommended_action": {"type": "END_TURN"},
                        "chain_preview": [{"type": "END_TURN"}],
                        "search_used": False,
                        "diagnostics": {"stop_reason": "turn_deadline"},
                    }
                else:
                    deadline_greedy["n"] += 1
                segment_stats.append(
                    {
                        "index": segment_index,
                        "mode": "deadline-greedy" if use_greedy else "deadline-end-turn",
                        "width": 1 if use_greedy else 0,
                        "width_requested": 0,
                        "chunks": 0,
                        "ms": int((self._clock() - seg_started) * 1000),
                        "unique_prefixes": 0,
                        "dedup_ratio": None,
                        "leaves_evaluated": 0,
                        "chunk_sizes": [],
                        "stage_seconds": {},
                        "chunk_timings": [],
                        "segment_budget_s": None,
                        "deadline_overshoot_ms": 0,
                        "stop_reason": (
                            "chunk_budget"
                            if cfg.chunk_budget is not None
                            and turn_chunks["n"] >= int(cfg.chunk_budget)
                            else "turn_deadline"
                        ),
                    }
                )
                with self._lock:
                    self._progress.deadline_segments += 1
                return rec

            width = self._segment_width(segment_index, started, cfg)
            if cfg.gear in CLOCKED_GEARS:
                remaining_s = max(0.0, global_deadline - seg_started)
                # exp16 Wave 4. `max(1.0, ...)` made the segment at
                # `mean_segments - 1` the notional last one and handed it
                # the entire remaining clock, so a turn that then hit one
                # more structural boundary played NOTHING: 10 of 10
                # four-segment turns in the Wave 3 archive committed zero
                # ops in their final segment, after paying for the roll
                # that created it. The prior still shapes the split, but a
                # reserve now bounds it, so the next segment's slice is
                # positive by construction.
                expected_remaining = max(1.0, float(cfg.mean_segments) - float(segment_index))
                reserve_s = min(
                    remaining_s * 0.5,
                    float(cfg.turn_budget_s) * float(cfg.tail_reserve_frac),
                )
                segment_budget_s = min(
                    remaining_s / expected_remaining,
                    max(0.0, remaining_s - reserve_s),
                )
                deadline_holder["segment_budget_s"] = segment_budget_s
                deadline_holder["segment"] = min(
                    global_deadline,
                    seg_started + segment_budget_s,
                )
            else:
                deadline_holder["segment"] = None
                deadline_holder["segment_budget_s"] = None
            with self._lock:
                self._progress.width = 0

            # What the width phase has generated so far, for the gear whose
            # width is a commitment rather than a budget. A plain holder rather
            # than `self._progress.width`, which is behind the lock and is a
            # reporting field.
            generated_so_far = {"n": 0}

            def _should_stop() -> bool:
                if int(self.generation) != int(gen):
                    return True
                if cfg.chunk_budget is not None:
                    return turn_chunks["n"] >= int(cfg.chunk_budget)
                # `resample-clock` pins the width and spends what is LEFT of
                # the slice on k, so the clock must not speak until the width
                # is in hand. Without this the slice cut the width phase short
                # and the gear recorded 2, 3 and 4 against a setting of 4 in
                # one game -- which is the comparability that the pinned-width
                # gears exist to provide, quietly gone. A segment may therefore
                # overrun its slice while generating its width; `measured` has
                # always done exactly that, with no clock at all.
                if (cfg.gear == GEAR_RESAMPLE_CLOCK
                        and int(generated_so_far["n"]) < int(width)
                        # ...but the TURN's own clock is still a hard backstop.
                        # Suspending the per-SEGMENT slice is the gear's promise
                        # ("the remainder of the slice goes to k"); suspending
                        # the turn budget as well would mean the AI can keep a
                        # human waiting without bound, which is worse than a
                        # width the turn budget truncated -- and the truncation
                        # is recorded when it happens.
                        and self._clock() < global_deadline):
                    return False
                if cfg.gear in CLOCKED_GEARS and self._clock() >= global_deadline:
                    return True
                deadline = deadline_holder["segment"]
                return deadline is not None and self._clock() >= deadline

            def _next_chunk_size(_info: dict[str, Any]) -> int:
                if not cfg.adaptive_chunks or cfg.gear != GEAR_FULL_CLOCK:
                    return int(cfg.chunk_size)
                deadline = deadline_holder["segment"]
                if deadline is None:
                    return int(cfg.chunk_size)
                remaining_s = max(0.0, float(deadline) - self._clock())
                per_candidate_s = max(1e-6, float(cost_ewma["candidate_s"]))
                safety_s = max(float(cfg.deadline_safety_s), per_candidate_s)
                usable_s = max(0.0, remaining_s - safety_s)
                selected = 1
                for size in cfg.adaptive_chunk_sizes:
                    if per_candidate_s * int(size) <= usable_s:
                        selected = int(size)
                return selected

            def _on_chunk(info: dict[str, Any]) -> None:
                chunk_counter["n"] = int(info.get("chunk", 0))
                turn_chunks["n"] += 1
                actual_size = max(1, int(info.get("size", 1)))
                actual_s = float(info.get("total_s", 0.0) or 0.0)
                if actual_s > 0.0:
                    observed = actual_s / float(actual_size)
                    alpha = min(1.0, max(0.0, float(cfg.chunk_ewma_alpha)))
                    cost_ewma["candidate_s"] = (
                        alpha * observed
                        + (1.0 - alpha) * float(cost_ewma["candidate_s"])
                    )
                generated_so_far["n"] = int(info.get("n_generated", 0))
                with self._lock:
                    self._progress.chunks = chunk_counter["n"]
                    self._progress.width = int(info.get("n_generated", 0))

            def _on_level(info: dict[str, Any]) -> None:
                with self._lock:
                    self._progress.realised_samples = int(
                        info.get("realised_stochastic_samples") or 0
                    )

            rec = agent.search.recommend_anytime(
                decision_state,
                chunk_size=int(cfg.chunk_size),
                should_stop=_should_stop,
                max_candidates=int(width),
                on_chunk=_on_chunk,
                on_level=_on_level,
                extra_samples_until_stop=(cfg.gear == GEAR_RESAMPLE_CLOCK),
                next_chunk_size=(
                    _next_chunk_size
                    if cfg.adaptive_chunks and cfg.gear == GEAR_FULL_CLOCK
                    else None
                ),
            )
            anytime = rec.get("search_anytime") or {}
            seg_finished = self._clock()
            segment_deadline = deadline_holder["segment"]
            searched_width = int(
                anytime.get("width_searched")
                or rec.get("search_n_generated")
                or 1
            )
            stopped = bool(anytime.get("stopped"))
            if cfg.chunk_budget is not None and turn_chunks["n"] >= int(cfg.chunk_budget):
                stop_reason = "chunk_budget"
            elif cfg.gear in CLOCKED_GEARS and seg_finished >= global_deadline:
                stop_reason = "turn_deadline"
            elif segment_deadline is not None and seg_finished >= segment_deadline:
                stop_reason = "segment_deadline"
            elif searched_width >= int(width):
                # Reaching a PINNED width is the setting being honoured, not
                # the safety cap binding, and only the gear whose width floats
                # can hit that cap. The label keeps its historical name so no
                # consumer learns a new value; it means "stopped because the
                # configured width was reached", which is true of both pinned
                # gears.
                stop_reason = (
                    "safety_cap" if cfg.gear in FLOATING_WIDTH_GEARS
                    else "measured_width"
                )
            elif stopped:
                stop_reason = "deadline"
            else:
                stop_reason = "search_complete"
            # Recorded independently of the label above: the cap and a deadline
            # can both be true of the same segment, and the label can only name
            # one of them. Wave 4 found a cap hit at or after the deadline
            # instant was labelled a deadline stop with safety_cap_hit False,
            # so the operator tuning the cap never learned it was binding.
            # FLOATING, not "not measured". `resample-clock` pins the width
            # exactly as `measured` does, so under the old test every one of
            # its segments reported its own setting as a cap hit and the turn
            # then claimed `safety_cap_hit`. Only a gear whose width floats can
            # be capped by the cap.
            width_capped = bool(
                cfg.gear in FLOATING_WIDTH_GEARS and searched_width >= int(width)
            )
            segment_stats.append(
                {
                    "index": segment_index,
                    "mode": "searched",
                    "width": searched_width,
                    "width_requested": int(width),
                    "chunks": int(anytime.get("chunks", chunk_counter["n"])),
                    "ms": int((seg_finished - seg_started) * 1000),
                    "unique_prefixes": int(
                        anytime.get("unique_prefixes")
                        or rec.get("search_n_dedup")
                        or 0
                    ),
                    "dedup_ratio": (
                        min(1.0, float(
                            int(
                                anytime.get("unique_prefixes")
                                or rec.get("search_n_dedup")
                                or 0
                            )
                        ) / float(searched_width))
                        if searched_width > 0
                        else None
                    ),
                    "leaves_evaluated": int(anytime.get("leaves_evaluated") or 0),
                    "chunk_sizes": list(anytime.get("chunk_sizes") or []),
                    "stage_seconds": copy.deepcopy(anytime.get("stage_seconds") or {}),
                    "chunk_timings": copy.deepcopy(anytime.get("chunk_timings") or []),
                    "segment_budget_s": deadline_holder["segment_budget_s"],
                    "deadline_overshoot_ms": (
                        max(0, int((seg_finished - segment_deadline) * 1000))
                        if segment_deadline is not None
                        else 0
                    ),
                    "stop_reason": stop_reason,
                    "width_capped": width_capped,
                    # exp16 A3. Scalars on purpose: `duel_app._live_segment_
                    # record` keeps scalars and drops structure, so these reach
                    # the page as well as the durable archive.
                    #
                    # Passed through RAW, so all five are `null` together on a
                    # segment whose search was skipped or degraded to greedy
                    # (`_annotate_skip` emits no completion fields). Coercing
                    # them to `bc_greedy`/1/0 would have that segment claim the
                    # old policy ran when no policy ran at all.
                    "completion_policy": rec.get("search_completion_policy"),
                    "completion_width": rec.get("search_completion_width"),
                    "completion_aggregate": rec.get("search_completion_aggregate"),
                    "completion_decided": rec.get("search_completion_decided"),
                    "completion_divergent": rec.get("search_completion_divergent"),
                    # W11b, under the same RAW pass-through rule as the five
                    # above: all null together on a segment whose search was
                    # skipped, rather than a zero that would claim the gear ran
                    # and added nothing.
                    "realised_stochastic_samples": rec.get("search_realised_stochastic_samples"),
                    "extra_sample_levels": rec.get("search_extra_sample_levels"),
                    # A scalar, because the live projection keeps scalars and
                    # drops structure. The mean is what answers "would another
                    # level have fit", and the list itself stays in the raw
                    # search record for anyone who wants the spread.
                    "extra_level_seconds_mean": (
                        (sum(rec.get("search_extra_level_seconds") or [])
                         / len(rec["search_extra_level_seconds"]))
                        if rec.get("search_extra_level_seconds")
                        else None
                    ),
                    "completion_boards_mean": rec.get("search_completion_boards_mean"),
                    "completion_dropped": rec.get("search_completion_dropped"),
                    "completion_second_chance_nodes": rec.get("search_completion_second_chance_nodes"),
                }
            )
            with self._lock:
                self._progress.searched_segments += 1
            return rec

        try:
            out = run_segmented_turn(
                state,
                bc=agent.search,
                honest=True,
                game_engine_seed=int(engine_seed),
                race_wins=int(wins),
                # exp16 PLAN risk 5: UNCONDITIONALLY, every segment. Without
                # it `_race_scalars` raises `vgame_race_wins_unset`, the V
                # leaf refuses to score and the search degrades to greedy BC
                # -- silently, and only in the direction of "weaker".
                set_race_context=agent.search.set_race_context,
                set_imagination_context=agent.search.set_imagination_context,
                decide=_decide,
                on_committed_op=_committed,
                max_segments=int(cfg.max_segments),
            )
            if int(self.generation) != int(gen):
                raise _StaleTurn(f"stale:{gen}")
        except _StaleTurn:
            # Deliberately does NOT touch the status: a newer turn may already
            # be queued behind this one and must not inherit a stale verdict.
            return _failure(gen=gen, turn=turn, code="ai_stale_generation", cfg=cfg)
        except Exception as exc:
            self._finish(STATUS_FAILED, gen=gen)
            return _failure(
                gen=gen,
                turn=turn,
                code=f"ai_turn_crashed:{type(exc).__name__}:{exc}",
                cfg=cfg,
            )

        elapsed_ms = int((self._clock() - started) * 1000)
        finished_at = self._clock()
        with self._lock:
            stop_at = self._stop_at
        stop_latency_ms = (
            int((self._clock() - stop_at) * 1000) if stop_at is not None else None
        )

        segments = out.segments or []
        for stat in segment_stats:
            index = int(stat["index"])
            if index < len(segments):
                stat["boundary_reason"] = segments[index].get("boundary_reason")
                stat["committed_types"] = list(segments[index].get("committed_types") or [])
                stat["search_used"] = bool(segments[index].get("search_used"))
                stat["search_error"] = segments[index].get("search_error")
                stat["n_dedup"] = segments[index].get("search_n_dedup")
            stat["committed_stochastic_reasons"] = list(reasons_by_segment.get(index, []))
            stat["committed_stochastic_structural"] = list(
                structural_by_segment.get(index, [])
            )

        search_errors = [
            s.get("search_error") for s in segments if s.get("search_error")
        ]
        failure_code = None
        if out.decode_failed:
            failure_code = f"decode_failed:{out.stop_reason}"
        elif out.replay_diverged_at:
            failure_code = f"chain_replay_diverged:{out.replay_diverged_at}"
        elif search_errors:
            failure_code = f"search_error:{search_errors[0]}"

        result = {
            "ok": failure_code is None,
            "gen": int(gen),
            "turn": int(turn),
            "error": failure_code,
            "actions": (
                [] if failure_code is not None else [copy.deepcopy(op) for op in out.committed_ops]
            ),
            "partial_actions": [copy.deepcopy(op) for op in out.committed_ops],
            "segments": segment_stats,
            "n_segments": len(segments),
            "searched_segments": sum(1 for s in segment_stats if s["mode"] == "searched"),
            "deadline_segments": sum(
                1
                for s in segment_stats
                if s["mode"] in ("deadline-end-turn", "deadline-greedy")
            ),
            "deadline_greedy_segments": sum(
                1 for s in segment_stats if s["mode"] == "deadline-greedy"
            ),
            "greedy_segments": 0,
            # exp16 A3, turn totals. `decided` is the DENOMINATOR (imagined
            # samples that had more than one completion to choose between) and
            # is 0 under `bc_greedy`, which is the honest reading: nothing was
            # decided, rather than everything agreeing.
            "completion_decided": sum(
                int(s.get("completion_decided") or 0) for s in segment_stats
            ),
            "completion_divergent": sum(
                int(s.get("completion_divergent") or 0) for s in segment_stats
            ),
            # W11b turn totals. `realised_stochastic_samples` is a per-segment
            # LEVEL, not a count of things, so summing it would mean nothing:
            # the turn reports the highest any segment reached, plus the total
            # extra levels bought. None when no segment searched, so a turn that
            # never ran the gear does not read as "ran it and got zero".
            "realised_stochastic_samples": max(
                (int(s["realised_stochastic_samples"]) for s in segment_stats
                 if s.get("realised_stochastic_samples") is not None),
                default=None,
            ),
            "extra_sample_levels": (
                sum(int(s["extra_sample_levels"]) for s in segment_stats
                    if s.get("extra_sample_levels") is not None)
                if any(s.get("extra_sample_levels") is not None for s in segment_stats)
                else None
            ),
            "completion_dropped": (
                sum(int(s["completion_dropped"]) for s in segment_stats
                    if s.get("completion_dropped") is not None)
                if any(s.get("completion_dropped") is not None for s in segment_stats)
                else None
            ),
            "completion_second_chance_nodes": (
                sum(int(s["completion_second_chance_nodes"]) for s in segment_stats
                    if s.get("completion_second_chance_nodes") is not None)
                if any(s.get("completion_second_chance_nodes") is not None for s in segment_stats)
                else None
            ),
            "segments_capped": bool(out.segments_capped),
            "replay_diverged_at": out.replay_diverged_at,
            "decode_failed": bool(out.decode_failed),
            "search_used": any(bool(s.get("search_used")) for s in segments),
            "search_error": (search_errors[0] if search_errors else None),
            "chain_types": list(out.chain_types),
            "elapsed_ms": elapsed_ms,
            "turn_budget_s": float(cfg.turn_budget_s),
            "budget_utilization": (
                (float(elapsed_ms) / 1000.0) / float(cfg.turn_budget_s)
                if cfg.gear in CLOCKED_GEARS
                else None
            ),
            "deadline_overshoot_ms": (
                max(0, int((finished_at - global_deadline) * 1000))
                if cfg.gear in CLOCKED_GEARS
                else 0
            ),
            "effective_unique_prefixes": sum(
                int(s.get("unique_prefixes") or 0) for s in segment_stats
            ),
            "dedup_ratio": (
                min(
                    1.0,
                    float(sum(int(s.get("unique_prefixes") or 0) for s in segment_stats))
                    / float(sum(int(s.get("width") or 0) for s in segment_stats)),
                )
                if sum(int(s.get("width") or 0) for s in segment_stats) > 0
                else None
            ),
            "leaves_evaluated": sum(
                int(s.get("leaves_evaluated") or 0) for s in segment_stats
            ),
            "safety_cap_hit": any(
                s.get("stop_reason") == "safety_cap" or s.get("width_capped")
                for s in segment_stats
            ),
            "human_waiting": stop_at is not None,
            "stop_latency_ms": stop_latency_ms,
            # `cfg`, NOT `self.config`: exp16 W5 lets the page swap the config
            # mid-game, and a turn already in flight keeps the one it started
            # under (`cfg` is bound once, at the top of this method). Reporting
            # the LIVE config here would label a turn with a gear it never ran
            # on -- which is exactly the "how hard did it think" number the duel
            # page shows the human.
            "gear": cfg.gear,
            "finish_turn": bool(cfg.finish_turn),
            "chunk_size": int(cfg.chunk_size),
            "adaptive_chunks": bool(cfg.adaptive_chunks),
            "adaptive_chunk_sizes": list(cfg.adaptive_chunk_sizes),
            "stochastic_samples": int(getattr(agent.search, "stochastic_samples", 0)),
        }
        self._finish(STATUS_READY if result["ok"] else STATUS_FAILED, result, gen=gen)
        return result

    def _finish(
        self, status: str, result: dict[str, Any] | None = None, *, gen: int | None = None
    ) -> None:
        with self._lock:
            if gen is not None and int(gen) != int(self._generation):
                return  # fenced off; a newer turn owns the status now
            self._status = status
            self._result = result


def candidate_stream_key(*, engine_seed: int, turn: int, segment_index: int) -> int:
    """The decision's candidate-sampling stream (`set_candidate_stream`).

    Keyed on the same `(engine_seed, turn, segment_index)` tuple A1's
    imagination stream is, and for the same reason: a decision must be a
    pure function of where it is in THIS game, never of how many decisions
    the process happened to serve before it. sha256, not `hash()`, because
    `hash()` is salted per process and would defeat the whole point.
    """
    key = f"exp16_candidate_stream:{int(engine_seed)}:{int(turn)}:{int(segment_index)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def _engine_seed(state: dict[str, Any]) -> int:
    """A1 ruling 1's `engine_seed`: the GAME's own seed.

    In a duel each side's seed is fixed for the whole game
    (`DuelSession(ai_seed=...)` writes it once into `meta.seed` and the
    engine chains it forward from there). Reading it off the state's CURRENT
    `meta.seed` at turn N would key imagination on the play stream's
    position, which is the coupling A1 removes -- so the duel passes it via
    `submit(engine_seed=...)`. This fallback exists for callers that have no
    seed to give (a fixture, a unit test) and is never the duel's path.
    """
    meta = state.get("meta") if isinstance(state, dict) else None
    if not isinstance(meta, dict):
        return 0
    try:
        return int(meta.get("seed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _failure(
    *, gen: int, turn: int, code: str, cfg: AiTurnConfig | None = None
) -> dict[str, Any]:
    out = {
        "ok": False,
        "gen": int(gen),
        "turn": int(turn),
        "error": code,
        "actions": [],
        "partial_actions": [],
        "segments": [],
        "n_segments": 0,
        "searched_segments": 0,
        "deadline_segments": 0,
        "greedy_segments": 0,
        "segments_capped": False,
        "replay_diverged_at": None,
        "decode_failed": False,
        "search_used": False,
        "search_error": None,
        "chain_types": [],
        "elapsed_ms": 0,
        "stop_requested": False,
        "stop_latency_ms": None,
    }
    if cfg is not None:
        out.update(
            {
                "turn_budget_s": float(cfg.turn_budget_s),
                "budget_utilization": 0.0,
                "deadline_overshoot_ms": 0,
                "effective_unique_prefixes": 0,
                "dedup_ratio": None,
                "leaves_evaluated": 0,
                "safety_cap_hit": False,
                "gear": cfg.gear,
                "finish_turn": bool(cfg.finish_turn),
                "chunk_size": int(cfg.chunk_size),
                "adaptive_chunks": bool(cfg.adaptive_chunks),
                "adaptive_chunk_sizes": list(cfg.adaptive_chunk_sizes),
            }
        )
    return out
