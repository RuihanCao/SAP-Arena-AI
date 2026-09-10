"""Deterministic ability event queue runtime."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import (
    AbilityEvent,
    TRIGGER_AFTER_FAINT,
    TRIGGER_EATS_FOOD,
    TRIGGER_FAINT,
    TRIGGER_FRIEND_AHEAD_FAINTS,
    TRIGGER_FRIEND_FAINTS,
    TRIGGER_FRIEND_SUMMONED,
    TRIGGER_FRIENDLY_ATE_FOOD,
    TRIGGER_FRIENDS_HURT_COUNTER,
    TRIGGER_HURT,
    TRIGGER_LEVEL_UP,
    TRIGGER_PURCHASE_FOOD,
    TRIGGER_SUMMONED,
)
from .registry import is_declared_trigger, resolve_handler

ABILITY_TRIGGER_PRIORITIES: dict[str, int] = {
    TRIGGER_LEVEL_UP: 1,
    TRIGGER_HURT: 3,
    TRIGGER_FRIENDS_HURT_COUNTER: 4,
    TRIGGER_SUMMONED: 6,
    TRIGGER_FRIEND_SUMMONED: 7,
    TRIGGER_FAINT: 9,
    TRIGGER_FRIEND_AHEAD_FAINTS: 10,
    TRIGGER_FRIEND_FAINTS: 11,
    TRIGGER_PURCHASE_FOOD: 17,
    TRIGGER_EATS_FOOD: 17,
    TRIGGER_FRIENDLY_ATE_FOOD: 17,
    TRIGGER_AFTER_FAINT: 25,
}


def sort_ability_events_for_runtime(
    state: dict[str, Any], events: list[AbilityEvent], rng: Any
) -> tuple[list[AbilityEvent], bool]:
    """Order one simultaneous event group and report whether a random tie was used.

    exp13 PLAN Amendment A2.6: an event that resolves to NO HANDLER takes no
    part in the tie test. `AbilityRuntime.run` resolves such an event by doing
    nothing to the board at all, so its position in the group cannot change an
    outcome, and it has no business pulling a random number and forcing the
    reseed that follows.

    This is where the engine emitted most of its randomness: `RESULTS_W0d.md`
    measured 87.9% of the resolutions raising `ability_randomness` as groups in
    which not a single event could resolve a handler, because the engine emits
    `buy_friend`, `buy_tier1_pet`, `purchase_food`, `friendly_ate_food` and
    `friend_summoned` for triggers no Turtle pet implements, and then ties them.

    The no-handler events are still RETURNED, and still reach the runtime: a
    DECLARED trigger with no implementation is how `unsupported_effect` gets
    raised, and dropping the event here would silently drop that signal.

    Unlike A2.1 and A4 this DOES change the board a given seed produces. Two
    legs, both recorded in the amendment: skipping the draw removes the reseed
    that followed it, which moves `meta.seed` and therefore every later shop
    roll; and where a group still ties, its real handlers now draw from a
    different position in the stream, so their order among themselves can move.
    The first is arithmetic. The second rests on A2 finding 4 -- tie order came
    back board-, topology- and legal-mask-invariant in 3,106 of 3,106
    exhaustive permutations and 1,350 of 1,350 alternate-seed replays.
    """
    if len(events) <= 1:
        return list(events), False

    decorated: list[tuple[int, int, float, AbilityEvent]] = []
    grouped: dict[tuple[int, int], list[int]] = {}
    for i, event in enumerate(events):
        trigger_priority = int(ABILITY_TRIGGER_PRIORITIES.get(str(event.trigger), 999))
        idx = event.actor_team_index
        if idx is not None:
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                idx = None
        if idx is not None and 0 <= idx < len(state.get("team", [])):
            actor_priority = int(state["team"][idx].get("attack", 0))
        else:
            payload = event.payload if isinstance(event.payload, dict) else {}
            actor_priority = int(payload.get("actor_attack", 0))
        decorated.append((trigger_priority, actor_priority, 0.0, event))
        # A2.6: no handler, no vote. The event keeps its place in `decorated`
        # (it still has to reach the runtime) but is left out of the tie test,
        # so a group of nothing but no-handler events draws nothing at all.
        if resolve_handler(event.actor_pet_id, event.trigger) is None:
            continue
        grouped.setdefault((trigger_priority, actor_priority), []).append(i)

    random_tie_used = False
    for indices in grouped.values():
        if len(indices) <= 1:
            continue
        random_tie_used = True
        for i in indices:
            trigger_priority, actor_priority, _tie, event = decorated[i]
            decorated[i] = (trigger_priority, actor_priority, float(rng.random()), event)

    decorated.sort(key=lambda row: (int(row[0]), -int(row[1]), float(row[2])))
    return [row[3] for row in decorated], random_tie_used


@dataclass
class AbilityRuntimeContext:
    """Per-resolution scratch space handed to every ability handler.

    `random_used` says a handler consumed a random number.
    `structural_used` (exp13 PLAN Amendment A2.1) says a handler wrote a field
    `legal_actions` reads: shop slot `slot_type`/`item_id`/`cost`/`frozen` and
    the slot count, `gold`, team occupancy/`pet_id`/`level`. The two are
    independent -- the seven stat-buff callers that route through
    `effects.add_slot_stats` set the first and not the second, because
    `legal_actions` never reads attack or health, while a deterministic Pig
    sell sets the second and not the first.

    `causality` is Amendment A4, as revised after the W1a-3 review. It is the
    STEP's taint object (`engine.StepCausality`), not this runtime's: the
    first cut of A4 kept the causal scope here, inside a context that dies with
    its runtime, and the read-set writes that matter most -- a faint clearing a
    slot -- happen in `engine._resolve_shop_hurt_faint_chain`, runtimes later.
    A random stat buff decided who survived and nothing was left alive to say
    so. So handlers still declare through `effects.mark_random` and
    `effects.mark_structural`, and those now speak to an object that outlives
    every runtime in the step.

    Handlers declare rather than being named, so a future random effect that
    grants gold or discounts the shop becomes a cut point the day it lands
    (A1 ruling 2: never a whitelist).

    `structural_used` stays local to the runtime as telemetry -- "did a handler
    in THIS resolution write the read-set" -- and is what the targeted tests
    assert against.

    Nested events emitted by a handler are inserted immediately after that
    handler, with the same simultaneous-group ordering and RNG as top-level
    events. This matches synchronous perk-gain food triggers without starting
    a second runtime or reseeding the state twice.
    """

    state: dict[str, Any]
    rng: Any
    notes: list[str]
    causality: Any = None
    random_used: bool = False
    structural_used: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)
    _emit_groups: Callable[[list[list[AbilityEvent]]], None] | None = field(default=None, repr=False)

    def emit_groups(self, *groups: list[AbilityEvent]) -> None:
        if self._emit_groups is None:
            raise RuntimeError("ability runtime is not attached")
        self._emit_groups([list(group) for group in groups])


class AbilityRuntime:
    """Small FIFO queue for ability trigger execution."""

    def __init__(
        self,
        state: dict[str, Any],
        rng: Any,
        notes: list[str],
        causality: Any = None,
        max_events: int = 128,
    ) -> None:
        # A4 revision 2: no per-event taint any more. The step's taint is
        # sticky, so an event emitted after a draw is covered by the draw
        # having armed the step, not by anything carried alongside the event.
        self._queue: deque[AbilityEvent] = deque()
        self._nested_events: list[AbilityEvent] = []
        self.ctx = AbilityRuntimeContext(state=state, rng=rng, notes=notes, causality=causality)
        self.ctx._emit_groups = self._emit_nested_groups
        self.max_events = int(max_events)
        self.unsupported = False

    @property
    def random_used(self) -> bool:
        return self.ctx.random_used

    @property
    def structural_used(self) -> bool:
        return self.ctx.structural_used

    @property
    def trace(self) -> list[dict[str, Any]]:
        return self.ctx.trace

    def emit(self, event: AbilityEvent) -> None:
        self._queue.append(event)

    def _emit_nested_groups(self, groups: list[list[AbilityEvent]]) -> None:
        ordered: list[AbilityEvent] = []
        for group in groups:
            sorted_group, random_tie_used = sort_ability_events_for_runtime(
                self.ctx.state, group, self.ctx.rng
            )
            if random_tie_used:
                # A4.2 as revised: an order draw among LIVE handlers arms the
                # step, exactly as the top-level tie in
                # `engine._run_ability_events` does. A2.6 already removed the
                # groups where nothing could act on the order.
                self.ctx.random_used = True
                if self.ctx.causality is not None:
                    self.ctx.causality.arm()
            ordered.extend(sorted_group)
        self._nested_events.extend(ordered)

    def run(self) -> None:
        steps = 0
        while self._queue:
            if steps >= self.max_events:
                self.unsupported = True
                self.ctx.notes.append("unsupported_effect:ability_queue_limit_reached")
                break
            steps += 1

            event = self._queue.popleft()
            event_view = {
                "trigger": str(event.trigger),
                "pet_id": str(event.actor_pet_id),
                "level": int(event.actor_level),
                "team_index": event.actor_team_index,
            }
            self.ctx.trace.append({"event": event_view, "status": "received"})

            handler = resolve_handler(event.actor_pet_id, event.trigger)
            if handler is None:
                if is_declared_trigger(event.actor_pet_id, event.trigger):
                    self.unsupported = True
                    self.ctx.notes.append(
                        f"unsupported_effect:ability_unimplemented:{event.actor_pet_id}:{event.trigger}"
                    )
                    self.ctx.trace.append({"event": event_view, "status": "unsupported"})
                continue

            self.ctx.notes.append(
                f"ability_event:{event.trigger}:{event.actor_pet_id}:team={event.actor_team_index}"
            )
            handler(self.ctx.state, event, self.ctx)
            self.ctx.trace.append({"event": event_view, "status": "applied"})
            if self._nested_events:
                for nested_event in reversed(self._nested_events):
                    self._queue.appendleft(nested_event)
                self._nested_events.clear()
