"""Headless human-vs-AI duel core (exp16 W2).

Two live shop sessions, one shared battle per turn. This module owns the
GAME: two sides, their own shop / gold / engine seed, whose turn it is, when
somebody has lost, and who won. It owns none of the AI: the AI's shop phase
arrives as an injected `decide(state) -> [action, ...]` callable, which in
this wave is a stub. W3 replaces that callable with the real segmented
search driver; nothing here imports torch, and nothing here should need to
once it does.

What is deliberately NOT re-derived here:

- the battle -> lives arithmetic and the turn-3 heal, which live in
  `versus_lives.py` and reach us through `end_turn.resolve_duel_turn`
  (which this module calls once per turn, and which calls the battle oracle
  exactly once for the pair);
- the terminal rule, which is `train/env.py::TrainingEnv._is_done` -- the
  same predicate the RL loop and the full-game driver already use. The only
  thing this module adds on top is the turn cap it hands that predicate,
  and the mapping from "somebody is done" to "who won", which no existing
  module has because no existing frame is head-to-head.

Duel rules (internal design notes): versus, 6 lives a side,
the loser of a battle drops one, first side to 0 loses, both sides heal 1
(capped) on the start of turn 3, and a game still alive after turn 30 is a
draw. `--duel-rules arena` is plumbed through to exp13's real-arena life
cap; see `resolve_duel_rules`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from ... import versus_lives
from ...ability.effects import ensure_stat_fields
from ...api import legal_actions, step, validate_state
from ...end_turn import resolve_duel_turn
from ...engine import resolve_end_turn_post_battle, resolve_end_turn_pre_battle
from ...oracles.sap_calc_battle_oracle import run_battle_oracle_with_config, team_to_pet_configs
from ...train.env import TrainingEnv, _set_last_opponent_team
from .compose import apply_group, compose_actions

HUMAN = "human"
AI = "ai"
SIDES = (HUMAN, AI)

DEFAULT_TURN_CAP = 30
DEFAULT_DUEL_RULES = "versus"
DUEL_RULES_CHOICES = ("versus", "arena")

# exp13 W0a lands `ARENA_START_LIVES` (5) and the `max_lives` argument on
# `versus_lives.py` together, on branch `exp13/w0-arena-harness`. Neither is
# on this line's base yet, so `--duel-rules arena` is resolved lazily and
# refuses loudly rather than silently falling back to the versus cap.
_ARENA_START_LIVES: int | None = getattr(versus_lives, "ARENA_START_LIVES", None)
_LIVES_FN_TAKES_MAX_LIVES = "max_lives" in inspect.signature(
    versus_lives.apply_battle_outcome_to_lives
).parameters


class DuelRulesUnavailable(RuntimeError):
    """A known rule set that this checkout cannot honor yet."""


@dataclass(frozen=True)
class DuelRules:
    """One ruler for a duel.

    `game_mode` is always `"versus"`, including for the arena ruler, and
    that is not an oversight -- see `end_turn.resolve_duel_turn`'s docstring.
    A duel is two-sided by construction: each side's
    `meta.versus.opponent_lives` mirrors the other's `lives`, and that
    mirror is what `TrainingEnv._is_done` reads. exp13's arena ruler is a
    POOL ruler (5 lives, 10 trophies, opponents drawn from a pool); the part
    of it that means anything head-to-head is its life cap, which is exactly
    what `max_lives` selects here.

    `max_lives=None` means "leave `versus_lives.apply_battle_outcome_to_lives`
    on its own default", which is 6.
    """

    name: str
    game_mode: str
    start_lives: int
    max_lives: int | None
    turn_cap: int


def resolve_duel_rules(rules: str | DuelRules = DEFAULT_DUEL_RULES, *, turn_cap: int = DEFAULT_TURN_CAP) -> DuelRules:
    """Name -> `DuelRules`. Pass a `DuelRules` through unchanged.

    Raises `ValueError` for an unknown name and `DuelRulesUnavailable` for
    `"arena"` on a checkout without exp13 W0a's `versus_lives` additions --
    the alternative, quietly capping arena lives at 6, is the kind of
    silent wrong number this wave exists to avoid.
    """
    if isinstance(rules, DuelRules):
        return rules
    name = str(rules or "").strip().lower()
    if name == "versus":
        return DuelRules(
            name="versus",
            game_mode="versus",
            start_lives=int(versus_lives.MAX_LIVES),
            max_lives=None,
            turn_cap=int(turn_cap),
        )
    if name == "arena":
        if _ARENA_START_LIVES is None or not _LIVES_FN_TAKES_MAX_LIVES:
            raise DuelRulesUnavailable(
                "duel_rules_arena_unavailable: needs versus_lives.ARENA_START_LIVES and "
                "apply_battle_outcome_to_lives(max_lives=...), both of which land with exp13 W0a "
                "(branch exp13/w0-arena-harness). Until that merges, only --duel-rules versus runs."
            )
        return DuelRules(
            name="arena",
            game_mode="versus",
            start_lives=int(_ARENA_START_LIVES),
            max_lives=int(_ARENA_START_LIVES),
            turn_cap=int(turn_cap),
        )
    raise ValueError(f"invalid_duel_rules:{name}")


def add_duel_rules_argument(
    parser: argparse.ArgumentParser,
    *,
    flag: str = "--duel-rules",
    default: str = DEFAULT_DUEL_RULES,
) -> argparse.ArgumentParser:
    """Add `--duel-rules {versus,arena}` to a parser.

    One definition so every future entry point (W3's duel smoke runner,
    W5's server) spells the flag the same way. Selecting `arena` on a
    checkout that cannot honor it fails at `resolve_duel_rules`, not here,
    so `--help` always lists both.
    """
    parser.add_argument(
        flag,
        type=str,
        choices=list(DUEL_RULES_CHOICES),
        default=default,
        help=(
            "life ruler for the duel: versus = 6 lives a side (default); "
            "arena = exp13's real-arena life cap (requires exp13 W0a)"
        ),
    )
    return parser


def duel_terminal_status(
    human_state: dict[str, Any],
    ai_state: dict[str, Any],
    *,
    turn_cap: int = DEFAULT_TURN_CAP,
) -> dict[str, Any]:
    """Is the duel over, and who won?

    "Over" is `TrainingEnv._is_done` on each side, not a second copy of the
    rule: it already covers `lives <= 0`, the mirrored
    `meta.versus.opponent_lives <= 0`, and the post-advance turn cap
    (inclusive of `turn_cap` itself -- turn `turn_cap` is still playable).
    Both sides are checked and OR-ed; with a consistent mirror they always
    agree, and `resolve_duel_turn` refuses to run on an inconsistent one.

    `winner` is `None` for a draw, which is both the turn-cap ending and
    the (unreachable in practice) double knockout.
    """
    done = bool(
        TrainingEnv._is_done(human_state, int(turn_cap))
        or TrainingEnv._is_done(ai_state, int(turn_cap))
    )
    human_lives = int(human_state.get("lives", 0))
    ai_lives = int(ai_state.get("lives", 0))

    winner: str | None = None
    end_reason: str | None = None
    if human_lives <= 0 and ai_lives <= 0:
        end_reason = "both_lives_0"
    elif human_lives <= 0:
        winner = AI
        end_reason = "human_lives_0"
    elif ai_lives <= 0:
        winner = HUMAN
        end_reason = "ai_lives_0"
    elif done:
        end_reason = "turn_cap"

    if not done:
        winner = None
        end_reason = None
    return {
        "done": done,
        "winner": winner,
        "end_reason": end_reason,
        "human_lives": human_lives,
        "ai_lives": ai_lives,
    }


@dataclass
class DuelSide:
    """One side's live session: the `SessionState` shape a duel needs.

    Intentionally NOT `play_web/app.py::SessionState` itself. That dataclass
    carries the browser's rendering state (image versions, predicted-board
    rows, the sampled replay battle) and importing it would drag the
    planner, the predictor and the replay player into a module whose whole
    point is to stay light enough for a background worker thread. W5 wires
    the two together at the App layer; the fields below are the overlap.
    """

    name: str
    state: dict[str, Any]
    seed: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    # One entry per `history` entry: the composed group a transition belongs
    # to, or None. Beside the transition, not inside it, because
    # `schemas/transition_v1.json` is `additionalProperties: false`.
    history_group_ids: list[str | None] = field(default_factory=list)
    # The real engine transition that produced this turn's opening shop.
    # Turn 1 is the seeded initial deal; later turns reuse the committed
    # END_TURN transition whose state_after is this side's current state.
    turn_start_transition: dict[str, Any] | None = None
    battle_rows: list[dict[str, Any]] = field(default_factory=list)
    last_battle: dict[str, Any] | None = None
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def push(self, tr: dict[str, Any], group_id: str | None = None) -> None:
        """The only way a transition enters a side's history.

        `history` and `history_group_ids` must stay the same length; one
        entry point is what keeps them from drifting."""
        self.history.append(copy.deepcopy(tr))
        self.history_group_ids.append(group_id)

    @property
    def lives(self) -> int:
        return int(self.state.get("lives", 0))

    @property
    def turn(self) -> int:
        return int(self.state.get("turn", 1))


def _reroll_shop_transition_for_seed(state: dict[str, Any]) -> dict[str, Any]:
    """Regenerate the shop from the state's own seed, gold unchanged.

    Same trick as `play_web/app.py::App._roll_no_cost` (clear the shop, pay
    for a roll out of a temporarily inflated purse, put the gold back), with
    one deliberate difference: play-web forces `meta.seed_known = False` so
    its shop is un-reproducible, and a duel needs the opposite -- both
    sides' openings must be a pure function of their seeds so a game can be
    replayed bit-for-bit (exp16 W3/W6).
    """
    work = copy.deepcopy(state)
    target_gold = int(work.get("gold", 10))
    work["shop"] = []
    work["gold"] = target_gold + 1
    tr = step(work, {"type": "ROLL"})
    if not tr.get("legal"):
        raise RuntimeError(f"duel_shop_roll_failed:{tr.get('engine_notes')}")
    if int((tr.get("state_after") or {}).get("gold", -1)) != target_gold:
        raise RuntimeError("duel_shop_roll_gold_mismatch")
    return copy.deepcopy(tr)


def _reroll_shop_for_seed(state: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper returning only the seeded deal's state."""
    return copy.deepcopy(_reroll_shop_transition_for_seed(state)["state_after"])


def prepare_duel_side_state(
    base_state: dict[str, Any],
    *,
    rules: DuelRules,
    seed: int | None = None,
) -> dict[str, Any]:
    """Turn a fixture's `initial_state` into one side's turn-1 duel state.

    Sets the mode and both life counters from `rules`, clears the carried
    battle context so turn 1 starts from nothing (`meta.last_battle_result`
    feeds Snail's end-of-turn ability; an inherited "loss" would buff a
    board that never lost), empties the last-seen opponent board through the
    normal writer, and -- when a seed is given -- rolls the opening shop
    from that seed so the two sides do not open on the same board.

    `meta.game_rules` is written for a NON-versus ruler only, matching
    exp13 W0a's own convention (`tools/eval_versus_fullgame.py`, which
    writes it "under arena rules ONLY, so a versus state is byte-identical
    to before this flag existed"). It is not decoration:
    `schemas/state_v1.json` pins turn 1 to exactly 6 lives, and exp13
    widens that to "exactly 5 when `meta.game_rules == arena`" -- so an
    arena duel that skipped this flag would be rejected by `validate_state`
    on its very first turn. `game_mode` stays `"versus"` regardless (see
    `DuelRules`): the flag says WHICH ruler, the mode says how the lives are
    booked, and a duel books them two-sidedly under either ruler.
    """
    state, _transition = _prepare_duel_side_state_with_transition(
        base_state,
        rules=rules,
        seed=seed,
    )
    return state


def _prepare_duel_side_state_with_transition(
    base_state: dict[str, Any], *, rules: DuelRules, seed: int | None = None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    state = copy.deepcopy(base_state)
    for slot in state.get("team", []) or []:
        ensure_stat_fields(slot)

    meta = state.setdefault("meta", {})
    meta["game_mode"] = rules.game_mode
    if rules.name != "versus":
        meta["game_rules"] = rules.name
    meta["last_battle_result"] = ""
    versus = meta.setdefault("versus", {})
    if not isinstance(versus, dict):
        versus = {}
        meta["versus"] = versus
    state["lives"] = int(rules.start_lives)
    versus["opponent_lives"] = int(rules.start_lives)
    versus.pop("current_opponent_participation_id", None)
    _set_last_opponent_team(state, None)

    transition: dict[str, Any] | None = None
    if seed is not None:
        meta["seed_known"] = True
        meta["seed"] = int(seed)
        transition = _reroll_shop_transition_for_seed(state)
        state = copy.deepcopy(transition["state_after"])

    validate_state(state)
    return state, transition


def duel_battle_seed(seed: int, turn: int) -> int:
    """A per-(game, turn) battle seed, so a duel can be replayed.

    The battle oracle is an UNSEEDED Monte-Carlo simulator by default: the
    same pair of boards can go 18-2 over 20 runs, which is why exp16's PLAN
    locks undo after a turn resolves (otherwise a lost battle could simply
    be re-rolled). The calculator does accept `config.seed` and swaps in a
    seeded `Math.random` for the run (`simulation-runner.ts` ->
    `applySeededRandom`), so passing one makes ONE battle a pure function of
    (game seed, turn) -- which is what "the same seeds reproduce the game"
    and W6's replay archive need, and which also closes the reroll hole
    rather than papering over it.

    sha256 rather than the builtin `hash()`, for the same reason
    `honest_frame.imagination_seed` uses it: `hash()` is salted per process.
    Truncated to 32 bits unsigned because the calculator's PRNG state is
    `seed >>> 0`.
    """
    key = f"exp16_duel_battle:{int(seed)}:{int(turn)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def seeded_battle_fn(
    seed: int,
    *,
    run_battle_fn: Callable[[dict[str, Any]], dict[str, Any]] = run_battle_oracle_with_config,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """`run_battle_fn` for a `DuelSession`, made reproducible by `seed`.

    Opt-in: a session built without it keeps today's unseeded Monte-Carlo
    behaviour byte for byte.
    """

    def _run(config: dict[str, Any]) -> dict[str, Any]:
        payload = dict(config)
        payload["seed"] = duel_battle_seed(seed, int(config.get("turn", 0) or 0))
        return run_battle_fn(payload)

    return _run


def _noop_decide(_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Default AI: buy nothing, just end the turn.

    The seam W3 replaces. Kept trivial and side-effect free so a duel is
    playable (and testable) end to end with no model on the box.
    """
    return []


class DuelSession:
    """A full human-vs-AI game: two sides, one battle a turn.

    The human side is driven action by action (`apply_human_action`), the
    way a browser drives play-web. The AI side is driven once per turn, at
    `end_turn()`, by the injected `ai_decide`. Both shop phases are applied
    to their own state; then ONE call to `resolve_duel_turn` runs the single
    shared battle and both sides' post-battle bookkeeping.

    `end_turn()` is all-or-nothing: an illegal AI action or a failed battle
    leaves the session exactly as it was and returns `ok=False` with a code,
    so the caller (W3) can apply its own "the AI stands still this turn"
    policy instead of the session silently playing a weakened AI.
    """

    def __init__(
        self,
        initial_state: dict[str, Any],
        *,
        rules: str | DuelRules = DEFAULT_DUEL_RULES,
        turn_cap: int = DEFAULT_TURN_CAP,
        human_seed: int | None = None,
        ai_seed: int | None = None,
        ai_initial_state: dict[str, Any] | None = None,
        ai_decide: Callable[[dict[str, Any]], list[dict[str, Any]]] | None = None,
        pre_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_pre_battle,
        post_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_post_battle,
        run_battle_fn: Callable[[dict[str, Any]], dict[str, Any]] = run_battle_oracle_with_config,
        team_to_pet_configs_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]] = team_to_pet_configs,
        resolve_duel_turn_fn: Callable[..., dict[str, Any]] = resolve_duel_turn,
        simulation_count: int = 1,
        battle_logs_enabled: bool = False,
    ) -> None:
        self.rules = resolve_duel_rules(rules, turn_cap=turn_cap)
        self.ai_decide = ai_decide if ai_decide is not None else _noop_decide
        self.pre_battle_fn = pre_battle_fn
        self.post_battle_fn = post_battle_fn
        self.run_battle_fn = run_battle_fn
        self.team_to_pet_configs_fn = team_to_pet_configs_fn
        self.resolve_duel_turn_fn = resolve_duel_turn_fn
        self.simulation_count = int(simulation_count)
        self.battle_logs_enabled = bool(battle_logs_enabled)

        human_state, human_start = _prepare_duel_side_state_with_transition(
            initial_state, rules=self.rules, seed=human_seed
        )
        self.human = DuelSide(
            name=HUMAN,
            state=human_state,
            seed=human_seed,
            turn_start_transition=human_start,
        )
        ai_state, ai_start = _prepare_duel_side_state_with_transition(
            ai_initial_state if ai_initial_state is not None else initial_state,
            rules=self.rules,
            seed=ai_seed,
        )
        self.ai = DuelSide(
            name=AI,
            state=ai_state,
            seed=ai_seed,
            turn_start_transition=ai_start,
        )
        self.turn_results: list[dict[str, Any]] = []
        self._group_seq = 0
        # Where the CURRENT turn's human history starts. `undo_human_action`
        # will not pop below it, which is what keeps undo inside the turn
        # (exp16 PLAN risk 7: the battle is a sample, so an undo that reached
        # back across a resolved turn could re-roll a lost battle).
        self._human_turn_mark = 0
        self._status = duel_terminal_status(self.human.state, self.ai.state, turn_cap=self.rules.turn_cap)

    # -- read-only view -------------------------------------------------
    @property
    def turn(self) -> int:
        return int(self.human.state.get("turn", 1))

    @property
    def done(self) -> bool:
        return bool(self._status.get("done"))

    @property
    def winner(self) -> str | None:
        return self._status.get("winner")

    @property
    def end_reason(self) -> str | None:
        return self._status.get("end_reason")

    def side(self, name: str) -> DuelSide:
        key = str(name or "").strip().lower()
        if key == HUMAN:
            return self.human
        if key == AI:
            return self.ai
        raise ValueError(f"invalid_duel_side:{key}")

    def legal_actions(self, name: str = HUMAN) -> list[dict[str, Any]]:
        return legal_actions(self.side(name).state)

    # -- the human's shop phase -----------------------------------------
    def apply_human_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Apply one shop action to the human side.

        `END_TURN` is not a shop action here -- it is the whole-duel step,
        `end_turn()`, because it moves both sides.
        """
        if self.done:
            return {"ok": False, "error": f"duel_over:{self.end_reason}", "transition": None}
        act = copy.deepcopy(action)
        if str(act.get("type", "")).strip().upper() == "END_TURN":
            return {"ok": False, "error": "use_end_turn_for_duel_step", "transition": None}
        tr = step(self.human.state, act)
        if not tr.get("legal"):
            return {"ok": False, "error": "illegal_action", "transition": tr}
        self.human.state = copy.deepcopy(tr["state_after"])
        self.human.push(tr)
        return {"ok": True, "error": None, "transition": tr}

    def undo_human_action(self) -> dict[str, Any]:
        """Undo the human's last shop action -- WITHIN this turn only (W5).

        A composed group (W4's "buy into slot k") unwinds whole, the same way
        `play_web/app.py::App.undo` unwinds it: the group is contiguous and
        was appended together, so popping to its head is "keep popping while
        the entry below carries the same id".

        Refuses at `_human_turn_mark`, the index this turn's history starts
        at. That refusal is the rule, not a convenience: `end_turn` resolves
        ONE sampled battle per turn, so an undo that could reach back past a
        resolved turn would let a lost battle be replayed until it was won
        (exp16 PLAN risk 7). The AI's own thinking is untouched by this --
        it is thinking about ITS board, which the human's shop cannot reach.
        """
        if self.done:
            return {"ok": False, "error": f"duel_over:{self.end_reason}", "transition": None}
        if len(self.human.history) <= int(self._human_turn_mark):
            return {"ok": False, "error": "no_history_to_undo_this_turn", "transition": None}

        tr = self.human.history.pop()
        group_id = (
            self.human.history_group_ids.pop() if self.human.history_group_ids else None
        )
        if group_id is not None:
            while (
                len(self.human.history) > int(self._human_turn_mark)
                and self.human.history_group_ids
                and self.human.history_group_ids[-1] == group_id
            ):
                tr = self.human.history.pop()
                self.human.history_group_ids.pop()
        self.human.state = copy.deepcopy(tr["state_before"])
        validate_state(self.human.state)
        return {"ok": True, "error": None, "transition": tr, "group_id": group_id}

    def _next_group_id(self) -> str:
        self._group_seq += 1
        return f"grp-{self._group_seq}"

    def apply_human_composition(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply a `{"compose": ...}` payload to the human side atomically.

        Same contract as `play_web/app.py::App.apply_compose` -- same
        composer, same `apply_group`, same all-or-nothing rollback and the
        same shared group id in the history -- because the duel page and the
        sandbox page must not disagree about what "buy into slot 4" does.
        """
        if self.done:
            return {
                "ok": False,
                "error": f"duel_over:{self.end_reason}",
                "compose": None,
                "transition": None,
                "transitions": [],
                "group_id": None,
            }
        kind, actions, err = compose_actions(self.human.state, payload)
        if err is not None:
            return {
                "ok": False,
                "error": err,
                "compose": kind,
                "transition": None,
                "transitions": [],
                "group_id": None,
            }
        ok, group_err, state_after, transitions = apply_group(self.human.state, actions)
        if not ok:
            return {
                "ok": False,
                "error": group_err,
                "compose": kind,
                "transition": None,
                "transitions": [],
                "group_id": None,
            }
        self.human.state = state_after
        group_id = self._next_group_id()
        for tr in transitions:
            self.human.push(tr, group_id)
        return {
            "ok": True,
            "error": None,
            "compose": kind,
            "transition": transitions[-1],
            "transitions": transitions,
            "group_id": group_id,
        }

    # -- the AI's shop phase --------------------------------------------
    def _play_ai_shop_phase(self) -> dict[str, Any]:
        """Run `ai_decide` and apply its chain to a COPY of the AI state.

        Nothing is committed here; the caller commits only if the whole turn
        resolves. An illegal action aborts the chain with its index and type
        so a failure is attributable to one action, not to "the AI".
        """
        decided = self.ai_decide(copy.deepcopy(self.ai.state))
        actions = [copy.deepcopy(a) for a in (decided or [])]
        work = copy.deepcopy(self.ai.state)
        applied: list[dict[str, Any]] = []
        for index, act in enumerate(actions):
            if str(act.get("type", "")).strip().upper() == "END_TURN":
                return {
                    "ok": False,
                    "error": f"ai_action_end_turn_in_chain:{index}",
                    "actions": actions,
                    "applied": applied,
                    "state": work,
                }
            tr = step(work, act)
            if not tr.get("legal"):
                return {
                    "ok": False,
                    "error": f"ai_action_illegal:{index}:{act.get('type')}",
                    "actions": actions,
                    "applied": applied,
                    "state": work,
                }
            work = copy.deepcopy(tr["state_after"])
            applied.append(copy.deepcopy(tr))
        return {"ok": True, "error": None, "actions": actions, "applied": applied, "state": work}

    # -- the duel step ---------------------------------------------------
    def end_turn(self) -> dict[str, Any]:
        """Play the AI's shop phase, resolve the shared battle, advance both."""
        if self.done:
            return {"ok": False, "error": f"duel_over:{self.end_reason}", "turn": self.turn, "result": None}

        turn = self.turn
        ai_phase = self._play_ai_shop_phase()
        if not ai_phase.get("ok"):
            return {
                "ok": False,
                "error": str(ai_phase.get("error")),
                "turn": turn,
                "result": None,
                "ai_actions": ai_phase.get("actions"),
            }

        ai_state_after_shop = ai_phase["state"]
        resolved = self.resolve_duel_turn_fn(
            self.human.state,
            ai_state_after_shop,
            game_mode=self.rules.game_mode,
            pre_battle_fn=self.pre_battle_fn,
            post_battle_fn=self.post_battle_fn,
            run_battle_fn=self.run_battle_fn,
            team_to_pet_configs_fn=self.team_to_pet_configs_fn,
            simulation_count=self.simulation_count,
            battle_logs_enabled=self.battle_logs_enabled,
            max_lives=self.rules.max_lives,
        )
        if not resolved.get("ok"):
            return {
                "ok": False,
                "error": str(resolved.get("error")),
                "turn": turn,
                "result": None,
                "ai_actions": ai_phase.get("actions"),
            }

        # Commit both sides together. Up to this point the session is
        # untouched, so any failure above leaves a replayable state.
        human_payload = resolved["human"]
        ai_payload = resolved["ai"]
        battle = resolved.get("battle") if isinstance(resolved.get("battle"), dict) else {}

        for ai_tr in ai_phase["applied"]:
            self.ai.push(ai_tr)
        self.human.state = copy.deepcopy(human_payload["state_after"])
        self.ai.state = copy.deepcopy(ai_payload["state_after"])
        self.human.push(human_payload["transition"])
        self.ai.push(ai_payload["transition"])
        self.human.turn_start_transition = copy.deepcopy(
            human_payload["transition"]
        )
        self.ai.turn_start_transition = copy.deepcopy(ai_payload["transition"])

        for sidepay, side in ((human_payload, self.human), (ai_payload, self.ai)):
            outcome = str(sidepay["outcome"])
            if outcome == "win":
                side.wins += 1
            elif outcome == "loss":
                side.losses += 1
            else:
                side.draws += 1
            side.last_battle = copy.deepcopy(battle)

        # Same reversed convention `play_web/app.py` feeds
        # `render_replay_image_from_calc_rows` (exp16 W6 renders from these).
        human_pets = list(human_payload["pet_configs"])
        ai_pets = list(ai_payload["pet_configs"])
        self.human.battle_rows.append(
            {
                "turn": int(turn),
                "outcome": str(human_payload["outcome"]),
                "opponentName": "AI",
                "playerPets": list(reversed(copy.deepcopy(human_pets))),
                "opponentPets": list(reversed(copy.deepcopy(ai_pets))),
            }
        )
        self.ai.battle_rows.append(
            {
                "turn": int(turn),
                "outcome": str(ai_payload["outcome"]),
                "opponentName": "Human",
                "playerPets": list(reversed(copy.deepcopy(ai_pets))),
                "opponentPets": list(reversed(copy.deepcopy(human_pets))),
            }
        )

        # This turn is committed, so the human's undo horizon moves up to here
        # (see `undo_human_action`).
        self._human_turn_mark = len(self.human.history)

        self._status = duel_terminal_status(self.human.state, self.ai.state, turn_cap=self.rules.turn_cap)
        result = {
            "turn": int(turn),
            "outcome": str(resolved.get("outcome")),
            "ai_actions": copy.deepcopy(ai_phase.get("actions") or []),
            "human": {
                "lives": int(human_payload["lives"]),
                "opponent_lives": int(human_payload["opponent_lives"]),
                "outcome": str(human_payload["outcome"]),
                "life_bonus_applied": bool(human_payload["life_bonus_applied"]),
                "board_pre_battle": copy.deepcopy(human_pets),
                # The same board as `board_pre_battle`, in the ENGINE's own
                # slot shape rather than the oracle's pet-config shape, so a
                # browser can render it with the skin it already has (W5) and
                # W6 can replay it through the engine.
                "board_pre_battle_team": copy.deepcopy(
                    (human_payload.get("battle_state") or {}).get("team") or []
                ),
            },
            "ai": {
                "lives": int(ai_payload["lives"]),
                "opponent_lives": int(ai_payload["opponent_lives"]),
                "outcome": str(ai_payload["outcome"]),
                "life_bonus_applied": bool(ai_payload["life_bonus_applied"]),
                "board_pre_battle": copy.deepcopy(ai_pets),
                "board_pre_battle_team": copy.deepcopy(
                    (ai_payload.get("battle_state") or {}).get("team") or []
                ),
            },
            "battle": copy.deepcopy(battle),
            "timing_ms": copy.deepcopy(resolved.get("timing_ms") or {}),
            **self._status,
        }
        self.turn_results.append(result)
        return {"ok": True, "error": None, "turn": int(turn), "result": result}

    def snapshot(self) -> dict[str, Any]:
        """JSON-able view of the whole duel (exp16 W6's archive reads this)."""
        return {
            "rules": {
                "name": self.rules.name,
                "game_mode": self.rules.game_mode,
                "start_lives": self.rules.start_lives,
                "max_lives": self.rules.max_lives,
                "turn_cap": self.rules.turn_cap,
            },
            "seeds": {HUMAN: self.human.seed, AI: self.ai.seed},
            "turn": self.turn,
            "done": self.done,
            "winner": self.winner,
            "end_reason": self.end_reason,
            "final": {
                HUMAN: {
                    "lives": self.human.lives,
                    "wins": self.human.wins,
                    "losses": self.human.losses,
                    "draws": self.human.draws,
                },
                AI: {
                    "lives": self.ai.lives,
                    "wins": self.ai.wins,
                    "losses": self.ai.losses,
                    "draws": self.ai.draws,
                },
            },
            "turns": copy.deepcopy(self.turn_results),
        }
