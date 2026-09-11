"""So the rules live here, once, as a pure function over plain ints:
`end_turn.py` calls it, the human reference calls it, and neither can drift.
Nothing in this module imports anything from the rest of `sap_ppo` -- it is
arithmetic over five scalars.

The rules themselves are UNCHANGED from the inline code this replaces
(`test_versus_lives.py` pins that with an independent re-implementation of
the old inline block, plus `test_end_turn_runtime.py`'s pre-existing
`test_resolver_versus_win_decrements_opponent_lives_only` /
`test_resolver_versus_turn3_bonus_applies_to_both_sides`):

- versus: `win` -> opponent loses a life; `loss` -> you lose a life;
  `draw` -> nobody does. Floored at 0.
- arena: `win` -> +1 trophy (capped at 10); `loss` -> you lose a life;
  `draw` -> nothing.
- BOTH modes, once the turn counter has advanced to 3 (i.e. immediately
  after turn 2's battle): each side below `max_lives` regains one, capped at
  `max_lives` (default 6). The player's heal is mode-independent (arena
  games get it too, which is what the inline code did); the opponent's only
  exists in versus, where an opponent life total exists at all.

`turn_after_battle` is the turn number AFTER the engine's post-battle
advance (`engine.resolve_end_turn_post_battle` does `turn += 1`), because
that is the state the original inline turn-3 check read. A caller that never
touches the engine (the human reference) passes `turn + 1` for the turn
whose battle it just resolved."""

from __future__ import annotations

from dataclasses import dataclass

GAME_MODE_VERSUS = "versus"
GAME_MODE_ARENA = "arena"

OUTCOME_WIN = "win"
OUTCOME_LOSS = "loss"
OUTCOME_DRAW = "draw"

MAX_LIVES = 6
MAX_TROPHIES = 10
# "On start of turn 3, restore 1 life (up to max 6)" -- the turn counter has
# already advanced by the time this fires, so the battle that triggers it is
# turn 2's.
LIFE_RECOVERY_TURN = 3


ARENA_START_LIVES = 5
ARENA_MAX_LIVES = ARENA_START_LIVES
VERSUS_START_LIVES = MAX_LIVES


@dataclass(frozen=True)
class BattleLivesUpdate:
    """Result of `apply_battle_outcome_to_lives`. All fields are the values
    AFTER the outcome and the turn-3 recovery have both been applied.

    `life_bonus_applied` / `opponent_life_bonus_applied` exist so a caller
    can reproduce `end_turn.py`'s two `turn_start_rule:*` engine notes
    without re-deriving the condition.
    """

    lives: int
    opponent_lives: int
    trophies: int
    life_bonus_applied: bool
    opponent_life_bonus_applied: bool


def apply_battle_outcome_to_lives(
    outcome: str,
    *,
    lives: int,
    opponent_lives: int = 0,
    trophies: int = 0,
    turn_after_battle: int,
    game_mode: str = GAME_MODE_VERSUS,
    max_lives: int = MAX_LIVES,
) -> BattleLivesUpdate:
    """Apply one resolved battle's outcome to the life/trophy totals.

    Pure: no state dict, no I/O, no engine. See the module docstring for the
    rules and for what `turn_after_battle` means.

    `outcome` is the oracle's discrete verdict (`"win"`/`"loss"`/`"draw"`);
    anything else is treated as a draw (no life moves), matching the inline
    code this replaces -- `end_turn.py` rejects unknown outcomes before it
    ever gets here, so that branch is defensive only."""
    mode = str(game_mode or GAME_MODE_ARENA).strip().lower()
    verdict = str(outcome or "").strip().lower()

    lives_out = int(lives)
    opponent_lives_out = int(opponent_lives)
    trophies_out = int(trophies)

    if mode == GAME_MODE_VERSUS:
        if verdict == OUTCOME_WIN:
            opponent_lives_out = max(0, opponent_lives_out - 1)
        elif verdict == OUTCOME_LOSS:
            lives_out = max(0, lives_out - 1)
    else:
        if verdict == OUTCOME_WIN:
            trophies_out = min(MAX_TROPHIES, trophies_out + 1)
        elif verdict == OUTCOME_LOSS:
            lives_out = max(0, lives_out - 1)

    cap = int(max_lives)

    life_bonus_applied = False
    if int(turn_after_battle) == LIFE_RECOVERY_TURN and lives_out < cap:
        lives_out = min(cap, lives_out + 1)
        life_bonus_applied = True

    opponent_life_bonus_applied = False
    if (
        mode == GAME_MODE_VERSUS
        and int(turn_after_battle) == LIFE_RECOVERY_TURN
        and opponent_lives_out < cap
    ):
        opponent_lives_out = min(cap, opponent_lives_out + 1)
        opponent_life_bonus_applied = True

    return BattleLivesUpdate(
        lives=lives_out,
        opponent_lives=opponent_lives_out,
        trophies=trophies_out,
        life_bonus_applied=life_bonus_applied,
        opponent_life_bonus_applied=opponent_life_bonus_applied,
    )
