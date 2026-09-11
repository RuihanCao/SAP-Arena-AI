"""Gymnasium wrapper over TrainingEnv for PPO training."""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any, NamedTuple

import numpy as np

from ..catalog import load_turtle_catalog, tier_for_turn
from .env import TrainingEnv
from .observation import DEFAULT_OBSERVATION_MODE, build_state_encoder

try:  # pragma: no cover - optional training dependency
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None
    spaces = None


_ODD_TURN_LEVELUP_BONUS_TURNS = {3, 5, 7, 9}
_EARLY_BOARD_FILL_BONUS_TURNS = {1, 2}

REWARD_MODE_LEGACY = "legacy"
REWARD_MODE_PBRS_V1 = "pbrs_v1"


REWARD_MODE_VERSUS_LIVES = "versus_lives"


REWARD_MODE_VERSUS_LIVES_KSIM = "versus_lives_ksim"
REWARD_MODE_CHOICES = (
    REWARD_MODE_LEGACY,
    REWARD_MODE_PBRS_V1,
    REWARD_MODE_VERSUS_LIVES,
    REWARD_MODE_VERSUS_LIVES_KSIM,
)
DEFAULT_REWARD_MODE = REWARD_MODE_LEGACY
DEFAULT_VERSUS_TURN_SCALE = 1.0
DEFAULT_VERSUS_TERMINAL_SCALE = 5.0
DEFAULT_KSIM_K = 16


VERSUS_TURN12_MODE_FULL = "full"
VERSUS_TURN12_MODE_STAKE = "stake"
VERSUS_TURN12_MODE_CHOICES = (VERSUS_TURN12_MODE_FULL, VERSUS_TURN12_MODE_STAKE)
DEFAULT_VERSUS_TURN12_MODE = VERSUS_TURN12_MODE_FULL

VERSUS_TURN12_STAKE_TURNS = (1, 2)


def normalize_versus_turn12_mode(
    value: str | None, *, default: str = DEFAULT_VERSUS_TURN12_MODE
) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = str(default).strip().lower()
    if raw not in VERSUS_TURN12_MODE_CHOICES:
        allowed = ",".join(VERSUS_TURN12_MODE_CHOICES)
        raise ValueError(f"invalid_versus_turn12_mode:{value}:allowed={allowed}")
    return raw


class KsimFought(NamedTuple):
    """Ksimfought."""

    term: float
    player_wins: int
    opponent_wins: int
    draws: int
    p_hat: float
    q_hat: float
    d_hat: float
    debug: dict[str, float]


def ksim_turn12_stake_scale(*, p_hat: float, q_hat: float) -> float:
    """Multiplying the plain EV term `turn_scale * (p_hat - q_hat)` by this gives
    `turn_scale * 0.5 * (p_hat + q_hat) * (p_hat - q_hat)`, i.e. half the real
    two-turn stake `(p+q)(p-q)`. Draws are absorbed automatically: they enter
    neither p_hat nor q_hat, so a draw-heavy battle prices near zero."""
    return 0.5 * (float(p_hat) + float(q_hat))


def ksim_lives_outcome(
    *,
    player_wins: int,
    opponent_wins: int,
    k: int,
    baseline_player_wins: int | None = None,
    baseline_opponent_wins: int | None = None,
    turn_scale: float = DEFAULT_VERSUS_TURN_SCALE,
) -> float:
    """Pure math of the ksim reward term (unit-tested separately from the
    oracle plumbing): expected lives outcome of the fought battle, minus the
    optional noop-baseline expectation, scaled. Draws contribute 0 on both
    sides, matching versus lives rules (a draw costs nobody a life)."""
    kk = max(1, int(k))
    ev = (float(player_wins) - float(opponent_wins)) / float(kk)
    if baseline_player_wins is not None and baseline_opponent_wins is not None:
        ev -= (float(baseline_player_wins) - float(baseline_opponent_wins)) / float(kk)
    return float(turn_scale) * ev


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))


def normalize_reward_mode(value: str | None, *, default: str = DEFAULT_REWARD_MODE) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = str(default).strip().lower()
    aliases = {
        "pbrs": REWARD_MODE_PBRS_V1,
        "pbrs1": REWARD_MODE_PBRS_V1,
        "pbrs_v1": REWARD_MODE_PBRS_V1,
        "versus": REWARD_MODE_VERSUS_LIVES,
        "lives": REWARD_MODE_VERSUS_LIVES,
        "ksim": REWARD_MODE_VERSUS_LIVES_KSIM,
        "crn": REWARD_MODE_VERSUS_LIVES_KSIM,
        "versus_lives_crn": REWARD_MODE_VERSUS_LIVES_KSIM,
    }
    mode = aliases.get(raw, raw)
    if mode not in REWARD_MODE_CHOICES:
        allowed = ",".join(REWARD_MODE_CHOICES)
        raise ValueError(f"invalid_reward_mode:{value}:allowed={allowed}")
    return mode


def _team_pet_count(state: dict[str, Any]) -> int:
    team = state.get("team")
    if not isinstance(team, list):
        return 0
    return sum(1 for slot in team if isinstance(slot, dict) and slot.get("pet_id"))


def _iter_team_slots(state: dict[str, Any]) -> list[dict[str, Any]]:
    team = state.get("team")
    if not isinstance(team, list):
        return []
    return [slot for slot in team if isinstance(slot, dict)]


def _shop_slot_for_index(state: dict[str, Any], shop_index: int) -> dict[str, Any] | None:
    for slot in state.get("shop", []):
        if not isinstance(slot, dict):
            continue
        if _safe_int(slot.get("shop_index"), -1) == int(shop_index):
            return slot
    return None


def _count_frozen_shop_slots(state: dict[str, Any]) -> int:
    shop = state.get("shop")
    if not isinstance(shop, list):
        return 0
    return sum(1 for slot in shop if isinstance(slot, dict) and bool(slot.get("frozen", False)))


def _count_level_ups(prev_state: dict[str, Any], curr_state: dict[str, Any]) -> int:
    prev_team = _iter_team_slots(prev_state)
    curr_team = _iter_team_slots(curr_state)
    events = 0
    for idx in range(min(len(prev_team), len(curr_team), 5)):
        prev_slot = prev_team[idx]
        curr_slot = curr_team[idx]
        if curr_slot.get("pet_id") is None:
            continue
        delta = _safe_int(curr_slot.get("level"), 1) - _safe_int(prev_slot.get("level"), 1)
        if delta > 0:
            events += int(delta)
    return int(events)


@lru_cache(maxsize=1)
def _pet_tier_by_id() -> dict[str, int]:
    catalog = load_turtle_catalog()
    by_tier = catalog.get("pets", {}).get("by_tier", {}) or {}
    out: dict[str, int] = {}
    for tier_str, pet_ids in by_tier.items():
        tier = _safe_int(tier_str, 1)
        if not isinstance(pet_ids, list):
            continue
        for pet_id in pet_ids:
            key = str(pet_id)
            if key and key not in out:
                out[key] = int(tier)
    return out


def _has_copy_target(prev_state: dict[str, Any], pet_id: str) -> bool:
    target = str(pet_id)
    if not target:
        return False
    for slot in _iter_team_slots(prev_state):
        if str(slot.get("pet_id", "")) != target:
            continue
        if _safe_int(slot.get("level"), 1) >= 3:
            continue
        return True
    return False


def _top2_perm_power(state: dict[str, Any]) -> int:
    base_stats = load_turtle_catalog().get("pets", {}).get("base_stats", {}) or {}
    powers: list[int] = []
    for slot in _iter_team_slots(state):
        pet_id = str(slot.get("pet_id", "") or "")
        if not pet_id:
            continue
        base = base_stats.get(pet_id, {}) if isinstance(base_stats, dict) else {}
        base_attack = _safe_int((base or {}).get("attack"), _safe_int(slot.get("attack"), 0))
        base_health = _safe_int((base or {}).get("health"), _safe_int(slot.get("health"), 0))
        perm_attack = _safe_int(slot.get("perm_attack"), _safe_int(slot.get("attack"), 0))
        perm_health = _safe_int(slot.get("perm_health"), _safe_int(slot.get("health"), 0))
        add_power = (perm_attack - base_attack) + (perm_health - base_health)
        powers.append(max(0, int(add_power)))
    powers.sort(reverse=True)
    return int(sum(powers[:2]))


def _ensure_turn_tracker(tracker: dict[str, Any], *, turn: int) -> dict[str, float]:
    curr_turn = _safe_int(tracker.get("turn"), -1)
    if curr_turn != int(turn):
        tracker["turn"] = int(turn)
        tracker["acc"] = {}
    acc = tracker.get("acc")
    if not isinstance(acc, dict):
        acc = {}
        tracker["acc"] = acc
    return acc


def _apply_turn_capped_increment(
    tracker: dict[str, Any],
    *,
    turn: int,
    key: str,
    delta: float,
    min_cap: float | None = None,
    max_cap: float | None = None,
) -> float:
    if abs(float(delta)) <= 1e-12:
        return 0.0
    acc = _ensure_turn_tracker(tracker, turn=int(turn))
    current = float(acc.get(key, 0.0))
    target = current + float(delta)
    bounded = float(target)
    if max_cap is not None:
        bounded = min(bounded, float(max_cap))
    if min_cap is not None:
        bounded = max(bounded, float(min_cap))
    applied = float(bounded - current)
    acc[key] = float(bounded)
    return applied


def _reward_terms_from_transition_legacy(
    prev_state: dict[str, Any],
    curr_state: dict[str, Any],
    *,
    action: dict[str, Any] | None,
    action_type: str | None,
    turn_tracker: dict[str, Any],
    step_ok: bool,
    done: bool,
) -> dict[str, float]:
    action_kind = str(action_type or "").strip().upper()
    prev_turn = _safe_int(prev_state.get("turn"), 1)
    trophies_delta = _safe_int(curr_state.get("trophies"), 0) - _safe_int(prev_state.get("trophies"), 0)
    lives_delta = _safe_int(curr_state.get("lives"), 0) - _safe_int(prev_state.get("lives"), 0)
    curr_pet_count = _team_pet_count(curr_state)

    reward_terms: dict[str, float] = {
        "trophy_term": 1.5 * float(trophies_delta),
        "lives_term": 0.5 * float(lives_delta),
        "step_term": 0.0,
        "illegal_term": 0.0,
        "terminal_term": 0.0,
        "end_turn_board_floor_term": 0.0,
        "freeze_unfreeze_term": 0.0,
        "reorder_term": 0.0,
        "roll_frozen_penalty": 0.0,
        "roll_10plus_penalty": 0.0,
        "early_board_fill_bonus": 0.0,
        "odd_turn_levelup_bonus": 0.0,
        "above_tier_buy_bonus": 0.0,
        "low_tier_noncopy_buy_penalty": 0.0,
        "top2_perm_power_bonus": 0.0,
    }

    if not step_ok:
        reward_terms["illegal_term"] = -0.1

    if done:
        if _safe_int(curr_state.get("trophies"), 0) >= 7:
            reward_terms["terminal_term"] = 20.0
        elif _safe_int(curr_state.get("lives"), 0) <= 0:
            reward_terms["terminal_term"] = -20.0

    if not step_ok:
        return reward_terms

    if action_kind == "END_TURN":
        if curr_pet_count <= 0:
            reward_terms["end_turn_board_floor_term"] = -1.0
        elif curr_pet_count == 1:
            reward_terms["end_turn_board_floor_term"] = -0.25

    if action_kind in {"FREEZE", "UNFREEZE"}:
        reward_terms["freeze_unfreeze_term"] = -0.02

    if action_kind == "REORDER":
        reward_terms["reorder_term"] = -0.02

    if action_kind == "ROLL":
        roll_acc = _ensure_turn_tracker(turn_tracker, turn=prev_turn)
        roll_count = _safe_int(roll_acc.get("_roll_count"), 0) + 1
        roll_acc["_roll_count"] = int(roll_count)
        frozen_before = _count_frozen_shop_slots(prev_state)
        reward_terms["roll_frozen_penalty"] = -min(0.03, 0.005 * float(frozen_before))
        if int(roll_count) >= 10:
            reward_terms["roll_10plus_penalty"] = -0.10

    board_fill_delta = max(0, int(curr_pet_count - _team_pet_count(prev_state)))
    if prev_turn in _EARLY_BOARD_FILL_BONUS_TURNS and board_fill_delta > 0:
        reward_terms["early_board_fill_bonus"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="early_board_fill_bonus",
            delta=0.25 * float(board_fill_delta),
            max_cap=0.45,
        )

    level_up_events = _count_level_ups(prev_state, curr_state)
    if prev_turn in _ODD_TURN_LEVELUP_BONUS_TURNS and level_up_events > 0:
        raw = 0.05 * float(level_up_events)
        reward_terms["odd_turn_levelup_bonus"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="odd_turn_levelup_bonus",
            delta=raw,
            max_cap=0.10,
        )

    if action_kind in {"BUY_PET", "BUY_COMBINE"} and isinstance(action, dict):
        shop_index = _safe_int(action.get("shop_index"), -1)
        shop_slot = _shop_slot_for_index(prev_state, shop_index)
        if shop_slot is not None and str(shop_slot.get("slot_type", "")) == "pet":
            pet_id = str(shop_slot.get("item_id", "") or "")
            pet_tier = _pet_tier_by_id().get(pet_id)
            current_tier = int(tier_for_turn(int(prev_turn)))
            if pet_tier is not None:
                if int(pet_tier) > current_tier:
                    reward_terms["above_tier_buy_bonus"] = _apply_turn_capped_increment(
                        turn_tracker,
                        turn=prev_turn,
                        key="above_tier_buy_bonus",
                        delta=0.02,
                        max_cap=0.06,
                    )
                if int(prev_turn) >= 7 and int(pet_tier) < current_tier:
                    copy_target = action_kind == "BUY_COMBINE" or _has_copy_target(prev_state, pet_id)
                    if not copy_target:
                        reward_terms["low_tier_noncopy_buy_penalty"] = _apply_turn_capped_increment(
                            turn_tracker,
                            turn=prev_turn,
                            key="low_tier_noncopy_buy_penalty",
                            delta=-0.02,
                            min_cap=-0.06,
                        )

    perm_delta = _top2_perm_power(curr_state) - _top2_perm_power(prev_state)
    if perm_delta > 0:
        raw_perm_bonus = 0.01 * float(perm_delta)
        reward_terms["top2_perm_power_bonus"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="top2_perm_power_bonus",
            delta=raw_perm_bonus,
            max_cap=0.08,
        )

    return reward_terms


def _versus_opponent_lives(state: dict[str, Any], default: int = 0) -> int:
    meta = state.get("meta")
    versus = meta.get("versus") if isinstance(meta, dict) else None
    if not isinstance(versus, dict):
        return int(default)
    return _safe_int(versus.get("opponent_lives"), int(default))


def _reward_terms_from_transition_versus_lives(
    prev_state: dict[str, Any],
    curr_state: dict[str, Any],
    *,
    step_ok: bool,
    done: bool,
    turn_scale: float,
    terminal_scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """- `lives_outcome_term` = turn_scale * ((opponent lives lost) - (own lives
      lost)) across this transition. Lives only move inside END_TURN's battle
      resolution (including the turn-3 recovery, which enters symmetrically as
      a negative "loss"), so every non-END_TURN action's reward is exactly 0
      without any action-type special-casing.
    - `terminal_term` = +terminal_scale when the game ends with the opponent
      at 0 lives, -terminal_scale when it ends with the player at 0. An
      episode ended by the 30-turn cap (nobody at 0) gets 0."""
    prev_own = _safe_int(prev_state.get("lives"), 0)
    curr_own = _safe_int(curr_state.get("lives"), 0)
    prev_opp = _versus_opponent_lives(prev_state, default=0)
    curr_opp = _versus_opponent_lives(curr_state, default=0)

    reward_terms: dict[str, float] = {
        "lives_outcome_term": 0.0,
        "terminal_term": 0.0,
    }
    reward_debug: dict[str, float] = {
        "versus_prev_lives": float(prev_own),
        "versus_curr_lives": float(curr_own),
        "versus_prev_opponent_lives": float(prev_opp),
        "versus_curr_opponent_lives": float(curr_opp),
        "versus_turn_scale": float(turn_scale),
        "versus_terminal_scale": float(terminal_scale),


        "versus_raw_lives_delta": 0.0,
    }

    if step_ok:
        own_lost = float(prev_own - curr_own)
        opp_lost = float(prev_opp - curr_opp)
        raw_lives_delta = opp_lost - own_lost
        reward_debug["versus_raw_lives_delta"] = float(raw_lives_delta)
        reward_terms["lives_outcome_term"] = float(turn_scale) * raw_lives_delta

    if done:
        if curr_opp <= 0:
            reward_terms["terminal_term"] = float(terminal_scale)
        elif curr_own <= 0:
            reward_terms["terminal_term"] = -float(terminal_scale)

    return reward_terms, reward_debug


def _state_total_board_power(state: dict[str, Any]) -> int:
    total = 0
    for slot in _iter_team_slots(state):
        if not slot.get("pet_id"):
            continue
        total += max(0, _safe_int(slot.get("attack"), 0))
        total += max(0, _safe_int(slot.get("health"), 0))
    return int(total)


def _state_copies_focus_top3_norm(state: dict[str, Any], *, beta: float) -> float:
    invested_by_pet: dict[str, float] = {}
    for slot in _iter_team_slots(state):
        pet_id = str(slot.get("pet_id") or "")
        if not pet_id:
            continue
        exp = max(0, _safe_int(slot.get("exp"), 0))
        invested = 1.0 + float(exp)
        invested_by_pet[pet_id] = float(invested_by_pet.get(pet_id, 0.0) + invested)

    line_scores: list[float] = []
    beta_value = max(0.0, float(beta))
    for invested_copies in invested_by_pet.values():
        c = max(0.0, float(invested_copies))
        lvl2_focus = min(c, 3.0) / 3.0
        lvl3_tail = beta_value * (min(max(c - 3.0, 0.0), 3.0) / 3.0)
        line_scores.append(float(lvl2_focus + lvl3_tail))
    line_scores.sort(reverse=True)
    top3_sum = float(sum(line_scores[:3]))
    max_sum = max(1e-6, 3.0 * (1.0 + beta_value))
    return _clip01(top3_sum / max_sum)


def _state_above_tier_on_board_count(state: dict[str, Any], *, turn: int) -> int:
    current_tier = max(1, _safe_int(tier_for_turn(max(1, int(turn))), 1))
    tiers = _pet_tier_by_id()
    count = 0
    for slot in _iter_team_slots(state):
        pet_id = str(slot.get("pet_id") or "")
        if not pet_id:
            continue
        pet_tier = _safe_int(tiers.get(pet_id), 0)
        if int(pet_tier) > int(current_tier):
            count += 1
    return int(count)


def _pbrs_turn_weights(turn: int) -> tuple[float, float, float]:
    t = max(1, int(turn))
    if t <= 2:
        return 0.25, 0.50, 0.25
    if t <= 8:
        return 0.50, 0.25, 0.25
    return 0.30, 0.45, 0.25


def _pbrs_phi(state: dict[str, Any], *, beta: float) -> tuple[float, dict[str, float]]:
    turn = max(1, _safe_int(state.get("turn"), 1))
    w_perm, w_power, w_copies = _pbrs_turn_weights(turn)
    phi_perm = _clip01(float(_top2_perm_power(state)) / max(1.0, 8.0 + (4.0 * float(turn))))
    phi_power = _clip01(float(_state_total_board_power(state)) / max(1.0, 20.0 + (12.0 * float(turn))))
    phi_copies = _state_copies_focus_top3_norm(state, beta=float(beta))
    phi = _clip01((w_perm * phi_perm) + (w_power * phi_power) + (w_copies * phi_copies))
    return float(phi), {
        "turn": float(turn),
        "w_perm": float(w_perm),
        "w_power": float(w_power),
        "w_copies": float(w_copies),
        "phi_perm": float(phi_perm),
        "phi_power": float(phi_power),
        "phi_copies": float(phi_copies),
        "phi": float(phi),
    }


def _reward_terms_from_transition_pbrs(
    prev_state: dict[str, Any],
    curr_state: dict[str, Any],
    *,
    action_type: str | None,
    turn_tracker: dict[str, Any],
    step_ok: bool,
    done: bool,
    pbrs_alpha: float,
    pbrs_gamma: float,
    pbrs_beta: float,
    pbrs_enable_aux: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    action_kind = str(action_type or "").strip().upper()
    prev_turn = max(1, _safe_int(prev_state.get("turn"), 1))
    trophies_delta = _safe_int(curr_state.get("trophies"), 0) - _safe_int(prev_state.get("trophies"), 0)
    lives_delta = _safe_int(curr_state.get("lives"), 0) - _safe_int(prev_state.get("lives"), 0)
    curr_pet_count = _team_pet_count(curr_state)

    reward_terms: dict[str, float] = {
        "trophy_term": 1.5 * float(trophies_delta),
        "lives_term": 0.5 * float(lives_delta),
        "illegal_term": 0.0,
        "terminal_term": 0.0,
        "end_turn_board_floor_term": 0.0,
        "pbrs_term": 0.0,
        "above_tier_on_board_bonus": 0.0,
        "reorder_penalty": 0.0,
        "roll_frozen_penalty": 0.0,
        "roll_10plus_penalty": 0.0,
    }
    reward_debug: dict[str, float] = {
        "pbrs_alpha": float(pbrs_alpha),
        "pbrs_gamma": float(pbrs_gamma),
        "pbrs_beta": float(pbrs_beta),
        "pbrs_enabled": 1.0,
    }

    if not step_ok:
        reward_terms["illegal_term"] = -0.1

    if done:
        if _safe_int(curr_state.get("trophies"), 0) >= 7:
            reward_terms["terminal_term"] = 20.0
        elif _safe_int(curr_state.get("lives"), 0) <= 0:
            reward_terms["terminal_term"] = -20.0

    if not step_ok:
        return reward_terms, reward_debug

    if action_kind == "END_TURN":
        if curr_pet_count <= 0:
            reward_terms["end_turn_board_floor_term"] = -1.0
        elif curr_pet_count == 1:
            reward_terms["end_turn_board_floor_term"] = -0.25

    if action_kind in {"ROLL", "END_TURN"}:
        phi_prev, phi_prev_debug = _pbrs_phi(prev_state, beta=float(pbrs_beta))
        phi_next, phi_next_debug = _pbrs_phi(curr_state, beta=float(pbrs_beta))
        pbrs_delta = (float(pbrs_gamma) * float(phi_next)) - float(phi_prev)
        reward_terms["pbrs_term"] = float(pbrs_alpha) * float(pbrs_delta)
        reward_debug.update(
            {
                "phi_prev": float(phi_prev),
                "phi_next": float(phi_next),
                "phi_delta": float(pbrs_delta),
                "phi_prev_perm": float(phi_prev_debug.get("phi_perm", 0.0)),
                "phi_prev_power": float(phi_prev_debug.get("phi_power", 0.0)),
                "phi_prev_copies": float(phi_prev_debug.get("phi_copies", 0.0)),
                "phi_next_perm": float(phi_next_debug.get("phi_perm", 0.0)),
                "phi_next_power": float(phi_next_debug.get("phi_power", 0.0)),
                "phi_next_copies": float(phi_next_debug.get("phi_copies", 0.0)),
            }
        )

    if not bool(pbrs_enable_aux):
        return reward_terms, reward_debug

    if action_kind == "REORDER":
        reward_terms["reorder_penalty"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="pbrs_reorder_penalty",
            delta=-0.003,
            min_cap=-0.015,
        )

    if action_kind == "ROLL":
        roll_acc = _ensure_turn_tracker(turn_tracker, turn=prev_turn)
        roll_count = _safe_int(roll_acc.get("_roll_count"), 0) + 1
        roll_acc["_roll_count"] = int(roll_count)
        frozen_before = _count_frozen_shop_slots(prev_state)
        per_roll = -min(0.01, 0.0015 * float(max(0, frozen_before)))
        reward_terms["roll_frozen_penalty"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="pbrs_roll_frozen_penalty",
            delta=per_roll,
            min_cap=-0.03,
        )
        if int(roll_count) >= 10:
            reward_terms["roll_10plus_penalty"] = _apply_turn_capped_increment(
                turn_tracker,
                turn=prev_turn,
                key="pbrs_roll_10plus_penalty",
                delta=-0.02,
                min_cap=-0.08,
            )

    if action_kind == "END_TURN":
        above_tier_count = _state_above_tier_on_board_count(curr_state, turn=prev_turn)
        reward_terms["above_tier_on_board_bonus"] = _apply_turn_capped_increment(
            turn_tracker,
            turn=prev_turn,
            key="pbrs_above_tier_on_board_bonus",
            delta=0.005 * float(max(0, above_tier_count)),
            max_cap=0.02,
        )
        reward_debug["above_tier_on_board_count"] = float(max(0, above_tier_count))

    return reward_terms, reward_debug


def _reward_terms_from_transition(
    prev_state: dict[str, Any],
    curr_state: dict[str, Any],
    *,
    action: dict[str, Any] | None,
    action_type: str | None,
    turn_tracker: dict[str, Any],
    step_ok: bool,
    done: bool,
) -> dict[str, float]:
    # Backward-compatible helper retained for existing unit tests and Step 5 behavior checks.
    return _reward_terms_from_transition_legacy(
        prev_state,
        curr_state,
        action=action,
        action_type=action_type,
        turn_tracker=turn_tracker,
        step_ok=step_ok,
        done=done,
    )


def _reward_terms_with_mode(
    prev_state: dict[str, Any],
    curr_state: dict[str, Any],
    *,
    action: dict[str, Any] | None,
    action_type: str | None,
    turn_tracker: dict[str, Any],
    step_ok: bool,
    done: bool,
    reward_mode: str,
    pbrs_alpha: float,
    pbrs_gamma: float,
    pbrs_beta: float,
    pbrs_enable_aux: bool,
    versus_turn_scale: float = DEFAULT_VERSUS_TURN_SCALE,
    versus_terminal_scale: float = DEFAULT_VERSUS_TERMINAL_SCALE,
) -> tuple[dict[str, float], dict[str, float]]:
    mode = normalize_reward_mode(reward_mode, default=DEFAULT_REWARD_MODE)
    if mode in (REWARD_MODE_VERSUS_LIVES, REWARD_MODE_VERSUS_LIVES_KSIM):
        # ksim shares the state-side terms (terminal, debug); the wrapper
        # overrides the lives-outcome term with the oracle k-sim estimate.
        return _reward_terms_from_transition_versus_lives(
            prev_state,
            curr_state,
            step_ok=bool(step_ok),
            done=bool(done),
            turn_scale=float(versus_turn_scale),
            terminal_scale=float(versus_terminal_scale),
        )
    if mode == REWARD_MODE_PBRS_V1:
        return _reward_terms_from_transition_pbrs(
            prev_state,
            curr_state,
            action_type=action_type,
            turn_tracker=turn_tracker,
            step_ok=bool(step_ok),
            done=bool(done),
            pbrs_alpha=float(pbrs_alpha),
            pbrs_gamma=float(pbrs_gamma),
            pbrs_beta=float(pbrs_beta),
            pbrs_enable_aux=bool(pbrs_enable_aux),
        )
    return (
        _reward_terms_from_transition_legacy(
            prev_state,
            curr_state,
            action=action,
            action_type=action_type,
            turn_tracker=turn_tracker,
            step_ok=bool(step_ok),
            done=bool(done),
        ),
        {"reward_mode": 0.0},
    )


if gym is not None:  # pragma: no cover - tested via import + simple runtime smoke only

    class SapPpoGymEnv(gym.Env):
        """Discrete-action maskable RL environment."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            core_env: TrainingEnv,
            encoder: Any | None = None,
            *,
            reward_mode: str = DEFAULT_REWARD_MODE,
            pbrs_alpha: float = 0.25,
            pbrs_gamma: float = 0.99,
            pbrs_beta: float = 0.20,
            pbrs_enable_aux: bool = True,
            versus_turn_scale: float = DEFAULT_VERSUS_TURN_SCALE,
            versus_terminal_scale: float = DEFAULT_VERSUS_TERMINAL_SCALE,
            versus_turn12_mode: str = DEFAULT_VERSUS_TURN12_MODE,
            ksim_k: int = DEFAULT_KSIM_K,
            ksim_noop_baseline: bool = True,
            ksim_battle_fn: Any | None = None,
        ):
            super().__init__()
            self.core_env = core_env
            default_max_turn = int(core_env.max_turn) if core_env.max_turn is not None else 15
            self.encoder = encoder or build_state_encoder(
                observation_mode=DEFAULT_OBSERVATION_MODE,
                max_turn=default_max_turn,
            )
            self.reward_mode = normalize_reward_mode(reward_mode, default=DEFAULT_REWARD_MODE)
            self.pbrs_alpha = float(pbrs_alpha)
            self.pbrs_gamma = float(pbrs_gamma)
            self.pbrs_beta = float(pbrs_beta)
            self.pbrs_enable_aux = bool(pbrs_enable_aux)
            self.versus_turn_scale = float(versus_turn_scale)
            self.versus_terminal_scale = float(versus_terminal_scale)


            self.ksim_k = max(1, int(ksim_k))
            self.ksim_noop_baseline = bool(ksim_noop_baseline)


            self.versus_turn12_mode = normalize_versus_turn12_mode(versus_turn12_mode)
            if self.versus_turn12_mode == VERSUS_TURN12_MODE_STAKE and self.ksim_noop_baseline:
                raise ValueError(
                    "versus_turn12_mode_stake_requires_no_ksim_noop_baseline: "
                    f"versus_turn12_mode={self.versus_turn12_mode!r}, "
                    f"ksim_noop_baseline={self.ksim_noop_baseline!r}"
                )
            self._ksim_battle_fn = ksim_battle_fn
            self._turn_start_team = copy.deepcopy(core_env.state.get("team", []))
            if self.reward_mode in (REWARD_MODE_VERSUS_LIVES, REWARD_MODE_VERSUS_LIVES_KSIM):
                # Loud construction-time guard: this reward reads versus lives
                # bookkeeping; on a non-versus fixture it would silently train
                # on all-zero rewards.
                meta = core_env.state.get("meta") if isinstance(core_env.state, dict) else None
                game_mode = str((meta or {}).get("game_mode") or "").strip().lower()
                versus = (meta or {}).get("versus") if isinstance(meta, dict) else None
                if game_mode != "versus" or not isinstance(versus, dict) or versus.get("opponent_lives") is None:
                    raise ValueError(
                        "versus_lives_reward_requires_versus_fixture: "
                        f"game_mode={game_mode!r}, meta.versus.opponent_lives="
                        f"{(versus or {}).get('opponent_lives')!r}"
                    )
            self.action_space = spaces.Discrete(self.core_env.action_space_size)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.encoder.size,), dtype=np.float32)
            self._last_state = copy.deepcopy(self.core_env.state)
            self._reward_turn_tracker: dict[str, Any] = {
                "turn": _safe_int(self.core_env.state.get("turn"), 1),
                "acc": {},
            }


            self._episode_index = 0

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            del options
            state = self.core_env.reset(seed=seed, episode_index=self._episode_index)
            self._episode_index += 1
            self._last_state = copy.deepcopy(state)
            self._reward_turn_tracker = {
                "turn": _safe_int(state.get("turn"), 1),
                "acc": {},
            }
            self._turn_start_team = copy.deepcopy(state.get("team", []))
            obs = self.encoder.encode(state)
            info = {"legal_action_mask": self.action_masks()}
            return obs, info

        def _ksim_estimate(self, fought_config: dict[str, Any]) -> KsimFought | None:
            """k-sim expected lives outcome for the battle that just resolved."""
            fn = self._ksim_battle_fn
            if fn is None:
                from ..oracles.sap_calc_battle_oracle import run_battle_oracle_with_config

                fn = self._ksim_battle_fn = run_battle_oracle_with_config
            cfg = copy.deepcopy(fought_config)
            cfg["simulationCount"] = int(self.ksim_k)
            cfg["logsEnabled"] = False
            out = fn(cfg)
            result = out.get("result") if isinstance(out, dict) else None
            if not (isinstance(out, dict) and out.get("ok") and isinstance(result, dict)):
                return None
            p_wins = _safe_int(result.get("playerWins"), 0)
            o_wins = _safe_int(result.get("opponentWins"), 0)


            if result.get("draws") is None:
                draws = max(0, int(self.ksim_k) - p_wins - o_wins)
            else:
                draws = max(0, _safe_int(result.get("draws"), 0))
            kk = float(max(1, int(self.ksim_k)))
            p_hat = float(p_wins) / kk
            q_hat = float(o_wins) / kk
            d_hat = float(draws) / kk
            debug: dict[str, float] = {
                "ksim_k": float(self.ksim_k),
                "ksim_player_wins": float(p_wins),
                "ksim_opponent_wins": float(o_wins),
                "ksim_draws": float(draws),
                "ksim_p_hat": p_hat,
                "ksim_q_hat": q_hat,
                "ksim_d_hat": d_hat,
            }
            base_p: int | None = None
            base_o: int | None = None
            if self.ksim_noop_baseline:
                from ..oracles.sap_calc_battle_oracle import team_to_pet_configs

                base_cfg = copy.deepcopy(cfg)
                base_cfg["playerPets"] = team_to_pet_configs(self._turn_start_team)
                base_out = fn(base_cfg)
                base_result = base_out.get("result") if isinstance(base_out, dict) else None
                if isinstance(base_out, dict) and base_out.get("ok") and isinstance(base_result, dict):
                    base_p = _safe_int(base_result.get("playerWins"), 0)
                    base_o = _safe_int(base_result.get("opponentWins"), 0)
                    debug["ksim_baseline_player_wins"] = float(base_p)
                    debug["ksim_baseline_opponent_wins"] = float(base_o)
                else:
                    debug["ksim_baseline_failed"] = 1.0
            term = ksim_lives_outcome(
                player_wins=p_wins,
                opponent_wins=o_wins,
                k=self.ksim_k,
                baseline_player_wins=base_p,
                baseline_opponent_wins=base_o,
                turn_scale=self.versus_turn_scale,
            )
            return KsimFought(
                term=float(term),
                player_wins=int(p_wins),
                opponent_wins=int(o_wins),
                draws=int(draws),
                p_hat=p_hat,
                q_hat=q_hat,
                d_hat=d_hat,
                debug=debug,
            )

        def step(self, action: int):
            prev = copy.deepcopy(self.core_env.state)
            out = self.core_env.step(int(action))
            curr = copy.deepcopy(out["state"])
            done = bool(out["done"])
            action_type: str | None = None
            action_payload: dict[str, Any] | None = None
            transition = out.get("transition")
            if isinstance(transition, dict):
                payload = transition.get("action")
                if isinstance(payload, dict):
                    action_payload = copy.deepcopy(payload)
            if action_payload is None:
                try:
                    decoded = self.core_env.decode_action(int(action))
                except Exception:
                    decoded = None
                if isinstance(decoded, dict):
                    action_payload = copy.deepcopy(decoded)
            if isinstance(action_payload, dict):
                action_type = str(action_payload.get("type", "")).strip() or None
            reward_terms, reward_debug = _reward_terms_with_mode(
                prev,
                curr,
                action=action_payload,
                action_type=action_type,
                turn_tracker=self._reward_turn_tracker,
                step_ok=bool(out["ok"]),
                done=done,
                reward_mode=self.reward_mode,
                pbrs_alpha=float(self.pbrs_alpha),
                pbrs_gamma=float(self.pbrs_gamma),
                pbrs_beta=float(self.pbrs_beta),
                pbrs_enable_aux=bool(self.pbrs_enable_aux),
                versus_turn_scale=float(self.versus_turn_scale),
                versus_terminal_scale=float(self.versus_terminal_scale),
            )
            info = dict(out.get("info", {}))


            raw_lives_outcome_term: float | None = None
            if self.reward_mode in (REWARD_MODE_VERSUS_LIVES, REWARD_MODE_VERSUS_LIVES_KSIM):


                raw_lives_outcome_term = float(reward_debug.get("versus_raw_lives_delta", 0.0))


            if (
                self.reward_mode == REWARD_MODE_VERSUS_LIVES_KSIM
                and bool(out.get("ok"))
                and isinstance(info.get("battle"), dict)
                and isinstance(info["battle"].get("config"), dict)
            ):
                if float(self.versus_turn_scale) == 0.0:


                    reward_debug["ksim_skipped_zero_turn_scale"] = 1.0
                else:
                    est = self._ksim_estimate(info["battle"]["config"])
                    if est is None:
                        reward_debug["ksim_fallback_single_draw"] = 1.0
                    else:
                        reward_debug.update(est.debug)
                        ksim_term = float(est.term)


                        if (
                            self.versus_turn12_mode == VERSUS_TURN12_MODE_STAKE
                            and _safe_int(prev.get("turn"), 1) in VERSUS_TURN12_STAKE_TURNS
                        ):
                            stake_scale = ksim_turn12_stake_scale(p_hat=est.p_hat, q_hat=est.q_hat)
                            ksim_term *= stake_scale
                            reward_debug["ksim_turn12_stake_scale"] = float(stake_scale)
                        reward_terms["lives_outcome_term"] = float(ksim_term)
            # Turn boundary: refresh the noop-baseline board only AFTER the
            # just-ended turn's reward used the previous turn-start team.
            if _safe_int(curr.get("turn"), 1) != _safe_int(prev.get("turn"), 1):
                self._turn_start_team = copy.deepcopy(curr.get("team", []))
            reward = float(sum(float(v) for v in reward_terms.values()))
            obs = self.encoder.encode(curr)
            info["step_ok"] = bool(out.get("ok"))


            info["action_index"] = int(action)


            if raw_lives_outcome_term is not None:
                reward_terms["lives_raw_term"] = float(raw_lives_outcome_term)
            info["reward_terms"] = reward_terms
            info["reward_debug"] = reward_debug
            info["reward_mode"] = str(self.reward_mode)
            info["legal_action_mask"] = self.action_masks()
            self._last_state = copy.deepcopy(curr)
            return obs, float(reward), done, False, info

        def action_masks(self) -> np.ndarray:
            return np.asarray(self.core_env.legal_action_mask(), dtype=np.bool_)

        def render(self):
            return None

        def close(self):
            return None

else:

    class SapPpoGymEnv:  # pragma: no cover - deterministic error path
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("gymnasium_not_installed: install with `pip install -e .[train]`")
