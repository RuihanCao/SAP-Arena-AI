"""Training/evaluation metrics helpers for Step 3 PPO."""

from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_core_env(env: Any) -> Any:
    if hasattr(env, "core_env"):
        return getattr(env, "core_env")
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "core_env"):
        return getattr(inner, "core_env")
    inner_env = getattr(inner, "env", None)
    if inner_env is not None and hasattr(inner_env, "core_env"):
        return getattr(inner_env, "core_env")
    return None


def _coerce_action_index(action: Any) -> int | None:
    try:
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)
    except Exception:
        return None


def _action_type_for_index(core_env: Any, action_index: int | None) -> str:
    if core_env is None or action_index is None:
        return "UNKNOWN"
    if not hasattr(core_env, "decode_action"):
        return "UNKNOWN"
    try:
        action_obj = core_env.decode_action(int(action_index))
    except Exception:
        return "UNKNOWN"
    action_type = str(action_obj.get("type", "")).strip()
    return action_type or "UNKNOWN"


def classify_episode_outcome(final_state: dict[str, Any]) -> tuple[str, str]:
    """(outcome, win_definition) for one finished episode's final state."""
    trophies = int(final_state.get("trophies", 0) or 0)
    lives = int(final_state.get("lives", 0) or 0)
    meta = final_state.get("meta") if isinstance(final_state.get("meta"), dict) else {}
    game_mode = str((meta or {}).get("game_mode") or "arena").strip().lower()
    versus = (meta or {}).get("versus") if isinstance(meta, dict) else None
    opponent_lives: int | None = None
    if isinstance(versus, dict) and versus.get("opponent_lives") is not None:
        try:
            opponent_lives = int(versus.get("opponent_lives"))
        except (TypeError, ValueError):
            opponent_lives = None

    if game_mode == "versus" and opponent_lives is not None:
        definition = "versus_opponent_lives_0"
        if opponent_lives <= 0 and lives > 0:
            return "win", definition
        if lives <= 0:
            return "loss", definition
        return "draw", definition

    definition = "arena_trophies_7"
    if trophies >= 7:
        return "win", definition
    if lives <= 0:
        return "loss", definition
    return "draw", definition


def evaluate_maskable_model(
    *,
    model: Any,
    env: Any,
    episodes: int,
    seed: int,
    deterministic: bool,
    provider: Any | None = None,
    max_steps_per_episode: int = 512,
) -> dict[str, Any]:
    episode_trophies: list[int] = []
    episode_lengths: list[int] = []
    episode_lives: list[int] = []
    episode_turns: list[int] = []
    episode_rewards: list[float] = []
    wins = 0
    losses = 0
    win_definition = "arena_trophies_7"
    invalid_steps = 0
    total_steps = 0
    action_type_counts_total: dict[str, int] = {}
    end_turn_actions = 0
    end_turn_zero_pet_actions = 0
    reward_breakdown_totals: dict[str, float] = {}
    forced_done_steps = 0
    forced_done_reasons: dict[str, int] = {}

    core_env = _resolve_core_env(env)

    for ep in range(int(episodes)):
        if provider is not None and hasattr(provider, "set_episode"):
            provider.set_episode(ep)

        obs, _info = env.reset(seed=int(seed) + ep)
        done = False
        steps = 0
        ep_reward = 0.0

        while not done:
            state_before = {}
            if core_env is not None and isinstance(getattr(core_env, "state", None), dict):
                state_before = dict(getattr(core_env, "state", {}))
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=bool(deterministic))
            action_index = _coerce_action_index(action)
            action_type = _action_type_for_index(core_env, action_index)
            action_type_counts_total[action_type] = int(action_type_counts_total.get(action_type, 0)) + 1
            if action_type == "END_TURN":
                end_turn_actions += 1
                team_before = state_before.get("team")
                pet_count_before = 0
                if isinstance(team_before, list):
                    pet_count_before = sum(
                        1
                        for slot in team_before
                        if isinstance(slot, dict) and slot.get("pet_id") is not None
                    )
                if int(pet_count_before) <= 0:
                    end_turn_zero_pet_actions += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            steps += 1
            total_steps += 1
            ep_reward += float(reward)

            if not bool(info.get("step_ok", True)):
                invalid_steps += 1
            if bool(info.get("forced_done", False)):
                forced_done_steps += 1
                reason = str(info.get("forced_done_reason") or "unknown")
                forced_done_reasons[reason] = int(forced_done_reasons.get(reason, 0)) + 1

            terms = info.get("reward_terms")
            if isinstance(terms, dict):
                for key, value in terms.items():
                    k = str(key)
                    reward_breakdown_totals[k] = float(reward_breakdown_totals.get(k, 0.0)) + float(value)

            if steps >= int(max_steps_per_episode):
                done = True

        final_state = getattr(core_env, "state", {}) if core_env is not None else {}
        trophies = int(final_state.get("trophies", 0))
        lives = int(final_state.get("lives", 0))
        turns = int(final_state.get("turn", 1))
        episode_trophies.append(trophies)
        episode_lives.append(lives)
        episode_turns.append(turns)
        episode_lengths.append(steps)
        episode_rewards.append(float(ep_reward))

        outcome, outcome_definition = classify_episode_outcome(final_state)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        win_definition = outcome_definition

    draws = max(0, int(episodes) - wins - losses)
    action_type_counts_sorted = {
        k: int(v) for k, v in sorted(action_type_counts_total.items(), key=lambda kv: kv[0])
    }
    action_type_avg_per_episode = {
        k: float(v) / max(1, int(episodes)) for k, v in action_type_counts_sorted.items()
    }
    summary = {
        "episodes": int(episodes),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_definition": str(win_definition),
        "win_rate": float(wins) / max(1, int(episodes)),
        "avg_trophies": float(np.mean(episode_trophies)) if episode_trophies else 0.0,
        "avg_lives": float(np.mean(episode_lives)) if episode_lives else 0.0,
        "avg_turn_reached": float(np.mean(episode_turns)) if episode_turns else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "avg_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "invalid_action_rate": float(invalid_steps) / max(1, total_steps),
        "end_turn_actions": int(end_turn_actions),
        "end_turn_zero_pet_actions": int(end_turn_zero_pet_actions),
        "end_turn_zero_pet_rate": float(end_turn_zero_pet_actions) / max(1, end_turn_actions),
        "reward_breakdown": reward_breakdown_totals,
        "action_type_counts_total": action_type_counts_sorted,
        "action_type_avg_per_episode": action_type_avg_per_episode,
        "forced_done_steps": int(forced_done_steps),
        "forced_done_reasons": {k: int(v) for k, v in sorted(forced_done_reasons.items(), key=lambda kv: kv[0])},
    }
    return summary
