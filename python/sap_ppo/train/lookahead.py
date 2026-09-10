"""Deterministic top-k lookahead planner for shop-phase action chains.

This planner is intentionally lightweight:
- expands only deterministic actions,
- stops at terminal stochastic actions (`ROLL`, `END_TURN`),
- uses heuristic scoring plus optional policy-first-action bias.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import TrainingEnv


TERMINAL_ACTION_TYPES = {"ROLL", "END_TURN"}


@dataclass(frozen=True)
class PlannerConfig:
    top_k: int = 8
    depth: int = 8
    node_cap: int = 256
    time_ms_cap: int = 30


def _coerce_action_index(action: Any) -> int | None:
    try:
        if hasattr(action, "item"):
            return int(action.item())
        return int(action)
    except Exception:
        return None


def _team_metrics(state: dict[str, Any]) -> tuple[int, int]:
    team = state.get("team")
    if not isinstance(team, list):
        return 0, 0
    pet_count = 0
    power = 0
    for slot in team:
        if not isinstance(slot, dict):
            continue
        if slot.get("pet_id") is None:
            continue
        pet_count += 1
        power += int(slot.get("attack", 0)) + int(slot.get("health", 0))
    return int(pet_count), int(power)


def _frozen_count(state: dict[str, Any]) -> int:
    shop = state.get("shop")
    if not isinstance(shop, list):
        return 0
    return sum(1 for slot in shop if isinstance(slot, dict) and bool(slot.get("frozen", False)))


def _state_heuristic_score(state: dict[str, Any]) -> float:
    pet_count, power = _team_metrics(state)
    gold = int(state.get("gold", 0))
    trophies = int(state.get("trophies", 0))
    lives = int(state.get("lives", 0))
    frozen = _frozen_count(state)
    # Conservative heuristic for shop-phase quality.
    return (
        0.40 * float(pet_count)
        + 0.02 * float(power)
        + 0.05 * float(10 - gold)
        + 0.20 * float(trophies)
        + 0.05 * float(lives)
        - 0.02 * float(frozen)
    )


def _action_bias(action_type: str) -> float:
    typ = str(action_type or "").strip().upper()
    return {
        "BUY_COMBINE": 0.50,
        "BUY_PET": 0.30,
        "BUY_FOOD": 0.20,
        "COMBINE": 0.15,
        "SELL": -0.15,
        "FREEZE": -0.08,
        "UNFREEZE": -0.08,
        "REORDER": -0.06,
        "ROLL": 0.05,
        "END_TURN": 0.00,
    }.get(typ, 0.0)


def _terminal_bonus(action_type: str, state: dict[str, Any]) -> float:
    typ = str(action_type or "").strip().upper()
    gold = int(state.get("gold", 0))
    if typ == "END_TURN":
        return 0.10 if gold == 0 else -0.20
    if typ == "ROLL":
        return 0.05
    return 0.0


def _policy_suggested_action_index(
    env: TrainingEnv,
    *,
    model: Any | None,
    encoder: Any | None,
    deterministic: bool,
) -> int | None:
    if model is None or encoder is None:
        return None
    try:
        obs = encoder.encode(env.state)
        masks = np.asarray(env.legal_action_mask(), dtype=np.bool_)
        if masks.size == 0 or not bool(masks.any()):
            return None
        action, _ = model.predict(obs, action_masks=masks, deterministic=bool(deterministic))
        return _coerce_action_index(action)
    except Exception:
        return None


def plan_deterministic_chain(
    env: TrainingEnv,
    *,
    model: Any | None = None,
    encoder: Any | None = None,
    deterministic_policy: bool = True,
    config: PlannerConfig = PlannerConfig(),
) -> dict[str, Any]:
    started = time.perf_counter()
    root_env = copy.deepcopy(env)
    top_k = max(1, int(config.top_k))
    depth_cap = max(1, int(config.depth))
    node_cap = max(1, int(config.node_cap))
    time_ms_cap = max(1, int(config.time_ms_cap))

    policy_hint = _policy_suggested_action_index(
        root_env,
        model=model,
        encoder=encoder,
        deterministic=bool(deterministic_policy),
    )

    root_state_score = _state_heuristic_score(root_env.state)
    frontier: list[dict[str, Any]] = [
        {
            "env": root_env,
            "chain": [],
            "score": float(root_state_score),
            "depth": 0,
        }
    ]
    leaves: list[dict[str, Any]] = []
    root_candidates_trace: list[dict[str, Any]] = []
    timed_out = False
    node_budget_hit = False
    nodes_expanded = 0
    actions_scored = 0
    expansions = 0

    while frontier:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms >= float(time_ms_cap):
            timed_out = True
            break
        next_frontier: list[dict[str, Any]] = []

        for node in frontier:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms >= float(time_ms_cap):
                timed_out = True
                break
            if nodes_expanded >= int(node_cap):
                node_budget_hit = True
                break

            depth = int(node["depth"])
            if depth >= depth_cap:
                leaves.append({**node, "stop_reason": "depth_cap"})
                continue

            node_env: TrainingEnv = node["env"]
            legal_mask = np.asarray(node_env.legal_action_mask(), dtype=np.bool_)
            legal_indices = [int(i) for i, bit in enumerate(legal_mask.tolist()) if bool(bit)]
            if not legal_indices:
                leaves.append({**node, "stop_reason": "no_legal_actions"})
                continue

            scored: list[tuple[float, int, dict[str, Any]]] = []
            for action_index in legal_indices:
                action_obj = node_env.decode_action(action_index)
                action_type = str(action_obj.get("type", ""))
                score = float(node["score"]) + _action_bias(action_type)
                if policy_hint is not None and int(action_index) == int(policy_hint) and depth == 0:
                    score += 0.25
                scored.append((score, int(action_index), action_obj))
                actions_scored += 1

            scored.sort(key=lambda row: float(row[0]), reverse=True)
            selected = scored[:top_k]
            if depth == 0:
                root_candidates_trace = [
                    {
                        "action_index": int(idx),
                        "action": copy.deepcopy(action),
                        "score_pre_expand": float(score),
                    }
                    for score, idx, action in selected[: min(len(selected), 16)]
                ]

            for pre_score, action_index, action_obj in selected:
                action_type = str(action_obj.get("type", "")).strip().upper()
                chain_next = list(node["chain"]) + [int(action_index)]
                if action_type in TERMINAL_ACTION_TYPES:
                    leaf_score = float(pre_score) + _terminal_bonus(action_type, node_env.state)
                    leaves.append(
                        {
                            "env": node_env,
                            "chain": chain_next,
                            "score": float(leaf_score),
                            "depth": depth + 1,
                            "stop_reason": f"terminal:{action_type}",
                        }
                    )
                    continue

                child_env = copy.deepcopy(node_env)
                out = child_env.step(int(action_index))
                if not bool(out.get("ok", False)):
                    continue
                tr = out.get("transition")
                if isinstance(tr, dict) and not bool(tr.get("deterministic", False)):
                    # Deterministic planner does not continue past stochastic branches.
                    leaves.append(
                        {
                            "env": child_env,
                            "chain": chain_next,
                            "score": float(pre_score),
                            "depth": depth + 1,
                            "stop_reason": "stochastic_transition",
                        }
                    )
                    continue

                child_score = float(pre_score) + _state_heuristic_score(child_env.state)
                next_frontier.append(
                    {
                        "env": child_env,
                        "chain": chain_next,
                        "score": float(child_score),
                        "depth": depth + 1,
                    }
                )
                expansions += 1

            nodes_expanded += 1

        if timed_out or node_budget_hit:
            break
        if not next_frontier:
            break
        next_frontier.sort(key=lambda n: float(n["score"]), reverse=True)
        frontier = next_frontier[:top_k]

    if frontier:
        leaves.extend({**n, "stop_reason": n.get("stop_reason", "frontier_remaining")} for n in frontier)

    def _leaf_sort_key(item: dict[str, Any]) -> tuple[int, float]:
        chain = item.get("chain") or []
        stop_reason = str(item.get("stop_reason") or "")
        terminal_flag = 1 if stop_reason.startswith("terminal:") and chain else 0
        return (terminal_flag, float(item.get("score", 0.0)))

    leaves_sorted = sorted(leaves, key=_leaf_sort_key, reverse=True)
    best = leaves_sorted[0] if leaves_sorted else None

    if not isinstance(best, dict) or not best.get("chain"):
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "ok": False,
            "error": "planner_no_chain",
            "selected_chain_indices": [],
            "selected_chain_actions": [],
            "stats": {
                "planner_ms": float(elapsed),
                "nodes_expanded": int(nodes_expanded),
                "actions_scored": int(actions_scored),
                "expansions": int(expansions),
                "timed_out": bool(timed_out),
                "node_budget_hit": bool(node_budget_hit),
                "depth_cap": int(depth_cap),
                "node_cap": int(node_cap),
                "top_k": int(top_k),
                "policy_hint_action_index": (int(policy_hint) if policy_hint is not None else None),
            },
            "trace": {
                "root_candidates": root_candidates_trace,
                "best_chain": [],
                "leaf_count": int(len(leaves_sorted)),
            },
        }

    selected_chain_indices = [int(x) for x in list(best["chain"])]
    selected_chain_actions = [root_env.decode_action(i) for i in selected_chain_indices]
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "ok": True,
        "error": None,
        "selected_chain_indices": selected_chain_indices,
        "selected_chain_actions": [copy.deepcopy(a) for a in selected_chain_actions],
        "stats": {
            "planner_ms": float(elapsed),
            "nodes_expanded": int(nodes_expanded),
            "actions_scored": int(actions_scored),
            "expansions": int(expansions),
            "timed_out": bool(timed_out),
            "node_budget_hit": bool(node_budget_hit),
            "depth_cap": int(depth_cap),
            "node_cap": int(node_cap),
            "top_k": int(top_k),
            "policy_hint_action_index": (int(policy_hint) if policy_hint is not None else None),
            "selected_terminal_action_type": (
                str(selected_chain_actions[-1].get("type", "")) if selected_chain_actions else None
            ),
        },
        "trace": {
            "root_candidates": root_candidates_trace,
            "best_chain": [
                {"action_index": int(i), "action": copy.deepcopy(a)}
                for i, a in zip(selected_chain_indices, selected_chain_actions)
            ],
            "leaf_count": int(len(leaves_sorted)),
            "top_leaf_scores": [
                {
                    "score": float(leaf.get("score", 0.0)),
                    "stop_reason": str(leaf.get("stop_reason", "")),
                    "chain_len": int(len(leaf.get("chain") or [])),
                }
                for leaf in leaves_sorted[:8]
            ],
        },
    }

