"""Opponent providers for Step 3 training runtime."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..catalog import load_turtle_catalog, tier_for_turn
from ..constants import NON_ROLLABLE_PET_IDS
from ..oracles.sap_calc_battle_oracle import team_to_pet_configs
from .snapshots import load_snapshot


class OpponentProvider(Protocol):
    """Training opponent source contract."""

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        """Return opponent payload matching replay sampler envelope."""


@dataclass
class ReplayDBOpponentProvider:
    """Live replay-db backed provider (best for tooling, not rollout hot loops)."""

    database_url: str | None = None

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        # Call-time import: this provider is the only thing in the training
        # package that needs Postgres, and module scope would put it in the
        # import closure of everything that touches training.
        from ..opponents.replay_db import (
            sample_opponent_team_for_pid,
            sample_random_opponent_team,
        )

        turn_i = int(turn)
        pid = str(forced_pid or "").strip()
        if pid:
            return sample_opponent_team_for_pid(pid, turn_i, database_url=self.database_url)
        return sample_random_opponent_team(turn_i, database_url=self.database_url)


@dataclass
class StaticTurnOpponentProvider:
    """Local in-memory turn-indexed provider (recommended for training loops)."""

    by_turn: dict[int, list[dict[str, Any]]]
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        turn_i = int(turn)
        rows = [r for r in self.by_turn.get(turn_i, []) if isinstance(r, dict)]
        if not rows:
            return {
                "ok": False,
                "error": f"no_static_opponent_for_turn:{turn_i}",
                "team": [],
                "replay_id": None,
                "participation_id": str(forced_pid or "").strip() or None,
                "created_at": None,
                "side": None,
                "source": "static",
            }

        pid = str(forced_pid or "").strip()
        if pid:
            filtered = [r for r in rows if str(r.get("participation_id") or "").strip() == pid]
            if not filtered:
                return {
                    "ok": False,
                    "error": f"forced_pid_not_found:{pid}:turn={turn_i}",
                    "team": [],
                    "replay_id": None,
                    "participation_id": pid,
                    "created_at": None,
                    "side": None,
                    "source": "static",
                }
            rows = filtered

        picked = copy.deepcopy(self._rng.choice(rows))
        picked.setdefault("ok", True)
        picked.setdefault("error", None)
        picked.setdefault("source", "static")
        return picked


@dataclass
class ReplaySnapshotProvider:
    """File-backed turn-indexed provider for rollout workers (no live DB calls)."""

    snapshot_path: Path
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _by_turn: dict[int, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._by_turn = load_snapshot(Path(self.snapshot_path))

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        turn_i = int(turn)
        rows = [r for r in self._by_turn.get(turn_i, []) if isinstance(r, dict)]
        if not rows:
            return {
                "ok": False,
                "error": f"no_snapshot_opponent_for_turn:{turn_i}",
                "team": [],
                "replay_id": None,
                "participation_id": str(forced_pid or "").strip() or None,
                "created_at": None,
                "side": None,
                "source": "snapshot",
            }

        pid = str(forced_pid or "").strip()
        if pid:
            rows = [r for r in rows if str(r.get("participation_id") or "").strip() == pid]
            if not rows:
                return {
                    "ok": False,
                    "error": f"forced_pid_not_found_in_snapshot:{pid}:turn={turn_i}",
                    "team": [],
                    "replay_id": None,
                    "participation_id": pid,
                    "created_at": None,
                    "side": None,
                    "source": "snapshot",
                }

        picked = copy.deepcopy(self._rng.choice(rows))
        picked.setdefault("ok", True)
        picked.setdefault("error", None)
        picked.setdefault("source", "snapshot")
        return picked


def _level_to_exp(level: int) -> int:
    if int(level) >= 3:
        return 5
    if int(level) == 2:
        return 2
    return 0


def _dummy_battle_payload(turn: int) -> dict[str, Any]:
    return {
        "UserBoard": {"Mins": {"Items": []}, "Pack": 0, "Tur": int(turn)},
        "OpponentBoard": {"Mins": {"Items": []}, "Pack": 0, "Tur": int(turn)},
    }


@dataclass
class RandomOpponentProvider:
    """Synthetic random-opponent generator for curriculum stage A."""

    seed: int | None = None
    pack: str = "Turtle"
    min_team_size: int = 1
    max_team_size: int = 5
    _rng: random.Random = field(init=False, repr=False)
    _catalog: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._catalog = load_turtle_catalog()

    def _pet_pool_for_turn(self, turn: int) -> list[str]:
        upto = max(1, min(6, int(tier_for_turn(int(turn)))))
        by_tier = self._catalog.get("pets", {}).get("by_tier", {}) or {}
        pool: list[str] = []
        for tier in range(1, upto + 1):
            for pet_id in by_tier.get(str(tier), []) or []:
                pid = str(pet_id)
                if pid in NON_ROLLABLE_PET_IDS:
                    continue
                pool.append(pid)
        return pool

    def _sample_level(self, turn: int) -> int:
        t = int(turn)
        roll = self._rng.random()
        if t >= 9 and roll < 0.12:
            return 3
        if t >= 5 and roll < 0.35:
            return 2
        return 1

    def _build_team(self, turn: int) -> list[dict[str, Any]]:
        base_stats = self._catalog.get("pets", {}).get("base_stats", {}) or {}
        pool = self._pet_pool_for_turn(turn)
        team = []
        for idx in range(5):
            team.append(
                {
                    "slot_index": idx,
                    "pet_id": None,
                    "attack": 0,
                    "health": 0,
                    "sell_value": 0,
                    "level": 1,
                    "exp": 0,
                    "equipment_id": None,
                    "status_effects": [],
                }
            )
        if not pool:
            return team

        min_size = max(0, min(5, int(self.min_team_size)))
        max_size = max(min_size, min(5, int(self.max_team_size)))
        team_size = self._rng.randint(min_size, max_size)
        for idx in range(team_size):
            pet_id = str(self._rng.choice(pool))
            stats = base_stats.get(pet_id, {"attack": 2, "health": 2})
            level = self._sample_level(turn)
            exp = _level_to_exp(level)
            bonus = max(0, int(turn) - 1)
            atk = int(stats.get("attack", 2)) + self._rng.randint(0, bonus)
            hp = int(stats.get("health", 2)) + self._rng.randint(0, bonus)
            team[idx] = {
                "slot_index": idx,
                "pet_id": pet_id,
                "attack": max(1, atk),
                "health": max(1, hp),
                "sell_value": max(1, int(level)),
                "level": int(level),
                "exp": int(exp),
                "equipment_id": None,
                "status_effects": [],
            }
        return team

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        pid = str(forced_pid or "").strip()
        if pid:
            return {
                "ok": False,
                "error": f"forced_pid_not_supported:random:{pid}",
                "team": [],
                "replay_id": None,
                "participation_id": pid,
                "created_at": None,
                "side": None,
                "source": "random",
            }

        turn_i = int(turn)
        team = self._build_team(turn_i)
        return {
            "ok": True,
            "error": None,
            "team": team,
            "replay_id": f"random-turn-{turn_i}",
            "participation_id": None,
            "created_at": None,
            "side_pack": str(self.pack),
            "side": "opponent",
            "source": "random",
            "battle": _dummy_battle_payload(turn_i),
            "build_model": None,
            "parsed_state": {
                "playerPack": str(self.pack),
                "opponentPack": str(self.pack),
                "turn": int(turn_i),
                "playerPets": [None, None, None, None, None],
                "opponentPets": team_to_pet_configs(team),
            },
        }


@dataclass
class SelfPlayPoolProvider:
    """Self-play pool provider (same by-turn snapshot shape, distinct source tag)."""

    pool_path: Path
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _by_turn: dict[int, list[dict[str, Any]]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._by_turn = load_snapshot(Path(self.pool_path))

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        turn_i = int(turn)
        rows = [r for r in self._by_turn.get(turn_i, []) if isinstance(r, dict)]
        if not rows:
            return {
                "ok": False,
                "error": f"no_self_play_opponent_for_turn:{turn_i}",
                "team": [],
                "replay_id": None,
                "participation_id": str(forced_pid or "").strip() or None,
                "created_at": None,
                "side": None,
                "source": "self_play",
            }

        pid = str(forced_pid or "").strip()
        if pid:
            rows = [r for r in rows if str(r.get("participation_id") or "").strip() == pid]
            if not rows:
                return {
                    "ok": False,
                    "error": f"forced_pid_not_found_in_self_play_pool:{pid}:turn={turn_i}",
                    "team": [],
                    "replay_id": None,
                    "participation_id": pid,
                    "created_at": None,
                    "side": None,
                    "source": "self_play",
                }

        picked = copy.deepcopy(self._rng.choice(rows))
        picked.setdefault("ok", True)
        picked.setdefault("error", None)


        picked.setdefault("source", "self_play")
        return picked


@dataclass
class CurriculumOpponentProvider:
    """Weighted provider mixer controlled by training stage schedule."""

    providers: dict[str, OpponentProvider]
    stages: list[dict[str, Any]]
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _episode_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.providers:
            raise ValueError("curriculum_requires_at_least_one_provider")
        if not self.stages:
            raise ValueError("curriculum_requires_at_least_one_stage")
        self._rng = random.Random(self.seed)

    def set_episode(self, episode_idx: int) -> None:
        self._episode_idx = max(0, int(episode_idx))

    def _current_stage(self) -> dict[str, Any]:
        current = self.stages[-1]
        for stage in self.stages:
            until = stage.get("until_episode")
            if until is None:
                current = stage
                break
            if self._episode_idx < int(until):
                current = stage
                break
        return current

    def _weighted_provider_keys(self, stage: dict[str, Any]) -> list[str]:
        weights = stage.get("weights", {})
        if not isinstance(weights, dict):
            return list(self.providers.keys())
        keys: list[str] = []
        for name, weight in weights.items():
            if name not in self.providers:
                continue
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            if w <= 0:
                continue
            keys.append(str(name))
        return keys or list(self.providers.keys())

    def sample(self, turn: int, *, forced_pid: str | None = None) -> dict[str, Any]:
        stage = self._current_stage()
        names = self._weighted_provider_keys(stage)
        weights_raw = stage.get("weights", {}) if isinstance(stage.get("weights", {}), dict) else {}
        weights = [max(0.0, float(weights_raw.get(name, 1.0))) for name in names]
        if sum(weights) <= 0:
            weights = [1.0 for _ in names]

        tried: set[str] = set()
        ordered = self._rng.choices(names, weights=weights, k=len(names))
        for name in names:
            if name not in ordered:
                ordered.append(name)

        last_error = "no_provider_selected"
        for name in ordered:
            if name in tried:
                continue
            tried.add(name)
            provider = self.providers[name]
            sampled = provider.sample(int(turn), forced_pid=forced_pid)
            if sampled.get("ok"):
                sampled = copy.deepcopy(sampled)
                sampled.setdefault("source", name)
                sampled["curriculum_provider"] = name
                sampled["curriculum_episode"] = int(self._episode_idx)
                sampled["curriculum_stage"] = str(stage.get("name") or "unnamed")
                return sampled
            last_error = str(sampled.get("error") or f"provider_failed:{name}")

        return {
            "ok": False,
            "error": f"curriculum_sampling_failed:{last_error}",
            "team": [],
            "replay_id": None,
            "participation_id": str(forced_pid or "").strip() or None,
            "created_at": None,
            "side": None,
            "source": "curriculum",
            "curriculum_episode": int(self._episode_idx),
            "curriculum_stage": str(stage.get("name") or "unnamed"),
        }
