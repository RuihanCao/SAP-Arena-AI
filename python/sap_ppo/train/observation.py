"""State vector encoders and observation compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..catalog import load_turtle_catalog, tier_for_turn
from ..constants import EQUIPMENT_STATUS_BY_FOOD_ID
from ..tempo.features import DEFAULT_HASH_BUCKETS, encode_last_opponent_context

OBSERVATION_MODE_V1 = "v1_compact"
OBSERVATION_MODE_V2 = "v2_onehot"
OBSERVATION_MODE_V3 = "v3_pbrs"
OBSERVATION_MODE_V4 = "v4_one_turn_context"
OBSERVATION_MODE_AUTO = "auto"
OBSERVATION_MODE_CHOICES = (OBSERVATION_MODE_V1, OBSERVATION_MODE_V2, OBSERVATION_MODE_V3, OBSERVATION_MODE_V4)
OBSERVATION_MODE_CHOICES_WITH_AUTO = (OBSERVATION_MODE_AUTO,) + OBSERVATION_MODE_CHOICES
DEFAULT_OBSERVATION_MODE = OBSERVATION_MODE_V2

OBSERVATION_SPEC_KEYS = (
    "observation_mode",
    "observation_size",
    "observation_vocab_fingerprint",
)

_MAX_SHOP_TIER = 6
_MAX_TEAM_SLOTS = 5
_MAX_SHOP_SLOTS = 9
_MAX_TEAM_PAIR_COUNT = (_MAX_TEAM_SLOTS * (_MAX_TEAM_SLOTS - 1)) // 2
_MAX_SHOP_TEAM_PAIR_COUNT = _MAX_TEAM_SLOTS * _MAX_SHOP_SLOTS

# Trigger counters with capped usages that are important for action quality.
_TRIGGER_RULES: dict[str, tuple[str, tuple[int, int, int] | int]] = {
    "pet-ox": ("friend_ahead_faints", (1, 2, 3)),
    "pet-fly": ("friend_faints", 3),
    "pet-gorilla": ("hurt", (1, 2, 3)),
    "pet-rabbit": ("friendly_ate_food", 3),
    "pet-cat": ("purchase_food", 2),
    "pet-dragon": ("buy_tier1_pet", 4),
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _norm(value: int | float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(max(0.0, min(1.0, float(value) / float(scale))))


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def normalize_observation_mode(
    value: str | None,
    *,
    default: str = DEFAULT_OBSERVATION_MODE,
    allow_auto: bool = False,
) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        raw = str(default).strip().lower()
    aliases = {
        "v1": OBSERVATION_MODE_V1,
        "compact": OBSERVATION_MODE_V1,
        "v2": OBSERVATION_MODE_V2,
        "onehot": OBSERVATION_MODE_V2,
        "one_hot": OBSERVATION_MODE_V2,
        "v3": OBSERVATION_MODE_V3,
        "pbrs": OBSERVATION_MODE_V3,
        "pbrs_v3": OBSERVATION_MODE_V3,
        "v4": OBSERVATION_MODE_V4,
        "one_turn": OBSERVATION_MODE_V4,
        "one_turn_context": OBSERVATION_MODE_V4,
    }
    mode = aliases.get(raw, raw)
    if allow_auto and mode == OBSERVATION_MODE_AUTO:
        return mode
    if mode not in OBSERVATION_MODE_CHOICES:
        allowed = ",".join(
            OBSERVATION_MODE_CHOICES_WITH_AUTO if allow_auto else OBSERVATION_MODE_CHOICES
        )
        raise ValueError(f"invalid_observation_mode:{value}:allowed={allowed}")
    return mode


@lru_cache(maxsize=1)
def _vocab_tables() -> dict[str, Any]:
    catalog = load_turtle_catalog()
    pets_id_to_name = catalog.get("pets", {}).get("id_to_name_id", {}) or {}
    foods_id_to_name = catalog.get("foods", {}).get("id_to_name_id", {}) or {}
    pets_by_tier = catalog.get("pets", {}).get("by_tier", {}) or {}
    foods_by_tier = catalog.get("foods", {}).get("by_tier", {}) or {}

    pet_ids = sorted(str(pid) for pid in pets_id_to_name.keys())
    food_ids = sorted(str(fid) for fid in foods_id_to_name.keys())
    equip_ids = sorted(set(food_ids))
    status_ids = sorted(set(str(v) for v in EQUIPMENT_STATUS_BY_FOOD_ID.values()))

    pet_tier_by_id: dict[str, int] = {}
    for tier_str, ids in pets_by_tier.items():
        tier = max(1, min(_MAX_SHOP_TIER, _safe_int(tier_str, 1)))
        if not isinstance(ids, list):
            continue
        for pid in ids:
            key = str(pid)
            if key not in pet_tier_by_id:
                pet_tier_by_id[key] = int(tier)

    food_tier_by_id: dict[str, int] = {}
    for tier_str, ids in foods_by_tier.items():
        tier = max(1, min(_MAX_SHOP_TIER, _safe_int(tier_str, 1)))
        if not isinstance(ids, list):
            continue
        for fid in ids:
            key = str(fid)
            if key not in food_tier_by_id:
                food_tier_by_id[key] = int(tier)

    return {
        "pet_ids": pet_ids,
        "food_ids": food_ids,
        "equip_ids": equip_ids,
        "status_ids": status_ids,
        "pet_tier_by_id": pet_tier_by_id,
        "food_tier_by_id": food_tier_by_id,
    }


def _shop_by_index(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    shop = state.get("shop")
    if not isinstance(shop, list):
        return out
    for slot in shop:
        if not isinstance(slot, dict):
            continue
        idx = _safe_int(slot.get("shop_index"), -1)
        if 0 <= idx < _MAX_SHOP_SLOTS:
            out[int(idx)] = slot
    return out


def _team_slots(state: dict[str, Any]) -> list[dict[str, Any]]:
    team = state.get("team")
    if not isinstance(team, list):
        return []
    return [slot for slot in team if isinstance(slot, dict)]


def _ability_counter(state: dict[str, Any], trigger: str, team_index: int) -> int:
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return 0
    store = meta.get("ability_counters")
    if not isinstance(store, dict):
        return 0
    by_trigger = store.get(str(trigger))
    if not isinstance(by_trigger, dict):
        return 0
    return max(0, _safe_int(by_trigger.get(str(team_index)), 0))


def _trigger_limit(pet_id: str, level: int) -> tuple[str, int] | None:
    spec = _TRIGGER_RULES.get(str(pet_id))
    if spec is None:
        return None
    trigger, raw_limit = spec
    lvl = max(1, min(3, _safe_int(level, 1)))
    if isinstance(raw_limit, tuple):
        limit = int(raw_limit[lvl - 1])
    else:
        limit = int(raw_limit)
    return str(trigger), max(0, int(limit))


def _team_combine_pair_count(state: dict[str, Any]) -> tuple[int, set[int]]:
    team = _team_slots(state)
    counts: dict[str, int] = {}
    slots_by_id: dict[str, list[int]] = {}
    for idx in range(min(_MAX_TEAM_SLOTS, len(team))):
        slot = team[idx]
        pet_id = str(slot.get("pet_id") or "")
        level = _safe_int(slot.get("level"), 1)
        if not pet_id or level >= 3:
            continue
        counts[pet_id] = int(counts.get(pet_id, 0)) + 1
        slots_by_id.setdefault(pet_id, []).append(int(idx))
    pair_count = sum(int(n * (n - 1) // 2) for n in counts.values() if int(n) >= 2)
    enabled_slots = {idx for pet_id, indices in slots_by_id.items() if counts.get(pet_id, 0) >= 2 for idx in indices}
    return int(pair_count), enabled_slots


def _shop_link_groups(state: dict[str, Any]) -> dict[int, int]:
    groups: dict[str, list[int]] = {}
    by_index = _shop_by_index(state)
    for idx in range(_MAX_SHOP_SLOTS):
        slot = by_index.get(int(idx))
        if not isinstance(slot, dict):
            continue
        link_id = str(slot.get("link_id") or "").strip()
        if not link_id:
            continue
        groups.setdefault(link_id, []).append(int(idx))
    out: dict[int, int] = {}
    for indices in groups.values():
        size = int(len(indices))
        for idx in indices:
            out[int(idx)] = int(size)
    return out


def _turns_to_next_tier(turn: int, max_turn: int) -> int:
    t = max(1, int(turn))
    curr = int(tier_for_turn(t))
    cap = max(int(max_turn), 50)
    for next_t in range(t + 1, cap + 1):
        if int(tier_for_turn(next_t)) != curr:
            return int(next_t - t)
    return 0


@dataclass
class StateVectorEncoder:
    """Legacy compact v1 encoder (kept as fallback)."""

    max_turn: int = 15
    max_gold: int = 10
    max_lives: int = 6
    max_trophies: int = 10
    max_stat: int = 50
    max_sell: int = 10

    def __post_init__(self) -> None:
        vocab = _vocab_tables()
        pet_ids = list(vocab["pet_ids"])
        food_ids = list(vocab["food_ids"])
        equip_ids = list(vocab["equip_ids"])
        self._pet_index = {"": 0}
        self._food_index = {"": 0}
        self._equip_index = {"": 0}
        for i, pid in enumerate(pet_ids, start=1):
            self._pet_index[pid] = i
        for i, fid in enumerate(food_ids, start=1):
            self._food_index[fid] = i
        for i, eid in enumerate(equip_ids, start=1):
            self._equip_index[eid] = i

        # 4 global + 5 team slots * 8 features + 9 shop slots * 6 features
        self.size = 4 + (_MAX_TEAM_SLOTS * 8) + (_MAX_SHOP_SLOTS * 6)
        self.mode = OBSERVATION_MODE_V1
        self.vocab_fingerprint = _fingerprint(
            {
                "mode": self.mode,
                "pet_ids": pet_ids,
                "food_ids": food_ids,
                "equip_ids": equip_ids,
                "feature_layout": "legacy_v1_compact",
            }
        )

    def _team_slot(self, slot: dict[str, Any]) -> list[float]:
        pet_id = str(slot.get("pet_id") or "")
        equip_id = str(slot.get("equipment_id") or "")
        pet_idx = self._pet_index.get(pet_id, 0)
        equip_idx = self._equip_index.get(equip_id, 0)
        return [
            1.0 if pet_id else 0.0,
            _norm(pet_idx, max(1, len(self._pet_index) - 1)),
            _norm(_safe_int(slot.get("attack"), 0), self.max_stat),
            _norm(_safe_int(slot.get("health"), 0), self.max_stat),
            _norm(_safe_int(slot.get("level"), 1), 3),
            _norm(_safe_int(slot.get("exp"), 0), 5),
            _norm(_safe_int(slot.get("sell_value"), 0), self.max_sell),
            _norm(equip_idx, max(1, len(self._equip_index) - 1)),
        ]

    def _shop_slot(self, slot: dict[str, Any] | None) -> list[float]:
        if not isinstance(slot, dict):
            return [0.0] * 6
        slot_type = str(slot.get("slot_type") or "")
        is_pet = 1.0 if slot_type == "pet" else 0.0
        is_food = 1.0 if slot_type == "food" else 0.0
        item_id = str(slot.get("item_id") or "")
        if slot_type == "pet":
            item_idx = self._pet_index.get(item_id, 0)
            item_norm = _norm(item_idx, max(1, len(self._pet_index) - 1))
        else:
            item_idx = self._food_index.get(item_id, 0)
            item_norm = _norm(item_idx, max(1, len(self._food_index) - 1))
        return [
            is_pet,
            is_food,
            item_norm,
            _norm(_safe_int(slot.get("cost"), 0), 3),
            1.0 if bool(slot.get("frozen")) else 0.0,
            1.0 if item_id else 0.0,
        ]

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        vec: list[float] = []
        vec.extend(
            [
                _norm(_safe_int(state.get("turn"), 1), self.max_turn),
                _norm(_safe_int(state.get("gold"), 0), self.max_gold),
                _norm(_safe_int(state.get("lives"), 0), self.max_lives),
                _norm(_safe_int(state.get("trophies"), 0), self.max_trophies),
            ]
        )

        team = state.get("team", [])
        for i in range(_MAX_TEAM_SLOTS):
            slot = team[i] if isinstance(team, list) and i < len(team) and isinstance(team[i], dict) else {}
            vec.extend(self._team_slot(slot))

        by_index = _shop_by_index(state)
        for i in range(_MAX_SHOP_SLOTS):
            vec.extend(self._shop_slot(by_index.get(i)))

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (self.size,):
            raise ValueError(f"observation_size_mismatch:got={arr.shape}:expected={(self.size,)}")
        return arr


StateVectorEncoderV1 = StateVectorEncoder


@dataclass
class StateVectorEncoderV2:
    """One-hot + rich-state v2 encoder."""

    max_turn: int = 15
    max_gold: int = 10
    max_lives: int = 6
    max_trophies: int = 10
    max_stat: int = 50
    max_sell: int = 10

    def __post_init__(self) -> None:
        vocab = _vocab_tables()
        self.mode = OBSERVATION_MODE_V2
        self._pet_ids = list(vocab["pet_ids"])
        self._food_ids = list(vocab["food_ids"])
        self._equip_ids = list(vocab["equip_ids"])
        self._status_ids = list(vocab["status_ids"])
        self._pet_tier_by_id = dict(vocab["pet_tier_by_id"])
        self._food_tier_by_id = dict(vocab["food_tier_by_id"])

        self._pet_onehot_team = {pid: i + 1 for i, pid in enumerate(self._pet_ids)}
        self._pet_onehot_shop = {pid: i for i, pid in enumerate(self._pet_ids)}
        self._food_onehot_shop = {fid: i for i, fid in enumerate(self._food_ids)}
        self._equip_onehot = {eid: i + 1 for i, eid in enumerate(self._equip_ids)}
        self._status_index = {sid: i for i, sid in enumerate(self._status_ids)}

        self._team_slot_size = (
            (1 + len(self._pet_ids))  # pet one-hot with explicit empty bucket
            + (1 + len(self._equip_ids))  # equipment one-hot with explicit none bucket
            + len(self._status_ids)  # status multi-hot
            + 15  # numeric + combine + trigger features
        )
        self._shop_slot_size = (
            2  # slot type one-hot
            + len(self._pet_ids)  # pet one-hot (empty -> all zero)
            + len(self._food_ids)  # food one-hot (empty -> all zero)
            + 11  # numeric/link/combine features
        )
        self._global_size = 6
        self._summary_size = 16
        self.size = (
            self._global_size
            + (_MAX_TEAM_SLOTS * self._team_slot_size)
            + (_MAX_SHOP_SLOTS * self._shop_slot_size)
            + self._summary_size
        )
        self.vocab_fingerprint = _fingerprint(
            {
                "mode": self.mode,
                "pet_ids": self._pet_ids,
                "food_ids": self._food_ids,
                "equip_ids": self._equip_ids,
                "status_ids": self._status_ids,
                "feature_layout": "v2_onehot_rich_state_v1",
            }
        )

    def _team_combine_flags(self, state: dict[str, Any]) -> tuple[set[int], dict[int, int], int]:
        team_pair_count, team_indices_enabled = _team_combine_pair_count(state)
        team = _team_slots(state)
        counts_level_lt3: dict[str, int] = {}
        for idx in range(min(_MAX_TEAM_SLOTS, len(team))):
            slot = team[idx]
            pet_id = str(slot.get("pet_id") or "")
            if not pet_id:
                continue
            if _safe_int(slot.get("level"), 1) >= 3:
                continue
            counts_level_lt3[pet_id] = int(counts_level_lt3.get(pet_id, 0)) + 1
        return team_indices_enabled, counts_level_lt3, int(team_pair_count)

    def _shop_combine_counts(self, state: dict[str, Any], team_counts: dict[str, int]) -> tuple[dict[int, int], int]:
        by_index = _shop_by_index(state)
        per_shop: dict[int, int] = {}
        total = 0
        for idx in range(_MAX_SHOP_SLOTS):
            slot = by_index.get(int(idx))
            if not isinstance(slot, dict):
                per_shop[int(idx)] = 0
                continue
            if str(slot.get("slot_type") or "") != "pet":
                per_shop[int(idx)] = 0
                continue
            pet_id = str(slot.get("item_id") or "")
            count = max(0, int(team_counts.get(pet_id, 0)))
            per_shop[int(idx)] = int(count)
            total += int(count)
        return per_shop, int(total)

    def _team_slot(self, state: dict[str, Any], slot: dict[str, Any], team_index: int, *, combine_team: set[int], combine_shop: bool) -> list[float]:
        pet_id = str(slot.get("pet_id") or "")
        equip_id = str(slot.get("equipment_id") or "")
        status_values = slot.get("status_effects")
        status_set = {str(x) for x in status_values if str(x)} if isinstance(status_values, list) else set()
        attack = max(0, _safe_int(slot.get("attack"), 0))
        health = max(0, _safe_int(slot.get("health"), 0))

        perm_attack_raw = slot.get("perm_attack")
        perm_health_raw = slot.get("perm_health")
        perm_attack = max(0, _safe_int(perm_attack_raw, attack))
        perm_health = max(0, _safe_int(perm_health_raw, health))

        temp_attack_raw = slot.get("temp_attack")
        temp_health_raw = slot.get("temp_health")
        temp_attack = max(0, _safe_int(temp_attack_raw, max(0, attack - perm_attack)))
        temp_health = max(0, _safe_int(temp_health_raw, max(0, health - perm_health)))

        level = max(1, min(3, _safe_int(slot.get("level"), 1)))
        exp = max(0, _safe_int(slot.get("exp"), 0))
        sell_value = max(0, _safe_int(slot.get("sell_value"), 0))
        pet_tier = max(0, int(self._pet_tier_by_id.get(pet_id, 0)))

        pet_onehot = [0.0] * (1 + len(self._pet_ids))
        pet_idx = self._pet_onehot_team.get(pet_id, 0)
        pet_onehot[int(pet_idx)] = 1.0

        equip_onehot = [0.0] * (1 + len(self._equip_ids))
        equip_idx = self._equip_onehot.get(equip_id, 0)
        equip_onehot[int(equip_idx)] = 1.0

        status_multi = [0.0] * len(self._status_ids)
        for status in status_set:
            idx = self._status_index.get(status)
            if idx is not None:
                status_multi[int(idx)] = 1.0

        trigger_limit = 0
        trigger_consumed = 0
        trigger_left = 0
        trigger_spec = _trigger_limit(pet_id, level)
        if trigger_spec is not None:
            trigger_key, limit = trigger_spec
            consumed = _ability_counter(state, trigger_key, int(team_index))
            trigger_limit = int(limit)
            trigger_consumed = max(0, min(int(limit), int(consumed)))
            trigger_left = max(0, int(limit) - int(trigger_consumed))

        out = []
        out.extend(pet_onehot)
        out.extend(equip_onehot)
        out.extend(status_multi)
        out.extend(
            [
                _norm(attack, self.max_stat),
                _norm(health, self.max_stat),
                _norm(perm_attack, self.max_stat),
                _norm(perm_health, self.max_stat),
                _norm(temp_attack, self.max_stat),
                _norm(temp_health, self.max_stat),
                _norm(level, 3),
                _norm(exp, 5),
                _norm(sell_value, self.max_sell),
                _norm(pet_tier, _MAX_SHOP_TIER),
                1.0 if int(team_index) in combine_team else 0.0,
                1.0 if bool(combine_shop) else 0.0,
                _norm(trigger_limit, 5),
                (_norm(trigger_consumed, trigger_limit) if trigger_limit > 0 else 0.0),
                (_norm(trigger_left, trigger_limit) if trigger_limit > 0 else 0.0),
            ]
        )
        return out

    def _shop_slot(
        self,
        slot: dict[str, Any] | None,
        *,
        link_group_size: int,
        buy_combine_target_count: int,
    ) -> list[float]:
        if not isinstance(slot, dict):
            return [0.0] * self._shop_slot_size

        slot_type = str(slot.get("slot_type") or "")
        item_id = str(slot.get("item_id") or "")
        slot_type_onehot = [0.0, 0.0]
        if slot_type == "pet":
            slot_type_onehot[0] = 1.0
        elif slot_type == "food":
            slot_type_onehot[1] = 1.0

        pet_onehot = [0.0] * len(self._pet_ids)
        food_onehot = [0.0] * len(self._food_ids)
        if slot_type == "pet":
            pet_idx = self._pet_onehot_shop.get(item_id)
            if pet_idx is not None:
                pet_onehot[int(pet_idx)] = 1.0
        elif slot_type == "food":
            food_idx = self._food_onehot_shop.get(item_id)
            if food_idx is not None:
                food_onehot[int(food_idx)] = 1.0

        pet_attack = _safe_int(slot.get("attack"), 0) if slot_type == "pet" else 0
        pet_health = _safe_int(slot.get("health"), 0) if slot_type == "pet" else 0
        shop_bonus_at = _safe_int(slot.get("at"), 0)
        shop_bonus_hp = _safe_int(slot.get("hp"), 0)
        if slot_type == "pet":
            item_tier = _safe_int(self._pet_tier_by_id.get(item_id, 0), 0)
        elif slot_type == "food":
            item_tier = _safe_int(self._food_tier_by_id.get(item_id, 0), 0)
        else:
            item_tier = 0

        is_linked = int(link_group_size > 0)
        linked_partner_exists = int(link_group_size > 1)

        out: list[float] = []
        out.extend(slot_type_onehot)
        out.extend(pet_onehot)
        out.extend(food_onehot)
        out.extend(
            [
                _norm(_safe_int(slot.get("cost"), 0), 3),
                1.0 if bool(slot.get("frozen")) else 0.0,
                _norm(pet_attack, self.max_stat),
                _norm(pet_health, self.max_stat),
                _norm(shop_bonus_at, self.max_stat),
                _norm(shop_bonus_hp, self.max_stat),
                _norm(item_tier, _MAX_SHOP_TIER),
                float(is_linked),
                float(linked_partner_exists),
                _norm(link_group_size, _MAX_SHOP_SLOTS),
                _norm(buy_combine_target_count, _MAX_TEAM_SLOTS),
            ]
        )
        return out

    def _summary_features(
        self,
        state: dict[str, Any],
        *,
        team_pair_count: int,
        shop_team_pair_count: int,
    ) -> list[float]:
        team = _team_slots(state)
        pet_count = 0
        total_attack = 0
        total_health = 0
        total_perm_attack = 0
        total_perm_health = 0
        total_temp_attack = 0
        total_temp_health = 0
        level2 = 0
        level3 = 0
        for slot in team[:_MAX_TEAM_SLOTS]:
            pet_id = str(slot.get("pet_id") or "")
            if not pet_id:
                continue
            pet_count += 1
            attack = max(0, _safe_int(slot.get("attack"), 0))
            health = max(0, _safe_int(slot.get("health"), 0))
            total_attack += int(attack)
            total_health += int(health)

            perm_attack = max(0, _safe_int(slot.get("perm_attack"), attack))
            perm_health = max(0, _safe_int(slot.get("perm_health"), health))
            temp_attack = max(0, _safe_int(slot.get("temp_attack"), max(0, attack - perm_attack)))
            temp_health = max(0, _safe_int(slot.get("temp_health"), max(0, health - perm_health)))
            total_perm_attack += int(perm_attack)
            total_perm_health += int(perm_health)
            total_temp_attack += int(temp_attack)
            total_temp_health += int(temp_health)

            level = _safe_int(slot.get("level"), 1)
            if level >= 3:
                level3 += 1
            elif level == 2:
                level2 += 1

        empty_slots = max(0, _MAX_TEAM_SLOTS - int(pet_count))
        by_index = _shop_by_index(state)
        frozen_count = sum(1 for idx in range(_MAX_SHOP_SLOTS) if bool((by_index.get(idx) or {}).get("frozen")))
        linked_count = sum(
            1
            for idx in range(_MAX_SHOP_SLOTS)
            if bool(str((by_index.get(idx) or {}).get("link_id") or "").strip())
        )
        total_power = int(total_attack + total_health)

        return [
            _norm(pet_count, _MAX_TEAM_SLOTS),
            _norm(empty_slots, _MAX_TEAM_SLOTS),
            _norm(total_attack, _MAX_TEAM_SLOTS * self.max_stat),
            _norm(total_health, _MAX_TEAM_SLOTS * self.max_stat),
            _norm(total_power, _MAX_TEAM_SLOTS * self.max_stat * 2),
            _norm(_safe_int(state.get("gold"), 0), self.max_gold),
            _norm(frozen_count, _MAX_SHOP_SLOTS),
            _norm(linked_count, _MAX_SHOP_SLOTS),
            _norm(level2, _MAX_TEAM_SLOTS),
            _norm(level3, _MAX_TEAM_SLOTS),
            _norm(team_pair_count, _MAX_TEAM_PAIR_COUNT),
            _norm(shop_team_pair_count, _MAX_SHOP_TEAM_PAIR_COUNT),
            _norm(total_perm_attack, _MAX_TEAM_SLOTS * self.max_stat),
            _norm(total_perm_health, _MAX_TEAM_SLOTS * self.max_stat),
            _norm(total_temp_attack, _MAX_TEAM_SLOTS * self.max_stat),
            _norm(total_temp_health, _MAX_TEAM_SLOTS * self.max_stat),
        ]

    def encode(self, state: dict[str, Any]) -> np.ndarray:
        turn = max(1, _safe_int(state.get("turn"), 1))
        shop_tier = max(1, min(_MAX_SHOP_TIER, _safe_int(tier_for_turn(turn), 1)))
        turns_to_next = _turns_to_next_tier(turn, self.max_turn)

        team_combine_flags, team_level_counts, team_pair_count = self._team_combine_flags(state)
        shop_combine_counts, shop_team_pair_count = self._shop_combine_counts(state, team_level_counts)
        link_group_sizes = _shop_link_groups(state)

        vec: list[float] = [
            _norm(turn, self.max_turn),
            _norm(_safe_int(state.get("gold"), 0), self.max_gold),
            _norm(_safe_int(state.get("lives"), 0), self.max_lives),
            _norm(_safe_int(state.get("trophies"), 0), self.max_trophies),
            _norm(shop_tier, _MAX_SHOP_TIER),
            _norm(turns_to_next, max(1, self.max_turn)),
        ]

        team = _team_slots(state)
        by_index = _shop_by_index(state)

        for i in range(_MAX_TEAM_SLOTS):
            slot = team[i] if i < len(team) else {}
            pet_id = str(slot.get("pet_id") or "")
            combine_shop_flag = bool(
                pet_id
                and _safe_int(slot.get("level"), 1) < 3
                and any(
                    str((by_index.get(shop_idx) or {}).get("slot_type") or "") == "pet"
                    and str((by_index.get(shop_idx) or {}).get("item_id") or "") == pet_id
                    for shop_idx in range(_MAX_SHOP_SLOTS)
                )
            )
            vec.extend(
                self._team_slot(
                    state,
                    slot,
                    i,
                    combine_team=team_combine_flags,
                    combine_shop=combine_shop_flag,
                )
            )

        for i in range(_MAX_SHOP_SLOTS):
            slot = by_index.get(i)
            vec.extend(
                self._shop_slot(
                    slot,
                    link_group_size=int(link_group_sizes.get(i, 0)),
                    buy_combine_target_count=int(shop_combine_counts.get(i, 0)),
                )
            )

        vec.extend(
            self._summary_features(
                state,
                team_pair_count=int(team_pair_count),
                shop_team_pair_count=int(shop_team_pair_count),
            )
        )

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape != (self.size,):
            raise ValueError(f"observation_size_mismatch:got={arr.shape}:expected={(self.size,)}")
        return arr


@dataclass
class StateVectorEncoderV3(StateVectorEncoderV2):
    """v3 encoder extends v2 with PBRS-visible summary features."""

    copy_focus_beta: float = 0.20
    max_rolls_this_turn: int = 15

    def __post_init__(self) -> None:
        super().__post_init__()
        self.mode = OBSERVATION_MODE_V3
        self._summary_size = int(self._summary_size) + 4
        self.size = (
            self._global_size
            + (_MAX_TEAM_SLOTS * self._team_slot_size)
            + (_MAX_SHOP_SLOTS * self._shop_slot_size)
            + self._summary_size
        )
        self.vocab_fingerprint = _fingerprint(
            {
                "mode": self.mode,
                "pet_ids": self._pet_ids,
                "food_ids": self._food_ids,
                "equip_ids": self._equip_ids,
                "status_ids": self._status_ids,
                "feature_layout": "v3_pbrs_summary_v1",
                "copy_focus_beta": float(self.copy_focus_beta),
                "max_rolls_this_turn": int(self.max_rolls_this_turn),
            }
        )

    def _phi_perm_norm(self, state: dict[str, Any], *, turn: int) -> float:
        base_stats = load_turtle_catalog().get("pets", {}).get("base_stats", {}) or {}
        powers: list[int] = []
        for slot in _team_slots(state)[:_MAX_TEAM_SLOTS]:
            pet_id = str(slot.get("pet_id") or "")
            if not pet_id:
                continue
            base = base_stats.get(pet_id, {}) if isinstance(base_stats, dict) else {}
            base_attack = _safe_int((base or {}).get("attack"), _safe_int(slot.get("attack"), 0))
            base_health = _safe_int((base or {}).get("health"), _safe_int(slot.get("health"), 0))
            perm_attack = _safe_int(slot.get("perm_attack"), _safe_int(slot.get("attack"), 0))
            perm_health = _safe_int(slot.get("perm_health"), _safe_int(slot.get("health"), 0))
            add_power = max(0, (perm_attack - base_attack) + (perm_health - base_health))
            powers.append(int(add_power))
        powers.sort(reverse=True)
        top2_perm = int(sum(powers[:2]))
        ref = max(1.0, 8.0 + (4.0 * float(max(1, int(turn)))))
        return float(max(0.0, min(1.0, float(top2_perm) / ref)))

    def _phi_copies_norm(self, state: dict[str, Any]) -> float:
        invested_by_pet: dict[str, float] = {}
        for slot in _team_slots(state)[:_MAX_TEAM_SLOTS]:
            pet_id = str(slot.get("pet_id") or "")
            if not pet_id:
                continue
            exp = max(0, _safe_int(slot.get("exp"), 0))
            invested = 1.0 + float(exp)
            invested_by_pet[pet_id] = float(invested_by_pet.get(pet_id, 0.0) + invested)
        line_scores: list[float] = []
        beta = max(0.0, float(self.copy_focus_beta))
        for invested_copies in invested_by_pet.values():
            c = max(0.0, float(invested_copies))
            lvl2_focus = min(c, 3.0) / 3.0
            lvl3_tail = beta * (min(max(c - 3.0, 0.0), 3.0) / 3.0)
            line_scores.append(float(lvl2_focus + lvl3_tail))
        line_scores.sort(reverse=True)
        top3_sum = float(sum(line_scores[:3]))
        max_sum = max(1e-6, 3.0 * (1.0 + beta))
        return float(max(0.0, min(1.0, top3_sum / max_sum)))

    def _above_tier_on_board_count_norm(self, state: dict[str, Any], *, turn: int) -> float:
        shop_tier = max(1, min(_MAX_SHOP_TIER, _safe_int(tier_for_turn(max(1, int(turn))), 1)))
        count = 0
        for slot in _team_slots(state)[:_MAX_TEAM_SLOTS]:
            pet_id = str(slot.get("pet_id") or "")
            if not pet_id:
                continue
            pet_tier = _safe_int(self._pet_tier_by_id.get(pet_id, 0), 0)
            if int(pet_tier) > int(shop_tier):
                count += 1
        return _norm(count, _MAX_TEAM_SLOTS)

    def _rolls_this_turn_norm(self, state: dict[str, Any]) -> float:
        meta = state.get("meta")
        if not isinstance(meta, dict):
            return 0.0
        rolls = max(0, _safe_int(meta.get("training_rolls_this_turn"), 0))
        return _norm(rolls, max(1, int(self.max_rolls_this_turn)))

    def _summary_features(
        self,
        state: dict[str, Any],
        *,
        team_pair_count: int,
        shop_team_pair_count: int,
    ) -> list[float]:
        base = super()._summary_features(
            state,
            team_pair_count=int(team_pair_count),
            shop_team_pair_count=int(shop_team_pair_count),
        )
        turn = max(1, _safe_int(state.get("turn"), 1))
        base.extend(
            [
                self._phi_perm_norm(state, turn=turn),
                self._phi_copies_norm(state),
                self._above_tier_on_board_count_norm(state, turn=turn),
                self._rolls_this_turn_norm(state),
            ]
        )
        return base


@dataclass
class StateVectorEncoderV4(StateVectorEncoderV3):
    """v4 encoder extends v3 with one-turn opponent-context features."""

    opponent_hash_buckets: int = DEFAULT_HASH_BUCKETS

    def __post_init__(self) -> None:
        super().__post_init__()
        self.mode = OBSERVATION_MODE_V4
        self._opponent_context_size = (int(self.opponent_hash_buckets) * 3) + 6
        self._summary_size = int(self._summary_size) + int(self._opponent_context_size)
        self.size = (
            self._global_size
            + (_MAX_TEAM_SLOTS * self._team_slot_size)
            + (_MAX_SHOP_SLOTS * self._shop_slot_size)
            + self._summary_size
        )
        self.vocab_fingerprint = _fingerprint(
            {
                "mode": self.mode,
                "pet_ids": self._pet_ids,
                "food_ids": self._food_ids,
                "equip_ids": self._equip_ids,
                "status_ids": self._status_ids,
                "feature_layout": "v4_one_turn_context_v1",
                "copy_focus_beta": float(self.copy_focus_beta),
                "max_rolls_this_turn": int(self.max_rolls_this_turn),
                "opponent_hash_buckets": int(self.opponent_hash_buckets),
            }
        )

    def _summary_features(
        self,
        state: dict[str, Any],
        *,
        team_pair_count: int,
        shop_team_pair_count: int,
    ) -> list[float]:
        base = super()._summary_features(
            state,
            team_pair_count=int(team_pair_count),
            shop_team_pair_count=int(shop_team_pair_count),
        )
        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        versus = meta.get("versus") if isinstance(meta.get("versus"), dict) else {}
        last_opponent_team = versus.get("last_opponent_team") if isinstance(versus.get("last_opponent_team"), list) else []
        opponent_lives = max(0, _safe_int(versus.get("opponent_lives"), 0))
        context_vec = encode_last_opponent_context(
            last_opponent_team,
            opponent_lives=int(opponent_lives),
            turn=max(1, _safe_int(state.get("turn"), 1)),
            buckets=int(self.opponent_hash_buckets),
        )
        base.extend(context_vec.astype(np.float32).tolist())
        return base


def build_state_encoder(
    *,
    observation_mode: str,
    max_turn: int,
) -> StateVectorEncoder | StateVectorEncoderV2 | StateVectorEncoderV3 | StateVectorEncoderV4:
    mode = normalize_observation_mode(observation_mode, default=DEFAULT_OBSERVATION_MODE)
    if mode == OBSERVATION_MODE_V1:
        return StateVectorEncoder(max_turn=int(max_turn))
    if mode == OBSERVATION_MODE_V2:
        return StateVectorEncoderV2(max_turn=int(max_turn))
    if mode == OBSERVATION_MODE_V3:
        return StateVectorEncoderV3(max_turn=int(max_turn))
    if mode == OBSERVATION_MODE_V4:
        return StateVectorEncoderV4(max_turn=int(max_turn))
    raise ValueError(f"unknown_observation_mode:{mode}")


def encoder_observation_spec(encoder: Any) -> dict[str, Any]:
    mode = str(getattr(encoder, "mode", OBSERVATION_MODE_V1))
    size = int(getattr(encoder, "size", 0))
    fingerprint = str(getattr(encoder, "vocab_fingerprint", "")).strip()
    if not fingerprint:
        fingerprint = _fingerprint(
            {
                "mode": mode,
                "size": size,
                "class": str(type(encoder).__name__),
            }
        )
    return {
        "observation_mode": mode,
        "observation_size": size,
        "observation_vocab_fingerprint": fingerprint,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_model_observation_metadata(model_path: Path) -> dict[str, Any] | None:
    run_dir = Path(model_path).resolve().parent
    candidates: list[dict[str, Any]] = []
    for name in ("metadata.json", "init_metadata.json"):
        path = run_dir / name
        payload = _read_json(path)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["_source_metadata_file"] = str(path)
            candidates.append(payload)
    for payload in candidates:
        if all(payload.get(k) is not None for k in OBSERVATION_SPEC_KEYS):
            return payload
    if candidates:
        return candidates[0]
    return None


def resolve_observation_mode_for_model(
    *,
    model_path: Path,
    requested_mode: str,
    default_mode: str = DEFAULT_OBSERVATION_MODE,
) -> str:
    requested = normalize_observation_mode(
        requested_mode,
        default=default_mode,
        allow_auto=True,
    )
    if requested != OBSERVATION_MODE_AUTO:
        return requested

    meta = load_model_observation_metadata(Path(model_path))
    if isinstance(meta, dict):
        mode_raw = meta.get("observation_mode")
        if isinstance(mode_raw, str) and mode_raw.strip():
            try:
                return normalize_observation_mode(mode_raw, default=default_mode)
            except Exception:
                pass
    # Legacy fallback for old checkpoints without observation metadata.
    return OBSERVATION_MODE_V1


def assert_model_observation_compatible(
    *,
    model_path: Path,
    encoder: Any,
    context: str,
) -> dict[str, Any]:
    """Raise iff `encoder` is not what `model_path`'s saved run produced.

    exp09 W5 P0 (fingerprint robustness, codex finding): `OBSERVATION_SPEC_KEYS`
    / `vocab_fingerprint` never depended on `max_turn` -- two encoders built
    with different horizons but the same mode/vocab hash identically, so a
    horizon mismatch used to pass this check silently even though `max_turn`
    changes real encoded VALUES (`_norm(turn, self.max_turn)`,
    `_turns_to_next_tier`). Closed below as a SEPARATE comparison against the
    plain top-level `"max_turn"` hyperparameter field `train_ppo.py` /
    `train_bc_warmstart.py` already write into every run's metadata.json
    (already read back elsewhere too: `tools/build_model_pool.py::
    _load_encoder_max_turn`) -- deliberately NOT folded into
    `encoder_observation_spec()` / `OBSERVATION_SPEC_KEYS` itself, since that
    dict is ALSO compared by a separate direct `==` elsewhere against a BC
    dataset cache's OWN recorded spec (`train_bc_warmstart.py`'s
    `bc_dataset_observation_spec_mismatch` check, against
    `chain_bc_dataset.py`'s manifest); adding a key there would make every
    dataset cache built before this fix spuriously "mismatch" too. This
    checkpoint-metadata check is unaffected by that concern and every real
    checkpoint metadata this function is ever called against already
    carries the field (verified: both `train_ppo.py` write sites do).
    """
    spec_expected = encoder_observation_spec(encoder)
    meta = load_model_observation_metadata(Path(model_path))
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"incompatible_observation_config:{context}:metadata_missing:{Path(model_path)}"
        )
    missing = [k for k in OBSERVATION_SPEC_KEYS if meta.get(k) is None]
    if missing:
        raise RuntimeError(
            "incompatible_observation_config:"
            f"{context}:metadata_missing_fields={','.join(missing)}:"
            f"source={meta.get('_source_metadata_file')}"
        )
    mismatches: list[str] = []
    for key in OBSERVATION_SPEC_KEYS:
        lhs = str(meta.get(key))
        rhs = str(spec_expected.get(key))
        if lhs != rhs:
            mismatches.append(f"{key}:{lhs}!={rhs}")

    # exp09 W5 P0 Fix A: chain-PPO now DECOUPLES the V4 encoder's own horizon
    # from the episode-length cap (`runtime.resolve_encoder_max_turn`), so a
    # run's plain top-level `"max_turn"` metadata field (the episode cap) can
    # legitimately differ from the encoder it was actually built with.
    # `train_ppo.py` / `init_ppo_model.py` / `train_bc_warmstart.py` all now
    # ALSO write a separate `"encoder_max_turn"` field recording exactly
    # that -- prefer it here, falling back to the legacy plain `"max_turn"`
    # only for metadata written before this fix (where the two were always
    # identical by construction, so the fallback is exactly correct for
    # them, not an approximation).
    meta_encoder_max_turn = meta.get("encoder_max_turn", meta.get("max_turn"))
    if meta_encoder_max_turn is None:
        mismatches.append("max_turn:missing_in_metadata")
    else:
        try:
            encoder_max_turn: int | None = int(getattr(encoder, "max_turn"))
        except Exception:
            encoder_max_turn = None
        if encoder_max_turn is None or int(meta_encoder_max_turn) != encoder_max_turn:
            mismatches.append(f"max_turn:{meta_encoder_max_turn}!={encoder_max_turn}")

    if mismatches:
        raise RuntimeError(
            "incompatible_observation_config:"
            f"{context}:mismatch={';'.join(mismatches)}:"
            f"source={meta.get('_source_metadata_file')}"
        )
    return {
        "source_metadata_file": str(meta.get("_source_metadata_file")),
        **{k: meta.get(k) for k in OBSERVATION_SPEC_KEYS},
        "max_turn": meta_encoder_max_turn,
    }
