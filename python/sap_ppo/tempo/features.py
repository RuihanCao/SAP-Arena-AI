"""Feature builders for one-turn tempo predictor/value models."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_HASH_BUCKETS = 128

# v2 (position-aware, ability-aware) value-feature encoder constants.
VALUE_FEATURE_MODE_V1 = "v1"
VALUE_FEATURE_MODE_V2 = "v2"
VALUE_FEATURE_MODES = (VALUE_FEATURE_MODE_V1, VALUE_FEATURE_MODE_V2)
_V2_MAX_TEAM_SLOTS = 5
# Per-slot v2 width = occupied flag + pet one-hot (catalog vocab + 1 OOV bucket)
# + [attack, health, level, exp, has_equipment].
_V2_STAT_FIELD_COUNT = 5


def _hash_bucket(token: str, buckets: int) -> int:
    key = str(token or "").strip().lower().encode("utf-8")
    digest = hashlib.sha1(key).digest()
    return int.from_bytes(digest[:4], "big") % max(1, int(buckets))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_team_slots(team: Any) -> list[dict[str, Any]]:
    rows = [r for r in (team or []) if isinstance(r, dict)]
    out: list[dict[str, Any]] = []
    for idx in range(5):
        slot = rows[idx] if idx < len(rows) else {}
        out.append(
            {
                "slot_index": idx,
                "pet_id": slot.get("pet_id"),
                "pet_name": slot.get("pet_name"),
                "pet_name_id": slot.get("pet_name_id"),
                "attack": _safe_int(slot.get("attack"), 0),
                "health": _safe_int(slot.get("health"), 0),
                "level": _safe_int(slot.get("level"), 1),
                "exp": _safe_int(slot.get("exp"), 0),
                "equipment_id": slot.get("equipment_id"),
            }
        )
    return out


def parsed_pets_to_team(parsed_pets: Any) -> list[dict[str, Any]]:
    """Convert replaybot-style pet configs into canonical slot rows."""
    out: list[dict[str, Any]] = []
    source = list(parsed_pets) if isinstance(parsed_pets, list) else []
    for idx in range(5):
        pet = source[idx] if idx < len(source) else None
        if not isinstance(pet, dict):
            out.append(
                {
                    "slot_index": idx,
                    "pet_id": None,
                    "pet_name": None,
                    "pet_name_id": None,
                    "attack": 0,
                    "health": 0,
                    "level": 1,
                    "exp": 0,
                    "equipment_id": None,
                }
            )
            continue
        out.append(
            {
                "slot_index": idx,
                "pet_id": None,
                "pet_name": str(pet.get("name") or "").strip() or None,
                "pet_name_id": None,
                "attack": _safe_int(pet.get("attack"), 0),
                "health": _safe_int(pet.get("health"), 0),
                "level": 1 + (_safe_int(pet.get("exp"), 0) // 2),
                "exp": _safe_int(pet.get("exp"), 0),
                "equipment_id": (
                    str((pet.get("equipment") or {}).get("name") or "").strip() if isinstance(pet.get("equipment"), dict) else None
                ),
            }
        )
    return out


def _slot_token(slot: dict[str, Any]) -> str:
    pet_id = str(slot.get("pet_id") or "").strip()
    pet_name_id = str(slot.get("pet_name_id") or "").strip()
    pet_name = str(slot.get("pet_name") or "").strip()
    if pet_id:
        return pet_id
    if pet_name_id:
        return f"nameid:{pet_name_id.lower()}"
    if pet_name:
        return f"name:{pet_name.lower()}"
    return "empty"


def _team_hash_vector(team: Any, *, buckets: int, prefix: str) -> np.ndarray:
    rows = normalize_team_slots(team)
    out = np.zeros(max(1, int(buckets)) * 3, dtype=np.float32)
    for slot in rows:
        token = f"{prefix}:{_slot_token(slot)}"
        b = _hash_bucket(token, buckets)
        atk = max(0.0, float(_safe_int(slot.get("attack"), 0)))
        hp = max(0.0, float(_safe_int(slot.get("health"), 0)))
        out[b] += 1.0
        out[int(buckets) + b] += atk / 50.0
        out[int(buckets) * 2 + b] += hp / 50.0
    return out


def _team_summary(team: Any) -> np.ndarray:
    rows = normalize_team_slots(team)
    occupied = [r for r in rows if _slot_token(r) != "empty"]
    pet_count = float(len(occupied))
    total_attack = float(sum(max(0, _safe_int(r.get("attack"), 0)) for r in occupied))
    total_health = float(sum(max(0, _safe_int(r.get("health"), 0)) for r in occupied))
    total_power = total_attack + total_health
    avg_level = (float(sum(max(1, _safe_int(r.get("level"), 1)) for r in occupied)) / pet_count) if pet_count > 0 else 0.0
    return np.asarray(
        [
            pet_count / 5.0,
            total_attack / 250.0,
            total_health / 250.0,
            total_power / 500.0,
            avg_level / 3.0,
        ],
        dtype=np.float32,
    )


def encode_predictor_context(last_opponent_team: Any, turn: int, *, buckets: int = DEFAULT_HASH_BUCKETS) -> np.ndarray:
    turn_vec = np.asarray([max(1, int(turn)) / 25.0], dtype=np.float32)
    return np.concatenate(
        [
            turn_vec,
            _team_summary(last_opponent_team),
            _team_hash_vector(last_opponent_team, buckets=buckets, prefix="opp_last"),
        ],
        axis=0,
    ).astype(np.float32)


def encode_last_opponent_context(
    last_opponent_team: Any,
    *,
    opponent_lives: int,
    turn: int,
    buckets: int = DEFAULT_HASH_BUCKETS,
) -> np.ndarray:
    predictor_ctx = encode_predictor_context(last_opponent_team, int(turn), buckets=buckets)
    team_ctx = predictor_ctx[1:] if int(predictor_ctx.shape[0]) > 1 else np.zeros(0, dtype=np.float32)
    opp_lives_vec = np.asarray([float(max(0, int(opponent_lives))) / 6.0], dtype=np.float32)
    return np.concatenate([team_ctx, opp_lives_vec], axis=0).astype(np.float32)


def encode_value_features(
    state: dict[str, Any],
    opponent_team: Any,
    *,
    buckets: int = DEFAULT_HASH_BUCKETS,
) -> np.ndarray:
    st = state if isinstance(state, dict) else {}
    turn = max(1, _safe_int(st.get("turn"), 1))
    gold = max(0, _safe_int(st.get("gold"), 0))
    lives = max(0, _safe_int(st.get("lives"), 0))
    trophies = max(0, _safe_int(st.get("trophies"), 0))

    meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
    versus_meta = meta.get("versus") if isinstance(meta.get("versus"), dict) else {}
    try:
        opp_lives = int(versus_meta.get("opponent_lives", 0))
    except Exception:
        opp_lives = 0

    global_vec = np.asarray(
        [
            float(turn) / 25.0,
            float(gold) / 10.0,
            float(lives) / 6.0,
            float(trophies) / 10.0,
            float(max(0, opp_lives)) / 6.0,
        ],
        dtype=np.float32,
    )

    player_team = st.get("team") if isinstance(st.get("team"), list) else []
    player_summary = _team_summary(player_team)
    opp_summary = _team_summary(opponent_team)
    player_hash = _team_hash_vector(player_team, buckets=buckets, prefix="player")
    opp_hash = _team_hash_vector(opponent_team, buckets=buckets, prefix="opp")

    return np.concatenate([global_vec, player_summary, opp_summary, player_hash, opp_hash], axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _v2_pet_vocab() -> tuple[str, ...]:
    """Deterministic pet-id vocabulary for the v2 encoder: the same Turtle-pack
    catalog + sort order `train/observation.py` already uses for its one-hot
    (`load_turtle_catalog()["pets"]["id_to_name_id"]`, sorted by pet_id).
    Opponent pets from non-Turtle packs (observed in ~52% of the fair-frame
    manifest's opponent boards) are not covered by this catalog and fall back
    to a trailing OOV bucket (see `_v2_resolve_pet_index`).
    """
    from ..catalog import load_turtle_catalog

    catalog = load_turtle_catalog()
    pets = catalog.get("pets", {}) if isinstance(catalog.get("pets"), dict) else {}
    id_to_name = pets.get("id_to_name_id") if isinstance(pets.get("id_to_name_id"), dict) else {}
    return tuple(sorted(str(pid) for pid in id_to_name.keys()))


@lru_cache(maxsize=1)
def _v2_name_to_pet_id() -> dict[str, str]:
    """Lowercased name_id/pet_id -> canonical pet_id lookup, built from the same
    catalog. Covers both raw-replay slots (only `pet_name` populated, e.g.
    "Ant") and already-canonical slots (`pet_id` populated, e.g. "pet-ant").
    """
    from ..catalog import load_turtle_catalog

    catalog = load_turtle_catalog()
    pets = catalog.get("pets", {}) if isinstance(catalog.get("pets"), dict) else {}
    name_to_id = pets.get("name_id_to_id") if isinstance(pets.get("name_id_to_id"), dict) else {}
    id_to_name = pets.get("id_to_name_id") if isinstance(pets.get("id_to_name_id"), dict) else {}

    out: dict[str, str] = {}
    for name_id, pid in (name_to_id or {}).items():
        key = str(name_id).strip().lower()
        if key:
            out[key] = str(pid)
    for pid in (id_to_name or {}).keys():
        key = str(pid).strip().lower()
        out.setdefault(key, str(pid))
    return out


def _v2_slugify_pet_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").strip().lower()).strip("-")
    return f"pet-{slug}" if slug else ""


def _v2_resolve_pet_index(slot: dict[str, Any], *, vocab_index: dict[str, int], name_lookup: dict[str, str]) -> int | None:
    """Resolve a normalized team slot to a vocab index, or None if the slot is
    occupied by a pet outside the v2 catalog vocab (OOV; e.g. a non-Turtle-pack
    opponent pet). Mirrors `eval_tempo_planner._resolve_pet_id`'s
    exact-match-then-slugify strategy (verified against the actual replay
    corpus: base pets match name_id exactly; the "Zombie Cricket"/"Zombie Fly"
    summon tokens only resolve via the slug fallback, since the catalog's
    name_id for those is an internal token ("CricketToken"/"FlyToken") that
    never appears in replay data verbatim).
    """
    pet_id = str(slot.get("pet_id") or "").strip()
    if pet_id:
        key = pet_id.lower()
        resolved = name_lookup.get(key, pet_id)
        if resolved in vocab_index:
            return vocab_index[resolved]

    for field in ("pet_name_id", "pet_name"):
        raw = str(slot.get(field) or "").strip()
        if not raw:
            continue
        resolved = name_lookup.get(raw.lower())
        if resolved is not None and resolved in vocab_index:
            return vocab_index[resolved]
        slug = _v2_slugify_pet_name(raw)
        if slug and slug in vocab_index:
            return vocab_index[slug]
    return None


def _encode_team_slots_v2(
    team: Any,
    *,
    vocab_size: int,
    vocab_index: dict[str, int],
    name_lookup: dict[str, str],
) -> np.ndarray:
    rows = normalize_team_slots(team)
    onehot_width = vocab_size + 1  # +1 trailing OOV bucket
    per_slot_width = 1 + onehot_width + _V2_STAT_FIELD_COUNT
    out = np.zeros(len(rows) * per_slot_width, dtype=np.float32)
    for i, slot in enumerate(rows):
        if _slot_token(slot) == "empty":
            continue  # leave this slot's whole span at zero
        base = i * per_slot_width
        out[base] = 1.0  # occupied flag

        onehot_base = base + 1
        idx = _v2_resolve_pet_index(slot, vocab_index=vocab_index, name_lookup=name_lookup)
        out[onehot_base + (idx if idx is not None else vocab_size)] = 1.0

        stat_base = onehot_base + onehot_width
        out[stat_base + 0] = max(0.0, float(_safe_int(slot.get("attack"), 0))) / 50.0
        out[stat_base + 1] = max(0.0, float(_safe_int(slot.get("health"), 0))) / 50.0
        out[stat_base + 2] = max(1.0, float(_safe_int(slot.get("level"), 1))) / 3.0
        out[stat_base + 3] = max(0.0, float(_safe_int(slot.get("exp"), 0))) / 5.0
        out[stat_base + 4] = 1.0 if slot.get("equipment_id") else 0.0
    return out


def value_feature_width(mode: str = VALUE_FEATURE_MODE_V1, *, buckets: int = DEFAULT_HASH_BUCKETS) -> int:
    """Fixed feature-vector width for a given `--value-feature-mode`, without
    building a dummy state. Used to sanity-check `feature_dim` after training.
    """
    mode_norm = str(mode or VALUE_FEATURE_MODE_V1).strip().lower()
    global_and_summary = 5 + 5 + 5
    if mode_norm == VALUE_FEATURE_MODE_V1:
        return global_and_summary + max(1, int(buckets)) * 3 * 2
    if mode_norm == VALUE_FEATURE_MODE_V2:
        vocab_size = len(_v2_pet_vocab())
        per_slot_width = 1 + (vocab_size + 1) + _V2_STAT_FIELD_COUNT
        return global_and_summary + per_slot_width * _V2_MAX_TEAM_SLOTS * 2
    raise ValueError(f"unknown_value_feature_mode:{mode}:allowed={VALUE_FEATURE_MODES}")


def encode_value_features_v2(state: dict[str, Any], opponent_team: Any) -> np.ndarray:
    """Keeps v1's global scalars (turn/gold/lives/trophies/opponent_lives) and
    team summary stats (count/attack/health/power/avg_level) unchanged, but
    replaces the position-blind, hashed bag-of-pets block with a per-slot
    block (occupied flag + one-hot pet identity over the real catalog vocab,
    NOT a 128-bucket hash + attack/health/level/exp/has_equipment), for both
    the player's team (slot order preserved) and the opponent's team."""
    st = state if isinstance(state, dict) else {}
    turn = max(1, _safe_int(st.get("turn"), 1))
    gold = max(0, _safe_int(st.get("gold"), 0))
    lives = max(0, _safe_int(st.get("lives"), 0))
    trophies = max(0, _safe_int(st.get("trophies"), 0))

    meta = st.get("meta") if isinstance(st.get("meta"), dict) else {}
    versus_meta = meta.get("versus") if isinstance(meta.get("versus"), dict) else {}
    try:
        opp_lives = int(versus_meta.get("opponent_lives", 0))
    except Exception:
        opp_lives = 0

    global_vec = np.asarray(
        [
            float(turn) / 25.0,
            float(gold) / 10.0,
            float(lives) / 6.0,
            float(trophies) / 10.0,
            float(max(0, opp_lives)) / 6.0,
        ],
        dtype=np.float32,
    )

    player_team = st.get("team") if isinstance(st.get("team"), list) else []
    player_summary = _team_summary(player_team)
    opp_summary = _team_summary(opponent_team)

    vocab = _v2_pet_vocab()
    vocab_index = {pid: i for i, pid in enumerate(vocab)}
    name_lookup = _v2_name_to_pet_id()

    player_slots = _encode_team_slots_v2(player_team, vocab_size=len(vocab), vocab_index=vocab_index, name_lookup=name_lookup)
    opp_slots = _encode_team_slots_v2(opponent_team, vocab_size=len(vocab), vocab_index=vocab_index, name_lookup=name_lookup)

    return np.concatenate(
        [global_vec, player_summary, opp_summary, player_slots, opp_slots], axis=0
    ).astype(np.float32)


def encode_value_features_for_mode(
    state: dict[str, Any],
    opponent_team: Any,
    *,
    mode: str = VALUE_FEATURE_MODE_V1,
    buckets: int = DEFAULT_HASH_BUCKETS,
) -> np.ndarray:
    """Encode value features for mode."""
    mode_norm = str(mode or VALUE_FEATURE_MODE_V1).strip().lower()
    if mode_norm == VALUE_FEATURE_MODE_V1:
        return encode_value_features(state, opponent_team, buckets=buckets)
    if mode_norm == VALUE_FEATURE_MODE_V2:
        return encode_value_features_v2(state, opponent_team)
    raise ValueError(f"unknown_value_feature_mode:{mode}:allowed={VALUE_FEATURE_MODES}")


def scalar_tempo_target(win_prob: float, draw_prob: float) -> float:
    return float(win_prob) + (0.5 * float(draw_prob))

