"""Usage (PYTHONPATH=python, from the checkout root, venv python):"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from .chain_bc_dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SPLIT_SEED,
)
from .chain_bc_dataset import DEFAULT_MAX_TURN as DATASET_DEFAULT_MAX_TURN
from .chain_bc_dataset import (
    DEFAULT_VAL_FRACTION,
    OBSERVATION_MODE_V4,
    WEIGHT_MODE_CHOICES,
    WEIGHT_MODE_RANK,
    ChainBcCache,
    compute_train_weights,
    encode_extra_dataset_rows,
    exclude_games,
    load_cache,
    rank_weight_report,
    split_by_game,
)
from .env import ACTION_CATALOG, build_action_type_ids
from .kl_ppo import load_bc_policy_weights

DEFAULT_LR = 3e-4
DEFAULT_BATCH = 1024
DEFAULT_EPOCHS = 20
DEFAULT_SEED = 42
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_STATUS_INTERVAL_EPOCHS = 5
DEFAULT_TORCH_THREADS = 4
POLICY_MODE_CHOICES = ("type_balanced", "hierarchical")


DEFAULT_NET_ARCH = "64,64"
MIX_MODE_BALANCED = "balanced"
MIX_MODE_CHOICES = (MIX_MODE_BALANCED,)
# Teacher-row confidence weight: 1.0 + clip(teacher_margin, 0, 1) -- a
# confident teacher pick (margin >= 1.0) weighs up to 2x a flat human row.
TEACHER_WEIGHT_BASE = 1.0
TEACHER_MARGIN_CLIP_LO = 0.0
TEACHER_MARGIN_CLIP_HI = 1.0


OVERRIDE_WEIGHT_BASE = 0.5


DEFAULT_FEATURES = "flat"
FEATURES_CHOICES = ("flat", "slot_v1", "slot_attn")
DEFAULT_EMBED_DIM = 32
DEFAULT_ITEM_DIM = 16
DEFAULT_SLOT_DIM = 48
DEFAULT_OPP_DIM = 32
DEFAULT_ATTN_HEADS = 2


def _status(msg: str) -> None:
    print(f"[status] {msg}", flush=True)


def _batch_slices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, int(indices.shape[0]), step):
        out.append(indices[start : start + step])
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_net_arch(spec: str) -> list[int]:
    """"64,64" -> [64, 64]; "256,256" -> [256, 256]. Comma-separated positive
    ints, at least one. Raises ValueError with the offending spec on any
    malformed input (never silently falls back to a default -- a typo'd
    `--net-arch` must fail loudly, not silently train the wrong arm)."""
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"net_arch_spec_empty:{spec!r}")
    try:
        sizes = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"net_arch_spec_not_ints:{spec!r}:{exc}") from exc
    if any(s <= 0 for s in sizes):
        raise ValueError(f"net_arch_spec_non_positive:{spec!r}")
    return sizes


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_teacher_weights(
    teacher_margin: np.ndarray, *, human_weight_sum: float, mix_mode: str = MIX_MODE_BALANCED
) -> tuple[np.ndarray, float]:
    """Per-teacher-row weight = `TEACHER_WEIGHT_BASE + clip(teacher_margin, 0, 1)`
    (a confident teacher pick, margin >= 1.0, weighs up to 2x a flat human
    row), then -- for "balanced" mix-mode (the only mode currently
    supported) -- rescaled by ONE global factor so
    `sum(returned_weights) == human_weight_sum`, i.e. the human and teacher
    sources contribute ~equal TOTAL weight per epoch (design item 3b).

    Pure/stateless (no I/O, no model/dataset construction) so this is
    directly unit-testable without a real training run -- see
    `test_w2_distill.py::TestMixModeWeighting`.

    Returns `(weights, mix_scale)`.
    """
    teacher_weights_raw = TEACHER_WEIGHT_BASE + np.clip(
        np.asarray(teacher_margin, dtype=np.float64), TEACHER_MARGIN_CLIP_LO, TEACHER_MARGIN_CLIP_HI
    )
    teacher_weight_sum_raw = float(teacher_weights_raw.sum())
    if mix_mode == MIX_MODE_BALANCED:
        mix_scale = float(human_weight_sum) / max(teacher_weight_sum_raw, 1e-9)
    else:
        raise ValueError(f"unknown_mix_mode:{mix_mode}:allowed={MIX_MODE_CHOICES}")
    return teacher_weights_raw * mix_scale, mix_scale


def compute_override_weights(teacher_margin: np.ndarray) -> np.ndarray:
    """Pure/stateless (no I/O, no model/dataset construction) so this is
    directly unit-testable without a real training run, mirroring
    `compute_teacher_weights`'s own testability design."""
    return OVERRIDE_WEIGHT_BASE + np.clip(
        np.asarray(teacher_margin, dtype=np.float64), TEACHER_MARGIN_CLIP_LO, TEACHER_MARGIN_CLIP_HI
    )


def _resolve_features_config(args: argparse.Namespace) -> dict[str, Any]:
    """Keys deliberately match `_build_model`'s own parameter names (d_pet
    etc.), not the CLI flag names, so `train()` can splat-adjacent them
    without a second renaming table."""
    return {
        "features": str(getattr(args, "features", DEFAULT_FEATURES)),
        "d_pet": int(getattr(args, "embed_dim", DEFAULT_EMBED_DIM)),
        "d_item": int(getattr(args, "item_dim", DEFAULT_ITEM_DIM)),
        "d_slot": int(getattr(args, "slot_dim", DEFAULT_SLOT_DIM)),
        "d_opp": int(getattr(args, "opp_dim", DEFAULT_OPP_DIM)),
        "attn_heads": int(getattr(args, "attn_heads", DEFAULT_ATTN_HEADS)),
    }


def _parse_cache_max_turn(manifest: dict[str, Any]) -> int | None:
    """Parse cache max turn."""
    try:
        return int(manifest.get("max_turn") or DATASET_DEFAULT_MAX_TURN)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Minimal Gym shell env: only its observation_space/action_space matter.
# See module docstring -- MaskablePPO never resets/steps it during BC training.
# ---------------------------------------------------------------------------

def _build_dummy_vec_env(obs_size: int, n_actions: int) -> Any:
    import gymnasium as gym
    from gymnasium import spaces
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _DummyBcShellEnv(gym.Env):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
            self.action_space = spaces.Discrete(n_actions)

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action: int):
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        def action_masks(self) -> np.ndarray:
            return np.ones(n_actions, dtype=bool)

    def _mask_fn(env: Any) -> np.ndarray:
        return env.action_masks()

    def _make_env():
        return ActionMasker(_DummyBcShellEnv(), _mask_fn)

    return DummyVecEnv([_make_env])


def _build_model(
    *,
    obs_size: int,
    n_actions: int,
    action_type_ids: list[int],
    policy_mode: str,
    type_balance_strength: float,
    lr: float,
    seed: int,
    device: str,
    net_arch: list[int] | None = None,
    features: str = DEFAULT_FEATURES,
    observation_mode: str | None = None,
    max_turn: int | None = None,
    d_pet: int = DEFAULT_EMBED_DIM,
    d_item: int = DEFAULT_ITEM_DIM,
    d_slot: int = DEFAULT_SLOT_DIM,
    d_opp: int = DEFAULT_OPP_DIM,
    attn_heads: int = DEFAULT_ATTN_HEADS,
) -> Any:
    from sb3_contrib import MaskablePPO

    from .policy import HierarchicalMaskablePolicy, TypeBalancedMaskablePolicy

    policy_class = HierarchicalMaskablePolicy if policy_mode == "hierarchical" else TypeBalancedMaskablePolicy
    vec_env = _build_dummy_vec_env(obs_size, n_actions)
    policy_kwargs: dict[str, Any] = {
        "action_type_ids": action_type_ids,
        "type_balance_strength": float(type_balance_strength),
    }


    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)


    if str(features) != DEFAULT_FEATURES:
        from .features import SlotEmbeddingExtractor, build_observation_layout

        if observation_mode is None or max_turn is None:
            raise ValueError(
                "build_model_slot_features_requires_manifest_context:"
                f"features={features!r}:observation_mode={observation_mode!r}:max_turn={max_turn!r}"
            )
        layout = build_observation_layout(str(observation_mode), int(max_turn))
        if int(layout["total"]) != int(obs_size):
            raise ValueError(
                "build_model_slot_features_layout_size_mismatch:"
                f"layout_total={layout['total']}!=obs_size={obs_size}:"
                f"observation_mode={observation_mode}:max_turn={max_turn}"
            )
        policy_kwargs["features_extractor_class"] = SlotEmbeddingExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "layout": layout,
            "d_pet": int(d_pet),
            "d_item": int(d_item),
            "d_slot": int(d_slot),
            "d_opp": int(d_opp),
            "arch": str(features),
            "attn_heads": int(attn_heads),
        }

    model = MaskablePPO(
        policy_class,
        vec_env,
        learning_rate=float(lr),
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
        clip_range=0.2,
        device=str(device),
        seed=int(seed),
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    return model


def _set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    for group in getattr(optimizer, "param_groups", []):
        group["lr"] = float(learning_rate)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _action_type_names_and_ids() -> tuple[list[int], list[str]]:
    return build_action_type_ids()


def _val_metrics(
    *,
    policy: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    device: th.device,
    action_type_ids: list[int],
    type_names: list[str],
) -> dict[str, Any]:
    policy.set_training_mode(False)
    n = int(X_val.shape[0])
    type_ids_arr = np.asarray(action_type_ids, dtype=np.int64)
    loss_sum = 0.0
    entropy_sum = 0.0
    target_prob_sum = 0.0
    top1_correct = 0
    top3_correct = 0
    pred_type_counts = np.zeros(len(type_names), dtype=np.int64)
    human_type_counts = np.zeros(len(type_names), dtype=np.int64)

    indices = np.arange(n, dtype=np.int64)
    with th.no_grad():
        for batch_idx in _batch_slices(indices, batch_size):
            obs_t = th.as_tensor(X_val[batch_idx], dtype=th.float32, device=device)
            actions_t = th.as_tensor(y_val[batch_idx], dtype=th.long, device=device)

            _values, log_prob, entropy = policy.evaluate_actions(obs_t, actions_t, action_masks=None)
            dist = policy.get_distribution(obs_t, action_masks=None)
            probs = dist.distribution.probs

            batch_nll = -log_prob
            loss_sum += float(batch_nll.sum().item())
            entropy_sum += float((entropy if entropy is not None else th.zeros_like(batch_nll)).sum().item())
            target_prob_sum += float(th.exp(log_prob).sum().item())

            top1_pred = th.argmax(probs, dim=1)
            top1_correct += int((top1_pred == actions_t).sum().item())
            top3_pred = th.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
            top3_correct += int((top3_pred == actions_t.unsqueeze(1)).any(dim=1).sum().item())

            pred_types = type_ids_arr[top1_pred.cpu().numpy()]
            human_types = type_ids_arr[y_val[batch_idx]]
            pred_type_counts += np.bincount(pred_types, minlength=len(type_names))
            human_type_counts += np.bincount(human_types, minlength=len(type_names))

    denom = max(1, n)
    action_mix = {
        type_names[i]: {
            "human_frac": float(human_type_counts[i]) / float(denom),
            "pred_frac": float(pred_type_counts[i]) / float(denom),
        }
        for i in range(len(type_names))
    }
    return {
        "rows": int(n),
        "avg_nll": float(loss_sum / denom),
        "avg_entropy": float(entropy_sum / denom),
        "avg_target_prob": float(target_prob_sum / denom),
        "top1_acc": float(top1_correct / denom),
        "top3_acc": float(top3_correct / denom),
        "action_mix": action_mix,
    }


def _format_action_mix_table(action_mix: dict[str, dict[str, float]]) -> str:
    rows = sorted(action_mix.items(), key=lambda kv: -kv[1]["human_frac"])
    lines = [f"{'TYPE':16s} {'human%':>8s} {'argmax%':>8s}"]
    for name, fracs in rows:
        lines.append(f"{name:16s} {100.0 * fracs['human_frac']:8.2f} {100.0 * fracs['pred_frac']:8.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def _load_human_plus_teacher_data(args: argparse.Namespace) -> dict[str, Any]:
    """Returns a dict with the same shape `_load_extra_dataset_only_data`
    returns (`X_train`/`y_train`/`weights_train`/`X_val`/`y_val`/`obs_size`/
    `n_actions`/`provenance`), so `train()` can call either loader
    interchangeably."""
    n_actions = len(ACTION_CATALOG)
    cache: ChainBcCache = load_cache(args.cache_dir)
    obs_size = int(cache.X.shape[1])
    if int(cache.y.max()) >= n_actions or int(cache.y.min()) < 0:
        raise RuntimeError(f"cache_action_index_out_of_range:min={cache.y.min()}:max={cache.y.max()}:n_actions={n_actions}")
    nan_count = int(np.isnan(cache.X).sum())
    inf_count = int(np.isinf(cache.X).sum())
    if nan_count or inf_count:
        raise RuntimeError(f"cache_observation_has_nan_or_inf:nan={nan_count}:inf={inf_count}")


    cache_observation_mode = str(cache.manifest.get("observation_mode") or OBSERVATION_MODE_V4)
    cache_max_turn = _parse_cache_max_turn(cache.manifest)


    holdout_ids: set[str] = set()
    holdout_report: dict[str, Any] | None = None
    if args.holdout_games is not None:
        holdout_ids = set(json.loads(Path(args.holdout_games).read_text(encoding="utf-8")))
        cache, holdout_report = exclude_games(cache, holdout_ids)
        _status(
            f"holdout_games={args.holdout_games}: requested={holdout_report['excluded_game_ids_requested']} "
            f"found_in_cache={holdout_report['excluded_game_ids_found_in_cache']} "
            f"not_in_cache={holdout_report['excluded_game_ids_not_in_cache']} | "
            f"games {holdout_report['games_before']}->{holdout_report['games_after']} "
            f"samples {holdout_report['samples_before']}->{holdout_report['samples_after']} "
            f"(excluded {holdout_report['samples_excluded']})"
        )

    split = split_by_game(cache.game_id, val_fraction=float(args.val_fraction), seed=int(args.split_seed))
    train_mask, val_mask = split["train_mask"], split["val_mask"]
    _status(
        f"split: train games={split['n_games_train']} samples={split['n_samples_train']} | "
        f"val games={split['n_games_val']} samples={split['n_samples_val']} (seed={split['seed']})"
    )

    if holdout_ids:
        train_games = set(cache.game_id[train_mask].tolist())
        val_games = set(cache.game_id[val_mask].tolist())
        leaked_train = holdout_ids & train_games
        leaked_val = holdout_ids & val_games
        if leaked_train or leaked_val:
            raise RuntimeError(
                f"holdout_games_leaked_into_split:train={sorted(leaked_train)[:5]}:val={sorted(leaked_val)[:5]}"
            )
        _status(f"holdout_assert_ok: 0 of {len(holdout_ids)} manifest games present in train or val split")

    X_train, y_train = cache.X[train_mask], cache.y[train_mask]
    X_val, y_val = cache.X[val_mask], cache.y[val_mask]
    skill_train = cache.player_skill[train_mask]

    weights_train = compute_train_weights(skill_train, mode=str(args.weight_mode))
    weight_report = (
        rank_weight_report(skill_train, [1500.0, 1745.0, 1852.0, 1908.0])
        if str(args.weight_mode) == WEIGHT_MODE_RANK
        else {"note": "flat mode: all weights are 1.0"}
    )
    _status(f"weight_mode={args.weight_mode} mean_weight={float(np.mean(weights_train)):.6f}")


    n_human_train = int(X_train.shape[0])
    mix_stats: dict[str, Any] | None = None
    extra_dataset_file_hashes: dict[str, str] = {}
    if args.extra_dataset:


        if cache_max_turn is None:
            raise RuntimeError(
                "extra_dataset_requires_parseable_cache_max_turn:"
                f"manifest_value={cache.manifest.get('max_turn')!r}"
            )
        extra = encode_extra_dataset_rows(
            [str(p) for p in args.extra_dataset],
            observation_mode=cache_observation_mode,
            max_turn=cache_max_turn,
        )
        if extra.n_miss:
            raise RuntimeError(
                f"extra_dataset_action_index_coverage_regression:{extra.n_miss}_misses:"
                f"examples={json.dumps(extra.miss_examples[:5])} -- merge_distill_dataset.py's "
                "gate should have caught this before training; do not proceed"
            )
        if int(extra.X.shape[1]) != int(X_train.shape[1]):
            raise RuntimeError(
                f"extra_dataset_observation_size_mismatch:extra={extra.X.shape[1]}:"
                f"human_cache={X_train.shape[1]} (encoder/max_turn drifted between the two builds)"
            )
        n_teacher_train = int(extra.X.shape[0])
        if n_teacher_train == 0:
            raise RuntimeError(f"extra_dataset_empty_after_encoding:{extra.paths}")

        human_weight_sum = float(weights_train.sum())
        teacher_weights, mix_scale = compute_teacher_weights(
            extra.teacher_margin, human_weight_sum=human_weight_sum, mix_mode=str(args.mix_mode)
        )
        teacher_weight_sum_raw = float(
            (TEACHER_WEIGHT_BASE + np.clip(extra.teacher_margin, TEACHER_MARGIN_CLIP_LO, TEACHER_MARGIN_CLIP_HI)).sum()
        )
        teacher_weight_sum_rescaled = float(teacher_weights.sum())

        X_train = np.concatenate([X_train, extra.X], axis=0)
        y_train = np.concatenate([y_train, extra.y], axis=0)
        weights_train = np.concatenate([weights_train, teacher_weights.astype(np.float64)], axis=0)

        margin_sorted = np.sort(extra.teacher_margin.astype(np.float64))
        extra_dataset_file_hashes = {str(p): _sha256_file(Path(p)) for p in args.extra_dataset}
        mix_stats = {
            "mix_mode": str(args.mix_mode),
            "n_human_train": n_human_train,
            "n_teacher_train": n_teacher_train,
            "n_train_combined": int(X_train.shape[0]),
            "human_weight_sum": human_weight_sum,
            "teacher_weight_sum_raw": teacher_weight_sum_raw,
            "teacher_weight_sum_rescaled": teacher_weight_sum_rescaled,
            "mix_scale": float(mix_scale),
            "teacher_margin_stats": {
                "min": float(margin_sorted[0]),
                "median": float(np.median(margin_sorted)),
                "mean": float(margin_sorted.mean()),
                "max": float(margin_sorted[-1]),
            },
            "teacher_score_mean": float(extra.teacher_score.mean()) if n_teacher_train else None,
            "myopic_score_mean": float(extra.myopic_score.mean()) if n_teacher_train else None,
            "candidates_n_mean": float(extra.candidates_n.mean()) if n_teacher_train else None,
            "extra_dataset_source_tag_counts": {
                str(tag): int(count)
                for tag, count in zip(*np.unique(extra.source, return_counts=True))
            },
        }
        _status(
            f"extra_dataset: n_teacher_train={n_teacher_train} n_human_train={n_human_train} "
            f"mix_mode={args.mix_mode} mix_scale={mix_scale:.6f} "
            f"human_weight_sum={human_weight_sum:.3f} teacher_weight_sum_rescaled={teacher_weight_sum_rescaled:.3f}"
        )

    provenance = {
        "extra_dataset_only": False,
        "cache_dir": str(args.cache_dir),
        "source_dataset_dir": cache.manifest.get("source_dataset_dir"),
        "source_schema_version": cache.manifest.get("source_schema_version"),
        "observation_mode": cache.manifest.get("observation_mode"),


        "max_turn": cache_max_turn,
        "observation_vocab_fingerprint": cache.manifest.get("observation_vocab_fingerprint"),
        "weight_mode": str(args.weight_mode),
        "weight_report": weight_report,


        "extra_dataset_paths": [str(p) for p in args.extra_dataset] if args.extra_dataset else [],
        "extra_val_dataset_paths": [],
        "extra_dataset_file_hashes": extra_dataset_file_hashes,
        "mix_mode": str(args.mix_mode) if args.extra_dataset else None,
        "mix_stats": mix_stats,
        "holdout_games_path": (str(args.holdout_games) if args.holdout_games is not None else None),
        "holdout_report": holdout_report,
        "split_seed": int(args.split_seed),
        "val_fraction": float(args.val_fraction),
        "n_games_train": int(split["n_games_train"]),
        "n_games_val": int(split["n_games_val"]),
        "n_samples_train": int(split["n_samples_train"]),
        "n_samples_val": int(split["n_samples_val"]),
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "weights_train": weights_train,
        "X_val": X_val,
        "y_val": y_val,
        "obs_size": obs_size,
        "n_actions": n_actions,
        "provenance": provenance,
    }


def _load_extra_dataset_only_data(args: argparse.Namespace) -> dict[str, Any]:
    """Same return shape as `_load_human_plus_teacher_data` (see its
    docstring)."""
    n_actions = len(ACTION_CATALOG)
    if not args.extra_dataset:
        raise RuntimeError("extra_dataset_only_requires_extra_dataset")
    if not args.extra_val_dataset:
        raise RuntimeError("extra_dataset_only_requires_extra_val_dataset")

    extra_train = encode_extra_dataset_rows(
        [str(p) for p in args.extra_dataset], observation_mode=OBSERVATION_MODE_V4, max_turn=DATASET_DEFAULT_MAX_TURN
    )
    extra_val = encode_extra_dataset_rows(
        [str(p) for p in args.extra_val_dataset], observation_mode=OBSERVATION_MODE_V4, max_turn=DATASET_DEFAULT_MAX_TURN
    )
    if extra_train.n_miss or extra_val.n_miss:
        raise RuntimeError(
            f"extra_dataset_only_action_index_coverage_regression:"
            f"train_misses={extra_train.n_miss}:val_misses={extra_val.n_miss}:"
            "build_override_finetune_set.py's own gate should have caught this -- do not proceed"
        )
    n_train = int(extra_train.X.shape[0])
    n_val = int(extra_val.X.shape[0])
    if n_train == 0:
        raise RuntimeError(f"extra_dataset_only_train_empty:{extra_train.paths}")
    if n_val == 0:
        raise RuntimeError(f"extra_dataset_only_val_empty:{extra_val.paths}")
    if int(extra_train.X.shape[1]) != int(extra_val.X.shape[1]):
        raise RuntimeError(
            f"extra_dataset_only_observation_size_mismatch:train={extra_train.X.shape[1]}:val={extra_val.X.shape[1]}"
        )
    if int(extra_train.y.max()) >= n_actions or int(extra_train.y.min()) < 0:
        raise RuntimeError(
            f"extra_dataset_only_action_index_out_of_range:"
            f"min={extra_train.y.min()}:max={extra_train.y.max()}:n_actions={n_actions}"
        )
    nan_count = int(np.isnan(extra_train.X).sum()) + int(np.isnan(extra_val.X).sum())
    inf_count = int(np.isinf(extra_train.X).sum()) + int(np.isinf(extra_val.X).sum())
    if nan_count or inf_count:
        raise RuntimeError(f"extra_dataset_only_observation_has_nan_or_inf:nan={nan_count}:inf={inf_count}")

    weights_train = compute_override_weights(extra_train.teacher_margin).astype(np.float64)
    n_games_train = int(len(set(extra_train.game_id.tolist())))
    n_games_val = int(len(set(extra_val.game_id.tolist())))
    _status(
        f"extra_dataset_only: train rows={n_train} (games={n_games_train}) "
        f"val rows={n_val} (games={n_games_val}) weight_mean={float(weights_train.mean()):.6f}"
    )

    margin_sorted = np.sort(extra_train.teacher_margin.astype(np.float64))
    weight_report = {
        "note": "extra_dataset_only mode: weight = compute_override_weights(teacher_margin) "
        "= 0.5+clip(margin,0,1); no rank/flat human weight mode applies",
        "min": float(weights_train.min()),
        "median": float(np.median(weights_train)),
        "mean": float(weights_train.mean()),
        "max": float(weights_train.max()),
    }
    mix_stats = {
        "mix_mode": "extra_dataset_only",
        "n_human_train": 0,
        "n_teacher_train": n_train,
        "n_train_combined": n_train,
        "teacher_margin_stats": {
            "min": float(margin_sorted[0]),
            "median": float(np.median(margin_sorted)),
            "mean": float(margin_sorted.mean()),
            "max": float(margin_sorted[-1]),
        },
        "teacher_score_mean": float(extra_train.teacher_score.mean()) if n_train else None,
        "myopic_score_mean": float(extra_train.myopic_score.mean()) if n_train else None,
        "candidates_n_mean": float(extra_train.candidates_n.mean()) if n_train else None,
        "extra_dataset_source_tag_counts": {
            str(tag): int(count) for tag, count in zip(*np.unique(extra_train.source, return_counts=True))
        },
    }
    extra_dataset_file_hashes = {
        str(p): _sha256_file(Path(p)) for p in (list(args.extra_dataset) + list(args.extra_val_dataset))
    }

    provenance = {
        "extra_dataset_only": True,
        "cache_dir": None,
        "source_dataset_dir": None,
        "source_schema_version": None,
        "observation_mode": OBSERVATION_MODE_V4,


        "max_turn": DATASET_DEFAULT_MAX_TURN,
        "observation_vocab_fingerprint": None,
        "weight_mode": "override_margin",
        "weight_report": weight_report,
        "extra_dataset_paths": [str(p) for p in args.extra_dataset],
        "extra_val_dataset_paths": [str(p) for p in args.extra_val_dataset],
        "extra_dataset_file_hashes": extra_dataset_file_hashes,
        "mix_mode": None,
        "mix_stats": mix_stats,
        "holdout_games_path": None,
        "holdout_report": None,
        "split_seed": None,
        "val_fraction": None,
        "n_games_train": n_games_train,
        "n_games_val": n_games_val,
        "n_samples_train": n_train,
        "n_samples_val": n_val,
    }

    return {
        "X_train": extra_train.X,
        "y_train": extra_train.y,
        "weights_train": weights_train,
        "X_val": extra_val.X,
        "y_val": extra_val.y,
        "obs_size": int(extra_train.X.shape[1]),
        "n_actions": n_actions,
        "provenance": provenance,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    # See module docstring "Perf note" -- this policy is tiny relative to this
    # box's core count; the default intra-op thread pool badly oversubscribes
    # matmuls this small (~100x slower wall-clock, identical numeric result).
    th.set_num_threads(max(1, int(args.torch_threads)))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    th.manual_seed(int(args.seed))


    data = (
        _load_extra_dataset_only_data(args)
        if bool(args.extra_dataset_only)
        else _load_human_plus_teacher_data(args)
    )
    X_train, y_train, weights_train = data["X_train"], data["y_train"], data["weights_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    obs_size, n_actions = int(data["obs_size"]), int(data["n_actions"])
    provenance = data["provenance"]

    action_type_ids, type_names = _action_type_names_and_ids()
    net_arch = parse_net_arch(str(args.net_arch))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "DONE").unlink(missing_ok=True)


    features_config = _resolve_features_config(args)
    observation_mode_for_layout = str(provenance.get("observation_mode") or OBSERVATION_MODE_V4)
    _raw_max_turn = provenance.get("max_turn")
    max_turn_for_layout = int(_raw_max_turn) if _raw_max_turn is not None else None

    model = _build_model(
        obs_size=obs_size,
        n_actions=n_actions,
        action_type_ids=action_type_ids,
        policy_mode=str(args.policy_mode),
        type_balance_strength=float(args.type_balance_strength),
        lr=float(args.lr),
        seed=int(args.seed),
        device=str(args.device),
        net_arch=net_arch,
        features=features_config["features"],
        observation_mode=observation_mode_for_layout,
        max_turn=max_turn_for_layout,
        d_pet=features_config["d_pet"],
        d_item=features_config["d_item"],
        d_slot=features_config["d_slot"],
        d_opp=features_config["d_opp"],
        attn_heads=features_config["attn_heads"],
    )
    device = next(model.policy.parameters()).device
    optimizer = model.policy.optimizer
    _set_optimizer_lr(optimizer, float(args.lr))


    init_checkpoint_report: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        _bc_model, init_checkpoint_report = load_bc_policy_weights(model, args.init_checkpoint, device=str(device))
        init_checkpoint_report["checkpoint_sha256"] = _sha256_file(Path(args.init_checkpoint))
        _status(
            f"init_checkpoint: {args.init_checkpoint} loaded (strict load_state_dict), "
            f"n_params={init_checkpoint_report['n_params']} "
            f"checksum_after={init_checkpoint_report['policy_weight_checksum_after']:.6f} "
            f"checksum_bc={init_checkpoint_report['policy_weight_checksum_bc']:.6f}"
        )

    writer = None
    if bool(args.tensorboard):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    weights_t_full = th.as_tensor(weights_train, dtype=th.float32, device=device)
    y_train_t_full = th.as_tensor(y_train, dtype=th.long, device=device)
    X_train_t_full = th.as_tensor(X_train, dtype=th.float32, device=device)

    rng = np.random.default_rng(int(args.seed))
    epoch_history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    started_at = time.time()

    initial_val = _val_metrics(
        policy=model.policy,
        X_val=X_val,
        y_val=y_val,
        batch_size=int(args.batch),
        device=device,
        action_type_ids=action_type_ids,
        type_names=type_names,
    )
    _status(
        f"epoch 0 (init): val_loss={initial_val['avg_nll']:.4f} "
        f"val_top1={initial_val['top1_acc']:.4f} val_top3={initial_val['top3_acc']:.4f} "
        f"val_target_prob={initial_val['avg_target_prob']:.4f}"
    )

    n_train = int(X_train.shape[0])
    for epoch in range(1, int(args.epochs) + 1):
        model.policy.set_training_mode(True)
        order = rng.permutation(n_train).astype(np.int64)
        batch_losses: list[float] = []
        batch_target_probs: list[float] = []
        batch_entropies: list[float] = []

        for batch_idx_np in _batch_slices(order, int(args.batch)):
            batch_idx = th.as_tensor(batch_idx_np, dtype=th.long, device=device)
            obs_t = X_train_t_full.index_select(0, batch_idx)
            actions_t = y_train_t_full.index_select(0, batch_idx)
            weights_t = weights_t_full.index_select(0, batch_idx)
            weight_denom = th.clamp(weights_t.sum(), min=1e-6)

            optimizer.zero_grad(set_to_none=True)
            _values, log_prob, entropy = model.policy.evaluate_actions(obs_t, actions_t, action_masks=None)
            batch_nll = -log_prob
            loss = (weights_t * batch_nll).sum() / weight_denom
            loss.backward()
            if float(args.max_grad_norm) > 0.0:
                th.nn.utils.clip_grad_norm_(model.policy.parameters(), float(args.max_grad_norm))
            optimizer.step()

            with th.no_grad():
                batch_losses.append(float(loss.item()))
                ent = entropy if entropy is not None else th.zeros_like(batch_nll)
                batch_entropies.append(float(((weights_t * ent).sum() / weight_denom).item()))
                batch_target_probs.append(float(((th.exp(log_prob) * weights_t).sum() / weight_denom).item()))

        val = _val_metrics(
            policy=model.policy,
            X_val=X_val,
            y_val=y_val,
            batch_size=int(args.batch),
            device=device,
            action_type_ids=action_type_ids,
            type_names=type_names,
        )
        train_loss = float(np.mean(batch_losses)) if batch_losses else None
        train_entropy = float(np.mean(batch_entropies)) if batch_entropies else None
        train_target_prob = float(np.mean(batch_target_probs)) if batch_target_probs else None

        record = {
            "epoch": int(epoch),
            "train_weighted_loss": train_loss,
            "train_weighted_entropy": train_entropy,
            "train_weighted_target_prob": train_target_prob,
            "val_loss": val["avg_nll"],
            "val_top1_acc": val["top1_acc"],
            "val_top3_acc": val["top3_acc"],
            "val_target_prob": val["avg_target_prob"],
            "val_entropy": val["avg_entropy"],
            "val_action_mix": val["action_mix"],
        }
        epoch_history.append(record)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/entropy", train_entropy, epoch)
            writer.add_scalar("train/target_prob", train_target_prob, epoch)
            writer.add_scalar("val/loss", val["avg_nll"], epoch)
            writer.add_scalar("val/top1_acc", val["top1_acc"], epoch)
            writer.add_scalar("val/top3_acc", val["top3_acc"], epoch)
            writer.add_scalar("val/target_prob", val["avg_target_prob"], epoch)
            writer.add_scalar("val/entropy", val["avg_entropy"], epoch)
            for type_name, fracs in val["action_mix"].items():
                writer.add_scalar(f"action_mix/human/{type_name}", fracs["human_frac"], epoch)
                writer.add_scalar(f"action_mix/pred/{type_name}", fracs["pred_frac"], epoch)

        interval = max(1, int(args.status_interval_epochs))
        if epoch == 1 or epoch == int(args.epochs) or (epoch % interval) == 0:
            _status(
                f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val['avg_nll']:.4f} "
                f"val_top1={val['top1_acc']:.4f} val_top3={val['top3_acc']:.4f} "
                f"val_target_prob={val['avg_target_prob']:.4f}"
            )
            if writer is not None:
                writer.add_text("action_mix_table", _format_action_mix_table(val["action_mix"]), epoch)

        if val["avg_nll"] < best_val_loss:
            best_val_loss = val["avg_nll"]
            best_epoch = epoch
            model.save(str(out_dir / "checkpoint_best.zip"))

    model.save(str(out_dir / "checkpoint_final.zip"))
    elapsed = time.time() - started_at

    final_val = epoch_history[-1] if epoch_history else initial_val
    _status(
        f"FINAL action-mix table (epoch {final_val.get('epoch', 0)}):\n"
        + _format_action_mix_table(final_val.get("val_action_mix", initial_val["action_mix"]))
    )

    metadata = {
        "run_name": str(args.run_name or out_dir.name),
        **provenance,
        "observation_size": int(obs_size),
        "action_space_size": int(n_actions),
        "action_type_names": type_names,
        "policy_mode_requested": str(args.policy_mode),
        "policy_class": type(model.policy).__name__,
        "type_balance_strength": float(args.type_balance_strength),
        "net_arch": list(net_arch),
        "net_arch_requested": str(args.net_arch),


        "features": features_config["features"],
        "features_extractor_class": type(model.policy.features_extractor).__name__,
        "features_dims": {
            "d_pet": features_config["d_pet"],
            "d_item": features_config["d_item"],
            "d_slot": features_config["d_slot"],
            "d_opp": features_config["d_opp"],
            "attn_heads": features_config["attn_heads"],
        },
        "lr": float(args.lr),
        "batch": int(args.batch),
        "epochs": int(args.epochs),
        "max_grad_norm": float(args.max_grad_norm),
        "torch_threads": int(args.torch_threads),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "initial_val_metrics": initial_val,
        "final_val_metrics": final_val,
        "elapsed_seconds": float(round(elapsed, 3)),
        "checkpoint_best_path": str(out_dir / "checkpoint_best.zip"),
        "checkpoint_final_path": str(out_dir / "checkpoint_final.zip"),
        "unmasked_training": True,
        "bc_trained": True,


        "init_checkpoint_path": (str(args.init_checkpoint) if args.init_checkpoint is not None else None),
        "init_checkpoint_report": init_checkpoint_report,
    }
    _write_json(out_dir / "metadata.json", metadata)
    _write_json(out_dir / "bc_stats.json", {"epoch_history": epoch_history, "initial_metrics": initial_val})
    (out_dir / "DONE").write_text(
        json.dumps({"ok": True, "finished_at": time.time(), "elapsed_seconds": elapsed, "best_epoch": best_epoch}),
        encoding="utf-8",
    )
    if writer is not None:
        writer.close()
    _status(f"DONE: {out_dir} (elapsed={elapsed:.1f}s, best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")
    return metadata


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight-mode", type=str, choices=WEIGHT_MODE_CHOICES, default=WEIGHT_MODE_RANK)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    ap.add_argument(
        "--holdout-games",
        type=Path,
        default=None,
        help=(
            "JSON list of game_ids to exclude before splitting the training dataset."
        ),
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--policy-mode", type=str, choices=POLICY_MODE_CHOICES, default="type_balanced")
    ap.add_argument("--type-balance-strength", type=float, default=1.0)
    ap.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--status-interval-epochs", type=int, default=DEFAULT_STATUS_INTERVAL_EPOCHS)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help="Intra-op thread count for torch (see 'Perf note' above -- this tiny "
        "policy runs ~100x faster at a small thread count than at the core count).",
    )
    ap.add_argument(
        "--net-arch",
        type=str,
        default=DEFAULT_NET_ARCH,
        help="comma-separated hidden layer sizes, e.g. '64,64' (default, reproduces the pre-W2 implicit sb3 default byte-for-byte) or '256,256' (the W2 wide arm). Shared pi/vf trunk, passed straight into policy_kwargs['net_arch'].",
    )
    ap.add_argument(
        "--extra-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="Additional teacher-row JSONL dataset; repeat for multiple files.",
    )
    ap.add_argument(
        "--mix-mode",
        type=str,
        choices=MIX_MODE_CHOICES,
        default=MIX_MODE_BALANCED,
        help="only meaningful with --extra-dataset. 'balanced' (only choice): rescale all teacher-row weights by one global factor so human and teacher sources contribute ~equal total weight per epoch.",
    )
    ap.add_argument(
        "--extra-dataset-only",
        action="store_true",
        default=False,
        help="Train exclusively from the additional teacher-row dataset.",
    )
    ap.add_argument(
        "--extra-val-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="required iff --extra-dataset-only. One or more teacher-row JSONL files used as the FIXED validation set (best-val-loss checkpoint selection) -- e.g. tools/build_override_finetune_set.py's override_val.jsonl.",
    )
    ap.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Warm-start policy weights from an existing BC checkpoint.",
    )
    ap.add_argument(
        "--features",
        type=str,
        choices=FEATURES_CHOICES,
        default=DEFAULT_FEATURES,
        help="Feature extractor: flat observations, slot encoders, or slot attention.",
    )
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help="pet embedding width (d_pet). Only used when --features != flat.",
    )
    ap.add_argument(
        "--item-dim",
        type=int,
        default=DEFAULT_ITEM_DIM,
        help="equipment/food embedding width (d_item). Only used when --features != flat.",
    )
    ap.add_argument(
        "--slot-dim",
        type=int,
        default=DEFAULT_SLOT_DIM,
        help="per-slot encoder output width (d_slot). Only used when --features != flat.",
    )
    ap.add_argument(
        "--opp-dim",
        type=int,
        default=DEFAULT_OPP_DIM,
        help="opponent-context block output width (d_opp). Only used when --features != flat.",
    )
    ap.add_argument(
        "--attn-heads",
        type=int,
        default=DEFAULT_ATTN_HEADS,
        help="attention head count, 'slot_attn' only (ignored by 'slot_v1'/'flat').",
    )
    args = ap.parse_args(argv)

    metadata = train(args)
    print(json.dumps({"ok": True, "out_dir": str(args.out), "best_val_loss": metadata["best_val_loss"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
