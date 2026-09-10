"""BC trainer for exp09 W4a Phase B: weighted cloning of dataset_v2 samples.

Consumes the `chain_bc_dataset.py` cache (an internal dataset path, encoded
v4 states + ACTION_CATALOG indices) and trains a policy of the EXACT SAME
class/construction (Maskable)PPO W5 KL-PPO will initialize from --
`TypeBalancedMaskablePolicy` / `HierarchicalMaskablePolicy` over the 1993-dim
observation space + 309 discrete actions (see `sap_ppo.train.policy`,
`sap_ppo.train.env.ACTION_CATALOG`) -- so the saved checkpoint is directly
loadable via `MaskablePPO.load(...)` as a PPO warm-start.

Policy-shell construction note: MaskablePPO/its policy only need an env for
its `observation_space`/`action_space` shape at construction time (see
`MaskablePPO._setup_model`, which never calls `env.reset()`/`.step()`) -- the
real `TrainingEnv`/opponent-provider stack (fixtures, replay DB, ...) plays no
role in supervised BC and is deliberately NOT pulled in here. A minimal
`_DummyBcShellEnv` with the right Gym spaces stands in; W5 will supply the
REAL env when it loads this checkpoint for actual PPO rollouts.

Training is UNMASKED end to end: `evaluate_actions(..., action_masks=None)`
skips `MaskableDistribution.apply_masking` entirely (see
`TypeBalancedMaskablePolicy._apply_masking_with_debug`: `if action_masks is
None: return`), so the log-prob of the human's action is always computed
against the full 309-way softmax. The existing "type-balance" prior
(`type_balance_strength`, a fixed architectural bias toward equal probability
mass per ACTION_CATALOG type, e.g. ROLL vs REORDER's 120 parameterizations)
still applies with `action_masks=None` (it falls back to the GLOBAL per-type
counts) -- this is not masking, it is the policy class's own design, kept
because W5 will use the same class with the same default.

Usage (PYTHONPATH=python, from the checkout root, venv python):

  python -m sap_ppo.train.train_chain_bc \\
      --cache-dir an internal dataset path --out an internal dataset path \\
      --weight-mode rank --lr 3e-4 --batch 1024 --epochs 20

Perf note (smoke finding, 2026-07-13): this policy is tiny (net_arch [64,64],
~284k params) relative to this box's 32 cores. Torch's default intra-op
thread pool (one thread per core) causes severe OpenMP oversubscription
overhead on matmuls this small -- observed ~100x slower wall-clock than
`--torch-threads 4` for numerically IDENTICAL results (best_val_loss matched
to 12 significant figures across the two configs). `--torch-threads` defaults
to a small value for exactly this reason; do not raise it back toward the
core count without re-timing.

exp09 W2 (distillation round 1) additions -- everything above is UNCHANGED
(every new flag defaults to the pre-W2 behavior byte-for-byte):

- `--net-arch` (default "64,64", i.e. `[64, 64]` passed EXPLICITLY into
  `policy_kwargs["net_arch"]` -- verified to reproduce the previous implicit
  sb3 default byte-for-byte: same param count / same `mlp_extractor` module
  structure / identical `best_val_loss` to 12 significant figures on a fixed-
  seed smoke run before vs. after this flag existed). `"256,256"` is the W2
  wide arm.
- `--extra-dataset` (repeatable path(s) to `tools/gen_distill_dataset.py`
  teacher-row JSONL files, default None = no behavior change at all) +
  `--mix-mode balanced` (only choice, only meaningful with `--extra-dataset`):
  teacher rows are loaded via `chain_bc_dataset.encode_extra_dataset_rows`
  (same v4 encoder, same `ACTION_CATALOG` index map as the human cache) and
  folded into the TRAIN split only -- never the val split, so best-val-loss
  checkpoint selection keeps measuring fidelity to real human play, unchanged
  from every prior run. Per-row weight: human rows keep whatever
  `--weight-mode` already gives them (the W2 launch scripts pass `flat`, i.e.
  1.0 each, per the segment design); teacher rows get
  `1.0 + clip(teacher_margin, 0, 1)` (a confident teacher pick, margin>=1.0,
  weighs up to 2x a flat human row). "balanced" mix-mode then rescales EVERY
  teacher weight by one global factor `human_weight_sum / teacher_weight_sum`
  so the two sources contribute ~equal TOTAL weight per epoch (this is a
  weight-sum balance, not a resampling scheme -- the existing per-epoch
  shuffle-and-batch loop is otherwise untouched, now just drawing from the
  concatenated pool). `--extra-dataset` file(s) get a sha256 recorded in
  `metadata.json` (`extra_dataset_file_hashes`) alongside every mix
  statistic, so a run's provenance is fully pinned.

exp09 W2b (override-only fine-tune from flat_v2, round 2) additions --
everything above (including every W2 flag) is UNCHANGED; every new flag
below defaults to no-behavior-change:

- `--extra-dataset-only` (default False): train ONLY on `--extra-dataset`
  rows -- NO human dataset_v2 cache at all (`--cache-dir` is ignored
  entirely; `_load_human_plus_teacher_data`'s whole code path, including
  `--weight-mode`/`--holdout-games`/`split_by_game`, never runs). Requires
  `--extra-val-dataset` (one or more teacher-row JSONL files used AS THE
  fixed validation set for best-val-loss checkpoint selection -- e.g.
  `tools/build_override_finetune_set.py`'s `override_val.jsonl`, a
  game-grouped split already carved out upstream, ONCE, at extraction
  time -- this trainer never re-splits). Per-row weight comes from
  `compute_override_weights(teacher_margin)` (base 0.5, not W2's base 1.0,
  and NO "balanced" global rescaling -- there is only one data source
  here, nothing to balance against).
- `--init-checkpoint` (default None): warm-start `model.policy`'s weights
  from an existing `train_chain_bc.py` checkpoint (e.g.
  `flat_v2/checkpoint_best.zip`) via `kl_ppo.load_bc_policy_weights` --
  the EXACT strict-`load_state_dict` + before/after/bc checksum-log
  contract W5's PPO warm-start already relies on, reused verbatim rather
  than a second copy of that logic. Orthogonal to `--extra-dataset-only`
  (either data-loading path can warm-start or not); `net_arch` must match
  the checkpoint's own architecture exactly or the strict load raises.
  The loaded checksum report (`n_params`, checksums before/after/bc, a
  sha256 of the checkpoint file) is recorded in `metadata.json`
  (`init_checkpoint_report`) for provenance, matching `extra_dataset_file_hashes`'s
  pinning convention.

Data loading was extracted into two functions during this addition
(`_load_human_plus_teacher_data` = the pre-W2b code, moved verbatim, no
logic changed; `_load_extra_dataset_only_data` = new) so `train()`'s
model-construction/training-loop/metadata-writing tail is shared,
unchanged, byte-for-byte between both data sources -- only which rows
end up in `X_train`/`y_train`/`weights_train`/`X_val`/`y_val` differs.

exp10 W4.1 (representation upgrade, PLAN.md "Wave 4: emb-bc" section)
additions -- everything above is UNCHANGED; the new flag defaults to the
exact pre-W4.1 behavior byte-for-byte:

- `--features` (default "flat"): "flat" reproduces the pre-W4.1 sb3 default
  EXACTLY -- `policy_kwargs` gains no new key at all, not even a no-op one
  (see `_build_model`'s W4.1 comment, mirroring how `net_arch=None` is
  handled above). "slot_v1"/"slot_attn" attach
  `features.py::SlotEmbeddingExtractor` instead of sb3's stock
  `FlattenExtractor`: shared learned pet/food/equipment embeddings + shared
  per-slot encoders over the SAME 1993-dim v4 layout (observation interface,
  cache, action space all UNCHANGED -- only how the flat vector is turned
  into MLP-trunk input changes). "slot_attn" additionally runs one shared
  self-attention layer over the 14 slot vectors before the trunk.
  `--embed-dim`/`--item-dim`/`--slot-dim`/`--opp-dim`/`--attn-heads` size the
  new extractor's internal embeddings (ignored when `--features flat`). The
  v4 encoder's `observation_mode`/`max_turn` needed to rebuild the layout are
  read from the cache manifest (`provenance["observation_mode"]`/
  `["max_turn"]`, both loader functions now carry the latter), never
  hardcoded. New checkpoints trained with a non-flat `--features` are NOT
  weight-compatible with `flat_v2` (a different architecture) --
  `--init-checkpoint` strict-loading across the two will raise loudly, as
  intended.
  Hardening (codex cross-review): `train()` reads the six new attributes
  via `getattr` with module defaults (`_resolve_features_config`) so
  pre-W4.1 PROGRAMMATIC `argparse.Namespace` callers that lack them keep
  working; and the manifest `max_turn` parse is LENIENT
  (`_parse_cache_max_turn`: missing key -> the module default, un-int-able
  junk -> None) so a legacy cache manifest keeps working for flat runs
  exactly as pre-W4.1, with a loud error only where the value is genuinely
  required (`--extra-dataset` encoding, slot_v1/slot_attn layout
  construction).
"""

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

# exp09 W2 additions (see module docstring).
DEFAULT_NET_ARCH = "64,64"
MIX_MODE_BALANCED = "balanced"
MIX_MODE_CHOICES = (MIX_MODE_BALANCED,)
# Teacher-row confidence weight: 1.0 + clip(teacher_margin, 0, 1) -- a
# confident teacher pick (margin >= 1.0) weighs up to 2x a flat human row.
TEACHER_WEIGHT_BASE = 1.0
TEACHER_MARGIN_CLIP_LO = 0.0
TEACHER_MARGIN_CLIP_HI = 1.0

# exp09 W2b additions (see module docstring "exp09 W2b" section).
# Override-row confidence weight: 0.5 + clip(teacher_margin, 0, 1) -- rows
# reaching this trainer already have teacher_margin > 0 (ties dropped by
# tools/build_override_finetune_set.py), so w in (0.5, 1.5]. Base is 0.5,
# NOT W2's 1.0 -- a different weighting scheme for a different data
# source (override-only, no human mix to balance against).
OVERRIDE_WEIGHT_BASE = 0.5

# exp10 W4.1 additions (see module docstring "exp10 W4.1" section).
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
    """exp09 W2b: per-row weight for override-only fine-tuning (round 2) =
    `OVERRIDE_WEIGHT_BASE + clip(teacher_margin, 0, 1)`, i.e. w in [0.5, 1.5]
    given `tools/build_override_finetune_set.py` only ever emits rows with
    teacher_margin > 0 (ties dropped upstream, so w is always strictly
    above 0.5 in practice).

    Unlike `compute_teacher_weights` (W2's human+teacher mix-BALANCING,
    which rescales by one global factor so two sources contribute equal
    total weight), there is no second data source to balance against here
    -- `--extra-dataset-only` fine-tuning trains on NOTHING but these rows,
    so the raw per-row weight is used directly, no global rescaling.

    Pure/stateless (no I/O, no model/dataset construction) so this is
    directly unit-testable without a real training run, mirroring
    `compute_teacher_weights`'s own testability design.
    """
    return OVERRIDE_WEIGHT_BASE + np.clip(
        np.asarray(teacher_margin, dtype=np.float64), TEACHER_MARGIN_CLIP_LO, TEACHER_MARGIN_CLIP_HI
    )


def _resolve_features_config(args: argparse.Namespace) -> dict[str, Any]:
    """exp10 W4.1 hardening (codex cross-review): resolve the six W4.1
    attributes via `getattr` + module defaults, NOT direct attribute access.
    `train()` has pre-W4.1 PROGRAMMATIC callers that hand-build an
    `argparse.Namespace` without going through `main()`'s parser (which
    always defines every flag); those Namespaces lack the new attributes
    entirely and must keep resolving to the exact flat default rather than
    raising AttributeError. Pure/stateless for direct unit testing
    (`test_w4_features.py::TestResolveFeaturesConfig`), following
    `compute_teacher_weights`'s testability convention.

    Keys deliberately match `_build_model`'s own parameter names (d_pet
    etc.), not the CLI flag names, so `train()` can splat-adjacent them
    without a second renaming table.
    """
    return {
        "features": str(getattr(args, "features", DEFAULT_FEATURES)),
        "d_pet": int(getattr(args, "embed_dim", DEFAULT_EMBED_DIM)),
        "d_item": int(getattr(args, "item_dim", DEFAULT_ITEM_DIM)),
        "d_slot": int(getattr(args, "slot_dim", DEFAULT_SLOT_DIM)),
        "d_opp": int(getattr(args, "opp_dim", DEFAULT_OPP_DIM)),
        "attn_heads": int(getattr(args, "attn_heads", DEFAULT_ATTN_HEADS)),
    }


def _parse_cache_max_turn(manifest: dict[str, Any]) -> int | None:
    """exp10 W4.1 hardening (codex cross-review): LENIENT parse of the cache
    manifest's `max_turn`. Pre-W4.1 this field was only ever parsed inside
    the `--extra-dataset` branch, so a plain flat run never touched it --
    hoisting the parse to unconditional scope (for `provenance["max_turn"]`)
    must not let a legacy/odd manifest start crashing flat runs. Missing or
    falsy key -> `DATASET_DEFAULT_MAX_TURN` (the exact pre-W4.1
    `or`-fallback rule); un-int-able junk -> None instead of the raw
    `int()` ValueError. None is then rejected LOUDLY only where a real
    value is genuinely required: the `--extra-dataset` encode call
    (RuntimeError in `_load_human_plus_teacher_data`, preserving that
    branch's pre-W4.1 crash-on-junk behavior) and `--features slot_v1`/
    `slot_attn` layout construction (`_build_model`'s existing ValueError).
    """
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
    # exp09 W2: `net_arch` defaults to None -- OMITTED from policy_kwargs
    # entirely (not even a [64, 64] default) so every pre-W2 caller of this
    # function (this module's own `train()` before W2, and OTHER experiments'
    # tests/training code that import `_build_model` directly -- e.g.
    # `test_w0b_probes.py`, `test_w5_p1_wiring.py`'s KL-anchor construction --
    # verified via direct import grep) gets the EXACT prior sb3 implicit
    # default, unchanged, with zero risk of this addition silently altering
    # an architecture those callers depend on matching bit-for-bit. Only
    # `train()`'s own W2 code path passes `net_arch` explicitly (parsed from
    # `--net-arch`, default "64,64" -- verified to reproduce that same
    # implicit default byte-for-byte, see module docstring); "256,256" is the
    # W2 wide arm.
    if net_arch is not None:
        policy_kwargs["net_arch"] = list(net_arch)

    # exp10 W4.1: `features` defaults to "flat" -- OMITTED from policy_kwargs
    # entirely (same byte-identical-default contract as `net_arch` above), so
    # every existing caller (flat_v2, W5 PPO warm-start, every test that
    # builds `_build_model` directly) is completely unaffected. Only when a
    # caller explicitly opts into "slot_v1"/"slot_attn" do we build the v4
    # segment layout (from the SAME cache manifest observation_mode/max_turn
    # `train()` already reads into `provenance`) and attach the custom
    # SlotEmbeddingExtractor -- see features.py's module docstring for the
    # architecture (exp10 PLAN.md "Wave 4" section).
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
    """Pre-W2b data loading -- extracted from `train()` VERBATIM (exp09 W2b
    refactor): human dataset_v2 cache (+ optional `--holdout-games`
    exclusion), split by game, rank/flat sample weights, optional
    `--extra-dataset` teacher rows folded into the TRAIN split only
    (balanced mix-mode). Every computation below is the SAME code that ran
    inside `train()` before this function existed -- only the enclosing
    `def`/`return` are new, so this is a pure extraction, not a behavior
    change (the full pre-existing suite + the W2 exact-config smoke both
    exercise this exact path and must still pass byte-for-byte).

    Returns a dict with the same shape `_load_extra_dataset_only_data`
    returns (`X_train`/`y_train`/`weights_train`/`X_val`/`y_val`/`obs_size`/
    `n_actions`/`provenance`), so `train()` can call either loader
    interchangeably.
    """
    n_actions = len(ACTION_CATALOG)
    cache: ChainBcCache = load_cache(args.cache_dir)
    obs_size = int(cache.X.shape[1])
    if int(cache.y.max()) >= n_actions or int(cache.y.min()) < 0:
        raise RuntimeError(f"cache_action_index_out_of_range:min={cache.y.min()}:max={cache.y.max()}:n_actions={n_actions}")
    nan_count = int(np.isnan(cache.X).sum())
    inf_count = int(np.isinf(cache.X).sum())
    if nan_count or inf_count:
        raise RuntimeError(f"cache_observation_has_nan_or_inf:nan={nan_count}:inf={inf_count}")

    # exp10 W4.1: read once, unconditionally (previously only computed inside
    # the `if args.extra_dataset:` block below, purely for encoding teacher
    # rows) -- `train()` now also needs this to build the `--features
    # slot_v1`/`slot_attn` observation layout, whether or not
    # `--extra-dataset` is in play. max_turn parses LENIENTLY (None on
    # un-int-able junk, missing key -> the module default; see
    # `_parse_cache_max_turn`) so a legacy manifest keeps working for flat
    # runs exactly as pre-W4.1, when this field was not parsed here at all.
    cache_observation_mode = str(cache.manifest.get("observation_mode") or OBSERVATION_MODE_V4)
    cache_max_turn = _parse_cache_max_turn(cache.manifest)

    # exp05c eval-manifest holdout (this session's bugfix): exclude these
    # game_ids from the WHOLE cache before split_by_game ever sees them, so
    # they land in neither train nor the monitoring-val split -- they are the
    # external held-out eval set (build with
    # `tools/build_manifest_holdout_gameids.py`). Previously this flag did not
    # exist and no exclusion happened at all: all 163 manifest games were
    # present in the training population and most landed in the train split
    # (leakage).
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

    # --- exp09 W2: fold in the teacher-distillation dataset (TRAIN split
    # only -- see module docstring). `mix_stats`/`extra_dataset_file_hashes`
    # stay None/empty when `--extra-dataset` is omitted, the fully
    # backward-compatible default.
    n_human_train = int(X_train.shape[0])
    mix_stats: dict[str, Any] | None = None
    extra_dataset_file_hashes: dict[str, str] = {}
    if args.extra_dataset:
        # exp10 W4.1 hardening: this branch encodes teacher rows with the
        # cache's own max_turn, so it genuinely REQUIRES a parseable value
        # -- pre-W4.1 the raw `int()` crashed here on junk, and lenient
        # parsing must not turn that into a silently wrong encoding.
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
        # exp10 W4.1: int (missing manifest key -> the module default), or
        # None when the manifest value was unparseable junk -- see
        # `_parse_cache_max_turn`. `train()` reads this straight into
        # `_build_model(max_turn=...)` for the `--features slot_v1`/
        # `slot_attn` layout (which rejects None loudly); flat runs carry
        # it as provenance only, never acting on it.
        "max_turn": cache_max_turn,
        "observation_vocab_fingerprint": cache.manifest.get("observation_vocab_fingerprint"),
        "weight_mode": str(args.weight_mode),
        "weight_report": weight_report,
        # exp09 W2: None/{} when --extra-dataset was not given (no behavior
        # change from every prior run).
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
    """exp09 W2b data loading: override-only fine-tuning, NO human
    dataset_v2 cache at all. Train/val come entirely from `--extra-dataset`
    / `--extra-val-dataset` (`tools/build_override_finetune_set.py`'s own
    pre-split `override_train.jsonl` / `override_val.jsonl` -- the
    game-grouped 90/10 split already happened there, once, at extraction
    time; this loader does not re-split anything, unlike the human-cache
    path's `split_by_game`). Per-row weight =
    `compute_override_weights(teacher_margin)` (module docstring "exp09
    W2b"), not `compute_teacher_weights` -- there is no second data source
    to balance against.

    Same return shape as `_load_human_plus_teacher_data` (see its
    docstring).
    """
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
        # exp10 W4.1: this loader always encodes with the fixed default (see
        # the `encode_extra_dataset_rows(..., max_turn=DATASET_DEFAULT_MAX_TURN)`
        # calls above) -- no manifest to read from, so no fallback needed.
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

    # exp09 W2b: two interchangeable data loaders (see their docstrings) --
    # `--extra-dataset-only` (default False) picks which one runs; every
    # pre-W2b caller leaves it unset and gets the exact prior code path.
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

    # exp10 W4.1: layout context from the provenance fields both data
    # loaders now carry unconditionally (see their docstrings/comments),
    # plus the six features attributes resolved getattr-safely
    # (`_resolve_features_config`, codex hardening: pre-W4.1 programmatic
    # Namespace callers of train() lack them entirely). observation_mode
    # keeps the pre-W4.1 `or`-default -- a wrong default THERE is caught by
    # `_build_model`'s layout-total-vs-obs_size assert. max_turn
    # deliberately does NOT re-default here: `_parse_cache_max_turn`
    # already applied the missing-key default upstream, so None means the
    # manifest value was junk, and silently substituting the default would
    # be caught by NO size check (max_turn changes encoded VALUES, never
    # the observation size). None flows through as-is; `_build_model`
    # ignores it for "flat" and rejects it loudly for slot_v1/slot_attn.
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

    # exp09 W2b: optional BC-init warm start -- orthogonal to
    # extra_dataset_only (works with either data-loading path above);
    # default None = no behavior change for every pre-W2b caller. Reuses
    # kl_ppo.py's load_bc_policy_weights VERBATIM (the same strict-load +
    # checksum-log contract W5's PPO warm-start already relies on) rather
    # than a second copy of that logic. `load_state_dict` only overwrites
    # the EXISTING parameter tensors `_build_model`'s optimizer already
    # references, so `_set_optimizer_lr` above stays correct without
    # needing to be called again after this.
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
        # exp10 W4.1: "flat" (default) alongside every existing field above --
        # "features_extractor_class" mirrors the existing "policy_class" field
        # (read straight off the constructed model, so it reports what
        # actually got built); the rest echo `_resolve_features_config`'s
        # getattr-safe resolution (the same values `_build_model` received,
        # valid even for pre-W4.1 Namespace callers missing the attributes).
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
        # exp09 W2b: None when --init-checkpoint was not given (no behavior
        # change from every prior run).
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
            "Path to a JSON list of game_ids (this cache's own game_id join key, i.e. the "
            "raw replay's outer 'id' field -- see tools/build_manifest_holdout_gameids.py) "
            "to EXCLUDE from the dataset before split_by_game. Used to carve the exp05c "
            "eval manifest's games out of BC training entirely (external held-out eval set)."
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
        help="exp09 W2: comma-separated hidden layer sizes, e.g. '64,64' (default, "
        "reproduces the pre-W2 implicit sb3 default byte-for-byte) or '256,256' "
        "(the W2 wide arm). Shared pi/vf trunk, passed straight into "
        "policy_kwargs['net_arch'].",
    )
    ap.add_argument(
        "--extra-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="exp09 W2: one or more tools/gen_distill_dataset.py teacher-row JSONL "
        "files (dataset_v2 sample shape + source/teacher_margin/teacher_score/"
        "myopic_score/candidates_n). Folded into the TRAIN split only, weighted per "
        "--mix-mode. Default None: no behavior change at all.",
    )
    ap.add_argument(
        "--mix-mode",
        type=str,
        choices=MIX_MODE_CHOICES,
        default=MIX_MODE_BALANCED,
        help="exp09 W2: only meaningful with --extra-dataset. 'balanced' (only "
        "choice): rescale all teacher-row weights by one global factor so human and "
        "teacher sources contribute ~equal total weight per epoch.",
    )
    ap.add_argument(
        "--extra-dataset-only",
        action="store_true",
        default=False,
        help="exp09 W2b: train ONLY on --extra-dataset rows -- no human dataset_v2 cache "
        "at all (--cache-dir is ignored). Requires --extra-val-dataset for the validation "
        "split. Weights come from compute_override_weights(teacher_margin), not the "
        "rank/flat human weight modes. Default False: no behavior change from every "
        "pre-W2b run.",
    )
    ap.add_argument(
        "--extra-val-dataset",
        type=Path,
        nargs="+",
        default=None,
        help="exp09 W2b: required iff --extra-dataset-only. One or more teacher-row "
        "JSONL files used as the FIXED validation set (best-val-loss checkpoint "
        "selection) -- e.g. tools/build_override_finetune_set.py's override_val.jsonl.",
    )
    ap.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="exp09 W2b: warm-start the model's policy weights from an existing "
        "train_chain_bc.py checkpoint (e.g. flat_v2/checkpoint_best.zip) via a strict "
        "load_state_dict + checksum verification (kl_ppo.load_bc_policy_weights, the "
        "same warm-start contract W5's PPO already relies on). Orthogonal to "
        "--extra-dataset-only. Default None: fresh random init, no behavior change from "
        "every pre-W2b run.",
    )
    ap.add_argument(
        "--features",
        type=str,
        choices=FEATURES_CHOICES,
        default=DEFAULT_FEATURES,
        help="exp10 W4.1: 'flat' (default, byte-identical sb3 FlattenExtractor -- no "
        "behavior change from every pre-W4.1 run) or 'slot_v1'/'slot_attn' (custom "
        "features.py::SlotEmbeddingExtractor with learned pet/food/equipment "
        "embeddings + shared per-slot encoders over the same v4 1993-dim layout; "
        "'slot_attn' additionally attends across the 14 slot vectors).",
    )
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=DEFAULT_EMBED_DIM,
        help="exp10 W4.1: pet embedding width (d_pet). Only used when --features != flat.",
    )
    ap.add_argument(
        "--item-dim",
        type=int,
        default=DEFAULT_ITEM_DIM,
        help="exp10 W4.1: equipment/food embedding width (d_item). Only used when "
        "--features != flat.",
    )
    ap.add_argument(
        "--slot-dim",
        type=int,
        default=DEFAULT_SLOT_DIM,
        help="exp10 W4.1: per-slot encoder output width (d_slot). Only used when "
        "--features != flat.",
    )
    ap.add_argument(
        "--opp-dim",
        type=int,
        default=DEFAULT_OPP_DIM,
        help="exp10 W4.1: opponent-context block output width (d_opp). Only used when "
        "--features != flat.",
    )
    ap.add_argument(
        "--attn-heads",
        type=int,
        default=DEFAULT_ATTN_HEADS,
        help="exp10 W4.1: attention head count, 'slot_attn' only (ignored by "
        "'slot_v1'/'flat').",
    )
    args = ap.parse_args(argv)

    metadata = train(args)
    print(json.dumps({"ok": True, "out_dir": str(args.out), "best_val_loss": metadata["best_val_loss"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
