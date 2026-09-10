"""Custom Maskable PPO policies for SAP-Arena training."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Distribution as TorchDistribution
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

# Disable strict simplex argument checks globally for masked categorical usage.
# With large-negative logit masking, tiny FP drift can exceed 1e-6 tolerance and
# raise false-positive errors during train/eval despite valid masks.
TorchDistribution.set_default_validate_args(False)


class TypeBalancedMaskablePolicy(MaskableActorCriticPolicy):
    """
    Adds a legal-action-type balancing prior to actor logits.

    For each state, it computes how many legal actions exist per action type and applies
    a bias of `-log(count_per_type)` to each action in that type. With equal raw logits,
    this yields roughly equal probability mass per action type instead of mass being
    dominated by types that have many parameterized variants (e.g., REORDER).
    """

    def __init__(
        self,
        *args: Any,
        action_type_ids: list[int],
        type_balance_strength: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        action_dim = int(getattr(self.action_space, "n", -1))
        if action_dim <= 0:
            raise ValueError("type_balanced_policy_requires_discrete_action_space")
        if len(action_type_ids) != action_dim:
            raise ValueError(
                f"type_balanced_policy_action_type_ids_mismatch: expected={action_dim} got={len(action_type_ids)}"
            )

        type_ids = th.as_tensor([int(x) for x in action_type_ids], dtype=th.long)
        num_types = int(type_ids.max().item()) + 1 if type_ids.numel() > 0 else 1
        type_one_hot = F.one_hot(type_ids, num_classes=num_types).to(dtype=th.float32)
        global_type_counts = type_one_hot.sum(dim=0).clamp_min(1.0)

        self.type_balance_strength = float(type_balance_strength)
        self.register_buffer("_action_type_ids", type_ids, persistent=True)
        self.register_buffer("_action_type_one_hot", type_one_hot, persistent=True)
        self.register_buffer("_global_type_counts", global_type_counts, persistent=True)

    @staticmethod
    def _tensor_summary(tensor: th.Tensor | None) -> dict[str, Any] | None:
        if not isinstance(tensor, th.Tensor):
            return None
        t = tensor.detach()
        summary: dict[str, Any] = {
            "shape": [int(x) for x in t.shape],
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": int(t.numel()),
        }
        if t.numel() == 0:
            return summary
        summary["nan_count"] = int(th.isnan(t).sum().item())
        summary["inf_count"] = int(th.isinf(t).sum().item())
        finite = th.isfinite(t)
        finite_count = int(finite.sum().item())
        summary["finite_count"] = finite_count
        if finite_count > 0:
            finite_t = t[finite]
            summary["finite_min"] = float(finite_t.min().item())
            summary["finite_max"] = float(finite_t.max().item())
            summary["finite_mean"] = float(finite_t.mean().item())
        return summary

    @staticmethod
    def _mask_summary(
        action_masks: np.ndarray | th.Tensor | None,
        *,
        expected_batch: int | None,
        expected_action_dim: int | None,
        device: th.device | None,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {"present": action_masks is not None}
        if action_masks is None:
            return summary
        mask_t = th.as_tensor(action_masks, device=device)
        summary["input_shape"] = [int(x) for x in mask_t.shape]
        if mask_t.ndim == 1:
            mask_2d = mask_t.unsqueeze(0)
        elif mask_t.ndim == 2:
            mask_2d = mask_t
        else:
            mask_2d = mask_t.reshape(mask_t.shape[0], -1)
            summary["reshaped_from_ndim"] = int(mask_t.ndim)
        mask_2d = mask_2d.to(dtype=th.bool)
        summary["normalized_shape"] = [int(x) for x in mask_2d.shape]
        if expected_batch is not None:
            summary["expected_batch"] = int(expected_batch)
            summary["batch_matches"] = int(mask_2d.shape[0]) == int(expected_batch)
        if expected_action_dim is not None:
            summary["expected_action_dim"] = int(expected_action_dim)
            summary["action_dim_matches"] = int(mask_2d.shape[1]) == int(expected_action_dim)
        legal_counts = mask_2d.sum(dim=-1)
        summary["legal_min"] = int(legal_counts.min().item())
        summary["legal_max"] = int(legal_counts.max().item())
        summary["zero_legal_rows"] = int((legal_counts == 0).sum().item())
        return summary

    def _collect_mask_debug_context(
        self,
        *,
        distribution: MaskableDistribution,
        action_masks: np.ndarray | th.Tensor | None,
        context: str,
    ) -> dict[str, Any]:
        dist = getattr(distribution, "distribution", None)
        logits = getattr(dist, "logits", None)
        probs = getattr(dist, "probs", None)
        original_logits = getattr(dist, "_original_logits", None)
        expected_batch = int(logits.shape[0]) if isinstance(logits, th.Tensor) and logits.ndim >= 2 else None
        expected_action_dim = (
            int(logits.shape[-1]) if isinstance(logits, th.Tensor) and logits.ndim >= 1 else int(getattr(self.action_space, "n", -1))
        )
        device = logits.device if isinstance(logits, th.Tensor) else None

        debug: dict[str, Any] = {
            "context": context,
            "mask": self._mask_summary(
                action_masks,
                expected_batch=expected_batch,
                expected_action_dim=expected_action_dim,
                device=device,
            ),
            "logits": self._tensor_summary(logits),
            "original_logits": self._tensor_summary(original_logits),
            "probs": self._tensor_summary(probs),
        }

        if isinstance(probs, th.Tensor) and probs.ndim == 2 and probs.numel() > 0:
            row_sums = probs.detach().sum(dim=-1)
            bad = ~th.isfinite(row_sums) | (th.abs(row_sums - 1.0) > 1e-3)
            bad_count = int(bad.sum().item())
            debug["probs_row_sums"] = {
                "min": float(row_sums.min().item()),
                "max": float(row_sums.max().item()),
                "bad_count": bad_count,
            }
            if bad_count > 0:
                first_bad = int(th.nonzero(bad, as_tuple=False).flatten()[0].item())
                row = probs.detach()[first_bad]
                debug["first_bad_row"] = {
                    "index": first_bad,
                    "sum": float(row.sum().item()),
                    "min": float(row.min().item()),
                    "max": float(row.max().item()),
                    "nan_count": int(th.isnan(row).sum().item()),
                    "inf_count": int(th.isinf(row).sum().item()),
                }
        return debug

    def _apply_masking_with_debug(
        self,
        distribution: MaskableDistribution,
        action_masks: np.ndarray | th.Tensor | None,
        *,
        context: str,
    ) -> None:
        if action_masks is None:
            return
        try:
            distribution.apply_masking(action_masks)
        except Exception as exc:
            debug = self._collect_mask_debug_context(
                distribution=distribution,
                action_masks=action_masks,
                context=context,
            )
            raise RuntimeError(
                f"mask_apply_failed:{context}:{type(exc).__name__}:{exc}|debug={json.dumps(debug, sort_keys=True)}"
            ) from exc

    def _balanced_logits(
        self,
        latent_pi: th.Tensor,
        *,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> th.Tensor:
        action_logits = self.action_net(latent_pi)
        strength = float(self.type_balance_strength)
        if strength <= 0.0:
            return action_logits

        batch_size = int(action_logits.shape[0])
        device = action_logits.device
        type_ids = self._action_type_ids.to(device=device)
        type_one_hot = self._action_type_one_hot.to(device=device)

        if action_masks is None:
            counts_per_type = self._global_type_counts.to(device=device).unsqueeze(0).expand(batch_size, -1)
        else:
            mask_t = th.as_tensor(action_masks, device=device)
            if mask_t.ndim == 1:
                mask_t = mask_t.unsqueeze(0)
            mask_t = mask_t.to(dtype=th.float32)
            if int(mask_t.shape[0]) != batch_size:
                if int(mask_t.shape[0]) == 1:
                    mask_t = mask_t.expand(batch_size, -1)
                else:
                    mask_t = mask_t[:batch_size]
            counts_per_type = mask_t @ type_one_hot
            counts_per_type = counts_per_type.clamp_min(1.0)

        per_type_bias = -th.log(counts_per_type)
        index = type_ids.unsqueeze(0).expand(int(per_type_bias.shape[0]), -1)
        per_action_bias = th.gather(per_type_bias, 1, index)
        if int(per_action_bias.shape[0]) != batch_size:
            if int(per_action_bias.shape[0]) == 1:
                per_action_bias = per_action_bias.expand(batch_size, -1)
            else:
                per_action_bias = per_action_bias[:batch_size]

        return action_logits + (strength * per_action_bias)

    def _get_action_dist_from_latent(
        self,
        latent_pi: th.Tensor,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> MaskableDistribution:
        action_logits = self._balanced_logits(latent_pi, action_masks=action_masks)
        if not bool(th.isfinite(action_logits).all().item()):
            debug = {
                "context": "_get_action_dist_from_latent",
                "action_logits": self._tensor_summary(action_logits),
                "mask": self._mask_summary(
                    action_masks,
                    expected_batch=int(action_logits.shape[0]),
                    expected_action_dim=int(action_logits.shape[-1]),
                    device=action_logits.device,
                ),
            }
            raise RuntimeError(f"action_logits_invalid:{json.dumps(debug, sort_keys=True)}")
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_masks)
        self._apply_masking_with_debug(distribution, action_masks, context="forward")
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_distribution(
        self,
        obs: th.Tensor,
        action_masks: np.ndarray | None = None,
    ) -> MaskableDistribution:
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_masks)
        self._apply_masking_with_debug(distribution, action_masks, context="get_distribution")
        return distribution

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi, action_masks=action_masks)
        self._apply_masking_with_debug(distribution, action_masks, context="evaluate_actions")
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()


class HierarchicalMaskablePolicy(TypeBalancedMaskablePolicy):
    """
    Factorized action policy compatible with flat discrete MaskablePPO.

    Stage 1 predicts action-type logits. Stage 2 predicts action logits within each type.
    We collapse the factorization into flat logits over the original action catalog:

        log P(action) = log P(type) + log P(action | type)

    This preserves sb3-contrib's flat masked categorical interface while reducing
    probability-mass distortion from action-count imbalance.
    """

    def __init__(
        self,
        *args: Any,
        action_type_ids: list[int],
        type_balance_strength: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            action_type_ids=action_type_ids,
            type_balance_strength=type_balance_strength,
            **kwargs,
        )
        num_types = int(self._action_type_one_hot.shape[1]) if self._action_type_one_hot.ndim == 2 else 1
        latent_dim_pi = int(getattr(self.mlp_extractor, "latent_dim_pi"))
        self.type_net = th.nn.Linear(latent_dim_pi, num_types)
        # Parent __init__ builds the optimizer before this head exists; rebuild so type_net trains.
        try:
            if bool(getattr(self, "ortho_init", False)):
                self.init_weights(self.type_net, gain=0.01)
            lr_schedule = getattr(self, "lr_schedule", None)
            optimizer_class = getattr(self, "optimizer_class", None)
            optimizer_kwargs = dict(getattr(self, "optimizer_kwargs", {}) or {})
            if callable(lr_schedule) and optimizer_class is not None:
                self.optimizer = optimizer_class(self.parameters(), lr=lr_schedule(1), **optimizer_kwargs)
        except Exception as exc:
            raise RuntimeError(f"hierarchical_policy_optimizer_rebuild_failed:{type(exc).__name__}:{exc}") from exc

    def _normalize_action_mask(
        self,
        *,
        action_masks: np.ndarray | th.Tensor | None,
        batch_size: int,
        action_dim: int,
        device: th.device,
    ) -> th.Tensor:
        if action_masks is None:
            return th.ones((batch_size, action_dim), dtype=th.bool, device=device)
        mask_t = th.as_tensor(action_masks, device=device)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)
        elif mask_t.ndim > 2:
            mask_t = mask_t.reshape(mask_t.shape[0], -1)
        mask_t = mask_t.to(dtype=th.bool)
        if int(mask_t.shape[1]) != action_dim:
            if int(mask_t.shape[1]) > action_dim:
                mask_t = mask_t[:, :action_dim]
            else:
                pad = th.zeros((int(mask_t.shape[0]), action_dim - int(mask_t.shape[1])), dtype=th.bool, device=device)
                mask_t = th.cat([mask_t, pad], dim=1)
        if int(mask_t.shape[0]) != batch_size:
            if int(mask_t.shape[0]) == 1:
                mask_t = mask_t.expand(batch_size, -1)
            else:
                mask_t = mask_t[:batch_size]
        return mask_t

    def _hierarchical_logits(
        self,
        latent_pi: th.Tensor,
        *,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> th.Tensor:
        action_logits = self.action_net(latent_pi)
        type_logits = self.type_net(latent_pi)
        if not bool(th.isfinite(action_logits).all().item()) or not bool(th.isfinite(type_logits).all().item()):
            debug = {
                "context": "_hierarchical_logits_input_invalid",
                "action_logits": self._tensor_summary(action_logits),
                "type_logits": self._tensor_summary(type_logits),
            }
            raise RuntimeError(f"hierarchical_logits_invalid:{json.dumps(debug, sort_keys=True)}")

        batch_size, action_dim = int(action_logits.shape[0]), int(action_logits.shape[1])
        device = action_logits.device
        mask_bool = self._normalize_action_mask(
            action_masks=action_masks,
            batch_size=batch_size,
            action_dim=action_dim,
            device=device,
        )
        if not bool(mask_bool.any(dim=1).all().item()):
            # Keep runtime explicit; maskable categorical cannot handle all-illegal rows.
            debug = {
                "context": "_hierarchical_logits_no_legal_actions",
                "mask": self._mask_summary(
                    mask_bool,
                    expected_batch=batch_size,
                    expected_action_dim=action_dim,
                    device=device,
                ),
            }
            raise RuntimeError(f"hierarchical_policy_no_legal_actions:{json.dumps(debug, sort_keys=True)}")

        type_ids = self._action_type_ids.to(device=device)
        type_one_hot = self._action_type_one_hot.to(device=device)
        num_types = int(type_one_hot.shape[1])

        # Legal action counts per type define the stage-1 type mask.
        mask_f = mask_bool.to(dtype=th.float32)
        legal_counts_per_type = (mask_f @ type_one_hot).clamp_min(0.0)
        legal_type_mask = legal_counts_per_type > 0.0

        neg_large = th.finfo(action_logits.dtype).min / 4.0
        masked_action_logits = action_logits.masked_fill(~mask_bool, neg_large)

        # Stage 2: normalize only among legal actions within the selected type.
        type_log_norms: list[th.Tensor] = []
        for t in range(num_types):
            type_mask = (type_ids == int(t))
            if not bool(type_mask.any().item()):
                type_log_norms.append(th.zeros((batch_size,), dtype=action_logits.dtype, device=device))
                continue
            logits_t = masked_action_logits[:, type_mask]
            lse_t = th.logsumexp(logits_t, dim=1)
            type_log_norms.append(lse_t)
        type_log_norm = th.stack(type_log_norms, dim=1)
        # Clamp pathological "no legal in type" rows before subtraction; those rows are masked at stage 1 anyway.
        type_log_norm = th.where(legal_type_mask, type_log_norm, th.zeros_like(type_log_norm))

        stage2_cond_logits = action_logits - th.gather(
            type_log_norm,
            1,
            type_ids.unsqueeze(0).expand(batch_size, -1),
        )

        # Stage 1: normalize among legal action types only.
        masked_type_logits = type_logits.masked_fill(~legal_type_mask, neg_large)
        stage1_log_probs = masked_type_logits - th.logsumexp(masked_type_logits, dim=1, keepdim=True)
        per_action_stage1_logp = th.gather(
            stage1_log_probs,
            1,
            type_ids.unsqueeze(0).expand(batch_size, -1),
        )

        flat_logits = per_action_stage1_logp + stage2_cond_logits
        flat_logits = flat_logits.masked_fill(~mask_bool, neg_large)

        if not bool(th.isfinite(flat_logits).all().item()):
            debug = {
                "context": "_hierarchical_logits_output_invalid",
                "flat_logits": self._tensor_summary(flat_logits),
                "stage1_log_probs": self._tensor_summary(stage1_log_probs),
                "stage2_cond_logits": self._tensor_summary(stage2_cond_logits),
                "legal_type_mask": self._tensor_summary(legal_type_mask.to(dtype=th.float32)),
                "mask": self._mask_summary(
                    mask_bool,
                    expected_batch=batch_size,
                    expected_action_dim=action_dim,
                    device=device,
                ),
            }
            raise RuntimeError(f"hierarchical_logits_output_invalid:{json.dumps(debug, sort_keys=True)}")
        return flat_logits

    def _balanced_logits(
        self,
        latent_pi: th.Tensor,
        *,
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> th.Tensor:
        # Override the type-balanced flat prior with an explicit factorized type->action policy.
        return self._hierarchical_logits(latent_pi, action_masks=action_masks)
