"""KL-anchored Maskable PPO for exp09 W5 (versus-mode chain PPO).

Implements the piKL / Generals.io-2025 recipe PLAN.md's "W5 reward" section
froze: standard MaskablePPO, PLUS a `kl_coef * KL(pi_current || pi_BC)` term
added to the training loss, where `pi_BC` is the FROZEN behavior-cloning
policy (`flat_v2`) the run was warm-started from, and `kl_coef` DECAYS over
training (linear from `--kl-coef-init` to `--kl-coef-final` on SB3's
`_current_progress_remaining`, i.e. init at step 0 -> final at the last
step). The KL is computed on the SAME masked action distributions the policy
trains on (`rollout_data.action_masks` from the maskable rollout buffer), so
anchor and policy always compare over the identical legal-action support.

Why loss-level KL and not a reward bonus: PLAN.md Rev 5 specifies the
Generals.io / piKL formulation ("add KL(pi||pi_BC) penalty to the LOSS with a
decaying coefficient"), which regularizes the optimization directly and
leaves the REWARD purely the versus-lives outcome (`gym_env.py`'s
`versus_lives` mode) -- reward stays interpretable, the leash lives in the
loss.

`train()` below is copied VERBATIM from the installed sb3-contrib 2.9.0
`MaskablePPO.train` (checked against `inspect.getsource` on this box) with
ONE clearly-marked insertion (the KL-to-anchor block) and its logging.
If sb3-contrib is ever upgraded past 2.9.x, re-diff this copy against the
new upstream `train()` (see `docs/dependency-pins.md` for the pin policy).

The anchor is deliberately EXCLUDED from checkpoint pickling
(`_excluded_save_params`): a saved W5 model zip stays loadable by plain
`MaskablePPO.load` (BcRecommender, eval_ppo, model-pool tools), and a RESUME
with the KL term re-attaches the anchor from `--kl-anchor` at launch time
instead of trusting a pickled copy.

exp09 W0b (decay probe matrix) added two things on top of the above,
investigating a reproducible pathology: KL-anchored PPO warm-started from a
BC checkpoint DECAYS below its BC init as the KL anchor releases.

1. `freeze_actor_params`: freezes the actor side of the policy (called by
   `train_ppo.py` right after BC init, gated on `--freeze-actor 1`) so ONLY
   the critic trains -- a sanity arm (probe B4) that must reproduce the BC
   floor exactly, since the action distribution never moves.
2. `train()` telemetry: pre-normalization advantage stats + a per-epoch
   gradient-norm decomposition, both marked `exp09 W0b` below, logged via the
   existing `self.logger` every update. These are OBSERVATIONAL ONLY -- see
   the telemetry block's own comment for why the decomposition cannot change
   the actual parameter update.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.utils import explained_variance

from sb3_contrib import MaskablePPO


def linear_kl_coef(
    *, coef_init: float, coef_final: float, progress_remaining: float, hold_frac: float = 0.0
) -> float:
    """Hold-then-linear-decay schedule for the KL coefficient.

    `coef_init` is HELD for the first `hold_frac` of training (exp09 W5
    Rev 6: the BC checkpoint's value head starts random, so the leash stays
    tight through the critic's warmup), then decays linearly to `coef_final`
    over the remaining `1 - hold_frac`. `hold_frac=0.0` reproduces the
    original run-1 schedule exactly (pure linear from step 0).
    """
    p = float(min(1.0, max(0.0, progress_remaining)))
    h = float(min(0.99, max(0.0, hold_frac)))
    done = 1.0 - p
    if done <= h:
        return float(coef_init)
    t = (done - h) / (1.0 - h)
    return float(coef_init) + (float(coef_final) - float(coef_init)) * t


def _policy_weight_checksum(policy: Any) -> float:
    """Cheap deterministic scalar over all policy params (init-verification aid)."""
    with th.no_grad():
        return float(sum(p.detach().double().abs().sum().item() for p in policy.parameters()))


def assert_policy_frames_compatible(model: Any, other_model: Any, *, context: str) -> None:
    """Raise unless `other_model` shares `model`'s observation shape, action
    space, and policy class. Used by BOTH the BC warm-start load and the
    standalone `--kl-anchor` attach (exp09 W5 P1, cross-model review finding:
    a same-class anchor with a different encoder previously failed only at
    the first training minibatch, AFTER a full rollout was collected, or
    silently regularized toward wrong semantics when shapes coincided)."""
    model_obs = tuple(int(x) for x in model.observation_space.shape)
    other_obs = tuple(int(x) for x in other_model.observation_space.shape)
    if model_obs != other_obs:
        raise ValueError(f"{context}_observation_shape_mismatch:model={model_obs}:other={other_obs}")
    model_n = int(getattr(model.action_space, "n", -1))
    other_n = int(getattr(other_model.action_space, "n", -1))
    if model_n != other_n:
        raise ValueError(f"{context}_action_space_mismatch:model={model_n}:other={other_n}")
    model_cls = type(model.policy).__name__
    other_cls = type(other_model.policy).__name__
    if model_cls != other_cls:
        raise ValueError(
            f"{context}_policy_class_mismatch:model={model_cls}:other={other_cls} "
            "(--policy-mode must match the checkpoint's policy_mode)"
        )


def load_bc_policy_weights(model: Any, bc_checkpoint_path: str | Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Warm-start `model.policy` from a BC checkpoint saved by
    `train_chain_bc.py` (a full MaskablePPO zip of the SAME policy class /
    architecture -- that trainer exists specifically to make this load work,
    see its module docstring).

    Uses `load_state_dict(strict=True)` so ANY architecture/class drift fails
    loudly instead of silently part-loading. Deliberately does NOT call
    `observation.assert_model_observation_compatible` here: BC metadata.json
    carries no `max_turn` key (BC training has no episode-cap concept), so
    that assert would reject every real BC checkpoint including `flat_v2` --
    see `tools/bc_recommender.py.__init__`'s note on the same decision. The
    caller (`train_ppo.py`) separately cross-checks the BC metadata's
    observation mode/size/vocab-fingerprint against its own encoder spec.

    Returns `(bc_model, report)`; the caller can reuse `bc_model.policy` as
    the frozen KL anchor so the checkpoint is only loaded once.
    """
    path = Path(bc_checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"bc_init_checkpoint_not_found:{path}")
    bc_model = MaskablePPO.load(str(path), device=str(device))

    assert_policy_frames_compatible(model, bc_model, context="bc_init")
    model_cls = type(model.policy).__name__

    checksum_before = _policy_weight_checksum(model.policy)
    bc_state = bc_model.policy.state_dict()
    model.policy.load_state_dict(bc_state, strict=True)
    checksum_after = _policy_weight_checksum(model.policy)
    checksum_bc = _policy_weight_checksum(bc_model.policy)
    if abs(checksum_after - checksum_bc) > 1e-6:
        raise RuntimeError(
            f"bc_init_checksum_mismatch_after_load:after={checksum_after!r}:bc={checksum_bc!r}"
        )

    report = {
        "bc_checkpoint": str(path),
        "policy_class": model_cls,
        "n_params": int(sum(p.numel() for p in model.policy.parameters())),
        "policy_weight_checksum_before": checksum_before,
        "policy_weight_checksum_after": checksum_after,
        "policy_weight_checksum_bc": checksum_bc,
    }
    return bc_model, report


def freeze_actor_params(policy: Any) -> dict[str, Any]:
    """Freeze the actor (policy) side of an ActorCriticPolicy-family module,
    leaving only the critic (value) side trainable (exp09 W0b probe B4,
    `--freeze-actor 1`).

    With the actor frozen, the policy's action distribution for any fixed
    input is bit-identical throughout training -- this is a sanity arm for
    the W0b decay-probe matrix: if PPO's post-BC decay were an
    implementation bug elsewhere (e.g. in the reward, the KL term, or the
    rollout plumbing) rather than genuinely entropy/advantage-noise driven,
    a frozen actor might STILL drift, which would show up as an eval score
    outside the BC floor's CI and mean something upstream is broken.

    Freezes, if present on `policy`:
    - `mlp_extractor.policy_net` (the actor's hidden MLP torso)
    - `action_net` (the flat action-logit head; TypeBalancedMaskablePolicy's
      own `_balanced_logits`/`_hierarchical_logits` only ADD a deterministic,
      parameter-free bias on top of this, so freezing it is sufficient there)
    - `type_net` (HierarchicalMaskablePolicy's stage-1 type head, absent on
      TypeBalancedMaskablePolicy)
    - `log_std` (continuous-action policies only; absent here, this env is
      Discrete, kept for generality since a probe script should not silently
      mis-freeze if the policy class ever changes)
    - the actor-side features extractor: `pi_features_extractor` when actor
      and critic use SEPARATE extractors, or the single shared
      `features_extractor` when `share_features_extractor` is True. This
      matters because a SHARED extractor that keeps training via the
      critic's gradient would silently change the actor's input features on
      every forward pass, breaking the "frozen actor" invariant even with
      policy_net/action_net themselves frozen bit-for-bit. In this repo the
      default extractor is a parameter-free `FlattenExtractor` (flat vector
      observations, no CNN/embedding), so this freeze is a no-op in practice
      -- kept for correctness if that ever changes.

    Returns a report dict (component names actually frozen + total param
    count) for metadata/telemetry; safe to call multiple times (idempotent).
    """
    frozen_components: list[str] = []
    frozen_param_count = 0

    def _freeze_module(name: str, module: Any) -> None:
        nonlocal frozen_param_count
        if module is None:
            return
        n = 0
        for p in module.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                n += 1
        if n > 0:
            frozen_components.append(name)
            frozen_param_count += n

    mlp_extractor = getattr(policy, "mlp_extractor", None)
    _freeze_module("mlp_extractor.policy_net", getattr(mlp_extractor, "policy_net", None))
    _freeze_module("action_net", getattr(policy, "action_net", None))
    _freeze_module("type_net", getattr(policy, "type_net", None))

    log_std = getattr(policy, "log_std", None)
    if isinstance(log_std, th.nn.Parameter) and log_std.requires_grad:
        log_std.requires_grad_(False)
        frozen_components.append("log_std")
        frozen_param_count += 1

    if bool(getattr(policy, "share_features_extractor", False)):
        _freeze_module("features_extractor(shared)", getattr(policy, "features_extractor", None))
    else:
        _freeze_module("pi_features_extractor", getattr(policy, "pi_features_extractor", None))

    return {
        "frozen_components": frozen_components,
        "frozen_param_count": int(frozen_param_count),
    }


class KLAnchoredMaskablePPO(MaskablePPO):
    """MaskablePPO + decaying KL(pi || pi_anchor) loss term (exp09 W5).

    With no anchor attached (or a zero coefficient) this trains EXACTLY like
    MaskablePPO -- the KL block is skipped entirely -- so constructing this
    class unconditionally on the W5 path is safe.
    """

    def __init__(
        self,
        *args: Any,
        kl_coef_init: float = 0.0,
        kl_coef_final: float = 0.0,
        kl_coef_hold_frac: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kl_coef_init = float(kl_coef_init)
        self.kl_coef_final = float(kl_coef_final)
        self.kl_coef_hold_frac = float(kl_coef_hold_frac)
        self._kl_anchor_policy: Any | None = None

    def set_kl_anchor(self, anchor_policy: Any) -> None:
        """Attach the FROZEN anchor policy (same class/arch as `self.policy`).

        Moves it to this model's device, switches it to eval mode, and turns
        every grad off -- the anchor never trains, it only answers
        `get_distribution` queries inside `train()`.
        """
        anchor_cls = type(anchor_policy).__name__
        own_cls = type(self.policy).__name__
        if anchor_cls != own_cls:
            raise ValueError(f"kl_anchor_policy_class_mismatch:model={own_cls}:anchor={anchor_cls}")
        anchor_policy = anchor_policy.to(self.device)
        anchor_policy.set_training_mode(False)
        for p in anchor_policy.parameters():
            p.requires_grad_(False)
        self._kl_anchor_policy = anchor_policy

    def _current_kl_coef(self) -> float:
        return linear_kl_coef(
            coef_init=self.kl_coef_init,
            coef_final=self.kl_coef_final,
            progress_remaining=float(self._current_progress_remaining),
            hold_frac=float(getattr(self, "kl_coef_hold_frac", 0.0)),
        )

    def _excluded_save_params(self) -> list[str]:
        # The anchor is re-attached from --kl-anchor at launch; never pickle it
        # into the checkpoint zip (keeps W5 checkpoints loadable by plain
        # MaskablePPO.load and the zip small).
        return [*super()._excluded_save_params(), "_kl_anchor_policy"]

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.

        Copied from sb3-contrib 2.9.0 MaskablePPO.train with TWO marked
        insertions: the original `KL-to-anchor (exp09 W5)` block, and the
        `W0b telemetry (exp09 W0b)` additions described in the module
        docstring. The telemetry is purely observational: the gradient
        decomposition reconstructs the EXACT same total gradient a plain
        `loss.backward()` would (by linearity of differentiation), so the
        parameter update a minibatch receives is bit-for-bit unchanged
        whether or not the decomposition runs on it.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_anchor_values = []  # exp09 W5: logged per-minibatch KL(pi||pi_BC)
        kl_coef = self._current_kl_coef()  # exp09 W5: one coef per train() call

        # --- W0b telemetry (exp09 W0b): accumulators -----------------------
        # Pre-normalization advantage stats (cheap: no extra forward/backward,
        # just stats on a tensor already being computed) collected on EVERY
        # minibatch. The gradient-norm decomposition is collected ONCE per
        # epoch (last minibatch only) to keep overhead small -- see below.
        adv_pre_norm_means: list[float] = []
        adv_pre_norm_stds: list[float] = []
        adv_pre_norm_mean_abs: list[float] = []
        grad_norm_policy_by_epoch: list[float] = []
        grad_norm_entropy_by_epoch: list[float] = []
        grad_norm_kl_by_epoch: list[float] = []
        grad_norm_value_by_epoch: list[float] = []
        # `rollout_buffer.get(batch_size)` yields ceil(buffer_size*n_envs /
        # batch_size) minibatches per epoch (sb3_contrib's MaskableRolloutBuffer
        # .get: constant given the buffer shape) -- computed ONCE so detecting
        # "last minibatch of the epoch" below never has to consume (and
        # reshuffle) the generator an extra time.
        total_transitions = int(self.rollout_buffer.buffer_size) * int(self.rollout_buffer.n_envs)
        effective_batch_size = int(self.batch_size) if self.batch_size is not None else total_transitions
        n_minibatches_per_epoch = max(1, math.ceil(total_transitions / max(1, effective_batch_size)))
        # ---------------------------------------------------------------------

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for mb_idx, rollout_data in enumerate(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # exp09 W0b: pre-normalization stats BEFORE the optional
                # normalize_advantage rescaling below. The decay-probe
                # hypothesis is about the RAW advantage signal being ~0 for
                # ~97% all-loss episodes; normalizing first would hide that
                # by construction (every minibatch gets unit variance).
                with th.no_grad():
                    adv_pre_norm_means.append(float(advantages.mean().item()))
                    adv_pre_norm_stds.append(float(advantages.std().item()))
                    adv_pre_norm_mean_abs.append(float(advantages.abs().mean().item()))
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- KL-to-anchor (exp09 W5): the ONE insertion -------------
                # KL(pi_current || pi_BC) over the SAME masked support, added
                # to the loss with the decaying coefficient. Grad flows only
                # through the CURRENT policy's distribution (the anchor is
                # frozen + no_grad).
                kl_anchor = None
                if self._kl_anchor_policy is not None and kl_coef > 0.0:
                    current_dist = self.policy.get_distribution(
                        rollout_data.observations, action_masks=rollout_data.action_masks
                    )
                    with th.no_grad():
                        anchor_dist = self._kl_anchor_policy.get_distribution(
                            rollout_data.observations, action_masks=rollout_data.action_masks
                        )
                    kl_anchor = th.distributions.kl_divergence(
                        current_dist.distribution, anchor_dist.distribution
                    ).mean()
                    loss = loss + kl_coef * kl_anchor
                    kl_anchor_values.append(float(kl_anchor.item()))
                # -------------------------------------------------------------

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                is_last_minibatch = mb_idx == (n_minibatches_per_epoch - 1)
                if is_last_minibatch:
                    # --- W0b gradient-norm decomposition (exp09 W0b) --------
                    # Cheapest correct approach: reuse THIS minibatch's
                    # already-built graph (no extra forward pass), pull each
                    # loss component's gradient separately via
                    # autograd.grad(retain_graph=...) over the SAME trainable
                    # params, then manually sum them into `.grad` before the
                    # SAME clip+step the undecomposed branch takes below. By
                    # linearity, sum(grad(component_i)) == grad(sum(component_i))
                    # == grad(loss) exactly, so this branch's parameter update
                    # is identical to plain loss.backward() -- only observed,
                    # never perturbed. Runs once per epoch (last minibatch
                    # only), not every minibatch, to keep overhead small.
                    grad_params = [p for p in self.policy.parameters() if p.requires_grad]
                    components: list[tuple[str, th.Tensor]] = [
                        ("policy", policy_loss),
                        ("entropy", self.ent_coef * entropy_loss),
                        ("value", self.vf_coef * value_loss),
                    ]
                    if kl_anchor is not None:
                        components.append(("kl", kl_coef * kl_anchor))

                    # A component whose ENTIRE graph lives in frozen params is
                    # a constant tensor with no grad_fn (`--freeze-actor 1`
                    # makes policy/entropy/KL exactly that), and autograd.grad
                    # on it raises "element 0 of tensors does not require
                    # grad" -- this crashed probe B4's first train() call
                    # (2026-07-19, an internal dataset path
                    # train.log.failed1). allow_unused only forgives UNUSED
                    # PARAMS, not a graphless OUTPUT. A graphless component
                    # contributes exactly zero gradient by definition, so:
                    # differentiate only components that require grad, and log
                    # 0.0 for the rest. (grad_params empty would be the
                    # everything-frozen edge; no component can require grad
                    # then, so diff_components is [] and no autograd call
                    # happens at all.)
                    diff_components = [
                        (comp_name, comp_value)
                        for comp_name, comp_value in components
                        if isinstance(comp_value, th.Tensor) and comp_value.requires_grad
                    ]
                    if not grad_params:
                        diff_components = []
                    summed_grads: list[th.Tensor | None] = [None] * len(grad_params)
                    component_norm_sq: dict[str, float] = {name: 0.0 for name, _ in components}
                    for comp_idx, (comp_name, comp_value) in enumerate(diff_components):
                        is_last_component = comp_idx == (len(diff_components) - 1)
                        comp_grads = th.autograd.grad(
                            comp_value,
                            grad_params,
                            retain_graph=not is_last_component,
                            allow_unused=True,
                        )
                        sq_sum = 0.0
                        for i, g in enumerate(comp_grads):
                            if g is None:
                                continue
                            sq_sum += float(g.detach().double().pow(2).sum().item())
                            if summed_grads[i] is None:
                                summed_grads[i] = g.detach().clone()
                            else:
                                summed_grads[i] = summed_grads[i] + g.detach()
                        component_norm_sq[comp_name] = sq_sum

                    for name, bucket in (
                        ("policy", grad_norm_policy_by_epoch),
                        ("entropy", grad_norm_entropy_by_epoch),
                        ("value", grad_norm_value_by_epoch),
                        ("kl", grad_norm_kl_by_epoch),
                    ):
                        bucket.append(float(component_norm_sq.get(name, 0.0)) ** 0.5)

                    for p, g in zip(grad_params, summed_grads):
                        p.grad = g
                    # ---------------------------------------------------------
                else:
                    loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        # exp09 W5: KL-to-anchor telemetry (0-length when anchor off)
        self.logger.record("train/kl_anchor_coef", float(kl_coef))
        if kl_anchor_values:
            self.logger.record("train/kl_anchor", float(np.mean(kl_anchor_values)))
        # --- W0b telemetry (exp09 W0b): decay-probe diagnostics ------------
        # See collect_w0b.py for the hypothesis-verdict readout of these
        # fields (entropy-driven / noise-normalization / anchor-was-the-only-
        # thing / implementation-bug).
        self.logger.record("train/ent_coef", float(self.ent_coef))
        if adv_pre_norm_means:
            self.logger.record("train/adv_pre_norm_mean", float(np.mean(adv_pre_norm_means)))
            self.logger.record("train/adv_pre_norm_std", float(np.mean(adv_pre_norm_stds)))
            self.logger.record("train/adv_pre_norm_mean_abs", float(np.mean(adv_pre_norm_mean_abs)))
            near_zero_frac = float(np.mean([abs(m) < 1e-3 for m in adv_pre_norm_means]))
            self.logger.record("train/adv_near_zero_frac", near_zero_frac)
        if grad_norm_policy_by_epoch:
            self.logger.record("train/grad_norm_policy", float(np.mean(grad_norm_policy_by_epoch)))
            self.logger.record("train/grad_norm_entropy", float(np.mean(grad_norm_entropy_by_epoch)))
            self.logger.record("train/grad_norm_value", float(np.mean(grad_norm_value_by_epoch)))
            self.logger.record("train/grad_norm_kl", float(np.mean(grad_norm_kl_by_epoch)))
        # ---------------------------------------------------------------------
