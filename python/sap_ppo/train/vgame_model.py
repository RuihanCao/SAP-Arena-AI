"""V_game model (exp12 W1c): frozen a2_attn features + P(win)/battle/margin heads.

The value function that replaces the search's Monte-Carlo leaf (PLAN.md
W1). Representation transfer is the whole bet: the a2_attn BC checkpoint's
`SlotEmbeddingExtractor` (exp10 W4.1, `train/features.py`) already learned
pet-identity embeddings + slot attention from 217k human decisions, so
V_game FREEZES it and trains only a small trunk + three heads on the 22.5k
afterstate rows of `build_afterstate_value_dataset.py`:

- `p_win`: P(player wins the whole game | afterstate), the search-leaf
  scalar. Main head, BCE on the W1a game label.
- `battle_wdl`: this turn's battle W/D/L (aux 0.3): a denser, nearer
  signal over the same features; rows whose turn has no recorded battle
  (battle_wdl = -1) are masked out of this loss, never out of the batch.
- `margin`: final lives margin / 6 (aux 0.1): regression tail that
  separates "barely wins" from "stomps".

Freezing is LITERAL: `load_frozen_extractor` rebuilds the extractor from
code, strict-loads the checkpoint's weights key-for-key (so a drift in
`features.py` fails loudly instead of silently re-randomizing), then sets
`requires_grad=False` + eval mode. `assert_frozen` re-checks both at use
time. Because the extractor never trains, `tools/train_vgame_model.py`
precomputes its 736-dim features once and trains heads on the cache; the
serve path (`VGameModel.forward`) runs the full extractor + heads and is
pinned equal to the cached path by a unit test.

## Two artifact kinds (exp12 W1'c, PLAN amendment 2026-07-29)

W1'c runs TWO training arms, so `save_heads` writes two artifact kinds and
`load_vgame_model` reads both. `mode` in the payload says which (absent ==
`"frozen"`, so every W1 artifact still loads):

- `"frozen"` (arm i, v1 apparatus unchanged): heads only. The extractor
  comes entirely from `--extractor`'s BC checkpoint, and the payload's
  `extractor_sha256` pins THAT file: a mismatch means the heads are being
  served on top of weights they never saw, so the load raises.
- `"unfrozen"` (arm ii, route a): heads PLUS a fine-tuned COPY of the
  extractor's weights (`extractor_state`). The deployed BC checkpoint file
  is never written -- `clone_trainable_extractor` deep-copies it in memory
  and only the copy trains. `extractor_sha256` still pins the ORIGIN
  checkpoint, and the load still rejects a mismatch: the origin file is
  where the architecture, observation space and provenance come from, so a
  different file means a different model even though the weights ride
  along in the artifact.

Both kinds serve identically: `load_vgame_model` returns a `VGameModel`
whose extractor is frozen + eval, so `assert_frozen` holds at serve time
for the fine-tuned copy too.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn as nn

from .observation import OBSERVATION_MODE_V4

# The a2_attn sweep arm's construction params (metadata.json next to the
# checkpoint; re-asserted by the strict state-dict load -- a mismatch here
# changes the key set and fails loudly).
A2_ATTN_FEATURES_KWARGS: dict[str, Any] = {
    "d_pet": 32,
    "d_item": 16,
    "d_slot": 48,
    "d_opp": 32,
    "arch": "slot_attn",
    "attn_heads": 2,
}
A2_ATTN_FEATURES_DIM = 736  # 32 + (5 team + 9 shop) * 48 + 32

DEFAULT_TRUNK_HIDDEN = 256
HEADS_CHECKPOINT_NAME = "vgame_heads.pt"

# Artifact kinds -- see the module docstring's "Two artifact kinds".
ARTIFACT_MODE_FROZEN = "frozen"
ARTIFACT_MODE_UNFROZEN = "unfrozen"
ARTIFACT_MODES = (ARTIFACT_MODE_FROZEN, ARTIFACT_MODE_UNFROZEN)

# Scalar bypass fed straight to the heads (W1c finding: the BC-frozen
# extractor was trained while state lives was a CONSTANT 6 and turn-1
# opponent_lives a constant 0, so its weights on exactly the race-state
# dims are untrained noise and the frozen features cannot carry them --
# measured: naive lives-opp margin alone out-AUCed a bypass-less V_game
# 0.6511 vs 0.6108 on val). The encoder stays frozen and untouched; the
# race scalars just enter AFTER it. Order and normalizers are part of the
# serve contract -- W2's scorer must call `bypass_features` with the TRUE
# engine race values (RESULTS_W1 finding 9).
BYPASS_DIM = 4
BYPASS_TURN_NORM = 15.0
BYPASS_LIVES_NORM = 6.0
BYPASS_WINS_NORM = 15.0


def bypass_features(turn, lives, opponent_lives, wins) -> "th.Tensor":
    """[N, 4] float32 bypass block: [turn/15, lives/6, opp_lives/6, wins/15].
    Accepts numpy arrays or tensors; the single place the order/normalizers
    live (training and serve both call this)."""
    cols = [
        th.as_tensor(np.asarray(turn), dtype=th.float32) / BYPASS_TURN_NORM,
        th.as_tensor(np.asarray(lives), dtype=th.float32) / BYPASS_LIVES_NORM,
        th.as_tensor(np.asarray(opponent_lives), dtype=th.float32) / BYPASS_LIVES_NORM,
        th.as_tensor(np.asarray(wins), dtype=th.float32) / BYPASS_WINS_NORM,
    ]
    return th.stack(cols, dim=-1)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_extractor(
    checkpoint_path: str | Path,
    *,
    observation_mode: str = OBSERVATION_MODE_V4,
    max_turn: int = 15,
    features_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Frozen `SlotEmbeddingExtractor` carrying the checkpoint's weights.

    Path: load the MaskablePPO zip, lift its features-extractor state dict,
    strict-load it into a FRESHLY constructed extractor (same code path
    `train_chain_bc.py::_build_model` uses), freeze, and report key count +
    features_dim for the caller's metadata. Any key/shape drift between the
    checkpoint and current `features.py` raises inside `load_state_dict`.
    """
    from sb3_contrib import MaskablePPO

    from .features import SlotEmbeddingExtractor, build_observation_layout

    model = MaskablePPO.load(str(checkpoint_path), device="cpu")
    source = model.policy.features_extractor
    if type(source).__name__ != "SlotEmbeddingExtractor":
        raise RuntimeError(
            f"vgame_extractor_wrong_class:{type(source).__name__}:"
            f"checkpoint={checkpoint_path}")

    layout = build_observation_layout(observation_mode, max_turn=int(max_turn))
    extractor = SlotEmbeddingExtractor(
        model.observation_space, layout, **(features_kwargs or A2_ATTN_FEATURES_KWARGS))
    source_state = source.state_dict()
    extractor.load_state_dict(source_state, strict=True)

    extractor.requires_grad_(False)
    extractor.eval()

    features_dim = int(extractor.features_dim)
    report = {
        "checkpoint": str(checkpoint_path),
        "extractor_class": "SlotEmbeddingExtractor",
        "state_dict_keys": len(source_state),
        "features_dim": features_dim,
        "observation_size": int(np.prod(model.observation_space.shape)),
        "features_kwargs": dict(features_kwargs or A2_ATTN_FEATURES_KWARGS),
    }
    if features_dim != A2_ATTN_FEATURES_DIM:
        raise RuntimeError(
            f"vgame_features_dim_mismatch:{features_dim}:expected={A2_ATTN_FEATURES_DIM}")
    return extractor, report


def assert_frozen(module: nn.Module) -> None:
    """Loud check that a module is actually frozen (use-time guard)."""
    if module.training:
        raise RuntimeError("vgame_extractor_not_in_eval_mode")
    for name, p in module.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"vgame_extractor_param_requires_grad:{name}")


def clone_trainable_extractor(extractor: nn.Module) -> nn.Module:
    """A deep COPY of `extractor`, re-enabled for training (route a, W1'c
    arm ii).

    The point of the copy is the one thing the amendment made
    non-negotiable: W2 still deploys the a2_attn BC checkpoint, so the
    fine-tune must never reach it. Nothing here writes the checkpoint file
    -- `load_frozen_extractor` opened it read-only, this deep-copies the
    module it built, and only the copy ever sees a gradient. The original
    module stays frozen + eval, and `extractor_param_delta` below is the
    after-the-fact proof that the copy moved and (if the caller passes the
    original twice) that the original did not.
    """
    clone = copy.deepcopy(extractor)
    clone.requires_grad_(True)
    clone.train()
    return clone


def extractor_param_delta(before: nn.Module, after: nn.Module) -> dict[str, Any]:
    """How far `after`'s parameters moved from `before`'s: total L2, max
    absolute element change, and how many tensors changed at all.

    Used as training-time evidence (metadata) and as the unit tests' teeth:
    the unfrozen arm must report `n_changed > 0`, the frozen arm exactly 0.
    """
    a = dict(before.named_parameters())
    b = dict(after.named_parameters())
    if a.keys() != b.keys():
        raise ValueError(f"extractor_param_delta_key_mismatch:"
                         f"{sorted(set(a) ^ set(b))[:4]}")
    sq_total = 0.0
    max_abs = 0.0
    n_changed = 0
    for name, pa in a.items():
        d = (b[name].detach() - pa.detach()).float()
        sq_total += float((d * d).sum())
        if d.numel():
            max_abs = max(max_abs, float(d.abs().max()))
        n_changed += int(bool((d != 0).any()))
    return {"l2": float(sq_total ** 0.5), "max_abs": max_abs,
            "n_changed_tensors": n_changed, "n_tensors": len(a)}


class VGameHeads(nn.Module):
    """Trainable part: shared trunk + p_win / battle_wdl / margin heads.

    `forward` takes the frozen extractor features AND the scalar bypass
    block (see `bypass_features`); `bypass_dim=0` turns the bypass off for
    ablations."""

    def __init__(self, features_dim: int, *, hidden: int = DEFAULT_TRUNK_HIDDEN,
                 bypass_dim: int = BYPASS_DIM, dropout: float = 0.0) -> None:
        super().__init__()
        self.features_dim = int(features_dim)
        self.bypass_dim = int(bypass_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(self.features_dim + self.bypass_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.p_win = nn.Linear(self.hidden, 1)
        self.battle_wdl = nn.Linear(self.hidden, 3)
        self.margin = nn.Linear(self.hidden, 1)
        # Feature standardization (train-tier mu/sd, set once by the trainer
        # via set_feature_norm, identity by default): the frozen extractor's
        # 736 dims are unnormalized ReLU/attention outputs whose scale
        # otherwise swamps the 4-dim bypass block. Buffers, so they ride
        # along in the state dict to the serve path.
        self.register_buffer("feat_mu", th.zeros(self.features_dim))
        self.register_buffer("feat_sd", th.ones(self.features_dim))

    def set_feature_norm(self, mu: th.Tensor, sd: th.Tensor) -> None:
        if tuple(mu.shape) != (self.features_dim,) or tuple(sd.shape) != (self.features_dim,):
            raise ValueError(f"vgame_feature_norm_shape:{tuple(mu.shape)}:{tuple(sd.shape)}")
        self.feat_mu.copy_(mu)
        self.feat_sd.copy_(th.clamp(sd, min=1e-6))

    def forward(self, features: th.Tensor,
                bypass: th.Tensor | None = None) -> dict[str, th.Tensor]:
        f = (features - self.feat_mu) / self.feat_sd
        if self.bypass_dim:
            if bypass is None or bypass.shape[-1] != self.bypass_dim:
                raise ValueError(
                    f"vgame_heads_bypass_shape:{None if bypass is None else tuple(bypass.shape)}:"
                    f"expected_last_dim={self.bypass_dim}")
            z = self.trunk(th.cat([f, bypass], dim=-1))
        else:
            z = self.trunk(f)
        return {
            "p_win_logit": self.p_win(z).squeeze(-1),
            "battle_wdl_logits": self.battle_wdl(z),
            "margin": self.margin(z).squeeze(-1),
        }


class VGameModel(nn.Module):
    """Serve-path module: frozen extractor + heads, obs vector in, P(win) out.

    W2's vgame_scorer feeds states whose lives/wins/opponent_lives are the
    TRUE engine race values at decision time (RESULTS_W1 finding 9), encoded
    by the same v4 encoder as training.
    """

    def __init__(self, extractor: nn.Module, heads: VGameHeads) -> None:
        super().__init__()
        self.extractor = extractor
        self.heads = heads

    @th.no_grad()
    def forward(self, obs: th.Tensor, bypass: th.Tensor | None = None) -> dict[str, th.Tensor]:
        assert_frozen(self.extractor)
        out = self.heads(self.extractor(obs), bypass)
        out["p_win"] = th.sigmoid(out["p_win_logit"])
        return out


def masked_vgame_loss(
    out: dict[str, th.Tensor],
    *,
    y_win: th.Tensor,
    battle_wdl: th.Tensor,
    margin: th.Tensor,
    aux_battle: float = 0.3,
    aux_margin: float = 0.1,
) -> dict[str, th.Tensor]:
    """Total + per-part losses. `battle_wdl == -1` rows are masked out of the
    aux battle CE only (the row still trains p_win/margin); an all-masked
    batch contributes exactly 0 there."""
    p_win_loss = nn.functional.binary_cross_entropy_with_logits(
        out["p_win_logit"], y_win.float())
    mask = battle_wdl >= 0
    if bool(mask.any()):
        battle_loss = nn.functional.cross_entropy(
            out["battle_wdl_logits"][mask], battle_wdl[mask].long())
    else:
        battle_loss = th.zeros((), dtype=out["p_win_logit"].dtype)
    margin_loss = nn.functional.mse_loss(out["margin"], margin.float() / 6.0)
    total = p_win_loss + float(aux_battle) * battle_loss + float(aux_margin) * margin_loss
    return {"total": total, "p_win": p_win_loss, "battle": battle_loss,
            "margin": margin_loss}


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via average ranks (tie-safe, no sklearn dependency).
    Equals P(score(pos) > score(neg)) + 0.5 * P(tie)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"binary_auc_degenerate:pos={n_pos}:neg={n_neg}")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def save_heads(out_dir: Path, heads: VGameHeads, metadata: dict[str, Any],
               *, extractor_sha256: Optional[str] = None,
               mode: str = ARTIFACT_MODE_FROZEN,
               extractor_state: Optional[dict[str, Any]] = None) -> Path:
    """Write `<out_dir>/vgame_heads.pt` for either artifact kind.

    `mode="unfrozen"` REQUIRES `extractor_state` (the fine-tuned copy's
    weights) and `mode="frozen"` forbids it, so a caller cannot half-write
    an unfrozen artifact that would then silently serve the origin weights.
    `extractor_sha256` pins the origin BC checkpoint in both kinds.
    """
    mode = str(mode)
    if mode not in ARTIFACT_MODES:
        raise ValueError(f"vgame_artifact_bad_mode:{mode}")
    if mode == ARTIFACT_MODE_UNFROZEN and not extractor_state:
        raise ValueError("vgame_artifact_unfrozen_without_extractor_state")
    if mode == ARTIFACT_MODE_FROZEN and extractor_state:
        raise ValueError("vgame_artifact_frozen_with_extractor_state")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / HEADS_CHECKPOINT_NAME
    th.save(
        {
            "mode": mode,
            "extractor_sha256": extractor_sha256,
            "extractor_state": (
                {k: v.detach().clone() for k, v in extractor_state.items()}
                if extractor_state else None),
            "heads_state": heads.state_dict(),
            "features_dim": int(heads.features_dim),
            "bypass_dim": int(heads.bypass_dim),
            "hidden": int(heads.hidden),
            "dropout": float(heads.dropout),
            "metadata": metadata,
        },
        path,
    )
    return path


def load_vgame_model(
    heads_path: str | Path,
    extractor_checkpoint: str | Path,
    *,
    observation_mode: str = OBSERVATION_MODE_V4,
    max_turn: int = 15,
) -> tuple[VGameModel, dict[str, Any]]:
    """Serve-path loader for BOTH artifact kinds (module docstring).

    Order is deliberate: read the artifact, check the sha pin, and only
    then build the extractor -- a mismatch should fail before the expensive
    `MaskablePPO.load`, and it must fail for the unfrozen kind too (the
    origin checkpoint is still where the architecture and observation space
    come from, so a different file is a different model).

    For `mode="unfrozen"` the origin weights are then OVERWRITTEN by the
    artifact's fine-tuned `extractor_state` (strict, so an architecture
    drift raises), and the result is re-frozen for serving.
    """
    payload = th.load(str(heads_path), map_location="cpu", weights_only=False)
    mode = str(payload.get("mode") or ARTIFACT_MODE_FROZEN)
    if mode not in ARTIFACT_MODES:
        raise RuntimeError(f"vgame_artifact_bad_mode:{mode}:path={heads_path}")
    pinned = payload.get("extractor_sha256")
    if pinned:
        actual = file_sha256(extractor_checkpoint)
        if actual != pinned:
            raise RuntimeError(
                f"vgame_extractor_checkpoint_mismatch:mode={mode}:pinned={pinned[:12]}:"
                f"got={actual[:12]}:path={extractor_checkpoint}")

    extractor, ext_report = load_frozen_extractor(
        extractor_checkpoint, observation_mode=observation_mode, max_turn=int(max_turn))
    ext_report["mode"] = mode
    if mode == ARTIFACT_MODE_UNFROZEN:
        state = payload.get("extractor_state")
        if not state:
            raise RuntimeError(
                f"vgame_artifact_unfrozen_without_extractor_state:path={heads_path}")
        extractor.load_state_dict(state, strict=True)
        extractor.requires_grad_(False)
        extractor.eval()
        ext_report["extractor_weights_source"] = "artifact_finetuned_copy"
    else:
        ext_report["extractor_weights_source"] = "origin_checkpoint"

    heads = VGameHeads(int(payload["features_dim"]), hidden=int(payload["hidden"]),
                       bypass_dim=int(payload.get("bypass_dim", 0)),
                       dropout=float(payload.get("dropout", 0.0)))
    heads.load_state_dict(payload["heads_state"], strict=True)
    heads.eval()
    model = VGameModel(extractor, heads)
    return model, {"mode": mode, "extractor": ext_report,
                   "metadata": payload.get("metadata", {})}
