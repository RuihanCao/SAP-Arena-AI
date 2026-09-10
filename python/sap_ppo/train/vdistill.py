"""Score-distillation objective + group metrics (exp12 route a, wave A3).

`train/vgame_model.py` stays exactly as W1'c left it -- the extractor
loader, the freeze/clone contract, `VGameHeads`, the 4-dim race bypass and
both artifact kinds are reused verbatim. What route a changes is the TARGET
and the CURRENCY, and that is all this module adds:

- the LINK between the rollout teacher's score and the head's output;
- a within-GROUP pairwise ranking loss, because the currency search needs is
  within-decision ranking (RESULTS_W1 finding 19), not pooled regression
  accuracy;
- the group metrics that early stopping and the gates are decided on.

## The link (documented choice, PLAN A3 leaves it open)

The teacher score is `_rollout_score_candidate`'s own
`mean_outcome + 0.01 * mean_lives_diff`. Both terms are STRUCTURALLY
bounded: `mean_outcome` averages per-repeat values in {0, 0.5, 1}, and
`mean_lives_diff` averages `player_lives - opponent_lives` with each side in
[0, 6]. So the score lives in `[-0.06, 1.06]` by construction, not by
observation -- and the A2 build measures exactly that range on all 566244
rows.

We therefore train with a LOGISTIC link on the affinely squashed score:

    y = (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)   in [0, 1]
    loss = BCE-with-logits(p_win_logit, y)              (soft targets)
    predicted score = SCORE_MIN + sigmoid(logit) * (SCORE_MAX - SCORE_MIN)

Why this over plain MSE on the raw score:

1. It reuses the head, the sigmoid serve path and both artifact kinds
   unchanged, so nothing downstream (`load_vgame_model`, the W2 scorer, the
   agreement-set harness) needs a new code path -- `p_win` simply now means
   "the teacher's normalized rollout score", which is monotone in the
   teacher's own estimate of P(win) and is recorded as such in metadata.
2. The prediction cannot leave the target's structural range, so a leaf that
   is later blended with the myopic score or shaded by a pessimism term
   stays on the scale those knobs were designed against.
3. Soft-target cross-entropy is the standard distillation loss, and its
   gradient does not vanish where MSE's does near the boundaries -- and the
   target mass here IS near a boundary (A2: mean 0.0923, sd 0.1992).

`--link mse` keeps plain MSE on the RAW score available as a sweep option so
the choice stays falsifiable rather than merely argued; the metrics below
are link-independent and both links are reported in the same units.

## The ranking loss

Pairwise logistic (RankNet), inside a group only:

    L_rank = mean over pairs i<j of the SAME decision with t_i != t_j of
             softplus( -sign(t_i - t_j) * (z_i - z_j) )

on the raw logits `z`, so it is invariant to the link's affine scale. Ties
in the teacher's own score are excluded (the teacher is indifferent there,
so there is nothing to learn). The mixing weight is a sweep dimension:

    total = link_loss + rank_weight * L_rank

## The metrics

`group_metrics` computes, per decision group, the three quantities the wave
is judged on, with EXACTLY the tie convention `eval_vgame_gates.
agreement_metrics` already uses (a pick agrees when its teacher score equals
the group max, because a tie means the teacher is indifferent):

- top-1 agreement (the early-stop signal and the G2b' gate's quantity);
- mean per-group Spearman (undefined groups excluded and counted);
- mean teacher-score regret of the pick.

`top1_chance_with_ties` is the matching chance floor: the probability a
uniform random pick lands on a teacher-argmax, which is where the
pre-registered 0.2865 on the W1d agreement set comes from (a plain 1/n would
say 0.25 and would overstate the model's lift).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn as nn

# Structural bounds of `search_recommender._rollout_score_candidate`'s score
# (module docstring). NOT fitted to data: an empirical min/max would move
# between waves and silently change what a trained artifact means.
SCORE_MIN = -0.06
SCORE_MAX = 1.06
SCORE_SPAN = SCORE_MAX - SCORE_MIN

LINK_LOGIT = "logit"
LINK_MSE = "mse"
LINK_CHOICES = (LINK_LOGIT, LINK_MSE)


def squash_score(score: Any) -> Any:
    """Raw teacher score -> [0, 1] target for the logistic link."""
    return (score - SCORE_MIN) / SCORE_SPAN


def unsquash_score(prob: Any) -> Any:
    """Head probability -> the teacher-score scale (the inverse of
    `squash_score`); the serve-side number a search leaf would use."""
    return SCORE_MIN + prob * SCORE_SPAN


def distill_loss(logits: th.Tensor, target_score: th.Tensor, *,
                 link: str = LINK_LOGIT) -> th.Tensor:
    """The per-row regression term. `target_score` is the RAW teacher score
    in both links, so the two are directly comparable in reports."""
    if link == LINK_LOGIT:
        y = th.clamp(squash_score(target_score.float()), 0.0, 1.0)
        return nn.functional.binary_cross_entropy_with_logits(logits, y)
    if link == LINK_MSE:
        pred = unsquash_score(th.sigmoid(logits))
        return nn.functional.mse_loss(pred, target_score.float())
    raise ValueError(f"vdistill_bad_link:{link}")


def predicted_score(logits: th.Tensor) -> th.Tensor:
    """Logits -> the teacher-score scale (both links serve identically)."""
    return unsquash_score(th.sigmoid(logits))


def pairwise_rank_loss(z: th.Tensor, t: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Within-group pairwise logistic ranking loss over PADDED groups.

    `z` (predicted logits), `t` (teacher scores) and `mask` are all
    `[n_groups, max_candidates]`; padded slots are masked out. Pairs whose
    teacher scores TIE are excluded -- the teacher expresses no preference
    there, and including them would pull the model toward a constant.
    Returns a 0-dim tensor; a batch with no comparable pair contributes
    exactly 0 (and keeps the graph, so callers need no branch).
    """
    if z.ndim != 2 or z.shape != t.shape or z.shape != mask.shape:
        raise ValueError(f"vdistill_rank_shape:{tuple(z.shape)}:{tuple(t.shape)}:"
                         f"{tuple(mask.shape)}")
    dz = z.unsqueeze(2) - z.unsqueeze(1)
    dt = t.unsqueeze(2) - t.unsqueeze(1)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    upper = th.triu(th.ones(z.shape[1], z.shape[1], dtype=th.bool, device=z.device),
                    diagonal=1)
    valid = pair_mask & upper.unsqueeze(0) & (dt != 0)
    n_pairs = valid.sum()
    if int(n_pairs) == 0:
        return (z * 0.0).sum()
    losses = nn.functional.softplus(-th.sign(dt) * dz)
    return (losses * valid.float()).sum() / n_pairs.float()


def top1_chance_with_ties(teacher: np.ndarray, group_start: np.ndarray,
                          group_size: np.ndarray) -> Optional[float]:
    """Chance floor of top-1 agreement: the mean, over groups, of
    (#teacher-argmax candidates / #candidates).

    This is the number a UNIFORM random picker scores, and it is strictly
    above 1/n whenever the teacher ties -- which it does often enough that
    the W1d agreement set's floor is 0.2865, not 0.25. Every route-a claim
    of "above chance" is measured against this.
    """
    fracs = []
    for start, size in zip(group_start.tolist(), group_size.tolist()):
        if int(size) < 2:
            continue
        block = teacher[int(start): int(start) + int(size)]
        best = float(np.nanmax(block))
        fracs.append(float((block == best).sum()) / float(size))
    return float(np.mean(fracs)) if fracs else None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.size, dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def group_metrics(pred: np.ndarray, teacher: np.ndarray, group_start: np.ndarray,
                  group_size: np.ndarray, *, turns: Optional[np.ndarray] = None,
                  turn_buckets: Optional[tuple[tuple[str, int, Optional[int]], ...]] = None,
                  ) -> dict[str, Any]:
    """Top-1 agreement / Spearman / regret over decision groups.

    Tie convention (identical to `eval_vgame_gates.agreement_metrics`): the
    pick agrees when ITS teacher score equals the group's maximum, so a
    teacher indifference is never scored as a model error.
    """
    pred = np.asarray(pred, dtype=np.float64).ravel()
    teacher = np.asarray(teacher, dtype=np.float64).ravel()
    agree = 0
    used = 0
    skipped_small = 0
    spearmans: list[float] = []
    undefined = 0
    regrets: list[float] = []
    per_group_agree: list[int] = []
    per_group_turn: list[int] = []
    for start, size in zip(group_start.tolist(), group_size.tolist()):
        start, size = int(start), int(size)
        if size < 2:
            skipped_small += 1
            continue
        used += 1
        t = teacher[start:start + size]
        p = pred[start:start + size]
        pick = int(np.argmax(p))
        best = float(np.nanmax(t))
        hit = int(float(t[pick]) == best)
        agree += hit
        regrets.append(best - float(t[pick]))
        rp, rt = _average_ranks(p), _average_ranks(t)
        if float(rp.std()) == 0.0 or float(rt.std()) == 0.0:
            undefined += 1
        else:
            spearmans.append(float(np.corrcoef(rp, rt)[0, 1]))
        per_group_agree.append(hit)
        if turns is not None:
            per_group_turn.append(int(np.asarray(turns).ravel()[start]))

    out: dict[str, Any] = {
        "n_groups": int(group_start.size),
        "n_groups_used": used,
        "n_groups_skipped_lt2": skipped_small,
        "n_spearman_undefined": undefined,
        "top1_agreement": (agree / used) if used else None,
        "top1_chance_with_ties": top1_chance_with_ties(teacher, group_start, group_size),
        "spearman_mean": (float(np.mean(spearmans)) if spearmans else None),
        "regret_mean": (float(np.mean(regrets)) if regrets else None),
    }
    if turns is not None and turn_buckets:
        buckets: dict[str, Any] = {}
        agree_arr = np.asarray(per_group_agree, dtype=np.float64)
        turn_arr = np.asarray(per_group_turn, dtype=np.int64)
        for name, lo, hi in turn_buckets:
            m = (turn_arr >= lo) if hi is None else ((turn_arr >= lo) & (turn_arr <= hi))
            n = int(m.sum())
            buckets[name] = {"groups": n,
                             "top1_agreement": (float(agree_arr[m].mean()) if n else None)}
        out["turn_buckets"] = buckets
    return out


def padded_group_index(group_start: np.ndarray, group_size: np.ndarray,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """`[n_groups, max_size]` row-index matrix + validity mask.

    Built once per run: batching by GROUP is what makes the ranking loss and
    the group metrics expressible, and this is the only structure the batch
    loop needs (the A2 build guarantees a group's rows are contiguous).
    """
    n_groups = int(group_start.size)
    width = int(group_size.max()) if n_groups else 0
    idx = np.zeros((n_groups, width), dtype=np.int64)
    mask = np.zeros((n_groups, width), dtype=bool)
    for g in range(n_groups):
        start, size = int(group_start[g]), int(group_size[g])
        idx[g, :size] = np.arange(start, start + size, dtype=np.int64)
        mask[g, :size] = True
    return idx, mask
