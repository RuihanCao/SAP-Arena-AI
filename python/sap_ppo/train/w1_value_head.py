"""exp13 W1c: the bounded-trophy value head, its objective and its metrics.

`PLAN_W1.md` §"Why the reward is expected trophies and not completion
probability" changes three things about the trained V and leaves everything
else alone. This module is those three things, and nothing else:

1. THE LINK. "The trained head's link changes from the `[-0.06, 1.06]`
   sigmoid to a bounded regression on `[0, 10]`." The head itself is
   unchanged -- `VGameHeads` still emits one logit -- so the change lives
   entirely in what that logit MEANS:

       value = TROPHY_MAX * sigmoid(logit)          in [0, 10]

   which is the same shape as `vdistill.unsquash_score`, on the operator's
   scale instead of the teacher's. Reusing the sigmoid rather than, say, a
   clamped linear head is what keeps `VGameHeads`, both artifact kinds and
   the serve path byte-identical; a prediction also cannot leave the target's
   structural range, which is the property `B_1`'s own arithmetic has.

2. THE RANKING TERM, WHICH IS TWO-LEVEL HERE AND WAS ONE-LEVEL IN exp12.
   "The within-group pairwise ranking term is unchanged in form, because
   per-candidate targets still exist." In exp12 route a a GROUP was a
   decision and its members were candidates, so `vdistill.pairwise_rank_loss`
   ranked rows directly. In exp13 a row is a COMPLETION AFTERSTATE, a prefix
   group owns k of them, and §Data unit and schema says "prefix-level ranking
   uses the mean prediction and mean target across that prefix's completion
   boards". So the pairwise term is applied one level up, over prefix groups
   inside a decision, and `vdistill.pairwise_rank_loss` is reused verbatim on
   the aggregated tensors -- unchanged in form, moved up a level.

   ON WHICH QUANTITY THE MEAN IS TAKEN, an ambiguity the plan does not
   resolve. "Mean prediction" could be the mean logit or the mean predicted
   trophies, and the two differ because the sigmoid is not affine. The rank
   loss here takes the mean LOGIT, because RankNet's `softplus(-sign(dt)*dz)`
   is scale-free in `z` and the trophy scale would put a typical pair
   difference deep in the linear tail of the softplus, which is a different
   loss and not "unchanged in form". Every REPORTED quantity -- top-1
   agreement, regret, calibration -- uses the mean predicted TROPHIES, which
   is what a reader means by a prefix's predicted value. Both are recorded
   next to each other by `prefix_frame`, so the choice is inspectable rather
   than buried.

3. THE SELECTOR'S CURRENCY. "The arm selector's third tie-break changes from
   MC8-anchor top-choice regret to MC8-anchor top-1 agreement, because
   regret's scale moved with the head." `prefix_group_metrics` therefore
   takes the reference target as an argument, so the same function computes
   the operator's own top-1 agreement and the MC8-anchor one without a second
   implementation that could drift from the first.

WHAT IS DELIBERATELY NOT HERE. No training loop, no data loading and no
selector policy: `tools/run_exp13_w1_train.py` owns those. This file is pure
tensor and array math so the ordering claims above can be tested without a
checkpoint, an encoder or a battle process, the same split
`w1_bellman_targets.py` keeps from `w1_arena_frame.py`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch as th
import torch.nn as nn

from .vdistill import pairwise_rank_loss, top1_chance_with_ties


# The operator's structural range (`PLAN_W1.md` §Operator: 10 trophies
# completes the game, no discount, so a return-to-go cannot leave [0, 10]).
# NOT fitted to data, for the same reason `vdistill.SCORE_MIN` is not: an
# empirical range would move between waves and silently redefine what a
# trained artifact means.
TROPHY_MIN = 0.0
TROPHY_MAX = 10.0
TROPHY_SPAN = TROPHY_MAX - TROPHY_MIN

# `metadata["target"]` of an artifact trained here. `tools/vgame_scorer.py`
# branches on it to pick the head-output-to-leaf-value map, so a W1c artifact
# served through an older scorer fails loudly instead of serving a probability
# where a trophy count is expected.
TARGET_BELLMAN_TROPHIES = "bellman_trophies"

LINK_LOGIT = "logit"
LINK_MSE = "mse"
LINK_CHOICES = (LINK_LOGIT, LINK_MSE)


def squash_trophies(value: Any) -> Any:
    """Trophies -> the [0, 1] target the logistic link regresses on."""
    return (value - TROPHY_MIN) / TROPHY_SPAN


def unsquash_trophies(prob: Any) -> Any:
    """Head probability -> trophies. The serve-side leaf value."""
    return TROPHY_MIN + prob * TROPHY_SPAN


def predicted_trophies(logits: th.Tensor) -> th.Tensor:
    """Logits -> trophies. Both links serve identically."""
    return unsquash_trophies(th.sigmoid(logits))


def bounded_value_loss(
    logits: th.Tensor, target_trophies: th.Tensor, *, link: str = LINK_LOGIT
) -> th.Tensor:
    """The per-row regression term, in trophies on both links.

    `LINK_LOGIT` is soft-target cross entropy on `target/10`, the same
    distillation loss `vdistill.distill_loss` uses and for the same reason:
    the gradient does not vanish at the boundary, and the target mass here
    sits near one (`RESULTS_W1a3.md` pins mean trophies at about 4 with the
    bulk of games dying at 0 lives, so returns-to-go pile up at the low end).
    `LINK_MSE` keeps plain squared error on the trophy scale available, so
    the choice stays falsifiable instead of merely argued.
    """
    target = target_trophies.float()
    if link == LINK_LOGIT:
        y = th.clamp(squash_trophies(target), 0.0, 1.0)
        return nn.functional.binary_cross_entropy_with_logits(logits, y)
    if link == LINK_MSE:
        return nn.functional.mse_loss(predicted_trophies(logits), target)
    raise ValueError(f"w1_value_head_bad_link:{link}")


def segment_mean(values: th.Tensor, segment_of_row: th.Tensor, n_segments: int) -> th.Tensor:
    """Mean of `values` per segment id, as a `[n_segments]` tensor.

    Differentiable and index-based rather than a python loop, because this
    runs once per batch inside the training step. A segment with no rows
    comes out 0 and is expected to be masked out by the caller; `counts` is
    clamped at 1 only to keep the division finite, never to invent a value.
    """
    if values.ndim != 1 or segment_of_row.ndim != 1:
        raise ValueError(
            f"w1_segment_mean_shape:{tuple(values.shape)}:{tuple(segment_of_row.shape)}"
        )
    totals = th.zeros(int(n_segments), dtype=values.dtype).index_add_(
        0, segment_of_row, values
    )
    counts = th.zeros(int(n_segments), dtype=values.dtype).index_add_(
        0, segment_of_row, th.ones_like(values)
    )
    return totals / th.clamp(counts, min=1.0)


def prefix_rank_loss(
    prefix_logit_mean: th.Tensor,
    prefix_target_mean: th.Tensor,
    padded_prefix_index: th.Tensor,
    padded_mask: th.Tensor,
) -> th.Tensor:
    """`vdistill.pairwise_rank_loss`, one level up: prefixes inside a decision.

    `padded_prefix_index` and `padded_mask` are `[n_decisions, max_prefixes]`;
    the index matrix may hold anything at masked slots, so it is zeroed
    through the mask before the gather rather than trusted.
    """
    if padded_prefix_index.shape != padded_mask.shape:
        raise ValueError(
            f"w1_prefix_rank_shape:{tuple(padded_prefix_index.shape)}:"
            f"{tuple(padded_mask.shape)}"
        )
    safe = padded_prefix_index * padded_mask.long()
    z = prefix_logit_mean[safe] * padded_mask.to(prefix_logit_mean.dtype)
    t = prefix_target_mean[safe] * padded_mask.to(prefix_target_mean.dtype)
    return pairwise_rank_loss(z, t, padded_mask)


def prefix_frame(
    row_values: np.ndarray, prefix_of_row: np.ndarray, n_prefixes: int
) -> np.ndarray:
    """Per-prefix mean of a per-row array. The numpy face of `segment_mean`."""
    values = np.asarray(row_values, dtype=np.float64).ravel()
    prefix = np.asarray(prefix_of_row, dtype=np.int64).ravel()
    totals = np.zeros(int(n_prefixes), dtype=np.float64)
    counts = np.zeros(int(n_prefixes), dtype=np.float64)
    np.add.at(totals, prefix, values)
    np.add.at(counts, prefix, 1.0)
    return totals / np.clip(counts, 1.0, None)


def prefix_group_metrics(
    prefix_pred: np.ndarray,
    prefix_reference: np.ndarray,
    decision_start: np.ndarray,
    decision_size: np.ndarray,
    *,
    turns: Optional[np.ndarray] = None,
    tie_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Top-1 agreement, regret and Spearman over PREFIX GROUPS in a decision.

    `prefix_reference` is a parameter and not a fixed column so the operator's
    own target and the MC8 anchor go through one implementation: the arm
    selector reads the first and its third tie-break reads the second
    (`PLAN_W1.md` §Model training and selection), and two implementations of
    "top-1 agreement" would be two chances to disagree.

    Tie convention is `vdistill.group_metrics`'s, unchanged: a pick agrees
    when ITS reference value equals the decision's maximum, because a tie
    means the reference is indifferent and a model is not wrong to break it.

    A decision with fewer than two prefix groups is skipped and counted: there
    is no ranking question there, and including it would inflate agreement
    with free hits.

    `tie_tolerance` widens "equals the maximum" to "is within tolerance of the
    maximum", and DEFAULTS TO 0.0, which is the exact comparison every existing
    caller already gets. It exists because a caller can be comparing two
    orderings of the SAME quantity computed on two paths, where a difference of
    a few 1e-9 is float noise rather than a ranking difference. Reported back as
    `tie_tolerance`, and the exact agreement is reported alongside as
    `top1_agreement_exact` so a tolerant number never hides the strict one.
    Callers that grade a MODEL against a LABEL must leave it at 0.0: there the
    two quantities are genuinely different and a near-tie is a real near-tie.
    """
    pred = np.asarray(prefix_pred, dtype=np.float64).ravel()
    ref = np.asarray(prefix_reference, dtype=np.float64).ravel()
    starts = np.asarray(decision_start, dtype=np.int64).ravel()
    sizes = np.asarray(decision_size, dtype=np.int64).ravel()

    agree = 0
    agree_exact = 0
    used = 0
    skipped_small = 0
    regrets: list[float] = []
    spearmans: list[float] = []
    undefined = 0
    per_decision_agree: list[int] = []
    per_decision_turn: list[int] = []
    for start, size in zip(starts.tolist(), sizes.tolist()):
        start, size = int(start), int(size)
        if size < 2:
            skipped_small += 1
            continue
        used += 1
        block_ref = ref[start:start + size]
        block_pred = pred[start:start + size]
        pick = int(np.argmax(block_pred))
        best = float(np.nanmax(block_ref))
        shortfall = best - float(block_ref[pick])
        # `shortfall <= 0.0` at the default tolerance is `== best` for every
        # finite value, and stays False for NaN, so the default is a no-op.
        hit = int(shortfall <= float(tie_tolerance))
        agree += hit
        agree_exact += int(float(block_ref[pick]) == best)
        regrets.append(shortfall)
        rank_pred = _average_ranks(block_pred)
        rank_ref = _average_ranks(block_ref)
        if float(rank_pred.std()) == 0.0 or float(rank_ref.std()) == 0.0:
            undefined += 1
        else:
            spearmans.append(float(np.corrcoef(rank_pred, rank_ref)[0, 1]))
        per_decision_agree.append(hit)
        if turns is not None:
            per_decision_turn.append(int(np.asarray(turns).ravel()[start]))

    out: dict[str, Any] = {
        "n_decisions": int(starts.size),
        "n_decisions_used": used,
        "n_decisions_skipped_lt2_prefixes": skipped_small,
        "n_spearman_undefined": undefined,
        "top1_agreement": (agree / used) if used else None,
        "top1_agreement_exact": (agree_exact / used) if used else None,
        "tie_tolerance": float(tie_tolerance),
        "top1_chance_with_ties": top1_chance_with_ties(ref, starts, sizes),
        "regret_mean_trophies": (float(np.mean(regrets)) if regrets else None),
        "regret_median_trophies": (float(np.median(regrets)) if regrets else None),
        "spearman_mean": (float(np.mean(spearmans)) if spearmans else None),
    }
    if turns is not None and per_decision_turn:
        agree_arr = np.asarray(per_decision_agree, dtype=np.float64)
        turn_arr = np.asarray(per_decision_turn, dtype=np.int64)
        out["per_decision_agreement_by_turn"] = {
            str(turn): {
                "n_decisions": int((turn_arr == turn).sum()),
                "top1_agreement": float(agree_arr[turn_arr == turn].mean()),
            }
            for turn in sorted(set(turn_arr.tolist()))
        }
    return out


def regression_metrics(pred_trophies: np.ndarray, target_trophies: np.ndarray) -> dict[str, Any]:
    """Held-out regression error ON THE BOUNDED TROPHY SCALE.

    This is the arm selector's FIRST tie-break, so it is denominated in the
    units the plan names ("median held-out regression error on the bounded
    trophy scale") and not in the link's internal [0, 1] coordinates. The
    median absolute error is the selector's quantity; the mean and the RMSE
    are reported beside it because a median alone hides a tail.
    """
    pred = np.asarray(pred_trophies, dtype=np.float64).ravel()
    target = np.asarray(target_trophies, dtype=np.float64).ravel()
    if pred.size != target.size:
        raise ValueError(f"w1_regression_metrics_shape:{pred.size}:{target.size}")
    if pred.size == 0:
        return {"n_rows": 0, "median_abs_error": None, "mean_abs_error": None,
                "rmse": None, "bias_mean": None}
    err = pred - target
    return {
        "n_rows": int(pred.size),
        "median_abs_error": float(np.median(np.abs(err))),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_mean": float(np.mean(err)),
    }


def saturation_metrics(pred_trophies: np.ndarray, *, edge: float = 0.05) -> dict[str, Any]:
    """Where the bounded head's mass sits, and how much of it is at the rails.

    `PLAN_W1.md` §Model training and selection lists "prediction histograms
    and fraction at the bounds of the trophy scale" as mandatory red-flag
    telemetry and Checkpoint A has a bar on unexplained saturation, so this is
    computed by the trainer rather than reconstructed later from a checkpoint.
    `edge` is a fraction of the span: the default counts a prediction within
    0.5 trophies of a rail as sitting on it.
    """
    pred = np.asarray(pred_trophies, dtype=np.float64).ravel()
    if pred.size == 0:
        return {"n_rows": 0}
    margin = float(edge) * TROPHY_SPAN
    hist, edges = np.histogram(pred, bins=20, range=(TROPHY_MIN, TROPHY_MAX))
    return {
        "n_rows": int(pred.size),
        "mean": float(pred.mean()),
        "sd": float(pred.std()),
        "min": float(pred.min()),
        "max": float(pred.max()),
        "fraction_at_low_bound": float((pred <= TROPHY_MIN + margin).mean()),
        "fraction_at_high_bound": float((pred >= TROPHY_MAX - margin).mean()),
        "bound_margin_trophies": margin,
        "histogram_counts": [int(x) for x in hist],
        "histogram_edges": [float(x) for x in edges],
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-safe average ranks. Same convention as `vdistill._average_ranks`,
    duplicated here only because that one is private to its module."""
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
