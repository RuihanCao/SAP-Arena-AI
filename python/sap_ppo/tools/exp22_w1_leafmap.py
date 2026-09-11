"""Fixed monotone calibration from value scores to trophy targets.

The default fit uses twenty equal-count bins and piecewise-linear interpolation.
Endpoint knots extend it across the reachable score domain using bounded slopes.
The resulting map is fixed during head training; interpolation remains
differentiable between knots. A full-point isotonic fit is also available.

The schema identifier is retained for compatibility with existing checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .w1_isotonic import compress_knots, fit_isotonic


SCHEMA_VERSION = "exp22_w1_leafmap_v1"

METHOD_VENTILE_MEANS = "ventile_means"
METHOD_ISOTONIC_FULL = "isotonic_full"
METHODS = (METHOD_VENTILE_MEANS, METHOD_ISOTONIC_FULL)

TAIL_LINEAR = "linear"
TAIL_CLAMP = "clamp"
TAILS = (TAIL_LINEAR, TAIL_CLAMP)

# `metadata["target"]` of an artifact whose head output is carried onto
# trophies by one of these maps. `vgame_scorer.VALUE_KINDS` branches on it.
TARGET_MC8_LEAFMAP = "mc8_leafmap_trophies"


TROPHY_MIN = 0.0
TROPHY_MAX = 10.0


SCORE_MIN = -0.06
SCORE_MAX = 1.06


class LeafMapError(RuntimeError):
    """Leafmaperror."""


def assert_fit_rows_disjoint(
    fit_game_ids: Iterable[int],
    graded_game_ids: Iterable[int],
    *,
    where: str = "gate_0c",
) -> dict[str, Any]:
    """Assert fit rows disjoint."""
    fit = {int(g) for g in fit_game_ids}
    graded = {int(g) for g in graded_game_ids}
    overlap = sorted(fit & graded)
    report = {
        "gate": "0c_leaf_map_fitting_rows",
        "n_fit_games": len(fit),
        "n_graded_games": len(graded),
        "n_overlapping_games": len(overlap),
        "overlapping_game_ids": overlap[:20],
        "unit": "whole game",
        "pass": not overlap,
        "why": (
            "a monotone leaf map is not rank-neutral where gate 1 grades: a "
            "prefix group's score is a mean over completions taken before the "
            "transform, f(mean) != mean(f), and RESULTS_W1b.md:178 measured "
            "the argmax flip rate at 15.61% [15.00%, 16.22%] over 500 games"
        ),
    }
    if overlap:
        raise LeafMapError(
            f"{where}_fit_rows_intersect_graded_rows:"
            f"n_overlapping_games={len(overlap)}:"
            f"first={overlap[:10]}"
        )
    if not fit:
        raise LeafMapError(f"{where}_no_fitting_games")
    return report


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------


def ventile_monotonicity(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 20
) -> dict[str, Any]:
    """`E[y | x]` over equal-count bins, and where it goes backwards."""
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    if xs.size != ys.size:
        raise LeafMapError(f"leafmap_ventile_shape:{xs.size}:{ys.size}")
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    if xs.size < n_bins:
        return {"n_bins": 0, "n_rows": int(xs.size), "means": [],
                "n_violations": None, "strictly_increasing": None,
                "note": "fewer rows than bins; the check is not defined here"}
    order = np.argsort(xs, kind="mergesort")
    bins = np.array_split(order, int(n_bins))
    means = [float(ys[idx].mean()) for idx in bins if idx.size]
    violations = [
        {"bin": i, "prev": means[i - 1], "this": means[i]}
        for i in range(1, len(means))
        if not means[i] > means[i - 1]
    ]
    return {
        "n_bins": len(means),
        "n_rows": int(xs.size),
        "means": [round(m, 6) for m in means],
        "n_violations": len(violations),
        "violations": violations[:10],
        "strictly_increasing": not violations,
    }


def fit_ventile_knots(
    x: np.ndarray, y: np.ndarray, *, n_bins: int
) -> tuple[list[float], list[float], dict[str, Any]]:
    """`E[y | x]` over equal-count bins, made monotone, as knots."""
    order = np.argsort(x, kind="mergesort")
    bins = [idx for idx in np.array_split(order, int(n_bins)) if idx.size]
    if len(bins) < 2:
        raise LeafMapError(f"leafmap_too_few_bins:{len(bins)}:n_rows={x.size}")
    xm = [float(x[idx].mean()) for idx in bins]
    ym = [float(y[idx].mean()) for idx in bins]
    counts = [int(idx.size) for idx in bins]

    # Weighted PAVA over the bin means. `fit_isotonic` is unweighted, and the
    # bins are equal-count by construction, so it is fed the bin means with
    # their own order as x and is exact here; the weights are recorded so an
    # unequal final bin is visible rather than assumed away.
    _x_sorted, fitted = fit_isotonic(list(range(len(ym))), ym)
    pooled = sum(1 for a, b in zip(fitted, fitted[1:]) if b == a)

    knots_x: list[float] = []
    knots_y: list[float] = []
    for xv, yv in zip(xm, fitted):
        if knots_y and yv == knots_y[-1]:
            # A pooled pair is one flat segment; keep its far end only, so the
            # map stays a function and does not acquire a zero-slope interior.
            knots_x[-1] = xv
            continue
        knots_x.append(xv)
        knots_y.append(float(yv))
    report = {
        "n_bins": len(bins),
        "bin_counts": counts,
        "n_pooled_by_pava": pooled,
        "raw_bin_means_y": [round(v, 6) for v in ym],
        "n_knots": len(knots_x),
    }
    if len(knots_x) < 2:
        raise LeafMapError(
            f"leafmap_degenerate_after_pava:n_knots={len(knots_x)}. Every bin "
            "pooled to one value, so E[y|x] is constant and no monotone map exists."
        )
    return knots_x, knots_y, report


def add_endpoint_knots(
    x_knots: list[float], y_knots: list[float]
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Extend the fit to the head's reachable input bounds, and stop there.

    This is what `tail="linear"` MEANS here, and it is deliberately not a
    special case inside the evaluator. Two synthetic knots are added, at
    `SCORE_MIN` and `SCORE_MAX`, carrying the end segments' own slopes,
    with each slope capped so the extrapolated value lands exactly on the
    operator's `[0, 10]` rather than past it.

    Three properties fall out at once, and each replaces something worse:

    A side whose end segment is flat cannot be extended without staying flat,
    so it is left alone and SAID so rather than given an invented slope."""
    xs = [float(v) for v in x_knots]
    ys = [float(v) for v in y_knots]
    report: dict[str, Any] = {"tail_left": "clamped", "tail_right": "clamped"}

    left_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    if left_slope > 0 and xs[0] > SCORE_MIN and ys[0] > TROPHY_MIN:
        capped = min(left_slope, (ys[0] - TROPHY_MIN) / (xs[0] - SCORE_MIN))
        xs.insert(0, SCORE_MIN)
        ys.insert(0, ys[0] - capped * (xs[1] - SCORE_MIN))
        report["tail_left"] = "linear"
        report["tail_left_slope"] = capped
        report["tail_left_slope_capped"] = bool(capped < left_slope)

    right_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    if right_slope > 0 and xs[-1] < SCORE_MAX and ys[-1] < TROPHY_MAX:
        capped = min(right_slope, (TROPHY_MAX - ys[-1]) / (SCORE_MAX - xs[-1]))
        xs.append(SCORE_MAX)
        ys.append(ys[-1] + capped * (SCORE_MAX - xs[-2]))
        report["tail_right"] = "linear"
        report["tail_right_slope"] = capped
        report["tail_right_slope_capped"] = bool(capped < right_slope)

    report["reachable_input_range"] = [SCORE_MIN, SCORE_MAX]
    report["reachable_output_range"] = [ys[0], ys[-1]]
    return xs, ys, report


def fit_leaf_map(
    v0_score: np.ndarray,
    target: np.ndarray,
    *,
    fit_game_ids: np.ndarray,
    graded_game_ids: Iterable[int],
    target_mode: str,
    method: str = METHOD_VENTILE_MEANS,
    n_bins: int = 20,
    tail: str = TAIL_LINEAR,
    provenance: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Fit leaf map."""
    gate = assert_fit_rows_disjoint(fit_game_ids, graded_game_ids)
    if method not in METHODS:
        raise LeafMapError(f"leafmap_bad_method:{method}:known={list(METHODS)}")
    if tail not in TAILS:
        raise LeafMapError(f"leafmap_bad_tail:{tail}:known={list(TAILS)}")

    xs = np.asarray(v0_score, dtype=np.float64).ravel()
    ys = np.asarray(target, dtype=np.float64).ravel()
    if xs.size != ys.size:
        raise LeafMapError(f"leafmap_fit_shape:{xs.size}:{ys.size}")
    finite = np.isfinite(xs) & np.isfinite(ys)
    n_dropped = int((~finite).sum())
    xs, ys = xs[finite], ys[finite]
    if xs.size == 0:
        raise LeafMapError("leafmap_fit_no_finite_rows")

    if method == METHOD_VENTILE_MEANS:
        x_knots, y_knots, fit_report = fit_ventile_knots(xs, ys, n_bins=int(n_bins))
        n_knots_full = fit_report["n_bins"]
        description = (
            f"E[y | x] over {int(n_bins)} equal-count bins of x, made monotone by "
            "pool-adjacent-violators over the bin means; linear interpolation "
            f"between knots; tails {tail}; output clipped to the operator's [0, 10]"
        )
    else:
        x_full, y_full = fit_isotonic(xs.tolist(), ys.tolist())
        x_knots, y_knots = compress_knots(x_full, y_full)
        n_knots_full = len(x_full)
        fit_report = {"n_distinct_fitted_values": len(set(y_full))}
        description = (
            "pool-adjacent-violators isotonic regression over every point; "
            f"linear interpolation between knots; tails {tail}; output clipped "
            "to the operator's [0, 10]"
        )

    # Recorded BEFORE the tails are added: this is the range the conditional
    # mean actually covers, and it is the dispersion number to read. The range
    # after the tails is [0, 10] by construction and says nothing.
    fit_report["fitted_x_range"] = [float(x_knots[0]), float(x_knots[-1])]
    fit_report["fitted_y_range"] = [float(y_knots[0]), float(y_knots[-1])]
    if tail == TAIL_LINEAR:
        x_knots, y_knots, tail_report = add_endpoint_knots(x_knots, y_knots)
        fit_report.update(tail_report)

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": method,
        "tail": tail,
        "method": description,
        "x": "leaf score emitted by vdistill.unsquash_score",
        "y": (
            f"{target_mode}: return-to-go in trophies on [0, 10] under the "
            "continuation that label mode pins"
        ),
        "purpose": (
             "A fixed monotone map that "
            "maps the value head output onto the training target, fitted once and "
            "frozen, with the head's parameters carried over"
        ),
        "target_mode": str(target_mode),
        "knots": {"x": [float(v) for v in x_knots], "y": [float(v) for v in y_knots]},
        "gate_0c": gate,
        "fit": {
            "n_points": int(xs.size),
            "n_rows_dropped_non_finite": n_dropped,
            "n_knots_full": n_knots_full,
            "n_knots_stored": len(x_knots),
            "n_fit_games": gate["n_fit_games"],
            **fit_report,
            "x_range": [float(xs.min()), float(xs.max())],
            "y_range": [float(min(y_knots)), float(max(y_knots))],
            "y_label_range": [float(ys.min()), float(ys.max())],
            "y_label_sd": float(ys.std()),
            "split": "the training side of the held-out whole-game split",
            **(provenance or {}),
        },
        "ventiles": ventile_monotonicity(xs, ys, n_bins=int(n_bins)),
        "dispersion_note": (
            "y_range is the range of E[y | x], which is under-dispersed "
            "relative to the label by construction; the label's own range and "
            "SD are recorded beside it. With linear tails the composed model "
            "is not confined to y_range: it is bounded only by the operator's "
            "[0, 10]. Read a narrow y_range as what the map's fitted band "
            "covers, not as what the model can predict."
        ),
    }
    artifact["sha256"] = leaf_map_sha256(artifact)
    return artifact


def leaf_map_sha256(artifact: dict[str, Any]) -> str:
    """Hash of the map's DEFINING content: the knots and what they mean.

    Provenance counters are excluded so an identical curve fitted twice hashes
    the same, which is what makes the pin checkable rather than incidental.
    """
    payload = {
        "schema_version": artifact["schema_version"],
        "kind": artifact["kind"],
        "tail": artifact.get("tail", TAIL_LINEAR),
        "target_mode": artifact["target_mode"],
        "knots": artifact["knots"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_leaf_map(path: Path, artifact: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The evaluator, in both currencies the arm needs
# --------------------------------------------------------------------------


class LeafMap:
    """The frozen map, evaluated identically by numpy and by torch."""

    def __init__(self, x_knots: list[float], y_knots: list[float],
                 *, tail: str = TAIL_LINEAR,
                 fitted_x_range: Optional[list[float]] = None,
                 meta: Optional[dict[str, Any]] = None) -> None:
        if len(x_knots) != len(y_knots):
            raise LeafMapError("leafmap_knot_length_mismatch")
        if len(x_knots) < 2:
            raise LeafMapError(f"leafmap_too_few_knots:{len(x_knots)}")
        if any(b <= a for a, b in zip(x_knots, x_knots[1:])):
            raise LeafMapError("leafmap_x_knots_not_strictly_increasing")
        if any(b < a for a, b in zip(y_knots, y_knots[1:])):
            raise LeafMapError("leafmap_y_knots_not_monotone")
        if tail not in TAILS:
            raise LeafMapError(f"leafmap_bad_tail:{tail}")
        self.x_knots = [float(v) for v in x_knots]
        self.y_knots = [float(v) for v in y_knots]
        self.tail = str(tail)
        self.fitted_x_range = (
            [float(fitted_x_range[0]), float(fitted_x_range[1])]
            if fitted_x_range else [self.x_knots[0], self.x_knots[-1]]
        )
        self.meta = dict(meta or {})

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "LeafMap":
        if str(artifact.get("schema_version")) != SCHEMA_VERSION:
            raise LeafMapError(
                f"leafmap_bad_schema:{artifact.get('schema_version')}:expected={SCHEMA_VERSION}"
            )
        declared = artifact.get("sha256")
        actual = leaf_map_sha256(artifact)
        if declared and str(declared) != actual:
            raise LeafMapError(f"leafmap_sha256_mismatch:{declared}:{actual}")
        knots = artifact["knots"]
        return cls(
            list(knots["x"]), list(knots["y"]),
            tail=str(artifact.get("tail", TAIL_LINEAR)),
            # `fitted_x_range` is the band the data actually supported, as
            # distinct from the knot range, which the tails extend to the
            # head's reachable bounds. Everything outside it is extrapolation,
            # and `extrapolation_report` is how much of that a run is doing.
            fitted_x_range=(artifact.get("fit") or {}).get("fitted_x_range"),
            meta={"sha256": actual, "target_mode": artifact.get("target_mode"),
                  "kind": artifact.get("kind")},
        )

    @classmethod
    def from_path(cls, path: Path) -> "LeafMap":
        return cls.from_artifact(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def y_range(self) -> tuple[float, float]:
        return (self.y_knots[0], self.y_knots[-1])

    def __call__(self, values: Any) -> np.ndarray:
        """numpy face. Clamp into the knot range, then interpolate."""
        xk = np.asarray(self.x_knots, dtype=np.float64)
        yk = np.asarray(self.y_knots, dtype=np.float64)
        x = np.clip(np.asarray(values, dtype=np.float64), xk[0], xk[-1])
        idx = np.clip(np.searchsorted(xk, x, side="right"), 1, xk.size - 1)
        x0, x1 = xk[idx - 1], xk[idx]
        y0, y1 = yk[idx - 1], yk[idx]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def torch(self, values: Any) -> Any:
        """torch face, differentiable in `values`. The same formula, verbatim.

        The segment search runs on the DETACHED input because an index is not
        differentiable; the gradient flows through the interpolation weight,
        which is the piecewise-linear derivative and is what a chain rule
        through this map means.

        Under `TAIL_LINEAR` the knots already span everything a sigmoid can
        produce, so the clamp is unreachable from a real forward pass and the
        derivative is positive across the whole reachable domain. That is why
        this can be one formula with no tail branch: the tail rule lives in
        the KNOTS, not here.
        """
        import torch as th

        x_in = values if isinstance(values, th.Tensor) else th.as_tensor(values)
        xk = th.as_tensor(self.x_knots, dtype=x_in.dtype, device=x_in.device)
        yk = th.as_tensor(self.y_knots, dtype=x_in.dtype, device=x_in.device)
        x = th.clamp(x_in, float(self.x_knots[0]), float(self.x_knots[-1]))
        idx = th.searchsorted(xk, x.detach().contiguous(), right=True)
        idx = th.clamp(idx, 1, xk.numel() - 1)
        x0, x1 = xk[idx - 1], xk[idx]
        y0, y1 = yk[idx - 1], yk[idx]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def extrapolation_report(self, values: Any) -> dict[str, Any]:
        """How much of a score population lands OUTSIDE the fitted band.

        The tails are principled rather than flat, but a run whose scores
        mostly land in them is a run whose leaf values are mostly
        EXTRAPOLATION from about twenty conditional means, and that is a
        caveat on every number computed from it rather than a detail. It is
        measured per curve point and reported rather than assumed small.

        Read it against `fitted_x_range`, not against the knot range: the
        knots were deliberately extended to the head's reachable bounds, so
        "inside the knots" is nearly vacuous.
        """
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        low, high = self.fitted_x_range
        if arr.size == 0:
            return {"n": 0, "fraction_outside": None, "fitted_x_range": [low, high]}
        below = int((arr < low).sum())
        above = int((arr > high).sum())
        return {
            "n": int(arr.size),
            "n_below": below,
            "n_above": above,
            "fraction_below": below / arr.size,
            "fraction_above": above / arr.size,
            "fraction_outside": (below + above) / arr.size,
            "fitted_x_range": [low, high],
            "observed_range": [float(arr.min()), float(arr.max())],
        }

    def slope_range(self) -> tuple[float, float]:
        slopes = [(b - a) / (d - c) for a, b, c, d
                  in zip(self.y_knots, self.y_knots[1:], self.x_knots, self.x_knots[1:])]
        return (min(slopes), max(slopes))

    def slope_profile(self) -> dict[str, Any]:
        """Segment slopes, split into the fitted interior and the tails.

        This is the COMPRESSION mechanism stated without any data: a tail is a
        single segment whose slope was capped so the extrapolation lands on
        `[0, 10]`, and if that cap is far below the interior's typical slope
        then predictions in the tail are squeezed into a narrower output range
        than the same spread of inputs would get inside the fit. Monotonicity
        survives that; RESOLUTION does not, and a ranking is what the search
        consumes.

        A segment is a tail when its midpoint lies outside `fitted_x_range`,
        so this does not depend on remembering which knots were synthetic.
        """
        lo, hi = self.fitted_x_range
        interior: list[float] = []
        tails: list[dict[str, Any]] = []
        for x0, x1, y0, y1 in zip(self.x_knots, self.x_knots[1:],
                                  self.y_knots, self.y_knots[1:]):
            slope = (y1 - y0) / (x1 - x0)
            mid = 0.5 * (x0 + x1)
            if lo <= mid <= hi:
                interior.append(slope)
            else:
                tails.append({"x_range": [x0, x1], "slope": slope,
                              "side": ("left" if mid < lo else "right")})
        interior_sorted = sorted(interior)
        median = (interior_sorted[len(interior_sorted) // 2]
                  if interior_sorted else None)
        out: dict[str, Any] = {
            "n_interior_segments": len(interior),
            "interior_slope_median": median,
            "interior_slope_min": (min(interior) if interior else None),
            "interior_slope_max": (max(interior) if interior else None),
            "tail_segments": tails,
        }
        for entry in tails:
            key = f"tail_{entry['side']}_slope"
            out[key] = entry["slope"]
            if median:
                out[f"tail_{entry['side']}_over_interior_median"] = entry["slope"] / median
        return out

    def describe(self) -> dict[str, Any]:
        low, high = self.slope_range()
        return {
            "n_knots": len(self.x_knots),
            "tail": self.tail,
            "x_range": [self.x_knots[0], self.x_knots[-1]],
            "fitted_x_range": list(self.fitted_x_range),
            "y_range": [self.y_knots[0], self.y_knots[-1]],
            "slope_min": low,
            "slope_max": high,
            "output_bounds": [TROPHY_MIN, TROPHY_MAX],
            **self.meta,
        }
