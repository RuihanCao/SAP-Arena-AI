"""WHAT IS NEW HERE AND WHY. `isotonic()` used to be reachable only as a closure
over the fit's 33,612 points, which cannot be written to a file. A curve that
lives only inside the process that fitted it cannot be frozen, hashed or
pinned in a manifest, so a label run would have to refit from whatever data it
was labelling -- an in-the-loop fit of the target on the data being targeted.
Splitting the fit from the evaluator is what makes the curve an artifact:

    fit_isotonic(xs, ys) -> (x_knots, y_knots)   the fit, once, offline
    compress_knots(...)  -> (x_knots, y_knots)   the same function, smaller
    interpolator(...)    -> Callable[[float], float]   the evaluator, anywhere

`isotonic(xs, ys)` is still exactly `interpolator(*fit_isotonic(xs, ys))`, so
every existing caller keeps its meaning."""

from __future__ import annotations

from typing import Callable


def fit_isotonic(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators isotonic regression, as knots.

    Monotone by construction, fitted from value scores to realised subsequent
    trophies, nothing else assumed about its shape. Returns the sorted x and
    the fitted y at each of them, which is the full uncompressed curve.
    """
    if not xs:
        raise ValueError("isotonic fit needs at least one point")
    pairs = sorted(zip(xs, ys))
    x_sorted = [p[0] for p in pairs]
    blocks: list[list[float]] = []  # [sum, weight, value]
    for _, y in pairs:
        blocks.append([float(y), 1.0, float(y)])
        while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
            b = blocks.pop()
            a = blocks.pop()
            total = a[0] + b[0]
            weight = a[1] + b[1]
            blocks.append([total, weight, total / weight])
    fitted: list[float] = []
    for block in blocks:
        fitted.extend([block[2]] * int(block[1]))
    return x_sorted, fitted


def interpolator(
    x_knots: list[float], y_knots: list[float]
) -> Callable[[float], float]:
    """Linear interpolation between knots, clamped outside them.

    Defined everywhere, so a leaf score outside the fit's range still maps to
    a number rather than raising in the middle of a labelling shard.
    """
    if not x_knots:
        raise ValueError("an isotonic curve needs at least one knot")
    if len(x_knots) != len(y_knots):
        raise ValueError("knot arrays must have equal length")

    def f(x: float) -> float:
        if x <= x_knots[0]:
            return y_knots[0]
        if x >= x_knots[-1]:
            return y_knots[-1]
        low, high = 0, len(x_knots) - 1
        while high - low > 1:
            mid = (low + high) // 2
            if x_knots[mid] <= x:
                low = mid
            else:
                high = mid
        span = x_knots[high] - x_knots[low]
        if span <= 0:
            return y_knots[low]
        weight = (x - x_knots[low]) / span
        return y_knots[low] + (y_knots[high] - y_knots[low]) * weight

    return f


def compress_knots(
    x_knots: list[float], y_knots: list[float]
) -> tuple[list[float], list[float]]:
    """Drop the knots `interpolator` cannot distinguish. Exactly, not nearly."""
    n = len(x_knots)
    if n != len(y_knots):
        raise ValueError("knot arrays must have equal length")
    keep: list[int] = []
    index = 0
    while index < n:
        last = index
        while last + 1 < n and y_knots[last + 1] == y_knots[index]:
            last += 1
        keep.append(index)
        if last != index:
            keep.append(last)
        index = last + 1
    return [x_knots[i] for i in keep], [y_knots[i] for i in keep]


def isotonic(xs: list[float], ys: list[float]) -> Callable[[float], float]:
    """Fit and return the curve as a callable. The pre-consolidation signature."""
    return interpolator(*fit_isotonic(xs, ys))
