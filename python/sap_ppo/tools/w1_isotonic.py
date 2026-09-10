"""The ONE isotonic fit exp13 W1 uses, and the one evaluator it serialises to.

WHY THIS MODULE EXISTS. `RESULTS_probes.md` red flag 13: the probe branch was
cut off `main` while the W1b readout branch was unmerged, so the same
pool-adjacent-violators fit, the same knot interpolation and the same
return-to-go definition were written twice -- in `w1_readouts.py` and in
`w1_recalibration_weights.py`. The flag said one copy should go once the
branches merged. They have merged (`main` @ `64af0cc`), so this is that one
copy, and both former sites now import it.

The definitions are unchanged, deliberately and to the last bit. Readout 1's
published curve is the artifact every later number is compared against, so a
consolidation that moved any value would silently invalidate probe 2's
reproduction gate. `test_w1_recalibration_curve.py` pins the eleven fitted
deciles readout 1 published against this implementation, and the exactness is
the point: the tolerance is 0.0, not 1e-12.

WHAT IS NEW HERE AND WHY. `isotonic()` used to be reachable only as a closure
over the fit's 33,612 points, which cannot be written to a file. A curve that
lives only inside the process that fitted it cannot be frozen, hashed or
pinned in a manifest, so a label run would have to refit from whatever data it
was labelling -- an in-the-loop fit of the target on the data being targeted.
Splitting the fit from the evaluator is what makes the curve an artifact:

    fit_isotonic(xs, ys) -> (x_knots, y_knots)   the fit, once, offline
    compress_knots(...)  -> (x_knots, y_knots)   the same function, smaller
    interpolator(...)    -> Callable[[float], float]   the evaluator, anywhere

`isotonic(xs, ys)` is still exactly `interpolator(*fit_isotonic(xs, ys))`, so
every existing caller keeps its meaning.
"""

from __future__ import annotations

from typing import Callable


def fit_isotonic(xs: list[float], ys: list[float]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators isotonic regression, as knots.

    Monotone by construction, fitted from V0 score to realised subsequent
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
    """Drop the knots `interpolator` cannot distinguish. Exactly, not nearly.

    A PAVA fit is a step function: on the canonical W1 curve 33,612 fitted
    points collapse to 165 knots. Inside a block, interpolating between the block's
    first and last x gives the same constant every interior knot would have
    given, and at a block boundary the two neighbouring knots are both kept,
    so the linear segment between blocks is the same segment. Keeping the
    first and last knot of each run is therefore not an approximation -- it is
    the same function with the redundant knots removed, which is what makes a
    small artifact possible instead of a 1.4 MB one.

    Exactness is not argued from this docstring: the fit and its compression
    are checked against each other on every point of the real fit by
    an internal analysis script, and on the committed
    curve by the unit tests.
    """
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
    """Fit and return the curve as a callable. The pre-consolidation signature.

    `w1_readouts.py` and `w1_recalibration_weights.py` both import this name;
    it is byte-for-byte the behaviour each of them used to define locally.
    """
    return interpolator(*fit_isotonic(xs, ys))
