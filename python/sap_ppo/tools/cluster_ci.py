"""Shared cluster-robust bootstrap CI helper for one-turn tempo eval probes."""

from __future__ import annotations

import random
import statistics as st
from typing import Sequence, Tuple


def cluster_ci(
    pairs: Sequence[Tuple[object, float]],
    iters: int = 5000,
    seed: int = 41,
) -> Tuple[float, float, float, int]:
    """Block-bootstrap a cluster-robust CI for the mean of `pairs` values.

    `pairs` is a sequence of (game_id, value) tuples. Resampling draws whole
    game ids with replacement (all of a game's values move together), which
    is what makes this "cluster-robust" rather than a naive per-row bootstrap.

    Returns (mean, lo95, hi95, n_games):
      - `mean` is the plain arithmetic mean of all `value`s (not a bootstrap
        statistic).
      - `lo95`/`hi95` are the 2.5th/97.5th percentiles of the game-clustered
        bootstrap distribution of the mean.
      - `n_games` is the number of distinct game ids.
    """
    by_game: dict[object, list[float]] = {}
    for game_id, value in pairs:
        by_game.setdefault(game_id, []).append(float(value))
    game_ids = list(by_game)

    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(int(iters)):
        sampled_ids = [rng.choice(game_ids) for _ in game_ids]
        values = [v for gid in sampled_ids for v in by_game[gid]]
        boots.append(sum(values) / len(values))
    boots.sort()

    lo = boots[int(0.025 * iters)]
    hi = boots[int(0.975 * iters)]
    mean = st.mean(value for _, value in pairs)
    return mean, lo, hi, len(game_ids)
