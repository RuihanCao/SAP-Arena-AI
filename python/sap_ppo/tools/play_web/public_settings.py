"""Search settings offered by the local demo."""

import math

ROOT_WIDTH = 72
COMPLETION_WIDTH = 4
INITIAL_SAMPLES = 12
DEFAULT_TURN_SECONDS = 105.0
SEARCH_MODES = ("grow-k", "fixed-all")
MODE_LABELS = {"grow-k": "Grow k", "fixed-all": "Fixed-all"}
_ENGINE_GEARS = {"grow-k": "resample-clock", "fixed-all": "measured"}


def engine_gear(mode: str) -> str:
    if mode not in SEARCH_MODES:
        raise ValueError("Choose Grow k or Fixed-all.")
    return _ENGINE_GEARS[mode]


def public_mode(gear: str) -> str:
    return {value: key for key, value in _ENGINE_GEARS.items()}.get(gear, gear)


def turn_seconds(value: float) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 1 <= seconds <= 600:
        raise ValueError("AI turn time must be between 1 and 600 seconds.")
    return seconds
