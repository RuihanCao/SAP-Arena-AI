"""Opponent sampling helpers."""

# `replay_db` is deliberately NOT re-exported: it reaches Postgres, and a
# re-export here put it in core `end_turn.py`'s import closure. Import
# `sap_ppo.opponents.replay_db` directly if you need it.
from .replaybot_bridge import parse_replay_for_calculator_state
from .replaybot_render_bridge import render_replay_image_from_calc_rows, render_replay_image_from_raw_battles

__all__ = [
    "parse_replay_for_calculator_state",
    "render_replay_image_from_raw_battles",
    "render_replay_image_from_calc_rows",
]
