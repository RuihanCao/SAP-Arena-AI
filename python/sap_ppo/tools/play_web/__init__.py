"""Local web simulator for interactive SAP shop-phase testing."""

from __future__ import annotations

from ...engine import resolve_end_turn_pre_battle
from ...oracles.sap_calc_battle_oracle import run_battle_oracle_with_config
from ...opponents import (
    parse_replay_for_calculator_state,
    render_replay_image_from_calc_rows,
    render_replay_image_from_raw_battles,
)
from ...train.opponents import ReplaySnapshotProvider


# The two replay-database samplers are wrappers rather than re-exports: a
# re-export imports the Postgres layer whenever anything imports play_web, which
# put it in the public-release closure. These keep the patch seam described
# below identical, because the attribute still lives on this package.
def sample_opponent_team_for_pid(pid, turn, **kwargs):
    """Sample a specific participation's opponent board. Imports at call time."""
    from ...opponents.replay_db import sample_opponent_team_for_pid as _fn

    return _fn(pid, turn, **kwargs)


def sample_random_opponent_team(turn, **kwargs):
    """Sample any opponent board for a turn. See `sample_opponent_team_for_pid`."""
    from ...opponents.replay_db import sample_random_opponent_team as _fn

    return _fn(turn, **kwargs)

# Re-exported so `unittest.mock.patch("sap_ppo.tools.play_web.<name>", ...)`
# keeps working after the split: app.py looks these eight names up through
# this package's own namespace (see app.py's `_play_web_pkg` import) rather
# than binding a private copy, so patching the attribute here is what
# actually changes App's behavior.
__all__ = [
    "App",
    "Handler",
    "main",
    "_build_arg_parser",
    "ReplaySnapshotProvider",
    "parse_replay_for_calculator_state",
    "render_replay_image_from_calc_rows",
    "render_replay_image_from_raw_battles",
    "resolve_end_turn_pre_battle",
    "run_battle_oracle_with_config",
    "sample_opponent_team_for_pid",
    "sample_random_opponent_team",
]

from .app import App  # noqa: E402
from .http_app import Handler  # noqa: E402
from .cli import _build_arg_parser, main  # noqa: E402
