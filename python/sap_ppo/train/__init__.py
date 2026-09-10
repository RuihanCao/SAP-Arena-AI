"""Step 3 training scaffolding."""

from .env import TrainingEnv, load_initial_state_from_fixture
from .gym_env import SapPpoGymEnv
from .metrics import evaluate_maskable_model
from .observation import StateVectorEncoder
from .opponents import (
    CurriculumOpponentProvider,
    OpponentProvider,
    RandomOpponentProvider,
    ReplayDBOpponentProvider,
    ReplaySnapshotProvider,
    SelfPlayPoolProvider,
    StaticTurnOpponentProvider,
)
from .runtime import build_opponent_provider, build_training_env
from .snapshots import SNAPSHOT_VERSION, load_snapshot, save_snapshot

__all__ = [
    "TrainingEnv",
    "SapPpoGymEnv",
    "evaluate_maskable_model",
    "StateVectorEncoder",
    "load_initial_state_from_fixture",
    "OpponentProvider",
    "RandomOpponentProvider",
    "ReplayDBOpponentProvider",
    "ReplaySnapshotProvider",
    "SelfPlayPoolProvider",
    "StaticTurnOpponentProvider",
    "CurriculumOpponentProvider",
    "build_opponent_provider",
    "build_training_env",
    "SNAPSHOT_VERSION",
    "load_snapshot",
    "save_snapshot",
]
