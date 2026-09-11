"""Load the BC proposer and compatible value model for the local demo.

Model imports and checkpoint loads are deferred until needed. The value model
checks the BC feature-extractor hash before scoring states."""

from __future__ import annotations

import copy

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from ..honest_frame import (
    DEFAULT_STOCHASTIC_SAMPLES,
    TURN_MODE_SEGMENTED_HONEST,
)
from .agent_identity import (
    CURRENT_AGENT_ID,
    CURRENT_AGENT_NAME,
    CURRENT_BASELINE_ID,
    CURRENT_SEARCH_REVISION,
    MODEL_REVISION_R0,
)


from ..._artifact_defaults import (  # noqa: E402
    DUEL_VGAME_HEADS,
    PLAY_WEB_BC_CHECKPOINT as DEFAULT_BC_CHECKPOINT,
    VGAME_HEADS as DEFAULT_VGAME_HEADS,
)
DEFAULT_WIDTH = 72
DEFAULT_KSIM = 16
DEFAULT_MAX_TURN = 30
DEFAULT_TORCH_THREADS = 1


@dataclass(frozen=True)
class AgentConfig:
    """Every knob that changes what the agent PLAYS, in one place."""

    agent_id: str = CURRENT_AGENT_ID
    agent_name: str = CURRENT_AGENT_NAME
    model_revision: str = MODEL_REVISION_R0
    search_revision: str = CURRENT_SEARCH_REVISION
    baseline_id: str = CURRENT_BASELINE_ID
    bc_checkpoint: str = DEFAULT_BC_CHECKPOINT
    vgame_heads: tuple[str, ...] = DEFAULT_VGAME_HEADS
    # None => the BC checkpoint, which is what the heads were trained on.
    vgame_extractor: str | None = None
    blend: float = 0.0
    pessimism: float = 0.0
    width: int = DEFAULT_WIDTH
    ksim: int = DEFAULT_KSIM
    max_turn: int = DEFAULT_MAX_TURN
    seed: int = 0
    turn_mode: str = TURN_MODE_SEGMENTED_HONEST
    stochastic_samples: int = DEFAULT_STOCHASTIC_SAMPLES


    completion_policy: str = "bc_greedy"
    completion_width: int = 1


    completion_tail: str = "resample"
    torch_threads: int = DEFAULT_TORCH_THREADS


    skip_imagined_validation: bool = False

    def __post_init__(self) -> None:
        expected_id = f"bc-value-{self.model_revision}-{self.search_revision}"
        if self.agent_id != expected_id or not self.agent_name or not self.baseline_id:
            raise ValueError("agent_identity_inconsistent")
        if self.agent_id == CURRENT_AGENT_ID:
            if (
                self.agent_name != CURRENT_AGENT_NAME
                or self.model_revision != MODEL_REVISION_R0
                or self.search_revision != CURRENT_SEARCH_REVISION
                or self.baseline_id != CURRENT_BASELINE_ID
            ):
                raise ValueError("current_agent_identity_inconsistent")
            if (
                self.bc_checkpoint != DEFAULT_BC_CHECKPOINT
                or tuple(self.vgame_heads) != DEFAULT_VGAME_HEADS
                or self.extractor_path != DEFAULT_BC_CHECKPOINT
            ):
                raise ValueError(
                    "current_agent_identity_requires_pinned_r0_weights"
                )

    @property
    def extractor_path(self) -> str:
        return str(self.vgame_extractor or self.bc_checkpoint)


def demo_agent_config(**overrides: Any) -> AgentConfig:
    """Interactive defaults with configurable model paths.

    Model paths can be overridden by the operator. The archive records their
    actual hashes; the generic identity below does not assert a checkpoint hash.
    Matching these budgets alone does not make a timed duel an arena benchmark.
    """
    values: dict[str, Any] = {
        "agent_id": "bc-value-configured-interactive",
        "agent_name": "SAP-Arena-AI",
        "model_revision": "configured",
        "search_revision": "interactive",
        "baseline_id": "interactive-configured-budget",
        "vgame_heads": DUEL_VGAME_HEADS,
        "width": 72,
        "completion_policy": "v_search",
        "completion_width": 4,
        "stochastic_samples": 12,
        "skip_imagined_validation": True,
    }
    values.update(overrides)
    if int(values["stochastic_samples"]) < 1:
        raise ValueError("stochastic_samples must be positive")
    return AgentConfig(**values)


def _sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    sha = out.stdout.strip()
    return sha or None


def _honest_driver_commit(repo_root: Path) -> str | None:
    """Read from git rather than hard-coded, so a rebase cannot leave the
    archive pointing at a commit that is not in this history."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", "python/sap_ppo/tools/honest_frame.py"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    sha = out.stdout.strip()
    return sha or None


@dataclass
class AgentHandle:
    """The built agent plus the block that identifies it in an archive."""

    config: AgentConfig
    bc: Any
    search: Any
    vgame_scorer: Any
    repo_root: Path
    _digests: dict[str, Any] = field(default_factory=dict)
    _digest_key: tuple[Any, ...] | None = None

    def describe(self) -> dict[str, Any]:
        """Only the EXPENSIVE half is cached (three checkpoint digests plus two
        git lookups), keyed on the paths it was computed from; everything
        derived from `config` is rebuilt on every call."""
        cfg = self.config
        digest_key = (cfg.bc_checkpoint, tuple(cfg.vgame_heads), cfg.extractor_path)
        if self._digest_key != digest_key:
            self._digests = {
                "bc_checkpoint": {
                    "path": Path(cfg.bc_checkpoint).name,
                    "sha256": _sha256(self.repo_root / cfg.bc_checkpoint),
                },
                "vgame_heads": [
                    {"path": Path(p).name, "sha256": _sha256(self.repo_root / p)}
                    for p in cfg.vgame_heads
                ],
                "vgame_extractor": {
                    "path": Path(cfg.extractor_path).name,
                    "sha256": _sha256(self.repo_root / cfg.extractor_path),
                },
                "honest_driver_commit": _honest_driver_commit(self.repo_root),
                "git_sha": _git_sha(self.repo_root),
            }
            self._digest_key = digest_key
        description = {
            "agent": cfg.agent_name,
            "agent_id": cfg.agent_id,
            "agent_name": cfg.agent_name,
            "model_revision": cfg.model_revision,
            "search_revision": cfg.search_revision,
            "baseline_id": cfg.baseline_id,
            "inference_time_revision": "fixed-clock-v1",
            "search": {
                "scoring": "vgame",
                "width": int(cfg.width),
                "ksim": int(cfg.ksim),
                "max_turn": int(cfg.max_turn),
                "seed": int(cfg.seed),
                "blend": float(cfg.blend),
                "pessimism": float(cfg.pessimism),
            },
            "turn_mode": cfg.turn_mode,
            "stochastic_samples": int(cfg.stochastic_samples),
            "completion_policy": str(cfg.completion_policy),
            "completion_width": int(cfg.completion_width),
            "completion_tail": str(cfg.completion_tail),
            # Record the independent simulation stream's key structure.
            "imagination_key": "sha256(engine_seed, turn, segment_index, sample_r)",
            "torch_threads": int(cfg.torch_threads),
            "skip_imagined_validation": bool(cfg.skip_imagined_validation),
        }
        description.update(copy.deepcopy(self._digests))
        return description


def repo_root_from_here() -> Path:
    """`<repo>/python/sap_ppo/tools/play_web/agent.py` -> `<repo>`."""
    return Path(__file__).resolve().parents[4]


def build_agent(cfg: AgentConfig | None = None, *, repo_root: Path | None = None) -> AgentHandle:
    """Load the checkpoints and wire the honest-frame search over them.

    Every heavy import is INSIDE this function. Calling it is the only thing
    in play-web that needs torch; nothing else in the package does.
    """
    cfg = cfg or AgentConfig()
    root = Path(repo_root) if repo_root is not None else repo_root_from_here()

    import torch


    torch.set_num_threads(int(cfg.torch_threads))

    from ..bc_recommender import BcRecommender
    from ..search_recommender import SearchRecommender
    from ..vgame_scorer import VGameLeafScorer

    if cfg.skip_imagined_validation:
        from ...api import set_skip_imagined_validation

        set_skip_imagined_validation(True)

    # No `max_turn` here on purpose. `BcRecommender.max_turn` is the
    # OBSERVATION ENCODER's turn normaliser (15, the value the checkpoint
    # was trained under and the value `VGameLeafScorer` pins its own encoder
    # to); `SearchRecommender.max_turn` below is the game-length cap the
    # rollout leaf uses (30). Same word, two quantities -- the driver
    # likewise leaves this one at its default and only passes the other.
    bc = BcRecommender(str(root / cfg.bc_checkpoint))
    scorer = VGameLeafScorer.from_checkpoints(
        [str(root / p) for p in cfg.vgame_heads],
        str(root / cfg.extractor_path),
        blend=float(cfg.blend),
        pessimism=float(cfg.pessimism),
    )
    search = build_search(bc, scorer, cfg)
    return AgentHandle(config=cfg, bc=bc, search=search, vgame_scorer=scorer, repo_root=root)


def build_search(bc: Any, scorer: Any, cfg: AgentConfig) -> Any:
    """The honest-frame search over an ALREADY-LOADED (bc, scorer) pair.

    Split out of `build_agent` so a per-game settings change can rebuild just
    this object without touching torch or re-reading a checkpoint. Building a
    fresh one rather than assigning onto the live one is deliberate: the AI
    worker dereferences `agent.search` several times inside a single turn, so
    a half-updated recommender is a state no setting describes -- the same
    reasoning `DuelApp.set_config` gives for swapping a whole `AiTurnConfig`.

    All validation of the (policy, width) pair lives in `SearchRecommender`'s
    constructor, so this is also the only place a caller needs to go through
    to have a candidate setting checked.
    """


    from ..search_recommender import SearchRecommender

    return SearchRecommender(
        bc,
        n_candidates=int(cfg.width),
        ksim=int(cfg.ksim),
        seed=int(cfg.seed),
        scoring="vgame",
        max_turn=int(cfg.max_turn),
        vgame_scorer=scorer,
        turn_mode=cfg.turn_mode,
        stochastic_samples=int(cfg.stochastic_samples),
        completion_policy=str(cfg.completion_policy),
        completion_width=int(cfg.completion_width),
        completion_tail=str(cfg.completion_tail),
    )


def build_readonly_models(
    cfg: AgentConfig, *, repo_root: Path | None = None
) -> tuple[Any, Any]:
    """Load a separate BC/value pair for read-only board probes.

The search worker changes per-call decoder settings, so sharing its objects
with a concurrent probe would introduce a race. Independent models keep probe
requests from altering the search. This costs another copy of the same weights."""
    root = Path(repo_root) if repo_root is not None else repo_root_from_here()

    import torch

    torch.set_num_threads(int(cfg.torch_threads))

    from ..bc_recommender import BcRecommender
    from ..vgame_scorer import VGameLeafScorer

    bc = BcRecommender(str(root / cfg.bc_checkpoint))
    scorer = VGameLeafScorer.from_checkpoints(
        [str(root / p) for p in cfg.vgame_heads],
        str(root / cfg.extractor_path),
        blend=float(cfg.blend),
        pessimism=float(cfg.pessimism),
    )
    return bc, scorer
