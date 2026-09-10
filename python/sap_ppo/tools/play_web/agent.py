"""exp16 W3: the real agent behind the duel, built lazily.

WHAT THIS BUILDS. The exp13 baseline, which is exp12's champion arm C:

    BcRecommender(an internal dataset path)
      -> the chain proposer (one whole-turn chain per candidate)
    VGameLeafScorer.from_checkpoints([an internal dataset path],
                                     <the same BC checkpoint as extractor>,
                                     blend=0.0, pessimism=0.0)
      -> the learned leaf that replaced the Monte-Carlo rollout
    SearchRecommender(..., scoring="vgame", ksim=16, max_turn=30,
                      turn_mode="segmented-honest", stochastic_samples=3)
      -> best-of-N over the proposer, on exp13 Amendment A2's structural frame

The V heads were trained with that BC checkpoint's feature extractor, so the
extractor path is the BC checkpoint by default and `from_checkpoints` pins
the sha of what it actually loaded; a mismatch raises there rather than
scoring quietly wrong boards.

WHY TORCH IS IMPORTED INSIDE THE FUNCTION. play-web must start, serve
`/sandbox` and answer `/api/state` on a box with no checkpoints and no
torch: exp14's browser gate runs against that surface, and the sandbox is
Ruihan's manual inspection page. So this module imports nothing heavier than
the standard library at module scope, and every torch/sb3 import lives
inside `build_agent`. `python/tests/test_exp16_agent.py` constructs the app
with no agent and hits `/api/state` to keep that true.

WHY ONE TORCH THREAD. W0 measured width-72 decisions at 4.70 / 4.73 / 4.78 /
4.74 s on 1 / 4 / 8 / 16 threads -- 1.5% across the whole range, because the
cost is hundreds of small BC forwards, not one big matmul. One thread is
therefore free, and it is the only setting under which two decodes of the
same state agree bit for bit, which W3's "same seeds reproduce the game" and
W6's replay archive both depend on.
"""

from __future__ import annotations

import copy

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# exp13's honest-frame mode enum and k. Imported at module scope deliberately:
# `honest_frame` is pure stdlib, so it costs nothing and it means the
# defaults below cannot drift from exp13's own constants.
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

# The exp13 baseline == exp12 champion arm C. Paths are relative to the repo
# root (where `data` is the symlink to the data disk), matching every other
# tool in the tree.
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
    """Every knob that changes what the agent PLAYS, in one place.

    Anything here that differs between two games makes them two different
    agents, which is why `AgentHandle.describe()` echoes the whole thing
    into the archive (W6) rather than a hand-picked subset.
    """

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
    # exp16 A3. What fills the rest of the turn behind a chance node while the
    # search is still ranking. `bc_greedy` + width 1 is every run before
    # 2026-08-19 and is the default HERE ON PURPOSE, so every caller that does
    # not ask for deepening keeps a byte-identical agent; the deployed :8766
    # command line is what turns it on. `search_recommender` owns the
    # validation (the two fields are coupled: v_search needs width >= 2,
    # bc_greedy requires width == 1) and refuses a bad pair loudly.
    completion_policy: str = "bc_greedy"
    completion_width: int = 1
    # exp16 W0. What a SAMPLED completion does when its walk would give up
    # before the turn is over. Default is the behaviour every run before
    # 2026-08-20 had; `search_recommender` owns the definitions and the
    # measurement that motivates changing it.
    #: W0, 2026-08-21. `resample` rather than `stop`: see
    #: `search_recommender.COMPLETION_TAIL_STOP` for the ruling and the three
    #: readings. `stop` reproduces every result measured before that date.
    completion_tail: str = "resample"
    torch_threads: int = DEFAULT_TORCH_THREADS
    # exp13 W0b': skip schema validation on IMAGINED walks only (BC decode,
    # candidate replay, prefix walks, resample completions). Measured 2.7x
    # there; the committed ops this line plays go through `api.step` and
    # cannot inherit it. Off by default, on for interactive play.
    skip_imagined_validation: bool = False

    def __post_init__(self) -> None:
        expected_id = f"arm-c-{self.model_revision}-{self.search_revision}"
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
    """Interactive defaults; AgentConfig keeps the historical research defaults.

    Model paths can be overridden by the operator. The archive records their
    actual hashes; the generic identity below does not assert a checkpoint hash.
    Matching these budgets alone does not make a timed duel an arena benchmark.
    """
    values: dict[str, Any] = {
        "agent_id": "arm-c-configured-interactive",
        "agent_name": "BC + V (checkpoint hashes in archive)",
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
    """The commit that introduced exp13's honest frame, as the archive's
    provenance for WHICH segmented driver played the game.

    Read from git rather than hard-coded, so a rebase cannot leave the
    archive pointing at a commit that is not in this history.
    """
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
        """The `ai_version` block exp16 W6 archives with every game.

        Only the EXPENSIVE half is cached (three checkpoint digests plus two
        git lookups), keyed on the paths it was computed from; everything
        derived from `config` is rebuilt on every call.

        The split is not an optimisation, it is what keeps the block honest
        after `DuelApp` installs a new config per game (Amendment 6). Caching
        the whole block, which is what this did until 2026-08-20, would keep
        serving the completion policy the FIRST game of the process ran under.
        That block is exactly what the archive records, so every later game
        would carry a wrong number, and nothing else on the page contradicts
        it -- the deepening line reads the live config, not the archive.
        """
        cfg = self.config
        digest_key = (cfg.bc_checkpoint, tuple(cfg.vgame_heads), cfg.extractor_path)
        if self._digest_key != digest_key:
            self._digests = {
                "bc_checkpoint": {
                    "path": cfg.bc_checkpoint,
                    "sha256": _sha256(self.repo_root / cfg.bc_checkpoint),
                },
                "vgame_heads": [
                    {"path": p, "sha256": _sha256(self.repo_root / p)}
                    for p in cfg.vgame_heads
                ],
                "vgame_extractor": {
                    "path": cfg.extractor_path,
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
            # A1 ruling 1's key shape, spelled out so an archive says
            # what stream imagination ran on without reading the code.
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

    # W0: 1/4/8/16 threads are within 1.5% of each other and 1 is the only
    # bit-reproducible one. See the module docstring.
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
    # Imported here, not at module scope, for the same reason `build_agent`
    # keeps its heavy imports inside: importing this module must not pull in
    # torch, or `/sandbox` and `/api/state` stop working on a box with no
    # checkpoints (exp16 PLAN risk 7).
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
    """A SECOND, independent (BC, V) pair from the same pinned checkpoints.

    For exp16's value readout (`play_web/value_probe.py`), which answers HTTP
    requests on the server's own threads while the AI worker is searching on
    its own.

    WHY A SECOND COPY RATHER THAN THE AGENT'S. `SearchRecommender.
    _call_bc_recommend` temporarily mutates the shared `BcRecommender`'s
    `deterministic` and `decode_mode` and restores them in a `finally`, and both
    objects carry per-call torch state; calling `bc.recommend` on the same
    instance from another thread is a race on the object the game is being
    played with. This was MEASURED, not reasoned about: with the readout sharing
    the agent's models, one seeded duel panel played three different games
    (an internal analysis script, two `on_probed`
    runs and `off_a`), and the duel's whole archive story is that the same seeds
    reproduce the game. With an independent pair the same panel reproduces.

    The price is one extra checkpoint load (~1.6 s, once) and the resident
    memory of a second copy of the same weights. The alternative -- taking a
    lock the AI worker holds -- would make the readout unavailable for the whole
    of a 105 s turn, which is exactly when the human is shopping and wants it.

    Same paths, so the same sha256s: `AgentHandle.describe()` stays the honest
    identity of what produced the number.
    """
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
