"""The LEARNED search leaf: `--search-scoring vgame` (exp12 W2, run in A4).

Spec: PLAN.md W2 ("`--search-scoring vgame` in SearchRecommender: one batched
V forward over deduped end boards, score = (1-blend)*V + blend*myopic -
pessimism*ensemble_std; never-raise contract kept; turn-1 skip kept for
comparability") and the A4 amendment that deploys it.

This module owns everything the leaf needs that `SearchRecommender` must not
grow: the checkpoint load, the encoder, the bypass block and the score
formula. `SearchRecommender` only ever calls `score_boards`, so it keeps its
torch-free import surface and this file is the single place the SERVE
CONTRACT lives.

## The serve contract (RESULTS_W1 findings 9, 11 and 14, enforced here)

1. The encoder input is the candidate's END BOARD as-is, which is exactly
   what A2 stored (`build_teacher_distill_dataset.py` encodes the recorded
   afterstate AS STORED) and exactly what `search_recommender._apply_chain`
   hands scoring. No patch step exists and none may be added.
2. The race scalars do NOT come from that board. They come through the 4-dim
   BYPASS block as the DRIVER-TRUE engine values at decision time:
   `[turn/15, lives/6, opp_lives/6, wins/15]` via
   `vgame_model.bypass_features`, the single place that order and those
   normalizers live. Finding 11 is why the bypass exists at all (the frozen
   extractor was trained while `lives` was a constant 6, so its weights on
   the race dims are untrained noise), and finding 9 is the instruction that
   a scorer must feed it TRUE values. `wins` is the only one not readable off
   a board, so the driver supplies it (`SearchRecommender.set_race_context`);
   a scorer that is never told it refuses to score rather than guessing 0.
3. The encoding is fingerprint-checked at construction against the encoder
   the heads were TRAINED with (`metadata["encoder"]["vocab_fingerprint"]`
   and `size`), so an `observation.py` change that alters the vocabulary
   fails loudly here instead of silently serving a different feature space.
4. The extractor checkpoint is sha256-pinned by `load_vgame_model` for both
   artifact kinds, and the heads file's own sha256 is reported so a run's
   report says on its face which weights played it.

## What V means, and why that is checked rather than assumed

A route-a artifact (`metadata["target"] == "teacher_score"`) does NOT emit a
win probability: its `p_win` head carries the AFFINELY SQUASHED teacher
score, so the leaf value is `unsquash_score(p_win)` and lives on the
teacher's own `[-0.06, 1.06]` scale. An outcome-trained artifact (W1/W1'c)
emits a real probability and is used as-is. An exp13 W1c artifact
(`metadata["target"] == "bellman_trophies"`) carries a BOUNDED TROPHY VALUE,
so the leaf is `10 * p_win` on `[0, 10]` -- `PLAN_W1.md` §Operator changed the
head's link when the reward became expected trophies. An exp22 W1 artifact
(`metadata["target"] == "mc8_leafmap_trophies"`) is the fourth: its head was
CARRIED from V0 rather than retrained, so its logit still speaks the squashed
teacher score, and a FROZEN monotone leaf map carried in the artifact's own
metadata lifts that onto trophies. The map is applied per board, before any
averaging, because `f(mean) != mean(f)` and that difference was measured to
flip 15.61% of argmaxes. The artifact says which, this module branches on it,
and `describe()` reports the branch it took.

The four scales are an order of magnitude apart, and every one of them is a
plausible small number, so an artifact whose target string this module does
not recognise is REFUSED at construction rather than served on a default.
Serving a W1c head as if it were a probability would put a V1 arena arm on a
tenth of its own scale and the paired gate would read as a catastrophic
regression that never happened.

## The blend and the pessimism knobs

`score = (1 - blend) * V + blend * myopic - pessimism * ensemble_std`, the
formula frozen in W2, with both knobs defaulting to 0 so the deployed arm is
a pure V leaf.

Two properties of that formula are stated rather than smoothed over. First,
`V` and `myopic` are on DIFFERENT scales for a route-a artifact (teacher
score in [-0.06, 1.06] against `(playerWins - opponentWins) / ksim` in
[-1, 1]), so a nonzero blend mixes two units; the formula is frozen, so it is
implemented as written and this is documented, not silently rescaled.
Second, `ensemble_std` is only defined for an ensemble, so `pessimism > 0`
with a single member is rejected at CONSTRUCTION rather than being a no-op
that quietly turns a pessimism run into a plain run.

Turning `blend` on is also what makes the stage-1 myopic oracle calls
necessary at all: at `blend == 0` the leaf never reads them, so
`SearchRecommender` skips them entirely, which is where the cost saving the
W2 rule measures actually comes from. `needs_myopic` is how that is
communicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch as th

from ..train.observation import OBSERVATION_MODE_V4, build_state_encoder
from ..train.vgame_model import bypass_features, file_sha256, load_vgame_model

DEFAULT_BLEND = 0.0
DEFAULT_PESSIMISM = 0.0
from .._artifact_defaults import VGAME_EXTRACTOR as DEFAULT_EXTRACTOR  # noqa: E402

TARGET_TEACHER_SCORE = "teacher_score"
TARGET_GAME_OUTCOME = "game_outcome"
TARGET_BELLMAN_TROPHIES = "bellman_trophies"
# exp22 W1: a head CARRIED from V0 and fine-tuned on MC8. Its logit still
# speaks V0's squashed teacher score, and a frozen monotone leaf map lifts
# that onto trophies (`19-trophy-levers/PLAN_W1.md` §3d case 2). It is a
# distinct target string precisely because serving it as `bellman_trophies`
# would apply `unsquash_trophies` to a teacher score, which is off by the
# whole map and is exactly the silent units error this table exists to stop.
TARGET_MC8_LEAFMAP = "mc8_leafmap_trophies"

# `metadata["target"]` -> which map turns the head's probability into a leaf
# value. A target string absent from here is refused rather than defaulted:
# see the module docstring's "What V means".
VALUE_KINDS: dict[str, str] = {
    TARGET_GAME_OUTCOME: "probability",
    TARGET_TEACHER_SCORE: "squashed_teacher_score",
    TARGET_BELLMAN_TROPHIES: "bounded_trophies",
    TARGET_MC8_LEAFMAP: "leafmap_trophies",
}
VALUE_KIND_PROBABILITY = "probability"
VALUE_KIND_TEACHER = "squashed_teacher_score"
VALUE_KIND_TROPHIES = "bounded_trophies"
VALUE_KIND_LEAFMAP = "leafmap_trophies"

# The value kinds that need a leaf map carried in the artifact's metadata.
VALUE_KINDS_NEEDING_LEAF_MAP = frozenset({VALUE_KIND_LEAFMAP})


def check_encoder_pin(encoder: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    """The serve contract's encoder pin (module docstring, item 3).

    Compares the LIVE encoder's vocabulary fingerprint and observation size
    against the ones recorded in the heads artifact's own metadata, and
    raises rather than serving a different feature space than the heads were
    trained on. Split out of `from_checkpoints` so it is testable without a
    real BC checkpoint on disk.
    """
    pinned_fp = str(cfg.get("vocab_fingerprint") or "").strip()
    live_fp = str(getattr(encoder, "vocab_fingerprint", "") or "").strip()
    if pinned_fp and live_fp and pinned_fp != live_fp:
        raise RuntimeError(
            f"vgame_scorer_encoder_fingerprint_mismatch:pinned={pinned_fp}:live={live_fp}")
    pinned_size = cfg.get("size")
    live_size = int(getattr(encoder, "size", 0) or 0)
    if pinned_size is not None and live_size and int(pinned_size) != live_size:
        raise RuntimeError(
            f"vgame_scorer_encoder_size_mismatch:pinned={pinned_size}:live={live_size}")
    return {"fingerprint": live_fp, "size": live_size, "size_pinned": pinned_size}


class VGameLeafScorer:
    """One batched V forward over a decision's deduped candidate end boards.

    Constructed by `from_checkpoints`; `SearchRecommender` only calls
    `score_boards` and `describe`.
    """

    def __init__(
        self,
        models: Sequence[Any],
        encoder: Any,
        *,
        blend: float = DEFAULT_BLEND,
        pessimism: float = DEFAULT_PESSIMISM,
        value_is_squashed_teacher_score: Optional[bool] = None,
        value_kind: Optional[str] = None,
        leaf_map: Optional[Any] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        self.models = list(models)
        if not self.models:
            raise ValueError("vgame_scorer_no_models")
        self.encoder = encoder
        self.blend = float(blend)
        self.pessimism = float(pessimism)
        if self.pessimism != 0.0 and len(self.models) < 2:
            raise ValueError(
                f"vgame_scorer_pessimism_without_ensemble:members={len(self.models)}:"
                f"pessimism={self.pessimism} (ensemble_std is identically 0 with one "
                f"member, so the knob would be a silent no-op)")
        # Two ways in, one field out. `value_kind` is the current spelling and
        # the boolean is what every pre-W1c caller passes; giving both is only
        # allowed when they agree, so a caller cannot half-migrate and end up
        # serving on a scale it did not ask for.
        if value_kind is None and value_is_squashed_teacher_score is None:
            raise ValueError("vgame_scorer_value_kind_unset")
        derived = (
            VALUE_KIND_TEACHER if value_is_squashed_teacher_score else VALUE_KIND_PROBABILITY
        ) if value_is_squashed_teacher_score is not None else None
        if value_kind is not None and derived is not None and value_kind != derived:
            raise ValueError(
                f"vgame_scorer_value_kind_conflict:{value_kind}:{derived}"
            )
        self.value_kind = str(value_kind or derived)
        if self.value_kind not in set(VALUE_KINDS.values()):
            raise ValueError(f"vgame_scorer_unknown_value_kind:{self.value_kind}")
        self.value_is_squashed_teacher_score = self.value_kind == VALUE_KIND_TEACHER
        # A leaf-mapped head without its map is not servable at all: the map is
        # part of the model, not a decoration on it, so this refuses rather
        # than falling back to the identity, which would serve a teacher score
        # where a trophy count is expected.
        self.leaf_map = leaf_map
        if self.value_kind in VALUE_KINDS_NEEDING_LEAF_MAP and self.leaf_map is None:
            raise ValueError(
                f"vgame_scorer_missing_leaf_map:{self.value_kind}. The artifact's "
                "metadata must carry a 'leaf_map' block; see exp22_w1_leafmap.")
        if self.value_kind not in VALUE_KINDS_NEEDING_LEAF_MAP and self.leaf_map is not None:
            raise ValueError(
                f"vgame_scorer_unexpected_leaf_map:{self.value_kind}")
        self.meta = dict(meta or {})
        self.n_scored_calls = 0
        self.n_boards_scored = 0

    # -- construction ------------------------------------------------------

    @classmethod
    def from_checkpoints(
        cls,
        heads_paths: Sequence[str | Path],
        extractor_path: str | Path,
        *,
        blend: float = DEFAULT_BLEND,
        pessimism: float = DEFAULT_PESSIMISM,
    ) -> "VGameLeafScorer":
        """Strict load of every head artifact + one shared encoder.

        `load_vgame_model` does the strict extractor rebuild, the strict
        heads load and the extractor-sha pin for both artifact kinds; this
        adds the encoder fingerprint check and the target branch.
        """
        heads_paths = [Path(p) for p in heads_paths]
        if not heads_paths:
            raise ValueError("vgame_scorer_no_heads_paths")

        models: list[Any] = []
        reports: list[dict[str, Any]] = []
        encoder_cfgs: list[dict[str, Any]] = []
        targets: set[str] = set()
        leaf_map_shas: set[str] = set()
        leaf_map_blocks: list[dict[str, Any]] = []
        for path in heads_paths:
            enc_cfg = cls._encoder_cfg(path)
            model, report = load_vgame_model(
                path, extractor_path,
                observation_mode=str(enc_cfg.get("observation_mode") or OBSERVATION_MODE_V4),
                max_turn=int(enc_cfg.get("max_turn") or 15))
            heads_meta = report.get("metadata") or {}
            targets.add(str(heads_meta.get("target") or "game_outcome"))
            block = heads_meta.get("leaf_map")
            if block:
                leaf_map_blocks.append(dict(block))
                leaf_map_shas.add(str(block.get("sha256") or ""))
            models.append(model)
            reports.append(report)
            encoder_cfgs.append(enc_cfg)

        if len({(c.get("observation_mode"), c.get("max_turn"), c.get("vocab_fingerprint"))
                for c in encoder_cfgs}) != 1:
            raise RuntimeError(f"vgame_scorer_ensemble_encoder_disagreement:{encoder_cfgs}")
        if len(targets) != 1:
            raise RuntimeError(f"vgame_scorer_ensemble_target_disagreement:{sorted(targets)}")

        cfg = encoder_cfgs[0]
        observation_mode = str(cfg.get("observation_mode") or OBSERVATION_MODE_V4)
        max_turn = int(cfg.get("max_turn") or 15)
        encoder = build_state_encoder(observation_mode=observation_mode, max_turn=max_turn)

        pin = check_encoder_pin(encoder, cfg)
        live_fp, live_size, pinned_size = pin["fingerprint"], pin["size"], pin["size_pinned"]

        target = sorted(targets)[0]
        if target not in VALUE_KINDS:
            raise RuntimeError(
                f"vgame_scorer_unknown_target:{target}:known={sorted(VALUE_KINDS)}. "
                "The head-output-to-leaf-value map is not guessable from the "
                "numbers: a probability, a squashed teacher score and a bounded "
                "trophy value are all plausible small floats on different scales."
            )
        leaf_map = None
        if VALUE_KINDS[target] in VALUE_KINDS_NEEDING_LEAF_MAP:
            from .exp22_w1_leafmap import LeafMap

            if not leaf_map_blocks:
                raise RuntimeError(
                    f"vgame_scorer_leaf_map_missing_in_artifact:{target}. The head "
                    "declares a leaf-mapped target but carries no 'leaf_map' block.")
            if len(leaf_map_shas) != 1:
                raise RuntimeError(
                    f"vgame_scorer_ensemble_leaf_map_disagreement:{sorted(leaf_map_shas)}. "
                    "Ensemble members were fitted against different maps, so their "
                    "leaf values are not on one scale and averaging them is meaningless.")
            leaf_map = LeafMap.from_artifact(leaf_map_blocks[0])

        meta = {
            "heads": [str(p) for p in heads_paths],
            "heads_sha256": [file_sha256(p) for p in heads_paths],
            "extractor": str(extractor_path),
            "extractor_sha256": file_sha256(extractor_path),
            "artifact_modes": [str(r.get("mode")) for r in reports],
            "arms": [str((r.get("metadata") or {}).get("arm")) for r in reports],
            "target": target,
            "leaf_map": (leaf_map.describe() if leaf_map is not None else None),
            "encoder": {"observation_mode": observation_mode, "max_turn": max_turn,
                        "vocab_fingerprint": live_fp, "size": live_size,
                        "size_pinned": pinned_size},
            "ensemble_size": len(models),
        }
        return cls(models, encoder, blend=blend, pessimism=pessimism,
                   value_kind=VALUE_KINDS[target], leaf_map=leaf_map, meta=meta)

    @staticmethod
    def _encoder_cfg(heads_path: Path) -> dict[str, Any]:
        payload = th.load(str(heads_path), map_location="cpu", weights_only=False)
        return dict(((payload.get("metadata") or {}).get("encoder") or {}))

    # -- scoring -----------------------------------------------------------

    @property
    def needs_myopic(self) -> bool:
        """Whether the stage-1 myopic oracle scores are read by the formula.
        False at `blend == 0`, which is what lets `SearchRecommender` skip
        the oracle calls entirely."""
        return self.blend != 0.0

    @th.no_grad()
    def score_boards(
        self,
        boards: Sequence[dict[str, Any]],
        *,
        turn: int,
        lives: int,
        opponent_lives: int,
        wins: int,
        myopic_scores: Optional[Sequence[Optional[float]]] = None,
    ) -> dict[str, Any]:
        """ONE batched forward over `boards`; returns the leaf scores.

        The race scalars are the DECISION's own driver-true values and are
        therefore identical for every candidate, which is exactly how A2
        stored them (one `race` block per decision, repeated per candidate
        row). Raises on any failure; the caller owns the never-raise
        contract and turns a raise into a skipped search.
        """
        n = len(boards)
        if n == 0:
            raise ValueError("vgame_scorer_no_boards")
        if wins is None:
            raise ValueError("vgame_scorer_wins_unset")

        x = np.asarray([self.encoder.encode(b) for b in boards], dtype=np.float32)
        obs = th.as_tensor(x, dtype=th.float32)
        bypass = bypass_features(
            np.full(n, int(turn), dtype=np.int64),
            np.full(n, int(lives), dtype=np.int64),
            np.full(n, int(opponent_lives), dtype=np.int64),
            np.full(n, int(wins), dtype=np.int64),
        )

        per_model: list[np.ndarray] = []
        for model in self.models:
            p = model(obs, bypass)["p_win"].numpy().astype(np.float64)
            per_model.append(self._to_value(p))
        stack = np.stack(per_model, axis=0)
        v = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=0) if stack.shape[0] > 1 else np.zeros(n, dtype=np.float64)

        if self.needs_myopic:
            if myopic_scores is None or any(m is None for m in myopic_scores):
                raise ValueError("vgame_scorer_blend_needs_myopic_scores")
            myopic = np.asarray([float(m) for m in myopic_scores], dtype=np.float64)
            score = (1.0 - self.blend) * v + self.blend * myopic
        else:
            myopic = None
            score = v.copy()
        if self.pessimism:
            score = score - self.pessimism * std

        self.n_scored_calls += 1
        self.n_boards_scored += n
        return {
            "score": [float(s) for s in score],
            "v": [float(x_) for x_ in v],
            "ensemble_std": [float(s) for s in std],
            "myopic": (None if myopic is None else [float(m) for m in myopic]),
            "n_boards": n,
        }

    def _to_value(self, p_win: np.ndarray) -> np.ndarray:
        """Head output -> leaf value (module docstring, "What V means")."""
        if self.value_kind == VALUE_KIND_PROBABILITY:
            return p_win
        if self.value_kind == VALUE_KIND_TEACHER:
            from ..train.vdistill import unsquash_score

            return np.asarray(unsquash_score(p_win), dtype=np.float64)
        if self.value_kind == VALUE_KIND_TROPHIES:
            from ..train.w1_value_head import unsquash_trophies

            return np.asarray(unsquash_trophies(p_win), dtype=np.float64)
        if self.value_kind == VALUE_KIND_LEAFMAP:
            from ..train.vdistill import unsquash_score

            # V0's units first, then the frozen monotone map. The order is the
            # arm's definition, not an implementation choice: the map was
            # fitted from `v0_score` to the label, so it must be handed a
            # `v0_score`. Applied PER BOARD, before any averaging the search
            # does, because `f(mean) != mean(f)` and the argmax flip rate that
            # difference produces was measured at 15.61%.
            return np.asarray(
                self.leaf_map(np.asarray(unsquash_score(p_win), dtype=np.float64)),
                dtype=np.float64,
            )
        raise RuntimeError(f"vgame_scorer_unknown_value_kind:{self.value_kind}")

    # -- telemetry ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """The block every A4 report echoes, so a number says on its face
        which weights and which knobs produced it."""
        out = dict(self.meta)
        out.update({
            "blend": self.blend,
            "pessimism": self.pessimism,
            "needs_myopic": self.needs_myopic,
            "value_kind": self.value_kind,
            "value_is_squashed_teacher_score": self.value_is_squashed_teacher_score,
            "leaf_map": (self.leaf_map.describe() if self.leaf_map is not None else None),
            "scored_calls": int(self.n_scored_calls),
            "boards_scored": int(self.n_boards_scored),
        })
        return out
