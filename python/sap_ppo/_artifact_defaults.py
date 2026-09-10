"""Default locations of trained artifacts, public-release edition.

The private tree keeps these pointing at its own experiment directories. Here
they point at a plain `models/` directory, because the published weights are
distributed separately rather than living in this repository. Every value can be
overridden by environment variable.

See the README for where to download the weights.
"""
from __future__ import annotations

import os

BC_CHECKPOINT = os.environ.get("SAP_BC_CHECKPOINT", "models/bc_attn_v4.zip")
PLAY_WEB_BC_CHECKPOINT = os.environ.get(
    "SAP_PLAY_WEB_BC_CHECKPOINT", "models/bc_attn_v4.zip")
VGAME_EXTRACTOR = os.environ.get("SAP_VGAME_EXTRACTOR", "models/bc_attn_v4.zip")
# The shipped configuration's value heads. NOT `vgame_heads.pt`: that file
# declares `target = teacher_score` and this one declares
# `target = mc8_leafmap_trophies`, which is a different value scale, not a newer
# version of the same one. See MODELS.md.
VGAME_HEADS: tuple[str, ...] = tuple(
    p for p in os.environ.get(
        "SAP_VGAME_HEADS", "models/vgame_heads_w3b.pt").split(os.pathsep) if p)
DUEL_VGAME_HEADS: tuple[str, ...] = tuple(
    p for p in os.environ.get(
        "SAP_DUEL_VGAME_HEADS", os.environ.get(
            "SAP_VGAME_HEADS", "models/vgame_heads_w3b.pt",
        )).split(os.pathsep) if p)
TROPHY_CURVE_REL_PATH = os.environ.get(
    "SAP_TROPHY_CURVE", "models/trophy_recalibration_curve_v1.json")
