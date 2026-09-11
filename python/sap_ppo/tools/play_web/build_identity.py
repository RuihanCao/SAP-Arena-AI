"""Public version and commit of the loaded demo code, cached per process."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

# <root>/python/sap_ppo/tools/play_web/build_identity.py -> <root>
_REPO_ROOT = Path(__file__).resolve().parents[4]

_GIT_TIMEOUT_S = 10


def _git(root: Path, *args: str) -> str | None:
    """One `git` call under `root`, or None if git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


@lru_cache(maxsize=1)
def build_identity() -> dict[str, Any]:
    """Return version and commit without exposing local paths or branch names."""
    root = _REPO_ROOT
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "version": "0.1.0",
        "commit": commit,
        "commit_short": commit[:7] if commit else None,
        "dirty": None if status is None else bool(status),
        "resolved": commit is not None,
    }
