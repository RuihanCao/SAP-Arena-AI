"""Which tree is this process actually serving?

A play_web server is started from a worktree with `PYTHONPATH=python`, while
the venv installs `sap_ppo` editable against the MAIN checkout. Drop that
prefix and the page looks entirely normal while serving `main`'s code --
SAP-Arena/CLAUDE.md records that exact failure, and `ops/play-web/dev8766.sh`
carries the prefix for the same reason.

A branch name in a runbook cannot tell those two apart, and neither can the
worktree the launcher *meant* to use. So the answer is read off the LOADED
module object and reported by the running server at `/api/build`: whatever
tree `sap_ppo.__file__` resolves into is, by construction, the tree whose
code is executing.

Resolved once per process and cached: it is a `git` call, and every snapshot
carries the result.
"""

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
    """The provenance of the code this process loaded.

    `module_path` is the load-bearing field: it is where Python actually
    found the package, so a missing `PYTHONPATH=python` shows up as a path
    under the main checkout no matter which worktree the launcher `cd`-ed
    into. `commit` is then that tree's HEAD, not a branch anyone typed.
    """
    root = _REPO_ROOT
    commit = _git(root, "rev-parse", "HEAD")
    head_ref = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain")
    return {
        "module_path": str(Path(__file__).resolve()),
        "repo_root": str(root),
        "commit": commit,
        "commit_short": commit[:7] if commit else None,
        # "HEAD" means detached, which is how the main-tracking deploy
        # worktrees are meant to sit.
        "head_ref": head_ref,
        "dirty": None if status is None else bool(status),
        "resolved": commit is not None,
    }
