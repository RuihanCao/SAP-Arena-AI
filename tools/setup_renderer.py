"""Install the original pinned replay-bot renderer without starting the bot."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "third_party" / "sap-replay-bot"
RUNTIME = ROOT / "tools" / "replay_renderer"
REMOTE = "https://github.com/RuihanCao/sap-replay-bot.git"
PIN = "ca06ba34bf647b7876d15922c9b24ab0d1dd16a3"


def run(*args: str, cwd: Path = ROOT, capture: bool = False) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return (result.stdout or "").strip()


def main() -> int:
    git = shutil.which("git")
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not git or not npm or not shutil.which("node"):
        print("Install Git and Node.js (including npm), then run this command again.")
        return 1
    try:
        if not RENDERER.exists():
            run(git, "init", str(RENDERER))
            run(git, "-C", str(RENDERER), "remote", "add", "origin", REMOTE)
        if not (RENDERER / ".git").exists():
            raise RuntimeError(f"Not a Git checkout: {RENDERER}")
        if run(git, "-C", str(RENDERER), "status", "--porcelain", capture=True):
            raise RuntimeError("Replay-bot checkout has local changes; leaving it untouched.")
        remote = run(git, "-C", str(RENDERER), "remote", "get-url", "origin", capture=True)
        if remote != REMOTE:
            raise RuntimeError("Replay-bot checkout has a different origin; leaving it untouched.")
        head = subprocess.run([git, "-C", str(RENDERER), "rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode == 0 and head.stdout.strip() != PIN:
            raise RuntimeError("Replay-bot checkout is at a different revision; leaving it untouched.")
        if head.returncode != 0:
            run(git, "-C", str(RENDERER), "fetch", "--depth", "1", "origin", PIN)
            run(git, "-C", str(RENDERER), "checkout", "--detach", PIN)
        # Only Canvas is needed. Do not install or launch Discord, Playwright,
        # or the bot's separately configured battle simulator.
        run(npm, "ci", "--no-audit", "--no-fund", cwd=RUNTIME)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Renderer setup failed: {exc}")
        return 1
    print("Original replay renderer is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
