#!/usr/bin/env python3
"""Download the two release weights: python tools/fetch_models.py."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

REPOSITORY = "RuihanCao/SAP-Arena-AI"
RELEASE = "v0.1.0"
DEFAULT_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE}"
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models"

# Runtime bundle: the attention BC proposer and its compatible value head.
# The head carries its leaf-value mapping; no external curve is required.
FILES = {
    "bc_attn_v4.zip": "5ae652854896aa92273523b741c65fe45c10e461adffb7082f853b99221a64cc",
    # Retain the existing download filename; the checksum identifies the model.
    "vgame_heads_w3b.pt": "baed69f0956ce88d2bebd886aaab1e5939ae7901eeb58e99ad690897c6a0ac39",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, target: Path, base: str) -> None:
    """Prefer the public URL; private releases may use an authenticated gh CLI.

    Tokens are never placed in a URL, passed as command arguments, or forwarded
    to an override host. GitHub CLI handles its own stored credentials.
    """
    try:
        with urllib.request.urlopen(f"{base}/{name}", timeout=30) as response:
            with target.open("wb") as out:
                shutil.copyfileobj(response, out)
        return
    except OSError:
        if base != DEFAULT_URL:
            raise RuntimeError("download failed from SAP_MODELS_URL") from None
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "release download failed; check your connection. For private access, "
            "install GitHub CLI and run gh auth login"
        )
    try:
        # gh writes the named asset in an isolated directory; never clobber the
        # user's existing checkpoint until the new bytes pass verification.
        result = subprocess.run(
            [gh, "release", "download", RELEASE, "--repo", REPOSITORY,
             "--pattern", name, "--dir", str(target.parent), "--clobber"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("GitHub CLI download failed or timed out") from None
    if result.returncode != 0 or not target.is_file():
        raise RuntimeError(
            "GitHub release unavailable; check your connection and repository "
            "access, then run gh auth login if needed"
        )


def main() -> int:
    base = os.environ.get("SAP_MODELS_URL", DEFAULT_URL).rstrip("/")
    dest = Path(os.environ.get("SAP_MODELS_DIR", str(DEFAULT_DIR)))
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Cannot create the model directory; check SAP_MODELS_DIR and permissions.")
        return 1

    bad = 0
    for name, want in FILES.items():
        target = dest / name
        try:
            if target.is_file() and sha256(target) == want:
                print(f"  {name}: already present")
                continue
            print(f"  {name}: downloading")
            with tempfile.TemporaryDirectory(prefix=".download-", dir=dest) as temp:
                pending = Path(temp) / name
                download(name, pending, base)
                if sha256(pending) != want:
                    raise RuntimeError("downloaded file has an unexpected checksum")
                pending.replace(target)
        except (OSError, RuntimeError) as exc:
            # Do not echo network errors: they may include a signed URL or a
            # user-supplied URL containing credentials.
            detail = str(exc) if isinstance(exc, RuntimeError) else "file operation failed"
            print(f"  {name}: {detail}")
            bad += 1
        else:
            print(f"  {name}: ready")

    if bad:
        print(f"\n{bad} file(s) missing or failed verification")
        return 1
    print("\nBoth models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
