#!/usr/bin/env python3
"""Print what this export actually contains.

The module and file counts of the public tree are not a constant anyone may
quote. Three different numbers for them were in circulation at once inside the
project, because each was written down by hand on a different day while the tree
kept moving. They are computed by the exporter, recorded in `release_info.json`
at the root of this tree, and read back here.

usage: release_info.py [path/to/release_info.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "release_info.json"

if not path.is_file():
    # Not a warning: this file is written by the exporter, so a tree without one
    # was not produced by the documented path and its provenance is unknown.
    print(f"FATAL: no release_info.json at {path}. This tree was not produced by "
          f"the release exporter, so its contents are not accounted for.", file=sys.stderr)
    raise SystemExit(1)

info = json.loads(path.read_text(encoding="utf-8"))
width = max(len(k) for k in info)
for key in sorted(info):
    print(f"{key:{width}}  {info[key]}")
