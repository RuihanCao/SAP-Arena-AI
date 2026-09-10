"""Stand-in for the recorded-replay viewer, which this release does not ship.

The viewer reads the raw replay corpus and pulls the corpus-build pipeline with
it, so it is not part of the public tree. `play_web/app.py` already treats it as
optional and disables one page when it cannot be imported. `play_web/http_app.py`
does not: it imports `REPLAY_HTML` at module level, so with nothing here the demo
server does not start at all.

This module therefore supplies that one constant and nothing else. The names
`app.py` looks for -- `DEFAULT_REPLAY_CACHE` and `ReplayPlayerSession` -- are
deliberately absent, so its `except ImportError` branch still fires and the
viewer stays disabled rather than half-present.
"""

from __future__ import annotations

REPLAY_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Replay viewer not included</title>
  <style>
    body { font: 15px/1.6 system-ui, sans-serif; margin: 3rem auto; max-width: 34rem;
           color: #222; background: #fafafa; }
    a { color: #0b5; }
  </style>
</head>
<body>
  <h1>Replay viewer not included</h1>
  <p>This page plays back recorded human games, and it reads the replay corpus
     that is not published with this release. The datasheet marks that corpus
     available on request.</p>
  <p>Everything else on this server works without it. <a href="/">Back to the
     simulator</a>.</p>
</body>
</html>
"""
