#!/usr/bin/env python3
"""Play N shop phases with a random legal agent.

This is the smallest end-to-end exercise of the rules engine: every action it
takes comes from `legal_actions`, and every transition goes through `step`, so a
transport error anywhere in the engine shows up as an illegal transition rather
than as a plausible-looking board.

    python examples/random_agent.py --games 20

Battles are not resolved here. Ending a turn needs an opponent board, and the
opponent pool ships with the evaluation harness; see the README.
"""
from __future__ import annotations

import argparse
import random

from sap_ppo import api
from sap_ppo.train.opening_source import build_varied_opening_source


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-actions", type=int, default=200)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    source = build_varied_opening_source(log=lambda _m: None)

    total_actions = 0
    illegal = 0
    for game in range(args.games):
        state = source.state_for_game(game)
        for _ in range(args.max_actions):
            actions = api.legal_actions(state)
            if not actions:
                break
            action = rng.choice(actions)
            if action.get("type") == "END_TURN":
                break
            transition = api.step(state, action)
            if not transition["legal"]:
                illegal += 1
                break
            state = transition["state_after"]
            total_actions += 1
        else:
            print(f"game {game}: hit the action cap without ending the turn")

    print(f"games={args.games} actions={total_actions} illegal_transitions={illegal}")
    if illegal:
        print("FAIL: the engine accepted an action it had offered as legal")
        return 1
    if total_actions == 0:
        print("FAIL: no actions were taken, so nothing was exercised")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
