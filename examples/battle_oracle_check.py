#!/usr/bin/env python3
"""Prove the battle oracle is wired up: two boards in, one outcome out.

Battles are not simulated in Python. Both boards are handed to a pinned
SAP-Calculator checkout, which runs the fight. This script fails loudly if that
checkout is missing or at the wrong commit, which is the single most likely
setup problem on a fresh machine.

    python examples/battle_oracle_check.py
"""
from __future__ import annotations

import random

from sap_ppo import api
from sap_ppo.oracles.sap_calc_battle_oracle import (
    SAP_CALC_CLI,
    SAP_CALC_DIR,
    run_battle_oracle_with_config,
    team_to_pet_configs,
)
from sap_ppo.train.opening_source import build_varied_opening_source


def _board(game_index: int, rng: random.Random) -> list[dict]:
    """A board built by buying whatever the shop offers first."""
    state = build_varied_opening_source(log=lambda _m: None).state_for_game(game_index)
    for _ in range(40):
        actions = [a for a in api.legal_actions(state) if a.get("type") != "END_TURN"]
        if not actions:
            break
        transition = api.step(state, rng.choice(actions))
        if not transition["legal"]:
            break
        state = transition["state_after"]
    return state["team"]


def main() -> int:
    if not SAP_CALC_CLI.exists():
        print(f"FAIL: SAP-Calculator not found at {SAP_CALC_DIR}")
        print("      Clone it next to this repository; see the README.")
        return 1

    rng = random.Random(0)
    left, right = _board(0, rng), _board(1, rng)
    config = {
        "playerPets": team_to_pet_configs(left),
        "opponentPets": team_to_pet_configs(right),
        "simulationCount": 8,
    }
    result = run_battle_oracle_with_config(config)
    if not result.get("ok", True) and result.get("error"):
        print(f"FAIL: battle oracle error: {result['error']}")
        return 1
    print(f"battle result: {result.get('outcome')}  raw={ {k: result.get(k) for k in ('playerWins', 'opponentWins', 'draws')} }")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
