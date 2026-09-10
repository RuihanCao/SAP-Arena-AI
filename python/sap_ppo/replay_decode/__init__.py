"""sap_ppo.replay_decode -- SAP Versus replay -> full observed state + imitable chain.

Single source of truth (exp08 T7 code convergence, 2026-07-10) for:
  * reconstruct_shop -- the exp02 seed->shop RNG crack (DotNetRandom, pools);
  * decode_turn      -- the exp08 turn decoder: per-turn state records with
                        every shop the player saw + the concrete action chain;
  * verify           -- one-command regression gate pinning the wave-5 settled
                        numbers over the frozen 2,143-game Turtle cache;
  * pools.json / minion_enum.json / spell_enum.json -- the pool/enum data.

Experiment history, findings and evidence: internal project notes
(the crack) and internal project notes (the decoder, waves 1-5).

CLI:
  python -m sap_ppo.replay_decode.decode_turn --report-only   # self-check report
  python -m sap_ppo.replay_decode.verify                      # frozen regression gate
"""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
MINION_ENUM_PATH = DATA_DIR / "minion_enum.json"
SPELL_ENUM_PATH = DATA_DIR / "spell_enum.json"

from .reconstruct_shop import (  # noqa: E402
    POOLS_PATH, DotNetRandom, load_pools, reconstruct_shop, shop_tier,
    verify as verify_shop_roll,
)

# decode_turn re-exports are lazy (PEP 562) so `python -m
# sap_ppo.replay_decode.decode_turn` does not double-execute the module.
_DECODE_TURN_NAMES = {"decode_game", "genesis_shop_items", "iter_games", "run"}


def __getattr__(name: str):
    if name in _DECODE_TURN_NAMES:
        from . import decode_turn as _dt
        return getattr(_dt, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DATA_DIR", "MINION_ENUM_PATH", "SPELL_ENUM_PATH", "POOLS_PATH",
    "DotNetRandom", "load_pools", "reconstruct_shop", "shop_tier",
    "verify_shop_roll", "decode_game", "genesis_shop_items", "iter_games",
    "run",
]
