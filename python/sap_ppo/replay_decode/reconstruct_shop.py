"""Reconstruct a SAP Versus shop from a server seed (replay `Response.Data.Seed`).

Cracked 2026-07-08 (exp02 shop-seed-crack); moved here from
internal project notes in T7 code convergence (2026-07-10). Full
logic in internal project notesghidra/INDEX.md:
    playerSeed = DotNetRandom(serverSeed).Next(int.MaxValue)
    rng        = DotNetRandom(playerSeed)
    pets  = [ minion_pool[tier][ rng.Next(0, 10*tier) ] for each pet slot ]   # with replacement
    rng.NextDouble()                                                          # Sloth check (1 draw)
      -> if < 1e-4: pets[0] = Sloth (MinionEnum 71)
    foods = [ food_pool[tier][ rng.Next(0, foodN[tier]) ] for each food slot ]
    shop  = [frozen] + [drawn]

Held-out validation: tier1-3 pets ~98%, foods ~93-96%. Pools in pools.json.
Use verify()/reconstruct_verified() to keep only rolls that reproduce exactly (zero contamination).
"""
from __future__ import annotations
import json
from collections import Counter
import os
from pathlib import Path

MBIG = 2147483647
MSEED = 161803398
SLOTH_ENU = 71
POOLS_PATH = Path(__file__).with_name("pools.json")


class DotNetRandom:
    """Faithful legacy .NET System.Random (== CoreRandom), robust for large seeds."""

    def __init__(self, seed: int):
        self.a = [0] * 56
        sub = MBIG if seed == -2147483648 else abs(int(seed))
        mj = MSEED - sub
        self.a[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.a[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += MBIG
            mj = self.a[ii]
        while self.a[55] < 0:
            self.a[55] += MBIG
        for _ in range(1, 5):
            for i in range(1, 56):
                self.a[i] -= self.a[1 + (i + 30) % 55]
                while self.a[i] < 0:
                    self.a[i] += MBIG
        self.inext, self.inextp = 0, 21

    def _sample_int(self) -> int:
        ni = 1 if self.inext + 1 >= 56 else self.inext + 1
        nip = 1 if self.inextp + 1 >= 56 else self.inextp + 1
        ret = self.a[ni] - self.a[nip]
        if ret == MBIG:
            ret -= 1
        while ret < 0:
            ret += MBIG
        self.a[ni] = ret
        self.inext, self.inextp = ni, nip
        return ret

    def next_double(self) -> float:
        return self._sample_int() * (1.0 / MBIG)

    def next_int(self, maxv: int) -> int:      # System.Random.Next(maxValue)
        v = int(self.next_double() * maxv)
        return v if v < maxv else maxv - 1


def shop_tier(turn: int) -> int:
    return min(6, (turn + 1) // 2)


_POOLS = None


def load_pools(path=POOLS_PATH):
    global _POOLS
    if _POOLS is None:
        _POOLS = json.loads(Path(path).read_text())
    return _POOLS


def reconstruct_shop(server_seed: int, turn: int, *, n_pets=None, n_foods=None,
                     frozen_pets=(), frozen_foods=(), pools=None) -> dict:
    """Return {"pets":[enu...], "foods":[enu...]} the shop shown after a roll with this server seed.
    n_pets/n_foods default to the turn's capacity; pass observed counts for exact verification.
    Draw count = capacity - frozen (frozen items are carried, not re-drawn)."""
    pools = pools or load_pools()
    t = str(shop_tier(turn))
    mpool, fpool = pools["minion"][t], pools["food"][t]
    minN, foodN = len(mpool), len(fpool)
    cap_p = pools["pet_capacity"][t] if n_pets is None else n_pets
    cap_f = pools["food_capacity"][t] if n_foods is None else n_foods
    draw_p = max(0, cap_p - len(frozen_pets))
    draw_f = max(0, cap_f - len(frozen_foods))

    rng = DotNetRandom(DotNetRandom(int(server_seed)).next_int(MBIG))   # playerSeed hop
    pets = [mpool[rng.next_int(minN)] for _ in range(draw_p)]
    if rng.next_double() < 1e-4 and pets:                              # Sloth (draw always consumed)
        pets[0] = SLOTH_ENU
    foods = [fpool[rng.next_int(foodN)] for _ in range(draw_f)]
    return {"pets": list(frozen_pets) + pets, "foods": list(frozen_foods) + foods}


def verify(server_seed, turn, actual_pets, actual_foods, *, frozen_pets=(), frozen_foods=()) -> bool:
    """True iff the reconstruction reproduces the observed shop exactly (safety-net gate)."""
    r = reconstruct_shop(server_seed, turn,
                         n_pets=len(actual_pets), n_foods=len(actual_foods),
                         frozen_pets=frozen_pets, frozen_foods=frozen_foods)
    return (Counter(r["pets"]) == Counter(int(x) for x in actual_pets)
            and Counter(r["foods"]) == Counter(int(x) for x in actual_foods))


if __name__ == "__main__":
    # end-to-end: full shop (pets+foods) exact-match vs the harvested pairs
    # Developer harness: point these at a local harvest with
    # SAP_RE_PAIRS_FOOD and SAP_RE_POOLS.
    pairs = json.loads(Path(os.environ["SAP_RE_PAIRS_FOOD"]).read_text())
    load_pools(os.environ["SAP_RE_POOLS"])
    for tier in (1, 2, 3, 4, 5, 6):
        rows = [p for p in pairs if shop_tier(p["turn"]) == tier
                and isinstance(p.get("roll_seed"), int) and p["pet_enus"]]
        if len(rows) < 20:
            continue
        both = pets = foods = 0
        for p in rows:
            r = reconstruct_shop(p["roll_seed"], p["turn"], n_pets=len(p["pet_enus"]), n_foods=len(p["food_enus"]))
            mp = Counter(r["pets"]) == Counter(p["pet_enus"])
            mf = Counter(r["foods"]) == Counter(p["food_enus"])
            pets += mp; foods += mf; both += (mp and mf)
        n = len(rows)
        print(f"tier{tier}: n={n}  pets {pets/n:5.1%}  foods {foods/n:5.1%}  FULL-shop {both/n:5.1%}")
