"""sap_ppo.replay_decode.decode_turn -- replay turn -> state + imitable action chain.

1. Freezing is NOT an action type. The client batches the current freeze state
   onto the NEXT request of any type: Request.Data.BoardFreezes appears on
   Types 5/6/7/8/9/11 and is a FULL-STATE enumeration of the current shop
   ("ItemId":{BoId,Uni} entries; "Freeze":true marks frozen, absent means
   unfrozen). It replaces, never increments.
2. Consumption clears freeze: a frozen shop item that gets bought/eaten leaves
   the frozen set (Type 6 MinionId.Uni / Type 7 SourceMinionId.Uni /
   Type 8 SpellId.Uni). Selling (Type 9) is board-side and never touches it.
3. Frozen items keep their Uni across rolls and across turns; end-of-turn
   snapshot items carry Id.Uni + Fro, so the tracker cross-checks itself
   against every end snapshot (dev cache: 1975/1977 turns match; the 2 misses
   are single frozen unis vanishing without a consuming action, counted, not
   modeled). WFro on snapshot items = "was frozen at turn start" (carry-in
   marker), not used by the tracker.
4. Shop Unis are allocated sequentially in draw order (pets then foods): on
   clean-last-roll turns, snapshot items sorted by Uni reproduce the
   reconstruction ORDER exactly (130/130 dev turns; the 3 apparent misses all
   had mid-turn frozen carries, i.e. Uni gaps). Slot-level alignment is
   therefore available wherever a block's base Uni is pinned.

REMAINING ASSUMPTIONS (read before trusting a record):

4. Action ordering: CreatedOn is assumed monotonic non-decreasing within a
   turn in the raw Actions array order. This is verified at runtime; any turn
   that violates it has its action list stable-sorted by (CreatedOn, original
   index) instead, and the violation is counted and reported (never silently
   trusted).

5. At most one Type-0 (battle snapshot) per turn is assumed (confirmed on the
   163-game dev cache: exactly 1977 Type-0 actions across exactly 1977 turns).
   The first Type-0 found in a turn's (chronological) action list is used."""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
import os
from pathlib import Path

from .reconstruct_shop import (
    POOLS_PATH, load_pools, reconstruct_shop, shop_tier,
)

DEFAULT_CACHE = Path(os.environ.get(
    "SAP_REPLAY_MANIFEST_CACHE",
    str(Path(__file__).resolve().parents[3] / "data" / "raw_replays_manifest.json.gz"),
))


SHOP_ACTION_TYPES = {5, 6, 7, 8, 9, 10}


COW_ENU, PIGEON_ENU, WORM_ENU = 15, 559, 82
CHOCOLATE_ENU = 23
CRUMB_ENU = 139
MILK_BY_LVL = {1: 49, 2: 102, 3: 103}
CHOCOMILK_BY_LVL = {1: 130, 2: 131, 3: 132}
WORM_APPLE_BY_LVL = {1: 0, 2: 134, 3: 135}


MILK_ENUS = set(MILK_BY_LVL.values())
CHOCOMILK_ENUS = set(CHOCOMILK_BY_LVL.values())


INJ_FOOD_FAMILY = {49: "milk", 102: "milk", 103: "milk",
                   130: "chocomilk", 131: "chocomilk", 132: "chocomilk",
                   139: "crumb", 134: "worm-apple", 135: "worm-apple"}


PILL_ENU = 92
FLY_ENU = 30
RAT_ENU = 57            # faint spawns on the OPPONENT side -- consumption unknown
# shop-phase pill faint: tokens spawned on the player's board (consume unis).
SPAWN_ON_PILL = {17: 1, 68: 2, 20: 1, 74: 1}   # Cricket/Sheep/Deer/Spider
SPAWN_ON_PILL_BY_LVL = {63: 1}                 # Rooster: `level` Chicks
CASCADE_ON_PILL = {2, 37}   # Badger/Hedgehog: faint damage may kill neighbours
# perks that spawn on a shop-phase faint (dump.cs `Perk`): Honey -> Bee,
# Mushroom -> self-copy. Both are faint-triggered, so a pet that SURVIVED its
# battle still carries them -- the previous end snapshot is a safe source.
PERK_HONEY, PERK_MUSHROOM = 8, 1
PERK_UNKNOWN = -1       # sentinel: a fed food we could not map -> state unknown
_NAME_TO_SPELL = {v: int(k) for k, v in json.loads(
    (Path(__file__).with_name("spell_enum.json")).read_text()).items()}
# spell -> the Perk it equips (Turtle-pack equipables; dump.cs `Perk` consts)
PERK_BY_SPELL = {_NAME_TO_SPELL[n]: p for n, p in (
    ("Honey", PERK_HONEY), ("MeatBone", 6), ("Garlic", 9), ("Chili", 5),
    ("Melon", 13), ("Steak", 12), ("Mushroom", PERK_MUSHROOM)) if n in _NAME_TO_SPELL}
# consumable foods that do NOT touch the perk slot (stat/exp/pill/injected)
NONPERK_SPELLS = {_NAME_TO_SPELL[n] for n in (
    "Apple", "Apple2", "Apple3", "Chocolate", "Pill", "Cupcake", "CupCake",
    "Salad", "SaladBowl", "CannedFood", "Pear", "Sushi", "Pizza",
    "Milk", "Milk2", "Milk3", "ChocolateMilk", "ChocolateMilk2",
    "ChocolateMilk3", "BreadCrumbs") if n in _NAME_TO_SPELL}
# consume subtypes NOT yet allowed to be chained across even when n is known
# (pre-calibration hypotheses live here; emptied as the probe validates them)
CHAIN_BARRIER_WHYS: set = set()


CHAIN_ALLOW_1DIR = True


# --------------------------------------------------------------------------
# cache loading (both formats)
# --------------------------------------------------------------------------

def _sniff_jsonl(path: Path) -> bool:
    """True if `path` looks like the format-2 JSONL cache (one record per
    line, each a JSON object with a "raw" key) rather than format-1's single
    JSON dict spanning the whole file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
    try:
        obj = json.loads(first_line)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and "raw" in obj


def iter_games(path: Path, max_games: int | None = None):
    """Yield (game_id, game_dict) pairs from either cache format."""
    path = Path(path)
    if _sniff_jsonl(path):
        n = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                game = rec.get("raw")
                if not isinstance(game, dict):
                    continue
                yield str(rec.get("id", n)), game
                n += 1
                if max_games is not None and n >= max_games:
                    return
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            games = json.load(f)
        n = 0
        for gid, game in games.items():
            yield gid, game
            n += 1
            if max_games is not None and n >= max_games:
                return


# --------------------------------------------------------------------------
# low-level action field parsing
# --------------------------------------------------------------------------

def _parse_json_field(action: dict, key: str):
    raw = action.get(key)
    if raw is None or raw == "":
        return None
    if isinstance(raw, (dict, list)):  # defensive: already-parsed value
        return raw
    return json.loads(raw)


def action_seed(action: dict):
    """Response.Data.Seed, or None. Present on all Type-4 and Type-5 actions."""
    resp = _parse_json_field(action, "Response")
    if not isinstance(resp, dict):
        return None
    data = resp.get("Data")
    if not isinstance(data, dict):
        return None
    return data.get("Seed")


def snapshot_shop(action: dict):
    """(pet_items, food_items) from a Type-0 action's Build.Bor.MiSh/.SpSh, or
    None. NB the Bor nesting -- items at Build top level do not exist."""
    build = _parse_json_field(action, "Build")
    if not isinstance(build, dict):
        return None
    bor = build.get("Bor")
    if not isinstance(bor, dict):
        return None
    mish = [i for i in (bor.get("MiSh") or []) if isinstance(i, dict)]
    spsh = [i for i in (bor.get("SpSh") or []) if isinstance(i, dict)]
    return mish, spsh


def item_enu(item: dict) -> int:
    return int(item.get("Enu", 0))  # Ant bug: Enu is OMITTED (not 0) when the item is Ant


def item_frozen(item: dict) -> bool:
    return bool(item.get("Fro"))


def item_uni(item: dict):
    u = (item.get("Id") or {}).get("Uni")
    return int(u) if u is not None else None


def genesis_shop_items(game: dict):
    """The stored turn-1 opening shop, from the replay's GenesisBuildModel.

    T3 genesis is a READ, not a crack: the game persists the match-created
    opening shop verbatim in this field (a serialized BuildModel with the same
    Bor.MiSh/SpSh item shape as a Type-0 snapshot). It is NOT rolled from any
    replay-exposed seed -- a brute force over the turn-1 Type-4 seed, MatchId
    and UserId (all derivations x pre-draw skips 0-40 x opponent-shop-first)
    reproduces it only at chance, and the field's stored Unis (pets 1,2,3 +
    food 4) confirm genesis is the start of the sequential draw counter.

    Returns [(uni, kind, enu, fro, pri), ...] ordered by the stored Uni
    (= draw order), or None if the field is absent.
    """
    g = game.get("GenesisBuildModel")
    if g is None:
        return None
    g = json.loads(g) if isinstance(g, str) else g
    bor = g.get("Bor") if isinstance(g, dict) else None
    if not isinstance(bor, dict):
        return None
    items = []
    for kind, key in (("pet", "MiSh"), ("food", "SpSh")):
        for it in bor.get(key) or []:
            if isinstance(it, dict):
                items.append((item_uni(it), kind, item_enu(it),
                              item_frozen(it), it.get("Pri")))
    items.sort(key=lambda x: (x[0] is None, x[0]))   # by Uni = draw order
    return items


def parse_created_on(action: dict):
    v = action.get("CreatedOn")
    if not v:
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# T6: request payload parsing (freeze / order / target fields)
# --------------------------------------------------------------------------

def action_request(action: dict) -> dict:
    req = _parse_json_field(action, "Request")
    return req if isinstance(req, dict) else {}


def board_freezes_frozen_unis(req: dict):
    """The FULL-STATE frozen-uni set carried by this request, or None if the
    request has no BoardFreezes batch (semantics #1 in the module docstring:
    entries without "Freeze":true are explicitly UNfrozen)."""
    bf = (req.get("Data") or {}).get("BoardFreezes")
    if not bf:
        return None
    out = set()
    for e in bf:
        if isinstance(e, dict) and e.get("Freeze"):
            u = (e.get("ItemId") or {}).get("Uni")
            if u is not None:
                out.add(int(u))
    return out


def board_orders_slots(req: dict) -> dict:
    """Board orders slots."""
    out = {}
    for e in (req.get("Data") or {}).get("BoardOrders") or []:
        if not isinstance(e, dict):
            continue
        u = (e.get("MinionId") or {}).get("Uni")
        if u is None:
            continue
        x = (e.get("Point") or {}).get("x") or 0
        out[int(u)] = 4 - int(x)
    return out


def consumed_shop_uni(action_type, req: dict):
    """The shop-item Uni this action removes from the shop (freeze semantics
    #2), or None. Type 6's MinionId may be a BOARD uni (a move) -- discarding
    a board uni from the frozen set is a no-op, so callers need not care."""
    if action_type == 6:
        u = (req.get("MinionId") or {}).get("Uni")
    elif action_type == 7:
        u = (req.get("SourceMinionId") or {}).get("Uni")
    elif action_type == 8:
        u = (req.get("SpellId") or {}).get("Uni")
    else:
        u = None
    return int(u) if u is not None else None


def food_enu_from_response(action: dict):
    """Food enu from response."""
    resp = _parse_json_field(action, "Response")
    if not isinstance(resp, dict):
        return None
    enu = ((((resp.get("Event") or {}).get("Event") or {}).get("Spell") or {}).get("Enu"))
    return int(enu) if enu is not None else None


def point_to_slot(req: dict) -> int:
    """Board target slot of a Type-6 request (slot = 4 - Point.x; Point
    omitted means x=0, i.e. the back slot)."""
    x = (req.get("Point") or {}).get("x") or 0
    return 4 - int(x)


def snapshot_uni_entries(action: dict):
    """[(uni, kind, enu)] for every identifiable item in a Type-0 snapshot:
    shop pets (Bor.MiSh), shop foods (Bor.SpSh) and board pets
    (Bor.Mins.Items). Feeds the game-wide Uni->Enu map."""
    build = _parse_json_field(action, "Build")
    if not isinstance(build, dict):
        return []
    bor = build.get("Bor")
    if not isinstance(bor, dict):
        return []
    out = []
    for kind, items in (("pet", bor.get("MiSh")), ("food", bor.get("SpSh"))):
        for i in items or []:
            if not isinstance(i, dict):
                continue
            u = (i.get("Id") or {}).get("Uni")
            if u is not None:
                out.append((int(u), kind, item_enu(i)))
    mins = bor.get("Mins")
    items = mins.get("Items") if isinstance(mins, dict) else None
    for i in items or []:
        if not isinstance(i, dict):
            continue
        u = (i.get("Id") or {}).get("Uni")
        if u is not None:
            out.append((int(u), "pet", item_enu(i)))
    return out


# --------------------------------------------------------------------------
# per-game turn grouping
# --------------------------------------------------------------------------

def group_actions_by_turn(actions: list):
    """Group a game's Actions by Turn. Returns (by_turn: {turn: [action,...]},
    violations: int). Array order is trusted as chronological UNLESS CreatedOn
    is found to go backwards inside a turn's slice, in which case that turn's
    actions are stable-sorted by (CreatedOn, original index) and the violation
    is counted (assumption #4 in the module docstring)."""
    raw = defaultdict(list)
    for a in actions:
        turn = a.get("Turn")
        if not isinstance(turn, int):
            continue
        raw[turn].append(a)

    by_turn = {}
    violations = 0
    for turn, acts in raw.items():
        times = [parse_created_on(a) for a in acts]
        bad_here = sum(
            1 for i in range(1, len(times))
            if times[i] is not None and times[i - 1] is not None and times[i] < times[i - 1]
        )
        violations += bad_here
        if bad_here:
            order = sorted(range(len(acts)), key=lambda i: (times[i] is None, times[i] or 0, i))
            acts = [acts[i] for i in order]
        by_turn[turn] = acts
    return by_turn, violations


def turn_end_snapshot(acts: list):
    """{"pets": [{"enu","fro"}...], "foods": [...]} from this turn's Type-0
    Build, or None if absent (e.g. the game ended before a final battle).
    Assumes at most one Type-0 per turn (module docstring assumption #5)."""
    for a in acts:
        if a.get("Type") == 0:
            shop = snapshot_shop(a)
            if shop is None:
                return None
            mish, spsh = shop
            return {
                "pets": [{"enu": item_enu(i), "fro": item_frozen(i), "uni": item_uni(i)} for i in mish],
                "foods": [{"enu": item_enu(i), "fro": item_frozen(i), "uni": item_uni(i)} for i in spsh],
            }
    return None


def find_last_clean_roll(acts: list):
    """Find last clean roll."""
    shop_acts = [(i, a) for i, a in enumerate(acts) if a.get("Type") in SHOP_ACTION_TYPES]
    if not shop_acts or shop_acts[-1][1].get("Type") != 5:
        return None, False
    li = shop_acts[-1][0]
    roll_seed = action_seed(shop_acts[-1][1])
    if not isinstance(roll_seed, int):
        return None, False
    for j in range(li + 1, len(acts)):
        t = acts[j].get("Type")
        if t == 0:
            return roll_seed, True
        if t in SHOP_ACTION_TYPES:
            return None, False
    return None, False  # ran off the end of the turn without a Type-0


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------

def is_sub_multiset(small_items, big_items) -> bool:
    small, big = Counter(small_items), Counter(big_items)
    return all(big[k] >= v for k, v in small.items())


def _example(seed, carry_pets, carry_foods, snap_pets, snap_foods, recon_pets, recon_foods) -> dict:
    return {
        "seed": seed,
        "carry_pets": sorted(carry_pets), "carry_foods": sorted(carry_foods),
        "snap_pets": sorted(snap_pets), "recon_pets": sorted(recon_pets),
        "snap_foods": sorted(snap_foods), "recon_foods": sorted(recon_foods),
    }


def classify_validation(acts, end_snapshot, turn, carry_status, carry_pets, carry_foods,
                         start_shop, pools, last_roll=None) -> dict:
    """Bucket this turn into a/b/c/d/e (or None = outside self-check scope)
    and check it. Returns {"bucket", "pass", "note", "detail"}; "note"
    explains an unbucketed turn, "detail" is a compact failing example
    (seed/carry/snap-vs-recon), populated only on a bucketed FAILURE."""
    if end_snapshot is None:
        return {"bucket": None, "pass": None, "note": "no-end-snapshot", "detail": None}

    snap_pets = [i["enu"] for i in end_snapshot["pets"]]
    snap_foods = [i["enu"] for i in end_snapshot["foods"]]
    end_fp = [i["enu"] for i in end_snapshot["pets"] if i["fro"]]
    end_ff = [i["enu"] for i in end_snapshot["foods"] if i["fro"]]
    end_frozen_empty = not end_fp and not end_ff

    types = [a.get("Type") for a in acts]
    has_roll = 5 in types
    roll_seed, clean_ok = find_last_clean_roll(acts)

    def tracked_recon_ok(legacy_pets, legacy_foods, legacy_ok):
        """Tracked recon ok."""
        if last_roll is None or last_roll.get("seed") != roll_seed or last_roll["unresolved"]:
            return None
        fp, ff = last_roll["frozen_pets"], last_roll["frozen_foods"]
        if Counter(fp) == Counter(legacy_pets) and Counter(ff) == Counter(legacy_foods):
            return legacy_ok
        r = reconstruct_shop(roll_seed, turn, n_pets=len(snap_pets), n_foods=len(snap_foods),
                              frozen_pets=fp, frozen_foods=ff, pools=pools)
        return Counter(r["pets"]) == Counter(snap_pets) and Counter(r["foods"]) == Counter(snap_foods)

    def e_lane():
        """(e): reconstruct the last clean roll under its TRACKED frozen set."""
        if last_roll is None or last_roll.get("seed") != roll_seed:
            return {"bucket": None, "pass": None, "note": "e-no-tracked-roll", "detail": None}
        if last_roll["unresolved"]:
            return {"bucket": None, "pass": None, "note": "e-frozen-unresolved", "detail": None}
        fp, ff = last_roll["frozen_pets"], last_roll["frozen_foods"]
        r = reconstruct_shop(roll_seed, turn, n_pets=len(snap_pets), n_foods=len(snap_foods),
                              frozen_pets=fp, frozen_foods=ff, pools=pools)
        ok = Counter(r["pets"]) == Counter(snap_pets) and Counter(r["foods"]) == Counter(snap_foods)
        detail = None if ok else _example(roll_seed, fp, ff, snap_pets, snap_foods, r["pets"], r["foods"])
        return {"bucket": "e", "pass": ok, "note": None, "detail": detail}

    if clean_ok and roll_seed is not None:
        if carry_status == "empty" and end_frozen_empty:
            r = reconstruct_shop(roll_seed, turn, n_pets=len(snap_pets), n_foods=len(snap_foods), pools=pools)
            ok = Counter(r["pets"]) == Counter(snap_pets) and Counter(r["foods"]) == Counter(snap_foods)
            detail = None if ok else _example(roll_seed, [], [], snap_pets, snap_foods, r["pets"], r["foods"])
            return {"bucket": "a", "pass": ok, "note": None, "detail": detail,
                    "pass_tracked": tracked_recon_ok([], [], ok)}
        if carry_status == "nonempty":
            same_frozen = Counter(end_fp) == Counter(carry_pets) and Counter(end_ff) == Counter(carry_foods)
            if same_frozen:
                r = reconstruct_shop(roll_seed, turn, n_pets=len(snap_pets), n_foods=len(snap_foods),
                                      frozen_pets=carry_pets, frozen_foods=carry_foods, pools=pools)
                ok = Counter(r["pets"]) == Counter(snap_pets) and Counter(r["foods"]) == Counter(snap_foods)
                detail = (None if ok else
                          _example(roll_seed, carry_pets, carry_foods, snap_pets, snap_foods, r["pets"], r["foods"]))
                return {"bucket": "d", "pass": ok, "note": None, "detail": detail,
                        "pass_tracked": tracked_recon_ok(carry_pets, carry_foods, ok)}
            return e_lane()   # carry-in present but the freeze state changed mid-turn
        if carry_status == "unknown":
            return {"bucket": None, "pass": None, "note": "roll-clean-but-carry-unknown", "detail": None}
        return e_lane()       # empty carry-in but new freeze(s) at end

    if not has_roll:
        if start_shop["provenance"] in ("reconstructed", "genesis"):
            if start_shop["provenance"] == "genesis":
                bucket = "g"      # turn-1 stored opening shop (T3); subset test
            else:
                bucket = "b" if carry_status == "empty" else "c"
            ok = is_sub_multiset(snap_pets, start_shop["pets"]) and is_sub_multiset(snap_foods, start_shop["foods"])
            # T5 companion column: pass if the snapshot subsets the FULL shop.
            # Available foods = rolled ++ start-of-turn injections (Worm/Pigeon).
            # Any end-snapshot food that is an injection-ONLY enu (milk / crumb /
            # apple2-3 / chocomilk) is self-evidently a mid-turn Cow stock or a
            # start injection and is always explained; the rest must subset the
            # available multiset. Pets must still subset the rolled pets.
            inj_foods = list(start_shop["foods"]) + [it["enu"] for it in (start_shop.get("injected") or [])]
            avail = Counter(inj_foods)
            pool_surplus = Counter(f for f in snap_foods if not INJ_FOOD_FAMILY.get(f))
            ok_inj = (is_sub_multiset(snap_pets, start_shop["pets"]) and
                      all(avail[e] >= k for e, k in pool_surplus.items()))
            detail = (None if ok else
                      _example(None, carry_pets, carry_foods, snap_pets, snap_foods,
                               start_shop["pets"], inj_foods))
            return {"bucket": bucket, "pass": ok, "pass_inj": ok_inj, "note": None, "detail": detail}
        return {"bucket": None, "pass": None, "note": f"no-roll-start-shop-{start_shop['provenance']}", "detail": None}

    return {"bucket": None, "pass": None, "note": "roll-not-last-shop-action", "detail": None}


# --------------------------------------------------------------------------
# T6: per-turn action walk (freeze timeline + concrete op chain + membership)
# --------------------------------------------------------------------------

def build_game_uni_map(actions: list) -> dict:
    """Game-wide {uni: (kind, enu)} from every Type-0 snapshot (shop items +
    board pets) plus Type-8 responses (consumed shop foods, whose identity is
    in Response.Event.Event.Spell.Enu and which may never reach a snapshot).
    First sighting wins (unis are unique per board, so later sightings agree)."""
    m = {}
    for a in actions:
        ty = a.get("Type")
        if ty == 0:
            for u, kind, enu in snapshot_uni_entries(a):
                m.setdefault(u, (kind, enu))
        elif ty == 8:
            req = action_request(a)
            u = (req.get("SpellId") or {}).get("Uni")
            enu = food_enu_from_response(a)
            if u is not None and enu is not None:
                m.setdefault(int(u), ("food", enu))
    return m


def snapshot_board_unis(action: dict) -> set:
    """Unis of the player's BOARD pets in a Type-0 snapshot (Bor.Mins.Items)."""
    build = _parse_json_field(action, "Build")
    bor = build.get("Bor") if isinstance(build, dict) else None
    mins = bor.get("Mins") if isinstance(bor, dict) else None
    items = mins.get("Items") if isinstance(mins, dict) else None
    out = set()
    for i in items or []:
        if isinstance(i, dict):
            u = item_uni(i)
            if u is not None:
                out.add(u)
    return out


def snapshot_board_pets(action: dict) -> list:
    """Snapshot board pets."""
    build = _parse_json_field(action, "Build")
    bor = build.get("Bor") if isinstance(build, dict) else None
    mins = bor.get("Mins") if isinstance(bor, dict) else None
    if isinstance(mins, dict):
        mins = mins.get("Items")
    out = []
    for i in mins or []:
        if not isinstance(i, dict):
            continue
        exp = i.get("Exp") or 0


        slot = 4 - int((i.get("Poi") or {}).get("x") or 0)
        out.append({"uni": item_uni(i), "enu": item_enu(i),
                    "lvl": 1 + (1 if exp >= 2 else 0) + (1 if exp >= 5 else 0),
                    "exp": exp, "perk": i.get("Perk"), "slot": slot})
    return out


def start_of_turn_injections(prev_board_pets: list) -> list:
    """T5 START-OF-TURN injected foods given the board ENTERING the turn
    (= previous end snapshot's board). Returns [{"enu","src","pri"}...]:
    a board **Worm** stocks 1 discounted apple (2g, level variant). Deterministic.

    NOTE: Pigeon's BreadCrumb is NOT here -- it is SELL-triggered, not
    start-of-turn (verified on the full cache: 788/792 crumb turns have a sell,
    crumbs per sell = pigeon level; SAP-Calculator turtle/tier-1/pigeon.class.ts:
    triggers ['ThisSold'], count = this.level). It is emitted in the walk's
    Type-9 handler."""
    inj = []
    for p in prev_board_pets or []:
        if p["enu"] == WORM_ENU:
            inj.append({"enu": WORM_APPLE_BY_LVL[p["lvl"]],
                        "src": f"Worm L{p['lvl']}", "pri": 2})
    return inj


def resolve_frozen_ids(frozen_unis, carry_map: dict, uni_map: dict):
    """Resolve a frozen-uni set to ({"pets": [enu...], "foods": [enu...]},
    n_unresolved), in uni order (= draw/allocation order). carry_map (this
    turn's frozen carry-in, from the previous end snapshot) takes precedence;
    the game-wide map covers items frozen mid-turn."""
    pets, foods = [], []
    unresolved = 0
    for u in sorted(frozen_unis):
        ent = carry_map.get(u) or uni_map.get(u)
        if ent is None:
            unresolved += 1
            continue
        kind, enu = ent
        (pets if kind == "pet" else foods).append(enu)
    return {"pets": pets, "foods": foods}, unresolved


def new_membership() -> dict:
    return {
        "pet_buys": 0, "pet_checked": 0, "pet_ok": 0,
        "food_buys": 0, "food_checked": 0, "food_ok": 0,
        "food_ok_injected": 0,       # T5: matched via an injection (family token or stocked item)
        "pet_ok_crossref": 0,        # identity pass: enu came from the uni-counter cross-ref
        "food_ok_crossref": 0,
        "pet_ok_crossref_1dir": 0,   # subset resolved via ONE-direction chain pins (tagged slice)
        "food_ok_crossref_1dir": 0,
        "pet_ok_crossref_1anchor": 0,
        "food_ok_crossref_1anchor": 0,
        "pet_ok_crossref_position": 0,
        "food_ok_crossref_position": 0,
        "pet_ok_crossref_milk": 0,
        "food_ok_crossref_milk": 0,
        "pet_ok_crossref_crumb": 0,
        "food_ok_crossref_crumb": 0,
        "op_kind_conflict": 0,

        "food_miss_enu": Counter(),
        "miss_examples": [],
    }


def decode_turn_actions(acts: list, turn: int, carry_map: dict, uni_map: dict,
                         board_unis: set, start_shop: dict, pools: dict,
                         membership: dict, where: str = "",
                         crossref_unis: set | None = None,
                         crossref_1dir_unis: set | None = None,
                         crossref_1anchor_unis: set | None = None,
                         board_pet_levels: dict | None = None,
                         orphan_combine_unis: set | None = None,
                         board_pet_slots: dict | None = None,
                         crossref_position_unis: set | None = None,
                         crossref_milk_unis: set | None = None,
                         crossref_crumb_unis: set | None = None,
                         end_crumb_leftover: int = 0,
                         pos_stats=None):
    """Walk one turn's actions applying the T6 freeze semantics (module
    docstring #1-#3) and decoding each action to a concrete op.

    board_unis is UPDATED IN PLACE (buys add, sells/board-merges remove) --
    it is how Type-6 buy vs move is split. membership is the run-wide
    buy-membership accumulator (updated in place).

    Returns {"chain", "rolls", "final_frozen_unis", "n_freeze_events",
    "stream"}. rolls[i] = {seed, frozen_unis, frozen_pets|None,
    frozen_foods|None, unresolved, pets|None, foods|None, provenance} --
    reconstruction under the TRACKED frozen state at that roll
    ("frozen-tracked"), or None when a frozen uni has no identity anywhere
    ("frozen-unresolved")."""
    cur_frozen = set(carry_map)
    if crossref_unis is None:
        crossref_unis = set()
    if crossref_1dir_unis is None:
        crossref_1dir_unis = set()
    if crossref_1anchor_unis is None:
        crossref_1anchor_unis = set()
    if orphan_combine_unis is None:
        orphan_combine_unis = set()
    if crossref_position_unis is None:
        crossref_position_unis = set()
    if crossref_milk_unis is None:
        crossref_milk_unis = set()
    if crossref_crumb_unis is None:
        crossref_crumb_unis = set()
    if pos_stats is None:
        pos_stats = Counter()

    def id_src(u):
        """Id src."""
        if u in crossref_position_unis:
            return "crossref-position"
        if u in crossref_milk_unis:
            return "crossref-milk"
        if u in crossref_crumb_unis:
            return "crossref-crumb"
        if u in crossref_1dir_unis:
            return "crossref-1dir"
        if u in crossref_1anchor_unis:
            return "crossref-1anchor"
        if u in crossref_unis:
            return "crossref"
        return None
    # {uni: [enu, exp, perk, fuzz]} of the CURRENT board (enu/exp None =
    # unknown; perk: None = bare, PERK_UNKNOWN = unmappable fed food; fuzz =
    # max unattributed +1-exp absorbs targeted at THIS pet). Seeded from the
    # entering board (previous end snapshot -- battle changes neither exp nor
    # the two faint-perks we care about) and updated on buys/merges/foods/
    # sells/pills. Exp is the absorbed-copy count; Lvl = 1 + (exp>=2) +
    # (exp>=5) (the snapshot Exp convention). Feeds sell levels, Pigeon crumb
    # counts, milk variants, level-up choice mints and pill spawn counts.
    # Negative keys are mid-turn spawn PLACEHOLDERS (pill tokens etc.): real
    # uni unknown, tracked so the possible-target scans stay complete.
    board = {u: list(ent) + [0] for u, ent in (board_pet_levels or {}).items()}
    board_slop = 0   # unattributed +1-exp absorbs that landed on SOME pet
    placeholder_n = 0


    pos = {u: s for u, s in (board_pet_slots or {}).items()}
    slot_certain = board_pet_slots is not None
    pos_mints: dict = {}   # {uni: (kind, enu)} minted from a certain position


    milk_fresh_buys = 0
    choco_fresh_buys = 0
    milk_variants: set = set()
    cow_stock_fired = False
    pigeon_stock_fired = False
    plain_null_buys: list = []
    sold_this_turn: set = set()
    unres_sell_unis: list = []


    crumb_buys_total = 0
    for _a in acts:
        if _a.get("Type") == 8 and food_enu_from_response(_a) == CRUMB_ENU:
            _cu = (action_request(_a).get("SpellId") or {}).get("Uni")
            if _cu is None or int(_cu) not in carry_map:
                crumb_buys_total += 1

    def occ_at(slot):
        for pu, ps in pos.items():
            if pu > 0 and ps == slot:
                return pu
        return None


    _end_snap_a = next((a for a in acts if a.get("Type") == 0), None)
    _end_snap_exp = ({b["uni"]: b["exp"] for b in snapshot_board_pets(_end_snap_a)
                      if b["uni"] is not None} if _end_snap_a is not None else {})
    _entering_exp = {u2: e2[1] for u2, e2 in (board_pet_levels or {}).items()}
    _bo_by_index = [board_orders_slots(action_request(a)) for a in acts]

    def recovery_vetoed(bought, occ, act_i):
        """(reason|None) veto a position recovery of buy `bought` onto `occ` at
        action index `act_i` -- see the two vetoes documented just above."""
        for bo2 in _bo_by_index[act_i + 1:]:
            if bought in bo2:
                return "later_batch"
        eo, do = _entering_exp.get(occ), _end_snap_exp.get(occ)
        if eo is not None and do is not None and do <= eo:
            return "truth"
        return None

    chain = []
    rolls = []
    n_freeze_events = 0
    buy_by_uni = {}


    stream = []

    def emit_block(seq):
        if seq:
            stream.append({"kind": "block", "seq": seq})

    def emit_consume(n, why):
        stream.append({"kind": "consume", "n": n, "why": why})

    def lvl_of(exp):
        return None if exp is None else 1 + (1 if exp >= 2 else 0) + (1 if exp >= 5 else 0)

    def exp_range(ent):
        """Possible current-exp interval [lo, hi] of a tracked pet (fuzz =
        pet-targeted unknowns, board_slop = board-wide unknowns), or None."""
        if ent is None or ent[1] is None:
            return None
        return ent[1], min(5, ent[1] + ent[3] + board_slop)

    def lvl_range(ent):
        r = exp_range(ent)
        if r is None:
            return None
        return {lvl_of(e) for e in range(r[0], r[1] + 1)}

    def board_lvl(u, default=None):
        ent = board.get(u) if u is not None else None
        lv = lvl_of(ent[1]) if ent else None
        return lv if lv is not None else default

    def cow_level(u):
        """Cow level."""
        lr = lvl_range(board.get(u))
        return lr.pop() if lr is not None and len(lr) == 1 else None

    def ups_for(pre, add):
        new = min(5, pre + add)
        return (1 if pre < 2 <= new else 0) + (1 if pre < 5 <= new else 0)

    def add_placeholder(enu, perk=None):
        """Track a mid-turn board spawn (unknown uni) so possible-target and
        bystander scans stay complete. enu=-1 = known-not-special token."""
        nonlocal placeholder_n
        placeholder_n += 1
        board[-placeholder_n] = [enu, 0, perk, 0]

    def bump_exp(u, add, dbl2="no"):
        """dbl2 (review C5): when two LEVEL-2 pets board-merge, the real engine
        SUPPRESSES the level-up shop reward (engine.py
        'levelup_reward_suppressed:double_level2_combine') -- the crossing
        mints nothing. "yes" = both sides determinately L2 -> exact 0;
        "maybe" = a double-L2 scenario is possible but not certain -> any
        would-be crossing is an honest barrier, never a guessed 2; "no" =
        normal semantics (buys/chocolate/buy-merges: the absorbed copy is
        fresh L1, so only board merges can be dbl2)."""
        ent = board.get(u) if u is not None else None
        r = exp_range(ent)
        if ent is None or r is None or add is None:
            if ent is not None:
                ent[1] = None
            emit_consume(None, "levelup_unknown")
            return
        alo, ahi = add if isinstance(add, tuple) else (add, add)
        ups_set = {ups_for(pre, a)
                   for pre in range(r[0], r[1] + 1) for a in range(alo, ahi + 1)}
        ent[1] = min(5, ent[1] + alo)
        ent[3] += ahi - alo   # widened uncertainty sticks to the destination
        if ups_set == {0}:
            return
        if dbl2 == "yes":
            return          # offer suppressed by the engine: exact 0, no mint
        if dbl2 == "maybe":
            emit_consume(None, "levelup_dbl2_ambiguous")
        elif ups_set == {1}:
            emit_consume(2, "levelup")
        elif ups_set == {2}:
            # two boundaries in one action -- offer semantics uncalibrated
            emit_consume(None, "levelup_double")
        else:
            emit_consume(None, "levelup_unknown")

    def pill_spawn_count(aim):
        """(n, why) a shop-phase pill on `aim` would mint: species tokens
        (Cricket/Sheep/Deer/Spider fixed, Rooster level-scaled), +1 Bee for a
        Honey perk, +1 self-copy for a Mushroom perk. A bystander Fly (or any
        bystander whose species we cannot name) and faint-damage cascades are
        uncalibrated -> (None, why)."""
        ent = board.get(aim) if aim is not None else None
        if ent is None or ent[0] is None:
            return None, "pill_target_unknown"
        enu, _exp, perk = ent[0], ent[1], ent[2]
        if enu in CASCADE_ON_PILL or enu == RAT_ENU:
            return None, "pill_cascade"
        if any(u2 != aim and e2[0] == FLY_ENU for u2, e2 in board.items()):
            return None, "pill_with_fly"
        if any(u2 != aim and e2[0] is None for u2, e2 in board.items()):
            return None, "pill_bystander_unknown"
        n = SPAWN_ON_PILL.get(enu, 0)
        if enu in SPAWN_ON_PILL_BY_LVL:
            lvls = lvl_range(ent)
            if lvls is None or len(lvls) != 1:
                return None, "pill_spawner_lvl_unknown"
            n += SPAWN_ON_PILL_BY_LVL[enu] * lvls.pop()
        if perk == PERK_UNKNOWN:
            return None, "pill_perk_unknown"
        if perk in (PERK_HONEY, PERK_MUSHROOM):
            n += 1
        return n, ("pill_spawn" if n else "pill_no_spawn")

    def pill_spawn_event(aim):
        n, why = pill_spawn_count(aim)
        emit_consume(n, why)
        if n:   # track what landed on the board (uni unknown -> placeholders)
            ent = board.get(aim)
            enu, perk = (ent[0], ent[2]) if ent else (None, None)
            if perk == PERK_MUSHROOM:
                add_placeholder(enu)          # self-copy respawn, same species
            if perk == PERK_HONEY:
                add_placeholder(4)            # Bee token
            if enu in SPAWN_ON_PILL or enu in SPAWN_ON_PILL_BY_LVL:
                tok = None if enu == 74 else -1   # Spider's token species is RNG
                for _ in range(SPAWN_ON_PILL.get(enu, 0) or 0):
                    add_placeholder(tok)
                if enu in SPAWN_ON_PILL_BY_LVL:
                    lvls = lvl_range(ent)
                    for _ in range((lvls.pop() if lvls and len(lvls) == 1 else 0)):
                        add_placeholder(-1)

    def levelup_possible_anywhere(add=1):
        """Could a +`add` exp absorb on ANY tracked pet cross a boundary?
        (True also when any pet's exp is unknown or the board is untracked.)"""
        if not board:
            return True
        for e2 in board.values():
            r2 = exp_range(e2)
            if r2 is None:
                return True
            if any(ups_for(pre, add) for pre in range(r2[0], r2[1] + 1)):
                return True
        return False

    # Tracked shop multisets for the buy-membership cross-check. None = state
    # unknown (turn-1 genesis, unresolved roll, or an unidentifiable buy made
    # the multiset stale -- conservative reset, so misses are never artifacts).
    shop_pets = Counter(start_shop["pets"]) if start_shop.get("pets") is not None else None
    shop_foods = Counter(start_shop["foods"]) if start_shop.get("foods") is not None else None
    if shop_foods is not None:
        for it in start_shop.get("injected") or []:    # T5 start-of-turn stocks
            shop_foods[it["enu"]] += 1

    def check_buy(kind: str, enu, src=None):
        nonlocal shop_pets, shop_foods
        membership[f"{kind}_buys"] += 1
        # T5: injection-ONLY foods (milk/chocomilk/crumb/apple2-3, see
        # INJ_FOOD_FAMILY) are never in the roll pool, so a buy of one is
        # self-evidently an ability injection -- explained without tracing the
        # trigger (the Cow/Pigeon/Worm buy may itself be an unresolved uni).
        # Plain Apple (enu 0) is a POOL food too, so it's NOT here -- it falls
        # through to the normal shop check (Worm-L1 apples are added to
        # shop_foods as start injections above).
        if kind == "food" and INJ_FOOD_FAMILY.get(enu):
            membership["food_checked"] += 1
            membership["food_ok"] += 1
            membership["food_ok_injected"] += 1
            return
        shop = shop_pets if kind == "pet" else shop_foods
        if enu is None or shop is None:
            if kind == "pet":
                shop_pets = None   # unknown item left the shop -> state stale
            else:
                shop_foods = None
            return
        membership[f"{kind}_checked"] += 1
        if shop[enu] > 0:
            membership[f"{kind}_ok"] += 1
            shop[enu] -= 1
            if src:

                membership[f"{kind}_ok_crossref"] += 1
                if src == "crossref-1dir":
                    membership[f"{kind}_ok_crossref_1dir"] += 1
                elif src == "crossref-1anchor":
                    membership[f"{kind}_ok_crossref_1anchor"] += 1
                elif src == "crossref-position":
                    membership[f"{kind}_ok_crossref_position"] += 1
                elif src == "crossref-milk":
                    membership[f"{kind}_ok_crossref_milk"] += 1
                elif src == "crossref-crumb":
                    membership[f"{kind}_ok_crossref_crumb"] += 1
        else:
            if kind == "food":
                membership["food_miss_enu"][enu] += 1
            if len(membership["miss_examples"]) < 5:
                membership["miss_examples"].append(
                    {"where": where, "kind": kind, "enu": enu,
                     "shop": sorted(shop.elements())})

    def cow_stock_event(family: str, cow_lvl=1):
        """Cow bought / fed Chocolate: the UNFROZEN food shop is replaced by
        2x (Choco)Milk_cowlevel (free). Frozen foods survive (T6 law: only
        consumption clears freeze). Emits the chain op for the artifact, resets
        the tracked food multiset so a post-stock POOL-food buy is honestly
        flagged (the milk buys themselves match by enu), and registers the 2
        milk as an injection block so their Unis resolve in the anchoring pass.
        cow_lvl=None = the cow's level (hence the milk VARIANT) is ambiguous
        (e.g. an ambiguous combine target, review C4): the 2 mints are counted
        exactly but content-unknown -- a consume, not a block, so chaining
        survives without guessing an enu."""
        nonlocal shop_foods, cow_stock_fired
        cow_stock_fired = True
        if shop_foods is not None:
            ids, unresolved = resolve_frozen_ids(cur_frozen, carry_map, uni_map)
            shop_foods = None if unresolved else Counter(ids["foods"])
        if cow_lvl is None:
            emit_consume(2, f"{family}_variant_unknown")
        else:
            table = CHOCOMILK_BY_LVL if family == "chocomilk" else MILK_BY_LVL
            enu = table.get(cow_lvl, table[1])
            emit_block([("food", enu), ("food", enu)])
        chain.append({"op": "ability_stock", "source": "Cow", "family": family,
                      "count": 2, "replaces_foods": True})

    for act_idx, a in enumerate(acts):
        ty = a.get("Type")
        req = action_request(a)

        bf = board_freezes_frozen_unis(req)
        if bf is not None:
            newly, gone = sorted(bf - cur_frozen), sorted(cur_frozen - bf)
            if newly:
                chain.append({"op": "freeze", "unis": newly})
                n_freeze_events += 1
            if gone:
                chain.append({"op": "unfreeze", "unis": gone})
                n_freeze_events += 1
            cur_frozen = set(bf)

        bo = board_orders_slots(req)
        # H4: apply this request's BoardOrders BEFORE the op (empirically the
        # PRE-action full-state positions of the current board -- the sold/acted
        # uni appears in its own batch at its slot). A batch that covers every
        # tracked board pet is a full-state enumeration: trust it wholesale
        # (dropping stale unlisted unis) and RESTORE certainty; a partial batch
        # only refreshes the unis it names.
        if bo:
            board_positive = {u2 for u2 in board if u2 > 0}
            if board_positive <= set(bo):
                pos = {u2: s2 for u2, s2 in bo.items()}
                slot_certain = True
            else:
                for u2, s2 in bo.items():
                    pos[u2] = s2
        op = None
        after_op = None

        if ty == 4:
            op = {"op": "start_turn", "seed": action_seed(a)}
        elif ty == 5:
            seed = action_seed(a)
            ids, unresolved = resolve_frozen_ids(cur_frozen, carry_map, uni_map)
            roll = {"seed": seed, "frozen_unis": sorted(cur_frozen), "unresolved": unresolved,
                    "frozen_pets": None, "frozen_foods": None,
                    "pets": None, "foods": None, "provenance": "frozen-unresolved"}
            if isinstance(seed, int) and unresolved == 0:
                r = reconstruct_shop(seed, turn, frozen_pets=ids["pets"],
                                      frozen_foods=ids["foods"], pools=pools)
                roll.update(frozen_pets=ids["pets"], frozen_foods=ids["foods"],
                            pets=r["pets"], foods=r["foods"], provenance="frozen-tracked")
                shop_pets, shop_foods = Counter(r["pets"]), Counter(r["foods"])
                # drawn (non-frozen) slice of this roll, for the anchoring pass
                emit_block([("pet", e) for e in r["pets"][len(ids["pets"]):]] +
                           [("food", e) for e in r["foods"][len(ids["foods"]):]])
            else:
                shop_pets = shop_foods = None
                # contents unknown, but the DRAW COUNT is not: capacity minus
                # the frozen carries (frozen items keep their unis)
                t = str(shop_tier(turn))
                cap = pools["pet_capacity"][t] + pools["food_capacity"][t]
                emit_consume(max(0, cap - len(cur_frozen)), "roll_unresolved")
            rolls.append(roll)
            op = {"op": "roll", "seed": seed}
        elif ty == 6:
            u = (req.get("MinionId") or {}).get("Uni")
            u = int(u) if u is not None else None
            slot = point_to_slot(req)
            if u is not None and u in board_unis:
                op = {"op": "move", "uni": u, "to_slot": slot}
                pos[u] = slot   # H4: dragged pet's new position (0 in v45 cache)
            else:
                ent = uni_map.get(u) if u is not None else None
                if ent is not None and ent[0] != "pet":
                    # H1 (review C1): a Type-6 buy consumes a PET shop slot; a
                    # map entry claiming food is a wrong pin, never a species.
                    membership["op_kind_conflict"] += 1
                    ent = None
                enu = ent[1] if ent else None
                src = id_src(u) if ent else None


                combine_real: list = []
                combine_ph: list = []
                combine_unk: list = []
                if u in orphan_combine_unis and enu is not None:
                    combine_real = [bu for bu, eb in board.items()
                                    if bu > 0 and eb[0] == enu]
                    combine_ph = [bu for bu, eb in board.items()
                                  if bu < 0 and eb[0] in (enu, None)]
                    # a REAL board pet whose species is unresolved at scan
                    # time could be the same species too (verify-round
                    # finding: 6 full-cache cases, 2 proven wrong picks) --
                    # it blocks a "confident" target exactly like a
                    # placeholder does.
                    combine_unk = [bu for bu, eb in board.items()
                                   if bu > 0 and eb[0] is None]


                occ = None
                if u in orphan_combine_unis and slot_certain:
                    o = occ_at(slot)
                    if o is not None and o > 0 and o in board and o != u:
                        occ = o
                recov_onto = recov_enu = recov_mint = None
                recov_conflict = False
                if occ is not None:
                    occ_enu = board[occ][0]
                    occ_lvls = lvl_range(board.get(occ))
                    lvl_ok = occ_lvls is None or any(lv < 3 for lv in occ_lvls)
                    if enu is not None:
                        if occ_enu == enu and lvl_ok:
                            recov_onto, recov_enu = occ, enu
                        elif occ_enu is None and lvl_ok:
                            # the combine PROVES occ's species == enu
                            recov_onto, recov_enu = occ, enu
                            recov_mint = (occ, ("pet", enu))
                        elif occ_enu is not None and occ_enu != enu:
                            recov_conflict = True
                    elif occ_enu is not None and lvl_ok:
                        # bought species unknown -> the combine PROVES the BUY
                        # was occ's species (engine same-species legality)
                        recov_onto, recov_enu = occ, occ_enu
                        recov_mint = (u, ("pet", occ_enu))


                if recov_onto is not None:
                    _veto = recovery_vetoed(u, recov_onto, act_idx)
                    if _veto is not None:
                        pos_stats["position_veto_later_batch" if _veto == "later_batch"
                                  else "position_veto_truth"] += 1
                        recov_onto = recov_enu = recov_mint = None
                if len(combine_real) == 1 and not combine_ph and not combine_unk:
                    combine_uni = combine_real[0]
                    if occ is not None and occ != combine_uni:
                        # lane 2c tripwire: a CERTAIN tracker disagrees with the
                        # lone same-species candidate. Never-guess: fall back to
                        # the ambiguous barrier and count it (expected ~0 -- a
                        # tracker-quality signal, investigate before shipping if
                        # it grows).
                        pos_stats["position_confident_disagree"] += 1
                        op = {"op": "buy_combine", "uni": u, "enu": enu,
                              "onto_uni": None, "to_slot": slot,
                              "combine_target_ambiguous": True}
                        if src:
                            op["identity_src"] = src
                        check_buy("pet", enu, src)
                        r_c = exp_range(board.get(combine_uni))
                        if r_c is None or any(ups_for(pre, 1)
                                              for pre in range(r_c[0], r_c[1] + 1)):
                            emit_consume(None, "combine_target_unknown")
                        board[combine_uni][3] += 1
                        if enu == COW_ENU:
                            cow_stock_event("milk", None)
                    else:
                        if occ is not None:
                            pos_stats["position_confident_agree"] += 1
                        bump_exp(combine_uni, 1)   # may mint a choice pair (+2 unis)
                        op = {"op": "buy_combine", "uni": u, "enu": enu,
                              "onto_uni": combine_uni, "to_slot": slot}
                        if src:
                            op["identity_src"] = src
                        check_buy("pet", enu, src)
                        if enu == COW_ENU:          # buying a Cow (even to combine) = ThisBought
                            cow_stock_event("milk", cow_level(combine_uni))
                elif recov_onto is not None:
                    # position recovery of an AMBIGUOUS or orphan-unknown
                    # combine: exact bump_exp REPLACES the combine_target_unknown
                    # barrier; the candidates keep no fuzz for this event; a
                    # mint (occ's proven species, or the bought pet's) rides the
                    # rounds loop (crossref-position).
                    if recov_mint is not None:
                        mu, ment = recov_mint
                        pos_mints[mu] = ment
                        if mu == occ:
                            board[occ][0] = ment[1]   # species now known in-turn
                    bump_exp(recov_onto, 1)
                    op = {"op": "buy_combine", "uni": u, "enu": recov_enu,
                          "onto_uni": recov_onto, "to_slot": slot,
                          "combine_target_recovered": True}
                    s_r = id_src(u) if recov_enu is not None else None
                    if s_r:
                        op["identity_src"] = s_r
                    check_buy("pet", recov_enu, s_r)
                    if recov_enu == COW_ENU:
                        cow_stock_event("milk", cow_level(recov_onto))
                    pos_stats["position_recovered"] += 1
                    pos_stats["position_ambiguous_recovered"
                              if (combine_real or combine_ph)
                              else "position_orphan_recovered"] += 1
                elif combine_real or combine_ph:
                    # ambiguous combine target: it IS a combine, but WHICH
                    # same-species stack absorbed the copy is a guess we
                    # refuse to make (review C4: 64% wrong when adjudicable).
                    # H4 could not recover it here (tracker uncertain, slot
                    # empty, or a species contradiction -> honest barrier).
                    if recov_conflict:
                        pos_stats["position_species_conflict"] += 1
                    cands = combine_real + combine_ph + combine_unk
                    op = {"op": "buy_combine", "uni": u, "enu": enu,
                          "onto_uni": None, "to_slot": slot,
                          "combine_target_ambiguous": True}
                    if src:
                        op["identity_src"] = src
                    check_buy("pet", enu, src)
                    # stream: the +1 exp landed on ONE of the candidates. If
                    # no candidate could cross a level boundary the mint count
                    # is provably 0 (each candidate keeps a fuzz upper bound);
                    # otherwise an honest barrier.
                    could_mint = False
                    for bu in cands:
                        r_c = exp_range(board.get(bu))
                        if r_c is None or any(ups_for(pre, 1)
                                              for pre in range(r_c[0], r_c[1] + 1)):
                            could_mint = True
                    if could_mint:
                        emit_consume(None, "combine_target_unknown")
                    for bu in cands:      # each MAY have absorbed the +1 exp
                        board[bu][3] += 1
                    if enu == COW_ENU:    # ThisBought fires whatever the target
                        lvls: set = set()
                        for bu in cands:
                            lr = lvl_range(board.get(bu))
                            lvls |= lr if lr else {None}
                        cow_stock_event("milk", lvls.pop()
                                        if (None not in lvls and len(lvls) == 1)
                                        else None)
                elif u in orphan_combine_unis:


                    if recov_conflict:
                        pos_stats["position_species_conflict"] += 1
                    op = {"op": "buy_pet", "uni": u, "enu": enu, "to_slot": slot}
                    if src:
                        op["identity_src"] = src
                    check_buy("pet", enu, src)
                    if enu == COW_ENU:
                        cow_stock_event("milk", 1)
                    if enu is not None:
                        # species known but target untracked: mint unknowable
                        emit_consume(None, "combine_target_unknown")
                    elif (any(e2[0] in (COW_ENU, None) for e2 in board.values())
                          or levelup_possible_anywhere(1)):
                        emit_consume(None, "combine_target_unknown")
                    else:
                        board_slop += 1
                else:
                    # plain buy: the bought pet drops on `slot`. If that slot
                    # was OCCUPIED the insertion pushes neighbours in an
                    # unmodeled direction (~6% of buys) -> poison position
                    # certainty until the next full-state BoardOrders batch.
                    occupied = any(ps == slot for ps in pos.values())
                    op = {"op": "buy_pet", "uni": u, "enu": enu, "to_slot": slot}
                    if src:
                        op["identity_src"] = src
                    if u is not None:
                        board_unis.add(u)
                        buy_by_uni[u] = op
                        board[u] = [enu, 0, None, 0]   # a fresh buy: level 1, bare
                        pos[u] = slot
                        if enu is None:


                            plain_null_buys.append(u)
                    check_buy("pet", enu, src)
                    if enu == COW_ENU:
                        cow_stock_event("milk", 1)
                    if occupied:
                        slot_certain = False
        elif ty == 7:
            su = (req.get("SourceMinionId") or {}).get("Uni")
            du = (req.get("TargetMinionId") or {}).get("Uni")
            su = int(su) if su is not None else None
            du = int(du) if du is not None else None
            buy_merge = su is not None and su not in board_unis
            op = {"op": "merge", "src_uni": su, "dst_uni": du, "buy_merge": buy_merge}
            # A merge is same-species by rule -> the destination board pet's
            # identity is also the source's. Use it (a) to name a buy-merge whose
            # fresh shop Uni never snapshots, and (b) to BACK-FILL a same-turn
            # buy_pet that is now being merged away (its Uni likewise never
            # reaches a snapshot). Both are the dominant unresolved-identity case.
            dent = uni_map.get(du) if du is not None else None
            dst_enu = dent[1] if dent and dent[0] == "pet" else None
            if du is not None and du in board and board[du][0] is None and dst_enu is not None:
                board[du][0] = dst_enu   # dst identity known from the truth map
            if buy_merge:
                ent = uni_map.get(su)
                if ent is not None and ent[0] != "pet":
                    # H1 (review C1): a Type-7 source is a PET shop slot.
                    membership["op_kind_conflict"] += 1
                    ent = None
                src_enu = ent[1] if ent else dst_enu
                op["src_enu"] = src_enu
                s7 = id_src(su) if ent else None
                if s7:
                    op["identity_src"] = s7
                check_buy("pet", src_enu, s7)
                bump_exp(du, 1)   # the bought shop copy is fresh: +1 absorbed exp
                if src_enu == COW_ENU or dst_enu == COW_ENU:
                    # buying INTO a cow stack is still a Buy; the post-merge
                    # cow level picks the milk enu
                    cow_stock_event("milk", cow_level(du))
            elif su is not None:
                src_ent = board.get(su)
                src_r = exp_range(src_ent)
                src_gain = None if src_r is None else (src_r[0] + 1, src_r[1] + 1)
                src_lvls = lvl_range(src_ent)
                prior = buy_by_uni.get(su)
                if prior is not None and prior.get("enu") is None and dst_enu is not None:
                    prior["enu"] = dst_enu   # retroactively name the consumed buy
                board_unis.discard(su)
                board.pop(su, None)
                pos.pop(su, None)   # H4: merged-away source leaves its slot
                # M1 (review C5): two L2 pets merging = the engine suppresses
                # the level-up offer ('double_level2_combine') -> the crossing
                # mints 0. Determinately both-L2 -> exact 0; possibly-L2 on
                # both sides -> barrier instead of a possibly-phantom +2.
                dst_lvls = lvl_range(board.get(du))
                if src_lvls == {2} and dst_lvls == {2}:
                    dbl2 = "yes"
                elif ((src_lvls is None or 2 in src_lvls)
                        and (dst_lvls is None or 2 in dst_lvls)):
                    dbl2 = "maybe"
                else:
                    dbl2 = "no"
                # a board merge absorbs the source's copies (its exp + 1)
                bump_exp(du, src_gain, dbl2=dbl2)
        elif ty == 8:
            u = (req.get("SpellId") or {}).get("Uni")
            u = int(u) if u is not None else None
            enu = food_enu_from_response(a)


            if enu in MILK_ENUS and (u is None or u not in cur_frozen):
                milk_fresh_buys += 1
                milk_variants.add(enu)
            elif enu in CHOCOMILK_ENUS and (u is None or u not in cur_frozen):
                choco_fresh_buys += 1
            if enu is None and u is not None:
                ent = uni_map.get(u)
                if ent is not None and ent[0] != "food":
                    # H1 (review C1): a Type-8 spell slot is a FOOD.
                    membership["op_kind_conflict"] += 1
                    ent = None
                enu = ent[1] if ent else None
            aim = (req.get("Aim") or {}).get("Uni")
            aim = int(aim) if aim is not None else None
            op = {"op": "buy_food", "uni": u, "enu": enu,
                  "target_uni": aim}
            s8 = id_src(u) if enu is not None else None
            if s8:
                op["identity_src"] = s8
            check_buy("food", enu, s8)
            if enu == CHOCOLATE_ENU and aim is not None:
                bump_exp(aim, 1)


                aim_ent_b = board.get(aim)
                if aim_ent_b is not None and aim_ent_b[0] == COW_ENU:
                    # Chocolate fed to a Cow; post-bump level picks the variant
                    cow_stock_event("chocomilk", cow_level(aim))
            elif enu == PILL_ENU:
                pill_spawn_event(aim)   # spawn tokens consume counter values
                if aim is not None:
                    board_unis.discard(aim)   # the pilled pet faints
                    board.pop(aim, None)
                    pos.pop(aim, None)
                slot_certain = False   # H4: faint + spawns land at unmodeled slots
            elif enu is None and aim is not None:
                # unknown targeted food: could be a chocolate (exp -> possible
                # choice mint), an equipable (perk), or even a pill (spawns).
                # Provably-quiet iff a chocolate could not cross a boundary,
                # the target is not (possibly) a Cow, and a hypothetical pill
                # would spawn nothing.
                # H4: an unknown food on a target could be a pill (faint +
                # spawns) -> we cannot model the resulting positions -> poison.
                slot_certain = False
                ent_t = board.get(aim)
                r_t = exp_range(ent_t)
                quiet = (ent_t is not None and r_t is not None
                         and ent_t[0] not in (COW_ENU, None)
                         and not any(ups_for(pre, 1) for pre in range(r_t[0], r_t[1] + 1))
                         and pill_spawn_count(aim)[0] == 0)
                if quiet:
                    ent_t[3] += 1               # may have been chocolate
                    ent_t[2] = PERK_UNKNOWN     # may have been an equipable
                else:
                    if ent_t is not None:
                        ent_t[1] = None
                        ent_t[2] = PERK_UNKNOWN
                    emit_consume(None, "food_unknown_fed")
            elif aim is not None and aim in board:
                if enu in PERK_BY_SPELL:
                    board[aim][2] = PERK_BY_SPELL[enu]
                elif enu not in NONPERK_SPELLS:
                    board[aim][2] = PERK_UNKNOWN
        elif ty == 9:
            u = (req.get("Minion") or {}).get("Uni")
            u = int(u) if u is not None else None
            ent = uni_map.get(u) if u is not None else None
            if ent is not None and ent[0] != "pet":
                # H1 (review C1): a Type-9 sell removes a board PET; a map
                # entry claiming food is a wrong pin -- never a species.
                membership["op_kind_conflict"] += 1
                ent = None
            ent_b = board.get(u) if u is not None else None
            # species: exact tracker first, truth map fallback
            sold_enu = ent_b[0] if (ent_b and ent_b[0] is not None) else (
                ent[1] if ent else None)
            lvls_b = lvl_range(ent_b) if ent_b else None
            sold_lvl = lvls_b.pop() if lvls_b is not None and len(lvls_b) == 1 else None
            # op level: tracked exact, else legacy "identity known -> level 1"
            sold_lvl_op = sold_lvl if sold_lvl is not None else (
                1 if sold_enu is not None else None)
            op = {"op": "sell", "uni": u, "enu": ent[1] if ent else None,
                  "level": sold_lvl_op}   # tracked level (sell gold = level)
            s9 = id_src(u) if ent else None
            if s9:
                op["identity_src"] = s9
            if u is not None:
                board_unis.discard(u)
                board.pop(u, None)
                pos.pop(u, None)
                sold_this_turn.add(u)


            if sold_enu == PIGEON_ENU:


                if u is not None and u in crossref_crumb_unis:
                    count_ok = (sold_lvl is not None
                                and crumb_buys_total <= sold_lvl
                                and crumb_buys_total + end_crumb_leftover == sold_lvl)
                else:
                    count_ok = sold_lvl is not None
                if count_ok:
                    n_crumb = sold_lvl
                    emit_block([("food", CRUMB_ENU)] * n_crumb)
                else:
                    n_crumb = sold_lvl_op or 1   # legacy artifact count
                    emit_consume(None, "crumbs_unknown")
                pigeon_stock_fired = True
                after_op = {"op": "ability_stock", "source": "Pigeon",
                            "family": "crumb", "count": n_crumb, "replaces_foods": False}
            elif sold_enu is None and u is not None:
                # could be an untracked Pigeon -> crumb consumption unknown
                emit_consume(None, "sell_species_unknown")
                unres_sell_unis.append(u)
        elif ty == 11:
            op = {"op": "end_turn"}

        if op is not None:
            if bo:
                op["board_orders"] = bo
            chain.append(op)
        if after_op is not None:
            chain.append(after_op)

        cu = consumed_shop_uni(ty, req)
        if cu is not None:
            cur_frozen.discard(cu)


    milk_mints: dict = {}
    crumb_mints: dict = {}
    # MILK: exactly one cow-stock worth of FRESH Milk-49 buys (level-1 variant,
    # a plain fresh Cow buy) with NO resolved Cow stock this turn, and exactly
    # one enu-unresolved plain buy that is SOLD this turn (the classic buy-Cow
    # -> stock milk -> sell-Cow hole). Level-2/3 variants imply a combine onto a
    # stack -- left to H4's position proof, never claimed here.
    if (not cow_stock_fired and choco_fresh_buys == 0
            and milk_fresh_buys == 2 and milk_variants == {MILK_BY_LVL[1]}):
        cands = {mu for mu in plain_null_buys if mu in sold_this_turn}
        if len(cands) == 1:
            milk_mints[next(iter(cands))] = ("pet", COW_ENU)


    if (not pigeon_stock_fired and crumb_buys_total >= 1
            and len(set(unres_sell_unis)) == 1):
        crumb_mints[unres_sell_unis[0]] = ("pet", PIGEON_ENU)

    return {"chain": chain, "rolls": rolls,
            "final_frozen_unis": sorted(cur_frozen), "n_freeze_events": n_freeze_events,
            "stream": stream,


            "pos_mints": pos_mints, "final_slots": dict(pos),
            "slots_certain": slot_certain, "pos_stats": pos_stats,

            "milk_mints": milk_mints, "crumb_mints": crumb_mints}


def anchor_uni_blocks(stream: list, uni_map: dict,
                      action_kinds: dict | None = None) -> tuple:
    """Pin each drawn block's base uni against the snapshot-truth map.

    ACCURACY: every chain prediction that lands on an INDEPENDENTLY
    pass-1-anchored block doubles as a holdout -- chain_pred_{pure,counted}
    count predictions (pure = zero-consumption gaps, counted = crossed a
    minting event), *_wrong count disagreements: the direct wrong-mint bound
    for chained identities. chain_pred_xturn splits out predictions that
    crossed a turn boundary. The calibration sweep ignores CHAIN_BARRIER_WHYS
    (it must MEASURE unproven subtypes); only the pinning pass honours them.

    action_kinds ({uni: "pet"|"food"}, review C1): the ACTION TYPE that used a
    uni implies its kind (T6/7/9 = pet, T8 = food) -- independent evidence that
    VETOES any candidate base placing a contradicting kind on that uni (pass 1,
    pass 2 and re-basing alike).

    stream entries: {"kind": "block", "seq": [(kind, enu), ...]} |
    {"kind": "consume", "n": int|None, "why": str} (extra tags pass through).
    Block entries are annotated in place with "base"/"src" ("anchor"|"chain").
    Returns (crossref: {uni: (kind, enu)} for unis NOT in uni_map, stats)."""
    idx = defaultdict(list)
    for u, ent in uni_map.items():
        idx[ent].append(u)
    action_kinds = action_kinds or {}

    def kinds_ok(base, seq):
        return all(action_kinds.get(base + i) in (None, ent[0])
                   for i, ent in enumerate(seq))

    entries = [e for e in stream if e["kind"] == "consume" or e.get("seq")]

    stats = {"blocks": 0, "anchored": 0, "anchored_multi": 0, "anchored_single": 0,
             "ambiguous": 0, "no_anchor": 0,
             "holdout_n": 0, "holdout_robust": 0, "holdout_wrong": 0,
             "chain_pred_pure": 0, "chain_pred_pure_wrong": 0,
             "chain_pred_counted": 0, "chain_pred_counted_wrong": 0,
             "chain_pred_xturn": 0, "chain_pred_xturn_wrong": 0,
             # M4 (review C8): endpoint-quality split -- _mm = source AND
             # target are multi-anchor (measures the chained arithmetic),
             # _s1 = at least one endpoint is single-anchor (dominated by
             # single-anchor base placement noise, NOT chain error).
             "chain_pred_pure_mm": 0, "chain_pred_pure_mm_wrong": 0,
             "chain_pred_pure_s1": 0, "chain_pred_pure_s1_wrong": 0,
             "chain_pred_counted_mm": 0, "chain_pred_counted_mm_wrong": 0,
             "chain_pred_counted_s1": 0, "chain_pred_counted_s1_wrong": 0,
             "chain_pred_xturn_mm": 0, "chain_pred_xturn_mm_wrong": 0,
             "chain_pred_xturn_s1": 0, "chain_pred_xturn_s1_wrong": 0,
             "chain_pinned_pure": 0, "chain_pinned_counted": 0,
             "chain_disambiguated": 0, "chain_conflict": 0,
             "chain_content_skip": 0, "crossref_chained": 0,
             "chain_holdout2": 0, "chain_holdout2_wrong": 0,
             "chain_1dir_skipped": 0, "chain_pinned_1dir": 0,
             "chain_vs_1anchor": 0, "chain_vs_1anchor_disagree": 0,
             # H1/H2/H3 additions (review C1/C2/C3)
             "p1_kind_veto": 0, "chain_kind_veto": 0,
             "anchor1_line_agree": 0, "anchor1_rebased": 0, "anchor1_dropped": 0,
             "chain_overlap_evict": 0, "overlap_anchor_dropped": 0,
             "crossref_minted_1anchor": 0, "minted_positions": 0}


    prev_end = 0
    for e in entries:
        if e["kind"] != "block":
            continue
        seq = e["seq"]
        stats["blocks"] += 1
        cands = {u - i for i, ent in enumerate(seq) for u in idx.get(tuple(ent), ())}
        valid = [b for b in cands if b > prev_end
                 and all(uni_map.get(b + i) in (None, tuple(ent)) for i, ent in enumerate(seq))]
        # H1 (review C1): action-type kind evidence vetoes candidate bases
        kept = [b for b in valid if kinds_ok(b, seq)]
        if kept != valid:
            stats["p1_kind_veto"] += 1
        valid = kept
        if not valid:
            stats["no_anchor"] += 1
            e["p1"] = "no_anchor"
            continue
        if len(valid) > 1:
            stats["ambiguous"] += 1
            e["p1"] = "ambiguous"
            continue
        base = valid[0]
        # how many of this block's positions are pinned by an actual snapshot uni?
        n_anchor = sum(1 for i, ent in enumerate(seq) if base + i in uni_map)
        stats["anchored_multi" if n_anchor >= 2 else "anchored_single"] += 1
        e["multi"] = n_anchor >= 2   # order-validated anchor (calibration split)
        # holdout accuracy: on multi-anchor blocks, drop one snapshot uni and
        # confirm re-anchoring reproduces its identity from the rest.
        if n_anchor >= 2:
            pinned = [i for i, ent in enumerate(seq) if base + i in uni_map]
            drop = pinned[0]
            rest = {u - i for i, ent in enumerate(seq) if i != drop
                    for u in idx.get(tuple(ent), ())}
            rvalid = [b for b in rest if b > prev_end
                      and all(uni_map.get(b + i) in (None, tuple(ent))
                              for i, ent in enumerate(seq) if i != drop)]
            stats["holdout_n"] += 1
            if len(rvalid) == 1 and rvalid[0] == base:
                stats["holdout_robust"] += 1     # over-determined: still uniquely == base
            elif len(rvalid) == 1 and rvalid[0] != base:
                stats["holdout_wrong"] += 1       # would MINT a wrong base -> real error signal
            # len(rvalid) > 1 -> ambiguous -> would be SKIPPED (conservative), not wrong
        e["base"] = base
        e["src"] = "anchor"
        stats["anchored"] += 1
        prev_end = base + len(seq) - 1

    # ---- calibration sweep: chain predictions vs independent pass-1 anchors.
    # Raw n (barriers ignored): unproven subtypes must be measured too. An
    # unanchored block's LENGTH is exact (drawn count), so the frontier walks
    # straight through it -- only unknown-count consumes break the arithmetic.
    nxt = None          # next counter value if fully chained from the left
    counted = False     # crossed a >0 consume since the source anchor
    xturn = False       # crossed a turn boundary since the source anchor
    src_multi = False   # was the lane's SOURCE anchor multi-anchored? (M4/C8)
    for e in entries:
        if e["kind"] == "consume":
            nxt = None if (e["n"] is None or nxt is None) else nxt + e["n"]
            if nxt is not None:
                counted = counted or e["n"] > 0
                xturn = xturn or e.get("why") == "turn_boundary"
            continue
        if e.get("src") == "anchor":
            if nxt is not None:
                lane = "counted" if counted else "pure"
                # M4 (review C8): split by ENDPOINT quality -- only the
                # multi->multi lane measures the chained arithmetic; a
                # single-anchor endpoint mostly measures its own base noise.
                q = "mm" if (src_multi and e.get("multi")) else "s1"
                wrong = e["base"] != nxt
                stats[f"chain_pred_{lane}"] += 1
                stats[f"chain_pred_{lane}_{q}"] += 1
                if wrong:
                    stats[f"chain_pred_{lane}_wrong"] += 1
                    stats[f"chain_pred_{lane}_{q}_wrong"] += 1
                if xturn:
                    stats["chain_pred_xturn"] += 1
                    stats[f"chain_pred_xturn_{q}"] += 1
                    if wrong:
                        stats["chain_pred_xturn_wrong"] += 1
                        stats[f"chain_pred_xturn_{q}_wrong"] += 1
            nxt = e["base"] + len(e["seq"])
            counted = xturn = False
            src_multi = bool(e.get("multi"))
        elif nxt is not None:
            nxt += len(e["seq"])

    # ---- pass 2: STRICT double-confirmed chained pinning ----
    # Frontier sources are MULTI-anchor blocks only (pair calibration: single-
    # anchor bases are order-unvalidated and measurably off; they pass through
    # by LENGTH, their base is neither trusted nor extended). A block is
    # pinned ONLY when the forward and backward arithmetic BOTH reach it and
    # AGREE: an unmodeled mint on either side shifts exactly one direction,
    # so disagreement (-> skip) filters the residual error sources the
    # calibration can't yet name (rare conditional offers around tier-unlock
    # boundaries etc.). Truth-agreement + overlap guards apply on top.
    def consume_n(e):
        if e["n"] is None or e.get("why") in CHAIN_BARRIER_WHYS:
            return None
        return e["n"]

    def is_source(e):
        return e.get("src") == "anchor" and e.get("multi")

    fwd, bwd = {}, {}          # id(block) -> (pred_base, crossed_counted)
    fwd_at, bwd_at = {}, {}    # arriving prediction AT a source block (holdout)
    frontier = None
    counted = False
    for e in entries:
        if e["kind"] == "consume":
            n = consume_n(e)
            frontier = None if (n is None or frontier is None) else frontier + n
            if frontier is not None and n > 0:
                counted = True
            continue
        L = len(e["seq"])
        if is_source(e):
            if frontier is not None:
                fwd_at[id(e)] = (frontier, counted)
            frontier = e["base"] + L
            counted = False
        else:
            if frontier is not None:
                fwd[id(e)] = (frontier, counted)
                frontier += L
            # frontier stays None if it was None (no source yet / barrier)
    frontier = None
    counted = False
    for e in reversed(entries):
        if e["kind"] == "consume":
            n = consume_n(e)
            frontier = None if (n is None or frontier is None) else frontier - n
            if frontier is not None and n > 0:
                counted = True
            continue
        L = len(e["seq"])
        if is_source(e):
            if frontier is not None:
                bwd_at[id(e)] = (frontier - L, counted)
            frontier = e["base"]
            counted = False
        else:
            if frontier is not None:
                bwd[id(e)] = (frontier - L, counted)
                frontier -= L

    # double-confirmation holdout: multi-anchor blocks reached by BOTH
    # directions (excluding themselves) with agreeing arithmetic -- the
    # direct accuracy of exactly the confirmations pass 2 pins on.
    for e in entries:
        if e["kind"] != "block" or e.get("src") != "anchor":
            continue
        if e.get("multi"):
            f, b = fwd_at.get(id(e)), bwd_at.get(id(e))
            if f is not None and b is not None and f[0] == b[0]:
                stats["chain_holdout2"] += 1
                if f[0] != e["base"]:
                    stats["chain_holdout2_wrong"] += 1
        else:


            f, b = fwd.get(id(e)), bwd.get(id(e))
            if f is not None and b is not None and f[0] == b[0]:
                stats["chain_vs_1anchor"] += 1
                if f[0] != e["base"]:
                    stats["chain_vs_1anchor_disagree"] += 1
                    # H2 (review C2): the single snapshot sighting contradicts
                    # the double-confirmed line (583/584 adjudicated cases =
                    # the anchor wrong, the line right). Re-base onto the line
                    # where truth AND action-kind evidence agree there; else
                    # drop the pin entirely -- never mint from a measured-off
                    # base.
                    nb, seq = f[0], e["seq"]
                    if (nb > 0
                            and all(uni_map.get(nb + i) in (None, tuple(ent))
                                    for i, ent in enumerate(seq))
                            and kinds_ok(nb, seq)):
                        e["base"] = nb
                        e["src"] = "chain"
                        e["chain_counted"] = f[1] or b[1]
                        e["rebased_1anchor"] = True
                    else:
                        e.pop("base")
                        e["dead"] = "anchor_offline"
                        stats["anchor1_dropped"] += 1
                else:
                    # line-validated single anchor: same trust as the
                    # double-confirmed slice -> stays untagged
                    e["line_agree"] = True
                    stats["anchor1_line_agree"] += 1

    for e in entries:
        if e["kind"] != "block" or "base" in e or e.get("dead"):
            continue   # dead: e.g. an off-line single anchor dropped by H2
        f, b = fwd.get(id(e)), bwd.get(id(e))
        if f is None or b is None:
            if f is None and b is None:
                continue
            if not CHAIN_ALLOW_1DIR:
                stats["chain_1dir_skipped"] += 1
                continue
        elif f[0] != b[0]:
            e["dead"] = "conflict"
            continue
        base = (f or b)[0]
        seq = e["seq"]
        if base <= 0:
            e["dead"] = "conflict"
            continue
        if not all(uni_map.get(base + i) in (None, tuple(ent))
                   for i, ent in enumerate(seq)):
            e["dead"] = "content"
            continue
        if not kinds_ok(base, seq):
            # H1 (review C1): the action that consumed one of these unis
            # implies a kind that contradicts the pinned content -> veto.
            e["dead"] = "kind"
            continue
        e["base"] = base
        e["src"] = "chain"
        e["chain_counted"] = (f[1] if f else False) or (b[1] if b else False)
        if f is None or b is None:
            e["chain_1dir"] = True


    def _strength(e):
        if e.get("src") == "anchor":
            if e.get("multi"):
                return 5              # order self-validated by >=2 snapshots
            return 4 if e.get("line_agree") else 1   # unvalidated single anchor
        if e.get("rebased_1anchor") or not e.get("chain_1dir"):
            return 3                  # on the double-confirmed line
        return 2                      # 1-dir (tagged) chain pin

    cand_pins = [(i, e) for i, e in enumerate(entries)
                 if e["kind"] == "block" and "base" in e]
    acc_idx: list = []                # accepted stream indices, sorted
    acc_iv: dict = {}                 # stream index -> (start_uni, end_uni)

    def _fits(i, s, t):
        pos = bisect.bisect_left(acc_idx, i)
        if pos > 0 and s <= acc_iv[acc_idx[pos - 1]][1]:
            return False
        if pos < len(acc_idx) and t >= acc_iv[acc_idx[pos]][0]:
            return False
        return True

    def _drop(e):
        e.pop("base", None)
        e.pop("chain_counted", None)
        if e.get("src") == "anchor":
            e["dead"] = "overlap_anchor"   # lost to STRONGER evidence
        else:
            e["dead"] = "overlap"

    for stg in (5, 4, 3, 2, 1):
        group = [(i, e) for i, e in cand_pins
                 if "base" in e and _strength(e) == stg]
        fit = [(i, e, e["base"], e["base"] + len(e["seq"]) - 1) for i, e in group
               if _fits(i, e["base"], e["base"] + len(e["seq"]) - 1)]
        for i, e in group:                 # rejects vs stronger accepted pins
            if not any(i == fi for fi, _, _, _ in fit):
                _drop(e)


        involved = set()
        for x in range(len(fit)):
            i1, _, s1, t1 = fit[x]
            for y in range(x + 1, len(fit)):
                i2, _, s2, t2 = fit[y]
                lo, hi = (x, y) if i1 < i2 else (y, x)
                _, _, _, thi = fit[lo]
                _, _, slo, _ = fit[hi]
                if thi >= slo:             # stream order must match uni order
                    involved.add(x)
                    involved.add(y)
        for x, (i, e, s, t) in enumerate(fit):
            if x in involved:
                _drop(e)
            else:
                bisect.insort(acc_idx, i)
                acc_iv[i] = (s, t)

    for e in entries:
        if e["kind"] != "block":
            continue
        if e.get("src") == "chain" and "base" in e:
            if e.get("rebased_1anchor"):
                stats["anchor1_rebased"] += 1
                continue
            stats["chain_pinned_counted" if e.get("chain_counted") else "chain_pinned_pure"] += 1
            if e.get("chain_1dir"):
                stats["chain_pinned_1dir"] += 1
            if e.get("p1") == "ambiguous":
                stats["chain_disambiguated"] += 1
        elif e.get("dead") == "conflict":
            stats["chain_conflict"] += 1           # true fwd/bwd disagreement
        elif e.get("dead") == "overlap":
            stats["chain_overlap_evict"] += 1      # C9: report separately
        elif e.get("dead") == "overlap_anchor":
            stats["overlap_anchor_dropped"] += 1
        elif e.get("dead") == "content":
            stats["chain_content_skip"] += 1
        elif e.get("dead") == "kind":
            stats["chain_kind_veto"] += 1
        # dead == "anchor_offline" is counted at drop time (anchor1_dropped)

    crossref = {}
    for e in entries:
        if e["kind"] != "block" or "base" not in e:
            continue
        # unvalidated single-anchor mints are TAGGED (review C2): the risk
        # boundary follows measured risk, not mechanism.
        tag_1anchor = (e.get("src") == "anchor" and not e.get("multi")
                       and not e.get("line_agree"))
        for i, (kind, enu) in enumerate(e["seq"]):
            u = e["base"] + i
            if u not in uni_map:
                crossref[u] = (kind, enu)
                stats["minted_positions"] += 1     # C3 invariant: == len(crossref)
                if e["src"] == "chain":
                    stats["crossref_chained"] += 1
                if tag_1anchor:
                    stats["crossref_minted_1anchor"] += 1
    return crossref, stats


# --------------------------------------------------------------------------
# per-game decode
# --------------------------------------------------------------------------

def run_harvest_rounds(harvest, uni_map: dict, action_kinds: dict | None = None,
                       max_rounds: int = 4, pos_out: dict | None = None):
    """Adaptive harvest rounds. Each round's minted identities resolve sold
    pets / frozen unis / pill targets / combine identities in the RE-WALK
    (`harvest(umap)` rebuilds the stream under the enriched map), turning
    unknown consumes into exact ones and unlocking more blocks next round.
    Anchoring TRUTH stays the SNAPSHOT map every round -- minted identities
    never anchor themselves -- but their consume-COUNT evidence CAN unlock a
    previously-barriered direction (the feedback lane), so 1-dir / 1-anchor
    provenance is accumulated as a UNION across rounds and never upgraded
    (review C6). Stops when the mint count stops growing (typically 2-3
    rounds)."""
    stream_r1 = xref_r1 = None
    stream = xref_stats = None
    crossref: dict = {}


    evidence: dict = {"position": {}, "milk": {}, "crumb": {}}
    seen: dict = {"position": set(), "milk": set(), "crumb": set()}


    if pos_out is not None:
        for name in evidence:
            pos_out[f"{name}_crossref"] = evidence[name]
            pos_out[f"seen_{name}_unis"] = seen[name]
    seen_1dir_unis: set = set()
    seen_1anchor_unis: set = set()
    prev_n = -1
    rounds = 0
    while (len(crossref) + sum(len(v) for v in evidence.values())) > prev_n and rounds < max_rounds:
        prev_n = len(crossref) + sum(len(v) for v in evidence.values())
        umap = dict(uni_map)
        umap.update(crossref)
        for _b in evidence.values():
            umap.update(_b)
        result = harvest(umap)
        if isinstance(result, tuple):
            stream, mints = result
        else:
            stream, mints = result, {}
        for name, d in mints.items():
            evidence[name].update(d)
            seen[name] |= set(d)
        crossref, xref_stats = anchor_uni_blocks(stream, uni_map, action_kinds)
        for e in stream:
            if e["kind"] != "block" or "base" not in e:
                continue
            span = range(e["base"], e["base"] + len(e["seq"]))
            if e.get("chain_1dir"):
                seen_1dir_unis.update(span)
            elif (e.get("src") == "anchor" and not e.get("multi")
                    and not e.get("line_agree")):
                seen_1anchor_unis.update(span)
        rounds += 1
        if rounds == 1:
            stream_r1, xref_r1 = stream, xref_stats
    xref_stats["harvest_rounds"] = rounds
    for k in ("chain_pinned_pure", "chain_pinned_counted", "crossref_chained",
              "chain_holdout2", "chain_holdout2_wrong"):
        xref_stats[f"r1_{k}"] = xref_r1[k]
    # direct 1-dir accuracy lane (verify-round finding: the 1-dir slice had
    # no post-fix direct measurement): round-1 one-direction mints that the
    # FINAL round re-derives through a non-1-dir pin (double-confirmed line /
    # anchor) are graded against that independent re-derivation. Partial
    # coverage by construction (never-promoted 1-dir pins stay unmeasured --
    # that is what the crossref-1dir tag is for), but it is a real, shipped-
    # pipeline error rate for the promoted subset.
    r1_1dir = {}
    for e in stream_r1 or []:
        if e.get("kind") == "block" and "base" in e and e.get("chain_1dir"):
            for i, ent in enumerate(e["seq"]):
                r1_1dir[e["base"] + i] = tuple(ent)
    n_prom = n_prom_wrong = 0
    if r1_1dir:
        for e in stream:
            if e.get("kind") != "block" or "base" not in e or e.get("chain_1dir"):
                continue
            for i, ent in enumerate(e["seq"]):
                u = e["base"] + i
                prev = r1_1dir.get(u)
                if prev is not None and u in crossref:
                    n_prom += 1
                    if tuple(ent) != prev:
                        n_prom_wrong += 1
    xref_stats["onedir_promoted"] = n_prom
    xref_stats["onedir_promoted_wrong"] = n_prom_wrong
    # pos_out references were bound live at the top of the loop (in-place
    # accumulation), so nothing more to fill here.
    return (stream, stream_r1, xref_stats, xref_r1, crossref,
            seen_1dir_unis, seen_1anchor_unis)


def scan_orphan_combines(actions: list):
    """Returns (orphan_combine_unis, action_kinds, n_excluded_aim,
    n_excluded_pill). action_kinds is the H1 (review C1) per-uni action-type
    kind evidence (T6/T7/T9 = pet, T8 = food; contradictions omitted)."""
    type6_unis, sold_unis, t7_unis, board_unis_seen = set(), set(), set(), set()
    food_action_unis, type8_aim_unis, pill_aim_unis = set(), set(), set()
    for a in actions:
        ty = a.get("Type")
        req = action_request(a)
        if ty == 0:
            board_unis_seen |= snapshot_board_unis(a)   # unis that reach the BOARD
        elif ty == 6:
            u = (req.get("MinionId") or {}).get("Uni")
            if u is not None:
                type6_unis.add(int(u))
        elif ty == 9:
            u = (req.get("Minion") or {}).get("Uni")
            if u is not None:
                sold_unis.add(int(u))
        elif ty == 7:
            for k in ("SourceMinionId", "TargetMinionId"):
                u = (req.get(k) or {}).get("Uni")
                if u is not None:
                    t7_unis.add(int(u))
        elif ty == 8:
            u = (req.get("SpellId") or {}).get("Uni")
            if u is not None:
                food_action_unis.add(int(u))
            au = (req.get("Aim") or {}).get("Uni")
            if au is not None:
                type8_aim_unis.add(int(au))
                if food_enu_from_response(a) == PILL_ENU:
                    pill_aim_unis.add(int(au))
    # H1 (review C1): the action TYPE that consumed a uni is kind evidence in
    # its own right -- T6 buys/moves, T7 merge endpoints and T9 sells are PETS;
    # T8 spells are FOODS. Unis with contradictory action evidence veto nothing.
    pet_action_unis = type6_unis | sold_unis | t7_unis
    action_kinds = {u: "pet" for u in pet_action_unis - food_action_unis}
    action_kinds.update({u: "food" for u in food_action_unis - pet_action_unis})
    orphan_raw = type6_unis - board_unis_seen - sold_unis - t7_unis
    orphan_combine_unis = orphan_raw - type8_aim_unis   # veto (c)
    return (orphan_combine_unis, action_kinds,
            len(orphan_raw & type8_aim_unis), len(orphan_raw & pill_aim_unis))


def decode_game(game_id: str, game: dict, pools: dict):
    """Decode game."""
    actions = game.get("Actions") or []
    by_turn, violations = group_actions_by_turn(actions)
    turns_sorted = sorted(by_turn)
    end_snaps = {t: turn_end_snapshot(by_turn[t]) for t in turns_sorted}
    uni_map = build_game_uni_map(actions)
    # T3 genesis: the stored turn-1 opening shop carries its own Unis (pets
    # 1,2,3 + food 4); fold them into the truth map so turn-1 buys resolve by
    # Uni (incl. a genesis pet bought-then-sold that never snapshots).
    gen_items = genesis_shop_items(game)
    for u, kind, enu, _fro, _pri in gen_items or []:
        if u is not None:
            uni_map.setdefault(u, (kind, enu))


    (orphan_combine_unis, action_kinds,
     n_orphan_excluded_aim, n_orphan_excluded_pill) = scan_orphan_combines(actions)

    # ---- shared per-turn prep: carry state, start shop, T5 start injections ----
    prep = {}
    for t in turns_sorted:
        acts = by_turn[t]
        type4_seeds = [action_seed(a) for a in acts if a.get("Type") == 4]
        type4_seed = next((s for s in type4_seeds if isinstance(s, int)), None)
        roll_seeds = [s for s in (action_seed(a) for a in acts if a.get("Type") == 5) if isinstance(s, int)]

        # frozen carry-IN (previous turn's end-snapshot Fro items)
        carry_map = {}
        if t == 1:
            carry_status, carry_pets, carry_foods = "empty", [], []
        else:
            prev_snap = end_snaps.get(t - 1)
            if prev_snap is None:
                carry_status, carry_pets, carry_foods = "unknown", [], []
            else:
                carry_pets = [i["enu"] for i in prev_snap["pets"] if i["fro"]]
                carry_foods = [i["enu"] for i in prev_snap["foods"] if i["fro"]]
                carry_status = "nonempty" if (carry_pets or carry_foods) else "empty"
                for kind, key in (("pet", "pets"), ("food", "foods")):
                    for i in prev_snap[key]:
                        if i["fro"] and i.get("uni") is not None:
                            carry_map[i["uni"]] = (kind, i["enu"])


        prev_snap_action = next((a for a in by_turn.get(t - 1, []) if a.get("Type") == 0), None)
        start_block = []
        start_consume = None


        tt = str(shop_tier(t))
        cap_total = pools["pet_capacity"][tt] + pools["food_capacity"][tt]
        if t == 1:
            if gen_items:
                start_shop = {
                    "pets": [enu for u, kind, enu, fro, pri in gen_items if kind == "pet"],
                    "foods": [enu for u, kind, enu, fro, pri in gen_items if kind == "food"],
                    "provenance": "genesis", "injected": []}
                # genesis is the first drawn block (unis 1,2,3 + 4). Feed it so
                # roll blocks anchor AFTER it (prev_end monotonicity); its unis
                # are already in uni_map, so this pins ordering, never mints.
                start_block = [(kind, enu) for u, kind, enu, fro, pri in gen_items]
            else:
                start_shop = {"pets": None, "foods": None,
                              "provenance": "genesis-unknown", "injected": []}
                start_consume = (cap_total, "start_genesis_unknown")
        elif type4_seed is None:
            start_shop = {"pets": None, "foods": None, "provenance": "no-seed", "injected": []}
            if carry_status == "unknown":
                start_consume = (None, "start_unknown")
            else:
                n_inj = len(start_of_turn_injections(snapshot_board_pets(prev_snap_action))
                            if prev_snap_action is not None else [])
                start_consume = (max(0, cap_total - len(carry_pets) - len(carry_foods)) + n_inj,
                                 "start_no_seed")
        elif carry_status == "unknown":
            start_shop = {"pets": None, "foods": None, "provenance": "unknown-carry", "injected": []}
            start_consume = (None, "start_unknown")
        else:
            r = reconstruct_shop(type4_seed, t, frozen_pets=carry_pets, frozen_foods=carry_foods, pools=pools)
            injected = (start_of_turn_injections(snapshot_board_pets(prev_snap_action))
                        if prev_snap_action is not None else [])
            start_shop = {"pets": r["pets"], "foods": r["foods"],
                          "provenance": "reconstructed", "injected": injected}
            # drawn-then-injected uni sequence (frozen carries keep their old
            # unis; injected stocks occupy counter slots after the rolled block)
            start_block = ([("pet", e) for e in r["pets"][len(carry_pets):]] +
                           [("food", e) for e in r["foods"][len(carry_foods):]] +
                           [("food", it["enu"]) for it in injected])

        # entering-board tracker seed {uni: (enu, exp, perk)} -- exact levels
        # for sells/crumbs/milk/choice-mints, perks for pill spawn counts
        entering_pets = (snapshot_board_pets(prev_snap_action) if prev_snap_action else [])
        board_pet_levels = {b["uni"]: (b["enu"], b["exp"], b["perk"])
                            for b in entering_pets if b["uni"] is not None}


        if prev_snap_action is not None:
            board_pet_slots = {b["uni"]: b["slot"] for b in entering_pets if b["uni"] is not None}
        elif t == 1:
            board_pet_slots = {}
        else:
            board_pet_slots = None


        _end = end_snaps.get(t)
        end_crumb_leftover = (sum(1 for i in _end["foods"] if i["enu"] == CRUMB_ENU)
                              if _end is not None else 0)

        prep[t] = {"acts": acts, "type4_seed": type4_seed, "roll_seeds": roll_seeds,
                   "carry_map": carry_map, "carry_status": carry_status,
                   "carry_pets": carry_pets, "carry_foods": carry_foods,
                   "start_shop": start_shop, "start_block": start_block,
                   "start_consume": start_consume,
                   "prev_snap_action": prev_snap_action,
                   "board_pet_levels": board_pet_levels,
                   "board_pet_slots": board_pet_slots,
                   "end_crumb_leftover": end_crumb_leftover}

    # ---- pass A: harvest the game-wide uni-ALLOCATION stream (start block /
    # start consume first, then the walk's blocks and consume events in ACTION
    # order, with turn-boundary markers between turns -- the battle mints
    # nothing on the player's counter, see the chain_pred_xturn calibration).
    pos_out: dict = {}   # filled live by run_harvest_rounds (evidence buckets)
    def harvest(umap):
        stream = []
        pos_mints: dict = {}
        milk_mints: dict = {}
        crumb_mints: dict = {}
        board_unis: set = set()
        mem = new_membership()

        crumb_unis = pos_out.get("seen_crumb_unis", set())
        milk_unis = pos_out.get("seen_milk_unis", set())
        prev_t = None
        for t in turns_sorted:
            p = prep[t]
            if prev_t is not None:
                stream.append({"kind": "consume",
                               "n": 0 if t == prev_t + 1 else None,
                               "why": "turn_boundary" if t == prev_t + 1 else "turn_gap",
                               "turn": t})
            if p["start_block"]:
                stream.append({"kind": "block", "seq": p["start_block"], "turn": t})
            elif p["start_consume"] is not None:
                n, why = p["start_consume"]
                stream.append({"kind": "consume", "n": n, "why": why, "turn": t})
            if p["prev_snap_action"] is not None:
                board_unis = snapshot_board_unis(p["prev_snap_action"])
            elif t == 1:
                board_unis = set()
            walk = decode_turn_actions(p["acts"], t, p["carry_map"], umap, board_unis,
                                        p["start_shop"], pools, mem,
                                        board_pet_levels=p["board_pet_levels"],
                                        board_pet_slots=p["board_pet_slots"],
                                        crossref_milk_unis=milk_unis,
                                        crossref_crumb_unis=crumb_unis,
                                        end_crumb_leftover=p["end_crumb_leftover"],
                                        orphan_combine_unis=orphan_combine_unis)
            for e in walk["stream"]:
                e["turn"] = t
                stream.append(e)
            pos_mints.update(walk["pos_mints"])
            milk_mints.update(walk["milk_mints"])
            crumb_mints.update(walk["crumb_mints"])
            prev_t = t
        return stream, {"position": pos_mints, "milk": milk_mints, "crumb": crumb_mints}

    (stream, stream_r1, xref_stats, xref_r1, crossref,
     seen_1dir_unis, seen_1anchor_unis) = run_harvest_rounds(
        harvest, uni_map, action_kinds, pos_out=pos_out)
    position_crossref = pos_out.get("position_crossref", {})
    seen_position_unis = pos_out.get("seen_position_unis", set())
    milk_crossref = pos_out.get("milk_crossref", {})
    seen_milk_unis = pos_out.get("seen_milk_unis", set())
    crumb_crossref = pos_out.get("crumb_crossref", {})
    seen_crumb_unis = pos_out.get("seen_crumb_unis", set())
    full_map = dict(uni_map)
    full_map.update(crossref)          # crossref only fills unis ABSENT from truth
    full_map.update(position_crossref)  # H4: position mints (kept OUT of the
                                        # anchoring crossref -> minted_positions
                                        # == crossref_unis invariant untouched)
    full_map.update(milk_crossref)      # D2: milk-evidence Cow mints (crossref-milk)
    full_map.update(crumb_crossref)     # D2: crumb-evidence Pigeon mints (crossref-crumb)
    crossref_unis = set(crossref)
    xref_stats["crossref_unis"] = len(crossref_unis)


    crossref_1dir_unis = seen_1dir_unis & crossref_unis
    crossref_1anchor_unis = (seen_1anchor_unis & crossref_unis) - crossref_1dir_unis
    xref_stats["crossref_chained_1dir"] = len(crossref_1dir_unis)
    xref_stats["crossref_1anchor_cum"] = len(crossref_1anchor_unis)


    crossref_position_unis = seen_position_unis & set(position_crossref)
    xref_stats["crossref_position_unis"] = len(crossref_position_unis)


    xref_stats["orphan_excluded_aim"] = n_orphan_excluded_aim
    xref_stats["orphan_excluded_pill"] = n_orphan_excluded_pill


    crossref_milk_unis = seen_milk_unis & set(milk_crossref)
    crossref_crumb_unis = seen_crumb_unis & set(crumb_crossref)
    xref_stats["crossref_milk_unis"] = len(crossref_milk_unis)
    xref_stats["crossref_crumb_unis"] = len(crossref_crumb_unis)

    # ---- pass B: final decode with the enriched map ----
    membership = new_membership()
    freeze_counts = Counter()
    freeze_examples = []
    records = []
    board_unis = set()
    pos_stats = Counter()   # H4 position quality/recovery lanes -> xref
    for t in turns_sorted:
        p = prep[t]
        acts = p["acts"]
        start_shop = p["start_shop"]
        if p["prev_snap_action"] is not None:
            board_unis = snapshot_board_unis(p["prev_snap_action"])  # authoritative re-sync
        elif t == 1:
            board_unis = set()
        walk = decode_turn_actions(acts, t, p["carry_map"], full_map, board_unis,
                                    start_shop, pools, membership,
                                    where=f"{game_id}:t{t}", crossref_unis=crossref_unis,
                                    crossref_1dir_unis=crossref_1dir_unis,
                                    crossref_1anchor_unis=crossref_1anchor_unis,
                                    board_pet_levels=p["board_pet_levels"],
                                    board_pet_slots=p["board_pet_slots"],
                                    crossref_position_unis=crossref_position_unis,
                                    crossref_milk_unis=crossref_milk_unis,
                                    crossref_crumb_unis=crossref_crumb_unis,
                                    end_crumb_leftover=p["end_crumb_leftover"],
                                    orphan_combine_unis=orphan_combine_unis,
                                    pos_stats=pos_stats)

        # --- postrack holdout: end-of-turn tracked slots vs the end snapshot,
        # on turns whose tracker is STILL certain (expect ~1 mismatch cache-
        # wide -- the naive-tracker fidelity residual). ---
        this_snap_action = next((a for a in acts if a.get("Type") == 0), None)
        if walk["slots_certain"] and this_snap_action is not None:
            end_slots = {b["uni"]: b["slot"]
                         for b in snapshot_board_pets(this_snap_action)
                         if b["uni"] is not None}
            got_slots = {pu: ps for pu, ps in walk["final_slots"].items() if pu > 0}
            pos_stats["postrack_turns_certain"] += 1
            if got_slots != end_slots:
                pos_stats["postrack_end_mismatch"] += 1

        end_snapshot = end_snaps[t]

        # --- freeze-tracking cross-check vs the end snapshot ---
        track_ok = None
        if end_snapshot is not None:
            end_frozen_unis = sorted(i["uni"] for k in ("pets", "foods") for i in end_snapshot[k]
                                     if i["fro"] and i.get("uni") is not None)
            track_ok = end_frozen_unis == walk["final_frozen_unis"]
            freeze_counts["ok" if track_ok else "mismatch"] += 1
            if not track_ok and len(freeze_examples) < 3:
                freeze_examples.append({"where": f"{game_id}:t{t}",
                                        "tracked": walk["final_frozen_unis"],
                                        "end": end_frozen_unis})
        else:
            freeze_counts["no-snapshot"] += 1

        validation = classify_validation(acts, end_snapshot, t, p["carry_status"],
                                          p["carry_pets"], p["carry_foods"],
                                          start_shop, pools,
                                          last_roll=walk["rolls"][-1] if walk["rolls"] else None)

        records.append({
            "game_id": game_id,
            "turn": t,
            "tier": shop_tier(t),
            "type4_seed": p["type4_seed"],
            "roll_seeds": p["roll_seeds"],
            "carry_in": {"status": p["carry_status"], "frozen_pets": p["carry_pets"],
                          "frozen_foods": p["carry_foods"], "frozen_unis": sorted(p["carry_map"])},
            "start_shop": start_shop,
            "roll_shops": walk["rolls"],
            "chain": walk["chain"],
            "freeze": {"final_frozen_unis": walk["final_frozen_unis"],
                        "n_freeze_events": walk["n_freeze_events"],
                        "track_ok": track_ok},
            "end_snapshot": end_snapshot,
            "validation": validation,
        })


    snap_pets_by_turn = {}
    for t in turns_sorted:
        a0 = next((a for a in by_turn[t] if a.get("Type") == 0), None)
        if a0 is not None:
            snap_pets_by_turn[t] = {b["uni"]: {"slot": b["slot"], "enu": b["enu"], "exp": b["exp"]}
                                    for b in snapshot_board_pets(a0) if b["uni"] is not None}
    enu_of = {u: e for u, (k, e) in uni_map.items() if k == "pet"}

    def _is_amb(op):
        return op["op"] == "buy_combine" and (
            op.get("combine_target_ambiguous") or op.get("combine_target_recovered"))

    for rec in records:
        t = rec["turn"]
        chain = rec["chain"]
        prev = snap_pets_by_turn.get(t - 1, {}) if t > 1 else {}
        end = snap_pets_by_turn.get(t)
        posH = {u: d["slot"] for u, d in prev.items()}
        dirty = False
        exp_touched = Counter()   # unis exp-touched by ops OTHER than the
        for op in chain:          # adjudicated (recovered) combine
            o = op["op"]
            if o == "merge" and not op.get("buy_merge"):
                exp_touched[op.get("dst_uni")] += 99   # board-merge exp jump -> exclude
            elif o == "merge":
                exp_touched[op.get("dst_uni")] += 1
            elif (o == "buy_combine" and op.get("onto_uni") is not None
                    and not op.get("combine_target_recovered")):
                exp_touched[op["onto_uni"]] += 1
            elif o == "buy_food" and op.get("enu") == CHOCOLATE_ENU:
                exp_touched[op.get("target_uni")] += 1
            elif o == "buy_food" and op.get("enu") is None:
                dirty = True
        n_amb_same_enu = Counter(op.get("enu") for op in chain if _is_amb(op))
        amb_ops = []
        for op in chain:
            o = op["op"]
            bo = op.get("board_orders")
            if bo:
                posH.update({int(bu): s for bu, s in bo.items()})
            if o == "buy_pet" and op.get("uni") is not None:
                if any(v == op["to_slot"] for v in posH.values()):
                    dirty = True
                posH[op["uni"]] = op["to_slot"]
            elif _is_amb(op):
                enu = op.get("enu")
                cands = [u for u in posH if enu_of.get(u) == enu]
                unk = [u for u in posH if u > 0 and enu_of.get(u) is None]
                tgt = op.get("to_slot")
                at_slot = [u for u in cands if posH.get(u) == tgt]
                unk_at = [u for u in unk if posH.get(u) == tgt]
                if len(at_slot) == 1 and not unk_at and len(cands) >= 2:
                    amb_ops.append((op, list(cands), at_slot[0]))
            elif o == "merge" and not op.get("buy_merge") and op.get("src_uni") is not None:
                posH.pop(op["src_uni"], None)
            elif o == "sell" and op.get("uni") is not None:
                posH.pop(op["uni"], None)
            elif o == "buy_food" and op.get("enu") == PILL_ENU and op.get("target_uni") is not None:
                posH.pop(op["target_uni"], None)
                dirty = True
        if end is None or dirty:
            continue
        for op, cands, picked in amb_ops:
            enu = op.get("enu")
            if n_amb_same_enu[enu] != 1:
                continue
            if not all(u in prev and u in end for u in cands):
                continue
            if any(exp_touched.get(u) for u in cands):
                continue
            gained = [u for u in cands if end[u]["exp"] == prev[u]["exp"] + 1]
            same = [u for u in cands if end[u]["exp"] == prev[u]["exp"]]
            if len(gained) == 1 and len(same) == len(cands) - 1:
                pos_stats["position_holdout"] += 1
                if gained[0] != picked:
                    pos_stats["position_holdout_wrong"] += 1


    sold_all_g: set = set()
    for a in actions:
        if a.get("Type") == 9:
            su = (action_request(a).get("Minion") or {}).get("Uni")
            if su is not None:
                sold_all_g.add(int(su))
    for rec in records:
        t = rec["turn"]
        chain = rec["chain"]
        # map-independent milk/crumb signals from raw actions (fresh only)
        carry_frozen: set = set()
        prev_snap = end_snaps.get(t - 1)
        if prev_snap is not None:
            for k in ("pets", "foods"):
                for i in prev_snap[k]:
                    if i["fro"] and i.get("uni") is not None:
                        carry_frozen.add(i["uni"])
        milk_f = choco_f = crumb_f = 0
        milk_vars: set = set()
        for a in by_turn[t]:
            if a.get("Type") != 8:
                continue
            e = food_enu_from_response(a)
            mu = (action_request(a).get("SpellId") or {}).get("Uni")
            fresh = mu is None or int(mu) not in carry_frozen
            if e in MILK_ENUS and fresh:
                milk_f += 1
                milk_vars.add(e)
            elif e in CHOCOMILK_ENUS and fresh:
                choco_f += 1
            elif e == CRUMB_ENU and fresh:
                crumb_f += 1
        cow_stock = any(op.get("op") == "ability_stock" and op.get("source") == "Cow"
                        for op in chain)
        pigeon_stock = any(op.get("op") == "ability_stock" and op.get("source") == "Pigeon"
                           for op in chain)
        unres_plain_sold = [op["uni"] for op in chain
                            if op["op"] == "buy_pet" and op.get("enu") is None
                            and op.get("uni") in sold_all_g]
        cow_buys = [op for op in chain
                    if op["op"] in ("buy_pet", "buy_combine") and op.get("enu") == COW_ENU]
        unres_sell = [op["uni"] for op in chain
                      if op["op"] == "sell" and op.get("enu") is None]
        pig_sells = [op["uni"] for op in chain
                     if op["op"] == "sell" and op.get("enu") == PIGEON_ENU]

        # MILK lane: the resolved Cow buy is the truth. The rule's candidate set
        # is the sold unresolved plain buys, plus the resolved Cow iff it was
        # itself a sold plain buy. A unique candidate that is NOT the resolved
        # Cow = a wrong pick (the decoy-beats-truth failure mode).
        if (milk_f == 2 and choco_f == 0 and milk_vars == {MILK_BY_LVL[1]}
                and cow_stock and len(cow_buys) == 1):
            cb = cow_buys[0]
            cands = set(unres_plain_sold)
            if cb["op"] == "buy_pet" and cb.get("uni") in sold_all_g:
                cands.add(cb.get("uni"))
            if len(cands) == 1:
                pos_stats["milk_holdout"] += 1
                if next(iter(cands)) != cb.get("uni"):
                    pos_stats["milk_holdout_wrong"] += 1
        # MILK negative control: a fully-explained milk turn with a decoy sold
        # unresolved plain buy -> production (gated on `not cow_stock`) stays
        # silent by construction; count the guarded population.
        if milk_f >= 2 and choco_f == 0 and cow_stock and unres_plain_sold:
            pos_stats["milk_negctrl"] += 1

        # CRUMB lane: the resolved Pigeon sell is the truth; the rule nominates
        # the unique unresolved sell. Wrong iff != the resolved Pigeon.
        if crumb_f >= 1 and pigeon_stock and len(pig_sells) == 1:
            pig_uni = pig_sells[0]
            cands = set(unres_sell) | {pig_uni}
            if len(cands) == 1:
                pos_stats["crumb_holdout"] += 1
                if next(iter(cands)) != pig_uni:
                    pos_stats["crumb_holdout_wrong"] += 1
        # CRUMB negative control: a resolved-Pigeon crumb turn with a decoy
        # unresolved sell -> production (gated on `not pigeon_stock`) stays silent.
        if crumb_f >= 1 and pigeon_stock and unres_sell:
            pos_stats["crumb_negctrl"] += 1

    xref_stats.update(dict(pos_stats))

    game_stats = {"membership": membership, "freeze_counts": freeze_counts,
                  "freeze_examples": freeze_examples, "xref": xref_stats,
                  "crossref": crossref,   # {uni: (kind, enu)} minted beyond snapshots
                  # annotated allocation streams (probe/debug surface; run()
                  # aggregation ignores them)
                  "stream": stream, "stream_r1": stream_r1}
    return records, violations, game_stats


# --------------------------------------------------------------------------
# aggregate report
# --------------------------------------------------------------------------

BUCKET_LABELS = {
    "a": "(a) CLEAN-LAST-ROLL, no carry-in, no end-frozen    [target >=90%, expect ~96-98%]",
    "b": "(b) NO-ROLL START, no carry-in                     [target >=90%, expect ~100%]",
    "c": "(c) NO-ROLL START, WITH carry-in (T2 1st measure)  [no target -- report honestly]",
    "d": "(d) CARRY-IN LAST-ROLL (T2 1st measure)             [no target -- report honestly]",
    "e": "(e) FREEZE-CHANGED LAST-ROLL, tracked frozen (T6)   [wave-1: unbucketed; expect ~(a)]",
    "g": "(g) TURN-1 GENESIS shop (stored GenesisBuildModel,T3) [read not cracked; expect ~100%]",
}
BUCKET_TARGETS = {"a": 0.90, "b": 0.90}


def aggregate_report(all_records: list, total_violations: int, run_stats: dict) -> dict:
    n_total = len(all_records)
    n_type4 = sum(1 for r in all_records if r["type4_seed"] is not None)
    n_no_snapshot = sum(1 for r in all_records if r["end_snapshot"] is None)
    n_turn1 = sum(1 for r in all_records if r["turn"] == 1)
    n_turn1_genesis = sum(1 for r in all_records if r["turn"] == 1
                          and r["start_shop"]["provenance"] == "genesis")
    n_carry_unknown = sum(1 for r in all_records if r["carry_in"]["status"] == "unknown")

    buckets = {k: {"n": 0, "pass": 0, "pass_tracked": 0, "pass_inj": 0, "examples": []} for k in "abcdeg"}
    unbucketed_notes = Counter()
    for r in all_records:
        v = r["validation"]
        b = v.get("bucket")
        if b in buckets:
            buckets[b]["n"] += 1
            if v.get("pass"):
                buckets[b]["pass"] += 1
            elif v.get("detail") is not None and len(buckets[b]["examples"]) < 3:
                buckets[b]["examples"].append(v["detail"])
            pt = v.get("pass_tracked")
            if pt if pt is not None else v.get("pass"):
                buckets[b]["pass_tracked"] += 1
            pi = v.get("pass_inj")
            if pi if pi is not None else v.get("pass"):
                buckets[b]["pass_inj"] += 1
        else:
            unbucketed_notes[v.get("note", "unknown")] += 1

    return {
        "n_total": n_total, "n_type4": n_type4, "n_no_snapshot": n_no_snapshot,
        "n_turn1": n_turn1, "n_turn1_genesis": n_turn1_genesis,
        "n_carry_unknown": n_carry_unknown,
        "buckets": buckets, "unbucketed_notes": unbucketed_notes,
        "createdon_violations": total_violations,
        "run_stats": run_stats,
    }


def _pct(n, d) -> str:
    return f"{n / d:.1%}" if d else "n/a"


def print_report(agg: dict, n_games: int, cache_path) -> None:
    print("=" * 78)
    print("decode_turn.py -- wave-2 (T3+T2 state, T6 action/freeze) self-validation report")
    print("=" * 78)
    print(f"cache:            {cache_path}")
    print(f"games decoded:    {n_games}")
    print(f"turns total:      {agg['n_total']}")
    print(f"  with type4 seed:            {agg['n_type4']}  ({_pct(agg['n_type4'], agg['n_total'])})")
    print(f"  no end-of-turn snapshot:    {agg['n_no_snapshot']}")
    print(f"  turn-1 total:               {agg['n_turn1']}  "
          f"(genesis read from GenesisBuildModel: {agg['n_turn1_genesis']}; "
          f"unknown: {agg['n_turn1'] - agg['n_turn1_genesis']})")
    print(f"  carry-in status unknown:    {agg['n_carry_unknown']}")
    print(f"  CreatedOn violations (re-sorted within turn): {agg['createdon_violations']}")
    print()
    print("-- self-validation buckets --")
    for k in "abcdeg":
        b = agg["buckets"][k]
        n, p = b["n"], b["pass"]
        rate = _pct(p, n)
        flag = ""
        target = BUCKET_TARGETS.get(k)
        if target is not None and n:
            flag = "  OK" if (p / n) >= target else "  *** BELOW TARGET ***"
        print(BUCKET_LABELS[k])
        print(f"    n={n}  pass={p}  rate={rate}{flag}")
        if k in "ad" and n and b["pass_tracked"] != p:
            print(f"    ({k}') same population under TRACKED roll-time frozen set: "
                  f"pass={b['pass_tracked']}  rate={_pct(b['pass_tracked'], n)}")
        if k in "bc" and n and b["pass_inj"] != p:
            print(f"    ({k}+) allowing unclaimed mid-turn Cow stocks (T5): "
                  f"pass={b['pass_inj']}  rate={_pct(b['pass_inj'], n)}")
        for ex in b["examples"]:
            print(f"    fail example: {json.dumps(ex)}")
    print()
    n_unbucketed = sum(agg["unbucketed_notes"].values())
    print(f"unbucketed turns (outside self-check scope): {n_unbucketed}")
    for note, c in agg["unbucketed_notes"].most_common():
        print(f"    {c:5d}  {note}")

    rs = agg.get("run_stats") or {}
    fc = rs.get("freeze_counts") or {}
    m = rs.get("membership") or {}
    print()
    print("-- wave-2 chain metrics (T6) --")
    n_ok, n_bad = fc.get("ok", 0), fc.get("mismatch", 0)
    print(f"freeze tracking vs end snapshots: ok={n_ok}  mismatch={n_bad}  "
          f"rate={_pct(n_ok, n_ok + n_bad)}  (no-snapshot: {fc.get('no-snapshot', 0)})")
    for ex in rs.get("freeze_examples") or []:
        print(f"    mismatch example: {json.dumps(ex)}")
    for kind in ("pet", "food"):
        buys = m.get(f"{kind}_buys", 0)
        chk = m.get(f"{kind}_checked", 0)
        ok = m.get(f"{kind}_ok", 0)
        print(f"buy-membership [{kind}]: buys={buys}  checked={chk} ({_pct(chk, buys)} of buys)  "
              f"in-shop={ok} ({_pct(ok, chk)} of checked)")
    print(f"    of which matched via T5 injections: {m.get('food_ok_injected', 0)} food; "
          f"resolved via uni cross-ref: {m.get('pet_ok_crossref', 0)} pet / "
          f"{m.get('food_ok_crossref', 0)} food "
          f"(1-dir-tagged slice: {m.get('pet_ok_crossref_1dir', 0)} pet / "
          f"{m.get('food_ok_crossref_1dir', 0)} food; 1-anchor-tagged: "
          f"{m.get('pet_ok_crossref_1anchor', 0)} pet / "
          f"{m.get('food_ok_crossref_1anchor', 0)} food)")
    print(f"    op kind conflicts (map kind vs action-type kind, review C1): "
          f"{m.get('op_kind_conflict', 0)}")
    fme = m.get("food_miss_enu") or {}
    if fme:
        top = ", ".join(f"{e}={c}" for e, c in Counter(fme).most_common(8))
        print(f"    residual food-miss by enu (top): {top}")
    for ex in (m.get("miss_examples") or [])[:3]:
        print(f"    membership miss: {json.dumps(ex)}")
    print("    (misses include tier-up buys -- the T4 census, see docstring)")
    xr = rs.get("xref") or {}
    print()
    print("-- wave-4 identity pass (uni-counter block anchoring) --")
    print(f"drawn blocks: {xr.get('blocks', 0)}  anchored={xr.get('anchored', 0)} "
          f"({_pct(xr.get('anchored', 0), xr.get('blocks', 0))})  "
          f"ambiguous={xr.get('ambiguous', 0)}  no-anchor={xr.get('no_anchor', 0)}")
    print(f"  anchored by >=2 snapshot unis (order self-validated): {xr.get('anchored_multi', 0)}  "
          f"| single-anchor: {xr.get('anchored_single', 0)}")
    hn = xr.get('holdout_n', 0)
    print(f"  holdout (drop 1 anchor, re-anchor on the rest): robust={xr.get('holdout_robust', 0)} "
          f"({_pct(xr.get('holdout_robust', 0), hn)})  "
          f"WRONG-mint={xr.get('holdout_wrong', 0)} ({_pct(xr.get('holdout_wrong', 0), hn)})  "
          f"(remainder = became ambiguous -> skipped, not wrong)")
    print()
    print("-- Chained anchoring (uni-counter contiguity) --")
    n, w = xr.get("chain_holdout2", 0), xr.get("chain_holdout2_wrong", 0)
    rn, rw = xr.get("r1_chain_holdout2", 0), xr.get("r1_chain_holdout2_wrong", 0)
    print(f"  DOUBLE-CONFIRM holdout (fwd+bwd agree on a multi-anchor block): "
          f"n={n}  WRONG={w} ({_pct(w, n)})")
    print(f"    direct lane (round 1, no feedback): n={rn}  WRONG={rw} ({_pct(rw, rn)})"
          "   (final minus r1 = feedback-enabled lane, review C6)")
    pn, pw = xr.get("onedir_promoted", 0), xr.get("onedir_promoted_wrong", 0)
    print(f"  1-dir DIRECT lane (r1 1-dir mints re-derived non-1-dir in the final round): "
          f"n={pn}  WRONG={pw} ({_pct(pw, pn)})   (promoted subset only; "
          f"never-promoted 1-dir pins stay tag-sliced, not measured)")
    for lane in ("pure", "counted"):
        for q, qlab in (("mm", "multi->multi"), ("s1", ">=1 single-anchor endpt")):
            nn = xr.get(f"chain_pred_{lane}_{q}", 0)
            ww = xr.get(f"chain_pred_{lane}_{q}_wrong", 0)
            print(f"  calibration [{lane:7s}|{qlab:24s}]: preds={nn}  wrong={ww} "
                  f"({_pct(ww, nn)})")
    nn, ww = xr.get("chain_pred_xturn_mm", 0), xr.get("chain_pred_xturn_mm_wrong", 0)
    print(f"  calibration [cross-turn subset, multi->multi]: preds={nn}  wrong={ww} ({_pct(ww, nn)})")
    print("    (mm lanes measure the chained arithmetic; s1 lanes are dominated by")
    print("     single-anchor base placement noise -- review C8; measured, NOT pinned on)")
    print(f"  chained pins: pure={xr.get('chain_pinned_pure', 0)}  "
          f"counted={xr.get('chain_pinned_counted', 0)}  "
          f"(round-1: {xr.get('r1_chain_pinned_pure', 0)}/{xr.get('r1_chain_pinned_counted', 0)})  "
          f"disambiguated={xr.get('chain_disambiguated', 0)}  "
          f"1-dir-TAGGED={xr.get('chain_pinned_1dir', 0)}")
    print(f"  chain skips (split, review C9): fwd/bwd-DIRECTION-conflicts="
          f"{xr.get('chain_conflict', 0)}  overlap-evictions={xr.get('chain_overlap_evict', 0)}  "
          f"content={xr.get('chain_content_skip', 0)}  kind-veto={xr.get('chain_kind_veto', 0)}  "
          f"1-direction-only={xr.get('chain_1dir_skipped', 0)} (0 when CHAIN_ALLOW_1DIR)")
    n1, w1 = xr.get("chain_vs_1anchor", 0), xr.get("chain_vs_1anchor_disagree", 0)
    print(f"  single-anchor bases vs the confirmed line: n={n1}  "
          f"OFF-LINE={w1} ({_pct(w1, n1)})  -> re-based={xr.get('anchor1_rebased', 0)} "
          f"dropped={xr.get('anchor1_dropped', 0)}; line-validated={xr.get('anchor1_line_agree', 0)}  "
          f"(review C2; p1 kind-vetoes: {xr.get('p1_kind_veto', 0)}; "
          f"anchors evicted by stronger pins: {xr.get('overlap_anchor_dropped', 0)})")
    print(f"cross-ref unis minted (identities beyond any snapshot): {xr.get('crossref_unis', 0)}  "
          f"(via chaining: {xr.get('crossref_chained', 0)}; round-1: {xr.get('r1_crossref_chained', 0)}; "
          f"1-dir-TAGGED: {xr.get('crossref_chained_1dir', 0)}; "
          f"1-anchor-TAGGED: {xr.get('crossref_1anchor_cum', 0)})")
    print(f"  minted-position invariant (== crossref unis, review C3): "
          f"{xr.get('minted_positions', 0)}")
    print("=" * 78)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run(cache: Path, max_games: int | None = None):
    """Decode every game in `cache`; return (all_records, agg, n_games).

    The exact aggregation behind the CLI report -- verify.py gates on the
    returned `agg` dict instead of parsing printed output. agg is None when
    the cache yielded no games.
    """
    pools = load_pools(POOLS_PATH)

    all_records = []
    total_violations = 0
    n_games = 0
    run_stats = {"membership": new_membership(), "freeze_counts": Counter(),
                 "freeze_examples": [], "xref": Counter()}
    for gid, game in iter_games(cache, max_games=max_games):
        recs, viol, gstats = decode_game(gid, game, pools)
        all_records.extend(recs)
        total_violations += viol
        n_games += 1
        gm = gstats["membership"]
        for k, v in gm.items():
            if k == "miss_examples":
                space = 5 - len(run_stats["membership"]["miss_examples"])
                if space > 0:
                    run_stats["membership"]["miss_examples"].extend(v[:space])
            elif k == "food_miss_enu":
                run_stats["membership"]["food_miss_enu"].update(v)
            else:
                run_stats["membership"][k] += v
        run_stats["freeze_counts"].update(gstats["freeze_counts"])
        run_stats["xref"].update(gstats["xref"])
        space = 3 - len(run_stats["freeze_examples"])
        if space > 0:
            run_stats["freeze_examples"].extend(gstats["freeze_examples"][:space])

    agg = (aggregate_report(all_records, total_violations, run_stats)
           if n_games else None)
    return all_records, agg, n_games


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Decode SAP replay turns into per-turn state records + concrete "
                     "action chains (wave-2: T3/T2 state + T6 freeze/actions) and self-validate.")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                     help="manifest .json.gz or full .jsonl.gz cache (auto-detected by content)")
    ap.add_argument("--out", type=Path, default=None,
                     help="write decoded records as a gzipped JSON list")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--report-only", action="store_true",
                     help="print the aggregate report only; skip writing --out")
    args = ap.parse_args(argv)

    all_records, agg, n_games = run(args.cache, max_games=args.max_games)

    if n_games == 0:
        print(f"WARNING: no games loaded from {args.cache} -- check the path/format.", file=sys.stderr)
        return 2

    print_report(agg, n_games, args.cache)

    if args.out and not args.report_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.out, "wt", encoding="utf-8") as f:
            json.dump(all_records, f)
        print(f"\nwrote {len(all_records)} turn records -> {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
