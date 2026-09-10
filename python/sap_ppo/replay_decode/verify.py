"""sap_ppo.replay_decode.verify -- frozen regression gate for the replay decoder.

Runs the full decode over the frozen 2,143-game Turtle cache and asserts the
settled numbers EXACTLY. The decode is deterministic on a fixed cache, so any
drift -- even by one turn -- is a behavior change that must be understood
before it ships (see internal design notes for what each
number means and how it was settled; the 2026-07-10 exp09-chaining re-pin is
documented there as an addendum).

Checked surfaces (pinned to the 2026-07-12 W3a identity-recovery baseline --
D1 H4 position tracking + D2 milk/crumb evidence rules on top of the 2026-07-11
review-fix-batch (C1-C6) baseline; lanes and freeze tracking unchanged since
wave 5). 104 -> 116 -> 120 checks (W3a +12 evidence lanes; 2026-07-12
post-adversarial-review fix round +4 F1 orphan-exclusion tripwires). Per-value
mechanisms for every drifted pin are in the RESULTS.md 2026-07-12 appendix
(W3a section + the "post-adversarial-review fix round" subsection):
  * self-validation lanes (a)(a')(b)(b+)(c)(c+)(d)(d')(e)(g) -- exact n/pass;
  * unbucketed-turn census (roll-not-last / e-frozen-unresolved / no-snapshot);
  * freeze tracking vs end snapshots (25,216 ok / 36 known mismatches);
  * buy-membership censuses (pet + food, incl. T5-injection / cross-ref /
    1-dir / 1-anchor splits + the op kind-conflict counter);
  * wave-4 identity pass (block anchoring, holdout WRONG-mint must stay 0);
  * exp09 chained anchoring (double-confirm holdout incl. the round-1 direct
    lane, pins, skip censuses split conflict/overlap/content/kind, single-
    anchor line checks, mm/s1 calibration lanes, minted-position invariant --
    loose+TAGGED default: CHAIN_ALLOW_1DIR=True, 1-dir + 1-anchor tagged);
  * W3a evidence-quality lanes (PINNED_EVIDENCE): position BLIND-holdout
    wrong-mint bounds; milk/crumb HINDSIGHT-CONSISTENCY lanes over the converged
    chain (NOT blind re-decodes); provenance tag-set sizes; + the F1 orphan-
    exclusion / recovery-veto tripwires (fix round);
  * coverage over the decoded records (identity 97.5%, genesis 2,143/0, ...).

NOT gated here: the gold-reconcile census (artifact layer) -- run
  python an internal analysis script --reconcile-census
and expect reconciled 24,328 / 25,239 snapshot turns (96.39%, fix round; +4 vs
the W3a 24,324 = the F1 coverage cascade reaching a few more sell-placements;
+11 vs the fix-batch 24,317 -- a mostly-orthogonal gold-effect surface).

Usage:
  python -m sap_ppo.replay_decode.verify              # full gate (~30s), exit 1 on drift
  python -m sap_ppo.replay_decode.verify --cache X    # report-only (no gate) on another cache
  python -m sap_ppo.replay_decode.verify --max-games N  # smoke run, report-only
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
import os
from pathlib import Path

from .decode_turn import run
from .reconstruct_shop import POOLS_PATH, load_pools

PINNED_CACHE = Path(os.environ.get(
    "SAP_REPLAY_PINNED_CACHE",
    str(Path(__file__).resolve().parents[3] / "data" / "raw_replays_turtle_full.jsonl.gz"),
))

# Wave-5 baseline, full cache, 2026-07-10 (branch exp08/impl-wave6 cut from
# the PR #18 merge). Every value is an exact count, not a rate.
PINNED_TOTALS = {
    "games": 2143,
    "turns": 25255,
    "with_type4_seed": 25255,
    "no_end_snapshot": 3,
    "turn1_total": 2143,
    "turn1_genesis": 2143,
    "carry_unknown": 0,
    "createdon_violations": 0,
}
# lane -> (n, pass); tracked/inj variants pinned separately where they differ.
PINNED_LANES = {
    "a": (1746, 1673), "b": (268, 244), "c": (1291, 1118),
    "d": (470, 413), "e": (9175, 9163), "g": (8, 8),
}
PINNED_LANE_TRACKED = {"a": 1746, "d": 470}   # (a') (d')
PINNED_LANE_INJ = {"b": 251, "c": 1145}       # (b+) (c+)
PINNED_UNBUCKETED = {
    "roll-not-last-shop-action": 12277,
    "e-frozen-unresolved": 17,
    "no-end-snapshot": 3,
}
PINNED_FREEZE = {"ok": 25216, "mismatch": 36, "no-snapshot": 3}
# 2026-07-12 W3a: _checked/_ok grow because H4/D2 mints resolve op["enu"] for
# buys that used to be dead ends (Family A coverage cascade -- newly-resolved
# ops ride the same enu-is-not-None gate as any snapshot buy). The _1dir /
# _1anchor sub-buckets SHRINK (retagging, Family B): id_src() ranks the new
# crossref-position/-milk/-crumb proofs above the counter tags, so a uni
# resolved by BOTH keeps its identity but is relabelled to the stronger tag --
# same species, zero net coverage loss (it stays inside pet_ok_crossref).
PINNED_MEMBERSHIP = {
    "pet_buys": 52401, "pet_checked": 51295, "pet_ok": 43203,
    "food_buys": 29562, "food_checked": 29381, "food_ok": 29362,
    "food_ok_injected": 5893, "pet_ok_crossref": 9501, "food_ok_crossref": 1431,
    "pet_ok_crossref_1dir": 1416, "food_ok_crossref_1dir": 557,
    "pet_ok_crossref_1anchor": 264, "food_ok_crossref_1anchor": 109,
    # review C1: with pin-level kind vetoes in place, ZERO op-level kind
    # conflicts remain -- any growth = a new kind-blind path shipped.
    "op_kind_conflict": 0,
}
# 2026-07-12 W3a: block-population growth (Family C, extended chain reach) --
# once a mechanic identity becomes known (Cow behind a milk stock, Pigeon
# behind crumbs, combine target via H4), the WALK emits a real content block
# instead of an opaque barrier, so `blocks` and every pass-1 split grow.
# holdout_wrong MUST stay 0 (a wrong-mint on a >=2-snapshot block = real error).
PINNED_XREF = {
    "blocks": 93141, "anchored": 30464, "ambiguous": 11110, "no_anchor": 51567,
    "anchored_multi": 18411, "anchored_single": 12053,
    "holdout_robust": 16910, "holdout_wrong": 0,
}
# exp09 chained anchoring. chain_holdout2_wrong is the direct wrong-mint
# bound of the DOUBLE-confirmed pins (the 1-dir tagged slice carries ~1-2%,
# measured by guarded holdouts -- see STATUS.md). Chained-arithmetic drift is
# watched by the multi->multi calibration lanes (chain_pred_*_mm*); the *_s1
# lanes are dominated by single-anchor base placement noise and measure THAT,
# not the chain (review C8). Any GROWTH in a *_wrong value is a red flag,
# shrinkage is progress (re-pin + document).
# 2026-07-12 W3a identity-recovery re-pin (branch exp09/w3a-identity-recovery;
# per-value mechanisms in internal design notes 2026-07-12
# appendix). Headline surge (crossref 260,630 -> 283,517; identity 91.3% ->
# 97.2%) is the H4/D2 blocker-removal + mint-feedback effect: combine-target
# and plain-buy-Cow None-barriers become EXACT counted consumes, so the
# counter chain reaches further (more blocks pinned) and a large slice of
# previously counter-derived identities is retagged to the stronger
# crossref-position/-milk/-crumb provenance (1dir/1anchor sub-buckets shrink).
# *_wrong growth adjudicated to sample level (see RESULTS.md deep-dives):
#   onedir_promoted_wrong 109 -> 323: 0 grade-flips on the overlapping
#     population; the +214 is all NEW grading opportunities (round-1 1-dir
#     guesses now have an independent final-round re-derivation to grade
#     against) -- hole-closing shifts the round-1 frontier; the lane converges
#     toward its already-published 0.45-1.6% expected band.
#   chain_holdout2_wrong 13 -> 27 / r1 10 -> 23: ZERO flips in the common
#     population; +14 are newly-double-confirmable blocks (barrier removal)
#     inheriting pre-existing unmodeled gaps at a selection-biased higher rate.
#   chain_conflict 800 -> 1,114: barrier removal reveals fwd/bwd-disagreeing
#     stretches that are safely QUARANTINED (never minted -- conflict entries
#     never acquire a "base"); D2 milk-hole fix nets -255 vs D1's 1,369.
#   chain_pred_counted_s1_wrong 1092 -> 1528 (rate 29.8% -> 35.3%): the
#     single-anchor-endpoint / cross-turn PLACEMENT-noise calibration lane
#     (measurement only, no mints). All wrongness localises to single-anchor
#     endpoints (MM 40/2281 = 1.75%, IMPROVED from 2.73%; MS 16.8% / SM 35.7%
#     / SS 61.7%) -- the chain ARITHMETIC did not regress. Growth is the
#     barrier->counted-consume recomposition pushing more predictions into the
#     cross-turn regime (100% of sampled wrongs were xturn). NOT chain error.
PINNED_CHAIN = {
    "chain_holdout2": 13235, "chain_holdout2_wrong": 29,
    "r1_chain_holdout2": 10003, "r1_chain_holdout2_wrong": 24,  # direct lane (C6)
    # verify-round lane: r1 1-dir mints graded against independent non-1-dir
    # re-derivation in the final round (promoted subset): 357/47,399 = 0.75%
    # (W3a: 0 flips on the overlap; fix round: F1 orphan-exclusion extends the
    # promoted population, wrong-RATE IMPROVES 0.95% -> 0.75%, 0 quality flips).
    "onedir_promoted": 47399, "onedir_promoted_wrong": 357,
    "chain_pinned_pure": 18653, "chain_pinned_counted": 38161,
    "chain_pinned_1dir": 11687, "chain_disambiguated": 10433,
    "chain_conflict": 1294, "chain_overlap_evict": 143,
    "overlap_anchor_dropped": 742, "chain_content_skip": 281,
    "chain_kind_veto": 12, "p1_kind_veto": 647,
    "chain_1dir_skipped": 0,
    "chain_vs_1anchor": 10247, "chain_vs_1anchor_disagree": 1743,
    "anchor1_line_agree": 8504, "anchor1_rebased": 1743, "anchor1_dropped": 0,
    "crossref_chained": 240084, "crossref_chained_1dir": 92608,
    "crossref_1anchor_cum": 10500,
    "crossref_unis": 292251,
    # calibration lanes, endpoint-quality split (review C8): mm = the chained
    # arithmetic (watch these for real regressions), s1 = single-anchor base
    # placement + cross-turn noise (expected ~15-35% wrong, NOT chain error;
    # measurement only, mints nothing). W3a + fix round: every mm lane held or
    # IMPROVED (rates: pure_mm 0.80%, counted_mm 1.77%, xturn_mm 1.37%).
    "chain_pred_pure": 20714, "chain_pred_pure_wrong": 2237,
    "chain_pred_pure_mm": 6411, "chain_pred_pure_mm_wrong": 51,
    "chain_pred_pure_s1": 14303, "chain_pred_pure_s1_wrong": 2186,
    "chain_pred_counted": 6989, "chain_pred_counted_wrong": 1760,
    "chain_pred_counted_mm": 2377, "chain_pred_counted_mm_wrong": 42,
    "chain_pred_counted_s1": 4612, "chain_pred_counted_s1_wrong": 1718,
    "chain_pred_xturn": 15412, "chain_pred_xturn_wrong": 3049,
    "chain_pred_xturn_mm": 5902, "chain_pred_xturn_mm_wrong": 81,
}
# 2026-07-12 W3a Family A coverage cascade: H4/D2 mints resolve op["enu"] for
# buys that used to be dead ends, so identity coverage climbs 91.3% -> 97.2%
# (+1,467 turns) and buy_unresolved_identity falls 2,132 -> 661. has_tierup
# grows +89 as newly-resolved above-tier buys become countable; shop_gap
# falls as frozen carry-ins let more roll-shops reconstruct. Fix round (F1
# pilled-orphan exclusion) extends the SAME cascade one step: fully-decoded
# 24,536 -> 24,619 (97.2% -> 97.5%), buy_unresolved 661 -> 584, shop_gap
# 108 -> 98, has_tierup(T4) 6,895 -> 6,886.
PINNED_COVERAGE = {
    "fully_decoded_identity": 24619,
    "fully_decoded_rolled_only": 16012,
    "turn1_genesis_decoded": 2143,
    "turn1_genesis_unknown": 0,
    "shop_gap(genesis/frozen-unresolved)": 98,
    "buy_unresolved_identity": 584,
    "has_tierup_buy(T4)": 6886,
    "has_ability_buy(T5)": 3194,
}

# 2026-07-12 W3a evidence-quality lanes (D1 H4 position recovery + D2 milk/crumb
# ability evidence). POSITION lanes are true BLIND picks (the slot choice graded
# against exp truth): position_holdout n = blind adjudications, _wrong = the
# direct wrong-mint bound (0). MILK/CRUMB lanes are HINDSIGHT-CONSISTENCY lanes
# over the final converged chain -- the candidate set is already collapsed by ALL
# mechanisms and the evidence rule's nominee is checked against the independently-
# RESOLVED Cow/Pigeon; they are NOT masked blind re-decodes, so rule-fired
# precision is the production-relevant measure (adversarial-round note + the
# masked-rerun confound are in the exp08 RESULTS.md 2026-07-12 appendix).
# crossref_*_unis = provenance tag-set sizes (unis whose species was PROVEN by
# that evidence class). All keys live in run_stats["xref"] (pos_stats merged in).
# Fix round (F1 pilled-orphan exclusion): position_species_conflict 16 -> 0 (the
# 16 clashes were Type-8 pilled false-orphans, now excluded before H4);
# crossref_position_unis 5,706 -> 5,666 (40 phantom mints removed); the cascade
# nudges position_holdout 449 -> 451 / milk 1,332 -> 1,333 / crumb 866 -> 875.
PINNED_EVIDENCE = {
    # H4 position recovery (combine-target certainty tracker)
    "crossref_position_unis": 5666,
    "position_holdout": 451, "position_holdout_wrong": 0,   # blind slot-pick vs exp truth: 0 wrong
    "position_confident_disagree": 0,   # tracker-quality tripwire; downgrades to barrier, never guesses
    "position_species_conflict": 0,     # F1: pilled false-orphans excluded pre-H4 -> 0 clashes (was 16)
    "postrack_end_mismatch": 1,         # tracked slots vs end-snapshot on still-certain turns (~1 cache-wide)
    # F1 orphan-exclusion tripwires: Type-8 pilled/fed unis kept OUT of the
    # orphan_combine set (root fix), + defense-in-depth recovery vetoes. The
    # veto lanes read 0 on this cache (veto (c) subsumes the population); they
    # guard non-pilled false orphans + future data, and fire in the unit tests.
    "orphan_excluded_aim": 1000, "orphan_excluded_pill": 1000,
    "position_veto_later_batch": 0, "position_veto_truth": 0,
    # D2 milk evidence (buy-Cow -> stock 2xMilk-49 hole)
    "crossref_milk_unis": 607,
    # milk_holdout_wrong=1 is a HINDSIGHT-LANE-ONLY artifact (the resolved Cow is
    # excluded from the candidate set and a decoy sold null-buy is the unique
    # nominee); in production a kept Cow snapshot-resolves so cow_stock fires and
    # the rule stays silent -- 0 real occurrences.
    "milk_holdout": 1333, "milk_holdout_wrong": 1,
    # D2 crumb evidence (Pigeon sell -> fresh BreadCrumbs stock)
    "crossref_crumb_unis": 148,
    "crumb_holdout": 875, "crumb_holdout_wrong": 0,
}


def coverage_stats(records: list, pools: dict | None = None) -> Counter:
    """Coverage census over decoded records (moved from exp08 coverage.py):
    of all turns, how many are fully decoded (every shop state reconstructed +
    every buy identity-resolved), and where the rest fall (shop gaps /
    tier-up-or-ability buys = T4/T5 / unresolved identity)."""
    pools = pools or load_pools(POOLS_PATH)
    pet6 = set(pools["minion"]["6"])
    food6 = set(pools["food"]["6"])
    pet_cum = {t: set(pools["minion"][str(t)]) for t in range(1, 7)}

    c = Counter()
    for r in records:
        ss = r["start_shop"]["provenance"]
        # "genesis" = turn-1 opening shop read from the stored
        # GenesisBuildModel (T3, wave 5); a known shop like a seed-recon one.
        shops_ok = ss in ("reconstructed", "genesis")
        for roll in r["roll_shops"]:
            if roll["provenance"] != "frozen-tracked":
                shops_ok = False
        buys_ident = True
        has_tierup = has_ability = False
        for op in r["chain"]:
            if op["op"] in ("buy_pet", "buy_combine"):
                e = op.get("enu")
            elif op["op"] == "merge" and op.get("buy_merge"):
                e = op.get("src_enu")
            elif op["op"] == "buy_food":
                e = op.get("enu")
            else:
                continue
            if e is None:
                buys_ident = False
            elif op["op"] == "buy_food":
                if e not in food6:
                    has_ability = True
            else:  # pet buy
                if e not in pet_cum[r["tier"]] and e in pet6:
                    has_tierup = True
                elif e not in pet6:
                    has_ability = True

        if r["turn"] == 1:
            c["turn1_genesis_decoded" if ss == "genesis" else "turn1_genesis_unknown"] += 1
        if shops_ok and buys_ident:
            c["fully_decoded_identity"] += 1
            if not has_tierup and not has_ability:
                c["fully_decoded_rolled_only"] += 1
        else:
            if not shops_ok:
                c["shop_gap(genesis/frozen-unresolved)"] += 1
            if not buys_ident:
                c["buy_unresolved_identity"] += 1
        if has_tierup:
            c["has_tierup_buy(T4)"] += 1
        if has_ability:
            c["has_ability_buy(T5)"] += 1
    return c


def print_coverage(cov: Counter, n: int) -> None:
    print(f"total turns: {n}")
    for k in PINNED_COVERAGE:
        v = cov[k]
        print(f"  {k:42s} {v:6d}  ({v / n:.1%})" if n else f"  {k:42s} {v:6d}")


def _gate(checks: list, name: str, got, want) -> None:
    checks.append((name, got, want))


def collect_checks(agg: dict, n_games: int, cov: Counter) -> list:
    """Compare every pinned surface; return [(name, got, want), ...]."""
    checks: list = []
    _gate(checks, "games decoded", n_games, PINNED_TOTALS["games"])
    _gate(checks, "turns total", agg["n_total"], PINNED_TOTALS["turns"])
    _gate(checks, "turns with type4 seed", agg["n_type4"], PINNED_TOTALS["with_type4_seed"])
    _gate(checks, "no end-of-turn snapshot", agg["n_no_snapshot"], PINNED_TOTALS["no_end_snapshot"])
    _gate(checks, "turn-1 total", agg["n_turn1"], PINNED_TOTALS["turn1_total"])
    _gate(checks, "turn-1 genesis decoded", agg["n_turn1_genesis"], PINNED_TOTALS["turn1_genesis"])
    _gate(checks, "carry-in unknown", agg["n_carry_unknown"], PINNED_TOTALS["carry_unknown"])
    _gate(checks, "CreatedOn violations", agg["createdon_violations"],
          PINNED_TOTALS["createdon_violations"])

    for k, (n, p) in PINNED_LANES.items():
        b = agg["buckets"][k]
        _gate(checks, f"lane ({k}) n", b["n"], n)
        _gate(checks, f"lane ({k}) pass", b["pass"], p)
    for k, p in PINNED_LANE_TRACKED.items():
        _gate(checks, f"lane ({k}') pass_tracked", agg["buckets"][k]["pass_tracked"], p)
    for k, p in PINNED_LANE_INJ.items():
        _gate(checks, f"lane ({k}+) pass_inj", agg["buckets"][k]["pass_inj"], p)

    for note, n in PINNED_UNBUCKETED.items():
        _gate(checks, f"unbucketed [{note}]", agg["unbucketed_notes"].get(note, 0), n)
    _gate(checks, "unbucketed total", sum(agg["unbucketed_notes"].values()),
          sum(PINNED_UNBUCKETED.values()))

    fc = agg["run_stats"]["freeze_counts"]
    for k, v in PINNED_FREEZE.items():
        _gate(checks, f"freeze tracker [{k}]", fc.get(k, 0), v)
    m = agg["run_stats"]["membership"]
    for k, v in PINNED_MEMBERSHIP.items():
        _gate(checks, f"buy-membership [{k}]", m.get(k, 0), v)
    xr = agg["run_stats"]["xref"]
    for k, v in PINNED_XREF.items():
        _gate(checks, f"identity xref [{k}]", xr.get(k, 0), v)
    for k, v in PINNED_CHAIN.items():
        _gate(checks, f"chained anchoring [{k}]", xr.get(k, 0), v)
    # W3a evidence-quality lanes (position/milk/crumb), also in xref via pos_stats.
    for k, v in PINNED_EVIDENCE.items():
        _gate(checks, f"W3a evidence [{k}]", xr.get(k, 0), v)
    # structural invariant, not a pin (review C3): every crossref uni comes
    # from exactly one kept block -- double-claims would break the equality.
    _gate(checks, "INVARIANT minted positions == crossref unis",
          xr.get("minted_positions", 0), xr.get("crossref_unis", 0))

    for k, v in PINNED_COVERAGE.items():
        _gate(checks, f"coverage [{k}]", cov.get(k, 0), v)
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Frozen regression gate: decode the pinned Turtle cache and "
                     "assert the wave-5 settled numbers exactly.")
    ap.add_argument("--cache", type=Path, default=PINNED_CACHE,
                    help=f"replay cache (gate only fires on the pinned one: {PINNED_CACHE})")
    ap.add_argument("--max-games", type=int, default=None,
                    help="smoke run on a prefix of the cache (report-only, no gate)")
    args = ap.parse_args(argv)

    gated = (args.cache == PINNED_CACHE and args.max_games is None)
    t0 = time.time()
    records, agg, n_games = run(args.cache, max_games=args.max_games)
    if n_games == 0:
        print(f"FAIL: no games loaded from {args.cache}", file=sys.stderr)
        return 2
    cov = coverage_stats(records)
    dt = time.time() - t0

    print(f"cache: {args.cache}   games: {n_games}   turns: {agg['n_total']}   "
          f"decode+census: {dt:.0f}s")
    print()
    print_coverage(cov, agg["n_total"])
    print()

    if not gated:
        print("NON-PINNED input (custom --cache or --max-games): report only, no gate.")
        print("full self-check report: python -m sap_ppo.replay_decode.decode_turn --report-only")
        return 0

    checks = collect_checks(agg, n_games, cov)
    fails = [(name, got, want) for name, got, want in checks if got != want]
    for name, got, want in checks:
        mark = "ok  " if got == want else "FAIL"
        line = f"  {mark}  {name:45s} {got}"
        if got != want:
            line += f"   (pinned: {want})"
        print(line)
    print()
    if fails:
        print(f"*** REGRESSION: {len(fails)}/{len(checks)} pinned checks drifted. ***")
        print("If the change is intended, re-pin in sap_ppo/replay_decode/verify.py and")
        print("record why the counts moved before trusting the new baseline.")
        return 1
    print(f"PASS: all {len(checks)} pinned checks exact (W3a identity-recovery baseline, 2026-07-12).")
    print("reminder (artifact layer, not gated here): build_verification.py "
          "--reconcile-census -> expect 24328/25239 (96.39%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
