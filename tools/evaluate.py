#!/usr/bin/env python3
"""Evaluate complete arena games against the shipped same-turn opponent pool.

This is the benchmark entry point. It plays whole games: each turn the policy
acts in the shop, then the turn is ended against a board drawn from a real human
game at that same turn, and the battle is resolved by the pinned calculator.

    python tools/evaluate.py --policy random --games 20
    python tools/evaluate.py --policy search --games 20

`--policy random` needs nothing but the base install. `bc` and `search` load
trained weights and need the `agents` extra plus the published checkpoints; see
the README.

`--policy search` defaults to the PUBLISHED configuration, and every one of its
knobs is copied from that run's own manifest rather than from prose about it:
root width 72, ksim 16, `vgame` leaf scoring with blend and pessimism both 0,
segmented-honest turns, k=12 stochastic samples, and a `v_search` completion of
width 4 aggregated by `max` (then mean across chance samples).

This runner reports uncensored full-game ITT, not a human-matched score.
The two conventions for this runner are stated rather than assumed:

* mean trophies is ITT over the games LAUNCHED, so a game that fails to complete
  contributes 0 and still counts in the denominator;
* a draw is not a win.

To evaluate your own agent, implement the recommender contract illustrated by
`RandomPolicy.recommend` and pass it to `run_games`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from collections import Counter
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sap_ppo import api
from sap_ppo import _artifact_defaults
from sap_ppo.end_turn import resolve_end_turn_with_sampled_battle
from sap_ppo.opponents.chain_snapshot import ChainSnapshotSource
from sap_ppo.train.opening_source import build_varied_opening_source
# Reused, not reimplemented. The arena frame is not just "pass game_mode=arena":
# it starts on 5 lives rather than 6, sets a marker distinguishing this arena
# from the training env's 7-trophy one, applies the race convention, and sets
# seed_known=True so a replayed chain cannot land on a different shop. Rebuilding
# that by hand is how a frame silently drifts and every number becomes
# incomparable.
from sap_ppo.tools.eval_versus_fullgame import _new_game_state
from sap_ppo.tools.cluster_ci import cluster_ci

DEFAULT_POOL = "data/opponents/arena_val_pool_deidentified.json.gz"
MAX_TURN = 30
MAX_ACTIONS_PER_TURN = 60

# The published configuration, copied from the run manifest of the run that
# produced the reported number, not from a description of it.
PUBLISHED_SEARCH = {
    "search_candidates": 72,
    "search_ksim": 16,
    "search_scoring": "vgame",
    "turn_mode": "segmented-honest",
    "stochastic_samples": 12,
    "completion_policy": "v_search",
    "completion_width": 4,
    "completion_aggregate": "max",
}
# What the released value heads declare their output to MEAN. Checked rather
# than assumed: a head on a different value scale can silently change search.
RELEASED_VALUE_TARGET = "mc8_leafmap_trophies"


DEFAULT_BC_CHECKPOINT = _artifact_defaults.PLAY_WEB_BC_CHECKPOINT

# Digests of the released model files. The value file has one non-runtime
# training-path field removed; all parameters and serving metadata are unchanged.
# Identity has to be checked by DIGEST rather than by path, because the default
# above reads an environment variable: pointing SAP_PLAY_WEB_BC_CHECKPOINT at a
# different policy moves `DEFAULT_BC_CHECKPOINT` with it, so a path comparison
# would find no drift and the run would announce the published configuration
# while proposing with another agent.
PUBLISHED_BC_SHA256 = (
    "5ae652854896aa92273523b741c65fe45c10e461adffb7082f853b99221a64cc")
PUBLISHED_V_SHA256 = "baed69f0956ce88d2bebd886aaab1e5939ae7901eeb58e99ad690897c6a0ac39"
RELEASE_VERSION = "0.1.0"
ARENA_TERMINAL_REASONS = {"player_lives_0", "trophies_10"}
CI_ITERS, CI_SEED = 5000, 41
ARENA_RACE_CONVENTION = "const6"


class RandomPolicy:
    """Uniform over legal actions. The floor any real agent must clear."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def recommend(self, state: dict) -> dict:
        chain = []
        cur = state
        for _ in range(MAX_ACTIONS_PER_TURN):
            actions = api.legal_actions(cur)
            if not actions:
                break
            action = self._rng.choice(actions)
            if action.get("type") == "END_TURN":
                break
            transition = api.step(cur, action)
            if not transition["legal"]:
                break
            cur = transition["state_after"]
            chain.append(action)
        chain.append({"type": "END_TURN"})
        # The harness expects the project's recommender contract, not a bare
        # chain. Returning `{"chain": ...}` made every game fail at turn 0 with
        # `decode_failed`, which is what a policy that does not speak this
        # protocol looks like from the outside.
        return {
            "ok": True,
            "error": None,
            "recommended_action": chain[0] if chain else None,
            "chain_preview": chain,
            "wdl_probs": None,
            "diagnostics": {"stop_reason": "end_turn", "chain_length": len(chain)},
        }


def build_policy(name: str, args: argparse.Namespace):
    if name == "random":
        return RandomPolicy(seed=args.seed)
    # `--bc-checkpoint` defaults to None at the parser, and `BcRecommender` takes
    # the path as a required positional with no default of its own, so an unset
    # flag used to reach the loader as None and die on 'None.zip' -- from the
    # command README and REPRODUCE.md hand a reader verbatim. Resolved here, and
    # a missing file is refused with the same message the value heads use rather
    # than as a traceback about a filename nobody wrote. Checked BEFORE the torch
    # import below: refusing a path that is not there should not cost a
    # multi-second import, and it lets the refusal be tested without one.
    if not args.bc_checkpoint:
        raise SystemExit(
            "no behaviour-cloning checkpoint: --bc-checkpoint was not given and "
            "no default is configured (SAP_PLAY_WEB_BC_CHECKPOINT).")
    if not Path(args.bc_checkpoint).is_file():
        raise SystemExit(
            f"behaviour-cloning checkpoint not found: {args.bc_checkpoint}\n"
            "Fetch the weights first (see README.md):\n"
            "    python tools/fetch_models.py")
    print(f"  BC proposer: {args.bc_checkpoint}")
    # Imported lazily: these pull in torch, which the base install does not have.
    from sap_ppo.tools.bc_recommender import BcRecommender

    if name == "search":
        # Performance-only setting used by the reference execution. Committed
        # actions remain validated by the driver.
        api.set_skip_imagined_validation(True)
    bc = BcRecommender(checkpoint_path=args.bc_checkpoint)
    if name == "bc":
        return bc
    if name == "search":
        from sap_ppo.tools.search_recommender import SearchRecommender

        # The reported agent scores leaves with the released value heads
        # (`--search-scoring vgame`). Without them the search still runs, but it
        # falls back to myopic leaf scoring and is a WEAKER configuration than
        # the published one, so the difference is stated rather than hidden.
        vgame = None
        if args.search_scoring == "vgame":
            if not args.vgame_heads:
                raise SystemExit("--search-scoring vgame needs --vgame-heads")
            if not Path(args.vgame_heads).is_file():
                raise SystemExit(
                    f"value heads not found: {args.vgame_heads}\n"
                    "Fetch the weights first (see README.md):\n"
                    "    python tools/fetch_models.py")
            from sap_ppo.tools.vgame_scorer import VGameLeafScorer

            vgame = VGameLeafScorer.from_checkpoints(
                [args.vgame_heads],
                args.vgame_extractor or args.bc_checkpoint,
                blend=0.0,
                pessimism=0.0,
            )
            described = vgame.describe()
            target = str(described.get("target") or "")
            print(f"  value heads loaded: {args.vgame_heads}")
            print(f"  value target: {target or '<none declared>'}")
            # Only enforced when the shipped default is in use. Someone
            # evaluating their OWN heads is doing something legitimate; someone
            # running the default and getting a different scale is not, and the
            # numbers would look plausible either way.
            if args.vgame_heads == default_vgame_heads():
                if target != RELEASED_VALUE_TARGET:
                    raise SystemExit(
                        f"the default value heads declare target {target!r}, but the "
                        f"published configuration is {RELEASED_VALUE_TARGET!r}. These "
                        "are different value scales, so this would report a number "
                        "that is not the published one. Refusing rather than running.")
            elif target != RELEASED_VALUE_TARGET:
                print("  NOTE: these heads are not on the published value scale, "
                      "so this run is not the published configuration.")
        return SearchRecommender(
            bc,
            n_candidates=args.search_candidates,
            ksim=args.search_ksim,
            seed=args.seed,
            scoring=args.search_scoring,
            vgame_scorer=vgame,
            turn_mode=args.turn_mode,
            stochastic_samples=args.stochastic_samples,
            completion_policy=args.completion_policy,
            completion_width=args.completion_width,
            completion_aggregate=args.completion_aggregate,
            game_rules="arena",
            arena_race_convention=ARENA_RACE_CONVENTION,
            mc_rerank_k=0,
        )
    raise SystemExit(f"unknown policy: {name}")


def _sha256_or_none(path: str | None) -> str | None:
    """Digest of a file, or None if there is no readable file at `path`.

    Returns None rather than raising: this records provenance for a run that has
    already happened, and losing a finished measurement to a digest is worse than
    recording that the digest could not be taken.
    """
    if not path:
        return None
    try:
        with open(path, "rb") as fh:
            h = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def configuration_drift(args: argparse.Namespace) -> dict:
    """Which parts of the published configuration this run is NOT using.

    A function rather than four lines inside `main` so the predicate can be
    exercised without loading a checkpoint. The proposer belongs in here beside
    the eight knobs: the same search over a different proposer is a different
    agent, and it would otherwise be the one part of the configuration whose
    change is reported nowhere.

    The proposer is compared by CONTENT, not by path. `DEFAULT_BC_CHECKPOINT`
    reads an environment variable, so pointing that variable at another policy
    moves the default with it: a path comparison would then find no drift and the
    run would announce the published configuration while proposing with a
    different agent. A digest comparison also gets the benign case right, which a
    path comparison does not -- a reader who keeps the published weights in
    another directory is not running a different configuration.
    """
    drift = {k: (v, getattr(args, k)) for k, v in PUBLISHED_SEARCH.items()
             if getattr(args, k) != v}
    got = _sha256_or_none(args.bc_checkpoint)
    if got != PUBLISHED_BC_SHA256:
        drift["bc_checkpoint"] = (
            f"sha256:{PUBLISHED_BC_SHA256[:16]}",
            f"sha256:{got[:16]}" if got else f"unreadable ({args.bc_checkpoint})")
    for key, expected in (("vgame_heads", PUBLISHED_V_SHA256),
                          ("vgame_extractor", PUBLISHED_BC_SHA256)):
        path = getattr(args, key, None)
        got = _sha256_or_none(path)
        if got != expected:
            drift[key] = (f"sha256:{expected[:16]}",
                          f"sha256:{got[:16]}" if got else f"unreadable ({path})")
    return drift


def default_vgame_heads() -> str:
    """The first entry of the shipped default, or "" if none is configured."""
    heads = tuple(_artifact_defaults.VGAME_HEADS)
    return heads[0] if heads else ""


def run_games(policy, pool: str, games: int, split: str = "val",
              seed: int = 0, max_turn: int = 30,
              turn_mode: str = PUBLISHED_SEARCH["turn_mode"]) -> dict:
    """Play `games` whole games through the project's own evaluation loop.

    Deliberately NOT a private game loop. `run_versus_eval` carries the arena
    frame -- 5 starting lives, the trophy-target marker, the race convention, the
    opponent chain, one battle simulation per committed turn, and the honest turn
    decomposition. Every one of those is a way for a reimplementation to look
    fine and measure something else.
    """
    from sap_ppo.tools.eval_versus_fullgame import run_versus_eval

    source = ChainSnapshotSource(pool, split=split)
    opening = build_varied_opening_source(log=lambda _m: None)
    rows = run_versus_eval(
        policy,
        source,
        opening,
        num_games=games,
        max_turn=max_turn,
        seed=seed,
        log=True,
        opponent_mode="arena",
        arena_source=source,
        game_rules="arena",
        arena_race_convention=ARENA_RACE_CONVENTION,
        # The driver has its OWN turn mode, and it is not enough to set it on the
        # recommender: a segmented-honest search driven by a whole-determinized
        # loop is a third configuration that is neither arm.
        turn_mode=turn_mode,
    )
    return {**summarize(rows, launched=games), "games": rows}


def summarize(rows: list, launched: int) -> dict:
    """Terminal trophies, zero for nonterminal stops, over all launched games.

    Resolved battles before a failure still enter the pooled battle rate.
    Missing or malformed records are errors, not fabricated zero scores.
    """
    if type(launched) is not int or launched < 1 or len(rows) != launched:
        raise ValueError("expected exactly one result for each launched game")
    indices, trophy_pairs, completed_turns = set(), [], []
    ends, failures, outcomes = Counter(), Counter(), Counter()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid game record")
        index = row.get("game_index")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError("missing, invalid or duplicate game_index")
        indices.add(index)
        reason = row.get("end_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"game {index}: missing end_reason")
        turns = row.get("per_turn")
        if not isinstance(turns, list):
            raise ValueError(f"game {index}: missing per_turn")
        complete = reason in ARENA_TERMINAL_REASONS
        ends[reason] += 1
        if complete:
            trophies = row.get("trophies")
            survived = row.get("turns_survived")
            if type(trophies) is not int or not 0 <= trophies <= 10:
                raise ValueError(f"game {index}: invalid terminal trophies")
            if type(survived) is not int or survived < 1 or survived != len(turns):
                raise ValueError(f"game {index}: invalid turns_survived")
            if reason == "trophies_10" and trophies != 10:
                raise ValueError(f"game {index}: trophies_10 without ten trophies")
            completed_turns.append(survived)
        else:
            trophies = 0
            failures[reason] += 1
        trophy_pairs.append((index, trophies))
        wins = 0
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(f"game {index}: invalid turn record")
            outcome = turn.get("outcome")
            if outcome in {"win", "loss", "draw"}:
                outcomes[outcome] += 1
                wins += outcome == "win"
            elif complete or outcome not in (None, "unknown"):
                raise ValueError(f"game {index}: missing or invalid battle outcome")
        if complete and wins != trophies:
            raise ValueError(f"game {index}: trophies disagree with resolved wins")
    mean = sum(v for _, v in trophy_pairs) / launched
    ci = {"mean": mean, "lo95": None, "hi95": None, "n_games": launched,
          "method": "whole-game percentile bootstrap", "iters": CI_ITERS,
          "seed": CI_SEED, "unavailable_reason": None}
    if launched >= 2:
        _, ci["lo95"], ci["hi95"], _ = cluster_ci(sorted(trophy_pairs), iters=CI_ITERS, seed=CI_SEED)
    else:
        ci["unavailable_reason"] = "at least two games are required"
    battles = sum(outcomes.values())
    return {
        "games_launched": launched,
        "games_completed": len(completed_turns),
        "games_failed": sum(failures.values()),
        "end_reason_distribution": dict(sorted(ends.items())),
        "failures_by_reason": dict(sorted(failures.items())),
        "mean_trophies_itt": mean,
        "mean_trophies_itt_ci95": ci,
        "battle_outcomes": {k: outcomes[k] for k in ("win", "loss", "draw")},
        "battle_win_rate": outcomes["win"] / battles if battles else None,
        "battle_win_rate_unavailable_reason": None if battles else "no resolved battles",
        "n_battles": battles,
        "avg_turns_survived": sum(completed_turns) / len(completed_turns) if completed_turns else None,
        "metric_policy": "uncensored terminal trophies; nonterminal stops contribute zero; battle rate includes all resolved battles",
    }


#: Policies that load a torch model. `random` deliberately loads nothing, and
#: the README promises it "needs nothing but the base install" -- so the thread
#: cap below must not be the thing that drags torch into that promise.
_POLICIES_NEEDING_TORCH = ("bc", "search")


def intra_op_threads(requested: int) -> int:
    """Intra-op threads to use. `requested` of 0 means "pick the cap".

    A function rather than an expression inside `main` so the choice can be
    tested without loading a checkpoint, and so a mutant that removes the cap
    is caught by a test of the OUTPUT rather than by a test that a particular
    line exists. The cap is never raised above what the machine reports, so a
    two-core laptop gets two, not eight.
    """
    if requested:
        return max(1, int(requested))
    return max(1, min(8, os.cpu_count() or 1))


def apply_thread_cap(policy: str, requested: int) -> int | None:
    """Cap torch's intra-op pool for the policies that use torch. Returns the
    value set, or None when the policy loads no model.

    The import lives HERE, behind the policy test, rather than at the top of
    `main`. An earlier version of this fix imported torch unconditionally and
    broke `--policy random` in the acceptance rig's isolated venv, which does
    not install torch -- the cap is a performance fix and it must not quietly
    add a dependency to the one policy documented as needing none.
    """
    if policy not in _POLICIES_NEEDING_TORCH:
        return None
    import torch

    want = intra_op_threads(requested)
    torch.set_num_threads(want)
    return int(torch.get_num_threads())


def render_saved_gallery(records: Path, output: Path, games: int) -> dict:
    """Only consume the saved records; no policy or simulator is passed in."""
    spec = importlib.util.spec_from_file_location(
        "public_gallery", Path(__file__).with_name("gallery.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_records(records, output, max_games=games)


def _update_own_summary(path: Path, payload: dict) -> None:
    """Atomically update the result file created by this invocation."""
    pending = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            pending = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(pending, path)
    finally:
        if pending is not None and pending.exists():
            pending.unlink()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="search", choices=("random", "bc", "search"))
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--pool", default=DEFAULT_POOL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="intra-op threads for the value model. 0 means the "
                         "capped default below, which is deliberately NOT "
                         "torch's own default; see the note at its use.")
    ap.add_argument("--bc-checkpoint", default=None)
    ap.add_argument("--search-candidates", type=int, default=PUBLISHED_SEARCH["search_candidates"])
    ap.add_argument("--search-ksim", type=int, default=PUBLISHED_SEARCH["search_ksim"])
    ap.add_argument("--search-scoring", default=PUBLISHED_SEARCH["search_scoring"],
                    choices=("myopic", "vgame"))
    ap.add_argument("--turn-mode", default=PUBLISHED_SEARCH["turn_mode"],
                    choices=("whole-determinized", "segmented-honest"))
    ap.add_argument("--stochastic-samples", type=int, default=PUBLISHED_SEARCH["stochastic_samples"])
    ap.add_argument("--completion-policy", default=PUBLISHED_SEARCH["completion_policy"],
                    choices=("bc_greedy", "v_search"))
    ap.add_argument("--completion-width", type=int, default=PUBLISHED_SEARCH["completion_width"])
    ap.add_argument("--completion-aggregate", default=PUBLISHED_SEARCH["completion_aggregate"],
                    choices=("max", "mean"))
    ap.add_argument("--vgame-heads", default=None)
    ap.add_argument("--vgame-extractor", default=None)
    ap.add_argument("--out", default=None, help="summary JSON; also writes a sibling .games.jsonl")
    ap.add_argument("--records", default=None, help="per-game JSONL path (overrides the --out-derived path)")
    ap.add_argument("--gallery-games", type=int, default=5,
                    help="render the first N recorded games after saving results; 0 disables")
    ap.add_argument("--gallery-out", type=Path, default=None,
                    help="new gallery directory; default <out-stem>.gallery")
    ap.add_argument("--print-config", action="store_true",
                    help="print the published configuration as JSON and exit, without "
                         "loading the pool or any checkpoint")
    args = ap.parse_args()
    if args.games < 1:
        ap.error("--games must be positive")
    if args.gallery_games < 0:
        ap.error("--gallery-games must be nonnegative")
    if args.gallery_out is not None and not args.gallery_games:
        ap.error("--gallery-out requires --gallery-games greater than zero")
    if args.vgame_heads is None:
        args.vgame_heads = default_vgame_heads() or None
    if args.vgame_extractor is None:
        args.vgame_extractor = _artifact_defaults.VGAME_EXTRACTOR
    if args.bc_checkpoint is None:
        args.bc_checkpoint = DEFAULT_BC_CHECKPOINT
    if args.print_config:
        # Printed from the same dict argparse takes its defaults from, so this
        # cannot drift from what a run actually uses. Reading the values out of
        # a docstring or a table could.
        print(json.dumps({"published_configuration": PUBLISHED_SEARCH,
                          "requested_configuration": {key: getattr(args, key) for key in PUBLISHED_SEARCH},
                          "policy": args.policy,
                          "vgame_blend": 0.0, "vgame_pessimism": 0.0, "mc_rerank_k": 0,
                          "game_rules": "arena", "arena_race_convention": ARENA_RACE_CONVENTION,
                          "skip_imagined_validation": args.policy == "search",
                          "release_version": RELEASE_VERSION,
                          "bc_checkpoint": args.bc_checkpoint,
                          "vgame_heads": args.vgame_heads,
                          "vgame_extractor": args.vgame_extractor,
                          "expected_bc_sha256": PUBLISHED_BC_SHA256,
                          "expected_vgame_sha256": PUBLISHED_V_SHA256,
                          "released_value_target": RELEASED_VALUE_TARGET}, indent=2,
                         sort_keys=True))
        return 0
    auto_output = not args.out and args.gallery_games > 0
    if auto_output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        args.out = str(Path("results") / f"eval-{stamp}-{uuid.uuid4().hex[:8]}" / "summary.json")
    records_path = Path(args.records) if args.records else (
        Path(args.out).with_suffix(".games.jsonl") if args.out else None)
    gallery_path = (args.gallery_out or Path(args.out).with_suffix(".gallery")) if args.gallery_games else None
    destinations = ([Path(args.out)] if args.out else []) + ([records_path] if records_path else [])
    if len({p.resolve() for p in destinations}) != len(destinations):
        ap.error("summary and per-game records must have different paths")
    for path in destinations:
        if path.exists() or path.is_symlink():
            ap.error(f"refusing to overwrite existing result: {path}")
        if not path.parent.is_dir() and not (auto_output and path.parent == Path(args.out).parent):
            ap.error(f"result directory does not exist: {path.parent}")
    if gallery_path is not None:
        if gallery_path.resolve() in {p.resolve() for p in destinations}:
            ap.error("gallery, summary and per-game records must have different paths")
        if gallery_path.exists() or gallery_path.is_symlink():
            ap.error(f"refusing to overwrite existing gallery: {gallery_path}")

    if not Path(args.pool).is_file():
        print(f"FATAL: opponent pool not found: {args.pool}", file=sys.stderr)
        return 2

    # Torch defaults its intra-op pool to the machine's core count, and for THIS
    # workload that default is not merely suboptimal, it is fatal. One search
    # decision scores thousands of tiny leaves, one forward pass each, so a
    # parallel region is opened and joined thousands of times over arithmetic far
    # too small to spread; the scheduling cost is what scales with core count.
    # Measured in this exported tree under a 64-core allowance: two turns of
    # `--policy search` took 56.6 s at 4 and at 8 threads, 59.2 s at 1, and did
    # NOT finish inside 900 s at torch's own default of 64. A whole game takes
    # about 450 s once the pool is capped.
    #
    # So the cap is part of the shipped command, not advice in a doc. Two
    # segments of this project read the uncapped behaviour as "this configuration
    # is expensive" and once as "the machine was busy", and neither was true.
    # A command that never returns is not slow, it is broken, and nothing in this
    # repository would have said so.
    set_to = apply_thread_cap(args.policy, args.torch_threads)
    if set_to is not None:
        print(f"  torch intra-op threads: {set_to}"
              f" (machine reports {os.cpu_count()})")

    policy = build_policy(args.policy, args)
    if auto_output:
        Path(args.out).parent.mkdir(parents=True, exist_ok=False)
    print(f"policy={args.policy} games={args.games} pool={args.pool}")
    if args.policy == "search":
        drift = configuration_drift(args)
        if drift:
            print("  NOT the published configuration; these knobs were changed:")
            for k, (want, got) in sorted(drift.items()):
                print(f"    {k}: published={want} here={got}")
        else:
            print("  configuration: root72/w4/k12 with the released BC and value model hashes verified")
    result = run_games(policy, args.pool, args.games, seed=args.seed,
                       turn_mode=args.turn_mode)
    rows = result.pop("games")
    summary = result
    if records_path:
        with records_path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary["records_file"] = str(records_path)
        summary["records_sha256"] = _sha256_or_none(str(records_path))
    print()
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if args.out:
        # Record the whole configuration, not just the policy name. A myopic and
        # a value-guided search at the same width are different measurements and
        # were previously written to files that looked identical.
        search = args.policy == "search"
        config = {
            "policy": args.policy,
            "games": args.games,
            "pool": args.pool,
            "pool_sha256": _sha256_or_none(args.pool),
            "pool_split": "val", "game_rules": "arena", "opponent_mode": "arena",
            "arena_race_convention": ARENA_RACE_CONVENTION,
            "skip_imagined_validation": search,
            "opening_mode": "varied",
            "max_turn": MAX_TURN, "torch_threads": set_to,
            "release_version": RELEASE_VERSION,
            "release_source": (json.loads(Path("release_info.json").read_text())
                               if Path("release_info.json").is_file() else None),
            "seed": args.seed,
            "gallery_games_requested": args.gallery_games,
            "gallery_output": str(gallery_path) if gallery_path else None,
            "search_scoring": args.search_scoring if search else None,
            "search_candidates": args.search_candidates if search else None,
            "search_ksim": args.search_ksim if search else None,
            # Recorded because a run at a different completion or turn mode is a
            # different measurement, and files that omitted them looked identical.
            "turn_mode": args.turn_mode if search else None,
            "stochastic_samples": args.stochastic_samples if search else None,
            "completion_policy": args.completion_policy if search else None,
            "completion_width": args.completion_width if search else None,
            "completion_aggregate": args.completion_aggregate if search else None,
            "vgame_blend": 0.0 if search else None,
            "vgame_pessimism": 0.0 if search else None,
            "mc_rerank_k": 0 if search else None,
            # The same predicate the printed drift report uses, so the file and
            # the console cannot disagree about whether this was the published
            # run -- and so this flag covers the proposer's identity too.
            "is_published_config": search and not configuration_drift(args),
            "bc_checkpoint": args.bc_checkpoint,
            "vgame_heads": args.vgame_heads,
            "vgame_extractor": args.vgame_extractor,
            # Paths are not identity. Two runs can name the same file and load
            # different bytes, and the value-head files in particular are a pair
            # whose scales differ by an order of magnitude while both look
            # ordinary. The digest is what a later reader can check MODELS.md
            # against.
            "bc_checkpoint_sha256": _sha256_or_none(args.bc_checkpoint),
            "vgame_heads_sha256": _sha256_or_none(args.vgame_heads) if search else None,
            "vgame_extractor_sha256": _sha256_or_none(args.vgame_extractor) if search else None,
        }
        payload = {"config": config, **summary,
                   "gallery": {"status": "pending" if args.gallery_games else "disabled"}}
        with Path(args.out).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        print(f"  wrote {args.out}")
    gallery_failed = False
    if args.gallery_games:
        # Persist all evaluation evidence BEFORE loading or calling a renderer.
        try:
            gallery = render_saved_gallery(records_path, gallery_path, args.gallery_games)
        except Exception as exc:
            gallery_failed = True
            gallery = {"status": "failed", "error": str(exc), "output": str(gallery_path)}
            print(f"GALLERY FAILED: {exc}; evaluation results retained at {args.out}", file=sys.stderr)
        payload["gallery"] = gallery
        _update_own_summary(Path(args.out), payload)
        print("  gallery: " + json.dumps(gallery, sort_keys=True))
    if summary["games_failed"]:
        print("FAIL: one or more games stopped before an arena terminal; results retained")
        return 1
    return 2 if gallery_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
