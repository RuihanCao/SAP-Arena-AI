"""Command-line entry point and server bootstrap for the web simulator."""

from __future__ import annotations

import argparse
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

from ...api import skip_imagined_validation
from ..search_recommender import COMPLETION_TAILS
from .agent import demo_agent_config
from .ai_worker import DEFAULT_TURN_BUDGET_S, GEAR_RESAMPLE_CLOCK, GEARS, AiTurnConfig
from .app import (
    DEFAULT_GAME_MODE,
    DEFAULT_PREDICTOR_PATH,
    DEFAULT_REPLAY_SNAPSHOT_PATH,
    DEFAULT_SURFACE,
    DEFAULT_VALUE_MODEL_PATH,
    SURFACE_AGENT,
    SURFACES,
    App,
)
from .assets import ROOT
from .build_identity import build_identity
from .archive import DEFAULT_DUEL_ARCHIVE_ROOT
from .duel import DEFAULT_DUEL_RULES, add_duel_rules_argument
from .duel_app import DEFAULT_VALUE_ACTION_CAP, DuelApp
from .http_app import Handler

DEFAULT_PLAY_WEB_OPPONENT_SOURCE = "snapshot"
OPPONENT_SOURCE_ENV = "SAP_PPO_OPPONENT_SOURCE"
SNAPSHOT_PATH_ENV = "SAP_PPO_REPLAY_SNAPSHOT_PATH"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SAP web simulator")
    parser.add_argument(
        "--surface", type=str, choices=list(SURFACES), default=DEFAULT_SURFACE,
        help=(
            "who this server is for. 'human' is the full page: the menu, the "
            "duel at /play, and the composed drag gestures. 'agent' is the "
            "agent's environment made visible -- the shop page and the engine "
            "API only, offering exactly constants.ACTION_TYPES, with no duel "
            "and no composed gestures. 'agent' implies --no-duel."
        ),
    )
    parser.add_argument("--fixture", type=Path, default=ROOT / "fixtures" / "parity_cases" / "sample_case.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--opponent-source",
        type=str,
        choices=["db", "snapshot"],
        default=str(os.getenv(OPPONENT_SOURCE_ENV, DEFAULT_PLAY_WEB_OPPONENT_SOURCE)).strip().lower(),
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path(str(os.getenv(SNAPSHOT_PATH_ENV, str(DEFAULT_REPLAY_SNAPSHOT_PATH))).strip()),
    )
    parser.add_argument("--game-mode", type=str, choices=["versus", "arena"], default=DEFAULT_GAME_MODE)
    parser.add_argument("--predictor-path", type=Path, default=DEFAULT_PREDICTOR_PATH)
    parser.add_argument("--value-model-path", type=Path, default=DEFAULT_VALUE_MODEL_PATH)
    parser.add_argument("--planner-depth", type=int, default=6)
    parser.add_argument("--planner-beam-width", type=int, default=12)
    parser.add_argument("--planner-samples-per-stochastic-action", type=int, default=3)
    parser.add_argument("--planner-oracle-rerank-top", type=int, default=5)
    parser.add_argument("--planner-oracle-sims", type=int, default=250)
    parser.add_argument("--planner-predictor-top-k", type=int, default=5)
    # Default None rather than the viewer's own constant: resolving that
    # constant here would import the viewer at startup, which a tree that does
    # not ship it cannot do. The App resolves the real default when the viewer
    # is available.
    parser.add_argument("--replay-cache", type=Path, default=None,
                        help="raw-replay jsonl.gz for the /replay player "
                             "(optional; the viewer is disabled without it)")

    # exp16 W5: the duel at /play. Constructing it is cheap (a fixture read and
    # the catalog); the agent and torch are not loaded until the first game.
    parser.add_argument("--no-duel", dest="duel", action="store_false", default=True,
                        help="serve /sandbox and /replay only; /play answers duel_unavailable")
    add_duel_rules_argument(parser, default=DEFAULT_DUEL_RULES)
    parser.add_argument("--duel-gear", type=str, choices=list(GEARS), default=GEAR_RESAMPLE_CLOCK,
                        help="resample-clock: grow k (default); measured: fixed-all; full-clock: laboratory width growth")
    parser.add_argument("--duel-width", type=int, default=None,
                        help="root proposals per segment for fixed-all and grow-k (default: 72)")
    parser.add_argument("--duel-stochastic-samples", type=int, default=12,
                        help="initial chance samples per stochastic group (default: 12); grow-k adds complete equal-k levels")
    parser.add_argument("--duel-chunk-size", type=int, default=None,
                        help="fixed laboratory chunk size (default: 1)")
    parser.add_argument(
        "--duel-adaptive-chunks",
        action="store_true",
        default=False,
        help="laboratory opt-in to adaptive batching after benchmark review",
    )
    parser.add_argument(
        "--duel-turn-budget-s",
        type=float,
        default=DEFAULT_TURN_BUDGET_S,
        help="wall-clock seconds per AI turn (default: 105)",
    )
    parser.add_argument(
        "--duel-completion-policy",
        type=str,
        choices=("bc_greedy", "v_search"),
        default="v_search",
        help=(
            "what fills the rest of the turn behind a chance node while the "
            "search is still ranking: v_search (default) takes the maximum V "
            "over --duel-completion-width completions for each chance sample; "
            "bc_greedy is a laboratory alternative requiring width 1"
        ),
    )
    parser.add_argument(
        "--duel-completion-tail",
        default=None,
        choices=list(COMPLETION_TAILS),
        help=(
            "What a sampled completion does when its walk would stop before "
            "the turn is over. Default `resample` (draw again past a repeat "
            "instead of giving up, and do not break at a chance node inside a "
            "completion). `finish_greedy` hands the tail to the greedy "
            "decoder; `stop` is the pre-2026-08-21 behaviour and is what "
            "reproduces results measured before then."
        ),
    )
    parser.add_argument(
        "--duel-completion-width",
        type=int,
        default=4,
        help=(
            "imagined completions per chance outcome (default 4); "
            "must be >= 2 under v_search, or exactly 1 for bc_greedy"
        ),
    )
    parser.add_argument(
        "--duel-max-width",
        type=int,
        default=None,
        help=(
            "safety-valve ceiling on candidates generated per segment under the "
            "fixed clock (default: the code's own, currently 32768). This is NOT "
            "a search budget: the segment deadline is the normal stop, and a run "
            "that stops here instead records stop_reason=safety_cap so it is "
            "visible. Lower it only to reproduce an older run"
        ),
    )
    parser.add_argument("--duel-turn-cap", type=int, default=30)
    parser.add_argument(
        "--duel-archive-root",
        type=Path,
        default=DEFAULT_DUEL_ARCHIVE_ROOT,
        help="durable per-turn duel archive root",
    )
    parser.add_argument(
        "--no-duel-archive",
        action="store_true",
        default=False,
        help="disable writing games played at /play",
    )
    # exp16 show-value. On by default -- it is the feature. Off is what
    # `DESIGN_show_value.md` acceptance 4's byte-identity arm runs on, and what
    # a server with no recalibration curve on disk would want.
    parser.add_argument(
        "--no-duel-show-value",
        dest="duel_show_value",
        action="store_false",
        default=True,
        help="hide the agent's valuation of the human's board at /play",
    )
    parser.add_argument(
        "--duel-value-action-cap",
        type=int,
        default=DEFAULT_VALUE_ACTION_CAP,
        help=(
            "how many legal actions one per-action valuation will score "
            f"(default: {DEFAULT_VALUE_ACTION_CAP}); each one costs a greedy "
            "BC completion"
        ),
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.duel_stochastic_samples < 1:
        parser.error("--duel-stochastic-samples must be positive")

    if not args.fixture.exists():
        raise FileNotFoundError(f"Fixture not found: {args.fixture}")

    agent_surface = args.surface == SURFACE_AGENT
    # The duel is the human entry point AND the only thing that turns the
    # imagined-walk validation skip on, so the agent surface declines it
    # rather than relying on the operator to pass --no-duel as well.
    serve_duel = bool(args.duel) and not agent_surface

    app = App(
        args.fixture,
        surface=args.surface,
        opponent_source=args.opponent_source,
        snapshot_path=args.snapshot_path,
        game_mode=args.game_mode,
        predictor_path=args.predictor_path,
        value_model_path=args.value_model_path,
        planner_depth=args.planner_depth,
        planner_beam_width=args.planner_beam_width,
        planner_samples_per_stochastic_action=args.planner_samples_per_stochastic_action,
        planner_oracle_rerank_top=args.planner_oracle_rerank_top,
        planner_oracle_simulation_count=args.planner_oracle_sims,
        planner_predictor_top_k=args.planner_predictor_top_k,
        replay_cache=args.replay_cache,
    )
    duel_app: DuelApp | None = None
    if serve_duel:
        agent_config = demo_agent_config(
            # exp13 W0b' skip lever: imagined walks only, never a committed op.
            # On for interactive play (W3 measured 2.0x on this path); the
            # numbers a duel produces are per-turn timings, not eval results.
            skip_imagined_validation=True,
            completion_policy=str(args.duel_completion_policy),
            completion_width=int(args.duel_completion_width),
            stochastic_samples=int(args.duel_stochastic_samples),
            **({"completion_tail": str(args.duel_completion_tail)}
               if args.duel_completion_tail else {}),
            **({"width": int(args.duel_width)} if args.duel_width else {}),
        )
        ai_config = AiTurnConfig(
            gear=str(args.duel_gear),
            width=agent_config.width,
            turn_budget_s=float(args.duel_turn_budget_s),
            adaptive_chunks=bool(args.duel_adaptive_chunks),
            **({"chunk_size": int(args.duel_chunk_size)} if args.duel_chunk_size else {}),
            **(
                {"full_clock_max_width": int(args.duel_max_width)}
                if args.duel_max_width
                else {}
            ),
        )
        duel_app = DuelApp(
            args.fixture,
            agent_config=agent_config,
            ai_config=ai_config,
            rules=str(args.duel_rules),
            turn_cap=int(args.duel_turn_cap),
            archive_root=None if args.no_duel_archive else args.duel_archive_root,
            show_value=bool(args.duel_show_value),
            value_action_cap=int(args.duel_value_action_cap),
        )

    if agent_surface and skip_imagined_validation():
        # Nothing in this entry point should be able to reach here: the only
        # caller of `set_skip_imagined_validation` on this path is the duel's
        # agent, which the agent surface does not build. If some future import
        # flips the process-wide switch anyway, refuse to serve rather than
        # advertise an environment whose imagined walks are unvalidated.
        raise RuntimeError(
            "--surface agent refuses to start with the imagined-walk validation "
            "skip already on: this surface exists to show the agent's "
            "environment honestly, and that includes its validators."
        )

    Handler.app = app
    Handler.duel_app = duel_app
    if duel_app is not None:
        # THE REDEPLOY LINE. `skip_imagined_validation` is set above and has
        # been for a while, but exp13 PR #101 widened what the flag covers: it
        # now also skips schema validation inside `legal_actions`, so a service
        # redeployed from a tree containing that change stops validating the
        # legal-action mask -- with nothing on screen, in the log, or in the
        # archive saying so. internal design notes has carried
        # that as a pending hazard since 2026-08-06 precisely because neither
        # `:8765` nor `:8766` had been redeployed yet.
        #
        # The evidence that it does not change what gets played is exp13's own
        # byte-identity gate over 63 real decisions
        # (internal design notes section 7). This line
        # does not decide anything; it makes the next redeploy announce the
        # change instead of making it silently.
        #
        # STDERR AND `flush=True`, both load-bearing. `ops/play-web/dev8766.sh`
        # starts the server under `nohup`, where stdout is block-buffered and a
        # startup line can sit unwritten for the life of the process -- which is
        # why none of this file's other startup prints appear in `dev.log` until
        # something flushes them. `BaseHTTPRequestHandler.log_message` writes to
        # stderr, which is where the log an operator actually reads comes from.
        # A warning nobody can see is not a warning.
        print(
            "imagined-walk schema validation: "
            + (
                "SKIPPED (api.step AND legal_actions, since exp13 PR #101; "
                "byte-identity gate: see the speedup results notes)"
                if agent_config.skip_imagined_validation
                else "ON"
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "value readout at /play: "
            + ("on" if duel_app.show_value else "off")
            + f" (POST /api/infer/value, action cap {duel_app.value_action_cap})",
            file=sys.stderr,
            flush=True,
        )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    build = build_identity()
    print(
        f"surface={args.surface} duel={'on' if duel_app is not None else 'off'} "
        f"compose_gestures={'on' if app.compose_enabled else 'off'} "
        f"imagined_validation_skipped={skip_imagined_validation()}"
    )
    print(
        f"serving commit {build['commit_short']} ({build['head_ref']}) "
        f"from {build['repo_root']}  [module: {build['module_path']}]"
    )
    print(
        f"SAP web simulator on http://{args.host}:{args.port} "
        f"(mode={args.game_mode}, opponents={args.opponent_source}"
        + (f", snapshot={args.snapshot_path}" if args.opponent_source == "snapshot" else "")
        + f", predictor_loaded={bool(app.tempo_predictor is not None)}, value_model_loaded={bool(app.tempo_value_model is not None)}"
        + ")"
    )
    print(f"replay player (exp09 W2): http://{args.host}:{args.port}/replay "
          f"(cache={args.replay_cache})")
    if agent_surface:
        print(
            f"agent surface: http://{args.host}:{args.port}/ (= /sandbox)  |  "
            "no /play, no /replays, no composed drag gestures  |  "
            f"build: http://{args.host}:{args.port}/api/build"
        )
    else:
        print(
            f"menu: http://{args.host}:{args.port}/  |  duel: /play "
            + (
                f"(rules={args.duel_rules}, gear={duel_app.ai_config.gear}, "
                f"turn_budget_s={duel_app.ai_config.turn_budget_s}, "
                f"adaptive_chunks={duel_app.ai_config.adaptive_chunks}) "
                if duel_app is not None
                else "(disabled) "
            )
            + " |  free practice: /sandbox"
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if duel_app is not None:
            duel_app.shutdown()
