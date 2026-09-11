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
from .public_settings import SEARCH_MODES, engine_gear
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
        choices=["snapshot"],
        default=str(os.getenv(OPPONENT_SOURCE_ENV, DEFAULT_PLAY_WEB_OPPONENT_SOURCE)).strip().lower(),
    )
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=Path(str(os.getenv(SNAPSHOT_PATH_ENV, str(DEFAULT_REPLAY_SNAPSHOT_PATH))).strip()),
    )
    parser.add_argument("--game-mode", type=str, choices=["versus", "arena"], default=DEFAULT_GAME_MODE)
    parser.set_defaults(
        predictor_path=None, value_model_path=None, replay_cache=None,
        planner_depth=6, planner_beam_width=12, planner_samples_per_stochastic_action=3,
        planner_oracle_rerank_top=5, planner_oracle_sims=250, planner_predictor_top_k=5,
    )


    parser.add_argument("--no-duel", dest="duel", action="store_false", default=True,
                        help="disable the AI duel; Sandbox remains available")
    add_duel_rules_argument(parser, default=DEFAULT_DUEL_RULES)
    parser.add_argument("--duel-gear", type=str, choices=list(SEARCH_MODES), default="grow-k",
                        help="grow-k (default) or fixed-all; both use root=72, w=4, initial k=12")
    parser.add_argument("--duel-turn-budget-s", type=float, default=DEFAULT_TURN_BUDGET_S,
                        help="seconds per AI turn in Grow k (default: 105; range: 1-600)")
    parser.set_defaults(
        duel_width=72, duel_stochastic_samples=12, duel_completion_policy="v_search",
        duel_completion_width=4, duel_completion_tail=None, duel_chunk_size=None,
        duel_adaptive_chunks=False, duel_max_width=None,
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
    if not 1 <= args.duel_turn_budget_s <= 600:
        parser.error("--duel-turn-budget-s must be between 1 and 600")

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


            skip_imagined_validation=True,
            completion_policy=str(args.duel_completion_policy),
            completion_width=int(args.duel_completion_width),
            stochastic_samples=int(args.duel_stochastic_samples),
            **({"completion_tail": str(args.duel_completion_tail)}
               if args.duel_completion_tail else {}),
            **({"width": int(args.duel_width)} if args.duel_width else {}),
        )
        ai_config = AiTurnConfig(
            gear=engine_gear(args.duel_gear),
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


        print(
            "imagined-walk schema validation: "
            + (
                "SKIPPED (api.step and legal_actions on imagined states)"
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
        f"SAP-Arena-AI {build['version']} · commit {build['commit_short']}"
    )
    print(
        f"SAP web simulator on http://{args.host}:{args.port} "
        f"(mode={args.game_mode}, opponents={args.opponent_source}"
        + (f", snapshot={args.snapshot_path}" if args.opponent_source == "snapshot" else "")
        + f", predictor_loaded={bool(app.tempo_predictor is not None)}, value_model_loaded={bool(app.tempo_value_model is not None)}"
        + ")"
    )
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
