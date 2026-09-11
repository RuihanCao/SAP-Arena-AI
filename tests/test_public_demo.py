"""Public demo presets and response metadata; no model load or live games."""

import copy
from dataclasses import replace
import io
import json
import threading
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sap_ppo.tools.play_web.agent import demo_agent_config
from sap_ppo.tools.play_web.archive import DuelArchive
from sap_ppo.tools.play_web.assets import static_path
from sap_ppo.tools.play_web.ai_worker import AiTurnConfig, CLOCKED_GEARS
from sap_ppo.tools.play_web.cli import _build_arg_parser
from sap_ppo.tools.play_web.duel_app import DuelApp
from sap_ppo.tools.play_web.http_app import Handler
from sap_ppo.tools.play_web.public_metadata import public_payload
from sap_ppo.tools.play_web.public_settings import engine_gear, turn_seconds


class DemoPresetTests(unittest.TestCase):
    def test_sandbox_end_screen_script_is_served(self):
        script = static_path("sandbox.js")
        self.assertIsNotNone(script)
        self.assertTrue(script.is_file())

    def make_app(self):
        app = object.__new__(DuelApp)
        app._lock = threading.RLock()
        app.agent_config = demo_agent_config(width=999, completion_width=8, stochastic_samples=3)
        app.ai_config = AiTurnConfig(gear="full-clock", width=999, turn_budget_s=599)
        app.agent = SimpleNamespace(bc=object(), vgame_scorer=object())
        app.worker = SimpleNamespace(config=app.ai_config)
        app._build_search_fn = Mock(return_value=object())
        app.snapshot = lambda: {"gear": app.gear_config()}
        return app

    def test_cli_defaults_and_only_two_modes(self):
        parser = _build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual((args.duel_gear, args.duel_turn_budget_s), ("grow-k", 105))
        self.assertEqual((args.duel_width, args.duel_completion_width, args.duel_stochastic_samples), (72, 4, 12))
        self.assertEqual(args.duel_rules, "versus")
        with patch("sys.stderr", new=io.StringIO()):
            for options in (["--duel-gear", "full-clock"], ["--duel-width", "9"], ["--duel-completion-width", "8"]):
                with self.subTest(options=options), self.assertRaises(SystemExit):
                    parser.parse_args(options)

    def test_new_game_presets_reset_old_parameters_and_clock(self):
        app = self.make_app()
        for mode, seconds in [("fixed-all", None), ("grow-k", 11), ("fixed-all", None), ("grow-k", None)]:
            pending = app._plan_settings(None, seconds, mode, None)
            app._install_settings(pending)
            cfg = app.gear_config()
            self.assertEqual(cfg["gear"], mode)
            self.assertEqual(cfg["gears"], ["grow-k", "fixed-all"])
            self.assertEqual((cfg["width"], cfg["completion_width"], cfg["stochastic_samples"]), (72, 4, 12))
            self.assertEqual(cfg["turn_budget_s"], seconds or 105)
            self.assertEqual(app.ai_config.gear in CLOCKED_GEARS, mode == "grow-k")
            self.assertIs(app.worker.config, app.ai_config)
            self.assertEqual(app.agent.config, app.agent_config)

    def test_obsolete_default_falls_back_but_explicit_invalid_modes_rejected(self):
        app = self.make_app()
        self.assertEqual(app._plan_settings(None, None)["gear"], engine_gear("grow-k"))
        for mode in ("full-clock", "measured", "resample-clock", "unknown"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                app._plan_settings(None, None, mode)
        for args in [(8, None, "grow-k", None), (None, None, "grow-k", 73)]:
            with self.assertRaises(ValueError):
                app._plan_settings(*args)

    def test_time_validation(self):
        for value in (0, -1, 601, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                turn_seconds(value)
        self.assertEqual(turn_seconds(105), 105)


class PublicMetadataTests(unittest.TestCase):
    def test_projection_preserves_numeric_data_and_original_records(self):
        data = {"head_ref": "personal-branch", "repo_root": "/home/owner/repo", "turns": [{
            "gear": "resample-clock", "stochastic_samples": 12,
            "score": 4.715, "board": {"gold": 7},
        }], "ai_version": {"agent_id": "arm-c-r0-a2", "agent_name": "Arm C",
            "bc_checkpoint": {"path": r"C:\models\bc_attn_v4.zip", "sha256": "abc"}},
            "error": "FileNotFoundError: /root/workspace/model.pt"}
        before = copy.deepcopy(data)
        out = public_payload(data)
        self.assertEqual(data, before)
        self.assertNotIn("head_ref", out)
        self.assertNotIn("repo_root", out)
        self.assertEqual(out["turns"][0], {"gear": "grow-k", "stochastic_samples": 12, "score": 4.715, "board": {"gold": 7}})
        self.assertEqual(out["ai_version"]["bc_checkpoint"], {"path": "bc_attn_v4.zip", "sha256": "abc"})
        self.assertEqual(out["ai_version"]["agent_id"], "bc-value-r0-a2")
        self.assertEqual(out["error"], "FileNotFoundError: <local path>")
        self.assertEqual(public_payload({"url": "https://github.com/example/project", "package": "sap_ppo"}),
                         {"url": "https://github.com/example/project", "package": "sap_ppo"})

    def test_http_json_boundary_uses_projection(self):
        handler = object.__new__(Handler)
        handler._send = Mock()
        handler._send_json(200, {"repo_root": "/home/owner/repo", "gear": "measured", "width": 72})
        args = handler._send.call_args.args
        self.assertEqual(json.loads(args[1]), {"gear": "fixed-all", "width": 72})

    def test_public_api_rejects_midgame_configuration_changes(self):
        handler = object.__new__(Handler)
        handler.path = "/api/duel/config"
        handler.app = SimpleNamespace(surface="human")
        handler._parse_post = lambda: {"gear": "grow-k", "search_width": 999}
        duel = SimpleNamespace(snapshot=Mock(return_value={}), set_config=Mock())
        handler._duel = lambda: duel
        handler._send_state_json = Mock()
        handler.do_POST()
        self.assertEqual(handler._send_state_json.call_args.args[0], 400)
        self.assertFalse(handler._send_state_json.call_args.args[1]["ok"])
        duel.set_config.assert_not_called()


class ArchivePresetTests(unittest.TestCase):
    def test_new_archive_uses_public_modes_and_keeps_actual_search_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = DuelArchive(Path(temp) / "archive")
            game_id = archive.begin_game({"seeds": {"game": 7}, "rules": {"start_lives": 6},
                                         "gear_start": {"gear": "measured", "width": 72}})
            record = {"turn": 1, "gear": "measured", "stochastic_samples": 12,
                      "completion_width": 4, "segment_widths": [{"width": 72}]}
            original = copy.deepcopy(record)
            archive.record_turn(game_id, record, final={}, done=True, winner="human", end_reason="lives")
            game = archive.load_game(game_id)
            self.assertEqual(record, original)
            self.assertEqual(game["gear_start"]["gear"], "fixed-all")
            self.assertEqual(game["turns"][0], {**record, "gear": "fixed-all"})
            self.assertEqual(archive.list_games()[0]["gear"], "fixed-all")


if __name__ == "__main__":
    unittest.main()
