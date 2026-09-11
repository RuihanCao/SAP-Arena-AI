"""Render-payload compatibility checks; no renderer install or games required."""

import copy
import gzip
import io
import json
import shutil
import unittest
from unittest.mock import patch

from sap_ppo.opponents import replaybot_render_bridge as bridge


class RawBattleRenderPayloadTests(unittest.TestCase):
    def test_missing_identity_gets_generic_label_without_changing_battle(self):
        battle = {
            "Outcome": 1,
            "Seed": 123,
            "UserBoard": {"Mins": {"Items": []}, "Rel": {"Items": []}},
            "OpponentBoard": {"Mins": {"Items": []}, "Rel": {"Items": []}},
        }
        original = copy.deepcopy(battle)
        expected_result = {"ok": True, "image": b"test-image"}
        with patch.object(bridge, "_run_render", return_value=expected_result) as run:
            result = bridge.render_replay_image_from_raw_battles([battle])

        payload = run.call_args.args[0]
        rendered_battle = payload["battles"][0]
        self.assertEqual(result, expected_result)
        self.assertEqual(payload["mode"], "battle_json")
        self.assertEqual(rendered_battle, {**original, "Opponent": {"DisplayName": "Opponent"}})
        self.assertEqual(battle, original)

    def test_optional_identity_forms_and_existing_names(self):
        for identity, expected_name in [
            (None, "Opponent"),
            ({}, "Opponent"),
            ({"DisplayName": None}, "Opponent"),
            ({"DisplayName": ""}, "Opponent"),
            ({"DisplayName": "Existing name", "other": 7}, "Existing name"),
        ]:
            with self.subTest(identity=identity):
                battle = {"Opponent": identity}
                original = copy.deepcopy(battle)
                with patch.object(bridge, "_run_render") as run:
                    bridge.render_replay_image_from_raw_battles([battle])
                opponent = run.call_args.args[0]["battles"][0]["Opponent"]
                self.assertEqual(opponent["DisplayName"], expected_name)
                if isinstance(identity, dict):
                    for key, value in identity.items():
                        if key != "DisplayName":
                            self.assertEqual(opponent[key], value)
                    self.assertIsNot(opponent, identity)
                self.assertEqual(battle, original)


class InstalledRendererTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node")
        and bridge._ensure_paths() is None
        and (bridge.RENDER_RUNTIME / "canvas" / "package.json").is_file(),
        "Requires the separately installed replay renderer; does not install it",
    )
    def test_shipped_anonymous_battle_renders_png(self):
        from PIL import Image

        pool_path = bridge.ROOT / "data" / "opponents" / "arena_val_pool_deidentified.json.gz"
        with gzip.open(pool_path, "rt", encoding="utf-8") as stream:
            pool = json.load(stream)
        battle = pool["games"][0]["turns"][0]["battle"]
        original = copy.deepcopy(battle)
        self.assertNotIn("Opponent", battle)

        result = bridge.render_replay_image_from_raw_battles([battle])

        self.assertTrue(result["ok"], result.get("error"))
        with Image.open(io.BytesIO(result["image"])) as image:
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (1250, 125))
            image.load()
        self.assertEqual(battle, original)


if __name__ == "__main__":
    unittest.main()
