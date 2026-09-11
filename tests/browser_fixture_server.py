"""Read-only, loopback-only fixture server for isolated browser checks."""
from http.server import ThreadingHTTPServer
from pathlib import Path

from sap_ppo.tools.play_web.app import App
from sap_ppo.tools.play_web.http_app import Handler

ROOT = Path(__file__).resolve().parents[1]


class FixtureHandler(Handler):
    def do_POST(self):
        self._send_json(405, {"ok": False, "error": "Fixture server is read-only."})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    FixtureHandler.app = App(
        ROOT / "fixtures/parity_cases/sample_case.json",
        opponent_source="snapshot",
        snapshot_path=ROOT / "data/opponents/demo_snapshot.json.gz",
        game_mode="arena",
        predictor_path=None, value_model_path=None,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    print(server.server_port, flush=True)
    server.serve_forever()
