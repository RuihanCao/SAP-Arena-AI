"""App state and business logic for the SAP-Arena web simulator."""

from __future__ import annotations

import copy
import hashlib
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...api import legal_actions, skip_imagined_validation, step, validate_state
from ...ability.effects import ensure_stat_fields
from ...catalog import load_turtle_catalog
from ...constants import shop_item_cost
from ...end_turn import resolve_end_turn_with_sampled_battle
from ...engine import resolve_end_turn_post_battle
from ...oracles.sap_calc_battle_oracle import team_to_pet_configs
from ...tempo.features import parsed_pets_to_team
from ...train.opponents import ReplaySnapshotProvider
from .assets import (
    DEFAULT_SCENE_BACKGROUND,
    FONT_MAP,
    ICON_MAP,
    icon_or_placeholder,
    ROOT,
    SAP_CALC_ART,
    SAP_CALC_BACKGROUNDS,
    _pack_ability_text,
)
from .build_identity import build_identity
from .compose import apply_group, compose_actions

# Self-referential import of this package (sap_ppo.tools.play_web itself, not
# a submodule) so the eight names below are looked up dynamically through the
# package's own namespace rather than bound once at import time. Tests patch
# them as `sap_ppo.tools.play_web.<name>`; a plain `from ...x import name`
# here would bind a private copy in this module that a patch on the package
# object could never reach.
from .. import play_web as _play_web_pkg  # noqa: E402


SURFACE_HUMAN = "human"
SURFACE_AGENT = "agent"
SURFACES = (SURFACE_HUMAN, SURFACE_AGENT)
DEFAULT_SURFACE = SURFACE_HUMAN

DEFAULT_OPPONENT_SOURCE = "db"
DEFAULT_REPLAY_SNAPSHOT_PATH = ROOT / "data" / "opponents" / "replay_snapshot_v1.json"
DEFAULT_GAME_MODE = "versus"
DEFAULT_PREDICTOR_PATH = ROOT / "artifacts" / "tempo" / "next_board_predictor_v2.json"
DEFAULT_VALUE_MODEL_PATH = ROOT / "artifacts" / "tempo" / "one_turn_value_model_v1.json"


@dataclass
class SessionState:
    fixture_path: Path
    state: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    # One entry per `history` entry, in lockstep: the composed group this
    # transition belongs to, or None for a plain single action. Kept beside
    # the transition rather than inside it because `schemas/transition_v1.json`
    # is `additionalProperties: false`, so an extra key would make every
    # stored transition fail its own schema.
    history_group_ids: list[str | None] = field(default_factory=list)
    group_seq: int = 0
    last_battle: dict[str, Any] | None = None
    sampled_turn_battle: dict[str, Any] | None = None
    battle_rows: list[dict[str, Any]] = field(default_factory=list)
    predicted_next_board_row: dict[str, Any] | None = None
    predicted_next_board_end_row: dict[str, Any] | None = None
    image_version: int = 0


def catalog_debug_items(catalog: dict[str, Any], slot_type: str) -> list[dict[str, Any]]:
    """The catalog rows the skin's selection card reads (item id, tier, ability).

    Module level rather than a method because the duel page
    (`duel_app.py::DuelApp`) serves the same skin and needs the same rows, and
    two copies of "which ability sentence belongs to which item" is exactly
    the kind of drift that shows up as a wrong card in one page only.
    """
    if slot_type == "pet":
        by_tier = catalog.get("pets", {}).get("by_tier", {})
        id_to_name = catalog.get("pets", {}).get("id_to_name_id", {})
    else:
        by_tier = catalog.get("foods", {}).get("by_tier", {})
        id_to_name = catalog.get("foods", {}).get("id_to_name_id", {})
    ability_by_name = _pack_ability_text(slot_type)
    rows: list[dict[str, Any]] = []
    for tier_str, ids in by_tier.items():
        try:
            tier = int(tier_str)
        except (TypeError, ValueError):
            continue
        for item_id in ids:
            row: dict[str, Any] = {"item_id": str(item_id), "tier": tier}


            ability = ability_by_name.get(str(id_to_name.get(str(item_id), "")))
            if ability:
                row["ability"] = ability
            rows.append(row)
    rows.sort(key=lambda r: (int(r["tier"]), str(r["item_id"])))
    return rows


def _canonical_bytes(value: Any) -> bytes:
    """A value's content, encoded the one way that makes two equal values equal.

    Only ever hashed, never sent, so it is compact and key-ordered rather than
    readable -- the opposite of `http_app.py::_json_bytes`.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def catalog_document(catalog: dict[str, Any]) -> dict[str, Any]:
    """The half of the snapshot that cannot change while a server is running."""
    return {
        "catalog_base_stats": catalog.get("pets", {}).get("base_stats", {}),
        "debug_catalog": {
            "pets": catalog_debug_items(catalog, "pet"),
            "foods": catalog_debug_items(catalog, "food"),
        },
    }


def catalog_fingerprint(document: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(document)).hexdigest()[:16]


def catalog_url(document: dict[str, Any]) -> str:
    """Where a client fetches `document`, with its content in the URL.

    The hash is what makes the `immutable` cache header on that route true
    rather than merely convenient. A server restarted on a different pack
    serves a different `catalog_url`, so a stale copy is never asked for again
    instead of being trusted for a day at a URL that outlived its contents.
    """
    return f"/api/catalog?v={catalog_fingerprint(document)}"


def history_token(entries: list[dict[str, Any]], count: int | None = None) -> str:
    """Content address of the first `count` history entries (all, by default).

    Deliberately derived from the entries themselves rather than from a version
    counter the mutating call sites have to remember to bump. `history` is
    appended to by `_push_history`, shortened by `_pop_history` and `undo`,
    emptied by `reset` and `new_game`, and rewound a whole turn at a time by
    `duel.py`; a counter missed at any one of those sites leaves a client
    showing a log that silently disagrees with the server, while a hash missed
    anywhere just costs one full resend.
    """
    digest = hashlib.sha256()
    for entry in entries if count is None else entries[:count]:
        digest.update(_canonical_bytes(entry))
    return digest.hexdigest()[:16]


def client_history_entries(
    history: list[dict[str, Any]], group_ids: list[str | None]
) -> list[dict[str, Any]]:
    """The session log exactly as a snapshot hands it to the browser.

    ONE function rather than the two identical comprehensions it replaced (here
    and in `duel_app.py::snapshot`), because `history_precondition_error` below
    has to hash the same bytes the client hashed: `history_token` runs over
    THIS projection and not over the raw transitions, so a second copy that
    drifted by one field would turn every shop click into `stale_board`.
    """
    padded = list(group_ids)
    if len(padded) < len(history):
        padded = [None] * (len(history) - len(padded)) + padded
    return [
        {
            "action": tr["action"],
            "legal": tr["legal"],
            "deterministic": tr["deterministic"],
            "stochastic_reason": tr["stochastic_reason"],
            "stochastic_structural": tr.get("stochastic_structural"),
            "engine_notes": tr["engine_notes"],
            "group_id": group_id,
        }
        for tr, group_id in zip(history, padded)
    ]


STALE_BOARD_ERROR = "stale_board"
MALFORMED_PRECONDITION_ERROR = "stale_board:malformed_precondition"


def history_precondition_error(
    payload: Any, entries: list[dict[str, Any]]
) -> str | None:
    """Why this mutating request must NOT be applied, or None to go ahead.

    The page's own in-flight guard closes the common case. It cannot close a
    reload, a reconnect or a second tab, because those are two clients, so the
    server needs its own answer. It already has the state code to do it with:
    the log gains exactly one entry per applied action (`_push_history` is its
    only writer), and every snapshot carries `history_token`, the content
    address of the whole log. A mutating request may therefore name the board
    it was decided on, and a second click issued before the first reply landed
    necessarily names the pre-click board.

    Missing preconditions are allowed for API clients. Malformed or mismatched
    preconditions reject the operation without changing the board.
    """
    if not isinstance(payload, dict):
        return None
    if "expect_history_len" not in payload and "expect_history_token" not in payload:
        return None
    raw_len = payload.get("expect_history_len")
    raw_token = payload.get("expect_history_token")
    if isinstance(raw_len, bool) or not isinstance(raw_len, int) or raw_len < 0:
        return MALFORMED_PRECONDITION_ERROR
    if not isinstance(raw_token, str) or not raw_token:
        return MALFORMED_PRECONDITION_ERROR
    if raw_len != len(entries) or raw_token != history_token(entries):
        return STALE_BOARD_ERROR
    return None


def _normalize_game_mode(raw: str | None) -> str:
    mode = str(raw or DEFAULT_GAME_MODE).strip().lower()
    if mode not in {"arena", "versus"}:
        raise ValueError(f"invalid_game_mode:{mode}")
    return mode


class App:
    def __init__(
        self,
        fixture_path: Path,
        *,
        opponent_source: str = DEFAULT_OPPONENT_SOURCE,
        snapshot_path: Path = DEFAULT_REPLAY_SNAPSHOT_PATH,
        game_mode: str = DEFAULT_GAME_MODE,
        predictor_path: Path | None = DEFAULT_PREDICTOR_PATH,
        value_model_path: Path | None = DEFAULT_VALUE_MODEL_PATH,
        planner_depth: int = 6,
        planner_beam_width: int = 12,
        planner_samples_per_stochastic_action: int = 3,
        planner_oracle_rerank_top: int = 5,
        planner_oracle_simulation_count: int = 250,
        planner_predictor_top_k: int = 5,
        replay_cache: Path | None = None,
        surface: str = DEFAULT_SURFACE,
    ):

        try:
            from ...tempo.planner import TempoPlannerConfig
        except ImportError:

            TempoPlannerConfig = None
        # The viewer is optional. Its absence disables one page rather than
        # preventing the server from starting.
        try:
            from ..replay_player import DEFAULT_REPLAY_CACHE, ReplayPlayerSession
        except ImportError:
            self.replay_session = None
            DEFAULT_REPLAY_CACHE = None
            ReplayPlayerSession = None

        if surface not in SURFACES:
            raise ValueError(f"unknown surface: {surface!r} (expected one of {SURFACES})")
        self.surface = surface
        # Composed groups (`buy_pet_at`, `move_pet`) are server-side gestures
        # built out of several engine actions -- see compose.py. They are real
        # and they are exact, but they are not actions the AGENT has, so the
        # agent surface refuses them and the browser falls back to what
        # `legal_actions` enumerates.
        self.compose_enabled = surface != SURFACE_AGENT
        self.fixture_path = fixture_path


        self.replay_session = None if ReplayPlayerSession is None else ReplayPlayerSession(
            Path(replay_cache) if replay_cache is not None else DEFAULT_REPLAY_CACHE
        )
        self.game_mode = _normalize_game_mode(game_mode)
        self.predictor_path = Path(predictor_path) if predictor_path is not None else None
        self.value_model_path = Path(value_model_path) if value_model_path is not None else None

        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_initial = payload["initial_state"]
        for slot in raw_initial.get("team", []):
            ensure_stat_fields(slot)
        self.initial_state = self._normalize_state_for_mode(copy.deepcopy(raw_initial))
        self.session = SessionState(fixture_path=fixture_path, state=copy.deepcopy(self.initial_state))

        self.catalog = load_turtle_catalog()
        # Built once, here, because the pack is pinned for the life of the
        # process: every snapshot then costs a 40-byte pointer to it instead of
        # a fresh 18.8 KB copy of it.
        self.catalog_document = catalog_document(self.catalog)
        self.catalog_fingerprint = catalog_fingerprint(self.catalog_document)
        self.catalog_url = catalog_url(self.catalog_document)
        self.opponent_source = str(opponent_source or DEFAULT_OPPONENT_SOURCE).strip().lower()
        if self.opponent_source not in {"db", "snapshot"}:
            raise ValueError(f"invalid_opponent_source:{self.opponent_source}")
        self.snapshot_path = Path(snapshot_path)
        self._snapshot_provider: ReplaySnapshotProvider | None = None
        if self.opponent_source == "snapshot":
            if not self.snapshot_path.exists():
                raise FileNotFoundError(f"snapshot_not_found:{self.snapshot_path}")
            self._snapshot_provider = _play_web_pkg.ReplaySnapshotProvider(snapshot_path=self.snapshot_path)

        planner_fields = dict(
            depth=int(max(1, planner_depth)),
            beam_width=int(max(1, planner_beam_width)),
            samples_per_stochastic_action=int(max(1, planner_samples_per_stochastic_action)),
            oracle_rerank_top=int(max(1, planner_oracle_rerank_top)),
            oracle_simulation_count=int(max(1, planner_oracle_simulation_count)),
            predictor_top_k=int(max(1, planner_predictor_top_k)),
        )
        if TempoPlannerConfig is None:


            from types import SimpleNamespace

            self.tempo_planner_config = SimpleNamespace(**planner_fields)
        else:
            self.tempo_planner_config = TempoPlannerConfig(**planner_fields)
        self.tempo_predictor = self._load_predictor_artifact(self.predictor_path)
        self.tempo_value_model = self._load_value_model_artifact(self.value_model_path)

        self._end_turn_parse_cache: dict[str, dict[str, Any]] = {}
        self.session.state = self._reset_state()

    @staticmethod
    def _append_unique(items: list[str], code: str) -> None:
        value = str(code or "").strip()
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _snapshot_error_allows_live_db_fallback(error: str | None) -> bool:
        value = str(error or "").strip()
        if not value:
            return False
        return value.startswith("no_snapshot_opponent_for_turn:") or value.startswith("forced_pid_not_found_in_snapshot:")

    @staticmethod
    def _annotate_live_db_fallback(
        sampled: dict[str, Any],
        *,
        route: str,
        trigger_error: str | None,
    ) -> dict[str, Any]:
        out = copy.deepcopy(sampled) if isinstance(sampled, dict) else {"ok": False, "error": "live_db_sample_failed"}
        out["live_db_fallback"] = True
        out["live_db_fallback_from"] = "snapshot"
        out["live_db_fallback_route"] = str(route or "unknown")
        out["live_db_fallback_reason"] = "snapshot_exhausted"
        out["live_db_fallback_trigger_error"] = str(trigger_error or "").strip() or None
        return out

    @staticmethod
    def _empty_opponent_team() -> list[dict[str, Any]]:
        return parsed_pets_to_team([None, None, None, None, None])

    def _load_predictor_artifact(self, path: Path | None) -> dict[str, Any] | None:


        if path is None or not Path(path).exists():
            return None
        try:
            from ...tempo.predictor import load_predictor_artifact

            return load_predictor_artifact(Path(path))
        except Exception:
            return None

    def _load_value_model_artifact(self, path: Path | None) -> dict[str, Any] | None:
        # See `_load_predictor_artifact`: check the path before importing.
        if path is None or not Path(path).exists():
            return None
        try:
            from ...tempo.value_model import load_value_model

            return load_value_model(Path(path))
        except Exception:
            return None

    def _normalize_state_for_mode(self, state: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(state)
        meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
        out["meta"] = meta
        meta["game_mode"] = self.game_mode
        if self.game_mode == "versus":
            versus = meta.get("versus") if isinstance(meta.get("versus"), dict) else None
            if versus is None:
                raise ValueError("missing_required_field:meta.versus.opponent_lives")
            try:
                opponent_lives = int(versus.get("opponent_lives"))
            except Exception as exc:
                raise ValueError("missing_required_field:meta.versus.opponent_lives") from exc
            versus["opponent_lives"] = max(0, int(opponent_lives))
            meta["versus"] = versus
        return out

    def _roll_no_cost(self, state: dict[str, Any], *, clear_shop: bool, target_gold: int) -> tuple[dict[str, Any], list[str]]:
        """Apply shop reroll logic while keeping caller-specified gold unchanged."""
        work = copy.deepcopy(state)
        if clear_shop:
            work["shop"] = []
        work.setdefault("meta", {})
        # Web simulator should not be stuck on fixture seed values.
        work["meta"]["seed_known"] = False
        work["gold"] = int(target_gold) + 1

        tr = step(work, {"type": "ROLL"})
        if not tr["legal"]:
            raise RuntimeError(f"roll_refresh_failed:{tr.get('engine_notes')}")

        out = copy.deepcopy(tr["state_after"])
        out["gold"] = int(target_gold)
        return out, list(tr.get("engine_notes", []))

    def _reset_state(self) -> dict[str, Any]:
        """Reset to initial scalar/team state but with a freshly generated shop."""
        base = copy.deepcopy(self.initial_state)
        target_gold = int(base.get("gold", 10))
        refreshed, _notes = self._roll_no_cost(base, clear_shop=True, target_gold=target_gold)
        return self._normalize_state_for_mode(refreshed)

    def _local_replay_image_url(self, mode: str) -> str:
        return f"/api/replay_image?mode={urllib.parse.quote(str(mode), safe='')}&v={int(self.session.image_version)}"

    def _bump_image_version(self) -> None:
        self.session.image_version = int(self.session.image_version) + 1

    def _clear_predicted_next_board_rows(self) -> None:
        changed = bool(self.session.predicted_next_board_row is not None or self.session.predicted_next_board_end_row is not None)
        self.session.predicted_next_board_row = None
        self.session.predicted_next_board_end_row = None
        if changed:
            self._bump_image_version()

    def _set_predicted_next_board_row(self, row: dict[str, Any] | None) -> None:
        normalized = copy.deepcopy(row) if isinstance(row, dict) else None
        if self.session.predicted_next_board_row == normalized:
            return
        self.session.predicted_next_board_row = normalized
        self._bump_image_version()

    def _set_predicted_next_board_end_row(self, row: dict[str, Any] | None) -> None:
        normalized = copy.deepcopy(row) if isinstance(row, dict) else None
        if self.session.predicted_next_board_end_row == normalized:
            return
        self.session.predicted_next_board_end_row = normalized
        self._bump_image_version()

    @staticmethod
    def _to_replay_order_pets(team: Any) -> list[dict[str, Any] | None]:
        slots = list(team) if isinstance(team, list) else []
        normalized: list[dict[str, Any]] = []
        for idx in range(5):
            slot = slots[idx] if idx < len(slots) and isinstance(slots[idx], dict) else {}
            normalized.append(copy.deepcopy(slot))

        return list(reversed(team_to_pet_configs(normalized)))

    @staticmethod
    def _to_predictor_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_prob = -1.0
        for row in candidates:
            if not isinstance(row, dict):
                continue
            if not isinstance(row.get("team"), list):
                continue
            try:
                prob = float(row.get("prob", 0.0))
            except Exception:
                prob = 0.0
            if best is None or prob > best_prob:
                best = row
                best_prob = prob
        return copy.deepcopy(best) if isinstance(best, dict) else None

    def _simulate_state_after_chain_preview(self, state: dict[str, Any], chain_preview: list[dict[str, Any]]) -> dict[str, Any]:
        work = copy.deepcopy(state) if isinstance(state, dict) else {}
        for action in chain_preview:
            if not isinstance(action, dict):
                continue
            if str(action.get("type") or "").strip().upper() == "END_TURN":
                break
            try:
                tr = step(work, copy.deepcopy(action))
            except Exception:
                break
            if not bool(tr.get("legal", False)):
                break
            next_state = tr.get("state_after") if isinstance(tr.get("state_after"), dict) else None
            if not isinstance(next_state, dict):
                break
            work = copy.deepcopy(next_state)
        return work

    def _build_predicted_next_board(
        self,
        *,
        state: dict[str, Any],
        candidates: list[dict[str, Any]],
        context_source: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        top = self._to_predictor_candidate(candidates)
        if top is None:
            self._clear_predicted_next_board_rows()
            return None, None

        team = top.get("team")
        if not isinstance(team, list):
            self._clear_predicted_next_board_rows()
            return None, None

        try:
            prob = max(0.0, float(top.get("prob", 0.0)))
        except Exception:
            prob = 0.0

        try:
            candidate_turn = int(top.get("turn", state.get("turn", 1)))
        except Exception:
            candidate_turn = int(state.get("turn", 1))

        pid = str(top.get("participation_id") or "").strip() or None
        candidate_calc_state = top.get("opponent_calc_state") if isinstance(top.get("opponent_calc_state"), dict) else None

        preview_row = {
            "turn": int(state.get("turn", 1)),
            "outcome": "draw",
            "opponentName": "Predicted Next Board (Start)",
            "playerPets": self._to_replay_order_pets(state.get("team")),
            "opponentPets": self._to_replay_order_pets(team),
            "opponentToy": (
                str(candidate_calc_state.get("opponentToy")).strip()
                if isinstance(candidate_calc_state, dict)
                and isinstance(candidate_calc_state.get("opponentToy"), str)
                and str(candidate_calc_state.get("opponentToy")).strip()
                else None
            ),
            "opponentToyLevel": (
                self._coerce_int(candidate_calc_state.get("opponentToyLevel"), 0)
                if isinstance(candidate_calc_state, dict)
                else 0
            ),
            "sourceMode": "calc_rows_simulated",
        }
        self._set_predicted_next_board_row(preview_row)
        self._set_predicted_next_board_end_row(None)

        predicted_next_board = {
            "team": copy.deepcopy(team),
            "prob": float(prob),
            "source": str(top.get("source") or context_source or "unknown"),
            "rank": int(top["rank"]) if isinstance(top.get("rank"), int) else None,
            "similarity": float(top["similarity"])
            if isinstance(top.get("similarity"), (int, float))
            else None,
            "turn": int(candidate_turn),
            "participation_id": pid,
            "opponent_calc_state": copy.deepcopy(candidate_calc_state) if isinstance(candidate_calc_state, dict) else None,
        }
        return predicted_next_board, self._local_replay_image_url("predicted_next")

    def _build_predicted_next_board_end_image(
        self,
        *,
        state: dict[str, Any],
        predicted_next_board: dict[str, Any] | None,
        chain_preview: list[dict[str, Any]],
    ) -> str | None:
        team = predicted_next_board.get("team") if isinstance(predicted_next_board, dict) else None
        if not isinstance(team, list):
            self._set_predicted_next_board_end_row(None)
            return None

        simulated = self._simulate_state_after_chain_preview(state, chain_preview)
        candidate_calc_state = (
            predicted_next_board.get("opponent_calc_state")
            if isinstance(predicted_next_board.get("opponent_calc_state"), dict)
            else None
        )
        end_row = {
            "turn": int(state.get("turn", 1)),
            "outcome": "draw",
            "opponentName": "Predicted Next Board (End)",
            "playerPets": self._to_replay_order_pets(simulated.get("team")),
            "opponentPets": self._to_replay_order_pets(team),
            "opponentToy": (
                str(candidate_calc_state.get("opponentToy")).strip()
                if isinstance(candidate_calc_state, dict)
                and isinstance(candidate_calc_state.get("opponentToy"), str)
                and str(candidate_calc_state.get("opponentToy")).strip()
                else None
            ),
            "opponentToyLevel": (
                self._coerce_int(candidate_calc_state.get("opponentToyLevel"), 0)
                if isinstance(candidate_calc_state, dict)
                else 0
            ),
            "sourceMode": "calc_rows_simulated",
        }
        self._set_predicted_next_board_end_row(end_row)
        return self._local_replay_image_url("predicted_end")

    def _sampled_image_headers(self) -> tuple[str | None, str | None]:
        battle = self.session.sampled_turn_battle
        if not isinstance(battle, dict):
            return None, None
        user_name = ((battle.get("User") or {}).get("DisplayName")) if isinstance(battle.get("User"), dict) else None
        opp_name = (
            ((battle.get("Opponent") or {}).get("DisplayName"))
            if isinstance(battle.get("Opponent"), dict)
            else None
        )
        user = str(user_name).strip() if isinstance(user_name, str) and user_name.strip() else None
        opp = str(opp_name).strip() if isinstance(opp_name, str) and opp_name.strip() else None
        return user, opp

    def _sample_random_opponent(self, turn: int) -> dict[str, Any]:
        turn_i = int(turn)
        if self._snapshot_provider is not None:
            sampled = self._snapshot_provider.sample(turn_i, forced_pid=None)
            if bool(sampled.get("ok")):
                return sampled
            trigger_error = str(sampled.get("error") or "").strip()
            if self._snapshot_error_allows_live_db_fallback(trigger_error):
                live = _play_web_pkg.sample_random_opponent_team(turn_i)
                return self._annotate_live_db_fallback(live, route="random", trigger_error=trigger_error)
            return sampled
        return _play_web_pkg.sample_random_opponent_team(turn_i)

    def _sample_opponent_for_pid(self, pid: str, turn: int) -> dict[str, Any]:
        forced_pid = str(pid or "").strip()
        turn_i = int(turn)
        if self._snapshot_provider is not None:
            sampled = self._snapshot_provider.sample(turn_i, forced_pid=forced_pid)
            if bool(sampled.get("ok")):
                return sampled
            trigger_error = str(sampled.get("error") or "").strip()
            if self._snapshot_error_allows_live_db_fallback(trigger_error):
                live = _play_web_pkg.sample_opponent_team_for_pid(forced_pid, turn_i)
                return self._annotate_live_db_fallback(live, route="forced_pid", trigger_error=trigger_error)
            return sampled
        return _play_web_pkg.sample_opponent_team_for_pid(forced_pid, turn_i)

    def _apply_end_turn_with_battle(self) -> dict[str, Any]:
        before = copy.deepcopy(self.session.state)
        forced_pid: str | None = None

        def _fail_end_turn(
            error: str,
            *,
            sampled: dict[str, Any] | None = None,
            battle: dict[str, Any] | None = None,
            parse_mode: str | None = None,
            parse_error: str | None = None,
            timing_ms: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            sampled = sampled or {}
            battle = battle or {"ok": False, "error": error, "outcome": "unknown", "result": None, "calculator_link": None}
            sampled_replay_id = str(sampled.get("replay_id") or "").strip() or None
            sampled_created_at = str(sampled.get("created_at") or "").strip() or None
            sampled_participation_id = str(sampled.get("participation_id") or "").strip() or None
            live_db_fallback = bool(sampled.get("live_db_fallback"))
            self.session.last_battle = {
                "turn": int(before.get("turn", 1)),
                "result": "unknown",
                "oracle_ok": False,
                "oracle_error": error,
                "calculator_link": battle.get("calculator_link"),
                "opponent_source": sampled.get("source"),
                "opponent_replay_id": sampled_replay_id,
                "opponent_participation_id": sampled_participation_id,
                "opponent_created_at": sampled_created_at,
                "opponent_pack": sampled.get("side_pack"),
                "opponent_side": sampled.get("side"),
                "opponent_ok": bool(sampled.get("ok")),
                "opponent_error": sampled.get("error"),
                "sampled_turn_image_url": self._local_replay_image_url("sampled")
                if self.session.sampled_turn_battle
                else None,
                "session_replay_image_url": self._local_replay_image_url("session") if self.session.battle_rows else None,
                "forced_pid": forced_pid or None,
                "oracle_result": battle.get("result"),
                "parse_mode": parse_mode,
                "parse_error": parse_error,
                "timing_ms": timing_ms if isinstance(timing_ms, dict) else None,
                "live_db_fallback": live_db_fallback,
                "live_db_fallback_route": sampled.get("live_db_fallback_route") if live_db_fallback else None,
                "live_db_fallback_reason": sampled.get("live_db_fallback_reason") if live_db_fallback else None,
                "live_db_fallback_trigger_error": sampled.get("live_db_fallback_trigger_error") if live_db_fallback else None,
            }
            return {
                "ok": False,
                "error": error,
                "transition": None,
                "state": self._snapshot(),
            }

        resolved = resolve_end_turn_with_sampled_battle(
            before,
            game_mode=self.game_mode,
            pre_battle_fn=_play_web_pkg.resolve_end_turn_pre_battle,
            post_battle_fn=resolve_end_turn_post_battle,
            sample_random_fn=self._sample_random_opponent,
            sample_for_pid_fn=self._sample_opponent_for_pid,
            parse_replay_fn=_play_web_pkg.parse_replay_for_calculator_state,
            run_battle_fn=_play_web_pkg.run_battle_oracle_with_config,
            team_to_pet_configs_fn=team_to_pet_configs,
            parse_cache=self._end_turn_parse_cache,
        )
        forced_pid = str(resolved.get("forced_pid") or "").strip() or None
        if not resolved.get("ok"):
            sampled = resolved.get("sampled")
            battle = resolved.get("battle")
            return _fail_end_turn(
                str(resolved.get("error") or "end_turn_resolver_failed"),
                sampled=(sampled if isinstance(sampled, dict) else None),
                battle=(battle if isinstance(battle, dict) else None),
                parse_mode=(str(resolved.get("parse_mode")) if resolved.get("parse_mode") is not None else None),
                parse_error=(str(resolved.get("parse_error")) if resolved.get("parse_error") is not None else None),
                timing_ms=(resolved.get("timing_ms") if isinstance(resolved.get("timing_ms"), dict) else None),
            )

        tr = copy.deepcopy(resolved["transition"])
        after = copy.deepcopy(tr["state_after"])
        sampled = resolved.get("sampled") if isinstance(resolved.get("sampled"), dict) else {}
        battle = resolved.get("battle") if isinstance(resolved.get("battle"), dict) else {}
        parse_mode = str(resolved.get("parse_mode") or "")
        parse_error = str(resolved.get("parse_error") or "") or None
        replay_battle = resolved.get("replay_battle") if isinstance(resolved.get("replay_battle"), dict) else None
        parsed_state = resolved.get("parsed_state") if isinstance(resolved.get("parsed_state"), dict) else {}
        battle_state = resolved.get("battle_state") if isinstance(resolved.get("battle_state"), dict) else before

        self.session.state = self._normalize_state_for_mode(after)
        if self.game_mode == "versus":
            seen_team = parsed_pets_to_team(parsed_state.get("opponentPets"))
            if not self._team_has_any_pet(seen_team):
                fallback_team = self._parsed_opponent_team_from_sampled(sampled)
                if isinstance(fallback_team, list):
                    seen_team = copy.deepcopy(fallback_team)
            meta = self.session.state.setdefault("meta", {})
            versus_meta = meta.setdefault("versus", {})
            if self._has_opponent_calc_state(parsed_state):
                versus_meta["last_opponent_calc_state"] = copy.deepcopy(parsed_state)
            if self._team_has_any_pet(seen_team):
                versus_meta["last_opponent_team"] = copy.deepcopy(seen_team)
                versus_meta["last_opponent_turn"] = int(battle_state.get("turn", before.get("turn", 1)))
        tr["state_after"] = copy.deepcopy(self.session.state)
        self._push_history(tr)

        sampled_replay_id = str(sampled.get("replay_id") or "").strip() or None
        sampled_created_at = str(sampled.get("created_at") or "").strip() or None
        sampled_participation_id = str(sampled.get("participation_id") or "").strip() or None
        live_db_fallback = bool(sampled.get("live_db_fallback"))

        # Replay image row rendering (battle_json path via replay-bot getBattleInfo)
        # uses the opposite order from parseReplayForCalculator player/opponent pets.
        # Keep session rows in that same visual convention for consistency.
        player_pets_for_replay = list(reversed(team_to_pet_configs(battle_state.get("team", []))))
        opponent_pets_for_replay = list(reversed(list(parsed_state.get("opponentPets") or [])))
        session_row = {
            "turn": int(before.get("turn", 1)),
            "outcome": str(battle.get("outcome", "unknown")),
            "opponentName": f"Replay {sampled_replay_id or 'n/a'} / PID {sampled_participation_id or 'n/a'}",
            "playerPets": player_pets_for_replay,
            "opponentPets": copy.deepcopy(opponent_pets_for_replay),
        }
        if replay_battle is not None:
            self.session.sampled_turn_battle = copy.deepcopy(replay_battle)
        self.session.battle_rows.append(session_row)
        if len(self.session.battle_rows) > 30:
            self.session.battle_rows = self.session.battle_rows[-30:]
        self._bump_image_version()

        if live_db_fallback:
            tr_notes = tr.get("engine_notes")
            if isinstance(tr_notes, list):
                route = str(sampled.get("live_db_fallback_route") or "unknown")
                trigger = str(
                    sampled.get("live_db_fallback_trigger_error")
                    or sampled.get("live_db_fallback_reason")
                    or "snapshot_exhausted"
                )
                tr_notes.append(f"end_turn_snapshot_fallback_live_db:{route}:{trigger}")

        self.session.last_battle = {
            "turn": int(before.get("turn", 1)),
            "result": str(battle.get("outcome", "unknown")),
            "oracle_ok": bool(battle.get("ok")),
            "oracle_error": battle.get("error"),
            "calculator_link": battle.get("calculator_link"),
            "opponent_source": sampled.get("source"),
            "opponent_replay_id": sampled_replay_id,
            "opponent_participation_id": sampled_participation_id,
            "opponent_created_at": sampled_created_at,
            "opponent_pack": sampled.get("side_pack"),
            "opponent_side": sampled.get("side"),
            "opponent_ok": bool(sampled.get("ok")),
            "opponent_error": sampled.get("error"),
            "sampled_turn_image_url": self._local_replay_image_url("sampled"),
            "session_replay_image_url": self._local_replay_image_url("session"),
            "forced_pid": forced_pid or None,
            "oracle_result": battle.get("result"),
            "parse_mode": (parse_mode or None),
            "parse_error": parse_error,
            "timing_ms": (resolved.get("timing_ms") if isinstance(resolved.get("timing_ms"), dict) else None),
            "live_db_fallback": live_db_fallback,
            "live_db_fallback_route": sampled.get("live_db_fallback_route") if live_db_fallback else None,
            "live_db_fallback_reason": sampled.get("live_db_fallback_reason") if live_db_fallback else None,
            "live_db_fallback_trigger_error": sampled.get("live_db_fallback_trigger_error") if live_db_fallback else None,
        }

        return {
            "ok": True,
            "error": None,
            "transition": tr,
            "state": self._snapshot(),
        }

    def _label_action(self, action: dict[str, Any]) -> str:
        return json.dumps(action, separators=(",", ":"))

    def _catalog_debug_items(self, slot_type: str) -> list[dict[str, Any]]:
        return catalog_debug_items(self.catalog, slot_type)

    def replay_image_bytes(self, mode: str) -> bytes:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode == "sampled":
            if not isinstance(self.session.sampled_turn_battle, dict):
                raise ValueError("replay_image_unavailable:sampled")
            player_name, opponent_name = self._sampled_image_headers()
            rendered = _play_web_pkg.render_replay_image_from_raw_battles(
                [self.session.sampled_turn_battle],
                max_lives=6,
                player_name=player_name,
                header_opponent_name=opponent_name,
            )
            if not rendered.get("ok"):
                raise ValueError(str(rendered.get("error") or "replay_image_render_failed:sampled"))
            image = rendered.get("image")
            if not isinstance(image, (bytes, bytearray)):
                raise ValueError("replay_image_render_failed:sampled_output_empty")
            return bytes(image)
        if normalized_mode == "session":
            rows = [r for r in self.session.battle_rows if isinstance(r, dict)]
            if not rows:
                raise ValueError("replay_image_unavailable:session")
            rendered = _play_web_pkg.render_replay_image_from_calc_rows(
                rows,
                max_lives=6,
                player_name="Player",
                header_opponent_name="Sampled Opponent",
            )
            if not rendered.get("ok"):
                raise ValueError(str(rendered.get("error") or "replay_image_render_failed:session"))
            image = rendered.get("image")
            if not isinstance(image, (bytes, bytearray)):
                raise ValueError("replay_image_render_failed:session_output_empty")
            return bytes(image)
        if normalized_mode in {"predicted_next", "predicted_start"}:
            row = self.session.predicted_next_board_row if isinstance(self.session.predicted_next_board_row, dict) else None
            if row is None:
                raise ValueError("replay_image_unavailable:predicted_start")
            rendered = _play_web_pkg.render_replay_image_from_calc_rows(
                [row],
                max_lives=6,
                player_name="Player",
                header_opponent_name="Predicted Opponent (Start)",
            )
            if not rendered.get("ok"):
                raise ValueError(str(rendered.get("error") or "replay_image_render_failed:predicted_start"))
            image = rendered.get("image")
            if not isinstance(image, (bytes, bytearray)):
                raise ValueError("replay_image_render_failed:predicted_start_output_empty")
            return bytes(image)
        if normalized_mode == "predicted_end":
            row = self.session.predicted_next_board_end_row if isinstance(self.session.predicted_next_board_end_row, dict) else None
            if row is None:
                raise ValueError("replay_image_unavailable:predicted_end")
            rendered = _play_web_pkg.render_replay_image_from_calc_rows(
                [row],
                max_lives=6,
                player_name="Player",
                header_opponent_name="Predicted Opponent (End)",
            )
            if not rendered.get("ok"):
                raise ValueError(str(rendered.get("error") or "replay_image_render_failed:predicted_end"))
            image = rendered.get("image")
            if not isinstance(image, (bytes, bytearray)):
                raise ValueError("replay_image_render_failed:predicted_end_output_empty")
            return bytes(image)
        raise ValueError(f"unknown_replay_mode:{mode}")

    def _push_history(self, tr: dict[str, Any], group_id: str | None = None) -> None:
        """The ONLY way a transition enters the history.

        `history` and `history_group_ids` are two lists that must stay the
        same length; funnelling every append (and every pop, below) through
        one place is what keeps them from drifting.
        """
        self.session.history.append(copy.deepcopy(tr))
        self.session.history_group_ids.append(group_id)

    def _pop_history(self) -> tuple[dict[str, Any], str | None]:
        tr = self.session.history.pop()
        group_id = self.session.history_group_ids.pop() if self.session.history_group_ids else None
        return tr, group_id

    def _next_group_id(self) -> str:
        """Next group id."""
        self.session.group_seq = int(self.session.group_seq) + 1
        return f"grp-{self.session.group_seq}"

    def _snapshot(self) -> dict[str, Any]:
        validate_state(self.session.state)
        actions = legal_actions(self.session.state)
        return {
            "state": self.session.state,
            # Not the catalog itself: `/api/catalog`, addressed by content. The
            # page fetches that once and reuses it for the whole session.
            "catalog_url": self.catalog_url,
            "legal_actions": [
                {"index": i, "action": action, "label": self._label_action(action)} for i, action in enumerate(actions)
            ],
            "history": client_history_entries(
                self.session.history, self._history_group_ids()
            ),
            "last_battle": self.session.last_battle,
            "game_mode": self.game_mode,
            "opponent_source_mode": self.opponent_source,
            "snapshot_path": str(self.snapshot_path) if self.opponent_source == "snapshot" else None,
            "predictor_path": str(self.predictor_path) if self.predictor_path is not None else None,
            "value_model_path": str(self.value_model_path) if self.value_model_path is not None else None,
            "predictor_loaded": bool(self.tempo_predictor is not None),
            "value_model_loaded": bool(self.tempo_value_model is not None),
            # Everything the page (and a bare `curl`) needs to say WHAT it is
            # looking at, read out of the running process rather than out of
            # the branch someone believes is deployed.
            #
            # `imagined_validation_skipped` is live, not a startup constant:
            # `api.set_skip_imagined_validation` is a process-wide switch that
            # the duel's agent flips on when it is built, and nothing turns it
            # back off. It does NOT reach the mask in `legal_actions` above --
            # that call validates every time -- and
            # `test_play_web_agent_surface.py` pins both halves of that
            # sentence. It is reported because "you cannot tell from here" is
            # what made it a hazard.
            "surface": {
                "name": self.surface,
                "compose_enabled": bool(self.compose_enabled),
                "imagined_validation_skipped": bool(skip_imagined_validation()),
                "build": build_identity(),
            },
        }

    def reset(self) -> dict[str, Any]:
        self.session.state = self._reset_state()
        self.session.history = []
        self.session.history_group_ids = []
        self.session.last_battle = None
        self.session.sampled_turn_battle = None
        self.session.battle_rows = []
        self._clear_predicted_next_board_rows()
        self._bump_image_version()
        return self._snapshot()

    def _history_group_ids(self) -> list[str | None]:
        """`history_group_ids` padded to the history length."""
        ids = list(self.session.history_group_ids)
        if len(ids) < len(self.session.history):
            ids = [None] * (len(self.session.history) - len(ids)) + ids
        return ids[len(ids) - len(self.session.history):] if self.session.history else []

    def undo(self) -> tuple[bool, str | None, dict[str, Any]]:
        if not self.session.history:
            return False, "no_history_to_undo", self._snapshot()
        self.session.history_group_ids = self._history_group_ids()
        tr, group_id = self._pop_history()
        # A composed buy is two transitions; popping one would leave the pet
        # bought and sitting in the wrong slot. The group is contiguous and
        # was appended together, so unwinding to the head of the group is
        # "keep popping while the entry below carries the same id".
        if group_id is not None:
            while self.session.history and self.session.history_group_ids[-1] == group_id:
                tr, _gid = self._pop_history()
        self.session.state = self._normalize_state_for_mode(copy.deepcopy(tr["state_before"]))
        self.session.last_battle = None
        self.session.sampled_turn_battle = None
        self.session.battle_rows = []
        self._clear_predicted_next_board_rows()
        self._bump_image_version()
        return True, None, self._snapshot()

    def _debug_insert_shop_item(
        self,
        *,
        slot_type: str,
        item_id: str,
        cost: int,
        frozen: bool = False,
    ) -> tuple[bool, str | None]:
        if slot_type not in {"pet", "food"}:
            return False, f"invalid_slot_type:{slot_type}"
        shop = copy.deepcopy(self.session.state["shop"])
        insert_at = 0 if slot_type == "pet" else len(shop)
        shop.insert(
            insert_at,
            {
                "shop_index": -1,
                "slot_type": slot_type,
                "item_id": str(item_id),
                "cost": int(cost),
                "frozen": bool(frozen),
            },
        )

        if len(shop) > 9:
            removable = [i for i, slot in enumerate(shop) if not slot.get("frozen")]
            if removable:
                ordered = sorted(removable, reverse=(slot_type == "pet"))
                drop_idx = next((i for i in ordered if i != insert_at), None)
                if drop_idx is None:
                    # no real evictable slot except the inserted one => no-op
                    shop.pop(insert_at)
                else:
                    shop.pop(drop_idx)
            else:
                # all frozen => skip insertion
                shop.pop(insert_at)

        for idx, slot in enumerate(shop):
            slot["shop_index"] = idx
        self.session.state["shop"] = shop
        return True, None

    def add_debug_shop_item(
        self,
        slot_type: str,
        item_id: str,
        cost: int | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        item = str(item_id or "").strip()
        if not item:
            return False, "missing_item_id", self._snapshot()
        if slot_type == "pet":
            known = {str(x["item_id"]) for x in self._catalog_debug_items("pet")}
        elif slot_type == "food":
            known = {str(x["item_id"]) for x in self._catalog_debug_items("food")}
        else:
            return False, f"invalid_slot_type:{slot_type}", self._snapshot()
        if item not in known:
            return False, f"unknown_item_id:{item}", self._snapshot()

        # A missing cost means "whatever the game charges", so a scripted or
        # gallery caller gets an honest shop slot. The dev drawer passes 0
        # explicitly: composing a board by hand is the reason it exists, and
        # paying for it caps what you can build at the gold you happen to hold.
        #
        # Both are valid states. `validate_state` is schema-only here and
        # state_v1.json allows a cost of 0..3; the rule that a non-frozen slot
        # costs 3 belongs to `api._compare_roll_invariants`, which checks an
        # engine ROLL against the oracle and never sees a debug add.
        resolved_cost = shop_item_cost(slot_type, item) if cost is None else int(cost)
        ok, err = self._debug_insert_shop_item(slot_type=slot_type, item_id=item, cost=resolved_cost, frozen=False)
        if not ok:
            return False, err, self._snapshot()
        try:
            validate_state(self.session.state)
        except Exception as exc:
            return False, f"debug_add_item_failed:{type(exc).__name__}:{exc}", self._snapshot()
        return True, None, self._snapshot()

    def add_debug_chocolate(self) -> tuple[bool, str | None, dict[str, Any]]:
        ok, err = self._debug_insert_shop_item(
            slot_type="food",
            item_id="food-chocolate",
            # Free, like the drawer's other adds; see add_debug_shop_item.
            cost=0,
            frozen=False,
        )
        if not ok:
            return False, err, self._snapshot()
        try:
            validate_state(self.session.state)
        except Exception as exc:
            return False, f"debug_add_chocolate_failed:{type(exc).__name__}:{exc}", self._snapshot()
        return True, None, self._snapshot()

    def _resolve_action(self, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        if "action_index" in payload:
            idx = payload["action_index"]
            if not isinstance(idx, int):
                return None, "action_index_must_be_int"
            actions = legal_actions(self.session.state)
            if idx < 0 or idx >= len(actions):
                return None, f"action_index_out_of_range:{idx}"
            return actions[idx], None

        if "action" not in payload:
            return None, "missing_action_or_action_index"
        action = payload["action"]
        if not isinstance(action, dict):
            return None, "action_must_be_object"
        return action, None

    def apply_compose(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply a `{"compose": ...}` payload as one atomic group.

        Response shape matches `apply()` with two additions: `transitions`
        is the whole group in order, and `group_id` is the tag `undo` pops
        by. `transition` is the LAST op, so its `state_after` is the state
        the snapshot shows -- a client that needs the buy (for the shop
        lane) reads `transitions[0]`.
        """
        if not self.compose_enabled:
            # Refused on the SERVER, not merely hidden in the page: the
            # transparency surface must not execute a gesture the agent has
            # no action for, however the request reached it.
            return {
                "ok": False,
                "error": "compose_disabled_on_agent_surface",
                "compose": str(payload.get("compose") or "") or None,
                "transition": None,
                "transitions": [],
                "group_id": None,
                "state": self._snapshot(),
            }
        kind, actions, err = compose_actions(self.session.state, payload)
        if err is not None:
            return {
                "ok": False,
                "error": err,
                "compose": kind,
                "transition": None,
                "transitions": [],
                "group_id": None,
                "state": self._snapshot(),
            }

        ok, group_err, state_after, transitions = apply_group(
            self.session.state,
            actions,
            normalize_fn=self._normalize_state_for_mode,
        )
        if not ok:
            return {
                "ok": False,
                "error": group_err,
                "compose": kind,
                "transition": None,
                "transitions": [],
                "group_id": None,
                "state": self._snapshot(),
            }

        self._clear_predicted_next_board_rows()
        self.session.state = state_after
        group_id = self._next_group_id()
        for tr in transitions:
            self._push_history(tr, group_id)
        return {
            "ok": True,
            "error": None,
            "compose": kind,
            "transition": transitions[-1],
            "transitions": transitions,
            "group_id": group_id,
            "state": self._snapshot(),
        }

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Before the compose branch, so a composed gesture is guarded too.
        stale = history_precondition_error(
            payload, client_history_entries(
                self.session.history, self._history_group_ids())
        )
        if stale is not None:
            return {
                "ok": False,
                "error": stale,
                "transition": None,
                "state": self._snapshot(),
            }

        if "compose" in payload:
            return self.apply_compose(payload)

        action, err = self._resolve_action(payload)
        if err is not None:
            return {
                "ok": False,
                "error": err,
                "transition": None,
                "state": self._snapshot(),
            }

        if action.get("type") == "END_TURN":
            self._clear_predicted_next_board_rows()
            return self._apply_end_turn_with_battle()

        try:
            tr = step(self.session.state, action)
        except Exception as exc:  # schema validation and malformed action errors
            return {
                "ok": False,
                "error": f"action_rejected:{type(exc).__name__}:{exc}",
                "transition": None,
                "state": self._snapshot(),
            }

        if tr["legal"]:
            self._clear_predicted_next_board_rows()
            self.session.state = self._normalize_state_for_mode(copy.deepcopy(tr["state_after"]))
            tr["state_after"] = copy.deepcopy(self.session.state)
            self._push_history(tr)
            return {
                "ok": True,
                "error": None,
                "transition": tr,
                "state": self._snapshot(),
            }

        return {
            "ok": False,
            "error": tr["engine_notes"][0] if tr["engine_notes"] else "illegal_action",
            "transition": tr,
            "state": self._snapshot(),
        }

    @staticmethod
    def _team_has_any_pet(team: Any) -> bool:
        if not isinstance(team, list):
            return False
        for slot in team:
            if not isinstance(slot, dict):
                continue
            if slot.get("pet_id") or slot.get("pet_name") or slot.get("name"):
                return True
        return False

    @staticmethod
    def _has_opponent_calc_state(value: Any) -> bool:
        return isinstance(value, dict) and isinstance(value.get("opponentPets"), list)

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except Exception:
            return int(default)

    def _parsed_opponent_team_from_sampled(self, sampled: dict[str, Any]) -> list[dict[str, Any]] | None:
        parsed_state = sampled.get("parsed_state") if isinstance(sampled.get("parsed_state"), dict) else None
        if parsed_state is not None:
            team_from_parsed = parsed_pets_to_team(parsed_state.get("opponentPets"))
            if self._team_has_any_pet(team_from_parsed):
                return team_from_parsed

        replay_battle = sampled.get("battle") if isinstance(sampled.get("battle"), dict) else None
        if replay_battle is not None:
            build_model = sampled.get("build_model") if isinstance(sampled.get("build_model"), dict) else None
            parsed = _play_web_pkg.parse_replay_for_calculator_state(replay_battle, build_model)
            if bool(parsed.get("ok", False)) and isinstance(parsed.get("state"), dict):
                team_from_parse = parsed_pets_to_team(parsed["state"].get("opponentPets"))
                if self._team_has_any_pet(team_from_parse):
                    return team_from_parse

        sampled_team = sampled.get("team")
        if isinstance(sampled_team, list):
            return copy.deepcopy(sampled_team)
        return None

    def recommend(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:

        from ...tempo.planner import recommend_next_action, resolve_opponent_candidates

        encountered_errors: list[str] = []
        fallbacks_used: list[str] = []
        context_source = "no_context"
        predicted_next_board: dict[str, Any] | None = None
        predicted_next_board_image_url: str | None = None
        predicted_next_board_start_image_url: str | None = None
        predicted_next_board_end_image_url: str | None = None
        self._clear_predicted_next_board_rows()

        def _response(
            *,
            ok: bool,
            error: str | None,
            recommended_action: dict[str, Any] | None,
            chain_preview: list[dict[str, Any]],
            wdl_probs: dict[str, Any] | None,
            diagnostics: dict[str, Any],
            context_source_value: str,
        ) -> dict[str, Any]:
            return {
                "ok": bool(ok),
                "error": error,
                "recommended_action": copy.deepcopy(recommended_action) if isinstance(recommended_action, dict) else None,
                "chain_preview": copy.deepcopy(chain_preview),
                "wdl_probs": copy.deepcopy(wdl_probs) if isinstance(wdl_probs, dict) else None,
                "diagnostics": copy.deepcopy(diagnostics),
                "encountered_errors": list(encountered_errors),
                "fallbacks_used": list(fallbacks_used),
                "context_source": str(context_source_value),
                "predicted_next_board": copy.deepcopy(predicted_next_board)
                if isinstance(predicted_next_board, dict)
                else None,
                "predicted_next_board_image_url": str(predicted_next_board_image_url)
                if isinstance(predicted_next_board_image_url, str) and predicted_next_board_image_url.strip()
                else None,
                "predicted_next_board_start_image_url": str(predicted_next_board_start_image_url)
                if isinstance(predicted_next_board_start_image_url, str) and predicted_next_board_start_image_url.strip()
                else None,
                "predicted_next_board_end_image_url": str(predicted_next_board_end_image_url)
                if isinstance(predicted_next_board_end_image_url, str) and predicted_next_board_end_image_url.strip()
                else None,
                "updated_at_unix_ms": int(time.time() * 1000),
            }

        request = payload if isinstance(payload, dict) else {}
        state_payload = request.get("state")
        if state_payload is None:
            state = copy.deepcopy(self.session.state)
        elif isinstance(state_payload, dict):
            state = copy.deepcopy(state_payload)
        else:
            return _response(
                ok=False,
                error="state_must_be_object",
                recommended_action=None,
                chain_preview=[],
                wdl_probs=None,
                diagnostics={},
                context_source_value=context_source,
            )

        try:
            mode = _normalize_game_mode(request.get("game_mode", self.game_mode))
        except ValueError as exc:
            return _response(
                ok=False,
                error=str(exc),
                recommended_action=None,
                chain_preview=[],
                wdl_probs=None,
                diagnostics={},
                context_source_value=context_source,
            )

        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        state["meta"] = meta
        meta["game_mode"] = mode

        versus_meta: dict[str, Any] | None = None
        if mode == "versus":
            versus = meta.get("versus") if isinstance(meta.get("versus"), dict) else None
            if versus is None:
                return _response(
                    ok=False,
                    error="missing_required_field:meta.versus.opponent_lives",
                    recommended_action=None,
                    chain_preview=[],
                    wdl_probs=None,
                    diagnostics={},
                    context_source_value=context_source,
                )
            try:
                versus["opponent_lives"] = int(versus.get("opponent_lives"))
            except Exception:
                return _response(
                    ok=False,
                    error="missing_required_field:meta.versus.opponent_lives",
                    recommended_action=None,
                    chain_preview=[],
                    wdl_probs=None,
                    diagnostics={},
                    context_source_value=context_source,
                )
            meta["versus"] = versus
            versus_meta = versus

        opponent_context = request.get("opponent_context")
        ctx: dict[str, Any] = copy.deepcopy(opponent_context) if isinstance(opponent_context, dict) else {}
        predictor_loaded = bool(self.tempo_predictor is not None)
        value_model_loaded = bool(self.tempo_value_model is not None)
        if not value_model_loaded:
            self._append_unique(encountered_errors, "value_model_not_loaded")
            self._append_unique(fallbacks_used, "fallback_used_value_model_missing")

        turn = int(state.get("turn", 1))
        candidates: list[dict[str, Any]] = []
        explicit_board_payload = isinstance(ctx.get("team"), list) or isinstance(ctx.get("teams"), list)

        session_meta = self.session.state.get("meta") if isinstance(self.session.state.get("meta"), dict) else {}
        session_versus = session_meta.get("versus") if isinstance(session_meta.get("versus"), dict) else {}
        history_calc_state = (
            copy.deepcopy(versus_meta.get("last_opponent_calc_state"))
            if isinstance(versus_meta, dict) and self._has_opponent_calc_state(versus_meta.get("last_opponent_calc_state"))
            else (
                copy.deepcopy(session_versus.get("last_opponent_calc_state"))
                if self._has_opponent_calc_state(session_versus.get("last_opponent_calc_state"))
                else None
            )
        )
        if (
            explicit_board_payload
            and not self._has_opponent_calc_state(ctx.get("opponent_calc_state"))
            and self._has_opponent_calc_state(history_calc_state)
        ):
            ctx["opponent_calc_state"] = copy.deepcopy(history_calc_state)

        if explicit_board_payload:
            context_source = "explicit"
            candidates = resolve_opponent_candidates(
                state=state,
                opponent_context=ctx,
                predictor_artifact=self.tempo_predictor,
                default_team=None,
                predictor_top_k=int(self.tempo_planner_config.predictor_top_k),
            )
            if not candidates:
                self._append_unique(encountered_errors, "explicit_context_invalid")

        history_team = (
            copy.deepcopy(versus_meta.get("last_opponent_team"))
            if isinstance(versus_meta, dict) and isinstance(versus_meta.get("last_opponent_team"), list)
            else (
                copy.deepcopy(session_versus.get("last_opponent_team"))
                if isinstance(session_versus.get("last_opponent_team"), list)
                else None
            )
        )
        if not isinstance(history_team, list) and self._has_opponent_calc_state(history_calc_state):
            parsed_history_team = parsed_pets_to_team(history_calc_state.get("opponentPets"))
            if self._team_has_any_pet(parsed_history_team):
                history_team = copy.deepcopy(parsed_history_team)

        if mode == "versus" and not explicit_board_payload and not self._has_opponent_calc_state(history_calc_state):
            context_source = "history_missing_calc_state"
            self._append_unique(encountered_errors, "planner_missing_opponent_calc_state")
            planner_diag = {
                "context_source": context_source,
                "predictor_loaded": predictor_loaded,
                "value_model_loaded": value_model_loaded,
                "candidate_count_input": 0,
                "strict_opponent_calc_state": True,
                "encountered_errors": list(encountered_errors),
                "fallbacks_used": list(fallbacks_used),
            }
            return _response(
                ok=False,
                error="planner_missing_opponent_calc_state",
                recommended_action=None,
                chain_preview=[],
                wdl_probs=None,
                diagnostics=planner_diag,
                context_source_value=context_source,
            )

        if not candidates and isinstance(history_team, list):
            if predictor_loaded:
                history_ctx = {
                    "last_opponent_team": copy.deepcopy(history_team),
                    "turn": int(turn),
                    "opponent_calc_state": copy.deepcopy(history_calc_state),
                }
                history_preds = resolve_opponent_candidates(
                    state=state,
                    opponent_context=history_ctx,
                    predictor_artifact=self.tempo_predictor,
                    default_team=None,
                    predictor_top_k=int(self.tempo_planner_config.predictor_top_k),
                )
                if history_preds:
                    candidates = history_preds
                    context_source = "predictor_history"
                else:
                    self._append_unique(encountered_errors, "predictor_no_candidates")
                    self._append_unique(fallbacks_used, "fallback_used_predictor_empty")
            else:
                self._append_unique(encountered_errors, "predictor_not_loaded")
                self._append_unique(fallbacks_used, "fallback_used_predictor_missing")

            if not candidates:
                context_source = "history"
                candidate_row: dict[str, Any] = {
                    "team": copy.deepcopy(history_team),
                    "prob": 1.0,
                    "source": "history_fallback",
                }
                if self._has_opponent_calc_state(history_calc_state):
                    candidate_row["opponent_calc_state"] = copy.deepcopy(history_calc_state)
                candidates = [candidate_row]

        if not candidates and not isinstance(history_team, list):
            self._append_unique(encountered_errors, "history_context_missing")
            self._append_unique(fallbacks_used, "fallback_used_no_history_context")

            if predictor_loaded:
                cold_start_ctx = {"last_opponent_team": self._empty_opponent_team(), "turn": int(turn)}
                cold_preds = resolve_opponent_candidates(
                    state=state,
                    opponent_context=cold_start_ctx,
                    predictor_artifact=self.tempo_predictor,
                    default_team=None,
                    predictor_top_k=int(self.tempo_planner_config.predictor_top_k),
                )
                if cold_preds:
                    candidates = cold_preds
                    context_source = "predictor_cold_start"
                else:
                    self._append_unique(encountered_errors, "predictor_no_candidates")
                    self._append_unique(fallbacks_used, "fallback_used_predictor_empty")
            else:
                self._append_unique(encountered_errors, "predictor_not_loaded")
                self._append_unique(fallbacks_used, "fallback_used_predictor_missing")

        if not candidates:
            context_source = "no_context"
            self._append_unique(fallbacks_used, "fallback_used_no_context_planner")

        predicted_next_board, predicted_next_board_start_image_url = self._build_predicted_next_board(
            state=state,
            candidates=candidates,
            context_source=context_source,
        )
        predicted_next_board_image_url = predicted_next_board_start_image_url
        predicted_next_board_end_image_url = None

        out = recommend_next_action(
            state=state,
            opponent_candidates=candidates,
            value_model=self.tempo_value_model,
            config=self.tempo_planner_config,
        )
        if not bool(out.get("ok", False)):
            planner_diag = copy.deepcopy(out.get("diagnostics")) if isinstance(out.get("diagnostics"), dict) else {}
            for code in planner_diag.get("encountered_errors", []) if isinstance(planner_diag.get("encountered_errors"), list) else []:
                self._append_unique(encountered_errors, str(code))
            for code in planner_diag.get("fallbacks_used", []) if isinstance(planner_diag.get("fallbacks_used"), list) else []:
                self._append_unique(fallbacks_used, str(code))
            planner_diag["context_source"] = context_source
            planner_diag["predictor_loaded"] = predictor_loaded
            planner_diag["value_model_loaded"] = value_model_loaded
            planner_diag["candidate_count_input"] = int(len(candidates))
            planner_diag["encountered_errors"] = list(encountered_errors)
            planner_diag["fallbacks_used"] = list(fallbacks_used)
            return _response(
                ok=False,
                error=str(out.get("error") or "planner_failed"),
                recommended_action=None,
                chain_preview=[],
                wdl_probs=None,
                diagnostics=planner_diag,
                context_source_value=context_source,
            )

        diagnostics = copy.deepcopy(out.get("diagnostics")) if isinstance(out.get("diagnostics"), dict) else {}
        for code in diagnostics.get("encountered_errors", []) if isinstance(diagnostics.get("encountered_errors"), list) else []:
            self._append_unique(encountered_errors, str(code))
        for code in diagnostics.get("fallbacks_used", []) if isinstance(diagnostics.get("fallbacks_used"), list) else []:
            self._append_unique(fallbacks_used, str(code))
        diagnostics["game_mode"] = mode
        diagnostics["context_source"] = context_source
        diagnostics["predictor_loaded"] = predictor_loaded
        diagnostics["value_model_loaded"] = value_model_loaded
        diagnostics["candidate_count_input"] = int(len(candidates))

        top_prob = 0.0
        for row in candidates:
            try:
                p = float(row.get("prob", 0.0))
            except Exception:
                p = 0.0
            top_prob = max(top_prob, p)
        diagnostics["candidate_top_prob"] = float(top_prob)
        diagnostics["encountered_errors"] = list(encountered_errors)
        diagnostics["fallbacks_used"] = list(fallbacks_used)

        chain_preview_payload = [copy.deepcopy(a) for a in (out.get("chain_preview") or []) if isinstance(a, dict)]
        predicted_next_board_end_image_url = self._build_predicted_next_board_end_image(
            state=state,
            predicted_next_board=predicted_next_board,
            chain_preview=chain_preview_payload,
        )

        return _response(
            ok=True,
            error=None,
            recommended_action=(copy.deepcopy(out.get("recommended_action")) if isinstance(out.get("recommended_action"), dict) else None),
            chain_preview=chain_preview_payload,
            wdl_probs=(copy.deepcopy(out.get("wdl_probs")) if isinstance(out.get("wdl_probs"), dict) else None),
            diagnostics=diagnostics,
            context_source_value=context_source,
        )

    def _name_id(self, slot_type: str, item_id: str) -> str | None:
        if slot_type == "pet":
            return self.catalog.get("pets", {}).get("id_to_name_id", {}).get(item_id)
        if slot_type == "food":
            return self.catalog.get("foods", {}).get("id_to_name_id", {}).get(item_id)
        return None

    def image_path(self, slot_type: str, item_id: str) -> Path | None:
        name_id = self._name_id(slot_type, item_id)
        if not name_id:
            return None

        if slot_type == "pet":
            path = SAP_CALC_ART / "Pets" / f"{name_id}.png"
        else:
            path = SAP_CALC_ART / "Food" / f"{name_id}.png"

        if path.exists() and path.is_file():
            return path
        return None

    def icon_path(self, name: str) -> Path | None:
        return icon_or_placeholder(name)

    def font_path(self, name: str) -> Path | None:
        path = FONT_MAP.get(name)
        if path is None:
            return None
        if path.exists() and path.is_file():
            return path
        return None

    def background_path(self, name: str) -> Path | None:
        """Resolve a shop-scene background inside the pinned art pack.

        Read-only and name-restricted: only bare alphanumeric asset names resolve,
        so the query string cannot walk out of the Background directory.
        """
        clean = (name or DEFAULT_SCENE_BACKGROUND).strip() or DEFAULT_SCENE_BACKGROUND
        if not clean.isalnum():
            return None
        path = SAP_CALC_BACKGROUNDS / f"{clean}.png"
        if path.exists() and path.is_file():
            return path
        return None
