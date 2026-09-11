"""Minimal Step 3 training environment scaffolding."""

from __future__ import annotations

import copy
import json
import random
import warnings
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

from ..action_keys import NONE_TYPE as _NONE_TYPE
from ..action_keys import action_key_seq as _action_key_seq
from ..action_keys import action_key_value as _action_key_value
from ..action_keys import hashable_action_key as _action_index_key
from ..api import legal_actions, step, validate_state
from ..end_turn import resolve_end_turn_with_sampled_battle
from ..engine import resolve_end_turn_post_battle, resolve_end_turn_pre_battle
from ..tempo.features import parsed_pets_to_team
from ..visited_guard import set_training_rolls_this_turn as _set_training_rolls_this_turn
from ..visited_guard import state_signature as _state_signature
from .opening_source import FixedOpeningSource, VariedOpeningSource
from .opening_source import training_opening_index as _training_opening_index
from .opponents import OpponentProvider, ReplayDBOpponentProvider


def _action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _build_action_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []

    for shop_index in range(9):
        for team_index in range(5):
            catalog.append({"type": "BUY_PET", "shop_index": shop_index, "team_index": team_index})
            catalog.append({"type": "BUY_COMBINE", "shop_index": shop_index, "team_index": team_index})
            catalog.append({"type": "BUY_FOOD", "shop_index": shop_index, "team_index": team_index})
        catalog.append({"type": "BUY_FOOD", "shop_index": shop_index, "team_index": None})

    for team_index in range(5):
        catalog.append({"type": "SELL", "team_index": team_index})

    for src in range(5):
        for dst in range(5):
            if src == dst:
                continue
            catalog.append({"type": "COMBINE", "src_team_index": src, "dst_team_index": dst})

    for order in permutations(range(5)):
        catalog.append({"type": "REORDER", "order": list(order)})

    for shop_index in range(9):
        catalog.append({"type": "FREEZE", "shop_index": shop_index})
        catalog.append({"type": "UNFREEZE", "shop_index": shop_index})

    catalog.append({"type": "ROLL"})
    catalog.append({"type": "END_TURN"})
    return catalog


ACTION_CATALOG: list[dict[str, Any]] = _build_action_catalog()
ACTION_INDEX_BY_KEY: dict[str, int] = {_action_key(action): idx for idx, action in enumerate(ACTION_CATALOG)}


ACTION_INDEX_BY_TUPLE_KEY: dict[tuple[Any, ...], int] = {
    _action_index_key(action): idx for idx, action in enumerate(ACTION_CATALOG)
}
if len(ACTION_INDEX_BY_TUPLE_KEY) != len(ACTION_CATALOG):
    raise RuntimeError(
        f"action_index_key_collision:{len(ACTION_INDEX_BY_TUPLE_KEY)}!={len(ACTION_CATALOG)}"
    )


_VISITED_GUARD_PROGRESS_CAP = 25


_unmapped_action_types_warned: set[str] = set()


def _warn_unmapped_action_once(action: dict[str, Any]) -> None:
    """Emit ONE `RuntimeWarning` per distinct engine-legal action `type` that
    has no `ACTION_CATALOG` index, then stay silent for that type for the
    rest of the process. Never raises (a measurement/diagnostics aid must
    not be able to break a rollout) -- mirrors `bc_recommender.
    _maybe_log_decode`'s "never raises" contract for its own best-effort
    sink, just via `warnings.warn` instead of a file.
    """
    type_name = str(action.get("type", "?"))
    if type_name in _unmapped_action_types_warned:
        return
    _unmapped_action_types_warned.add(type_name)
    try:
        warnings.warn(
            f"TrainingEnv.legal_actions(): engine-legal action type {type_name!r} "
            f"has no ACTION_CATALOG index (example={action!r}); silently excluded "
            "from legal_actions()/legal_action_mask() so a rare catalog gap cannot "
            "crash a training run (contrast tools.bc_recommender.legal_mask, which "
            "raises at eval time). Logged once per "
            "action type per process.",
            RuntimeWarning,
            stacklevel=2,
        )
    except Exception:
        pass


def build_action_type_ids(catalog: list[dict[str, Any]] | None = None) -> tuple[list[int], list[str]]:
    """Return (type_id_by_action_index, type_names_by_id) with stable first-seen ordering."""
    source = ACTION_CATALOG if catalog is None else catalog
    type_names: list[str] = []
    type_name_to_id: dict[str, int] = {}
    type_ids: list[int] = []

    for action in source:
        type_name = str(action.get("type", "")).strip().upper() or "UNKNOWN"
        if type_name not in type_name_to_id:
            type_name_to_id[type_name] = len(type_names)
            type_names.append(type_name)
        type_ids.append(int(type_name_to_id[type_name]))

    return type_ids, type_names


def load_initial_state_from_fixture(fixture_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    return copy.deepcopy(payload["initial_state"])


def _ensure_meta_dict(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("meta")
    if isinstance(meta, dict):
        return meta
    meta = {}
    state["meta"] = meta
    return meta


def _set_last_opponent_team(state: dict[str, Any], team: Any) -> None:
    meta = _ensure_meta_dict(state)
    versus = meta.get("versus")
    if not isinstance(versus, dict):
        versus = {}
        meta["versus"] = versus
    versus["last_opponent_team"] = [copy.deepcopy(slot) for slot in parsed_pets_to_team(team)]


class TrainingEnv:
    """Gym-style core loop without external framework dependency."""

    def __init__(
        self,
        initial_state: dict[str, Any],
        *,
        opponent_provider: OpponentProvider | None = None,
        end_turn_resolver: Callable[..., dict[str, Any]] = resolve_end_turn_with_sampled_battle,
        max_turn: int | None = None,
        one_turn_mode: bool = False,
        opening_source: VariedOpeningSource | FixedOpeningSource | None = None,
        opening_env_seed: int | None = None,
    ) -> None:
        validate_state(initial_state)
        self._initial_state = copy.deepcopy(initial_state)
        self.state = copy.deepcopy(initial_state)
        self.history: list[dict[str, Any]] = []
        self.opponent_provider = opponent_provider or ReplayDBOpponentProvider()
        self.end_turn_resolver = end_turn_resolver
        self._end_turn_parse_cache: dict[str, dict[str, Any]] = {}
        self.max_turn = int(max_turn) if max_turn is not None else None
        self.one_turn_mode = bool(one_turn_mode)
        # Full-game frame fix (see train/opening_source.py): when set, `reset()`
        # draws each episode's turn-1 state from THIS instead of always
        # re-copying `self._initial_state` -- gated to versus, non-one-turn
        # episodes only (see `reset()`). Default None is a complete no-op:
        # every existing caller that doesn't pass this is byte-for-byte
        # unaffected, including arena/one-turn callers that pass it anyway.
        self._opening_source = opening_source
        # Full-game frame fix, parallel-TRAINING decorrelation (review
        # finding 1): default None means "use the bare episode_index
        # directly in reset() below" -- byte-for-byte the ORIGINAL behavior
        # every existing caller keeps getting, since none of them pass
        # this (eval_versus_fullgame.py's play_one_game calls `opening_
        # source.state_for_game(game_index)` directly, never through this
        # class; eval_ppo.py's single, non-parallel core_env; this env's
        # own arena/one-turn callers). Only train_ppo.py's parallel-rollout
        # `_make_env(rank)` factory passes a real value (its per-rank
        # `env_seed = args.seed + rank*9973`, the SAME value already
        # namespacing that rank's opponent provider), so ONLY those
        # training envs opt into the decorrelated `training_opening_
        # index()` hashing in reset().
        self._opening_env_seed = opening_env_seed


        self.visited: set[str] = {_state_signature(self.state)}
        self._progress_action_count = 0


        self._episode_index = 0


        self._seed_base = 0
        _set_training_rolls_this_turn(self._initial_state, 0)
        _set_training_rolls_this_turn(self.state, 0)

    def reset(self, seed: int | None = None, *, episode_index: int | None = None) -> dict[str, Any]:
        """Reset to the fixture's initial state.

        Full-game frame fix (train/opening_source.py): `ep_index` is now
        resolved BEFORE the state is built (previously right after) because
        the turn-1 state itself can depend on it. When `self._opening_source`
        is set AND this is a versus, non-one-turn episode -- EXACTLY the
        guard `initial_pid_fn` below already uses -- the episode starts from
        `self._opening_source.state_for_game(ep_index)` (a fresh, seeded
        engine roll -- see opening_source.py's module docstring) instead of
        always re-copying the single `self._initial_state` fixture. Arena /
        one-turn-mode / no-opening-source callers take the `else` branch,
        which is the OLD line, unchanged.

        Review finding 1 (parallel-training decorrelation): when
        `self._opening_env_seed` is also set (see `__init__`), the index fed
        to `state_for_game` is `training_opening_index(self._opening_env_seed,
        ep_index)` instead of the bare `ep_index` -- otherwise every
        parallel `--num-envs` worker, each counting its own episodes from 0,
        would draw the IDENTICAL opening sequence (the bug this closes).
        `opening_env_seed=None` (every caller except train_ppo.py's rollout
        workers) keeps the bare-`ep_index` path byte-for-byte, including
        every EVAL call site (`eval_versus_fullgame.py`, `eval_ppo.py`) --
        eval's own reproducibility depends on `game_index` 0, 1, 2, ...
        mapping to a FIXED, repeatable sequence, which only the bare-index
        path guarantees."""
        ep_index = int(episode_index) if episode_index is not None else int(self._episode_index)

        base_game_mode = str(self._initial_state.get("meta", {}).get("game_mode") or "arena").strip().lower()
        if self._opening_source is not None and base_game_mode == "versus" and not self.one_turn_mode:
            if self._opening_env_seed is not None:
                opening_index = _training_opening_index(self._opening_env_seed, ep_index)
            else:
                opening_index = ep_index
            self.state = self._opening_source.state_for_game(opening_index)
        else:
            self.state = copy.deepcopy(self._initial_state)
        self.history = []
        self._end_turn_parse_cache = {}
        self.visited = {_state_signature(self.state)}
        self._progress_action_count = 0
        _set_training_rolls_this_turn(self.state, 0)
        if seed is not None:
            self._seed_base = int(seed)
            random.seed(int(seed))


            for rng_attr in ("_rng", "_random_rng"):
                rng = getattr(self.opponent_provider, rng_attr, None)
                if hasattr(rng, "seed"):
                    rng.seed(int(seed))

        meta = _ensure_meta_dict(self.state)
        meta["seed_known"] = True
        meta["seed"] = int(self._seed_base) + int(ep_index)

        game_mode = str(meta.get("game_mode") or "arena").strip().lower()
        initial_pid_fn = getattr(self.opponent_provider, "initial_pid_for_game", None)
        if game_mode == "versus" and not self.one_turn_mode and callable(initial_pid_fn):
            followed_pid = initial_pid_fn(ep_index)
            if followed_pid:
                versus = meta.setdefault("versus", {})
                versus["current_opponent_participation_id"] = str(followed_pid)
        self._episode_index = ep_index + 1

        return copy.deepcopy(self.state)

    def legal_actions(self, *, apply_visited_guard: bool = True) -> list[dict[str, Any]]:
        """`apply_visited_guard` (default True -- this is what every real
        caller wants, including `SapPpoGymEnv.action_masks()`, so Part B is
        "on" unless a caller deliberately opts out): set False to see PURE
        Part-A legality (engine-true only, no within-turn guard/cap at
        all). This exists because Part B's guard has no equivalent STATIC
        mask function on the eval side to diff against --
        `bc_recommender.legal_mask()` is (and must stay) a pure, stateless,
        engine-true-only function; the eval decoder's OWN guard lives
        inside `BcRecommender.recommend()`'s decode LOOP (skip-and-try-
        next-candidate over a probability ranking), never as a second mask
        function. `tools/frame_parity_harness.py`'s `legality` dimension
        uses `apply_visited_guard=False` to isolate exactly what it always
        tested (engine-true legality vs the OLD RL anti-spam heuristics,
        Part A); Part B's guard/cap RULE is verified by its own focused
        test (`python/tests/test_visited_guard.py`), not that harness.

        Part B (NOT a game rule -- this env's own within-turn bookkeeping,
        mirroring `bc_recommender.py`'s eval-time decode guard exactly: same
        `_state_signature`, same cap):
          1. any non-END_TURN action whose forward-simulated result would
             revisit an already-visited within-turn state (`self.visited`)
             is excluded. If EVERY non-END_TURN action would cycle, nothing
             is excluded from END_TURN itself (it is never subject to this
             check), so it becomes the only remaining legal action --
             "forcing" END_TURN falls out of this rule for free, no
             special-case code needed.
          2. once `_VISITED_GUARD_PROGRESS_CAP` non-END_TURN actions have
             actually been applied this turn (`self._progress_action_count`,
             see `_record_progress_action`), every non-END_TURN action is
             excluded, again leaving END_TURN as the only legal action.
        `legal_action_mask()` / `legal_action_indices()` derive from this
        method (unchanged), so they automatically inherit both parts."""
        allowed: list[dict[str, Any]] = []
        cap_reached = apply_visited_guard and self._progress_action_count >= _VISITED_GUARD_PROGRESS_CAP
        for action in legal_actions(self.state):
            if _action_key(action) not in ACTION_INDEX_BY_KEY:
                _warn_unmapped_action_once(action)
                continue
            action_type = str(action.get("type", "")).strip().upper()
            if action_type == "END_TURN":
                allowed.append(action)
                continue
            if cap_reached:
                continue
            if apply_visited_guard:
                next_sig = self._forward_signature(action)
                if next_sig is not None and next_sig in self.visited:
                    continue
            allowed.append(action)
        return allowed

    def _forward_signature(self, action: dict[str, Any]) -> str | None:
        """`_state_signature` after forward-simulating `action` from
        `self.state` on a deep copy (never mutates `self.state`).

        Returns None if the engine itself refuses to apply the action --
        defense-in-depth only, since `action` always comes from THIS
        state's own engine-true `legal_actions()` and so should already be
        applicable; mirrors `bc_recommender._choose_next_action`'s
        identical `except Exception: continue` / `if not legal: continue`
        guards against the same theoretical case.
        """
        try:
            trans = step(copy.deepcopy(self.state), action)
        except Exception:
            return None
        if not trans.get("legal"):
            return None
        next_state = trans.get("state_after")
        if not isinstance(next_state, dict):
            return None
        return _state_signature(next_state)

    def _record_progress_action(self) -> None:
        """After a successful non-END_TURN `step()`: extend `self.visited`
        with the new current-state signature and bump the progress counter
        (Part B) -- mirrors `bc_recommender.recommend()`'s own
        `visited.add(_state_signature(work))` after each accepted
        "progress" step of its decode walk.
        """
        self.visited.add(_state_signature(self.state))
        self._progress_action_count += 1

    def _reset_turn_guard(self) -> None:
        """After END_TURN (either resolver path, or `advance_turn_no_battle`):
        the anti-cycle guard is scoped to ONE turn, so a new turn starts a
        fresh `visited` set (seeded with the new turn-start signature, same
        as `__init__`/`reset()`) and a fresh progress-action counter.
        """
        self.visited = {_state_signature(self.state)}
        self._progress_action_count = 0

    @property
    def action_space_size(self) -> int:
        return len(ACTION_CATALOG)

    def action_catalog(self) -> list[dict[str, Any]]:
        return copy.deepcopy(ACTION_CATALOG)

    def decode_action(self, action_index: int) -> dict[str, Any]:
        idx = int(action_index)
        if idx < 0 or idx >= len(ACTION_CATALOG):
            raise IndexError(f"action_index_out_of_range:{idx}")
        return copy.deepcopy(ACTION_CATALOG[idx])

    def encode_action(self, action: dict[str, Any]) -> int | None:
        return ACTION_INDEX_BY_KEY.get(_action_key(action))

    def legal_action_mask(self, *, apply_visited_guard: bool = True) -> list[int]:
        mask = [0] * len(ACTION_CATALOG)
        for action in self.legal_actions(apply_visited_guard=apply_visited_guard):
            idx = ACTION_INDEX_BY_KEY.get(_action_key(action))
            if idx is not None:
                mask[idx] = 1
        return mask

    def legal_action_indices(self, *, apply_visited_guard: bool = True) -> list[int]:
        return [
            i
            for i, bit in enumerate(self.legal_action_mask(apply_visited_guard=apply_visited_guard))
            if bit == 1
        ]

    def _sample_random(self, turn: int) -> dict[str, Any]:
        return self.opponent_provider.sample(int(turn), forced_pid=None)

    def _sample_for_pid(self, pid: str, turn: int) -> dict[str, Any]:
        return self.opponent_provider.sample(int(turn), forced_pid=str(pid))

    def _training_rule_violation(self, action: dict[str, Any], *, apply_visited_guard: bool = True) -> str | None:
        """Env-side rule-violation pre-check inside `step()` (defense in
        depth, run before the action ever reaches the real engine).

        What is left is Part B's within-turn anti-cycle guard + progress
        cap, RE-CHECKED here (the primary enforcement point is
        `legal_actions()` / `legal_action_mask()`, consulted by
        `SapPpoGymEnv.action_masks()` before sb3-contrib ever samples an
        action) so a caller that steps a raw action index or a hand-built
        action dict WITHOUT going through the mask first still cannot force
        a within-turn cycle or exceed the cap. `step()` always calls this
        with the default (True); `apply_visited_guard=False` exists only so
        `frame_parity_harness.py`'s Part-A-only diagnostic (see
        `legal_actions()`'s docstring) stays honest when explaining a mask
        mismatch under that same mode."""
        action_type = str(action.get("type", "")).strip().upper()
        if action_type == "END_TURN" or not apply_visited_guard:
            return None
        if self._progress_action_count >= _VISITED_GUARD_PROGRESS_CAP:
            return "progress_action_cap_reached"
        next_sig = self._forward_signature(action)
        if next_sig is not None and next_sig in self.visited:
            return "would_revisit_within_turn_state"
        return None

    def step(self, action: dict[str, Any] | int) -> dict[str, Any]:
        action_index: int | None = None
        action_obj: dict[str, Any]
        if isinstance(action, int):
            action_index = int(action)
            if action_index < 0 or action_index >= len(ACTION_CATALOG):
                return {
                    "ok": False,
                    "error": f"invalid_action_index:{action_index}",
                    "transition": None,
                    "state": copy.deepcopy(self.state),
                    "done": bool(self._is_done(self.state, self.max_turn)),
                    "info": {"action_index": action_index},
                }
            action_obj = copy.deepcopy(ACTION_CATALOG[action_index])
        else:
            action_obj = copy.deepcopy(action)

        info: dict[str, Any] = {}
        if action_index is not None:
            info["action_index"] = action_index

        rule_error = self._training_rule_violation(action_obj)
        if rule_error is not None:
            return {
                "ok": False,
                "error": rule_error,
                "transition": None,
                "state": copy.deepcopy(self.state),
                "done": bool(self._is_done(self.state, self.max_turn)),
                "info": info,
            }

        if str(action_obj.get("type", "")) != "END_TURN":
            tr = step(self.state, action_obj)
            if tr.get("legal"):
                self.state = copy.deepcopy(tr["state_after"])
                self.history.append(copy.deepcopy(tr))
                self._record_progress_action()


                _set_training_rolls_this_turn(self.state, 0)
            return {
                "ok": bool(tr.get("legal")),
                "error": (None if tr.get("legal") else "illegal_action"),
                "transition": tr,
                "state": copy.deepcopy(self.state),
                "done": bool(self._is_done(self.state, self.max_turn)),
                "info": info,
            }

        meta = self.state.get("meta") if isinstance(self.state.get("meta"), dict) else {}
        game_mode = str(meta.get("game_mode") or "arena").strip().lower()
        resolved = self.end_turn_resolver(
            self.state,
            game_mode=game_mode,
            sample_random_fn=self._sample_random,
            sample_for_pid_fn=self._sample_for_pid,
            parse_cache=self._end_turn_parse_cache,
        )
        if not resolved.get("ok"):
            return {
                "ok": False,
                "error": str(resolved.get("error") or "end_turn_failed"),
                "transition": None,
                "state": copy.deepcopy(self.state),
                # END_TURN resolution failures can otherwise trap rollouts in
                # repeated illegal END_TURN attempts on an unchanged state.
                "done": True,
                "info": {
                    **info,
                    "forced_done": True,
                    "forced_done_reason": "end_turn_failed",
                    "sampled": copy.deepcopy(resolved.get("sampled")),
                    "battle": copy.deepcopy(resolved.get("battle")),
                    "parse_mode": resolved.get("parse_mode"),
                    "parse_error": resolved.get("parse_error"),
                    "forced_pid": resolved.get("forced_pid"),
                    "replay_battle": copy.deepcopy(resolved.get("replay_battle")),
                    "parsed_state": copy.deepcopy(resolved.get("parsed_state")),
                    "battle_state": copy.deepcopy(resolved.get("battle_state")),
                },
            }

        tr = copy.deepcopy(resolved["transition"])
        self.state = copy.deepcopy(tr["state_after"])
        parsed_state = resolved.get("parsed_state")
        if isinstance(parsed_state, dict):
            _set_last_opponent_team(self.state, parsed_state.get("opponentPets"))
        self.history.append(copy.deepcopy(tr))
        self._reset_turn_guard()


        _set_training_rolls_this_turn(self.state, 0)
        return {
            "ok": True,
            "error": None,
            "transition": tr,
            "state": copy.deepcopy(self.state),
            "done": bool(self._is_done(self.state, self.max_turn, one_turn_mode=self.one_turn_mode, transition=tr)),
            "info": {
                **info,
                "sampled": copy.deepcopy(resolved.get("sampled")),
                "battle": copy.deepcopy(resolved.get("battle")),
                "parse_mode": resolved.get("parse_mode"),
                "parse_error": resolved.get("parse_error"),
                "forced_pid": resolved.get("forced_pid"),
                "replay_battle": copy.deepcopy(resolved.get("replay_battle")),
                "parsed_state": copy.deepcopy(resolved.get("parsed_state")),
                "battle_state": copy.deepcopy(resolved.get("battle_state")),
            },
        }

    def advance_turn_no_battle(self) -> dict[str, Any]:
        """Advance to next shop turn without sampled opponent battle.

        Intended for offline pool generation where battle outcomes are irrelevant.
        """
        pre = resolve_end_turn_pre_battle(self.state)
        if not pre.legal:
            return {
                "ok": False,
                "error": f"end_turn_pre_failed:{pre.notes}",
                "state": copy.deepcopy(self.state),
                "done": bool(self._is_done(self.state, self.max_turn)),
            }
        post = resolve_end_turn_post_battle(pre.state_after)
        if not post.legal:
            return {
                "ok": False,
                "error": f"end_turn_post_failed:{post.notes}",
                "state": copy.deepcopy(self.state),
                "done": bool(self._is_done(self.state, self.max_turn)),
            }
        self.state = copy.deepcopy(post.state_after)
        self._reset_turn_guard()
        _set_training_rolls_this_turn(self.state, 0)
        return {
            "ok": True,
            "error": None,
            "state": copy.deepcopy(self.state),
            "done": bool(self._is_done(self.state, self.max_turn)),
            "transition": {
                "state_before": copy.deepcopy(pre.state_after),
                "action": {"type": "END_TURN_NO_BATTLE"},
                "state_after": copy.deepcopy(self.state),
                "deterministic": bool(post.deterministic and pre.deterministic),
                "stochastic_reason": (post.stochastic_reason or pre.stochastic_reason),
                "legal": True,
                "engine_notes": list(pre.notes) + list(post.notes),
            },
        }

    @staticmethod
    def _is_done(
        state: dict[str, Any],
        max_turn: int | None = None,
        *,
        one_turn_mode: bool = False,
        transition: dict[str, Any] | None = None,
    ) -> bool:
        if bool(one_turn_mode):
            action_type = str(((transition or {}).get("action") or {}).get("type") or "").strip().upper()
            if action_type == "END_TURN":
                return True
        if int(state.get("lives", 0)) <= 0:
            return True

        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        mode = str(meta.get("game_mode") or "arena").strip().lower()
        if mode == "versus":
            versus_meta = meta.get("versus") if isinstance(meta.get("versus"), dict) else {}
            try:
                opponent_lives = int(versus_meta.get("opponent_lives", 0))
            except Exception:
                opponent_lives = 0
            if opponent_lives <= 0:
                return True
        else:
            if int(state.get("trophies", 0)) >= 7:
                return True


        if max_turn is not None and int(state.get("turn", 1)) > int(max_turn):
            return True
        return False
