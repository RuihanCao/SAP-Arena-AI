"""Shared end-turn battle resolver for tools and training runtimes."""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Callable

from .engine import resolve_end_turn_post_battle, resolve_end_turn_pre_battle
from .opponents import parse_replay_for_calculator_state
from .oracles.sap_calc_battle_oracle import run_battle_oracle_with_config, team_to_pet_configs
from .versus_lives import BattleLivesUpdate, apply_battle_outcome_to_lives

DUEL_GAME_MODES = ("versus",)


def _sample_random_opponent_team(turn: int) -> dict[str, Any]:
    """Default opponent sampler: the replay database.

    Imported at call time, not module scope, so that importing the engine does
    not import the Postgres layer. Callers that pass their own
    `sample_random_fn` (the eval harness does) never reach this.
    """
    from .opponents.replay_db import sample_random_opponent_team

    return sample_random_opponent_team(turn)


def _sample_opponent_team_for_pid(pid: str, turn: int) -> dict[str, Any]:
    """Default per-participation sampler. See `_sample_random_opponent_team`."""
    from .opponents.replay_db import sample_opponent_team_for_pid

    return sample_opponent_team_for_pid(pid, turn)


def _merge_reason(existing: str | None, incoming: str | None) -> str | None:
    if existing is None:
        return incoming
    if incoming is None or incoming == existing:
        return existing
    return f"{existing}+{incoming}"


def _invert_outcome(outcome: str) -> str:
    """Flip a battle verdict to the other side's point of view."""
    verdict = str(outcome or "").strip().lower()
    if verdict == "win":
        return "loss"
    if verdict == "loss":
        return "win"
    return verdict


def apply_resolved_battle_to_state(
    battle_state: dict[str, Any],
    *,
    battle_outcome: str,
    game_mode: str,
    post_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_post_battle,
    max_lives: int | None = None,
) -> tuple[dict[str, Any], list[str], BattleLivesUpdate, Any]:
    """Advance ONE side from its pre-battle board past an ALREADY-RESOLVED battle.

    `battle_state` is the PRE-battle board (post `pre_battle_fn`, i.e. after
    end-of-turn abilities). What this does, in the original's order:

    1. `post_battle_fn` (the engine's `resolve_end_turn_post_battle`:
       advance the turn, reset gold, reroll the shop, run start-of-turn
       abilities);
    2. `versus_lives.apply_battle_outcome_to_lives` for the win/loss/draw
       life-and-trophy rules AND the turn-3 heal, which keys off the
       ADVANCED turn number -- hence after step 1;
    3. write `lives` / `meta.versus.opponent_lives` / `trophies`, the two
       `turn_start_rule:*` engine notes, and `meta.last_battle_result` +
       `meta.game_mode`.

    Returns `(state_after, notes, lives_update, post_outcome)`.
    `notes` is `post_battle_fn`'s own notes followed by the zero-to-two
    `turn_start_rule:*` notes, i.e. exactly the contiguous slice the caller
    used to splice into its engine-note list. The fourth element is
    `post_battle_fn`'s raw `StepOutcome`; the caller needs its
    `stochastic_reason` for the transition it builds, and that is the one
    thing the first three cannot carry."""
    after_battle = copy.deepcopy(battle_state)
    battle_versus_meta = after_battle.get("meta", {}).get("versus", {}) if isinstance(after_battle.get("meta"), dict) else {}
    if not isinstance(battle_versus_meta, dict):
        battle_versus_meta = {}

    mode = str(game_mode or "arena").strip().lower()
    outcome = str(battle_outcome or "").strip().lower()

    post = post_battle_fn(after_battle)
    after = copy.deepcopy(post.state_after)

    lives_kwargs: dict[str, Any] = {}
    if max_lives is not None:
        lives_kwargs["max_lives"] = int(max_lives)
    lives_update = apply_battle_outcome_to_lives(
        outcome,
        lives=int(after_battle.get("lives", 0)),
        opponent_lives=int(battle_versus_meta.get("opponent_lives", 0)),
        trophies=int(after_battle.get("trophies", 0)),
        turn_after_battle=int(after.get("turn", 1)),
        game_mode=mode,
        **lives_kwargs,
    )
    after["lives"] = lives_update.lives
    if mode == "versus":
        versus_meta = after.setdefault("meta", {}).setdefault("versus", {})
        versus_meta["opponent_lives"] = lives_update.opponent_lives
    else:
        after["trophies"] = lives_update.trophies

    notes = list(post.notes)
    # On start of turn 3, restore 1 life (up to max 6).
    if lives_update.life_bonus_applied:
        notes.append(f"turn_start_rule:turn3_life_bonus:lives={lives_update.lives}")
    if lives_update.opponent_life_bonus_applied:
        notes.append(
            f"turn_start_rule:turn3_opponent_life_bonus:opponent_lives={lives_update.opponent_lives}"
        )

    after_meta = after.setdefault("meta", {})
    after_meta["last_battle_result"] = outcome
    after_meta["game_mode"] = mode
    return after, notes, lives_update, post


def resolve_end_turn_with_sampled_battle(
    before_state: dict[str, Any],
    *,
    forced_pid: str | None = None,
    game_mode: str = "arena",
    pre_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_pre_battle,
    post_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_post_battle,
    sample_random_fn: Callable[[int], dict[str, Any]] = _sample_random_opponent_team,
    sample_for_pid_fn: Callable[[str, int], dict[str, Any]] = _sample_opponent_team_for_pid,
    parse_replay_fn: Callable[[dict[str, Any] | None, dict[str, Any] | None], dict[str, Any]] = parse_replay_for_calculator_state,
    run_battle_fn: Callable[[dict[str, Any]], dict[str, Any]] = run_battle_oracle_with_config,
    team_to_pet_configs_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]] = team_to_pet_configs,
    parse_cache: dict[str, dict[str, Any]] | None = None,
    simulation_count: int = 1,
    battle_logs_enabled: bool = False,
    max_lives: int | None = None,
) -> dict[str, Any]:
    """Resolve END_TURN using sampled replay opponents + battle oracle.

    Returns a dict envelope:
    - ok: bool
    - error: str | None
    - transition: canonical transition dict when ok, else None
    - sampled: sampled opponent payload when available
    - replay_battle: sampled replay battle dict when available
    - parsed_state: replay-bot parsed calculator state when available
    - battle: oracle result payload when available
    - parse_mode / parse_error
    - forced_pid"""
    before = copy.deepcopy(before_state)
    before_meta = before.get("meta") if isinstance(before.get("meta"), dict) else {}
    before_versus_meta = before_meta.get("versus") if isinstance(before_meta.get("versus"), dict) else {}

    mode = str(game_mode or "arena").strip().lower()
    if mode not in {"arena", "versus"}:
        return {
            "ok": False,
            "error": f"invalid_game_mode:{mode}",
            "transition": None,
            "battle_state": before,
            "sampled": None,
            "replay_battle": None,
            "parsed_state": None,
            "battle": {"ok": False, "error": "invalid_game_mode"},
            "parse_mode": None,
            "parse_error": None,
            "forced_pid": None,
            "timing_ms": {},
        }

    if mode == "versus":
        raw_opp_lives = before_versus_meta.get("opponent_lives") if isinstance(before_versus_meta, dict) else None
        try:
            _ = int(raw_opp_lives)
        except Exception:
            return {
                "ok": False,
                "error": "missing_required_field:meta.versus.opponent_lives",
                "transition": None,
                "battle_state": before,
                "sampled": None,
                "replay_battle": None,
                "parsed_state": None,
                "battle": {"ok": False, "error": "missing_required_field:meta.versus.opponent_lives"},
                "parse_mode": None,
                "parse_error": None,
                "forced_pid": None,
                "timing_ms": {},
            }

    forced_pid_val = str(forced_pid or "").strip()
    forced_pid_source: str | None = "explicit" if forced_pid_val else None
    if not forced_pid_val:
        forced_pid_val = str(before_meta.get("debug_opponent_pid") or os.getenv("SAP_PPO_DEBUG_OPPONENT_PID") or "").strip()
        if forced_pid_val:
            forced_pid_source = "debug"
    if not forced_pid_val and mode == "versus":
        forced_pid_val = str(before_versus_meta.get("current_opponent_participation_id") or "").strip()
        if forced_pid_val:
            forced_pid_source = "versus_chain"
    forced_pid_out = forced_pid_val if forced_pid_source in {"explicit", "debug"} else None

    parse_mode = "replaybot_battle_config"
    parse_error: str | None = None
    t_total_start = time.perf_counter()
    timing_ms: dict[str, int] = {}

    pre = pre_battle_fn(before)
    battle_state = copy.deepcopy(pre.state_after)

    t_sample_start = time.perf_counter()
    sampled_chain_fallback = False
    if forced_pid_val:
        sampled = sample_for_pid_fn(forced_pid_val, int(battle_state.get("turn", 1)))
        if forced_pid_source == "versus_chain" and not bool(sampled.get("ok", False)):
            sampled_chain_fallback = True
            sampled = sample_random_fn(int(battle_state.get("turn", 1)))
    else:
        sampled = sample_random_fn(int(battle_state.get("turn", 1)))
    timing_ms["sample"] = int((time.perf_counter() - t_sample_start) * 1000)

    if not sampled.get("ok"):
        sampling_error_prefix = "opponent_sampling_failed"
        if sampled_chain_fallback:
            sampling_error_prefix = "opponent_sampling_failed:chain_then_random"
        return {
            "ok": False,
            "error": f"{sampling_error_prefix}:{sampled.get('error')}",
            "transition": None,
            "battle_state": battle_state,
            "sampled": sampled,
            "replay_battle": None,
            "parsed_state": None,
            "battle": {"ok": False, "error": "opponent_sampling_failed"},
            "parse_mode": parse_mode,
            "parse_error": parse_error,
            "forced_pid": forced_pid_out,
            "timing_ms": timing_ms,
        }

    # Strict mode: no team-based fallback.
    replay_battle = sampled.get("battle")
    build_model = sampled.get("build_model")
    if not isinstance(replay_battle, dict):
        return {
            "ok": False,
            "error": "opponent_missing_battle_payload",
            "transition": None,
            "battle_state": battle_state,
            "sampled": sampled,
            "replay_battle": None,
            "parsed_state": None,
            "battle": {"ok": False, "error": "opponent_missing_battle_payload"},
            "parse_mode": parse_mode,
            "parse_error": parse_error,
            "forced_pid": forced_pid_out,
            "timing_ms": timing_ms,
        }

    parsed_state: dict[str, Any] | None = None
    cache_key: str | None = None
    replay_id = str(sampled.get("replay_id") or "").strip()
    side = str(sampled.get("side") or "").strip()
    turn_i = int(battle_state.get("turn", 1))
    if replay_id:
        cache_key = f"{replay_id}|{side}|{turn_i}"

    if isinstance(sampled.get("parsed_state"), dict):
        parsed_state = copy.deepcopy(sampled["parsed_state"])
        parse_mode = "snapshot_preparsed"
        timing_ms["parse"] = 0
        if parse_cache is not None and cache_key:
            parse_cache[cache_key] = copy.deepcopy(parsed_state)
    elif parse_cache is not None and cache_key and isinstance(parse_cache.get(cache_key), dict):
        parsed_state = copy.deepcopy(parse_cache[cache_key])
        parse_mode = "replaybot_battle_config_cached"
        timing_ms["parse"] = 0
    else:
        t_parse_start = time.perf_counter()
        parsed = parse_replay_fn(
            replay_battle,
            build_model if isinstance(build_model, dict) else None,
        )
        timing_ms["parse"] = int((time.perf_counter() - t_parse_start) * 1000)
        if not parsed.get("ok") or not isinstance(parsed.get("state"), dict):
            parse_error = str(parsed.get("error") or "replaybot_parse_failed")
            return {
                "ok": False,
                "error": f"opponent_parse_failed:{parse_error}",
                "transition": None,
                "battle_state": battle_state,
                "sampled": sampled,
                "replay_battle": replay_battle,
                "parsed_state": None,
                "battle": {"ok": False, "error": "opponent_parse_failed"},
                "parse_mode": parse_mode,
                "parse_error": parse_error,
                "forced_pid": forced_pid_out,
                "timing_ms": timing_ms,
            }
        parsed_state = copy.deepcopy(parsed["state"])
        if parse_cache is not None and cache_key:
            parse_cache[cache_key] = copy.deepcopy(parsed_state)

    if not isinstance(parsed_state, dict):
        return {
            "ok": False,
            "error": "opponent_parse_failed:parsed_state_missing",
            "transition": None,
            "battle_state": battle_state,
            "sampled": sampled,
            "replay_battle": replay_battle,
            "parsed_state": None,
            "battle": {"ok": False, "error": "opponent_parse_failed"},
            "parse_mode": parse_mode,
            "parse_error": parse_error,
            "forced_pid": forced_pid_out,
            "timing_ms": timing_ms,
        }

    config = copy.deepcopy(parsed_state)
    config["playerPets"] = team_to_pet_configs_fn(battle_state.get("team", []))
    config["playerPack"] = str(battle_state.get("pack") or config.get("playerPack") or "Turtle")
    config["turn"] = int(battle_state.get("turn", config.get("turn", 1)))
    if not config.get("opponentPack"):
        config["opponentPack"] = str(sampled.get("side_pack") or "Turtle")
    config["simulationCount"] = int(simulation_count)
    config["logsEnabled"] = bool(battle_logs_enabled)
    if battle_logs_enabled:
        config["maxLoggedBattles"] = 1

    t_battle_start = time.perf_counter()
    battle = run_battle_fn(config)
    timing_ms["battle"] = int((time.perf_counter() - t_battle_start) * 1000)
    if not battle.get("ok"):
        return {
            "ok": False,
            "error": f"battle_oracle_failed:{battle.get('error')}",
            "transition": None,
            "battle_state": battle_state,
            "sampled": sampled,
            "replay_battle": replay_battle,
            "parsed_state": parsed_state,
            "battle": battle,
            "parse_mode": parse_mode,
            "parse_error": parse_error,
            "forced_pid": forced_pid_out,
            "timing_ms": timing_ms,
        }

    battle_outcome = str(battle.get("outcome", "unknown")).lower()
    if battle_outcome not in {"win", "loss", "draw"}:
        return {
            "ok": False,
            "error": f"battle_outcome_unknown:{battle_outcome}",
            "transition": None,
            "battle_state": battle_state,
            "sampled": sampled,
            "replay_battle": replay_battle,
            "parsed_state": parsed_state,
            "battle": battle,
            "parse_mode": parse_mode,
            "parse_error": parse_error,
            "forced_pid": forced_pid_out,
            "timing_ms": timing_ms,
        }

    after, applied_notes, lives_update, post = apply_resolved_battle_to_state(
        battle_state,
        battle_outcome=battle_outcome,
        game_mode=mode,
        post_battle_fn=post_battle_fn,
        max_lives=max_lives,
    )

    after_meta = after.setdefault("meta", {})
    if mode == "versus":
        versus_meta = after_meta.setdefault("versus", {})
        sampled_pid = str(sampled.get("participation_id") or "").strip()
        if sampled_pid:
            versus_meta["current_opponent_participation_id"] = sampled_pid

    notes = list(pre.notes)
    notes.append(f"end_turn_battle_result:{battle_outcome}")
    notes.append(f"end_turn_game_mode:{mode}")
    notes.append(
        f"end_turn_battle_opponent_source:{sampled.get('source')}:{sampled.get('replay_id')}:{sampled.get('side')}"
    )
    if parse_mode:
        notes.append(f"end_turn_battle_parse_mode:{parse_mode}")
    if parse_error:
        notes.append(f"end_turn_battle_parse_error:{parse_error}")
    if battle.get("ok"):
        notes.append("end_turn_battle_oracle_ok")
    else:
        notes.append(f"end_turn_battle_oracle_failed:{battle.get('error')}")
    notes.extend(applied_notes)
    if sampled_chain_fallback:
        notes.append("end_turn_versus_chain_fallback_random")
    if mode == "versus" and forced_pid_source == "versus_chain" and forced_pid_val:
        sampled_pid = str(sampled.get("participation_id") or "").strip()
        if sampled_pid and sampled_pid != forced_pid_val:
            notes.append(f"end_turn_versus_chain_switched:{forced_pid_val}->{sampled_pid}")

    stochastic_reason = _merge_reason(pre.stochastic_reason, "end_turn_battle_sampled")
    stochastic_reason = _merge_reason(stochastic_reason, post.stochastic_reason)
    transition = {
        "state_before": before,
        "action": {"type": "END_TURN"},
        "state_after": after,
        "deterministic": False,
        "stochastic_reason": stochastic_reason,


        "stochastic_structural": bool(
            getattr(pre, "stochastic_structural", False)
            or getattr(post, "stochastic_structural", False)
        ),
        "legal": True,
        "engine_notes": notes,
    }
    timing_ms["total"] = int((time.perf_counter() - t_total_start) * 1000)

    return {
        "ok": True,
        "error": None,
        "transition": transition,
        "battle_state": battle_state,
        "sampled": sampled,
        "replay_battle": replay_battle,
        "parsed_state": parsed_state,
        "battle": battle,
        "parse_mode": parse_mode,
        "parse_error": parse_error,
        "forced_pid": forced_pid_out,
        "timing_ms": timing_ms,
    }


def _duel_failure(
    error: str,
    *,
    battle: dict[str, Any] | None = None,
    timing_ms: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Error envelope with the same key set as a successful duel turn."""
    return {
        "ok": False,
        "error": error,
        "outcome": None,
        "human": None,
        "ai": None,
        "battle": battle if isinstance(battle, dict) else {"ok": False, "error": error},
        "timing_ms": dict(timing_ms or {}),
    }


def _set_duel_last_opponent_team(state: dict[str, Any], pet_configs: list[dict[str, Any] | None]) -> None:
    """Write one side's last-seen opponent board through the TRAINING path.

    Deliberately `train.env._set_last_opponent_team` and not a local dict
    build, and deliberately fed `team_to_pet_configs(engine_team)` rather
    than the engine team itself. `tempo/features.py::_slot_token` keys on
    `pet_id` FIRST and only falls back to `name:<lower>`; every board the
    models were ever trained on came through `parsed_pets_to_team`, which
    sets `pet_id=None` and `pet_name="Cricket"`, so the token is
    `name:cricket`. Handing over a raw engine team (`pet_id="pet-cricket"`)
    puts all 390 opponent dimensions into hash buckets the model has never
    seen -- with no exception, no warning and no visible symptom other than
    the agent quietly getting worse. `play_web/app.py`'s replay path does
    the same round-trip; `test_duel_game.py` pins the shape.

    The import is function-local because `train/env.py` imports THIS module
    at module scope; a top-level import here would be a cycle.
    """
    from .train.env import _set_last_opponent_team

    _set_last_opponent_team(state, pet_configs)


def resolve_duel_turn(
    human_before: dict[str, Any],
    ai_before: dict[str, Any],
    *,
    game_mode: str = "versus",
    pre_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_pre_battle,
    post_battle_fn: Callable[[dict[str, Any]], Any] = resolve_end_turn_post_battle,
    run_battle_fn: Callable[[dict[str, Any]], dict[str, Any]] = run_battle_oracle_with_config,
    team_to_pet_configs_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any] | None]] = team_to_pet_configs,
    simulation_count: int = 1,
    battle_logs_enabled: bool = False,
    max_lives: int | None = None,
) -> dict[str, Any]:
    """The sibling of `resolve_end_turn_with_sampled_battle`, with the opponent
    replaced by a second live state instead of a sampled human replay: no
    sampling, no replay parsing, no opponent participation-id chain.

    Three things here are rules, not implementation details:

    Returns one envelope for the pair:
    - ok / error
    - outcome: the battle verdict FROM THE HUMAN'S SIDE
    - human / ai: per-side payloads (`state_before`, `battle_state` = the
      pre-battle board, `state_after`, `pet_configs` as sent to the oracle,
      that side's `outcome`, its lives/trophies after the update, the
      `turn_start_rule` flags, `notes`, and a canonical END_TURN
      `transition`)
    - battle: the oracle payload
    - timing_ms"""
    t_total_start = time.perf_counter()
    timing_ms: dict[str, int] = {}

    mode = str(game_mode or "").strip().lower()
    if mode not in DUEL_GAME_MODES:
        return _duel_failure(f"invalid_game_mode:{mode}", timing_ms=timing_ms)

    human_start = copy.deepcopy(human_before)
    ai_start = copy.deepcopy(ai_before)

    sides: dict[str, dict[str, Any]] = {"human": human_start, "ai": ai_start}
    opponent_lives_in: dict[str, int] = {}
    for name, state in sides.items():
        meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
        versus_meta = meta.get("versus") if isinstance(meta.get("versus"), dict) else {}
        try:
            opponent_lives_in[name] = int(versus_meta.get("opponent_lives"))
        except Exception:
            return _duel_failure(
                f"missing_required_field:{name}.meta.versus.opponent_lives",
                timing_ms=timing_ms,
            )

    human_turn = int(human_start.get("turn", 1))
    ai_turn = int(ai_start.get("turn", 1))
    if human_turn != ai_turn:
        return _duel_failure(f"duel_turn_mismatch:human={human_turn}:ai={ai_turn}", timing_ms=timing_ms)

    # Each side's mirror must already agree with the other side's real life
    # total, or the two `apply_battle_outcome_to_lives` calls below start
    # from different books and the game ends on whichever copy is consulted.
    human_lives_in = int(human_start.get("lives", 0))
    ai_lives_in = int(ai_start.get("lives", 0))
    if opponent_lives_in["human"] != ai_lives_in or opponent_lives_in["ai"] != human_lives_in:
        return _duel_failure(
            "duel_lives_mirror_mismatch:"
            f"human_lives={human_lives_in}:human_sees={opponent_lives_in['human']}:"
            f"ai_lives={ai_lives_in}:ai_sees={opponent_lives_in['ai']}",
            timing_ms=timing_ms,
        )

    t_pre_start = time.perf_counter()
    human_pre = pre_battle_fn(human_start)
    ai_pre = pre_battle_fn(ai_start)
    timing_ms["pre"] = int((time.perf_counter() - t_pre_start) * 1000)

    human_battle_state = copy.deepcopy(human_pre.state_after)
    ai_battle_state = copy.deepcopy(ai_pre.state_after)

    human_pets = team_to_pet_configs_fn(human_battle_state.get("team", []))
    ai_pets = team_to_pet_configs_fn(ai_battle_state.get("team", []))

    config: dict[str, Any] = {
        "playerPack": str(human_battle_state.get("pack") or "Turtle"),
        "opponentPack": str(ai_battle_state.get("pack") or "Turtle"),
        "turn": int(human_battle_state.get("turn", human_turn)),
        "playerPets": copy.deepcopy(human_pets),
        "opponentPets": copy.deepcopy(ai_pets),
        "simulationCount": int(simulation_count),
        "logsEnabled": bool(battle_logs_enabled),
    }
    if battle_logs_enabled:
        config["maxLoggedBattles"] = 1

    t_battle_start = time.perf_counter()
    battle = run_battle_fn(config)
    timing_ms["battle"] = int((time.perf_counter() - t_battle_start) * 1000)
    if not battle.get("ok"):
        return _duel_failure(f"battle_oracle_failed:{battle.get('error')}", battle=battle, timing_ms=timing_ms)

    human_outcome = str(battle.get("outcome", "unknown")).lower()
    if human_outcome not in {"win", "loss", "draw"}:
        return _duel_failure(
            f"battle_outcome_unknown:{human_outcome}", battle=battle, timing_ms=timing_ms
        )
    ai_outcome = _invert_outcome(human_outcome)

    t_post_start = time.perf_counter()
    applied: dict[str, dict[str, Any]] = {}
    for name, other, state_before, battle_state, pets, side_outcome, pre in (
        ("human", "ai", human_start, human_battle_state, human_pets, human_outcome, human_pre),
        ("ai", "human", ai_start, ai_battle_state, ai_pets, ai_outcome, ai_pre),
    ):
        after, applied_notes, lives_update, post = apply_resolved_battle_to_state(
            battle_state,
            battle_outcome=side_outcome,
            game_mode=mode,
            post_battle_fn=post_battle_fn,
            max_lives=max_lives,
        )

        notes = list(pre.notes)
        notes.append(f"end_turn_battle_result:{side_outcome}")
        notes.append(f"end_turn_game_mode:{mode}")
        notes.append(f"duel_battle_opponent_side:{other}")
        notes.append("end_turn_battle_oracle_ok")
        notes.extend(applied_notes)

        stochastic_reason = _merge_reason(pre.stochastic_reason, "duel_battle_resolved")
        stochastic_reason = _merge_reason(stochastic_reason, post.stochastic_reason)
        applied[name] = {
            "state_before": state_before,
            "battle_state": battle_state,
            "state_after": after,
            "pet_configs": copy.deepcopy(pets),
            "outcome": side_outcome,
            "lives_update": lives_update,
            "notes": notes,
            "stochastic_reason": stochastic_reason,
            "stochastic_structural": bool(
                getattr(pre, "stochastic_structural", False)
                or getattr(post, "stochastic_structural", False)
            ),
        }
    timing_ms["post"] = int((time.perf_counter() - t_post_start) * 1000)

    # Cross-feed: each side's next turn sees the board it just fought, which
    # is the OTHER side's PRE-battle board (what training's
    # `last_opponent_team` has always meant), fed through the round-trip.
    _set_duel_last_opponent_team(applied["human"]["state_after"], ai_pets)
    _set_duel_last_opponent_team(applied["ai"]["state_after"], human_pets)

    side_payloads: dict[str, dict[str, Any]] = {}
    for name in ("human", "ai"):
        entry = applied[name]
        lives_update = entry["lives_update"]
        side_payloads[name] = {
            "state_before": entry["state_before"],
            "battle_state": entry["battle_state"],
            "state_after": entry["state_after"],
            "pet_configs": entry["pet_configs"],
            "outcome": entry["outcome"],
            "lives": lives_update.lives,
            "opponent_lives": lives_update.opponent_lives,
            "trophies": lives_update.trophies,
            "life_bonus_applied": bool(lives_update.life_bonus_applied),
            "opponent_life_bonus_applied": bool(lives_update.opponent_life_bonus_applied),
            "notes": entry["notes"],
            "transition": {
                "state_before": entry["state_before"],
                "action": {"type": "END_TURN"},
                "state_after": entry["state_after"],
                "deterministic": False,
                "stochastic_reason": entry["stochastic_reason"],
                "legal": True,
                "stochastic_structural": entry["stochastic_structural"],
                "engine_notes": entry["notes"],
            },
        }

    timing_ms["total"] = int((time.perf_counter() - t_total_start) * 1000)
    return {
        "ok": True,
        "error": None,
        "outcome": human_outcome,
        "human": side_payloads["human"],
        "ai": side_payloads["ai"],
        "battle": battle,
        "timing_ms": timing_ms,
    }
