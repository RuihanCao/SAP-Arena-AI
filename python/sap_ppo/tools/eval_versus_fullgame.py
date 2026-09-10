"""exp09 W4c We1: full-game versus eval driver for a BC MaskablePPO checkpoint.

See internal design notes, section "W4c -- full-game versus eval"
(added Rev 5, 2026-07-14) for the full design. This module plays N whole
**versus** games (win = opponent_lives -> 0, loss = lives -> 0, both start at
6) with a trained BC policy and reports full-game metrics. It is the
BEFORE-W5 yardstick: nothing here is new game logic -- every piece of the
game model (turn-3 life recovery, versus life bookkeeping, battle
resolution) already lives in `end_turn.py` / `engine.py` and is reused
as-is, exactly like `TrainingEnv.step`'s END_TURN branch (`train/env.py`,
the END_TURN branch under `TrainingEnv.step`) and
`PlayWebSession._apply_end_turn_with_battle` (`tools/play_web.py`) already
do for their own callers. The ONE new piece is `ChainSnapshotSource`
(`sap_ppo.opponents.chain_snapshot`, see its docstring), imported below.

Decode discipline (Rev 5, non-negotiable, matches PLAN.md exactly):
engine-true `api.legal_actions` mask + the visited-state anti-cycle guard
as the SOLE guard + deterministic decode. This is exactly what
`BcRecommender` (`tools/bc_recommender.py`) already implements, so it is
reused unchanged here -- this module never touches `TrainingEnv.legal_actions`
(that bakes in RL anti-spam training heuristics that are not game rules,
see `bc_recommender.py`'s own docstring) and never uses
`evaluate_maskable_model` (that harness drives a gym env with env-side
masking + arena/trophy win semantics, wrong for this versus-lives frame).

Per-turn loop: `BcRecommender.recommend(state)` decodes one whole turn's
action chain (ending in END_TURN when the policy chose to stop on its own);
every non-END_TURN op in that chain is re-applied here via `api.step`
(imported as `engine_step`, mirroring `bc_recommender.py`'s own import
alias) starting from the SAME state the recommender was given -- this
reproduces bit-identical intermediate states to what the recommender
internally walked through, because `engine.step` is a pure function of
(state, action) with any RNG drawn from `state["meta"]["seed"]` (see
`engine._rng_from_state`/`_reseed_meta`), never from hidden global state,
and this is exactly the same re-simulation pattern `eval_tempo_planner.py`'s
`_simulate_state_after_chain` already uses to score `chain_preview` outputs
from this same recommender. END_TURN itself is never applied via
`engine_step` -- `resolve_end_turn_with_sampled_battle` (`end_turn.py`)
resolves the whole end-of-turn (battle + life bookkeeping + turn advance)
in one call, exactly as `TrainingEnv.step` and `PlayWebSession` do.

A defensive addition beyond the literal per-turn spec: if a chain op that
the recommender already validated internally somehow fails to re-apply here
(illegal or raises), the game is ended with `end_reason` prefixed
`chain_replay_diverged` instead of crashing or silently truncating the
chain. This should not fire in practice (see the determinism note above)
but the codebase's own `_simulate_state_after_chain` carries the identical
defensive posture (it also breaks out on `not tr.get("legal")`), so this
mirrors established practice rather than inventing new tolerance for a
real bug.

We2 additions (metrics + galleries; the We1 core above is UNCHANGED):
`play_one_game` gained an optional `capture_detail=True` path that records
each turn's pre-battle board / opponent / battle payload (see
`_build_turn_detail`) without altering the decode/replay/battle-resolution
above -- it only copies values that already existed. `run_versus_eval` is
the N-game loop (factored out of `main`, same behavior, plus a
`capture_detail` passthrough). `_compute_aggregate` adds a game-clustered
bootstrap winrate CI (`cluster_ci`, one cluster per game since games are
independent) on top of We1's plain-text aggregate.
`select_representative_games` / `render_selected_games` /
`_full_game_rows_and_sidecar` select a handful of already-played games from
`run_versus_eval`'s own `results` (captured with `capture_detail=True` -- see
below for why this is NOT a re-run) and render each as ONE WHOLE-GAME image,
one row per turn down the page (the exp08
`build_verification.py::full_game_<gid>.png` format: end-of-turn board player
vs opponent, W/L row colour, and the actual lives race in the heart), via a
SINGLE `render_replay_image_from_calc_rows` call per game (all the game's
turns in one call so `render.js` carries the running life total down the
rows). Each PNG is paired with a `full_game_<pid8>.json` sidecar (per-turn
chain/calc-link/lives). `--render-from-jsonl` re-runs JUST this rendering off
a prior run's `--out` JSONL, replaying nothing. The human-reference empirical
winrate (PLAN.md "Run + deliverables") is a SEPARATE script,
`human_reference_winrate.py` -- it reads the raw replay cache directly and
never touches the engine/BC, so it does not belong in this file.

Gotcha found while building the render path, and why rendering does NOT
re-run games: an early version re-ran the `k` selected games afterward
(`capture_detail=True` only for those) to avoid paying the capture cost on
every game. That turned out to be unrecoverably wrong, for two stacked
reasons (both verified directly, not assumed):
1. `BcRecommender`'s forward pass is not bit-reproducible across separate
   calls under torch's default multi-threaded matmul (32 threads on this
   box) -- fixable with `--torch-threads 1` (kept as the default; it is
   also simply faster for this small a model on this box).
2. FATAL even after fixing (1): the battle oracle
   (`run_battle_oracle_with_config`, `simulationCount=1`) is an UNSEEDED
   Monte Carlo draw, not a deterministic function of its board config --
   calling it 20 times on one byte-identical real config from this driver
   gave 18 draws and 2 wins. `engine_step`'s purity (`_new_game_state`'s
   docstring) covers the shop-phase ops only; battle resolution never goes
   through `engine_step` and nothing seeds the JS simulator's own RNG. So a
   re-run can diverge from the original at ANY turn whose fight is close,
   no matter how carefully the rest of the pipeline is pinned.
The fix: `run_versus_eval(..., capture_detail=True)` records every turn's
render-needed detail DURING the one authoritative pass whenever
`--render-dir` is set (see `main`), and `render_selected_games` only
selects and renders from that -- it never simulates anything itself.

We3 addition (exp09 W6a search recommender integration; We1/We2 above are
UNCHANGED): `--recommender {bc,search}` (default `bc`, fully backward
compatible) optionally wraps the constructed `BcRecommender` in
`tools/search_recommender.py::SearchRecommender` before it gets passed to
`run_versus_eval`/`play_one_game` -- both of those only ever call
`.recommend(state)` on whatever they were handed and read the returned
dict, so this is a pure substitution at ONE call site (`main`), not a new
code path through the rest of the file. `--search-candidates`/
`--search-ksim` control the wrapper's `n_candidates`/`ksim` (see that
module for the full best-of-N search algorithm and its `search_*`
bookkeeping keys). `_turn_record` now threads `rec.get("search_used")`
into `per_turn[i]["search_used"]` and `play_one_game` sums it into a
`search_used_turns` count on its return dict (cheap: `rec` was already in
scope at every `_turn_record` call site) -- always present (False/0 for
plain `--recommender bc`, since `BcRecommender.recommend`'s return dict
has no `search_used` key at all and `.get(..., False)` covers that).

We4 addition (exp09 W0c dual-decode probe; We1-We3 above are UNCHANGED):
`--decode-mode {ranked,sample}` (default `ranked`, fully backward
compatible) is passed straight through to the `BcRecommender` this driver
constructs -- see that module's docstring for the full rationale. `ranked`
reproduces We1's original scan byte-for-byte; `sample` (tuned by
`--sample-temperature`/`--sample-seed`) re-decodes the SAME checkpoints
under sampling instead, to tell apart a real PPO regression from a
ranked/greedy-decode artifact. Every one of these three flags is recorded
in `--report-json`'s `metadata` block alongside the existing
`recommender`/`search_candidates`/`search_ksim` fields, so a report is
self-describing regardless of which decode produced it.

We5 addition (exp09 W1 "teacher ceiling" rollout scorer; We1-We4 above are
UNCHANGED): `play_one_game`'s per-turn while-loop and end-of-game
bookkeeping are factored out into `play_out_game` (this module) + the
smaller `_resolve_versus_turn` helper -- `play_one_game` now just builds the
turn-1 state (`_new_game_state`) and delegates the rest. This is a PURE
refactor (the moved code is unchanged; only its call sites changed), done so
`search_recommender.py`'s new `--search-scoring rollout` mode (best-of-N
search reranked by simulating a shortlisted candidate's REST OF THE GAME,
not just its immediate battle) can call the SAME "what happens turn to
turn" logic the real driver uses, instead of a second, driftable copy of
the versus lives/turn-cap/win rules -- see `search_recommender.py`'s module
docstring for the full rollout algorithm. `play_out_game` takes
`sample_for_pid_fn`/`sample_random_fn` as plain callables (not an
`opp_source` object) so the rollout scorer can pass an ISOLATED fallback
sampler (`chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`)
without perturbing the real game's own shared `ChainSnapshotSource`
instance -- a throwaway rollout continuation must never advance that
instance's `_random_rng`, or the real game's later (for-real) fallback
draws would silently depend on how many rollouts happened to run first.
`_resolve_versus_turn` additionally takes `simulation_count` (default 1,
forwarded to `end_turn.py::resolve_end_turn_with_sampled_battle`'s own new
parameter of the same name), which the rollout scorer sets to
`--rollout-ksim` for a shortlisted candidate's OWN turn only -- every
subsequent simulated turn in that same continuation reverts to the default
(the driver's normal single-draw mechanics). New CLI flags
`--search-scoring {myopic,rollout}` (default `myopic`, fully backward
compatible -- byte-for-byte the existing We3 behavior), `--rollout-shortlist`,
`--rollout-repeats`, `--rollout-ksim` are threaded straight through to
`SearchRecommender`'s new constructor kwargs and recorded in
`--report-json`'s `metadata` alongside every existing flag.

We6 addition (exp09 W2 distillation data generation; We1-We5 above are
UNCHANGED): `play_out_game` gains an optional `on_turn` callback (default
None -- every existing caller, including `search_recommender.py`'s own
rollout continuations, omits it and pays zero extra cost: the new
`hook_state_before` snapshot and per-step `step_records` list are only
built `if on_turn is not None`). When set, `on_turn` is invoked exactly
once per turn ATTEMPTED (mirroring the existing one-`per_turn.append`-per-
turn invariant), after that turn's outcome is fully known, with a dict:
`{"turn": int, "state_before": the state at the TOP of this turn (BEFORE
`bc.recommend` acted on it, independent of and always deep-copied
regardless of `capture_detail`), "rec": the full `bc.recommend(state)`
result dict UNCHANGED (chain_preview, search_used, search_diagnostics,
...), "step_records": a list of {"state_before": state, "action": action}
for every non-END_TURN op in `rec["chain_preview"]` that was actually
committed during this turn's chain replay (in play order; empty if decode
failed before any op, or if `rec["chain_preview"]` had none), "ok": bool
(whether this turn resolved a battle cleanly -- False for the three
infra-failure branches below), "end_reason": set (only) on a
terminal/infra-failure turn, "state_after": the resulting state if `ok`
else None, "board_pre_battle": the board immediately before END_TURN
resolves (same value `_build_turn_detail` already captures under that name
for `capture_detail` -- the state a genuine END_TURN decision would be
"before", since `step_records` itself never includes one; None when no
board was ever reached (the decode-failed branch))}`. This is the
machinery `tools/gen_distill_dataset.py` (PLAN.md W2) uses to emit one
training row per within-turn step of the CHOSEN (teacher-search-winning)
chain (PLUS one more for the END_TURN decision itself, mirroring
dataset_v2's own "end_turn IS a trainable label" convention,
`replay_compile/serialize.py`'s module docstring), plus a per-turn sidecar
of every candidate the teacher considered -- see that module's docstring.
Purely additive: no existing branch/return value/bookkeeping below changes
when `on_turn` is None.

We7 addition (exp10 W2 opponent-source ablation; We1-We6 above are
UNCHANGED): new CLI flag `--rollout-opponent-mode {true,pool_random,
retrieval}` (default `"true"`, fully backward compatible -- byte-for-byte
the existing We5 rollout behavior), threaded straight through to
`SearchRecommender`'s new constructor kwarg of the same name and recorded
in `--report-json`'s `metadata` alongside every existing flag. This module
itself is UNCHANGED beyond that one extra kwarg pass-through and print/
metadata line -- the actual opponent-substitution logic lives entirely in
`search_recommender.py::SearchRecommender._rollout_score_candidate` /
`_choose_rollout_opponent_pid` (see that module's docstring, "exp10 W2"
section), which only ever calls this module's EXISTING, unmodified
`_resolve_versus_turn` / `play_out_game` / `ChainSnapshotSource.
sample_for_pid` plumbing with a different starting pid spliced into the
board it hands them.

We8 addition (exp12 W0 "the ruler"; We1-We7 above are UNCHANGED):
`--opponent-mode {chain,arena}` (default `chain`, byte-identical to every
run before this flag existed) selects WHICH opponent a turn is scored
against.

- `chain` (the historical ruler): one real player's recorded game is
  followed for the whole game -- `_new_game_state` seeds
  `meta.versus.current_opponent_participation_id`, `end_turn.py` re-forces
  that pid every turn and rewrites it after each resolved turn, and the
  random fallback only fires when the followed chain runs out of turns.
  You fight ONE opponent's whole build arc.
- `arena` (exp12 W0): the real game's actual pairing rule -- a FRESH
  opponent every turn, drawn uniformly from everyone still in the lobby at
  that turn. Mechanically this is just "no forced pid": `play_out_game`
  pops `current_opponent_participation_id` off the state at the top of each
  turn, so `end_turn.py`'s existing `sample_random_fn` branch (the one the
  chain ruler only reaches on exhaustion) becomes the normal path. The
  sampler handed in is a per-game-seeded `sample_random_with_rng` closure,
  so a given (seed, game_index) always draws the same opponent sequence.

Arena pool (Ruihan's ruling, 2026-07-28): the arena opponent pool is built
from the SAME `--chain-snapshot` file and the same `--opponent-pack`, but
with ALL splits and NO rank cap BY DEFAULT -- the full pool, because a real
lobby is not rank-filtered. `--opponent-split`/`--opponent-rank-max` keep
governing the CHAIN pool exactly as before (and, in arena mode, still govern
the `initial_pid_for_game` draw that is deliberately kept -- see below); they
never reach the arena pool, which in arena mode is the ONLY pool a turn's
opponent comes from. The arena pool has its own selectors:
`--arena-pool-rank-max` rank-caps it, for the "what does a rank-matched lobby
look like" bracket probe, and `--arena-pool-split` (exp13 W0b) restricts it
to ONE split -- without which a val-only gate ruler could not be expressed at
all, since `--opponent-split` only filters a pool arena mode discards (exp13
PLAN.md D1 requires zero overlap between the gate pool and the train-split
generation pool). The effective arena pool (split, where that split came
from, rank cap, game count, per-turn sizes) is echoed into `--report-json` so
a report is self-describing.

Two deliberate non-changes in arena mode, both so the two rulers stay
comparable game-for-game:
- `initial_pid_for_game(game_index)` is STILL called (and still seeds
  `_new_game_state`), purely so every downstream RNG stream stays aligned
  with a chain-mode run of the same seed. The pid it returns is popped
  before it can ever be used; the per-game result records it with
  `initial_pid_unused: true` so nobody mistakes it for an opponent.
- The fallback counters stay in the output but are structurally 0 (a chain
  can never "run out" when no chain is being followed), so
  `fallback_semantics: "n/a_arena"` is emitted alongside them rather than
  letting a 0.000 fallback rate read as "this ruler never needed a
  fallback".

SCOPE NOTE (exp12 W0): `--opponent-mode` governs the DRIVER's per-turn
opponent draw only. `SearchRecommender`'s internal rollout continuations
(`--recommender search --search-scoring rollout`) still follow the CHAIN
pool they are handed (`opp_source=`), i.e. an arena-ruler search run
searches against chain-style continuations. That is intentional for W0
(SearchRecommender is explicitly untouched this wave) and is recorded in
the report metadata rather than silently assumed.

Also We8: `--game-index-start` (default 0) offsets `run_versus_eval`'s game
loop to `range(start, start + num_games)`, so an N=1000 run can be sharded
across cores as disjoint game-index windows of the SAME frame (each shard's
opponents/openings/engine seeds are functions of game_index, so the union
of shards is exactly the unsharded run).

We8 RNG change, and the ONE way exp12 is not byte-compatible with earlier
chain runs (codex review F2): the chain ruler's random-fallback draws used
to come from `ChainSnapshotSource._random_rng`, a single stream seeded per
PROCESS. That is incompatible with the sharding above -- shard 1 would
replay shard 0's fallback sequence, and neither would match the unsharded
run -- so BOTH rulers now draw from a per-game
`random.Random("<salt>:{seed}:{game_index}")` (distinct salts per ruler).
Same distribution, different sample: a pre-exp12 chain number is
reproducible only statistically across this boundary, which is why the W0
acceptance for the chain arm is "greedy N=1000 reproduces 0.118 within CI"
rather than an exact match. Reports record `fallback_rng: "per_game"`.

We9 addition (exp12 W1'a self-play generation recorder; We1-We8 above are
UNCHANGED): `--afterstate-out PATH` (default None, fully backward
compatible -- every existing caller/run is byte-for-byte unaffected) opens
PATH in APPEND mode and streams one JSONL row per turn whose shop-phase
chain fully replayed and reached END_TURN resolution, plus one final
outcome row per game. This is the dataset W1'b trains V_game v2 on (PLAN.md
"W1': V_game v2 from self-generated games").

Row shapes (both carry `game_index`, injected by `run_versus_eval`/`main`,
the ONLY piece `play_out_game` itself does not know):
- Per-turn: `{"game_index", "turn", "stop_reason", "lives", "opp_lives",
  "wins", "state"}`. `state` is a DEEP COPY of `board` at the exact point
  `play_out_game` calls `_resolve_versus_turn(board, ...)` -- i.e. the
  literal input to battle resolution, `meta.versus` intact -- NOT
  `turn_resolution["battle_state"]` (see `_build_turn_detail`'s docstring:
  that value is `resolve_end_turn_with_sampled_battle`'s own internal
  normalization of the board fed to the oracle, a different object that can
  drop the wider state envelope). `lives`/`opp_lives` are the driver-tracked
  engine-frame values at the END_TURN decision (`pre_turn_lives`/
  `pre_turn_opp_lives`, already computed at the top of this turn's loop
  iteration for the existing failure-branch `_turn_record` calls below --
  reused here, not recomputed). `wins` is the PRE-BATTLE cumulative count of
  this game's own prior turns whose `outcome == "win"` (Vic semantics,
  RESULTS_W1 finding 2's `Battle.UserBoard.Vic`: pre-battle cumulative
  wins) -- tracked by a local `wins_so_far` counter, incremented only after
  a turn resolves `ok=True` with `outcome == "win"`, so turn N's row always
  reflects turns 1..N-1. A row is emitted whenever `_resolve_versus_turn` is
  actually called, regardless of whether that call then succeeds -- turns
  that never reach it (`decode_failed:*`, `chain_replay_diverged:*`) have no
  pre-battle afterstate to record and are correctly absent.
- Final (one per game, after `play_out_game`'s loop exits): `{"game_index",
  "final": true, "win", "end_reason", "turns_survived"}` -- the same values
  the game's own `--out` JSONL row already carries, so the two files agree
  by construction (both read off `play_out_game`'s own return locals).

Anti-buffering discipline (the reason this is a NEW mechanism and not
`capture_detail`, which whole-run-buffers every `per_turn[i]["detail"]` in
`results` until `main` returns -- fine at N~=300, not at a 2000-game shard):
`play_out_game` never appends afterstate rows to any list. Each row is
handed to an `afterstate_sink` callback the instant it is known and nothing
downstream retains it; `run_versus_eval` builds one PER-GAME closure
(mirrors `gen_distill_dataset.py`'s own `on_turn`-wrapping `_TurnEmitter`
pattern -- the established way this repo injects `game_index`, which the
inner per-turn machinery never knows, into a lower-level per-turn hook) that
writes + flushes the file handle on every row, so a killed shard's file
ends at the last fully-written row, never a half-written game.

This is a pure ADDITION at three call sites -- `play_out_game` (the sink
invocations + `wins_so_far`), `play_one_game` (passthrough), `run_versus_eval`
(the per-game closure + `afterstate_out_fh` passthrough) -- and one CLI flag
+ open/close in `main`. `capture_detail`, `on_turn`, and every existing
branch/return value are untouched.

Wa addition (exp12 route a, wave A0 -- the ROLLOUT-TEACHER recorder; We1-We9
above are UNCHANGED and every existing run is byte-for-byte unaffected):
`--teacher-record-out PATH` streams, per SEARCHED decision of a
`--recommender search --search-scoring rollout` run, one gzipped JSONL row
carrying the whole decision: every scored candidate's afterstate (the exact
board handed to scoring), its myopic score, its rollout (teacher) score with
the per-repeat outcomes behind it, the CRN keys those repeats ran under, the
chosen flag, and the driver-true race scalars. Route a regresses V on those
continuous teacher scores instead of on 0/1 game outcomes (RESULTS_W1
finding 19: the currency search actually needs is WITHIN-DECISION ranking,
and 0/1 outcomes carry neither the contrast nor the resolution for it).

Two companion flags:
- `--teacher-rollouts N` (default 8): rollout repeats per candidate AT LABEL
  TIME. Deliberately a separate knob from `--rollout-repeats` (the deployable
  preset's 4), so recording can buy a lower-variance label without editing
  the preset every other run reads; it only takes effect together with
  `--teacher-record-out`.
- `--rollout-crn` / `--no-rollout-crn` (default: ON when recording, OFF
  otherwise): common random numbers across the sibling candidates of one
  decision -- see `search_recommender.py`'s "exp12 route a, wave A0" section.
  Defaulting it OFF outside the recorder is what keeps the live W2c width
  arms on their own frame.

Row shapes (both carry `game_index`, injected by `run_versus_eval`):
- Per decision: `{"schema", "game_index", "turn", "decision_id", "seed",
  "race": {turn, lives, opp_lives, wins}, "played_signature",
  "played_chain", "start_state", "n_generated", "n_dedup", "n_candidates",
  "shortlist_size", "chosen_index", "chosen_signature",
  "chosen_teacher_score", "myopic_chosen_index", "crn", "candidates": [...],
  "self_check": {...}}`. `race` is the driver's own `pre_turn_lives` /
  `pre_turn_opp_lives` / `wins_so_far` (Vic semantics, RESULTS_W1 finding 2)
  -- the identical values the We9 afterstate row records, read off the same
  locals. `start_state` is the board handed to `bc.recommend`, so every
  candidate afterstate is re-derivable (and the checker's D1-style gate
  re-derives them). `played_signature` is `visited_guard.state_signature` of
  the board the driver ACTUALLY replayed this turn, so a record whose chosen
  candidate is not the board that got fought is detectable from the file
  alone.
- Final (one per game): `{"game_index", "final": true, "win", "end_reason",
  "turns_survived"}` -- same locals as the We9 final row, so teacher scores
  can be joined against realized outcomes for A3's calibration report.

`self_check` is computed inline (chosen-score bit-exactness, argmax
consistency, group completeness, chosen-board identity) and written into the
row, but it is NOT the gate: `tools/check_teacher_record.py` RE-derives all
four from the file itself and only cross-checks the inline flags, so a bug
in the recorder cannot certify itself.

Anti-buffering discipline is We9's, unchanged and for the same reason (a
teacher row is far bigger than an afterstate row -- it carries N boards, not
one): rows are written and flushed the instant they are known, nothing is
retained, and `teacher_record` rides on the recommend result's TOP level
(never inside `search_diagnostics`, which `_turn_record` keeps for the whole
run) so it is dropped as soon as the turn ends.

Wb addition (exp13 W0a -- the REAL-ARENA RULER; We1-We9/Wa above are
UNCHANGED and every existing run is byte-for-byte unaffected):
`--game-rules {versus,arena}` (default `versus`) selects the GAME the driver
is playing, a separate axis from `--opponent-mode`, which selects WHO you
play each turn.

- `versus` (default): 6 starting lives on both sides, you win when the
  opponent's life bar empties. Every prior run of this driver.
- `arena` (internal design notes D1): the real arena. 5
  starting lives, no opponent life bar at all; a battle win is a TROPHY,
  10 trophies COMPLETES the run (`END_REASON_TROPHIES_10`, a win), 0 lives
  ends it as a loss, and the `--max-turn` cap still ends it as a non-win.
  The turn-3 heal caps at 5 (the starting value) instead of 6 --
  `versus_lives.apply_battle_outcome_to_lives`'s new `max_lives`, forwarded
  through `end_turn.py`, so BOTH rulers and the human reference keep sharing
  the one implementation of "who loses a life, and when does the heal fire".

Three consequences worth stating out loud rather than leaving to be
rediscovered:

1. Arena rules require `--opponent-mode arena`. `end_turn.py` only follows
   `meta.versus.current_opponent_participation_id` when it is resolving a
   VERSUS battle, so an arena-rules game cannot follow a chain; rather than
   silently turning a `--opponent-mode chain` run into a random-opponent
   run, `main` rejects the combination.
2. Arena rules refuse `--search-scoring rollout`. `SearchRecommender`'s
   rollout continuations resolve their own turns under VERSUS rules (they
   call `play_out_game` with its default), so under arena rules they would
   score every continuation against a life race that this game does not
   have -- `_versus_win(lives, opp_lives)` with no opponent life bar reads
   as an instant win. Wiring the scorer to the arena frame is exp13 W1a;
   until then `main` rejects the combination instead of producing numbers
   that look fine and are not.
3. The RACE BLOCK the learned leaf consumes is a pre-registered CONVENTION
   under arena rules, not a measurement -- see `--arena-race-convention`
   and PLAN.md D2. V's 4-dim bypass is `[turn/15, lives/6, opp_lives/6,
   wins/15]`, and arena has no `opp_lives`; the convention says what to feed
   that slot. `play_out_game` writes it onto `meta.versus.opponent_lives` at
   the top of every turn, which is where `SearchRecommender._race_scalars`
   already reads it from, so the value the leaf scores with, the value the
   recorders store, and the value this driver reports are one number with
   one definition. Under arena rules that field is therefore a RACE FEATURE,
   never a life total (nothing in `end_turn.py`'s arena branch reads or
   writes it), and `wins` is the trophy count.

Durability of that stream is per GAME, not per row, and deliberately so:
gzip's per-row `flush()` does not terminate a member, so a single streamed
member left unterminated by a killed shard makes the WHOLE file raise
`EOFError` in any `gzip.open` reader. `TeacherRecordWriter` therefore closes
one complete gzip member per game (concatenated members, read back
transparently), the per-row flush is kept on top of it, and
`check_teacher_record.iter_rows` salvages a truncated trailing member instead
of raising -- so a killed shard costs at most the rows of the game that was
in flight. Measured overhead of the framing: +1.9% (~509 B/game) on the A0
smoke record.

Wd addition (exp13 PLAN Amendment A1, `--turn-mode`; the default
`whole-determinized` is byte-identical to every run before this flag, and
the pin `test_exp13_honest_frame.TestDefaultModeByteIdentity` exists to keep
it that way):

`--turn-mode segmented-honest` is the HONEST-ROLL frame. Read
`tools/honest_frame.py`'s module docstring for the finding it answers -- in
one line, `_new_game_state` turns `meta.seed_known` on (it has to: the BC
decode walk and this driver's own chain replay must land on the same board),
which makes the rest of the game's randomness a deterministic function of
the state, so a search layer replaying candidate chains through the engine
sees this turn's ROLL BEFORE deciding whether to roll. Humans never had
that. Two things change here, and nothing else does:

1. STREAM SEPARATION. The state handed to `bc.recommend` each segment is an
   IMAGINED CLONE whose `meta.seed` is `S(engine_seed, turn, segment_index,
   0)` (`honest_frame.imagined_clone`). Play itself keeps stream P
   untouched: ops are committed against the REAL board, so We8
   reproducibility, `--game-index-start` sharding and the galleries all
   survive. Every engine walk downstream of `recommend` -- the BC decode,
   search's sampled decode, search's prefix replay, its resample
   completions -- inherits S from that clone, so no imagination path can
   read P's position.
2. SEGMENTED TURNS. The chosen chain is committed OP BY OP, and after any op
   whose engine transition reports a STRUCTURAL stochastic resolution
   (`stochastic_structural`, exp13 PLAN Amendment A2.1) the loop STOPS,
   keeps the real outcome, and re-searches from the observed state with
   `segment_index + 1`. Segments per turn = 1 + realized structural
   resolutions. Detection is by that field alone, never an action-type or
   pet whitelist, so a future randomness source that writes
   `legal_actions`'s read-set is a boundary the day it is added, and
   `search_recommender.py::_prefix_walk` filters on the SAME boolean, so the
   search cut and this cut are identical by construction. This keeps
   `chain_replay_diverged`'s only real cause structurally removed: nothing
   that could change the legal action set is ever committed blind.

   Amendment A4 (2026-08-06) made that flag CAUSAL: a read-set write the dice
   did not steer no longer raises it, because the search already imagined that
   board when it planned the chain. The loop here is unchanged.

   A2 narrowed the criterion from A1's "the engine consumed a random number"
   to "the engine wrote a field `legal_actions` reads". `RESULTS_W0d.md` is
   the evidence: A1's rule ran 5.304 segments per turn at a measured 1.604 s
   each, of which `ability_randomness` was 44.6% -- and 96.8% of the
   resolutions raising it were the event-order tie-break alone, board- and
   mask-invariant in 3,106 of 3,106 exhaustive permutations, 87.9% of them
   unable to resolve a single handler. ROLL, the level-up reward slot, a
   random summon and `unsupported_effect` still cut; a stat buff on random
   friends, a random-target food and a tie order no longer do.

Per-turn records gain `segments` (one entry per segment, with that
segment's own `chain_preview`, what was actually committed, the boundary
reason and the imagination seed), `n_segments` and `segments_capped`; the
per-game row gains `turn_mode`, `segments_by_turn`, `n_segments_total`,
`n_searched_segments` and `boundary_reason_counts`; the report gains a
`segments` block. ALL of those keys are emitted ONLY under the honest frame
-- the same arena-only-tag discipline `_new_game_state` uses for
`meta.game_rules` -- which is what keeps a determinized run byte-identical.

Expectation scoring at the root (A1 section 3) lives in
`search_recommender.py`; this module refuses the flag combinations where it
does not exist rather than running a biased leaf (see
`_turn_mode_refusal`).

We addition (exp13 W0b', the codex review of the above), two guards and one
performance flag:

- `_turn_mode_agreement_refusal`: the RECOMMENDER's frame and the DRIVER's
  frame must be the same one. The pre-existing guard read only `.scoring`, so
  a determinized-built `SearchRecommender` under an honest driver ran
  silently, and the reverse truncated every chance-node turn without
  re-searching. See that function for both failure shapes.
- `_honest_frame_aggregate` REFUSES a mixed determinized+honest results list
  instead of labelling itself honest and quietly reporting the honest subset.
  A determinized row carries no `turn_mode` by design, which is exactly what
  made the mislabel invisible.
- `--skip-imagined-validation` (off by default, PERFORMANCE ONLY): schema
  validation is 70.6% of this driver's wall time at arm C width, so the flag
  turns it off for IMAGINED walks -- the BC decode, search's candidate replay,
  prefix walks, resample completions. This module's own committed replay
  imports `api.step` directly and cannot inherit it. Pinned to move no number
  by `test_exp13_imagined_validation.TestByteIdentity`, and recorded in the
  report metadata either way.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

# `step`, deliberately NOT `imagined_step`: this module owns the COMMITTED
# replay (every op that actually enters a game's history goes through here),
# and exp13 W0b's validation-skip lever must never reach it. See
# `api._SKIP_IMAGINED_VALIDATION` and `--skip-imagined-validation`.
from ..api import set_skip_imagined_validation, step as engine_step
from ..end_turn import resolve_end_turn_with_sampled_battle
from ..opponents.chain_snapshot import (
    CHAIN_SNAPSHOT_VERSION,
    DEFAULT_CHAIN_SNAPSHOT,
    DEFAULT_LONG_MIN,
    DEFAULT_SEED,
    SPLIT_NAMES,
    ChainSnapshotSource,
)
from ..oracles.sap_calc_battle_oracle import battle_worker_stats
from ..train.env import TrainingEnv, _set_last_opponent_team, load_initial_state_from_fixture
from ..versus_lives import (
    ARENA_MAX_LIVES,
    ARENA_START_LIVES,
    GAME_MODE_ARENA,
    GAME_MODE_VERSUS,
    MAX_TROPHIES,
    VERSUS_START_LIVES,
)
from ..train.opening_source import (
    FixedOpeningSource,
    VariedOpeningSource,
    build_varied_opening_source,
    fixed_opening_source,
)
from ..visited_guard import state_signature
from .honest_frame import (
    DEFAULT_STOCHASTIC_SAMPLES,
    DEFAULT_TURN_MODE,
    MAX_SEGMENTS_PER_TURN,
    PROPOSAL_SAMPLE_R,
    TURN_MODE_SEGMENTED_HONEST,
    TURN_MODES,
    imagination_seed,
    imagined_clone,
    is_honest,
    normalize_turn_mode,
    read_engine_seed,
)
from .cat_trigger_audit import empty_counts as _cat_audit_empty, sum_counts as _cat_audit_sum
from .segmented_turn import bc_decide, run_segmented_turn
from .bc_recommender import DECODE_MODES as BC_DECODE_MODES
from .bc_recommender import DEFAULT_DECODE_MODE as BC_DEFAULT_DECODE_MODE
from .bc_recommender import DEFAULT_SAMPLE_SEED as BC_DEFAULT_SAMPLE_SEED
from .bc_recommender import DEFAULT_SAMPLE_TEMPERATURE as BC_DEFAULT_SAMPLE_TEMPERATURE
from .bc_recommender import BcRecommender
from .cluster_ci import cluster_ci
from .search_recommender import COMPLETION_AGG_MAX as SEARCH_COMPLETION_AGG_MAX
from .search_recommender import COMPLETION_AGGREGATES as SEARCH_COMPLETION_AGGREGATES
from .search_recommender import COMPLETION_BC_GREEDY as SEARCH_COMPLETION_BC_GREEDY
from .search_recommender import COMPLETION_POLICIES as SEARCH_COMPLETION_POLICIES
from .search_recommender import DEFAULT_KSIM as SEARCH_DEFAULT_KSIM
from .search_recommender import DEFAULT_N_CANDIDATES as SEARCH_DEFAULT_N_CANDIDATES
from .search_recommender import DEFAULT_ROLLOUT_KSIM as SEARCH_DEFAULT_ROLLOUT_KSIM
from .search_recommender import DEFAULT_ROLLOUT_OPPONENT_MODE as SEARCH_DEFAULT_ROLLOUT_OPPONENT_MODE
from .search_recommender import DEFAULT_ROLLOUT_REPEATS as SEARCH_DEFAULT_ROLLOUT_REPEATS
from .search_recommender import DEFAULT_ROLLOUT_SHORTLIST as SEARCH_DEFAULT_ROLLOUT_SHORTLIST
from .search_recommender import DEFAULT_SCORING as SEARCH_DEFAULT_SCORING
from .search_recommender import ROLLOUT_OPPONENT_MODES as SEARCH_ROLLOUT_OPPONENT_MODES
from .search_recommender import SCORING_MODES as SEARCH_SCORING_MODES
from .search_recommender import SCORING_ROLLOUT as SEARCH_SCORING_ROLLOUT
from .search_recommender import SCORING_VGAME as SEARCH_SCORING_VGAME
from .search_recommender import SearchRecommender

from .._artifact_defaults import BC_CHECKPOINT as DEFAULT_BC_CHECKPOINT  # noqa: E402
DEFAULT_MAX_TURN = 30

# Wa (exp12 route a, wave A0): label-time rollout repeats per candidate, used
# only when --teacher-record-out is set. 8 (not the deployable preset's 4)
# because the recorded number is a regression TARGET, not a move choice: its
# standard error is what a distilled V can at best reproduce.
DEFAULT_TEACHER_ROLLOUTS = 8

# exp12 W2 (wave A4): the vgame leaf's two knobs, as LITERALS rather than
# imported from `tools/vgame_scorer.py`, because that module imports torch and
# this driver has non-torch entry points (`--render-from-jsonl`, and
# `--recommender llm` without a BC fallback). The scorer is imported lazily,
# inside the `--search-scoring vgame` construction branch, so a run that does
# not ask for the learned leaf never pays for (or fails on) that dependency.
# `tests/test_vgame_scoring.py` pins these against the scorer's own constants.
VGAME_DEFAULT_BLEND = 0.0
VGAME_DEFAULT_PESSIMISM = 0.0

# Turn-1 versus fixture (verified: turn=1, lives=6, meta.game_mode="versus",
# meta.versus.opponent_lives=6 -- see fixtures/parity_cases/sample_case.json).
# Relative to repo root, matching this repo's established convention of
# repo-root-relative default paths for fixtures/manifests (e.g.
# eval_tempo_planner.py's `--case-manifest` default).
FIXTURE_PATH = Path("fixtures/parity_cases/sample_case.json")

# The full closed set PLAN.md enumerates, plus one defensive addition
# (`chain_replay_diverged`, see module docstring) that is not expected to
# fire in practice.
END_REASON_PLAYER_LIVES_0 = "player_lives_0"
END_REASON_OPPONENT_LIVES_0 = "opponent_lives_0"
END_REASON_TURN_CAP = "turn_cap"
END_REASON_DECODE_FAILED = "decode_failed"
END_REASON_END_TURN_FAILED = "end_turn_failed"
END_REASON_CHAIN_REPLAY_DIVERGED = "chain_replay_diverged"
# Wb (exp13 W0a): the arena ruler's own completed verdict -- 10 trophies
# reached, i.e. the run was COMPLETED and that is a WIN. Structurally
# impossible under `--game-rules versus` (no trophy is ever awarded there),
# so adding it to the completed set below leaves versus reports untouched.
END_REASON_TROPHIES_10 = "trophies_10"
# exp13 F1 (Ruihan's ruling, 2026-08-07): the ARENA opponent pool ran out of
# candidates at this turn, so the game cannot be carried further -- but the
# trophies already earned are KEPT, exactly as they are when the game hits
# `--max-turn`. Reaching the end of the pool means the agent survived that
# long; being punished for it inverts the thing the gate measures. This is
# the residual path only: `effective_arena_max_turn` normally ends such a
# game as `turn_cap` one turn earlier, and this reason fires only when the
# pool has a HOLE at some turn below its own maximum rather than a flat tail
# truncation. Structurally impossible under `--game-rules versus` and under
# `--opponent-mode chain` (see `play_out_game`'s guard), so adding it to the
# completed set below leaves every non-arena report untouched.
END_REASON_OPPONENT_POOL_EXHAUSTED = "opponent_pool_exhausted"

# A game's `end_reason` is a GENUINE completed-game verdict iff it is exactly
# one of these five. Everything else -- `decode_failed:*`, `end_turn_failed:*`,
# `chain_replay_diverged:*` -- is an INFRASTRUCTURE failure (the eval harness
# could not carry the game to a lives verdict), NOT a policy loss. Those get
# `win=False` too, so counting them in the winrate denominator would silently
# bias the number DOWN (FIX 1). The winrate + its CI + the policy-outcome
# averages are therefore computed over completed games ONLY; incomplete games
# are counted and broken down separately so the drop is visible, never hidden.
# The completed reasons are always emitted as these EXACT strings (the
# failure reasons alone carry a `:suffix` -- `decode_failed` included since
# exp13 W1c, see `_decode_failure_reason`), so exact-set membership is the
# correct, unambiguous test.
#
# `opponent_pool_exhausted` is a completed verdict and not a failure because
# of what the two classes MEAN downstream: a failure is scored 0 trophies by
# `validate_exp13_report.py`, and running out of opponents is not a reason to
# erase a game's trophies. See that constant's own comment.
COMPLETED_END_REASONS = frozenset(
    {
        END_REASON_PLAYER_LIVES_0,
        END_REASON_OPPONENT_LIVES_0,
        END_REASON_TURN_CAP,
        END_REASON_TROPHIES_10,
        END_REASON_OPPONENT_POOL_EXHAUSTED,
    }
)

# The `resolve_end_turn_with_sampled_battle` error suffix that means "the
# snapshot pool holds no candidate at this turn" (`chain_snapshot.py`'s
# `sample_random_with_rng`). Matched as a substring rather than a prefix
# because `end_turn.py` wraps it in one of two `opponent_sampling_failed`
# prefixes and `play_out_game` wraps THAT in `end_turn_failed:`.
POOL_EXHAUSTED_ERROR_TOKEN = "no_snapshot_opponent_for_turn:"

# We8 (exp12 W0 "the ruler") -- see module docstring's "We8 addition".
OPPONENT_MODE_CHAIN = "chain"
OPPONENT_MODE_ARENA = "arena"
OPPONENT_MODES: tuple[str, ...] = (OPPONENT_MODE_CHAIN, OPPONENT_MODE_ARENA)

# What a run's `num_fallbacks`/`fallback_turns`/`fallback.rate_games` MEAN
# under each ruler. Emitted next to those counters (per game and in the
# aggregate) because under `arena` they are structurally 0 -- there is no
# followed chain that could run out -- and a bare 0.000 would otherwise read
# as "this ruler happened to need no fallbacks", a very different claim.
FALLBACK_SEMANTICS_CHAIN = "followed_chain_exhausted_random_resample"
FALLBACK_SEMANTICS_ARENA = "n/a_arena"


def _fallback_semantics(opponent_mode: str) -> str:
    return FALLBACK_SEMANTICS_ARENA if opponent_mode == OPPONENT_MODE_ARENA else FALLBACK_SEMANTICS_CHAIN


# Wb (exp13 W0a) -- see the module docstring's "Wb addition". WHICH GAME is
# being played, orthogonal to `OPPONENT_MODES` (WHO you play).
GAME_RULES_VERSUS = "versus"
GAME_RULES_ARENA = "arena"
GAME_RULES: tuple[str, ...] = (GAME_RULES_VERSUS, GAME_RULES_ARENA)

# PLAN.md D2, the two pre-registered race conventions for the 4-dim V bypass
# under arena rules. `wins` := trophies and `lives` := true lives under BOTH;
# they differ only in what fills the `opp_lives` slot:
#   - trophies_mapped: min(6, 10 - trophies), i.e. "distance to completing
#     the run", mapped onto the feature that carried that semantic in versus
#     ("how many more battles until the other bar empties").
#   - const6: a constant 6, i.e. "tell V nothing" -- the null convention.
# W0c's probe picks one on the gate ruler and pins it; every later round
# reuses the pinned value, which is what makes V self-consistent with the
# frame it was trained under.
ARENA_RACE_TROPHIES_MAPPED = "trophies_mapped"
ARENA_RACE_CONST6 = "const6"
ARENA_RACE_CONVENTIONS: tuple[str, ...] = (ARENA_RACE_TROPHIES_MAPPED, ARENA_RACE_CONST6)
DEFAULT_ARENA_RACE_CONVENTION = ARENA_RACE_TROPHIES_MAPPED

# What a report's trophy block MEANS under each ruler, emitted next to it for
# the same reason `FALLBACK_SEMANTICS_*` is: under versus rules there is no
# trophy race at all, and the block is None rather than a pile of zeros that
# would read as "this policy earned no trophies".
TROPHY_SEMANTICS_ARENA = "arena_10_trophies_completes_the_run"
TROPHY_SEMANTICS_VERSUS = "n/a_versus"


def _normalize_game_rules(game_rules: str | None) -> str:
    rules = str(game_rules or GAME_RULES_VERSUS).strip().lower()
    if rules not in GAME_RULES:
        raise ValueError(f"unknown_game_rules:{game_rules}:valid={list(GAME_RULES)}")
    return rules


def _game_rules_refusal(
    *,
    game_rules: str | None,
    opponent_mode: str | None,
    scoring: str | None,
) -> str | None:
    """The two combinations that would silently produce wrong numbers rather
    than fail -- returned as a message, or None when the combination is fine.

    Both are consequences of code exp13 W0a deliberately does NOT change:

    - `end_turn.py` only follows `meta.versus.current_opponent_participation_id`
      while resolving a VERSUS battle, so an arena-rules game cannot follow a
      chain. Left unchecked, arena rules + chain opponents would run as a
      random-opponent eval that still calls itself "chain".
    - `SearchRecommender`'s rollout continuations resolve their own turns
      through `play_out_game`'s DEFAULT (versus) rules, and score them with
      `_versus_win(lives, opp_lives)`. Under arena rules there is no opponent
      life bar, so that verdict reads as an instant win for every
      continuation. Wiring the scorer to the arena frame is exp13 W1a; until
      then this refuses instead of scoring nonsense.

    `scoring` is the leaf-scoring mode of the recommender that will play
    (`SearchRecommender.scoring`; None for anything without one).

    Two callers, two faces, ONE rule (exp13 W0b, codex review P2): the CLI
    face `_check_game_rules_compatibility` raises `SystemExit` before any
    checkpoint is loaded, and the API face `_assert_game_rules_supported`
    raises `ValueError` inside `run_versus_eval` / `play_one_game`, so a
    direct caller of either entry point cannot bypass the check by never
    going through `main()`.
    """
    if _normalize_game_rules(game_rules) != GAME_RULES_ARENA:
        return None
    if str(opponent_mode or OPPONENT_MODE_CHAIN).strip().lower() != OPPONENT_MODE_ARENA:
        return (
            "--game-rules arena requires --opponent-mode arena (game_rules=/opponent_mode= "
            "on the API): end_turn.py only follows a chain's forced pid under versus "
            "rules, so an arena-rules game cannot follow one -- it would silently become "
            "a random-opponent run labelled 'chain'."
        )
    if str(scoring or "").strip().lower() == SEARCH_SCORING_ROLLOUT:
        return (
            "--game-rules arena does not support --search-scoring rollout yet (exp13 W1a): "
            "SearchRecommender's rollout continuations resolve under VERSUS rules and score "
            "with _versus_win(lives, opp_lives), which has no meaning when there is no "
            "opponent life bar -- every continuation would read as an instant win."
        )
    return None


def _turn_mode_refusal(*, turn_mode: str | None, scoring: str | None) -> str | None:
    """Wd (exp13 A1): the leaf-scoring modes the honest frame does not have.

    A1 section 3 defines expectation scoring -- score a prefix that ends at a
    chance node as the mean of k imagined completions -- for the LEARNED
    LEAF only. A myopic or rollout leaf under this frame would still rank
    candidates by whatever the single proposal-stream roll happened to
    produce, and since each candidate's roll is no longer shared with play,
    the argmax would systematically pick the candidate that got lucky: a
    number that OVERSTATES the value of rolling, produced silently.

    So the combination is refused, exactly like the two `_game_rules_refusal`
    cases. `--recommender bc` (no candidate ranking at all, therefore no
    lucky-roll selection) and `--recommender llm` pass: `scoring` is None for
    anything without a leaf.

    Same two faces / one rule structure as `_game_rules_refusal`:
    `_check_turn_mode_compatibility` (CLI, `SystemExit`) and
    `_assert_game_rules_supported` (API, `ValueError`).
    """
    if not is_honest(turn_mode):
        return None
    leaf = str(scoring or "").strip().lower()
    if leaf and leaf != SEARCH_SCORING_VGAME:
        return (
            f"--turn-mode {TURN_MODE_SEGMENTED_HONEST} does not support --search-scoring "
            f"{leaf} (exp13 PLAN Amendment A1 section 3): expectation scoring over the k "
            f"imagined completions of a stochastic prefix is implemented for the "
            f"{SEARCH_SCORING_VGAME} leaf only. A {leaf} leaf under this frame would rank "
            "candidates on one imagined roll each and systematically pick whichever "
            "candidate rolled well, overstating the value of rolling."
        )
    return None


def _turn_mode_agreement_refusal(
    *, recommender_turn_mode: str | None, driver_turn_mode: str | None
) -> str | None:
    """We (exp13 W0b', codex finding 1): the recommender and the driver must
    be on the SAME frame.

    A1's frame is not one setting, it is two halves that only mean anything
    together. The driver owns segmentation (commit op by op, stop at the first
    realized `stochastic_reason`, re-search) and owns the imagined clone it
    hands over each segment; the recommender owns prefix dedup and expectation
    scoring over the k completions. Building a `SearchRecommender` with one
    `turn_mode` and running the driver with the other yields a run that is
    NEITHER frame, and until this check existed it did so silently:

    - determinized recommender under an honest driver: candidates are ranked
      as WHOLE turns, one imagined roll each -- the pick-the-lucky-roll bias
      A1 exists to remove -- while the driver commits only the prefix, so the
      post-boundary ops the winner was chosen FOR are thrown away and never
      re-searched. The report still says `turn_mode: segmented-honest`.
    - honest recommender under a determinized driver: `chain_preview` comes
      back TRUNCATED to the prefix (`_prefix_chain`), and the determinized
      driver commits one chain and resolves the turn -- so the agent stops
      playing mid-turn on every turn that crosses a chance node, with no
      re-search and no report field saying so.

    Refused at the API face, next to the rules and leaf checks, for the same
    reason they are: `run_versus_eval`/`play_one_game` are reusable entry
    points and a direct caller must not be able to assemble this by hand.
    Anything without a `turn_mode` attribute (a plain `BcRecommender`, an
    `LlmRecommender`) is unaffected -- it has no imagination of its own to
    disagree with, which is exactly why `--recommender bc` is honest under the
    honest driver on segmentation alone.

    `main()` passes `args.turn_mode` to both halves, so the CLI cannot build a
    mismatch and there is deliberately no second CLI face here.
    """
    if recommender_turn_mode is None:
        return None
    built = normalize_turn_mode(recommender_turn_mode)
    driving = normalize_turn_mode(driver_turn_mode)
    if built == driving:
        return None
    return (
        f"turn_mode_mismatch:recommender={built}:driver={driving} (exp13 PLAN "
        "Amendment A1): the recommender was CONSTRUCTED for one frame and the driver "
        "is RUNNING the other. Segmentation lives in the driver and expectation "
        "scoring lives in the recommender, so a mismatched pair is neither frame -- "
        "pass the same turn_mode to both."
    )


def _assert_game_rules_supported(
    recommender: Any,
    *,
    game_rules: str | None,
    opponent_mode: str | None,
    turn_mode: str | None = DEFAULT_TURN_MODE,
) -> None:
    """API face of `_game_rules_refusal` + `_turn_mode_refusal` +
    `_turn_mode_agreement_refusal` -- raises `ValueError` (this module's
    established library-level error, as in `unknown_opponent_mode`).

    The scoring mode and the frame are read off the recommender the same
    duck-typed way `play_one_game` reads `set_decision_context`: a plain
    `BcRecommender` has neither `.scoring` nor `.turn_mode`, and is unaffected
    by both checks.

    `turn_mode` (Wd, exp13 A1) defaults to the determinized frame, so every
    pre-A1 caller of this function is unaffected.
    """
    scoring = getattr(recommender, "scoring", None)
    message = _game_rules_refusal(
        game_rules=game_rules,
        opponent_mode=opponent_mode,
        scoring=scoring,
    )
    if message is None:
        message = _turn_mode_refusal(turn_mode=turn_mode, scoring=scoring)
    if message is None:
        message = _turn_mode_agreement_refusal(
            recommender_turn_mode=getattr(recommender, "turn_mode", None),
            driver_turn_mode=turn_mode,
        )
    if message is not None:
        raise ValueError(message)


def _normalize_arena_race_convention(convention: str | None) -> str:
    name = str(convention or DEFAULT_ARENA_RACE_CONVENTION).strip().lower()
    if name not in ARENA_RACE_CONVENTIONS:
        raise ValueError(
            f"unknown_arena_race_convention:{convention}:valid={list(ARENA_RACE_CONVENTIONS)}"
        )
    return name


def _arena_race_opp_lives(trophies: int, convention: str) -> int:
    """PLAN.md D2: what fills the bypass's `opp_lives` slot under arena rules.

    Pure arithmetic over the trophy count, so the convention is one
    expression in one place -- the driver, the recorders and (through the
    state field below) the learned leaf all read the same number.
    """
    if convention == ARENA_RACE_CONST6:
        return VERSUS_START_LIVES
    # `trophies` is capped at MAX_TROPHIES by the lives helper, so this is
    # already non-negative; the max() is a guard, not a rule.
    return max(0, min(VERSUS_START_LIVES, MAX_TROPHIES - int(trophies)))


def _apply_arena_race_context(state: dict[str, Any], *, convention: str) -> int:
    """Write this turn's D2 race convention onto `meta.versus.opponent_lives`
    and return it.

    That field is where `SearchRecommender._race_scalars` (and therefore the
    V bypass) reads `opp_lives` from, and where this driver reads
    `pre_turn_opp_lives` for the afterstate/teacher recorders -- so writing
    the convention THERE is what makes "the number V scored with" and "the
    number the dataset stores" the same object, instead of two definitions
    that can drift.

    Under arena rules the field is a RACE FEATURE, never a life total:
    `end_turn.py`'s arena branch neither reads nor writes it (only its
    `mode == "versus"` branches touch `meta.versus`), so nothing downstream
    can mistake it for one.
    """
    trophies = int(state.get("trophies", 0) or 0)
    opp_lives = _arena_race_opp_lives(trophies, convention)
    meta = state.setdefault("meta", {})
    versus_meta = meta.setdefault("versus", {})
    versus_meta["opponent_lives"] = int(opp_lives)
    return int(opp_lives)


def _is_completed_end_reason(end_reason: str | None) -> bool:
    """True iff `end_reason` is a genuine completed-game verdict (see
    `COMPLETED_END_REASONS`), not an infrastructure failure."""
    return str(end_reason) in COMPLETED_END_REASONS


def _decode_failure_reason(rec: dict[str, Any] | None, stop_reason: str | None) -> str:
    """WHY the recommender refused to act, as the suffix of `decode_failed:*`.

    `decode_failed` on its own says a game died and nothing else. exp13 W1c
    hit one in 180 search games and the reason had to be dug out by hand from
    the recorded board, which is not a thing a run's artifact should require:
    the reasoned form this builds is the one the rest of the codebase already
    assumes exists (`_new_game_state`'s own comment names
    `decode_failed:legal_mask_failed`, and `play_web/ai_worker.py` emits
    `decode_failed:<stop_reason>` for the duel).

    `error` is read FIRST and `diagnostics.stop_reason` is the fallback.
    `BcRecommender` sets `error` to exactly the stop reason on every failure,
    so the BC path yields that anticipated form either way; but
    `SearchRecommender`'s hard-failure shape carries
    `search_greedy_call_failed:<exc>` in `error` and an EMPTY diagnostics
    block, so reading the stop reason first would throw away the only thing
    that says what happened. Collapsed to one line and truncated, because
    this string ends up in `end_reason_distribution` keys.
    """
    for value in ((rec or {}).get("error"), stop_reason):
        text = " ".join(str(value or "").split())
        if text:
            return text[:120]
    return "unknown"


def _is_pool_exhausted_end_reason(end_reason: str | None) -> bool:
    """True iff this `end_turn_failed:*` reason is "the arena pool holds no
    candidate at that turn" rather than a real infrastructure failure.

    exp13 F1. The reason string this matches is built in three layers:
    `chain_snapshot.sample_random_with_rng` -> `no_snapshot_opponent_for_turn:N`,
    `end_turn.resolve_end_turn_with_sampled_battle` -> `opponent_sampling_failed:...`
    (or `opponent_sampling_failed:chain_then_random:...`), and
    `_resolve_versus_turn` -> `end_turn_failed:...`. Only the innermost token
    is load-bearing, so only it is matched; the two middle prefixes both pass.
    """
    return POOL_EXHAUSTED_ERROR_TOKEN in str(end_reason or "")


def effective_arena_max_turn(
    max_turn: int,
    arena_source: "ChainSnapshotSource | None",
    *,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
) -> int:
    """The turn cap a game against `arena_source` can actually be played to.

    exp13 F1 (Ruihan's ruling, 2026-08-07). The configured cap is the GAME's
    rule (30 in `PLAN.md`); the pool's depth is the HARNESS's rule, and the
    binding one is whichever is smaller. Returning `min` of the two means a
    game that would have outlived the pool now ends as `turn_cap` on its last
    servable turn, which already keeps its trophies, instead of dying as
    `end_turn_failed:opponent_sampling_failed:...` on the first unservable
    one, which `validate_exp13_report.py` scores 0.

    The depth is read off the pool at run time (`max_turn_with_candidates`),
    never hard-coded: it is 21 on the v45 val pool and 25 on the v45 train
    pool as measured on 2026-08-07, and it moves with the snapshot, the
    split, the opponent pack and the rank bound.

    Only `opponent_mode="arena"` is clamped. Under chain mode the opponent
    comes from the FOLLOWED chain and running out is the normal, designed
    event that the random fallback exists to absorb, so a pool depth there is
    not a turn cap. `arena_source is None` (the chain-mode default) and an
    empty pool both return `max_turn` unchanged.
    """
    if opponent_mode != OPPONENT_MODE_ARENA or arena_source is None:
        return int(max_turn)
    depth = arena_source.max_turn_with_candidates
    if depth is None:
        return int(max_turn)
    return min(int(max_turn), int(depth))


# `ChainSnapshotSource` (formerly defined inline here as `SnapshotOpponentSource`)
# now lives in `sap_ppo.opponents.chain_snapshot` (exp09 W5 P0) so
# `train/runtime.py::build_opponent_provider`'s new `chain_snapshot` mode can
# share it -- see that module's docstring for the extraction rationale and the
# held-out split it adds. Imported above; no behavior change for this eval
# driver (constructed the same way, same three methods, `split=None` default).


def _versus_meta(state: dict[str, Any]) -> dict[str, Any]:
    meta = state.get("meta")
    return meta.get("versus", {}) if isinstance(meta, dict) and isinstance(meta.get("versus"), dict) else {}


def _assert_fixture_shape(state: dict[str, Any]) -> None:
    """Verify the turn-1 versus fixture actually matches what this driver requires."""
    versus = _versus_meta(state)
    meta = state.get("meta") if isinstance(state.get("meta"), dict) else {}
    problems = []
    if int(state.get("turn", -1)) != 1:
        problems.append(f"turn={state.get('turn')}")
    if int(state.get("lives", -1)) != 6:
        problems.append(f"lives={state.get('lives')}")
    if str(meta.get("game_mode", "")).strip().lower() != "versus":
        problems.append(f"game_mode={meta.get('game_mode')}")
    if int(versus.get("opponent_lives", -1)) != 6:
        problems.append(f"opponent_lives={versus.get('opponent_lives')}")
    if problems:
        raise ValueError(f"fixture_shape_mismatch:{FIXTURE_PATH}:{','.join(problems)}")


def _new_game_state(
    fixture_initial_state: dict[str, Any],
    followed_pid: str,
    *,
    engine_seed: int,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
) -> dict[str, Any]:
    """Deep-copy the fixture and wire up this game's followed opponent + versus life state.

    Also sets `meta.seed_known=True` + a concrete `meta.seed`. The fixture ships
    `seed_known=False`, which makes `engine._rng_from_state` seed every RNG draw
    (ROLL's shop regen, food-target selection, ...) from OS entropy instead of
    from the state. That is harmless for a single one-shot decode, but this
    driver's per-turn loop calls `engine_step` TWICE over the same chain of
    non-END_TURN ops: once implicitly inside `BcRecommender.recommend()`'s own
    internal decode walk, and once explicitly here to replay `chain_preview`
    and reconstruct the pre-END_TURN board (see module docstring). With
    `seed_known=False` those two walks draw independent OS entropy at any
    ROLL, so they can land on different shops -- and a later shop-position
    action in the SAME chain (BUY_PET/BUY_COMBINE/BUY_FOOD/FREEZE/UNFREEZE
    after the ROLL) then replays against the wrong shop and can turn illegal.
    Verified directly: an un-seeded game hit exactly this
    (`chain_replay_diverged:BUY_COMBINE:illegal_on_replay` on turn 3, chain
    `FREEZE, BUY_COMBINE, BUY_FOOD, BUY_COMBINE, ROLL, FREEZE, FREEZE`); the
    identical game with `seed_known=True` replayed all 9 turns (every one
    ROLL-then-shop-op) with zero divergence. `eval_tempo_planner.py` hits the
    same requirement for its own chain-replay (`_simulate_state_after_chain`)
    and already sets `seed_known=True` + a concrete seed for exactly this
    reason (see its `_run_one_case`/`_run_one_start_case` `base_state`
    construction) -- mirrored here rather than reinvented.
    `engine._reseed_meta` chains the seed forward deterministically turn to
    turn once `seed_known` is True, so setting it once at game start covers
    the whole game.

    `game_rules` (Wb, exp13 W0a, default `versus` = byte-identical to every
    run before it existed): under `arena` the versus block below is written
    first and then OVERRIDDEN -- deliberately in that order, so the versus
    path is literally the unchanged code. The arena overrides are exactly
    PLAN.md D1's start state: 5 lives, 0 trophies, `meta.game_mode="arena"`
    (which is what routes `end_turn.py` to its trophy branch), and the D2
    race convention written onto `meta.versus.opponent_lives` (see
    `_apply_arena_race_context` -- under arena rules that field is a race
    FEATURE, not a life total). `current_opponent_participation_id` is still
    seeded so the state shape is identical under both rulers; arena rules
    require `--opponent-mode arena`, whose loop pops it every turn.
    `meta.game_rules` is set under arena ONLY, so `schemas/state_v1.json` can
    keep asserting an EXACT turn-1 life total per ruler -- see the comment at
    that assignment for why `game_mode` could not be the discriminator.
    """
    state = copy.deepcopy(fixture_initial_state)
    state["turn"] = 1
    state["lives"] = 6
    meta = state.setdefault("meta", {})
    meta["game_mode"] = "versus"
    meta["seed_known"] = True
    meta["seed"] = int(engine_seed)
    versus = meta.setdefault("versus", {})
    versus["opponent_lives"] = 6
    versus["current_opponent_participation_id"] = followed_pid
    if _normalize_game_rules(game_rules) == GAME_RULES_ARENA:
        state["lives"] = ARENA_START_LIVES
        state["trophies"] = 0
        meta["game_mode"] = GAME_MODE_ARENA
        # WHICH arena. `game_mode` alone is ambiguous: the RL training env's
        # episodes are also `game_mode="arena"` and legitimately start on 6
        # lives with a 7-trophy target (`train/env.py::_is_done`), while THIS
        # ruler starts on 5 with a 10-trophy target. `schemas/state_v1.json`
        # needs to tell them apart to keep its turn-1 start-of-game invariant
        # exact (turn 1 -> exactly 5 lives here, exactly 6 everywhere else)
        # rather than weakened to "5 or 6". Without it every arena game died
        # on turn 1 with `decode_failed:legal_mask_failed` -- `legal_actions`
        # validates the state. Written under arena rules ONLY, so a versus
        # state is byte-identical to before this flag existed.
        meta["game_rules"] = GAME_RULES_ARENA
        _apply_arena_race_context(
            state, convention=_normalize_arena_race_convention(arena_race_convention)
        )
    return state


def _search_width_counts(turn_entry: dict[str, Any], key: str) -> list[int]:
    """This turn's per-DECISION candidate-width counts (`search_n_generated`
    / `search_n_dedup`), for the exp12 W2c width telemetry.

    Under the determinized frame a turn IS one decision, and this returns the
    single pre-A1 value (or nothing, for a recommender that reports no
    counts) -- byte-identical.

    Wd (exp13 A1): under the honest frame a turn is SEVERAL decisions, so the
    counts are read off the segment records instead. Without this the width
    block would quote whichever segment happened to be last and silently
    describe a fraction of the run's decisions -- on the W0a' smoke, 16 of
    88 -- while still printing `searched_turns` as if it covered them all.
    The field names keep saying `by_turn` for JSONL continuity; what they
    count is decisions.
    """
    segments = turn_entry.get("segments")
    if segments is None:
        if turn_entry.get("search_used") and turn_entry.get(key) is not None:
            return [int(turn_entry[key])]
        return []
    return [
        int(segment[key])
        for segment in segments
        if segment.get("search_used") and segment.get(key) is not None
    ]


def _turn_record(
    *,
    turn: int,
    chain_types: list[str],
    stop_reason: str,
    outcome: str | None,
    opponent_pets: Any,
    lives: int,
    opp_lives: int,
    detail: dict[str, Any] | None = None,
    search_used: bool = False,
    search_n_generated: int | None = None,
    search_n_dedup: int | None = None,
    search_diagnostics: dict[str, Any] | None = None,
    segments: list[dict[str, Any]] | None = None,
    segments_capped: bool = False,
    cat_trigger_audit: dict[str, int] | None = None,
) -> dict[str, Any]:
    record = {
        "turn": int(turn),
        "chain_types": list(chain_types),
        "stop_reason": str(stop_reason),
        "outcome": outcome,
        "opponent_pets": opponent_pets,
        "lives": int(lives),
        "opp_lives": int(opp_lives),
        # None unless `play_one_game(..., capture_detail=True)` (We2 render
        # path); always present (rather than an absent key) so every
        # `per_turn` record has the same schema regardless of mode.
        "detail": detail,
        # We3 (exp09 W6a): whether `SearchRecommender` actually searched
        # this turn (False for every turn under plain `--recommender bc`,
        # since that recommender's result dict has no `search_used` key at
        # all -- see module docstring).
        "search_used": bool(search_used),
        # exp12 W2c (width curve): the requested candidate width and the
        # number of DISTINCT end boards those candidates collapsed to on
        # this turn. None whenever the recommender does not report them
        # (plain `--recommender bc`, or an older captured run), so a width
        # arm's report can never silently read a missing count as zero.
        "search_n_generated": (int(search_n_generated) if search_n_generated is not None else None),
        "search_n_dedup": (int(search_n_dedup) if search_n_dedup is not None else None),
        # We5 (exp09 W1): free-form diagnostics `SearchRecommender.recommend`
        # optionally attaches (`rec.get("search_diagnostics")`) -- myopic
        # scores, the rollout shortlist + its scores, which index each stage
        # picked. None whenever the recommender didn't set it (plain
        # `--recommender bc`, or `--search-scoring myopic`, which does not
        # populate this key -- see search_recommender.py).
        "search_diagnostics": search_diagnostics,
        # exp19 (PLAN_W1 step 6): this turn's four-condition Cat trigger
        # opportunity counts, from `tools/cat_trigger_audit.py`. Always
        # present (all-zero when no Cat bought a food), because the whole
        # point is that a zero is readable: it separates "the fix changed
        # nothing" from "the situation never arose", which no count of
        # divergences can do.
        "cat_trigger_audit": dict(cat_trigger_audit or _cat_audit_empty()),
    }
    # Wd (exp13 A1): segment accounting, added ONLY under the honest frame
    # (`segments is None` under the determinized default). A determinized
    # turn is one segment by construction, so carrying the block there would
    # be pure schema churn -- and every pre-A1 per-turn record would stop
    # matching the byte-identity pin. Same arena-only-tag discipline
    # `_new_game_state` uses for `meta.game_rules`.
    if segments is not None:
        record["segments"] = segments
        record["n_segments"] = len(segments)
        record["segments_capped"] = bool(segments_capped)
    return record


def _build_turn_detail(
    *,
    state_before: dict[str, Any] | None,
    board_pre_battle: dict[str, Any] | None,
    parsed_state: dict[str, Any] | None,
    battle: dict[str, Any] | None,
    state_after: dict[str, Any] | None,
    full: bool,
) -> dict[str, Any]:
    """Per-turn board/battle snapshot. `full` is the `--render-games` form (We2).

    exp19 (PLAN_W1 step 7) made the LIGHT form unconditional: every metric row
    now carries `board_pre_battle` and nothing else, and `full=True` (i.e.
    `play_one_game(..., capture_detail=True)`) adds the other four. `detail`
    used to be null on the metrics path entirely, which is why every
    agent-side behavioural number in this campaign rests on the 12 games that
    happened to be gallery-captured: W0-D read `detail` on 0 of 10,890 turn
    rows, and W0-D2 priced this exact change as the one that takes the agent
    side from 12 clusters to the whole run. One board per turn is the cheap
    half of the capture; the four heavy fields stay behind the flag.

    Every argument is already computed by the verified per-turn loop in
    `play_one_game` -- this function does not compute anything new, it only
    deep-copies for safekeeping across the rest of the game's turns:
    - `state_before`: the engine state at the TOP of the turn, before
      `bc.recommend` was even called (for the "Start of turn" render row).
    - `board_pre_battle`: `resolved["battle_state"]`, the EXACT board
      `resolve_end_turn_with_sampled_battle` fed the battle oracle as
      `config["playerPets"]` -- not the locally re-derived `board` var,
      which can differ from it by whatever `resolve_end_turn_pre_battle`
      normalizes internally (see that function's own pre-battle step).
    - `parsed_state`: `resolved["parsed_state"]` as-is (never flipped, see
      `ChainSnapshotSource`'s docstring) -- `["opponentPets"]` is the
      duel opponent's board the BC actually fought this turn.
    - `battle`: `resolved["battle"]`, the oracle payload (`outcome` +
      `calculator_link` already computed against the real fought boards).
    - `state_after`: the engine state once this turn's battle + life
      bookkeeping + turn-3 recovery are all applied (i.e. `state` right
      after the caller's `state = resolved["transition"]["state_after"]`).
    """
    return {
        "state_before": (
            copy.deepcopy(state_before) if full and isinstance(state_before, dict) else None
        ),
        # The one field the light form carries: the EXACT board that was
        # fought, on every row, whatever mode the run is in.
        "board_pre_battle": copy.deepcopy(board_pre_battle) if isinstance(board_pre_battle, dict) else None,
        "parsed_state": (
            copy.deepcopy(parsed_state) if full and isinstance(parsed_state, dict) else None
        ),
        "battle": copy.deepcopy(battle) if full and isinstance(battle, dict) else None,
        "state_after": (
            copy.deepcopy(state_after) if full and isinstance(state_after, dict) else None
        ),
    }


def _teacher_self_check(record: dict[str, Any], *, played_signature: str) -> dict[str, Any]:
    """Wa (exp12 route a, wave A0): the four per-decision invariants, computed
    from the record the recommender just produced.

    Written into the row for triage, NOT trusted as the gate --
    `tools/check_teacher_record.py` re-derives every one of them off the file
    so the recorder cannot certify itself (that tool additionally runs the
    D1-style re-encode spot check, which needs the encoder and so does not
    belong on the hot path).

    - `chosen_score_matches`: the chosen candidate's own recorded
      `teacher_score` IS the decision-level `chosen_teacher_score` the driver
      acted on, bit-exact (catches an index misalignment between the deduped
      candidate list and the shortlist).
    - `argmax_matches`: recomputing the argmax over the rolled-out candidates
      with the same `(rollout score, myopic score)` key `_search_rollout`
      uses lands on `chosen_index`.
    - `group_complete`: one entry per scored candidate, indices 0..n-1 once
      each, exactly one `chosen`, and the shortlist fully rolled out.
    - `chosen_board_matches`: the chosen candidate's afterstate signature is
      the board the driver actually replayed and is about to fight.
    """
    cands = record.get("candidates") or []
    n = int(record.get("n_candidates") or 0)
    chosen_index = record.get("chosen_index")
    rolled = [c for c in cands if c.get("rolled_out")]

    indices_ok = sorted(int(c.get("index", -1)) for c in cands) == list(range(len(cands)))
    group_complete = bool(
        len(cands) == n
        and n > 0
        and indices_ok
        and sum(1 for c in cands if c.get("chosen")) == 1
        and len(rolled) == int(record.get("shortlist_size") or -1)
    )

    chosen = None
    if isinstance(chosen_index, int) and 0 <= chosen_index < len(cands):
        chosen = cands[chosen_index]
    chosen_score_matches = bool(
        chosen is not None
        and chosen.get("teacher_score") is not None
        and float(chosen["teacher_score"]) == float(record.get("chosen_teacher_score"))
        and bool(chosen.get("chosen"))
    )

    argmax_matches = False
    if rolled:
        best = max(rolled, key=lambda c: (float(c["teacher_score"]), float(c["myopic_score"])))
        argmax_matches = int(best["index"]) == chosen_index

    chosen_board_matches = bool(
        chosen is not None and str(chosen.get("signature") or "") == str(played_signature)
    )

    return {
        "chosen_score_matches": chosen_score_matches,
        "argmax_matches": argmax_matches,
        "group_complete": group_complete,
        "chosen_board_matches": chosen_board_matches,
    }


def _versus_win(lives: int, opp_lives: int) -> bool:
    """Versus win rule, shared so it is spelled out in exactly one place:
    the driver's `play_out_game` tail AND `search_recommender.py`'s rollout
    scorer (which needs the identical rule for a continuation that ends the
    same turn its shortlisted candidate's own battle resolves)."""
    return bool(int(opp_lives) <= 0 and int(lives) > 0)


def _arena_win(lives: int, trophies: int) -> bool:
    """Wb (exp13 W0a) arena win rule, PLAN.md D1: 10 trophies COMPLETES the
    run, and completing it is the win. Spelled out next to `_versus_win` so
    the two rulers' verdicts sit side by side, and shared with
    `human_vs_pool_reference.py` for the same anti-drift reason."""
    return bool(int(trophies) >= MAX_TROPHIES and int(lives) > 0)


def _resolve_versus_turn(
    board: dict[str, Any],
    *,
    sample_for_pid_fn: Callable[[str, int], dict[str, Any]],
    sample_random_fn: Callable[[int], dict[str, Any]],
    parse_cache: dict[str, Any] | None = None,
    simulation_count: int = 1,
    battle_logs_enabled: bool = False,
    game_mode: str = GAME_MODE_VERSUS,
    max_lives: int | None = None,
) -> dict[str, Any]:
    """Resolve END_TURN on an ALREADY-DECODED-AND-APPLIED `board` (this
    turn's shop-phase actions are already committed) via
    `end_turn.py::resolve_end_turn_with_sampled_battle` -- the single source
    of truth for versus battle resolution + lives/turn-3-recovery
    bookkeeping. This helper only adds the "read the result into a
    state_after that carries `last_opponent_team` forward, plus whether a
    random fallback fired" glue `play_out_game`'s per-turn loop needs, ONCE,
    so it is identical regardless of which caller invokes it: the real
    per-game loop (`play_out_game`, `simulation_count=1`, its default), or
    `search_recommender.py`'s rollout scorer resolving a shortlisted
    candidate's OWN turn (`simulation_count=--rollout-ksim`; every
    subsequent turn of that same continuation calls `play_out_game`, which
    reverts to the default).

    `game_mode` / `max_lives` (Wb, exp13 W0a, defaults `"versus"` / None =
    byte-identical to every prior caller): forwarded to
    `resolve_end_turn_with_sampled_battle`, which is what selects its
    (already existing) trophy branch and the turn-3 heal cap. Under
    `game_mode="arena"` that resolver never reads
    `meta.versus.current_opponent_participation_id`, i.e. no chain is
    followed -- which is why arena RULES require arena OPPONENTS (see the
    module docstring). The function name keeps its `_versus_` prefix because
    the exp12-era rollout scorer in `search_recommender.py` calls it by that
    name; what it resolves is "this driver's turn", under either ruler.

    `simulation_count` (We5, exp09 W1, default 1 -- matches every existing
    caller byte-for-byte): forwarded to `resolve_end_turn_with_sampled_battle`.
    A count > 1 still resolves to ONE discrete `outcome` (majority vote over
    the k sims, see that function's own docstring), so the win/loss/draw
    bookkeeping below is unaffected by the count.

    Returns:
    - `{"ok": False, "end_reason": "end_turn_failed:<error>", "resolved": <envelope>}`
    - `{"ok": True, "state_after": dict, "outcome": str, "parsed_state":
      dict | None, "battle": dict, "battle_state": dict (the board actually
      fought, == `board`), "fallback_used": bool, "resolved": <envelope>}`
    """
    resolve_kwargs: dict[str, Any] = {
        "game_mode": game_mode,
        "sample_for_pid_fn": sample_for_pid_fn,
        "sample_random_fn": sample_random_fn,
        "parse_cache": parse_cache,
        "simulation_count": simulation_count,
    }
    if battle_logs_enabled:
        resolve_kwargs["battle_logs_enabled"] = True
    if max_lives is not None:
        resolve_kwargs["max_lives"] = int(max_lives)
    resolved = resolve_end_turn_with_sampled_battle(board, **resolve_kwargs)
    if not resolved.get("ok"):
        return {
            "ok": False,
            "end_reason": f"{END_REASON_END_TURN_FAILED}:{resolved.get('error')}",
            "resolved": resolved,
        }

    battle = resolved.get("battle") if isinstance(resolved.get("battle"), dict) else {}
    outcome = str(battle.get("outcome", "unknown"))
    parsed_state = resolved.get("parsed_state")

    state_after = resolved["transition"]["state_after"]
    if isinstance(parsed_state, dict):
        _set_last_opponent_team(state_after, parsed_state.get("opponentPets"))

    engine_notes = resolved["transition"].get("engine_notes") or []
    fallback_used = "end_turn_versus_chain_fallback_random" in engine_notes

    return {
        "ok": True,
        "state_after": state_after,
        "outcome": outcome,
        "parsed_state": parsed_state,
        "battle": battle,
        "battle_state": resolved.get("battle_state"),
        "fallback_used": fallback_used,
        "resolved": resolved,
    }


def play_out_game(
    state: dict[str, Any],
    bc: BcRecommender | SearchRecommender,
    *,
    initial_pid: str,
    sample_for_pid_fn: Callable[[str, int], dict[str, Any]],
    sample_random_fn: Callable[[int], dict[str, Any]],
    max_turn: int = DEFAULT_MAX_TURN,
    parse_cache: dict[str, Any] | None = None,
    capture_detail: bool = False,
    on_turn: Callable[[dict[str, Any]], None] | None = None,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    afterstate_sink: Callable[[dict[str, Any]], None] | None = None,
    teacher_sink: Callable[[dict[str, Any]], None] | None = None,
    segment_sink: Callable[[dict[str, Any]], None] | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> dict[str, Any]:
    """Play `state` to a lives verdict (or the turn cap), one turn at a
    time: decode (`bc.recommend`) -> replay non-END_TURN ops through the
    engine -> resolve END_TURN (`_resolve_versus_turn`, `simulation_count=1`,
    the driver's normal single-draw mechanics) -> bookkeeping -> repeat.

    We5 (exp09 W1): factored out of `play_one_game` (which now just builds
    the turn-1 state and delegates here) so this is the SINGLE place "what
    happens turn to turn in a versus game" is implemented -- both the real
    per-game driver loop and `search_recommender.py`'s rollout scorer
    (`--search-scoring rollout`, simulating a shortlisted candidate's
    rest-of-game from whatever state its own already-resolved current-turn
    battle reached) call this, so the rules can never drift between the two.
    This is a PURE code-motion refactor of We1-We4's `play_one_game` body
    (see that function's own note); every line below is unchanged from
    before this split, just parameterized instead of closing over
    `opp_source`/`game_index`/`seed` directly.

    `sample_for_pid_fn`/`sample_random_fn` (plain callables, not an
    `opp_source` object) so a caller can substitute an ISOLATED fallback
    sampler without mutating a shared `ChainSnapshotSource`'s own
    `_random_rng` -- required by the rollout scorer, whose throwaway
    continuations must never perturb the real game's own later (for-real)
    draws from that same shared source (see
    `chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`).

    `initial_pid`: the fallback value for `final_followed_pid` if
    `state`'s own `meta.versus.current_opponent_participation_id` is ever
    absent (defensive only -- `_new_game_state` always sets it, so this is
    never hit via `play_one_game`; the rollout scorer passes whatever pid
    `state` is already following when the continuation starts).

    `capture_detail` (We2, default False): see `play_one_game`'s docstring
    -- unchanged meaning, just threaded through as a parameter now.

    `on_turn` (We6, exp09 W2, default None): see module docstring's "We6
    addition" section for the full per-turn payload shape. Independent of
    `capture_detail` (a caller can use either, both, or neither).

    `opponent_mode` (We8, exp12 W0, default `"chain"` = byte-identical to
    every run before this parameter existed): under `"arena"` the loop pops
    `meta.versus.current_opponent_participation_id` off `state` at the top
    of EVERY iteration, before anything else reads it. That single line is
    the whole mechanism: with no forced pid on the state,
    `end_turn.py::resolve_end_turn_with_sampled_battle` takes its existing
    `sample_random_fn(turn)` branch instead of `sample_for_pid_fn`, so the
    caller's sampler (a per-game-seeded `sample_random_with_rng` closure --
    see `play_one_game`) decides the opponent fresh each turn. The pop has
    to be per-iteration, not once up front, because `end_turn.py` WRITES
    the pid it just used back onto the resulting state after every resolved
    turn; leaving that in place would silently re-chain from turn 2 on.

    `afterstate_sink` (We9, exp12 W1'a, default None): see module docstring's
    "We9 addition" for the full row schema and the anti-buffering rationale.
    Independent of `capture_detail`/`on_turn` (a caller can use any subset).
    Called once per turn that reaches `_resolve_versus_turn` (a per-turn row,
    game_index NOT included -- the caller injects it) and once more, right
    before this function returns, with the final outcome row (also without
    game_index).

    `teacher_sink` (Wa, exp12 route a, default None): see module docstring's
    "Wa addition". Called at the SAME point as `afterstate_sink` -- after
    this turn's chosen chain has fully replayed, before its battle resolves --
    but only on turns where the recommender actually produced a
    `teacher_record` (i.e. rollout-scored search turns; turn-1 skips and
    myopic turns produce none and are correctly absent). Also called once
    with the per-game final row, for the same join-against-outcomes reason
    the We9 recorder emits one.

    `game_rules` / `arena_race_convention` (Wb, exp13 W0a, defaults
    `"versus"` / `"trophies_mapped"` = byte-identical to every run before
    these existed): see the module docstring's "Wb addition". Under `arena`
    three things change and nothing else does --

    1. the battle resolves with `game_mode="arena"` and the 5-life heal cap,
       so `versus_lives.py`'s existing arena branch does the bookkeeping
       (win -> trophy, loss -> life) instead of the versus one;
    2. the D2 race convention is (re)written onto the state at the TOP of
       every turn, before `bc.recommend` sees it and before this loop reads
       `pre_turn_opp_lives` off it, so the leaf, the recorders and the report
       all carry one number with one definition;
    3. the terminal rule is arena's: 10 trophies -> win, 0 lives -> loss,
       `max_turn` -> non-win. `TrainingEnv._is_done` is deliberately NOT
       consulted under arena rules -- its arena branch ends the episode at
       SEVEN trophies (the RL env's own frame, `train/env.py`), which is not
       this ruler.

    `turn_mode` (Wd, exp13 A1, default `whole-determinized` = byte-identical
    to every run before it existed): see the module docstring's "Wd
    addition". Under `segmented-honest` the decode/replay body becomes a
    SEGMENT loop -- the recommender is handed an imagined clone (stream S)
    instead of the real board, the chosen chain is committed op by op
    against the real board (stream P), and the loop re-enters from the
    observed state after any op whose transition reports a
    `stochastic_reason`. Everything else in this function -- battle
    resolution, lives/trophy bookkeeping, the terminal rule, every sink --
    is untouched, and every key the honest frame adds is emitted only under
    it.

    Returns the SAME keys `play_one_game` returns minus the game-identity
    fields (`game_index`/`followed_pid`/`initial_followed_pid`, which only
    that function's caller knows): `final_followed_pid`, `win`,
    `player_lives`, `opponent_lives`, `turns_survived`, `end_reason`,
    `num_fallbacks`, `fallback_turns`, `search_used_turns`, `stop_reasons`,
    `per_turn`, plus We8's `opponent_mode` / `fallback_semantics` and Wb's
    `game_rules` / `arena_race_convention` / `trophies`.
    """
    # Call-time import: segment recording is experiment tooling, not part of the public release.
    from .exp13_w1_records import SegmentRecorder

    rules = _normalize_game_rules(game_rules)
    race_convention = _normalize_arena_race_convention(arena_race_convention)
    is_arena = rules == GAME_RULES_ARENA
    # None under versus, so `_resolve_versus_turn`'s forwarded kwargs are
    # literally the pre-exp13 set (see that function).
    engine_game_mode = GAME_MODE_ARENA if is_arena else GAME_MODE_VERSUS
    heal_cap = ARENA_MAX_LIVES if is_arena else None
    per_turn: list[dict[str, Any]] = []
    stop_reason_counts: Counter[str] = Counter()
    fallback_turns: list[int] = []
    turns_completed = 0
    end_reason: str | None = None
    # exp13 F1: the raw `end_turn_failed:...` string and the turn it fired at,
    # kept whenever `end_reason` is re-labelled to
    # `END_REASON_OPPONENT_POOL_EXHAUSTED` below, so re-labelling never costs
    # the diagnosis. Stays None on every other path and is emitted only when
    # set, so no existing row gains a key.
    pool_exhausted_detail: dict[str, Any] | None = None
    # We9 (exp12 W1'a): pre-battle cumulative win count, Vic semantics
    # (RESULTS_W1 finding 2) -- turn N's afterstate row records wins_so_far
    # BEFORE turn N's own battle is resolved, i.e. wins from turns 1..N-1
    # only. Incremented below only after a turn resolves ok with outcome
    # "win" (never on a loss/draw/infra-failure turn).
    wins_so_far = 0
    # exp12 W2 (wave A4): resolved once, called every turn -- see the call
    # site below and `SearchRecommender.set_race_context`.
    set_race_context = getattr(bc, "set_race_context", None)
    if not callable(set_race_context):
        set_race_context = None
    # Wd (exp13 A1): the honest frame, resolved once. `game_engine_seed` is
    # read HERE, off the state as handed in (turn 1 via `play_one_game`),
    # BEFORE play chains `meta.seed` forward -- so it is the game's own
    # engine seed, a pure function of `(--seed, game_index)`, and every
    # imagination key derives from it. Deriving keys from the state's
    # CURRENT seed instead would key imagination on the play stream's
    # position, which is exactly the coupling A1 removes.
    turn_mode = normalize_turn_mode(turn_mode)
    honest = is_honest(turn_mode)
    game_engine_seed = read_engine_seed(state)
    set_imagination_context = getattr(bc, "set_imagination_context", None)
    if not callable(set_imagination_context):
        set_imagination_context = None

    while True:
        # We8 (exp12 W0): arena ruler == "no forced pid, ever". See this
        # function's docstring for why this must run every iteration.
        if opponent_mode == OPPONENT_MODE_ARENA:
            live_versus_meta = _versus_meta(state)
            live_versus_meta.pop("current_opponent_participation_id", None)

        # Wb (exp13 W0a, PLAN.md D2): the driver-true race block. Under arena
        # rules `wins` IS the trophy count and `opp_lives` is the pinned
        # convention; under versus rules this is exactly the pre-exp13
        # `wins_so_far` and the untouched engine life race. Refreshed here,
        # at the top of the turn, so every consumer downstream -- the V
        # bypass via `set_race_context` + the state, `pre_turn_opp_lives`,
        # the afterstate/teacher rows -- reads ONE number.
        if is_arena:
            race_wins = int(state.get("trophies", 0) or 0)
            _apply_arena_race_context(state, convention=race_convention)
        else:
            race_wins = wins_so_far

        # `capture_detail`-only: the board as of the TOP of this turn, before
        # `bc.recommend` acts on it -- the "Start of turn" render row. Cheap
        # to skip entirely in the default (fast, N~=300-game) path.
        state_before_snapshot = copy.deepcopy(state) if capture_detail else None
        # We6 (exp09 W2): `on_turn`'s own top-of-turn snapshot, independent
        # of `capture_detail` (a caller can want one hook without the other).
        hook_state_before = copy.deepcopy(state) if on_turn is not None else None

        pre_turn_lives = int(state.get("lives", 0))
        pre_turn_opp_lives = int(_versus_meta(state).get("opponent_lives", 0))

        def _count_stop_reason(reason: str) -> None:
            stop_reason_counts[reason] += 1

        # We6 (exp09 W2): `step_records` is only allocated when a caller
        # actually wants it -- see the module docstring's "We6 addition".
        # Each entry is the board immediately BEFORE that op was applied,
        # paired with the op itself; only ops that actually committed are
        # appended -- a chain that diverges partway still gets step_records
        # for its successfully-applied prefix, matching
        # `search_recommender.py::_apply_chain`'s "the board reached so far
        # still counts" contract. Under the honest frame it spans the whole
        # turn's segments, which is what a renderer wants.
        step_records: list[dict[str, Any]] | None = [] if on_turn is not None else None

        # Wd (exp13 A1): the SEGMENT loop. Under the determinized default it
        # runs EXACTLY ONCE and its body is line-for-line the pre-A1 turn
        # body (decode -> replay every non-END_TURN op -> fall through);
        # under `segmented-honest` the chain is committed op by op against
        # the REAL board and the loop re-enters after every realized
        # stochastic resolution, from the observed state, on a fresh
        # imagination stream. See the module docstring's "Wd addition".
        #
        # exp16 W3 moved that loop VERBATIM into `tools/segmented_turn.py`
        # so the play-vs-AI duel drives the AI side through the same code
        # instead of a second copy (both PLANs require the reuse: two
        # implementations of segmentation + stream separation drift, and
        # this is the silent-error surface of both lines). Nothing about
        # this driver's behaviour changed with the move.

        # exp13 W1: the per-segment recorder hangs off the SHARED loop's own
        # hooks -- `decide` for the imagined decision state and the
        # recommendation, `on_committed_op` for every op that reached the
        # real board. It used to hang off a copy of that loop which lived
        # here; exp16 W3 moved the loop out, which would have left the
        # recorder hooked to dead code and recording nothing without ever
        # raising. See `SegmentRecorder`. Off entirely (`segment_sink=None`)
        # unless a run asked for `--segment-record-out`, so an ordinary
        # eval pays nothing and behaves identically.
        segment_recorder = (
            SegmentRecorder(segment_sink, engine_seed=game_engine_seed, honest=honest)
            if segment_sink is not None
            else None
        )
        decide = None
        if segment_recorder is not None:
            segment_recorder.begin_turn(state)
            decide = segment_recorder.wrap_decide(bc_decide(bc))

        turn_out = run_segmented_turn(
            state,
            bc=bc,
            honest=honest,
            game_engine_seed=game_engine_seed,
            race_wins=race_wins,
            set_race_context=set_race_context,
            set_imagination_context=set_imagination_context,
            decide=decide,
            step_records=step_records,
            stop_reason_sink=_count_stop_reason,
            on_committed_op=(
                segment_recorder.on_committed_op
                if segment_recorder is not None
                else None
            ),
        )
        board = turn_out.board
        rec = turn_out.rec
        chain_preview = turn_out.chain_preview
        committed_ops = turn_out.committed_ops
        chain_types = turn_out.chain_types
        stop_reason = turn_out.stop_reason
        replay_diverged_at = turn_out.replay_diverged_at
        decode_failed = turn_out.decode_failed
        segments = turn_out.segments
        segments_capped = turn_out.segments_capped
        if segment_recorder is not None:
            # Before the `decode_failed` / `replay_diverged_at`
            # branches below, both of which break out of the turn
            # loop -- a failed turn's segments are recorded too.
            segment_recorder.end_turn(turn_out)

        if decode_failed:
            # The REASON travels with the failure. See `_decode_failure_reason`.
            end_reason = (
                f"{END_REASON_DECODE_FAILED}:{_decode_failure_reason(rec, stop_reason)}"
            )
            per_turn.append(
                _turn_record(
                    turn=state.get("turn", 0),
                    # Empty under the determinized frame (nothing was decoded
                    # at all); under the honest frame it carries whatever
                    # earlier segments of this turn already committed.
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        board_pre_battle=None,
                        parsed_state=None,
                        battle=None,
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": state.get("turn", 0),
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": None,
                    }
                )
            break

        if replay_diverged_at is not None:
            end_reason = f"{END_REASON_CHAIN_REPLAY_DIVERGED}:{replay_diverged_at}"
            per_turn.append(
                _turn_record(
                    turn=state.get("turn", 0),
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        board_pre_battle=board,
                        parsed_state=None,
                        battle=None,
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": state.get("turn", 0),
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": step_records or [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": board,
                    }
                )
            break

        turn_played = int(board.get("turn", state.get("turn", 1)))

        # We9 (exp12 W1'a): the pre-battle afterstate row -- `board` here is
        # the EXACT object about to be handed to `_resolve_versus_turn`
        # (its `meta.versus` intact), not the oracle-normalized
        # `battle_state` a caller only gets back AFTER resolution (see
        # `_build_turn_detail`'s docstring for that distinction). Emitted
        # regardless of whether the resolution below then succeeds, so an
        # `end_turn_failed` turn's genuine pre-battle board is still
        # recorded (rare, S2-gated near-zero in practice).
        if afterstate_sink is not None:
            afterstate_sink(
                {
                    "turn": turn_played,
                    "stop_reason": stop_reason,
                    "lives": pre_turn_lives,
                    "opp_lives": pre_turn_opp_lives,
                    "wins": race_wins,
                    "state": copy.deepcopy(board),
                }
            )

        # Wa (exp12 route a, wave A0): the rollout-teacher decision row. Same
        # emission point as the We9 row above, so `board` is exactly the
        # afterstate the chosen candidate promised and the race scalars are
        # the same driver-true locals. `teacher_record` rides on `rec`'s TOP
        # level (never `search_diagnostics`, which `_turn_record` retains for
        # the whole run), so nothing here is kept after this turn ends.
        if teacher_sink is not None:
            teacher_record = rec.get("teacher_record")
            if isinstance(teacher_record, dict):
                played_signature = state_signature(board)
                teacher_sink(
                    {
                        **teacher_record,
                        "turn": turn_played,
                        "race": {
                            "turn": turn_played,
                            "lives": pre_turn_lives,
                            "opp_lives": pre_turn_opp_lives,
                            "wins": race_wins,
                        },
                        "stop_reason": stop_reason,
                        "played_signature": played_signature,
                        # Wd (exp13 A1): under the honest frame one turn is
                        # several proposals, only whose prefixes were
                        # committed, so `chain_preview` (the LAST segment's)
                        # is not what was played -- the committed ops are.
                        # Structurally unreachable today (the teacher record
                        # requires rollout scoring, which this frame refuses),
                        # kept correct so enabling it later cannot mislabel.
                        "played_chain": copy.deepcopy(
                            committed_ops if honest else chain_preview
                        ),
                        # The board handed to `bc.recommend` this turn: every
                        # candidate afterstate is `_apply_chain(start_state,
                        # candidate.chain)`, which is what the checker's
                        # D1-style gate re-derives.
                        "start_state": copy.deepcopy(state),
                        "self_check": _teacher_self_check(
                            teacher_record, played_signature=played_signature
                        ),
                    }
                )

        turn_resolution = _resolve_versus_turn(
            board,
            sample_for_pid_fn=sample_for_pid_fn,
            sample_random_fn=sample_random_fn,
            parse_cache=parse_cache,
            simulation_count=1,
            game_mode=engine_game_mode,
            max_lives=heal_cap,
        )

        if not turn_resolution["ok"]:
            end_reason = turn_resolution["end_reason"]
            # exp13 F1 (Ruihan's ruling, 2026-08-07): under the ARENA ruler,
            # "the pool has no opponent at this turn" is not an agent failure
            # and must not erase the trophies already earned. Re-label it as
            # the completed verdict `opponent_pool_exhausted` so it takes the
            # same scoring path `turn_cap` already takes.
            #
            # This is the RESIDUAL path. `effective_arena_max_turn` clamps the
            # cap to the pool's depth, so a flat tail truncation ends the game
            # as `turn_cap` one turn before this branch can fire; what reaches
            # here is a pool with a HOLE at a turn below its own maximum. Both
            # are covered deliberately -- the clamp alone would leave the hole
            # case scoring 0.
            #
            # Scoped to arena: under chain rules a sampling failure means the
            # followed chain AND the random fallback both came up empty, which
            # is a genuine infrastructure failure there, and versus games have
            # no trophies to preserve in the first place.
            pool_exhausted = is_arena and _is_pool_exhausted_end_reason(end_reason)
            if pool_exhausted:
                pool_exhausted_detail = {
                    "turn": turn_played,
                    "raw_end_reason": str(end_reason),
                }
                end_reason = END_REASON_OPPONENT_POOL_EXHAUSTED
            resolved = turn_resolution["resolved"]
            per_turn.append(
                _turn_record(
                    turn=turn_played,
                    chain_types=chain_types,
                    stop_reason=stop_reason,
                    outcome=None,
                    opponent_pets=None,
                    lives=pre_turn_lives,
                    opp_lives=pre_turn_opp_lives,
                    detail=_build_turn_detail(
                        state_before=state_before_snapshot,
                        # Best-effort: `resolved` failed at some point
                        # after `battle_state` was computed, so it may or
                        # may not carry `parsed_state`/`battle` depending
                        # on which stage failed (see
                        # `resolve_end_turn_with_sampled_battle`'s own
                        # per-branch envelopes) -- fall back to the
                        # locally replayed `board` if `battle_state`
                        # itself is absent.
                        board_pre_battle=(
                            resolved.get("battle_state")
                            if isinstance(resolved.get("battle_state"), dict)
                            else board
                        ),
                        parsed_state=resolved.get("parsed_state"),
                        battle=resolved.get("battle"),
                        state_after=None,
                        full=capture_detail,
                    ),
                    search_used=bool(rec.get("search_used", False)),
                    search_n_generated=rec.get("search_n_generated"),
                    search_n_dedup=rec.get("search_n_dedup"),
                    search_diagnostics=rec.get("search_diagnostics"),
                    segments=segments,
                    segments_capped=segments_capped,
                    cat_trigger_audit=turn_out.cat_trigger_audit,
                )
            )
            if on_turn is not None:
                on_turn(
                    {
                        "turn": turn_played,
                        "state_before": hook_state_before,
                        "rec": rec,
                        "step_records": step_records or [],
                        "ok": False,
                        "end_reason": end_reason,
                        "state_after": None,
                        "board_pre_battle": (
                            resolved.get("battle_state") if isinstance(resolved.get("battle_state"), dict) else board
                        ),
                    }
                )
            break

        state = turn_resolution["state_after"]
        if turn_resolution["fallback_used"]:
            fallback_turns.append(turn_played)

        turns_completed += 1
        # We9 (exp12 W1'a): advance the pre-battle win counter AFTER this
        # turn's outcome is known, so it is ready for turn N+1's afterstate
        # row (never this turn's own, already emitted above pre-battle).
        if turn_resolution["outcome"] == "win":
            wins_so_far += 1
        parsed_state = turn_resolution["parsed_state"]
        per_turn.append(
            _turn_record(
                turn=turn_played,
                chain_types=chain_types,
                stop_reason=stop_reason,
                outcome=turn_resolution["outcome"],
                opponent_pets=(parsed_state.get("opponentPets") if isinstance(parsed_state, dict) else None),
                lives=int(state.get("lives", 0)),
                opp_lives=int(_versus_meta(state).get("opponent_lives", 0)),
                detail=_build_turn_detail(
                    state_before=state_before_snapshot,
                    board_pre_battle=turn_resolution["battle_state"],
                    parsed_state=parsed_state,
                    battle=turn_resolution["battle"],
                    # `state` was already reassigned to
                    # `turn_resolution["state_after"]` above, so it IS
                    # this turn's true state_after.
                    state_after=state,
                    full=capture_detail,
                ),
                search_used=bool(rec.get("search_used", False)),
                search_n_generated=rec.get("search_n_generated"),
                search_n_dedup=rec.get("search_n_dedup"),
                search_diagnostics=rec.get("search_diagnostics"),
                segments=segments,
                segments_capped=segments_capped,
                cat_trigger_audit=turn_out.cat_trigger_audit,
            )
        )
        if on_turn is not None:
            on_turn(
                {
                    "turn": turn_played,
                    "state_before": hook_state_before,
                    "rec": rec,
                    "step_records": step_records or [],
                    "ok": True,
                    "end_reason": None,
                    "state_after": state,
                    "board_pre_battle": turn_resolution["battle_state"],
                }
            )

        # FIX 5: the LIVES verdict uses the verified `TrainingEnv._is_done`
        # (passed `max_turn=None` so its own turn check is disabled), and the
        # turn cap is checked here on `turn_played` -- the turn whose battle
        # we JUST resolved -- rather than on `_is_done`'s post-advance
        # `state["turn"]`. `_is_done(state, max_turn)` breaks once the
        # advanced turn reaches `max_turn`, so the last battle it ever allows
        # is at `max_turn - 1` (off-by-one, exclusive). Checking
        # `turn_played >= max_turn` makes the final battle AT `max_turn`
        # inclusive. Behaviorally inert for the runs that matter (games end
        # by lives ~turn 10, far below the 30-turn cap; the smoke never
        # reaches it), so the verified per-turn loop's observable behavior is
        # unchanged -- this only tightens the rarely-hit cap boundary.
        #
        # Wb (exp13 W0a): under arena rules the verdict is trophies/lives and
        # `_is_done` is NOT usable -- its arena branch ends at SEVEN trophies
        # (the RL env's frame), which would silently cut this ruler's games
        # three trophies short.
        if is_arena:
            lives_done = bool(
                int(state.get("lives", 0)) <= 0
                or int(state.get("trophies", 0) or 0) >= MAX_TROPHIES
            )
        else:
            lives_done = bool(TrainingEnv._is_done(state, None))
        if lives_done or turn_played >= max_turn:
            break

    lives_final = int(state.get("lives", 0))
    opp_lives_final = int(_versus_meta(state).get("opponent_lives", 0))
    trophies_final = int(state.get("trophies", 0) or 0) if is_arena else None

    if end_reason is None:
        if is_arena:
            # A turn awards a trophy or costs a life, never both, so these
            # two branches cannot both be true; lives is checked first to
            # mirror the versus ordering below.
            if lives_final <= 0:
                end_reason = END_REASON_PLAYER_LIVES_0
            elif int(trophies_final or 0) >= MAX_TROPHIES:
                end_reason = END_REASON_TROPHIES_10
            else:
                end_reason = END_REASON_TURN_CAP
        elif lives_final <= 0:
            end_reason = END_REASON_PLAYER_LIVES_0
        elif opp_lives_final <= 0:
            end_reason = END_REASON_OPPONENT_LIVES_0
        else:
            end_reason = END_REASON_TURN_CAP

    win = (
        _arena_win(lives_final, int(trophies_final or 0))
        if is_arena
        else _versus_win(lives_final, opp_lives_final)
    )

    # FIX 4: `final_followed_pid` records which pid the versus chain was
    # actually on at game end -- it differs from `initial_pid` exactly when
    # a fallback switched the followed game
    # (`resolve_end_turn_with_sampled_battle` rewrites
    # `meta.versus.current_opponent_participation_id` on each resolved turn),
    # so the two together make the fallback's effect on opponent identity
    # visible without cross-referencing `fallback_turns`.
    final_followed_pid = str(_versus_meta(state).get("current_opponent_participation_id") or initial_pid)

    # We9 (exp12 W1'a): the per-game final outcome row, same values the
    # `--out` JSONL row for this game already carries below (`win`,
    # `end_reason`, `turns_survived` all read off the exact same locals), so
    # the two files agree by construction.
    if afterstate_sink is not None:
        afterstate_sink(
            {
                "final": True,
                "win": win,
                "end_reason": end_reason,
                "turns_survived": turns_completed,
                # Wc (exp13 W0b, codex review P2): stamp the RULER on the
                # per-game final row -- under ARENA ONLY, the same
                # arena-only-tag discipline `_new_game_state` uses for
                # `meta.game_rules`, so a versus record stays byte-identical
                # (its exact key set is pinned in test_afterstate_recorder).
                # This is what lets a dataset builder REFUSE arena records
                # rather than silently mislabel them: the builders' `margin`
                # is `player_lives - opponent_lives`, and under arena rules
                # `opponent_lives` is the PLAN D2 race feature, not a life
                # bar (see build_afterstate_value_dataset's own guard).
                **({"game_rules": rules} if is_arena else {}),
            }
        )

    # Wa (exp12 route a): the same final row for the teacher stream, so a
    # decision's teacher scores can be joined against the outcome the game
    # actually reached (A3's calibration-vs-realized-outcomes report).
    if teacher_sink is not None:
        teacher_sink(
            {
                "final": True,
                "win": win,
                "end_reason": end_reason,
                "turns_survived": turns_completed,
                # Wc (exp13 W0b): same arena-only ruler stamp as the
                # afterstate final row above, for the same reason -- see
                # that comment. The teacher stream has no `margin` column,
                # but it stores the same `opponent_lives` race scalar.
                **({"game_rules": rules} if is_arena else {}),
            }
        )

    return {
        "final_followed_pid": final_followed_pid,
        "win": win,
        "player_lives": lives_final,
        # Wb (exp13 W0a): under ARENA rules this is the D2 race-feature value
        # (`_apply_arena_race_context`), NOT a life total -- arena has no
        # opponent life bar. `trophies` below is the arena race; it is None
        # under versus rules, where no trophy is ever awarded, rather than a
        # 0 that would read as "earned none".
        "opponent_lives": opp_lives_final,
        "trophies": trophies_final,
        "game_rules": rules,
        "arena_race_convention": (race_convention if is_arena else None),
        "turns_survived": turns_completed,
        "end_reason": end_reason,
        # exp13 F1: present ONLY on the games whose `end_reason` was
        # re-labelled to `opponent_pool_exhausted`, so every other row keeps
        # exactly the key set it had before. Carries the turn the pool ran
        # dry and the raw `end_turn_failed:...` string it was re-labelled
        # from, because "these games kept their trophies" is only auditable
        # if you can still see which games they were and where they stopped.
        **(
            {"pool_exhausted": pool_exhausted_detail}
            if pool_exhausted_detail is not None
            else {}
        ),
        "num_fallbacks": len(fallback_turns),
        "fallback_turns": fallback_turns,
        # We8 (exp12 W0): which ruler produced this game, and what the two
        # fallback fields above MEAN under it (structurally 0 under arena --
        # see FALLBACK_SEMANTICS_ARENA).
        "opponent_mode": opponent_mode,
        "fallback_semantics": _fallback_semantics(opponent_mode),
        # We3 (exp09 W6a): count of turns where `SearchRecommender` actually
        # searched (see `_turn_record`'s `search_used` field); always 0
        # under plain `--recommender bc`.
        "search_used_turns": sum(1 for t in per_turn if t.get("search_used")),
        # exp12 W2c (width curve): per-SEARCHED-turn candidate dedup counts,
        # in turn order. A game-level list (not just a mean) so the report
        # can show the whole distribution -- the width arms are only
        # meaningful if the proposer actually produces distinct candidates
        # at the requested width.
        "search_dedup_by_turn": [
            v for t in per_turn for v in _search_width_counts(t, "search_n_dedup")
        ],
        "search_generated_by_turn": [
            v for t in per_turn for v in _search_width_counts(t, "search_n_generated")
        ],
        # Wd (exp13 A1): segment accounting, present ONLY under the honest
        # frame -- same discipline as the per-turn `segments` block, so a
        # determinized per-game row stays byte-identical to every pre-A1 run.
        **(
            {
                "turn_mode": turn_mode,
                "segments_by_turn": [int(t.get("n_segments") or 0) for t in per_turn],
                "n_segments_total": sum(int(t.get("n_segments") or 0) for t in per_turn),
                # How many SEARCHES this game paid for: one per segment that
                # actually searched. Under the determinized frame this would
                # equal `search_used_turns` by construction; the honest frame
                # is where the two come apart, and that gap IS the frame's
                # compute cost (A1 ruling 6(ii)/(iii) budget both off it).
                "n_searched_segments": sum(
                    1
                    for t in per_turn
                    for s in (t.get("segments") or [])
                    if s.get("search_used")
                ),
                # WHICH randomness actually forced a re-search, by engine
                # reason -- the distribution A1 ruling 6(i) pre-registers.
                "boundary_reason_counts": dict(
                    sorted(
                        Counter(
                            str(s["boundary_reason"])
                            for t in per_turn
                            for s in (t.get("segments") or [])
                            if s.get("boundary_reason")
                        ).items()
                    )
                ),
            }
            if honest
            else {}
        ),
        "stop_reasons": dict(stop_reason_counts),
        # exp19 (PLAN_W1 step 6): the game's Cat trigger opportunity counts,
        # summed off the per-turn rows so the two can never disagree.
        "cat_trigger_audit": _cat_audit_sum(t.get("cat_trigger_audit") for t in per_turn),
        "per_turn": per_turn,
    }


def play_one_game(
    bc: BcRecommender | SearchRecommender,
    opp_source: ChainSnapshotSource,
    opening: VariedOpeningSource | FixedOpeningSource | dict[str, Any],
    *,
    game_index: int,
    max_turn: int = DEFAULT_MAX_TURN,
    seed: int = DEFAULT_SEED,
    parse_cache: dict[str, Any] | None = None,
    capture_detail: bool = False,
    on_turn: Callable[[dict[str, Any]], None] | None = None,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    arena_source: ChainSnapshotSource | None = None,
    afterstate_sink: Callable[[dict[str, Any]], None] | None = None,
    teacher_sink: Callable[[dict[str, Any]], None] | None = None,
    segment_sink: Callable[[dict[str, Any]], None] | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> dict[str, Any]:
    """Play one whole game with the BC policy to a lives/trophy verdict.

    `opening` (full-game frame fix, `train/opening_source.py`): either an
    opening-source object (`.state_for_game(game_index)` -- a VARIED,
    per-game turn-1 state; `main()`'s new default) or, for full backward
    compatibility with existing direct callers that still pass a raw fixture
    dict (`tools/gen_distill_dataset.py`, some tests -- neither of which this
    fix touches), a plain `dict`, silently wrapped as a `FixedOpeningSource`
    so every game gets that SAME state -- byte-for-byte the old
    single-fixture behavior those callers already depend on.

    `bc` (We3, exp09 W6a): may be a plain `BcRecommender` OR a
    `SearchRecommender` wrapping one (`--recommender search`, see module
    docstring) -- this function only ever calls `bc.recommend(state)` and
    reads the returned dict, both of which are API-compatible, so nothing
    below needs to know or care which one it was actually given.

    Builds the turn-1 state (`_new_game_state`) then delegates the whole
    per-turn loop + end-of-game bookkeeping to `play_out_game` (We5, exp09
    W1 -- see that function's docstring for the full per-turn algorithm,
    unchanged from before this split).

    `capture_detail` (We2, default False): when True, every `per_turn` entry
    also carries a `detail` sub-dict (`_build_turn_detail`) with the boards
    and battle payload needed to render a gallery image for that turn. This
    changes ONLY what gets recorded, never the decode/replay/battle-
    resolution above -- it never reads its own output, so turning it on
    cannot change which action gets chosen or which battle gets fought.
    It is NOT, however, a "replay this game later" switch: the battle
    oracle this turn's END_TURN calls into (`resolve_end_turn_with_sampled_battle`
    -> `run_battle_oracle_with_config`) is an unseeded Monte Carlo draw
    (verified: the identical board config produced different W/L outcomes
    across repeated calls), so re-running `game_index` a second time -- with
    or without `capture_detail` -- is NOT guaranteed to reproduce the same
    game. Callers that need to render a SPECIFIC game must set
    `capture_detail=True` on the ONE pass that plays it (see
    `render_selected_games`'s docstring), not capture it after the fact.

    `on_turn` (We6, exp09 W2, default None): passed straight through to
    `play_out_game` -- see that function's docstring for the full per-turn
    payload shape.

    `afterstate_sink` (We9, exp12 W1'a, default None): passed straight
    through to `play_out_game` -- see that function's / the module
    docstring's "We9 addition" for the row schema.

    `teacher_sink` (Wa, exp12 route a, default None): passed straight through
    to `play_out_game` -- see the module docstring's "Wa addition". This
    function additionally calls `bc.set_decision_context(game_index=...)`
    when the recommender defines it (duck-typed, so a plain `BcRecommender`
    is untouched), which is what makes `SearchRecommender`'s CRN keys a pure
    function of `(seed, game_index, turn, repeat)` and therefore identical
    under any `--game-index-start` sharding.

    `opponent_mode` / `arena_source` (We8, exp12 W0, defaults `"chain"` /
    None = byte-identical to every run before these existed): under
    `"arena"`, `sample_random_fn` becomes a closure over a PER-GAME
    `random.Random(f"exp12_arena_opponent:{seed}:{game_index}")` drawing
    from `arena_source` (the full, all-splits pool -- required, and a
    different object from `opp_source`), so the whole opponent sequence of
    a game is a deterministic function of (seed, game_index) and no two
    games share an RNG stream. `initial_pid_for_game(game_index)` is still
    drawn from `opp_source` (never from `arena_source`) even though arena
    mode discards it, so that both rulers consume the same draws in the
    same order for the same seed and their per-game_index openings/engine
    seeds line up -- the result records it as `initial_followed_pid` with
    `initial_pid_unused: True`.

    `max_turn` (exp13 F1, 2026-08-07): under arena mode the value handed to
    `play_out_game` is `effective_arena_max_turn(max_turn, arena_source)`,
    i.e. clamped to the pool's own depth. The configured cap is the game's
    rule; the pool's depth is the harness's, and a game may not be played
    past the last turn its opponent supply can serve. Unclamped under chain
    mode and whenever the pool is deeper than the cap, so every pre-F1 run
    reproduces byte for byte.

    `game_rules` / `arena_race_convention` (Wb, exp13 W0a, defaults
    `"versus"` / `"trophies_mapped"`): passed straight through to
    `_new_game_state` (the arena start state) and `play_out_game` (the
    bookkeeping + terminal rule) -- see their docstrings. NOTE this is a
    different axis from `opponent_mode`: rules say WHICH GAME, opponent_mode
    says WHO you play. Every RNG stream here is a pure function of (seed,
    game_index) and is drawn BEFORE either is consulted, so the two rulers of
    the same (seed, game_index) face the identical opponent sequence.
    """
    # Wc (exp13 W0b, codex review P2): the same refusals `main()` applies,
    # applied HERE too -- this is a reusable entry point, and a direct caller
    # (a tool, a notebook, a later wave's harness) that never goes through
    # `main()` must not be able to run arena rules against a chain opponent
    # or with a rollout scorer. See `_game_rules_refusal` for both.
    _assert_game_rules_supported(
        bc, game_rules=game_rules, opponent_mode=opponent_mode, turn_mode=turn_mode
    )

    opening_source = opening if hasattr(opening, "state_for_game") else fixed_opening_source(opening)
    fixture_initial_state = opening_source.state_for_game(game_index)

    mode = str(opponent_mode or OPPONENT_MODE_CHAIN).strip().lower()
    if mode not in OPPONENT_MODES:
        raise ValueError(f"unknown_opponent_mode:{opponent_mode}:valid={list(OPPONENT_MODES)}")

    # Wa (exp12 route a, wave A0): tell a CRN-capable recommender which game
    # it is in, BEFORE the first `recommend` of this game. Duck-typed exactly
    # like every other optional recommender surface in this file.
    set_context = getattr(bc, "set_decision_context", None)
    if callable(set_context):
        set_context(game_index=int(game_index))

    followed_pid = opp_source.initial_pid_for_game(game_index)
    # Namespaced separately from `initial_pid_for_game`'s own RNG stream so
    # the two draws (which pid to follow vs. which engine seed to start
    # from) vary independently across games. See `_new_game_state` for why
    # an explicit engine seed is required at all.
    engine_seed_rng = random.Random(f"eval_versus_fullgame_engine_seed:{seed}:{game_index}")
    engine_seed = engine_seed_rng.randrange(0, 2**31)
    state = _new_game_state(
        fixture_initial_state,
        followed_pid,
        engine_seed=engine_seed,
        game_rules=game_rules,
        arena_race_convention=arena_race_convention,
    )

    # BOTH rulers draw their random opponents from a PER-GAME rng namespaced
    # by (seed, game_index), never from `opp_source`'s shared `_random_rng`.
    #
    # Arena needs it because that stream IS the ruler. Chain needs it because
    # of `--game-index-start` (exp12 W0 codex review F2): the shared
    # `_random_rng` is seeded per PROCESS, so a sharded chain run would give
    # shard 1 the same fallback draw sequence shard 0 got, and neither would
    # match the unsharded run. Per-game keying makes a game's opponents a
    # pure function of (seed, game_index) under either ruler, which is the
    # invariant `--game-index-start` sharding is built on.
    #
    # This DOES change which uniform draws a chain run makes versus pre-exp12
    # (same distribution, different sample), so a chain number is reproducible
    # only in the statistical sense across that boundary -- recorded as
    # `fallback_rng: "per_game"` in the report metadata. The two salts are
    # distinct so the two rulers never share a stream.
    if mode == OPPONENT_MODE_ARENA:
        if arena_source is None:
            raise ValueError("arena_opponent_mode_requires_arena_source")
        arena_rng = random.Random(f"exp12_arena_opponent:{seed}:{game_index}")

        def _arena_sample_random(turn: int) -> dict[str, Any]:
            return arena_source.sample_random_with_rng(int(turn), arena_rng)

        sample_random_fn: Callable[[int], dict[str, Any]] = _arena_sample_random
    else:
        chain_fallback_rng = random.Random(f"exp12_chain_fallback:{seed}:{game_index}")

        def _chain_sample_random(turn: int) -> dict[str, Any]:
            return opp_source.sample_random_with_rng(int(turn), chain_fallback_rng)

        sample_random_fn = _chain_sample_random

    # exp13 F1: the binding cap is the smaller of the game's rule and the
    # pool's depth. Applied HERE because this is the single function that
    # holds both `max_turn` and `arena_source`, so every caller of the API
    # face gets it, not only the CLI. Inert under chain mode and whenever the
    # pool is deeper than the configured cap; the CLI recomputes the same
    # number from the same helper for its log line and its report block.
    play_max_turn = effective_arena_max_turn(max_turn, arena_source, opponent_mode=mode)

    outcome = play_out_game(
        state,
        bc,
        initial_pid=followed_pid,
        sample_for_pid_fn=opp_source.sample_for_pid,
        sample_random_fn=sample_random_fn,
        max_turn=play_max_turn,
        parse_cache=parse_cache,
        capture_detail=capture_detail,
        on_turn=on_turn,
        opponent_mode=mode,
        afterstate_sink=afterstate_sink,
        teacher_sink=teacher_sink,
        segment_sink=segment_sink,
        game_rules=game_rules,
        arena_race_convention=arena_race_convention,
        # Wd (exp13 A1): this function owns the game's engine seed, but
        # `play_out_game` reads it back off the turn-1 state it is handed
        # (see there) rather than taking a second copy of it as a parameter,
        # so the two can never disagree about which seed keyed imagination.
        turn_mode=turn_mode,
    )

    return {
        "game_index": int(game_index),
        "followed_pid": followed_pid,
        "initial_followed_pid": followed_pid,
        # We8 (exp12 W0): under the arena ruler the two pids above are
        # BOOKKEEPING ONLY -- drawn (so the RNG streams stay aligned with a
        # chain run of the same seed) and then popped off the state before
        # turn 1 ever reads them. Flagged explicitly so no reader mistakes
        # `followed_pid` for "the opponent this game played against".
        "initial_pid_unused": mode == OPPONENT_MODE_ARENA,
        # Full-game frame fix: which opening this game actually got
        # ("seeded_roll" / "fixed_single_fixture"), so a run's own
        # games.jsonl is self-describing (CLAUDE.md "no naked numbers") --
        # see train/opening_source.py.
        "opening_source_kind": opening_source.provenance_for_game(game_index),
        **outcome,
    }


INCOMPLETE_FRACTION_WARN_THRESHOLD = 0.02


def _search_width_aggregate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """exp12 W2c (width curve): what the requested candidate width actually
    bought, over every SEARCHED turn of the run.

    A width arm is only real if the proposer produces distinct end boards at
    that width: `--search-candidates 72` that dedups to 9 boards a turn is a
    72-candidate arm in name and a 9-candidate arm in fact, and the width
    curve would then be measuring nothing. So every report carries the
    per-turn dedup distribution (mean / median / min / max / deciles) next
    to the requested width, and `dedup_ratio` = mean distinct / requested.

    Returns None when no turn reported counts (plain `--recommender bc`, or
    a JSONL captured before this telemetry existed) rather than a block of
    zeros that would read as "the proposer produced nothing".
    """
    dedup: list[int] = []
    generated: list[int] = []
    for r in results:
        dedup.extend(int(v) for v in (r.get("search_dedup_by_turn") or []))
        generated.extend(int(v) for v in (r.get("search_generated_by_turn") or []))
    if not dedup:
        return None
    ordered = sorted(dedup)
    requested = sorted(set(generated))
    mean_dedup = statistics.mean(ordered)
    return {
        "n_searched_turns": len(ordered),
        # Normally one value (the arm's width); a list keeps a hand-merged
        # multi-arm JSONL honest instead of silently quoting one of them.
        "requested_candidates": requested,
        "dedup_mean": mean_dedup,
        "dedup_median": statistics.median(ordered),
        "dedup_min": ordered[0],
        "dedup_max": ordered[-1],
        "dedup_deciles": [ordered[min(len(ordered) - 1, (len(ordered) * d) // 10)] for d in range(1, 10)],
        "dedup_ratio": (mean_dedup / float(requested[0]) if len(requested) == 1 and requested[0] > 0 else None),
        "dedup_histogram": dict(sorted(Counter(ordered).items())),
    }


def _honest_frame_aggregate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Wd (exp13 A1): what the honest frame cost and why, over the run.

    Returns None -- so the report gains NO key at all -- whenever no game
    carries a `turn_mode`, which is every determinized run and every report
    re-derived from a pre-A1 JSONL. That is what keeps the byte-identity pin
    green; a block of zeros would instead read as "this run segmented
    nothing", a different claim from "this run was not on that frame".

    `mean_segments_per_turn` is the honest frame's headline cost (1.0 means
    no turn ever crossed a randomness boundary), `mean_searches_per_game`
    prices it in search calls, and `boundary_reason_counts` says WHICH engine
    randomness forced the re-searches -- the three numbers A1 ruling 6
    pre-registers before any round-0 pin.

    MIXED INPUTS ARE REFUSED (exp13 W0b', codex finding 4). "No `turn_mode`"
    means "determinized", not "no frame", so a hand-merged determinized +
    honest JSONL used to land here as a set of ONE tagged mode: the report
    called itself `segmented-honest` while `n_games` covered both arms, and
    every number in this block silently described the honest subset only. The
    `"mixed"` treatment the sibling aggregates use (opponent_mode,
    game_rules) is not available as a fix, because these are SUMS and MEANS
    over per-game lists rather than a label -- there is no honest mixed value
    for `mean_segments_per_turn`. So a mixed list raises instead: the only way
    to produce one is to merge two arms by hand, and that merge is the bug.
    """
    tagged = [r for r in results if r.get("turn_mode")]
    if not tagged:
        return None
    if len(tagged) != len(results):
        raise ValueError(
            f"honest_frame_aggregate_mixed_frames:tagged={len(tagged)}:total={len(results)} -- "
            f"{len(results) - len(tagged)} game rows carry no turn_mode (i.e. they were "
            "played on the whole-determinized frame) while others do. Aggregate each "
            "frame's rows separately; the two arms are different rulers, not one run."
        )
    modes = sorted({str(r["turn_mode"]) for r in tagged})
    segments = [int(v) for r in results for v in (r.get("segments_by_turn") or [])]
    searched = [int(r.get("n_searched_segments") or 0) for r in results]
    boundary: Counter[str] = Counter()
    for r in results:
        for reason, count in (r.get("boundary_reason_counts") or {}).items():
            boundary[str(reason)] += int(count)
    return {
        "turn_mode": modes[0] if len(modes) == 1 else "mixed:" + ",".join(modes),
        "n_turns": len(segments),
        "n_segments": sum(segments),
        "mean_segments_per_turn": (statistics.mean(segments) if segments else None),
        "segments_histogram": {str(k): int(v) for k, v in sorted(Counter(segments).items())},
        "n_searched_segments": sum(searched),
        "mean_searches_per_game": (statistics.mean(searched) if searched else None),
        "boundary_reason_counts": dict(sorted(boundary.items())),
    }


def _compute_aggregate(
    results: list[dict[str, Any]],
    *,
    arena_pool_size_by_turn: dict[int, int] | None = None,
) -> dict[str, Any]:
    """JSON-serializable aggregate metrics over `results` (We2 `--report-json`).

    FIX 1 (metric integrity): every game is classified `completed`
    (`end_reason` in `COMPLETED_END_REASONS`) or `incomplete` (an
    infrastructure failure: decode/end_turn/chain_replay). The POLICY-OUTCOME
    metrics -- winrate + its CI, avg player/opponent lives, avg turns
    survived -- are computed over COMPLETED games ONLY, because an incomplete
    game carries `win=False` and truncated lives/turns that are artifacts of
    the harness giving up, not of the policy losing; including them would
    bias every one of these down. `n_completed`/`n_incomplete`/
    `incomplete_by_reason`/`incomplete_fraction` are reported alongside so
    the drop is explicit. The DIAGNOSTIC distributions (end_reason across ALL
    games, stop_reason, chain_length, fallback) stay over ALL games on
    purpose -- the whole point of `end_reason_distribution` is to SHOW the
    infrastructure failures, and fallback/chain/stop stats describe the
    opponent source and decoder regardless of how the game ended.

    Winrate gets a game-clustered 95% bootstrap CI via the shared
    `cluster_ci` helper, each game its OWN cluster (PLAN.md "bootstrap CI
    (games are independent)") -- passing `(game_index, win_as_0_or_1)` pairs
    makes every cluster a singleton, i.e. a plain per-game bootstrap of the
    mean, through the same reused helper as every other cluster-CI here.

    We8 (exp12 W0): `opponent_mode` + `fallback_semantics` are read back off
    the results themselves (every `play_out_game` stamps both onto every
    game) rather than passed in, so a report re-derived from an existing
    `--out` JSONL says which ruler produced it too. A results list mixing
    rulers -- which nothing in this driver can produce, but a hand-merged
    JSONL could -- reports `"mixed"` rather than silently picking one.
    `arena_pool_size_by_turn` is the ONE piece the results cannot carry (it
    describes the pool, not any game), so `main` passes it in.

    Wb (exp13 W0a): `game_rules` / `arena_race_convention` are read back off
    the results the same way, and the TROPHY BLOCK (`mean_trophies` + its
    game-clustered CI, `completion_rate` = the share of completed runs that
    reached 10 trophies, and the full `trophies_histogram`) is computed over
    COMPLETED games only, exactly like the winrate above it. Under versus
    rules that whole block is None rather than zeros -- there is no trophy
    race to report, and a 0.0 would read as "earned none" (`trophy_semantics`
    says which of the two a report is).

    Note `completion_rate` is about the ARENA RUN being completed (10
    trophies); `n_completed`/`incomplete_*` above are about the HARNESS
    carrying a game to any verdict at all. Different questions, both kept.
    """
    n = len(results)
    modes = sorted({str(r.get("opponent_mode") or OPPONENT_MODE_CHAIN) for r in results})
    if len(modes) == 1:
        opponent_mode: str | None = modes[0]
    elif not modes:
        opponent_mode = None
    else:
        opponent_mode = "mixed:" + ",".join(modes)
    rules_seen = sorted({str(r.get("game_rules") or GAME_RULES_VERSUS) for r in results})
    if len(rules_seen) == 1:
        game_rules: str | None = rules_seen[0]
    elif not rules_seen:
        game_rules = None
    else:
        game_rules = "mixed:" + ",".join(rules_seen)
    conventions = sorted(
        {str(r["arena_race_convention"]) for r in results if r.get("arena_race_convention")}
    )
    ruler_block = {
        "opponent_mode": opponent_mode,
        # Wb (exp13 W0a): WHICH GAME these results are of, and (arena only)
        # under which PLAN D2 race convention.
        "game_rules": game_rules,
        "arena_race_convention": (
            conventions[0]
            if len(conventions) == 1
            else ("mixed:" + ",".join(conventions) if conventions else None)
        ),
        "trophy_semantics": (
            TROPHY_SEMANTICS_ARENA if game_rules == GAME_RULES_ARENA else TROPHY_SEMANTICS_VERSUS
        ),
        "fallback_semantics": (_fallback_semantics(opponent_mode) if len(modes) == 1 else None),
        # {turn: n_candidate_games} of the ARENA pool -- None under the
        # chain ruler (no arena pool is built) so a chain report never
        # implies one existed.
        "arena_pool_size_by_turn": (
            {str(int(t)): int(v) for t, v in sorted(arena_pool_size_by_turn.items())}
            if isinstance(arena_pool_size_by_turn, dict)
            else None
        ),
    }
    if n == 0:
        return {"n_games": 0, "n_completed": 0, "n_incomplete": 0, **ruler_block}

    completed = [r for r in results if _is_completed_end_reason(r["end_reason"])]
    incomplete = [r for r in results if not _is_completed_end_reason(r["end_reason"])]
    n_completed = len(completed)
    n_incomplete = len(incomplete)
    incomplete_by_reason = dict(sorted(Counter(r["end_reason"] for r in incomplete).items()))

    if completed:
        win_pairs = [(int(r["game_index"]), 1.0 if r["win"] else 0.0) for r in completed]
        win_mean, win_lo, win_hi, win_n_clusters = cluster_ci(win_pairs)
        wins = sum(1 for r in completed if r["win"])
        winrate = {
            "mean": win_mean,
            "lo95": win_lo,
            "hi95": win_hi,
            "n_games": win_n_clusters,
            "n_wins": wins,
        }
        avg_turns_survived = statistics.mean(r["turns_survived"] for r in completed)
        avg_player_lives = statistics.mean(r["player_lives"] for r in completed)
        # Wb (exp13 W0a): under arena rules each game's `opponent_lives` is
        # the D2 RACE FEATURE, not a life total, so an average of it would
        # print as "the opponent ended on 1.0 lives" -- a number about a life
        # bar this ruler does not have. None instead; `mean_trophies` below
        # is the arena race's actual summary.
        avg_opponent_lives = (
            None
            if game_rules == GAME_RULES_ARENA
            else statistics.mean(r["opponent_lives"] for r in completed)
        )
    else:
        winrate = None
        avg_turns_survived = avg_player_lives = avg_opponent_lives = None

    # Wb (exp13 W0a): the arena ruler's headline (mean trophies) + secondary
    # (completion rate) metrics, over COMPLETED games, with the whole
    # distribution next to them -- PLAN.md "Headline metric".
    mean_trophies: float | None = None
    mean_trophies_ci: dict[str, Any] | None = None
    trophies_histogram: dict[str, int] | None = None
    completion_rate: float | None = None
    if game_rules == GAME_RULES_ARENA and completed:
        trophy_values = [int(r.get("trophies") or 0) for r in completed]
        t_mean, t_lo, t_hi, t_n = cluster_ci(
            [(int(r["game_index"]), float(int(r.get("trophies") or 0))) for r in completed]
        )
        mean_trophies = t_mean
        mean_trophies_ci = {"mean": t_mean, "lo95": t_lo, "hi95": t_hi, "n_games": t_n}
        trophies_histogram = {str(k): int(v) for k, v in sorted(Counter(trophy_values).items())}
        completion_rate = (
            sum(1 for r in completed if r["end_reason"] == END_REASON_TROPHIES_10) / len(completed)
        )

    # Diagnostic distributions -- over ALL games (see docstring).
    games_with_fallback = sum(1 for r in results if r["num_fallbacks"] > 0)
    total_fallback_turns = sum(r["num_fallbacks"] for r in results)
    all_fallback_turns = sorted(t for r in results for t in r["fallback_turns"])
    end_reason_counts: Counter[str] = Counter(r["end_reason"] for r in results)

    # We3 (exp09 W6a): same shape as `fallback` above, over ALL games. Every
    # game has 0 under plain `--recommender bc` (see `play_one_game`'s
    # `search_used_turns`); non-zero only tells you search actually had an
    # opponent to score against on at least one turn, not that it changed
    # the chosen action -- see `SearchRecommender`'s own `search_scores`
    # per-turn detail for that.
    games_with_search_used = sum(1 for r in results if r["search_used_turns"] > 0)
    total_search_used_turns = sum(r["search_used_turns"] for r in results)
    search_width = _search_width_aggregate(results)

    stop_reason_totals: Counter[str] = Counter()
    chain_lengths: list[int] = []
    for r in results:
        for turn_entry in r["per_turn"]:
            stop_reason_totals[turn_entry["stop_reason"]] += 1
            chain_lengths.append(len(turn_entry["chain_types"]))
    sorted_lengths = sorted(chain_lengths)

    report: dict[str, Any] = {
        **ruler_block,
        "n_games": n,
        "n_completed": n_completed,
        "n_incomplete": n_incomplete,
        "incomplete_fraction": n_incomplete / n,
        "incomplete_by_reason": incomplete_by_reason,
        # winrate + these three averages are over COMPLETED games only.
        "winrate": winrate,
        "avg_turns_survived": avg_turns_survived,
        "avg_player_lives": avg_player_lives,
        "avg_opponent_lives": avg_opponent_lives,
        # Wb (exp13 W0a): the trophy block, completed games only. All four
        # are None under versus rules -- see the docstring.
        "mean_trophies": mean_trophies,
        "mean_trophies_ci": mean_trophies_ci,
        "trophies_histogram": trophies_histogram,
        "completion_rate": completion_rate,
        # fallback + the distributions below are over ALL games.
        "fallback": {
            "rate_games": games_with_fallback / n,
            "n_games_with_fallback": games_with_fallback,
            "total_fallback_turns": total_fallback_turns,
            "fallback_turn_numbers": all_fallback_turns,
        },
        "search_used": {
            "rate_games": games_with_search_used / n,
            "n_games_with_search_used": games_with_search_used,
            "total_search_used_turns": total_search_used_turns,
        },
        # exp12 W2c: candidate-width telemetry -- see `_search_width_aggregate`.
        # None under any recommender that does not report per-turn counts.
        "search_width": search_width,
        # exp19 (PLAN_W1 step 6): the run's Cat trigger opportunity counts,
        # over ALL games like the distributions around it.
        "cat_trigger_audit": _cat_audit_sum(r.get("cat_trigger_audit") for r in results),
        "end_reason_distribution": dict(sorted(end_reason_counts.items())),
        "stop_reason_distribution": dict(sorted(stop_reason_totals.items())),
        "chain_length": (
            {
                "min": sorted_lengths[0],
                "median": statistics.median(sorted_lengths),
                "max": sorted_lengths[-1],
            }
            if sorted_lengths
            else None
        ),
    }

    # Wd (exp13 A1): added ONLY when the results were produced on the honest
    # frame -- see `_honest_frame_aggregate` for why it is absent rather
    # than zeroed otherwise.
    honest_block = _honest_frame_aggregate(results)
    if honest_block is not None:
        report["turn_mode"] = honest_block.pop("turn_mode")
        report["segments"] = honest_block
    return report


def _print_aggregate(
    results: list[dict[str, Any]],
    *,
    arena_pool_size_by_turn: dict[int, int] | None = None,
) -> None:
    print("=== aggregate ===", flush=True)
    agg = _compute_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)
    print(
        f"game_rules: {agg['game_rules']}  arena_race_convention: {agg['arena_race_convention']}  "
        f"opponent_mode: {agg['opponent_mode']}  fallback_semantics: {agg['fallback_semantics']}",
        flush=True,
    )
    if agg["arena_pool_size_by_turn"]:
        sizes = agg["arena_pool_size_by_turn"]
        head = {t: sizes[t] for t in list(sizes)[:8]}
        print(f"arena_pool_size_by_turn (first 8 turns): {head}", flush=True)
    if agg["n_games"] == 0:
        print("no_games_played", flush=True)
        return

    n = agg["n_games"]
    n_completed = agg["n_completed"]
    n_incomplete = agg["n_incomplete"]
    fb = agg["fallback"]

    print(f"games_total: {n}  completed: {n_completed}  incomplete: {n_incomplete}", flush=True)
    if n_incomplete:
        print(f"incomplete_by_reason: {agg['incomplete_by_reason']}", flush=True)
    if agg["incomplete_fraction"] > INCOMPLETE_FRACTION_WARN_THRESHOLD:
        print(
            f"WARNING: incomplete_fraction={agg['incomplete_fraction']:.3f} "
            f"(> {INCOMPLETE_FRACTION_WARN_THRESHOLD:.2f}) -- the eval infra dropped "
            f"{n_incomplete}/{n} games before a lives verdict; policy metrics below are "
            f"over the {n_completed} COMPLETED games only, so treat them with care and "
            f"investigate the incomplete_by_reason breakdown.",
            flush=True,
        )

    if agg["winrate"] is None:
        print("winrate: n/a (no completed games)", flush=True)
    else:
        w = agg["winrate"]
        print(
            f"winrate (completed only): {w['mean']:.3f} ({w['n_wins']}/{n_completed})  "
            f"95% cluster-CI (by game): [{w['lo95']:.3f}, {w['hi95']:.3f}]",
            flush=True,
        )
        print(f"avg_turns_survived (completed): {agg['avg_turns_survived']:.2f}", flush=True)
        print(f"avg_player_lives (completed): {agg['avg_player_lives']:.2f}", flush=True)
        if agg["avg_opponent_lives"] is not None:
            print(f"avg_opponent_lives (completed): {agg['avg_opponent_lives']:.2f}", flush=True)

    # Wb (exp13 W0a): the arena ruler's headline. Absent (not zero) under
    # versus rules -- see `_compute_aggregate`.
    if agg.get("mean_trophies") is not None:
        ci = agg["mean_trophies_ci"]
        print(
            f"mean_trophies (completed): {agg['mean_trophies']:.3f}  "
            f"95% cluster-CI (by game): [{ci['lo95']:.3f}, {ci['hi95']:.3f}]",
            flush=True,
        )
        print(
            f"completion_rate (reached {MAX_TROPHIES} trophies, completed): "
            f"{agg['completion_rate']:.3f}",
            flush=True,
        )
        print(f"trophies_histogram (completed): {agg['trophies_histogram']}", flush=True)

    print(
        f"fallback_rate_games (all): {fb['rate_games']:.3f} ({fb['n_games_with_fallback']}/{n})  "
        f"fallback_turns_total: {fb['total_fallback_turns']}",
        flush=True,
    )
    su = agg["search_used"]
    print(
        f"search_used_rate_games (all): {su['rate_games']:.3f} ({su['n_games_with_search_used']}/{n})  "
        f"search_used_turns_total: {su['total_search_used_turns']}",
        flush=True,
    )
    sw = agg.get("search_width")
    if sw:
        ratio = "n/a" if sw["dedup_ratio"] is None else f"{sw['dedup_ratio']:.2f}"
        print(
            f"search_width: requested={sw['requested_candidates']} "
            f"dedup_mean={sw['dedup_mean']:.2f} dedup_median={sw['dedup_median']:.1f} "
            f"dedup_min={sw['dedup_min']} dedup_max={sw['dedup_max']} "
            f"dedup_ratio={ratio} searched_turns={sw['n_searched_turns']}",
            flush=True,
        )
    # Wd (exp13 A1): absent (not zeroed) under the determinized frame --
    # see `_honest_frame_aggregate`.
    seg = agg.get("segments")
    if seg:
        print(
            f"turn_mode: {agg['turn_mode']}  "
            f"mean_segments_per_turn: {seg['mean_segments_per_turn']:.3f} "
            f"({seg['n_segments']}/{seg['n_turns']} turns)  "
            f"mean_searches_per_game: {seg['mean_searches_per_game']:.2f}  "
            f"segments_histogram: {seg['segments_histogram']}",
            flush=True,
        )
        print(f"boundary_reason_counts: {seg['boundary_reason_counts']}", flush=True)
    print(f"end_reason_distribution (all): {agg['end_reason_distribution']}", flush=True)
    if agg["chain_length"] is not None:
        cl = agg["chain_length"]
        print(f"chain_length_min_median_max (all): {cl['min']}/{cl['median']:.1f}/{cl['max']}", flush=True)
    print(f"stop_reason_distribution (all): {agg['stop_reason_distribution']}", flush=True)


class TeacherRecordWriter:
    """Gzip JSONL sink for `--teacher-record-out` that closes ONE COMPLETE
    GZIP MEMBER PER GAME.

    exp12 Wa, codex review finding 1. The first version of this streamed a
    single `gzip.open(path, "at")` handle for the whole process and called
    `flush()` after every row, on the reasoning that a killed shard would
    then "end at the last fully-written row". It does not:
    `GzipFile.flush()` emits a Z_SYNC_FLUSH boundary but never the member
    TRAILER (crc32 + isize), so a process that dies leaves an unterminated
    member -- and `gzip.open`-based readers (`check_teacher_record.iter_rows`)
    raise `EOFError` when they reach it. Measured on this box: 5 rows written
    and flushed, process killed, `gzip.open(...).read()` ->
    `EOFError: Compressed file ended before the end-of-stream marker was
    reached`. The per-row durability claim was therefore backwards -- the
    file was all-or-nothing, and the "nothing" case was the normal one for
    any shard that got killed.

    Framing per GAME fixes the contract at the granularity the data actually
    has: every game that FINISHED is a self-contained, trailer-terminated
    member, python's gzip reads concatenated members transparently, and only
    the game in flight can be short. The per-row `flush()` is kept ON TOP of
    that, so the in-flight game's completed rows still reach the disk and are
    salvageable by `iter_rows`'s truncation path.

    Cost, measured on the real A0 smoke record (3 games, 33 rows, 1.86 MB of
    JSON): 81,649 B as one stream vs 83,177 B framed per game -- +1,528 B,
    +1.9%, ~509 B per game against ~27 KB of payload per game. Almost all of
    it is the fresh deflate dictionary each member starts with (the header +
    trailer are only 18 B of it), so the ratio falls as games get longer.

    `write` opens the current member lazily, so a game that emits no rows
    costs no member at all.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh: Any = None
        self.members_written = 0

    def write(self, text: str) -> None:
        if self._fh is None:
            self._fh = gzip.open(self.path, "at", encoding="utf-8")
        self._fh.write(text)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def end_game(self) -> None:
        """Close the current member, writing its gzip trailer. Idempotent, so
        the caller's `finally` can fire on a game that wrote nothing."""
        fh, self._fh = self._fh, None
        if fh is None:
            return
        fh.close()
        self.members_written += 1

    def close(self) -> None:
        self.end_game()


def run_versus_eval(
    bc: BcRecommender | SearchRecommender,
    opp_source: ChainSnapshotSource,
    opening: VariedOpeningSource | FixedOpeningSource | dict[str, Any],
    *,
    num_games: int,
    max_turn: int = DEFAULT_MAX_TURN,
    seed: int = DEFAULT_SEED,
    parse_cache: dict[str, Any] | None = None,
    out_fh: Any = None,
    log: bool = True,
    capture_detail: bool = False,
    game_index_start: int = 0,
    opponent_mode: str = OPPONENT_MODE_CHAIN,
    arena_source: ChainSnapshotSource | None = None,
    afterstate_out_fh: Any = None,
    teacher_writer: TeacherRecordWriter | None = None,
    segment_writer: TeacherRecordWriter | None = None,
    game_rules: str = GAME_RULES_VERSUS,
    arena_race_convention: str = DEFAULT_ARENA_RACE_CONVENTION,
    turn_mode: str = DEFAULT_TURN_MODE,
) -> list[dict[str, Any]]:
    """Play `num_games` full games and return their result dicts.

    Factored out of `main` (We2). Behavior for the plain N-game path is
    unchanged from We1's inline loop: same per-game progress line, same
    optional `--out` JSONL write.

    `opening` (full-game frame fix): passed straight through to
    `play_one_game` on every iteration -- see that function's docstring for
    the accepted types (an opening-source object, the new default from
    `main()`, or a raw dict for backward compatibility with existing direct
    callers).

    `parse_cache` (FIX 2, default None): passed straight through to every
    `play_one_game` -> `resolve_end_turn_with_sampled_battle`. It is left
    None here and by `main`, and this loop does NOT lazily create a shared
    `{}` for it, because the chain snapshot ships EVERY row pre-parsed
    (`sampled["parsed_state"]` is always a dict), so `end_turn.py` never
    READS the cache -- yet it still deep-copies each parsed_state INTO a
    non-None cache on every turn, which over N~=300 games x ~10 turns is
    thousands of retained parsed-state copies (unbounded growth, pure waste).
    None makes that write a no-op. (A caller with a NON-preparsed source can
    still pass its own dict.)

    `capture_detail` (We2, default False): passed straight through to every
    `play_one_game` call. `main` only turns it on when `--render-dir` is
    set -- see `render_selected_games`'s docstring for why rendering reads
    directly off THIS pass's `results` instead of re-running specific games
    afterward (the battle oracle is not reproducible across separate calls,
    so a re-run cannot be trusted to reproduce the exact game these results
    already describe).

    `game_index_start` (We8, exp12 W0, default 0 = unchanged): the loop runs
    `range(start, start + num_games)` instead of `range(num_games)`.
    EVERYTHING that varies per game -- the opening (`state_for_game`), the
    engine seed, the initial pid, the arena opponent RNG -- is a pure
    function of `game_index` (plus `seed`), so N games split into disjoint
    index windows across processes reproduce exactly the same N games one
    process would have played. That is what makes an N=1000 stage shardable
    across this box's cores without changing the frame.

    `opponent_mode` / `arena_source` (We8, exp12 W0): passed straight
    through to every `play_one_game` -- see that function's docstring.

    `afterstate_out_fh` (We9, exp12 W1'a, default None): an already-open
    file handle (`main` opens it in APPEND mode -- see `--afterstate-out`'s
    help). For each game this builds a fresh closure (mirrors
    `gen_distill_dataset.py`'s own `on_turn`-wrapping pattern) that injects
    THIS game's `game_index` into every row `play_out_game`'s
    `afterstate_sink` hands it and writes + flushes it immediately -- rows
    are never accumulated in memory, on top of `play_out_game` itself never
    buffering them (see module docstring "We9 addition").

    `teacher_writer` (Wa, exp12 route a, default None): the same per-game
    closure pattern for the rollout-teacher stream, over a
    `TeacherRecordWriter` rather than a bare file handle -- the rows carry N
    full boards each, so that stream is GZIP (uncompressed JSONL is not an
    option on this box's data disk), and gzip needs FRAMING to survive a
    kill. Rows are still written + flushed per row; the writer additionally
    closes one complete gzip member per game (see its docstring), which is
    why this loop owns a `finally: teacher_writer.end_game()`.
    """
    # Wc (exp13 W0b, codex review P2): refuse the unsupported rulers ONCE,
    # before the first game, rather than N times from inside the loop -- see
    # `_game_rules_refusal`. `play_one_game` re-checks per game (it is its
    # own entry point); the check is a string compare, so the duplication
    # costs nothing and neither door is left open.
    _assert_game_rules_supported(
        bc, game_rules=game_rules, opponent_mode=opponent_mode, turn_mode=turn_mode
    )

    results: list[dict[str, Any]] = []
    start_index = int(game_index_start)
    for game_index in range(start_index, start_index + int(num_games)):
        afterstate_sink = None
        if afterstate_out_fh is not None:

            def afterstate_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:
                payload = {"game_index": int(_game_index), **row}
                afterstate_out_fh.write(json.dumps(payload, sort_keys=True) + "\n")
                afterstate_out_fh.flush()

        teacher_sink = None
        if teacher_writer is not None:

            def teacher_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:
                payload = {"game_index": int(_game_index), **row}
                teacher_writer.write(json.dumps(payload, sort_keys=True) + "\n")
                teacher_writer.flush()

        segment_sink = None
        if segment_writer is not None:

            def segment_sink(row: dict[str, Any], *, _game_index: int = game_index) -> None:
                # Call-time import: segment recording is experiment tooling, not part of the public release.
                from .exp13_w1_records import build_segment_record

                payload = build_segment_record(row, game_id=int(_game_index))
                segment_writer.write(json.dumps(payload, sort_keys=True) + "\n")
                segment_writer.flush()

        t_game = time.monotonic()
        try:
            result = play_one_game(
                bc,
                opp_source,
                opening,
                game_index=game_index,
                max_turn=max_turn,
                seed=seed,
                parse_cache=parse_cache,
                capture_detail=capture_detail,
                opponent_mode=opponent_mode,
                arena_source=arena_source,
                afterstate_sink=afterstate_sink,
                teacher_sink=teacher_sink,
                segment_sink=segment_sink,
                game_rules=game_rules,
                arena_race_convention=arena_race_convention,
                turn_mode=turn_mode,
            )
        finally:
            # Close THIS game's gzip member (writing its trailer) whether the
            # game finished or blew up -- see `TeacherRecordWriter`.
            if teacher_writer is not None:
                teacher_writer.end_game()
            if segment_writer is not None:
                segment_writer.end_game()
        elapsed = time.monotonic() - t_game
        if log:
            print(
                f"game_done:idx={game_index}:pid={result['followed_pid']}:win={result['win']}:"
                f"turns_survived={result['turns_survived']}:end_reason={result['end_reason']}:"
                f"player_lives={result['player_lives']}:opponent_lives={result['opponent_lives']}:"
                # Wb (exp13 W0a): the arena ruler's headline metric, on the
                # live progress line a 24h+ generation shard is watched by.
                # None under versus rules, where no trophy exists.
                f"trophies={result['trophies']}:"
                f"num_fallbacks={result['num_fallbacks']}:elapsed={elapsed:.1f}s",
                flush=True,
            )
        results.append(result)
        if out_fh is not None:
            out_fh.write(json.dumps(result, sort_keys=True) + "\n")
            out_fh.flush()
    return results


def select_representative_games(results: list[dict[str, Any]], k: int) -> list[tuple[str, int]]:
    """Pick up to `k` game indices for `--render-games` (We2, PLAN.md "Run +
    deliverables": "Render 3 FULL games ... turn-by-turn").

    If ANY game won, always include the first win (a 5-game smoke went 0/5,
    so a win is the rarer, more informative case -- see the task background).
    Remaining slots are filled from the games that did NOT win, spread
    across `turns_survived` (worst, then best, then median of what is left)
    so the gallery shows the RANGE of how this checkpoint's games actually
    go, not three near-duplicates. Returns `(reason, game_index)` pairs in
    the order they should be rendered.
    """
    n = len(results)
    if n == 0 or k <= 0:
        return []

    chosen: list[tuple[str, int]] = []
    used: set[int] = set()

    win_idx = next((i for i, r in enumerate(results) if r["win"]), None)
    if win_idx is not None:
        chosen.append(("win", win_idx))
        used.add(win_idx)

    labels = ["worst", "best", "median"]
    slot = 0
    while len(chosen) < k:
        pool = sorted((i for i in range(n) if i not in used), key=lambda i: (results[i]["turns_survived"], i))
        if not pool:
            break
        if slot == 0:
            pick = pool[0]
        elif slot == 1:
            pick = pool[-1]
        else:
            pick = pool[len(pool) // 2]
        chosen.append((labels[min(slot, len(labels) - 1)], pick))
        used.add(pick)
        slot += 1

    return chosen[:k]


def _replay_order_helpers() -> tuple[Callable[[Any], list[Any]], Callable[[Any], list[Any]]]:
    """Lazy import of `eval_tempo_planner`'s pet-row reorientation helpers.

    `render_replay_image_from_calc_rows` (replay-bot bridge) expects the
    OPPOSITE slot order from both engine `team` lists and replaybot-parsed
    `playerPets`/`opponentPets` lists (see both helpers' own one-line
    comments in `eval_tempo_planner.py`). That reversal was already solved
    once there and is verified by its own production gallery
    (RESULTS_W4a.md's 3-case spot-check), so this reuses it verbatim rather
    than re-deriving the orientation. Imported lazily (only when
    `--render-dir` is used) so the N~=300-game metrics path never pays for
    `eval_tempo_planner`'s wider import surface (tempo predictor/value-model
    modules, unused here).
    """
    from .eval_tempo_planner import _replay_order_from_parsed_pets, _replay_order_from_state_team

    return _replay_order_from_state_team, _replay_order_from_parsed_pets


def _pretty_pet_name(slot: dict[str, Any]) -> str | None:
    """A readable pet name for a sidecar board row: the explicit `pet_name`
    if the engine slot carries one, else the `pet_id` de-prefixed
    (`pet-fairy-armadillo` -> `Fairy Armadillo`) so the sidecar reads in
    display names, not raw ids (the PNG already shows the sprite)."""
    name = slot.get("pet_name") or slot.get("pet_name_id")
    if isinstance(name, str) and name.strip():
        return name.strip()
    pet_id = slot.get("pet_id")
    if isinstance(pet_id, str) and pet_id.strip():
        base = pet_id.strip()
        if base.startswith("pet-"):
            base = base[len("pet-"):]
        return base.replace("-", " ").replace("_", " ").title() or pet_id.strip()
    return None


def _engine_board_summary(team: Any) -> list[dict[str, Any]]:
    """Compact per-pet summary of an engine-format board (`state.team` slots)
    for the sidecar JSON. Front-to-back, empty slots dropped."""
    out: list[dict[str, Any]] = []
    for slot in team or []:
        if not isinstance(slot, dict):
            continue
        name = _pretty_pet_name(slot)
        if not name:
            continue
        out.append(
            {
                "name": name,
                "attack": slot.get("attack"),
                "health": slot.get("health"),
                "level": slot.get("level"),
                "equipment": slot.get("equipment_id"),
            }
        )
    return out


def _calc_opponent_summary(opponent_pets: Any) -> list[dict[str, Any]]:
    """Compact per-pet summary of a calculator-format opponent board
    (`parsed_state.opponentPets`) for the sidecar JSON."""
    out: list[dict[str, Any]] = []
    for pet in opponent_pets or []:
        if not isinstance(pet, dict):
            continue
        name = pet.get("name")
        if not name:
            continue
        equip = pet.get("equipment")
        equip_name = equip.get("name") if isinstance(equip, dict) else equip
        out.append(
            {
                "name": str(name),
                "attack": pet.get("attack"),
                "health": pet.get("health"),
                "equipment": equip_name,
            }
        )
    return out


def _render_heart_lives(outcomes: list[str], *, max_lives: int = 6) -> list[int]:
    """The exact per-row heart value the whole-game PNG shows, replicated in
    Python for the sidecar cross-check.

    This mirrors `SAP-Replay-Bot/lib/render.js`'s `renderReplayImage` life
    model EXACTLY (read directly before relying on it): `currentLives` starts
    at `maxLives`; at row index 2 (turn 3) it regains 1 if below max; the
    value is DRAWN (lives going INTO that turn's battle); THEN a LOSS
    decrements it by 1 (WIN/DRAW leave it). That model is identical to this
    driver's own versus life bookkeeping (`end_turn.py`: -1 life per loss,
    draws free, turn-3 +1 recovery), which is WHY feeding the real per-turn
    outcome sequence to the renderer reproduces the true lives race with no
    explicit lives field -- verified on 3 real games (the drawn sequence ends
    at each game's captured final `player_lives`)."""
    lives = int(max_lives)
    drawn: list[int] = []
    for i, outcome in enumerate(outcomes):
        if i == 2 and lives < max_lives:
            lives += 1
        drawn.append(lives)
        if str(outcome).strip().lower() == "loss":
            lives -= 1
    return drawn


def _full_game_rows_and_sidecar(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (calc_rows, sidecar_turns) for ONE whole-game image.

    ONE row per turn = that turn's END-OF-TURN board that was actually scored
    (`detail["board_pre_battle"]`, player left) vs the duel opponent it fought
    (`detail["parsed_state"]["opponentPets"]`, right), carrying that turn's
    real W/L outcome. The renderer stacks the rows into one image and derives
    the per-row heart lives from the outcome sequence (see
    `_render_heart_lives`), which is the exp08 `full_game_<gid>.png` format.

    Sidecar turns carry the per-turn text the plan wants alongside the PNG:
    the chain the policy played, the sap-calculator link, and the start/end
    lives (plus start/end board summaries so the board transformation is
    legible without the PNG). NOTE: `chain` here is the action-TYPE sequence
    (`chain_types`) only -- this run's capture did not record each op's
    shop_index/team_index params; the start_board -> end_board delta shows
    what the chain achieved.
    """
    replay_order_state_team, replay_order_parsed_pets = _replay_order_helpers()

    rows: list[dict[str, Any]] = []
    sidecar: list[dict[str, Any]] = []
    for entry in result.get("per_turn", []):
        detail = entry.get("detail") or {}
        board_pre_battle = detail.get("board_pre_battle")
        parsed_state = detail.get("parsed_state")
        battle = detail.get("battle") if isinstance(detail.get("battle"), dict) else {}
        state_before = detail.get("state_before")
        # A whole-game row needs the scored board + the fought opponent. Every
        # turn of a COMPLETED game has both; a turn that ended the game on an
        # infrastructure failure (no battle) is skipped so it doesn't inject a
        # blank row that would desync render.js's row-index turn numbering.
        if not isinstance(board_pre_battle, dict) or not isinstance(parsed_state, dict):
            continue

        outcome = str(entry.get("outcome") or battle.get("outcome") or "unknown")
        rows.append(
            {
                "turn": int(entry.get("turn", len(rows) + 1)),
                "turnLabel": "",
                "outcome": outcome,
                "opponentName": "",
                "playerPets": replay_order_state_team(board_pre_battle.get("team")),
                "opponentPets": replay_order_parsed_pets(parsed_state.get("opponentPets")),
            }
        )
        sidecar.append(
            {
                "turn": int(entry.get("turn", len(sidecar) + 1)),
                "outcome": outcome,
                "player_lives_after_battle": entry.get("lives"),
                "opponent_lives_after_battle": entry.get("opp_lives"),
                "chain_types": entry.get("chain_types"),
                "stop_reason": entry.get("stop_reason"),
                "calc_link": battle.get("calculator_link"),
                "start_board": _engine_board_summary(
                    state_before.get("team") if isinstance(state_before, dict) else None
                ),
                "end_board": _engine_board_summary(board_pre_battle.get("team")),
                "opponent_board": _calc_opponent_summary(parsed_state.get("opponentPets")),
            }
        )

    heart_lives = _render_heart_lives([r["outcome"] for r in rows])
    for turn_entry, hl in zip(sidecar, heart_lives):
        # The value the heart actually shows on this row (lives going INTO the
        # battle, render.js convention) -- distinct from
        # `player_lives_after_battle` (post-battle), both kept for the cross-check.
        turn_entry["heart_lives_shown"] = hl
    return rows, sidecar


def render_selected_games(
    results: list[dict[str, Any]],
    *,
    k: int,
    out_dir: Path,
) -> dict[str, Any]:
    """Render `k` representative games as ONE WHOLE-GAME image each (exp08
    `full_game_<gid>.png` format), plus a per-game sidecar JSON.

    FORMAT (project-owner correction, matches
    an internal analysis script): each game is a SINGLE
    image, ONE ROW PER TURN down the page -- that turn's end-of-turn board
    (player left, opponent right), the W/L for the turn (green row bg for WIN,
    red for LOSS, grey for TIE), and the ACTUAL player lives that turn in the
    heart. The heart shows the lives RACE across the game because all of a
    game's turns are rendered in a SINGLE `render_replay_image_from_calc_rows`
    call: `render.js` carries one running `currentLives` down the rows (start
    6, +1 at turn 3, -1 per LOSS -- see `_render_heart_lives`), so the heart
    counts down exactly as the game's lives did. (The earlier per-turn-PNG
    version rendered each turn in its own call, which reset `currentLives`
    every image and flattened the race to a constant 6 -- that was the bug
    this correction fixes.)

    Reads DIRECTLY off `results` (the captured per-turn detail) -- it does NOT
    re-run any game. Re-running cannot reproduce a specific past game because
    the battle oracle (`run_battle_oracle_with_config`, `simulationCount=1`)
    is an UNSEEDED Monte Carlo draw (verified: the byte-identical board config
    gave 18 draws / 2 wins over 20 calls), so the ONLY faithful source is the
    detail captured during the original playthrough
    (`run_versus_eval(..., capture_detail=True)`, or a prior run's
    `games.jsonl` via `--render-from-jsonl`).

    Outputs (in `<out_dir>/images/`): `full_game_<pid8>.png` (whole-game
    image, `pid8` = initial followed pid) + `full_game_<pid8>.json` (per-turn
    chain/calc-link/lives sidecar).
    """
    from ..opponents import render_replay_image_from_calc_rows

    chosen = select_representative_games(results, k)
    print(
        f"render_games_selected: "
        f"{[(label, idx, results[idx]['followed_pid']) for label, idx in chosen]}",
        flush=True,
    )
    if not chosen:
        return {"k_requested": k, "games": []}

    images_dir = Path(out_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    games_report: list[dict[str, Any]] = []
    for label, game_index in chosen:
        detailed = results[game_index]
        pid8 = "".join(
            ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(detailed["followed_pid"])[:8]
        )
        rows, sidecar_turns = _full_game_rows_and_sidecar(detailed)

        # exp09 W5 P1 fix: the filename carries the game index too. Two
        # selected games can follow the SAME pid (observed in the W5 P0
        # held-out gallery: median games 102 and 105 both followed
        # 2410227f..., so the second render silently overwrote the first and
        # 12 selected games left only 11 PNGs on disk). `game_index` is
        # unique within a run, making collisions impossible.
        stem = f"full_game_{int(game_index):03d}_{pid8}"
        png_path = images_dir / f"{stem}.png"
        json_path = images_dir / f"{stem}.json"
        render_error: str | None = None

        if not rows:
            render_error = "no_scored_turns_to_render"
        else:
            # No header/odds -> matches the exp08 look exactly (bare stacked
            # rows). max_lives=6 is the versus starting life the heart counts
            # down from.
            rendered = render_replay_image_from_calc_rows(
                rows,
                max_lives=6,
                player_name=None,
                header_opponent_name=None,
                include_odds=False,
            )
            if not bool(rendered.get("ok", False)):
                render_error = str(rendered.get("error") or "render_failed")
            else:
                image = rendered.get("image")
                if not isinstance(image, (bytes, bytearray)):
                    render_error = "render_output_empty"
                else:
                    png_path.write_bytes(bytes(image))

        sidecar = {
            "reason": label,
            "game_index": game_index,
            "followed_pid": detailed["followed_pid"],
            "initial_followed_pid": detailed.get("initial_followed_pid", detailed["followed_pid"]),
            "final_followed_pid": detailed.get("final_followed_pid"),
            "win": detailed["win"],
            "player_lives": detailed["player_lives"],
            "opponent_lives": detailed["opponent_lives"],
            "turns_survived": detailed["turns_survived"],
            "end_reason": detailed["end_reason"],
            "num_fallbacks": detailed.get("num_fallbacks"),
            "fallback_turns": detailed.get("fallback_turns"),
            "n_rows_rendered": len(rows),
            "chain_params_captured": False,
            "chain_note": (
                "chain_types is the action-TYPE sequence only; this run did not "
                "capture per-op shop_index/team_index params. start_board -> "
                "end_board shows what the chain achieved."
            ),
            "image": str(png_path) if render_error is None else None,
            "render_error": render_error,
            "turns": sidecar_turns,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")

        if render_error is not None:
            print(f"render_full_game_error:game_index={game_index}:pid={pid8}:error={render_error}", flush=True)
        else:
            print(
                f"render_full_game_ok:reason={label}:game_index={game_index}:pid={pid8}:"
                f"rows={len(rows)}:png={png_path}",
                flush=True,
            )

        games_report.append(
            {
                "reason": label,
                "game_index": game_index,
                "followed_pid": detailed["followed_pid"],
                "win": detailed["win"],
                "player_lives": detailed["player_lives"],
                "opponent_lives": detailed["opponent_lives"],
                "turns_survived": detailed["turns_survived"],
                "end_reason": detailed["end_reason"],
                "image": str(png_path) if render_error is None else None,
                "sidecar": str(json_path),
                "n_rows_rendered": len(rows),
                "heart_lives_shown": [t["heart_lives_shown"] for t in sidecar_turns],
                "render_error": render_error,
            }
        )

    return {"k_requested": k, "games": games_report}


def load_results_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load per-game result dicts from a `--out` JSONL (one game per line),
    ordered by `game_index`. Lets `--render-from-jsonl` re-render an ALREADY
    captured run's galleries without replaying anything (the only faithful
    way -- see `render_selected_games`)."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("game_index", 0)))
    return rows


# ---------------------------------------------------------------------------
# exp11 W1: --recommender llm CLI wiring
# ---------------------------------------------------------------------------


def build_mock_end_turn_client() -> MockLlmClient:
    """Hidden CLI escape hatch (`--llm-provider mock`, both eval harnesses --
    `eval_tempo_planner.py`'s own `main` imports this): a `MockLlmClient`
    that always answers with exactly one `submit_chain([{"type":
    "END_TURN"}])` tool call, regardless of what it is asked -- the same
    tool-call shape `test_llm_agent_loop.py` scripts by hand. Lets the
    MockLLM E2E smoke (PLAN.md W1 "Check") exercise the FULL CLI (report
    JSON, transcript, render gallery) for zero API cost and zero network
    calls, with no bespoke script needed per caller. Never used by a unit
    test (those construct their own scripted `MockLlmClient`, see
    `test_llm_agent_loop.py`) and never used by a real run.
    """
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from ..llm_agent.client import (
        LlmResponse,
        LlmUsage,
        MockLlmClient,
        ToolCallRequest,
    )


    def _always_end_turn(call_index: int, **_kwargs: Any) -> LlmResponse:
        return LlmResponse(
            text=None,
            tool_calls=[
                ToolCallRequest(
                    id=f"mock-end-turn-{call_index}", name="submit_chain", input={"actions": [{"type": "END_TURN"}]}
                )
            ],
            stop_reason="tool_calls",
            usage=LlmUsage(input_tokens=0, output_tokens=0),
            model="mock-llm-always-end-turn",
            raw={},
        )

    return MockLlmClient(_always_end_turn, model="mock-llm-always-end-turn")


def _resolve_llm_transcript_path(args: argparse.Namespace) -> Path | None:
    """`--llm-transcript` default: next to `--out` (`<out>.transcript.jsonl`)
    whenever `--out` is given; transcripting stays off (`None`) if neither
    is set. An explicit `--llm-transcript` always wins over the derived
    default."""
    if args.llm_transcript is not None:
        return Path(args.llm_transcript)
    if args.out is not None:
        return Path(str(args.out) + ".transcript.jsonl")
    return None


def _build_llm_agent_config(args: argparse.Namespace, *, transcript_path: Path | None) -> LlmAgentConfig:
    """W1 review fix 2: every `--llm-*` override flag defaults to `None` at
    the parser (see each flag's own `add_argument`) and is applied here ONLY
    when it was actually given -- a flag left at its default must never
    clobber a value `--llm-config` already loaded from its JSON file. Before
    this fix, `--llm-fallback`/`--llm-cache-mode` had non-None argparse
    defaults ("end_turn"/"off") and were assigned unconditionally, so a
    config file's own `fallback_mode`/`llm_cache_mode` was silently
    overwritten by those defaults on every run that didn't repeat them on
    the CLI -- exactly the case a config file exists to avoid repeating.
    `transcript_path` follows the same rule: only overwrite when the
    resolved path (explicit `--llm-transcript`, or `--out`-derived --
    `_resolve_llm_transcript_path`) is non-None, so a bare
    `--llm-config`-only run keeps whatever `transcript_path` the file set.
    """
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from .llm_recommender import LlmAgentConfig

    config = LlmAgentConfig.from_json_file(args.llm_config) if args.llm_config is not None else LlmAgentConfig()
    if args.llm_model is not None:
        config.model = str(args.llm_model)
    if args.llm_fallback is not None:
        config.fallback_mode = str(args.llm_fallback)
    if args.llm_cache_dir is not None:
        config.llm_cache_dir = str(args.llm_cache_dir)
    if args.llm_cache_mode is not None:
        config.llm_cache_mode = str(args.llm_cache_mode)
    if transcript_path is not None:
        config.transcript_path = str(transcript_path)
    if args.llm_provider is not None:
        config.provider = str(args.llm_provider)
    return config



def _is_llm_recommender(recommender: object) -> bool:
    """True for `LlmRecommender`, identified by capability rather than class.

    `llm_recommender` is imported lazily inside `_build_llm_recommender` so that
    importing this module does not pull the LLM stack, which is what the public
    release needs. The report block below runs on EVERY evaluation, so it cannot
    name the class: doing so raised `NameError` on every run. `cost_totals` and
    `fallback_reason_counts` are defined only on `LlmRecommender`.
    """
    return hasattr(recommender, "cost_totals") and hasattr(recommender, "fallback_reason_counts")

def _build_llm_recommender(config: LlmAgentConfig, *, bc: BcRecommender | None) -> LlmRecommender:
    """`--recommender llm`'s construction, matching `LlmRecommender`'s actual
    constructor (part A, `tools/llm_recommender.py`): a fresh
    `GameMemoryStore` (a real, whole-games full-game eval keeps cross-turn
    scratchpad memory ON, unlike the one-shot `eval_tempo_planner.py` harness
    -- see that file's own `memory=None` wiring), `fallback=bc.recommend`
    only when a `BcRecommender` was actually constructed (`--llm-fallback
    bc`), and an explicit `MockLlmClient` when `--llm-provider mock` asks for
    the zero-cost escape hatch (`LlmRecommender` cannot build a mock client
    from `config` alone -- `provider="mock"` requires an explicit `client=`,
    see that class's own `_build_client`).

    `config` is built by the CALLER now (`main()`, once, before the
    `needs_bc` gate -- see that gate's own comment for why), not rebuilt in
    here -- W1 review fix 2's `needs_bc` correction needs the fully-resolved
    config (file + overrides) to decide whether a config-file-only
    `fallback_mode="bc"` needs a `BcRecommender` loaded, which is only
    knowable after `_build_llm_agent_config` runs.
    """
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from ..llm_agent.memory import GameMemoryStore
    from ..llm_agent.transcript import TranscriptWriter
    from .llm_recommender import PROVIDER_MOCK, LlmRecommender

    transcript_path = Path(config.transcript_path) if config.transcript_path else None
    client = build_mock_end_turn_client() if config.provider == PROVIDER_MOCK else None
    recommender = LlmRecommender(
        config,
        client=client,
        memory=GameMemoryStore(),
        fallback=(bc.recommend if bc is not None else None),
        transcript=(TranscriptWriter(transcript_path) if transcript_path is not None else None),
    )
    return recommender


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "exp09 W4c We1: play N full versus games with a BC MaskablePPO checkpoint "
            "(engine-true masked, visited-state-guarded, deterministic decode via "
            "BcRecommender) against real replay-chain opponents sourced from a local "
            "chain snapshot, and report full-game metrics."
        )
    )
    parser.add_argument("--bc-checkpoint", type=Path, default=Path(DEFAULT_BC_CHECKPOINT))
    parser.add_argument("--chain-snapshot", type=Path, default=Path(DEFAULT_CHAIN_SNAPSHOT))
    parser.add_argument(
        "--recommender", type=str, default="bc", choices=("bc", "search", "llm"),
        help="exp09 W6a: \"bc\" (default, fully backward compatible) plays the plain "
        "BcRecommender decode. \"search\" wraps it in "
        "tools/search_recommender.py::SearchRecommender -- a best-of-N chain search "
        "reranked by the battle oracle against the last-seen opponent (see that "
        "module's docstring); --search-candidates/--search-ksim tune it. exp11 W1: "
        "\"llm\" plays tools/llm_recommender.py::LlmRecommender (a tool-calling LLM "
        "agent loop) instead -- see the --llm-* flags below; --bc-checkpoint is only "
        "loaded for \"llm\" when --llm-fallback bc is also given.",
    )
    parser.add_argument(
        "--search-candidates", type=int, default=SEARCH_DEFAULT_N_CANDIDATES,
        help="--recommender search only: SearchRecommender's n_candidates, i.e. the "
        "candidate WIDTH of stage 1 (chain generation + end-board dedup + one ksim "
        "oracle call per distinct end board). Applies to BOTH --search-scoring modes: "
        "under rollout scoring this is the pool the --rollout-shortlist is taken FROM, "
        "so raising it widens what the rollout stage gets to choose among without "
        "changing the rollout budget (exp12 W2c width curve). Every report echoes the "
        "requested width and the per-turn dedup distribution it actually produced "
        "(report key search_width).",
    )
    parser.add_argument(
        "--search-ksim", type=int, default=SEARCH_DEFAULT_KSIM,
        help="--recommender search only: SearchRecommender's ksim (oracle "
        "simulation_count per candidate).",
    )
    parser.add_argument(
        "--search-scoring", type=str, default=SEARCH_DEFAULT_SCORING, choices=list(SEARCH_SCORING_MODES),
        help="exp09 W1: --recommender search only. \"myopic\" (default, fully backward "
        "compatible -- byte-for-byte the existing W6a behavior): rerank the n_candidates "
        "chains by ONE ksim oracle call against the LAST-SEEN opponent board. \"rollout\": "
        "additionally simulate the REST OF THE GAME for the top --rollout-shortlist "
        "candidates (by myopic score) against the actual recorded opponent chain, "
        "--rollout-repeats times each, and rerank by mean(final outcome) + a lives-diff "
        "tiebreak -- see search_recommender.py's module docstring for the full algorithm.",
    )
    parser.add_argument(
        "--rollout-shortlist", type=int, default=SEARCH_DEFAULT_ROLLOUT_SHORTLIST,
        help="--search-scoring rollout only: how many of the myopic-ranked deduped "
        "candidates get a full rest-of-game rollout.",
    )
    parser.add_argument(
        "--rollout-repeats", type=int, default=SEARCH_DEFAULT_ROLLOUT_REPEATS,
        help="--search-scoring rollout only: how many times each shortlisted candidate's "
        "rest-of-game is simulated (battle draws are stochastic; the score is the mean "
        "over these repeats).",
    )
    parser.add_argument(
        "--rollout-ksim", type=int, default=SEARCH_DEFAULT_ROLLOUT_KSIM,
        help="--search-scoring rollout only: simulation_count used to resolve a "
        "shortlisted candidate's OWN turn (a steadier signal for the turn actually being "
        "decided). Every SUBSEQUENT simulated turn in that same continuation reverts to "
        "the driver's normal single-draw mechanics (simulation_count=1) -- this flag "
        "does not affect them.",
    )
    parser.add_argument(
        "--rollout-opponent-mode", type=str, default=SEARCH_DEFAULT_ROLLOUT_OPPONENT_MODE,
        choices=list(SEARCH_ROLLOUT_OPPONENT_MODES),
        help="exp10 W2 (opponent-source ablation): --search-scoring rollout only. "
        "\"true\" (default, fully backward compatible -- byte-for-byte the existing W1 "
        "behavior): each rollout continuation replays the ACTUAL followed opponent's "
        "recorded rest-of-game (privileged, see search_recommender.py's module "
        "docstring). \"pool_random\": each continuation instead follows a RANDOM other "
        "pid's recorded chain, drawn from the same opponent pool (--opponent-split/"
        "--opponent-pack/--opponent-rank-max) and excluded from ever coinciding with the "
        "true pid. \"retrieval\": same as pool_random, additionally restricted to pids "
        "whose own chain reached at least the current turn. Measures how much of the "
        "rollout ceiling survives when the teacher cannot see the real future opponent.",
    )
    parser.add_argument(
        "--vgame-heads", type=Path, action="append", default=None,
        help="exp12 W2 (wave A4): --search-scoring vgame only. The trained V artifact "
        "(vgame_heads.pt) that replaces the search leaf. Repeatable, which makes the "
        "leaf an ENSEMBLE and is the only way --vgame-pessimism can mean anything "
        "(ensemble_std is identically 0 with one member, so a pessimism run with a "
        "single head is rejected rather than silently becoming a plain run). Required "
        "for, and only valid with, --search-scoring vgame.",
    )
    parser.add_argument(
        "--vgame-extractor", type=Path, default=None,
        help="exp12 W2: the ORIGIN BC checkpoint the V artifact pins by sha256 "
        "(default: --bc-checkpoint). The pin is enforced by load_vgame_model for both "
        "artifact kinds, so a mismatch fails at construction, never mid-run.",
    )
    parser.add_argument(
        "--vgame-blend", type=float, default=VGAME_DEFAULT_BLEND,
        help="exp12 W2: the frozen leaf formula is (1-blend)*V + blend*myopic - "
        "pessimism*ensemble_std. Default 0.0 = a pure V leaf, which is also what lets "
        "the stage-1 myopic ORACLE CALLS be skipped entirely (nothing reads them), and "
        "that skip is where the speed the W2 decision rule measures comes from. Any "
        "non-zero value turns those oracle calls back on.",
    )
    parser.add_argument(
        "--vgame-pessimism", type=float, default=VGAME_DEFAULT_PESSIMISM,
        help="exp12 W2: the pessimism coefficient on the ensemble standard deviation "
        "in the formula above. Default 0.0. Requires at least two --vgame-heads.",
    )
    parser.add_argument(
        "--num-games", type=int, default=None,
        help="number of versus games to play; required UNLESS --render-from-jsonl is given "
        "(that mode only re-renders an existing run's galleries and plays nothing)",
    )
    parser.add_argument(
        "--opponent-mode", type=str, default=OPPONENT_MODE_CHAIN, choices=list(OPPONENT_MODES),
        help="exp12 W0 (\"the ruler\"): WHICH opponent each turn is scored against. "
        "\"chain\" (default, byte-identical to every run before this flag existed): follow "
        "ONE real player's recorded game for the whole game, resampling only when their "
        "chain runs out of turns. \"arena\": the real game's pairing rule -- a FRESH "
        "opponent every turn, drawn uniformly from the arena pool at that turn. The arena "
        "pool is built from the SAME --chain-snapshot and --opponent-pack but with ALL "
        "splits and NO rank cap (--arena-pool-split and --arena-pool-rank-max are its "
        "OWN selectors); --opponent-split/--opponent-rank-max keep governing the chain "
        "pool, which in arena mode no opponent is drawn from at all. NOTE this "
        "governs the DRIVER's per-turn draw only -- --recommender search's internal "
        "rollout continuations still follow the chain pool (exp12 W0 scope).",
    )
    parser.add_argument(
        "--game-rules", type=str, default=GAME_RULES_VERSUS, choices=list(GAME_RULES),
        help="exp13 W0a (PLAN.md D1): WHICH GAME is played -- a separate axis from "
        "--opponent-mode, which picks WHO you play. \"versus\" (default, byte-identical "
        "to every run before this flag existed): 6 lives each, you win when the "
        "opponent's life bar empties. \"arena\": the REAL arena -- 5 starting lives, no "
        "opponent life bar, a battle win is a TROPHY, 10 trophies completes the run and "
        "IS the win (end_reason trophies_10), 0 lives loses, --max-turn still caps it as "
        "a non-win, and the turn-3 heal caps at 5. Requires --opponent-mode arena, and "
        "refuses --search-scoring rollout (see the module docstring for both).",
    )
    parser.add_argument(
        "--arena-race-convention", type=str, default=DEFAULT_ARENA_RACE_CONVENTION,
        choices=list(ARENA_RACE_CONVENTIONS),
        help="exp13 W0a (PLAN.md D2), --game-rules arena only: what fills the learned "
        "leaf's `opp_lives` bypass slot, which arena has no natural value for. "
        "\"trophies_mapped\" (default): min(6, 10 - trophies), i.e. distance to "
        "completing the run, mapped onto the feature that carried that semantic in "
        "versus. \"const6\": a constant 6, the null convention. Under BOTH, `wins` is "
        "the trophy count and `lives` is the true life count. W0c's probe pins one; "
        "every later round reuses the pinned value so V stays self-consistent with the "
        "frame it was trained under.",
    )
    parser.add_argument(
        "--turn-mode", type=str, default=DEFAULT_TURN_MODE, choices=list(TURN_MODES),
        help="exp13 PLAN Amendment A1: HOW a turn is played. \"whole-determinized\" "
        "(default, byte-identical to every run before this flag existed): decode one "
        "whole-turn chain and commit all of it. The engine seeds its RNG from the state, "
        "so a search layer replaying candidate chains SEES this turn's roll before "
        "deciding whether to roll -- foresight the human reference never had, which is "
        "why every pre-A1 agent number is a determinized-frame number. "
        "\"segmented-honest\": the recommender is handed an IMAGINED clone whose "
        "meta.seed is an independent derived stream (play itself stays on its own seeded "
        "stream, so reproducibility and --game-index-start sharding survive), and the "
        "chosen chain is committed op by op against the real board, stopping to "
        "re-search after any op whose engine transition reports a stochastic_reason. "
        "Requires --search-scoring vgame when searching (see --stochastic-samples). The "
        "determinized mode is kept runnable for the foresight-premium diagnostic and for "
        "reading exp12 history.",
    )
    parser.add_argument(
        "--capture-candidate-chains", action="store_true",
        help="exp13 A2.4, --recommender search only: persist per-candidate action "
        "chains in each turn's search_diagnostics (`candidate_chains` = one entry per "
        "scored group, `decoded_candidate_chains` = one entry per GENERATED candidate, "
        "before the prefix walk truncated it). Off by default because it multiplies "
        "report size; A2.4 requires it because the size of the decode saving it claims "
        "must be MEASURED rather than asserted, and per-candidate chain lengths are "
        "persisted nowhere else.",
    )
    parser.add_argument(
        "--completion-policy", default=SEARCH_COMPLETION_BC_GREEDY,
        choices=list(SEARCH_COMPLETION_POLICIES),
        help="exp22 W3, --turn-mode segmented-honest + --search-scoring vgame only: what "
        "fills in the rest of the turn behind a chance node WHILE RANKING. "
        "'bc_greedy' (default, every run before 2026-08-19) decodes one greedy BC "
        "completion, which lower-bounds the value of information because the real agent "
        "re-searches after the roll instead of playing greedily. 'v_search' proposes "
        "--completion-width completions and lets V pick, which is the fix "
        "search_recommender.py has named in its own 'Accepted residual biases' note "
        "since exp13 A1 and which no experiment had built.",
    )
    parser.add_argument(
        "--completion-width", type=int, default=1,
        help="exp22 W3: how many completions ONE imagined sample proposes under "
        "--completion-policy v_search; inner candidate 0 is always the greedy one, so "
        "the deepened arm keeps the outer search's 'never worse than greedy' property. "
        "Must be 1 for bc_greedy and >= 2 for v_search: a width-1 v_search arm would "
        "measure the old policy under the new name, which is the failure exp22's own "
        "extractor_travel gate exists to stop.",
    )
    parser.add_argument(
        "--completion-aggregate", default=SEARCH_COMPLETION_AGG_MAX,
        choices=list(SEARCH_COMPLETION_AGGREGATES),
        help="exp22 W3: how one imagined sample's completions collapse to one number. "
        "'max' IS the deepening. 'mean' is the CONTROL ARM: identical boards, identical "
        "compute, no optimism, so the difference between the two arms isolates 'taking a "
        "max over more estimates inflates them' from 'the search found a better "
        "continuation'. Inert under bc_greedy, where a sample has one board.",
    )
    parser.add_argument(
        "--mc-rerank-k", type=int, default=0,
        help="MC rerank (ledger A2.6): after V ranks the menu, re-rank its top K "
             "groups IN V ORDER by --rollout-repeats rollouts each, scored on "
             "trophies under arena rules. 0 (default) disables it and every run "
             "before exp22 is byte-identical. This is NOT --search-scoring rollout, "
             "which makes a rollout the leaf for every candidate (ledger C7 records "
             "that as a different design).",
    )
    parser.add_argument(
        "--stochastic-samples", type=int, default=DEFAULT_STOCHASTIC_SAMPLES,
        help="exp13 A1 section 3, --turn-mode segmented-honest + --search-scoring vgame "
        "only: k, how many imagined completions a candidate prefix ending at a chance "
        "node is scored over. The prefix's score is the MEAN of them, so search cannot "
        "pick whichever candidate happened to roll well; the resample keys carry "
        "(decision, r) and deliberately NOT the candidate index, so siblings at the same "
        "chance node face the same draw (common random numbers). Default 3; A1 ruling "
        "6(iii) pre-registers a k in {1,3,5} strength/wall-clock sweep before this is "
        "refrozen. k=1 is the cheap stopgap exp16's duel loop may run, and must be "
        "declared wherever its numbers appear.",
    )
    parser.add_argument(
        "--skip-imagined-validation", action="store_true",
        help="exp13 W0b' PERFORMANCE ONLY, off by default: skip jsonschema validation "
        "inside IMAGINED engine walks -- the BC decode that proposes a chain, search's "
        "candidate replay, prefix walks and the k resampled completions. The committed "
        "path (this driver's own op-by-op replay against the real board) always "
        "validates, so nothing that enters a game's history goes unchecked. Measured on "
        "the arm C profile shape: schema validation is 70.6%% of wall time and 98.9%% of "
        "api.step, and the validators are already compiled once, so this is the only "
        "lever left. It must not move a single number -- see the byte-identity probe in "
        "test_exp13_imagined_validation.py -- and the report records whether it was on.",
    )
    parser.add_argument(
        "--arena-pool-rank-max", type=int, default=None,
        help="exp12 W0, --opponent-mode arena only: rank-cap the ARENA pool (inclusive, "
        "same unranked-exclusion rule as --opponent-rank-max). Default None = no cap, i.e. "
        "the full pool -- a real lobby is not rank-filtered. Set it (e.g. 1500) for the "
        "\"rank-matched lobby\" bracket probe.",
    )
    parser.add_argument(
        "--arena-pool-split", type=str, default=None, choices=list(SPLIT_NAMES),
        help="exp13 W0b, --opponent-mode arena only: restrict the ARENA pool -- the pool "
        "every turn's opponent is actually drawn from -- to one snapshot split "
        "(\"train\"/\"val\"/\"test\"). Default None = every split, i.e. the pre-exp13 "
        "full pool, so existing arena runs are unchanged. --opponent-split does NOT do "
        "this: it filters the CHAIN pool, whose only remaining use in arena mode is the "
        "initial_pid draw the ruler discards. Set \"val\" for exp13's gate ruler, whose "
        "PLAN.md D1 contract is zero overlap with the train-split generation pool.",
    )
    parser.add_argument(
        "--game-index-start", type=int, default=0,
        help="exp12 W0: play game indices [start, start + --num-games) instead of "
        "[0, --num-games). Every per-game input (opening, engine seed, initial pid, arena "
        "opponent RNG) is a pure function of the game index, so disjoint windows run in "
        "parallel processes reproduce exactly the unsharded run -- this is how an N=1000 "
        "stage is sharded across cores on one frame.",
    )
    parser.add_argument("--max-turn", type=int, default=DEFAULT_MAX_TURN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--long-min", type=int, default=DEFAULT_LONG_MIN)
    parser.add_argument(
        "--opening-mode", type=str, default="varied", choices=("varied", "fixed"),
        help="Full-game frame fix (train/opening_source.py). \"varied\" (default) starts "
        "each game from a fresh, per-game_index-seeded engine roll (gold 10, empty team, "
        "before any purchase; engine._roll_shop_slots, the SAME roll mechanism a real "
        "turn-2+ advance uses), instead of always the same hardcoded FIXTURE_PATH shop. "
        "\"fixed\" restores the OLD single-fixture-every-game behavior (byte-for-byte), "
        "kept for parity/debug.",
    )
    parser.add_argument(
        "--opponent-pack", type=str, default=None,
        help="exp09 W5 Rev 6: keep only opponent chains whose game-level opponent_pack "
        "equals this (e.g. Turtle -- the turtle-only pool; the mixed snapshot is only "
        "~40%% Turtle opponents). Default None = all packs (every pre-Rev-6 run).",
    )
    parser.add_argument(
        "--opponent-rank-min", type=int, default=None,
        help="exp09 W5.3: keep only opponent chains whose game-level opponent_rank is "
        ">= this (inclusive). A game with no int opponent_rank is excluded whenever "
        "either rank bound is set (see ChainSnapshotSource's docstring). Default "
        "None = no lower bound.",
    )
    parser.add_argument(
        "--opponent-rank-max", type=int, default=None,
        help="exp09 W5.3: keep only opponent chains whose game-level opponent_rank is "
        "<= this (inclusive). See --opponent-rank-min for the unranked-exclusion "
        "rule. Default None = no upper bound.",
    )
    parser.add_argument(
        "--opponent-split", type=str, default=None, choices=list(SPLIT_NAMES),
        help="exp09 W5 P0: restrict the chain-snapshot opponent pool (initial followed pid "
        "+ fallback resampling, ChainSnapshotSource's `split=`) to one held-out split "
        "(\"train\"/\"val\"/\"test\" -- see chain_snapshot.py's 'Held-out pid split' and "
        "PLAN.md). Default None = ALL pids, unrestricted -- backward-compatible with every "
        "existing W4c run, which predates the split concept. Use \"test\" for a fair W5 "
        "floor/eval that never draws an opponent a chain_snapshot-mode PPO run trained "
        "against (that uses \"train\").",
    )
    parser.add_argument("--out", type=Path, default=None, help="optional output path for per-game JSONL")
    parser.add_argument(
        "--afterstate-out", type=Path, default=None,
        help="exp12 W1'a: optional path, APPEND-opened, to stream one JSONL row per "
        "played turn's pre-battle afterstate (the exact state handed to "
        "_resolve_versus_turn, meta.versus intact) plus driver-true race scalars "
        "(lives/opp_lives at the END_TURN decision, pre-battle cumulative wins -- "
        "Vic semantics, RESULTS_W1 finding 2), plus one final {win, end_reason, "
        "turns_survived} row per game. Written + flushed per row, never buffered "
        "across games (unlike --render-dir/capture_detail) -- see module docstring "
        "'We9 addition'. Off by default; every existing run is byte-for-byte "
        "unaffected.",
    )
    parser.add_argument(
        "--teacher-record-out", type=Path, default=None,
        help="exp12 route a (wave A0): optional path to stream the ROLLOUT-TEACHER "
        "record as GZIPPED JSONL -- one row per searched decision carrying every "
        "scored candidate's afterstate, myopic score, rollout (teacher) score and "
        "per-repeat outcomes, the CRN keys behind them, the chosen flag, and the "
        "driver-true race scalars; plus one final {win, end_reason, turns_survived} "
        "row per game. Requires --recommender search --search-scoring rollout (a "
        "myopic run produces no rollout scores to record). Turns --rollout-crn ON "
        "unless --no-rollout-crn is given. APPEND-opened, written + flushed per row, "
        "never buffered across games, and framed as ONE COMPLETE GZIP MEMBER PER GAME "
        "so a killed shard stays readable (costs ~1.9%% size; see TeacherRecordWriter "
        "and module docstring 'Wa addition'). Off by default.",
    )
    parser.add_argument(
        "--segment-record-out", type=Path, default=None,
        help="exp13 W1: optional GZIP JSONL stream with one row per real honest-frame "
        "segment decision. Keeps the real start and committed result separate from "
        "imagined prefix-completion afterstates, records all raw candidate chains and "
        "deduplicated groups, and adds play-independent S12/S16 label selection. "
        "Requires search+vgame+segmented-honest.",
    )
    parser.add_argument(
        "--teacher-rollouts", type=int, default=DEFAULT_TEACHER_ROLLOUTS,
        help="exp12 route a, --teacher-record-out only: rollout repeats per candidate "
        "AT LABEL TIME, replacing --rollout-repeats for the recorded run. Separate knob "
        "because the recorded score is a regression TARGET (its standard error bounds "
        "what a distilled V can reproduce), not just a move choice, so it wants a "
        "bigger budget than the deployable preset's 4 without editing that preset.",
    )
    parser.add_argument(
        "--rollout-crn", action=argparse.BooleanOptionalAction, default=None,
        help="exp12 route a (wave A0), --search-scoring rollout only: common random "
        "numbers across the SIBLING candidates of one decision. The r-th rollout of "
        "every candidate at a decision then shares the same future (random-fallback "
        "opponent draw, pool_random/retrieval pid draw, and the continuation's engine/"
        "shop seed), keyed on (seed, game_index, turn, repeat) with the candidate "
        "index deliberately absent; different repeats/turns/games stay independent. "
        "The battle oracle's own Monte-Carlo draw is NOT controllable (the JS "
        "simulator takes no seed) and remains the residual noise source. Default: ON "
        "when --teacher-record-out is set, OFF otherwise (so no existing measurement "
        "frame moves).",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="We2: torch.set_num_threads() pin, default 1. Not primarily a performance "
        "knob (it also happens to be faster here, not slower, for this small a model): "
        "BcRecommender's forward pass (bc_recommender.py::_masked_action_probs) is NOT "
        "bit-reproducible across separate calls under torch's default multi-threaded "
        "matmul (verified: two in-process decodes of the same state diverged at 32 "
        "threads, matched bit-for-bit at 1) -- this only matters when a near-tied "
        "top-2 action probability flips order between calls, rare per decode step but "
        "not rare over a whole game. NOTE this alone does NOT make a game reproducible "
        "end to end -- the battle oracle is a separate, unseeded source of "
        "non-determinism this flag cannot fix (see `render_selected_games`'s "
        "docstring); it only removes the decode side's contribution. Mirrors "
        "`train_chain_bc.py --torch-threads`'s existing pin for the same reason, at "
        "training time.",
    )
    parser.add_argument(
        "--report-json", type=Path, default=None,
        help="We2: optional path to write the aggregate metrics report (winrate + CI, "
        "lives, fallback, end/stop-reason distributions) as JSON",
    )
    parser.add_argument(
        "--render-games", type=int, default=3,
        help="We2: number of representative games to render (only takes effect if "
        "--render-dir is also given)",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=None,
        help="We2: base output dir for rendered galleries "
        "(one whole-game image per game at <dir>/images/full_game_<pid8>.png + a "
        "<dir>/images/full_game_<pid8>.json sidecar, summary at <dir>/render_report.json); "
        "rendering is SKIPPED entirely unless this is set",
    )
    parser.add_argument(
        "--render-from-jsonl", type=Path, default=None,
        help="We2: re-render galleries from an EXISTING run's per-game JSONL (a prior "
        "--out file that was written with detail capture on) WITHOUT replaying any game "
        "-- the only faithful way to regenerate a specific run's galleries, since the "
        "battle oracle is non-deterministic (see render_selected_games). Requires "
        "--render-dir; skips the eval, the BC checkpoint, and the chain snapshot load.",
    )
    parser.add_argument(
        "--decode-mode", type=str, default=BC_DEFAULT_DECODE_MODE, choices=list(BC_DECODE_MODES),
        help="We4 (exp09 W0c dual-decode probe): \"ranked\" (default, fully backward "
        "compatible) is BcRecommender's pre-existing deterministic probability-ranked "
        "scan. \"sample\" draws from the masked (optionally temperature-scaled) action "
        "distribution instead, retrying among the remaining legal candidates on a "
        "visited-state collision -- see bc_recommender.py's module docstring. Lets the "
        "SAME checkpoints be re-evaluated under a different decode, to tell apart a real "
        "PPO regression from a ranked/greedy-decode artifact.",
    )
    parser.add_argument(
        "--sample-temperature", type=float, default=BC_DEFAULT_SAMPLE_TEMPERATURE,
        help="--decode-mode sample only: softmax temperature applied to the masked action "
        "distribution before sampling (1.0 = distribution as predicted, unchanged).",
    )
    parser.add_argument(
        "--sample-seed", type=int, default=BC_DEFAULT_SAMPLE_SEED,
        help="--decode-mode sample only: base seed for the per-(game, turn, step) "
        "deterministic draw (see bc_recommender.py::_sample_rng). Independent of --seed, "
        "which only controls opponent/engine draws.",
    )
    parser.add_argument(
        "--llm-config", type=Path, default=None,
        help="exp11 W1, --recommender llm only: path to a JSON file matching "
        "LlmAgentConfig's fields (loaded via LlmAgentConfig.from_json_file). Omit to use "
        "LlmAgentConfig()'s own defaults plus whatever --llm-* overrides below are given.",
    )
    parser.add_argument(
        "--llm-model", type=str, default=None,
        help="exp11 W1, --recommender llm only: overrides LlmAgentConfig.model "
        "(default: from --llm-config, else LlmAgentConfig's own default deepseek-v4-flash).",
    )
    parser.add_argument(
        "--llm-fallback", type=str, default=None, choices=("end_turn", "bc"),
        help="exp11 W1, --recommender llm only: overrides LlmAgentConfig.fallback_mode "
        "(default when neither this flag nor --llm-config set one: LlmAgentConfig's own "
        "default, \"end_turn\"). \"end_turn\" -- every fallback ladder rung (illegal-chain "
        "retries exhausted, turn budget exhausted, provider error) submits a bare "
        "END_TURN. \"bc\" -- fall back to the BcRecommender this driver also constructs "
        "for that purpose (this is the ONE case where --recommender llm still loads "
        "--bc-checkpoint/imports torch).",
    )
    parser.add_argument(
        "--llm-cache-dir", type=Path, default=None,
        help="exp11 W1, --recommender llm only: overrides LlmAgentConfig.llm_cache_dir "
        "(the RecordedReplayClient cache directory; required whenever --llm-cache-mode "
        "is not \"off\").",
    )
    parser.add_argument(
        "--llm-cache-mode", type=str, default=None, choices=("off", "record", "replay", "record_missing"),
        help="exp11 W1, --recommender llm only: overrides LlmAgentConfig.llm_cache_mode "
        "(default when neither this flag nor --llm-config set one: LlmAgentConfig's own "
        "default, \"off\") -- see llm_agent/client.py::RecordedReplayClient for the 3 "
        "non-off modes (record writes, replay reads-only-ever and hard-raises on a miss, "
        "record_missing reads-if-cached else writes).",
    )
    parser.add_argument(
        "--llm-transcript", type=Path, default=None,
        help="exp11 W1, --recommender llm only: per-turn JSONL transcript path "
        "(llm_agent/transcript.py::TranscriptWriter). Default: <out>.transcript.jsonl "
        "next to --out when --out is given; transcripting stays off if neither is set.",
    )
    parser.add_argument(
        "--llm-provider", type=str, default=None, choices=("openai_compatible", "mock"),
        # Hidden CLI escape hatch for zero-cost MockLLM E2E smokes (see
        # build_mock_end_turn_client()) -- intentionally undocumented in --help.
        help=argparse.SUPPRESS,
    )
    return parser


def _run_render_only(args: argparse.Namespace) -> None:
    """`--render-from-jsonl` mode: re-render galleries from an existing run's
    captured JSONL, replaying NOTHING (no eval, no BC, no snapshot load)."""
    if args.render_dir is None:
        raise SystemExit("--render-from-jsonl requires --render-dir")
    print(f"render_from_jsonl:{args.render_from_jsonl}", flush=True)
    results = load_results_jsonl(args.render_from_jsonl)
    print(f"loaded_results:n_games={len(results)}", flush=True)
    render_report = render_selected_games(results, k=int(args.render_games), out_dir=args.render_dir)
    args.render_dir.mkdir(parents=True, exist_ok=True)
    render_report_path = args.render_dir / "render_report.json"
    render_report_path.write_text(json.dumps(render_report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"render_report_written:{render_report_path}", flush=True)


def _build_arena_pool(
    args: argparse.Namespace,
) -> tuple[ChainSnapshotSource | None, dict[int, int] | None]:
    """The arena ruler's own opponent pool (`--opponent-mode arena`), or
    (None, None) under chain mode.

    exp12 W0 (We8): SAME snapshot file and `--opponent-pack` as the chain
    pool, but its own split/rank selectors -- built as a SECOND
    `ChainSnapshotSource` rather than by relaxing `opp_source`'s filters,
    because both pools are live at once: `initial_pid_for_game` keeps drawing
    from the (rank/split-filtered) chain pool so the two rulers' RNG streams
    stay aligned, while the per-turn draw comes from this one. Costs a second
    parse of the snapshot (~1GB extra resident on the 262MB turtle file) --
    worth measuring before sharding many of these onto one box.

    exp13 W0b (codex review P1): `--arena-pool-split` selects THIS pool's
    split. It defaults to None = every split (the pre-exp13 behavior), and
    `ChainSnapshotSource` validates a requested split's existence and
    disjointness, so a gate run that names one gets a real held-out pool or a
    hard error -- never a silent full-pool fallback.

    Factored out of `main()` so both the flag wiring and the two
    only-applies-to-arena refusals are reachable from a test without loading
    a checkpoint. Raises `SystemExit` for a misapplied flag, matching the
    rest of this module's flag rejections.
    """
    if args.opponent_mode != OPPONENT_MODE_ARENA:
        for flag, value in (
            ("--arena-pool-rank-max", args.arena_pool_rank_max),
            ("--arena-pool-split", args.arena_pool_split),
        ):
            if value is not None:
                raise SystemExit(f"{flag} only applies to --opponent-mode arena")
        return None, None

    print(
        f"loading_arena_pool:{args.chain_snapshot}:"
        f"splits={args.arena_pool_split or 'all'}:"
        f"opponent_pack={args.opponent_pack}:arena_pool_rank_max={args.arena_pool_rank_max}",
        flush=True,
    )
    t0 = time.monotonic()
    arena_source = ChainSnapshotSource(
        args.chain_snapshot,
        long_min=int(args.long_min),
        seed=int(args.seed),
        split=args.arena_pool_split,
        opponent_pack=args.opponent_pack,
        opponent_rank_min=None,
        opponent_rank_max=args.arena_pool_rank_max,
    )
    arena_pool_size_by_turn = {int(t): len(rows) for t, rows in arena_source.by_turn.items()}
    # exp13 F1: `configured_max_turn` is the game's rule, `pool_depth` is the
    # supply, `effective_max_turn` is the one games are actually played to.
    # Printed on the same line the depth has always been printed on, because
    # the depth alone was already visible in every run log and it still took a
    # review to notice it was below the cap.
    pool_depth = arena_source.max_turn_with_candidates
    effective = effective_arena_max_turn(
        int(args.max_turn), arena_source, opponent_mode=OPPONENT_MODE_ARENA
    )
    print(
        f"arena_pool_loaded:games={len(arena_source.by_pid)}:"
        f"split={arena_source.split}:splits_source={arena_source.splits_source}:"
        f"turn1_candidates={len(arena_source.by_turn.get(1, []))}:"
        f"max_turn_with_candidates={pool_depth if pool_depth is not None else 0}:"
        f"configured_max_turn={int(args.max_turn)}:"
        f"effective_max_turn={effective}:"
        f"turn_cap_bound_by={'pool_depth' if effective < int(args.max_turn) else 'configured'}:"
        f"elapsed={time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return arena_source, arena_pool_size_by_turn


def _arena_pool_report_fields(
    args: argparse.Namespace,
    arena_source: ChainSnapshotSource | None,
) -> dict[str, Any]:
    """The `arena_pool_*` block of `--report-json`.

    Every key is present regardless of `--opponent-mode` (None under chain,
    where no arena pool exists), matching the metadata block's own
    "diffable field-by-field" convention.

    exp13 W0b (codex review P1): `arena_pool_splits` is the split the pool
    was ACTUALLY built with -- "all" when unrestricted (the pre-exp13 value,
    so old and new reports stay comparable) and the split name when
    `--arena-pool-split` restricted it -- with `arena_pool_splits_source`
    next to it, because "the gate pool was val-only" is only a real claim if
    the split assignment came from the snapshot's own builder-computed
    `splits` rather than a derived fallback.

    exp13 F1 (2026-08-07) adds three keys, for the same reason
    `arena_pool_splits_source` exists: "the games were played to turn 30" is
    only a real claim if the pool could serve turn 30. `arena_pool_max_turn`
    is the pool's depth, `arena_configured_max_turn` is `--max-turn`, and
    `arena_effective_max_turn` is the smaller of the two, which is what the
    games were actually played to. All three are None under chain mode, where
    no arena pool exists, matching this block's own convention.
    """
    effective = (
        None
        if arena_source is None
        else effective_arena_max_turn(
            int(args.max_turn), arena_source, opponent_mode=OPPONENT_MODE_ARENA
        )
    )
    return {
        "arena_pool_rank_max": args.arena_pool_rank_max,
        "arena_pool_splits": (None if arena_source is None else (arena_source.split or "all")),
        "arena_pool_splits_source": (
            None if arena_source is None or arena_source.split is None else arena_source.splits_source
        ),
        "arena_pool_opponent_pack": (args.opponent_pack if arena_source is not None else None),
        "arena_pool_games": (len(arena_source.by_pid) if arena_source is not None else None),
        "arena_pool_snapshot_version": (
            arena_source.snapshot_version if arena_source is not None else None
        ),
        "arena_pool_max_turn": (
            None if arena_source is None else arena_source.max_turn_with_candidates
        ),
        "arena_configured_max_turn": (None if arena_source is None else int(args.max_turn)),
        "arena_effective_max_turn": effective,
    }


def _check_game_rules_compatibility(args: argparse.Namespace) -> None:
    """Wb (exp13 W0a): CLI face of `_game_rules_refusal` -- see it for WHICH
    combinations are refused and why.

    Called from `main()` before any checkpoint or 262MB snapshot is loaded.
    Raises `SystemExit` (the established way this module rejects a flag
    combination -- see `--arena-pool-rank-max`). The API face
    (`_assert_game_rules_supported`, exp13 W0b) enforces the identical rule
    inside `run_versus_eval` / `play_one_game`, so the check is not something
    a caller can skip by not being the CLI.
    """
    message = _game_rules_refusal(
        game_rules=getattr(args, "game_rules", GAME_RULES_VERSUS),
        opponent_mode=getattr(args, "opponent_mode", OPPONENT_MODE_CHAIN),
        scoring=(
            getattr(args, "search_scoring", None)
            if str(getattr(args, "recommender", "bc")) == "search"
            else None
        ),
    )
    if message is not None:
        raise SystemExit(message)


def _check_turn_mode_compatibility(args: argparse.Namespace) -> None:
    """Wd (exp13 A1): CLI face of `_turn_mode_refusal` -- see it for WHICH
    combination is refused and why. Called from `main()` next to
    `_check_game_rules_compatibility`, before any checkpoint or the 262MB
    snapshot is loaded; the API face (`_assert_game_rules_supported`)
    enforces the same rule inside `run_versus_eval`/`play_one_game`.

    `--stochastic-samples` is validated here too: it is a k, and a k below 1
    would silently score every stochastic prefix on zero completions.
    """
    if int(getattr(args, "stochastic_samples", DEFAULT_STOCHASTIC_SAMPLES)) < 1:
        raise SystemExit(
            f"--stochastic-samples must be >= 1 (got {args.stochastic_samples})"
        )
    message = _turn_mode_refusal(
        turn_mode=getattr(args, "turn_mode", DEFAULT_TURN_MODE),
        scoring=(
            getattr(args, "search_scoring", None)
            if str(getattr(args, "recommender", "bc")) == "search"
            else None
        ),
    )
    if message is not None:
        raise SystemExit(message)


SEGMENT_RECORD_RECOMMENDERS: tuple[str, ...] = ("search", "bc")


def segment_record_flag_problem(
    *, recommender: str, search_scoring: str, turn_mode: str
) -> str | None:
    """Why this flag combination may not write segment records, or None.

    `PLAN_W1.md` Waves W1c asks for "exactly 200 exploratory games through the
    final recorder path, 180 search-policy and 20 BC0-only", so the BC0 anchor
    is a recorder arm by the plan's own words. This guard used to require
    `--recommender search`, which made the anchor unrecordable, which is why
    internal project notesconfigs/w1_pilot.json` carries
    `segment_record: false` on that arm: the manifest was following the code,
    not the plan. The #107 review found the gap.

    WHAT IS STILL REQUIRED, AND WHY EACH ONE IS.

    `--turn-mode segmented-honest` binds for EVERY recommender. The rows are
    per-SEGMENT: `SegmentRecorder` opens one per `decide` call, records the
    imagined clone the recommender was handed and the `imagination_seed` that
    clone ran on, and chains `decision_start_real` to `committed_result_real`
    across the segment boundary. Under the determinized frame there is one
    whole-turn decision, no imagined clone and no boundary, so a row would be
    a differently-shaped object wearing the same schema version.

    `--search-scoring vgame` binds for the SEARCH arm only. It is what puts
    scored candidate groups on the recommendation, and a search arm without
    them would produce rows whose `candidate_groups` are empty for a reason
    nobody could tell apart from a broken recorder. A BC arm has no candidates
    to score: its rows carry `candidate_groups: []` BY CONSTRUCTION, which
    `run_exp13_w1_pilot.record_census` counts on its own line
    (`n_decisions_without_candidate_groups`) and `exp13_w1_records.in_label_pool`
    keeps out of the label pool. So the anchor's rows can never be mistaken for
    trainable rows, which is what `PLAN_W1.md` Distribution contract requires of
    an anchor.
    """
    if recommender not in SEGMENT_RECORD_RECOMMENDERS:
        allowed = " ".join(SEGMENT_RECORD_RECOMMENDERS)
        return (
            f"--segment-record-out requires --recommender one of [{allowed}], "
            f"got {recommender!r}"
        )
    if turn_mode != TURN_MODE_SEGMENTED_HONEST:
        return (
            "--segment-record-out requires --turn-mode segmented-honest, got "
            f"{turn_mode!r}"
        )
    if recommender == "search" and search_scoring != SEARCH_SCORING_VGAME:
        return (
            "--segment-record-out with --recommender search requires "
            f"--search-scoring vgame, got {search_scoring!r}"
        )
    return None


def main() -> None:
    # Call-time import: the LLM agent line is not part of the public release, and module scope would put it in the closure.
    from .llm_recommender import FALLBACK_MODE_BC

    args = _build_arg_parser().parse_args()

    # Render-only re-run of an existing capture (no eval; see the flag's help).
    if args.render_from_jsonl is not None:
        _run_render_only(args)
        return

    if args.num_games is None:
        raise SystemExit("--num-games is required unless --render-from-jsonl is given")

    # Wb (exp13 W0a) + Wd (exp13 A1): before any checkpoint or 262MB
    # snapshot is loaded.
    _check_game_rules_compatibility(args)
    _check_turn_mode_compatibility(args)

    # exp13 W0b': process-level and PERFORMANCE ONLY -- see the flag's --help
    # and `api._SKIP_IMAGINED_VALIDATION`. Set here, before any recommender is
    # built, so every imagined walk in this process sees the same setting.
    if args.skip_imagined_validation:
        set_skip_imagined_validation(True)
        print("skip_imagined_validation:on", flush=True)

    fixture_state = load_initial_state_from_fixture(FIXTURE_PATH)
    _assert_fixture_shape(fixture_state)

    # Full-game frame fix (train/opening_source.py): "varied" (default)
    # builds a `VariedOpeningSource` (a per-game_index-seeded engine roll,
    # no file/pool dependency); "fixed" keeps the untouched single-fixture
    # `fixture_state` loaded above, wrapped so `play_one_game` sees the
    # same `.state_for_game()` interface either way.
    opening_mode = str(args.opening_mode)
    if opening_mode == "varied":
        opening: VariedOpeningSource | FixedOpeningSource = build_varied_opening_source()
    else:
        opening = fixed_opening_source(fixture_state)
        print(f"opening_source_built:mode=fixed:fixture={FIXTURE_PATH}", flush=True)

    # exp11 W1 review fix 2: build the LlmAgentConfig here, BEFORE the
    # needs_bc gate below, when --recommender llm -- needs_bc must read the
    # fully-RESOLVED config's fallback_mode (file + overrides), not the raw
    # --llm-fallback CLI value: a --llm-config file can set
    # fallback_mode="bc" on its own without repeating --llm-fallback bc on
    # the CLI (that is the whole point of fix 2), and if needs_bc still read
    # only the raw flag it would miss that case, skip loading BcRecommender,
    # and LlmRecommender's own construction would then hard-raise
    # (fallback_mode="bc" requires a fallback callable). Built once here and
    # reused by _build_llm_recommender below instead of rebuilt.
    early_llm_config: LlmAgentConfig | None = None
    if args.recommender == "llm":
        early_llm_config = _build_llm_agent_config(args, transcript_path=_resolve_llm_transcript_path(args))

    # exp11 W1: BC checkpoint construction (and the torch import/pin it
    # requires) is now CONDITIONAL. "bc"/"search" always need it -- byte-
    # identical to before this flag existed. "llm" needs it ONLY when its
    # RESOLVED fallback_mode is "bc" (--llm-fallback bc, or --llm-config
    # setting fallback_mode="bc"); a plain end_turn-fallback llm run has no
    # BcRecommender in its critical path at all (LlmRecommender's own
    # end_turn fallback needs nothing but the chain it already has) and must
    # not require --bc-checkpoint or pull in torch.
    needs_bc = args.recommender in ("bc", "search") or (
        args.recommender == "llm" and early_llm_config is not None and early_llm_config.fallback_mode == FALLBACK_MODE_BC
    )

    bc: BcRecommender | None = None
    if needs_bc:
        # Pin BEFORE constructing BcRecommender (see --torch-threads' own help
        # string for why this is a correctness pin, not a speed knob).
        import torch

        torch.set_num_threads(int(args.torch_threads))

        print(f"loading_bc_checkpoint:{args.bc_checkpoint}", flush=True)
        print(
            f"decode_mode={args.decode_mode}:sample_temperature={args.sample_temperature}:"
            f"sample_seed={args.sample_seed}",
            flush=True,
        )
        t0 = time.monotonic()
        bc = BcRecommender(
            args.bc_checkpoint,
            decode_mode=args.decode_mode,
            sample_temperature=args.sample_temperature,
            sample_seed=args.sample_seed,
        )
        print(f"bc_checkpoint_loaded:elapsed={time.monotonic() - t0:.1f}s", flush=True)

    # exp09 W1: the chain snapshot is now loaded BEFORE the recommender is
    # constructed (previously the other way around) -- --search-scoring
    # rollout needs the SAME ChainSnapshotSource instance the eval loop uses
    # so its continuations can sample the followed chain's real subsequent
    # boards (see search_recommender.py's module docstring). Cosmetic-only
    # reordering for every other mode: neither construction depends on the
    # other, so this only changes the ORDER these two print lines appear in
    # the log, never any behavior.
    print(
        f"loading_chain_snapshot:{args.chain_snapshot}:opponent_split={args.opponent_split}:"
        f"opponent_pack={args.opponent_pack}:opponent_rank_min={args.opponent_rank_min}:"
        f"opponent_rank_max={args.opponent_rank_max}",
        flush=True,
    )
    t0 = time.monotonic()
    opp_source = ChainSnapshotSource(
        args.chain_snapshot,
        long_min=int(args.long_min),
        seed=int(args.seed),
        split=args.opponent_split,
        opponent_pack=args.opponent_pack,
        opponent_rank_min=args.opponent_rank_min,
        opponent_rank_max=args.opponent_rank_max,
    )
    print(
        f"chain_snapshot_loaded:games={len(opp_source.by_pid)}:"
        f"long_pids={len(opp_source.long_pids)}:split={opp_source.split}:"
        f"splits_source={opp_source.splits_source}:elapsed={time.monotonic() - t0:.1f}s",
        flush=True,
    )

    # exp12 W0 (We8) + exp13 W0b: the arena ruler's own pool, and the
    # refusals for its flags under chain mode -- see `_build_arena_pool`.
    arena_source, arena_pool_size_by_turn = _build_arena_pool(args)

    # exp09 W6a: --recommender search wraps the just-loaded `bc` in a
    # SearchRecommender (the SAME model/encoder, no second checkpoint load
    # -- see that class's module docstring) before it ever reaches
    # `run_versus_eval`/`play_one_game`, which only ever call
    # `.recommend(state)` on whatever they were handed.
    # Wa (exp12 route a, wave A0): resolve the recorder's two defaults BEFORE
    # the recommender is built. CRN defaults ON with the recorder and OFF
    # without it, so the live W2c width arms (and every historical frame) keep
    # the pre-A0 keying unless somebody asks for the change explicitly.
    teacher_record_on = args.teacher_record_out is not None
    if teacher_record_on and (
        args.recommender != "search" or args.search_scoring != SEARCH_SCORING_ROLLOUT
    ):
        raise SystemExit(
            "--teacher-record-out requires --recommender search --search-scoring rollout"
        )
    segment_record_on = args.segment_record_out is not None
    if segment_record_on:
        problem = segment_record_flag_problem(
            recommender=args.recommender,
            search_scoring=args.search_scoring,
            turn_mode=args.turn_mode,
        )
        if problem is not None:
            raise SystemExit(problem)
    rollout_crn = bool(teacher_record_on if args.rollout_crn is None else args.rollout_crn)
    rollout_repeats = int(args.teacher_rollouts if teacher_record_on else args.rollout_repeats)

    # exp12 W2 (wave A4): the learned leaf. Built BEFORE the recommender so a
    # bad checkpoint, a stale encoder fingerprint or a pessimism run with one
    # head fails here, at construction, instead of on the first searched turn
    # of a sharded 300-game arm.
    vgame_scorer = None
    if args.search_scoring == SEARCH_SCORING_VGAME:
        from .vgame_scorer import VGameLeafScorer

        if args.recommender != "search":
            raise SystemExit("--search-scoring vgame requires --recommender search")
        if not args.vgame_heads:
            raise SystemExit("--search-scoring vgame requires --vgame-heads")
        vgame_scorer = VGameLeafScorer.from_checkpoints(
            args.vgame_heads,
            args.vgame_extractor or args.bc_checkpoint,
            blend=float(args.vgame_blend),
            pessimism=float(args.vgame_pessimism),
        )
        print(f"vgame_leaf_loaded:{json.dumps(vgame_scorer.describe(), sort_keys=True)}", flush=True)
    elif args.vgame_heads:
        raise SystemExit("--vgame-heads requires --search-scoring vgame")

    recommender: BcRecommender | SearchRecommender | LlmRecommender
    llm_config: LlmAgentConfig | None = None
    if args.recommender == "search":
        recommender = SearchRecommender(
            bc,
            n_candidates=int(args.search_candidates),
            ksim=int(args.search_ksim),
            seed=int(args.seed),
            scoring=args.search_scoring,
            rollout_shortlist=int(args.rollout_shortlist),
            rollout_repeats=rollout_repeats,
            rollout_ksim=int(args.rollout_ksim),
            # exp10 W2: opponent-source ablation, default "true" (byte-for-
            # byte the pre-existing W1 behavior) -- see search_recommender.
            # py's module docstring and this flag's own --help.
            rollout_opponent_mode=args.rollout_opponent_mode,
            # exp09 W1: only the rollout scorer needs the chain snapshot
            # (myopic scoring only ever looks at state's own
            # last_opponent_team, never samples a new turn) -- passed
            # unconditionally regardless of scoring mode since it's cheap
            # (already constructed) and keeps this call site simple; unused
            # by SearchRecommender whenever scoring="myopic".
            opp_source=opp_source,
            max_turn=int(args.max_turn),
            # Wa (exp12 route a, wave A0): see the flags' own --help.
            rollout_crn=rollout_crn,
            capture_teacher_record=teacher_record_on,
            # exp12 W2 (wave A4): None unless --search-scoring vgame.
            vgame_scorer=vgame_scorer,
            # Wd (exp13 A1): the honest frame. Default whole-determinized, in
            # which every line of that class is byte-for-byte its pre-A1 self.
            turn_mode=args.turn_mode,
            stochastic_samples=int(args.stochastic_samples),
            # exp22 AMENDMENT 28: the frame the ROLLOUT resolves under. The
            # recommender used to carry `turn_mode` and nothing else, so its
            # continuation ran under the versus default whatever frame the arm
            # was judged on -- which is why an arena rollout read a missing
            # opponent life bar as an instant win.
            game_rules=str(args.game_rules),
            arena_race_convention=str(args.arena_race_convention),
            mc_rerank_k=int(args.mc_rerank_k),
            # exp22 W3. Validated in the recommender's constructor, which is
            # the single authority on which combinations are coherent.
            completion_policy=str(args.completion_policy),
            completion_width=int(args.completion_width),
            completion_aggregate=str(args.completion_aggregate),
            # exp13 A2.4: off by default -- see the flag's own --help.
            capture_candidate_chains=bool(
                args.capture_candidate_chains or segment_record_on
            ),
        )
        print(
            f"search_recommender_wrapped:n_candidates={args.search_candidates}:"
            f"ksim={args.search_ksim}:scoring={args.search_scoring}:"
            f"rollout_shortlist={args.rollout_shortlist}:rollout_repeats={rollout_repeats}:"
            f"rollout_ksim={args.rollout_ksim}:rollout_opponent_mode={args.rollout_opponent_mode}:"
            f"rollout_crn={rollout_crn}:teacher_record={teacher_record_on}:"
            f"vgame_leaf={'yes' if vgame_scorer is not None else 'no'}:"
            f"turn_mode={args.turn_mode}:stochastic_samples={args.stochastic_samples}:"
            f"completion_policy={args.completion_policy}:"
            f"completion_width={args.completion_width}:"
            f"completion_aggregate={args.completion_aggregate}:"
            f"game_rules={args.game_rules}:mc_rerank_k={args.mc_rerank_k}",
            flush=True,
        )
        if teacher_record_on and int(args.rollout_shortlist) < int(args.search_candidates):
            # Not an error (a narrower shortlist is a legitimate, cheaper
            # recording), but it means most candidates carry NO teacher score
            # and the decision's within-group contrast is only as wide as the
            # shortlist -- which is the whole point of route a, so say so.
            print(
                f"teacher_record_warning:shortlist={args.rollout_shortlist}"
                f"<candidates={args.search_candidates}:"
                "only shortlisted candidates will carry a teacher score",
                flush=True,
            )
    elif args.recommender == "llm":
        # exp11 W1: `bc` is None here unless the RESOLVED fallback_mode is
        # "bc" (see `needs_bc` above) -- `_build_llm_recommender` only wires
        # it in as `fallback=bc.recommend` when it is actually present.
        assert early_llm_config is not None  # set above whenever args.recommender == "llm"
        llm_config = early_llm_config
        recommender = _build_llm_recommender(llm_config, bc=bc)
        print(
            f"llm_recommender_built:provider={llm_config.provider}:model={llm_config.model}:"
            f"fallback_mode={llm_config.fallback_mode}:llm_cache_mode={llm_config.llm_cache_mode}:"
            f"llm_cache_dir={llm_config.llm_cache_dir}:transcript_path={llm_config.transcript_path}",
            flush=True,
        )
    else:
        # needs_bc is True for plain "bc", so this is never None here.
        recommender = bc

    want_render = args.render_dir is not None and int(args.render_games) > 0

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.out.open("w", encoding="utf-8") if args.out else None
    # We9 (exp12 W1'a): APPEND-opened (see --afterstate-out's help) --
    # unlike --out, a shard resumed after a partial prior write must not
    # clobber whatever rows already reached disk.
    if args.afterstate_out is not None:
        args.afterstate_out.parent.mkdir(parents=True, exist_ok=True)
    afterstate_out_fh = args.afterstate_out.open("a", encoding="utf-8") if args.afterstate_out else None
    # Wa (exp12 route a): GZIP, APPEND, ONE MEMBER PER GAME -- the rows carry
    # one full board per candidate, so plain JSONL would not fit this box's
    # data disk at A1 volume, and a single streamed member would make the
    # whole shard unreadable the moment the process was killed. See
    # `TeacherRecordWriter`.
    if args.teacher_record_out is not None:
        args.teacher_record_out.parent.mkdir(parents=True, exist_ok=True)
    teacher_writer = (
        TeacherRecordWriter(args.teacher_record_out) if args.teacher_record_out else None
    )
    if args.segment_record_out is not None:
        args.segment_record_out.parent.mkdir(parents=True, exist_ok=True)
    segment_writer = (
        TeacherRecordWriter(args.segment_record_out) if args.segment_record_out else None
    )
    try:
        results = run_versus_eval(
            recommender,
            opp_source,
            opening,
            num_games=int(args.num_games),
            max_turn=int(args.max_turn),
            seed=int(args.seed),
            # FIX 2: no parse_cache -- the snapshot source is fully
            # pre-parsed, so a cache is never read, only written into
            # (unbounded growth over N games). None makes that write a no-op.
            parse_cache=None,
            out_fh=out_fh,
            # Only pay for full per-turn board capture when a gallery was
            # actually requested -- see `render_selected_games`'s docstring
            # for why rendering reads off THIS pass instead of re-running.
            capture_detail=want_render,
            game_index_start=int(args.game_index_start),
            opponent_mode=args.opponent_mode,
            arena_source=arena_source,
            afterstate_out_fh=afterstate_out_fh,
            teacher_writer=teacher_writer,
            segment_writer=segment_writer,
            game_rules=args.game_rules,
            arena_race_convention=args.arena_race_convention,
            turn_mode=args.turn_mode,
        )
    finally:
        if out_fh is not None:
            out_fh.close()
        if afterstate_out_fh is not None:
            afterstate_out_fh.close()
        if teacher_writer is not None:
            teacher_writer.close()
        if segment_writer is not None:
            segment_writer.close()

    _print_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)
    # exp12 W2c: the battle worker's lifetime for THIS process, in the log
    # as well as the report -- a shard that recycled its worker (or never
    # needed to) says so without anyone attaching to a live pid.
    print(f"battle_worker_stats:{json.dumps(battle_worker_stats(), sort_keys=True)}", flush=True)

    if args.report_json is not None:
        report = _compute_aggregate(results, arena_pool_size_by_turn=arena_pool_size_by_turn)
        # exp09 W5 P0: run provenance, so a held-out-split report (or any future
        # one) is self-identifying instead of relying on the caller to remember
        # which flags produced it (CLAUDE.md "no naked numbers").
        report["metadata"] = {
            "opponent_split": args.opponent_split,
            "opponent_pack": args.opponent_pack,
            "opponent_rank_min": args.opponent_rank_min,
            "opponent_rank_max": args.opponent_rank_max,
            "chain_snapshot": str(args.chain_snapshot),
            # exp12 W1'a: the FILTERED chain pool's size (opp_source's own
            # split/pack/rank_min/rank_max filters -- the same object every
            # run builds regardless of --opponent-mode, since arena mode
            # still draws `initial_pid_for_game` from it). Present
            # unconditionally so a chain-mode report is self-describing
            # (e.g. train/Turtle/rank<=1500 -> 914) without re-deriving the
            # count from the raw snapshot file by hand.
            "chain_pool_games": len(opp_source.by_pid),
            "bc_checkpoint": str(args.bc_checkpoint),
            "num_games": int(args.num_games),
            "seed": int(args.seed),
            "long_min": int(args.long_min),
            # exp09 W6a
            "recommender": args.recommender,
            "search_candidates": int(args.search_candidates),
            "search_ksim": int(args.search_ksim),
            # exp09 W0c dual-decode probe (We4)
            "decode_mode": args.decode_mode,
            "sample_temperature": float(args.sample_temperature),
            "sample_seed": int(args.sample_seed),
            # exp09 W1 teacher-ceiling rollout scorer (We5)
            "search_scoring": args.search_scoring,
            "rollout_shortlist": int(args.rollout_shortlist),
            # The EFFECTIVE repeats this run actually used: --teacher-rollouts
            # replaces --rollout-repeats whenever --teacher-record-out is set,
            # so echoing the raw flag here would misdescribe a recorded run.
            "rollout_repeats": int(rollout_repeats),
            "rollout_repeats_flag": int(args.rollout_repeats),
            "rollout_ksim": int(args.rollout_ksim),
            # Wa (exp12 route a, wave A0): always present, so any report says
            # on its face which RNG keying and which recorder produced it.
            "rollout_crn": bool(rollout_crn),
            "teacher_record_out": (
                str(args.teacher_record_out) if args.teacher_record_out is not None else None
            ),
            "segment_record_out": (
                str(args.segment_record_out) if args.segment_record_out is not None else None
            ),
            "teacher_rollouts": int(args.teacher_rollouts),
            # exp11 W1: --recommender llm. Present (None/empty) regardless of
            # --recommender, matching this block's existing "every key always
            # present" convention (e.g. search_candidates under plain
            # --recommender bc above). cost_totals/fallback_reason_counts are
            # the whole RUN's totals (LlmRecommender.cost_totals()/
            # fallback_reason_counts() accumulate across every recommend()
            # call this one instance served, i.e. every game/turn above).
            "llm_provider": (llm_config.provider if llm_config is not None else None),
            "llm_model": (llm_config.model if llm_config is not None else None),
            "llm_config_path": (str(args.llm_config) if args.llm_config is not None else None),
            "llm_fallback_mode": (llm_config.fallback_mode if llm_config is not None else None),
            "llm_cache_mode": (llm_config.llm_cache_mode if llm_config is not None else None),
            "llm_cache_dir": (llm_config.llm_cache_dir if llm_config is not None else None),
            "llm_transcript_path": (llm_config.transcript_path if llm_config is not None else None),
            "llm_cost_totals": (recommender.cost_totals() if _is_llm_recommender(recommender) else None),
            "llm_fallback_reason_counts": (
                recommender.fallback_reason_counts() if _is_llm_recommender(recommender) else None
            ),
            # exp10 W2 opponent-source ablation
            "rollout_opponent_mode": args.rollout_opponent_mode,
            # exp12 W2 (wave A4): present (None) on every non-vgame report,
            # matching this block's "every key always present" convention, so
            # a vgame arm and a rollout arm stay diffable field by field. The
            # dict carries the heads path + sha256, the extractor sha, the
            # encoder fingerprint, the blend/pessimism knobs and how many
            # boards the leaf actually scored.
            "vgame": (vgame_scorer.describe() if vgame_scorer is not None else None),
            # Full-game frame fix (train/opening_source.py)
            "opening_mode": opening_mode,
            # exp12 W0 (We8) "the ruler". Every key is present regardless of
            # --opponent-mode, matching this block's existing convention, so
            # a chain report and an arena report are diffable field-by-field.
            # The arena_pool_* fields describe the pool that was ACTUALLY
            # built (None under chain, where none is), which is what makes
            # an arena report self-describing without the launch command.
            "opponent_mode": args.opponent_mode,
            # exp13 W0a (Wb) "which game". Present on every report, so a
            # real-arena run and a versus/hybrid-arena run stay diffable
            # field by field; the race convention is None under versus,
            # where it does not apply.
            "game_rules": args.game_rules,
            "arena_race_convention": (
                args.arena_race_convention if args.game_rules == GAME_RULES_ARENA else None
            ),
            # exp13 A1 (Wd) "how a turn is played". Present on every report
            # (this block's convention) so a determinized arm and an honest
            # arm of the foresight-premium diagnostic stay diffable field by
            # field. `stochastic_samples` is echoed unconditionally for the
            # same reason, and is the k any honest number must be quoted with.
            "turn_mode": args.turn_mode,
            "stochastic_samples": int(args.stochastic_samples),
            "mc_rerank_k": int(args.mc_rerank_k),
            # exp22 W3, echoed unconditionally for the same reason as the two
            # above: a bc_greedy arm and a v_search arm have to stay diffable
            # field by field, and the policy is not inferable from the numbers.
            "completion_policy": str(args.completion_policy),
            "completion_width": int(args.completion_width),
            "completion_aggregate": str(args.completion_aggregate),
            # exp13 A2.4: recorded on every report so a run whose per-turn
            # diagnostics are unexpectedly large (or unexpectedly empty) can
            # say which it was without anyone finding the launch command.
            "capture_candidate_chains": bool(
                args.capture_candidate_chains or segment_record_on
            ),
            # exp13 W0b': a PERFORMANCE switch that is pinned to move no
            # number, recorded anyway -- a run whose numbers are ever doubted
            # must be able to say whether it validated its imagined walks
            # without anyone having to find the launch command.
            "skip_imagined_validation": bool(args.skip_imagined_validation),
            **_arena_pool_report_fields(args, arena_source),
            # exp12 W0 scope: --opponent-mode governs the DRIVER's per-turn
            # draw only; SearchRecommender's rollout continuations still
            # follow the chain pool (see the flag's --help). Only meaningful
            # when a rollout is actually scored -- myopic scoring never
            # samples a continuation at all, so claiming a "pool" for it
            # would be a fabricated provenance field (review F6).
            "search_rollout_opponent_pool": (
                "chain_pool"
                if (args.recommender == "search" and args.search_scoring == "rollout")
                else None
            ),
            # exp12 W0 review F2: opponent draws are keyed per (seed,
            # game_index), not per process -- see `play_one_game`. Present on
            # BOTH rulers so a report says which regime produced it.
            "fallback_rng": "per_game",
            "game_index_start": int(args.game_index_start),
            # exp12 W2c: this process's battle-worker lifetime (total oracle
            # calls, recycles and why, last RSS sample). RESULTS_W1 finding
            # 16 was diagnosed after the fact from RSS on live processes;
            # recording it makes "did the leak guard fire, and how often"
            # readable off any shard's own report.
            "battle_worker": battle_worker_stats(),
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"report_json_written:{args.report_json}", flush=True)

    if want_render:
        render_report = render_selected_games(results, k=int(args.render_games), out_dir=args.render_dir)
        args.render_dir.mkdir(parents=True, exist_ok=True)
        render_report_path = args.render_dir / "render_report.json"
        render_report_path.write_text(json.dumps(render_report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"render_report_written:{render_report_path}", flush=True)


if __name__ == "__main__":
    main()
