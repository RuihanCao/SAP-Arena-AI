"""exp09 W6a: best-of-N search layer over a loaded BC policy.

See internal design notes, the "W6 search layer" bullet and the
Rev 6.1 "W6 trajectory" addendum ("W6a = measure search gain on flat_v2 (the
turtle yardstick); W6b = expert iteration"). This module is the W6a
SUBSTRATE only -- a search-augmented recommender plus the driver flag to use
it (`tools/eval_versus_fullgame.py --recommender search`) -- not the
measurement run itself, which happens later on a fresh checkpoint load.

`SearchRecommender.recommend(state)` is API-compatible with
`BcRecommender.recommend` (`tools/bc_recommender.py`): same return-dict
shape (`ok`/`error`/`recommended_action`/`chain_preview`/`wdl_probs`/
`diagnostics`), plus `search_*` bookkeeping keys layered on top -- so
`eval_versus_fullgame.py::play_one_game` can call either recommender through
the exact same `rec = bc.recommend(state)` / `rec.get("chain_preview")`
site with no other change (mirrors how `BcRecommender` itself is already a
drop-in for `eval_tempo_planner.run_eval_cases`, see that class's own
docstring).

ALGORITHM: generate `n_candidates` full-turn action chains from `state`
(candidate 0 is always the wrapped recommender's OWN deterministic/greedy
chain -- byte-identical to what plain `--recommender bc` would have played,
so search can never do worse than plain BC; candidates 1..N-1 aim for
diversity, see "candidate diversity" below). Every candidate's chain is
re-applied from a FRESH `copy.deepcopy` of `state` via `api.step`
(`_apply_chain`, stopping early at the first illegal/failing step -- the
board reached so far still counts as that candidate's end board), exactly
the replay procedure `play_one_game` itself uses on the winning chain, so
the board this class scores a candidate on is guaranteed to be the board
the driver will actually reach if that candidate wins. End boards are
de-duplicated by `visited_guard.state_signature` before scoring, so a board
reached by more than one candidate costs ONE oracle call, not N. Each
unique end board is scored against the LAST-SEEN opponent
(`state["meta"]["versus"]["last_opponent_team"]`, written by
`train/env.py::_set_last_opponent_team` after every resolved turn -- absent
on turn 1, before any battle has happened yet) with one
`simulation_count=ksim` oracle call: `(playerWins - opponentWins) / ksim`
(PLAN.md's k-sim expected-outcome formula, reused here as a whole-chain
reranking score -- see `train/gym_env.py::ksim_lives_outcome` for the
training-time sibling of this same math). The winning candidate's ORIGINAL
result dict is returned UNCHANGED (so the driver's existing
`chain_preview`/`recommended_action` consumption keeps working) plus
`search_*` annotations. If the opponent is unknown (turn 1) or every oracle
call fails, scoring is skipped entirely and the greedy candidate's own
result comes back with `search_used=False` -- this can never make a turn
worse than plain BC, only occasionally no-better.

Candidate diversity (IMPORTANT -- read before assuming more `n_candidates`
buys more search): `tools/bc_recommender.py::BcRecommender.recommend` is,
as verified directly against that file, UNCONDITIONALLY a deterministic
probability-ranked walk -- its `__init__` stores `self.deterministic` but
the decode loop never reads it back (that class's own constructor comment:
"no longer consulted by recommend() ... kept only so an existing/future
caller passing this kwarg doesn't break"). Calling `recommend()` twice on
the same state therefore always returns a byte-identical chain: toggling
`bc_recommender.deterministic` around the call, by itself, produces ZERO
diversity against the real checkpoint. Rather than modify
`bc_recommender.py` (out of scope for this task -- the brief is a new file
plus a driver flag, not a BC-decoder refactor), this class carries its OWN
sampling decode (`_sample_candidate`), reusing the SAME loaded
model/encoder (`bc_recommender.model`, `bc_recommender.encoder`) and the
same engine-true `bc_recommender.legal_mask` + `visited_guard.
state_signature` the wrapped recommender's own greedy walk uses, but
drawing each step from the masked categorical distribution with a
candidate-local `numpy.random.Generator` instead of always taking the
arg-max. `_generate_candidate` takes this path automatically whenever the
wrapped object exposes that lower-level surface (i.e. a real
`BcRecommender`); a minimal duck-typed stand-in that only implements
`.deterministic` + `.recommend()` (as in this module's own unit tests, and
as an earlier draft of this class's spec assumed was sufficient) has no
`.model`/`.encoder`, so falls back to literally toggling `.deterministic`
and calling `.recommend()` again. That fallback path is what the unit
tests below exercise (their fakes script different chains per call
directly); it is NOT expected to add diversity against the real
checkpoint -- only the internal sampling path is.

exp09 W1 "teacher ceiling" ROLLOUT scoring (`scoring="rollout"`, driver flag
`--search-scoring rollout`; `scoring="myopic"`, the pre-existing behavior
above, stays the default and is completely UNCHANGED): a second scoring
mode for the SAME stage-1 candidates (generation + dedup + myopic ksim
scoring are shared, byte-for-byte, between both modes -- "myopic" simply
returns right after that shared stage; see `_search`). Purpose: myopic
scoring reranks candidates by ONE ksim oracle call against the LAST-SEEN
opponent board -- a purely MYOPIC (next-battle-only) signal that cannot see
whether a candidate trades this battle for a stronger position two turns
later (the project owner's tempo-bias concern). Rollout scoring measures
how much of that ceiling decision-time search can already reach by
replacing the myopic score, for a SHORTLIST only (full-game simulation is
too expensive for all 12 candidates), with an estimate of the candidate's
actual FULL-GAME outcome.

ALGORITHM (two stages; stage 1 identical to myopic):
1. Generate + dedup `n_candidates` end-boards exactly as myopic does, and
   score ALL of them with the existing single-ksim-call myopic score (this
   stage never changes based on `scoring`).
2. Take the top `rollout_shortlist` deduped candidates BY MYOPIC SCORE
   (`_search_rollout`). For each, `rollout_repeats` times
   (`_rollout_score_candidate`): resolve THIS turn's END_TURN battle against
   the followed opponent chain's actual board this turn
   (`eval_versus_fullgame.py::_resolve_versus_turn`, `simulation_count=
   rollout_ksim` -- a steadier signal for the turn actually being decided),
   then -- if the game is not already over -- play every SUBSEQUENT turn
   with the plain wrapped recommender's OWN standard decode
   (`eval_versus_fullgame.py::play_out_game`, `self.bc`, never re-searching)
   against the SAME chain's actual subsequent boards, to a lives verdict or
   the turn cap. Score = mean(final outcome: win=1.0, loss=0.0, cap/no-result
   =0.5) over the repeats, plus `0.01 * mean(own_lives - opp_lives at end)`
   as a tiebreak. Argmax wins; an EXACT tie breaks by the stage-1 myopic
   score (`_search_rollout`'s sort key).

ACTUAL-CHAIN-ONLY, NO FUTURE-AVERAGING (Ruihan's explicit design
constraint, 2026-07-19): every simulated turn -- the candidate's own turn
AND every subsequent turn of the continuation -- samples the SAME followed
opponent's ACTUALLY RECORDED chain (`self.opp_source.sample_for_pid`,
falling back to `.sample_random_with_rng` only when that chain runs out of
indexed turns, exactly like the real driver). This is privileged
training-time information (the real future is not observable at true
decision time) -- accepted for now because the entire point of this
measurement is a CEILING: "how good could search get if it could see far
enough ahead", not a deployable recommender. There is deliberately no
opponent-predictor, no ensemble over K plausible next-boards, and no
alternate-future branching of any kind -- one real future, replayed. This
also bounds the comparison: every candidate rolled out this turn is scored
against the exact SAME future (the outer game has not committed to any of
them yet), so this is a valid RERANKING of the current turn's choice, not a
prediction that plants a specific future.

NO shared-state mutation: each repeat runs the continuation on a FRESH
`copy.deepcopy` of the candidate's end board, and every random-fallback
draw during a continuation comes from an ISOLATED `random.Random` seeded
by `(self.seed, self._recommend_call_count, candidate_index, repeat_index)`
-- never `self.opp_source`'s own shared `_random_rng` (see
`chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`'s
docstring for why: a throwaway rollout simulation must not perturb the
REAL game's own later draws from that same shared source). The exact
(pid, turn) chain lookup (`sample_for_pid`) is a pure O(1) dict read with
no mutable state at all, so it is shared directly, unisolated.

exp09 W2 (distillation teacher data generation, `capture_candidate_chains`
constructor flag, default False -- fully backward compatible, every W1/W6a
caller is unaffected): when True, `recommend()`'s result additionally
carries each deduped stage-1 candidate's own `chain_preview` (the action
sequence, not just its score), aligned index-for-index with `search_scores`
(myopic mode: top-level key `search_candidate_chains`; rollout mode:
`search_diagnostics["candidate_chains"]`, aligned with that dict's own
`myopic_scores`). This costs one extra list of already-computed
`chain_preview` deep-copies per `recommend()` call -- cheap relative to the
oracle/rollout calls that dominate this class's cost -- but is skipped
entirely when the flag is off (default), so no existing measurement run
(W1, W6a) pays for or changes behavior from this addition. Purpose:
`tools/gen_distill_dataset.py`'s per-turn sidecar (PLAN.md W2) records the
full candidate set (not just scores) for later inspection/audit -- see that
module's docstring.

exp10 W2 (opponent-source ablation, `rollout_opponent_mode` constructor
kwarg, default `"true"` -- fully backward compatible, byte-identical to
every existing W1/W6a caller): W1's rollout ceiling (immediately above)
replays the ACTUAL followed opponent's recorded rest-of-game -- privileged
information a real decision-time search could never see. This addition
measures how much of that ceiling survives when the rollout continuation
follows a DIFFERENT opponent's recorded chain instead, via
`--rollout-opponent-mode {true,pool_random,retrieval}`
(`tools/eval_versus_fullgame.py`):

- `"true"` (default): unchanged -- `_rollout_score_candidate` never touches
  `board_copy`'s followed pid, so `sample_for_pid_fn=self.opp_source.
  sample_for_pid` keeps resolving against whatever pid the outer game is
  actually following, exactly as before this addition existed.
- `"pool_random"`: for each rollout repeat (ONE draw per repeat, same cost
  as `"true"` -- no K-averaging in this first pass), draw a random pid from
  `self.opp_source.all_pids` (already exactly the eval frame's own
  opponent pool -- e.g. val/Turtle/rank<=1500, whatever `opp_source` was
  constructed with -- see `chain_snapshot.py::ChainSnapshotSource`),
  EXCLUDING the true followed pid, and splice it into `board_copy["meta"]
  ["versus"]["current_opponent_participation_id"]` BEFORE this turn
  resolves. From there the EXISTING sampling plumbing
  (`resolve_end_turn_with_sampled_battle`'s forced-pid-then-random-
  fallback logic, completely unmodified) follows that pid's recorded
  chain turn by turn, falling back to the same isolated
  `sample_random_with_rng` draw the `"true"` path already uses whenever
  that chain runs short -- see `_choose_rollout_opponent_pid`.
- `"retrieval"`: identical to `"pool_random"`, additionally restricted to
  pids whose indexed chain length is >= the CURRENT turn (a "this
  candidate opponent's own game had actually reached this turn"
  plausibility filter -- rank/pack are already pool-wide invariants
  because `opp_source` itself was constructed with those bounds, so no
  separate rank/pack check is needed per draw).

Every draw uses an ISOLATED `random.Random` keyed on `(mode, seed,
recommend_call_count, candidate_index, repeat_index)` -- never
`self.opp_source`'s own shared state -- for the same reason the existing
`sample_random_with_rng` fallback draws are isolated (module docstring's
"NO shared-state mutation" paragraph above): a throwaway ablation draw
must never perturb the real game's own later draws. For auditability
(Ruihan's explicit ask: prove the arms actually differ), every mode's
`_rollout_score_candidate` result additionally carries `true_followed_pid`,
`repeat_chosen_opponent_pids`, `repeat_fallback_tiers`, and
`repeat_opponent_pid_traces` (the pid actually used to sample the
opponent board, per turn, per repeat -- captured via `play_out_game`'s
existing `on_turn` hook) -- present (empty-safe) for every mode so a
consumer never needs a schema branch.

exp12 route a, wave A0 -- COMMON RANDOM NUMBERS (`rollout_crn` constructor
kwarg, driver flag `--rollout-crn`, default False = byte-identical to every
run before it existed, including the live W2c width arms):

BEFORE this flag, every exogenous draw a rollout continuation makes was
keyed on `candidate_index`, so the r-th rollout of candidate 0 and the r-th
rollout of candidate 1 -- two estimates of the SAME decision, differing only
in the move under evaluation -- faced INDEPENDENT futures: a different
random-fallback opponent board, a different `pool_random`/`retrieval`
opponent pid, and a different engine (shop) RNG chain inherited from the
candidate's own end board. Comparing candidates then pays the full variance
of the future on top of the variance of the move, which is exactly the noise
a distillation target must not carry.

WITH `rollout_crn=True`, all three exogenous draws are keyed on
`(seed, decision_id, repeat_index)` where `decision_id` is `(game_index,
turn)` -- `candidate_index` is deliberately ABSENT:

- the isolated random-fallback sampler's seed;
- the `pool_random`/`retrieval` opponent-pid draw;
- the continuation's engine seed, written into `board_copy["meta"]["seed"]`
  before this turn resolves (only when the board already carries
  `seed_known`, never forced on) so every sibling's rest-of-game shops come
  off the same stream instead of off whatever chain that candidate's own
  shop actions happened to advance to.

So sibling candidates share repeat r's future, while different repeat
indices, different turns and different games stay independent. The battle
oracle itself is NOT under this flag's control and cannot be: the JS
simulator takes no seed (see `eval_versus_fullgame.py`'s module docstring,
"nothing seeds the JS simulator's own RNG"), so its Monte-Carlo draw is the
one residual per-candidate noise source. `--rollout-ksim` (a majority vote
over k sims) is what damps it.

`set_decision_context(game_index=...)` is how the driver tells this class
which game it is in; `turn` is read off the state each `recommend()` call.
Without the call (a direct/unit-test caller), `decision_id` falls back to
the per-process recommend counter, which is still constant across one
decision's candidates and distinct across decisions -- so the CRN property
holds, only cross-process reproducibility is lost.

exp12 route a, wave A0 -- TEACHER RECORD (`capture_teacher_record`
constructor kwarg, driver flag `--teacher-record-out`, default False):
when True the rollout result additionally carries a `teacher_record` block
with EVERY scored stage-1 candidate's own end board (the exact object handed
to scoring), its myopic score, its rollout score and per-repeat outcomes,
the CRN keys/seeds those repeats used, and which one was chosen. This is the
distillation training signal for route a (regress V on the teacher's
CONTINUOUS rollout scores instead of on 0/1 game outcomes); the driver's
`--teacher-record-out` turns it into a gzipped JSONL stream. See
`eval_versus_fullgame.py`'s "Wa addition" and
`tools/check_teacher_record.py` for the file's self-check gates.

exp12 W2 -- the LEARNED LEAF (`scoring="vgame"`, driver flag
`--search-scoring vgame`, constructor kwarg `vgame_scorer`; both other modes
are completely UNCHANGED and no run without the flag differs by a byte):

Stage 1 is shared, byte for byte, with `myopic` and `rollout`: the same
`n_candidates` chains are generated from the same wrapped recommender and
deduped by the same end-board signature, so a vgame arm at width W scores
exactly the candidate set a rollout arm at width W would have. What changes
is the LEAF. Instead of one ksim oracle call per candidate (myopic) or a
rest-of-game simulation per shortlisted candidate (rollout), every deduped
end board goes through ONE batched forward of a learned value function
(`tools/vgame_scorer.py::VGameLeafScorer.score_boards`) and the argmax of
`(1-blend)*V + blend*myopic - pessimism*ensemble_std` wins.

Two consequences worth stating because they are the point of the arm:

- At `blend == 0` (the default, and the deployed A4 configuration) NOTHING
  reads the stage-1 myopic score, so `_search` does not make the oracle
  calls at all. That is where the speed the W2 decision rule measures comes
  from, and it is why `search_greedy_score` and `search_diagnostics
  ["myopic_scores"]` are all None on such a run rather than being quietly
  filled with something the leaf did not use.
- The turn-1 skip is KEPT even though a V leaf does not need an opponent
  board, because W2 froze it that way for comparability with every other
  arm on this ruler.

The never-raise contract is strengthened rather than merely kept: a scorer
failure of ANY kind degrades that one turn to the greedy chain with
`search_used=False` and a `search_error` string, so a broken leaf costs
strength and is counted, never a crashed game.

The 4-dim race bypass the value function consumes needs pre-battle
cumulative wins, which no board carries; `set_race_context(wins=...)` is how
the driver supplies it once per turn, and a vgame search that was never told
it refuses to score rather than guessing (see that method).

exp13 AMENDMENT A1 -- THE HONEST FRAME (`turn_mode="segmented-honest"`,
driver flag `--turn-mode`; `"whole-determinized"` is the default and is
byte-for-byte everything above, so no run that does not ask for the new
frame moves):

Read `tools/honest_frame.py`'s module docstring first -- it owns the WHY
(search on a seeded engine sees this turn's roll before deciding whether to
roll) and the stream-S key. What changes HERE is stage 1 and the leaf, for
the vgame path only:

- The state this class is handed each segment is already an IMAGINED clone
  (the driver overrides `meta.seed` with `S(engine_seed, turn,
  segment_index, 0)`), so every engine walk this class does -- the wrapped
  recommender's greedy decode, `_sample_candidate`'s sampled decode,
  `_prefix_walk`'s replay -- runs on stream S. None of them can read the
  play stream's position. That is A1 ruling 1, and it costs this class no
  code: it inherits the separation from the state it is given.
- Candidates are deduped and ranked by their DETERMINISTIC PREFIX
  (`_prefix_walk`): every op up to and INCLUDING the first whose engine
  resolution reports a `stochastic_reason`. That prefix is the unit the
  driver actually commits before it stops and re-searches (A1 ruling 2), so
  it is the unit that must be scored. Boundary detection is by
  `stochastic_reason`, NEVER by an action-type whitelist, so a future engine
  randomness source is covered the day it is added.
- A prefix that ends in a stochastic action is scored as the MEAN of k
  imagined completions (`--stochastic-samples`, default 3): resample that
  one resolution under `S(..., sample_r=r)`, let the wrapped recommender
  greedily finish the turn from the resampled board, and score the end board
  with V. The sample keys carry `(decision, r)` and DELIBERATELY NOT the
  candidate index, so siblings that reach the same chance node face the same
  resampled outcome and only the move under evaluation differs (CRN). A
  prefix that is deterministic all the way to END_TURN is scored exactly as
  the determinized path scores it: one V forward on its end board.
- Why the mean and not the argmax of one sample: with the seed no longer
  shared between imagination and play, each candidate containing a ROLL
  would otherwise draw its OWN shop, and a plain argmax would
  systematically pick whichever candidate happened to roll well -- an
  overestimate of the value of rolling. Re-drawing the SAME chain k times
  is the opposite error (the original chain's post-roll buys turn illegal
  on a different shop and get truncated, systematically UNDER-valuing the
  roll), which is why the completion is regenerated greedily rather than
  replayed.
- Accepted residual biases, recorded rather than hidden (A1 ruling 4): a
  greedy completion lower-bounds the value of information, because the real
  agent re-searches after the roll instead of playing greedily; and argmax
  over k-sample means carries a small optimizer bias even after CRN. W3's
  depth-2 amendment upgrades the completion policy to a V-search over the
  chance node and is the principled fix for the first.

`scoring="myopic"`/`"rollout"` under the honest frame are REFUSED at
construction, not silently run: their leaves would rank candidates on the
single proposal-stream roll, which is exactly the pick-the-lucky-roll bias
above. Expectation scoring is implemented for the V leaf only (A1 ruling 3).

exp13 W0b' (the codex review of A1) tightened two things in that leaf, both
in the same direction -- an imagined completion must be imagined the way A1
says or not exist at all:

- The k completions decode GREEDILY whatever the proposal decoder is doing.
  `.deterministic` never controlled that (`BcRecommender.recommend` does not
  read it back; since the W0(c) dual-decode probe it branches on
  `.decode_mode`), so under `--decode-mode sample` the completions were
  SAMPLED and their noise entered the k-mean. `_call_bc_recommend`'s
  `force_ranked_decode` pins it, and only the completion path passes it:
  sampled PROPOSALS stay a legal, deliberate arm.
- A prefix whose completions ALL fail is DROPPED, and if it is the greedy
  chain's own prefix (or nothing survives) the whole decision degrades to
  greedy. It used to fall back to the proposal stream's board, which would
  have scored that one prefix on a single lucky/unlucky roll while its
  siblings were scored on k-means -- the exact bias A1 removes, reappearing
  only under failure, where nothing would have noticed.

exp13 AMENDMENT A2 -- THE STRUCTURAL CUT (`RESULTS_W0d.md` is the evidence):

A1 cut the turn at every reported `stochastic_reason`. The W0 addendum
measured what that bought: 5.304 segments per turn at 1.604 s each, with
`ability_randomness` 44.6% of all segments -- and 96.8% of the resolutions
that raised it were the event-order TIE-BREAK alone, 87.9% of which cannot
resolve a single handler. Tie order came back board-invariant in 3,106 of
3,106 exhaustive permutations. A2 therefore cuts on a STRUCTURAL criterion
instead: only when the resolution wrote a field `legal_actions` reads (the
engine reports it as `stochastic_structural`, see `engine.StepOutcome`).

exp13 AMENDMENT A4 (2026-08-06) makes that criterion CAUSAL. A2.1 checked
"randomness consumed" and "read-set written" separately over the whole step,
so a deterministic read-set write became a cut whenever anything else in the
same step happened to draw -- 1,218 of 30,015 searched segments across 697 of
1,000 games on the post-merge re-pin (`RESULTS_W1a2.md` section 9). The engine
now raises `stochastic_structural` only when the draw STEERED the write.
Nothing in this module changed: it reads the same boolean.

What changes in this module:

- A2.2. `_prefix_walk` cuts on `stochastic_structural`, so a prefix MAY now
  contain a non-structural stochastic resolution, resolved once on the
  proposal stream. The deterministic-prefix invariant is RETIRED: the prefix
  board is an ESTIMATE (it differs from the real committed board in which
  pets got buffed), not a prediction. The remaining chain still cannot turn
  illegal through that path, because `legal_actions` never reads stats, and
  `_imagined_completion` is mechanically unaffected -- it overrides
  `meta.seed` regardless and `pre_board` is still computed once for all k.
- A2.3. k applies at STRUCTURAL chance nodes only. Non-structural randomness
  inside a prefix resolves once, on the proposal stream, and is not averaged.
  Accepted bias, recorded: a candidate containing a stat resolution is scored
  on a single draw of it. All candidates in one decision share the proposal
  stream so the comparison stays paired, but the draw is not shared op-for-op
  between candidates whose chains differ; V's sensitivity to a few points of
  attack/health on random friends is low relative to what re-applying whole
  chains k times would cost.
- A2.4. `_sample_candidate` stops its walk at the first op whose transition
  reports a structural resolution, honest frame only. Decision-identical by
  construction: `_prefix_walk` truncates at the same op and `_prefix_group_key`
  is the prefix, so the winning candidate's committed ops are unchanged.
  Candidate 0 comes from `_call_bc_recommend`, not this path, so the
  search-failure fallback is untouched.
- A2.5. Candidate 0 passes `force_ranked_decode=True` unconditionally, so
  `--decode-mode sample` cannot take away the greedy anchor, the plain-BC
  failure fallback (`_annotate_skip`) or comparability with plain-BC arms.
  `.deterministic` never controlled the decoder (see `_call_bc_recommend`).
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from typing import Any, Callable

import numpy as np

# exp13 W0b': `imagined_step`, not `step`. Every engine walk in this module
# (`_apply_chain`, `_sample_candidate`, `_prefix_walk`, `_imagined_completion`)
# scores or proposes a board that the DRIVER then re-applies against its own
# real state; nothing here is ever committed. That makes these steps eligible
# for the opt-in schema-validation skip; with the skip off (the default) this
# is `api.step` exactly. See `api._SKIP_IMAGINED_VALIDATION`.
from ..api import imagined_step as engine_step
from ..oracles.sap_calc_battle_oracle import build_simulation_config
from ..train.env import ACTION_CATALOG, TrainingEnv
from ..visited_guard import set_training_rolls_this_turn, state_signature
from . import honest_frame
from .bc_recommender import DECODE_MODE_RANKED as _BC_DECODE_MODE_RANKED
from .bc_recommender import DEFAULT_MAX_CHAIN_STEPS as _BC_DEFAULT_MAX_CHAIN_STEPS
from .bc_recommender import legal_mask

# Constructor defaults, exported so callers (the eval driver's argparse
# defaults) share one source of truth instead of a second copy of "12"/"16".
DEFAULT_N_CANDIDATES = 12
DEFAULT_KSIM = 16
# exp16 W3: candidates per ANYTIME chunk, i.e. how much work a stop request
# has to wait out. W0 derived 4 from its 65.3 ms-per-candidate PROPOSAL cost
# (4 x 65.3 = 264 ms, inside exp16's 0.3 s stop budget), but it measured that
# under k=1 and priced the proposal only. Two costs it could not have
# included dominate under the honest frame:
#   1. a chunk also pays k=3 imagined completions per DISTINCT prefix, so one
#      candidate is ~110 ms end to end, not 65;
#   2. a stop is not over when the chunk is: the turn's REMAINING segments
#      still have to be greedy-finished before the AI has a turn to hand over
#      (5-45 ms each, and a turn runs 5-11 segments), which is another
#      50-250 ms and does not shrink with the chunk.
# Measured (duel_smoke, one torch thread, stop-latency median in seconds):
#   chunk   1      2      4      8
#   6 turns 0.12   0.23   0.30   0.43
#   24 games (276 turns): 0.152 / 0.163 at chunk 1, 0.209 / 0.235 at chunk 2
# 4 and 8 miss the budget outright. 2 passes on the 24-game sample but landed
# at 0.34 on one 36-turn sample, i.e. it makes the gate flaky rather than
# safe. 1 measured 0.154-0.185 across three independent 3-game samples with
# ~2x headroom, so that is the shipped default -- the exp16 PLAN's "if it
# misses, tune the chunk size". It costs ~125 vs ~110 ms per candidate of
# throughput, which only shows up when the AI has a long human turn to think
# through; `--chunk-size` trades it back.
DEFAULT_ANYTIME_CHUNK = 1
# Two group scores within this of each other are a TIE for the anytime
# path's argmax -- see `_search_anytime` for the batch-shape float noise
# it exists to absorb.
SCORE_TIE_EPS = 1e-6
DEFAULT_SEED = 0

# exp09 W1: `scoring=` choices + rollout-scoring defaults, exported for the
# same reason as the constants above (the eval driver's argparse defaults
# share this one source of truth).
SCORING_MYOPIC = "myopic"
SCORING_ROLLOUT = "rollout"
# exp12 W2 (deployed in wave A4): the LEARNED leaf. See the module
# docstring's "exp12 W2" section and `tools/vgame_scorer.py`.
SCORING_VGAME = "vgame"
SCORING_MODES: tuple[str, ...] = (SCORING_MYOPIC, SCORING_ROLLOUT, SCORING_VGAME)
DEFAULT_SCORING = SCORING_MYOPIC
DEFAULT_ROLLOUT_SHORTLIST = 4
DEFAULT_ROLLOUT_REPEATS = 4

#: The frame the ROLLOUT continuation resolves under. Plain strings rather than
#: imports from `eval_versus_fullgame`, which this module may only import lazily
#: (see the lazy import inside `_rollout_score_candidate`); they are checked
#: against that module's own constants at the point of use.
SEARCH_GAME_RULES_VERSUS = "versus"
SEARCH_GAME_RULES_ARENA = "arena"
SEARCH_GAME_RULES = (SEARCH_GAME_RULES_VERSUS, SEARCH_GAME_RULES_ARENA)
DEFAULT_SEARCH_GAME_RULES = SEARCH_GAME_RULES_VERSUS
#: Mirrors `eval_versus_fullgame.DEFAULT_ARENA_RACE_CONVENTION`.
DEFAULT_SEARCH_ARENA_RACE_CONVENTION = "trophies_mapped"

#: MC rerank (ledger A2.6): after V has ranked the menu, take the top
#: `mc_rerank_k` groups IN V ORDER and re-rank only those by `rollout_repeats`
#: rollouts each. 0 disables it, which is every run before exp22.
#:
#: NOT `scoring="rollout"`, which makes a rollout the leaf for EVERY candidate.
#: Ledger C7 records that as a different design (exp12 W2 section 9.3, 1482
#: oracle calls per game against 11) and says in as many words that it "is not
#: the same design" as this one, which is A2.6. The four refusals guarding
#: `scoring="rollout"` therefore stay exactly as they are.
DEFAULT_MC_RERANK_K = 0
DEFAULT_ROLLOUT_KSIM = 16
# Matches `eval_versus_fullgame.DEFAULT_MAX_TURN`; not imported from there
# (module-level import would cycle -- that module imports THIS one at its
# own top level) -- every real caller (`eval_versus_fullgame.py::main`)
# passes its own `--max-turn` explicitly, so this constant only matters for
# a caller/test that omits it.
DEFAULT_ROLLOUT_MAX_TURN = 30

# exp10 W2: `rollout_opponent_mode=` choices, exported for the same reason
# as the constants above (the eval driver's argparse default/choices share
# this one source of truth). See module docstring's "exp10 W2" section.
ROLLOUT_OPPONENT_TRUE = "true"
ROLLOUT_OPPONENT_POOL_RANDOM = "pool_random"
ROLLOUT_OPPONENT_RETRIEVAL = "retrieval"
ROLLOUT_OPPONENT_MODES: tuple[str, ...] = (
    ROLLOUT_OPPONENT_TRUE,
    ROLLOUT_OPPONENT_POOL_RANDOM,
    ROLLOUT_OPPONENT_RETRIEVAL,
)
DEFAULT_ROLLOUT_OPPONENT_MODE = ROLLOUT_OPPONENT_TRUE

# exp12 route a (wave A0): common-random-numbers keying for the rollout leaf
# + the teacher record, both OFF by default so no existing run (including the
# live W2c width arms) changes by a byte. See the module docstring's two
# "exp12 route a, wave A0" sections.
DEFAULT_ROLLOUT_CRN = False
DEFAULT_CAPTURE_TEACHER_RECORD = False
# Salts, exported so the checker tool can re-derive a recorded key instead of
# hard-coding a second copy of the format string.
CRN_FALLBACK_SALT = "exp12_wa_crn_fallback"
CRN_OPPONENT_SALT = "exp12_wa_crn_opponent"
CRN_ENGINE_SALT = "exp12_wa_crn_engine"
# Width of the CRN continuation seed, in bits -- see `_crn_engine_seed`.
# Exported so the test that pins "the range the engine accepts" and this
# derivation share one number instead of two copies of it.
CRN_ENGINE_SEED_BITS = 63
TEACHER_RECORD_SCHEMA = "exp12-teacher-record/v1"

# exp13 (2026-08-09): `_sample_candidate` draws again when a walk committed
# NOTHING because a DRAW was rejected. A walk that had nothing to draw
# (`no_legal_actions`, `legal_mask_failed`) is not retried -- it would draw
# the same nothing -- and neither is a walk that committed at least one op,
# which is already a candidate. 8 is a backstop, not a tuning knob: the
# measured rate of an empty walk is ~1 candidate in 5,000 (176 of 1,044,576
# on W1b, 74 of 386,280 on the W1c pilot), so the second attempt clears it
# on every board except one whose legal mass is nearly all cycles.
RESAMPLABLE_EMPTY_STOPS: frozenset[str] = frozenset(
    {"sampled_cycle", "sampled_action_illegal", "sampled_action_error"}
)
#: Every way `_sample_walk` can stop WITHOUT the decoder choosing to end the
#: turn. `end_turn_chosen` is the only one that means it finished, so this is
#: its complement, written out rather than inferred -- a completion is rescued
#: on a NAMED failure, never on "the reason is not the one string I know".
#: Getting that polarity wrong dropped 6 of 10 boards in exp13's honest-frame
#: fixture, whose duck-typed proposer spells its success `stub_end_turn`.
#:
#: `test_exp16_completion_tail` asserts this set still covers every literal
#: `_sample_walk` can emit, so a new stop reason fails loudly instead of
#: quietly counting as finished.
UNFINISHED_WALK_STOPS: frozenset[str] = frozenset(
    {
        "cap_reached",
        "legal_mask_failed",
        "no_legal_actions",
        "sampled_no_action_left",
        "sampled_action_error",
        "sampled_action_illegal",
        "sampled_cycle",
        "structural_boundary",
    }
)
MAX_SAMPLE_ATTEMPTS = 8
#: exp16 W11b safety valve on the `resample-clock` extra-sample loop. The
#: segment clock is the normal stop; this only bounds the pathological case.
MAX_EXTRA_SAMPLE_LEVELS = 4096


#: exp22 W3. What fills in the rest of the turn behind a chance node while the
#: search is still RANKING. `bc_greedy` is every run before 2026-08-19: one
#: greedy BC decode, which lower-bounds the value of information because the
#: real agent re-searches after the roll instead of playing greedily (this
#: module's own "Accepted residual biases" note). `v_search` is that note's
#: named fix: propose several completions and let V pick.
#: exp16 W0 (2026-08-20). What a SAMPLED completion does when its walk would
#: otherwise give up before the turn is over. Measured at the real call site
#: (525 sampled completions over 25 walks that stop at a chance node): 45.0%
#: reached `end_turn_chosen`, 38.7% stopped on `sampled_cycle`, 16.4% on
#: `structural_boundary`. So 55% never finished, the half-chain is NOT dropped
#: (`ok = bool(chain)`), and `_completion_agg`'s `max` then compares one
#: finished board against up to `completion_width - 1` unfinished ones while
#: the V leaf is trained on end-of-turn boards.
#:
#: `stop` is that behaviour. It was the default until 2026-08-21 and is kept as
#: a named value so a run can reproduce anything measured before then.
#:
#: THE DEFAULT IS NOW `resample`. Ruihan ruled the unfinished completion a
#: defect (`16-play-vs-ai/PLAN_W11.md` appendix W0-Z): a completion that quits
#: with gold unspent, a roll unused, or the turn unended describes a position
#: that cannot occur in a real game, so scoring it is scoring a fiction, and
#: that is true whether or not V notices. The measurements that had argued for
#: leaving it alone were all proxies for "did V's number move", and W0-Z voids
#: every one of them as an input to this decision.
#:
#: `resample` beat `finish_greedy` on the two axes the ruling leaves open,
#: measured paired in one window over 90 (walk, sample) units at inner width 8
#: (`16-play-vs-ai/probe_completion_pick.py`, raw in `w0_pick.json`):
#:   completeness  both reach END_TURN on 100% of alternatives, against 51.3%
#:                 for `stop`
#:   diversity     6.27 distinct boards of 8 and 81.0% differing from index 0,
#:                 against `finish_greedy`'s 5.58 and 74.8%
#:   cost          a tie: paired ratio 0.970x with an IQR of [0.877, 1.028],
#:                 which contains 1.0. The previously published 64% vs 78% gap
#:                 did NOT replicate under a paired estimator.
#: Both cost about 1.4x of `stop` per completion in that window, and that part
#: is real: under a fixed clock it buys fewer segments.
COMPLETION_TAIL_STOP = "stop"
#: Let the walk stop, then hand the tail to the greedy decoder -- what index 0
#: already does. Closes comparability; costs diversity, because every
#: alternative then ends the same way.
COMPLETION_TAIL_GREEDY = "finish_greedy"
#: Do not let it stop: no structural break inside a completion (that break's
#: own justification, "everything past it is provably discarded", is true of an
#: outer CANDIDATE and false here), and on a would-be revisit mask that action
#: and draw again from what is left instead of giving up. Ruihan's, 2026-08-20.
COMPLETION_TAIL_RESAMPLE = "resample"
COMPLETION_TAILS = (
    COMPLETION_TAIL_STOP,
    COMPLETION_TAIL_GREEDY,
    COMPLETION_TAIL_RESAMPLE,
)

COMPLETION_BC_GREEDY = "bc_greedy"
COMPLETION_V_SEARCH = "v_search"
COMPLETION_POLICIES = (COMPLETION_BC_GREEDY, COMPLETION_V_SEARCH)

#: How one imagined sample's inner completions collapse to one number.
#: `max` is the deepening, and it is also the SEMANTICALLY right aggregation:
#: the chance outcome has already been averaged over one layer out, by the
#: `stochastic_samples` loop, so a max here is not optimism about the roll --
#: it is the agent doing what it really does, picking the best continuation it
#: can find once the shop is known. `mean` is the deliberately-degraded
#: CONTROL: identical boards, identical compute, but the extra options are not
#: allowed to be chosen, so the arm pays for depth and then throws it away.
#:
#: RETRACTED 2026-08-19: an earlier version of this note called `mean` the
#: control that separates "the search found a better continuation" from "a max
#: over more estimates inflates them". It does not separate anything, because
#: the inflation it named would have to enter through the chance node, and the
#: chance node is handled by the outer mean.
COMPLETION_AGG_MAX = "max"
COMPLETION_AGG_MEAN = "mean"
COMPLETION_AGGREGATES = (COMPLETION_AGG_MAX, COMPLETION_AGG_MEAN)


def completion_agg(values: list[float], aggregate: str = COMPLETION_AGG_MAX) -> float:
    """One imagined sample's inner completions -> one number.

    The ONE definition of the inner aggregation. Both honest assembly sites
    (`_search_vgame_honest`, `_search_anytime`) reach it through
    `SearchRecommender._completion_agg`, and the Bellman labeller imports it
    directly (`w1_bellman_targets._score_group`), so the policy the label
    values and the policy the search plays cannot disagree about what a group
    score means. See `COMPLETION_AGG_MAX` for why `max` is the semantics and
    `mean` the degraded control.

    A single value returns unchanged, which is what makes `bc_greedy` (inner
    width 1) bit-identical to every run before deepening existed.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if aggregate == COMPLETION_AGG_MAX:
        return float(max(values))
    return sum(values) / float(len(values))

def _read_last_opponent_team(state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """`state["meta"]["versus"]["last_opponent_team"]`, deep-copied, or None
    if absent/wrong-typed/empty (turn 1, before any battle has resolved --
    see `train/env.py::_set_last_opponent_team`, this field's only writer).
    Never raises: every level of the path is defensively type-checked.
    """
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return None
    versus = meta.get("versus")
    if not isinstance(versus, dict):
        return None
    team = versus.get("last_opponent_team")
    if not isinstance(team, list) or not team:
        return None
    return copy.deepcopy(team)


def _read_current_opponent_pid(state: dict[str, Any]) -> str | None:
    """`state["meta"]["versus"]["current_opponent_participation_id"]`, or
    None if absent/wrong-typed/empty. This is the pid `resolve_end_turn_
    with_sampled_battle` will look up to resolve THIS turn's battle (see
    `end_turn.py`'s `forced_pid_source == "versus_chain"` branch) -- exp10
    W2's `"true"`-vs-substituted-mode correctness proof reads it both
    before a rollout continuation starts (the true followed pid to
    exclude/compare against) and after each turn resolves (mirrors
    `_read_last_opponent_team`'s defensive-read style). Never raises.
    """
    meta = state.get("meta")
    if not isinstance(meta, dict):
        return None
    versus = meta.get("versus")
    if not isinstance(versus, dict):
        return None
    pid = str(versus.get("current_opponent_participation_id") or "").strip()
    return pid or None


def _apply_chain(state: dict[str, Any], chain: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-apply `chain` (a list of action dicts, as found in a recommend-
    result's `chain_preview`) op-by-op from a FRESH `copy.deepcopy` of
    `state`, via `api.step` -- the identical replay procedure
    `tools/eval_versus_fullgame.py::play_one_game` itself uses on the
    winning chain (that function's per-turn loop: `ops = [op for op in
    chain_preview if ... != "END_TURN"]` then `engine_step(board, op)` in a
    loop). Reusing the SAME procedure here, rather than trusting whatever
    intermediate state a candidate generator tracked internally, guarantees
    the board this module scores a candidate on is bit-identical to the
    board the driver will actually reach if that candidate wins -- if the
    two diverged, search could pick a candidate based on a board the real
    replay never lands on.

    END_TURN is never applied (it does not change the shop-phase board;
    `play_one_game` skips it the same way). Stops at the first
    illegal/failing step -- the board reached so far still counts as this
    candidate's end board (a partially-applied chain is still a valid, if
    suboptimal, candidate rather than a hard failure).
    """
    work = copy.deepcopy(state)
    for action in chain:
        if not isinstance(action, dict):
            break
        if str(action.get("type") or "").strip().upper() == "END_TURN":
            break
        try:
            trans = engine_step(work, action)
        except Exception:
            break
        if not trans.get("legal"):
            break
        next_state = trans.get("state_after")
        if not isinstance(next_state, dict):
            break
        work = next_state
    return work


def _annotate_skip(greedy_result: dict[str, Any]) -> dict[str, Any]:
    """The greedy candidate's result, unchanged, with `search_*` keys added
    to record that NO search happened (turn 1 / no opponent yet, or every
    oracle call failed). Deliberately distinct from the `recommend()`-level
    exception fallback: this is a designed, expected outcome (not an
    error), so `search_error` is never set here.
    """
    result = dict(greedy_result)
    result.update(
        {
            "search_used": False,
            "search_n_candidates": 0,
            # exp12 W2c width telemetry: schema parity only -- nothing was
            # generated or deduped on a skipped turn.
            "search_n_generated": 0,
            "search_n_dedup": 0,
            "search_scores": [],
            "search_chosen_index": 0,
            "search_greedy_score": None,
            # exp09 W2: schema parity with the searched-and-scored result
            # shape (always present, always None here -- there was nothing
            # to capture chains FOR).
            "search_candidate_chains": None,
        }
    )
    return result


def _searched_verdict(chosen_chain: list[dict[str, Any]]) -> dict[str, Any]:
    """The four keys a COMPLETED prefix search owns on its own result.

    Both honest assembly sites (`_search_vgame_honest`, `_search_anytime`)
    build their result as `dict(winner)` and then overwrite what the SEARCH
    decided. `winner` is the raw candidate that was FIRST to reach the winning
    prefix group -- a bookkeeping pointer into `candidates`, not a verdict on
    this decision -- so `ok` and `error` have to be overwritten with the rest.
    They were not. A winner whose own decode committed nothing carries
    `ok=False` (`_sample_candidate`), `run_segmented_turn` reads exactly that
    field, and a search that had deduped, scored and ranked every prefix and
    picked one therefore ended the game as an agent terminal failure instead of
    playing its own choice. All 82 `decode_failed` games in exp13's 8,548
    honest games on disk are that, and nothing else (`census_empty_chain.py`).

    An empty winning prefix is a CHOICE, not a failure. Its dedup key is
    `("det", signature(start board))` -- the same key an END_TURN-first
    candidate walks to -- and the board it was scored on is the decision's own
    start board, so what won is "end the turn now with what is on the board".
    Which of the two chains the group carries is an accident of which candidate
    index reached that key first, so this says it the way the other one would:
    the driver filters END_TURN out of the ops it commits, so the turn resolves
    with nothing committed either way, and `recommended_action` stops being
    `None` for the consumers that read it (`play_web`'s `/api/infer/recommend`
    would otherwise answer `ok` with no action in it).

    `diagnostics` stays the winning candidate's on purpose: it is decode
    telemetry about that chain, and `run_segmented_turn`'s stop-reason sink --
    hence every `stop_reasons` histogram already on disk -- reads `stop_reason`
    off it.
    """
    chain = chosen_chain if chosen_chain else [{"type": "END_TURN"}]
    return {
        "ok": True,
        "error": None,
        "chain_preview": chain,
        "recommended_action": copy.deepcopy(chain[0]),
    }


class SearchRecommender:
    """Best-of-N search wrapper over a `BcRecommender` (or API-compatible
    stand-in) for exp09 W6a. See the module docstring for the full
    algorithm and the candidate-diversity design note.
    """

    def __init__(
        self,
        bc_recommender: Any,
        *,
        n_candidates: int = DEFAULT_N_CANDIDATES,
        ksim: int = DEFAULT_KSIM,
        seed: int = DEFAULT_SEED,
        battle_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        max_chain_steps: int | None = None,
        scoring: str = DEFAULT_SCORING,
        rollout_shortlist: int = DEFAULT_ROLLOUT_SHORTLIST,
        rollout_repeats: int = DEFAULT_ROLLOUT_REPEATS,
        rollout_ksim: int = DEFAULT_ROLLOUT_KSIM,
        rollout_opponent_mode: str = DEFAULT_ROLLOUT_OPPONENT_MODE,
        opp_source: Any | None = None,
        max_turn: int = DEFAULT_ROLLOUT_MAX_TURN,
        capture_candidate_chains: bool = False,
        rollout_crn: bool = DEFAULT_ROLLOUT_CRN,
        capture_teacher_record: bool = DEFAULT_CAPTURE_TEACHER_RECORD,
        vgame_scorer: Any | None = None,
        turn_mode: str = honest_frame.DEFAULT_TURN_MODE,
        stochastic_samples: int = honest_frame.DEFAULT_STOCHASTIC_SAMPLES,
        completion_policy: str = COMPLETION_BC_GREEDY,
        completion_width: int = 1,
        completion_tail: str = COMPLETION_TAIL_RESAMPLE,
        completion_aggregate: str = COMPLETION_AGG_MAX,
        game_rules: str = DEFAULT_SEARCH_GAME_RULES,
        arena_race_convention: str = DEFAULT_SEARCH_ARENA_RACE_CONVENTION,
        mc_rerank_k: int = DEFAULT_MC_RERANK_K,
    ) -> None:
        # The SAME loaded model -- this class never loads a checkpoint of
        # its own, only ever calls into the one it was handed (see module
        # docstring's "candidate diversity" section for exactly how).
        self.bc = bc_recommender
        # The frame the rollout continuation resolves under. The `versus`
        # default keeps every pre-exp22 run byte-identical.
        if game_rules not in SEARCH_GAME_RULES:
            raise ValueError(
                f"search_recommender_bad_game_rules:{game_rules!r}:"
                f"expected one of {SEARCH_GAME_RULES!r}"
            )
        self.game_rules = str(game_rules)
        self.arena_race_convention = str(arena_race_convention)
        self.mc_rerank_k = int(mc_rerank_k)
        if self.mc_rerank_k < 0:
            raise ValueError(
                f"search_recommender_bad_mc_rerank_k:{mc_rerank_k!r}:must_be_at_least_0"
            )
        self.n_candidates = int(n_candidates)
        self.ksim = int(ksim)
        self.seed = int(seed)
        # None => lazy-import `run_battle_oracle_with_config` on first use
        # (see `_resolve_battle_fn`); tests inject a stub here instead.
        self._battle_fn = battle_fn
        self.max_chain_steps = (
            int(max_chain_steps)
            if max_chain_steps is not None
            else int(getattr(bc_recommender, "max_chain_steps", _BC_DEFAULT_MAX_CHAIN_STEPS))
        )
        # Bumped once per `recommend()` call (once per turn during a game)
        # so each turn's candidate RNGs draw fresh entropy instead of
        # repeating the same per-candidate sequence turn after turn -- see
        # `_rng_for_candidate`.
        self._recommend_call_count = 0

        # exp09 W1 rollout scoring (module docstring's ROLLOUT section).
        if scoring not in SCORING_MODES:
            raise ValueError(f"search_recommender_bad_scoring:{scoring!r}:expected_one_of={SCORING_MODES}")
        self.scoring = str(scoring)
        self.rollout_shortlist = int(rollout_shortlist)
        self.rollout_repeats = int(rollout_repeats)
        self.rollout_ksim = int(rollout_ksim)
        # exp10 W2: see module docstring's "exp10 W2" section. Validated
        # unconditionally (not just when scoring="rollout") so a bad value
        # is caught at construction time regardless of which scoring mode
        # a caller also passed.
        if rollout_opponent_mode not in ROLLOUT_OPPONENT_MODES:
            raise ValueError(
                f"search_recommender_bad_rollout_opponent_mode:{rollout_opponent_mode!r}:"
                f"expected_one_of={ROLLOUT_OPPONENT_MODES}"
            )
        self.rollout_opponent_mode = str(rollout_opponent_mode)
        self.max_turn = int(max_turn)
        if self.scoring == SCORING_ROLLOUT and opp_source is None:
            # Rollout scoring must sample the followed chain's actual
            # subsequent boards (module docstring's "actual-chain-only, no
            # future-averaging" section) -- unlike myopic scoring, which
            # only ever reads `state["meta"]["versus"]["last_opponent_team"]`
            # and needs no opponent source at all, so this is required ONLY
            # for this mode, not a blanket constructor requirement.
            raise ValueError("search_recommender_scoring_rollout_requires_opp_source")
        self.opp_source = opp_source
        # exp09 W2: see module docstring's "capture_candidate_chains" section.
        self.capture_candidate_chains = bool(capture_candidate_chains)
        # exp12 route a (wave A0): see the module docstring's two
        # "exp12 route a, wave A0" sections.
        self.rollout_crn = bool(rollout_crn)
        self.capture_teacher_record = bool(capture_teacher_record)
        # Set by `set_decision_context` (the driver, once per game) and by
        # `recommend` (the turn, once per decision). Both None for a caller
        # that never sets them -- see `_crn_decision_id`.
        self._crn_game_index: int | None = None
        self._decision_turn: int | None = None
        # exp12 W2 (wave A4): the learned leaf, and the one race scalar no
        # board carries. See the module docstring's "exp12 W2" section.
        self.vgame_scorer = vgame_scorer
        if self.scoring == SCORING_VGAME and vgame_scorer is None:
            raise ValueError("search_recommender_scoring_vgame_requires_vgame_scorer")
        self._race_wins: int | None = None

        # exp13 A1: the honest frame. See the module docstring's own A1
        # section. Default `whole-determinized` == every line above,
        # unchanged, so nothing that does not ask for the new frame moves.
        self.turn_mode = honest_frame.normalize_turn_mode(turn_mode)
        self.honest = honest_frame.is_honest(self.turn_mode)
        self.stochastic_samples = int(stochastic_samples)
        if self.stochastic_samples < 1:
            raise ValueError(
                f"search_recommender_bad_stochastic_samples:{stochastic_samples!r}:must_be_at_least_1"
            )
        if self.mc_rerank_k and self.scoring != SCORING_VGAME:
            # The rerank re-orders a ranking V produced. Without a V ranking
            # there is no "top k in V order" to re-rank, and taking the
            # shortlist off a myopic score is the defect `DECISIONS.md:316`
            # named on 2026-08-25: the implemented rollout operator picks its
            # shortlist by stage-1 myopic score while the priced `Pick_MC(k,m)`
            # picks by V order, "so W3 is a wave of implementation".
            raise ValueError(
                f"search_recommender_mc_rerank_requires_vgame_scoring:{self.scoring!r}:"
                f"mc_rerank_k={self.mc_rerank_k} needs scoring={SCORING_VGAME!r} so the "
                f"shortlist is taken in V order"
            )
        if self.honest and self.scoring != SCORING_VGAME:
            # LOUD rather than silently biased: a myopic/rollout leaf under
            # the honest frame ranks candidates on the ONE proposal-stream
            # roll, i.e. it picks whichever candidate got lucky. Expectation
            # scoring (A1 ruling 3) is implemented for the V leaf only.
            raise ValueError(
                f"search_recommender_honest_frame_requires_vgame_scoring:{self.scoring!r}:"
                f"expectation scoring at the root (exp13 PLAN Amendment A1 section 3) is "
                f"implemented for scoring={SCORING_VGAME!r} only; a {self.scoring!r} leaf "
                f"under turn_mode={honest_frame.TURN_MODE_SEGMENTED_HONEST!r} would rank candidates on a "
                f"single imagined roll and systematically prefer the lucky one"
            )
        # exp22 W3: the completion policy behind a chance node.
        if completion_policy not in COMPLETION_POLICIES:
            raise ValueError(
                f"search_recommender_bad_completion_policy:{completion_policy!r}:"
                f"expected_one_of={COMPLETION_POLICIES}"
            )
        if completion_aggregate not in COMPLETION_AGGREGATES:
            raise ValueError(
                f"search_recommender_bad_completion_aggregate:{completion_aggregate!r}:"
                f"expected_one_of={COMPLETION_AGGREGATES}"
            )
        if completion_tail not in COMPLETION_TAILS:
            raise ValueError(
                f"search_recommender_bad_completion_tail:{completion_tail!r}:"
                f"expected_one_of={COMPLETION_TAILS}"
            )
        self.completion_tail = str(completion_tail)
        self.completion_policy = str(completion_policy)
        self.completion_aggregate = str(completion_aggregate)
        self.completion_width = int(completion_width)
        if self.completion_width < 1:
            raise ValueError(
                f"search_recommender_bad_completion_width:{completion_width!r}:must_be_at_least_1"
            )
        if self.completion_policy == COMPLETION_V_SEARCH:
            # A "deepening" arm at width 1 proposes exactly the greedy
            # completion and nothing else, i.e. it is the old arm wearing the
            # new name. exp22 already shipped one confident null about an
            # intervention that had barely happened (b1's 2.55% extractor
            # travel against exp12's 15.67%); refusing at construction is the
            # cheap end of that lesson.
            if self.completion_width < 2:
                raise ValueError(
                    f"search_recommender_v_search_needs_width_ge_2:{self.completion_width}:"
                    "width 1 proposes only the greedy completion, so the arm would "
                    "measure the bc_greedy policy under the v_search name"
                )
            if not self.honest:
                raise ValueError(
                    f"search_recommender_v_search_requires_honest_frame:{self.turn_mode!r}:"
                    "there is no chance node to search over under the determinized frame"
                )
            if self.scoring != SCORING_VGAME:
                raise ValueError(
                    f"search_recommender_v_search_requires_vgame_scoring:{self.scoring!r}"
                )
        elif self.completion_width != 1:
            raise ValueError(
                f"search_recommender_bc_greedy_width_must_be_1:{self.completion_width}:"
                "the greedy policy emits exactly one completion per imagined sample"
            )
        # Counted per decision, reported on the result, and gated upstream:
        # how often the inner search's pick differs from the greedy one.
        self._completion_divergent = 0
        self._completion_decided = 0
        # How many inner boards a sample ACTUALLY got, against how many the
        # config asked for. See `_imagined_completions` for why a configured
        # width is not evidence that the width happened.
        self._completion_boards = 0
        self._completion_samples = 0
        self._completion_dropped = 0
        # Direct evidence for the completion-policy contract. A returned board
        # is classified at the point where the walk's stop reason and tail
        # policy are still available, before V can score it.
        self._completion_terminal_boards = 0
        self._completion_unfinished_boards = 0
        # exp16 W11b. How many chance nodes a completion walked THROUGH, which
        # `MEASURED.md`'s draft row for this layer says nothing records: "there
        # is neither a tunable nor any telemetry saying how many chance nodes a
        # chain contains". Counted for every tail, so the three are comparable:
        # under `stop`/`finish_greedy` the walk breaks at the first one, under
        # `resample` it keeps going, and the difference is the point.
        self._completion_second_chance = 0
        #: Completions whose sampled walk gave up and were finished greedily so
        #: that no half-played board is ever scored. Reported on the search
        #: result rather than archived per segment: the measured incidence is
        #: zero, and a schema field for an event nothing produces is the shape
        #: this wave already declined once.
        self._completion_rescued = 0

        # Set by `set_imagination_context` (the driver, once per SEGMENT).
        # None => `_imagination_seed` falls back to the handed state's own
        # seed and says so in the diagnostics -- see that method.
        self._imagination_engine_seed: int | None = None
        self._imagination_segment_index: int = 0
        # The seed of the state the current `recommend()` was handed, read
        # defensively there; only ever used as the fallback key base above.
        self._state_seed: int = 0

    def set_decision_context(self, *, game_index: int | None, turn: int | None = None) -> None:
        """exp12 route a (wave A0): tell this recommender which GAME the
        decisions it is about to make belong to, so CRN keys and teacher
        records are a pure function of `(seed, game_index, turn,
        repeat_index)` rather than of how many `recommend()` calls this
        process happened to make first (which sharding breaks).

        Called once per game by `eval_versus_fullgame.py::play_one_game`,
        duck-typed, so a `BcRecommender` or any other recommender that does
        not define it is simply not called. Never raises.

        `turn` (exp12 route a, track b) is optional and exists for callers
        that score a RECORDED decision directly through
        `_rollout_score_candidate` instead of playing a game -- the offline
        re-scorer `tools/rescore_agreement_teacher.py` is the only one. A
        live `recommend()` call always overwrites it from the state it was
        handed, so this can never shadow the driver's own turn.
        """
        self._crn_game_index = None if game_index is None else int(game_index)
        if turn is not None:
            self._decision_turn = int(turn)

    def _crn_decision_id(self) -> str:
        """Stable id of the DECISION (never of a candidate) that every CRN
        key below is scoped to: `(game_index, turn)` when the driver has
        supplied both, else this process's recommend counter -- which is
        also constant across one decision's candidates and distinct across
        decisions, so the CRN property survives, only cross-process
        reproducibility does not.
        """
        if self._crn_game_index is not None and self._decision_turn is not None:
            return f"g{int(self._crn_game_index)}:t{int(self._decision_turn)}"
        return f"call{int(self._recommend_call_count)}"

    def _crn_fallback_key(self, *, candidate_index: int, repeat_index: int) -> str:
        """Seed string for a repeat's isolated random-fallback sampler.
        Under CRN the candidate index is ABSENT (that is the whole point);
        with CRN off this returns the pre-A0 key byte-for-byte."""
        if self.rollout_crn:
            return f"{CRN_FALLBACK_SALT}:{self.seed}:{self._crn_decision_id()}:{repeat_index}"
        return (
            f"w1_rollout_fallback:{self.seed}:{self._recommend_call_count}:"
            f"{candidate_index}:{repeat_index}"
        )

    def _crn_opponent_key(self, *, candidate_index: int, repeat_index: int) -> str:
        """Seed string for a repeat's `pool_random`/`retrieval` pid draw --
        same CRN rule as `_crn_fallback_key`."""
        if self.rollout_crn:
            return (
                f"{CRN_OPPONENT_SALT}:{self.rollout_opponent_mode}:{self.seed}:"
                f"{self._crn_decision_id()}:{repeat_index}"
            )
        return (
            f"w2_opponent_mode:{self.rollout_opponent_mode}:{self.seed}:"
            f"{self._recommend_call_count}:{candidate_index}:{repeat_index}"
        )

    def _crn_engine_seed(self, repeat_index: int) -> int:
        """The engine (shop) seed every sibling candidate's repeat-`r`
        continuation is started from under CRN. sha256 rather than
        `hash()` so it is stable across processes and python runs.

        Width (exp12 Wa, codex review finding 2): `CRN_ENGINE_SEED_BITS` = 63,
        NOT the 31 this first shipped with. 31 bits is the range the engine
        REGENERATES into (`engine._reseed_meta` draws
        `randrange(0, 2**31)`, the .NET-heritage `int.MaxValue` range), but
        nothing on the accept side is bound by it: `schemas/state_v1.json`
        types `meta.seed` as an unbounded integer, the rust mirror
        (`sap_engine_core::Meta`) types it `Option<u64>`, the only consumer
        is `engine._rng_from_state`'s `random.Random(seed)` which takes any
        int, and `meta` never crosses into the JS battle oracle (see
        `build_simulation_config`, which sends pets and packs only), so no
        float-precision boundary applies either. At A1 volume the narrow
        range mattered: ~320k (decision, repeat) keys drawn from 2**31 gives
        ~24 EXPECTED birthday collisions, i.e. ~24 pairs of decisions
        silently sharing a shop stream; from 2**63 the same draw expects
        5.5e-9 of one. `--rollout-crn` is what makes this seed authoritative,
        so a collision there is a coupled pair of teacher labels, not noise
        that averages out.
        """
        key = f"{CRN_ENGINE_SALT}:{self.seed}:{self._crn_decision_id()}:{repeat_index}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") >> (64 - CRN_ENGINE_SEED_BITS)

    def _resolve_battle_fn(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        if self._battle_fn is None:
            from ..oracles.sap_calc_battle_oracle import run_battle_oracle_with_config

            self._battle_fn = run_battle_oracle_with_config
        return self._battle_fn

    def _rng_for_completion(self, sample_r: int, index: int) -> np.random.Generator:
        """RNG for inner completion `index` of imagined sample `sample_r`.

        Keyed on the IMAGINATION seed, not on the play seed and not on the
        group, for two reasons that are both load-bearing. Imagination,
        because a completion proposal drawn off the play stream would be
        peeking at the shop the agent has not rolled yet. Not the group,
        because the outer frame deliberately keys `(decision, r)` without the
        candidate index so siblings reaching the same chance node face the
        same resampled shop (CRN); drawing the inner proposals per group
        would put an independent noise draw back under the argmax that CRN
        exists to remove.
        """
        seed_seq = np.random.SeedSequence(
            [
                int(self.seed),
                int(self._recommend_call_count),
                int(self._imagination_seed(sample_r)) & 0xFFFFFFFF,
                int(index),
            ]
        )
        return np.random.default_rng(seed_seq)

    def _rng_for_candidate(self, index: int) -> np.random.Generator:
        """Deterministic-given-(seed, turn, candidate) RNG stream: distinct
        across candidates within one turn AND across turns within one game
        (via `_recommend_call_count`), so repeated turns in a long game do
        not all sample the exact same noise pattern.
        """
        seed_seq = np.random.SeedSequence([int(self.seed), int(self._recommend_call_count), int(index)])
        return np.random.default_rng(seed_seq)

    def _call_bc_recommend(
        self,
        state: dict[str, Any],
        *,
        deterministic: bool,
        force_ranked_decode: bool = False,
    ) -> dict[str, Any]:
        """Call `self.bc.recommend(state)` with `.deterministic` temporarily
        set to `deterministic`, restoring the ORIGINAL value in a `finally`
        (matches the spec this class was built from: candidate 0 always
        uses `deterministic=True`; `_generate_candidate`'s fallback path
        uses `deterministic=False`). A no-op toggle if `self.bc` has no
        `deterministic` attribute at all (defensive only).

        `force_ranked_decode` (exp13 W0b', codex finding 2) additionally pins
        `.decode_mode` to `"ranked"` for the duration of the call. It exists
        because `.deterministic` DOES NOT CONTROL the real decoder: as this
        module's docstring already records, `BcRecommender.recommend` never
        reads that attribute back, and since the W0(c) dual-decode probe it
        branches on `self.decode_mode` instead. So under
        `--decode-mode sample` the toggle above is inert and every walk this
        class asks for comes back SAMPLED -- which is the arm's own choice for
        PROPOSALS (they stay sampled, deliberately), but is a frame violation
        for the k imagined COMPLETIONS of a stochastic prefix: A1 section 3
        defines them as greedy, and sampling them injects decoder noise into
        the k-mean that expectation scoring is supposed to average the ENGINE's
        randomness over. Only `_imagined_completion` passes it.
        """
        has_det = hasattr(self.bc, "deterministic")
        original_det = self.bc.deterministic if has_det else None
        has_decode = force_ranked_decode and hasattr(self.bc, "decode_mode")
        original_decode = self.bc.decode_mode if has_decode else None
        try:
            if has_det:
                self.bc.deterministic = bool(deterministic)
            if has_decode:
                self.bc.decode_mode = _BC_DECODE_MODE_RANKED
            return self.bc.recommend(state)
        finally:
            if has_det:
                self.bc.deterministic = original_det
            if has_decode:
                self.bc.decode_mode = original_decode

    def _sample_candidate(
        self,
        state: dict[str, Any],
        rng: np.random.Generator,
        *,
        completion_mode: bool = False,
        for_completion: bool = False,
    ) -> dict[str, Any]:
        """One sampled candidate: `_sample_walk`, drawn again while it is empty.

        A walk that rejects its very first draw has committed nothing, and an
        empty chain is not a candidate -- it is a candidate that was not
        produced. Emitting it anyway put `ok=False` into the search's candidate
        list, where `_prefix_walk` reduced it to an empty committable prefix
        that then competed on merit and, when it won, handed the driver a
        result that said both "here is my choice" and "I failed" (exp13 W1c
        game 647, and the 82 `decode_failed` games in the data on disk).

        So draw again. Each attempt is an independent walk from the SAME masked
        distribution, so what changes is how often this class of candidate
        occurs, not the distribution a returned chain is drawn from: the
        candidate is the sampler's own distribution conditioned on committing
        at least one op. It costs one extra decode on ~1 candidate in 5,000.

        Only DRAW rejections are retried (`RESAMPLABLE_EMPTY_STOPS`). A walk
        that had nothing to draw (`no_legal_actions`, `legal_mask_failed`)
        would draw the same nothing, and one that is still empty after
        `MAX_SAMPLE_ATTEMPTS` has had its chances; both come back `ok=False`
        exactly as before. An empty chain therefore stays REACHABLE, which is
        why `_searched_verdict` owns the search's verdict rather than trusting
        that this method no longer produces one.

        `diagnostics.sample_attempts` is how many walks this candidate cost, so
        a resample can never be invisible.
        """
        attempts = 0
        while True:
            attempts += 1
            chain, stop_reason = self._sample_walk(
                state, rng, completion_mode=completion_mode,
                for_completion=for_completion,
            )
            if (
                chain
                or attempts >= MAX_SAMPLE_ATTEMPTS
                or stop_reason not in RESAMPLABLE_EMPTY_STOPS
            ):
                break

        ok = bool(chain)
        return {
            "ok": ok,
            "error": None if ok else stop_reason,
            "recommended_action": copy.deepcopy(chain[0]) if chain else None,
            "chain_preview": chain,
            "wdl_probs": None,
            "diagnostics": {
                "stop_reason": stop_reason,
                "chain_length": len(chain),
                "sampled": True,
                "sample_attempts": attempts,
            },
        }

    def _sample_walk(
        self,
        state: dict[str, Any],
        rng: np.random.Generator,
        *,
        completion_mode: bool = False,
        for_completion: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """ONE stochastically-sampled walk from `state`: the same
        per-step machinery `BcRecommender.recommend` uses (`legal_mask`,
        `self.bc.encoder`/`self.bc.model` via `_masked_action_probs`, the
        within-turn `state_signature` anti-cycle set) but DRAWS each step
        from the masked categorical distribution (`rng.choice`) instead of
        always taking the arg-max -- see the module docstring's "candidate
        diversity" section.

        Unlike that method's guarded scan (which, on a would-be revisit,
        tries the NEXT-best legal action instead of stopping), a sampled
        walk that would revisit an already-seen board, or draws an action
        that turns out illegal to apply, simply STOPS there -- there is no
        well-defined "next best" for a single random draw the way there is
        for a full probability ranking. Stopping after at least one committed
        op still leaves a valid candidate -- `_apply_chain`'s own "the board
        reached so far still counts" contract applies equally here -- but
        stopping on the FIRST draw leaves nothing, which is what
        `_sample_candidate` draws again for.

        A2.4: under the HONEST frame the walk also stops at the first op whose
        transition reports a STRUCTURAL resolution, because everything past it
        is provably discarded -- `_prefix_walk` truncates the chain at exactly
        that op and `_prefix_group_key` keys on the prefix, so the tail never
        reaches a score, a dedup key or the driver. The measured discarded tail
        was a median of 0 ops (`RESULTS_W0d.md` part 1), and BC decode plus
        engine replay is 97% of wall clock. Off under the determinized frame,
        where the whole-turn chain IS the scored unit.

        `completion_mode` (exp16 W0, 2026-08-20) is for a walk that is filling
        in the rest of a turn behind a chance node rather than proposing a
        committable prefix. BOTH early stops above are wrong there and right
        here, which is why it is a parameter rather than a change:

        - the structural break's own justification is "everything past it is
          provably discarded", true of an outer candidate and FALSE of a
          completion, whose tail is exactly the board that gets scored;
        - the cycle stop gives up because a single draw has no well-defined
          "next best". A completion can simply mask that action and draw again
          from what is left, which keeps it a sample instead of a stump.

        With `completion_mode=False` nothing is banned, `pool is legal_idx`,
        and the draw is the identical `rng.choice` on the identical array, so
        the proposal path is byte-identical. `test_exp16_completion_tail`
        pins that.
        """
        work = copy.deepcopy(state)
        set_training_rolls_this_turn(work, 0)  # same observation-parity reset BcRecommender.recommend uses
        chain: list[dict[str, Any]] = []
        visited: set[str] = {state_signature(work)}
        stop_reason = "cap_reached"

        for _ in range(self.max_chain_steps):
            try:
                mask = legal_mask(work)
            except Exception:
                stop_reason = "legal_mask_failed"
                break
            legal_idx = np.flatnonzero(mask)
            if legal_idx.size == 0:
                stop_reason = "no_legal_actions"
                break

            obs = self.bc.encoder.encode(work)
            probs = np.asarray(self.bc._masked_action_probs(obs, mask), dtype=np.float64)

            # `banned` only ever grows in completion mode, so with it empty
            # `pool is legal_idx` and every draw below is the identical call.
            banned: set[int] = set()
            step = None
            while True:
                pool = legal_idx
                if banned:
                    pool = legal_idx[~np.isin(legal_idx, np.fromiter(banned, dtype=legal_idx.dtype))]
                    if pool.size == 0:
                        step = ("stop", "sampled_no_action_left")
                        break
                legal_probs = probs[pool]
                total = float(legal_probs.sum())
                legal_probs = (legal_probs / total) if total > 0.0 else np.full(pool.size, 1.0 / pool.size)
                chosen = int(rng.choice(pool, p=legal_probs))
                action = ACTION_CATALOG[chosen]

                if str(action.get("type") or "").strip().upper() == "END_TURN":
                    step = ("end_turn", action)
                    break

                try:
                    trans = engine_step(work, action)
                except Exception:
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_action_error")
                    break
                next_state = trans.get("state_after")
                if not trans.get("legal") or not isinstance(next_state, dict):
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_action_illegal")
                    break
                sig = state_signature(next_state)
                if sig in visited:
                    if completion_mode:
                        banned.add(chosen)
                        continue
                    step = ("stop", "sampled_cycle")
                    break
                step = ("commit", action, trans, next_state, sig)
                break

            if step[0] == "stop":
                stop_reason = step[1]
                break
            if step[0] == "end_turn":
                chain.append(copy.deepcopy(step[1]))
                stop_reason = "end_turn_chosen"
                break

            _, action, trans, next_state, sig = step
            chain.append(copy.deepcopy(action))
            work = next_state
            visited.add(sig)
            # W11b instrument, and it sits OUTSIDE the break below on purpose
            # so the three tails stay comparable: `stop` and `finish_greedy`
            # break here and so count at most one per walk, `resample` walks on
            # and counts each. Keyed on `for_completion` rather than
            # `completion_mode` because the latter is the BEHAVIOUR switch and
            # is only on under `resample`; this has to fire for all three.
            if for_completion and trans.get("stochastic_structural"):
                self._completion_second_chance += 1
            # The structural break belongs to the PROPOSAL frame only: see the
            # docstring for why its justification is false for a completion.
            if self.honest and not completion_mode and trans.get("stochastic_structural"):
                stop_reason = "structural_boundary"
                break
        else:
            stop_reason = "cap_reached"

        return chain, stop_reason

    def _generate_candidate(self, state: dict[str, Any], index: int) -> dict[str, Any]:
        """Candidate `index` (>= 1; candidate 0 is always the greedy call
        made directly in `recommend`, never routed through here). Prefers
        this class's own sampling decode (genuine diversity) when `self.bc`
        exposes the lower-level `.model`/`.encoder` surface a real
        `BcRecommender` has; falls back to toggling `.deterministic` and
        calling `.recommend()` again for minimal duck-typed stand-ins (this
        module's own unit tests) that do not expose it -- see the module
        docstring's "candidate diversity" section for why both paths exist.
        The internal path's `except Exception: pass` is deliberate (mirrors
        `bc_recommender.py`'s own "try the next-best/fallback path" style,
        e.g. `_choose_next_action`'s `except Exception: continue`): if
        `self.bc` claims the richer surface but it turns out unusable for
        any reason, this candidate is not lost, it is generated the
        guaranteed-safe way instead.
        """
        if hasattr(self.bc, "model") and hasattr(self.bc, "encoder"):
            try:
                return self._sample_candidate(state, self._rng_for_candidate(index))
            except Exception:
                pass
        return self._call_bc_recommend(state, deterministic=False)

    def _score_end_board(self, end_board: dict[str, Any], opponent_team: list[dict[str, Any]]) -> float:
        """One oracle call, `simulation_count=self.ksim`, scored as
        `(playerWins - opponentWins) / ksim` (PLAN.md's k-sim
        expected-outcome formula; see `train/gym_env.py::
        ksim_lives_outcome` for the training-time sibling of this same
        math). Raises on any failure -- never returns a sentinel -- so the
        caller (`_search`'s per-candidate scoring loop) can `except` it and
        exclude just that one candidate rather than the whole search.
        """
        config = build_simulation_config(
            end_board, opponent_team=copy.deepcopy(opponent_team), simulation_count=self.ksim
        )
        response = self._resolve_battle_fn()(config)
        if not isinstance(response, dict):
            raise RuntimeError(f"battle_fn_returned_non_dict:{type(response).__name__}")
        if "ok" in response and not response.get("ok"):
            raise RuntimeError(f"battle_fn_reported_failure:{response.get('error')}")
        payload = response.get("result")
        if not isinstance(payload, dict):
            payload = response  # tolerate a flat stub with no ok/result envelope
        player_wins = float(payload.get("playerWins", 0))
        opponent_wins = float(payload.get("opponentWins", 0))
        return (player_wins - opponent_wins) / float(self.ksim)

    def _search(self, state: dict[str, Any], greedy_result: dict[str, Any]) -> dict[str, Any]:
        opponent_team = _read_last_opponent_team(state)
        if not opponent_team:
            # Turn 1 (or any turn before a battle has ever resolved):
            # nothing to score candidates against yet. Skip generating the
            # other n_candidates-1 candidates entirely, not just the oracle
            # calls -- without an opponent there is nothing for them to be
            # scored against (see module docstring).
            return _annotate_skip(greedy_result)

        candidates = [greedy_result]
        for i in range(1, self.n_candidates):
            candidates.append(self._generate_candidate(state, i))

        # exp13 A1: under the honest frame the unit that gets scored is the
        # deterministic PREFIX, not the whole-turn end board, so stage 1's
        # end-board dedup below is replaced wholesale -- see
        # `_search_vgame_honest`. Only reachable with `scoring="vgame"` (the
        # constructor refuses every other leaf under this mode).
        if self.honest:
            return self._search_vgame_honest(
                state=state,
                greedy_result=greedy_result,
                candidates=candidates,
                opponent_team=opponent_team,
            )

        # De-duplicate by end-board signature (`_apply_chain`, the same
        # replay procedure the driver itself uses on the winning chain): a
        # board reached by more than one candidate costs one oracle call.
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []  # signatures, first-seen (== oracle-call) order
        sig_by_index: list[str] = []
        for i, cand in enumerate(candidates):
            end_board = _apply_chain(state, cand.get("chain_preview") or [])
            sig = state_signature(end_board)
            sig_by_index.append(sig)
            if sig not in groups:
                groups[sig] = {"end_board": end_board, "first_index": i, "score": None, "scored": False}
                order.append(sig)

        # exp12 W2: the LEARNED leaf does not read the myopic score at all
        # unless `blend != 0`, and the stage-1 oracle call is precisely the
        # cost the W2 rule's speed criterion is about, so it is SKIPPED when
        # nothing will read it. Every other mode (and a blended vgame leaf)
        # takes the unchanged path below, byte for byte.
        if self.scoring == SCORING_VGAME and not self._vgame_needs_myopic():
            for sig in order:
                groups[sig]["score"] = None
                groups[sig]["scored"] = True
            scored_sigs = list(order)
        else:
            for sig in order:
                group = groups[sig]
                try:
                    group["score"] = self._score_end_board(group["end_board"], opponent_team)
                    group["scored"] = True
                except Exception:
                    pass  # this group is excluded from the argmax below; others still count

            scored_sigs = [sig for sig in order if groups[sig]["scored"]]
            if not scored_sigs:
                # Every oracle call failed (e.g. the calculator CLI/worker is
                # down) -- nothing usable was searched, so behave exactly like
                # the no-opponent case above rather than picking a candidate on
                # no evidence. Applies to BOTH scoring modes: without even one
                # scored candidate there is nothing to shortlist for rollout
                # either.
                return _annotate_skip(greedy_result)

        greedy_group = groups[sig_by_index[0]]
        greedy_score = greedy_group["score"] if greedy_group["scored"] else None
        # None both when the greedy candidate's own oracle call failed and
        # (vgame, blend 0) when no oracle call was made at all.

        # exp09 W2: see module docstring's "capture_candidate_chains" section
        # -- None (the default) unless a caller opted in; skipped entirely
        # otherwise so no existing (W1/W6a) caller pays for or changes this.
        candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains or self.capture_teacher_record:
            candidate_chains = [
                copy.deepcopy(candidates[groups[sig]["first_index"]].get("chain_preview") or [])
                for sig in scored_sigs
            ]

        # exp12 W2: `scoring="vgame"` replaces the leaf -- see
        # `_search_vgame`'s docstring.
        if self.scoring == SCORING_VGAME:
            return self._search_vgame(
                state=state,
                greedy_result=greedy_result,
                candidates=candidates,
                groups=groups,
                scored_sigs=scored_sigs,
                greedy_score=greedy_score,
                candidate_chains=(candidate_chains if self.capture_candidate_chains else None),
                n_dedup=len(order),
            )

        # exp09 W1: `scoring="rollout"` takes over from here -- see
        # `_search_rollout`'s docstring. `scoring="myopic"` (default) falls
        # straight through to the UNCHANGED code below.
        if self.scoring == SCORING_ROLLOUT:
            return self._search_rollout(
                candidates=candidates,
                groups=groups,
                scored_sigs=scored_sigs,
                greedy_score=greedy_score,
                candidate_chains=(candidate_chains if self.capture_candidate_chains else None),
                teacher_chains=candidate_chains,
                n_dedup=len(order),
            )

        best_sig = max(scored_sigs, key=lambda sig: groups[sig]["score"])
        winner = candidates[groups[best_sig]["first_index"]]

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_n_candidates": len(scored_sigs),
                # exp12 W2c (width curve): how many chains were GENERATED at
                # this turn's requested width vs how many DISTINCT end
                # boards they collapsed to. `search_n_candidates` above is
                # the deduped set that actually got scored, so it also drops
                # any group whose oracle call failed; `search_n_dedup` is
                # the proposer's real diversity, which is what a width arm
                # is buying (see the module docstring's "candidate
                # diversity" note -- a width that dedups away is fake).
                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(len(order)),
                "search_scores": [float(groups[sig]["score"]) for sig in scored_sigs],
                "search_chosen_index": scored_sigs.index(best_sig),
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),
                # exp09 W2: None unless `capture_candidate_chains=True`.
                "search_candidate_chains": candidate_chains,
            }
        )
        return result

    def set_race_context(self, *, wins: int | None) -> None:
        """exp12 W2 (wave A4): the one race scalar no board carries.

        `turn`, `lives` and `opponent_lives` are all readable off the state
        search was handed, but PRE-BATTLE CUMULATIVE WINS (Vic semantics,
        RESULTS_W1 finding 2) lives only in the driver's own `wins_so_far`
        counter, and the V bypass block needs it (RESULTS_W1 finding 9's
        serve contract). `eval_versus_fullgame.py::play_out_game` calls this
        once per turn before `bc.recommend`, duck-typed, so a recommender
        that does not define it is simply never called.

        A vgame search that was never told `wins` REFUSES to score (the
        search is skipped and `search_error` says why) rather than guessing
        0, because a wrong bypass value is a silent train/serve skew and a
        skipped search is a loud one. Never raises.
        """
        self._race_wins = None if wins is None else int(wins)

    def _vgame_needs_myopic(self) -> bool:
        """Whether this vgame leaf reads the stage-1 myopic scores (i.e.
        `blend != 0`). Defensive: a scorer that does not expose the property
        is treated as needing them, so the oracle calls are only ever
        skipped on an explicit False."""
        return bool(getattr(self.vgame_scorer, "needs_myopic", True))

    def _race_scalars(self, state: dict[str, Any]) -> dict[str, int]:
        """The DECISION's driver-true race block for the V bypass.

        `lives` and `opponent_lives` are read off the state search was
        handed, which is exactly where `eval_versus_fullgame.py` reads its
        own `pre_turn_lives` / `pre_turn_opp_lives` from, so the two agree by
        construction. `wins` comes from `set_race_context`. Raises if `wins`
        was never supplied -- the caller turns that into a skipped search.
        """
        if self._race_wins is None:
            raise RuntimeError("vgame_race_wins_unset")
        meta = state.get("meta") if isinstance(state, dict) else None
        versus = (meta or {}).get("versus") if isinstance(meta, dict) else None
        return {
            "turn": int(state.get("turn", 0) or 0),
            "lives": int(state.get("lives", 0) or 0),
            "opponent_lives": int((versus or {}).get("opponent_lives", 0) or 0),
            "wins": int(self._race_wins),
        }

    # ------------------------------------------------------------------
    # exp13 Amendment A1: the honest frame (stream separation, prefix dedup,
    # expectation scoring). See the module docstring's own A1 section and
    # `tools/honest_frame.py`.
    # ------------------------------------------------------------------
    def set_imagination_context(
        self, *, engine_seed: int | None, segment_index: int = 0
    ) -> None:
        """exp13 A1 (ruling 1): which IMAGINATION STREAM this segment plans on.

        `engine_seed` is the GAME's own engine seed (`play_one_game`'s
        `engine_seed_rng` draw, a pure function of `(--seed, game_index)`),
        never the state's chained `meta.seed` -- keying on the latter would
        key on the play stream's position, which is the coupling A1 removes.
        `segment_index` counts stochastic boundaries already crossed THIS
        turn, so the re-search after a real ROLL imagines off a different
        stream than the search that chose to roll.

        Called once per SEGMENT by `eval_versus_fullgame.py::play_out_game`,
        duck-typed exactly like `set_decision_context`/`set_race_context`, so
        a plain `BcRecommender` is simply never called. Never raises.
        """
        self._imagination_engine_seed = None if engine_seed is None else int(engine_seed)
        self._imagination_segment_index = int(segment_index)

    def _imagination_seed(self, sample_r: int) -> int:
        """`S(engine_seed, turn, segment_index, sample_r)` for this decision.

        With the driver's context set (the only configuration exp13 runs),
        this is a pure function of `(--seed, game_index, turn, segment,
        sample_r)`. WITHOUT it -- a unit test, or a caller like exp16's duel
        loop that has not wired the hook yet -- it falls back to the seed of
        the state it was handed, which the driver has ALREADY overridden to
        `S(..., 0)`, so the derived sample streams are still a pure function
        of the same tuple and still independent of the play stream. The
        fallback is reported (`_imagination_key_source`) rather than assumed,
        so a report cannot claim the honest frame while running off it.
        """
        if self._imagination_engine_seed is not None:
            engine_seed = int(self._imagination_engine_seed)
            segment_index = int(self._imagination_segment_index)
        else:
            engine_seed = int(self._state_seed)
            segment_index = 0
        return honest_frame.imagination_seed(
            engine_seed=engine_seed,
            turn=int(self._decision_turn or 0),
            segment_index=segment_index,
            sample_r=int(sample_r),
        )

    def _imagination_key_source(self) -> str:
        return (
            honest_frame.KEY_SOURCE_DRIVER_CONTEXT
            if self._imagination_engine_seed is not None
            else honest_frame.KEY_SOURCE_STATE_SEED
        )

    def _prefix_walk(
        self, state: dict[str, Any], chain: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """The chain's COMMITTABLE PREFIX and where it stopped (A2.2).

        Replays `chain` op by op from a fresh copy of `state` through
        `api.step` -- the same procedure `_apply_chain` and the driver use --
        and stops at the first op whose transition reports a STRUCTURAL
        stochastic resolution (`stochastic_structural`), INCLUDING that op.
        Detection is by that field alone, never by an action-type or pet
        whitelist, so ROLL, the level-up reward slot, a random summon and any
        future randomness that writes `legal_actions`'s read-set are all
        boundaries the day they exist. The driver's commit loop filters on the
        SAME boolean, so the search cut and the execution cut are identical by
        construction.

        A2.2 RETIRES THE DETERMINISTIC-PREFIX INVARIANT, explicitly. Under A1
        this walk stopped at every `stochastic_reason`, so the prefix consumed
        no randomness and the board it reached was a PREDICTION of the board
        the driver would commit. Under A2 a prefix may contain a NON-structural
        resolution (a stat buff on random friends, a random-target food),
        resolved once on the proposal stream, so:
        - `board` and `pre_board` are ESTIMATES. They differ from the real
          committed board in which pets got buffed, never in what is legal,
          because `legal_actions` never reads attack or health.
        - the remaining chain still cannot become illegal through that path,
          so `chain_replay_diverged` stays reachable only through a structural
          resolution, which this walk still cuts.
        - `_imagined_completion` is mechanically unaffected: it overrides
          `meta.seed` regardless of what `pre_board` carries, and `pre_board`
          is still computed once and shared by all k samples.

        Returns `{"ops", "boundary", "board", "pre_board", "stop"}`:
        - `ops`: the prefix, END_TURN excluded (as everywhere else here).
        - `boundary`: the `stochastic_reason` of the STRUCTURAL op the walk
          stopped at, or None when the prefix ran to the end of the chain
          without one. Non-structural resolutions passed on the way are not
          reported here: they are not chance nodes for scoring (A2.3) and not
          cut points for execution.
        - `board`: the board the prefix reaches, on the PROPOSAL stream.
        - `pre_board`: the board just BEFORE the structural op (None when
          there is none).
        - `stop`: why the walk ended (`end_turn`/`stochastic_boundary`/
          `chain_end`, or one of the defensive `illegal_on_replay`/
          `step_error`/`no_state_after`/`malformed_action`).
        """
        work = copy.deepcopy(state)
        ops: list[dict[str, Any]] = []
        boundary: str | None = None
        pre_board: dict[str, Any] | None = None
        stop = "chain_end"
        for action in chain:
            if not isinstance(action, dict):
                stop = "malformed_action"
                break
            if str(action.get("type") or "").strip().upper() == "END_TURN":
                stop = "end_turn"
                break
            try:
                trans = engine_step(work, action)
            except Exception:
                stop = "step_error"
                break
            if not trans.get("legal"):
                stop = "illegal_on_replay"
                break
            next_state = trans.get("state_after")
            if not isinstance(next_state, dict):
                stop = "no_state_after"
                break
            reason = trans.get("stochastic_reason")
            ops.append(copy.deepcopy(action))
            if trans.get("stochastic_structural"):
                boundary = str(reason or "structural")
                pre_board = work
                work = next_state
                stop = "stochastic_boundary"
                break
            work = next_state
        return {
            "ops": ops,
            "boundary": boundary,
            "board": work,
            "pre_board": pre_board,
            "stop": stop,
        }

    @staticmethod
    def _prefix_group_key(walk: dict[str, Any]) -> tuple[str, ...]:
        """The dedup key: one entry per DISTINCT committable prefix.

        A deterministic prefix is keyed by the END BOARD it reaches, exactly
        as the determinized path keys its candidates, so two different op
        orders landing on the same board still cost one V forward.

        A stochastic prefix is keyed by `(board just before the chance node,
        the stochastic op itself)` rather than by the whole op sequence: two
        candidates that reach the same board by different orderings and then
        roll are the SAME decision, they face the same distribution, and
        collapsing them is what makes the k-sample cost scale with distinct
        chance nodes instead of with candidate width. The post-op board is
        deliberately NOT in the key -- it is a draw, not a decision.
        """
        if walk["boundary"] is None:
            return ("det", state_signature(walk["board"]))
        pre = walk["pre_board"] if isinstance(walk["pre_board"], dict) else walk["board"]
        last_op = walk["ops"][-1] if walk["ops"] else {}
        return (
            "stoch",
            state_signature(pre),
            json.dumps(last_op, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _prefix_chain(walk: dict[str, Any]) -> list[dict[str, Any]]:
        """What `chain_preview` becomes for a winning candidate: the prefix.

        The driver commits op by op and stops at the boundary anyway, so
        truncating here is belt-and-braces -- but it makes the contract exact
        ("what search chose IS what gets committed") for every other consumer
        of a recommend result too, and it means a post-boundary op decoded
        against an imagined shop can never be handed to anyone. END_TURN is
        re-appended only when the walk consumed the whole chain and that
        chain really did end on END_TURN, i.e. when the policy's own choice
        to stop is what ended the prefix.
        """
        chain = [copy.deepcopy(op) for op in walk["ops"]]
        if walk["stop"] == "end_turn":
            chain.append({"type": "END_TURN"})
        return chain

    def _reset_completion_counters(self) -> None:
        """Zero the per-search completion instruments.

        They are reported on every segment record, so they have to describe
        THAT search. They were only ever zeroed in `__init__`, which made
        `completion_dropped` a running total for the life of the process and
        `completion_boards_mean` a lifetime average: one drop in segment 3 and
        every later segment carried it. The first reading happened to be
        `(8.0, 0)`, and a zero total looks identical either way, so nothing
        said so. Called at the top of both search entry points.
        """
        self._completion_boards = 0
        self._completion_samples = 0
        self._completion_dropped = 0
        self._completion_terminal_boards = 0
        self._completion_unfinished_boards = 0
        self._completion_second_chance = 0
        #: Completions whose sampled walk gave up and were finished greedily so
        #: that no half-played board is ever scored. Reported on the search
        #: result rather than archived per segment: the measured incidence is
        #: zero, and a schema field for an event nothing produces is the shape
        #: this wave already declined once.
        self._completion_rescued = 0

    def _completion_instruments(self) -> dict[str, Any]:
        """The completion instruments as reported, defined once so the two
        search paths cannot drift into meaning different things by them."""
        return {
            "search_completion_samples": int(self._completion_samples),
            "search_completion_boards": int(self._completion_boards),
            "search_completion_boards_mean": (
                float(self._completion_boards) / float(self._completion_samples)
                if self._completion_samples
                else None
            ),
            "search_completion_terminal_boards": int(
                self._completion_terminal_boards
            ),
            "search_completion_unfinished_boards": int(
                self._completion_unfinished_boards
            ),
            "search_completion_terminal_rate": (
                float(self._completion_terminal_boards) / float(self._completion_boards)
                if self._completion_boards
                else None
            ),
            "search_completion_dropped": int(self._completion_dropped),
            "search_completion_second_chance_nodes": int(self._completion_second_chance),
            "search_completion_rescued": int(self._completion_rescued),
        }

    def _imagined_completions(self, walk: dict[str, Any], sample_r: int) -> list[dict[str, Any]]:
        """The boards one imagined sample contributes.

        Under `bc_greedy` (every run before 2026-08-19) that is exactly one
        board and this is the old `_imagined_completion` unchanged. Under
        `v_search` it is up to `completion_width` boards, of which index 0 is
        always the greedy one.

        One of the k imagined completions of a stochastic prefix.

        Re-applies the ONE stochastic op on a clone of `pre_board` seeded
        with `S(..., sample_r)` (everything earlier in the prefix was
        deterministic, so replaying it would land on the identical board),
        then lets the wrapped recommender greedily decode the REST of the
        turn from the resampled board and applies that chain through
        `_apply_chain`. The returned end board is what V scores.

        The completion runs on the same `S(..., sample_r)` stream, so a
        second roll inside the completion chains off it rather than off the
        play stream -- imagination stays imagination all the way down.

        The completion is GREEDY whatever the proposal decoder is doing
        (`force_ranked_decode`, exp13 W0b' codex finding 2) -- see
        `_call_bc_recommend` for why `deterministic=True` alone does not
        achieve that and why the two decodes are allowed to differ.
        """
        # The greedy board comes from `_imagined_completion` rather than being
        # inlined here, and that is deliberate: it keeps ONE failure site for
        # a sample, which is what the fail-closed tests monkeypatch. Inlining
        # it made those three tests pass while no longer reaching the code
        # they pin -- the same shape as a gate that reports green having
        # measured nothing.
        greedy_board = self._imagined_completion(walk, sample_r)
        if self.completion_policy == COMPLETION_BC_GREEDY:
            return [greedy_board]
        # Recomputing the resampled board costs one clone plus one engine step
        # and is a pure function of `(walk, sample_r)`, so it lands on the
        # identical board the greedy completion started from. That is cheap
        # next to the width-N decodes below, and it is the price of leaving
        # the singular method's contract untouched.
        after = self._resample_after(walk, sample_r)

        # exp22 W3. The greedy completion stays inner candidate 0, so the
        # deepened arm inherits the outer search's "can never do worse than
        # greedy" property: under `max`, a set whose every alternative scores
        # lower keeps index 0, and `max` returns the FIRST maximal element.
        boards = [greedy_board]
        terminal_boards = 1
        unfinished_boards = 0
        for j in range(1, self.completion_width):
            try:
                cand = self._sample_completion(after, sample_r, j)
                if self.completion_tail == COMPLETION_TAIL_RESAMPLE:
                    chain = (cand or {}).get("chain_preview") or []
                    # This tail's ENTIRE claim is that the board it hands to V
                    # is an end-of-turn board, and two walk outcomes break that
                    # while looking like success: `cap_reached` returns a
                    # non-empty chain that simply ran out of steps, and
                    # `sampled_no_action_left` (the pool emptied under the
                    # mask) is not in RESAMPLABLE_EMPTY_STOPS, so an EMPTY
                    # chain comes straight back -- and `_apply_chain(after, [])`
                    # would score the unchanged MID-TURN board. `ok` is not
                    # consulted at this call site and `_sample_candidate`'s own
                    # docstring says an empty chain stays reachable.
                    #
                    # So it is dropped and counted, exactly as a raising
                    # alternative already is: this sample degrades toward the
                    # greedy completion, which is the accepted degradation, and
                    # never toward a fiction. Found by the codex review,
                    # 2026-08-21; measured incidence in that window was 0 of
                    # 630, which is why it is a correctness fix and not a
                    # performance one.
                    reason = str(((cand or {}).get("diagnostics") or {}).get("stop_reason") or "")
                    if not chain or reason in UNFINISHED_WALK_STOPS:
                        # RESCUED, not dropped: dropping would return fewer than
                        # `completion_width` boards for this sample, and that
                        # count is a contract elsewhere. The greedy finisher is
                        # the same decoder index 0 already is, so the board is
                        # an end-of-turn board either way and only this rare
                        # alternative loses its sampled tail.
                        self._completion_rescued += 1
                        boards.append(
                            self._finish_completion_greedily(_apply_chain(after, chain))
                        )
                        terminal_boards += 1
                        continue
                board = _apply_chain(after, cand.get("chain_preview") or [])
                if self.completion_tail == COMPLETION_TAIL_GREEDY:
                    board = self._finish_completion_greedily(board)
            except Exception:
                # One failed alternative must not lose the sample: the greedy
                # completion is already in hand, so this degrades toward the
                # old policy for this sample rather than dropping it.
                #
                # But it is COUNTED (2026-08-20, found by the exp22 line). It
                # used to be a bare `continue`: the alternative vanished, and
                # the telemetry recorded `search_completion_width`, which is
                # the CONFIGURED width. So "inner width 8" was never checked
                # against how many boards a sample actually got -- it could be
                # 8 or 3 and the record said 8 either way. That is exactly the
                # rule exp22's `AMENDMENT 6` registered for arms ("name the
                # quantity that says the treatment took"), never applied to
                # this deployed knob.
                self._completion_dropped += 1
                continue
            boards.append(board)
            if self.completion_tail == COMPLETION_TAIL_STOP:
                reason = str(
                    ((cand or {}).get("diagnostics") or {}).get("stop_reason") or ""
                )
                if not ((cand or {}).get("chain_preview") or []) or reason in UNFINISHED_WALK_STOPS:
                    unfinished_boards += 1
                else:
                    terminal_boards += 1
            else:
                terminal_boards += 1
        if terminal_boards + unfinished_boards != len(boards):
            raise AssertionError(
                "search_recommender_completion_terminal_accounting_mismatch:"
                f"terminal={terminal_boards}:unfinished={unfinished_boards}:"
                f"boards={len(boards)}"
            )
        self._completion_boards += len(boards)
        self._completion_samples += 1
        self._completion_terminal_boards += terminal_boards
        self._completion_unfinished_boards += unfinished_boards
        return boards

    def _finish_completion_greedily(self, board: dict[str, Any]) -> dict[str, Any]:
        """Play the rest of the turn out greedily from wherever a SAMPLED
        completion stopped, so every completion this method returns is an
        end-of-turn board.

        Reached only under `completion_tail == "finish_greedy"`. The default
        is `"resample"`, which never gets here because its walk does not stop
        early in the first place; `"stop"` is the pre-2026-08-21 behaviour and
        does not get here either.

        WHY THIS EXISTS (2026-08-20). Without it the inner candidates are not
        comparable, and the measurement is not close: on one real mid-game
        board, the greedy completion stopped on `end_turn_chosen` with 0 gold
        left, while **0 of 40** sampled completions finished the turn -- 75%
        stopped at the next structurally random op, 25% on a repeated board,
        and 72% left gold unspent (mean 2.20). `_completion_agg` then took
        `max` over one finished board and up to `completion_width - 1`
        unfinished ones, and the V head is trained on end-of-turn boards, so
        the unfinished ones are off-distribution exactly where a max is most
        willing to believe them.

        `_sample_walk` stops early for two different reasons and BOTH are
        wrong HERE while being right where they came from:

        - `structural_boundary` is justified in the docstring by "everything
          past it is provably discarded", which is true of an outer CANDIDATE
          (the search truncates it at that op anyway) and false of a
          completion, whose whole purpose is to reach a scorable end board.
        - `sampled_cycle` stops because a single random draw has no
          well-defined "next best". The greedy scan does have one, so handing
          the tail to it is also what resolves that case.

        Deterministic given `board`, and `board` is a pure function of
        `(walk, sample_r, j)`, so CRN and reproducibility are unchanged.
        Index 0 does not come through here: it already runs to END_TURN.
        """
        finish = self._call_bc_recommend(
            board, deterministic=True, force_ranked_decode=True
        )
        return _apply_chain(board, finish.get("chain_preview") or [])

    def _sample_completion(
        self, after: dict[str, Any], sample_r: int, index: int
    ) -> dict[str, Any]:
        """One alternative completion of the turn from the resampled board.

        Same two-path shape as `_generate_candidate` for the same reason (a
        duck-typed stand-in without `.model`/`.encoder` still has to work),
        but the RNG comes from `_rng_for_completion`, which is keyed inside
        the imagination.
        """
        if hasattr(self.bc, "model") and hasattr(self.bc, "encoder"):
            try:
                return self._sample_candidate(
                    after,
                    self._rng_for_completion(sample_r, index),
                    completion_mode=(self.completion_tail == COMPLETION_TAIL_RESAMPLE),
                    for_completion=True,
                )
            except Exception:
                pass
        return self._call_bc_recommend(after, deterministic=False)

    def _resample_after(self, walk: dict[str, Any], sample_r: int) -> dict[str, Any]:
        """The board the ONE stochastic op lands on for imagined sample `r`.

        Everything earlier in the prefix was deterministic, so replaying it
        would reach the identical board; only the last op is re-applied, on a
        clone reseeded to `S(..., sample_r)`.
        """
        pre = walk["pre_board"] if isinstance(walk["pre_board"], dict) else walk["board"]
        work = honest_frame.imagined_clone(pre, seed=self._imagination_seed(sample_r))
        trans = engine_step(work, walk["ops"][-1])
        if not trans.get("legal"):
            raise RuntimeError("honest_resample_illegal")
        after = trans.get("state_after")
        if not isinstance(after, dict):
            raise RuntimeError("honest_resample_no_state_after")
        return after

    def _completion_agg(self, values: list[float]) -> float:
        """This recommender's configured inner aggregation.

        A thin bind of the module-level `completion_agg` to
        `self.completion_aggregate`. The arithmetic lives at module level
        because the Bellman labeller has to apply the SAME aggregation to the
        SAME boards; see `completion_agg`.
        """
        return completion_agg(values, self.completion_aggregate)

    @staticmethod
    def _count_completion_decisions(
        leaf_scores: list[float],
        sub_spans: list[list[tuple[int, int, int]]],
    ) -> tuple[int, int]:
        """`(decided, divergent)` over the imagined samples that had a CHOICE.

        Did the inner search actually change the pick, or did it re-derive the
        greedy completion every time. An arm that names an intervention has to
        carry a measurement that the intervention happened; this is it, and the
        gate on it lives upstream in the runner.

        A sample with fewer than two boards had nothing to decide and is
        counted in NEITHER total, so `bc_greedy` reports 0 of 0 rather than a
        vacuous 100% agreement -- a ratio computed off this pair therefore has
        no denominator to divide by rather than a misleading one.
        """
        decided = 0
        divergent = 0
        for subs in sub_spans:
            for (a, b, _r) in subs:
                if b - a < 2:
                    continue
                decided += 1
                window = leaf_scores[a:b]
                if max(range(len(window)), key=lambda i: window[i]) != 0:
                    divergent += 1
        return decided, divergent

    def _imagined_completion(self, walk: dict[str, Any], sample_r: int) -> dict[str, Any]:
        """The greedy imagined completion of one stochastic sample.

        Unchanged in name, signature and meaning: `w1_scoring_layer_probe`'s
        reconstruction gate and `test_exp13_honest_frame` both call it and
        both mean exactly this. Under `bc_greedy` it is also the only board a
        sample contributes, so W3 does not move those callers.
        """
        after = self._resample_after(walk, sample_r)
        completion = self._call_bc_recommend(after, deterministic=True, force_ranked_decode=True)
        return _apply_chain(after, completion.get("chain_preview") or [])

    def _honest_myopic_scores(
        self, boards: list[dict[str, Any]], opponent_team: list[dict[str, Any]]
    ) -> list[float | None]:
        """Stage-1 myopic scores for the honest leaf, or all-None at blend 0.

        Same skip rule as the determinized vgame path: at `blend == 0`
        nothing reads them, so the oracle is not called at all. At a non-zero
        blend every SCORED board needs one, which under this frame means one
        per imagined completion -- k times the determinized cost. That is the
        real price of blending under expectation scoring and it is paid
        honestly rather than approximated.
        """
        if not self._vgame_needs_myopic():
            return [None] * len(boards)
        scores: list[float | None] = []
        for board in boards:
            try:
                scores.append(self._score_end_board(board, opponent_team))
            except Exception:
                scores.append(None)
        return scores

    def _search_vgame_honest(
        self,
        *,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        candidates: list[dict[str, Any]],
        opponent_team: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """exp13 A1 ruling 3: prefix dedup + expectation scoring on the V leaf.

        1. Every candidate chain is reduced to its deterministic prefix
           (`_prefix_walk`) and the prefixes are deduped (`_prefix_group_key`)
           -- the committable unit is what gets ranked.
        2. Each group is turned into the boards V will score: ONE end board
           for a deterministic-to-END_TURN prefix (identical to the
           determinized path), or k imagined completions for a prefix that
           ends at a chance node (`_imagined_completion`).
        3. All of those boards go through ONE batched V forward, and a
           group's score is the MEAN over its own boards. Argmax wins, ties
           break by mean myopic when a blend made one, else by first-seen
           order, i.e. the greedy chain.

        NEVER-RAISE, same contract as `_search_vgame`: any failure degrades
        this turn to the greedy chain with `search_used=False` plus a
        `search_error`, so a broken leaf costs strength and is counted, never
        a crashed game. That degradation is also the FAIL-CLOSED route for a
        prefix whose completions all failed: it is dropped rather than scored
        on the proposal stream's own board -- see the drop site in step 2.
        """
        # `_reset_completion_counters` documents itself as "called at the top
        # of both search entry points" and was only ever wired into
        # `_search_anytime`. THIS is the path the arena arms take, so every
        # per-segment completion instrument they reported was a running total
        # for the life of the recommender. Measured on the shipped width-4 run
        # `w4/n1000-5cd4688`, game 0: samples climb 99 -> 159 -> 204 -> ...
        # across segments and never fall, and two consecutive segments that did
        # no completion work at all (turn 5 segments 4 and 5) both report 627.
        # Ratios survive that encoding -- which is why
        # `completion_boards_mean` always read exactly the configured width and
        # nothing ever said so -- but totals do not, and "did THIS segment do
        # the work" becomes unanswerable, which is the question the completion
        # telemetry gate exists to answer.
        self._reset_completion_counters()
        try:
            groups: dict[tuple[str, ...], dict[str, Any]] = {}
            order: list[tuple[str, ...]] = []
            for i, cand in enumerate(candidates):
                walk = self._prefix_walk(state, cand.get("chain_preview") or [])
                key = self._prefix_group_key(walk)
                if key in groups:
                    continue
                groups[key] = {"first_index": i, "walk": walk}
                order.append(key)

            boards: list[dict[str, Any]] = []
            spans: list[tuple[int, int]] = []
            # exp22 W3: one entry per SURVIVING imagined sample, each a
            # (start, stop) into `boards`. Under `bc_greedy` every entry has
            # length 1 and the aggregator below degenerates to identity, so
            # the group score is the same flat mean it has always been.
            sub_spans: list[list[tuple[int, int, int]]] = []
            kept: list[tuple[str, ...]] = []
            dropped: list[tuple[str, ...]] = []
            for key in order:
                walk = groups[key]["walk"]
                start = len(boards)
                subs: list[tuple[int, int, int]] = []
                if walk["boundary"] is None:
                    boards.append(walk["board"])
                    subs.append((start, len(boards), 0))
                else:
                    for sample_r in range(1, self.stochastic_samples + 1):
                        sub_start = len(boards)
                        try:
                            got = self._imagined_completions(walk, sample_r)
                            if not got:
                                raise RuntimeError("honest_resample_no_completion")
                            boards.extend(got)
                            subs.append((sub_start, len(boards), int(sample_r)))
                        except Exception:
                            # Defensive only: the resampled op's legality is a
                            # function of the PRE-op board, which is identical
                            # across samples, so this cannot fire for any
                            # engine randomness that exists today. Losing one
                            # sample must not lose the candidate -- the mean
                            # is simply taken over the samples that survived.
                            continue
                    if len(boards) == start:
                        # ...but losing them ALL must not fall back to
                        # `walk["board"]` (exp13 W0b', codex finding 3). That
                        # board is the PROPOSAL stream's outcome: the one
                        # roll the candidate generator happened to draw. Using
                        # it here would score this prefix on a lucky/unlucky
                        # single draw while its siblings are scored on a k-mean
                        # -- reintroducing exactly the pick-the-lucky-roll bias
                        # A1 removes, silently and only under failure. Fail
                        # CLOSED instead: drop the prefix, and if the dropped
                        # one is the greedy chain's (group 0, the tie-break
                        # anchor) or if nothing survives, degrade the whole
                        # decision to greedy via the outer handler.
                        dropped.append(key)
                        continue
                kept.append(key)
                spans.append((start, len(boards)))
                sub_spans.append(subs)

            if not kept or kept[0] != order[0]:
                raise RuntimeError(
                    f"honest_prefix_completions_failed:dropped={len(dropped)}/{len(order)}"
                )
            order = kept

            myopic_scores = self._honest_myopic_scores(boards, opponent_team)
            race = self._race_scalars(state)
            scored = self.vgame_scorer.score_boards(
                boards, myopic_scores=myopic_scores, **race
            )
            leaf_scores = [float(x) for x in scored["score"]]
            if len(leaf_scores) != len(boards):
                raise RuntimeError(
                    f"vgame_scorer_length_mismatch:{len(leaf_scores)}!={len(boards)}"
                )
            v_scores = [float(x) for x in scored["v"]]
            ensemble_std = [float(x) for x in scored["ensemble_std"]]
        except Exception as exc:
            result = _annotate_skip(greedy_result)
            result["search_scoring"] = SCORING_VGAME
            result["search_turn_mode"] = self.turn_mode
            result["search_error"] = f"honest_vgame_scoring_failed:{exc}"
            return result

        def _mean(values: list[float]) -> float:
            return sum(values) / float(len(values)) if values else 0.0

        _agg = self._completion_agg

        decided_here, divergent_here = self._count_completion_decisions(
            leaf_scores, sub_spans
        )
        self._completion_decided += decided_here
        self._completion_divergent += divergent_here

        group_scores = [
            _mean([_agg(leaf_scores[a:b]) for (a, b, _r) in subs]) for subs in sub_spans
        ]
        group_myopic: list[float | None] = []
        for (a, b) in spans:
            present = [float(m) for m in myopic_scores[a:b] if m is not None]
            group_myopic.append(_mean(present) if present else None)

        # ------------------------------------------------------ MC rerank
        # Ledger A2.6: V ranks the whole menu, then the top `mc_rerank_k`
        # groups -- IN V ORDER, which is the half `DECISIONS.md:316` found the
        # existing rollout operator getting wrong by shortlisting on a myopic
        # score -- are re-ranked by rollouts to terminal.
        #
        # THE ONE REAL CHOICE HERE, stated where it is made. Under the honest
        # frame a group whose prefix stops at a chance node has NO single
        # committed board: it exists only as its `k` imagined completions, and
        # at decision time the real roll has not happened. So a rollout for
        # such a group must start from imagined boards, and the question is how
        # to spend a budget of `m` across them. This spreads `m` round-robin
        # over the group's own boards, so the cost is exactly `k * m` per
        # decision -- the same grid the offline pricing used -- and the
        # estimate covers the group's board distribution rather than favouring
        # one completion. The alternative, `m` rollouts on EVERY board, is
        # unbiased per board but costs `boards * m` and would not be the priced
        # operator. Deterministic groups have one board and are unaffected.
        rerank_block: dict[str, Any] | None = None
        if self.mc_rerank_k > 0 and group_scores:
            v_ranked = sorted(
                range(len(order)),
                key=lambda i: (-float(group_scores[i]), int(groups[order[i]]["first_index"])),
            )
            shortlist = v_ranked[: int(self.mc_rerank_k)]
            budget = max(1, int(self.rollout_repeats))
            rerank_scores: dict[int, float] = {}
            rerank_rows: list[dict[str, Any]] = []
            for slot, group_i in enumerate(shortlist):
                start, stop = spans[group_i]
                group_boards = list(range(start, stop))
                if not group_boards:
                    continue
                # Round-robin split of the budget across this group's boards.
                per_board = [budget // len(group_boards)] * len(group_boards)
                for extra in range(budget % len(group_boards)):
                    per_board[extra] += 1
                totals: list[float] = []
                spent = 0
                for board_i, reps in zip(group_boards, per_board):
                    if reps <= 0:
                        continue
                    roll = self._rollout_score_candidate(
                        boards[board_i], candidate_index=group_i, repeats=reps
                    )
                    totals.append(float(roll["score"]) * reps)
                    spent += reps
                if not spent:
                    continue
                mc_score = sum(totals) / float(spent)
                rerank_scores[group_i] = mc_score
                rerank_rows.append({
                    "v_rank": slot,
                    "group_index": int(group_i),
                    "v_score": float(group_scores[group_i]),
                    "mc_score": float(mc_score),
                    "rollouts_spent": int(spent),
                    "n_boards": len(group_boards),
                    "stochastic": groups[order[group_i]]["walk"]["boundary"] is not None,
                })
            if rerank_scores:
                # The shortlist is re-ordered among ITSELF; everything below it
                # keeps the order V gave it. A group outside the shortlist can
                # never overtake one inside it, which is what makes this a
                # rerank of V's top-k rather than a second full ranking.
                reordered = sorted(
                    rerank_scores,
                    key=lambda i: (-rerank_scores[i], int(groups[order[i]]["first_index"])),
                )
                new_order_index = {g: p for p, g in enumerate(reordered)}
                bumped = max(group_scores) + 1.0
                for position, group_i in enumerate(reordered):
                    # Rewrite the score so the single argmax downstream picks
                    # the rerank winner, preserving V's ordering below.
                    group_scores[group_i] = bumped + float(len(reordered) - position)
                rerank_block = {
                    "mc_rerank_k": int(self.mc_rerank_k),
                    "mc_rerank_m": int(budget),
                    "shortlist_taken_by": "v_score",
                    "scored_on": (
                        "trophies" if self.game_rules == SEARCH_GAME_RULES_ARENA
                        else "versus_outcome"
                    ),
                    "game_rules": self.game_rules,
                    "shortlist_group_indices": [int(g) for g in shortlist],
                    "v_order_before": [int(g) for g in v_ranked[: int(self.mc_rerank_k)]],
                    "mc_order_after": [int(g) for g in reordered],
                    "changed_the_pick": bool(
                        reordered and v_ranked and reordered[0] != v_ranked[0]
                    ),
                    "rows": rerank_rows,
                    "total_rollouts": int(sum(r["rollouts_spent"] for r in rerank_rows)),
                }

        candidate_groups: list[dict[str, Any]] | None = None
        if self.capture_candidate_chains:
            ranked = sorted(
                range(len(order)),
                key=lambda i: (-float(group_scores[i]), int(groups[order[i]]["first_index"])),
            )
            rank_by_group = {group_i: rank for rank, group_i in enumerate(ranked)}
            candidate_groups = []
            for group_i, key in enumerate(order):
                walk = groups[key]["walk"]
                start, stop = spans[group_i]
                # The sample a board belongs to comes from `sub_spans`, not
                # from its position: under `v_search` one imagined sample
                # contributes several boards, so inferring `sample_r` from the
                # offset silently relabels every board after the first.
                completions: list[dict[str, Any]] = []
                for (a, b, sample_r) in sub_spans[group_i]:
                    for inner_index, board_i in enumerate(range(a, b)):
                        completions.append(
                            {
                                "completion_r": int(sample_r),
                                "completion_inner_index": int(inner_index),
                                "imagination_seed": (
                                    self._imagination_seed(sample_r) if sample_r > 0 else None
                                ),
                                "candidate_imagined_afterstate": copy.deepcopy(boards[board_i]),
                                "leaf_score": float(leaf_scores[board_i]),
                                "v0_score": float(v_scores[board_i]),
                                "ensemble_std": float(ensemble_std[board_i]),
                            }
                        )
                candidate_groups.append(
                    {
                        "group_index": int(group_i),
                        "raw_candidate_index": int(groups[key]["first_index"]),
                        "dedup_prefix_key": list(key),
                        "committed_ops": self._prefix_chain(walk),
                        "boundary_reason": walk["boundary"],
                        "prefix_stop": walk["stop"],
                        "completion_count": len(completions),
                        "v0_group_score": float(group_scores[group_i]),
                        "v0_rank": int(rank_by_group[group_i]),
                        "completions": completions,
                    }
                )

        # `max` returns the FIRST maximal element, and candidate 0 (the
        # greedy chain) is always group 0, so a full tie keeps plain BC --
        # the same "search can never do worse than greedy" property the
        # determinized path has.
        best_i = max(
            range(len(order)),
            key=lambda i: (group_scores[i], group_myopic[i] if group_myopic[i] is not None else 0.0),
        )
        best_key = order[best_i]
        best_walk = groups[best_key]["walk"]
        winner = candidates[groups[best_key]["first_index"]]
        chosen_chain = self._prefix_chain(best_walk)

        candidate_chains: list[list[dict[str, Any]]] | None = None
        decoded_candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains:
            candidate_chains = [self._prefix_chain(groups[key]["walk"]) for key in order]
            # A2.4's required telemetry: what the DECODER produced, before
            # `_prefix_walk` truncated it. `candidate_chains` above is one
            # entry per surviving GROUP and is already truncated, so it cannot
            # measure the decode saving; this is one entry per generated
            # candidate, so per-candidate chain lengths pin the size of the
            # saving instead of it being asserted.
            decoded_candidate_chains = [
                copy.deepcopy(cand.get("chain_preview") or []) for cand in candidates
            ]

        result = dict(winner)
        result.update(
            {
                **_searched_verdict(chosen_chain),
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_turn_mode": self.turn_mode,
                "search_n_candidates": len(order),
                "search_n_generated": int(self.n_candidates),
                # Under this frame "dedup" counts DISTINCT COMMITTABLE
                # PREFIXES, which is the width that is actually being bought
                # once the turn is segmented -- not distinct whole-turn end
                # boards, a unit this frame never commits.
                "search_n_dedup": len(order),
                # exp22 W3. The policy is on the row so a reader never has to
                # infer which arm produced it, and `decided`/`divergent` are
                # the measurement that the intervention happened at all: a
                # v_search arm whose inner search re-derives the greedy
                # completion every time did not deepen anything.
                "search_completion_policy": self.completion_policy,
                "search_completion_width": int(self.completion_width),
                "search_completion_aggregate": self.completion_aggregate,
                "search_completion_decided": int(decided_here),
                "search_completion_divergent": int(divergent_here),
                **self._completion_instruments(),
                "search_scores": group_scores,
                "search_chosen_index": best_i,
                "search_candidate_groups": candidate_groups,
                # Parallel to the determinized vgame path: this is the GREEDY
                # candidate's stage-1 myopic score, None at blend 0 where no
                # oracle call is made. Its leaf score is in the diagnostics.
                "search_greedy_score": group_myopic[0] if group_myopic else None,
                "search_candidate_chains": candidate_chains,
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "turn_mode": self.turn_mode,
                    "stochastic_samples": int(self.stochastic_samples),
                    # None when the rerank is off, which is every run before
                    # exp22. When on, it carries what the arm has to PROVE:
                    # the realised k and m, that the shortlist was taken in V
                    # order, what it was scored on, and the order before and
                    # after. A configured knob is not evidence the stage ran.
                    "mc_rerank": rerank_block,
                    "completion_boards_mean": (
                        float(self._completion_boards) / float(self._completion_samples)
                        if self._completion_samples
                        else None
                    ),
                    "completion_dropped": int(self._completion_dropped),
                    "imagination_key_source": self._imagination_key_source(),
                    "imagination_segment_index": int(self._imagination_segment_index),
                    "imagination_sample_seeds": [
                        self._imagination_seed(r)
                        for r in range(1, self.stochastic_samples + 1)
                    ],
                    "prefix_boundaries": [groups[key]["walk"]["boundary"] for key in order],
                    "prefix_stops": [groups[key]["walk"]["stop"] for key in order],
                    "prefix_lengths": [len(groups[key]["walk"]["ops"]) for key in order],
                    # SAMPLES, not boards. Under `v_search` a sample carries
                    # `completion_width` boards, so `b - a` stopped being the
                    # sample count the moment W3 landed.
                    "prefix_n_samples": [len(subs) for subs in sub_spans],
                    "prefix_n_boards": [b - a for (a, b) in spans],
                    # exp13 W0b' finding 3: prefixes whose k completions ALL
                    # failed and were therefore dropped instead of scored on
                    # the proposal stream. Always 0 on the engine as it stands;
                    # emitted so a non-zero can never be invisible.
                    "prefix_n_dropped": len(dropped),
                    "prefix_sample_scores": [leaf_scores[a:b] for (a, b) in spans],
                    "decoded_candidate_chains": decoded_candidate_chains,
                    "group_scores": group_scores,
                    "greedy_leaf_score": (group_scores[0] if group_scores else None),
                    "v_scores": v_scores,
                    "leaf_scores": leaf_scores,
                    "ensemble_std": ensemble_std,
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in myopic_scores
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                },
            }
        )
        return result

    def _search_vgame(
        self,
        *,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        candidates: list[dict[str, Any]],
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        greedy_score: float | None,
        candidate_chains: list[list[dict[str, Any]]] | None,
        n_dedup: int,
    ) -> dict[str, Any]:
        """exp12 W2's learned leaf: ONE batched V forward over the deduped
        stage-1 end boards, then argmax (module docstring's "exp12 W2").

        Stage 1 is shared with every other mode, so the candidate set this
        scores is bit-identical to the set a rollout run at the same width
        would have scored. The score itself is
        `(1-blend)*V + blend*myopic - pessimism*ensemble_std`, computed by
        `tools/vgame_scorer.py` (which owns the encoder, the bypass block and
        the checkpoint pins).

        NEVER-RAISE: any scorer failure -- a bad board, an encoder error, an
        unset `wins` -- degrades to the greedy candidate with the search
        marked unused, exactly as a turn-1 skip does, plus `search_error` so
        the failure is visible in the report instead of silent. A game never
        dies because the leaf did.
        """
        boards = [groups[sig]["end_board"] for sig in scored_sigs]
        myopic_scores = [groups[sig]["score"] for sig in scored_sigs]
        try:
            race = self._race_scalars(state)
            scored = self.vgame_scorer.score_boards(boards, myopic_scores=myopic_scores, **race)
            values = [float(x) for x in scored["score"]]
            if len(values) != len(scored_sigs):
                raise RuntimeError(
                    f"vgame_scorer_length_mismatch:{len(values)}!={len(scored_sigs)}")
        except Exception as exc:
            result = _annotate_skip(greedy_result)
            result["search_scoring"] = SCORING_VGAME
            # Deliberately distinct from a turn-1 skip: that one is a design
            # outcome, this one is a degraded turn and must be countable.
            result["search_error"] = f"vgame_scoring_failed:{exc}"
            return result

        # Argmax on the leaf score; an EXACT tie breaks by the stage-1 myopic
        # score when one exists (the rollout path's own convention), else by
        # first-seen order, which is candidate 0 = the greedy chain.
        best_i = max(
            range(len(values)),
            key=lambda i: (values[i], (myopic_scores[i] if myopic_scores[i] is not None else 0.0)),
        )
        best_sig = scored_sigs[best_i]
        winner = candidates[groups[best_sig]["first_index"]]

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_n_candidates": len(scored_sigs),
                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(n_dedup),
                "search_scores": values,
                "search_chosen_index": best_i,
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),
                "search_candidate_chains": candidate_chains,
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "v_scores": [float(x) for x in scored["v"]],
                    "leaf_scores": values,
                    "ensemble_std": [float(x) for x in scored["ensemble_std"]],
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in myopic_scores
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                },
            }
        )
        return result

    def _search_rollout(
        self,
        *,
        candidates: list[dict[str, Any]],
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        greedy_score: float | None,
        candidate_chains: list[list[dict[str, Any]]] | None = None,
        teacher_chains: list[list[dict[str, Any]]] | None = None,
        n_dedup: int | None = None,
    ) -> dict[str, Any]:
        """Stage 2 of `scoring="rollout"` (module docstring's ROLLOUT
        section): shortlist the top `self.rollout_shortlist` deduped
        candidates by their stage-1 myopic score, simulate each one's
        rest-of-game `self.rollout_repeats` times
        (`_rollout_score_candidate`), and pick the argmax rollout score --
        an EXACT tie breaks by the myopic score (both wrapped in one
        descending-sorted tuple key, so a plain `max()` does the right
        thing). `candidates`/`groups`/`scored_sigs`/`greedy_score` are the
        SAME stage-1 outputs the myopic path uses, passed in rather than
        recomputed.
        """
        myopic_ranked = sorted(scored_sigs, key=lambda sig: groups[sig]["score"], reverse=True)
        shortlist_sigs = myopic_ranked[: max(1, int(self.rollout_shortlist))]

        rollout_by_sig: dict[str, dict[str, Any]] = {}
        for sig in shortlist_sigs:
            candidate_index = groups[sig]["first_index"]
            rollout_by_sig[sig] = self._rollout_score_candidate(
                groups[sig]["end_board"], candidate_index=candidate_index
            )

        best_myopic_sig = max(scored_sigs, key=lambda sig: groups[sig]["score"])
        best_sig = max(shortlist_sigs, key=lambda sig: (rollout_by_sig[sig]["score"], groups[sig]["score"]))
        winner = candidates[groups[best_sig]["first_index"]]

        teacher_record = (
            self._build_teacher_record(
                groups=groups,
                scored_sigs=scored_sigs,
                shortlist_sigs=shortlist_sigs,
                rollout_by_sig=rollout_by_sig,
                best_sig=best_sig,
                best_myopic_sig=best_myopic_sig,
                teacher_chains=teacher_chains,
                n_dedup=n_dedup,
            )
            if self.capture_teacher_record
            else None
        )

        result = dict(winner)
        result.update(
            {
                "search_used": True,
                "search_scoring": SCORING_ROLLOUT,
                "search_n_candidates": len(scored_sigs),
                # exp12 W2c (width curve): same meaning as in the myopic
                # branch -- requested width vs distinct end boards.
                "search_n_generated": int(self.n_candidates),
                "search_n_dedup": int(len(scored_sigs) if n_dedup is None else n_dedup),
                "search_scores": [float(groups[sig]["score"]) for sig in scored_sigs],
                # Kept for continuity with the myopic schema: which deduped
                # candidate MYOPIC scoring alone would have picked (may
                # differ from the rollout-chosen one -- that divergence IS
                # the W1 measurement).
                "search_chosen_index": scored_sigs.index(best_myopic_sig),
                "search_greedy_score": (float(greedy_score) if greedy_score is not None else None),
                "search_diagnostics": {
                    "scoring": SCORING_ROLLOUT,
                    "myopic_scores": [float(groups[sig]["score"]) for sig in scored_sigs],
                    "myopic_chosen_index": scored_sigs.index(best_myopic_sig),
                    "shortlist_indices": [scored_sigs.index(sig) for sig in shortlist_sigs],
                    "rollout_scores": [rollout_by_sig[sig] for sig in shortlist_sigs],
                    "rollout_chosen_index": shortlist_sigs.index(best_sig),
                    "rollout_shortlist": int(self.rollout_shortlist),
                    "rollout_repeats": int(self.rollout_repeats),
                    "rollout_ksim": int(self.rollout_ksim),
                    # exp10 W2: which opponent-source mode produced these
                    # rollout scores -- see module docstring's "exp10 W2"
                    # section. Always "true" unless a caller opted into the
                    # ablation via the constructor kwarg of the same name.
                    "rollout_opponent_mode": self.rollout_opponent_mode,
                    # exp09 W2: None unless `capture_candidate_chains=True`;
                    # aligned index-for-index with `myopic_scores` above (the
                    # full stage-1 deduped candidate set, NOT just the
                    # shortlist) -- see module docstring.
                    "candidate_chains": candidate_chains,
                },
                # exp12 route a (wave A0): None unless
                # `capture_teacher_record=True`. See `_build_teacher_record`.
                "teacher_record": teacher_record,
            }
        )
        return result

    def _build_teacher_record(
        self,
        *,
        groups: dict[str, dict[str, Any]],
        scored_sigs: list[str],
        shortlist_sigs: list[str],
        rollout_by_sig: dict[str, dict[str, Any]],
        best_sig: str,
        best_myopic_sig: str,
        teacher_chains: list[list[dict[str, Any]]] | None,
        n_dedup: int | None,
    ) -> dict[str, Any]:
        """exp12 route a (wave A0): the per-decision distillation record.

        One entry per SCORED stage-1 candidate (index-aligned with
        `search_scores`/`myopic_scores`), each carrying the exact end board
        that was handed to scoring, the candidate's chain, its myopic score
        and -- for the shortlisted ones the teacher actually rolled out --
        its rollout score, per-repeat outcomes and the CRN keys those
        repeats used. `chosen_index` is the ROLLOUT winner, i.e. the
        candidate the driver is about to play (NOT `search_chosen_index`,
        which stays the myopic argmax for schema continuity with W6a).

        `chosen_teacher_score` is read off the DECISION path
        (`rollout_by_sig[best_sig]`) while the per-candidate scores are
        built in the loop below, so the checker tool's "the chosen
        candidate's recorded score is the score the driver acted on" gate
        compares two independently indexed reads and can actually catch an
        index misalignment.
        """
        candidates_out: list[dict[str, Any]] = []
        for i, sig in enumerate(scored_sigs):
            roll = rollout_by_sig.get(sig)
            chain = None
            if teacher_chains is not None and i < len(teacher_chains):
                chain = copy.deepcopy(teacher_chains[i])
            candidates_out.append(
                {
                    "index": i,
                    "signature": sig,
                    "myopic_score": float(groups[sig]["score"]),
                    "rolled_out": roll is not None,
                    "teacher_score": (float(roll["score"]) if roll is not None else None),
                    "mean_outcome": (float(roll["mean_outcome"]) if roll is not None else None),
                    "mean_lives_diff": (float(roll["mean_lives_diff"]) if roll is not None else None),
                    "n_repeats": (int(roll["n_repeats"]) if roll is not None else 0),
                    "repeat_outcomes": (list(roll["repeat_outcomes"]) if roll is not None else []),
                    "repeat_lives_diffs": (list(roll.get("repeat_lives_diffs") or []) if roll is not None else []),
                    "repeat_crn_keys": (list(roll.get("repeat_crn_keys") or []) if roll is not None else []),
                    "repeat_engine_seeds": (list(roll.get("repeat_engine_seeds") or []) if roll is not None else []),
                    "repeat_chosen_opponent_pids": (
                        list(roll.get("repeat_chosen_opponent_pids") or []) if roll is not None else []
                    ),
                    "chosen": sig == best_sig,
                    "chain": chain,
                    # The EXACT object scoring was handed -- `_apply_chain`'s
                    # output, pre-`resolve_end_turn_pre_battle`, which is the
                    # board both training (the W1'a afterstate contract) and
                    # serving (search's own hand) are on.
                    "state": copy.deepcopy(groups[sig]["end_board"]),
                }
            )
        return {
            "schema": TEACHER_RECORD_SCHEMA,
            "n_generated": int(self.n_candidates),
            "n_dedup": int(len(scored_sigs) if n_dedup is None else n_dedup),
            "n_candidates": len(scored_sigs),
            "shortlist_size": len(shortlist_sigs),
            "rollout_repeats": int(self.rollout_repeats),
            "rollout_ksim": int(self.rollout_ksim),
            "rollout_opponent_mode": self.rollout_opponent_mode,
            "chosen_index": scored_sigs.index(best_sig),
            "chosen_signature": best_sig,
            "chosen_teacher_score": float(rollout_by_sig[best_sig]["score"]),
            "myopic_chosen_index": scored_sigs.index(best_myopic_sig),
            "candidates": candidates_out,
            "crn": {
                "enabled": bool(self.rollout_crn),
                "base_seed": int(self.seed),
                "game_index": self._crn_game_index,
                "turn": self._decision_turn,
                "decision_id": self._crn_decision_id(),
            },
        }

    def _choose_rollout_opponent_pid(
        self, *, true_pid: str | None, current_turn: int, rng: random.Random
    ) -> dict[str, Any]:
        """exp10 W2 (opponent-source ablation): pick the pid whose recorded
        chain a rollout continuation should follow INSTEAD of `true_pid`,
        for `self.rollout_opponent_mode in {"pool_random", "retrieval"}` --
        see module docstring's "exp10 W2" section. Never called when
        `self.rollout_opponent_mode == ROLLOUT_OPPONENT_TRUE` (the caller,
        `_rollout_score_candidate`, skips this entirely for that mode).

        `"pool_random"`: uniform draw over `self.opp_source.all_pids` --
        already exactly the eval frame's own opponent pool (e.g.
        val/Turtle/rank<=1500, whatever `opp_source` was constructed with)
        -- EXCLUDING `true_pid`, so the draw can never silently coincide
        with the privileged `"true"` arm (the correctness proof this
        module's caller requires: the arms must actually use different
        opponents).

        `"retrieval"`: same pool, additionally restricted to pids whose
        indexed chain length (`len(self.opp_source.by_pid[pid])`, the same
        "chain length" measure `ChainSnapshotSource.long_min` already uses)
        is >= `current_turn` -- a "this candidate opponent's own game had
        actually reached this turn" plausibility filter. rank/pack are NOT
        re-checked here: they are pool-wide invariants already enforced by
        how `self.opp_source` itself was constructed (every pid in
        `all_pids` already satisfies them), so re-checking per draw would
        be redundant, not additionally correct.

        Degrades gracefully rather than raising: if the length filter
        leaves zero candidates, falls back to the unrestricted
        (`"pool_random"`-style) population; if THAT is also empty (a
        single-pid pool with `true_pid` excluded), gives up and returns
        `pid=None` -- the caller then leaves `board_copy` untouched for
        just that one repeat (equivalent to `"true"` for that draw only,
        never a hard failure).

        Returns `{"pid": str | None, "fallback_tier": str}`.
        """
        all_pids = self.opp_source.all_pids
        pool = [p for p in all_pids if p != true_pid] or list(all_pids)
        if not pool:
            return {"pid": None, "fallback_tier": "empty_pool"}

        if self.rollout_opponent_mode == ROLLOUT_OPPONENT_RETRIEVAL:
            eligible = [p for p in pool if len(self.opp_source.by_pid.get(p, {})) >= int(current_turn)]
            if eligible:
                return {"pid": str(rng.choice(eligible)), "fallback_tier": "retrieval_primary"}
            # Too few (or zero) same-tier-and-long-enough candidates at this
            # turn -- relax the length constraint rather than fail the
            # repeat outright (rank/pack still hold, see docstring).
            return {"pid": str(rng.choice(pool)), "fallback_tier": "retrieval_relaxed_length"}

        return {"pid": str(rng.choice(pool)), "fallback_tier": "pool_random"}

    def _rollout_score_candidate(
        self, end_board: dict[str, Any], *, candidate_index: int,
        repeats: int | None = None,
    ) -> dict[str, Any]:
        """Simulate the REST OF THE GAME `self.rollout_repeats` times for one
        shortlisted candidate's `end_board` (this turn's shop-phase actions
        already applied, END_TURN not yet resolved). See module docstring's
        ROLLOUT section for the full algorithm; this is stage 2's per-
        candidate inner loop.

        Each repeat runs on a FRESH `copy.deepcopy(end_board)` and an
        ISOLATED fallback sampler (`random.Random`, never
        `self.opp_source`'s own shared `_random_rng` -- see
        `chain_snapshot.py::ChainSnapshotSource.sample_random_with_rng`).
        This turn's own battle uses `simulation_count=self.rollout_ksim`;
        every subsequent turn (`eval_versus_fullgame.py::play_out_game`)
        reverts to that driver's default (1).

        exp10 W2 (opponent-source ablation, module docstring's "exp10 W2"
        section): when `self.rollout_opponent_mode != "true"`, ONE
        alternate pid is drawn per repeat (`_choose_rollout_opponent_pid`,
        an isolated RNG keyed on `(mode, seed, recommend_call_count,
        candidate_index, repeat_index)`) and spliced into
        `board_copy["meta"]["versus"]["current_opponent_participation_id"]`
        BEFORE this turn resolves -- the EXISTING sampling plumbing
        (`sample_for_pid_fn=self.opp_source.sample_for_pid`, completely
        unmodified/unwrapped) then follows THAT pid's recorded chain, with
        the SAME chain-then-random-fallback behavior the `"true"` mode
        already relies on. `"true"` mode (the default) never touches
        `board_copy`, so it is byte-identical to this method's pre-W2 body.

        Never raises: a repeat whose current-turn battle resolution itself
        fails (e.g. both the followed chain AND the isolated random
        fallback come up empty for this turn -- `end_turn_failed`) scores
        0.5 (treated as a no-result), the same bucket a genuine turn-cap
        gets, rather than aborting the whole candidate.

        Returns `{"score": float, "mean_outcome": float, "mean_lives_diff":
        float, "n_repeats": int, "repeat_outcomes": list[float],
        "rollout_opponent_mode": str, "true_followed_pid": str | None,
        "repeat_chosen_opponent_pids": list[str | None],
        "repeat_fallback_tiers": list[str],
        "repeat_opponent_pid_traces": list[list[{"turn": int,
        "opponent_pid_used": str | None}]]}` -- the last four keys are the
        exp10 W2 correctness-proof evidence (module docstring), present
        (empty-safe) for every mode so a consumer never needs a schema
        branch.
        """
        # Lazy import: avoids a module-level cycle (eval_versus_fullgame.py
        # imports THIS module at its own top level already) -- mirrors this
        # repo's own precedent for the identical reason
        # (`eval_versus_fullgame.py::_replay_order_helpers`'s lazy
        # cross-tools-module import).
        from . import eval_versus_fullgame as _evf

        true_pid = _read_current_opponent_pid(end_board)
        current_turn = int(end_board.get("turn", 1))

        outcomes: list[float] = []
        lives_diffs: list[float] = []
        # Arena's own estimand. Under versus this list stays empty and nothing
        # reads it, so the versus score below is unchanged.
        trophies_seen: list[float] = []
        is_arena = self.game_rules == SEARCH_GAME_RULES_ARENA
        repeat_chosen_opponent_pids: list[str | None] = []
        repeat_fallback_tiers: list[str] = []
        repeat_opponent_pid_traces: list[list[dict[str, Any]]] = []
        # exp12 route a (wave A0): the exact CRN key / engine seed each
        # repeat ran under, so a recorded decision proves on its own face
        # that its sibling candidates shared repeat r's future (the checker
        # tool's CRN gate compares these across a decision group) instead of
        # anyone having to trust the flag was on.
        repeat_crn_keys: list[str] = []
        repeat_engine_seeds: list[int | None] = []
        n_repeats = max(1, int(self.rollout_repeats if repeats is None else repeats))
        for repeat_index in range(n_repeats):
            fallback_key = self._crn_fallback_key(
                candidate_index=candidate_index, repeat_index=repeat_index
            )
            repeat_crn_keys.append(fallback_key)
            fallback_rng = random.Random(fallback_key)
            sample_random_fn = (
                lambda turn, _rng=fallback_rng: self.opp_source.sample_random_with_rng(turn, _rng)
            )

            board_copy = copy.deepcopy(end_board)

            # exp12 route a (wave A0): share the SHOP stream across siblings.
            # Without this each candidate's continuation inherits whatever
            # engine seed its own shop actions chained to, so candidate A and
            # candidate B see different turn-(t+1) shops -- pure comparison
            # noise. Only applied when the board already declares its seed
            # known (`_new_game_state` always does); never forced onto a
            # deliberately-unseeded state.
            engine_seed: int | None = None
            if self.rollout_crn:
                copy_meta = board_copy.setdefault("meta", {})
                if isinstance(copy_meta, dict) and copy_meta.get("seed_known"):
                    engine_seed = self._crn_engine_seed(repeat_index)
                    copy_meta["seed"] = int(engine_seed)
            repeat_engine_seeds.append(engine_seed)

            # exp10 W2: see this method's own docstring + module docstring's
            # "exp10 W2" section. No-op for the default "true" mode.
            chosen_pid: str | None = None
            fallback_tier = "true_mode_passthrough"
            if self.rollout_opponent_mode != ROLLOUT_OPPONENT_TRUE:
                opponent_rng = random.Random(
                    self._crn_opponent_key(
                        candidate_index=candidate_index, repeat_index=repeat_index
                    )
                )
                choice = self._choose_rollout_opponent_pid(
                    true_pid=true_pid, current_turn=current_turn, rng=opponent_rng
                )
                chosen_pid = choice["pid"]
                fallback_tier = choice["fallback_tier"]
                if chosen_pid:
                    versus_meta = board_copy.setdefault("meta", {}).setdefault("versus", {})
                    versus_meta["current_opponent_participation_id"] = chosen_pid
            repeat_chosen_opponent_pids.append(chosen_pid)
            repeat_fallback_tiers.append(fallback_tier)

            # exp10 W2: per-turn proof trace -- the pid `resolve_end_turn_
            # with_sampled_battle` actually used to sample the opponent
            # board this turn, read back from `state_after`'s own versus
            # meta (the single source of truth that field already is, see
            # `_read_current_opponent_pid`). Appended once per turn
            # regardless of mode.
            opponent_pid_trace: list[dict[str, Any]] = []

            def _record_pid_used(turn: Any, state_after: dict[str, Any] | None) -> None:
                pid_used = _read_current_opponent_pid(state_after) if isinstance(state_after, dict) else None
                opponent_pid_trace.append({"turn": int(turn) if turn is not None else None, "opponent_pid_used": pid_used})

            if is_arena:
                # What `play_out_game` does at the top of every arena turn:
                # arena has no engine life race, so the opponent life bar is a
                # pinned convention derived from the trophy count. Without it
                # `_versus_win(lives, opp_lives)` below reads a missing bar as
                # an instant win, which is exactly why the driver refuses an
                # arena rollout today.
                _evf._apply_arena_race_context(
                    board_copy, convention=self.arena_race_convention
                )
            this_turn = _evf._resolve_versus_turn(
                board_copy,
                sample_for_pid_fn=self.opp_source.sample_for_pid,
                sample_random_fn=sample_random_fn,
                parse_cache=None,
                simulation_count=self.rollout_ksim,
                game_mode=(_evf.GAME_MODE_ARENA if is_arena else _evf.GAME_MODE_VERSUS),
                max_lives=(_evf.ARENA_MAX_LIVES if is_arena else None),
            )
            if not this_turn["ok"]:
                outcomes.append(0.5)
                lives_diffs.append(0.0)
                repeat_opponent_pid_traces.append(opponent_pid_trace)
                continue

            state_after = this_turn["state_after"]
            _record_pid_used(current_turn, state_after)
            lives = int(state_after.get("lives", 0))
            opp_lives = int(_evf._versus_meta(state_after).get("opponent_lives", 0))
            if bool(TrainingEnv._is_done(state_after, None)):
                win = _evf._versus_win(lives, opp_lives)
                outcomes.append(1.0 if win else (0.0 if lives <= 0 else 0.5))
                lives_diffs.append(float(lives - opp_lives))
                if is_arena:
                    trophies_seen.append(float(state_after.get("trophies", 0) or 0))
                repeat_opponent_pid_traces.append(opponent_pid_trace)
                continue

            current_pid = str(_evf._versus_meta(state_after).get("current_opponent_participation_id") or "")
            remainder = _evf.play_out_game(
                state_after,
                self.bc,  # plain wrapped recommender -- NEVER self (no recursive search)
                initial_pid=current_pid,
                sample_for_pid_fn=self.opp_source.sample_for_pid,
                sample_random_fn=sample_random_fn,
                max_turn=self.max_turn,
                parse_cache=None,
                capture_detail=False,
                # The frame. `play_out_game` has taken all four of these all
                # along; this class simply never carried them, so the
                # continuation resolved under the versus default no matter
                # what frame the arm was actually being judged on.
                game_rules=self.game_rules,
                arena_race_convention=self.arena_race_convention,
                opponent_mode=(_evf.OPPONENT_MODE_ARENA if is_arena
                               else _evf.OPPONENT_MODE_CHAIN),
                turn_mode=self.turn_mode,
                # exp10 W2: reuses `play_out_game`'s existing W2(exp09)
                # `on_turn` hook (this repo's established per-turn tap point,
                # already zero-cost when unset) to extend the proof trace
                # across every subsequent turn of the continuation, not just
                # the candidate's own turn above.
                on_turn=lambda payload: _record_pid_used(payload.get("turn"), payload.get("state_after")),
            )
            win = bool(remainder["win"])
            loss = remainder["end_reason"] == _evf.END_REASON_PLAYER_LIVES_0
            outcomes.append(1.0 if win else (0.0 if loss else 0.5))
            if is_arena:
                trophies_seen.append(float(remainder.get("trophies", 0) or 0))
            lives_diffs.append(float(remainder["player_lives"] - remainder["opponent_lives"]))
            repeat_opponent_pid_traces.append(opponent_pid_trace)

        mean_outcome = sum(outcomes) / len(outcomes) if outcomes else 0.5
        mean_lives_diff = sum(lives_diffs) / len(lives_diffs) if lives_diffs else 0.0
        # ARENA IS SCORED ON TROPHIES, because that is what the arena
        # judgement's estimand is. Scoring an arena rollout on the versus
        # win/loss would rank candidates on a quantity no arena verdict reads,
        # and under arena a "win" means the 10-trophy completion, which almost
        # no game reaches -- so nearly every candidate would tie at 0.5.
        mean_trophies = (
            sum(trophies_seen) / len(trophies_seen) if trophies_seen else 0.0
        )
        score = (
            mean_trophies if is_arena
            else mean_outcome + 0.01 * mean_lives_diff
        )
        return {
            "score": score,
            "mean_trophies": mean_trophies if is_arena else None,
            "repeat_trophies": list(trophies_seen) if is_arena else None,
            "scored_on": "trophies" if is_arena else "versus_outcome",
            "mean_outcome": mean_outcome,
            "mean_lives_diff": mean_lives_diff,
            "n_repeats": len(outcomes),
            "repeat_outcomes": outcomes,
            "rollout_crn": bool(self.rollout_crn),
            "crn_decision_id": self._crn_decision_id(),
            # exp12 route a (wave A0): per-repeat lives margin (the tiebreak
            # term's own components) + the CRN evidence the checker's T4 gate
            # compares across a decision group. Only emitted for the RECORDER,
            # because these ride into `search_diagnostics` and therefore into
            # every per-game JSONL row: at the deployable preset they would add
            # ~7 KB per game to a file that is already ~98 KB per game, bought
            # by nobody who is not recording.
            **(
                {
                    "repeat_lives_diffs": lives_diffs,
                    "repeat_crn_keys": repeat_crn_keys,
                    "repeat_engine_seeds": repeat_engine_seeds,
                }
                if self.capture_teacher_record
                else {}
            ),
            # exp10 W2: correctness-proof evidence, see this method's own
            # docstring + module docstring's "exp10 W2" section.
            "rollout_opponent_mode": self.rollout_opponent_mode,
            "true_followed_pid": true_pid,
            "repeat_chosen_opponent_pids": repeat_chosen_opponent_pids,
            "repeat_fallback_tiers": repeat_fallback_tiers,
            "repeat_opponent_pid_traces": repeat_opponent_pid_traces,
        }

    # ------------------------------------------------------------------
    # exp16 W3: the ANYTIME face of the honest search. Added as NEW methods
    # only -- `recommend`, `_search` and `_search_vgame_honest` are untouched
    # (exp16 PLAN risk 6: exp13 is flying on those, so this line may only add).
    # ------------------------------------------------------------------
    def set_candidate_stream(self, key: int) -> None:
        """exp16 W3: pin candidate sampling to the DECISION, not to the process.

        `_rng_for_candidate` keys the per-candidate sampling RNG on
        `SeedSequence([seed, _recommend_call_count, index])`, and that counter
        is a PROCESS counter: it counts how many `recommend()` calls this
        object has served since it was constructed. That is the right default
        for an eval driver, where "don't repeat the same noise every turn" is
        all it has to buy -- but it makes a decision depend on how many
        decisions came before it in the same process.

        For a duel that has to be archivable and replayable (exp16 W6) that
        is fatal: replaying the same game a second time in the same process
        starts the counter where the first run left it, so the second run's
        candidates are different chains and the game diverges at turn 1.
        Measured exactly that way, and it was the ONLY source of divergence
        (`duel_smoke.py`'s replay check).

        So the duel worker calls this once per SEGMENT with a key derived
        from `(engine_seed, turn, segment_index)` -- the same tuple A1's
        imagination stream is keyed on, for the same reason -- and the next
        `recommend`/`recommend_anytime` call runs on exactly `key`. Nothing
        else calls it, so the driver's counter behaviour is untouched.
        """
        self._recommend_call_count = int(key) - 1

    def recommend_anytime(
        self,
        state: dict[str, Any],
        *,
        chunk_size: int = DEFAULT_ANYTIME_CHUNK,
        should_stop: Callable[[], bool] | None = None,
        max_candidates: int | None = None,
        on_chunk: Callable[[dict[str, Any]], None] | None = None,
        next_chunk_size: Callable[[dict[str, Any]], int] | None = None,
        extra_samples_until_stop: bool = False,
        on_level: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """`recommend()` for an interactive clock: score in CHUNKS, keep the
        best so far, and be interruptible between chunks.

        WHY. A duel turn runs on a fixed wall clock (`turn_budget_s`, 105 s
        by default) split into per-segment deadlines, so the search stops on
        TIME rather than at a preset width, and the caller needs a best-so-far
        answer at whatever moment that is.

        The human pressing "End turn" is NOT one of those stops, and an
        earlier version of this docstring said it was ("the AI has to commit
        within a fraction of a second"). `16-play-vs-ai/
        PLAN_A2_FIXED_CLOCK_8766.md` removed human-stop truncation and
        `ai_worker.request_stop` records it in one line ("Human End Turn no
        longer stops inference"); `_should_stop` reads only the generation, a
        chunk budget and two wall-clock deadlines.

        There is no interruption point INSIDE a chunk and this method does not
        pretend otherwise: `should_stop` is consulted at chunk boundaries only.
        `chunk_size` ships at 1, so that boundary is every single candidate.

        WHAT IS AND IS NOT NEW HERE. Every piece of A1 semantics is the same
        object the driver's path uses: `_prefix_walk` for the committable
        prefix, `_prefix_group_key` for the dedup, `_imagined_completion`
        for the k CRN-keyed resamples, `_honest_myopic_scores`,
        `vgame_scorer.score_boards`, `_prefix_chain`. What this method owns
        is the BOOKKEEPING that `_search_vgame_honest` does in one pass and
        this one does incrementally -- grouping, spans, argmax.

        EXACT AGREEMENT WHEN NOTHING STOPS IT. Candidates are generated in
        the same index order (`_generate_candidate` is a pure function of
        `(state, index)`), groups keep first-seen order, and the argmax uses
        the same `(group_score, group_myopic)` key resolved to the FIRST
        maximum -- so with no stop request and the full width this returns
        the same chain as `recommend()`. internal project notes
        duel_smoke.py --equivalence-states 20` is the check on real states.

        The one deliberate difference in the RESULT: `search_n_generated` is
        how many candidates this call actually generated, not the configured
        width, because under a stop those differ and the UI has to show the
        human the width the AI really got.
        """
        if not (self.honest and self.scoring == SCORING_VGAME):
            raise ValueError(
                f"search_recommender_anytime_requires_honest_vgame:"
                f"turn_mode={self.turn_mode!r}:scoring={self.scoring!r}"
            )
        chunk_size = max(1, int(chunk_size))

        # Identical preamble to `recommend()` -- same counters, same
        # defensive reads, so a chunked decision keys its CRN streams
        # exactly the way an unchunked one at the same call index would.
        self._recommend_call_count += 1
        turn_value = state.get("turn") if isinstance(state, dict) else None
        try:
            self._decision_turn = int(turn_value) if turn_value is not None else None
        except (TypeError, ValueError):
            self._decision_turn = None
        self._state_seed = honest_frame.read_engine_seed(state) if isinstance(state, dict) else 0

        try:
            greedy_result = self._call_bc_recommend(
                state, deterministic=True, force_ranked_decode=True
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"search_greedy_call_failed:{exc}",
                "recommended_action": None,
                "chain_preview": [],
                "wdl_probs": None,
                "diagnostics": {},
                "search_used": False,
                "search_n_candidates": 0,
                "search_scores": [],
                "search_chosen_index": 0,
                "search_greedy_score": None,
                "search_error": str(exc),
            }

        try:
            return self._search_anytime(
                state=state,
                greedy_result=greedy_result,
                chunk_size=chunk_size,
                should_stop=should_stop,
                max_candidates=max_candidates,
                on_chunk=on_chunk,
                next_chunk_size=next_chunk_size,
                extra_samples_until_stop=extra_samples_until_stop,
                on_level=on_level,
            )
        except Exception as exc:
            result = dict(greedy_result)
            result.update(
                {
                    "search_used": False,
                    "search_n_candidates": 0,
                    "search_scores": [],
                    "search_chosen_index": 0,
                    "search_greedy_score": None,
                    "search_error": str(exc),
                }
            )
            return result

    def _anytime_group_boards(
        self, walk: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
        """The boards V scores for ONE prefix group, and how they split by
        imagined sample.

        Returns `(boards, subs)`. `boards` is 1 board if the prefix is
        deterministic, else the surviving imagined samples' completions
        concatenated. `subs` carries one `(start, stop, sample_r)` per
        SURVIVING sample, as offsets into `boards`.

        exp16 A3: `subs` is the whole point. It is what lets the caller take
        the same TWO-LAYER score `_search_vgame_honest` takes -- mean over the
        imagined samples of `_completion_agg` over that sample's inner
        completions -- instead of one flat mean over every board. Under
        `bc_greedy` each sub-span has length 1, `_completion_agg` returns it
        unchanged, and the group score is the flat mean over samples this path
        has always computed.

        Empty `boards` means every completion failed, and the caller must DROP
        the group rather than fall back to the proposal stream's own board --
        exp13 W0b' codex finding 3, restated here because the failure is
        silent and only shows up under failure.
        """
        if walk["boundary"] is None:
            return [walk["board"]], [(0, 1, 0)]
        # exp16 A3 landed the sub-span bookkeeping this path was missing, so
        # `v_search` is no longer refused here. It buys depth by searching
        # less WIDTH inside the same clock -- a strength trade, not a latency
        # one, because human-stop truncation does not exist any more (see
        # `recommend_anytime`'s docstring). Chunking is untouched: a chunk is
        # one candidate, so the interruption grain is already the finest it
        # can be and a costlier candidate does not coarsen it.
        boards: list[dict[str, Any]] = []
        subs: list[tuple[int, int, int]] = []
        for sample_r in range(1, self.stochastic_samples + 1):
            got, span = self._anytime_sample_boards(walk, sample_r, len(boards))
            if span is None:
                continue
            boards.extend(got)
            subs.append(span)
        return boards, subs

    def _anytime_sample_boards(
        self, walk: dict[str, Any], sample_r: int, offset: int
    ) -> tuple[list[dict[str, Any]], tuple[int, int, int] | None]:
        """ONE imagined sample's inner boards, and its span at `offset`.

        Split out of `_anytime_group_boards` (W11b) so the `resample-clock`
        gear can add a LATER sample to a group that was already scored,
        through the identical code path the first `stochastic_samples` went
        through. A second implementation would be the obvious way to get a
        different frame by accident.

        `(boards, None)` means every completion of this sample failed. Losing
        ONE sample must not lose the candidate; losing them all is what the
        caller's drop handles.
        """
        try:
            got = self._imagined_completions(walk, sample_r)
            if not got:
                raise RuntimeError("honest_resample_no_completion")
        except Exception:
            return [], None
        return got, (int(offset), int(offset) + len(got), int(sample_r))

    def _search_anytime(
        self,
        *,
        extra_samples_until_stop: bool = False,
        state: dict[str, Any],
        greedy_result: dict[str, Any],
        chunk_size: int,
        should_stop: Callable[[], bool] | None,
        max_candidates: int | None,
        on_chunk: Callable[[dict[str, Any]], None] | None,
        next_chunk_size: Callable[[dict[str, Any]], int] | None,
        on_level: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        opponent_team = _read_last_opponent_team(state)
        if not opponent_team:
            # Turn 1 (or any turn before a battle has ever resolved): the
            # same designed skip `_search` takes, for the same reason.
            return _annotate_skip(greedy_result)

        width = int(self.n_candidates if max_candidates is None else max_candidates)
        width = max(1, width)
        race = self._race_scalars(state)
        self._reset_completion_counters()

        groups: dict[tuple[str, ...], dict[str, Any]] = {}
        order: list[tuple[str, ...]] = []
        group_scores: list[float] = []
        group_myopic: list[float | None] = []
        group_sample_scores: list[list[float]] = []
        # SAMPLES and BOARDS are two different counts once the inner width can
        # exceed 1, so both are carried rather than one being inferred from
        # the other -- the same pair `_search_vgame_honest` reports.
        group_n_samples: list[int] = []
        group_n_boards: list[int] = []
        # W11b. A later k-level reopens a group that was already scored, so the
        # per-SAMPLE aggregates are carried rather than only their mean: adding
        # a sample is then appending one number and re-taking the mean, over
        # the identical two-layer shape the first pass used.
        group_aggs: list[list[float]] = []
        group_keys_in_order: list[tuple[str, ...]] = []
        v_scores: list[float] = []
        ensemble_std: list[float] = []
        all_myopic: list[float | None] = []
        n_dropped = 0
        generated = 0
        chunks = 0
        stopped = False
        chunk_sizes: list[int] = []
        chunk_timings: list[dict[str, Any]] = []
        timing_totals = {
            "candidate_s": 0.0,
            "outcomes_s": 0.0,
            "myopic_s": 0.0,
            "value_s": 0.0,
            "total_s": 0.0,
        }
        leaves_evaluated = 0
        decided_here = 0
        divergent_here = 0

        while generated < width:
            requested_size = chunk_size
            if next_chunk_size is not None:
                requested_size = int(
                    next_chunk_size(
                        {
                            "chunks": int(chunks),
                            "n_generated": int(generated),
                            "n_dedup": int(len(order)),
                            "width_requested": int(width),
                        }
                    )
                )
            size = min(max(1, int(requested_size)), width - generated)
            chunk_started = time.perf_counter()
            chunk_indices = list(range(generated, generated + size))
            generated += size
            chunk_sizes.append(int(size))

            # Same candidates, same order, as the one-pass path: index 0 IS
            # the greedy result, every other index is a pure function of
            # `(state, index)`.
            candidate_started = time.perf_counter()
            new_keys: list[tuple[str, ...]] = []
            for index in chunk_indices:
                cand = greedy_result if index == 0 else self._generate_candidate(state, index)
                walk = self._prefix_walk(state, cand.get("chain_preview") or [])
                key = self._prefix_group_key(walk)
                if key in groups:
                    continue
                groups[key] = {"first_index": index, "walk": walk, "candidate": cand}
                new_keys.append(key)
            candidate_s = time.perf_counter() - candidate_started

            outcomes_started = time.perf_counter()
            boards: list[dict[str, Any]] = []
            spans: list[tuple[int, int]] = []
            # One entry per kept group, each a list of (start, stop, sample_r)
            # into THIS chunk's `boards`. Chunk-local, exactly like `boards`
            # and `spans`, because the V forward below is per chunk.
            sub_spans: list[list[tuple[int, int, int]]] = []
            kept: list[tuple[str, ...]] = []
            for key in new_keys:
                start = len(boards)
                group_boards, group_subs = self._anytime_group_boards(
                    groups[key]["walk"]
                )
                if not group_boards:
                    n_dropped += 1
                    if not order and not kept:
                        # The greedy chain's own group is the tie-break
                        # anchor; losing it degrades the whole decision,
                        # exactly as the one-pass path does.
                        raise RuntimeError(
                            f"honest_prefix_completions_failed:dropped={n_dropped}/{len(new_keys)}"
                        )
                    # Left in `groups` (so a later candidate with the same
                    # prefix is still deduped away, exactly as one-pass does)
                    # but never added to `order`, so it cannot be chosen.
                    continue
                boards.extend(group_boards)
                spans.append((start, len(boards)))
                sub_spans.append(
                    [(start + a, start + b, r) for (a, b, r) in group_subs]
                )
                kept.append(key)
            outcomes_s = time.perf_counter() - outcomes_started

            myopic_s = 0.0
            value_s = 0.0
            if boards:
                myopic_started = time.perf_counter()
                myopic = self._honest_myopic_scores(boards, opponent_team)
                myopic_s = time.perf_counter() - myopic_started
                value_started = time.perf_counter()
                scored = self.vgame_scorer.score_boards(boards, myopic_scores=myopic, **race)
                value_s = time.perf_counter() - value_started
                leaf = [float(x) for x in scored["score"]]
                if len(leaf) != len(boards):
                    raise RuntimeError(
                        f"vgame_scorer_length_mismatch:{len(leaf)}!={len(boards)}"
                    )
                v_scores.extend(float(x) for x in scored["v"])
                ensemble_std.extend(float(x) for x in scored["ensemble_std"])
                all_myopic.extend(myopic)
                chunk_decided, chunk_divergent = self._count_completion_decisions(
                    leaf, sub_spans
                )
                decided_here += chunk_decided
                divergent_here += chunk_divergent
                for key, (a, b), subs in zip(kept, spans, sub_spans):
                    order.append(key)
                    samples = leaf[a:b]
                    # TWO layers, the same shape `_search_vgame_honest` uses:
                    # mean over the imagined SAMPLES of the aggregate over that
                    # sample's inner completions. Under `bc_greedy` every
                    # sub-span is one board and `_completion_agg` returns it
                    # unchanged, so this reduces -- value by value, in the same
                    # order -- to the flat mean it has always been.
                    agged = [self._completion_agg(leaf[x:y]) for (x, y, _r) in subs]
                    group_scores.append(
                        sum(agged) / float(len(agged)) if agged else 0.0
                    )
                    present = [float(m) for m in myopic[a:b] if m is not None]
                    group_myopic.append(sum(present) / float(len(present)) if present else None)
                    group_sample_scores.append(samples)
                    group_n_samples.append(len(subs))
                    group_n_boards.append(b - a)
                    group_aggs.append(list(agged))
                    group_keys_in_order.append(key)
            leaves_evaluated += len(boards)

            chunks += 1
            total_s = time.perf_counter() - chunk_started
            chunk_info = {
                "chunk": int(chunks),
                "size": int(size),
                "n_generated": int(generated),
                "n_dedup": int(len(order)),
                "new_prefixes": int(len(kept)),
                "leaves": int(len(boards)),
                "candidate_s": float(candidate_s),
                "outcomes_s": float(outcomes_s),
                "myopic_s": float(myopic_s),
                "value_s": float(value_s),
                "total_s": float(total_s),
            }
            chunk_timings.append(chunk_info)
            for key in timing_totals:
                timing_totals[key] += float(chunk_info[key])
            if on_chunk is not None:
                on_chunk(copy.deepcopy(chunk_info))
            if should_stop is not None and should_stop():
                stopped = True
                break

        # ------------------------------------------------------ W11b: more k
        # The `resample-clock` gear pins the OUTER width and spends whatever is
        # left of the segment's slice on more imagined samples instead.
        #
        # LEVEL-SYNCHRONOUS on purpose. `group_scores` is a mean over samples,
        # so a group holding more samples than its siblings is not inflated the
        # way a bigger inner `max` would be -- but it does carry less variance,
        # and an argmax over unequal variances favours the noisy ones. Equal
        # samples costs nothing here and removes the question.
        #
        # A level is BUILT in full and only then committed, and `should_stop`
        # is consulted only BETWEEN levels. A half level would hand back
        # exactly the unequal-variance ranking this is avoiding.
        # Which groups are eligible to hold imagined samples at all. A
        # deterministic prefix contributes exactly one board and never takes
        # part, so it must not count toward either the synchrony check or the
        # "did any sampling happen" question.
        random_groups = [
            gi for gi, key in enumerate(order)
            if groups[key]["walk"].get("boundary") is not None
        ]
        extra_levels = 0
        # Seconds per level, kept per level rather than as a mean. PLAN_W11 W2
        # asks for this measured rather than extrapolated from the price table,
        # which is only trustworthy near width 72 and is off by a third at
        # 1024. It is also the operator-facing answer to "will another level
        # fit in this slice", which no other field answers.
        extra_level_seconds: list[float] = []
        if extra_samples_until_stop and not stopped and order:
            next_r = int(self.stochastic_samples)
            # A hard ceiling as well as the clock. The clock is the normal stop
            # and this never fires in production, but the loop is otherwise
            # bounded only by `should_stop` and by a level failing, and a
            # mutation run that removed the completeness guard turned it into
            # an infinite loop rather than a failing test. An unbounded loop in
            # the inference path is worth two lines to close.
            max_extra_levels = MAX_EXTRA_SAMPLE_LEVELS
            while extra_levels < max_extra_levels:
                if should_stop is None or should_stop():
                    if should_stop is not None:
                        stopped = True
                    break
                next_r += 1
                level_started = time.perf_counter()
                level: list[tuple[int, list[dict[str, Any]]]] = []
                complete = True
                for gi, key in enumerate(group_keys_in_order):
                    walk = groups[key]["walk"]
                    if walk.get("boundary") is None:
                        # A deterministic prefix is one board with no sample to
                        # add. Not a failure: it does not take part.
                        continue
                    got, span = self._anytime_sample_boards(walk, next_r, 0)
                    if span is None:
                        complete = False
                        break
                    level.append((gi, got))
                # BOTH conditions. `complete` was set and never read in the
                # first draft of this loop, so a level whose second group
                # failed still committed its first -- the exact half level the
                # comment above forbids, and nothing would have said so.
                if not complete or not level:
                    break
                level_boards: list[dict[str, Any]] = []
                level_spans: list[tuple[int, int]] = []
                for _gi, got in level:
                    level_spans.append((len(level_boards), len(level_boards) + len(got)))
                    level_boards.extend(got)
                level_myopic = self._honest_myopic_scores(level_boards, opponent_team)
                scored_level = self.vgame_scorer.score_boards(
                    level_boards, myopic_scores=level_myopic, **race
                )
                level_leaf = [float(x) for x in scored_level["score"]]
                if len(level_leaf) != len(level_boards):
                    raise RuntimeError(
                        f"vgame_scorer_length_mismatch:{len(level_leaf)}!={len(level_boards)}"
                    )
                v_scores.extend(float(x) for x in scored_level["v"])
                ensemble_std.extend(float(x) for x in scored_level["ensemble_std"])
                all_myopic.extend(level_myopic)
                lvl_decided, lvl_divergent = self._count_completion_decisions(
                    level_leaf, [[(a, b, next_r)] for (a, b) in level_spans]
                )
                decided_here += lvl_decided
                divergent_here += lvl_divergent
                for (gi, got), (a, b) in zip(level, level_spans):
                    group_aggs[gi].append(self._completion_agg(level_leaf[a:b]))
                    group_scores[gi] = sum(group_aggs[gi]) / float(len(group_aggs[gi]))
                    group_n_samples[gi] += 1
                    group_n_boards[gi] += len(got)
                leaves_evaluated += len(level_boards)
                extra_levels += 1
                extra_level_seconds.append(time.perf_counter() - level_started)
                if on_level is not None:
                    # AFTER the level is committed, never during: a half level
                    # does not enter the ranking, so it must not enter the
                    # readout either. Reporting a level in progress would show
                    # a k the search is not actually using.
                    on_level({
                        "extra_levels": int(extra_levels),
                        "realised_stochastic_samples": int(self.stochastic_samples) + int(extra_levels),
                        "level_seconds": extra_level_seconds[-1],
                        "leaves_evaluated": int(leaves_evaluated),
                    })

        self._completion_decided += decided_here
        self._completion_divergent += divergent_here

        if not order:
            # Nothing survived (only reachable if every group in every chunk
            # dropped, which the greedy-anchor guard above already refuses).
            return _annotate_skip(greedy_result)

        # Same ranking as the one-pass path -- `(group score, group myopic)`,
        # first maximum wins, so a full tie keeps the greedy chain (group 0)
        # and search can never do worse than greedy. The one difference is
        # the EPSILON, and it is not cosmetic: the one-pass path scores every
        # board in ONE batched V forward while this one scores a batch per
        # chunk, and torch's CPU reduction is not bit-identical across batch
        # shapes. Measured on a real duel state, two prefixes that tie to all
        # 17 digits in one pass came out 2.2e-8 apart when chunked, which
        # silently flipped the tie-break away from the earlier candidate.
        # 1e-6 is ~50x that noise and ~1000x below any V difference that
        # means anything, so this restores the first-maximum property
        # without ever overriding a real preference.
        best_i = 0
        for i in range(1, len(order)):
            gap = group_scores[i] - group_scores[best_i]
            if gap > SCORE_TIE_EPS:
                best_i = i
            elif abs(gap) <= SCORE_TIE_EPS:
                mine = group_myopic[i] if group_myopic[i] is not None else 0.0
                best = group_myopic[best_i] if group_myopic[best_i] is not None else 0.0
                if mine > best + SCORE_TIE_EPS:
                    best_i = i
        best_key = order[best_i]
        best_walk = groups[best_key]["walk"]
        winner = groups[best_key]["candidate"]
        chosen_chain = self._prefix_chain(best_walk)

        candidate_chains: list[list[dict[str, Any]]] | None = None
        if self.capture_candidate_chains:
            candidate_chains = [self._prefix_chain(groups[key]["walk"]) for key in order]

        result = dict(winner)
        result.update(
            {
                **_searched_verdict(chosen_chain),
                "search_used": True,
                "search_scoring": SCORING_VGAME,
                "search_turn_mode": self.turn_mode,
                "search_n_candidates": len(order),
                # Deliberately the REAL count, not `self.n_candidates` -- see
                # `recommend_anytime`'s docstring.
                "search_n_generated": int(generated),
                "search_n_dedup": len(order),
                # exp16 A3, mirroring the one-pass path field for field: the
                # policy is on the row so a reader never has to infer which arm
                # produced it, and `decided`/`divergent` are the measurement
                # that the intervention happened at all. A `v_search` arm whose
                # inner search re-derives the greedy completion every time
                # deepened nothing.
                "search_completion_policy": self.completion_policy,
                "search_completion_width": int(self.completion_width),
                "search_completion_aggregate": self.completion_aggregate,
                "search_completion_decided": int(decided_here),
                "search_completion_divergent": int(divergent_here),
                # W11b. What the `resample-clock` gear actually did. The gear is
                # the only thing that can raise these above the configured
                # `stochastic_samples`, so a non-zero level count IS its
                # treatment-took gate: name the quantity that says the
                # intervention happened, never infer it from the config.
                # None, not a number, when no random-prefix group contributed
                # a sample: a deterministic-only search did no sampling at all,
                # and reporting the CONFIGURED k there would render as
                # "3 (no extra)" on the replay page, i.e. as a measurement.
                # Absent is not zero, and this is the one place it is
                # load-bearing. (codex review, 2026-08-21.)
                "search_realised_stochastic_samples": (
                    int(self.stochastic_samples) + int(extra_levels)
                    if random_groups else None
                ),
                "search_extra_sample_levels": int(extra_levels) if random_groups else None,
                "search_extra_level_seconds": [round(float(x), 4) for x in extra_level_seconds],
                # Level synchrony is only meaningful ACROSS the loop; a group
                # can already arrive unequal, because a sample that raises is
                # skipped with `continue` in the baseline pass. Pre-existing and
                # not introduced here (the other gears simply never deepen), so
                # it is COUNTED rather than rewritten around: the ranking core
                # is not the place to fix a defect whose measured incidence is
                # still zero. Deterministic groups are excluded on purpose --
                # they legitimately hold exactly one board.
                "search_group_samples_unequal": bool(
                    len({group_n_samples[i] for i in random_groups}) > 1
                ) if random_groups else False,
                **self._completion_instruments(),
                "search_scores": group_scores,
                "search_chosen_index": best_i,
                "search_greedy_score": group_myopic[0] if group_myopic else None,
                "search_candidate_chains": candidate_chains,
                "search_anytime": {
                    "chunk_size": int(chunk_size),
                    "chunk_sizes": chunk_sizes,
                    "chunks": int(chunks),
                    "width_requested": int(width),
                    "width_searched": int(generated),
                    "unique_prefixes": int(len(order)),
                    "leaves_evaluated": int(leaves_evaluated),
                    "stage_seconds": timing_totals,
                    "chunk_timings": chunk_timings,
                    "stopped": bool(stopped),
                },
                "search_diagnostics": {
                    "scoring": SCORING_VGAME,
                    "turn_mode": self.turn_mode,
                    "stochastic_samples": int(self.stochastic_samples),
                    "realised_stochastic_samples": int(self.stochastic_samples) + int(extra_levels),
                    "extra_sample_levels": int(extra_levels),
                    "extra_level_seconds": [round(float(x), 4) for x in extra_level_seconds],
                    "completion_second_chance_nodes": int(self._completion_second_chance),
                    "completion_boards_mean": (
                        float(self._completion_boards) / float(self._completion_samples)
                        if self._completion_samples
                        else None
                    ),
                    "completion_dropped": int(self._completion_dropped),
                    "imagination_key_source": self._imagination_key_source(),
                    "imagination_segment_index": int(self._imagination_segment_index),
                    "imagination_sample_seeds": [
                        self._imagination_seed(r)
                        for r in range(1, self.stochastic_samples + 1)
                    ],
                    "prefix_boundaries": [groups[key]["walk"]["boundary"] for key in order],
                    "prefix_stops": [groups[key]["walk"]["stop"] for key in order],
                    "prefix_lengths": [len(groups[key]["walk"]["ops"]) for key in order],
                    # SAMPLES, not boards. Under `v_search` one sample
                    # carries up to `completion_width` boards, so the length of
                    # a group's score list stopped being its sample count.
                    "prefix_n_samples": list(group_n_samples),
                    "prefix_n_boards": list(group_n_boards),
                    "prefix_n_dropped": int(n_dropped),
                    "prefix_sample_scores": group_sample_scores,
                    "group_scores": group_scores,
                    "greedy_leaf_score": (group_scores[0] if group_scores else None),
                    "v_scores": v_scores,
                    "leaf_scores": [s for samples in group_sample_scores for s in samples],
                    "ensemble_std": ensemble_std,
                    "myopic_scores": [
                        (None if m is None else float(m)) for m in all_myopic
                    ],
                    "myopic_oracle_used": bool(self._vgame_needs_myopic()),
                    "blend": float(getattr(self.vgame_scorer, "blend", 0.0)),
                    "pessimism": float(getattr(self.vgame_scorer, "pessimism", 0.0)),
                    "race": race,
                    "candidate_chains": candidate_chains,
                    "anytime": {
                        "chunk_size": int(chunk_size),
                        "chunk_sizes": chunk_sizes,
                        "chunks": int(chunks),
                        "width_requested": int(width),
                        "width_searched": int(generated),
                        "unique_prefixes": int(len(order)),
                        "leaves_evaluated": int(leaves_evaluated),
                        "stage_seconds": timing_totals,
                        "chunk_timings": chunk_timings,
                        "stopped": bool(stopped),
                    },
                },
            }
        )
        return result

    def recommend(self, state: dict[str, Any]) -> dict[str, Any]:
        """API-compatible with `BcRecommender.recommend` -- see module
        docstring for the full algorithm. Never raises: any failure
        degrades to the greedy result (or, if even the greedy call itself
        fails, to a minimal `ok=False` shape mirroring `BcRecommender`'s
        own hard-failure contract) rather than propagating.
        """
        self._recommend_call_count += 1
        # exp12 route a (wave A0): the decision's turn, for the CRN key and
        # the teacher record. Read defensively (never raises) -- a state
        # without a usable turn simply leaves this None and `_crn_decision_id`
        # falls back to the recommend counter.
        turn_value = state.get("turn") if isinstance(state, dict) else None
        try:
            self._decision_turn = int(turn_value) if turn_value is not None else None
        except (TypeError, ValueError):
            self._decision_turn = None
        # exp13 A1: the fallback base for `_imagination_seed` when a caller
        # never set the driver context. Read here, once, off the state this
        # decision was handed (which under the driver is ALREADY the imagined
        # clone, i.e. S(engine_seed, turn, segment, 0)).
        self._state_seed = honest_frame.read_engine_seed(state) if isinstance(state, dict) else 0
        try:
            # A2.5: candidate 0 is the greedy ANCHOR -- the tie-break that
            # makes "search can never do worse than greedy" true, the
            # `_annotate_skip` fallback, and the thing a plain-BC arm is
            # comparable to. `.deterministic` does not reach the decoder
            # (see `_call_bc_recommend`), so under `--decode-mode sample` it
            # came back sampled and all three properties were lost.
            greedy_result = self._call_bc_recommend(
                state, deterministic=True, force_ranked_decode=True
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": f"search_greedy_call_failed:{exc}",
                "recommended_action": None,
                "chain_preview": [],
                "wdl_probs": None,
                "diagnostics": {},
                "search_used": False,
                "search_n_candidates": 0,
                "search_scores": [],
                "search_chosen_index": 0,
                "search_greedy_score": None,
                "search_error": str(exc),
            }

        try:
            return self._search(state, greedy_result)
        except Exception as exc:
            result = dict(greedy_result)
            result.update(
                {
                    "search_used": False,
                    "search_n_candidates": 0,
                    "search_scores": [],
                    "search_chosen_index": 0,
                    "search_greedy_score": None,
                    "search_error": str(exc),
                }
            )
            return result
