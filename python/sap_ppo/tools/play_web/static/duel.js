/* exp16 W5: the duel page's own chrome, on top of the exp14 shop skin.

   WHAT THIS FILE OWNS. Everything the skin does not know about: the opponent
   lane, the thinking badge, the gear switch, the per-turn record, the failure
   banner and the end screen. The human's own board and shop are rendered by
   `app.js`, unchanged, against `/api/duel/*` -- see `apiPath` there.

   THE OPPONENT LANE IS A STATE MACHINE WITH THREE STATES (W5.1), and which
   one it is in is on the element (`#ai-lane[data-phase]`) rather than in a
   variable only this file can see. Which turn's board is on screen rides on
   the same element as `data-seen-turn`:

     unseen     no battle has resolved yet, so there is nothing the human has
                been shown. The curtain is up. This is the ONLY state with a
                curtain, and a game passes through it once, at the start.
     revealed   the board that just FOUGHT, next to its result.
     last-seen  that same board, left standing while the next turn is played.

   W5.1 REPLACED THE OLD `pending` STATE. Until W5 the reveal ran on a 6 s
   timer that put the lane back behind the curtain, so most of a turn was spent
   looking at a hatched rectangle. What that curtain hid is exactly what the
   real client leaves standing -- the opponent's board as you last saw it -- so
   the dwell now ends in `last-seen`, and pressing End turn leaves the board
   alone instead of clearing it.

   THAT IS NOT A LEAK. The server only ever sends the board of a turn whose
   battle has already RESOLVED (`duel.turns[i].ai_board`); the board the AI is
   choosing right now never enters a snapshot, which
   `test_exp16_duel_app.py::TestOpponentIsHidden` greps for. So `data-seen-turn`
   is always a turn in the past, and `gate_duel.py` asserts it is exactly
   `turn - 1` before every battle.

   THE TWO BOARDS FACE EACH OTHER (W5.1). `renderBoard` takes a side. The
   human's board keeps the shop scene's own order and mirroring (slot 0 on the
   right, art mirrored so the front rank faces right); the AI is the ENEMY, so
   its slot 0 is on the LEFT, its art is not mirrored, and its level plaque
   mirrors with it to the top-right corner. Attack and health do NOT mirror:
   the real client draws attack left of health on both sides. Reference =
   exp15's captured battle frames, `strips_final/strip_00`, REAL row.

   NOTHING HERE INVENTS A NUMBER. The searched/deadline counts, the widths, the
   elapsed times and the lives all come from the server's per-turn record
   (`duel.turns[]`), which is the same record W6 archives. */

(function () {
  /* THE POLL'S TWO CADENCES, and why it is not one.

     This poll exists for one thing: the AI's clock is asynchronous, so the
     badge and the render card have to ask. Nothing the human does needs it --
     every mutation answers with the state it produced.

     It used to be a flat 250 ms with no in-flight guard, which is fine on
     loopback and hostile over a tunnel. At Ruihan's measured 244 ms RTT with
     20% packet loss the browser fired four polls a second and completed about
     one; on 2026-08-13 the server logged ~76 of the 240 polls a minute it was
     being sent. The other three quarters sat in Chrome's six-connections-per-
     host queue -- and a Roll or a Buy has to queue there too, behind polls
     that are only asking whether an idle AI is still idle.

     So: fast only while the AI's clock is actually running or a battle is
     resolving, slow while the human shops, and never more than one in flight.
     The badge's numbers come off the server payload, so the cadence IS their
     resolution; half a second on a seconds counter is not visible, and 250 ms
     was never buying anything a human could see. */
  const POLL_ACTIVE_MS = 500;
  const POLL_IDLE_MS = 1500;
  // exp16 show-value: how long the live board readout waits after the last
  // repaint before asking the server. Every buy/roll/reorder repaints, and a
  // request per keystroke of shopping would be a self-inflicted load.
  const VALUE_DEBOUNCE_MS = Number(window.__SAP_DUEL_VALUE_DEBOUNCE_MS || 220);
  const VALUE_PLACEHOLDER = '--';
  // The two wordings the server is allowed to send, keyed the way it keys them.
  // The page refuses to draw a number under a key it does not know, so a server
  // that grew a third meaning cannot ship a number with no sentence under it.
  const VALUE_MEANING_KEYS = ['end_of_turn', 'bc_completed'];
  /* The per-row form of the same statement. The full sentence is 270-odd
     characters, and thirteen copies of it down a list is a wall nobody reads --
     which is a way of not saying it. So the full sentence heads the list once,
     each row carries this short form, and the row's `title` is the full text.
     Keyed by the SAME key the full wording is keyed by, so a row can no more
     get a mismatched short label than a mismatched long one. */
  const VALUE_SHORT_MEANINGS = {
    end_of_turn: 'the board as it stands at end of turn',
    bc_completed: 'NOT this move’s value: take it, then BC finishes the turn',
  };
  // How long the board stays in its `revealed` styling after its battle before
  // it settles into `last-seen`. It does not go away at the end of it -- W5.1
  // -- so this is only how long the fresh result stays highlighted.
  // Overridable so a test can pick its own dwell without the page shipping a
  // different one.
  const REVEAL_MS = Number(window.__SAP_DUEL_REVEAL_MS || 6000);

  const SIDE_AI = 'ai';
  const SIDE_HUMAN = 'human';

  // The stand-in "worker" for a transition the page learned from the End turn
  // response rather than from a status sample. It never claims to be the AI.
  const HANDOVER_SAMPLE = {status: 'handover'};

  // What the badge showed, what phase the lane was in, and WHAT THE AI ITSELF
  // REPORTED at the same instant, per turn. The gate reads this to check the
  // badge against reality instead of trying to catch a 400 ms window by
  // polling the DOM from outside. `badgeVsWorker` is what makes "the badge
  // claimed the battle was resolving while the AI was still choosing its turn"
  // a checkable statement rather than something only a human watching the page
  // can notice -- which is how it survived until 2026-08-06.
  const observations = {byTurn: {}};
  window.__SAP_DUEL_OBS = observations;

  // The page's own clock, in one place so a probe can drive it. Everything it
  // times is a duration SHOWN to the human; nothing the server records is
  // measured here.
  function now() {
    return typeof window.__SAP_DUEL_NOW === 'function'
      ? Number(window.__SAP_DUEL_NOW())
      : Date.now();
  }

  let lanePhase = 'unseen';
  let laneTurn = null;      // which turn's board the lane is showing
  let endingTurn = false;
  // The AI's last reported status, so that the moment End turn is pressed the
  // badge can keep telling the truth instead of blanking to a fixed string.
  let lastWorker = {status: 'idle'};
  // When the BATTLE started resolving, or null while the AI is still choosing
  // its turn. It is a latch: inside one End turn it only ever goes thinking ->
  // resolving. `duel_app.py::end_turn` submits the NEXT turn to the worker
  // before its response is serialised, so without the latch the badge would
  // flip back to `thinking` for the tail of the request.
  //
  // WHAT THE LATCH IS ALLOWED TO FIRE ON matters as much as the latch. It used
  // to fire on ANY sample that was not literally `thinking`, which made one
  // out-of-order `/api/duel/status` response -- generated before this turn's
  // `_submit_ai_turn` and delivered after the human pressed End turn, carrying
  // `ready` or `idle` -- pin the badge to `resolving the battle…` for the whole
  // of the AI's 105 s turn. That is the 2026-08-06 defect again, wearing the
  // fix's clothes. It now fires only on a sample that is demonstrably about
  // the turn being ended or a later one; see `badgeStatusNow`.
  let resolvingSince = null;
  // The newest worker generation the page has accepted. `worker.gen` is
  // `DuelApp._gen`, bumped once per submitted turn and never reset by a New
  // game, so `gen < newestWorkerGen` is exactly "this response was overtaken by
  // a newer one" -- the only ordering guarantee a threaded server and a
  // `setInterval` poll with no in-flight guard can give us.
  let newestWorkerGen = -1;
  // What is on the badge right now, so a moment with no usable sample can keep
  // showing it instead of guessing.
  let paintedStatus = 'idle';
  // Which turn an End turn in flight belongs to. `end_turn` resolves the
  // battle and advances the session BEFORE it replies, so `status()` starts
  // reporting turn N+1 while the human is still waiting on turn N's battle and
  // the badge is still talking about it. Without this, that window's
  // observations land on the next turn's row and read as "the page announced
  // the battle before the AI had finished" on a turn where nothing of the sort
  // happened.
  let endingTurnNumber = null;
  let pollTimer = null;
  // Which cadence the interval is currently installed at, and whether a poll
  // is already out. See the two POLL_*_MS constants for why both exist.
  let pollPeriod = null;
  let pollInFlight = false;
  let revealTimer = null;
  let busy = false;
  let gameEpoch = 0;
  let serverGameGeneration = null;
  /* THE SERVER THIS PAGE IS TALKING TO HAS RESTARTED.

     `game_generation` counts games inside ONE process and only ever goes up
     (`DuelApp._game_status_token`, bumped by `new_game` and by nothing else),
     so a value BELOW the one this page pinned cannot happen without the
     process being replaced. Until now the page said nothing about it:
     `acceptsGameGeneration` dropped the response and the poll returned, so the
     badge froze, the board stayed up, and the whole thing looked exactly like
     an AI taking a long time. On 2026-08-19 a deploy orphaned Ruihan's open
     page and he read it as a stall.

     Latched rather than re-derived, because once it is true every later poll
     is about a game this page has never seen. Cleared only by New game, which
     is the one action that still works against the new process. */
  let serverRestarted = false;
  let renderGeneration = null;
  let activeRenderKey = null;
  let failedRenderKey = null;

  function q(id) {
    return document.getElementById(id);
  }

  function duelBlock() {
    return (typeof appState !== 'undefined' && appState && appState.duel) || null;
  }

  function observationRow(turn) {
    const key = String(turn);
    return observations.byTurn[key] || (observations.byTurn[key] = {
      statuses: [], phases: [], badgeVsWorker: [],
    });
  }

  /* WHAT MADE THE PAGE SAY THE BATTLE WAS RESOLVING, recorded the instant it
     decided. This is the claim that matters and the one a stream of
     `badge/worker` pairs cannot express: `resolving` next to a `thinking`
     worker is the 2026-08-06 defect when the worker is thinking about the turn
     being ended, and completely normal when it is already thinking about the
     next one (`end_turn` submits it before it replies). The turn numbers are
     what tells those apart, so they are what gets recorded. */
  function noteHandover(source, worker) {
    if (endingTurnNumber === null) return;
    const workerTurn = Number(worker && worker.turn);
    observationRow(endingTurnNumber).handover = {
      source,
      worker: String((worker && worker.status) || ''),
      worker_turn: Number.isInteger(workerTurn) ? workerTurn : null,
      ended_turn: endingTurnNumber,
    };
  }

  function observe(turn, status, phase, worker) {
    const row = observationRow(turn);
    if (row.statuses[row.statuses.length - 1] !== status) row.statuses.push(status);
    if (row.phases[row.phases.length - 1] !== phase) row.phases.push(phase);
    // The badge next to what the AI actually said at that instant.
    // `resolving/thinking` in here is precisely the 2026-08-06 defect.
    const pair = `${status}/${String((worker && worker.status) || '')}`;
    if (row.badgeVsWorker[row.badgeVsWorker.length - 1] !== pair) {
      row.badgeVsWorker.push(pair);
    }
    return row;
  }

  /* ---------------------------------------------------------------- boards */
  /* A read-only board card. Deliberately NOT `app.js::teamCardHTML`: that one
     reads the HUMAN session's selection, drag state and ability counters, all
     of which would be wrong on the opponent's board. The pieces it is built
     from (the slab, the sprite, the stat badges, the level plaque) are the
     same shared helpers the human's board uses, so the two boards are the same
     skin rather than a lookalike. */
  function boardCardHTML(slot, side) {
    // `is-enemy` is what carries the mirroring that is not the sprite's own:
    // the level plaque moves to the top-RIGHT corner (`duel.css`). The attack
    // and health badges are deliberately left alone.
    const enemy = side === SIDE_AI;
    const enemyCls = enemy ? ' is-enemy' : '';
    const itemId = slot && slot.pet_id;
    if (!itemId) {
      return `<div class="card team-card is-empty is-readonly${enemyCls}">` + SLAB_SVG + '</div>';
    }
    const atk = slot.attack ?? 0;
    const hp = slot.health ?? 0;
    const tip = `${prettyItemName(itemId)} ${atk}/${hp}, Lvl ${slot.level ?? 1}`;
    return [
      `<div class="card team-card is-readonly${enemyCls}" title="${tip}">`,
      SLAB_SVG,
      levelRowHTML(slot.level ?? 1, slot.exp ?? 0),
      spriteHTML('pet', itemId, enemy ? 'enemy' : 'own'),
      petStatsHTML(atk, hp),
      equipmentHTML(slot),
      '</div>',
    ].join('');
  }

  /* `side` is required at every call site rather than defaulted, because the
     wrong side is a silent bug: the board still renders, it just faces the
     wrong way. Anything that is not `SIDE_AI` is drawn as the human's own. */
  function renderBoard(el, team, side) {
    if (!el) return 0;
    const enemy = side === SIDE_AI;
    const slots = Array.isArray(team) ? team.slice() : [];
    // Where the front rank stands. The human's board matches the shop scene's
    // own lane: slot 0 on the RIGHT, facing right. The enemy is the mirror of
    // that -- slot 0 on the LEFT, facing left -- so the two front ranks meet
    // in the middle, which is how the real client draws a battle.
    slots.sort((a, b) => ((a.slot_index ?? 0) - (b.slot_index ?? 0)) * (enemy ? 1 : -1));
    el.innerHTML = slots.map((slot) => boardCardHTML(slot, side)).join('');
    if (el.dataset) el.dataset.side = enemy ? SIDE_AI : SIDE_HUMAN;
    return slots.filter((s) => s && s.pet_id).length;
  }

  /* Did this turn's battle actually RESOLVE?

     `duel_app.py::end_turn` appends a turn record even when the step failed,
     and `_turn_record` fills `outcome`, `lives`, `ai_board` and `human_board`
     ONLY when a battle produced a result. So a failed turn arrives as a record
     with none of them, and treating it like any other would clear the lane
     with its missing board and move `data-seen-turn` onto a turn that never
     fought -- the exact thing assertion 3 of `gate_duel.py` exists to forbid,
     on the one path a passing game never walks. */
  function isResolvedTurn(record) {
    return Boolean(record && record.ok && record.outcome && Array.isArray(record.ai_board));
  }

  /* ------------------------------------------------------------------ lane */
  /* The ONLY function that moves the lane. Three transitions matter:

       unseen    <- a new game, and nothing else
       revealed  <- a battle resolved (carries that turn's record)
       last-seen <- the reveal dwell ran out, or End turn was pressed

     `last-seen` deliberately touches neither the board nor the outcome: it is
     the same board under a different label, so asking for it without ever
     having revealed one falls back to `unseen` rather than showing an empty
     lane with no curtain. */
  function setLane(phase, turnRecord) {
    // A turn that did not resolve has no board to reveal, so the lane keeps
    // the one it is showing. Guarded here and not only at the call site: this
    // is the one function that moves the lane, so it is where the rule lives.
    if (phase === 'revealed' && !isResolvedTurn(turnRecord)) {
      phase = 'last-seen';
      turnRecord = null;
    }
    if (phase === 'last-seen' && laneTurn === null) phase = 'unseen';
    lanePhase = phase;
    const lane = q('ai-lane');
    if (lane) lane.dataset.phase = phase;
    const board = q('ai-board');
    const outcome = q('ai-outcome');
    const note = q('ai-seen-note');
    if (revealTimer) {
      clearTimeout(revealTimer);
      revealTimer = null;
    }
    if (phase === 'revealed' && turnRecord) {
      laneTurn = turnRecord.turn;
      if (lane) lane.dataset.seenTurn = String(turnRecord.turn);
      renderBoard(board, turnRecord.ai_board, SIDE_AI);
      if (outcome) {
        const verdict = String(turnRecord.outcome || '');
        // The human's outcome, said from the human's side.
        const text = verdict === 'win' ? 'You won the battle'
          : verdict === 'loss' ? 'You lost the battle'
          : verdict ? 'Draw' : '';
        outcome.textContent = text ? `Turn ${turnRecord.turn}: ${text}` : '';
        outcome.dataset.outcome = verdict;
      }
      if (note) note.textContent = `the board that just fought (turn ${turnRecord.turn})`;
      const duel = duelBlock();
      if (!(duel && duel.done)) {
        // The result stops being NEWS after the dwell, but the board stays:
        // it is what the human last saw, which is exactly what the real
        // client leaves on screen while the next turn is played.
        revealTimer = setTimeout(() => {
          revealTimer = null;
          setLane('last-seen', null);
        }, REVEAL_MS);
      }
    } else if (phase === 'last-seen') {
      // board, outcome and `data-seen-turn` are all left exactly as they are
      if (note) note.textContent = `last seen on turn ${laneTurn}`;
    } else {
      laneTurn = null;
      if (lane) delete lane.dataset.seenTurn;
      if (board) board.innerHTML = '';
      if (outcome) {
        outcome.textContent = '';
        outcome.dataset.outcome = '';
      }
      if (note) note.textContent = '';
    }
  }

  /* ----------------------------------------------------------------- badge */
  /* WHAT THE BADGE SAYS WHILE End turn IS IN FLIGHT (rewritten 2026-08-06,
     from Ruihan playing a real game).

     `POST /api/duel/end_turn` does two things in a row: it WAITS for the AI to
     finish the turn it has been thinking about since the top of the human's
     turn, and only then resolves the battle. The page used to paint a fixed
     `resolving the battle…` from the press until the response came back, and
     the poll hard-overrode the worker's real status with the same constant --
     so the entire wait, which is the part with something to watch, showed a
     static line. Measured on one game: 56.4 s of it on turn 1, 37.7 s on turn
     2, and 0 s on turn 3, where the AI was already done. Turn 3 looking
     informative and turn 1 looking dead was not two bugs; it was this branch
     hit two ways.

     So: while the AI is thinking, the badge is the thinking badge, numbers and
     all. `resolving the battle…` now means the battle, and carries its own
     clock rather than the AI's stopped one. */
  /* Is this status response worth believing, or was it overtaken?

     `pollStatus` fires on a bare `setInterval` with no in-flight guard, against
     a threaded server: two responses can land out of order, and one generated
     before this turn's `_submit_ai_turn` can arrive after it. Such a response
     carries an OLD `gen`, and a payload with no `worker` block at all carries
     no `gen` -- which used to read as a perfectly good `idle`. Neither is
     information about now, so neither is allowed to move the badge. */
  function acceptWorkerSample(worker) {
    const gen = Number(worker && worker.gen);
    const turn = Number(worker && worker.turn);
    if (!Number.isInteger(gen) || !Number.isInteger(turn)) return false;
    if (gen < newestWorkerGen) return false;
    newestWorkerGen = gen;
    return true;
  }

  /* What the badge should say, given a sample already judged fresh.

     Returns null when the sample says nothing about the turn being ended, in
     which case the caller leaves the badge alone rather than inventing a
     state. The three cases that DO mean something:

       worker.turn === the turn being ended, thinking   -> the AI is still on it
       worker.turn === the turn being ended, not thinking -> it handed over
       worker.turn  >  the turn being ended             -> it is already on the
                                                           next one, so ours is
                                                           long since handed over

     The second and third are the handover, and only they start the battle's
     clock. Nothing else does, which is what stops a stale `ready` from
     announcing a battle that has not happened. */
  function badgeStatusNow(worker) {
    const status = String((worker && worker.status) || 'idle');
    if (!endingTurn) return status;
    if (resolvingSince !== null) return 'resolving';
    if (endingTurnNumber === null) return status;
    const turn = Number(worker && worker.turn);
    if (!Number.isInteger(turn) || turn < endingTurnNumber) return null;
    if (turn === endingTurnNumber && status === 'thinking') return 'thinking';
    resolvingSince = now();
    noteHandover('sample', worker);
    return 'resolving';
  }

  function badgeText(status, worker) {
    const seconds = ((Number(worker && worker.elapsed_ms) || 0) / 1000).toFixed(1);
    if (status === 'resolving') {
      // The battle's own elapsed time. Not the AI's, which has stopped, and
      // not the time since End turn, which would include the AI's thinking.
      const held = ((now() - (resolvingSince === null ? now() : resolvingSince)) / 1000);
      return `resolving the battle… ${held.toFixed(1)}s`;
    }
    if (status === 'thinking') {
      /* All three knobs, because with only the outer width on show there was
         no way to watch `resample-clock` do the one thing it exists for.
         `outer` climbs during the width phase, `inner` is config and fixed for
         the game, `k` only moves under that gear and reads `–` until the first
         level commits -- a gear that has added nothing YET has not added zero. */
      const segment = Number((worker && worker.segment) || 0) + 1;
      const outer = Number((worker && worker.width) || 0);
      const inner = (worker || {}).inner_width;
      const k = (worker || {}).realised_samples;
      const parts = [
        `thinking ${seconds}s`,
        `segment ${segment}`,
        `segment width ${outer || '–'}`,
        `deepening width ${inner === null || inner === undefined ? '–' : Number(inner)}`,
        `k ${k === null || k === undefined ? '–' : Number(k)}`,
      ];
      return parts.join(' · ');
    }
    if (status === 'ready') return `turn ready (${seconds}s)`;
    if (status === 'failed') return 'the AI’s turn failed';
    return 'idle';
  }

  function paintBadge(status, worker) {
    paintedStatus = status;
    const badge = q('ai-badge');
    const text = q('ai-badge-text');
    if (badge) {
      badge.dataset.status = status;
      badge.classList.toggle('is-thinking', status === 'thinking');
    }
    if (text) text.textContent = badgeText(status, worker);
  }

  /* ------------------------------------------------------ battle rendering */
  function paintRender(render) {
    const card = q('duel-render-card');
    const status = q('duel-render-status');
    const placeholder = q('duel-render-placeholder');
    const image = q('duel-render-image');
    const link = q('duel-render-link');
    let info = render || {status: 'idle'};
    let phase = String(info.status || 'idle');
    const serverKey = phase === 'ready'
      ? `${info.generation ?? ''}:${info.turn ?? ''}:${info.url || ''}`
      : null;
    // An image URL that already failed stays quarantined across 250 ms status
    // polls. A later turn/generation gets a different key and can load normally.
    if (serverKey && failedRenderKey === serverKey) {
      info = {...info, status: 'failed', error: 'image_load_failed', calculator_link: null, url: null};
      phase = 'failed';
    }
    if (card) {
      card.dataset.status = phase;
      if (info.turn != null) card.dataset.turn = String(info.turn);
      else delete card.dataset.turn;
    }
    if (link) {
      link.hidden = !info.calculator_link;
      if (info.calculator_link) link.href = String(info.calculator_link);
      else link.removeAttribute('href');
    }
    if (phase === 'idle') {
      activeRenderKey = null;
      failedRenderKey = null;
      if (status) status.textContent = 'No battle has resolved yet.';
      if (image) {
        image.removeAttribute('src');
        delete image.dataset.turn;
        image.hidden = true;
      }
      if (placeholder) {
        placeholder.hidden = false;
        placeholder.textContent = 'The battle image will appear here after End turn.';
      }
      return;
    }
    if (phase === 'ready' && info.url) {
      activeRenderKey = serverKey;
      if (failedRenderKey !== serverKey) failedRenderKey = null;
      if (status) status.textContent = `Battles through turn ${info.turn}.`;
      if (image) {
        if (image.getAttribute('src') !== String(info.url)) image.src = String(info.url);
        image.dataset.turn = String(info.turn);
        image.hidden = false;
      }
      if (placeholder) placeholder.hidden = true;
      return;
    }
    if (phase === 'failed') {
      if (status) status.textContent = `render unavailable: ${info.error || 'unknown'}`;
    } else if (phase === 'pending') {
      if (status) status.textContent = `Rendering battles through turn ${info.turn}…`;
    } else if (status) {
      status.textContent = 'No battle has resolved yet.';
    }
    // A ready image from the preceding turn may remain while the next prefix
    // renders. On turn 1 there is no preceding image, so the placeholder stays.
    const hasImage = Boolean(image && image.getAttribute('src'));
    if (image) image.hidden = !hasImage;
    if (placeholder) {
      placeholder.hidden = hasImage;
      if (phase === 'failed' && !hasImage) {
        placeholder.textContent = `render unavailable: ${info.error || 'unknown'}`;
      } else if (!hasImage) {
        placeholder.textContent = phase === 'pending'
          ? `Rendering turn ${info.turn}…`
          : 'The battle image will appear here after End turn.';
      }
    }
  }

  function pinRenderGeneration(render) {
    const value = Number(render && render.generation);
    renderGeneration = Number.isInteger(value) ? value : null;
  }

  function acceptsRenderGeneration(render) {
    if (renderGeneration == null) return true;
    const value = Number(render && render.generation);
    return Number.isInteger(value) && value === renderGeneration;
  }

  function pinGameGeneration(duel) {
    const value = Number(duel && duel.game_generation);
    serverGameGeneration = Number.isInteger(value) ? value : null;
  }

  function acceptsGameGeneration(payload) {
    if (serverGameGeneration == null) return true;
    return Number(payload && payload.game_generation) === serverGameGeneration;
  }

  /* Has this page been orphaned? Returns true once it has, so the caller stops.

     Strictly BELOW, never merely different: a New game in a second tab raises
     the generation, and that is a live server, not a dead one. */
  function noticedRestartedServer(payload) {
    if (serverRestarted) return true;
    if (serverGameGeneration == null) return false;
    const seen = Number(payload && payload.game_generation);
    if (!Number.isInteger(seen) || seen >= serverGameGeneration) return false;
    serverRestarted = true;
    paintStaleServer(true);
    const endBtn = q('btn-end-turn');
    if (endBtn) endBtn.disabled = true;
    return true;
  }

  function paintStaleServer(on) {
    const el = q('duel-stale-server');
    if (!el) return;
    el.hidden = !on;
    el.textContent = on
      ? 'The server restarted, so the game on this page is over and nothing here is live. Press New game (or reload) to start one on the server that is running now.'
      : '';
  }

  function markImageUnavailable(code) {
    const image = q('duel-render-image');
    const card = q('duel-render-card');
    if (!activeRenderKey) return;
    failedRenderKey = activeRenderKey;
    const turn = image && image.dataset.turn
      ? Number(image.dataset.turn)
      : Number(card && card.dataset.turn);
    if (image) {
      image.removeAttribute('src');
      delete image.dataset.turn;
      image.hidden = true;
    }
    paintRender({
      status: 'failed',
      turn: Number.isFinite(turn) ? turn : null,
      error: String(code || 'image_load_failed'),
      calculator_link: null,
      url: null,
    });
  }

  /* ------------------------------------------------------------------- log */
  function searchLineText(record) {
    if (!record) return 'no turn played yet';
    const searched = Number(record.searched_segments || 0);
    const deadline = Number(record.deadline_segments || 0);
    const total = Number(record.n_segments || 0);
    const seconds = (Number(record.elapsed_ms || 0) / 1000).toFixed(1);
    const budget = Number(record.turn_budget_s || 0);
    return (
      `turn ${record.turn}: ${searched} searched, ${deadline} deadline-end `
      + `of ${total} segments · ${seconds}s of ${budget}s`
    );
  }

  /* exp16 A3's deepening, summed over the game.

     Game totals rather than the last turn's, because the question the human is
     asking of this line is "is the deepening on, and is it doing anything",
     and one turn's `decided` is often 0 on a small board. `decided` is the
     DENOMINATOR the worker already built per turn: imagined samples that had
     more than one completion to choose between. Under `bc_greedy` it is 0 by
     construction, which is the honest reading -- nothing was decided, rather
     than everything agreeing. */
  function deepenTotals(turns) {
    let decided = 0;
    let divergent = 0;
    (turns || []).forEach((t) => {
      decided += Number((t && t.completion_decided) || 0);
      divergent += Number((t && t.completion_divergent) || 0);
    });
    return {decided, divergent};
  }

  function deepenLineText(duel, totals) {
    const agent = (duel && duel.agent) || {};
    const policy = agent.completion_policy ? String(agent.completion_policy) : '';
    if (!policy) return 'deepening: unknown';
    const head = `deepening: ${policy} x${Number(agent.completion_width || 0)}`;
    if (!totals.decided) return `${head} \u00b7 no completion choice yet`;
    const pct = Math.round((totals.divergent / totals.decided) * 100);
    return `${head} \u00b7 V took the non-greedy completion `
      + `${totals.divergent} of ${totals.decided} (${pct}%)`;
  }

  function renderLog(turns) {
    const el = q('duel-log');
    if (!el) return;
    if (!turns.length) {
      el.innerHTML = '<span class="tiny">No turn has resolved yet.</span>';
      return;
    }
    // The cell wording lives in `search_telemetry.js`, shared with /replays,
    // so the two pages cannot start disagreeing about what these counts mean.
    // The markup stays here: this table carries `data-` attributes the browser
    // gate reads back against the server's own record, and the replay page's
    // does not.
    const telemetry = window.__SAP_SEARCH_TELEMETRY;
    const rows = turns.slice().reverse().map((t) => {
      const cells = telemetry.turnRowCells(t);
      return `<tr class="duel-log-row" data-turn="${t.turn}" data-outcome="${cells.outcome}">
        <td>${cells.turn}</td>
        <td class="verdict">${cells.outcome}</td>
        <td>${cells.lives}</td>
        <td class="searched" data-searched="${t.searched_segments}" data-total="${t.n_segments}" data-deadline="${t.deadline_segments || 0}">${cells.segments}</td>
        <td>${cells.aiTurn}</td>
        <td>${cells.endWait}</td>
        <td class="tiny">${cells.budget}</td>
        <td class="tiny widths">${cells.widths}</td>
      </tr>`;
    }).join('');
    el.innerHTML = `<table class="duel-log-table">
      <thead><tr><th>turn</th><th>battle</th><th>lives</th><th>segments</th><th>AI turn</th><th>End wait</th><th>budget</th><th>widths</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  /* ------------------------------------------------------------ end screen */
  function renderEnd(duel) {
    const overlay = q('duel-end');
    if (!overlay) return;
    if (!duel || !duel.done) {
      overlay.hidden = true;
      overlay.dataset.winner = '';
      return;
    }
    const winner = duel.winner || 'draw';
    const title = q('duel-end-title');
    if (title) {
      title.textContent = winner === 'human' ? 'You win' : winner === 'ai' ? 'The AI wins' : 'Draw';
    }
    const turns = Array.isArray(duel.turns) ? duel.turns : [];
    // The last turn that actually FOUGHT, which is not always the last one
    // recorded: a game can end ON a failed turn (two AI failures in a row),
    // and that record carries no board to put on the end screen.
    const fought = turns.filter(isResolvedTurn);
    const last = fought.length ? fought[fought.length - 1] : null;
    const sub = q('duel-end-sub');
    if (sub) {
      const human = duel.human || {};
      const ai = duel.ai || {};
      const reasons = {
        human_lives_0: 'you ran out of lives',
        ai_lives_0: 'the AI ran out of lives',
        both_lives_0: 'you both ran out of lives',
        turn_cap: 'the turn cap was reached',
        ai_failed: 'the AI failed two turns in a row',
      };
      const reason = reasons[String(duel.end_reason || '')] || String(duel.end_reason || '');
      sub.textContent =
        `${turns.length} turns played · lives ${human.lives ?? '-'} – ${ai.lives ?? '-'} `
        + `· battles won ${human.wins ?? 0} – ${ai.wins ?? 0}`
        + (reason ? ` · ${reason}` : '');
    }
    // The two boards of the end screen are the same two sides, so they get the
    // same treatment: yours faces right from the left column, the AI's faces
    // left from the right column.
    renderBoard(q('duel-end-human'), duel.final_human_board, SIDE_HUMAN);
    renderBoard(q('duel-end-ai'), last ? last.ai_board : [], SIDE_AI);
    const aiTitle = q('duel-end-ai-title');
    if (aiTitle && last) aiTitle.textContent = `The AI's board on turn ${last.turn}`;
    const replay = q('duel-end-replay');
    if (replay) {
      const gameId = duel.archive && duel.archive.game_id;
      replay.href = gameId ? `/replays?game=${encodeURIComponent(gameId)}` : '/replays';
    }
    overlay.hidden = false;
    overlay.dataset.winner = winner;
  }

  /* --------------------------------------------------------------- painting */
  function renderDuel() {
    const duel = duelBlock();
    const banner = q('duel-banner');
    if (!duel) {
      if (banner) banner.hidden = true;
      return;
    }
    const turns = Array.isArray(duel.turns) ? duel.turns : [];
    const last = turns.length ? turns[turns.length - 1] : null;

    const human = duel.human || {};
    const ai = duel.ai || {};
    const set = (id, value) => {
      const el = q(id);
      if (el) el.textContent = String(value);
    };
    set('duel-human-lives', human.lives ?? '-');
    set('duel-human-wins', human.wins ?? 0);
    set('duel-ai-lives', ai.lives ?? '-');
    set('duel-ai-wins', ai.wins ?? 0);
    set('duel-turn', duel.turn ? `Turn ${duel.turn}` : 'Turn -');

    const searchLine = q('ai-search-line');
    if (searchLine) {
      searchLine.textContent = searchLineText(last);
      if (last) {
        searchLine.dataset.turn = String(last.turn);
        searchLine.dataset.searched = String(last.searched_segments);
        searchLine.dataset.total = String(last.n_segments);
        searchLine.dataset.deadline = String(last.deadline_segments || 0);
      }
    }

    const deepenLine = q('ai-deepen-line');
    if (deepenLine) {
      const agent = duel.agent || {};
      const totals = deepenTotals(turns);
      deepenLine.textContent = deepenLineText(duel, totals);
      deepenLine.dataset.policy = String(agent.completion_policy ?? '');
      deepenLine.dataset.width = String(agent.completion_width ?? '');
      deepenLine.dataset.decided = String(totals.decided);
      deepenLine.dataset.divergent = String(totals.divergent);
    }

    // The loud failure line. `agent_error` is the "there is no AI at all" case
    // (no checkpoints, no torch); a per-turn banner is the "this turn failed"
    // case. Neither is ever swallowed into a weaker-but-quiet AI.
    if (banner) {
      const message = duel.agent_error
        ? `The AI could not be loaded: ${duel.agent_error}`
        : (last && last.banner) || null;
      banner.hidden = !message;
      banner.textContent = message || '';
    }

    const gear = duel.gear || {};
    paintGearReadout(gear);

    // The lane is deliberately NOT touched here. Its two transitions belong to
    // the two moments that mean something -- End turn pressed (pending) and a
    // turn resolved (revealed) -- and a repaint triggered by the human buying
    // a pet must not move it.

    paintRender(duel.render);
    renderLog(turns);
    renderValuePanel(duel);
    renderEnd(duel);

    const endBtn = q('btn-end-turn');
    if (endBtn && duel.done) endBtn.disabled = true;
    const rollBtn = q('btn-roll');
    if (rollBtn && duel.done) rollBtn.disabled = true;
  }
  window.__SAP_AFTER_RENDER = renderDuel;

  /* ------------------------------------ what the AI thinks of your board ----
     exp16 show-value. Three things happen here and only one of them is hard:

     1. the LIVE readout, refreshed off the same repaint every shop action
        already triggers, debounced;
     2. the PER-ACTION list, on an explicit press because each entry costs a
        greedy BC completion server-side;
     3. the per-turn predicted-versus-realised table, which is pure rendering of
        `duel.value.rows`.

     The hard one is that a number must never appear without the sentence that
     says what it means. `paintValue` is the ONLY place a number is written, it
     derives the number from the sentence rather than beside it, and the
     per-action rows go through the same function. DESIGN_show_value.md section
     3: the label is part of the feature, not decoration. */
  let valueBusy = false;
  let valueTimer = null;
  let valueEpoch = 0;
  let valueActionsOpen = false;
  // The board key the panel last asked the server about; see `valueBoardKey`.
  let valueBoardSeen = null;

  function fmtTrophies(value) {
    const n = Number(value);
    if (value === null || value === undefined || !isFinite(n)) return null;
    return n.toFixed(2);
  }

  function fmtRaw(value) {
    const n = Number(value);
    if (value === null || value === undefined || !isFinite(n)) return null;
    return n.toFixed(4);
  }

  /* The one place a valuation becomes text. Returns what it painted so the
     per-action list and the tests can use the identical derivation. */
  function valueText(row) {
    const usable = Boolean(
      row && row.ok !== false && row.meaning && VALUE_MEANING_KEYS.indexOf(row.meaning_key) >= 0,
    );
    const meaning = usable ? String(row.meaning) : '';
    const trophies = meaning ? fmtTrophies(row.trophies) : null;
    return {
      usable,
      meaning,
      // No sentence, no number. Both branches go through this expression, so
      // there is no path that paints one without the other.
      number: meaning && trophies !== null ? trophies : VALUE_PLACEHOLDER,
      units: meaning && trophies !== null ? 'trophies from here' : '',
      raw: meaning && fmtRaw(row.raw) !== null ? `raw ${fmtRaw(row.raw)} (${row.raw_scale || 'head output'})` : '',
      // `completed` marks a row that IS SHOWING a conditional number, which is
      // why it is gated on the sentence too: a row with nothing painted is not
      // showing a conditional number, it is showing nothing.
      completed: Boolean(meaning) && Boolean(row && row.completed),
      // Same gate as the number: no sentence, no short form either.
      shortMeaning: meaning ? (VALUE_SHORT_MEANINGS[row.meaning_key] || '') : '',
      flags: valueFlags(row, usable),
    };
  }

  function valueFlags(row, usable) {
    const flags = [];
    // `null` is "nothing has been asked yet" (a fresh game), which is not a
    // failure and must not read like one.
    if (row === null || row === undefined) return flags;
    if (!usable) {
      flags.push(
        row.error
          ? `no valuation: ${row.error}`
          : 'no valuation: the server did not say what the number would mean',
      );
      return flags;
    }
    if (row.trophies === null || row.trophies === undefined) {
      flags.push(
        'this head has no trophy scale, so only its raw output is shown '
        + `(${row.trophy_source || 'no curve'})`,
      );
    }
    if (row.clamped) {
      flags.push(
        `outside the range the recalibration curve was fitted on (clamped at the ${row.clamp_side} end) `
        + '-- a board built by a human is exactly where that happens',
      );
    }
    if (row.completion && row.completion.ok === false) {
      flags.push(`BC could not finish the turn cleanly: ${row.completion.stop_reason}`);
    }
    return flags;
  }

  function paintValue(row) {
    const painted = valueText(row);
    const number = q('duel-value-number');
    const units = q('duel-value-units');
    const raw = q('duel-value-raw');
    const meaning = q('duel-value-meaning');
    const flags = q('duel-value-flags');
    if (number) number.textContent = painted.number;
    if (units) units.textContent = painted.units;
    if (raw) raw.textContent = painted.raw;
    if (meaning) meaning.textContent = painted.meaning;
    if (flags) flags.textContent = painted.flags.join(' · ');
    const panel = q('duel-value-panel');
    if (panel) {
      panel.dataset.state = painted.usable ? 'ok' : 'unavailable';
      panel.dataset.completed = painted.completed ? '1' : '0';
    }
    return painted;
  }

  function paintProvenance(payload) {
    const el = q('duel-value-provenance');
    if (!el) return '';
    const provenance = (payload && payload.provenance) || {};
    const model = provenance.model || {};
    const scale = provenance.scale || {};
    const curve = scale.curve || null;
    const bits = [];
    if (model.agent_id) bits.push(String(model.agent_id));
    if (model.value_kind) bits.push(`head ${model.value_kind}`);
    if (curve && curve.path) {
      bits.push(`curve ${String(curve.path).split('/').pop()} @ ${String(curve.sha256 || '').slice(0, 12)}`);
    } else if (scale.source) {
      bits.push(`scale ${scale.source}`);
    }
    if (provenance.search && provenance.search.ran === false) bits.push('no search');
    if (payload && payload.elapsed_ms !== undefined) bits.push(`${payload.elapsed_ms} ms`);
    el.textContent = bits.join(' · ');
    el.title = provenance.search ? String(provenance.search.note || '') : '';
    return el.textContent;
  }

  function actionLabel(action) {
    if (!action || typeof action !== 'object') return '(action)';
    const type = String(action.type || '').toUpperCase();
    const parts = [type];
    ['shop_index', 'team_index', 'target_index', 'order'].forEach((key) => {
      if (action[key] !== undefined && action[key] !== null) {
        parts.push(`${key}=${JSON.stringify(action[key])}`);
      }
    });
    return parts.join(' ');
  }

  function paintActionList(rows) {
    const list = q('duel-value-list');
    if (!list) return 0;
    if (!rows || !rows.length) {
      list.innerHTML = '';
      return 0;
    }
    // Sorted best-first, but every row carries its own sentence: an ordered
    // list is exactly where "the top one is the best move" gets read into a
    // number that is not the value of the move.
    const painted = rows.map((row) => ({row, text: valueText(row)}));
    painted.sort((a, b) => {
      const av = a.text.number === VALUE_PLACEHOLDER ? -Infinity : Number(a.text.number);
      const bv = b.text.number === VALUE_PLACEHOLDER ? -Infinity : Number(b.text.number);
      return bv - av;
    });
    // The full sentence once, above the list, for the rows whose number is
    // conditional. Not instead of the per-row label: the two say the same thing
    // at two lengths, and a row can only show a number if it has both.
    const anyCompleted = painted.some((entry) => entry.text.completed);
    const header = anyCompleted
      ? `<div class="duel-value-list-note" id="duel-value-list-note">${escapeText(
        painted.find((entry) => entry.text.completed).text.meaning,
      )}</div>`
      : '';
    list.innerHTML = header + painted
      .map(
        (entry) => `<div class="duel-value-row" data-completed="${entry.text.completed ? '1' : '0'}"`
          + ` title="${escapeText(entry.text.meaning)}">`
          + `<span class="duel-value-row-number">${escapeText(entry.text.number)}</span>`
          + `<span class="duel-value-row-action">${escapeText(actionLabel(entry.row.action))}</span>`
          + `<span class="duel-value-row-meaning tiny">${escapeText(entry.text.shortMeaning)}</span>`
          + '</div>',
      )
      .join('');
    return painted.length;
  }

  function escapeText(value) {
    return String(value === undefined || value === null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function renderValueLog(rows) {
    const el = q('duel-value-log');
    if (!el) return;
    if (!rows || !rows.length) {
      el.innerHTML = '<span class="tiny">No turn has resolved yet.</span>';
      return;
    }
    el.innerHTML = rows
      .map((row) => {
        const you = fmtTrophies(row.human_predicted_trophies);
        const ai = fmtTrophies(row.ai_predicted_trophies);
        const got = row.human_realised_trophies === null || row.human_realised_trophies === undefined
          ? null
          : String(row.human_realised_trophies);
        // A clamped row is pinned to the top of the curve's fitted range, so
        // 5.00 there does NOT mean "five" -- it means "at least the best board
        // in the fit". Marked where it happens, because it happens most on the
        // AI's own boards, which is exactly where a reader would otherwise take
        // the number at face value.
        const youMark = row.human_clamped ? '≥' : '';
        const aiMark = row.ai_clamped ? '≥' : '';
        return `<div class="duel-value-logrow" data-clamped="${row.human_clamped || row.ai_clamped ? '1' : '0'}">`
          + `<span class="duel-value-logturn">T${escapeText(row.turn)}</span>`
          + `<span>you ${escapeText(youMark)}${escapeText(you === null ? VALUE_PLACEHOLDER : you)}</span>`
          + `<span>AI ${escapeText(aiMark)}${escapeText(ai === null ? VALUE_PLACEHOLDER : ai)}</span>`
          + `<span class="duel-value-logoutcome">${escapeText(row.outcome || '-')}</span>`
          + `<span class="tiny">${got === null ? 'realised: at game end' : `realised ${escapeText(got)}`}</span>`
          + (row.human_clamped || row.ai_clamped
            ? '<span class="tiny duel-value-clamp">≥ = off the top of the curve’s fitted range</span>'
            : '')
          + (row.error ? `<span class="tiny">${escapeText(row.error)}</span>` : '')
          + '</div>';
      })
      .join('');
  }

  function renderValuePanel(duel) {
    const panel = q('duel-value-panel');
    const value = (duel && duel.value) || null;
    if (!panel) return;
    if (!value || !value.enabled) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;
    renderValueLog(value.rows || []);
    if (value.error) {
      paintValue({ok: false, error: value.error});
      return;
    }
    // The probe values THE CURRENT BOARD, so it is worth re-running exactly
    // when the board has moved. This function runs on every repaint, and the
    // status poll repaints, so an unconditional refresh here meant a probe
    // every second of a turn nobody was touching: 19 to 27 `POST
    // /api/infer/value` a minute in minutes with zero human actions
    // (2026-08-13 `dev.log`). Cheap on the server at ~13 ms, but each one is a
    // request the human's own click then queues behind.
    const key = valueBoardKey();
    if (key !== null && key === valueBoardSeen) return;
    valueBoardSeen = key;
    scheduleValueRefresh();
  }

  /* What "the board has moved" means, in the state the page already holds.
     `history_len` counts applied actions within the turn, `history_token`
     changes when the server's log is replaced (New game, undo past the base),
     and `turn` covers the End turn that resets the first two. */
  function valueBoardKey() {
    const state = (typeof appState !== 'undefined' && appState) || null;
    if (!state) return null;
    const duel = state.duel || {};
    return [
      String(state.history_token || ''),
      Number(state.history_len ?? -1),
      Number(duel.turn ?? -1),
      duel.done ? 1 : 0,
      gameEpoch,
    ].join('|');
  }

  function scheduleValueRefresh() {
    if (valueTimer) clearTimeout(valueTimer);
    valueTimer = setTimeout(() => {
      valueTimer = null;
      refreshValue();
    }, VALUE_DEBOUNCE_MS);
  }

  async function requestValue(body) {
    const res = await fetch('/api/infer/value', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify(body || {}),
    });
    return res.json();
  }

  async function refreshValue() {
    const duel = duelBlock();
    if (!duel || !duel.value || !duel.value.enabled || !duel.active || duel.done) return;
    if (valueBusy) {
      // A probe for an older board is still out. Forget the key so the next
      // repaint schedules this one again, rather than dropping it silently.
      valueBoardSeen = null;
      return;
    }
    valueBusy = true;
    const epoch = gameEpoch;
    const seq = ++valueEpoch;
    try {
      const payload = await requestValue({});
      // A response for a board the human has already changed, or for a game
      // that has been replaced, must not paint: the number would be labelled
      // with the right sentence and be about the wrong board.
      if (epoch !== gameEpoch || seq !== valueEpoch) return;
      if (!payload || payload.ok === false) {
        paintValue({ok: false, error: (payload && payload.error) || 'value_request_failed'});
        paintProvenance(null);
        return;
      }
      paintValue(payload.value);
      paintProvenance(payload);
      if (valueActionsOpen) paintActionsNote(payload);
    } catch (err) {
      // A dropped request, not a refused one: the board is still worth asking
      // about, so let the next repaint try again.
      valueBoardSeen = null;
      paintValue({ok: false, error: 'value_request_failed'});
    } finally {
      valueBusy = false;
    }
  }

  function paintActionsNote(payload) {
    const note = q('duel-value-actions-note');
    if (!note) return;
    const bits = [];
    if (payload && payload.n_legal_actions) {
      bits.push(`${(payload.actions || []).length} of ${payload.n_legal_actions} legal moves`);
    }
    if (payload && payload.actions_capped) bits.push('capped');
    if (payload && payload.ai_worker_status === 'thinking') {
      bits.push('the AI is thinking; scoring your moves borrows some of its clock');
    }
    note.textContent = bits.join(' · ');
  }

  async function valueEachMove() {
    const btn = q('duel-value-actions-btn');
    const note = q('duel-value-actions-note');
    const duel = duelBlock();
    if (!duel || !duel.value || !duel.value.enabled || !duel.active || duel.done) return;
    if (btn) btn.disabled = true;
    if (note) note.textContent = 'scoring every legal move...';
    const epoch = gameEpoch;
    try {
      const payload = await requestValue({actions: true});
      if (epoch !== gameEpoch) return;
      if (!payload || payload.ok === false) {
        if (note) note.textContent = (payload && payload.error) || 'value_request_failed';
        return;
      }
      valueActionsOpen = true;
      paintValue(payload.value);
      paintProvenance(payload);
      paintActionList(payload.actions || []);
      paintActionsNote(payload);
    } catch (err) {
      if (note) note.textContent = 'value_request_failed';
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  /* ---------------------------------------------------------------- polling */
  /* Does the badge have a moving number to fetch right now? Only the AI's
     clock and the battle's reveal produce one. `endingTurn` is checked first
     because it opens before any sample about the new turn has arrived. */
  function pollIsActive() {
    if (endingTurn) return true;
    if (paintedStatus === 'thinking' || paintedStatus === 'resolving') return true;
    const status = String((lastWorker && lastWorker.status) || '');
    return status === 'thinking' || status === 'resolving';
  }

  /* Reinstall the interval only when the cadence really changes, so the usual
     poll costs no timer churn. Keeping `setInterval` as the tick source is
     deliberate: it is the seam `test_exp16_duel_badge.py` captures the poll
     through, so the shipped path stays the tested path. */
  function installPoll(period) {
    if (pollPeriod === period) return;
    if (pollTimer !== null) clearInterval(pollTimer);
    pollPeriod = period;
    pollTimer = setInterval(pollStatus, period);
  }

  async function pollStatus() {
    // Single-flight. Without it a slow link stacks polls faster than they come
    // back, and that pile is what leaves the human's own click with no free
    // connection to go out on.
    if (pollInFlight) return;
    pollInFlight = true;
    const epoch = gameEpoch;
    try {
      const res = await fetch('/api/duel/status');
      const payload = await res.json();
      // A status request issued before New game may finish afterwards. It
      // belongs to the old game and must not repaint that game's render card.
      if (epoch !== gameEpoch || (busy && !endingTurn)) return;
      // BEFORE the fence, because the fence's silence is the defect: it drops
      // exactly the responses that say the process was replaced.
      if (noticedRestartedServer(payload)) return;
      if (!acceptsGameGeneration(payload)) return;
      const worker = payload.worker || {};
      const fresh = acceptWorkerSample(worker);
      if (fresh) lastWorker = worker;
      if (!acceptsRenderGeneration(payload.render)) return;
      paintRender(payload.render);
      // A stale or worker-less response still carries a render block worth
      // painting; it just does not get to say anything about the AI.
      if (!fresh) return;
      const status = badgeStatusNow(worker);
      if (status === null) return;
      paintBadge(status, worker);
      // Attributed to the turn the badge is ABOUT, which during an End turn is
      // the turn being ended even after the server has moved on.
      const observedTurn = (endingTurn && endingTurnNumber !== null)
        ? endingTurnNumber
        : payload.turn;
      if (observedTurn) observe(observedTurn, status, lanePhase, worker);
    } catch (err) {
      /* a dropped poll is not an error worth a banner: another one is along
         shortly and every number the page SHOWS comes from the state payload. */
    } finally {
      pollInFlight = false;
      installPoll(pollIsActive() ? POLL_ACTIVE_MS : POLL_IDLE_MS);
    }
  }

  function startPolling() {
    installPoll(POLL_ACTIVE_MS);
  }

  /* ----------------------------------------------------------- the actions */
  async function post(path, body) {
    try {
      const res = await fetch(path, {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        // `withHistoryCursor` comes from app.js, which the duel page loads first.
        // Without it these two routes -- new game and end turn -- would be the
        // only ones still answering with the whole session log every time.
        body: JSON.stringify(withHistoryCursor(body)),
      });
      return await res.json();
    } catch (err) {
      /* Same envelope `apiPost` uses, from the same helper in app.js, because
         the three callers here already read `payload.state` and `payload.ok`
         and would otherwise be reading an exception thrown past them. `endTurn`
         wraps this in `try { } finally { }` with no `catch`: the finally puts
         the button back, so the page LOOKS ready while the turn silently did
         not happen. */
      return transportFailure(err);
    }
  }

  async function newGame(settings) {
    if (busy) return;
    busy = true;
    gameEpoch += 1;
    renderGeneration = null;
    serverGameGeneration = null;
    // The one action that works against a process this page has never met.
    serverRestarted = false;
    paintStaleServer(false);
    try {
      // A new game is the one moment the lane goes back to having nothing to
      // show, so it is the one caller of `unseen`.
      setLane('unseen', null);
      paintRender({status: 'idle'});
      endingTurn = false;
      resolvingSince = null;
      endingTurnNumber = null;
      lastWorker = {status: 'idle'};
      // F3: the stream is keyed by the bare turn number, so without this a
      // second game's turn 3 appends to the first game's turn 3 and anything
      // reading "the first resolving pair of turn 3" reads the PREVIOUS game.
      // On the two-game gate run that was 7 of the second game's 11 turns.
      observations.byTurn = {};
      // Same reason, for the valuation: the panel's per-action list and its
      // headline are about a board that no longer exists once a new game
      // starts, and a stale number under a correct sentence is still wrong.
      valueActionsOpen = false;
      if (valueTimer) {
        clearTimeout(valueTimer);
        valueTimer = null;
      }
      valueEpoch += 1;
      paintActionList([]);
      paintValue(null);
      paintProvenance(null);
      const valueNote = q('duel-value-actions-note');
      if (valueNote) valueNote.textContent = '';
      // A restarted server hands out generations from zero again; the fence is
      // only ever meant to reject responses from THIS game that overtook each
      // other.
      newestWorkerGen = -1;
      paintBadge('idle', lastWorker);
      const curtain = q('ai-curtain-text');
      if (curtain) curtain.textContent = 'Loading the agent and dealing the boards…';
      // Amendment 6: the two knobs travel WITH the new game, because they are
      // properties of the game rather than of the process. An empty object is
      // still valid and means "whatever the server already has", which is what
      // a reload or an old client sends.
      const payload = await post('/api/duel/new_game', settings || {});
      if (payload.state) {
        const duelState = payload.state.duel || {};
        pinGameGeneration(duelState);
        pinRenderGeneration(duelState.render);
        setAppState(payload.state, {resetFx: true});
        renderState();
      }
      if (curtain) {
        curtain.textContent = payload.ok
          ? 'You have not seen the AI’s board yet. It appears when the first battle resolves, and stays.'
          : String(payload.error || 'could not start a game');
      }
      const endBtn = q('btn-end-turn');
      if (endBtn && payload.ok) endBtn.disabled = false;
    } finally {
      busy = false;
    }
  }

  async function endTurn() {
    if (busy) return;
    busy = true;
    endingTurn = true;
    resolvingSince = null;
    const openTurn = (duelBlock() || {}).turn;
    endingTurnNumber = openTurn || null;
    // Settle the lane BEFORE the request: from the moment End turn is pressed
    // the result on screen is no longer this turn's news. The board itself
    // stays up -- it is the PREVIOUS turn's and hides nothing about this one.
    // Before the first battle there is nothing up, and `setLane` turns this
    // into `unseen` by itself.
    setLane('last-seen', null);
    // The badge keeps saying what the AI is actually doing. If the AI already
    // handed its turn over this latches straight into `resolving`; if it is
    // still thinking, the human goes on watching it think, because that is
    // what the wait IS.
    // `badgeStatusNow` answers null when the last sample says nothing about
    // this turn (a game just started, a response arrived with no worker block).
    // Then the badge keeps what it is showing rather than inventing a state.
    const pressStatus = badgeStatusNow(lastWorker) || paintedStatus;
    paintBadge(pressStatus, lastWorker);
    // Recorded here rather than left to the 250 ms poll: the resolving window
    // is often shorter than one poll, and what the badge said at the press is
    // a claim the gate checks.
    if (openTurn) {
      const row = observe(openTurn, pressStatus, lanePhase, lastWorker);
      const badgeTextEl = q('ai-badge-text');
      row.atEndTurn = {
        badge: pressStatus,
        worker: String((lastWorker && lastWorker.status) || ''),
        text: badgeTextEl ? String(badgeTextEl.textContent || '') : '',
      };
    }
    const endBtn = q('btn-end-turn');
    if (endBtn) endBtn.disabled = true;
    try {
      const payload = await post('/api/duel/end_turn', {});
      if (payload.state) {
        setAppState(payload.state, {resetFx: true});
      }
      const duel = duelBlock();
      const turns = (duel && Array.isArray(duel.turns) ? duel.turns : []);
      const last = turns.length ? turns[turns.length - 1] : null;
      const resolvedNow = isResolvedTurn(last);
      // THE RESPONSE IS THE AUTHORITATIVE TRANSITION, not a lucky poll.
      //
      // Everything between the AI handing its turn over and this response is
      // server-side (`end_turn`: await_turn -> battle -> bump_generation ->
      // archive sync -> `_submit_ai_turn`), and it is regularly shorter than
      // one 250 ms poll. When no poll landed in it, the old code left
      // `resolvingSince` null and the turn simply never had a resolving phase:
      // 1 of 33 turns across the two saved gate runs, worst on turn 1 where the
      // archive document is smallest. A response that carries a resolved turn
      // is proof the battle happened, so the state machine closes here instead
      // of hoping. The worker in the observation is named `handover` rather
      // than whatever the AI is doing now, because the transition came from the
      // response and not from a status sample.
      if (resolvedNow && resolvingSince === null) {
        resolvingSince = now();
        noteHandover('response', HANDOVER_SAMPLE);
        paintBadge('resolving', HANDOVER_SAMPLE);
        if (openTurn) observe(openTurn, 'resolving', lanePhase, HANDOVER_SAMPLE);
      }
      endingTurn = false;
      resolvingSince = null;
      endingTurnNumber = null;
      if (resolvedNow) {
        setLane('revealed', last);
        observe(last.turn, 'resolved', 'revealed');
      } else if (last) {
        /* The step failed, so no battle was fought this turn. The lane was
           settled to `last-seen` before the request and stays exactly there:
           the board on it is still the last one that DID fight, and
           `data-seen-turn` still names that turn. The failure reaches the
           human through the banner and the message line below. */
        observe(last.turn, 'failed', lanePhase);
      }
      renderState();
      if (!payload.ok && payload.error) {
        setMessage(String(payload.error), false);
      }
    } finally {
      endingTurn = false;
      resolvingSince = null;
      endingTurnNumber = null;
      busy = false;
      const duel = duelBlock();
      if (endBtn) endBtn.disabled = Boolean(duel && duel.done);
    }
  }
  window.__SAP_END_TURN = endTurn;

  /* ------------------------------------------------------- settings card */
  /* exp16 Amendment 6. Both knobs are set here, once, before a game, and the
     duel bar only reads them back. The alternative -- changing them mid-game,
     which is what the bar used to allow for the clock -- cannot be filed:
     `ai_version` records ONE setting per game, so a game that changed its own
     would be archived under a number half of it never ran. */

  function deepenText(gear) {
    const width = Number((gear || {}).deepen_width || 0);
    return width >= 2 ? `${width}` : 'off';
  }

  function paintGearReadout(gear) {
    const clock = q('duel-clock-readout');
    if (clock) clock.textContent = String(Number((gear || {}).turn_budget_s || 0));
    const deepen = q('duel-deepen-readout');
    if (deepen) deepen.textContent = deepenText(gear);
    const widthOut = q('duel-width-readout');
    if (widthOut) {
      const g = String((gear || {}).gear || '');
      const w = Number((gear || {}).width || 0);
      const k = (gear || {}).stochastic_samples;
      const initial = k === undefined || k === null ? 'not recorded' : String(k);
      if (g === 'resample-clock') widthOut.textContent = `root ${w}, grow k from ${initial}`;
      else if (g === 'measured') widthOut.textContent = `fixed root ${w}, k ${initial}`;
      else widthOut.textContent = 'laboratory: grow root width';
    }
  }

  /* Gears that PIN the outer width, so the width input applies to them.
     W11a had only `measured`; W11b's `resample-clock` pins it the same way and
     spends the remainder on k instead. Under `full-clock` the number is
     generated until the slice ends and the input would be a control that does
     nothing. Dimming beats hiding: the value is still visible, so switching
     gears does not look like it lost a setting. */
  const PINNED_WIDTH_GEARS = ['measured', 'resample-clock'];
  /* Mirrors `ai_worker.MAX_SETTABLE_SEARCH_WIDTH` and the input's own `max`. */
  const MAX_SEARCH_WIDTH = 4096;

  function syncWidthRow() {
    const row = q('setup-width-row');
    const gearSel = q('setup-gear');
    if (!row || !gearSel) return;
    const on = PINNED_WIDTH_GEARS.indexOf(String(gearSel.value)) >= 0;
    row.style.opacity = on ? '1' : '0.45';
    const width = q('setup-width');
    if (width) width.disabled = !on;
  }

  function setupError(message) {
    const box = q('duel-setup-error');
    if (!box) return;
    box.hidden = !message;
    box.textContent = message || '';
  }

  function openSetup(source) {
    const overlay = q('duel-setup');
    if (!overlay) return;
    // `source` is the status payload at boot, when the page has no app state
    // yet. Falling back to the HTML defaults instead would be the quiet
    // version of the bug this card exists to remove: a server launched with
    // DEEPEN=8 would offer 0, and Start would turn deepening off without ever
    // saying so.
    const duel = source || duelBlock() || {};
    const gear = duel.gear || {};
    const deepen = q('setup-deepen');
    if (deepen) deepen.value = String(Number(gear.deepen_width || 0));
    const budget = q('setup-budget');
    if (budget && gear.turn_budget_s) budget.value = String(Number(gear.turn_budget_s));
    const gearSel = q('setup-gear');
    if (gearSel && gear.gear) gearSel.value = String(gear.gear);
    const width = q('setup-width');
    if (width && gear.width) width.value = String(Number(gear.width));
    syncWidthRow();
    // Cancel exists only when there is a game to go back TO. On arrival at
    // /play with nothing running, a card you can dismiss leaves the human on
    // an empty board with no way back to it.
    const cancel = q('duel-setup-cancel');
    if (cancel) cancel.hidden = !Number(duel.game_generation);
    setupError(null);
    overlay.hidden = false;
    if (deepen) deepen.focus();
  }

  function closeSetup() {
    const overlay = q('duel-setup');
    if (overlay) overlay.hidden = true;
  }

  async function startFromSetup() {
    const deepen = q('setup-deepen');
    const budget = q('setup-budget');
    const gearSel = q('setup-gear');
    const widthEl = q('setup-width');
    const width = deepen ? Number(deepen.value) : NaN;
    const seconds = budget ? Number(budget.value) : NaN;
    const gearValue = gearSel ? String(gearSel.value) : null;
    const searchWidth = widthEl ? Number(widthEl.value) : NaN;
    // Refused here rather than sent, because `Number("abc")` is NaN and JSON
    // carries it as null, which the server cannot tell apart from "not sent".
    if (!Number.isFinite(width) || width < 0 || !Number.isInteger(width)) {
      setupError('Deepening width has to be a whole number, 0 or more. 0 turns it off.');
      return;
    }
    if (!Number.isFinite(seconds) || seconds <= 0) {
      setupError('The AI turn clock has to be a positive number of seconds.');
      return;
    }
    if (PINNED_WIDTH_GEARS.indexOf(String(gearValue)) >= 0
        && (!Number.isFinite(searchWidth) || searchWidth < 1
            || !Number.isInteger(searchWidth) || searchWidth > MAX_SEARCH_WIDTH)) {
      /* The ceiling matters as much as the floor under a pinned-width gear:
         the clock waits for the width, so an unbounded width is an unbounded
         turn. Same number as the input's `max` and as the server's own limit. */
      setupError(`Chains per segment has to be a whole number between 1 and ${MAX_SEARCH_WIDTH}.`);
      return;
    }
    setupError(null);
    closeSetup();
    const settings = {deepen_width: width, turn_budget_s: seconds};
    if (gearValue) settings.gear = gearValue;
    // Only sent under a gear that uses it, so a stale number left in the box
    // under `full-clock` cannot quietly become next game's width.
    if (PINNED_WIDTH_GEARS.indexOf(String(gearValue)) >= 0) settings.search_width = searchWidth;
    await newGame(settings);
    const payloadOk = Number((duelBlock() || {}).game_generation) > 0;
    if (!payloadOk) {
      openSetup();
      setupError('The server refused those settings, so nothing was started.');
    }
  }

  /* ------------------------------------------------------------------ boot */
  function boot() {
    const newBtn = q('duel-new-game');
    if (newBtn) newBtn.addEventListener('click', () => openSetup());
    const again = q('duel-end-again');
    if (again) again.addEventListener('click', () => openSetup());
    const start = q('duel-setup-start');
    if (start) start.addEventListener('click', () => startFromSetup());
    const gearSel = q('setup-gear');
    if (gearSel) gearSel.addEventListener('change', () => syncWidthRow());
    const cancel = q('duel-setup-cancel');
    if (cancel) cancel.addEventListener('click', () => closeSetup());
    const renderImage = q('duel-render-image');
    if (renderImage) {
      renderImage.addEventListener('error', () => markImageUnavailable('image_load_failed'));
    }
    const valueBtn = q('duel-value-actions-btn');
    if (valueBtn) valueBtn.addEventListener('click', () => valueEachMove());

    setLane('unseen', null);
    startPolling();
    bootSetup();
  }

  /* Arriving at /play opens the settings card instead of dealing a game.
     Until Amendment 6 this called `newGame()` directly, which had a second
     effect nobody asked for: a reload silently threw away the game in
     progress. Now the card offers to keep it. */
  async function bootSetup() {
    let status = null;
    try {
      const res = await fetch('/api/duel/status');
      status = await res.json();
    } catch (err) {
      // The card still opens; it just shows the markup's defaults, and Start
      // will fail loudly against a server that is not answering anyway.
      status = null;
    }
    if (status && status.gear) paintGearReadout(status.gear);
    openSetup(status);
  }

  /* The fourth seam, alongside `__SAP_DUEL_OBS`, `__SAP_END_TURN` and
     `__SAP_AFTER_RENDER`: the board renderer and the lane's one transition
     function, so `test_exp16_duel_board.py` can check the two sides and the
     three states in node against THIS file, without a browser.

     The badge deliberately has NO seam of its own: `test_exp16_duel_badge.py`
     drives it through `window.__SAP_END_TURN` and the same poll the page
     installs with `setInterval`, so it exercises the shipped path rather
     than a testing entry point beside it. (The cadence that `setInterval` is
     given now depends on whether the AI's clock is running -- see
     `POLL_ACTIVE_MS` -- but the seam is unchanged: it is still the function
     handed to `setInterval`.) */
  window.__SAP_DUEL_INTERNALS = {
    renderBoard,
    boardCardHTML,
    paintRender,
    pinRenderGeneration,
    acceptsRenderGeneration,
    pinGameGeneration,
    acceptsGameGeneration,
    noticedRestartedServer,
    paintStaleServer,
    restarted: () => serverRestarted,
    deepenTotals,
    deepenLineText,
    searchLineText,
    deepenText,
    openSetup,
    startFromSetup,
    markImageUnavailable,
    setLane,
    isResolvedTurn,
    sides: {ai: SIDE_AI, human: SIDE_HUMAN},
    phase: () => lanePhase,
    seenTurn: () => laneTurn,
  };

  /* exp16 show-value's seam. `valueText` is the derivation the whole panel goes
     through -- headline and every per-action row -- so a node probe that drives
     THIS function is driving what the page paints, not a copy of it. */
  window.__SAP_DUEL_VALUE_INTERNALS = {
    valueText,
    valueFlags,
    paintValue,
    paintActionList,
    paintProvenance,
    renderValueLog,
    renderValuePanel,
    actionLabel,
    placeholder: VALUE_PLACEHOLDER,
    meaningKeys: VALUE_MEANING_KEYS.slice(),
    shortMeanings: Object.assign({}, VALUE_SHORT_MEANINGS),
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
