(function () {
  'use strict';

  const q = (id) => document.getElementById(id);
  const esc = (value) => String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  let games = [];
  let selectedId = null;
  let detailRequest = 0;
  let refreshRequest = 0;
  let initialized = false;
  // exp16 replay UI (2026-08-20). The human's actions are the ones the reader
  // just played, so they start HIDDEN and the AI's start shown; only the initial
  // values move, the toggle mechanism below is untouched. `turnNumbers` backs
  // the jump control and is rebuilt on every painted game.
  let humanActionsHidden = true;
  let aiActionsHidden = false;
  let turnNumbers = [];
  let currentTurnIndex = 0;
  // exp16 show-value, replayed. Same contract as the live readout in duel.js:
  // the page refuses to draw a number under a meaning key it does not know, so
  // a server that grows a third wording cannot ship a number with no sentence
  // under it. Keys and placeholder are duplicated from duel.js deliberately --
  // the two pages are separate bundles -- and `check_replays_value.py` asserts
  // they still agree with the server's own `meaning_for`.
  const VALUE_PLACEHOLDER = '--';
  const VALUE_MEANING_KEYS = ['end_of_turn', 'bc_completed'];
  const VALUE_SHORT_MEANINGS = {
    end_of_turn: 'the board as it stood at end of turn',
    bc_completed: 'NOT this move’s value: take it, then BC finishes the turn',
  };
  let valueReadout = null;          // the whole-game envelope
  let valueByTurn = new Map();      // turn number -> its row

  function valueUsable() {
    return Boolean(
      valueReadout
      && valueReadout.available
      && valueReadout.meaning
      && VALUE_MEANING_KEYS.indexOf(valueReadout.meaning_key) >= 0
    );
  }

  function fmtTrophies(value) {
    return typeof value === 'number' && isFinite(value) ? value.toFixed(2) : null;
  }

  function valueHeaderHTML() {
    if (!valueReadout) return '';
    if (!valueUsable()) {
      // Absence is the ordinary case for anything played before show-value
      // shipped, and saying so beats an empty row that reads as "V had nothing
      // to say" when the truth is that V was not watching.
      const reason = valueReadout.reason === 'value_log_absent'
        ? 'This game was played before the value readout existed, and nothing back-fills it.'
        : `No value readout for this game (${esc(valueReadout.reason || 'unavailable')}).`;
      return `<p class="replay-value-absent">${esc(reason)}</p>`;
    }
    return `<div class="replay-value-legend">
      <b>What V thought, as the game was played</b>
      <p>${esc(valueReadout.meaning)}</p>
      <p class="replay-muted">Recorded at each end of turn while the game ran, not recomputed now.
      Predicted is on the trophy scale; realised is the trophies actually taken from that turn onward.</p>
    </div>`;
  }

  function valueRowHTML(turnNumber) {
    if (!valueUsable()) return '';
    const row = valueByTurn.get(Number(turnNumber));
    if (!row) return '';
    const human = fmtTrophies(row.human_predicted_trophies);
    const ai = fmtTrophies(row.ai_predicted_trophies);
    const realised = typeof row.human_realised_trophies === 'number'
      ? String(row.human_realised_trophies)
      : VALUE_PLACEHOLDER;
    const short = VALUE_SHORT_MEANINGS[valueReadout.meaning_key] || '';
    const clamp = (flag) => (flag ? '<span class="replay-value-clamp" title="at the curve ceiling">≥</span>' : '');
    return `<div class="replay-value" data-value-turn="${Number(turnNumber)}" title="${esc(valueReadout.meaning)}">
      <span class="replay-value-side" data-value-side="human"><b>You</b> ${clamp(row.human_clamped)}${esc(human === null ? VALUE_PLACEHOLDER : human)}</span>
      <span class="replay-value-side" data-value-side="ai"><b>AI</b> ${clamp(row.ai_clamped)}${esc(ai === null ? VALUE_PLACEHOLDER : ai)}</span>
      <span class="replay-value-actual"><b>battle</b> ${esc(row.outcome || 'unknown')}</span>
      <span class="replay-value-actual"><b>you actually took</b> ${esc(realised)}</span>
      <span class="replay-value-meaning tiny">${esc(short)}</span>
      ${row.error ? `<span class="replay-value-error">${esc(row.error)}</span>` : ''}
    </div>`;
  }
  // Turn number -> promise of that turn's record WITH the detail the list
  // response leaves out. One entry per turn whose telemetry or raw transition
  // someone has actually opened; cleared whenever a different game is painted.
  const turnDetail = new Map();

  function loadTurnDetail(gameId, turnNumber) {
    const key = Number(turnNumber);
    if (!turnDetail.has(key)) {
      const pending = fetch(`/api/duel/replay_turn?id=${encodeURIComponent(gameId)}&turn=${key}`)
        .then((res) => res.json())
        .then((payload) => {
          if (!payload.ok) throw new Error(payload.error || 'could not load turn detail');
          return payload.turn;
        })
        .catch((error) => {
          turnDetail.delete(key);   // a failed fetch must not be cached as the answer
          throw error;
        });
      turnDetail.set(key, pending);
    }
    return turnDetail.get(key);
  }

  /* exp16 Amendment 6. What the AI searched, per turn, readable without
     opening anything -- the same thing the live page has always shown in its
     own log, worded by the shared `search_telemetry.js` so the two cannot
     drift. The raw JSON fold below it is unchanged and still carries
     everything; this is the part the PLAN's replay contract calls "the main
     timeline", which a whole JSON block is not allowed to drown. */
  function searchRowHTML(turn) {
    const telemetry = window.__SAP_SEARCH_TELEMETRY;
    if (!telemetry) return '';
    const cells = telemetry.turnRowCells(turn);
    const pairs = [
      ['segments searched', cells.segments],
      ['width per segment', cells.widths || 'not recorded'],
      ['AI turn', cells.aiTurn],
      ['clock', cells.budget],
      ['deepening picked', cells.deepening],
      ['k samples', cells.samples],
    ];
    return `<div class="replay-search-row">
      ${pairs.map(([label, value]) =>
        `<span><b>${esc(label)}</b> ${esc(value)}</span>`).join('')}
    </div>`;
  }

  /* What this GAME was set to, from `ai_version`. Games archived before
     2026-08-19 carry no completion fields, and the PLAN's replay contract is
     explicit about that case: say the archive does not have it. Deriving it
     from `completion_decided > 0` would recover whether it was on but never
     the width, and half an answer sitting in a number's place is what gets
     quoted later as the whole one. */
  /* W11b. Which gear this GAME was played on, from `gear_start`, which
     `DuelApp` writes once when the game begins. Absent on anything archived
     before the settings card existed, and said so rather than defaulted: the
     process default at the time is not evidence of what that game ran. */
  function gameGearText(game) {
    const start = (game || {}).gear_start;
    if (!start || !start.gear) return 'unavailable from v1 archive';
    const gear = String(start.gear);
    const width = Number(start.width || 0);
    if (gear === 'resample-clock') return `${gear} · width ${width} · rest to rerolls`;
    if (gear === 'measured') return `${gear} · width ${width}`;
    return `${gear} · width floats`;
  }

  function gameDeepeningText(game) {
    const telemetry = window.__SAP_SEARCH_TELEMETRY;
    const ai = (game || {}).ai_version || {};
    const label = telemetry
      ? telemetry.deepeningLabel(ai.completion_policy, ai.completion_width)
      : null;
    return label === null ? 'unavailable from v1 archive' : label;
  }

  function telemetryOf(turn) {
    return {
      gear: turn.gear,
      turn_budget_s: turn.turn_budget_s,
      adaptive_chunks: turn.adaptive_chunks,
      segments: turn.segments,
      searched_segments: turn.searched_segments,
      deadline_segments: turn.deadline_segments,
      n_segments: turn.n_segments,
      search_used: turn.search_used,
      search_error: turn.search_error,
      // Amendment 6. These were missing, so the one place a replay could have
      // said anything about the deepening said nothing about it.
      completion_decided: turn.completion_decided,
      completion_divergent: turn.completion_divergent,
      // W11b. `realised_stochastic_samples` is the gear's treatment-took
      // reading, and the two counters below are the instruments for silent
      // loss inside a completion, so a replay can answer both without the
      // reader having to have been watching live.
      realised_stochastic_samples: turn.realised_stochastic_samples,
      extra_sample_levels: turn.extra_sample_levels,
      completion_dropped: turn.completion_dropped,
      completion_second_chance_nodes: turn.completion_second_chance_nodes,
      greedy_segments: turn.greedy_segments,
      chunk_size: turn.chunk_size,
      finish_turn: turn.finish_turn,
      dedup_ratio: turn.dedup_ratio,
      segments_capped: turn.segments_capped,
      stop_latency_ms: turn.stop_latency_ms,
      elapsed_ms: turn.elapsed_ms,
      end_wait_ms: turn.end_wait_ms,
      budget_utilization: turn.budget_utilization,
      deadline_overshoot_ms: turn.deadline_overshoot_ms,
      effective_unique_prefixes: turn.effective_unique_prefixes,
      leaves_evaluated: turn.leaves_evaluated,
      safety_cap_hit: turn.safety_cap_hit,
      end_to_result_ms: turn.end_to_result_ms,
      lives: turn.lives,
      wins: turn.wins,
      battle: turn.battle,
      timing_ms: turn.timing_ms,
    };
  }

  function transitionOf(turn, sideKey, eventIndex) {
    const chain = ((turn[sideKey] || {}).replay || {}).chain || [];
    const event = chain[eventIndex];
    return event ? event.transition : null;
  }

  function syncHumanActionsVisibility() {
    document.body.classList.toggle('is-human-actions-hidden', humanActionsHidden);
    const button = q('replay-toggle-human');
    if (!button) return;
    button.textContent = humanActionsHidden ? 'Show Human actions' : 'Hide Human actions';
    button.setAttribute('aria-pressed', humanActionsHidden ? 'true' : 'false');
  }

  function syncAiActionsVisibility() {
    document.body.classList.toggle('is-ai-actions-hidden', aiActionsHidden);
    const button = q('replay-toggle-ai');
    if (!button) return;
    button.textContent = aiActionsHidden ? 'Show AI actions' : 'Hide AI actions';
    button.setAttribute('aria-pressed', aiActionsHidden ? 'true' : 'false');
  }

  // One binder for both sides: the buttons live in `replay-meta`, which is
  // rewritten whenever another game is painted, so both have to be rebound and
  // both re-synchronized to the state the reader left them in.
  function bindActionToggles() {
    const human = q('replay-toggle-human');
    if (human) {
      human.onclick = () => {
        humanActionsHidden = !humanActionsHidden;
        syncHumanActionsVisibility();
      };
    }
    syncHumanActionsVisibility();
    const ai = q('replay-toggle-ai');
    if (ai) {
      ai.onclick = () => {
        aiActionsHidden = !aiActionsHidden;
        syncAiActionsVisibility();
      };
    }
    syncAiActionsVisibility();
  }

  // Jumping to a turn resets every fold to closed: a reader arriving at a turn
  // should meet the same short summary everywhere, not whatever was left open
  // on the turn before. The raw-transition and telemetry blocks are NOT touched
  // -- they carry their own classes and their own lazy load.
  function resetFolds() {
    const turns = q('replay-turns');
    if (!turns || typeof turns.querySelectorAll !== 'function') return;
    turns.querySelectorAll('details.replay-fold').forEach((details) => {
      details.open = false;
    });
  }

  function syncTurnNav() {
    const prev = q('replay-turn-prev');
    const next = q('replay-turn-next');
    if (prev) prev.disabled = currentTurnIndex <= 0;
    if (next) next.disabled = currentTurnIndex >= turnNumbers.length - 1;
  }

  function gotoTurnIndex(index) {
    if (!turnNumbers.length) return;
    currentTurnIndex = Math.max(0, Math.min(turnNumbers.length - 1, Number(index) || 0));
    resetFolds();
    const select = q('replay-turn-jump');
    if (select) select.value = String(turnNumbers[currentTurnIndex]);
    syncTurnNav();
    const card = q(`turn-${turnNumbers[currentTurnIndex]}`);
    if (card && typeof card.scrollIntoView === 'function') {
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function bindTurnNav() {
    const select = q('replay-turn-jump');
    if (select) {
      select.onchange = () => gotoTurnIndex(turnNumbers.indexOf(Number(select.value)));
    }
    const prev = q('replay-turn-prev');
    if (prev) prev.onclick = () => gotoTurnIndex(currentTurnIndex - 1);
    const next = q('replay-turn-next');
    if (next) next.onclick = () => gotoTurnIndex(currentTurnIndex + 1);
    syncTurnNav();
  }


  /* The list rows come from the archive INDEX, which is a flat summary rather
     than a game, so the fields sit at the top level here and under
     `ai_version` on the detail. Same absent-means-not-recorded rule. */
  function summaryDeepeningText(game) {
    const telemetry = window.__SAP_SEARCH_TELEMETRY;
    const row = game || {};
    const label = telemetry
      ? telemetry.deepeningLabel(row.completion_policy, row.completion_width)
      : null;
    return label === null ? 'not recorded' : label;
  }

  function gameLabel(game) {
    const stamp = String(game.created_at || '').replace('T', ' ').replace('Z', ' UTC');
    const result = game.winner ? `${game.winner} won` : String(game.end_reason || 'completed');
    const agent = game.agent_name || game.agent || 'unknown AI';
    const agentId = game.agent_id ? ` (${game.agent_id})` : '';
    const gear = game.gear ? ` · ${esc(String(game.gear))}` : '';
    return `<b>${esc(stamp)}</b><span>${esc(result)} · ${Number(game.turns || 0)} turns · seed ${esc(game.game_seed)} · ${esc(agent + agentId)} · deepening ${esc(summaryDeepeningText(game))}${gear}</span>`;
  }

  function paintList() {
    const list = q('replay-list');
    const empty = q('replays-empty');
    list.innerHTML = games.map((game) => (
      `<button class="replay-list-item${game.id === selectedId ? ' is-selected' : ''}" data-game-id="${esc(game.id)}">${gameLabel(game)}</button>`
    )).join('');
    empty.hidden = games.length > 0;
    list.querySelectorAll('[data-game-id]').forEach((button) => {
      button.addEventListener('click', () => {
        loadGame(button.dataset.gameId).catch(showError);
      });
    });
  }

  function safeCalculatorLink(value) {
    try {
      const url = new URL(String(value || ''));
      if (
        url.protocol === 'https:'
        && url.hostname === 'sap-calculator.com'
        && !url.username
        && !url.password
      ) return url.href;
    } catch (_error) {
      // Corrupt archives are rendered without an outbound link.
    }
    return null;
  }

  function itemName(item) {
    if (!item || item.empty) return 'empty';
    const stats = item.attack == null || item.health == null
      ? ''
      : ` ${Number(item.attack)}/${Number(item.health)}`;
    const level = Number(item.level || 1) > 1 ? ` L${Number(item.level)}` : '';
    return `${item.name || item.item_id || 'unknown'}${level}${stats}`;
  }

  function boardHTML(board, label, phase) {
    const slots = Array.isArray(board) ? board : [];
    return `<section class="replay-board-block" data-board-phase="${esc(phase)}">
      <h6>${esc(label)}</h6>
      <div class="replay-board-row">
        ${slots.map((slot) => `<div class="replay-board-slot${slot.empty ? ' is-empty' : ''}" data-board-slot="${Number(slot.slot)}">
          <b>slot ${Number(slot.slot)}</b>
          <span>${esc(itemName(slot))}</span>
          ${slot.equipment_id ? `<small>equipment ${esc(slot.equipment_id)}</small>` : ''}
        </div>`).join('')}
      </div>
    </section>`;
  }

  function shopHTML(shop, label) {
    const slots = (shop && shop.slots) || [];
    const provenance = (shop && shop.shop_provenance) || 'unknown';
    return `<section class="replay-shop-block" data-shop-provenance="${esc(provenance)}">
      <h6>${esc(label)} <small>${esc(provenance)}</small></h6>
      <div class="replay-shop-row">
        ${slots.map((slot) => `<div class="replay-shop-slot${slot.frozen ? ' is-frozen' : ''}" data-shop-slot="${Number(slot.shop_index)}">
          <b>${Number(slot.shop_index)} · ${esc(slot.slot_type)}</b>
          <span>${esc(itemName(slot))}</span>
          <small>${Number(slot.cost)} gold · ${esc(slot.provenance || provenance)}${slot.frozen ? ' · frozen' : ''}${slot.injected ? ' · ability injected' : ''}</small>
        </div>`).join('')}
      </div>
    </section>`;
  }

  // Every block a reader skips by default gets its OWN <details>, rather than
  // one page-wide "expand all": opening a board must not also open the shop
  // beside it. `replay-fold` is the class the turn jump closes again.
  function fold(label, innerHTML) {
    if (!innerHTML) return '';
    return `<details class="replay-fold"><summary>${esc(label)}</summary>${innerHTML}</details>`;
  }

  // The rule for the whole page (Ruihan, 2026-08-20): gold says what is LEFT.
  // Not what was spent, not a delta. `start_turn` carries it as `gold`, every
  // other op as `gold_after`.
  function goldLeft(event) {
    const value = event.gold_after == null ? event.gold : event.gold_after;
    return value == null ? '' : ` · gold ${Number(value)}`;
  }

  function itemChips(items) {
    const rows = Array.isArray(items) ? items : [];
    if (!rows.length) return '<span class="replay-muted">none</span>';
    return rows.map((item) => (
      `<span class="replay-item-chip">${esc(itemName(item))} · ${esc(item.provenance || item.identity_source || '')}</span>`
    )).join('');
  }

  function tierUpHTML(event) {
    if (!event || !event.tier_up_triggered) return '';
    const choices = Array.isArray(event.tier_up_choices) ? event.tier_up_choices : [];
    const content = choices.length
      ? itemChips(choices)
      : '<span class="replay-muted">choices unavailable in this archive</span>';
    return `<div class="replay-tier-up" data-tier-up-link="${esc(event.tier_up_link_id || '')}">
      <b>Tier Up choices</b>
      <div class="replay-item-chips">${content}</div>
    </div>`;
  }

  // The board an op leaves behind, unchanged from what the generic branch drew
  // before; it is shared now, because the short ops fold the same thing away.
  function afterBoardsHTML(event) {
    if (event.dst_after) return boardHTML([event.dst_after], 'Merge target after', 'merge-after');
    if (event.pet_after) return boardHTML([event.pet_after], 'Pet after buy', 'buy-after');
    if (event.target_after) return boardHTML([event.target_after], 'Target after food', 'food-after');
    if (Array.isArray(event.targets) && event.targets.length) {
      return boardHTML(event.targets.map((row) => row.after), 'Targets after food', 'food-after');
    }
    return '';
  }

  // What a frozen or unfrozen shop slot is CALLED -- display name, never an id.
  // `itemName` is the one place that decides that, so go through it.
  function frozenItemNames(event) {
    const rows = Array.isArray(event.items) ? event.items : [];
    if (!rows.length) return `shop slot ${Number(event.shop_index)}`;
    return rows.map((item) => itemName(item)).join(', ');
  }

  // buy_pet / buy_combine / sell: one short line a reader can scan, and one
  // fold holding the slot and the resulting board.
  //
  // The tier-up pair is deliberately NOT in that fold (Ruihan, 2026-08-25).
  // A level-up hands the shop a linked PAIR from the next tier, and one turn
  // can level up twice, so the pair on screen is not necessarily the pair that
  // produced the pet bought a few steps later. Folded on `buy_combine` -- the
  // op that triggers most level-ups -- the first pair was invisible while the
  // second, landing on a `merge`, was not, and the shop looked like it had
  // grown a pet from nowhere. Emitted in the same place as the generic branch
  // below, so every op that triggers a tier up shows its pair inline.
  function shortOpHTML(event, op) {
    const tierUp = event.tier_up_triggered ? ' · tier up' : '';
    let headline;
    let detail;
    let label;
    if (op === 'sell') {
      headline = `sell ${itemName(event.pet)}${goldLeft(event)}`;
      detail = `<p>from board slot ${esc(event.from_slot)}</p>`;
      label = 'Slot and the rest';
    } else if (op === 'buy_combine') {
      headline = `buy combine ${itemName(event.item)}${goldLeft(event)}${tierUp}`;
      detail = `<p>from shop slot ${esc(event.from_shop_slot)} onto board slot `
        + `${esc(event.onto && event.onto.slot)} ${esc(itemName(event.onto))}</p>`;
      label = 'Slot, boards and the rest';
    } else {
      headline = `buy ${itemName(event.item)}${goldLeft(event)}${tierUp}`;
      detail = `<p>from shop slot ${esc(event.from_shop_slot)} into board slot ${esc(event.to_slot)}</p>`;
      label = 'Slot, boards and the rest';
    }
    return `<p>${esc(headline)}</p>`
      + tierUpHTML(event)
      + fold(label, `${detail}${afterBoardsHTML(event)}`);
  }

  // The text for the ops whose LAYOUT the display spec left alone -- buy_food,
  // merge, and anything the archive grows later, ability_stock included.
  // buy_pet, buy_combine and sell go through `shortOpHTML` instead.
  //
  // Ruihan ruled on 2026-08-20 that the gold rule is the whole page's and not
  // only the rewritten ops': these carry what is LEFT too, and `Δgold` is gone
  // from the page altogether. The node probe pins that as a COUNT of zero
  // rather than op by op, which is what stops a delta creeping back into one
  // op while the others stay clean.
  function operationText(event) {
    const gold = goldLeft(event);
    if (event.op === 'buy_food') {
      const targets = Array.isArray(event.targets) ? event.targets : [];
      const targetText = targets.length
        ? targets.map((row) => `slot ${row.slot} ${itemName(row.before)}`).join(', ')
        : (event.target_mode === 'no_target' ? 'shop-wide effect' : 'no resolved target');
      return `Buy ${itemName(event.item)} from shop ${event.from_shop_slot} for ${targetText}${event.free ? ' · free' : ''}${gold}`;
    }
    if (event.op === 'merge') {
      return `Merge slot ${event.src_slot} ${itemName(event.src)} into slot ${event.dst_slot} ${itemName(event.dst)}${gold}`;
    }
    if (event.op === 'ability_stock') {
      return `${event.source || 'Ability'} stocked ${Number(event.count || 0)} × ${event.name || event.item_id} · ability injected`;
    }
    // An op nobody has taught this page yet: say what gold is left and nothing
    // it would have to invent.
    return gold.replace(/^ · /, '');
  }

  function rawTransitionHTML(event, turnNumber, sideKey, eventIndex) {
    // `transition_available` is the summary payload's marker; `transition` is
    // the field itself, which is still present under `detail=full`.
    if (!event.transition && !event.transition_available) return '';
    return `<details class="replay-raw" data-turn="${Number(turnNumber)}" data-side="${esc(sideKey)}" data-event="${Number(eventIndex)}"><summary>Raw engine transition</summary><pre>Open to load the archived transition.</pre></details>`;
  }

  function eventHTML(event, index, turnNumber, sideKey) {
    const op = String(event.op || 'unknown');
    let body = '';
    if (op === 'start_turn') {
      // Opening gold and the opening shop stay on screen -- they are what the
      // turn gets decided from. The board is one unfold away.
      body = `<p>Source <b>${esc(event.source)}</b>${goldLeft(event)}</p>
        ${fold('Board at turn start', boardHTML(event.board, 'Board at turn start', 'turn-start'))}
        ${shopHTML(event.shop, 'Shop after turn-start reroll')}`;
    } else if (op === 'roll') {
      // The rolled shop is the whole point of having rolled, so it stays open.
      body = `<p>roll${goldLeft(event)}</p>
        ${shopHTML(event.shop, 'Shop after roll')}`;
    } else if (op === 'freeze' || op === 'unfreeze') {
      body = `<p>${esc(op)} ${esc(frozenItemNames(event))}</p>
        ${fold(`Shop after ${op}`, `<p>shop slot ${esc(event.shop_index)}</p>
          <div class="replay-item-chips">${itemChips(event.items)}</div>
          ${shopHTML(event.shop, `Shop after ${op}`)}`)}`;
    } else if (op === 'reorder') {
      body = `<p>reorder</p>
      <div class="replay-board-pair">
        ${fold('Before reorder', boardHTML(event.board_before, 'Before reorder', 'reorder-before'))}
        ${fold('After reorder', boardHTML(event.board_after, 'After reorder', 'reorder-after'))}
      </div>`;
    } else if (op === 'end_turn') {
      // What the turn was FOR: the battle, and what it cost in lives.
      body = `<p>Battle result: <b>${esc((event.battle_result || {}).outcome || 'unknown')}</b> · lives ${esc((event.battle_result || {}).lives)}</p>
        ${fold('Frozen carried', `<p>Frozen carried: ${esc((event.frozen_carried || []).join(', ') || 'none')}</p>`)}
        ${fold('Next turn shop after automatic reroll', shopHTML(event.next_shop, 'Next turn shop after automatic reroll'))}`;
    } else if (op === 'buy_pet' || op === 'buy_combine' || op === 'sell') {
      body = shortOpHTML(event, op);
    } else {
      body = `<p>${esc(operationText(event))}</p>`;
      body += tierUpHTML(event);
      body += afterBoardsHTML(event);
    }
    return `<article class="replay-event" data-replay-op="${esc(op)}" data-replay-provenance="${esc(event.provenance || '')}" data-replay-injected="${event.injected ? 'true' : 'false'}" data-tier-up="${event.tier_up_triggered ? 'true' : 'false'}">
      <header><span class="replay-event-index">${index}</span><h5>${esc(op.replace(/_/g, ' '))}</h5>${event.tier_up_triggered ? '<span class="replay-tier-up-badge">Tier Up</span>' : ''}</header>
      ${body}
      ${rawTransitionHTML(event, turnNumber, sideKey, index)}
    </article>`;
  }

  function legacyActionsHTML(side) {
    const lines = (side && side.action_text) || [];
    if (!lines.length) return '<li class="replay-actions-empty">no archived action text</li>';
    return lines.map((line) => `<li data-replay-action="true">${esc(line)}</li>`).join('');
  }

  function sideReplayHTML(side, sideName, turnNumber, sideKey) {
    const replay = side && side.replay;
    if (!replay || !replay.detail_available) {
      const reason = replay && replay.unavailable_reason
        ? replay.unavailable_reason
        : 'transition detail unavailable';
      return `<section class="replay-side" data-side="${esc(sideName)}">
        <header><h4>${esc(sideName)}</h4><span class="replay-legacy-badge">legacy</span></header>
        <p class="replay-unavailable">${esc(reason)}. This v1 archive predates per-action transitions, so the UI will not infer missing shops or boards.</p>
        <ol class="replay-legacy-actions">${legacyActionsHTML(side)}</ol>
      </section>`;
    }
    const gold = replay.gold || {};
    const check = replay.self_check || {};
    return `<section class="replay-side" data-side="${esc(sideName)}">
      <header>
        <h4>${esc(sideName)}</h4>
        <span class="replay-check ${check.pass ? 'is-pass' : 'is-fail'}">self-check ${check.pass ? 'pass' : 'fail'}</span>
      </header>
      <div class="replay-gold">
        <span>base <b>${Number(gold.base || 0)}</b></span>
        <span>gained <b>${Number(gold.gained || 0)}</b></span>
        <span>spent <b>${Number(gold.spent || 0)}</b></span>
        <span>left <b>${Number(gold.left || 0)}</b></span>
        <span>rolls <b>${Number(gold.rolls || 0)}</b></span>
        <span>reconciled <b>${gold.reconciled ? 'yes' : 'no'}</b></span>
      </div>
      ${boardHTML(replay.board_in, 'Board in', 'board-in')}
      <div class="replay-chain">
        ${(replay.chain || []).map((event, index) => eventHTML(event, index, turnNumber, sideKey)).join('')}
      </div>
    </section>`;
  }

  function paintGame(game) {
    valueReadout = game.value_readout || null;
    valueByTurn = new Map(
      ((valueReadout && valueReadout.rows) || []).map((row) => [Number(row.turn), row])
    );
    // The jump control is painted inside `replay-meta`, so the turn list has to
    // be known before that innerHTML is written.
    turnNumbers = (game.turns || []).map((turn) => Number(turn.turn));
    currentTurnIndex = 0;
    const meta = q('replay-meta');
    const fullPng = game.full_game_png
      ? `<a class="replay-full-link" href="${esc(game.full_game_png)}" target="_blank">Open full-game battle PNG</a>`
      : '<span class="replay-muted">full-game PNG is still rendering</span>';
    meta.hidden = false;
    meta.innerHTML = `
      <h2>${esc(game.game_id || game.id)}</h2>
      <div class="replay-meta-grid">
        <span><b>result</b> ${esc(game.winner || game.end_reason || 'completed')}</span>
        <span><b>reason</b> ${esc(game.end_reason)}</span>
        <span><b>seed</b> ${esc((game.seeds || {}).game)}</span>
        <span><b>commit</b> <code>${esc(game.git_sha || 'unknown')}</code></span>
        <span><b>agent</b> ${esc((game.ai_version || {}).agent_name || (game.ai_version || {}).agent || 'unknown')} <code>${esc((game.ai_version || {}).agent_id || 'unknown')}</code></span>
        <span><b>turn cap</b> ${Number(game.max_turn || 0)}</span>
        <span><b>deepening</b> ${esc(gameDeepeningText(game))}</span>
        <span><b>gear</b> ${esc(gameGearText(game))}</span>
      </div>
      ${fullPng}
      ${valueHeaderHTML()}
      <div class="replay-view-controls">
        <button id="replay-toggle-human" class="menu-btn secondary replay-side-toggle replay-human-toggle" type="button" aria-pressed="${humanActionsHidden ? 'true' : 'false'}">${humanActionsHidden ? 'Show Human actions' : 'Hide Human actions'}</button>
        <button id="replay-toggle-ai" class="menu-btn secondary replay-side-toggle" type="button" aria-pressed="${aiActionsHidden ? 'true' : 'false'}">${aiActionsHidden ? 'Show AI actions' : 'Hide AI actions'}</button>
        <label class="replay-turn-jump-label" for="replay-turn-jump">Jump to turn</label>
        <select id="replay-turn-jump" class="replay-turn-jump" aria-label="Jump to turn">${turnNumbers.map((number, index) => `<option value="${Number(number)}"${index === currentTurnIndex ? ' selected' : ''}>Turn ${Number(number)}</option>`).join('')}</select>
        <button id="replay-turn-prev" class="menu-btn secondary replay-turn-step" type="button">Previous turn</button>
        <button id="replay-turn-next" class="menu-btn secondary replay-turn-step" type="button">Next turn</button>
      </div>`;
    bindActionToggles();
    turnDetail.clear();
    const gameId = game.id;
    q('replay-turns').innerHTML = (game.turns || []).map((turn) => {
      const turnNumber = Number(turn.turn);
      const render = turn.render || {};
      const image = render.path
        ? `<a href="/api/duel/replay_image?id=${encodeURIComponent(game.id)}&turn=${Number(turn.turn)}" target="_blank"><img class="replay-turn-image" loading="lazy" src="/api/duel/replay_image?id=${encodeURIComponent(game.id)}&turn=${Number(turn.turn)}" alt="Battles through turn ${Number(turn.turn)}" /></a>`
        : '<div class="replay-render-missing">battle PNG is still rendering or unavailable</div>';
      const calculatorLink = safeCalculatorLink(render.calculator_link);
      const calc = calculatorLink
        ? `<a class="replay-calc-link" href="${esc(calculatorLink)}" target="_blank" rel="noreferrer">Open this turn in SAP Calculator</a>`
        : '';
      return `<article class="replay-turn-card" id="turn-${Number(turn.turn)}">
        <header><div><h3>Turn ${Number(turn.turn)}</h3><span>tier ${Number(turn.tier || 0)}</span></div><span class="replay-outcome">${esc(turn.outcome || turn.error || 'unknown')}</span></header>
        ${image}${calc}
        ${valueRowHTML(turnNumber)}
        <div class="replay-sides">
          ${sideReplayHTML(turn.human, 'Human', turnNumber, 'human')}
          ${sideReplayHTML(turn.ai, 'AI', turnNumber, 'ai')}
        </div>
        ${searchRowHTML(turn)}
        <details class="replay-telemetry" data-turn="${turnNumber}"><summary>AI search, life, and battle telemetry</summary><pre>Open to load this turn's telemetry.</pre></details>
      </article>`;
    }).join('');
    bindTurnDetail(gameId);
    bindTurnNav();
  }

  /* Both collapsed blocks -- the raw engine transition and the search
     telemetry -- now fetch their own turn on first open instead of riding
     along in the whole-game payload. See `summarize_public_replay` in
     replay_view.py for the sizes that motivated it. */
  function bindTurnDetail(gameId) {
    q('replay-turns').querySelectorAll('details.replay-raw, details.replay-telemetry').forEach((details) => {
      details.addEventListener('toggle', () => {
        const pre = details.querySelector('pre');
        if (!details.open || !pre || pre.dataset.loaded) return;
        const isTelemetry = details.classList.contains('replay-telemetry');
        pre.dataset.loaded = 'true';
        pre.textContent = 'Loading…';
        loadTurnDetail(gameId, details.dataset.turn).then((turn) => {
          const value = isTelemetry
            ? telemetryOf(turn)
            : transitionOf(turn, details.dataset.side, Number(details.dataset.event));
          pre.textContent = JSON.stringify(value, null, 2);
        }).catch((error) => {
          pre.dataset.loaded = '';   // let a retry happen on the next open
          pre.textContent = `Could not load this turn's detail: ${error.message}`;
        });
      });
    });
  }

  async function loadGame(gameId) {
    const requestedId = String(gameId);
    const request = ++detailRequest;
    selectedId = requestedId;
    paintList();
    let payload;
    try {
      const res = await fetch(`/api/duel/replay?id=${encodeURIComponent(requestedId)}`);
      payload = await res.json();
      if (!payload.ok) throw new Error(payload.error || 'could not load replay');
    } catch (error) {
      if (request !== detailRequest || selectedId !== requestedId) return;
      throw error;
    }
    if (request !== detailRequest || selectedId !== requestedId) return;
    paintGame(payload.game);
    const url = new URL(window.location.href);
    url.searchParams.set('game', requestedId);
    window.history.replaceState(null, '', url);
  }

  function clearGame(message) {
    ++detailRequest;
    games = [];
    selectedId = null;
    paintList();
    q('replay-meta').hidden = true;
    q('replay-meta').innerHTML = '';
    q('replay-turns').innerHTML = '';
    const empty = q('replays-empty');
    empty.hidden = false;
    empty.textContent = message;
    const url = new URL(window.location.href);
    url.searchParams.delete('game');
    window.history.replaceState(null, '', url);
  }

  async function refresh() {
    const request = ++refreshRequest;
    const search = q('replay-search').value.trim();
    let payload;
    try {
      const res = await fetch(`/api/duel/replays?q=${encodeURIComponent(search)}&limit=200`);
      payload = await res.json();
      if (!payload.ok && payload.ok !== undefined) {
        throw new Error(payload.error || 'could not list replays');
      }
    } catch (error) {
      if (request !== refreshRequest) return;
      throw error;
    }
    if (request !== refreshRequest) return;
    games = payload.games || [];
    const requested = new URL(window.location.href).searchParams.get('game');
    if (!initialized) {
      selectedId = requested || (games.length ? games[0].id : null);
      initialized = true;
    } else if (!games.some((game) => game.id === selectedId)) {
      selectedId = games.length ? games[0].id : null;
    }
    paintList();
    if (selectedId) {
      await loadGame(selectedId);
    } else {
      clearGame(
        search
          ? 'No completed games match this search.'
          : 'No completed games have been archived yet. Finish a game in Play and it will appear here.'
      );
    }
  }

  q('replay-search').addEventListener('input', () => refresh().catch(showError));
  function showError(error) {
    clearGame(`Could not load replays: ${error.message || error}`);
  }
  refresh().catch(showError);
})();
