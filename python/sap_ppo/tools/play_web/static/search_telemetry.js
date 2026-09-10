/* exp16 Amendment 6: how hard the AI thought, in one wording for both pages.

   The live duel page has always shown this per turn. Ruihan asked for the
   same thing to be readable from a replay, and the reason this is a shared
   file rather than a second copy is that the two would drift: the live table
   and the replay table would start disagreeing about what "5 + 1d of 6"
   counts, and the replay is the one nobody is watching while it is written.

   Plain text out, no markup. Callers escape and lay out.

   The two pages hand in DIFFERENT shapes, deliberately. A live turn record
   carries whole segment objects; a replayed one carries only the compact
   `segment_widths` the transport projection keeps, because the segments
   themselves are the 22 MB this page was fixed for. `segmentWidths` below is
   the one place that difference is absorbed. */
(function () {
  'use strict';

  function segmentWidths(turn) {
    const record = turn || {};
    const full = Array.isArray(record.segments) ? record.segments : null;
    if (full) {
      return full.map((s) => ({
        width: Number((s || {}).width || 0),
        width_requested: (s || {}).width_requested,
        mode: String((s || {}).mode || ''),
      }));
    }
    const compact = Array.isArray(record.segment_widths) ? record.segment_widths : [];
    return compact.map((s) => ({
      width: Number((s || {}).width || 0),
      width_requested: (s || {}).width_requested,
      mode: String((s || {}).mode || ''),
    }));
  }

  /* A "d" prefix is a segment that ran only after the fixed turn clock was
     spent, so its width is the ranked minimum rather than what it searched.
     Same marker the live log has always used. */
  function widthsText(turn) {
    const widths = segmentWidths(turn);
    if (!widths.length) return '';
    return widths
      .map((s) => {
        const requested = Number(s.width_requested || 0);
        const truncated = requested > s.width && ['measured', 'resample-clock'].includes(String((turn || {}).gear));
        return `${s.mode === 'deadline-end-turn' ? 'd' : ''}${s.width}`
          + (truncated ? `/${requested} truncated` : '');
      })
      .join(' ');
  }

  function segmentsText(turn) {
    const record = turn || {};
    const searched = Number(record.searched_segments || 0);
    const deadline = Number(record.deadline_segments || 0);
    const total = Number(record.n_segments || 0);
    return `${searched} + ${deadline}d of ${total}`;
  }

  function secondsText(ms, digits) {
    return (Number(ms || 0) / 1000).toFixed(digits === undefined ? 1 : digits);
  }

  /* What the game was SET to, from a source that knows. Returns null when
     nothing knows, which is not the same as "off" and must not be shown as
     it: games archived before 2026-08-19 carry no completion fields at all,
     and printing "off" for those would be inventing a measurement. */
  function deepeningLabel(policy, width) {
    if (!policy) return null;
    const name = String(policy);
    if (name === 'bc_greedy') return 'off';
    return `${name} x${Number(width || 0)}`;
  }

  /* Per turn, what the deepening actually DID -- `decided` is the number of
     imagined samples that had more than one completion to choose between, so
     under `bc_greedy` it is 0 by construction rather than by agreement. */
  function deepeningWorkText(turn) {
    const record = turn || {};
    // Absent is not zero, and the difference is the whole claim. A turn
    // recorded before 2026-08-19 has no counter at all, and reporting that as
    // "no completion choice" says the deepening ran and never disagreed --
    // which is what a bc_greedy turn honestly says, and which nothing knows
    // about a legacy one. Verified against a real 2026-08-16 archived game:
    // the key is missing, not 0.
    if (record.completion_decided === undefined || record.completion_decided === null) {
      return 'not recorded';
    }
    const decided = Number(record.completion_decided || 0);
    const divergent = Number(record.completion_divergent || 0);
    if (!decided) return 'no completion choice';
    const pct = Math.round((divergent / decided) * 100);
    return `${divergent} of ${decided} (${pct}%)`;
  }

  /* W11b. What the reroll gear actually reached, per turn.

     Absent is NOT 3. A turn recorded before the gear existed has no counter at
     all, and printing the configured 3 there would say the gear ran and bought
     nothing -- which is what a `measured` turn honestly says and which nothing
     knows about a legacy one. Same rule the deepening column already follows.

     `realised` is the highest any segment reached and `levels` is the total
     bought across the turn, because a per-segment level is not a thing you can
     add up: two segments that each reached 5 did not reach 10. */
  function samplesText(turn) {
    const record = turn || {};
    const realised = record.realised_stochastic_samples;
    if (realised === undefined || realised === null) return 'not recorded';
    const levels = Number(record.extra_sample_levels || 0);
    if (!levels) return `${Number(realised)} (no extra)`;
    return `${Number(realised)} peak, +${levels} levels`;
  }

  /* The inner width this turn ran, which the row never showed: the widths
     column is the OUTER width and the samples column is k, so the third knob
     -- the one that multiplies against k and is therefore the expensive one --
     was invisible in both the live log and the replay.

     Read off the segments, not off the game's settings, so a turn reports what
     it RAN rather than what the card was set to; those differ whenever a
     segment was skipped or degraded. Absent stays absent. */
  function innerWidthText(turn) {
    const record = turn || {};
    const full = Array.isArray(record.segments) ? record.segments : null;
    const values = (full || [])
      .map((s) => (s || {}).completion_width)
      .filter((v) => v !== null && v !== undefined)
      .map(Number);
    if (!values.length) {
      const one = record.completion_width;
      if (one === null || one === undefined) return 'not recorded';
      return String(Number(one));
    }
    const distinct = Array.from(new Set(values));
    return distinct.length === 1 ? String(distinct[0]) : distinct.join('/');
  }

  const COLUMNS = [
    {key: 'turn', label: 'turn'},
    {key: 'outcome', label: 'battle'},
    {key: 'lives', label: 'lives'},
    {key: 'segments', label: 'segments'},
    {key: 'aiTurn', label: 'AI turn'},
    {key: 'endWait', label: 'End wait'},
    {key: 'budget', label: 'budget'},
    {key: 'widths', label: 'segment width'},
    {key: 'innerWidth', label: 'deepening width'},
    {key: 'deepening', label: 'deepening picked'},
    {key: 'samples', label: 'k samples'},
  ];

  function turnRowCells(turn) {
    const record = turn || {};
    const lives = record.lives;
    return {
      turn: Number(record.turn || 0),
      outcome: record.ok === false ? 'failed' : String(record.outcome || '-'),
      lives: lives ? `${lives.human} - ${lives.ai}` : '-',
      segments: segmentsText(record),
      aiTurn: `${secondsText(record.elapsed_ms)}s`,
      endWait: `${secondsText(record.end_wait_ms, 2)}s`,
      budget: `${Number(record.turn_budget_s || 0)}s${record.adaptive_chunks ? ' adaptive' : ' fixed chunk'}`,
      widths: widthsText(record),
      innerWidth: innerWidthText(record),
      deepening: deepeningWorkText(record),
      samples: samplesText(record),
    };
  }

  window.__SAP_SEARCH_TELEMETRY = {
    COLUMNS,
    segmentWidths,
    widthsText,
    segmentsText,
    deepeningLabel,
    deepeningWorkText,
    innerWidthText,
    samplesText,
    turnRowCells,
  };
}());
