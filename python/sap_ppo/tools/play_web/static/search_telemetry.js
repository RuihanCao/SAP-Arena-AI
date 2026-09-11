
(function () {
  'use strict';

  function modeLabel(mode) {
    if (['fixed-all', 'measured'].includes(mode)) return 'Fixed-all';
    if (['grow-k', 'resample-clock'].includes(mode)) return 'Grow k';
    return 'Custom';
  }

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
        const truncated = requested > s.width && ['fixed-all', 'grow-k', 'measured', 'resample-clock'].includes(String((turn || {}).gear));
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


  function deepeningLabel(policy, width) {
    if (!policy) return null;
    const name = String(policy);
    if (name === 'bc_greedy') return 'off';
    return `w=${Number(width || 0)}`;
  }

  /* Per turn, what the deepening actually DID -- `decided` is the number of
     imagined samples that had more than one completion to choose between, so
     under `bc_greedy` it is 0 by construction rather than by agreement. */
  function deepeningWorkText(turn) {
    const record = turn || {};

    if (record.completion_decided === undefined || record.completion_decided === null) {
      return 'not recorded';
    }
    const decided = Number(record.completion_decided || 0);
    const divergent = Number(record.completion_divergent || 0);
    if (!decided) return 'no completion choice';
    const pct = Math.round((divergent / decided) * 100);
    return `${divergent} of ${decided} (${pct}%)`;
  }


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
      budget: modeLabel(record.gear) === 'Fixed-all' ? 'Fixed-all' : `${Number(record.turn_budget_s || 0)}s`,
      widths: widthsText(record),
      innerWidth: innerWidthText(record),
      deepening: deepeningWorkText(record),
      samples: samplesText(record),
    };
  }

  window.__SAP_SEARCH_TELEMETRY = {
    modeLabel,
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
