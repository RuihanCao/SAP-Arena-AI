/* Sandbox end screen. Arena trophies and versus lives remain separate rules. */
(() => {
  'use strict';
  const q = id => document.getElementById(id);
  const overlay = q('sandbox-end');
  if (!overlay) return;

  function fitFinalBoard() {
    const board = q('sandbox-end-board');
    if (!overlay.hidden && board.scrollWidth) {
      const scale = Math.min(0.72, board.parentElement.clientWidth / board.scrollWidth);
      board.style.transform = `scale(${scale})`;
    }
  }

  function renderSandboxEnd() {
    const state = appState && appState.state;
    const mode = appState && appState.game_mode;
    const opponentLives = state && state.meta && state.meta.versus && state.meta.versus.opponent_lives;
    const lost = Boolean(state && Number(state.lives) <= 0);
    const won = Boolean(state && (mode === 'versus'
      ? opponentLives != null && Number(opponentLives) <= 0
      : Number(state.trophies) >= 10));
    const done = lost || won;
    const wasHidden = overlay.hidden;
    overlay.hidden = !done;
    // The finished shop is still visible behind the same overlay as /play,
    // but cannot receive a stray keyboard, click or drag action.
    q('stage').inert = done;
    if (!done) {
      q('sandbox-end-error').hidden = true;
      return;
    }
    for (const id of ['btn-roll', 'btn-end-turn', 'btn-context-action']) q(id).disabled = true;
    q('sandbox-end-title').textContent = won && !lost ? 'You win'
      : mode === 'versus' && won && lost ? 'Draw' : 'Game over';
    const battle = appState.last_battle;
    const turn = battle && battle.turn != null ? battle.turn : state.turn;
    const score = mode === 'versus'
      ? `lives ${state.lives} – ${opponentLives ?? '-'}`
      : `${state.trophies} / 10 trophies · ${state.lives} lives`;
    const reason = lost ? 'you ran out of lives' : mode === 'versus'
      ? 'the opponent ran out of lives' : 'you reached 10 trophies';
    q('sandbox-end-sub').textContent = `Turn ${turn} · ${score} · ${reason}`;
    renderReadOnlyBoard(q('sandbox-end-board'), state.team, 'human');
    fitFinalBoard();
    const replay = q('sandbox-end-replay');
    const replayUrl = battle && battle.session_replay_image_url;
    replay.hidden = !replayUrl;
    if (replayUrl) replay.href = replayUrl;
    else replay.removeAttribute('href');
    if (wasHidden) q('sandbox-end-again').focus();
  }

  q('sandbox-end-again').addEventListener('click', async () => {
    const error = q('sandbox-end-error');
    error.hidden = true;
    const result = await resetShopSession();
    if (!result.ok && !overlay.hidden) {
      error.textContent = result.transport
        ? 'No answer from the server. The board was rechecked; reset was not retried.'
        : result.error;
      error.hidden = false;
    }
  });
  window.__SAP_AFTER_RENDER = renderSandboxEnd;
  window.addEventListener('resize', fitFinalBoard);
  renderSandboxEnd();
})();
