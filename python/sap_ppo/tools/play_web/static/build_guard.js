/* Tell a long-lived tab that the server changed underneath it.

   These pages are served `no-store`, so a reload always gets the current
   bundle and there is no cache to fight. The failure this guards is the other
   one: a tab left open across a server restart keeps running the JavaScript it
   already has, against a server whose payloads have moved on. Nothing in the
   page notices, so a feature that depended on the old shape renders empty
   rather than broken, and it reads as "the app is buggy" for as long as the tab
   stays open. That is exactly how a debug drawer sat dead for three days after
   the catalog moved out of `/api/state`.

   So: remember which build answered when this page loaded, and say something
   the first time a different one answers. `/api/build` is the server's own
   report of the tree it loaded, it is cached per process, and it is deliberately
   cheap, so polling it costs nothing next to the traffic these pages already
   make. A server too old to have the route, or a tree git cannot resolve, both
   read as "no answer" and this stays silent rather than guessing.

   The shop page's "Serving" panel already shows this same identity, out of the
   `build` block on `/api/state`. That answers "which code is this?" when you
   think to look; this answers "the code changed while you were not looking",
   which is the case where nobody thinks to look. It lives in its own file, and
   polls rather than riding the state payload, so that one copy covers the shop
   page, the duel page and the replays list without being threaded through two
   separate bundles' state handling. */
(() => {
  'use strict';

  const POLL_MS = 30000;
  const BUILD_URL = '/api/build';

  let loadedCommit = null;
  let announced = false;
  let timer = null;
  let badge = null;

  async function readBuild() {
    try {
      const res = await fetch(BUILD_URL, {cache: 'no-store'});
      if (!res.ok) return null;
      const doc = await res.json();
      const build = doc && doc.build;
      const commit = build && build.commit;
      return typeof commit === 'string' && commit ? build : null;
    } catch (err) {
      // Offline, server down mid-restart, or no such route. All of them mean
      // "cannot tell", and a banner that fires on those would cry wolf.
      return null;
    }
  }


  function paintBadge(build) {
    if (!document.body) return;
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'sap-build-badge';
      badge.style.cssText = [
        'position:fixed', 'right:6px', 'bottom:4px', 'z-index:2147483646',
        'padding:2px 7px', 'border-radius:3px',
        'background:rgba(20,14,6,0.72)', 'color:#f2e6d0',
        'font:500 11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace',
        'letter-spacing:0.02em', 'pointer-events:none',
        'user-select:none', 'opacity:0.72',
      ].join(';');
      document.body.appendChild(badge);
    }
    const short = String(build.commit_short || build.commit || '').slice(0, 7);
    const port = window.location.port || '80';
    // `HEAD` is what a detached worktree reports, and printing it would read
    // as a branch name. The port and the commit still identify the server.
    badge.textContent = `SAP-Arena-AI ${build.version || ''} · ${short}${build.dirty ? ' (modified)' : ''}`;
  }

  function announce(was, now) {
    if (announced) return;
    announced = true;
    if (timer !== null) clearInterval(timer);

    const bar = document.createElement('div');
    bar.setAttribute('role', 'alert');
    bar.style.cssText = [
      'position:fixed', 'top:0', 'left:0', 'right:0', 'z-index:2147483647',
      'display:flex', 'align-items:center', 'justify-content:center',
      'gap:14px', 'flex-wrap:wrap',
      'padding:10px 16px',
      'background:#8a4b12', 'color:#fff',
      'font:600 14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
      'box-shadow:0 2px 10px rgba(0,0,0,0.35)'
    ].join(';');

    const text = document.createElement('span');
    text.textContent = 'This page was loaded from an older build of the server ('
      + was.slice(0, 7) + ', now ' + now.slice(0, 7) + '). Reload it before trusting what you see.';
    bar.appendChild(text);

    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Reload';
    button.style.cssText = [
      'padding:5px 14px', 'border-radius:4px',
      'border:1px solid rgba(255,255,255,0.65)',
      'background:transparent', 'color:#fff',
      'font:inherit', 'cursor:pointer'
    ].join(';');
    button.addEventListener('click', () => window.location.reload());
    bar.appendChild(button);

    document.body.appendChild(bar);

    // Push the page down instead of sitting on top of it: the bar stays until
    // the tab is reloaded, and covering the HUD would take controls away for
    // as long as it is up.
    const prev = window.getComputedStyle(document.body).paddingTop;
    document.body.style.paddingTop = (parseFloat(prev) || 0) + bar.offsetHeight + 'px';
  }

  async function check() {
    const build = await readBuild();
    if (!build) return;
    // Painted on every poll, not only the first: a server restart that keeps
    // the same port should move the commit on screen even while the stale-tab
    // banner is up, and after `announced` this is the only thing still
    // reporting.
    paintBadge(build);
    if (announced) return;
    const commit = build.commit;
    if (loadedCommit === null) {
      loadedCommit = commit;
      return;
    }
    if (commit !== loadedCommit) announce(loadedCommit, commit);
  }

  check();
  timer = setInterval(check, POLL_MS);
  // Coming back to a tab left open for days is the moment this matters most.
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') check();
  });
})();
