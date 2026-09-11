/* Called by browser_public_demo.cjs. All mutations are intercepted fixtures. */
const assert = require('node:assert/strict');
const path = require('node:path');

module.exports = async function checkSandbox({context, page, origin, snapshot, output}) {
  const fresh = structuredClone(snapshot);
  fresh.state.lives = 6;
  fresh.state.trophies = 0;
  fresh.state.turn = 1;
  fresh.last_battle = null;
  fresh.history = [];
  // Use real sprites and board-card layout, without simulating a battle.
  const team = fresh.state.team.map((slot, index) => ({...slot,
    pet_id: index === 0 ? 'pet-fish' : index === 4 ? 'pet-ant' : null,
    attack: index === 0 ? 12 : 3, health: index === 0 ? 15 : 2,
  }));
  let current = {...structuredClone(fresh), state:{...fresh.state,team,turn:12,lives:1,trophies:9}};
  let resets = 0;
  let resetMode = 'reject';
  let releaseReset;
  let resetStarted;
  const matches = name => url => url.pathname === '/api/'+name;
  const stateRoute = matches('state');
  const resetRoute = matches('reset');
  const applyRoute = matches('apply');
  await context.route(stateRoute, route => route.fulfill({json:current}));
  await context.route(applyRoute, async route => {
    assert.equal(route.request().postDataJSON().action.type, 'END_TURN');
    current = {...current, state:{...current.state,turn:13,lives:0}, last_battle:{
      turn:12,result:'loss',oracle_ok:true,
      session_replay_image_url:'/api/replay_image?mode=session&v=1',
    }};
    await route.fulfill({json:{ok:true,state:current,transition:{action:{type:'END_TURN'},legal:true}}});
  });
  await context.route(resetRoute, async route => {
    resets++;
    if (resetMode === 'reject') {
      await route.fulfill({json:{ok:false,error:'Fixture reset failed',state:current}});
    } else if (resetMode === 'offline') {
      await route.abort('failed');
    } else if (resetMode === 'reply-lost') {
      current = structuredClone(fresh);
      await route.abort('failed');
    } else {
      resetStarted();
      await new Promise(resolve => {releaseReset=resolve;});
      current = structuredClone(fresh);
      await route.fulfill({json:{ok:true,state:current}});
    }
  });
  try {
    await page.goto(origin+'/sandbox');
    await page.waitForFunction(() => appState && appState.state.trophies === 9);
    assert(!(await page.locator('#sandbox-end').isVisible()));
    await page.click('#btn-end-turn');
    await page.locator('#sandbox-end:not([hidden])').waitFor();
    assert.equal(await page.locator('#sandbox-end-title').innerText(),'Game over');
    assert.match(await page.locator('#sandbox-end-sub').innerText(),/Turn 12.*9 \/ 10 trophies.*0 lives/);
    assert(await page.locator('#stage').evaluate(el => el.inert));
    assert(await page.locator('#btn-end-turn').isDisabled());
    assert(await page.locator('#btn-roll').isDisabled());
    assert.equal(await page.locator('#sandbox-end-board .is-readonly').count(),5);
    assert.equal(await page.locator('#sandbox-end-board [draggable="true"]').count(),0);
    const cards = await page.locator('#sandbox-end-board .card').all();
    assert.match(await cards[0].getAttribute('title'),/Ant/);
    assert.match(await cards[4].getAttribute('title'),/Fish 12\/15/);
    assert.equal(await page.locator('#sandbox-end-replay').getAttribute('href'),'/api/replay_image?mode=session&v=1');
    await page.screenshot({path:path.join(output,'sandbox-game-over.png'),fullPage:true});

    // Reloading an already finished game must also show the end screen.
    await page.reload();
    await page.locator('#sandbox-end:not([hidden])').waitFor();
    await page.click('#sandbox-end-again');
    await page.locator('#sandbox-end-error:not([hidden])').waitFor();
    assert.equal(await page.locator('#sandbox-end-error').innerText(),'Fixture reset failed');
    assert(await page.locator('#sandbox-end').isVisible());
    assert.equal(resets,1);

    // A missing reply re-reads the board, but never retries the reset POST.
    resetMode = 'offline';
    await page.click('#sandbox-end-again');
    await page.waitForFunction(() => document.querySelector('#sandbox-end-error').textContent.includes('No answer'));
    assert(await page.locator('#sandbox-end').isVisible());
    assert.equal(resets,2);

    resetMode = 'success';
    const started = new Promise(resolve => {resetStarted=resolve;});
    await page.click('#sandbox-end-again');
    await started;
    assert(await page.locator('#sandbox-end-again').isDisabled());
    await page.evaluate(() => document.querySelector('#sandbox-end-again').click());
    assert.equal(resets,3);
    releaseReset();
    await page.locator('#sandbox-end[hidden]').waitFor({state:'attached'});
    assert(!(await page.locator('#stage').evaluate(el => el.inert)));
    assert(!(await page.locator('#btn-end-turn').isDisabled()));
    assert(!(await page.locator('#btn-roll').isDisabled()));
    assert.equal(await page.evaluate(() => appState.state.lives),6);
    assert.equal(await page.evaluate(() => appState.state.turn),1);
    assert.equal(await page.locator('#drawer-dev').getAttribute('open'),null);

    // The reset can succeed even if its reply is lost. Re-reading finds the
    // fresh game; neither a second reset nor an old end-screen error survives.
    current = {...structuredClone(fresh),state:{...fresh.state,lives:0}};
    resetMode = 'reply-lost';
    await page.reload();
    await page.locator('#sandbox-end:not([hidden])').waitFor();
    await page.click('#sandbox-end-again');
    await page.locator('#sandbox-end[hidden]').waitFor({state:'attached'});
    assert.equal(resets,4);
    assert.equal(await page.evaluate(() => appState.state.lives),6);

    // Trophy completion is Arena-only. Versus keeps its two-life-counter rule.
    current = {...structuredClone(fresh),state:{...fresh.state,team,turn:16,trophies:10}};
    await page.evaluate(() => refresh());
    await page.locator('#sandbox-end:not([hidden])').waitFor();
    assert.equal(await page.locator('#sandbox-end-title').innerText(),'You win');
    assert(!(await page.locator('#sandbox-end-error').isVisible()));
    assert(!(await page.locator('#sandbox-end-replay').isVisible()));
    await page.screenshot({path:path.join(output,'sandbox-win.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'sandbox-win-mobile.png'),fullPage:true});
    assert(await page.locator('#sandbox-end-again').isVisible());
    assert(await page.locator('#sandbox-end').evaluate(el => el.scrollWidth <= el.clientWidth));
    assert(await page.locator('#sandbox-end-board').evaluate(el =>
      el.lastElementChild.getBoundingClientRect().right <= el.parentElement.getBoundingClientRect().right + 1));
    await page.setViewportSize({width:1440,height:1080});

    current = {...current,game_mode:'versus',state:{...current.state,meta:{versus:{opponent_lives:2}}}};
    await page.reload();
    await page.waitForFunction(() => appState && appState.game_mode === 'versus');
    assert(!(await page.locator('#sandbox-end').isVisible()));
    current.state.meta.versus.opponent_lives = 0;
    await page.reload();
    await page.locator('#sandbox-end:not([hidden])').waitFor();
    assert.equal(await page.locator('#sandbox-end-title').innerText(),'You win');
    assert(!/trophies/.test(await page.locator('#sandbox-end-sub').innerText()));
  } catch (error) {
    console.error('Sandbox check state:', await page.evaluate(() => ({
      lives:appState && appState.state.lives,
      endHidden:document.querySelector('#sandbox-end')?.hidden,
      message:document.querySelector('#messages')?.textContent,
    })));
    await page.screenshot({path:path.join(output,'sandbox-check-failure.png'),fullPage:true});
    throw error;
  } finally {
    await context.unroute(stateRoute);
    await context.unroute(resetRoute);
    await context.unroute(applyRoute);
  }
  return {sandboxEndScreen:true,resets,screenshots:3};
};
