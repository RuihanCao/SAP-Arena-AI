/* Requires an existing Playwright installation and browser; never installs.
   Set PLAYWRIGHT_MODULE and DEMO_TEST_PYTHON to your installed runtime paths. */
const assert = require('node:assert/strict');
const {spawn, spawnSync} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');

(async () => {
  const server = spawn(process.env.DEMO_TEST_PYTHON || 'python', ['tests/browser_fixture_server.py'], {
    cwd: root, windowsHide: true, env: {...process.env, PYTHONPATH: path.join(root, 'python')},
    stdio: ['ignore','pipe','pipe'],
  });
  let browser;
  try {
    const port = await new Promise((resolve,reject) => {
      let out='', err='';
      const timer=setTimeout(() => reject(new Error('Fixture server timed out: '+err)), 30000);
      server.stderr.on('data', data => {err+=data;});
      server.once('exit', code => {clearTimeout(timer);reject(new Error('Fixture server exited '+code+': '+err));});
      server.stdout.on('data', data => {out+=data;if(out.includes('\n')) {clearTimeout(timer);resolve(Number(out.trim()));}});
    });
    const origin = `http://127.0.0.1:${port}`;
    browser = await chromium.launch({headless:true, executablePath:process.env.DEMO_TEST_CHROMIUM || undefined});
    const context = await browser.newContext({viewport:{width:1440,height:1080}});
    const page = await context.newPage();
    const errors=[];
    page.on('pageerror', error => {errors.push(error.message);console.error('Browser error:',error.message);});
    const snapshot = await (await context.request.get(origin+'/api/state')).json();
    let gear = {gear:'grow-k', gears:['grow-k','fixed-all'], width:72, completion_width:4,
      deepen_width:4, completion_policy:'v_search', stochastic_samples:12, turn_budget_s:105};
    let generation=0;
    let rejectNextGame=false;
    let replayMode=null;
    const submissions=[];
    const duel = () => ({gear, game_generation:generation, turn:1, done:false, turns:[],
      worker:{status:'idle'}, render:{status:'idle'}, human:{lives:6,wins:0}, ai:{lives:6,wins:0}});
    const duelSnapshot = () => ({...snapshot, state:{...snapshot.state,lives:6},
      game_mode:'versus', opponent_source_mode:'duel', duel:duel()});
    await context.route('**/api/duel/**', async route => {
      const url = new URL(route.request().url());
      let result;
      if (url.pathname.endsWith('/new_game')) {
        const body=route.request().postDataJSON();submissions.push(body);
        if(rejectNextGame) {
          rejectNextGame=false;
          await route.fulfill({json:{ok:false,error:'Fixture rejected settings',state:duelSnapshot()}});
          return;
        }
        gear={...gear, gear:body.gear, turn_budget_s:body.turn_budget_s || 105};
        generation++;
        result={ok:true,state:duelSnapshot()};
      } else if (url.pathname.endsWith('/status')) result=duel();
      else if (url.pathname.endsWith('/state')) result=duelSnapshot();
      else if (url.pathname.endsWith('/replays')) result={ok:true,games:replayMode ? [{id:'fixture-game',gear:replayMode,gear_width:72,agent_name:'SAP-Arena-AI'}] : []};
      else if (url.pathname.endsWith('/replay')) result={ok:true,game:{
        id:'fixture-game',done:true,winner:'human',end_reason:'lives',seeds:{game:7},max_turn:30,
        gear_start:{gear:replayMode,width:72},turns:[],
        ai_version:{agent_name:'SAP-Arena-AI',agent_id:'bc-value-configured-interactive',completion_policy:'v_search',completion_width:4},
      }};
      else result={ok:true,value:null};
      await route.fulfill({json:result});
    });

    await page.goto(origin);
    assert.equal(await page.title(),'SAP-Arena-AI');
    for (const [id,target] of [['menu-play','/play'],['menu-sandbox','/sandbox'],['menu-replays','/replays']]) {
      assert.equal(await page.locator('#'+id).getAttribute('href'),target);
      assert.equal((await context.request.get(origin+target)).status(),200);
    }
    const output=path.join(root,'artifacts','public-demo-check');fs.mkdirSync(output,{recursive:true});
    await page.screenshot({path:path.join(output,'home.png'),fullPage:true});

    await page.click('#menu-play');
    await page.locator('#duel-setup:not([hidden])').waitFor();
    assert.deepEqual(await page.locator('#setup-gear option').allTextContents(),['Grow k (default)','Fixed-all']);
    assert.equal(await page.locator('#setup-budget').inputValue(),'105');
    assert(await page.locator('#setup-budget-row').isVisible());
    assert.equal(await page.locator('#setup-width, #setup-deepen').count(),0);
    await page.screenshot({path:path.join(output,'grow-k.png'),fullPage:true});
    await page.selectOption('#setup-gear','fixed-all');
    assert(!(await page.locator('#setup-budget-row').isVisible()));
    assert(await page.locator('#setup-budget').isDisabled());
    await page.screenshot({path:path.join(output,'fixed-all.png'),fullPage:true});
    await page.click('#duel-setup-start');
    await page.waitForFunction(() => document.querySelector('#duel-mode-readout').textContent==='Fixed-all');
    assert.deepEqual(submissions.at(-1),{gear:'fixed-all'});

    rejectNextGame=true;
    await page.click('#duel-new-game');
    await page.click('#duel-setup-start');
    await page.locator('#duel-setup-error:not([hidden])').waitFor();
    assert.equal(await page.locator('#duel-setup-error').innerText(),'Fixture rejected settings');
    await page.click('#duel-setup-cancel');
    assert(!(await page.locator('#duel-time-readout').isVisible()));
    await page.click('#duel-new-game');
    await page.selectOption('#setup-gear','grow-k');
    await page.fill('#setup-budget','17');
    await page.click('#duel-setup-start');
    await page.waitForFunction(() => document.querySelector('#duel-mode-readout').textContent==='Grow k');
    assert.deepEqual(submissions.at(-1),{gear:'grow-k',turn_budget_s:17});
    await page.click('#duel-new-game');
    await page.selectOption('#setup-gear','fixed-all');
    await page.click('#duel-setup-start');
    await page.waitForFunction(() => document.querySelector('#duel-mode-readout').textContent==='Fixed-all');
    assert.deepEqual(submissions.at(-1),{gear:'fixed-all'});

    // Sharing read-only board rendering with Sandbox must leave /play intact.
    const playEnd = await page.evaluate(() => {
      const board=appState.state.team.map((slot,index) => ({...slot,
        pet_id:index===0?'pet-fish':null,attack:12,health:15}));
      appState.duel={...appState.duel,done:true,winner:'human',end_reason:'ai_lives_0',
        final_human_board:board,turns:[{turn:1,ok:true,outcome:'win',ai_board:board}]};
      window.__SAP_AFTER_RENDER();
      return {visible:!document.querySelector('#duel-end').hidden,
        title:document.querySelector('#duel-end-title').textContent,
        human:document.querySelector('#duel-end-human').dataset.side,
        ai:document.querySelector('#duel-end-ai').dataset.side,
        cards:document.querySelectorAll('#duel-end .is-readonly').length};
    });
    assert.deepEqual(playEnd,{visible:true,title:'You win',human:'human',ai:'ai',cards:10});

    for (const url of ['/play','/sandbox']) {
      await page.goto(origin+url);
      await page.waitForFunction(() => typeof setMessage==='function');
      await page.evaluate(() => {document.querySelector('#duel-setup')?.setAttribute('hidden','');});
      for (const message of ['Bought a pet.','Sold a pet.','Combined pets.','Fed a pet.','Rolled.','Frozen.','Undo complete.']) {
        const state=await page.evaluate(message => {
          hideToast();setMessage(message,true);
          return {shown:document.querySelector('#toast').classList.contains('show'), logged:document.querySelector('#messages')?.textContent};
        },message);
        assert.equal(state.shown,false);
        assert.equal(state.logged,message);
      }
      for (const message of ['Cannot buy: no gold.','Request failed.','Second click ignored.']) {
        await page.evaluate(message => setMessage(message,false), message);
        assert(await page.locator('#toast.show').isVisible());
      }
      await page.evaluate(() => {hideToast();setMessage('Select a team pet target for this food.',true,null,true);});
      assert(await page.locator('#toast.show').isVisible());
      assert.equal(await page.locator('#btn-recommend, #recommendation-panel').count(),0);
      assert.equal(await page.locator('#drawer-dev').getAttribute('open'),null);
      if(url==='/play') assert(!(await page.locator('#duel-render-image').isVisible()));
      assert(!/SAP-PPO|exp\d\d|Tempo Recommendation|head_ref|\/root\//i.test(await page.locator('body').innerText()));
      await page.screenshot({path:path.join(output,url.slice(1)+'-prompt.png'),fullPage:true});
    }
    const sandboxChecks = await require('./browser_sandbox_checks.cjs')({context,page,origin,snapshot,output});
    await page.goto(origin+'/replays');
    await page.locator('#replays-empty').waitFor();
    assert.match(await page.locator('#replays-empty').innerText(),/No completed games/);
    assert.equal(await page.locator('a[href="/replay"]').count(),0);
    await page.screenshot({path:path.join(output,'replays.png'),fullPage:true});
    for(const [mode,label] of [['fixed-all','Fixed-all'],['grow-k','Grow k']]) {
      replayMode=mode;
      await page.goto(origin+'/replays');
      await page.locator('#replay-meta:not([hidden])').waitFor();
      assert((await page.locator('#replay-meta').innerText()).includes(label+' · root 72'));
      assert(!/full-clock|resample-clock|measured/.test(await page.locator('#replay-meta').innerText()));
    }
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({ok:true,submissions,pages:4,screenshots:6+sandboxChecks.screenshots,sandboxChecks,consoleErrors:errors}));
  } finally {
    if(browser) await browser.close();
    // The Windows venv launcher starts another Python process. Close only
    // this fixture's process tree so inherited pipes cannot leave a server up.
    if(process.platform==='win32') {
      spawnSync('taskkill',['/pid',String(server.pid),'/t','/f'],{windowsHide:true,stdio:'ignore'});
    } else server.kill();
  }
})().catch(error => {console.error(error);process.exitCode=1;});
