    let appState = null;
    let selectedTeamIndex = null;
    let selectedShopIndex = null;
    let pendingFoodTargetShopIndex = null;
    let dragData = null;
    let contextActionMode = null;
    let teamFxBySlot = {};
    const FX_TTL_MS = 1200;
    let fxCleanupTimer = null;
    let debugSelectorsBootstrapped = false;
    let recommendationPending = false;
    let lastRecommendationSignature = null;
    let toastTimer = null;

    /* The two halves of a snapshot this page no longer makes the server repeat.

       `catalogBundle` is the pinned pack's stat table and ability rows -- 18.8 KB
       that used to ride along on every shop click, unchanged, because the page
       reads them off `appState`. They are fetched once from `/api/catalog` and
       merged back into every snapshot below, so the five call sites that read
       `appState.debug_catalog` / `appState.catalog_base_stats` are exactly as
       they were.

       `historyEntries` is the session log. The server sends only the entries
       this client has not seen (see `http_app.py::_apply_history_cursor`), and
       they are accumulated here into the whole log the history panel renders.
       `historyToken` is the server's content hash of that log, handed back as
       the cursor; the moment the two disagree -- an undo, a reset, a second
       tab, a restarted server -- the server ignores the cursor and sends the
       lot, and `absorbHistory` simply replaces what is here. */
    let catalogBundle = null;
    let catalogBundleUrl = null;
    let catalogFetchInFlight = null;
    let historyEntries = [];
    let historyToken = null;


    let mutationInFlight = null;


    function setHTML(id, html) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = html;
      return el;
    }

    function bind(id, event, handler) {
      const el = document.getElementById(id);
      if (el) el.addEventListener(event, handler);
      return el;
    }

    function afterRender() {
      if (typeof window !== 'undefined' && typeof window.__SAP_AFTER_RENDER === 'function') {
        window.__SAP_AFTER_RENDER();
      }
    }

    function prettyAction(action) {
      return JSON.stringify(action);
    }

    function formatPredictedTeam(team) {
      const source = Array.isArray(team) ? team : [];
      const labels = source.slice(0, 5).map((slot) => {
        if (!slot || typeof slot !== 'object') return '-';
        const rawName = String(slot.pet_name || slot.name || slot.pet_id || '').trim();
        if (!rawName) return '-';
        const atkParsed = Number.parseInt(slot.attack ?? 0, 10);
        const hpParsed = Number.parseInt(slot.health ?? 0, 10);
        const atk = Number.isNaN(atkParsed) ? 0 : atkParsed;
        const hp = Number.isNaN(hpParsed) ? 0 : hpParsed;
        return `${rawName}(${atk}/${hp})`;
      });
      while (labels.length < 5) {
        labels.push('-');
      }
      return labels.join(' | ');
    }

    function stableStringify(value) {
      if (value === null || typeof value !== 'object') {
        return JSON.stringify(value);
      }
      if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(',')}]`;
      }
      const keys = Object.keys(value).sort();
      return `{${keys.map((k) => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(',')}}`;
    }

    function mergeFxMaps(existing, incoming) {
      const out = {...(existing || {})};
      const now = Date.now();
      Object.keys(incoming || {}).forEach((key) => {
        const idx = Number.parseInt(key, 10);
        if (Number.isNaN(idx)) return;
        const prev = out[idx] || {};
        const next = incoming[idx] || {};
        out[idx] = {
          buff: Boolean(prev.buff || next.buff),
          levelup: Boolean(prev.levelup || next.levelup),
          atkPlus: Math.max(
            0,
            (Number.parseInt(prev.atkPlus ?? 0, 10) || 0) + (Number.parseInt(next.atkPlus ?? 0, 10) || 0),
          ),
          hpPlus: Math.max(
            0,
            (Number.parseInt(prev.hpPlus ?? 0, 10) || 0) + (Number.parseInt(next.hpPlus ?? 0, 10) || 0),
          ),
          until: Math.max(Number(prev.until || 0), Number(next.until || now + FX_TTL_MS)),
        };
      });
      return out;
    }

    function fxToInt(value, fallback = 0) {
      const parsed = Number.parseInt(value ?? fallback, 10);
      return Number.isNaN(parsed) ? fallback : parsed;
    }

    /* Everything about a pet a reposition CANNOT change. Two slots with the
       same signature are the same pet as far as any animation is concerned.
       Only the FALLBACK pairing below needs it: when a transition names the
       engine actions it was made of, the mapping is read off those instead of
       being guessed from the numbers. */
    function petFxSignature(slot) {
      const statuses = Array.isArray(slot.status_effects) ? [...slot.status_effects].sort() : [];
      return [
        String(slot.pet_id),
        fxToInt(slot.attack, 0),
        fxToInt(slot.health, 0),
        fxToInt(slot.level, 1),
        fxToInt(slot.exp, 0),
        fxToInt(slot.temp_attack, 0),
        fxToInt(slot.temp_health, 0),
        String(slot.equipment_id ?? ''),
        statuses.join(','),
      ].join('|');
    }

    /* The five slabs indexed by `slot_index`, empty ones as null, so two
       boards can be lined up slab for slab. */
    function teamBySlab(team) {
      const out = [null, null, null, null, null];
      (team || []).forEach((slot) => {
        const i = fxToInt(slot && slot.slot_index, -1);
        if (i >= 0 && i < 5 && slot.pet_id) out[i] = slot;
      });
      return out;
    }

    /* Engine actions that cannot move a pet between slabs other than through
       their own `REORDER` order.

       Read off `engine.py` rather than assumed. `_apply_team_permutation` is
       the only thing in the shop that shuffles the team, and the whole call
       graph into it is two chains:

         apply_action, `REORDER` branch          -- the order the action carries
         apply_action, `BUY_FOOD` branch
           -> _resolve_shop_hurt_faint_chain
           -> _apply_pending_summons
           -> _make_room_for_summon_slot
           -> _push_{forward,backward}_from_slot -> _apply_team_permutation

       So only a food can shove the board behind an action's back: a sleeping
       pill kills a pet, the faint summons one, and landing that summon on an
       occupied slab shifts the run next to it. Buying, selling, combining,
       rolling and freezing all write in place.

       `BUY_FOOD`, `END_TURN` and anything not listed are therefore NOT here:
       a map derived for those has to be checked against the two boards before
       it is believed. */
    const FX_SLAB_STABLE_ACTIONS = new Set([
      'REORDER', 'BUY_PET', 'BUY_COMBINE', 'COMBINE', 'SELL', 'ROLL', 'FREEZE', 'UNFREEZE',
    ]);

    /* Fold a transition's actions into `map[k] = the slab after-slab k's pet
       came from`. `REORDER` is `new[k] = old[order[k]]` (engine.py), so
       composing one onto the running map is a single index lookup.

       Returns null when the group says nothing usable -- no actions at all, or
       an order that is not a permutation of the five slabs. `proven` is false
       when the group contains an action that could have shoved the board on
       its own; see the set above. */
    function slotMapFromActions(actions) {
      const list = (Array.isArray(actions) ? actions : []).filter(Boolean);
      if (!list.length) return null;
      let map = [0, 1, 2, 3, 4];
      let proven = true;
      for (let n = 0; n < list.length; n += 1) {
        const type = String(list[n].type || '');
        if (type === 'REORDER') {
          const order = list[n].order;
          if (!Array.isArray(order) || order.length !== 5) return null;
          const next = [];
          for (let k = 0; k < 5; k += 1) {
            const from = fxToInt(order[k], -1);
            if (from < 0 || from > 4) return null;
            next[k] = map[from];
          }
          if (new Set(next).size !== 5) return null;
          map = next;
        } else if (!FX_SLAB_STABLE_ACTIONS.has(type)) {
          proven = false;
        }
      }
      return {map, proven};
    }

    /* Does `map` describe what actually happened to these two boards? A pet
       never changes its `pet_id`, so a map that lines a pet up with a
       different pet, with a slab that was empty, or that empties a slab which
       still has a pet on it, is not this transition's. Asked only of the
       groups the set above cannot vouch for, where a false answer means
       exactly "something fainted or was summoned, so the board was shoved". */
    function slotMapMatchesBoards(before, after, map) {
      for (let k = 0; k < 5; k += 1) {
        const a = after[k];
        const b = before[map[k]];
        if (Boolean(a) !== Boolean(b)) return false;
        if (a && String(a.pet_id) !== String(b.pet_id)) return false;
      }
      return true;
    }

    /* The exact before-slab -> after-slab mapping for this transition, or null
       when it does not have one to give. */
    function slotMapForTransition(actions, beforeTeam, afterTeam) {
      const derived = slotMapFromActions(actions);
      if (!derived) return null;
      if (derived.proven) return derived.map;
      return slotMapMatchesBoards(teamBySlab(beforeTeam), teamBySlab(afterTeam), derived.map)
        ? derived.map
        : null;
    }


    function pairTeamSlots(beforeTeam, afterTeam, slotMap) {
      const before = teamBySlab(beforeTeam);
      const after = teamBySlab(afterTeam);
      const pairs = [null, null, null, null, null];

      if (slotMap) {
        for (let i = 0; i < 5; i += 1) {
          const j = slotMap[i];
          if (!after[i] || !before[j]) continue;
          pairs[i] = {index: j, slot: before[j], moved: j !== i};
        }
        return pairs;
      }

      const beforeSig = before.map((slot) => (slot ? petFxSignature(slot) : null));
      const afterSig = after.map((slot) => (slot ? petFxSignature(slot) : null));
      const claimed = [false, false, false, false, false];
      const claim = (i, j, moved) => {
        claimed[j] = true;
        pairs[i] = {index: j, slot: before[j], moved};
      };

      for (let i = 0; i < 5; i += 1) {
        if (after[i] && before[i] && afterSig[i] === beforeSig[i]) claim(i, i, false);
      }
      for (let i = 0; i < 5; i += 1) {
        if (!after[i] || pairs[i]) continue;
        for (let j = 0; j < 5; j += 1) {
          if (claimed[j] || !before[j] || beforeSig[j] !== afterSig[i]) continue;
          claim(i, j, true);
          break;
        }
      }
      for (let i = 0; i < 5; i += 1) {
        if (!after[i] || pairs[i] || claimed[i] || !before[i]) continue;
        if (String(after[i].pet_id) === String(before[i].pet_id)) claim(i, i, false);
      }
      return pairs;
    }

    /* What `setAppState` pairs two boards with: the transition's own actions
       when they spell the mapping out, the signature passes when they do not. */
    function pairTeamsForTransition(beforeTeam, afterTeam, actions) {
      return pairTeamSlots(
        beforeTeam,
        afterTeam,
        slotMapForTransition(actions, beforeTeam, afterTeam),
      );
    }

    /* An fx entry is filed under a slab, so a transition that moved pets has
       to take each in-flight animation along to wherever its pet went, or the
       glow stays behind on whatever slid into the slab it left. A pet that is
       gone (sold, combined away) drops its fx with it. */
    function carryTeamFx(existing, pairs) {
      const out = {};
      pairs.forEach((pair, index) => {
        if (!pair) return;
        const fx = (existing || {})[pair.index];
        if (fx) out[index] = fx;
      });
      return out;
    }

    function computeTeamFx(beforeState, afterState, pairs) {
      if (!beforeState || !afterState) return {};
      const now = Date.now();
      const byIndex = {};
      const afterTeam = afterState.team || [];
      const paired = pairs || pairTeamSlots(beforeState.team || [], afterTeam);

      for (let i = 0; i < 5; i += 1) {
        const a = afterTeam.find((slot) => slot.slot_index === i) || null;
        if (!a || !a.pet_id) continue;
        const pair = paired[i];

        const aAtk = fxToInt(a.attack, 0);
        const aHp = fxToInt(a.health, 0);
        const aLvl = fxToInt(a.level, 1);
        let atkDelta = 0;
        let hpDelta = 0;
        let lvlDelta = 0;

        if (pair) {
          /* Diffed even when the pet changed slab. A pet that ONLY moved diffs
             to zero and stays quiet by itself, and one that moved AND gained
             in the same transition has to show what it gained -- which is why
             this is a diff and not a `moved` early-out. */
          const b = pair.slot;
          atkDelta = aAtk - fxToInt(b.attack, 0);
          hpDelta = aHp - fxToInt(b.health, 0);
          lvlDelta = aLvl - fxToInt(b.level, 1);
        } else {
          // New pet on this slab: compare against base stats so summon buffs still show.
          const baseStats = ((appState && appState.catalog_base_stats) || {})[String(a.pet_id)] || {};
          const baseAtk = fxToInt(baseStats.attack, 0);
          const baseHp = fxToInt(baseStats.health, 0);
          atkDelta = aAtk - baseAtk;
          hpDelta = aHp - baseHp;
          // Temporary summon buffs are tracked separately; include them explicitly.
          atkDelta = Math.max(atkDelta, fxToInt(a.temp_attack, 0));
          hpDelta = Math.max(hpDelta, fxToInt(a.temp_health, 0));
        }

        const atkPlus = Math.max(0, atkDelta);
        const hpPlus = Math.max(0, hpDelta);
        const buff = atkPlus > 0 || hpPlus > 0;
        const levelup = lvlDelta > 0;
        if (!buff && !levelup) continue;
        byIndex[i] = {buff, levelup, atkPlus, hpPlus, until: now + FX_TTL_MS};
      }
      return byIndex;
    }


    if (typeof window !== 'undefined') {
      window.__SAP_FX_INTERNALS = {
        computeTeamFx,
        pairTeamSlots,
        pairTeamsForTransition,
        slotMapFromActions,
        slotMapForTransition,
        carryTeamFx,
        petFxSignature,
        // The live map `teamCardHTML` reads to decide which cards animate --
        // so the probe can drive the REAL path (`setAppState` -> carry ->
        // merge) and judge what the page would have drawn, not just what the
        // diff returned.
        teamFx: () => teamFxBySlot,
        resetTeamFx: () => { teamFxBySlot = {}; },
      };
    }

    /* Composed gestures (`buy_pet_at`, `move_pet`) are several engine actions
       applied as one. The AGENT surface does not have them -- its vocabulary
       is exactly `constants.ACTION_TYPES` -- so on that surface every drop
       falls back to what `legal_actions` enumerates and an unreachable drop
       says so instead of quietly being made reachable.

       Defaults to TRUE for a server that does not report a surface, so an
       older page against a newer server, or the reverse, degrades into the
       human surface it has always been rather than silently losing gestures. */
    /* What this page is looking at, said on the page itself.

       The point is that every field comes from the RUNNING server -- the
       commit is the HEAD of the tree the loaded module was imported from, not
       a branch anyone typed into a runbook, which is the one failure this
       whole redeploy exists to stop repeating. */
    function renderBuildLine() {
      const el = document.getElementById('build-line');
      if (!el) return;
      const surface = (appState && appState.surface) || null;
      if (!surface) { el.textContent = ''; return; }
      const b = surface.build || {};
      const bits = [
        `surface: ${surface.name}`,
        `commit: ${b.commit_short || 'unknown'}${b.dirty ? ' (dirty)' : ''}`,
        `version: ${b.version || 'unknown'}`,
        `composed gestures: ${surface.compose_enabled ? 'on' : 'off'}`,
        `imagined-walk validation: ${surface.imagined_validation_skipped ? 'SKIPPED' : 'on'}`,
      ];
      el.textContent = bits.join('  ·  ');
      el.title = 'Local demo build';
    }

    function composeEnabled() {
      if (!appState || !appState.surface) return true;
      return appState.surface.compose_enabled !== false;
    }

    /* The REORDER permutation that swaps two slabs -- the only reposition the
       engine enumerates, and therefore the only one the agent surface offers. */
    function transposition(a, b) {
      const order = [0, 1, 2, 3, 4];
      order[a] = b;
      order[b] = a;
      return order;
    }

    function hasLegalAction(action) {
      if (!appState) return false;
      const key = stableStringify(action);
      return appState.legal_actions.some((entry) => stableStringify(entry.action) === key);
    }

    function findLegalAction(action) {
      if (!appState) return null;
      const key = stableStringify(action);
      const match = appState.legal_actions.find((entry) => stableStringify(entry.action) === key);
      return match ? match.action : null;
    }

    function teamSlot(idx) {
      if (!appState) return null;
      return appState.state.team.find((slot) => slot.slot_index === idx) || null;
    }

    function shopSlot(idx) {
      if (!appState) return null;
      return appState.state.shop.find((slot) => slot.shop_index === idx) || null;
    }

    /* The shop lane is the game's own 9-column grid at the team lane's left edge
       and pitch: pets fill columns from the left, the food group is right-aligned
       so a single food sits at column 8, x=1022, the measured 0.80 of the scene
       width. Buying does not close the gap: the emptied column keeps its bare
       slab exactly where it was, like the reference capture.

       The engine's shop array shrinks on a buy and re-indexes what is left, so
       shop_index alone cannot tell the lane which column went empty. The lane
       therefore keeps its own map from column to live shop_index, vacates the
       bought column when a buy is applied, and rebuilds whenever anything
       replaces the shop (roll, end turn, reset, undo, debug add). This is a
       render-layer fix: no engine or API change. */
    const SHOP_LANE_COLUMNS = 9;
    // Each column holds either null (vacated) or {idx, key}: the live shop_index
    // plus what that column was last seen holding.
    let shopLaneLayout = null;

    function liveShopSlots() {
      return (appState && appState.state && appState.state.shop) || [];
    }

    function shopSlotKey(slot) {
      return `${slot.slot_type}|${slot.item_id}`;
    }

    function rebuildShopLaneLayout() {
      const source = liveShopSlots();
      const cols = (kind) => source
        .filter((slot) => slot.slot_type === kind)
        .map((slot) => ({idx: slot.shop_index, key: shopSlotKey(slot)}));
      shopLaneLayout = {pets: cols('pet'), foods: cols('food')};
    }

    function layoutHeld() {
      if (!shopLaneLayout) return [];
      return [...shopLaneLayout.pets, ...shopLaneLayout.foods].filter(Boolean);
    }

    // The layout still describes the live shop when it holds exactly the live
    // indices, each in a column of its own kind and still holding its own item.
    function shopLaneLayoutMatches() {
      if (!shopLaneLayout) return false;
      const source = liveShopSlots();
      const held = layoutHeld();
      if (held.length !== source.length) return false;
      const byIndex = new Map(source.map((slot) => [slot.shop_index, slot]));
      const kindOk = (cols, kind) => cols.every((col) => {
        if (col === null) return true;
        const slot = byIndex.get(col.idx);
        return Boolean(slot) && slot.slot_type === kind && shopSlotKey(slot) === col.key;
      });
      return kindOk(shopLaneLayout.pets, 'pet') && kindOk(shopLaneLayout.foods, 'food');
    }

    /* One slot left the shop: blank its column and slide every higher live index
       down by one, the way the engine re-indexed the rest. */
    function vacateShopColumn(shopIndex) {
      if (!shopLaneLayout) return;
      const drop = (cols) => cols.map((col) => {
        if (col === null || col.idx === shopIndex) return null;
        return col.idx > shopIndex ? {idx: col.idx - 1, key: col.key} : col;
      });
      shopLaneLayout = {pets: drop(shopLaneLayout.pets), foods: drop(shopLaneLayout.foods)};
    }

    /* Fallback for a shop that shrank without the lane seeing the action (a plain
       refresh, an undo, a buy applied straight through the API): if what is left
       is still the old columns in order, keep them where they are and blank the
       ones that went. Anything else is a new shop and gets a fresh layout. */
    function reconcileShopLaneLayout() {
      if (!shopLaneLayout) return false;
      const source = liveShopSlots();
      if (source.length >= layoutHeld().length) return false;
      const align = (cols, kind) => {
        const incoming = source.filter((slot) => slot.slot_type === kind);
        const out = cols.map(() => null);
        let at = 0;
        for (const slot of incoming) {
          const key = shopSlotKey(slot);
          while (at < cols.length && (cols[at] === null || cols[at].key !== key)) at += 1;
          if (at >= cols.length) return null;
          out[at] = {idx: slot.shop_index, key};
          at += 1;
        }
        return out;
      };
      const pets = align(shopLaneLayout.pets, 'pet');
      const foods = align(shopLaneLayout.foods, 'food');
      if (pets === null || foods === null) return false;
      shopLaneLayout = {pets, foods};
      return true;
    }

    const SHOP_BUY_ACTIONS = new Set(['BUY_PET', 'BUY_COMBINE', 'BUY_FOOD']);

    function noteShopTransition(action) {
      if (!action || typeof action !== 'object') return;
      if (!SHOP_BUY_ACTIONS.has(String(action.type || ''))) return;
      const idx = Number.parseInt(action.shop_index ?? -1, 10);
      if (Number.isNaN(idx) || idx < 0) return;
      vacateShopColumn(idx);
    }

    function buildShopDisplaySlots() {
      if (!appState) return [];
      if (!shopLaneLayoutMatches() && !(reconcileShopLaneLayout() && shopLaneLayoutMatches())) {
        rebuildShopLaneLayout();
      }
      const byIndex = new Map(liveShopSlots().map((slot) => [slot.shop_index, slot]));
      const petCols = shopLaneLayout.pets;
      const foodCols = shopLaneLayout.foods;
      const out = [];
      const push = (col, slotType, displayIndex) => {
        const slot = col === null ? null : byIndex.get(col.idx);
        out.push(slot
          ? {...slot, _placeholder: false, display_index: displayIndex}
          : {_placeholder: true, slot_type: slotType, display_index: displayIndex});
      };
      petCols.forEach((col, i) => push(col, 'pet', i));
      const gaps = Math.max(0, SHOP_LANE_COLUMNS - petCols.length - foodCols.length);
      for (let i = 0; i < gaps; i += 1) out.push({_gap: true});
      foodCols.forEach((col, i) => push(col, 'food', petCols.length + i));
      return out;
    }

    function isRealItem(itemId) {
      return Boolean(itemId) && !String(itemId).endsWith('-none');
    }

    function itemImage(slotType, itemId) {
      if (!itemId) return "";
      return `/api/image?slot_type=${encodeURIComponent(slotType)}&item_id=${encodeURIComponent(itemId)}`;
    }

    /* ----------------------------------------------------------------------
       House-style glyph set. The HUD icons, the slab, the stat badges, the tier
       die, the freeze badge, the level plaque, the ice block and the button
       icons are drawn here in the game's own flat-with-thick-outline style,
       with the colours and proportions measured off the reference pack.
       Nothing is imported from outside the pinned art pack: these exist because
       the pack has no HUD icon sheet, no trophy, no slab and no melon at the
       fidelity of its own Apple. Keylines are pure #000000, like the game's.
       ---------------------------------------------------------------------- */

    /* ---- the slab, one shared asset for every lane slot ----
       An irregular hand-drawn blob at the measured 92x54, with a grey rim that
       is ~2px at the top and sides and ~5px along the bottom, plus a couple of
       chipped nicks scratched into the face. Both lanes and the food slots use
       this exact asset, so no slot is a CSS rounded rectangle. */
    const SLAB_OUTLINE = 'M8.5 14.5C12 6.5 24 2.2 46.5 2c21.5-0.2 34 3.5 38 11.5'
      + 'c4.3 8.5 6 19.5 2.5 28.5C83.5 50.5 69 53.2 45.5 53.2C22.5 53.2 8.2 50.2 5.2 42'
      + 'C2 33.5 5 22 8.5 14.5Z';
    const SLAB_FACE = 'M10.5 15.5C14 8.2 25.5 4.6 46.5 4.4c20.5-0.2 32 3.2 36 10.6'
      + 'c3.8 7.8 5.3 17 2.1 24.2C81.4 45.8 67.5 48 45.5 48C23.5 48 10.6 45.6 7.8 39.2'
      + 'C4.9 32.4 7.3 22.6 10.5 15.5Z';
    /* The viewBox is cropped to the outline's own ink and stretched, so the
       painted slab fills its whole 92x54 box. Measured on the reference at
       1px: outer 92 wide at a 96 pitch (4px of grass between slabs, a
       continuous stone path), face 88 wide, rim 2px on the top and sides and
       5px along the bottom. */
    const SLAB_SVG = '<span class="stone" aria-hidden="true">'
      + '<svg viewBox="3.9 1.9 85.0 51.3" preserveAspectRatio="none">'
      + `<path d="${SLAB_OUTLINE}" fill="#89816b"/>`
      + `<path d="${SLAB_FACE}" fill="#bab391"/>`
      // the pits and dashes chipped into the face, darker than the face the way
      // the art draws them, and weighted to the bottom half
      + '<g fill="none" stroke="#9a9174" stroke-width="2.2" stroke-linecap="round">'
      + '<path d="M62.5 41.5c3.6-0.6 6.4-2 7.8-3.9"/>'
      + '<path d="M16.5 22.5c1.8-2.6 4.4-4.3 7.6-5.1"/>'
      + '<path d="M34 45.4h9.5"/>'
      + '<path d="M74 21.5c1.6 2 2.4 4.2 2.5 6.6"/>'
      + '</g>'
      // selection / drop ring, stroked only through a state class
      + `<path class="slab-body" d="${SLAB_OUTLINE}" fill="none" stroke="none"/>`
      + '</svg></span>';

    /* Every HUD icon is drawn on a 50-unit grid but framed by a viewBox that is
       cropped to its own ink, so the icon fills its 33px badge the way the
       reference's coin fills 32 of its 33px. */
    const COIN_SVG = '<svg class="stat-icon" viewBox="1.5 1.5 47 47" aria-hidden="true">'
      + '<circle cx="25" cy="25" r="21" fill="#ffca49" stroke="#000000" stroke-width="4"/>'
      // rim highlight: the lighter arc the game paints on the coin's upper left
      + '<path d="M10.5 14.5a19 19 0 0 1 11-7.5" fill="none" stroke="#fff0bd"'
      + ' stroke-width="4" stroke-linecap="round"/>'
      + '<circle cx="25" cy="25" r="14.5" fill="#ffac33"/>'
      + '<g fill="#e48300">'
      + '<ellipse cx="25" cy="29" rx="6.8" ry="5.5"/>'
      + '<ellipse cx="18" cy="23" rx="2.8" ry="3.7"/>'
      + '<ellipse cx="22.7" cy="19.4" rx="2.8" ry="3.8"/>'
      + '<ellipse cx="27.9" cy="19.4" rx="2.8" ry="3.8"/>'
      + '<ellipse cx="32.3" cy="23" rx="2.8" ry="3.7"/>'
      + '</g></svg>';

    const HEART_PATH = 'M25 43c-10-7-16.5-13.3-16.5-20.5C8.5 15.2 13.5 10.6 19.2 10.6'
      + 'c3.3 0 5 1.7 5.8 3.3c0.8-1.6 2.5-3.3 5.8-3.3C36.5 10.6 41.5 15.2 41.5 22.5'
      + 'C41.5 29.7 35 36 25 43z';
    const HEART_SVG = '<svg class="stat-icon" viewBox="5.5 6.5 39 39" aria-hidden="true">'
      + `<path d="${HEART_PATH}" fill="#ff0808" stroke="#000000" stroke-width="4" stroke-linejoin="round"/>`
      // white specular dot, top left, exactly like the game's heart
      + '<ellipse cx="17.5" cy="19" rx="3.2" ry="2.6" fill="#ffffff" opacity="0.9"'
      + ' transform="rotate(-25 17.5 19)"/>'
      + '</svg>';

    const VERSUS_SVG = '<svg class="stat-icon" viewBox="5.5 6.5 39 39" aria-hidden="true">'
      + `<path d="${HEART_PATH}" fill="#2f7fc4" stroke="#000000" stroke-width="4" stroke-linejoin="round"/>`
      + '<ellipse cx="17.5" cy="19" rx="3.2" ry="2.6" fill="#ffffff" opacity="0.9"'
      + ' transform="rotate(-25 17.5 19)"/>'
      + '</svg>';
    /* Hourglass, redrawn on the reference: a compact rounded blue cap top and
       bottom, and a bulbous cream body that bows outwards to a narrow waist.
       It is drawn tall and narrow inside the square badge on purpose, which is
       what gives the reference its wider gap before the third chip. */
    const HOURGLASS_SVG = '<svg class="stat-icon" viewBox="0 0 50 50" aria-hidden="true">'
      + '<path d="M13 12C11.2 20.2 24 21.4 25 25C26 28.6 11.2 29.8 13 38L37 38'
      + 'C38.8 29.8 26 28.6 25 25C24 21.4 38.8 20.2 37 12Z"'
      + ' fill="#f9e3b2" stroke="#000000" stroke-width="4" stroke-linejoin="round"/>'
      // the sand: a wedge still held above the waist, a small pile below it
      + '<path d="M16.6 15.4C15.8 20.6 24 22.2 25 24.6C26 22.2 34.2 20.6 33.4 15.4Z" fill="#ffac34"/>'
      + '<path d="M25 30.6C26.6 32.6 32.4 32.8 33.6 35.2L16.4 35.2C17.6 32.8 23.4 32.6 25 30.6Z"'
      + ' fill="#ffac34"/>'
      + '<g fill="#3c88c2" stroke="#000000" stroke-width="3.6" stroke-linejoin="round">'
      + '<rect x="9" y="3.4" width="32" height="10.4" rx="4.6"/>'
      + '<rect x="9" y="36.2" width="32" height="10.4" rx="4.6"/>'
      + '</g>'
      // hard highlight on the top cap, the way the game shades its glass
      + '<path d="M13.6 7.2h9" fill="none" stroke="#9fd2f2" stroke-width="2.8" stroke-linecap="round"/>'
      + '</svg>';
    const TROPHY_SVG = '<svg class="stat-icon" viewBox="6.4 5 37.2 40" aria-hidden="true">'
      + '<g fill="#ffb636" stroke="#000000" stroke-width="4" stroke-linejoin="round" stroke-linecap="round">'
      + '<path d="M15 8.5h20v10a10 10 0 0 1-20 0z"/>'
      + '<path d="M15 11.5H9.4a5.6 5.6 0 0 0 5.6 8.2" fill="none"/>'
      + '<path d="M35 11.5h5.6a5.6 5.6 0 0 1-5.6 8.2" fill="none"/>'
      + '<path d="M22.5 28.5h5v6.5h-5z"/>'
      + '<path d="M16.2 42.5h17.6l-2.5-6.5h-12.6z"/>'
      + '</g>'
      // darker gold shading on the cup's right side, hard highlight on its left
      + '<path d="M30 10.5v8a6 6 0 0 1-3.5 5.5c4-0.4 6.5-3.4 6.5-7.5v-6z" fill="#d98a12"/>'
      + '<path d="M19 12.5v5.5" fill="none" stroke="#ffe6a8" stroke-width="3" stroke-linecap="round"/>'
      + '</svg>';
    const SWORDS_SVG = '<svg class="btn-icon-svg" viewBox="0 0 40 36" aria-hidden="true">'
      + '<g fill="#651f00">'
      + '<path d="M31 2l7 1-1 7-16 16-6-6z"/><path d="M3 27l6-6 6 6-6 6z"/>'
      + '<path d="M9 2L2 3l1 7 16 16 6-6z"/><path d="M37 27l-6-6-6 6 6 6z"/>'
      + '</g></svg>';
    // End turn wears the game's own asymmetric pair: a broad blunt axe blade
    // crossed with a slimmer pointed sword, not two mirrored swords.
    const END_TURN_SVG = '<svg class="btn-icon-svg" viewBox="0 0 44 40" aria-hidden="true">'
      + '<g fill="#651f00">'
      // the axe: a broad blunt head at the top left, haft running to the bottom
      // right, guard bar across its end
      + '<path d="M12.6 5.4L9.5 2.3C5.5 0.5 2 3.7 2.9 7.9c0.8 4.1 4.8 6.1 8.4 4.9z"/>'
      + '<path d="M13.1 4.9L38.1 29.9l-4.2 4.2L8.9 9.1z"/>'
      + '<path d="M37.7 22.3l4 4-11.4 11.4-4-4z"/>'
      // the sword: slimmer, pointed, crossing the other way with its own guard
      // and pommel, drawn over the axe
      + '<path d="M34.5 3.5L41 2.5l-1.5 6-21 21-5-5z"/>'
      + '<path d="M10.2 20L23 32.8l-4.2 4.2L6 24.2z"/>'
      + '<path d="M7 30l6 6-3 3-6-6z"/>'
      + '</g></svg>';
    // Sell wears a flat monochrome price tag, the same ink as the label, so the
    // button keeps one icon system instead of a full-colour coin.
    const SELL_TAG_SVG = '<svg class="btn-icon-svg" viewBox="0 0 44 40" aria-hidden="true">'
      + '<path d="M2.5 20L14 8.5h23.5a4 4 0 0 1 4 4v15a4 4 0 0 1-4 4H14z" fill="#651f00"/>'
      + '<circle cx="16.5" cy="20" r="4" fill="#ff6a00"/>'
      + '</svg>';

    const SNOWFLAKE_PATHS = '<g fill="none" stroke-width="5.2" stroke-linecap="round"'
      + ' stroke-linejoin="round">'
      + '<path d="M26.4 20.0L23.2 14.5L16.8 14.5L13.6 20.0L16.8 25.5L23.2 25.5Z"/>'
      + '<path d="M20.0 15.2 L20.0 3.4 M20.0 10.2 L15.4 6.6 M20.0 4.4 L16.3 1.1'
      + ' M20.0 10.2 L24.6 6.6 M20.0 4.4 L23.7 1.1'
      + ' M15.8 17.6 L5.6 11.7 M11.5 15.1 L6.1 17.3 M6.5 12.2 L1.7 13.7'
      + ' M11.5 15.1 L10.7 9.4 M6.5 12.2 L5.5 7.3'
      + ' M15.8 22.4 L5.6 28.3 M11.5 24.9 L10.7 30.6 M6.5 27.8 L5.5 32.7'
      + ' M11.5 24.9 L6.1 22.7 M6.5 27.8 L1.7 26.3'
      + ' M20.0 24.8 L20.0 36.6 M20.0 29.8 L24.6 33.4 M20.0 35.6 L23.7 38.9'
      + ' M20.0 29.8 L15.4 33.4 M20.0 35.6 L16.3 38.9'
      + ' M24.2 22.4 L34.4 28.3 M28.5 24.9 L33.9 22.7 M33.5 27.8 L38.3 26.3'
      + ' M28.5 24.9 L29.3 30.6 M33.5 27.8 L34.5 32.7'
      + ' M24.2 17.6 L34.4 11.7 M28.5 15.1 L29.3 9.4 M33.5 12.2 L34.5 7.3'
      + ' M28.5 15.1 L33.9 17.3 M33.5 12.2 L38.3 13.7"/>'
      + '</g>';
    const SNOWFLAKE_SVG = '<svg viewBox="-1.5 -1.5 43 43" aria-hidden="true">'
      + `<g class="flake">${SNOWFLAKE_PATHS}</g>`
      + '</svg>';
    // The badge plate both the tier die and the freeze toggle sit on: a 34px
    // near-square with a hand-drawn wobble, radius 7 (the measured ~20%), a 3px
    // black keyline and the 3px white sticker rim the game draws outside it.
    const BADGE_PLATE_PATH = 'M10.5 3.4L29.8 3c4.4 0 7.2 3.2 7.2 7.6l-0.4 18.8'
      + 'c0 4.6-3 7.4-7.4 7.5l-18.8 0.1C6 37 3.1 34 3 29.6L3.2 10.4C3.2 6 6.1 3.5 10.5 3.4Z';
    // Frozen unit: a chunky flat faceted ice block drawn IN FRONT of the pet.
    // The pet keeps every one of its own colours; the block is translucent so it
    // reads through, has a pure black keyline and hard white highlight streaks,
    // and carries no blur, no glow, no grayscale and no hairline ring.

    const ICE_SILHOUETTE = 'M54 4L86 14L100 46L94 92L62 114L30 110L6 84L4 38L28 10Z';
    const ICE_BLOCK_SVG = '<svg class="ice-block" viewBox="0 0 104 118" aria-hidden="true">'
      + `<path d="${ICE_SILHOUETTE}" fill="rgba(150,222,247,0.15)"/>`
      // two flat planes across the shard, each its own pale tone
      + '<path d="M28 10L54 4L46 60L14 70L4 38Z" fill="rgba(228,249,255,0.15)"/>'
      + '<path d="M54 4L86 14L100 46L94 92L58 78L46 60Z" fill="rgba(96,192,234,0.14)"/>'
      + '<g fill="none" stroke="#000000" stroke-width="5" stroke-linejoin="round"'
      + ' stroke-linecap="round">'
      + `<path d="${ICE_SILHOUETTE}"/>`
      + '<path d="M54 4L46 60L14 70"/>'
      + '<path d="M46 60L58 78L94 92"/>'
      + '<path d="M46 60L60 114"/>'
      + '</g>'
      // hard white highlight slivers, the way the art shades glass
      + '<g fill="#ffffff" opacity="0.6">'
      + '<path d="M26 24l6 2-3 34-6-2.5z"/>'
      + '<path d="M77 29l5 2-2.5 22-5-2.5z"/>'
      + '<path d="M69 91l4.5 1.5-2.5 13-4.5-2z"/>'
      + '</g></svg>';
    // Attack / health badges: sticker plates on the slab's front edge, white
    // numeral with a black outline, flat fill, no radial shading. Attack is an
    // irregular hand-drawn rock, not a regular polygon.
    /* The attack plate is a chipped rock, not a regular polygon: ten edges of
       deliberately unequal length, and a darker facet down its right side so
       the stone has a light/dark tonal split the way the reference paints it. */
    const ATK_ROCK_PATH = 'M5.4 17.6L11.2 7.6L21.4 4.0L30.8 5.2L38.6 11.0L42.2 20.6'
      + 'L38.0 29.8L31.2 36.6L21.4 39.0L12.0 34.4L6.4 26.2Z';
    const ATK_ROCK_FACET = 'M42.2 20.6L38.0 29.8L31.2 36.6L21.4 39.0L25.4 27.2'
      + 'L34.6 9.4L38.6 11.0Z';
    const HP_HEART_PATH = 'M23 37.5C12 30 6.5 25 6.5 17.6C6.5 11.6 11 7.8 15.8 7.8'
      + 'c2.8 0 4.8 1.6 5.9 3.2c1.1-1.6 3.1-3.2 5.9-3.2c4.8 0 9.3 3.8 9.3 9.8'
      + 'C38.9 25 34 30 23 37.5z';
    function statBadgeSVG(kind, value) {
      const num = String(value);
      const path = kind === 'atk' ? ATK_ROCK_PATH : HP_HEART_PATH;
      const fill = kind === 'atk' ? '#7a7a7a' : '#e5332e';
      const facet = kind === 'atk'
        ? `<path d="${ATK_ROCK_FACET}" fill="#575757"/>`
        : '';
      const cy = kind === 'atk' ? 23 : 24;
      return `<svg class="badge badge-${kind}" viewBox="0 0 46 42" aria-hidden="true">`
        // white sticker rim, then the plate, then the numeral
        + `<path d="${path}" fill="none" stroke="#ffffff" stroke-width="9.5" stroke-linejoin="round"/>`
        + `<path d="${path}" fill="${fill}" stroke="#000000" stroke-width="3.6" stroke-linejoin="round"/>`
        + facet
        + `<text x="23" y="${cy}" text-anchor="middle" dominant-baseline="middle"`
        + ' font-family="SAPUI, sans-serif" font-weight="700" font-size="22"'
        + ' paint-order="stroke" stroke="#000000" stroke-width="4.5" stroke-linejoin="round"'
        + ` fill="#ffffff">${num}</text></svg>`;
    }

    // Tier die: white face, black pips, 3px black keyline inside the 3px white
    // sticker rim the reference draws around it.
    const DIE_PIPS = {
      1: [[20, 20]],
      2: [[13, 13], [27, 27]],
      3: [[13, 13], [20, 20], [27, 27]],
      4: [[13, 13], [27, 13], [13, 27], [27, 27]],
      5: [[13, 13], [27, 13], [20, 20], [13, 27], [27, 27]],
      6: [[13, 12], [27, 12], [13, 20], [27, 20], [13, 28], [27, 28]],
    };
    function dieSVG(tier) {
      const t = Math.max(1, Math.min(6, Number.parseInt(tier ?? 1, 10) || 1));
      const pips = (DIE_PIPS[t] || DIE_PIPS[1])
        .map(([cx, cy]) => `<circle cx="${cx}" cy="${cy}" r="3.2" fill="#000000"/>`)
        .join('');
      return `<svg class="die" viewBox="0 0 40 40" aria-hidden="true" title="tier ${t}">`
        + `<path class="die-rim" d="${BADGE_PLATE_PATH}" fill="#ffffff" stroke="#ffffff" stroke-width="6" stroke-linejoin="round"/>`
        + `<path class="die-plate" d="${BADGE_PLATE_PATH}" fill="#ffffff" stroke="#000000" stroke-width="3" stroke-linejoin="round"/>`
        + pips + '</svg>';
    }

    /* Freeze affordance, folded onto the tier die: a small snowflake mark in
       the die's bottom-right corner, drawn on top of the die's own plate so the
       tier's pips stay readable. It appears while the slot is hovered and stays
       while the slot is frozen, when the die's plate also fills with ice. */
    function freezeBadgeSVG() {
      return '<svg viewBox="0 0 40 40" aria-hidden="true">'
        + '<g transform="translate(34.5 34.5) scale(0.5) translate(-20 -20)">'
        + `<g class="flake-halo">${SNOWFLAKE_PATHS}</g>`
        + `<g class="flake">${SNOWFLAKE_PATHS}</g>`
        + '</g></svg>';
    }

    // Roll's cost die: the game's dark-brown near-square, radius ~4 on 34 (the
    // measured 12%) with a hand-drawn wobble, and a single pip punched through
    // to the button's own orange.
    const ROLL_DIE_SVG = '<svg viewBox="0 0 34 33" aria-hidden="true">'
      + '<path d="M4.6 1.2L29.4 0.8c2.4 0 3.8 1.6 3.8 4l-0.4 23.6c0 2.4-1.4 3.8-3.8 3.9'
      + 'L4.8 32.5C2.4 32.5 0.9 31 0.8 28.6L1 4.6C1 2.2 2.2 1.3 4.6 1.2Z" fill="#651f00"/>'
      + '<circle cx="17" cy="16.5" r="4.6" fill="#ff6a00"/>'
      + '</svg>';


    const LVL_PLAQUE_PATH = 'M6.4 11.8L41.4 10.2C44.1 10.1 45.3 11.5 45.1 14.3L44.3 39.4'
      + 'C44.2 42.1 42.5 43.1 40.0 42.9L7.2 42.1C4.4 42.0 3.0 40.7 3.2 38.0L3.7 14.4'
      + 'C3.8 11.9 4.7 11.9 6.4 11.8Z';
    function lvlPlaqueSVG(prog) {
      const segs = [];
      const total = prog.required > 0 ? prog.required : 0;
      if (total > 0) {
        const x0 = 3.5;
        const x1 = 44.5;
        const gap = 1.0;
        const w = (x1 - x0 - gap * (total - 1)) / total;
        for (let i = 0; i < total; i += 1) {
          const filled = i < prog.current;
          segs.push(
            `<rect x="${(x0 + i * (w + gap)).toFixed(1)}" y="29" width="${w.toFixed(1)}"`
            + ` height="17" rx="5" fill="${filled ? '#ffca49' : '#7a4a12'}"`
            + ' stroke="#000000" stroke-width="2"/>'
          );
        }
      } else {
        segs.push('<text x="24" y="41" text-anchor="middle" font-family="SAPUI, sans-serif"'
          + ' font-weight="700" font-size="13" fill="#ffca49">MAX</text>');
      }
      return '<svg class="lvl-plaque" viewBox="0 0 48 48" aria-hidden="true">'
        + `<path d="${LVL_PLAQUE_PATH}" fill="none" stroke="#ffffff" stroke-width="6" stroke-linejoin="round"/>`
        + `<path d="${LVL_PLAQUE_PATH}" fill="#000000" stroke="#000000" stroke-width="2" stroke-linejoin="round"/>`
        + '<text x="6.5" y="25" font-family="SAPUI, sans-serif" font-weight="700" font-size="12"'
        + ' fill="#ffffff">Lvl</text>'
        // drawn last and tall enough to break above the plaque's own top edge
        + `<text x="42" y="25" text-anchor="end" font-family="SAPUI, sans-serif" font-weight="700"`
        + ` font-size="32" paint-order="stroke" stroke="#000000" stroke-width="3.5"`
        + ` stroke-linejoin="round" fill="#ffca49">${prog.level}</text>`
        + segs.join('')
        + '</svg>';
    }

    // One SAP HUD chip: a white rounded-rectangle plate with the icon badge
    // riding across its left edge, breaking the outline above and below.
    function hudPill(spec) {
      const value = spec.value ?? 'n/a';
      const title = spec.title || '';
      return `<span class="stat-chip" title="${title}"><span class="stat-badge">${spec.svg}</span>`
        + `<span class="stat-val">${value}</span></span>`;
    }


    function spriteClass(slotType, location) {
      if (slotType !== 'pet') return 'sprite';
      return location === 'enemy' ? 'sprite' : 'sprite mirror-x';
    }

    // Sprite markup: the pack's own art, straight off `/api/image`.
    function spriteHTML(slotType, itemId, location) {
      return `<img class="${spriteClass(slotType, location || 'any')}" src="${itemImage(slotType, itemId)}"`
        + ` onerror="this.style.display='none';this.nextElementSibling.style.display='grid';" />`
        + `<div class="fallback" style="display:none;">${itemId}</div>`;
    }

    function itemTier(slotType, itemId) {
      if (!appState || !itemId) return 1;
      const cat = appState.debug_catalog || {};
      const rows = slotType === 'food' ? cat.foods : cat.pets;
      if (!Array.isArray(rows)) return 1;
      const hit = rows.find((entry) => String(entry.item_id) === String(itemId));
      return hit ? (Number.parseInt(hit.tier ?? 1, 10) || 1) : 1;
    }

    // Attack and health sit as sticker plates on the stone's front edge, baseline
    // aligned side by side below the unit, the way the real shop draws them.
    function petStatsHTML(attack, health) {
      return `
        <div class="badges" title="attack ${attack} / health ${health}">
          ${statBadgeSVG('atk', attack)}${statBadgeSVG('hp', health)}
        </div>
      `;
    }

    function prettyItemName(itemId) {
      const raw = String(itemId || '');
      const body = raw.replace(/^(pet|food|status)-/, '');
      if (!body) return '';
      return body
        .split('-')
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
    }

    const STATUS_TO_FOOD_ICON = {
      'status-honey-bee': 'food-honey',
      'status-bone-attack': 'food-meat-bone',
      'status-garlic-armor': 'food-garlic',
      'status-splash-attack': 'food-chili',
      'status-melon-armor': 'food-melon',
      'status-extra-life': 'food-mushroom',
      'status-steak-attack': 'food-steak',
      'status-peanut': 'food-peanut',
      'status-coconut-shield': 'food-coconut',
    };

    function displayEquipmentFromSlot(slot) {
      const explicit = String(slot?.equipment_id || '').trim();
      if (explicit) {
        return {
          iconItemId: explicit,
          iconSlotType: 'food',
          label: prettyItemName(explicit) || explicit,
          title: explicit,
        };
      }
      const effects = Array.isArray(slot?.status_effects) ? slot.status_effects.map((x) => String(x)) : [];
      for (const effectId of effects) {
        const mappedFood = STATUS_TO_FOOD_ICON[effectId];
        if (mappedFood) {
          return {
            iconItemId: mappedFood,
            iconSlotType: 'food',
            label: prettyItemName(mappedFood) || mappedFood,
            title: effectId,
          };
        }
      }
      return null;
    }

    function equipmentHTML(slot) {
      if (!slot || !slot.pet_id) return '';
      const display = displayEquipmentFromSlot(slot);
      if (!display) return '';
      const icon = itemImage(display.iconSlotType, display.iconItemId);
      return `
        <div class="equip-row" title="${display.title}">
          <span class="equip-chip">
            <img src="${icon}" alt="${display.iconItemId}" onerror="this.style.display='none';" />
            <span>${display.label}</span>
          </span>
        </div>
      `;
    }

    function levelProgress(level, exp) {
      const lv = Math.max(1, Math.min(3, Number.parseInt(level ?? 1, 10) || 1));
      const ex = Math.max(0, Math.min(5, Number.parseInt(exp ?? 0, 10) || 0));
      if (lv >= 3) {
        return {level: lv, exp: ex, current: 0, required: 0, pct: 100, label: 'MAX'};
      }
      const start = lv === 1 ? 0 : 2;
      const end = lv === 1 ? 2 : 5;
      const required = end - start;
      const current = Math.max(0, Math.min(required, ex - start));
      const pct = required > 0 ? Math.round((current / required) * 100) : 100;
      return {level: lv, exp: ex, current, required, pct, label: `${current}/${required}`};
    }

    // The game's black level plaque, overlapping the pet's top-left corner:
    // small white "Lvl", large gold numeral, brown segmented experience bar.
    function levelRowHTML(level, exp) {
      const prog = levelProgress(level, exp);
      return lvlPlaqueSVG(prog);
    }

    function teamSellValue(slot) {
      if (!slot || !slot.pet_id) return 0;
      const raw = Number.parseInt(slot.sell_value ?? 1, 10);
      if (Number.isNaN(raw)) return 1;
      return Math.max(1, raw);
    }

    const TRIGGER_COUNTER_LABELS = {
      friend_faints: 'Fly',
      buy_tier1_pet: 'Dragon',
      purchase_food: 'Cat',
      friendly_ate_food: 'Rabbit',
      hurt: 'Gorilla',
      friend_ahead_faints: 'Ox',
      bison_team_end_of_turn: 'Bison',
    };

    function triggerCounterLabel(trigger) {
      const key = String(trigger || '');
      if (TRIGGER_COUNTER_LABELS[key]) return TRIGGER_COUNTER_LABELS[key];
      return key.replaceAll('_', ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
    }

    function triggerLimitsForSlot(slot) {
      const petId = String(slot?.pet_id || '');
      const level = Math.max(1, Math.min(3, Number.parseInt(slot?.level ?? 1, 10) || 1));
      if (!petId) return {};
      if (petId === 'pet-fly') return {friend_faints: 3};
      if (petId === 'pet-dragon') return {buy_tier1_pet: 4};
      if (petId === 'pet-cat') return {purchase_food: 2};
      if (petId === 'pet-rabbit') return {friendly_ate_food: 3};
      if (petId === 'pet-ox') return {friend_ahead_faints: level};
      if (petId === 'pet-gorilla') return {hurt: level};
      return {};
    }

    function teamAbilityCounters(teamIndex) {
      if (!appState || !appState.state || !appState.state.meta) return [];
      const slot = teamSlot(teamIndex);
      if (!slot || !slot.pet_id) return [];
      const counters = appState.state.meta.ability_counters;
      if (!counters || typeof counters !== 'object') return [];
      const key = String(teamIndex);
      const limits = triggerLimitsForSlot(slot);
      const out = [];

      Object.entries(limits).forEach(([trigger, maxTriggers]) => {
        const byTeam = counters && typeof counters === 'object' ? counters[trigger] : null;
        const consumedRaw = byTeam && typeof byTeam === 'object' ? byTeam[key] : 0;
        const consumed = Math.max(0, Number.parseInt(consumedRaw ?? 0, 10) || 0);
        const maxVal = Math.max(0, Number.parseInt(maxTriggers ?? 0, 10) || 0);
        const left = Math.max(0, maxVal - consumed);
        out.push({
          trigger: String(trigger),
          value: left,
          mode: 'left',
          consumed,
          limit: maxVal,
        });
      });

      Object.entries(counters).forEach(([trigger, byTeam]) => {
        if (Object.prototype.hasOwnProperty.call(limits, String(trigger))) return;
        if (!byTeam || typeof byTeam !== 'object') return;
        const rawValue = byTeam[key];
        const value = Number.parseInt(rawValue ?? 0, 10);
        if (Number.isNaN(value) || value <= 0) return;
        out.push({trigger: String(trigger), value, mode: 'used'});
      });
      out.sort((a, b) => a.trigger.localeCompare(b.trigger));
      return out;
    }

    function teamCounterRowHTML(teamIndex) {
      const counters = teamAbilityCounters(teamIndex);
      if (!counters.length) return '';
      const chips = counters
        .map((entry) => {
          if (entry.mode === 'left') {
            return `<span class="ability-counter-chip" title="${entry.trigger}: used ${entry.consumed}/${entry.limit}">${triggerCounterLabel(entry.trigger)} Left: ${entry.value}</span>`;
          }
          return `<span class="ability-counter-chip" title="${entry.trigger}">${triggerCounterLabel(entry.trigger)}: ${entry.value}</span>`;
        })
        .join('');
      return `<div class="ability-counter-row">${chips}</div>`;
    }

    function basePetStats(itemId) {
      if (!appState) return {attack: 0, health: 0};
      const stats = (appState.catalog_base_stats || {})[itemId] || {};
      return {
        attack: Number.parseInt(stats.attack ?? 0, 10) || 0,
        health: Number.parseInt(stats.health ?? 0, 10) || 0,
      };
    }

    function teamPopupHTML(slot) {
      if (!appState) return '';
      const idx = slot.slot_index;
      const occupied = Boolean(slot.pet_id);
      if (!occupied || selectedTeamIndex !== idx) return '';

      const combineActions = appState.legal_actions
        .map((entry) => entry.action)
        .filter((action) => action.type === 'COMBINE' && action.src_team_index === idx);
      // Only a real affordance gets a popup. What the pet IS now reads off the
      // selection info card, not off a "No combine targets." line on the grass.
      if (!combineActions.length) return '';
      const combineButtons = combineActions.map((action) => (
        `<button class="mini-btn" data-action="combine" data-src="${idx}" data-dst="${action.dst_team_index}">Combine -> ${action.dst_team_index}</button>`
      )).join('');

      return `
        <div class="popup-controls">
          ${combineButtons}
        </div>
      `;
    }

    function shopPopupHTML(slot) {
      return '';
    }

    // Read-only boards share the shop's sprites and badges, but not its
    // selection, drag state or ability counters. Used by both end screens.
    function readOnlyBoardCardHTML(slot, side) {
      const enemy = side === 'ai';
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

    function renderReadOnlyBoard(el, team, side) {
      if (!el) return 0;
      const enemy = side === 'ai';
      const slots = Array.isArray(team) ? team.slice() : [];
      // Human slot 0 faces right; the opponent's slot 0 faces left.
      slots.sort((a, b) => ((a.slot_index ?? 0) - (b.slot_index ?? 0)) * (enemy ? 1 : -1));
      el.innerHTML = slots.map((slot) => readOnlyBoardCardHTML(slot, side)).join('');
      if (el.dataset) el.dataset.side = enemy ? 'ai' : 'human';
      return slots.filter((slot) => slot && slot.pet_id).length;
    }

    function teamCardHTML(slot) {
      const itemId = slot.pet_id;
      const index = slot.slot_index;
      const fx = teamFxBySlot[index] || null;
      const fxClasses = [
        fx && fx.buff ? ' fx-buff' : '',
        fx && fx.levelup ? ' fx-levelup' : '',
      ].join('');
      const selectedCls = selectedTeamIndex === index ? ' selected' : '';
      const pendingTargetCls = pendingFoodTargetShopIndex !== null && itemId ? ' pending-food-target' : '';
      const atk = slot.attack ?? 0;
      const hp = slot.health ?? 0;
      const level = slot.level ?? 1;
      const exp = slot.exp ?? 0;
      const draggable = itemId ? 'true' : 'false';
      const emptyCls = itemId ? '' : ' is-empty';
      const levelRow = itemId ? levelRowHTML(level, exp) : '';
      const counterRow = itemId ? teamCounterRowHTML(index) : '';
      const image = itemId ? spriteHTML('pet', itemId) : '';
      const statsRow = itemId ? petStatsHTML(atk, hp) : '';
      const equipmentRow = itemId ? equipmentHTML(slot) : '';
      const sellValue = itemId ? teamSellValue(slot) : 0;
      const fxTags = fx
        ? `<div class="fx-tags">
             ${fx.levelup ? '<span class="fx-tag levelup">LEVEL UP</span>' : ''}
             ${
               fx.buff
                 ? `
                   ${Number.parseInt(fx.atkPlus ?? 0, 10) > 0 ? `<span class="fx-tag stat">${statBadgeSVG('atk', `+${Number.parseInt(fx.atkPlus ?? 0, 10)}`)}</span>` : ''}
                   ${Number.parseInt(fx.hpPlus ?? 0, 10) > 0 ? `<span class="fx-tag stat">${statBadgeSVG('hp', `+${Number.parseInt(fx.hpPlus ?? 0, 10)}`)}</span>` : ''}
                 `
                 : ''
             }
           </div>`
        : '';
      const tip = itemId
        ? `${prettyItemName(itemId)} ${atk}/${hp}, Lvl ${slot.level ?? 1}, sells for ${sellValue} gold`
        : `empty slot ${index}`;
      return `
        <div class="card team-card${selectedCls}${pendingTargetCls}${fxClasses}${emptyCls}" data-slot-kind="team" data-team-index="${index}" draggable="${draggable}" title="${tip}">
          ${SLAB_SVG}
          ${levelRow}
          ${counterRow}
          ${image}
          ${statsRow}
          ${equipmentRow}
          ${fxTags}
          ${teamPopupHTML(slot)}
        </div>
      `;
    }

    function shopCardHTML(slot) {
      const isPlaceholder = Boolean(slot._placeholder);
      const itemId = slot.item_id;
      const idx = slot.shop_index;
      const displayIndex = Number.isInteger(slot.display_index) ? slot.display_index : idx;
      const teamPetIds = new Set(
        (appState && appState.state && appState.state.team ? appState.state.team : [])
          .filter((teamSlot) => Boolean(teamSlot.pet_id))
          .map((teamSlot) => String(teamSlot.pet_id))
      );
      const copyOnBoard = !isPlaceholder
        && slot.slot_type === 'pet'
        && isRealItem(itemId)
        && teamPetIds.has(String(itemId));
      const selectedCls = !isPlaceholder && selectedShopIndex === idx ? ' selected' : '';
      const copyCls = copyOnBoard ? ' copy-on-board' : '';
      const draggable = !isPlaceholder && isRealItem(itemId) ? 'true' : 'false';
      const linkTitle = !isPlaceholder && slot.link_id ? ` link=${slot.link_id}` : '';
      const frozen = !isPlaceholder && Boolean(slot.frozen);
      const frozenCls = frozen ? ' is-frozen' : '';
      const emptyCls = isPlaceholder || !isRealItem(itemId) ? ' is-empty' : '';
      const real = !isPlaceholder && isRealItem(itemId);
      const copyBadge = copyOnBoard ? `<span class="copy-badge" title="Same pet exists on your team">COPY</span>` : '';
      const image = real ? spriteHTML(slot.slot_type, itemId) : '';
      let shopStats = null;
      if (real && slot.slot_type === 'pet') {
        if (Number.isFinite(Number(slot.attack)) && Number.isFinite(Number(slot.health))) {
          shopStats = {
            attack: Number.parseInt(slot.attack ?? 0, 10) || 0,
            health: Number.parseInt(slot.health ?? 0, 10) || 0,
          };
        } else {
          shopStats = basePetStats(itemId);
        }
      }
      // Two stories only: the tier die over the unit, the unit with its foot
      // badges on the slab. Price lives in the hover badge, the name and the
      // ability in the selection info card.
      const die = real ? dieSVG(itemTier(slot.slot_type, itemId)) : '';
      // Frozen: a flat faceted block in front of the unit. The unit itself is
      // not tinted, masked, greyed or ringed.
      const iceLayers = frozen ? ICE_BLOCK_SVG : '';
      const controls = real
        ? `
          <button class="freeze-toggle" data-action="freeze-shop" data-shop-index="${idx}" title="${frozen ? 'Unfreeze' : 'Freeze'} this slot">
            ${freezeBadgeSVG()}
          </button>
          <button class="buy-chip" data-action="buy-shop" data-shop-index="${idx}">
            ${COIN_SVG.replace('class="stat-icon"', '')}${slot.cost}
          </button>
          ${shopPopupHTML(slot)}
        `
        : '';
      const tip = real
        ? `Buy ${prettyItemName(itemId)} for ${slot.cost} gold`
          + `${shopStats ? ` (${shopStats.attack}/${shopStats.health})` : ''}`
          + `${frozen ? ', frozen' : ''}${linkTitle}`
        : `empty slot ${displayIndex}`;
      const spriteUrl = real ? `--sprite-url:url(${itemImage(slot.slot_type, itemId)});` : '';
      return `
        <div class="card shop-card${selectedCls}${copyCls}${frozenCls}${emptyCls}" data-slot-kind="shop" data-shop-index="${idx ?? ''}" data-shop-slot-type="${slot.slot_type}" draggable="${draggable}" title="${tip}" style="${spriteUrl}">
          ${SLAB_SVG}
          ${copyBadge}
          ${die}
          ${image}
          ${iceLayers}
          ${shopStats ? petStatsHTML(shopStats.attack, shopStats.health) : ''}
          ${controls}
        </div>
      `;
    }

    function clearDropHighlights() {
      document.querySelectorAll('.drop-hover').forEach((el) => el.classList.remove('drop-hover'));
      const contextBtn = document.getElementById('btn-context-action');
      if (contextBtn) {
        contextBtn.classList.remove('drop-hover');
      }
    }

    function populateDebugSelectors() {
      if (!appState) return;
      const debugCatalog = appState.debug_catalog || {};
      const petSelect = document.getElementById('debug-pet-select');
      const foodSelect = document.getElementById('debug-food-select');
      const petSearch = document.getElementById('debug-pet-search');
      const foodSearch = document.getElementById('debug-food-search');
      if (!petSelect || !foodSelect || !petSearch || !foodSearch) return;

      const pets = Array.isArray(debugCatalog.pets) ? debugCatalog.pets : [];
      const foods = Array.isArray(debugCatalog.foods) ? debugCatalog.foods : [];

      const prettyName = (itemId) => {
        const raw = String(itemId || '');
        const body = raw.replace(/^(pet|food)-/, '');
        return body.split('-').filter(Boolean).map((part) => (
          part.charAt(0).toUpperCase() + part.slice(1)
        )).join(' ');
      };

      const petQuery = String(petSearch.value || '').trim().toLowerCase();
      const foodQuery = String(foodSearch.value || '').trim().toLowerCase();

      const petFiltered = pets.filter((entry) => {
        const itemId = String(entry.item_id || '');
        const name = prettyName(itemId).toLowerCase();
        return !petQuery || itemId.toLowerCase().includes(petQuery) || name.includes(petQuery);
      });
      const foodFiltered = foods.filter((entry) => {
        const itemId = String(entry.item_id || '');
        const name = prettyName(itemId).toLowerCase();
        return !foodQuery || itemId.toLowerCase().includes(foodQuery) || name.includes(foodQuery);
      });

      const petMarkup = petFiltered.map((entry) => {
        const itemId = String(entry.item_id || '');
        const tier = Number.parseInt(entry.tier ?? 0, 10) || 0;
        return `<option value="${itemId}">T${tier} ${prettyName(itemId)} (${itemId})</option>`;
      }).join('');
      const foodMarkup = foodFiltered.map((entry) => {
        const itemId = String(entry.item_id || '');
        const tier = Number.parseInt(entry.tier ?? 0, 10) || 0;
        return `<option value="${itemId}">T${tier} ${prettyName(itemId)} (${itemId})</option>`;
      }).join('');

      const prevPetValue = petSelect.value;
      const prevFoodValue = foodSelect.value;
      petSelect.innerHTML = petMarkup;
      foodSelect.innerHTML = foodMarkup;
      if (petFiltered.some((entry) => String(entry.item_id) === prevPetValue)) {
        petSelect.value = prevPetValue;
      } else if (!debugSelectorsBootstrapped && petFiltered.length > 0) {
        petSelect.value = String(petFiltered[0].item_id || '');
      }
      if (foodFiltered.some((entry) => String(entry.item_id) === prevFoodValue)) {
        foodSelect.value = prevFoodValue;
      } else if (!debugSelectorsBootstrapped && foodFiltered.length > 0) {
        foodSelect.value = String(foodFiltered[0].item_id || '');
      }
      debugSelectorsBootstrapped = true;
    }

    function renderState() {
      // `state: null` is a real answer, not a bug: the duel page (/play) asks
      // for state before a game exists, and gets the catalog and an empty
      // board rather than an error. Its own chrome still repaints.
      if (!appState || !appState.state) {
        afterRender();
        return;
      }
      const s = appState.state;
      const now = Date.now();
      Object.keys(teamFxBySlot).forEach((key) => {
        const idx = Number.parseInt(key, 10);
        if (Number.isNaN(idx)) return;
        const fx = teamFxBySlot[idx];
        if (!fx || Number(fx.until || 0) < now) {
          delete teamFxBySlot[idx];
        }
      });

      if (selectedShopIndex !== null && !shopSlot(selectedShopIndex)) {
        selectedShopIndex = null;
      }
      if (pendingFoodTargetShopIndex !== null && !shopSlot(pendingFoodTargetShopIndex)) {
        pendingFoodTargetShopIndex = null;
      }

      const gameMode = String(appState.game_mode || "arena");
      const versusMeta = (s.meta && typeof s.meta === "object" && s.meta.versus && typeof s.meta.versus === "object") ? s.meta.versus : null;
      const opponentLives = versusMeta ? versusMeta.opponent_lives : null;
      // Fourth HUD chip: trophies in arena, the opponent's remaining lives in
      // versus (there is no trophy race there). Same chip shape either way, with
      // the versus heart carrying the crossed-swords mark so the two never read
      // alike.
      const scorePill = gameMode === "versus"
        ? hudPill({svg: VERSUS_SVG, value: opponentLives ?? 'n/a', title: 'opponent lives'})
        : hudPill({svg: TROPHY_SVG, value: s.trophies, title: 'trophies'});

      setHTML('stats', [
        hudPill({svg: COIN_SVG, value: s.gold, title: 'gold'}),
        hudPill({svg: HEART_SVG, value: s.lives, title: 'lives'}),
        hudPill({svg: HOURGLASS_SVG, value: s.turn, title: 'turn'}),
        scorePill,
      ].join(''));
      const modeTag = document.getElementById('mode-tag');
      if (modeTag) {
        modeTag.innerHTML = gameMode === 'versus' ? SWORDS_SVG : TROPHY_SVG;
        modeTag.title = `game mode: ${gameMode}`;
      }
      renderBuildLine();
      const endIcon = document.getElementById('end-turn-icon');
      if (endIcon && !endIcon.dataset.ready) {
        endIcon.innerHTML = END_TURN_SVG;
        endIcon.dataset.ready = '1';
      }
      const rollDie = document.getElementById('roll-cost-die');
      if (rollDie && !rollDie.dataset.ready) {
        rollDie.innerHTML = ROLL_DIE_SVG;
        rollDie.dataset.ready = '1';
      }

      const visualTeam = [...s.team].sort((a, b) => b.slot_index - a.slot_index);
      setHTML('team', visualTeam.map((slot) => teamCardHTML(slot)).join(''));
      const displayShop = buildShopDisplaySlots();
      setHTML('shop', displayShop
        .map((slot) => (slot._gap ? '<span class="shop-gap"></span>' : shopCardHTML(slot)))
        .join(''));

      const filterEl = document.getElementById('action-filter');
      const filter = filterEl ? filterEl.value : 'ALL';
      const legal = appState.legal_actions.filter((a) => filter === 'ALL' || a.action.type === filter);
      setHTML('legal-actions', legal.map((a) => (
        `<button class="action-item" data-index="${a.index}" data-kind="legal-action">#${a.index} ${prettyAction(a.action)}</button>`
      )).join(''));

      const hist = appState.history || [];
      setHTML('history', hist.length
        ? hist.map((h, i) => `${i}: ${prettyAction(h.action)} | legal=${h.legal} | deterministic=${h.deterministic} | notes=${(h.engine_notes || []).join('|')}`).join('<br/>')
        : '<span class="tiny">No actions yet.</span>');

      const battle = appState.last_battle || null;
      const battleEl = document.getElementById('last-battle');
      if (battleEl) {
        if (!battle) {
          battleEl.innerHTML = '<span class="tiny">No battle yet.</span>';
        } else {
          // The human line stays in the game's display face; every id, timing
          // and parse flag drops into the parchment console in a small mono
          // face, where a UUID is readable instead of decorative.
          const link = battle.calculator_link
            ? `<a class="plaque-link" href="${battle.calculator_link}" target="_blank" rel="noopener"><span>Open SAP-Calculator Link</span></a>`
            : '<span class="hint">Calculator link unavailable.</span>';
          const sampledTurnImage = battle.sampled_turn_image_url
            ? `<div style="margin-top:8px;"><span class="hint">Sampled Turn Image</span></div><div class="battle-replay-wrap"><a href="${battle.sampled_turn_image_url}" target="_blank" rel="noopener"><img class="battle-replay-image" src="${battle.sampled_turn_image_url}" alt="Sampled single-turn image" /></a></div>`
            : '<div style="margin-top:8px;"><span class="hint">Sampled turn image unavailable.</span></div>';
          const sessionReplayImage = battle.session_replay_image_url
            ? `<div style="margin-top:8px;"><span class="hint">Session Replay</span></div><div class="battle-replay-wrap"><a href="${battle.session_replay_image_url}" target="_blank" rel="noopener"><img class="battle-replay-image" src="${battle.session_replay_image_url}" alt="Session replay image" /></a></div>`
            : '<div style="margin-top:8px;"><span class="hint">Session replay image unavailable.</span></div>';
          const timing = (battle.timing_ms && typeof battle.timing_ms === 'object') ? battle.timing_ms : null;
          const consoleRows = [
            `forced_pid          ${battle.forced_pid || 'none'}`,
            `opponent_source     ${battle.opponent_source || 'n/a'}`,
            `opponent_replay_id  ${battle.opponent_replay_id || 'n/a'}`,
            `opponent_side       ${battle.opponent_side || 'n/a'}`,
            `opponent_pid        ${battle.opponent_participation_id || 'n/a'}`,
            `opponent_pack       ${battle.opponent_pack || 'n/a'}`,
            `opponent_error      ${battle.opponent_error || 'none'}`,
            `oracle              ${battle.oracle_ok ? 'ok' : 'failed'}${battle.oracle_error ? ` (${battle.oracle_error})` : ''}`,
            battle.live_db_fallback
              ? `live_db_fallback    yes (${battle.live_db_fallback_route || 'unknown'}; trigger=${battle.live_db_fallback_trigger_error || battle.live_db_fallback_reason || 'snapshot_exhausted'})`
              : 'live_db_fallback    no',
            timing
              ? `timing_ms           total=${timing.total ?? 'n/a'} sample=${timing.sample ?? 'n/a'} parse=${timing.parse ?? 'n/a'} battle=${timing.battle ?? 'n/a'}`
              : 'timing_ms           n/a',
            `parse_mode          ${battle.parse_mode || 'n/a'}`,
            `parse_error         ${battle.parse_error || 'none'}`,
          ].map((row) => `<div>${row}</div>`).join('');
          battleEl.innerHTML = [
            `<div class="battle-headline"><span>Turn ${battle.turn}</span><span>Result: ${battle.result}</span>${link}</div>`,
            `<div class="console-block">${consoleRows}</div>`,
            sampledTurnImage,
            sessionReplayImage,
          ].join('');
        }
      }
      populateDebugSelectors();

      const rollBtn = document.getElementById('btn-roll');
      const endBtn = document.getElementById('btn-end-turn');
      const contextBtn = document.getElementById('btn-context-action');
      const contextLabel = document.getElementById('btn-context-label');
      const contextIcon = document.getElementById('btn-context-icon');
      if (rollBtn) rollBtn.disabled = !hasLegalAction({type: 'ROLL'});
      if (endBtn) endBtn.disabled = !hasLegalAction({type: 'END_TURN'});

      contextActionMode = null;
      const selectedTeam = selectedTeamIndex !== null ? teamSlot(selectedTeamIndex) : null;
      const selectedShop = selectedShopIndex !== null ? shopSlot(selectedShopIndex) : null;

      // The contextual slab is the real game's middle button: Sell for a selected
      // team pet, Freeze / Unfreeze for a selected shop slot. Its visibility is a
      // single class on the element, never an inline style fighting a stylesheet
      // rule, so what the state says is what the button shows.
      if (!contextBtn || !contextLabel || !contextIcon) {
        renderInfoCard(selectedTeam, selectedShop);
        afterRender();
        return;
      }
      if (selectedTeam && selectedTeam.pet_id) {
        contextActionMode = 'SELL';
        const sellValue = teamSellValue(selectedTeam);
        contextBtn.classList.remove('is-idle');
        contextBtn.disabled = !hasLegalAction({type: 'SELL', team_index: selectedTeamIndex});
        contextBtn.dataset.mode = contextActionMode;
        // the game's own contextual wording: the value goes in parentheses
        contextLabel.textContent = `Sell (${sellValue})`;
        contextIcon.innerHTML = SELL_TAG_SVG;
      } else if (selectedShop && isRealItem(selectedShop.item_id)) {
        const willUnfreeze = Boolean(selectedShop.frozen);
        contextActionMode = willUnfreeze ? 'UNFREEZE' : 'FREEZE';
        contextBtn.classList.remove('is-idle');
        contextBtn.disabled = !hasLegalAction({type: contextActionMode, shop_index: selectedShopIndex});
        contextBtn.dataset.mode = contextActionMode;
        contextLabel.textContent = willUnfreeze ? 'Unfreeze' : 'Freeze';
        contextIcon.innerHTML = SNOWFLAKE_SVG
          .replace('<svg', '<svg class="btn-icon-svg"')
          .replace('class="flake"', `stroke="${willUnfreeze ? '#2fb8e8' : '#651f00'}"`);
      } else {
        contextBtn.classList.add('is-idle');
        contextBtn.disabled = true;
        contextBtn.dataset.mode = '';
        contextLabel.textContent = 'Context';
        contextIcon.innerHTML = '';
      }

      renderInfoCard(selectedTeam, selectedShop);
      afterRender();
    }

    /* The game's own selection card: a big white card with a thick black border,
       the name in orange caps, a gold tier circle and the ability sentence for
       the level being looked at. The ability text is the pinned SAP-Calculator
       pack's own copy, served read-only with the catalog. */
    function catalogRow(slotType, itemId) {
      if (!appState || !itemId) return null;
      const cat = appState.debug_catalog || {};
      const rows = slotType === 'food' ? cat.foods : cat.pets;
      if (!Array.isArray(rows)) return null;
      return rows.find((entry) => String(entry.item_id) === String(itemId)) || null;
    }

    function abilityText(slotType, itemId, level) {
      const row = catalogRow(slotType, itemId);
      const lines = row && Array.isArray(row.ability) ? row.ability : null;
      if (!lines || !lines.length) return '';
      const lv = Math.max(1, Math.min(lines.length, Number.parseInt(level ?? 1, 10) || 1));
      return String(lines[lv - 1] || lines[0] || '');
    }

    // Inline glyphs for the ability sentence: the reference writes the trigger,
    // then a solid black arrow, then the effect, and draws the stat icon inline
    // wherever the sentence says health or attack.
    const ABILITY_ARROW_SVG = '<svg class="ability-glyph arrow" viewBox="0 0 26 18"'
      + ' aria-hidden="true"><path d="M0 5.6h14V0.4L25.6 9 14 17.6V12.4H0z" fill="#000000"/></svg>';
    const ABILITY_HEART_SVG = '<svg class="ability-glyph" viewBox="4 5 42 42" aria-hidden="true">'
      + `<path d="${HEART_PATH}" fill="#e5332e" stroke="#000000" stroke-width="4.5"`
      + ' stroke-linejoin="round"/></svg>';
    const ABILITY_ATK_SVG = '<svg class="ability-glyph" viewBox="2 1 44 41" aria-hidden="true">'
      + `<path d="${ATK_ROCK_PATH}" fill="#7a7a7a" stroke="#000000" stroke-width="4"`
      + ' stroke-linejoin="round"/></svg>';

    function escapeText(value) {
      return String(value)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    /* The pack writes "Trigger: effect"; the game draws that colon as an arrow
       and puts the stat icon in front of the stat word. */
    function abilityHTML(text) {
      let out = escapeText(text);
      out = out.replace(/^([^:]{1,40}):\s*/, (_m, trigger) => `${trigger} ${ABILITY_ARROW_SVG} `);
      out = out.replace(/\bhealth\b/g, `${ABILITY_HEART_SVG} health`);
      out = out.replace(/\battack\b/g, `${ABILITY_ATK_SVG} attack`);
      return out;
    }

    // Tier badge on the info card: a shaded coin, a darker rim disc under a
    // lighter face, not one flat circle.
    function tierCoinSVG(tier) {
      return '<svg class="info-tier" viewBox="0 0 40 40" aria-hidden="true"'
        + ` title="tier ${tier}">`
        + '<circle cx="20" cy="20" r="17.4" fill="#e0951a" stroke="#000000" stroke-width="3.2"/>'
        + '<circle cx="20" cy="20" r="12.4" fill="#ffce55"/>'
        + '<text x="20" y="21" text-anchor="middle" dominant-baseline="middle"'
        + ' font-family="SAPUI, sans-serif" font-weight="700" font-size="21"'
        + ` fill="#e06a12">${tier}</text></svg>`;
    }

    function renderInfoCard(selectedTeam, selectedShop) {
      const el = document.getElementById('info-card');
      if (!el) return;
      let slotType = null;
      let itemId = null;
      let level = 1;
      if (selectedTeam && selectedTeam.pet_id) {
        slotType = 'pet';
        itemId = selectedTeam.pet_id;
        level = selectedTeam.level ?? 1;
      } else if (selectedShop && isRealItem(selectedShop.item_id)) {
        slotType = selectedShop.slot_type;
        itemId = selectedShop.item_id;
      }
      if (!itemId) {
        el.classList.remove('show');
        el.innerHTML = '';
        return;
      }
      const tier = itemTier(slotType, itemId);
      const text = abilityText(slotType, itemId, level);
      // The reference card has three parts and no stat row: the stats and the
      // price are already on the slot itself.
      el.innerHTML = `
        <div class="info-head">
          <img class="info-thumb" src="${itemImage(slotType, itemId)}" alt="" onerror="this.style.visibility='hidden';" />
          <span class="info-name">${prettyItemName(itemId)}</span>
          ${tierCoinSVG(tier)}
        </div>
        <div class="info-rule"><i></i></div>
        ${text ? `<div class="info-text">${abilityHTML(text)}</div>` : ''}
      `;
      el.classList.add('show');
    }

    // Keep every message in the dev drawer; only errors and necessary prompts
    // also appear inside the shop scene.
    function setMessage(msg, ok=true, toastMsg=null, prompt=false) {
      const el = document.getElementById('messages');
      if (el) {
        el.textContent = msg || '';
        el.className = `messages ${ok ? 'ok' : 'err'}`;
      }
      // The drawer keeps the exact machine-readable line; the in-scene plaque
      // speaks the game's language and never shows raw JSON.
      if (!ok || prompt) showToast(toastMsg === null ? msg : toastMsg, ok);
    }

    // Plain-language rendering of an action, for the in-scene plaque.
    function humanAction(action) {
      if (!action || typeof action !== 'object') return 'Done.';
      const petName = (idx) => {
        const slot = teamSlot(idx);
        return slot && slot.pet_id ? prettyItemName(slot.pet_id) : `slot ${idx}`;
      };
      switch (String(action.type)) {
        case 'ROLL': return 'Rolled the shop.';
        case 'END_TURN': return 'Turn ended.';
        case 'FREEZE': return 'Froze that shop slot.';
        case 'UNFREEZE': return 'Unfroze that shop slot.';
        case 'BUY_PET': return 'Bought a pet.';
        case 'BUY_COMBINE': return 'Bought and combined a pet.';
        case 'BUY_FOOD': return 'Fed the food to your pet.';
        case 'SELL': return `Sold ${petName(action.team_index)}.`;
        case 'COMBINE': return 'Combined two pets.';
        case 'REORDER': return 'Moved your pet.';
        default: return 'Done.';
      }
    }

    // Chunky close cross for the toast's attached tile.
    const TOAST_CLOSE_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true">'
      + '<path d="M4 7.6L7.6 4 12 8.4 16.4 4 20 7.6 15.6 12 20 16.4 16.4 20 12 15.6'
      + ' 7.6 20 4 16.4 8.4 12z" fill="#651f00"/></svg>';

    function hideToast() {
      const toast = document.getElementById('toast');
      if (toastTimer) {
        clearTimeout(toastTimer);
        toastTimer = null;
      }
      if (toast) toast.classList.remove('show');
    }

    function showToast(msg, ok=true) {
      const toast = document.getElementById('toast');
      const body = document.getElementById('toast-text');
      if (!toast || !body) return;
      const text = String(msg || '').trim();
      if (toastTimer) {
        clearTimeout(toastTimer);
        toastTimer = null;
      }
      if (!text) {
        toast.classList.remove('show');
        return;
      }
      body.textContent = text;
      toast.className = `toast show${ok ? '' : ' err'}`;
      toastTimer = setTimeout(() => {
        toastTimer = null;
        toast.classList.remove('show');
      }, ok ? 2600 : 5200);
    }


    function apiPath(path) {
      const base = (typeof window !== 'undefined' && window.__SAP_API_BASE) || '';
      return base && String(path).startsWith('/api/') ? base + String(path).slice(4) : path;
    }

    /* What this client already holds of the session log, in the shape the two
       transports want it. Null until there is something to claim, which is
       what makes the first request of a session ask for the whole log. */
    function historyCursor() {
      if (!historyToken || !historyEntries.length) return null;
      return {history_from: historyEntries.length, history_token: historyToken};
    }

    /* Exported for `duel.js`, which posts to `/api/duel/*` through a fetch of
       its own rather than through `apiPost`. Without this its end-turn and
       new-game responses would be the only ones still carrying a whole log. */
    function withHistoryCursor(body) {
      return Object.assign({}, body || {}, historyCursor() || {});
    }


    /* The action's own name for a failure sentence ("ROLL did not reach the
       server"). The engine's word, so the message and the history line below
       the scene call the same thing by the same name. */
    function actionLabelFor(action) {
      const type = action && action.type ? String(action.type) : '';
      return type || 'That action';
    }

    function transportFailure(err) {
      const detail = (err && (err.message || err.name)) || String(err);
      return {ok: false, transport: true, state: null, transition: null,
              error: `request_failed: ${detail}`};
    }

    async function apiGet(path) {
      const cursor = historyCursor();
      const query = cursor
        ? (path.indexOf('?') >= 0 ? '&' : '?') +
          `history_from=${cursor.history_from}&history_token=${encodeURIComponent(cursor.history_token)}`
        : '';
      try {
        const res = await fetch(apiPath(path) + query);
        return await res.json();
      } catch (err) {
        return transportFailure(err);
      }
    }

    async function apiPost(path, body) {
      try {
        const res = await fetch(apiPath(path), {
          method: 'POST',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify(withHistoryCursor(body))
        });
        return await res.json();
      } catch (err) {
        return transportFailure(err);
      }
    }

    /* THE BOARD THIS ACTION WAS DECIDED ON, named so the server can refuse
       to apply it to a different one.

       `historyToken` is the server's own content address of the whole session
       log, and the log gains exactly one entry per applied action, so the pair
       below is a state code that changes on every apply. A second click issued
       before the first reply landed still holds the PRE-click pair and is
       refused by `app.py::history_precondition_error`.

       Omitted while `historyToken` is null -- before the first snapshot, and
       after `absorbHistory` finds the two ends disagreeing. That fails OPEN on
       purpose: an unproven staleness must not block the human, and the
       in-flight guard below still covers the click-twice case that produced
       this. */
    function withBoardPrecondition(body) {
      if (!historyToken) return Object.assign({}, body || {});
      return Object.assign({}, body || {}, {
        expect_history_len: historyEntries.length,
        expect_history_token: historyToken,
      });
    }

    /* Every POST that CHANGES the board goes through here, and no other.

       Returns a `{busy: true}` envelope rather than throwing or silently
       resolving, so each caller declares what it does about a refused click
       instead of inheriting a behaviour from here. */
    async function applyPost(body, what) {
      if (mutationInFlight) {
        setMessage(
          `Still applying ${mutationInFlight}; that second click was ignored so it cannot be spent twice.`,
          false,
          `Still applying ${mutationInFlight}. Second click ignored.`);
        return {ok: false, busy: true, state: null, transition: null, error: 'busy'};
      }
      mutationInFlight = what || 'that action';
      try {
        return await apiPost('/api/apply', withBoardPrecondition(body));
      } finally {
        mutationInFlight = null;
      }
    }

    /* The server's refusal, in the human's words where there are any. The
       machine-readable code still reaches the drawer through `setMessage`'s
       own line; this only replaces what the message bar says. */
    function applyErrorMessage(payload) {
      const err = String((payload && payload.error) || 'invalid action');
      if (err === 'stale_board' || err.indexOf('stale_board:') === 0) {
        return 'The board had already moved on, so nothing was applied. What is on screen is the server\'s board.';
      }
      return err;
    }

    /* What to do after a request whose answer never arrived.

       The action may have been applied with only the reply lost, so the board
       on screen is of unknown age and the honest move is to re-read it rather
       than to guess or to retry. NOT retried automatically: a buy or a roll is
       not idempotent, and a retry after a lost reply spends the gold twice.

       Returns true when it handled the payload, so callers read as
       `if (await handledTransportFailure(payload)) return;`. */
    async function handledTransportFailure(payload, what) {
      if (!payload || payload.transport !== true) return false;
      const label = what ? `${what} ` : '';
      setMessage(
        `${label}did not reach the server (${payload.error}). Re-reading the board; the action may or may not have been applied.`,
        false,
        'No answer from the server. Re-reading the board.');
      try {
        await refresh();
      } catch (err) {
        setMessage('The server is not answering. Nothing was applied; try again in a moment.', false);
      }
      return true;
    }

    /* Fetch the pinned pack's catalog, once. The URL carries the catalog's own
       content hash, so this is a normal cacheable GET: a reload costs nothing,
       and a server restarted on a different pack hands out a different URL
       instead of leaving a stale copy to be trusted.

       Deliberately NOT routed through `apiPath`: like `/api/image` and the
       other asset routes, the catalog is a read-only file both pages share, so
       the duel page must not rewrite it to `/api/duel/catalog`. */
    async function ensureCatalog(url) {
      if (!url || (catalogBundleUrl === url && catalogBundle)) return;
      if (catalogFetchInFlight === url) return;
      catalogFetchInFlight = url;
      try {
        const res = await fetch(url);
        const doc = await res.json();
        catalogBundle = {
          catalog_base_stats: doc.catalog_base_stats || {},
          debug_catalog: doc.debug_catalog || {pets: [], foods: []}
        };
        catalogBundleUrl = url;
        // A snapshot may already be on screen without it: the first paint of a
        // duel comes from `new_game`, not from `refresh`.
        if (appState) {
          attachCatalog(appState);
          renderState();
        }
      } catch (err) {
        // Leave `catalogBundleUrl` unset so the next snapshot retries. The page
        // renders without the selection card's ability text meanwhile; it does
        // not throw, because every reader already tolerates an absent catalog.
        setMessage('Could not load the pet catalog; selection details may be missing.', false);
      } finally {
        catalogFetchInFlight = null;
      }
    }

    function attachCatalog(snapshot) {
      if (!snapshot || typeof snapshot !== 'object' || !catalogBundle) return;
      if (!snapshot.catalog_base_stats) snapshot.catalog_base_stats = catalogBundle.catalog_base_stats;
      if (!snapshot.debug_catalog) snapshot.debug_catalog = catalogBundle.debug_catalog;
    }

    /* Fold the entries the server just sent into the log this page keeps, and
       leave `snapshot.history` holding the WHOLE log -- `renderState` prints
       every line of it and must not learn that it arrives in pieces.

       `history_base` is the server's answer to "how much of your copy did I
       accept": 0 means it accepted none and this is the whole log. The length
       check afterwards is the cheap proof that the two ends agree; if they ever
       do not, this drops its cursor so the next request resyncs from scratch. */
    function absorbHistory(snapshot) {
      if (!snapshot || typeof snapshot !== 'object') return;
      if (!Array.isArray(snapshot.history)) return;
      const base = Number.isInteger(snapshot.history_base) ? snapshot.history_base : 0;
      if (base === 0) {
        historyEntries = snapshot.history.slice();
      } else if (base <= historyEntries.length) {
        historyEntries = historyEntries.slice(0, base).concat(snapshot.history);
      } else {
        /* Unreachable: the server only ever echoes back a cursor this client
           sent it. If it ever did happen, the tail on its own is NOT the log,
           and showing it would be the one outcome worse than showing a stale
           one -- so keep what is here, drop the cursor, resync next request. */
        historyToken = null;
        snapshot.history = historyEntries;
        return;
      }
      historyToken = typeof snapshot.history_token === 'string' ? snapshot.history_token : null;
      if (Number.isInteger(snapshot.history_len) && snapshot.history_len !== historyEntries.length) {
        historyToken = null;
      }
      snapshot.history = historyEntries;
    }

    function setAppState(nextState, options = {}) {
      /* A snapshot or nothing. Every mutation reply carries `state`, but a
         reply that never arrived carries `state: null` (see `transportFailure`),
         and a server-side refusal can carry none at all. Assigning either would
         blank a board that is still perfectly good and still correct: the last
         snapshot IS the truth until a newer one lands. */
      if (!nextState || typeof nextState !== 'object') return;
      const prevState = appState && appState.state ? appState.state : null;
      absorbHistory(nextState);
      attachCatalog(nextState);
      appState = nextState;
      if (nextState && nextState.catalog_url && catalogBundleUrl !== nextState.catalog_url) {
        // Not awaited: the board is ready to paint now, and the catalog only
        // feeds the selection card, which repaints when it lands.
        ensureCatalog(nextState.catalog_url);
      }
      /* The engine actions this transition was made of, in order. The shop
         lane is told about the FIRST of them -- a buy leaves a hole in the
         lane, every other transition replaces it -- while the team fx diff is
         given the WHOLE group, because that is what says which slab each pet
         ended on. */
      const actions = (Array.isArray(options.actions) ? options.actions : []).filter(Boolean);
      if (actions.length) {
        noteShopTransition(actions[0]);
      }
      if (options.computeFx && prevState && appState && appState.state) {
        // Pets may have changed slab in this transition, so the animations
        // already in flight travel with them before the new ones merge in.
        const pairs = pairTeamsForTransition(
          prevState.team || [], appState.state.team || [], actions);
        const carried = carryTeamFx(teamFxBySlot, pairs);
        const fx = computeTeamFx(prevState, appState.state, pairs);
        teamFxBySlot = mergeFxMaps(carried, fx);
        if (Object.keys(fx).length) {
          if (fxCleanupTimer) {
            clearTimeout(fxCleanupTimer);
          }
          fxCleanupTimer = setTimeout(() => {
            fxCleanupTimer = null;
            renderState();
          }, FX_TTL_MS + 30);
        }
      }
      if (options.resetFx) {
        teamFxBySlot = {};
        if (fxCleanupTimer) {
          clearTimeout(fxCleanupTimer);
          fxCleanupTimer = null;
        }
      }
    }

    async function refresh() {
      const next = await apiGet('/api/state');
      // The re-read can itself be lost. Say so and keep the board that is on
      // screen, which is a real snapshot of a real moment; `setAppState` would
      // ignore the envelope anyway, but leaving the selection cleared after a
      // refresh that did not happen is its own small lie.
      if (next && next.transport === true) {
        setMessage(`Could not re-read the board (${next.error}).`, false);
        return false;
      }
      // A refresh pulls state this client did not drive, so shop indices may now
      // point at different items. Dropping the selection stops it from silently
      // re-targeting whatever slid into the index.
      selectedShopIndex = null;
      selectedTeamIndex = null;
      pendingFoodTargetShopIndex = null;
      setAppState(next);
      renderState();
      return true;
    }

    function endTurnLiveDbFallbackMessage(payload) {
      if (!payload || !payload.transition || !payload.transition.action) return null;
      if (String(payload.transition.action.type || '') !== 'END_TURN') return null;
      const battle = payload.state && payload.state.last_battle && typeof payload.state.last_battle === 'object'
        ? payload.state.last_battle
        : null;
      if (!battle || !battle.live_db_fallback) return null;
      const route = String(battle.live_db_fallback_route || 'unknown');
      const trigger = String(battle.live_db_fallback_trigger_error || battle.live_db_fallback_reason || 'snapshot_exhausted');
      return `END_TURN used live DB fallback (${route}; trigger=${trigger}).`;
    }

    async function applyAction(action, successLabel='Applied') {
      const payload = await applyPost({action}, actionLabelFor(action));
      if (payload.busy) return false;
      if (await handledTransportFailure(payload, actionLabelFor(action))) return false;
      const applied = payload.ok && payload.transition ? payload.transition.action : null;
      setAppState(payload.state, {computeFx: true, actions: applied ? [applied] : []});
      if (payload.ok) {
        selectedShopIndex = null;
        selectedTeamIndex = null;
        pendingFoodTargetShopIndex = null;
      }
      renderState();
      if (!payload.ok) {
        setMessage(applyErrorMessage(payload), false);
      } else {
        const fallbackMsg = endTurnLiveDbFallbackMessage(payload);
        if (fallbackMsg) {
          setMessage(fallbackMsg, false);
        } else {
          setMessage(`${successLabel}: ${prettyAction(payload.transition.action)}`, true,
                     humanAction(payload.transition.action));
        }
      }
      return payload.ok;
    }


    async function applyCompose(compose, successLabel='Applied') {
      const payload = await applyPost(compose, 'that drag');
      if (payload.busy) return false;
      if (await handledTransportFailure(payload, 'That drag')) return false;
      const transitions = Array.isArray(payload.transitions) ? payload.transitions : [];
      /* The whole group, in order. The shop lane vacates the column the pet
         was BOUGHT from, so it is told about the group's first op and not the
         reorder that follows it; the fx diff reads every REORDER in the group
         to know which slab each pet ended on. */
      const group = payload.ok ? transitions.map((tr) => tr && tr.action).filter(Boolean) : [];
      const applied = group.length ? group[0] : null;
      setAppState(payload.state, {computeFx: true, actions: group});
      if (payload.ok) {
        selectedShopIndex = null;
        selectedTeamIndex = null;
        pendingFoodTargetShopIndex = null;
      }
      renderState();
      if (!payload.ok) {
        setMessage(applyErrorMessage(payload), false);
      } else {
        setMessage(`${successLabel}: ${prettyAction(applied)}`, true, humanAction(applied));
      }
      return payload.ok;
    }

    async function applyIfLegal(action, invalidMessage) {
      const legal = findLegalAction(action);
      if (!legal) {
        setMessage(invalidMessage || `Invalid action: ${prettyAction(action)}`, false,
                   invalidMessage || `That ${String(action.type || 'action').toLowerCase().replace(/_/g, ' ')} is not legal here.`);
        return false;
      }
      return applyAction(legal);
    }

    async function applyActionIndex(index) {
      const payload = await applyPost({action_index: index}, `legal action #${index}`);
      if (payload.busy) return;
      if (await handledTransportFailure(payload, `Legal action #${index}`)) return;
      const applied = payload.ok && payload.transition ? payload.transition.action : null;
      setAppState(payload.state, {computeFx: true, actions: applied ? [applied] : []});
      if (payload.ok) {
        selectedShopIndex = null;
        selectedTeamIndex = null;
        pendingFoodTargetShopIndex = null;
      }
      renderState();
      if (!payload.ok) {
        setMessage(applyErrorMessage(payload), false);
      } else {
        const fallbackMsg = endTurnLiveDbFallbackMessage(payload);
        if (fallbackMsg) {
          setMessage(fallbackMsg, false);
        } else {
          setMessage(`Applied #${index}: ${prettyAction(payload.transition.action)}`, true,
                     humanAction(payload.transition.action));
        }
      }
    }

    async function buyFromShop(shopIndex) {
      if (!appState) return;
      const slot = shopSlot(shopIndex);
      if (!slot) {
        setMessage(`Shop slot ${shopIndex} not found.`, false);
        return;
      }

      if (slot.slot_type === 'pet') {
        const buyCandidates = appState.legal_actions
          .map((entry) => entry.action)
          .filter((action) => action.type === 'BUY_PET' && action.shop_index === shopIndex);
        const combineCandidates = appState.legal_actions
          .map((entry) => entry.action)
          .filter((action) => action.type === 'BUY_COMBINE' && action.shop_index === shopIndex);
        if (buyCandidates.length) {
          await applyAction(buyCandidates[0], 'Bought pet');
          return;
        }
        if (combineCandidates.length === 1) {
          await applyAction(combineCandidates[0], 'Bought + combined pet');
          return;
        }
        if (combineCandidates.length > 1) {
          setMessage('Multiple combine targets available. Drag this shop pet onto a specific team pet.', false);
          return;
        }
        setMessage('Invalid buy: no legal pet target (team may be full, no combine target, or gold too low).', false);
        return;
      }

      const candidates = appState.legal_actions
        .map((entry) => entry.action)
        .filter((action) => action.type === 'BUY_FOOD' && action.shop_index === shopIndex);
      if (!candidates.length) {
        setMessage('Invalid buy: no legal food target (gold too low or board empty).', false);
        return;
      }

      const randomTarget = candidates.find((action) => action.team_index === null) || null;
      const targeted = candidates.filter((action) => action.team_index !== null);
      if (randomTarget && !targeted.length) {
        await applyAction(randomTarget, 'Bought random-target food');
        return;
      }
      if (targeted.length === 1) {
        await applyAction(targeted[0], 'Bought food');
        return;
      }

      pendingFoodTargetShopIndex = shopIndex;
      selectedShopIndex = shopIndex;
      selectedTeamIndex = null;
      renderState();
      setMessage('Select a team pet target for this food, or drag food onto a pet.', true, null, true);
    }

    async function toggleFreeze(shopIndex) {
      const slot = shopSlot(shopIndex);
      if (!slot) {
        setMessage(`Shop slot ${shopIndex} not found.`, false);
        return;
      }
      const action = {type: slot.frozen ? 'UNFREEZE' : 'FREEZE', shop_index: shopIndex};
      await applyIfLegal(action, `Invalid freeze toggle on shop slot ${shopIndex}.`);
    }

    async function applyCustomAction() {
      const raw = document.getElementById('custom-action').value;
      let action = null;
      try {
        action = JSON.parse(raw);
      } catch (err) {
        setMessage(`Invalid JSON: ${err}`, false);
        return;
      }
      const payload = await applyPost({action}, 'that custom action');
      if (payload.busy) return;
      setAppState(payload.state, {computeFx: true});
      renderState();
      if (!payload.ok) {
        setMessage(applyErrorMessage(payload), false);
      } else {
        const fallbackMsg = endTurnLiveDbFallbackMessage(payload);
        if (fallbackMsg) {
          setMessage(fallbackMsg, false);
        } else {
          setMessage(`Applied custom: ${prettyAction(payload.transition.action)}`, true,
                     humanAction(payload.transition.action));
        }
      }
    }

    function setRecommendPending(active) {
      recommendationPending = Boolean(active);
      const btn = document.getElementById('btn-recommend');
      if (!btn) return;
      const label = btn.querySelector('span');
      btn.disabled = recommendationPending;
      if (label) {
        label.textContent = recommendationPending ? 'Recommending...' : 'Recommend';
      }
    }

    function renderRecommendation(payload) {
      const el = document.getElementById('recommendation');
      if (!el) return;
      if (!payload || typeof payload !== 'object') {
        el.innerHTML = `<span class="tiny">recommendation_failed</span>`;
        return;
      }

      const rec = payload.recommended_action || null;
      const wdl = payload.wdl_probs || null;
      const chain = Array.isArray(payload.chain_preview) ? payload.chain_preview : [];
      const diag = (payload.diagnostics && typeof payload.diagnostics === 'object') ? payload.diagnostics : {};
      const encountered = Array.isArray(payload.encountered_errors) ? payload.encountered_errors : [];
      const fallbacks = Array.isArray(payload.fallbacks_used) ? payload.fallbacks_used : [];
      const contextSource = String(payload.context_source || 'no_context');
      const predicted = (payload.predicted_next_board && typeof payload.predicted_next_board === 'object')
        ? payload.predicted_next_board
        : null;
      const predictedLegacyImageUrlRaw = (typeof payload.predicted_next_board_image_url === 'string')
        ? payload.predicted_next_board_image_url.trim()
        : '';
      const predictedStartImageUrlRaw = (typeof payload.predicted_next_board_start_image_url === 'string')
        ? payload.predicted_next_board_start_image_url.trim()
        : '';
      const predictedEndImageUrlRaw = (typeof payload.predicted_next_board_end_image_url === 'string')
        ? payload.predicted_next_board_end_image_url.trim()
        : '';
      const predictedStartImageUrl = predictedStartImageUrlRaw || predictedLegacyImageUrlRaw || null;
      const predictedEndImageUrl = predictedEndImageUrlRaw || null;
      const predictedSource = predicted ? String(predicted.source || contextSource) : 'n/a';
      const predictedProbRaw = predicted ? Number(predicted.prob ?? 0) : Number.NaN;
      const predictedProbText = Number.isFinite(predictedProbRaw) ? predictedProbRaw.toFixed(3) : 'n/a';
      const predictedTeamText = predicted ? formatPredictedTeam(predicted.team) : 'none';
      const predictedStartImageHtml = predictedStartImageUrl
        ? `<div style="margin-top:6px;"><div class="tiny"><strong>Start Board:</strong> player + predicted opponent</div><img src="${predictedStartImageUrl}" alt="Predicted start board" style="max-width:100%; border:1px solid var(--border); border-radius:8px;" /></div>`
        : '';
      const predictedEndImageHtml = predictedEndImageUrl
        ? `<div style="margin-top:6px;"><div class="tiny"><strong>End Board:</strong> after tempo chain</div><img src="${predictedEndImageUrl}" alt="Predicted end board" style="max-width:100%; border:1px solid var(--border); border-radius:8px;" /></div>`
        : '';
      const predictedBlock = predicted
        ? [
            `<div><strong>Predicted Next Board:</strong> p=${predictedProbText} (${predictedSource})</div>`,
            `<div><strong>Predicted Team:</strong> ${predictedTeamText}</div>`,
            predictedStartImageHtml,
            predictedEndImageHtml,
          ].join('')
        : `<div><strong>Predicted Next Board:</strong> unavailable</div>`;
      const updatedAtRaw = Number(payload.updated_at_unix_ms || 0);
      const updatedText = Number.isFinite(updatedAtRaw) && updatedAtRaw > 0
        ? new Date(updatedAtRaw).toLocaleTimeString()
        : 'n/a';

      let changeStatus = 'n/a';
      if (payload.ok) {
        const signature = stableStringify({
          recommended_action: rec,
          chain_preview: chain,
          wdl_probs: wdl,
          context_source: contextSource,
          predicted_next_board: predicted,
          predicted_next_board_start_image_url: predictedStartImageUrl,
          predicted_next_board_end_image_url: predictedEndImageUrl,
        });
        if (lastRecommendationSignature === null) {
          changeStatus = 'Initial';
        } else {
          changeStatus = (lastRecommendationSignature === signature) ? 'Unchanged' : 'Changed';
        }
        lastRecommendationSignature = signature;
      }

      if (!payload.ok) {
        el.innerHTML = [
          `<div><strong>Status:</strong> Failed (${payload.error || 'recommendation_failed'})</div>`,
          `<div><strong>Updated:</strong> ${updatedText}</div>`,
          `<div><strong>Context Source:</strong> ${contextSource}</div>`,
          predictedBlock,
          `<div><strong>Encountered Errors:</strong> ${encountered.length ? encountered.join(', ') : 'none'}</div>`,
          `<div><strong>Fallbacks Used:</strong> ${fallbacks.length ? fallbacks.join(', ') : 'none'}</div>`,
        ].join('');
        return;
      }

      const w = wdl ? Number(wdl.win || 0).toFixed(3) : 'n/a';
      const d = wdl ? Number(wdl.draw || 0).toFixed(3) : 'n/a';
      const l = wdl ? Number(wdl.loss || 0).toFixed(3) : 'n/a';
      const chainText = chain.length ? chain.map((a) => prettyAction(a)).join(' -> ') : 'none';
      el.innerHTML = [
        `<div><strong>Status:</strong> ${changeStatus}</div>`,
        `<div><strong>Updated:</strong> ${updatedText}</div>`,
        `<div><strong>Context Source:</strong> ${contextSource}</div>`,
        `<div><strong>Recommended Action:</strong> ${rec ? prettyAction(rec) : 'none'}</div>`,
        `<div><strong>W/D/L:</strong> ${w} / ${d} / ${l}</div>`,
        `<div><strong>Chain Preview:</strong> ${chainText}</div>`,
        predictedBlock,
        `<div><strong>Diagnostics:</strong> candidates=${diag.candidate_count ?? 'n/a'}, reranked=${diag.reranked_count ?? 'n/a'}, sims=${diag.oracle_simulation_count ?? 'n/a'}</div>`,
        `<div><strong>Encountered Errors:</strong> ${encountered.length ? encountered.join(', ') : 'none'}</div>`,
        `<div><strong>Fallbacks Used:</strong> ${fallbacks.length ? fallbacks.join(', ') : 'none'}</div>`,
      ].join('');
    }

    async function requestRecommendation() {
      if (recommendationPending) {
        return;
      }
      const el = document.getElementById('recommendation');
      if (el) {
        el.innerHTML = `<span class="tiny">Computing recommendation...</span>`;
      }
      setRecommendPending(true);
      try {
        const payload = await apiPost('/api/infer/recommend', {});
        renderRecommendation(payload);
        if (!payload.ok) {
          setMessage(payload.error || 'Could not produce recommendation.', false);
        } else if ((Array.isArray(payload.encountered_errors) && payload.encountered_errors.length) || (Array.isArray(payload.fallbacks_used) && payload.fallbacks_used.length)) {
          setMessage('Recommendation updated with fallback/error signals (see panel).', false);
        } else {
          setMessage('Recommendation updated.', true);
        }
      } catch (err) {
        const message = String(err || 'recommendation_request_failed');
        renderRecommendation({
          ok: false,
          error: `recommendation_request_failed:${message}`,
          recommended_action: null,
          chain_preview: [],
          wdl_probs: null,
          diagnostics: {},
          encountered_errors: [`recommendation_request_failed:${message}`],
          fallbacks_used: [],
          context_source: 'no_context',
          updated_at_unix_ms: Date.now(),
        });
        setMessage(`Recommendation request failed: ${message}`, false);
      } finally {
        setRecommendPending(false);
      }
    }
    function readDragData(event) {
      if (dragData) return dragData;
      const raw = event.dataTransfer ? event.dataTransfer.getData('text/plain') : '';
      if (!raw) return null;
      try {
        return JSON.parse(raw);
      } catch (_err) {
        return null;
      }
    }

    document.addEventListener('dragstart', (event) => {
      const card = event.target.closest('.card[data-slot-kind]');
      if (!card) return;
      const kind = card.dataset.slotKind;
      let payload = null;
      if (kind === 'shop') {
        const shopIndex = Number.parseInt(card.dataset.shopIndex, 10);
        const slot = shopSlot(shopIndex);
        if (!slot || !isRealItem(slot.item_id)) {
          event.preventDefault();
          return;
        }
        payload = {kind: 'shop', shopIndex, slotType: slot.slot_type};
      } else if (kind === 'team') {
        const teamIndex = Number.parseInt(card.dataset.teamIndex, 10);
        const slot = teamSlot(teamIndex);
        if (!slot || !slot.pet_id) {
          event.preventDefault();
          return;
        }
        payload = {kind: 'team', teamIndex};
      }
      if (!payload) {
        event.preventDefault();
        return;
      }
      dragData = payload;
      if (event.dataTransfer) {
        event.dataTransfer.setData('text/plain', JSON.stringify(payload));
        event.dataTransfer.effectAllowed = 'move';
      }
    });

    document.addEventListener('dragend', () => {
      dragData = null;
      clearDropHighlights();
    });

    bind('team', 'dragover', (event) => {
      const payload = readDragData(event);
      if (!payload) return;
      event.preventDefault();
      clearDropHighlights();
      const card = event.target.closest('.team-card');
      if (card) {
        card.classList.add('drop-hover');
      }
    });

    bind('btn-context-action', 'dragover', (event) => {
      const payload = readDragData(event);
      if (!payload || payload.kind !== 'team') return;
      event.preventDefault();
      event.currentTarget.classList.add('drop-hover');
    });

    bind('btn-context-action', 'dragleave', (event) => {
      event.currentTarget.classList.remove('drop-hover');
    });

    bind('btn-context-action', 'drop', async (event) => {
      const payload = readDragData(event);
      if (!payload || payload.kind !== 'team') return;
      event.preventDefault();
      event.currentTarget.classList.remove('drop-hover');
      await applyIfLegal(
        {type: 'SELL', team_index: payload.teamIndex},
        `Cannot sell team slot ${payload.teamIndex}.`,
      );
    });

    bind('team', 'drop', async (event) => {
      const payload = readDragData(event);
      if (!payload) return;
      event.preventDefault();
      clearDropHighlights();

      const targetCard = event.target.closest('.team-card');
      if (targetCard) {
        const targetIndex = Number.parseInt(targetCard.dataset.teamIndex, 10);
        if (payload.kind === 'shop') {
          if (payload.slotType === 'pet') {
            const targetSlot = teamSlot(targetIndex);
            if (targetSlot && targetSlot.pet_id) {
              const combineAction = {type: 'BUY_COMBINE', shop_index: payload.shopIndex, team_index: targetIndex};
              if (hasLegalAction(combineAction)) {
                await applyAction(combineAction, 'Bought + combined pet');
                return;
              }
              await applyIfLegal(
                {type: 'BUY_PET', shop_index: payload.shopIndex, team_index: targetIndex},
                `Cannot buy pet into team slot ${targetIndex}.`,
              );
              return;
            }
            // Empty slab: buy into THIS slab, whichever one it is. The engine
            // only sells into the first empty slot, so the server composes
            // the buy with a reorder and applies both or neither.
            if (composeEnabled()) {
              await applyCompose(
                {compose: 'buy_pet_at', shop_index: payload.shopIndex, team_index: targetIndex},
                'Bought pet',
              );
              return;
            }
            // Agent surface: the plain engine action, which lands only in the
            // FIRST empty slot. Dropping on a later one is refused by the
            // engine's own rule, which is the rule the agent plays under.
            await applyIfLegal(
              {type: 'BUY_PET', shop_index: payload.shopIndex, team_index: targetIndex},
              `BUY_PET into team slot ${targetIndex} is not legal here: the engine buys into the first empty slot.`,
            );
            return;
          }
          await applyIfLegal(
            {type: 'BUY_FOOD', shop_index: payload.shopIndex, team_index: targetIndex},
            `Cannot feed team slot ${targetIndex} with this food.`,
          );
          return;
        }

        if (payload.kind === 'team') {
          const src = payload.teamIndex;
          const dst = targetIndex;
          if (src === dst) return;
          // Landing on a SAME-NAME pet is a COMBINE, which is an upgrade and
          // not a move at all, so it is asked about first.
          const combineAction = {type: 'COMBINE', src_team_index: src, dst_team_index: dst};
          if (hasLegalAction(combineAction)) {
            await applyAction(combineAction, 'Combined');
            return;
          }
          // Every other drop between slabs is ONE move, whether it lands on a
          // bare slab or on another pet: the pet is lifted out and inserted
          // where it was dropped, and whoever it displaces shifts over one
          // slab -- the same shove the engine gives the board to land a
          // summon. `legal_actions` enumerates neither an empty-slab reorder
          // nor an insert, so both go through the composer.
          if (composeEnabled()) {
            await applyCompose({compose: 'move_pet', src, dst}, 'Moved pet');
            return;
          }
          // Agent surface: the only reposition in the action space is a
          // REORDER permutation of OCCUPIED slots, so a drop between two pets
          // is that swap and a drop onto a bare slab has no action at all.
          await applyIfLegal(
            {type: 'REORDER', order: transposition(src, dst)},
            `REORDER ${src} <-> ${dst} is not legal here: the engine only permutes occupied slots.`,
          );
          return;
        }
        return;
      }

      if (payload.kind === 'shop' && payload.slotType === 'food') {
        const randomFood = {type: 'BUY_FOOD', shop_index: payload.shopIndex, team_index: null};
        if (hasLegalAction(randomFood)) {
          await applyAction(randomFood, 'Bought random-target food');
        } else {
          setMessage('This food requires a specific pet target. Drop it onto a pet card.', false);
        }
      }
    });

    bind('legal-actions', 'click', (event) => {
      const btn = event.target.closest('button[data-index]');
      if (!btn) return;
      applyActionIndex(Number.parseInt(btn.dataset.index, 10));
    });

    bind('shop', 'click', async (event) => {
      const button = event.target.closest('button[data-action]');
      if (button) {
        const action = button.dataset.action;
        const shopIndex = Number.parseInt(button.dataset.shopIndex, 10);
        if (Number.isNaN(shopIndex)) return;
        if (action === 'buy-shop') {
          await buyFromShop(shopIndex);
          return;
        }
        if (action === 'freeze-shop') {
          await toggleFreeze(shopIndex);
          return;
        }
      }

      const card = event.target.closest('.shop-card');
      if (!card) return;
      const shopIndex = Number.parseInt(card.dataset.shopIndex, 10);
      if (Number.isNaN(shopIndex)) return;
      selectedShopIndex = shopIndex;
      selectedTeamIndex = null;
      renderState();
    });

    bind('shop', 'contextmenu', async (event) => {
      const card = event.target.closest('.shop-card');
      if (!card) return;
      const shopIndex = Number.parseInt(card.dataset.shopIndex, 10);
      if (Number.isNaN(shopIndex)) return;
      event.preventDefault();
      selectedShopIndex = shopIndex;
      selectedTeamIndex = null;
      renderState();
      await toggleFreeze(shopIndex);
    });

    bind('team', 'click', async (event) => {
      const button = event.target.closest('button[data-action]');
      if (button) {
        const action = button.dataset.action;
        if (action === 'combine') {
          const src = Number.parseInt(button.dataset.src, 10);
          const dst = Number.parseInt(button.dataset.dst, 10);
          await applyIfLegal(
            {type: 'COMBINE', src_team_index: src, dst_team_index: dst},
            `Cannot combine slot ${src} into slot ${dst}.`,
          );
          return;
        }
      }

      const card = event.target.closest('.team-card');
      if (!card) return;
      const teamIndex = Number.parseInt(card.dataset.teamIndex, 10);
      if (Number.isNaN(teamIndex)) return;

      if (pendingFoodTargetShopIndex !== null) {
        const ok = await applyIfLegal(
          {type: 'BUY_FOOD', shop_index: pendingFoodTargetShopIndex, team_index: teamIndex},
          `Invalid food target at team slot ${teamIndex}.`,
        );
        if (!ok) {
          selectedTeamIndex = teamIndex;
          renderState();
        }
        return;
      }

      selectedTeamIndex = selectedTeamIndex === teamIndex ? null : teamIndex;
      selectedShopIndex = null;
      renderState();
    });

    // the toast's attached close tile, drawn once and wired to dismiss it
    setHTML('toast-close', TOAST_CLOSE_SVG);
    bind('toast-close', 'click', hideToast);

    bind('action-filter', 'change', renderState);
    bind('debug-pet-search', 'input', () => {
      populateDebugSelectors();
    });
    bind('debug-food-search', 'input', () => {
      populateDebugSelectors();
    });
    bind('btn-refresh', 'click', refresh);
    bind('btn-add-debug-pet', 'click', async () => {
      const select = document.getElementById('debug-pet-select');
      const itemId = select ? select.value : '';
      if (!itemId) {
        setMessage('No pet selected for debug add.', false);
        return;
      }
      const payload = await apiPost('/api/debug/add_shop_item', {slot_type: 'pet', item_id: itemId, cost: 0});
      setAppState(payload.state);
      renderState();
      if (!payload.ok) {
        setMessage(payload.error || 'Could not add selected pet.', false);
      } else {
        setMessage(`Debug: added ${itemId} to shop.`, true);
      }
    });
    bind('btn-add-debug-food', 'click', async () => {
      const select = document.getElementById('debug-food-select');
      const itemId = select ? select.value : '';
      if (!itemId) {
        setMessage('No food selected for debug add.', false);
        return;
      }
      const payload = await apiPost('/api/debug/add_shop_item', {slot_type: 'food', item_id: itemId, cost: 0});
      setAppState(payload.state);
      renderState();
      if (!payload.ok) {
        setMessage(payload.error || 'Could not add selected food.', false);
      } else {
        setMessage(`Debug: added ${itemId} to shop.`, true);
      }
    });
    bind('btn-add-chocolate', 'click', async () => {
      const payload = await apiPost('/api/debug/add_chocolate', {});
      setAppState(payload.state);
      renderState();
      if (!payload.ok) {
        setMessage(payload.error || 'Could not add chocolate.', false);
      } else {
        setMessage('Debug: added chocolate to shop.', true);
      }
    });
    bind('btn-roll', 'click', async () => {
      await applyIfLegal({type: 'ROLL'}, 'ROLL is not legal right now (likely no gold).');
    });
    bind('btn-context-action', 'click', async () => {
      const mode = contextActionMode;
      if (mode === 'SELL') {
        if (selectedTeamIndex === null) {
          setMessage('Select a team pet first to sell.', false);
          return;
        }
        await applyIfLegal({type: 'SELL', team_index: selectedTeamIndex}, `Cannot sell team slot ${selectedTeamIndex}.`);
        return;
      }
      if (mode === 'FREEZE' || mode === 'UNFREEZE') {
        if (selectedShopIndex === null) {
          setMessage('Select a shop slot first to freeze.', false);
          return;
        }
        await toggleFreeze(selectedShopIndex);
        return;
      }
      setMessage('No contextual action available.', false);
    });
    bind('btn-end-turn', 'click', async () => {

      if (typeof window !== 'undefined' && typeof window.__SAP_END_TURN === 'function') {
        await window.__SAP_END_TURN();
        return;
      }
      await applyIfLegal({type: 'END_TURN'}, 'END_TURN is not legal in this state.');
    });
    bind('btn-recommend', 'click', async () => {
      await requestRecommendation();
    });
    bind('btn-apply-custom', 'click', applyCustomAction);
    async function resetShopSession() {
      if (mutationInFlight) {
        const error = 'An action is still being applied. Please wait before starting again.';
        setMessage(error, false);
        return {ok: false, error};
      }
      mutationInFlight = 'reset';
      const buttons = ['btn-reset', 'sandbox-end-again'].map(id => document.getElementById(id)).filter(Boolean);
      buttons.forEach(button => { button.disabled = true; });
      try {
        const payload = await apiPost('/api/reset', {});
        if (await handledTransportFailure(payload, 'Reset')) return payload;
        if (!payload.ok || !payload.state) {
          const error = payload.error || 'Could not start a new game.';
          setMessage(error, false);
          return {ok: false, error};
        }
        setAppState(payload.state, {resetFx: true, actions: [{type: 'RESET'}]});
        selectedShopIndex = null;
        selectedTeamIndex = null;
        pendingFoodTargetShopIndex = null;
        hideToast();
        renderState();
        setMessage('Reset to a fresh random shop state.');
        return payload;
      } finally {
        mutationInFlight = null;
        buttons.forEach(button => { button.disabled = false; });
      }
    }
    bind('btn-reset', 'click', resetShopSession);
    bind('btn-undo', 'click', async () => {
      const payload = await apiPost('/api/undo', {});
      setAppState(payload.state, {resetFx: true});
      selectedShopIndex = null;
      selectedTeamIndex = null;
      pendingFoodTargetShopIndex = null;
      renderState();
      if (!payload.ok) {
        setMessage(payload.error || 'Cannot undo', false);
      } else {
        setMessage('Undo complete.');
      }
    });

    // The scene is authored at the live game's own 1280x760 so every measured
    // offset holds. On a narrower window it scales down as one unit instead of
    // reflowing, which would break the lane-to-terrain alignment.
    function fitStage() {
      const stage = document.getElementById('stage');
      const scene = document.getElementById('scene');
      if (!stage || !scene) return;
      const avail = stage.clientWidth - 8;
      const scale = Math.min(1, avail / 1280);
      scene.style.transform = scale < 1 ? `scale(${scale})` : 'none';
      stage.style.height = `${Math.round(760 * scale) + 8}px`;
    }
    window.addEventListener('resize', fitStage);
    fitStage();

    refresh();
