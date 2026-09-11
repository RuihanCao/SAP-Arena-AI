#!/usr/bin/env node
"use strict";

const fs = require("fs");
const { createRequire } = require("module");

// Keep the pinned renderer/drawing modules unchanged. Only adapt fields that
// the original replay format inferred but saved Arena records supply explicitly.
function adaptRecordedMetadata(renderPath, payload) {
  if (payload.mode !== "calc_rows") return;
  const requireRenderer = createRequire(renderPath);
  const Canvas = requireRenderer("canvas");
  const drawing = requireRenderer("./drawing");
  const { BATTLE_HEIGHT, PET_WIDTH } = requireRenderer("./config");
  const originalDrawPet = drawing.drawPet;
  drawing.drawPet = async (ctx, pet, ...args) => {
    if (!Number.isFinite(pet.xp)) {
      // roundRect in the pinned drawPet draws XP ticks only. Missing XP is
      // not zero XP: retain the original sprite/perk/level/stats, omit ticks.
      ctx = new Proxy(ctx, {
        get(target, key) {
          if (key === "roundRect") return () => {};
          const value = Reflect.get(target, key);
          return typeof value === "function" ? value.bind(target) : value;
        },
        set(target, key, value) { return Reflect.set(target, key, value); },
      });
    }
    return originalDrawPet(ctx, pet, ...args);
  };
  const createCanvas = Canvas.createCanvas;
  Canvas.createCanvas = (...args) => {
    const canvas = createCanvas(...args);
    const ctx = canvas.getContext("2d");
    const fillText = ctx.fillText.bind(ctx);
    const headerHeight = payload.playerName && payload.headerOpponentName ? 36 : 0;
    ctx.fillText = (text, x, y, ...rest) => {
      const heartX = (25 + PET_WIDTH) * 2 + PET_WIDTH / 2;
      const rowIndex = (y - headerHeight - (x === heartX ? 57 : 56)) / BATTLE_HEIGHT;
      const row = Number.isInteger(rowIndex) ? payload.battles[rowIndex] : null;
      if (row && x === 25 + PET_WIDTH + 15 && Number.isInteger(row.turn)) text = row.turn;
      if (row && x === heartX && Number.isInteger(row.livesBefore)) text = row.livesBefore;
      return fillText(text, x, y, ...rest);
    };
    return canvas;
  };
}

// Python launches this bridge synchronously. If that parent is force-killed,
// do not leave a detached 60-second renderer behind merely because Popen made
// a process group for targeted cancellation.
const launcherPid = process.ppid;
const parentWatch = setInterval(() => {
  if (process.ppid === 1 || process.ppid !== launcherPid) process.exit(143);
}, 100);
parentWatch.unref();

function toSafeNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function toOutcomeCode(value, outcomes) {
  if (Number.isFinite(Number(value))) {
    return Number(value);
  }
  const raw = String(value || "").trim().toLowerCase();
  if (raw === "win") {
    return outcomes.WIN;
  }
  if (raw === "loss") {
    return outcomes.LOSS;
  }
  if (raw === "draw" || raw === "tie") {
    return outcomes.TIE;
  }
  return outcomes.TIE;
}

function normalizeEquipmentName(value) {
  if (!value) {
    return null;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || null;
  }
  if (typeof value === "object") {
    const maybeName = String(value.name || "").trim();
    return maybeName || null;
  }
  return null;
}

function levelFromExp(exp) {
  if (exp >= 5) {
    return 3;
  }
  if (exp >= 2) {
    return 2;
  }
  return 1;
}

function normalizeLookupToken(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

async function main() {
  const [payloadPath, renderPath, battlePath, dataPath, configPath] = process.argv.slice(2);
  if (!payloadPath || !renderPath || !battlePath || !dataPath || !configPath) {
    throw new Error("missing_args");
  }

  const payload = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
  adaptRecordedMetadata(renderPath, payload);
  const { renderReplayImage } = require(renderPath);
  const { getBattleInfo } = require(battlePath);
  const { PETS, PERKS, TOYS } = require(dataPath);
  const { PLACEHOLDER_SPRITE, PLACEHOLDER_PERK, BATTLE_OUTCOMES } = require(configPath);

  const petsByName = new Map();
  for (const pet of Object.values(PETS || {})) {
    if (!pet || !pet.Name) {
      continue;
    }
    petsByName.set(String(pet.Name).toLowerCase(), pet);
    if (pet.NameId) petsByName.set(String(pet.NameId).toLowerCase(), pet);
  }

  const perksByName = new Map();
  for (const perk of Object.values(PERKS || {})) {
    if (!perk || !perk.Name) {
      continue;
    }
    perksByName.set(String(perk.Name).toLowerCase(), perk);
    if (perk.NameId) perksByName.set(String(perk.NameId).toLowerCase(), perk);
  }

  const toysByName = new Map();
  for (const toy of Object.values(TOYS || {})) {
    if (!toy || !toy.Name) {
      continue;
    }
    const name = String(toy.Name).trim();
    if (!name) {
      continue;
    }
    toysByName.set(name.toLowerCase(), toy);
    toysByName.set(normalizeLookupToken(name), toy);
    if (toy.NameId) {
      toysByName.set(String(toy.NameId).toLowerCase(), toy);
      toysByName.set(normalizeLookupToken(String(toy.NameId)), toy);
    }
  }

  function toRenderPet(rawPet) {
    if (!rawPet || typeof rawPet !== "object") {
      return null;
    }

    const name = String(rawPet.name || "").trim();
    if (!name) {
      return null;
    }

    const petMeta = petsByName.get(name.toLowerCase()) || null;
    const imagePath = petMeta && petMeta.NameId ? `Sprite/Pets/${petMeta.NameId}.png` : PLACEHOLDER_SPRITE;

    const equipmentName = normalizeEquipmentName(rawPet.equipment);
    const perkMeta = equipmentName ? perksByName.get(equipmentName.toLowerCase()) : null;
    const perkImagePath = equipmentName
      ? (perkMeta && perkMeta.NameId ? `Sprite/Food/${perkMeta.NameId}.png` : PLACEHOLDER_PERK)
      : null;

    const totalAttack = toSafeNumber(rawPet.attack, 0);
    const totalHealth = toSafeNumber(rawPet.health, 0);

    const tempAttackInput = (rawPet.tempAttack !== undefined) ? rawPet.tempAttack : rawPet.temp_attack;
    const tempHealthInput = (rawPet.tempHealth !== undefined) ? rawPet.tempHealth : rawPet.temp_health;
    const tempAttack = toSafeNumber(tempAttackInput, 0);
    const tempHealth = toSafeNumber(tempHealthInput, 0);

    const attack = Math.max(0, totalAttack - tempAttack);
    const health = Math.max(0, totalHealth - tempHealth);

    const knownExp = rawPet.exp !== null && rawPet.exp !== undefined;
    const exp = knownExp ? toSafeNumber(rawPet.exp, 0) : NaN;

    return {
      name,
      attack,
      health,
      tempAttack,
      tempHealth,
      level: knownExp ? levelFromExp(exp) : toSafeNumber(rawPet.level, 1),
      xp: exp,
      perk: equipmentName,
      imagePath,
      perkImagePath,
    };
  }

  function toBoardPets(rawPets) {
    const out = [];
    for (const rawPet of Array.isArray(rawPets) ? rawPets.slice(0, 5) : []) {
      const mapped = toRenderPet(rawPet);
      if (mapped) {
        out.push(mapped);
      }
    }
    return out;
  }

  function toRenderToy(toyNameRaw, toyLevelRaw) {
    const toyName = normalizeEquipmentName(toyNameRaw);
    if (!toyName) {
      return { imagePath: null, level: 0 };
    }
    const lookupKey = toyName.toLowerCase();
    const toyMeta = toysByName.get(lookupKey) || toysByName.get(normalizeLookupToken(toyName)) || null;
    const toyLevel = Math.max(0, toSafeNumber(toyLevelRaw, 0));
    return {
      imagePath: (toyMeta && toyMeta.NameId) ? `Sprite/Toys/${toyMeta.NameId}.png` : PLACEHOLDER_SPRITE,
      level: toyLevel,
    };
  }

  const battles = [];

  if (String(payload.mode || "").toLowerCase() === "battle_json") {
    for (const rawBattle of Array.isArray(payload.battles) ? payload.battles : []) {
      if (!rawBattle || typeof rawBattle !== "object") {
        continue;
      }
      const battleInfo = getBattleInfo(rawBattle);
      if (battleInfo && typeof battleInfo === "object") {
        battles.push(battleInfo);
      }
    }
  } else {
    for (const row of Array.isArray(payload.battles) ? payload.battles : []) {
      if (!row || typeof row !== "object") {
        continue;
      }
      const parsedTurn = Number(row.turn);
      battles.push({
        playerBoard: {
          boardPets: toBoardPets(row.playerPets),
          toy: toRenderToy(row.playerToy, row.playerToyLevel),
        },
        oppBoard: {
          boardPets: toBoardPets(row.opponentPets),
          toy: toRenderToy(row.opponentToy, row.opponentToyLevel),
        },
        outcome: toOutcomeCode(row.outcome, BATTLE_OUTCOMES),
        opponentName: String(row.opponentName || "Opponent"),
        turn: Number.isFinite(parsedTurn) ? parsedTurn : null,
        turnLabel: (typeof row.turnLabel === "string" && row.turnLabel.trim()) ? row.turnLabel.trim() : null,
      });
    }
  }

  if (battles.length === 0) {
    throw new Error("no_battles_to_render");
  }

  const image = await renderReplayImage({
    battles,
    battleOpponentInfo: [],
    maxLives: toSafeNumber(payload.maxLives, 6),
    includeOdds: Boolean(payload.includeOdds),
    winPercentResults: Array.isArray(payload.winPercentResults)
      ? payload.winPercentResults.map((entry) => {
        if (!entry || typeof entry !== "object") {
          return null;
        }
        return {
          player: (entry.player === undefined || entry.player === null) ? null : String(entry.player),
          opponent: (entry.opponent === undefined || entry.opponent === null) ? null : String(entry.opponent),
          draw: (entry.draw === undefined || entry.draw === null) ? null : String(entry.draw),
        };
      })
      : [],
    playerName: payload.playerName || null,
    headerOpponentName: payload.headerOpponentName || null,
  });

  process.stdout.write(image.toString("base64"));
}

main().catch((error) => {
  const message = error && error.stack ? error.stack : String(error);
  process.stderr.write(message);
  process.exit(1);
});
