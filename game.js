(() => {
  "use strict";

  /** @type {HTMLCanvasElement} */
  const canvas = document.getElementById("game");
  /** @type {CanvasRenderingContext2D} */
  const ctx = canvas.getContext("2d");

  const elScore = document.getElementById("score");
  const elLives = document.getElementById("lives");
  const elEnemies = document.getElementById("enemies");
  const overlay = document.getElementById("overlay");
  const overlayTitle = document.getElementById("overlayTitle");
  const overlayBody = document.getElementById("overlayBody");
  const btnResume = document.getElementById("btnResume");
  const btnRestart = document.getElementById("btnRestart");

  // --- Config
  const TILE = 32;
  const MAP_W = 26; // 26*32 = 832
  const MAP_H = 20; // 20*32 = 640
  const W = MAP_W * TILE;
  const H = MAP_H * TILE;

  const COLORS = {
    bg: "rgba(255,255,255,0.03)",
    grid: "rgba(255,255,255,0.04)",
    brick: "#b45309",
    brick2: "#92400e",
    steel: "#94a3b8",
    water: "rgba(59,130,246,0.35)",
    grass: "rgba(34,197,94,0.22)",
    shadow: "rgba(0,0,0,0.35)",
    player: "#22c55e",
    enemy: "#ef4444",
    bulletP: "#e2e8f0",
    bulletE: "#fda4af",
    ui: "rgba(255,255,255,0.85)",
  };

  /** @typedef {"up"|"down"|"left"|"right"} Dir */
  const DIRS = /** @type {const} */ (["up", "down", "left", "right"]);
  const DIR_VEC = {
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 },
  };

  const TileKind = {
    Empty: 0,
    Brick: 1, // destroyable
    Steel: 2, // indestructible
    Water: 3, // blocks movement, blocks bullets
    Grass: 4, // pass-through (visual)
  };

  const keys = new Set();
  const justPressed = new Set();
  let raf = 0;

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function aabbIntersects(a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  }

  function rect(x, y, w, h) {
    return { x, y, w, h };
  }

  function tileAt(map, tx, ty) {
    if (tx < 0 || ty < 0 || tx >= MAP_W || ty >= MAP_H) return TileKind.Steel; // treat outside as wall
    return map[ty][tx];
  }

  function setTile(map, tx, ty, kind) {
    if (tx < 0 || ty < 0 || tx >= MAP_W || ty >= MAP_H) return;
    map[ty][tx] = kind;
  }

  function rectToTileRange(r) {
    const x0 = Math.floor(r.x / TILE);
    const y0 = Math.floor(r.y / TILE);
    const x1 = Math.floor((r.x + r.w - 0.001) / TILE);
    const y1 = Math.floor((r.y + r.h - 0.001) / TILE);
    return { x0, y0, x1, y1 };
  }

  function isSolidTile(kind) {
    return kind === TileKind.Brick || kind === TileKind.Steel || kind === TileKind.Water;
  }

  function canMoveThrough(kind) {
    return kind === TileKind.Empty || kind === TileKind.Grass;
  }

  function collidesWithMap(map, r) {
    const tr = rectToTileRange(r);
    for (let ty = tr.y0; ty <= tr.y1; ty++) {
      for (let tx = tr.x0; tx <= tr.x1; tx++) {
        const kind = tileAt(map, tx, ty);
        if (isSolidTile(kind)) {
          const cell = rect(tx * TILE, ty * TILE, TILE, TILE);
          if (aabbIntersects(r, cell)) return true;
        }
      }
    }
    return false;
  }

  function rayHitTile(map, r) {
    // For bullets: find first solid tile overlapped by bullet rect.
    const tr = rectToTileRange(r);
    for (let ty = tr.y0; ty <= tr.y1; ty++) {
      for (let tx = tr.x0; tx <= tr.x1; tx++) {
        const kind = tileAt(map, tx, ty);
        if (isSolidTile(kind)) {
          const cell = rect(tx * TILE, ty * TILE, TILE, TILE);
          if (aabbIntersects(r, cell)) return { tx, ty, kind };
        }
      }
    }
    return null;
  }

  function drawRoundedRect(c, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    c.beginPath();
    c.moveTo(x + rr, y);
    c.arcTo(x + w, y, x + w, y + h, rr);
    c.arcTo(x + w, y + h, x, y + h, rr);
    c.arcTo(x, y + h, x, y, rr);
    c.arcTo(x, y, x + w, y, rr);
    c.closePath();
  }

  function drawTank(c, t, color) {
    // Body
    c.save();
    c.translate(t.x + t.w / 2, t.y + t.h / 2);

    const angle =
      t.dir === "up" ? -Math.PI / 2 : t.dir === "down" ? Math.PI / 2 : t.dir === "left" ? Math.PI : 0;
    c.rotate(angle);

    c.translate(-t.w / 2, -t.h / 2);
    c.fillStyle = "rgba(0,0,0,0.25)";
    drawRoundedRect(c, 2, 3, t.w, t.h, 7);
    c.fill();

    c.fillStyle = color;
    drawRoundedRect(c, 0, 0, t.w, t.h, 7);
    c.fill();

    // Tracks
    c.fillStyle = "rgba(15,23,42,0.65)";
    c.fillRect(2, 3, 6, t.h - 6);
    c.fillRect(t.w - 8, 3, 6, t.h - 6);

    // Turret
    c.fillStyle = "rgba(255,255,255,0.14)";
    drawRoundedRect(c, t.w * 0.23, t.h * 0.22, t.w * 0.54, t.h * 0.56, 8);
    c.fill();

    // Barrel (forward)
    c.fillStyle = "rgba(15,23,42,0.72)";
    c.fillRect(t.w * 0.46, -t.h * 0.12, t.w * 0.08, t.h * 0.36);
    c.restore();
  }

  function sfxBeep(freq = 180, durationMs = 60, type = "square", gain = 0.035) {
    // Tiny WebAudio beep. Safe fallback when blocked.
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ac = new AudioCtx();
      const o = ac.createOscillator();
      const g = ac.createGain();
      o.type = type;
      o.frequency.value = freq;
      g.gain.value = gain;
      o.connect(g);
      g.connect(ac.destination);
      o.start();
      o.stop(ac.currentTime + durationMs / 1000);
      setTimeout(() => ac.close().catch(() => {}), durationMs + 30);
    } catch {
      // ignore
    }
  }

  function createMap() {
    // simple curated map with a few lanes and obstacles
    const map = Array.from({ length: MAP_H }, () => Array.from({ length: MAP_W }, () => TileKind.Empty));

    // Border steel
    for (let x = 0; x < MAP_W; x++) {
      setTile(map, x, 0, TileKind.Steel);
      setTile(map, x, MAP_H - 1, TileKind.Steel);
    }
    for (let y = 0; y < MAP_H; y++) {
      setTile(map, 0, y, TileKind.Steel);
      setTile(map, MAP_W - 1, y, TileKind.Steel);
    }

    // Brick clusters
    const bricks = [
      [6, 4],
      [7, 4],
      [8, 4],
      [6, 5],
      [8, 5],
      [6, 6],
      [7, 6],
      [8, 6],

      [17, 4],
      [18, 4],
      [19, 4],
      [17, 5],
      [19, 5],
      [17, 6],
      [18, 6],
      [19, 6],

      [12, 9],
      [13, 9],
      [12, 10],
      [13, 10],
      [12, 11],
      [13, 11],
    ];
    for (const [x, y] of bricks) setTile(map, x, y, TileKind.Brick);

    // Steel pillars
    for (let y = 3; y <= 16; y += 2) {
      setTile(map, 3, y, TileKind.Steel);
      setTile(map, MAP_W - 4, y, TileKind.Steel);
    }

    // Water line
    for (let x = 9; x <= 16; x++) setTile(map, x, 14, TileKind.Water);

    // Grass patches (visual)
    for (let y = 2; y <= 7; y++) for (let x = 11; x <= 14; x++) if (tileAt(map, x, y) === TileKind.Empty) setTile(map, x, y, TileKind.Grass);
    for (let y = 15; y <= 18; y++) for (let x = 5; x <= 7; x++) if (tileAt(map, x, y) === TileKind.Empty) setTile(map, x, y, TileKind.Grass);

    // Base protection (bricks around spawn)
    const bx = Math.floor(MAP_W / 2);
    const by = MAP_H - 3;
    for (let y = by - 1; y <= by; y++) for (let x = bx - 2; x <= bx + 2; x++) setTile(map, x, y, TileKind.Brick);
    setTile(map, bx, by, TileKind.Empty);

    return map;
  }

  function spawnPlayer() {
    return {
      id: "player",
      x: Math.floor(MAP_W / 2) * TILE + 2,
      y: (MAP_H - 2) * TILE - 30,
      w: 28,
      h: 28,
      dir: /** @type {Dir} */ ("up"),
      speed: 160,
      fireCooldown: 0,
      invuln: 0,
      alive: true,
    };
  }

  function spawnEnemy(i) {
    const spawns = [
      { x: 2 * TILE + 2, y: 2 * TILE + 2 },
      { x: Math.floor(MAP_W / 2) * TILE + 2, y: 2 * TILE + 2 },
      { x: (MAP_W - 3) * TILE + 2, y: 2 * TILE + 2 },
    ];
    const p = spawns[i % spawns.length];
    return {
      id: `enemy_${i}_${Math.random().toString(16).slice(2)}`,
      x: p.x,
      y: p.y,
      w: 28,
      h: 28,
      dir: /** @type {Dir} */ ("down"),
      speed: 110,
      fireCooldown: 0.5 + Math.random() * 0.8,
      think: 0,
      alive: true,
    };
  }

  function spawnBullet(owner, x, y, dir) {
    const v = DIR_VEC[dir];
    const speed = owner === "player" ? 360 : 310;
    return {
      owner,
      x,
      y,
      w: 8,
      h: 8,
      vx: v.x * speed,
      vy: v.y * speed,
      alive: true,
    };
  }

  function tryFire(state, who) {
    if (!who.alive) return;
    if (who.fireCooldown > 0) return;
    const v = DIR_VEC[who.dir];
    const cx = who.x + who.w / 2;
    const cy = who.y + who.h / 2;
    const bx = cx + v.x * 18 - 4;
    const by = cy + v.y * 18 - 4;
    state.bullets.push(spawnBullet(who.id === "player" ? "player" : "enemy", bx, by, who.dir));
    who.fireCooldown = who.id === "player" ? 0.28 : 0.9 + Math.random() * 0.65;
    sfxBeep(who.id === "player" ? 320 : 200, 50, "square", 0.02);
  }

  function moveTank(state, t, dx, dy, dt) {
    if (!t.alive) return;
    const nx = t.x + dx * t.speed * dt;
    const ny = t.y + dy * t.speed * dt;

    // Move axis-separated for stable collisions
    if (dx !== 0) {
      const r = rect(nx, t.y, t.w, t.h);
      if (!collidesWithMap(state.map, r) && !collidesWithTanks(state, t, r)) t.x = nx;
    }
    if (dy !== 0) {
      const r = rect(t.x, ny, t.w, t.h);
      if (!collidesWithMap(state.map, r) && !collidesWithTanks(state, t, r)) t.y = ny;
    }
  }

  function collidesWithTanks(state, me, r) {
    const tanks = [state.player, ...state.enemies].filter((x) => x && x.alive);
    for (const t of tanks) {
      if (t === me) continue;
      if (aabbIntersects(r, rect(t.x, t.y, t.w, t.h))) return true;
    }
    return false;
  }

  function updateHud(state) {
    elScore.textContent = String(state.score);
    elLives.textContent = String(state.lives);
    elEnemies.textContent = String(state.enemies.filter((e) => e.alive).length);
  }

  // 简化需求：完全关闭弹窗相关逻辑（不再显示暂停/结束面板）
  function showOverlay(_title, _body) {
    // 不做任何事情，保持 overlay 一直隐藏
    overlay.hidden = true;
  }

  function hideOverlay() {
    overlay.hidden = true;
  }

  function reset() {
    const state = {
      map: createMap(),
      player: spawnPlayer(),
      enemies: Array.from({ length: 5 }, (_, i) => spawnEnemy(i)),
      bullets: [],
      particles: [],
      score: 0,
      lives: 3,
      gameOver: false,
      paused: false,
      time: 0,
    };
    updateHud(state);
    hideOverlay();
    return state;
  }

  function explode(state, x, y, color) {
    for (let i = 0; i < 18; i++) {
      const a = Math.random() * Math.PI * 2;
      const s = 60 + Math.random() * 160;
      state.particles.push({
        x,
        y,
        vx: Math.cos(a) * s,
        vy: Math.sin(a) * s,
        life: 0.45 + Math.random() * 0.35,
        t: 0,
        r: 2 + Math.random() * 2.5,
        color,
      });
    }
  }

  function updatePlayer(state, dt) {
    const p = state.player;
    if (!p.alive) return;
    let dx = 0;
    let dy = 0;

    const up = keys.has("ArrowUp") || keys.has("KeyW");
    const down = keys.has("ArrowDown") || keys.has("KeyS");
    const left = keys.has("ArrowLeft") || keys.has("KeyA");
    const right = keys.has("ArrowRight") || keys.has("KeyD");

    if (up) {
      dy -= 1;
      p.dir = "up";
    } else if (down) {
      dy += 1;
      p.dir = "down";
    }
    if (left) {
      dx -= 1;
      p.dir = "left";
    } else if (right) {
      dx += 1;
      p.dir = "right";
    }

    // normalize diagonal
    if (dx !== 0 && dy !== 0) {
      dx *= 0.7071;
      dy *= 0.7071;
    }
    moveTank(state, p, dx, dy, dt);

    const fire = justPressed.has("Space") || justPressed.has("Enter");
    if (fire) tryFire(state, p);

    p.fireCooldown = Math.max(0, p.fireCooldown - dt);
    p.invuln = Math.max(0, p.invuln - dt);
  }

  function chooseEnemyDir(state, e) {
    // pick a direction that's not immediately blocked
    const options = [...DIRS];
    // slight bias toward player
    const px = state.player.x + state.player.w / 2;
    const py = state.player.y + state.player.h / 2;
    const ex = e.x + e.w / 2;
    const ey = e.y + e.h / 2;
    const bias =
      Math.abs(px - ex) > Math.abs(py - ey) ? (px < ex ? "left" : "right") : py < ey ? "up" : "down";
    if (Math.random() < 0.45) options.unshift(bias);

    for (const d of options) {
      const v = DIR_VEC[d];
      const probe = rect(e.x + v.x * 6, e.y + v.y * 6, e.w, e.h);
      if (!collidesWithMap(state.map, probe) && !collidesWithTanks(state, e, probe)) return d;
    }
    return e.dir;
  }

  function updateEnemies(state, dt) {
    for (const e of state.enemies) {
      if (!e.alive) continue;
      e.think -= dt;
      if (e.think <= 0) {
        e.dir = chooseEnemyDir(state, e);
        e.think = 0.25 + Math.random() * 0.7;
      }

      const v = DIR_VEC[e.dir];
      moveTank(state, e, v.x, v.y, dt);

      // shoot sometimes, more likely if aligned with player
      e.fireCooldown = Math.max(0, e.fireCooldown - dt);
      if (e.fireCooldown <= 0) {
        const aligned =
          (Math.abs((e.x + e.w / 2) - (state.player.x + state.player.w / 2)) < TILE * 0.45 && Math.random() < 0.65) ||
          (Math.abs((e.y + e.h / 2) - (state.player.y + state.player.h / 2)) < TILE * 0.45 && Math.random() < 0.65);
        if (Math.random() < 0.38 || aligned) {
          // face toward player a bit
          if (aligned) e.dir = chooseEnemyDir(state, e);
          tryFire(state, e);
        } else {
          e.fireCooldown = 0.2 + Math.random() * 0.5;
        }
      }
    }
  }

  function updateBullets(state, dt) {
    for (const b of state.bullets) {
      if (!b.alive) continue;
      b.x += b.vx * dt;
      b.y += b.vy * dt;

      // out of bounds
      if (b.x < -20 || b.y < -20 || b.x > W + 20 || b.y > H + 20) {
        b.alive = false;
        continue;
      }

      const br = rect(b.x, b.y, b.w, b.h);

      // map collision
      const hit = rayHitTile(state.map, br);
      if (hit) {
        if (hit.kind === TileKind.Brick) {
          setTile(state.map, hit.tx, hit.ty, TileKind.Empty);
          explode(state, hit.tx * TILE + TILE / 2, hit.ty * TILE + TILE / 2, "rgba(245,158,11,0.9)");
          sfxBeep(140, 70, "square", 0.016);
        } else {
          explode(state, hit.tx * TILE + TILE / 2, hit.ty * TILE + TILE / 2, "rgba(148,163,184,0.9)");
          sfxBeep(110, 60, "square", 0.012);
        }
        b.alive = false;
        continue;
      }

      // tank collision
      if (b.owner === "enemy") {
        const p = state.player;
        if (p.alive && p.invuln <= 0 && aabbIntersects(br, rect(p.x, p.y, p.w, p.h))) {
          b.alive = false;
          p.alive = false;
          explode(state, p.x + p.w / 2, p.y + p.h / 2, "rgba(34,197,94,0.95)");
          sfxBeep(90, 120, "sawtooth", 0.02);
          state.lives -= 1;
          if (state.lives > 0) {
            setTimeout(() => {
              state.player = spawnPlayer();
              state.player.invuln = 1.4;
              updateHud(state);
            }, 450);
          } else {
            state.gameOver = true;
            state.paused = true;
            showOverlay("游戏结束", `分数：${state.score}\n\n按 R 或点击“重新开始”再来一局。`);
          }
          updateHud(state);
        }
      } else {
        for (const e of state.enemies) {
          if (!e.alive) continue;
          if (aabbIntersects(br, rect(e.x, e.y, e.w, e.h))) {
            b.alive = false;
            e.alive = false;
            explode(state, e.x + e.w / 2, e.y + e.h / 2, "rgba(239,68,68,0.95)");
            sfxBeep(120, 90, "square", 0.018);
            state.score += 100;
            updateHud(state);
            break;
          }
        }
      }
    }
    state.bullets = state.bullets.filter((x) => x.alive);
  }

  function updateParticles(state, dt) {
    for (const p of state.particles) {
      p.t += dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= Math.pow(0.25, dt);
      p.vy *= Math.pow(0.25, dt);
    }
    state.particles = state.particles.filter((p) => p.t < p.life);
  }

  function drawMap(state) {
    // background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, W, H);

    // subtle grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= W; x += TILE) {
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, H);
    }
    for (let y = 0; y <= H; y += TILE) {
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(W, y + 0.5);
    }
    ctx.stroke();

    // tiles
    for (let y = 0; y < MAP_H; y++) {
      for (let x = 0; x < MAP_W; x++) {
        const kind = state.map[y][x];
        const px = x * TILE;
        const py = y * TILE;
        if (kind === TileKind.Empty) continue;
        if (kind === TileKind.Grass) {
          ctx.fillStyle = COLORS.grass;
          ctx.fillRect(px + 2, py + 2, TILE - 4, TILE - 4);
        } else if (kind === TileKind.Water) {
          ctx.fillStyle = COLORS.water;
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = "rgba(255,255,255,0.08)";
          ctx.fillRect(px + 2, py + 6, TILE - 4, 6);
          ctx.fillRect(px + 6, py + 18, TILE - 12, 6);
        } else if (kind === TileKind.Steel) {
          ctx.fillStyle = COLORS.steel;
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = "rgba(15,23,42,0.18)";
          ctx.fillRect(px + 3, py + 3, TILE - 6, TILE - 6);
          ctx.strokeStyle = "rgba(255,255,255,0.18)";
          ctx.strokeRect(px + 0.5, py + 0.5, TILE - 1, TILE - 1);
        } else if (kind === TileKind.Brick) {
          ctx.fillStyle = COLORS.brick;
          ctx.fillRect(px, py, TILE, TILE);
          ctx.fillStyle = COLORS.brick2;
          for (let i = 0; i < 3; i++) ctx.fillRect(px + 2, py + 4 + i * 10, TILE - 4, 4);
          ctx.strokeStyle = "rgba(0,0,0,0.15)";
          ctx.strokeRect(px + 0.5, py + 0.5, TILE - 1, TILE - 1);
        }
      }
    }
  }

  function drawEntities(state) {
    // bullets
    for (const b of state.bullets) {
      ctx.fillStyle = b.owner === "player" ? COLORS.bulletP : COLORS.bulletE;
      ctx.fillRect(b.x, b.y, b.w, b.h);
      ctx.fillStyle = "rgba(0,0,0,0.22)";
      ctx.fillRect(b.x + 1, b.y + 1, b.w - 2, b.h - 2);
    }

    // enemies
    for (const e of state.enemies) {
      if (!e.alive) continue;
      drawTank(ctx, e, COLORS.enemy);
    }

    // player
    if (state.player.alive) {
      const p = state.player;
      const blink = p.invuln > 0 && Math.floor(state.time * 14) % 2 === 0;
      if (!blink) drawTank(ctx, p, COLORS.player);
      else {
        ctx.globalAlpha = 0.45;
        drawTank(ctx, p, COLORS.player);
        ctx.globalAlpha = 1;
      }
    }
  }

  function drawParticles(state) {
    for (const p of state.particles) {
      const k = 1 - p.t / p.life;
      ctx.globalAlpha = clamp(k, 0, 1);
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * (0.8 + 0.6 * k), 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }

  function drawUI(state) {
    // If won
    const aliveEnemies = state.enemies.filter((e) => e.alive).length;
    if (!state.gameOver && aliveEnemies === 0) {
      state.paused = true;
      showOverlay("你赢了！", `分数：${state.score}\n\n按 R 或点击“重新开始”再开一局。`);
    }
  }

  function tick(state, dt) {
    if (state.paused) return;
    state.time += dt;
    updatePlayer(state, dt);
    updateEnemies(state, dt);
    updateBullets(state, dt);
    updateParticles(state, dt);
  }

  function render(state) {
    // Ensure UI 与状态同步：如果没有暂停且不是游戏结束，就强制隐藏暂停层
    if (!state.paused && !state.gameOver && !overlay.hidden) {
      hideOverlay();
    }

    drawMap(state);
    drawEntities(state);
    drawParticles(state);
    drawUI(state);
  }

  function bindInputs(stateRef) {
    window.addEventListener("keydown", (e) => {
      const code = e.code;
      if (code === "Space" || code === "ArrowUp" || code === "ArrowDown" || code === "ArrowLeft" || code === "ArrowRight")
        e.preventDefault();

      if (!keys.has(code)) justPressed.add(code);
      keys.add(code);

      if (code === "KeyP" || code === "Escape") {
        stateRef.state.paused = !stateRef.state.paused;
        if (stateRef.state.paused && !stateRef.state.gameOver) showOverlay("已暂停", "按 Esc / P 继续。\n也可以点“继续”。");
        if (!stateRef.state.paused) hideOverlay();
      }

      if (code === "KeyR") {
        stateRef.state = reset();
      }
    });

    window.addEventListener("keyup", (e) => {
      keys.delete(e.code);
    });

    btnResume.addEventListener("click", () => {
      if (stateRef.state.gameOver) return;
      stateRef.state.paused = false;
      hideOverlay();
      canvas.focus?.();
    });

    btnRestart.addEventListener("click", () => {
      stateRef.state = reset();
      canvas.focus?.();
    });
  }

  function main() {
    canvas.width = W;
    canvas.height = H;

    const stateRef = { state: reset() };
    bindInputs(stateRef);

    let last = performance.now();
    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      const dt = clamp((now - last) / 1000, 0, 0.033);
      last = now;

      const st = stateRef.state;
      tick(st, dt);
      render(st);

      justPressed.clear();
    };
    raf = requestAnimationFrame(loop);
  }

  main();
})();

