// UI for the kids demo. All the timing lives in demo_server.py: this page polls /state and
// draws whatever phase the server is in (idle -> intro -> try -> result -> dream -> ... -> finale).

const $ = (id) => document.getElementById(id);
const FRAME = 64; // dataset frames are 64x64, sprites are all the frames of an episode side by side

let INFO = null;
let state = null;
let phaseId = -1;
let phaseLocalStart = 0; // performance.now() at the server's phase_started
const sprites = {};

// ------------------------------------------------------------------ texts (Italian)
const INTRO = [
  { emoji: '🤔', title: 'Prova 1', text: 'Il robot <b>non sa ancora</b> come far girare la ruota. Proviamo!' },
  { emoji: '💡', title: 'Prova 2', text: 'Ora prova quello che ha <b>immaginato nel sogno</b>!' },
  { emoji: '💪', title: 'Prova finale', text: 'Dopo tante prove e tanti sogni... <b>ce la farà?</b>' },
];
const CAPTION = ['Prova a caso...', 'Sta migliorando...', 'Ora sa cosa fare!'];

function resultText(i) {
  const res = state.results;
  const deg = Math.round(res[i] ?? 0);
  const last = i === INFO.tries.length - 1;
  if (deg >= INFO.goal_deg)
    return { emoji: '🎉', title: "Ce l'ha fatta!", text: `La ruota ha girato di <b>${deg}°</b>: più di mezzo giro!` };
  if (last)
    return { emoji: '💪', title: "Ci è andato vicino!", text: `La ruota ha girato di <b>${deg}°</b>. Con qualche altra prova ce la farà!` };
  if (i === 0)
    return { emoji: '😅', title: "Ops! Non ce l'ha fatta", text: `La ruota ha girato solo di <b>${deg}°</b>. Sbagliare è normale: adesso il robot <b>ci pensa su</b>...` };
  const prev = Math.round(res[i - 1] ?? 0);
  if (deg > prev)
    return { emoji: '🙂', title: 'Meglio di prima!', text: `<b>${deg}°</b> invece di ${prev}°. Sta imparando! Ma può fare ancora meglio...` };
  return { emoji: '🤔', title: 'Quasi...', text: `Non è facile! Il robot ci pensa ancora un po'...` };
}

// ------------------------------------------------------------------ setup
async function init() {
  INFO = await (await fetch('/info')).json();
  document.body.classList.add(INFO.mode === 'live' ? 'mode-live' : 'mode-dry');
  if (INFO.has_wheel) {
    document.body.classList.add('has-wheel');
    connectVideo($('wheelVideo'), '/wheel.mjpg');
  }
  if (INFO.mode === 'live') connectVideo($('liveVideo'), '/video.mjpg');
  else $('modeTag').textContent = 'modalità prova: senza robot (immagini registrate)';

  for (const t of INFO.tries) loadSprite(t.ep);
  for (const d of INFO.dreams) for (const c of d) loadSprite(c.ep);

  buildGear();
  buildStars();
  placeGoalFlag();

  $('startBtn').onclick = start;
  $('nextBtn').onclick = () => post('/next');
  $('stopBtn').onclick = () => post('/stop');
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') post('/stop');
    else if (e.key === ' ' && document.body.classList.contains('waiting')) { e.preventDefault(); post('/next'); }
    else if (e.key === 'Enter' && (state?.phase === 'idle' || state?.phase === 'finale')) start();
  });

  poll();
  requestAnimationFrame(frame);
}

// mjpeg stream in an <img>: if it drops (e.g. the server was restarted) reconnect after a second
function connectVideo(img, url) {
  img.onerror = () => setTimeout(() => { img.src = `${url}?t=${Date.now()}`; }, 1000);
  img.src = `${url}?t=${Date.now()}`;
}

function loadSprite(ep) {
  if (sprites[ep]) return;
  const img = new Image();
  img.src = `/sprite/${ep}.png`;
  sprites[ep] = img;
}

function post(url) { return fetch(url, { method: 'POST' }).catch(() => {}); }
function start() { post('/start'); }

async function poll() {
  try {
    const s = await (await fetch('/state')).json();
    s.recvLocal = performance.now();
    state = s;
    if (s.phase_id !== phaseId) {
      phaseId = s.phase_id;
      phaseLocalStart = performance.now() - (s.now - s.phase_started) * 1000;
      onPhase();
    }
    document.body.classList.toggle('waiting', !!s.waiting_click);
  } catch (e) { /* server restarting, keep polling */ }
  setTimeout(poll, 100);
}

// ------------------------------------------------------------------ phase changes
function onPhase() {
  const { phase, idx } = state;
  document.body.className = document.body.className.replace(/\bphase-\w+/g, '').trim();
  document.body.classList.add(`phase-${phase}`);
  document.body.classList.toggle('timed', (phase === 'intro' || phase === 'result') && !!state.duration);

  // journey: stage 2*i for the tries, 2*i+1 for the dreams, 5 for the end
  const stage = phase === 'idle' ? -1 : phase === 'finale' ? 5 : phase === 'dream' ? 2 * idx + 1 : 2 * idx;
  document.querySelectorAll('#journey li').forEach((li) => {
    const s = +li.dataset.stage;
    li.classList.toggle('active', s === stage);
    li.classList.toggle('done', s < stage || (phase === 'finale' && s === 5));
  });

  $('startBtn').textContent = phase === 'finale' ? '↺ Ancora!' : '▶ Inizia!';
  const VIEW = { top: "📷 Il robot visto dall'alto", front: '📷 Il robot visto davanti', recorded: '📷 Il robot visto dalla telecamera' };
  $('viewTitle').textContent = phase === 'dream' ? '💭 Il robot sogna...' : VIEW[INFO.main_view];
  $('caption').classList.toggle('show', phase === 'try');

  if (phase === 'idle') {
    setCard('🐙', 'Come impara un robot?',
      "Questo robot morbido deve imparare a <b>far girare la ruota</b> usando i suoi <b>muscoli d'aria</b>. Vediamo come ci riesce!");
    setPressure([0, 0, 0]);
    setProgress(0);
  } else if (phase === 'intro') {
    const t = INTRO[Math.min(idx, INTRO.length - 1)];
    const r = INFO.tries[idx].round;
    const extra = idx > 0 ? `<br><small>(è già la sua prova numero ${r}!)</small>` : '';
    setCard(t.emoji, t.title, t.text + extra);
    setProgress(0);
  } else if (phase === 'try') {
    $('caption').innerHTML = `<span class="rec">●</span> ${CAPTION[Math.min(idx, CAPTION.length - 1)]}`;
  } else if (phase === 'result') {
    const t = resultText(idx);
    setCard(t.emoji, t.title, t.text);
    if ((state.results[idx] ?? 0) >= INFO.goal_deg) confetti(2500);
  } else if (phase === 'dream') {
    buildTiles(INFO.dreams[idx]);
  } else if (phase === 'finale') {
    setCard('🏆', 'Bravo robot!',
      'Provare, sbagliare, <b>pensarci su</b> e riprovare: così impara il robot... <b>e così impariamo anche noi!</b>');
    buildChart();
    confetti(5000);
  }
}

function setCard(emoji, title, text) {
  $('cardEmoji').textContent = emoji;
  $('cardTitle').textContent = title;
  $('cardText').innerHTML = text;
}

// ------------------------------------------------------------------ per-frame drawing
function frame() {
  requestAnimationFrame(frame);
  if (!state || !INFO) return;
  const t = (performance.now() - phaseLocalStart) / 1000;
  const { phase, idx } = state;

  if (document.body.classList.contains('timed')) {
    $('cardTimer').firstElementChild.style.width = `${Math.min(100, (100 * t) / state.duration)}%`;
  }

  if (phase === 'dream') return drawDream(INFO.dreams[idx], t);

  // real robot phases: pressures/rotation come from the server
  if (phase === 'try' || phase === 'result') {
    setPressure(state.pressure);
    setProgress(state.progress_deg);
  } else if (phase === 'intro') {
    setPressure(state.pressure);
  }
  if (INFO.mode === 'dry') {
    const ep = INFO.tries[Math.min(idx, INFO.tries.length - 1)].ep;
    const step = phase === 'try' || phase === 'result' ? state.step : 0;
    drawSprite($('recCanvas'), ep, step);
  }
}

function drawSprite(canvas, ep, step) {
  const img = sprites[ep];
  if (!img || !img.complete || !img.naturalWidth) return;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, step * FRAME, 0, FRAME, FRAME, 0, 0, FRAME, FRAME);
}

function drawDream(clips, t) {
  // clip i lasts n_steps/fps seconds, then pauses a bit to show its score
  let acc = 0;
  for (let i = 0; i < clips.length; i++) {
    const c = clips[i];
    const play = c.n_steps / INFO.dream_fps;
    const len = play + INFO.dream_clip_pause;
    if (t < acc + len || i === clips.length - 1) {
      const local = Math.max(0, t - acc);
      const step = Math.min(c.n_steps - 1, Math.floor(local * INFO.dream_fps));
      drawSprite($('dreamCanvas'), c.ep, step);
      setPressure(c.pressures[step]);
      setProgress(c.progress[step]);
      $('dreamLabel').textContent = `Sogno ${i + 1} di ${clips.length}`;
      updateTiles(clips, i, local >= play);
      return;
    }
    acc += len;
  }
}

// ------------------------------------------------------------------ dream tiles
function stars(deg) {
  const n = Math.min(5, Math.round(deg / 45));
  return n === 0 ? '😴' : '⭐'.repeat(n);
}

function buildTiles(clips) {
  $('dreamTiles').innerHTML = clips.map((c, i) =>
    `<div class="tile" id="tile${i}"><div class="t-name">Sogno ${i + 1}</div>
     <div class="t-stars">?</div><div class="t-deg">&nbsp;</div></div>`).join('');
}

function updateTiles(clips, current, currentDone) {
  let best = -1;
  clips.forEach((c, i) => {
    const el = $(`tile${i}`);
    if (!el) return;
    const done = i < current || (i === current && currentDone);
    el.classList.toggle('now', i === current && !currentDone);
    el.classList.toggle('done', done);
    el.querySelector('.t-stars').textContent = done ? stars(c.final_deg) : i === current ? '💭' : '?';
    el.querySelector('.t-deg').innerHTML = done ? `${Math.round(c.final_deg)}°` : '&nbsp;';
    if (done && (best < 0 || c.final_deg > clips[best].final_deg)) best = i;
  });
  clips.forEach((c, i) => $(`tile${i}`)?.classList.toggle('best', i === best));
}

// ------------------------------------------------------------------ balloons + wheel
function setPressure(p) {
  if (!p) return;
  document.querySelectorAll('.balloon').forEach((b, i) => {
    const f = Math.max(0, Math.min(1, p[i] / INFO.max_pressure));
    b.style.setProperty('--p', f.toFixed(3));
    $(`pct${i}`).textContent = `${Math.round(f * 100)}%`;
  });
}

function polar(r, deg) {
  const a = (deg * Math.PI) / 180;
  return [r * Math.sin(a), -r * Math.cos(a)];
}

function setProgress(deg) {
  deg = Math.max(0, deg || 0);
  $('degNum').textContent = Math.round(deg);
  $('gear').style.transform = `rotate(${deg}deg)`;
  const shown = Math.min(deg, 359.9);
  const [x, y] = polar(96, shown);
  $('arc').setAttribute('d', shown < 0.5 ? '' : `M 0 -96 A 96 96 0 ${shown > 180 ? 1 : 0} 1 ${x.toFixed(2)} ${y.toFixed(2)}`);
  $('arc').classList.toggle('win', deg >= INFO.goal_deg);
}

function placeGoalFlag() {
  $('goalFlag').setAttribute('transform', `rotate(${INFO.goal_deg})`);
  $('goalText').textContent = INFO.goal_deg === 180 ? '🚩 Traguardo: mezzo giro' : `🚩 Traguardo: ${INFO.goal_deg}°`;
}

function buildGear() {
  const teeth = 14, rOut = 70, rIn = 58;
  let d = '';
  for (let i = 0; i < teeth; i++) {
    const a0 = (360 / teeth) * i;
    const pts = [[rIn, a0], [rOut, a0 + 4], [rOut, a0 + 360 / teeth / 2 - 2], [rIn, a0 + 360 / teeth / 2 + 2]];
    for (const [r, a] of pts) {
      const [x, y] = polar(r, a);
      d += `${d ? 'L' : 'M'} ${x.toFixed(2)} ${y.toFixed(2)} `;
    }
  }
  $('gear').innerHTML = `<path class="gear-body" d="${d}Z"/>
    <circle r="16" class="gear-hub"/><circle cy="-42" r="9" class="gear-mark"/>`;
}

function buildStars() {
  const box = $('stars');
  for (let i = 0; i < 70; i++) {
    const s = document.createElement('div');
    s.className = 'star';
    s.style.left = `${Math.random() * 100}%`;
    s.style.top = `${Math.random() * 100}%`;
    s.style.animationDelay = `${-Math.random() * 3}s`;
    const k = 0.5 + Math.random() * 1.2;
    s.style.width = s.style.height = `${3 * k}px`;
    box.appendChild(s);
  }
}

// ------------------------------------------------------------------ finale chart
function buildChart() {
  const res = INFO.tries.map((t, i) => state.results[i] ?? 0);
  const top = Math.max(INFO.goal_deg * 1.15, ...res);
  const maxPx = 150, nameH = 36;
  const chart = $('chart');
  chart.innerHTML = res.map((r, i) =>
    `<div class="bar-col"><div class="bar-val">${Math.round(r)}°</div>
     <div class="bar ${r >= INFO.goal_deg ? 'win' : ''}" data-h="${Math.max(6, (r / top) * maxPx)}"></div>
     <div class="bar-name">Prova ${i + 1}</div></div>`).join('') +
    `<div class="goal-line" style="bottom:${nameH + (INFO.goal_deg / top) * maxPx}px"><span>🚩 traguardo</span></div>`;
  requestAnimationFrame(() => requestAnimationFrame(() =>
    chart.querySelectorAll('.bar').forEach((b) => { b.style.height = `${b.dataset.h}px`; })));
}

// ------------------------------------------------------------------ confetti
function confetti(ms) {
  const cv = $('confetti');
  const ctx = cv.getContext('2d');
  cv.width = innerWidth; cv.height = innerHeight;
  const colors = ['#ff7a1a', '#ffa24d', '#1d5fd1', '#4f8ff0', '#ffffff'];
  const parts = Array.from({ length: 220 }, () => ({
    x: Math.random() * cv.width, y: -20 - Math.random() * cv.height * 0.6,
    vx: (Math.random() - 0.5) * 3, vy: 2 + Math.random() * 4,
    r: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.3,
    w: 8 + Math.random() * 8, h: 5 + Math.random() * 6, c: colors[(Math.random() * colors.length) | 0],
  }));
  const end = performance.now() + ms;
  (function tick() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    const alive = performance.now() < end;
    for (const p of parts) {
      p.x += p.vx; p.y += p.vy; p.r += p.vr;
      if (p.y > cv.height + 20 && alive) { p.y = -20; p.x = Math.random() * cv.width; }
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r);
      ctx.fillStyle = p.c; ctx.strokeStyle = 'rgba(18,48,95,.15)';
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); ctx.strokeRect(-p.w / 2, -p.h / 2, p.w, p.h);
      ctx.restore();
    }
    if (alive || parts.some((p) => p.y < cv.height + 20)) requestAnimationFrame(tick);
    else ctx.clearRect(0, 0, cv.width, cv.height);
  })();
}

init();
