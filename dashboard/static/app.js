/* BMS Live Console — self-contained canvas frontend (no external libs). */
"use strict";

/* ---------------------------------------------------------------- palette */

const INK = "#e6edf3", INK2 = "#9198a1", INK3 = "#6e7681";
const EST = "#4493f8", REP = "#cd750d";
const CRIT = "#f85149", WARN = "#d29922";
const GRIDLINE = "#21262d";

function lerpHex(a, b, t) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  const c = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function ramp(stops) {
  return (t) => {
    t = Math.min(1, Math.max(0, t));
    const x = t * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2);
    return lerpHex(stops[i], stops[i + 1], x - i);
  };
}
/* sequential single-hue ramps (dark surface: low = dark, high = bright) */
const RAMP_BLUE = ramp(["#101d33", "#1d4079", "#2f6cc4", "#6bb1ff"]);
const RAMP_CYAN = ramp(["#0c2429", "#155e69", "#22a3b4", "#7ee5f2"]);
const RAMP_PURPLE = ramp(["#1d1530", "#43307a", "#7a5fc4", "#c4a8ff"]);
const RAMP_HEAT = ramp(["#1c0e07", "#6b2c12", "#c2601c", "#f8a03c", "#ffe08a"]);
/* diverging: charge (blue) <- neutral -> drain (orange) */
function divergeBal(t) { /* t in [-1, 1] */
  const m = "#3a404b";
  return t >= 0 ? lerpHex(m, REP, Math.min(1, t)) : lerpHex(m, EST, Math.min(1, -t));
}

/* ------------------------------------------------------------------ state */

const S = {
  cfg: { rows: 6, cols: 8 },
  snap: null,
  hist: [],           // recent snapshots (cap ~1300 ≈ 4 min at 5 Hz)
  sel: 20,
  metric: "v",
  dirty: false,
};
const HIST_WINDOW_S = 120;

/* ------------------------------------------------------------ canvas prep */

function prep(id) {
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const w = cv.width, h = cv.height;
  cv.style.width = w + "px";
  cv.style.height = h + "px";
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  return { cv, ctx, w, h };
}
const grid = prep("grid"), thermal = prep("thermal"), bars = prep("bars");
const chV = prep("chart-v"), chSoc = prep("chart-soc"), chI = prep("chart-i");

/* ----------------------------------------------------------------- fmt */

const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
const cellName = (i) => `cell ${i}`;
const cellPos = (i) => `module ${Math.floor(i / S.cfg.cols)} · pos ${i % S.cfg.cols}`;

const METRICS = {
  v: { get: (c, i) => c.v_true[i], ramp: RAMP_BLUE, fmt: (x) => fmt(x, 3) + " V", span: (vals) => padSpan(vals, 0.01) },
  soc: { get: (c, i) => c.soc_est[i], ramp: RAMP_CYAN, fmt: (x) => fmt(100 * x, 1) + " %", span: (vals) => padSpan(vals, 0.005) },
  r0: { get: (c, i) => c.r0_mohm[i], ramp: RAMP_PURPLE, fmt: (x) => fmt(x, 1) + " mΩ", span: (vals) => padSpan(vals, 0.3) },
  q: { get: (c, i) => c.q_est_ah[i], ramp: RAMP_PURPLE, fmt: (x) => fmt(x, 2) + " Ah", span: (vals) => padSpan(vals, 0.02) },
  bal: { get: (c, i) => c.bal_a[i], ramp: null, fmt: (x) => fmt(x, 2) + " A", span: () => null },
};

function padSpan(vals, minPad) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 2 * minPad) { const m = (hi + lo) / 2; lo = m - minPad; hi = m + minPad; }
  return [lo, hi];
}

/* --------------------------------------------------------------- heatmaps */

function cellRect(p, i) {
  const { rows, cols } = S.cfg;
  const padL = 6, padT = 4, padB = 20;
  const cw = (p.w - padL * 2) / cols, chh = (p.h - padT - padB) / rows;
  const r = Math.floor(i / cols), c = i % cols;
  return [padL + c * cw + 1.5, padT + r * chh + 1.5, cw - 3, chh - 3];
}

function drawCellGrid(p, values, colorFn, textFn) {
  const { ctx } = p;
  ctx.clearRect(0, 0, p.w, p.h);
  const cells = S.snap.cells;
  for (let i = 0; i < values.length; i++) {
    const [x, y, w, h] = cellRect(p, i);
    ctx.fillStyle = colorFn(i);
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 3);
    ctx.fill();
    if (cells.flagged[i]) {
      ctx.strokeStyle = cells.excluded[i] ? WARN : CRIT;
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    if (i === S.sel) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x - 1.5, y - 1.5, w + 3, h + 3);
    }
    if (textFn) {
      ctx.fillStyle = "rgba(230,237,243,0.85)";
      ctx.font = "10px 'Segoe UI'";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(textFn(i), x + w / 2, y + h / 2);
    }
  }
  ctx.fillStyle = INK3;
  ctx.font = "10px 'Segoe UI'";
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function drawColorbar(p, rampFn, lo, hi, fmtFn) {
  const { ctx } = p;
  const y = p.h - 12, x0 = 8, x1 = p.w - 8;
  for (let x = x0; x < x1; x++) {
    ctx.fillStyle = rampFn((x - x0) / (x1 - x0));
    ctx.fillRect(x, y, 1, 6);
  }
  ctx.fillStyle = INK3;
  ctx.font = "10px 'Segoe UI'";
  ctx.textAlign = "left";
  ctx.fillText(fmtFn(lo), x0, y - 2);
  ctx.textAlign = "right";
  ctx.fillText(fmtFn(hi), x1, y - 2);
}

function renderGrid() {
  const m = METRICS[S.metric], c = S.snap.cells;
  const vals = c.v_true.map((_, i) => m.get(c, i));
  let colorFn, lo = 0, hi = 0;
  if (S.metric === "bal") {
    const amp = Math.max(0.2, ...vals.map(Math.abs));
    colorFn = (i) => divergeBal(vals[i] / amp);
    lo = -amp; hi = amp;
    drawCellGrid(grid, vals, colorFn, (i) => Math.abs(vals[i]) > 0.005 ? vals[i].toFixed(1) : "");
    const { ctx } = grid, y = grid.h - 12;
    for (let x = 8; x < grid.w - 8; x++) {
      ctx.fillStyle = divergeBal(((x - 8) / (grid.w - 16)) * 2 - 1);
      ctx.fillRect(x, y, 1, 6);
    }
    ctx.fillStyle = INK3; ctx.font = "10px 'Segoe UI'";
    ctx.textAlign = "left"; ctx.fillText("charge " + fmt(-amp, 1) + " A", 8, y - 2);
    ctx.textAlign = "right"; ctx.fillText("drain +" + fmt(amp, 1) + " A", grid.w - 8, y - 2);
  } else {
    [lo, hi] = m.span(vals);
    colorFn = (i) => m.ramp((vals[i] - lo) / (hi - lo || 1));
    drawCellGrid(grid, vals, colorFn, null);
    drawColorbar(grid, m.ramp, lo, hi, m.fmt);
  }
}

function renderThermal() {
  const t = S.snap.cells.temp;
  const lo = Math.min(25, ...t), hi = Math.max(40, ...t);
  drawCellGrid(thermal, t, (i) => RAMP_HEAT((t[i] - lo) / (hi - lo || 1)),
    (i) => t[i] >= 45 ? Math.round(t[i]) + "°" : "");
  drawColorbar(thermal, RAMP_HEAT, lo, hi, (x) => fmt(x, 0) + " °C");
}

/* ------------------------------------------------------------------- bars */

function renderBars() {
  const { ctx, w, h } = bars, c = S.snap.cells;
  ctx.clearRect(0, 0, w, h);
  const padL = 30, padB = 14, padT = 6;
  const n = c.soc_est.length;
  const bw = (w - padL - 8) / n;
  const soc = c.soc_est, q = S.snap.cells.q_est_ah;
  const lo = Math.max(0, Math.min(...soc) - 0.04), hi = Math.min(1, Math.max(...soc) + 0.04);
  const yOf = (v) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);

  ctx.strokeStyle = GRIDLINE;
  ctx.lineWidth = 1;
  for (const g of [lo, (lo + hi) / 2, hi]) {
    ctx.beginPath(); ctx.moveTo(padL, yOf(g)); ctx.lineTo(w - 8, yOf(g)); ctx.stroke();
    ctx.fillStyle = INK3; ctx.font = "10px 'Segoe UI'"; ctx.textAlign = "right";
    ctx.fillText((100 * g).toFixed(0) + "%", padL - 4, yOf(g) + 3);
  }
  let qs = 0, qsum = 0;
  for (let i = 0; i < n; i++) { qs += soc[i] * q[i]; qsum += q[i]; }
  const mean = qs / qsum;

  for (let i = 0; i < n; i++) {
    const x = padL + i * bw;
    const b = c.bal_a[i];
    ctx.fillStyle = Math.abs(b) < 0.01 ? "#4d5566" : divergeBal(b / 1.0);
    const y = yOf(soc[i]);
    ctx.beginPath();
    ctx.roundRect(x + 1, y, bw - 2, h - padB - y, [3, 3, 0, 0]);
    ctx.fill();
    if (i === S.sel) {
      ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
    }
  }
  ctx.strokeStyle = INK2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(padL, yOf(mean)); ctx.lineTo(w - 8, yOf(mean)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = INK3; ctx.textAlign = "left"; ctx.font = "10px 'Segoe UI'";
  ctx.fillText("cell 0", padL, h - 3);
  ctx.textAlign = "right";
  ctx.fillText("cell 47", w - 8, h - 3);
}

/* ------------------------------------------------------------ line charts */

function renderChart(p, seriesList, yFmt, readoutId) {
  const { ctx, w, h } = p;
  ctx.clearRect(0, 0, w, h);
  const padL = 38, padR = 6, padT = 6, padB = 16;
  const tNow = S.snap.t, t0 = Math.max(0, tNow - HIST_WINDOW_S);

  let lo = Infinity, hi = -Infinity;
  for (const s of seriesList) {
    for (const [t, v] of s.pts) {
      if (t >= t0 && v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
  }
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = Math.max((hi - lo) * 0.12, (s => s)(seriesList[0].minPad ?? 0.001));
  lo -= pad; hi += pad;

  const xOf = (t) => padL + ((t - t0) / (tNow - t0 || 1)) * (w - padL - padR);
  const yOf = (v) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);

  ctx.strokeStyle = GRIDLINE; ctx.lineWidth = 1;
  ctx.font = "10px 'Segoe UI'"; ctx.fillStyle = INK3;
  for (let k = 0; k <= 3; k++) {
    const v = lo + ((hi - lo) * k) / 3, y = yOf(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(yFmt(v), padL - 3, y + 3);
  }
  for (let k = 0; k <= 2; k++) {
    const t = t0 + ((tNow - t0) * k) / 2;
    ctx.textAlign = k === 0 ? "left" : k === 2 ? "right" : "center";
    ctx.fillText(fmt(t, 0) + " s", xOf(t), h - 4);
  }

  for (const s of seriesList) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width ?? 2;
    if (s.dash) ctx.setLineDash(s.dash);
    ctx.beginPath();
    let started = false;
    for (const [t, v] of s.pts) {
      if (t < t0 || v == null) { started = false; continue; }
      const x = xOf(t), y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }
  p._map = { xOf, yOf, t0, tNow, lo, hi };
}

function seriesFromHist(pick, opts) {
  return { pts: S.hist.map((s) => [s.t, pick(s)]), ...opts };
}

function renderCharts() {
  const i = S.sel;
  document.getElementById("chart-v-cell").textContent = cellName(i);
  document.getElementById("chart-soc-cell").textContent = cellName(i);
  renderChart(chV, [
    seriesFromHist((s) => s.cells.v_true[i], { color: INK, width: 2, minPad: 0.004 }),
    seriesFromHist((s) => s.cells.v_rep[i], { color: REP, width: 1.2 }),
  ], (v) => fmt(v, 2), "chart-v-readout");
  renderChart(chSoc, [
    seriesFromHist((s) => s.cells.soc_true[i], { color: INK, width: 2, minPad: 0.004 }),
    seriesFromHist((s) => s.cells.soc_est[i], { color: EST, width: 2 }),
  ], (v) => (100 * v).toFixed(0) + "%");
  renderChart(chI, [
    seriesFromHist((s) => s.pack.i, { color: EST, width: 2, minPad: 0.2 }),
    seriesFromHist((s) => s.pack.i_request, { color: INK3, width: 1.2, dash: [4, 3] }),
  ], (v) => fmt(v, 1) + "A");
}

/* --------------------------------------------------------------- side UI */

function renderSide() {
  const snap = S.snap;
  const dEl = document.getElementById("diags");
  if (!snap.diagnoses.length) {
    dEl.innerHTML = '<div class="diag-empty">no faults detected</div>';
  } else {
    dEl.innerHTML = snap.diagnoses.map((d) => `
      <div class="diag-card ${d.severity}">
        <span class="sev">${d.severity}</span>
        <span class="kind">${d.kind.replace("_", " ")} · cell ${d.cell}</span>
        <div class="detail">t=${d.t}s · ${d.subtype} — ${d.detail}</div>
      </div>`).join("");
  }
  const evEl = document.getElementById("events");
  evEl.innerHTML = snap.events.slice().reverse().map((e) => {
    const cls = e.msg.startsWith("SHUTDOWN") || e.msg.startsWith("DERATE")
      ? "alarm" : e.msg.startsWith("INJECTED") ? "inject" : "";
    return `<div class="${cls}">t=${e.t}s — <b>${e.msg}</b></div>`;
  }).join("");
}

function renderHeader() {
  const s = S.snap, p = s.pack;
  document.getElementById("c-t").textContent = fmt(s.t, 0) + " s";
  document.getElementById("c-v").textContent = fmt(p.v, 1) + " V";
  document.getElementById("c-i").textContent = fmt(p.i, 2) + " A";
  document.getElementById("c-soc").textContent = fmt(100 * p.soc_mean_est, 1) + " %";
  document.getElementById("c-spread").textContent = fmt(100 * p.soc_spread_true, 1) + " %";
  document.getElementById("c-limit").textContent = fmt(100 * p.limit, 0) + " %";
  document.getElementById("c-loss").textContent = fmt(p.bal_loss_wh, 2) + " Wh";
  const badge = document.getElementById("c-contactor");
  badge.className = "badge " + (p.contactor_open ? "open" : "closed");
  badge.textContent = p.contactor_open ? "CONTACTOR OPEN" : "CONTACTOR CLOSED";
}

function renderOps() {
  document.getElementById("ops-cell").textContent = cellName(S.sel);
  document.getElementById("ops-pos").textContent = "(" + cellPos(S.sel) + ")";
}

/* ----------------------------------------------------------- render loop */

function render() {
  if (!S.snap) return;
  renderHeader();
  renderGrid();
  renderThermal();
  renderBars();
  renderCharts();
  renderSide();
  renderOps();
}
/* Time-throttled direct rendering: requestAnimationFrame is throttled or
   suspended in hidden/embedded panes, so the draw is driven straight off
   the WebSocket stream instead. force=true for user interactions. */
let lastRender = 0;
function requestRender(force) {
  const now = performance.now();
  if (force || now - lastRender >= 150) { lastRender = now; render(); }
}

/* -------------------------------------------------------------- hover UI */

function hitCell(p, ev) {
  const rect = p.cv.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  for (let i = 0; i < 48; i++) {
    const [x, y, w, h] = cellRect(p, i);
    if (mx >= x && mx <= x + w && my >= y && my <= y + h) return i;
  }
  return null;
}

function wireHeatmapHover(p, readoutId, describe) {
  p.cv.addEventListener("mousemove", (ev) => {
    const i = hitCell(p, ev);
    document.getElementById(readoutId).textContent = i == null ? " " : describe(i);
  });
  p.cv.addEventListener("mouseleave", () => {
    document.getElementById(readoutId).innerHTML = "&nbsp;";
  });
}

function describeCell(i) {
  const c = S.snap.cells;
  const rep = c.v_rep[i] == null ? "–" : fmt(c.v_rep[i], 3);
  return `${cellName(i)} (${cellPos(i)}) · ${fmt(c.v_true[i], 3)} V (rep ${rep}) · `
    + `${fmt(100 * c.soc_est[i], 1)}% SOC · ${fmt(c.temp[i], 1)} °C · `
    + `R0 ${fmt(c.r0_mohm[i], 1)} mΩ · Q ${fmt(c.q_est_ah[i], 2)} Ah`
    + (c.flagged[i] ? " · FLAGGED" : "");
}

grid.cv.addEventListener("click", (ev) => {
  const i = hitCell(grid, ev);
  if (i != null) { S.sel = i; requestRender(true); }
});
thermal.cv.addEventListener("click", (ev) => {
  const i = hitCell(thermal, ev);
  if (i != null) { S.sel = i; requestRender(true); }
});
bars.cv.addEventListener("click", (ev) => {
  const rect = bars.cv.getBoundingClientRect();
  const i = Math.floor(((ev.clientX - rect.left) - 30) / ((bars.w - 38) / 48));
  if (i >= 0 && i < 48) { S.sel = i; requestRender(true); }
});
wireHeatmapHover(grid, "grid-readout", describeCell);
wireHeatmapHover(thermal, "thermal-readout", describeCell);
bars.cv.addEventListener("mousemove", (ev) => {
  const rect = bars.cv.getBoundingClientRect();
  const i = Math.floor(((ev.clientX - rect.left) - 30) / ((bars.w - 38) / 48));
  document.getElementById("bars-readout").textContent =
    i >= 0 && i < 48 && S.snap
      ? `${cellName(i)} · SOC est ${fmt(100 * S.snap.cells.soc_est[i], 1)}% · balancing ${fmt(S.snap.cells.bal_a[i], 2)} A`
      : " ";
});

function wireChartHover(p, readoutId, describe) {
  p.cv.addEventListener("mousemove", (ev) => {
    if (!p._map || !S.hist.length) return;
    const rect = p.cv.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const t = p._map.t0 + ((mx - 38) / (p.w - 44)) * (p._map.tNow - p._map.t0);
    let best = null, bd = Infinity;
    for (const s of S.hist) {
      const d = Math.abs(s.t - t);
      if (d < bd) { bd = d; best = s; }
    }
    if (best) document.getElementById(readoutId).textContent = describe(best);
  });
  p.cv.addEventListener("mouseleave", () => {
    document.getElementById(readoutId).innerHTML = "&nbsp;";
  });
}
wireChartHover(chV, "chart-v-readout", (s) =>
  `t=${fmt(s.t, 1)}s · truth ${fmt(s.cells.v_true[S.sel], 3)} V · reported ${s.cells.v_rep[S.sel] == null ? "–" : fmt(s.cells.v_rep[S.sel], 3)} V`);
wireChartHover(chSoc, "chart-soc-readout", (s) =>
  `t=${fmt(s.t, 1)}s · truth ${fmt(100 * s.cells.soc_true[S.sel], 1)}% · UKF ${fmt(100 * s.cells.soc_est[S.sel], 1)}%`);
wireChartHover(chI, "chart-i-readout", (s) =>
  `t=${fmt(s.t, 1)}s · applied ${fmt(s.pack.i, 2)} A · requested ${fmt(s.pack.i_request, 2)} A`);

/* --------------------------------------------------------------- controls */

let ws = null;
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

document.getElementById("ctl-mode").addEventListener("change", (e) =>
  send({ cmd: "mode", value: e.target.value }));
document.getElementById("ctl-speed").addEventListener("change", (e) =>
  send({ cmd: "speed", value: parseFloat(e.target.value) }));
document.getElementById("ctl-bal").addEventListener("change", (e) =>
  send({ cmd: "balancer", value: e.target.checked }));
document.getElementById("ctl-reset").addEventListener("click", () => {
  send({ cmd: "reset" });
  S.hist = [];
});
for (const btn of document.querySelectorAll(".ops-buttons button")) {
  btn.addEventListener("click", () =>
    send({ cmd: "inject", kind: btn.dataset.kind, cell: S.sel }));
}

/* -------------------------------------------------------------- websocket */

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  const conn = document.getElementById("conn");
  ws.onopen = () => { conn.textContent = "live"; conn.className = "conn"; };
  ws.onclose = () => {
    conn.textContent = "disconnected — retrying";
    conn.className = "conn down";
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "config") { S.cfg = msg; return; }
    if (msg.type !== "state") return;
    if (S.hist.length && msg.t < S.hist[S.hist.length - 1].t) S.hist = []; // reset
    S.snap = msg;
    S.hist.push(msg);
    if (S.hist.length > 1300) S.hist.splice(0, S.hist.length - 1300);
    const mode = document.getElementById("ctl-mode");
    if (document.activeElement !== mode) mode.value = msg.mode;
    const bal = document.getElementById("ctl-bal");
    if (document.activeElement !== bal) bal.checked = msg.balancer_on;
    requestRender();
  };
}
connect();
