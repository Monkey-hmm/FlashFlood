"""
====================================================
 Flash Flood Prediction — FastAPI Server
====================================================
Serves a browser dashboard + REST API.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              → HTML dashboard
    GET  /features      → list of expected feature names
    POST /predict       → single-row prediction
    POST /predict/batch → batch prediction (list of rows)
    GET  /health        → liveness check
"""
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from inference import FloodPredictor

# ── Startup / shutdown ───────────────────────────────────────────────
predictor: FloodPredictor | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    print("Loading models…")
    predictor = FloodPredictor()
    yield
    predictor = None

app = FastAPI(
    title="Flash Flood Prediction API",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: dict[str, float]
    history:  list[dict[str, float]] | None = None   # optional LSTM context

class BatchRequest(BaseModel):
    rows: list[PredictRequest]


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/features")
def get_features():
    if predictor is None:
        raise HTTPException(503, "Models not loaded")
    return {"features": predictor.feature_names(), "count": len(predictor.feature_names())}


@app.post("/predict")
def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(503, "Models not loaded")
    t0 = time.perf_counter()
    result = predictor.predict(req.features, req.history)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(result)


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if predictor is None:
        raise HTTPException(503, "Models not loaded")
    if len(req.rows) > 1000:
        raise HTTPException(400, "Batch size exceeds 1000")
    t0 = time.perf_counter()
    results = [predictor.predict(r.features, r.history) for r in req.rows]
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse({"results": results, "count": len(results), "latency_ms": elapsed})


# ── HTML Dashboard ───────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>FloodWatch — Flash Flood Prediction</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<style>
/* ── Reset & tokens ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #050a0f;
  --surface:   #0b1520;
  --border:    #1a2d42;
  --muted:     #2a4060;
  --text:      #c8dff0;
  --dim:       #5a7a99;
  --accent:    #00c8ff;
  --green:     #22c55e;
  --amber:     #f59e0b;
  --orange:    #f97316;
  --red:       #ef4444;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

html { scroll-behavior: smooth; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 13px;
  min-height: 100vh;
  overflow-x: hidden;
}

/* animated grid background */
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 48px 48px;
  opacity: 0.35;
  pointer-events: none;
  z-index: 0;
}

/* ── Layout ─────────────────────────────────────── */
.wrap {
  position: relative; z-index: 1;
  max-width: 1120px;
  margin: 0 auto;
  padding: 32px 24px 80px;
}

/* ── Header ─────────────────────────────────────── */
header {
  display: flex; align-items: center; gap: 18px;
  margin-bottom: 48px;
  animation: fadeDown .5s ease both;
}

.logo {
  width: 48px; height: 48px;
  background: linear-gradient(135deg, #00c8ff22, #00c8ff55);
  border: 1px solid var(--accent);
  border-radius: 10px;
  display: grid; place-items: center;
  font-size: 22px;
}

header h1 {
  font-family: var(--font-head);
  font-size: 28px; font-weight: 800;
  letter-spacing: -0.5px;
  color: #fff;
}

header p {
  color: var(--dim);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 2px;
}

.status-dot {
  margin-left: auto;
  display: flex; align-items: center; gap: 8px;
  font-size: 11px; color: var(--dim);
}
.dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--muted);
  transition: background .3s;
}
.dot.live { background: var(--green); box-shadow: 0 0 8px var(--green); }

/* ── Section titles ─────────────────────────────── */
.section-label {
  font-family: var(--font-head);
  font-size: 11px; font-weight: 700;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Cards ──────────────────────────────────────── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
}

/* ── Grid layouts ───────────────────────────────── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }

@media (max-width: 680px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}

/* ── Result gauges ──────────────────────────────── */
.result-block {
  margin-bottom: 28px;
  animation: fadeUp .4s ease both;
}

.gauge-row {
  display: flex; align-items: center; gap: 14px;
  margin-bottom: 10px;
}

.gauge-label {
  font-family: var(--font-head);
  font-size: 12px; font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--dim);
  width: 72px; flex-shrink: 0;
}

.gauge-track {
  flex: 1; height: 8px;
  background: var(--muted);
  border-radius: 99px;
  overflow: hidden;
}

.gauge-fill {
  height: 100%;
  border-radius: 99px;
  width: 0%;
  transition: width .8s cubic-bezier(.23,1,.32,1);
}

.gauge-pct {
  width: 48px; text-align: right;
  font-size: 13px; font-weight: 600;
  color: #fff;
}

.flag-chip {
  padding: 2px 10px;
  border-radius: 99px;
  font-size: 10px; font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  border: 1px solid currentColor;
}

/* ── Risk banner ─────────────────────────────────── */
.risk-banner {
  display: flex; align-items: center; gap: 18px;
  padding: 20px 24px;
  border-radius: 12px;
  border: 1px solid;
  margin-bottom: 28px;
  transition: all .4s ease;
}

.risk-icon { font-size: 32px; }

.risk-text .label {
  font-family: var(--font-head);
  font-size: 22px; font-weight: 800;
}

.risk-text .sublabel {
  font-size: 11px; color: var(--dim);
  text-transform: uppercase; letter-spacing: 0.1em;
  margin-top: 2px;
}

.risk-prob {
  margin-left: auto;
  font-family: var(--font-head);
  font-size: 36px; font-weight: 800;
}

/* ── Input area ─────────────────────────────────── */
textarea {
  width: 100%;
  min-height: 220px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: 12px;
  padding: 14px;
  resize: vertical;
  outline: none;
  transition: border-color .2s;
  line-height: 1.6;
}
textarea:focus { border-color: var(--accent); }

/* ── Buttons ────────────────────────────────────── */
.btn-row { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }

button {
  font-family: var(--font-mono);
  font-size: 12px; font-weight: 600;
  border: none; border-radius: 8px;
  padding: 10px 20px;
  cursor: pointer;
  transition: transform .15s, opacity .15s;
}
button:active { transform: scale(.97); }
button:disabled { opacity: .4; cursor: not-allowed; }

.btn-primary {
  background: var(--accent);
  color: var(--bg);
}
.btn-primary:hover:not(:disabled) { opacity: .85; }

.btn-ghost {
  background: transparent;
  color: var(--dim);
  border: 1px solid var(--border);
}
.btn-ghost:hover { color: var(--text); border-color: var(--dim); }

/* ── Stat tiles ─────────────────────────────────── */
.stat-tile {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 18px;
}
.stat-tile .val {
  font-family: var(--font-head);
  font-size: 26px; font-weight: 800;
  color: #fff;
}
.stat-tile .key {
  font-size: 10px; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--dim);
  margin-top: 4px;
}

/* ── Log ────────────────────────────────────────── */
.log {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  max-height: 160px;
  overflow-y: auto;
  font-size: 11px;
  color: var(--dim);
  line-height: 1.7;
}
.log .entry { display: flex; gap: 10px; }
.log .ts { color: var(--muted); flex-shrink: 0; }
.log .msg.ok   { color: var(--green); }
.log .msg.warn { color: var(--amber); }
.log .msg.err  { color: var(--red);   }

/* ── Placeholder ─────────────────────────────────── */
.placeholder {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 10px;
  min-height: 200px;
  color: var(--muted);
  font-size: 12px;
  text-align: center;
}
.placeholder .icon { font-size: 36px; opacity: .4; }

/* ── Animations ─────────────────────────────────── */
@keyframes fadeDown {
  from { opacity: 0; transform: translateY(-14px); }
  to   { opacity: 1; transform: none; }
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: none; }
}

/* ── Spinner ─────────────────────────────────────── */
.spinner {
  display: inline-block;
  width: 14px; height: 14px;
  border: 2px solid var(--muted);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .6s linear infinite;
  vertical-align: middle;
  margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Feature pill list ──────────────────────────── */
.pill-wrap {
  display: flex; flex-wrap: wrap; gap: 6px;
  max-height: 130px; overflow-y: auto;
}
.pill {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 99px;
  padding: 3px 10px;
  font-size: 10px;
  color: var(--dim);
}

/* scrollbars */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--muted); border-radius: 99px; }
</style>
</head>

<body>
<div class="wrap">

  <!-- Header -->
  <header>
    <div class="logo">🌊</div>
    <div>
      <h1>FloodWatch</h1>
      <p>Flash Flood Prediction — XGBoost + LSTM Ensemble</p>
    </div>
    <div class="status-dot">
      <div class="dot" id="dot"></div>
      <span id="status-text">connecting…</span>
    </div>
  </header>

  <!-- Stats row -->
  <div class="section-label">System</div>
  <div class="grid-3" style="margin-bottom:32px" id="stat-row">
    <div class="stat-tile"><div class="val" id="st-features">—</div><div class="key">Features</div></div>
    <div class="stat-tile"><div class="val" id="st-xgb-t">—</div><div class="key">XGB Threshold</div></div>
    <div class="stat-tile"><div class="val" id="st-lstm-t">—</div><div class="key">LSTM Threshold</div></div>
  </div>

  <!-- Main grid -->
  <div class="grid-2" style="margin-bottom:32px">

    <!-- Left: Input -->
    <div>
      <div class="section-label">Input Features (JSON)</div>
      <div class="card">
        <textarea id="feature-input" spellcheck="false" placeholder='{"lag_1h": 120.5, "roll_6h_mean": 98.2, ...}'></textarea>
        <div class="btn-row">
          <button class="btn-primary" id="btn-predict">▶ Predict</button>
          <button class="btn-ghost"   id="btn-fill">Fill Example</button>
          <button class="btn-ghost"   id="btn-clear">Clear</button>
        </div>
      </div>

      <!-- Known features -->
      <div style="margin-top:20px">
        <div class="section-label">Expected Features</div>
        <div class="card">
          <div class="pill-wrap" id="feat-pills">
            <span style="color:var(--muted);font-size:11px">Loading…</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Right: Results -->
    <div>
      <div class="section-label">Prediction Result</div>
      <div class="card" id="result-card">
        <div class="placeholder" id="placeholder">
          <div class="icon">📡</div>
          <div>Submit features to see predictions</div>
        </div>
        <div id="result-body" style="display:none">

          <!-- Risk banner -->
          <div class="risk-banner" id="risk-banner">
            <div class="risk-icon" id="risk-icon">—</div>
            <div class="risk-text">
              <div class="label" id="risk-label">—</div>
              <div class="sublabel">Ensemble Risk Level</div>
            </div>
            <div class="risk-prob" id="risk-prob">—</div>
          </div>

          <!-- Gauges -->
          <div class="result-block">
            <div class="gauge-row">
              <div class="gauge-label">XGBoost</div>
              <div class="gauge-track"><div class="gauge-fill" id="g-xgb" style="background:var(--accent)"></div></div>
              <div class="gauge-pct" id="p-xgb">—</div>
              <span class="flag-chip" id="f-xgb">—</span>
            </div>
            <div class="gauge-row">
              <div class="gauge-label">LSTM</div>
              <div class="gauge-track"><div class="gauge-fill" id="g-lstm" style="background:#a78bfa"></div></div>
              <div class="gauge-pct" id="p-lstm">—</div>
              <span class="flag-chip" id="f-lstm">—</span>
            </div>
            <div class="gauge-row">
              <div class="gauge-label">Ensemble</div>
              <div class="gauge-track"><div class="gauge-fill" id="g-ens" style="background:var(--green)"></div></div>
              <div class="gauge-pct" id="p-ens">—</div>
              <span class="flag-chip" id="f-ens">—</span>
            </div>
          </div>

          <!-- Latency -->
          <div style="font-size:11px;color:var(--dim);text-align:right" id="latency-line"></div>
        </div>
      </div>

      <!-- Log -->
      <div style="margin-top:20px">
        <div class="section-label">Request Log</div>
        <div class="log" id="log"></div>
      </div>
    </div>

  </div>
</div>

<script>
const API = '';   // same origin

// ── Helpers ─────────────────────────────────────────────────────────
function pct(v) { return (v * 100).toFixed(1) + '%'; }

function riskIcon(label) {
  return { Low: '✅', Moderate: '⚠️', High: '🔶', Critical: '🚨' }[label] ?? '❓';
}

function log(msg, cls = '') {
  const d   = document.getElementById('log');
  const ts  = new Date().toLocaleTimeString();
  const el  = document.createElement('div');
  el.className = 'entry';
  el.innerHTML = `<span class="ts">[${ts}]</span><span class="msg ${cls}">${msg}</span>`;
  d.prepend(el);
}

function setGauge(id, prob, fillId) {
  document.getElementById(id).textContent  = pct(prob);
  document.getElementById(fillId).style.width = pct(prob);
}

function setFlag(id, flag, color) {
  const el = document.getElementById(id);
  el.textContent    = flag ? 'FLOOD' : 'CLEAR';
  el.style.color    = flag ? color : 'var(--dim)';
  el.style.borderColor = flag ? color : 'var(--muted)';
}

// ── Health check ─────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    const j = await r.json();
    const ok = j.status === 'ok' && j.model_loaded;
    document.getElementById('dot').className   = 'dot' + (ok ? ' live' : '');
    document.getElementById('status-text').textContent = ok ? 'Models ready' : 'Models loading…';
  } catch { document.getElementById('status-text').textContent = 'Server unreachable'; }
}

// ── Load feature list ─────────────────────────────────────────────────
async function loadFeatures() {
  try {
    const r = await fetch(`${API}/features`);
    const j = await r.json();
    const wrap = document.getElementById('feat-pills');
    wrap.innerHTML = '';
    j.features.forEach(f => {
      const p = document.createElement('span');
      p.className = 'pill'; p.textContent = f;
      wrap.appendChild(p);
    });
    document.getElementById('st-features').textContent = j.count;
  } catch(e) { log('Could not load feature list: ' + e.message, 'err'); }
}

// ── Predict ──────────────────────────────────────────────────────────
async function predict() {
  const raw = document.getElementById('feature-input').value.trim();
  if (!raw) { log('No input provided', 'warn'); return; }

  let features;
  try { features = JSON.parse(raw); }
  catch(e) { log('Invalid JSON: ' + e.message, 'err'); return; }

  const btn = document.getElementById('btn-predict');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Running…';

  try {
    const resp = await fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail ?? resp.statusText);
    }
    const j = await resp.json();
    renderResult(j);
    log(`Flood prob ${pct(j.ensemble_prob)} — ${j.risk_label} risk`, j.ensemble_flag ? 'warn' : 'ok');

    // Populate thresholds first time
    if (document.getElementById('st-xgb-t').textContent === '—') {
      document.getElementById('st-xgb-t').textContent  = (j.xgb_threshold  * 100).toFixed(1) + '%';
      document.getElementById('st-lstm-t').textContent = (j.lstm_threshold * 100).toFixed(1) + '%';
    }
  } catch(e) {
    log('Predict error: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '▶ Predict';
  }
}

function renderResult(j) {
  document.getElementById('placeholder').style.display  = 'none';
  document.getElementById('result-body').style.display  = 'block';

  // Risk banner
  const banner = document.getElementById('risk-banner');
  banner.style.borderColor      = j.risk_color;
  banner.style.backgroundColor  = j.risk_color + '12';
  document.getElementById('risk-icon').textContent  = riskIcon(j.risk_label);
  document.getElementById('risk-label').textContent = j.risk_label;
  document.getElementById('risk-label').style.color = j.risk_color;
  document.getElementById('risk-prob').textContent  = pct(j.ensemble_prob);
  document.getElementById('risk-prob').style.color  = j.risk_color;

  // Gauges
  setGauge('p-xgb',  j.xgb_prob,      'g-xgb');
  setGauge('p-lstm', j.lstm_prob,     'g-lstm');
  setGauge('p-ens',  j.ensemble_prob, 'g-ens');

  setFlag('f-xgb',  j.xgb_flag,      'var(--accent)');
  setFlag('f-lstm', j.lstm_flag,      '#a78bfa');
  setFlag('f-ens',  j.ensemble_flag,  j.risk_color);

  // Colour the ensemble gauge to match risk
  document.getElementById('g-ens').style.background = j.risk_color;

  document.getElementById('latency-line').textContent = `⚡ ${j.latency_ms} ms`;
}

// ── Example data ──────────────────────────────────────────────────────
document.getElementById('btn-fill').addEventListener('click', async () => {
  try {
    const r = await fetch(`${API}/features`);
    const j = await r.json();
    const ex = {};
    j.features.forEach(f => { ex[f] = parseFloat((Math.random() * 200).toFixed(2)); });
    document.getElementById('feature-input').value = JSON.stringify(ex, null, 2);
  } catch {
    document.getElementById('feature-input').value = '{"lag_1h": 110.5, "roll_6h_mean": 95.0}';
  }
});

document.getElementById('btn-clear').addEventListener('click', () => {
  document.getElementById('feature-input').value = '';
  document.getElementById('placeholder').style.display = 'flex';
  document.getElementById('result-body').style.display  = 'none';
});

document.getElementById('btn-predict').addEventListener('click', predict);

document.getElementById('feature-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) predict();
});

// ── Init ──────────────────────────────────────────────────────────────
checkHealth();
loadFeatures();
setInterval(checkHealth, 15000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)