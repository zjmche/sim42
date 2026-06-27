"""PR-EOS Distillation Calculation Sheet.

Run:  python calc_sheet.py
Then open  http://localhost:5050  in your browser.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

# Make sure distcalc is importable from this directory
sys.path.insert(0, str(Path(__file__).parent))

from distcalc.column.mesh_bp import solve_bp
from distcalc.column.shortcut_fug import shortcut_column
from distcalc.column.specs import ColumnConfig, FeedSpec
from distcalc.components.loader import load_mixture
from distcalc.equilibrium.bubble_dew import bubble_T

app = Flask(__name__)

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PR-EOS Distillation Calculator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f2f5;
         color: #1a1a2e; font-size: 13.5px; }

  /* ── Header ── */
  header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
            color: #e0e0ff; padding: 14px 28px;
            display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: .5px; }
  header .badge { background: #e94560; color: #fff; border-radius: 4px;
                  padding: 2px 8px; font-size: 11px; font-weight: 700; }

  /* ── Layout ── */
  .container { display: grid; grid-template-columns: 340px 1fr;
               gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }

  /* ── Panels ── */
  .panel { background: #fff; border-radius: 8px;
           box-shadow: 0 1px 4px rgba(0,0,0,.10); overflow: hidden; }
  .panel-header { background: #1a1a2e; color: #c8d6ea;
                  font-size: 11px; font-weight: 700; letter-spacing: 1px;
                  text-transform: uppercase; padding: 8px 14px; }
  .panel-body { padding: 14px; }

  /* ── Form ── */
  .section-title { font-size: 10.5px; font-weight: 700; color: #0f3460;
                   text-transform: uppercase; letter-spacing: .8px;
                   border-bottom: 2px solid #0f3460; padding-bottom: 4px;
                   margin: 14px 0 10px; }
  .section-title:first-child { margin-top: 0; }
  .field { display: flex; align-items: center; margin-bottom: 7px; gap: 6px; }
  .field label { flex: 0 0 170px; font-size: 12.5px; color: #444; }
  .field input, .field select {
    flex: 1; padding: 5px 8px; border: 1px solid #ccd; border-radius: 4px;
    font-size: 12.5px; background: #f8f9fc; transition: border-color .15s; }
  .field input:focus, .field select:focus {
    outline: none; border-color: #0f3460; background: #fff; }
  .field .unit { flex: 0 0 52px; font-size: 11px; color: #888; text-align: right; }

  /* ── Composition row ── */
  .comp-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;
              margin-bottom: 7px; }
  .comp-row .comp-input { display: flex; flex-direction: column; gap: 3px; }
  .comp-row .comp-input label { font-size: 11px; color: #666; text-align: center; }
  .comp-row .comp-input input { text-align: center; padding: 5px 4px;
    border: 1px solid #ccd; border-radius: 4px; font-size: 13px;
    background: #f8f9fc; width: 100%; }
  .comp-row .comp-input input:focus { outline: none; border-color: #0f3460; background:#fff; }

  /* ── Buttons ── */
  .btn-row { display: flex; gap: 8px; margin-top: 14px; }
  .btn { padding: 9px 20px; border: none; border-radius: 5px; cursor: pointer;
         font-size: 13px; font-weight: 600; letter-spacing: .3px; transition: all .15s; }
  .btn-primary { background: #0f3460; color: #fff; flex: 1; }
  .btn-primary:hover { background: #1a4a8a; }
  .btn-secondary { background: #e94560; color: #fff; }
  .btn-secondary:hover { background: #c73652; }
  .btn:disabled { opacity: .55; cursor: not-allowed; }

  /* ── Status bar ── */
  #status { margin-top: 10px; padding: 7px 10px; border-radius: 4px;
            font-size: 12px; min-height: 28px; }
  .status-ok  { background: #e6f7ef; color: #1a6e3c; border: 1px solid #a3d9bc; }
  .status-err { background: #fdecea; color: #9b1c1c; border: 1px solid #f5b7b1; }
  .status-run { background: #e8f0fe; color: #174ea6; border: 1px solid #aac4f5; }

  /* ── Results: summary cards ── */
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
           margin-bottom: 14px; }
  .card { border-radius: 6px; padding: 10px 14px; }
  .card-blue  { background: #e8f0fe; border-left: 4px solid #0f3460; }
  .card-green { background: #e6f7ef; border-left: 4px solid #1e8449; }
  .card-red   { background: #fef9e7; border-left: 4px solid #e67e22; }
  .card-title { font-size: 10px; font-weight: 700; text-transform: uppercase;
                letter-spacing: .7px; color: #555; margin-bottom: 5px; }
  .card-val   { font-size: 20px; font-weight: 700; color: #1a1a2e; }
  .card-sub   { font-size: 11px; color: #666; margin-top: 2px; }

  /* ── Composition result table ── */
  .comp-result { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
                 margin-bottom: 14px; }
  .comp-box { border: 1px solid #e0e6ef; border-radius: 6px; overflow: hidden; }
  .comp-box-hdr { background: #f0f4fb; padding: 6px 12px;
                  font-size: 10.5px; font-weight: 700; color: #0f3460;
                  text-transform: uppercase; letter-spacing: .5px; }
  .comp-box table { width: 100%; border-collapse: collapse; }
  .comp-box td { padding: 5px 12px; font-size: 12.5px; border-bottom: 1px solid #f0f3f8; }
  .comp-box td:last-child { text-align: right; font-weight: 600; font-family: monospace; }
  .comp-box tr:last-child td { border-bottom: none; }

  /* ── Stage table ── */
  #stage-section { margin-top: 6px; }
  .tbl-wrap { overflow-x: auto; border-radius: 6px; border: 1px solid #e0e6ef; }
  table.stage-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.stage-tbl thead tr { background: #1a1a2e; color: #c8d6ea; }
  table.stage-tbl th { padding: 7px 10px; font-weight: 600; text-align: right;
                       white-space: nowrap; font-size: 11px; letter-spacing: .3px; }
  table.stage-tbl th:first-child { text-align: center; }
  table.stage-tbl tbody tr:nth-child(even) { background: #f7f9fc; }
  table.stage-tbl tbody tr:hover { background: #eef2fb; }
  table.stage-tbl td { padding: 5px 10px; text-align: right; border-bottom: 1px solid #eef; }
  table.stage-tbl td:first-child { text-align: center; font-weight: 600; color: #0f3460; }
  .feed-row { background: #fff8e1 !important; }
  .feed-row td:first-child::after { content: " ★"; color: #e67e22; font-size: 10px; }

  /* ── Shortcut section ── */
  .shortcut-result { display: grid; grid-template-columns: repeat(4,1fr); gap:10px; }

  /* ── Tabs ── */
  .tabs { display: flex; border-bottom: 2px solid #e0e6ef; margin-bottom: 14px; }
  .tab { padding: 8px 18px; cursor: pointer; font-size: 12.5px; font-weight: 600;
         color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px;
         transition: all .15s; }
  .tab.active { color: #0f3460; border-bottom-color: #0f3460; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  .placeholder { text-align: center; color: #aaa; padding: 40px 20px;
                 font-size: 13px; }
  .spinner { display: inline-block; width:16px; height:16px;
             border:2px solid #aac4f5; border-top-color:#0f3460;
             border-radius:50%; animation: spin .7s linear infinite;
             vertical-align: middle; margin-right:6px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <div>
    <h1>PR-EOS Distillation Calculator</h1>
    <div style="font-size:11px; color:#8899bb; margin-top:2px;">
      N₂ / O₂ / Ar · Peng-Robinson EOS · Wang-Henke BP MESH
    </div>
  </div>
  <span class="badge">v1.0</span>
</header>

<div class="container">

  <!-- ══════════════════ LEFT PANEL – INPUTS ══════════════════ -->
  <div class="panel">
    <div class="panel-header">Input Specification</div>
    <div class="panel-body">

      <!-- Feed -->
      <div class="section-title">Feed</div>
      <div style="font-size:11px; color:#888; margin-bottom:6px;">Mole fractions (auto-normalised)</div>
      <div class="comp-row">
        <div class="comp-input">
          <label>N₂</label>
          <input type="number" id="z_n2" value="0.78" min="0" max="1" step="0.01">
        </div>
        <div class="comp-input">
          <label>O₂</label>
          <input type="number" id="z_o2" value="0.21" min="0" max="1" step="0.01">
        </div>
        <div class="comp-input">
          <label>Ar</label>
          <input type="number" id="z_ar" value="0.01" min="0" max="1" step="0.01">
        </div>
      </div>

      <div class="field">
        <label>Feed flow</label>
        <input type="number" id="feed_flow" value="1.0" min="0.001" step="0.1">
        <span class="unit">mol/s</span>
      </div>
      <div class="field">
        <label>Feed temperature</label>
        <input type="number" id="feed_T" value="90.0" step="0.5">
        <span class="unit">K</span>
      </div>
      <div class="field">
        <label>Feed pressure</label>
        <input type="number" id="feed_P" value="101325" step="1000">
        <span class="unit">Pa</span>
      </div>
      <div class="field">
        <label>Feed quality <i>q</i></label>
        <select id="feed_q">
          <option value="1.0" selected>1.0 – Saturated liquid</option>
          <option value="0.5">0.5 – Partial vapour</option>
          <option value="0.0">0.0 – Saturated vapour</option>
        </select>
        <span class="unit"></span>
      </div>

      <!-- Column -->
      <div class="section-title">Column Configuration</div>
      <div class="field">
        <label>Number of stages <i>N</i></label>
        <input type="number" id="N_stages" value="15" min="3" max="100" step="1">
        <span class="unit">—</span>
      </div>
      <div class="field">
        <label>Feed stage (1-based)</label>
        <input type="number" id="feed_stage" value="8" min="2" step="1">
        <span class="unit">—</span>
      </div>
      <div class="field">
        <label>Column pressure</label>
        <input type="number" id="col_P" value="101325" step="1000">
        <span class="unit">Pa</span>
      </div>
      <div class="field">
        <label>Condenser type</label>
        <select id="condenser_type">
          <option value="total" selected>Total</option>
          <option value="partial">Partial</option>
        </select>
        <span class="unit"></span>
      </div>

      <!-- Operating -->
      <div class="section-title">Operating Parameters</div>
      <div class="field">
        <label>Distillate rate <i>D</i></label>
        <input type="number" id="D_rate" value="0.75" min="0.001" step="0.01">
        <span class="unit">mol/s</span>
      </div>
      <div class="field">
        <label>Reflux ratio <i>L/D</i></label>
        <input type="number" id="RR" value="5.0" min="0.1" step="0.5">
        <span class="unit">—</span>
      </div>

      <!-- Solver -->
      <div class="section-title">Solver Options</div>
      <div class="field">
        <label>Max iterations</label>
        <input type="number" id="max_iter" value="200" min="50" max="500" step="50">
        <span class="unit">—</span>
      </div>
      <div class="field">
        <label>Tolerance |ΔT|</label>
        <input type="number" id="tol_T" value="1e-4" step="1e-5">
        <span class="unit">K</span>
      </div>

      <div class="btn-row">
        <button class="btn btn-primary" id="btn-calc" onclick="calculate()">
          ▶  Calculate
        </button>
        <button class="btn btn-secondary" onclick="resetForm()">Reset</button>
      </div>
      <div id="status"></div>

    </div><!-- panel-body -->
  </div><!-- left panel -->

  <!-- ══════════════════ RIGHT PANEL – RESULTS ══════════════════ -->
  <div class="panel">
    <div class="panel-header">Results</div>
    <div class="panel-body">

      <div class="tabs">
        <div class="tab active" onclick="switchTab('rigorous')">Rigorous (MESH BP)</div>
        <div class="tab" onclick="switchTab('shortcut')">Shortcut (FUG)</div>
      </div>

      <!-- ── Rigorous tab ── -->
      <div class="tab-panel active" id="tab-rigorous">
        <div id="rigorous-placeholder" class="placeholder">
          Fill in the inputs and click <b>Calculate</b> to run the Wang-Henke solver.
        </div>
        <div id="rigorous-results" style="display:none">

          <!-- Summary cards -->
          <div class="cards">
            <div class="card card-blue">
              <div class="card-title">Converged</div>
              <div class="card-val" id="r-converged">—</div>
              <div class="card-sub" id="r-iters">—</div>
            </div>
            <div class="card card-green">
              <div class="card-title">Q condenser</div>
              <div class="card-val" id="r-qcond">—</div>
              <div class="card-sub">kW  (heat removed)</div>
            </div>
            <div class="card card-red">
              <div class="card-title">Q reboiler</div>
              <div class="card-val" id="r-qreb">—</div>
              <div class="card-sub">kW  (heat added)</div>
            </div>
          </div>

          <!-- Distillate / Bottoms compositions -->
          <div class="comp-result">
            <div class="comp-box">
              <div class="comp-box-hdr">Distillate  (x<sub>D</sub>)</div>
              <table>
                <tr><td>N₂</td><td id="d-n2">—</td></tr>
                <tr><td>O₂</td><td id="d-o2">—</td></tr>
                <tr><td>Ar</td><td id="d-ar">—</td></tr>
                <tr><td style="color:#888;font-size:11px">Flow D</td>
                    <td id="d-flow" style="color:#888;font-size:11px">—</td></tr>
              </table>
            </div>
            <div class="comp-box">
              <div class="comp-box-hdr">Bottoms  (x<sub>B</sub>)</div>
              <table>
                <tr><td>N₂</td><td id="b-n2">—</td></tr>
                <tr><td>O₂</td><td id="b-o2">—</td></tr>
                <tr><td>Ar</td><td id="b-ar">—</td></tr>
                <tr><td style="color:#888;font-size:11px">Flow B</td>
                    <td id="b-flow" style="color:#888;font-size:11px">—</td></tr>
              </table>
            </div>
          </div>

          <!-- Stage-by-stage table -->
          <div id="stage-section">
            <div class="section-title" style="margin-top:4px">Stage Profile</div>
            <div class="tbl-wrap">
              <table class="stage-tbl">
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th>T [K]</th>
                    <th>V [mol/s]</th>
                    <th>L [mol/s]</th>
                    <th>x<sub>N2</sub></th>
                    <th>x<sub>O2</sub></th>
                    <th>x<sub>Ar</sub></th>
                    <th>y<sub>N2</sub></th>
                    <th>y<sub>O2</sub></th>
                    <th>y<sub>Ar</sub></th>
                    <th>K<sub>N2</sub></th>
                    <th>K<sub>O2</sub></th>
                    <th>K<sub>Ar</sub></th>
                  </tr>
                </thead>
                <tbody id="stage-tbody"></tbody>
              </table>
            </div>
          </div>

        </div><!-- rigorous-results -->
      </div><!-- tab-rigorous -->

      <!-- ── Shortcut tab ── -->
      <div class="tab-panel" id="tab-shortcut">
        <div id="shortcut-placeholder" class="placeholder">
          Fill in product specifications below and click <b>Calculate</b>.
        </div>
        <div id="shortcut-results" style="display:none">
          <div class="shortcut-result">
            <div class="card card-blue">
              <div class="card-title">N<sub>min</sub></div>
              <div class="card-val" id="s-nmin">—</div>
              <div class="card-sub">Fenske</div>
            </div>
            <div class="card card-green">
              <div class="card-title">R<sub>min</sub></div>
              <div class="card-val" id="s-rmin">—</div>
              <div class="card-sub">Underwood</div>
            </div>
            <div class="card card-red">
              <div class="card-title">N<sub>actual</sub></div>
              <div class="card-val" id="s-nact">—</div>
              <div class="card-sub">Gilliland @ 1.5×R<sub>min</sub></div>
            </div>
            <div class="card card-blue">
              <div class="card-title">R<sub>actual</sub></div>
              <div class="card-val" id="s-ract">—</div>
              <div class="card-sub">1.5 × R<sub>min</sub></div>
            </div>
          </div>
          <div class="comp-box" style="margin-top:14px">
            <div class="comp-box-hdr">Relative Volatilities α (to heavy key)</div>
            <table>
              <tr><td>N₂</td><td id="s-aN2">—</td></tr>
              <tr><td>O₂</td><td id="s-aO2">—</td></tr>
              <tr><td>Ar</td><td id="s-aAr">—</td></tr>
            </table>
          </div>
        </div>

        <!-- Shortcut product specs -->
        <div class="section-title" style="margin-top:18px">Product Specification</div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:8px">
          <div>
            <div style="font-size:11px;color:#555;margin-bottom:6px;font-weight:600">
              Distillate x<sub>D</sub></div>
            <div class="comp-row">
              <div class="comp-input"><label>N₂</label>
                <input type="number" id="xd_n2" value="0.995" min="0" max="1" step="0.001"></div>
              <div class="comp-input"><label>O₂</label>
                <input type="number" id="xd_o2" value="0.004" min="0" max="1" step="0.001"></div>
              <div class="comp-input"><label>Ar</label>
                <input type="number" id="xd_ar" value="0.001" min="0" max="1" step="0.001"></div>
            </div>
          </div>
          <div>
            <div style="font-size:11px;color:#555;margin-bottom:6px;font-weight:600">
              Bottoms x<sub>B</sub></div>
            <div class="comp-row">
              <div class="comp-input"><label>N₂</label>
                <input type="number" id="xb_n2" value="0.001" min="0" max="1" step="0.001"></div>
              <div class="comp-input"><label>O₂</label>
                <input type="number" id="xb_o2" value="0.95" min="0" max="1" step="0.001"></div>
              <div class="comp-input"><label>Ar</label>
                <input type="number" id="xb_ar" value="0.049" min="0" max="1" step="0.001"></div>
            </div>
          </div>
        </div>
      </div><!-- tab-shortcut -->

    </div><!-- panel-body -->
  </div><!-- right panel -->

</div><!-- container -->

<script>
let activeTab = 'rigorous';

function switchTab(name) {
  activeTab = name;
  document.querySelectorAll('.tab').forEach((t,i) => {
    t.classList.toggle('active', ['rigorous','shortcut'][i] === name);
  });
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
}

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status-' + cls;
}

function fmt(v, dec=4) {
  if (v === null || v === undefined || isNaN(v)) return '—';
  if (!isFinite(v)) return '∞';
  return Number(v).toFixed(dec);
}

function fmtFlow(v) { return fmt(v, 3) + ' mol/s'; }
function fmtKW(v)   {
  if (v === null || v === undefined) return '—';
  return (v/1000).toFixed(2);
}

async function calculate() {
  const btn = document.getElementById('btn-calc');
  btn.disabled = true;
  setStatus('<span class="spinner"></span> Running…', 'run');

  const payload = {
    feed: {
      z_n2: +document.getElementById('z_n2').value,
      z_o2: +document.getElementById('z_o2').value,
      z_ar: +document.getElementById('z_ar').value,
      flow: +document.getElementById('feed_flow').value,
      T:    +document.getElementById('feed_T').value,
      P:    +document.getElementById('feed_P').value,
      q:    +document.getElementById('feed_q').value,
      stage:+document.getElementById('feed_stage').value,
    },
    column: {
      N_stages:       +document.getElementById('N_stages').value,
      P:              +document.getElementById('col_P').value,
      condenser_type:  document.getElementById('condenser_type').value,
      D_rate:         +document.getElementById('D_rate').value,
      RR:             +document.getElementById('RR').value,
    },
    solver: {
      max_iter: +document.getElementById('max_iter').value,
      tol_T:    +document.getElementById('tol_T').value,
    },
    shortcut: {
      xd: [+document.getElementById('xd_n2').value,
           +document.getElementById('xd_o2').value,
           +document.getElementById('xd_ar').value],
      xb: [+document.getElementById('xb_n2').value,
           +document.getElementById('xb_o2').value,
           +document.getElementById('xb_ar').value],
    },
  };

  try {
    const resp = await fetch('/calculate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    renderResults(data);
    setStatus(`✔  Done — ${data.rigorous.n_iter} iterations, max|ΔT| = ${Number(data.rigorous.max_dT).toExponential(2)} K`, 'ok');
  } catch(e) {
    setStatus('✖  ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

function renderResults(data) {
  const r = data.rigorous;
  const N = r.N_stages;
  const feedStage = data.feed_stage_0based;   // 0-based index

  // Cards
  document.getElementById('r-converged').textContent = r.converged ? '✔ Yes' : '✖ No';
  document.getElementById('r-iters').textContent = r.n_iter + ' iterations, max|ΔT| ' + Number(r.max_dT).toExponential(1) + ' K';
  document.getElementById('r-qcond').textContent = fmtKW(r.Q_condenser);
  document.getElementById('r-qreb').textContent  = fmtKW(r.Q_reboiler);

  // Compositions
  const xD = r.x_distillate, xB = r.x_bottoms;
  document.getElementById('d-n2').textContent = fmt(xD[0]);
  document.getElementById('d-o2').textContent = fmt(xD[1]);
  document.getElementById('d-ar').textContent = fmt(xD[2] ?? 0);
  document.getElementById('d-flow').textContent = fmtFlow(data.D_rate);
  document.getElementById('b-n2').textContent = fmt(xB[0]);
  document.getElementById('b-o2').textContent = fmt(xB[1]);
  document.getElementById('b-ar').textContent = fmt(xB[2] ?? 0);
  document.getElementById('b-flow').textContent = fmtFlow(data.B_rate);

  // Stage table
  const tbody = document.getElementById('stage-tbody');
  tbody.innerHTML = '';
  const nComp = r.x[0].length > 0 ? r.x.length : 2;  // number of components
  for (let j = 0; j < N; j++) {
    const tr = document.createElement('tr');
    if (j === feedStage) tr.className = 'feed-row';
    const label = j === 0 ? 'C (0)' : j === N-1 ? 'R ('+(N-1)+')' : j;
    const xN2 = r.x[0][j], xO2 = r.x[1][j], xAr = nComp > 2 ? r.x[2][j] : 0;
    const yN2 = r.y[0][j], yO2 = r.y[1][j], yAr = nComp > 2 ? r.y[2][j] : 0;
    const KN2 = r.K[0][j], KO2 = r.K[1][j], KAr = nComp > 2 ? r.K[2][j] : 0;
    tr.innerHTML = `
      <td>${label}</td>
      <td>${fmt(r.T[j],2)}</td>
      <td>${fmt(r.V[j],3)}</td>
      <td>${fmt(r.L[j],3)}</td>
      <td>${fmt(xN2)}</td><td>${fmt(xO2)}</td><td>${fmt(xAr)}</td>
      <td>${fmt(yN2)}</td><td>${fmt(yO2)}</td><td>${fmt(yAr)}</td>
      <td>${fmt(KN2,3)}</td><td>${fmt(KO2,3)}</td><td>${fmt(KAr,3)}</td>`;
    tbody.appendChild(tr);
  }

  // Shortcut
  const s = data.shortcut;
  if (s && !s.error) {
    document.getElementById('s-nmin').textContent = fmt(s.N_min, 1);
    document.getElementById('s-rmin').textContent = fmt(s.R_min, 3);
    document.getElementById('s-nact').textContent = isFinite(s.N_actual) ? fmt(s.N_actual,1) : '∞';
    document.getElementById('s-ract').textContent = fmt(s.R_actual, 3);
    document.getElementById('s-aN2').textContent = fmt(s.alpha[0], 3);
    document.getElementById('s-aO2').textContent = fmt(s.alpha[1], 3);
    document.getElementById('s-aAr').textContent = s.alpha.length > 2 ? fmt(s.alpha[2], 3) : '—';
    document.getElementById('shortcut-placeholder').style.display = 'none';
    document.getElementById('shortcut-results').style.display = '';
  }

  document.getElementById('rigorous-placeholder').style.display = 'none';
  document.getElementById('rigorous-results').style.display = '';
}

function resetForm() {
  document.getElementById('z_n2').value = '0.78';
  document.getElementById('z_o2').value = '0.21';
  document.getElementById('z_ar').value = '0.01';
  document.getElementById('feed_flow').value = '1.0';
  document.getElementById('feed_T').value = '90.0';
  document.getElementById('feed_P').value = '101325';
  document.getElementById('feed_q').value = '1.0';
  document.getElementById('feed_stage').value = '8';
  document.getElementById('N_stages').value = '15';
  document.getElementById('col_P').value = '101325';
  document.getElementById('D_rate').value = '0.75';
  document.getElementById('RR').value = '5.0';
  document.getElementById('max_iter').value = '200';
  document.getElementById('tol_T').value = '1e-4';
  document.getElementById('rigorous-placeholder').style.display = '';
  document.getElementById('rigorous-results').style.display = 'none';
  document.getElementById('shortcut-placeholder').style.display = '';
  document.getElementById('shortcut-results').style.display = 'none';
  document.getElementById('status').textContent = '';
  document.getElementById('status').className = '';
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

def _mix_from_z(z: list[float]):
    """Load N2/O2 or N2/O2/Ar mixture depending on Ar fraction."""
    ar = z[2] if len(z) > 2 else 0.0
    if ar > 1e-9:
        return load_mixture(["N2", "O2", "Ar"]), np.array([z[0], z[1], z[2]])
    else:
        return load_mixture(["N2", "O2"]), np.array([z[0], z[1]])


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        body = request.get_json(force=True)
        feed_in = body["feed"]
        col_in  = body["column"]
        sol_in  = body["solver"]
        sc_in   = body.get("shortcut", {})

        # ── Normalise feed composition ──────────────────────────────────────
        z_raw = [
            float(feed_in["z_n2"]),
            float(feed_in["z_o2"]),
            float(feed_in.get("z_ar", 0.0)),
        ]
        z_sum = sum(z_raw)
        if z_sum < 1e-9:
            return jsonify({"error": "Feed composition sums to zero."})
        z_raw = [v / z_sum for v in z_raw]

        mix, z = _mix_from_z(z_raw)
        n_comp = mix.n

        F        = float(feed_in["flow"])
        T_feed   = float(feed_in["T"])
        P_feed   = float(feed_in["P"])
        q_feed   = float(feed_in["q"])
        fs_1     = int(feed_in["stage"])   # 1-based

        N        = int(col_in["N_stages"])
        col_P    = float(col_in["P"])
        D_rate   = float(col_in["D_rate"])
        RR       = float(col_in["RR"])
        cond_t   = str(col_in.get("condenser_type", "total"))

        max_iter = int(sol_in.get("max_iter", 200))
        tol_T    = float(sol_in.get("tol_T", 1e-4))

        # ── Clamp feed stage ────────────────────────────────────────────────
        fs_1 = max(2, min(N - 1, fs_1))

        # ── Column config ───────────────────────────────────────────────────
        feed = FeedSpec(
            stage=fs_1, flow=F, z=z,
            T=T_feed, P=P_feed, q=q_feed,
        )
        P_profile = np.full(N, col_P)
        cfg = ColumnConfig(
            N_stages=N, feed=feed, P_profile=P_profile,
            condenser_type=cond_t,
            distillate_rate=D_rate, reflux_ratio=RR,
        )

        # ── Solve ────────────────────────────────────────────────────────────
        res = solve_bp(cfg, mix, tol_T=tol_T, max_iter=max_iter)

        # Pad K/x/y to 3 components for uniform JS handling
        def pad3(arr):
            """arr shape (n_comp, N) → lists of 3 lists each length N."""
            out = []
            for i in range(3):
                if i < n_comp:
                    out.append(arr[i, :].tolist())
                else:
                    out.append([0.0] * N)
            return out

        x_dist = res.x_distillate.tolist()
        x_bot  = res.x_bottoms.tolist()
        while len(x_dist) < 3: x_dist.append(0.0)
        while len(x_bot)  < 3: x_bot.append(0.0)

        rigorous_out = {
            "N_stages":    N,
            "converged":   bool(res.converged),
            "n_iter":      int(res.n_iter),
            "max_dT":      float(res.max_dT),
            "Q_condenser": float(res.Q_condenser),
            "Q_reboiler":  float(res.Q_reboiler),
            "T":  res.T.tolist(),
            "V":  res.V.tolist(),
            "L":  res.L.tolist(),
            "x":  pad3(res.x),
            "y":  pad3(res.y),
            "K":  pad3(res.K),
            "x_distillate": x_dist,
            "x_bottoms":    x_bot,
        }

        # ── Shortcut ─────────────────────────────────────────────────────────
        shortcut_out = {}
        try:
            xd_raw = sc_in.get("xd", [0.99, 0.005, 0.005])
            xb_raw = sc_in.get("xb", [0.005, 0.99, 0.005])

            # Trim to n_comp
            xd = np.array(xd_raw[:n_comp], dtype=float)
            xb = np.array(xb_raw[:n_comp], dtype=float)
            xd /= xd.sum(); xb /= xb.sum()

            sc = shortcut_column(z, xd, xb, P=col_P, q=q_feed, mix=mix)
            alpha_list = sc.alpha.tolist()
            while len(alpha_list) < 3: alpha_list.append(0.0)
            shortcut_out = {
                "N_min":    float(sc.N_min),
                "R_min":    float(sc.R_min),
                "N_actual": float(sc.N_actual),
                "R_actual": float(sc.R_actual),
                "alpha":    alpha_list,
            }
        except Exception as e:
            shortcut_out = {"error": str(e)}

        return jsonify({
            "rigorous": rigorous_out,
            "shortcut": shortcut_out,
            "D_rate": D_rate,
            "B_rate": float(cfg.B),
            "feed_stage_0based": fs_1 - 1,
        })

    except Exception as e:
        return jsonify({"error": traceback.format_exc(limit=4)})


if __name__ == "__main__":
    print()
    print("  ┌─────────────────────────────────────────────┐")
    print("  │  PR-EOS Distillation Calculation Sheet      │")
    print("  │  Open  http://localhost:5050  in browser    │")
    print("  └─────────────────────────────────────────────┘")
    print()
    app.run(host="0.0.0.0", port=5050, debug=False)
