"""ASU (Air Separation Unit) Interactive Calculation Sheet.

Full double-column (Linde process) + argon column:
  Lower column  — HP (~6 bar): separates compressed air
  Upper column  — LP (~1.3 bar): produces pure N₂ and O₂
  Argon column  — refines crude Ar side draw from upper column
  MCHE          — thermally couples upper condenser to lower reboiler

Run:  python asu_calc_sheet.py
Open: http://localhost:5051
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template_string, request

from distcalc.asu.specs import ASUConfig
from distcalc.asu.system import solve_asu
from distcalc.components.loader import load_mixture

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASU Distillation Calculator</title>
<style>
  :root{--bg:#0f1117;--panel:#1a1d27;--border:#2d3148;--accent:#4f8ef7;
        --green:#3dcb7f;--red:#f05050;--orange:#f5a623;--text:#d4d8f0;
        --muted:#8890b0;--mono:'Fira Mono',Consolas,monospace}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,sans-serif;
       font-size:13px;min-height:100vh}
  header{background:var(--panel);border-bottom:1px solid var(--border);
         padding:12px 20px;display:flex;align-items:center;gap:12px}
  header h1{font-size:17px;font-weight:600;color:#fff}
  header span{font-size:11px;color:var(--muted);background:#23263a;
              padding:2px 8px;border-radius:4px}
  .layout{display:grid;grid-template-columns:320px 1fr;gap:0;height:calc(100vh - 49px)}
  .sidebar{background:var(--panel);border-right:1px solid var(--border);
           padding:14px;overflow-y:auto}
  .main{overflow-y:auto;padding:16px}
  .section{margin-bottom:16px}
  .section-title{font-size:10px;font-weight:700;text-transform:uppercase;
                 letter-spacing:.08em;color:var(--muted);margin-bottom:8px;
                 padding-bottom:4px;border-bottom:1px solid var(--border)}
  label{display:block;font-size:11px;color:var(--muted);margin-bottom:2px;margin-top:6px}
  input,select{width:100%;background:#23263a;border:1px solid var(--border);
               border-radius:4px;color:var(--text);padding:5px 8px;font-size:12px}
  input:focus,select:focus{outline:none;border-color:var(--accent)}
  .row2{display:grid;grid-template-columns:1fr 1fr;gap:6px}
  .row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
  button#run{width:100%;margin-top:12px;padding:9px;background:var(--accent);
             color:#fff;border:none;border-radius:6px;font-size:13px;
             font-weight:600;cursor:pointer;letter-spacing:.03em}
  button#run:hover{background:#3a77e8}
  button#run:disabled{background:#2d3148;color:var(--muted);cursor:not-allowed}
  .status-bar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
        padding:10px 14px;min-width:130px;flex:1}
  .card .label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
  .card .value{font-size:20px;font-weight:700;margin-top:2px;font-family:var(--mono)}
  .card.good .value{color:var(--green)}
  .card.warn .value{color:var(--orange)}
  .card.bad  .value{color:var(--red)}
  .tabs{display:flex;gap:2px;margin-bottom:12px}
  .tab{padding:6px 14px;border-radius:6px 6px 0 0;border:1px solid var(--border);
       border-bottom:none;cursor:pointer;font-size:12px;background:#23263a;color:var(--muted)}
  .tab.active{background:var(--panel);color:var(--text);font-weight:600}
  .tab-content{display:none}.tab-content.active{display:block}
  table{width:100%;border-collapse:collapse;font-size:11.5px;font-family:var(--mono)}
  th{background:#23263a;color:var(--muted);text-align:right;padding:5px 8px;
     font-weight:600;font-size:10px;text-transform:uppercase;position:sticky;top:0}
  th:first-child{text-align:left}
  td{padding:4px 8px;border-bottom:1px solid #1e2134;text-align:right}
  td:first-child{text-align:left;color:var(--muted)}
  tr:hover td{background:#1e2134}
  tr.feed-stage td{background:#1a2540;color:#8ab4f8}
  tr.ar-stage td{background:#1f2a1a;color:#80c980}
  .err{background:#2a1a1a;border:1px solid var(--red);border-radius:6px;
       padding:10px 14px;font-family:var(--mono);font-size:11px;color:var(--red)}
  .col-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
  .col-box{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px}
  .col-box h3{font-size:12px;color:var(--accent);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em}
  .kv-row{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid #1e2134;font-size:11.5px}
  .kv-row:last-child{border-bottom:none}
  .kv-key{color:var(--muted)}
  .kv-val{font-family:var(--mono);font-weight:600}
  .good{color:var(--green)}.warn{color:var(--orange)}.bad{color:var(--red)}
</style>
</head>
<body>
<header>
  <h1>ASU Distillation Calculator</h1>
  <span>PR-EOS · Double Column + Ar Side Draw</span>
</header>
<div class="layout">
<!-- ── SIDEBAR ─────────────────────────────────── -->
<aside class="sidebar">
  <form id="form">

  <div class="section">
    <div class="section-title">Air Feed</div>
    <label>Total Flow [mol/s]</label>
    <input name="air_flow" type="number" step="0.01" value="1.0" min="0.01">
    <div class="row3">
      <div><label>N₂ [-]</label><input name="z_n2" type="number" step="0.001" value="0.7812" min="0" max="1"></div>
      <div><label>O₂ [-]</label><input name="z_o2" type="number" step="0.001" value="0.2096" min="0" max="1"></div>
      <div><label>Ar [-]</label><input name="z_ar" type="number" step="0.001" value="0.0092" min="0" max="1"></div>
    </div>
    <label>Feed Temp [K]</label>
    <input name="T_air" type="number" step="1" value="100" min="70" max="200">
    <label>Feed Pressure [bar]</label>
    <input name="P_air_bar" type="number" step="0.1" value="6.0" min="2" max="15">
  </div>

  <div class="section">
    <div class="section-title">Lower Column (HP)</div>
    <div class="row2">
      <div><label>Stages</label><input name="N_lower" type="number" value="25" min="5" max="60"></div>
      <div><label>Feed Stage</label><input name="feed_stage_lower" type="number" value="13" min="2"></div>
    </div>
    <div class="row2">
      <div><label>Reflux Ratio</label><input name="RR_lower" type="number" step="0.1" value="2.5" min="0.1"></div>
      <div><label>D/F [-]</label><input name="D_frac_lower" type="number" step="0.01" value="0.50" min="0.05" max="0.95"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Upper Column (LP)</div>
    <label>LP Pressure [bar]</label>
    <input name="P_upper_bar" type="number" step="0.05" value="1.3" min="0.5" max="3">
    <div class="row2">
      <div><label>Stages</label><input name="N_upper" type="number" value="40" min="8" max="80"></div>
      <div><label>D/F [-]</label><input name="D_frac_upper" type="number" step="0.01" value="0.78" min="0.1" max="0.99"></div>
    </div>
    <div class="row2">
      <div><label>Reflux Ratio</label><input name="RR_upper" type="number" step="0.1" value="3.0" min="0.1"></div>
      <div><label>N₂ Feed Stage</label><input name="n2_feed_stage" type="number" value="2" min="1"></div>
    </div>
    <div class="row2">
      <div><label>O₂ Feed Stage</label><input name="co2_feed_stage" type="number" value="0" min="0">
        <small style="color:var(--muted);font-size:10px">0 = auto</small></div>
      <div><label>Ar Draw [mol/s]</label><input name="ar_draw_flow" type="number" step="0.001" value="0" min="0">
        <small style="color:var(--muted);font-size:10px">0 = auto</small></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Argon Column</div>
    <div class="row3">
      <div><label>Stages</label><input name="N_argon" type="number" value="30" min="5" max="60"></div>
      <div><label>RR</label><input name="RR_argon" type="number" step="0.1" value="3.5" min="0.1"></div>
      <div><label>D/F</label><input name="D_frac_argon" type="number" step="0.01" value="0.90" min="0.1" max="0.99"></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Solver</div>
    <div class="row2">
      <div><label>Outer Iterations</label><input name="max_outer_iter" type="number" value="12" min="1" max="30"></div>
      <div><label>MCHE Tol</label><input name="tol_duty" type="number" step="0.005" value="0.03" min="0.001"></div>
    </div>
  </div>

  <button id="run" type="button" onclick="runCalc()">▶ RUN ASU CALCULATION</button>
  </form>
</aside>

<!-- ── MAIN ──────────────────────────────────── -->
<main class="main" id="main">
  <div id="result" style="color:var(--muted);padding:40px 0;text-align:center">
    Configure and click RUN to start the ASU calculation.
  </div>
</main>
</div>

<script>
async function runCalc(){
  const btn=document.getElementById('run');
  btn.disabled=true; btn.textContent='⏳ Solving…';
  const fd=new FormData(document.getElementById('form'));
  const body={};
  fd.forEach((v,k)=>body[k]=v);
  try{
    const r=await fetch('/calculate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    renderResult(d);
  }catch(e){
    document.getElementById('result').innerHTML='<div class="err">'+e+'</div>';
  }finally{
    btn.disabled=false; btn.textContent='▶ RUN ASU CALCULATION';
  }
}

function fmt(v,dec=4){return v==null?'—':Number(v).toFixed(dec)}
function pct(v,dec=2){return v==null?'—':(Number(v)*100).toFixed(dec)+'%'}

function renderResult(d){
  if(d.error){
    document.getElementById('result').innerHTML='<div class="err">'+d.error+'</div>';
    return;
  }

  // Status cards
  const convClass=d.converged?'good':'warn';
  const mche_kw=(d.mche_duty/1e3).toFixed(1);
  const imb_kw=(d.duty_imbalance/1e3).toFixed(2);
  const imbClass=Math.abs(d.duty_imbalance/Math.max(Math.abs(d.mche_duty),1))<0.05?'good':'warn';

  let html=`<div class="status-bar">
    <div class="card ${convClass}"><div class="label">Converged</div><div class="value">${d.converged?'YES':'NO'}</div></div>
    <div class="card"><div class="label">Outer Iter</div><div class="value">${d.n_outer_iter}</div></div>
    <div class="card good"><div class="label">N₂ Purity</div><div class="value">${pct(d.N2_purity,2)}</div></div>
    <div class="card good"><div class="label">O₂ Purity</div><div class="value">${pct(d.O2_purity,2)}</div></div>
    <div class="card ${d.Ar_purity>0.5?'good':'warn'}"><div class="label">Ar Purity</div><div class="value">${pct(d.Ar_purity,2)}</div></div>
    <div class="card"><div class="label">N₂ Recovery</div><div class="value">${pct(d.N2_recovery,1)}</div></div>
    <div class="card"><div class="label">O₂ Recovery</div><div class="value">${pct(d.O2_recovery,1)}</div></div>
    <div class="card ${imbClass}"><div class="label">MCHE Imbalance</div><div class="value">${imb_kw} kW</div></div>
  </div>`;

  // Stream summary boxes
  html+=`<div class="col-grid">`;

  // Lower column
  if(d.lower_col){
    const lc=d.lower_col;
    html+=`<div class="col-box"><h3>Lower Column (HP ${(d.P_lower/1e5).toFixed(1)} bar)</h3>`;
    html+=kvrow('Converged', lc.converged?'Yes':'No', lc.converged?'good':'bad');
    html+=kvrow('Stages / Iterations', `${lc.N} / ${lc.n_iter}`);
    html+=kvrow('max|ΔT|', fmt(lc.max_dT,4)+' K');
    html+=kvrow('Q_cond', (lc.Q_condenser/1e3).toFixed(2)+' kW');
    html+=kvrow('Q_reb',  (lc.Q_reboiler/1e3).toFixed(2)+' kW');
    html+=`<div style="margin-top:8px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">Distillate (N₂-rich)</div>`;
    html+=streamRow(['N₂','O₂','Ar'],lc.x_distillate, lc.D_rate);
    html+=`<div style="margin-top:6px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">Bottoms (crude O₂)</div>`;
    html+=streamRow(['N₂','O₂','Ar'],lc.x_bottoms, lc.B_rate);
    html+=`</div>`;
  }

  // Upper column
  if(d.upper_col){
    const uc=d.upper_col;
    html+=`<div class="col-box"><h3>Upper Column (LP ${(d.P_upper/1e5).toFixed(2)} bar)</h3>`;
    html+=kvrow('Converged', uc.converged?'Yes':'No', uc.converged?'good':'bad');
    html+=kvrow('Stages / Iterations', `${uc.N} / ${uc.n_iter}`);
    html+=kvrow('max|ΔT|', fmt(uc.max_dT,4)+' K');
    html+=kvrow('Q_cond', (uc.Q_condenser/1e3).toFixed(2)+' kW');
    html+=kvrow('Q_reb',  (uc.Q_reboiler/1e3).toFixed(2)+' kW');
    html+=`<div style="margin-top:8px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">N₂ Distillate</div>`;
    html+=streamRow(['N₂','O₂','Ar'],uc.x_distillate, uc.D_rate);
    html+=`<div style="margin-top:6px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em">O₂ Bottoms</div>`;
    html+=streamRow(['N₂','O₂','Ar'],uc.x_bottoms, uc.B_rate);
    if(d.ar_draw_stage){
      html+=`<div style="margin-top:6px;font-size:10px;color:var(--green);text-transform:uppercase;letter-spacing:.06em">Ar Side Draw (stage ${d.ar_draw_stage})</div>`;
      html+=streamRow(['N₂','O₂','Ar'],d.ar_draw_z, d.ar_draw_flow);
    }
    html+=`</div>`;
  }

  html+=`</div>`;

  // Argon column (if present)
  if(d.argon_col){
    const ac=d.argon_col;
    html+=`<div class="col-box" style="margin-bottom:12px"><h3>Argon Column</h3>`;
    html+=`<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">`;
    html+=`<div>`;
    html+=kvrow('Converged', ac.converged?'Yes':'No', ac.converged?'good':'bad');
    html+=kvrow('Stages / Iterations', `${ac.N} / ${ac.n_iter}`);
    html+=kvrow('max|ΔT|', fmt(ac.max_dT,4)+' K');
    html+=kvrow('Q_cond', (ac.Q_condenser/1e3).toFixed(3)+' kW');
    html+=kvrow('Q_reb',  (ac.Q_reboiler/1e3).toFixed(3)+' kW');
    html+=`</div><div>`;
    html+=`<div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Ar Product (distillate)</div>`;
    html+=streamRow(['N₂','O₂','Ar'],ac.x_distillate, ac.D_rate);
    html+=`<div style="margin-top:6px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">O₂-rich bottoms</div>`;
    html+=streamRow(['N₂','O₂','Ar'],ac.x_bottoms, ac.B_rate);
    html+=`</div></div></div>`;
  }

  // MCHE box
  html+=`<div class="col-box" style="margin-bottom:12px"><h3>MCHE (Main Condenser / Reboiler)</h3>
    <div style="display:flex;gap:20px;flex-wrap:wrap">
    ${kvrow('Lower Q_cond', (d.lower_Q_cond/1e3).toFixed(2)+' kW')}
    ${kvrow('Upper Q_reb',  (d.upper_Q_reb/1e3).toFixed(2)+' kW')}
    ${kvrow('Imbalance',    (d.duty_imbalance/1e3).toFixed(2)+' kW', imbClass)}
    </div>
  </div>`;

  // Stage profiles
  html+=`<div class="tabs">
    <div class="tab active" onclick="switchTab(this,'lower-tbl')">Lower Column Stages</div>
    <div class="tab" onclick="switchTab(this,'upper-tbl')">Upper Column Stages</div>
    ${d.argon_col?'<div class="tab" onclick="switchTab(this,\'ar-tbl\')">Argon Column Stages</div>':''}
  </div>`;

  // Lower stage table
  html+=`<div id="lower-tbl" class="tab-content active">`;
  html+=buildStageTable(d.lower_col, d.lower_feed_stage, null);
  html+=`</div>`;

  // Upper stage table
  html+=`<div id="upper-tbl" class="tab-content">`;
  html+=buildStageTable(d.upper_col, null, d.ar_draw_stage, d.upper_n2_feed_stage, d.upper_co2_feed_stage);
  html+=`</div>`;

  if(d.argon_col){
    html+=`<div id="ar-tbl" class="tab-content">`;
    html+=buildStageTable(d.argon_col, null, null);
    html+=`</div>`;
  }

  document.getElementById('result').innerHTML=html;
}

function kvrow(k,v,cls=''){
  return `<div class="kv-row"><span class="kv-key">${k}</span><span class="kv-val ${cls}">${v}</span></div>`;
}

function streamRow(names, z, flow){
  if(!z||!z.length) return '';
  let s=`<div style="font-family:var(--mono);font-size:11.5px;margin-top:2px">`;
  names.slice(0,z.length).forEach((n,i)=>{
    s+=`<span style="margin-right:10px;color:var(--text)">${n}: <b>${(z[i]*100).toFixed(2)}%</b></span>`;
  });
  if(flow!=null) s+=`<span style="color:var(--muted)">flow: ${Number(flow).toFixed(4)} mol/s</span>`;
  s+=`</div>`;
  return s;
}

function buildStageTable(col, feedStage, arStage, n2Stage, co2Stage){
  if(!col) return '<p style="color:var(--muted)">No data</p>';
  const T=col.T, x=col.x, y=col.y, K=col.K, V=col.V, L=col.L;
  const n=T.length;
  const nc=x.length; // n_comp
  const compNames=['N₂','O₂','Ar'].slice(0,nc);

  let th='<th>Stage</th><th>T [K]</th><th>V [mol/s]</th><th>L [mol/s]</th>';
  compNames.forEach(c=>th+=`<th>x(${c})</th>`);
  compNames.forEach(c=>th+=`<th>y(${c})</th>`);
  compNames.forEach(c=>th+=`<th>K(${c})</th>`);

  let rows='';
  for(let j=0;j<n;j++){
    const stg=j+1;
    let cls='';
    if(stg===feedStage||stg===n2Stage||stg===co2Stage) cls='feed-stage';
    else if(stg===arStage) cls='ar-stage';
    let label=stg===1?'C':stg===n?'R':String(stg);
    if(stg===feedStage) label+='★';
    if(stg===n2Stage) label+='↓N₂';
    if(stg===co2Stage) label+='↓O₂';
    if(stg===arStage) label+='→Ar';

    let td=`<td>${label}</td><td>${T[j].toFixed(3)}</td><td>${V[j].toFixed(4)}</td><td>${L[j].toFixed(4)}</td>`;
    for(let i=0;i<nc;i++) td+=`<td>${x[i][j].toFixed(5)}</td>`;
    for(let i=0;i<nc;i++) td+=`<td>${y[i][j].toFixed(5)}</td>`;
    for(let i=0;i<nc;i++) td+=`<td>${K[i][j].toFixed(4)}</td>`;
    rows+=`<tr class="${cls}">${td}</tr>`;
  }

  return `<div style="overflow-x:auto"><table>
    <thead><tr>${th}</tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}

function switchTab(el,id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById(id).classList.add('active');
}
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_HTML)


@app.route("/calculate", methods=["POST"])
def calculate():
    try:
        data = request.get_json(force=True)

        def f(key, default=0.0):
            v = data.get(key, default)
            return float(v) if v not in ("", None) else default

        def i(key, default=0):
            v = data.get(key, default)
            return int(float(v)) if v not in ("", None) else default

        # Normalise air composition
        z = np.array([f("z_n2", 0.7812), f("z_o2", 0.2096), f("z_ar", 0.0092)])
        z = np.clip(z, 0.0, 1.0)
        if z.sum() < 1e-6:
            return jsonify({"error": "Air composition sums to zero"})
        z /= z.sum()

        # Include Ar only if non-negligible
        components = ["N2", "O2"]
        if z[2] > 1e-4:
            components.append("Ar")
            z_eff = z
        else:
            z_eff = z[:2] / z[:2].sum()

        mix = load_mixture(components)

        ar_draw = f("ar_draw_flow", 0.0)
        co2_stage = i("co2_feed_stage", 0)

        cfg = ASUConfig(
            air_flow=f("air_flow", 1.0),
            z_air=z_eff,
            T_air=f("T_air", 100.0),
            P_air=f("P_air_bar", 6.0) * 1e5,
            P_lower=f("P_air_bar", 6.0) * 1e5,
            N_lower=i("N_lower", 25),
            feed_stage_lower=i("feed_stage_lower", 13),
            RR_lower=f("RR_lower", 2.5),
            D_frac_lower=f("D_frac_lower", 0.50),
            P_upper=f("P_upper_bar", 1.3) * 1e5,
            N_upper=i("N_upper", 40),
            RR_upper=f("RR_upper", 3.0),
            D_frac_upper=f("D_frac_upper", 0.78),
            n2_feed_stage_upper=i("n2_feed_stage", 2),
            co2_feed_stage_upper=co2_stage,
            ar_draw_flow=ar_draw,
            N_argon=i("N_argon", 30),
            RR_argon=f("RR_argon", 3.5),
            D_frac_argon=f("D_frac_argon", 0.90),
            max_outer_iter=i("max_outer_iter", 12),
            tol_duty=f("tol_duty", 0.03),
        )

        res = solve_asu(cfg, mix)

        def col_to_dict(col, D_rate, B_rate):
            if col is None:
                return None
            nc = col.x.shape[0]
            N  = col.x.shape[1]
            return {
                "converged": col.converged,
                "n_iter": int(col.n_iter),
                "max_dT": float(col.max_dT),
                "N": N,
                "T": col.T.tolist(),
                "V": col.V.tolist(),
                "L": col.L.tolist(),
                "x": [col.x[i].tolist() for i in range(nc)],
                "y": [col.y[i].tolist() for i in range(nc)],
                "K": [col.K[i].tolist() for i in range(nc)],
                "H_L": col.H_L.tolist(),
                "H_V": col.H_V.tolist(),
                "Q_condenser": float(col.Q_condenser),
                "Q_reboiler":  float(col.Q_reboiler),
                "x_distillate": col.x_distillate.tolist(),
                "x_bottoms":    col.x_bottoms.tolist(),
                "D_rate": float(D_rate),
                "B_rate": float(B_rate),
            }

        D_lower = cfg.air_flow * cfg.D_frac_lower
        B_lower = cfg.air_flow - D_lower
        D_upper = float(res.streams.upper_distillate.flow) if res.streams.upper_distillate else 0.0
        B_upper = float(res.streams.upper_bottoms.flow)    if res.streams.upper_bottoms    else 0.0
        D_argon = float(res.streams.ar_distillate.flow)    if res.streams.ar_distillate    else 0.0
        B_argon = float(res.streams.ar_bottoms.flow)       if res.streams.ar_bottoms       else 0.0

        ar_draw_z = []
        ar_draw_flow_out = 0.0
        if res.streams.upper_ar_draw:
            ar_draw_z = res.streams.upper_ar_draw.z.tolist()
            ar_draw_flow_out = float(res.streams.upper_ar_draw.flow)

        out = {
            "converged":    res.converged,
            "n_outer_iter": res.n_outer_iter,
            "N2_purity":    float(res.N2_purity),
            "O2_purity":    float(res.O2_purity),
            "Ar_purity":    float(res.Ar_purity),
            "N2_recovery":  float(res.N2_recovery),
            "O2_recovery":  float(res.O2_recovery),
            "duty_imbalance": float(res.duty_imbalance),
            "mche_duty":   float(res.streams.Q_mche),
            "lower_Q_cond": float(res.lower_col.Q_condenser) if res.lower_col else 0.0,
            "upper_Q_reb":  float(res.upper_col.Q_reboiler)  if res.upper_col else 0.0,
            "P_lower":      float(cfg.P_lower),
            "P_upper":      float(cfg.P_upper),
            "lower_feed_stage": cfg.feed_stage_lower,
            "upper_n2_feed_stage": cfg.n2_feed_stage_upper,
            "upper_co2_feed_stage": cfg.co2_feed_stage_upper,
            "ar_draw_stage": (
                res.streams.upper_ar_draw is not None and
                int(res.upper_col.x.shape[1]) - 1  # rough; actual in solve_asu
            ) if False else None,   # placeholder; overwritten below
            "ar_draw_z":     ar_draw_z,
            "ar_draw_flow":  ar_draw_flow_out,
            "lower_col": col_to_dict(res.lower_col, D_lower, B_lower),
            "upper_col": col_to_dict(res.upper_col, D_upper, B_upper),
            "argon_col": col_to_dict(res.argon_col, D_argon, B_argon) if res.argon_col else None,
        }
        # Placeholder for Ar stage — would need to be tracked in solve_asu
        out["ar_draw_stage"] = None

        return jsonify(out)

    except Exception:
        return jsonify({"error": traceback.format_exc()})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5051))
    print(f"ASU Calculator: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
