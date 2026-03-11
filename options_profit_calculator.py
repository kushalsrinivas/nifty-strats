#!/usr/bin/env python3
"""
NIFTY Options Profit Calculator
================================
Self-contained HTML page inspired by optionsprofitcalculator.com, built
from live NSE option-chain data already stored locally.

Usage
-----
    python options_profit_calculator.py
    python options_profit_calculator.py --csv output/NIFTY_*.csv
    python options_profit_calculator.py --expiry 17-03-2026 --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIFTY Options Profit Calculator")
    p.add_argument("--csv", default=None, help="Option-chain CSV path.")
    p.add_argument("--expiry", default=None, help="Expiry filter, e.g. '17-03-2026'.")
    p.add_argument("--strikes", type=int, default=50, help="Strikes to include around ATM.")
    p.add_argument("--output-dir", default="output", help="Output directory.")
    p.add_argument("--open", action="store_true", help="Open in browser.")
    return p.parse_args()


def find_latest_csv(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("*_option_chain_*.csv"))
    if not files:
        raise FileNotFoundError(f"No option-chain CSV in {output_dir}")
    return files[-1]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    num = [
        "strike_price", "last_price", "change", "pchange",
        "open_interest", "change_in_open_interest", "volume",
        "implied_volatility", "underlying_value",
        "best_bid_price", "best_ask_price",
    ]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["expiry_dt"] = pd.to_datetime(df["contract_expiry"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["expiry_dt"])


def choose_expiry(df: pd.DataFrame, expiry_filter: str | None) -> pd.Timestamp:
    today = pd.Timestamp(datetime.now().date())
    future = sorted(df[df["expiry_dt"] >= today]["expiry_dt"].unique().tolist())
    if expiry_filter:
        dt = pd.to_datetime(expiry_filter, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Cannot parse expiry: {expiry_filter}")
        return dt
    return future[0] if future else sorted(df["expiry_dt"].unique().tolist())[0]


def build_chain_json(df: pd.DataFrame, expiry: pd.Timestamp, spot: float, n_strikes: int) -> str:
    sub = df[df["expiry_dt"] == expiry]
    strikes_data = {}
    for _, row in sub.iterrows():
        s = float(row["strike_price"])
        otype = row["option_type"]
        key = int(s)
        if key not in strikes_data:
            strikes_data[key] = {"strike": s}
        prefix = "ce" if otype == "CE" else "pe"
        strikes_data[key][f"{prefix}_ltp"] = round(float(row["last_price"]), 2)
        strikes_data[key][f"{prefix}_iv"] = round(float(row["implied_volatility"]), 2)
        strikes_data[key][f"{prefix}_bid"] = round(float(row["best_bid_price"]), 2)
        strikes_data[key][f"{prefix}_ask"] = round(float(row["best_ask_price"]), 2)

    all_strikes = sorted(strikes_data.keys())
    if spot > 0 and all_strikes:
        atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
        half = n_strikes // 2
        lo = max(0, atm_idx - half)
        hi = min(len(all_strikes), atm_idx + half)
        all_strikes = all_strikes[lo:hi]

    result = [strikes_data[k] for k in all_strikes]
    return json.dumps(result, separators=(",", ":"))


def get_vix(output_dir: Path) -> float | None:
    vix_files = sorted(output_dir.glob("INDIAVIX*.csv"))
    if not vix_files:
        return None
    try:
        vdf = pd.read_csv(vix_files[-1])
        vdf.columns = [c.strip() for c in vdf.columns]
        vcol = next((c for c in vdf.columns if "close" in c.lower()), None)
        if vcol:
            return float(pd.to_numeric(vdf[vcol], errors="coerce").dropna().iloc[-1])
    except Exception:
        pass
    return None


def render_html(
    chain_json: str,
    spot: float,
    expiry: pd.Timestamp,
    all_expiries: list[str],
    snapshot_ts: str,
    vix: float | None,
    nifty_lot_size: int = 75,
) -> str:
    expiry_str = str(expiry.date())
    today_str = str(datetime.now().date())
    vix_val = vix if vix else 15.0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NIFTY Options Profit Calculator</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {{
  --bg:#08090d; --surface:#0f1118; --surface2:#161922; --surface3:#1d2030;
  --border:#252838; --border2:#2e3148; --text:#c9cdd8; --bright:#eef0f6;
  --muted:#505570; --accent:#6366f1; --accent2:#818cf8;
  --green:#10b981; --green-dim:#065f46; --red:#ef4444; --red-dim:#7f1d1d;
  --amber:#f59e0b; --cyan:#06b6d4; --r:8px;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:13px;min-height:100vh}}
.app{{display:grid;grid-template-columns:360px 1fr;grid-template-rows:auto 1fr;height:100vh}}

.header{{grid-column:1/-1;display:flex;align-items:center;gap:16px;padding:10px 24px;background:var(--surface);border-bottom:1px solid var(--border)}}
.logo{{font-size:15px;font-weight:800;color:var(--bright)}}
.logo span{{color:var(--accent)}}
.hbadge{{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:3px 14px;font-size:14px;font-weight:700;color:var(--bright);font-family:'JetBrains Mono',monospace}}
.htag{{font-size:11px;font-weight:600;padding:3px 10px;border-radius:4px;letter-spacing:.04em}}
.htag-g{{background:#052e16;color:var(--green);border:1px solid #14532d}}
.htag-a{{background:#451a03;color:var(--amber);border:1px solid #78350f}}
.htag-i{{background:#1e1b4b;color:var(--accent2);border:1px solid #312e81}}
.header-r{{margin-left:auto;color:var(--muted);font-size:11px}}

.sidebar{{background:var(--surface);border-right:1px solid var(--border);overflow-y:auto}}
.sec{{padding:14px 18px;border-bottom:1px solid var(--border)}}
.sec-t{{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin-bottom:10px}}

.presets{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.pbtn{{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);color:var(--text);font-size:11px;font-weight:500;padding:7px 9px;cursor:pointer;transition:all .15s;font-family:inherit;text-align:left}}
.pbtn:hover{{background:var(--surface3);border-color:var(--accent);color:var(--bright)}}
.pbtn.on{{background:#1e1b4b;border-color:var(--accent);color:var(--accent2)}}

select,input[type="number"]{{background:var(--surface2);border:1px solid var(--border);border-radius:6px;color:var(--bright);padding:6px 9px;font-size:12px;font-family:'JetBrains Mono',monospace;outline:none;width:100%;transition:border-color .15s}}
select:focus,input:focus{{border-color:var(--accent)}}
label{{display:block;font-size:11px;color:var(--muted);margin-bottom:3px;font-weight:500}}

.fr{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px}}
.fr3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px}}
.fg{{display:flex;flex-direction:column}}

.lc{{background:var(--surface2);border:1px solid var(--border);border-radius:var(--r);padding:10px;margin-bottom:6px;position:relative}}
.lc.buy{{border-left:3px solid var(--green)}}
.lc.sell{{border-left:3px solid var(--red)}}
.lh{{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}}
.ll{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.ll.b{{color:var(--green)}}
.ll.s{{color:var(--red)}}
.rx{{background:none;border:none;color:var(--muted);cursor:pointer;font-size:15px;padding:2px 5px;border-radius:4px}}
.rx:hover{{color:var(--red);background:var(--red-dim)}}

.abtn{{width:100%;background:var(--surface2);border:1px dashed var(--border2);border-radius:var(--r);color:var(--muted);font-size:12px;font-weight:500;padding:9px;cursor:pointer;transition:all .15s;font-family:inherit;margin-bottom:6px}}
.abtn:hover{{border-color:var(--accent);color:var(--accent2);background:#1e1b4b}}
.cbtn{{width:100%;background:var(--accent);border:none;border-radius:var(--r);color:#fff;font-size:13px;font-weight:700;padding:11px;cursor:pointer;font-family:inherit;letter-spacing:.02em}}
.cbtn:hover{{background:#4f46e5}}

.main{{overflow-y:auto;padding:18px 22px;display:flex;flex-direction:column;gap:16px}}

.sstrip{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px}}
.scard{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:12px 14px}}
.scard .sl{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:3px}}
.scard .sv{{font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace}}
.vg{{color:var(--green)}} .vr{{color:var(--red)}} .va{{color:var(--amber)}} .vi{{color:var(--accent2)}}

.box{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:18px}}
.bt{{font-size:12px;font-weight:700;color:var(--bright);margin-bottom:3px}}
.bs{{font-size:11px;color:var(--muted);margin-bottom:14px}}

.cw{{position:relative;height:350px}}

.empty{{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:400px;color:var(--muted);gap:10px}}
.empty .ei{{font-size:48px;opacity:.3}}
.empty .em{{font-size:14px;font-weight:500}}
.empty .es{{font-size:12px;opacity:.6}}

.ht{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11px}}
.ht th{{padding:5px 7px;font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;border-bottom:2px solid var(--border);text-align:center;background:var(--surface);position:sticky;top:0;z-index:2}}
.ht th.exp-col{{background:#1e1b4b;color:var(--accent2)}}
.ht td{{padding:4px 6px;text-align:center;font-size:11px;white-space:nowrap}}
.ht .rh{{text-align:right;font-weight:600;color:var(--bright);background:var(--surface2);position:sticky;left:0;z-index:1;border-right:1px solid var(--border)}}
.ht .pct{{color:var(--muted);font-size:10px;text-align:right;border-left:1px solid var(--border)}}

.gt{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px}}
.gt th{{padding:7px 9px;font-size:10px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;border-bottom:2px solid var(--border);text-align:right}}
.gt th:first-child{{text-align:left}}
.gt td{{padding:5px 9px;text-align:right;border-bottom:1px solid var(--border)}}
.gt td:first-child{{text-align:left;font-weight:600;color:var(--bright)}}
.gt tr:last-child{{font-weight:700;color:var(--bright)}}
.gt tr:last-child td{{border-top:2px solid var(--border2);border-bottom:none}}

::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px}}
@media(max-width:900px){{.app{{grid-template-columns:1fr}}.sidebar{{max-height:50vh}}}}
</style>
</head>
<body>
<div class="app">
<div class="header">
  <div class="logo">NIFTY <span>OPTIONS P&L</span></div>
  <div class="hbadge">&#8377; {spot:,.2f}</div>
  <span class="htag htag-i">LOT {nifty_lot_size}</span>
  <span class="htag {'htag-a' if (vix and vix > 18) else 'htag-g'}">VIX {vix_val:.1f}</span>
  <div class="header-r">{snapshot_ts}</div>
</div>

<div class="sidebar">
  <div class="sec">
    <div class="sec-t">Strategy Presets</div>
    <div class="presets">
      <button class="pbtn" onclick="lp('long_call',this)">Long Call</button>
      <button class="pbtn" onclick="lp('long_put',this)">Long Put</button>
      <button class="pbtn" onclick="lp('bull_call_spread',this)">Bull Call Spread</button>
      <button class="pbtn" onclick="lp('bear_put_spread',this)">Bear Put Spread</button>
      <button class="pbtn" onclick="lp('iron_condor',this)">Iron Condor</button>
      <button class="pbtn" onclick="lp('straddle',this)">Long Straddle</button>
      <button class="pbtn" onclick="lp('strangle',this)">Long Strangle</button>
      <button class="pbtn" onclick="lp('butterfly',this)">Butterfly</button>
      <button class="pbtn" onclick="lp('short_straddle',this)">Short Straddle</button>
      <button class="pbtn" onclick="lp('bear_call_spread',this)">Bear Call Spread</button>
    </div>
  </div>
  <div class="sec">
    <div class="sec-t">Expiry & Lots</div>
    <div class="fr">
      <div class="fg">
        <label>Expiry</label>
        <select id="expSel">{''.join(f'<option value="{e}" {"selected" if e==expiry_str else ""}>{e}</option>' for e in all_expiries)}</select>
      </div>
      <div class="fg">
        <label>Lots</label>
        <input type="number" id="lotsIn" value="1" min="1" max="100"/>
      </div>
    </div>
  </div>
  <div class="sec">
    <div class="sec-t">Position Legs</div>
    <div id="legBox"></div>
    <button class="abtn" id="addBtn">+ Add Leg</button>
    <button class="cbtn" id="calcBtn">Calculate P&L</button>
  </div>
</div>

<div class="main" id="mainP">
  <div class="empty" id="emptyS">
    <div class="ei">&#128200;</div>
    <div class="em">Select a strategy or add legs</div>
    <div class="es">Choose a preset from the sidebar to begin</div>
  </div>
  <div id="resP" style="display:none">
    <div class="sstrip" id="sumStrip"></div>
    <div class="box" style="margin-top:16px">
      <div class="bt">Payoff Diagram</div>
      <div class="bs">P&L across NIFTY prices — each line is a different date</div>
      <div class="cw"><canvas id="pChart"></canvas></div>
    </div>
    <div class="box">
      <div class="bt">P&L Table</div>
      <div class="bs">Estimated P&L (&#8377;) across dates and NIFTY prices</div>
      <div id="hmWrap" style="overflow:auto;max-height:500px"></div>
    </div>
    <div class="box">
      <div class="bt">Greeks</div>
      <div class="bs">Per-leg and net greeks at current spot</div>
      <div id="gkWrap"></div>
    </div>
  </div>
</div>
</div>

<script>
"use strict";
const C={chain_json};
const SPOT={spot},LOT={nifty_lot_size},EXP="{expiry_str}",TOD="{today_str}",VIX={vix_val},RF=.065;
let legs=[],chart=null;

// ── Black-Scholes ──
function ncdf(x){{const a1=.254829592,a2=-.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=.3275911;const s=x<0?-1:1;x=Math.abs(x)/Math.SQRT2;const t=1/(1+p*x);return .5*(1+s*(1-(((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x)))}}
function npdf(x){{return Math.exp(-.5*x*x)/Math.sqrt(2*Math.PI)}}
function bs(S,K,T,r,v,tp){{if(T<=0)return tp==='CE'?Math.max(0,S-K):Math.max(0,K-S);if(v<=0)v=.001;const sq=Math.sqrt(T),d1=(Math.log(S/K)+(r+.5*v*v)*T)/(v*sq),d2=d1-v*sq;return tp==='CE'?S*ncdf(d1)-K*Math.exp(-r*T)*ncdf(d2):K*Math.exp(-r*T)*ncdf(-d2)-S*ncdf(-d1)}}
function bsg(S,K,T,r,v,tp){{if(T<=1e-6){{const itm=tp==='CE'?S>K:S<K;return{{d:itm?(tp==='CE'?1:-1):0,g:0,t:0,v:0}}}}if(v<=0)v=.001;const sq=Math.sqrt(T),d1=(Math.log(S/K)+(r+.5*v*v)*T)/(v*sq),d2=d1-v*sq,nd=npdf(d1),eR=Math.exp(-r*T);let d,t;if(tp==='CE'){{d=ncdf(d1);t=(-S*nd*v/(2*sq)-r*K*eR*ncdf(d2))/365}}else{{d=ncdf(d1)-1;t=(-S*nd*v/(2*sq)+r*K*eR*ncdf(-d2))/365}}return{{d,g:nd/(S*v*sq),t,v:S*nd*sq/100}}}}

// ── Helpers ──
function dyrs(a,b){{return(new Date(b)-new Date(a))/(365.25*864e5)}}
function dadd(d,n){{const x=new Date(d);x.setDate(x.getDate()+n);return x.toISOString().slice(0,10)}}
function fmtd(iso){{const m=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];const d=new Date(iso);return d.getDate()+' '+m[d.getMonth()]}}
function cls(s,t){{let b=C[0];for(const c of C)if(Math.abs(c.strike-t)<Math.abs(b.strike-t))b=c;return b.strike}}
function gsd(s){{return C.find(c=>c.strike===s)||null}}
function gltp(s,t){{const d=gsd(s);if(!d)return 0;return t==='CE'?(d.ce_ltp||0):(d.pe_ltp||0)}}
function giv(s,t){{const d=gsd(s);if(!d)return VIX/100;const iv=t==='CE'?(d.ce_iv||0):(d.pe_iv||0);return iv>0?iv/100:VIX/100}}

// ── Legs ──
function mkleg(side,tp,strike){{if(!strike)strike=cls(0,SPOT);return{{side,tp,strike,prem:gltp(strike,tp),iv:giv(strike,tp)}}}}

const $=id=>document.getElementById(id);

function rndr(){{
  const box=$('legBox');box.innerHTML='';
  const opts=C.map(c=>`<option value="${{c.strike}}">${{c.strike.toLocaleString()}}</option>`).join('');
  legs.forEach((l,i)=>{{
    const d=document.createElement('div');d.className='lc '+(l.side==='buy'?'buy':'sell');
    d.innerHTML=`<div class="lh"><span class="ll ${{l.side==='buy'?'b':'s'}}">Leg ${{i+1}} — ${{l.side.toUpperCase()}} ${{l.tp}}</span><button class="rx" data-i="${{i}}">&times;</button></div>`+
    `<div class="fr3"><div class="fg"><label>Side</label><select data-i="${{i}}" data-f="side"><option value="buy" ${{l.side==='buy'?'selected':''}}>Buy</option><option value="sell" ${{l.side==='sell'?'selected':''}}>Sell</option></select></div>`+
    `<div class="fg"><label>Type</label><select data-i="${{i}}" data-f="tp"><option value="CE" ${{l.tp==='CE'?'selected':''}}>CALL</option><option value="PE" ${{l.tp==='PE'?'selected':''}}>PUT</option></select></div>`+
    `<div class="fg"><label>Strike</label><select data-i="${{i}}" data-f="strike">${{opts}}</select></div></div>`+
    `<div class="fr"><div class="fg"><label>Premium</label><input type="number" data-i="${{i}}" data-f="prem" value="${{l.prem.toFixed(2)}}" step="0.05"/></div>`+
    `<div class="fg"><label>IV %</label><input type="number" data-i="${{i}}" data-f="iv" value="${{(l.iv*100).toFixed(1)}}" step="0.1"/></div></div>`;
    d.querySelector(`select[data-f="strike"]`).value=l.strike;
    box.appendChild(d);
  }});
  box.addEventListener('change',onLegChange);
  box.addEventListener('click',e=>{{if(e.target.classList.contains('rx')){{legs.splice(+e.target.dataset.i,1);rndr();if(legs.length)calc()}}}});
}}

function onLegChange(e){{
  const el=e.target,i=+el.dataset.i,f=el.dataset.f;
  if(f==='side')legs[i].side=el.value;
  else if(f==='tp'){{legs[i].tp=el.value;legs[i].prem=gltp(legs[i].strike,el.value);legs[i].iv=giv(legs[i].strike,el.value)}}
  else if(f==='strike'){{legs[i].strike=+el.value;legs[i].prem=gltp(+el.value,legs[i].tp);legs[i].iv=giv(+el.value,legs[i].tp)}}
  else if(f==='prem')legs[i].prem=+el.value;
  else if(f==='iv')legs[i].iv=+el.value/100;
  rndr();
}}

$('addBtn').onclick=()=>{{if(legs.length<4){{legs.push(mkleg('buy','CE'));rndr()}}}};
$('calcBtn').onclick=calc;
$('lotsIn').onchange=()=>{{if(legs.length)calc()}};

function lp(n,btn){{
  document.querySelectorAll('.pbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  legs=[];
  const atm=cls(0,SPOT),step=C.length>1?Math.abs(C[1].strike-C[0].strike):50;
  const oc=cls(0,SPOT+4*step),op=cls(0,SPOT-4*step),fc=cls(0,SPOT+8*step),fp=cls(0,SPOT-8*step);
  switch(n){{
    case'long_call':legs=[mkleg('buy','CE',atm)];break;
    case'long_put':legs=[mkleg('buy','PE',atm)];break;
    case'bull_call_spread':legs=[mkleg('buy','CE',atm),mkleg('sell','CE',oc)];break;
    case'bear_put_spread':legs=[mkleg('buy','PE',atm),mkleg('sell','PE',op)];break;
    case'bear_call_spread':legs=[mkleg('sell','CE',atm),mkleg('buy','CE',oc)];break;
    case'iron_condor':legs=[mkleg('sell','PE',op),mkleg('buy','PE',fp),mkleg('sell','CE',oc),mkleg('buy','CE',fc)];break;
    case'straddle':legs=[mkleg('buy','CE',atm),mkleg('buy','PE',atm)];break;
    case'strangle':legs=[mkleg('buy','CE',oc),mkleg('buy','PE',op)];break;
    case'butterfly':legs=[mkleg('buy','CE',cls(0,SPOT-4*step)),mkleg('sell','CE',atm),mkleg('sell','CE',atm),mkleg('buy','CE',cls(0,SPOT+4*step))];break;
    case'short_straddle':legs=[mkleg('sell','CE',atm),mkleg('sell','PE',atm)];break;
  }}
  rndr();calc();
}}

// ── P&L engine ──
function pnl(S,T){{
  let tot=0;
  for(const l of legs){{
    const sgn=l.side==='buy'?1:-1;
    tot+=sgn*(bs(S,l.strike,T,RF,l.iv,l.tp)-l.prem);
  }}
  return tot;
}}

function calc(){{
  if(!legs.length)return;
  $('emptyS').style.display='none';
  $('resP').style.display='';

  const lots=+$('lotsIn').value||1;
  const Tf=Math.max(dyrs(TOD,EXP),1/365);
  const daysTotal=Math.max(1,Math.round(Tf*365));

  // Price range
  const stks=legs.map(l=>l.strike);
  const mn=Math.min(...stks),mx=Math.max(...stks);
  const rng=Math.max(mx-mn,SPOT*.03);
  const lo=Math.floor((SPOT-rng*2.5)/50)*50,hi=Math.ceil((SPOT+rng*2.5)/50)*50;
  const pStep=Math.max(50,Math.round((hi-lo)/80/50)*50);

  const prices=[];
  for(let p=lo;p<=hi;p+=pStep)prices.push(p);

  // Date lines: today, evenly spaced days, expiry
  const nLines=Math.min(daysTotal+1,6);
  const dayOffsets=[];
  for(let i=0;i<nLines;i++)dayOffsets.push(Math.round(i/(nLines-1)*daysTotal));
  if(dayOffsets[dayOffsets.length-1]!==daysTotal)dayOffsets.push(daysTotal);
  const uniqueDays=[...new Set(dayOffsets)].sort((a,b)=>a-b);

  // Compute P&L for each date line
  const lineData=uniqueDays.map(d=>{{
    const Tr=Math.max(0,Tf-d/365);
    return prices.map(S=>pnl(S,Tr)*LOT*lots);
  }});

  // Net premium
  let netPrem=0;
  for(const l of legs)netPrem+=(l.side==='buy'?-1:1)*l.prem;
  const netCost=netPrem*LOT*lots;

  // Max profit/loss/breakevens from expiry line
  const expLine=lineData[lineData.length-1];
  let maxP=-Infinity,maxL=Infinity;
  const bkevens=[];
  for(let i=0;i<prices.length;i++){{
    if(expLine[i]>maxP)maxP=expLine[i];
    if(expLine[i]<maxL)maxL=expLine[i];
    if(i>0&&((expLine[i-1]<0&&expLine[i]>=0)||(expLine[i-1]>=0&&expLine[i]<0))){{
      const r=Math.abs(expLine[i-1])/(Math.abs(expLine[i-1])+Math.abs(expLine[i]));
      bkevens.push(Math.round(prices[i-1]+r*pStep));
    }}
  }}
  if(maxP>5e8)maxP=Infinity;
  if(maxL<-5e8)maxL=-Infinity;

  // Summary
  $('sumStrip').innerHTML=
    `<div class="scard"><div class="sl">Net Premium</div><div class="sv ${{netCost>=0?'vg':'vr'}}">${{netCost>=0?'+':''}}&#8377;${{Math.abs(netCost).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div></div>`+
    `<div class="scard"><div class="sl">Max Profit</div><div class="sv vg">${{maxP===Infinity?'Unlimited':'&#8377;'+maxP.toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div></div>`+
    `<div class="scard"><div class="sl">Max Loss</div><div class="sv vr">${{maxL===-Infinity?'Unlimited':'&#8377;'+Math.abs(maxL).toLocaleString(undefined,{{maximumFractionDigits:0}})}}</div></div>`+
    `<div class="scard"><div class="sl">Breakeven</div><div class="sv vi">${{bkevens.length?bkevens.map(b=>b.toLocaleString()).join(', '):'—'}}</div></div>`+
    `<div class="scard"><div class="sl">Risk/Reward</div><div class="sv va">${{(maxP!==Infinity&&maxL!==-Infinity&&maxL!==0)?(maxP/Math.abs(maxL)).toFixed(2)+'x':'—'}}</div></div>`+
    `<div class="scard"><div class="sl">Days to Expiry</div><div class="sv vi">${{daysTotal}}</div></div>`;

  // Chart — multi-line payoff
  drawChart(prices,uniqueDays,lineData);

  // Heatmap
  drawHeatmap(prices,uniqueDays,lineData,daysTotal);

  // Greeks
  drawGreeks(Tf);
}}

// ── Chart ──
const LINE_COLORS=['#505570','#ef4444','#f59e0b','#10b981','#06b6d4','#6366f1','#8b5cf6','#ec4899'];

function drawChart(prices,days,lineData){{
  if(chart){{chart.destroy();chart=null}}
  const datasets=[];
  for(let i=0;i<days.length;i++){{
    const d=days[i];
    const isExp=i===days.length-1;
    const isToday=i===0;
    const lbl=isExp?'Expiry':isToday?'Current':fmtd(dadd(TOD,d));
    datasets.push({{
      label:lbl,
      data:lineData[i],
      borderColor:isExp?'#6366f1':LINE_COLORS[i%LINE_COLORS.length],
      backgroundColor:'transparent',
      borderWidth:isExp?2.5:1.5,
      borderDash:isToday?[5,3]:[],
      pointRadius:0,
      tension:.3,
    }});
  }}
  const ctx=$('pChart').getContext('2d');
  chart=new Chart(ctx,{{
    type:'line',
    data:{{labels:prices,datasets}},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      animation:false,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{
          position:'right',
          labels:{{color:'#505570',font:{{family:'Inter',size:11}},usePointStyle:true,pointStyle:'line',padding:8}}
        }},
        tooltip:{{
          backgroundColor:'#161922ee',
          borderColor:'#252838',
          borderWidth:1,
          titleColor:'#eef0f6',
          bodyColor:'#c9cdd8',
          titleFont:{{family:'JetBrains Mono',size:12,weight:'600'}},
          bodyFont:{{family:'JetBrains Mono',size:11}},
          callbacks:{{
            title:items=>'NIFTY @ '+prices[items[0]?.dataIndex]?.toLocaleString(),
            label:item=>{{
              const v=item.parsed.y;
              return ' '+item.dataset.label+': ₹'+(v>=0?'+':'')+Math.round(v).toLocaleString();
            }}
          }}
        }}
      }},
      scales:{{
        x:{{
          title:{{display:true,text:'NIFTY',color:'#505570',font:{{family:'Inter',size:11}}}},
          grid:{{color:'#14161f'}},
          ticks:{{color:'#505570',font:{{family:'JetBrains Mono',size:10}},maxTicksLimit:12,callback:(v,i)=>prices[i]?.toLocaleString()}}
        }},
        y:{{
          title:{{display:true,text:'P&L (₹)',color:'#505570',font:{{family:'Inter',size:11}}}},
          grid:{{color:'#14161f'}},
          ticks:{{color:'#505570',font:{{family:'JetBrains Mono',size:10}},callback:v=>'₹'+v.toLocaleString()}}
        }}
      }}
    }}
  }});
}}

// ── Heatmap ──
function drawHeatmap(prices,days,lineData,daysTotal){{
  let h='<table class="ht"><thead><tr><th>NIFTY</th>';
  for(let i=0;i<days.length;i++){{
    const d=days[i];
    const isExp=i===days.length-1;
    h+=`<th${{isExp?' class="exp-col"':''}}>${{isExp?'Exp':fmtd(dadd(TOD,d))}}</th>`;
  }}
  h+='<th class="pct">+/-%</th></tr></thead><tbody>';

  for(let p=0;p<prices.length;p++){{
    const price=prices[p];
    const isSpot=Math.abs(price-SPOT)<(prices[1]-prices[0]);
    h+=`<tr><td class="rh" style="${{isSpot?'color:var(--amber);':''}}">${{price.toLocaleString()}}.00</td>`;
    for(let i=0;i<days.length;i++){{
      const val=lineData[i][p];
      const rnd=Math.round(val);
      const intensity=Math.min(1,Math.abs(val)/(Math.abs(val)+8000)*.85+.1);
      let bg;
      if(rnd>0)bg=`rgba(16,185,129,${{intensity}})`;
      else if(rnd<0)bg=`rgba(239,68,68,${{intensity}})`;
      else bg='transparent';
      h+=`<td style="background:${{bg}};color:#fff">${{rnd>=0?'':''}}${{rnd.toLocaleString()}}</td>`;
    }}
    const pctChg=((price-SPOT)/SPOT*100).toFixed(2);
    h+=`<td class="pct">${{+pctChg>=0?'+':''}}${{pctChg}}%</td></tr>`;
  }}
  h+='</tbody></table>';
  $('hmWrap').innerHTML=h;
}}

// ── Greeks ──
function drawGreeks(Tf){{
  const lots=+$('lotsIn').value||1;
  const rows=[];
  let net={{d:0,g:0,t:0,v:0}};
  for(const l of legs){{
    const sgn=l.side==='buy'?1:-1;
    const g=bsg(SPOT,l.strike,Tf,RF,l.iv,l.tp);
    const r={{
      lbl:l.side.toUpperCase()+' '+l.tp+' '+l.strike.toLocaleString(),
      d:g.d*sgn*lots*LOT,g:g.g*sgn*lots*LOT,
      t:g.t*sgn*lots*LOT,v:g.v*sgn*lots*LOT
    }};
    rows.push(r);
    net.d+=r.d;net.g+=r.g;net.t+=r.t;net.v+=r.v;
  }}
  rows.push({{lbl:'NET',...net}});
  let h='<table class="gt"><thead><tr><th>Leg</th><th>Delta</th><th>Gamma</th><th>Theta</th><th>Vega</th></tr></thead><tbody>';
  for(const r of rows){{
    h+=`<tr><td>${{r.lbl}}</td><td>${{r.d.toFixed(2)}}</td><td>${{r.g.toFixed(4)}}</td>`+
      `<td style="color:${{r.t>=0?'var(--green)':'var(--red)'}}">${{r.t.toFixed(2)}}</td><td>${{r.v.toFixed(2)}}</td></tr>`;
  }}
  h+='</tbody></table>';
  $('gkWrap').innerHTML=h;
}}
</script>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)

    csv_path = Path(args.csv) if args.csv else find_latest_csv(output_dir)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.", file=sys.stderr)
        return 1

    print(f"Loading: {csv_path.name}")
    df = load(csv_path)

    expiry = choose_expiry(df, args.expiry)
    spot = float(df[df["expiry_dt"] == expiry]["underlying_value"].median()) or 0.0
    snapshot_ts = df["snapshot_timestamp"].iloc[0] if "snapshot_timestamp" in df.columns else "N/A"

    today = pd.Timestamp(datetime.now().date())
    future = sorted(df[df["expiry_dt"] >= today]["expiry_dt"].unique().tolist())
    all_expiries = [str(e.date()) for e in future]

    chain_json = build_chain_json(df, expiry, spot, args.strikes)
    vix = get_vix(output_dir)

    html = render_html(
        chain_json=chain_json,
        spot=spot,
        expiry=expiry,
        all_expiries=all_expiries,
        snapshot_ts=snapshot_ts,
        vix=vix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"options_profit_calc_{stamp}.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"Spot     : {spot:,.2f}")
    print(f"Expiry   : {expiry.date()}")
    print(f"VIX      : {vix}")
    print(f"Strikes  : {len(json.loads(chain_json))}")
    print(f"Output   : {out_path.resolve()}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
