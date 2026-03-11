#!/usr/bin/env python3
"""
NIFTY Option Chain Visualiser — Sensibull-style table
======================================================
Produces a self-contained HTML page that looks like the Sensibull option chain:
  - CALLS columns on the left, Strike + IV in the centre, PUTS columns on the right
  - ITM rows highlighted differently from OTM rows
  - Inline OI bar (relative width) inside the OI cell
  - Expiry switcher dropdown at the top
  - Colour-coded LTP change (green / red)
  - Summary bar: spot, VIX, PCR, max-pain, ATM

Usage
-----
    python visualise_option_chain.py                            # auto-pick latest CSV
    python visualise_option_chain.py --csv output/NIFTY_*.csv
    python visualise_option_chain.py --expiry 17-03-2026
    python visualise_option_chain.py --strikes 30 --open
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIFTY option-chain HTML table viewer.")
    p.add_argument("--csv", default=None, help="Path to option-chain CSV. Auto-detects if omitted.")
    p.add_argument("--expiry", default=None, help="Expiry to display, e.g. '17-03-2026'.")
    p.add_argument("--strikes", type=int, default=40, help="Strikes to show centred on ATM (default 40).")
    p.add_argument("--output-dir", default="output", help="Where to save HTML. Default: output")
    p.add_argument("--open", action="store_true", help="Open HTML in browser after generating.")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────

def find_latest_csv(output_dir: Path) -> Path:
    files = sorted(output_dir.glob("*_option_chain_*.csv"))
    if not files:
        raise FileNotFoundError(f"No option-chain CSV found in {output_dir}.")
    return files[-1]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    num = ["strike_price", "last_price", "change", "pchange",
           "open_interest", "change_in_open_interest", "volume",
           "implied_volatility", "underlying_value",
           "best_bid_price", "best_ask_price",
           "total_buy_quantity", "total_sell_quantity"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["expiry_dt"] = pd.to_datetime(df["contract_expiry"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["expiry_dt"])


def choose_expiry(df: pd.DataFrame, expiry_filter: str | None) -> pd.Timestamp:
    today = pd.Timestamp(datetime.now().date())
    future = sorted(df[df["expiry_dt"] >= today]["expiry_dt"].unique().tolist())
    all_exp = sorted(df["expiry_dt"].unique().tolist())
    if expiry_filter:
        dt = pd.to_datetime(expiry_filter, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"Cannot parse expiry: {expiry_filter}")
        return dt
    return future[0] if future else all_exp[0]


def build_rows(df: pd.DataFrame, expiry: pd.Timestamp, n_strikes: int, spot: float) -> list[dict]:
    sub = df[df["expiry_dt"] == expiry]
    ce = sub[sub["option_type"] == "CE"].set_index("strike_price")
    pe = sub[sub["option_type"] == "PE"].set_index("strike_price")

    all_strikes = sorted(set(ce.index) | set(pe.index))
    if spot > 0 and all_strikes:
        atm = min(all_strikes, key=lambda s: abs(s - spot))
        idx = all_strikes.index(atm)
        half = n_strikes // 2
        all_strikes = all_strikes[max(0, idx - half): idx + half]

    def g(frame, strike, col, default=0.0):
        if strike in frame.index:
            v = frame.loc[strike, col]
            if isinstance(v, pd.Series):
                v = v.iloc[0]
            try:
                return float(v)
            except Exception:
                return default
        return default

    rows = []
    for s in all_strikes:
        rows.append({
            "strike": s,
            # CALL side
            "ce_ltp":    g(ce, s, "last_price"),
            "ce_chg":    g(ce, s, "change"),
            "ce_pchg":   g(ce, s, "pchange"),
            "ce_oi":     g(ce, s, "open_interest"),
            "ce_doi":    g(ce, s, "change_in_open_interest"),
            "ce_vol":    g(ce, s, "volume"),
            "ce_iv":     g(ce, s, "implied_volatility"),
            "ce_bid":    g(ce, s, "best_bid_price"),
            "ce_ask":    g(ce, s, "best_ask_price"),
            # PUT side
            "pe_ltp":    g(pe, s, "last_price"),
            "pe_chg":    g(pe, s, "change"),
            "pe_pchg":   g(pe, s, "pchange"),
            "pe_oi":     g(pe, s, "open_interest"),
            "pe_doi":    g(pe, s, "change_in_open_interest"),
            "pe_vol":    g(pe, s, "volume"),
            "pe_iv":     g(pe, s, "implied_volatility"),
            "pe_bid":    g(pe, s, "best_bid_price"),
            "pe_ask":    g(pe, s, "best_ask_price"),
        })
    return rows


def max_pain(df: pd.DataFrame, expiry: pd.Timestamp) -> float:
    sub = df[df["expiry_dt"] == expiry]
    ce = sub[sub["option_type"] == "CE"].set_index("strike_price")
    pe = sub[sub["option_type"] == "PE"].set_index("strike_price")
    strikes = sorted(set(ce.index) | set(pe.index))
    if not strikes:
        return 0.0
    best, best_loss = strikes[0], float("inf")
    for s in strikes:
        loss = sum(max(0.0, s - k) * (ce.loc[k, "open_interest"] if k in ce.index else 0) for k in strikes)
        loss += sum(max(0.0, k - s) * (pe.loc[k, "open_interest"] if k in pe.index else 0) for k in strikes)
        if loss < best_loss:
            best_loss, best = loss, s
    return best


# ── HTML ─────────────────────────────────────────────────────────────────────

def render(
    rows: list[dict],
    spot: float,
    expiry: pd.Timestamp,
    all_expiries: list[str],
    snapshot_ts: str,
    total_ce_oi: float,
    total_pe_oi: float,
    max_pain_strike: float,
    vix: float | None,
) -> str:
    atm = min((r["strike"] for r in rows), key=lambda s: abs(s - spot), default=0)
    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else None
    max_oi = max((max(r["ce_oi"], r["pe_oi"]) for r in rows), default=1) or 1

    vix_str = f"{vix:.2f}" if vix else "N/A"
    vix_class = "warn" if (vix and vix > 18) else "bull"
    pcr_str = str(pcr) if pcr else "N/A"
    pcr_class = "bear" if (pcr and pcr > 1) else "bull"
    expiry_opts = "\n".join(
        f'<option value="{e}" {"selected" if e == str(expiry.date()) else ""}>{e}</option>'
        for e in all_expiries
    )

    def fmt(v: float, decimals: int = 2) -> str:
        if v == 0:
            return "-"
        return f"{v:,.{decimals}f}"

    def chg_span(chg: float, pchg: float) -> str:
        if chg == 0 and pchg == 0:
            return '<span class="muted">-</span>'
        cls = "up" if chg >= 0 else "dn"
        sign = "+" if chg >= 0 else ""
        return f'<span class="{cls}">{sign}{chg:,.2f} ({sign}{pchg:.1f}%)</span>'

    def oi_cell(oi: float, doi: float, side: str) -> str:
        bar_pct = min(100, int(oi / max_oi * 100))
        bar_col = "#00c896" if side == "ce" else "#ff4d6d"
        doi_sign = "+" if doi >= 0 else ""
        doi_cls = "up" if doi >= 0 else "dn"
        doi_str = f'<span class="doi {doi_cls}">{doi_sign}{doi/1000:.1f}K</span>' if doi != 0 else ""
        bar_dir = "left" if side == "ce" else "right"
        return (
            f'<div class="oi-wrap">'
            f'<div class="oi-bar-bg">'
            f'<div class="oi-bar oi-bar-{bar_dir}" style="width:{bar_pct}%;background:{bar_col}"></div>'
            f'</div>'
            f'<span class="oi-num">{oi/1000:.1f}K</span>{doi_str}'
            f'</div>'
        )

    tbody = ""
    for r in rows:
        s = r["strike"]
        itm_ce = s < spot   # call is ITM if strike < spot
        itm_pe = s > spot   # put is ITM if strike > spot
        is_atm = s == atm

        row_class = "atm" if is_atm else ""
        ce_class = "itm-ce" if itm_ce else "otm-ce"
        pe_class = "itm-pe" if itm_pe else "otm-pe"

        # intrinsic value
        ce_intrinsic = max(0.0, spot - s)
        pe_intrinsic = max(0.0, s - spot)
        ce_tv = max(0.0, r["ce_ltp"] - ce_intrinsic)
        pe_tv = max(0.0, r["pe_ltp"] - pe_intrinsic)
        ce_be = s + r["ce_ltp"] if r["ce_ltp"] > 0 else 0
        pe_be = s - r["pe_ltp"] if r["pe_ltp"] > 0 else 0

        tbody += f"""
        <tr class="{row_class}" data-strike="{s}">
          <td class="{ce_class} num">{fmt(r['ce_ltp'])}<br>{chg_span(r['ce_chg'], r['ce_pchg'])}</td>
          <td class="{ce_class} num iv-td">{fmt(r['ce_iv'])}%</td>
          <td class="{ce_class} oi-td">{oi_cell(r['ce_oi'], r['ce_doi'], 'ce')}</td>
          <td class="{ce_class} num vol-td">{fmt(r['ce_vol'], 0)}</td>
          <td class="{ce_class} num">{fmt(ce_be)}</td>
          <td class="{ce_class} num">{fmt(ce_tv)}</td>
          <td class="{ce_class} num">{fmt(ce_intrinsic)}</td>
          <td class="strike-cell {'atm-strike' if is_atm else ''}">{int(s):,}</td>
          <td class="{pe_class} num">{fmt(pe_intrinsic)}</td>
          <td class="{pe_class} num">{fmt(pe_tv)}</td>
          <td class="{pe_class} num">{fmt(pe_be)}</td>
          <td class="{pe_class} oi-td">{oi_cell(r['pe_oi'], r['pe_doi'], 'pe')}</td>
          <td class="{pe_class} num iv-td">{fmt(r['pe_iv'])}%</td>
          <td class="{pe_class} num">{fmt(r['pe_ltp'])}<br>{chg_span(r['pe_chg'], r['pe_pchg'])}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>NIFTY Option Chain — {expiry.date()}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {{
  --bg:        #0e1117;
  --surface:   #161b27;
  --surface2:  #1c2235;
  --border:    #252d42;
  --text:      #c9d1e0;
  --muted:     #4a5568;
  --head:      #8892a4;
  --ce-itm:    #0d2b20;
  --ce-otm:    #0e1117;
  --pe-itm:    #2b0d1a;
  --pe-otm:    #0e1117;
  --atm-bg:    #1a1f30;
  --atm-str:   #e8b84b;
  --up:        #00c896;
  --dn:        #ff4d6d;
  --ce-accent: #00c896;
  --pe-accent: #ff4d6d;
  --warn:      #f59e0b;
  --neutral:   #7b8cde;
  --r: 6px;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  min-height: 100vh;
}}

/* ── Top bar ── */
.topbar {{
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 20px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}}
.topbar-brand {{
  font-size: 15px;
  font-weight: 700;
  color: #fff;
  letter-spacing: 0.04em;
}}
.topbar-brand span {{ color: var(--ce-accent); }}

.spot-badge {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 15px;
  font-weight: 700;
  color: #fff;
}}

.expiry-select {{
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text);
  padding: 5px 10px;
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
  outline: none;
}}
.expiry-select:focus {{ border-color: var(--neutral); }}

.tag {{
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
}}
.tag-vix  {{ background: #2a1f0a; color: var(--warn); border: 1px solid #5a3f10; }}
.tag-bull {{ background: #0a2010; color: var(--up);   border: 1px solid #0f4025; }}
.tag-bear {{ background: #2a0a14; color: var(--dn);   border: 1px solid #5a1025; }}
.tag-neutral {{ background: var(--surface2); color: var(--neutral); border: 1px solid var(--border); }}

.topbar-right {{ margin-left: auto; color: var(--muted); font-size: 11px; }}

/* ── Stats strip ── */
.stats {{
  display: flex;
  gap: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  overflow-x: auto;
}}
.stat {{
  padding: 8px 20px;
  border-right: 1px solid var(--border);
  white-space: nowrap;
}}
.stat-lbl {{ color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px; }}
.stat-val {{ font-size: 14px; font-weight: 700; }}
.ce-col   {{ color: var(--ce-accent); }}
.pe-col   {{ color: var(--pe-accent); }}
.warn-col {{ color: var(--warn); }}
.neutral-col {{ color: var(--neutral); }}

/* ── Table wrapper ── */
.table-wrap {{
  overflow-x: auto;
  overflow-y: auto;
  max-height: calc(100vh - 130px);
}}

table {{
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}}

/* ── Column header ── */
thead {{
  position: sticky;
  top: 0;
  z-index: 10;
}}
.calls-head th {{
  background: #0d2218;
  color: var(--ce-accent);
  font-size: 11px;
  font-weight: 600;
  padding: 8px 6px;
  text-align: center;
  border-bottom: 2px solid var(--ce-accent);
}}
.puts-head th {{
  background: #2b0d18;
  color: var(--pe-accent);
  font-size: 11px;
  font-weight: 600;
  padding: 8px 6px;
  text-align: center;
  border-bottom: 2px solid var(--pe-accent);
}}
.strike-head th {{
  background: var(--surface);
  color: var(--head);
  font-size: 11px;
  font-weight: 600;
  padding: 8px 6px;
  text-align: center;
  border-bottom: 2px solid var(--border);
}}

/* ── Data cells ── */
td {{
  padding: 5px 6px;
  text-align: right;
  vertical-align: middle;
  border-bottom: 1px solid #1a2030;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

/* ITM / OTM colouring */
.itm-ce {{ background: var(--ce-itm); }}
.otm-ce {{ background: var(--ce-otm); }}
.itm-pe {{ background: var(--pe-itm); }}
.otm-pe {{ background: var(--pe-otm); }}

/* ATM row */
tr.atm td {{ background: var(--atm-bg) !important; }}
tr.atm {{ outline: 1px solid #2e3a5a; }}

/* Strike centre cell */
.strike-cell {{
  text-align: center;
  font-weight: 700;
  font-size: 13px;
  color: var(--text);
  background: var(--surface2) !important;
  border-left: 1px solid var(--border);
  border-right: 1px solid var(--border);
  padding: 5px 8px;
}}
.atm-strike {{
  color: var(--atm-str) !important;
  font-size: 13px;
}}

/* Numbers */
.num {{ font-variant-numeric: tabular-nums; font-size: 12px; }}

/* Change colours */
.up   {{ color: var(--up); font-size: 10px; }}
.dn   {{ color: var(--dn); font-size: 10px; }}
.muted {{ color: var(--muted); }}

/* IV */
.iv-td {{ color: var(--neutral); font-size: 11px; text-align: center; }}

/* OI bar */
.oi-td {{ padding: 4px 6px; min-width: 100px; }}
.oi-wrap {{
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 90px;
}}
.oi-bar-bg {{
  height: 4px;
  background: #1f2535;
  border-radius: 2px;
  overflow: hidden;
}}
.oi-bar {{ height: 4px; border-radius: 2px; transition: width 0.3s; }}
.oi-bar-left  {{ float: right; }}
.oi-bar-right {{ float: left; }}
.oi-num {{ font-size: 11px; color: var(--text); }}
.doi {{ font-size: 10px; }}

/* Volume */
.vol-td {{ font-size: 11px; color: var(--muted); }}

/* Hover */
tbody tr:hover td {{ filter: brightness(1.25); cursor: default; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: var(--bg); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-brand">NIFTY <span>OPTION CHAIN</span></div>
  <div class="spot-badge">&#8377; {spot:,.2f}</div>
  <label style="color:var(--muted);font-size:11px">Expiry</label>
  <select class="expiry-select" onchange="location.href='#'+this.value">
    {expiry_opts}
  </select>
  <span class="tag tag-neutral">PCR <strong class="{pcr_class}-col">{pcr_str}</strong></span>
  <span class="tag {'tag-vix' if (vix and vix > 18) else 'tag-bull'}">India VIX {vix_str}</span>
  <span class="tag tag-neutral">Max Pain <strong class="warn-col">{int(max_pain_strike):,}</strong></span>
  <div class="topbar-right">Snapshot: {snapshot_ts}</div>
</div>

<!-- Stats strip -->
<div class="stats">
  <div class="stat"><div class="stat-lbl">Total CE OI</div><div class="stat-val ce-col">{total_ce_oi/100000:.2f}L</div></div>
  <div class="stat"><div class="stat-lbl">Total PE OI</div><div class="stat-val pe-col">{total_pe_oi/100000:.2f}L</div></div>
  <div class="stat"><div class="stat-lbl">Put-Call Ratio</div><div class="stat-val {'pe-col' if pcr and pcr>1 else 'ce-col'}">{pcr_str}</div></div>
  <div class="stat"><div class="stat-lbl">Max Pain</div><div class="stat-val warn-col">{int(max_pain_strike):,}</div></div>
  <div class="stat"><div class="stat-lbl">ATM Strike</div><div class="stat-val neutral-col">{int(atm):,}</div></div>
  <div class="stat"><div class="stat-lbl">India VIX</div><div class="stat-val {'warn-col' if (vix and vix>18) else 'ce-col'}">{vix_str}</div></div>
</div>

<!-- Table -->
<div class="table-wrap">
<table>
<colgroup>
  <!-- CE: LTP | IV | OI | Vol | BE | TV | IntVal | Strike | IntVal | TV | BE | OI | IV | LTP :PE -->
  <col style="width:110px"><col style="width:60px"><col style="width:110px"><col style="width:70px">
  <col style="width:80px"><col style="width:70px"><col style="width:70px">
  <col style="width:80px">
  <col style="width:70px"><col style="width:70px"><col style="width:80px">
  <col style="width:110px"><col style="width:60px"><col style="width:110px">
</colgroup>
<thead>
  <tr class="calls-head">
    <th>LTP (Chg)</th><th>IV</th><th>OI (ΔOI)</th><th>Volume</th>
    <th>Breakeven</th><th>Time Val</th><th>Int Val</th>
    <th class="strike-head" style="background:var(--surface);color:var(--head);border-bottom:2px solid var(--border)">STRIKE</th>
    <th>Int Val</th><th>Time Val</th><th>Breakeven</th>
    <th>OI (ΔOI)</th><th>IV</th><th>LTP (Chg)</th>
  </tr>
  <tr class="calls-head" style="font-size:10px;opacity:0.6">
    <th colspan="7" style="text-align:center;letter-spacing:0.15em">— CALLS —</th>
    <th></th>
    <th colspan="6" style="text-align:center;letter-spacing:0.15em">— PUTS —</th>
  </tr>
</thead>
<tbody>
{tbody}
</tbody>
</table>
</div>

<script>
// Scroll ATM row into view on load
window.addEventListener('load', () => {{
  const atm = document.querySelector('.atm');
  if (atm) atm.scrollIntoView({{ block: 'center', behavior: 'smooth' }});
}});
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

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

    rows = build_rows(df, expiry, args.strikes, spot)

    sub = df[df["expiry_dt"] == expiry]
    total_ce_oi = float(sub[sub["option_type"] == "CE"]["open_interest"].sum())
    total_pe_oi = float(sub[sub["option_type"] == "PE"]["open_interest"].sum())
    mp = max_pain(df, expiry)

    # Try to load VIX from sibling CSV
    vix: float | None = None
    vix_candidates = sorted(output_dir.glob("INDIAVIX*.csv"))
    if vix_candidates:
        try:
            vdf = pd.read_csv(vix_candidates[-1])
            vdf.columns = [c.strip() for c in vdf.columns]
            vcol = next((c for c in vdf.columns if "close" in c.lower()), None)
            if vcol:
                vix = float(pd.to_numeric(vdf[vcol], errors="coerce").dropna().iloc[-1])
        except Exception:
            pass

    html = render(
        rows=rows,
        spot=spot,
        expiry=expiry,
        all_expiries=all_expiries,
        snapshot_ts=snapshot_ts,
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
        max_pain_strike=mp,
        vix=vix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    expiry_slug = str(expiry.date()).replace("-", "")
    out_path = output_dir / f"option_chain_viz_{expiry_slug}_{stamp}.html"
    out_path.write_text(html, encoding="utf-8")

    atm = min((r["strike"] for r in rows), key=lambda s: abs(s - spot), default=0)
    pcr = round(total_pe_oi / total_ce_oi, 3) if total_ce_oi > 0 else None
    print(f"Expiry   : {expiry.date()}")
    print(f"Spot     : {spot:,.2f}   ATM: {int(atm):,}   Max Pain: {int(mp):,}")
    print(f"PCR      : {pcr}   VIX: {vix}")
    print(f"Output   : {out_path.resolve()}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
