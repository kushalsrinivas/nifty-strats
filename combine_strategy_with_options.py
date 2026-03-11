#!/usr/bin/env python3
"""Map strategy combo signals to tradable NIFTY option candidates."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine strategy combo signals with an NSE option-chain snapshot "
            "to produce a trade plan JSON."
        )
    )
    parser.add_argument(
        "--signals-csv",
        required=True,
        help="Path to strategy_combo_signals_*.csv",
    )
    parser.add_argument(
        "--options-csv",
        required=True,
        help="Path to NIFTY_indices_option_chain_*.csv",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=30000.0,
        help="Trading capital in INR. Default: 30000",
    )
    parser.add_argument(
        "--lot-size",
        type=int,
        default=65,
        help="NIFTY lot size. Default: 65",
    )
    parser.add_argument(
        "--max-lots",
        type=int,
        default=1,
        help="Hard cap on lots to buy. Default: 1",
    )
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=6.0,
        help="Risk budget per trade as percent of capital. Default: 6",
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=25.0,
        help="Assumed option premium stop loss percent. Default: 25",
    )
    parser.add_argument(
        "--expiry-rank",
        type=int,
        default=0,
        help="0=nearest expiry, 1=next expiry, etc. Default: 0",
    )
    parser.add_argument(
        "--use-latest-nonzero-signal",
        action="store_true",
        help="Use latest non-zero combined signal if current bar is hold.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for JSON plan. Default: output",
    )
    parser.add_argument(
        "--vix-csv",
        default=None,
        help=(
            "Path to India VIX CSV (e.g. output/INDIAVIX_3y_1d.csv). "
            "If supplied, VIX is read and checked against --vix-warn-above "
            "and --vix-block-above thresholds."
        ),
    )
    parser.add_argument(
        "--vix-warn-above",
        type=float,
        default=18.0,
        help=(
            "India VIX above this level adds a vix_warning to the plan JSON. "
            "Premiums are likely inflated. Default: 18.0"
        ),
    )
    parser.add_argument(
        "--vix-block-above",
        type=float,
        default=None,
        help=(
            "If set and India VIX exceeds this value, the plan is aborted with "
            "status=vix_blocked. Leave unset to only warn (recommended for daily use)."
        ),
    )
    return parser.parse_args()


def load_signals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "combined_signal" not in frame.columns:
        raise ValueError("signals CSV missing required column: combined_signal")
    if "Close" not in frame.columns:
        raise ValueError("signals CSV missing required column: Close")
    return frame


def pick_signal(signal_frame: pd.DataFrame, use_latest_nonzero: bool) -> tuple[int, int]:
    latest_signal = int(signal_frame["combined_signal"].iloc[-1])
    row_index = int(len(signal_frame) - 1)

    if latest_signal != 0:
        return latest_signal, row_index

    if not use_latest_nonzero:
        return latest_signal, row_index

    nz = signal_frame.index[signal_frame["combined_signal"] != 0].tolist()
    if not nz:
        return 0, row_index
    row_index = int(nz[-1])
    return int(signal_frame.loc[row_index, "combined_signal"]), row_index


def load_options(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = [
        "contract_expiry",
        "option_type",
        "strike_price",
        "last_price",
        "best_bid_price",
        "best_ask_price",
        "volume",
        "open_interest",
        "identifier",
        "underlying_value",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            "options CSV missing required columns: " + ", ".join(missing)
        )

    frame = frame.copy()
    frame["contract_expiry_dt"] = pd.to_datetime(
        frame["contract_expiry"], dayfirst=True, errors="coerce"
    )
    frame = frame.dropna(subset=["contract_expiry_dt", "identifier"])
    frame["last_price"] = pd.to_numeric(frame["last_price"], errors="coerce")
    frame["best_bid_price"] = pd.to_numeric(frame["best_bid_price"], errors="coerce")
    frame["best_ask_price"] = pd.to_numeric(frame["best_ask_price"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    frame["open_interest"] = pd.to_numeric(frame["open_interest"], errors="coerce").fillna(0)
    frame["strike_price"] = pd.to_numeric(frame["strike_price"], errors="coerce")
    frame["underlying_value"] = pd.to_numeric(frame["underlying_value"], errors="coerce")
    frame = frame.dropna(subset=["strike_price", "last_price"])
    return frame


def load_vix(path: Path) -> float | None:
    """
    Load India VIX CSV and return the most recent closing VIX value.

    The CSV is expected to have a 'Close' column (case-insensitive).
    Returns None if the file cannot be parsed or contains no data.
    """
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        # Accept "Close", "VIX_Close", or any column whose lowercased name ends
        # with "close". Fall back to the first numeric column.
        close_col = next(
            (c for c in df.columns if c.lower().endswith("close")),
            None,
        )
        if close_col is None:
            # Last resort: first column that can be parsed as numeric
            for c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                if not s.empty:
                    close_col = c
                    break
        if close_col is None:
            return None
        series = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])
    except Exception:
        return None


def vix_assessment(
    vix_value: float | None,
    warn_above: float,
    block_above: float | None,
) -> dict:
    """
    Return a VIX context dict to embed in the plan JSON.

    Fields:
        vix_value        : latest VIX reading (None if unavailable)
        vix_warning      : True if vix_value > warn_above
        vix_blocked      : True if vix_value > block_above (and block_above is set)
        vix_note         : human-readable explanation
    """
    if vix_value is None:
        return {
            "vix_value": None,
            "vix_warning": False,
            "vix_blocked": False,
            "vix_note": "VIX CSV not provided or could not be parsed.",
        }

    warning = vix_value > warn_above
    blocked = bool(block_above is not None and vix_value > block_above)

    if blocked:
        note = (
            f"India VIX = {vix_value:.2f} exceeds block threshold {block_above:.2f}. "
            "Option premiums are severely inflated — trade blocked."
        )
    elif warning:
        note = (
            f"India VIX = {vix_value:.2f} exceeds warning threshold {warn_above:.2f}. "
            "Option premiums are elevated; even a correct direction call may lose money. "
            "Consider reducing position size or waiting for VIX to cool below 18."
        )
    else:
        note = (
            f"India VIX = {vix_value:.2f} is within normal range (below {warn_above:.2f}). "
            "Premium environment is acceptable."
        )

    return {
        "vix_value": round(vix_value, 4),
        "vix_warning": warning,
        "vix_blocked": blocked,
        "vix_note": note,
    }


def choose_expiry(frame: pd.DataFrame, rank: int) -> pd.Timestamp:
    expiries = sorted(frame["contract_expiry_dt"].dropna().unique().tolist())
    if not expiries:
        raise ValueError("No valid expiries found in options CSV.")

    today = pd.Timestamp(datetime.now().date())
    future_expiries = [exp for exp in expiries if exp >= today]
    active = future_expiries if future_expiries else expiries
    rank = max(rank, 0)
    if rank >= len(active):
        rank = len(active) - 1
    return active[rank]


def option_side_from_signal(signal: int) -> str | None:
    if signal > 0:
        return "CE"
    if signal < 0:
        return "PE"
    return None


def estimate_entry_price(row: pd.Series) -> float:
    ask = float(row.get("best_ask_price", np.nan))
    bid = float(row.get("best_bid_price", np.nan))
    last = float(row.get("last_price", np.nan))

    if math.isfinite(ask) and ask > 0:
        return ask
    if math.isfinite(last) and last > 0:
        return last
    if math.isfinite(bid) and bid > 0:
        return bid
    return float("nan")


def rank_candidates(
    frame: pd.DataFrame,
    side: str,
    expiry: pd.Timestamp,
    spot: float,
) -> pd.DataFrame:
    subset = frame[
        (frame["option_type"] == side) & (frame["contract_expiry_dt"] == expiry)
    ].copy()
    if subset.empty:
        return subset

    subset["entry_price"] = subset.apply(estimate_entry_price, axis=1)
    subset = subset[np.isfinite(subset["entry_price"])]
    subset = subset[subset["entry_price"] > 0]
    if subset.empty:
        return subset

    subset["mid_price"] = (
        subset["best_bid_price"].fillna(0) + subset["best_ask_price"].fillna(0)
    ) / 2.0
    subset["spread_pct"] = np.where(
        subset["mid_price"] > 0,
        (subset["best_ask_price"] - subset["best_bid_price"]) / subset["mid_price"],
        10.0,
    )
    subset["strike_dist_pct"] = (subset["strike_price"] - spot).abs() / max(abs(spot), 1e-9)

    # Prefer near-ATM, tighter spread, then higher liquidity.
    subset.sort_values(
        by=["strike_dist_pct", "spread_pct", "volume", "open_interest"],
        ascending=[True, True, False, False],
        inplace=True,
    )
    return subset


def lot_plan(
    entry_price: float,
    lot_size: int,
    capital: float,
    max_lots: int,
    risk_per_trade_pct: float,
    stop_loss_pct: float,
) -> dict[str, Any]:
    premium_per_lot = entry_price * lot_size
    affordable_lots = int(capital // premium_per_lot) if premium_per_lot > 0 else 0

    risk_budget = capital * (risk_per_trade_pct / 100.0)
    per_lot_risk = premium_per_lot * (stop_loss_pct / 100.0)
    risk_lots = int(risk_budget // per_lot_risk) if per_lot_risk > 0 else 0

    chosen_lots = max(0, min(affordable_lots, risk_lots, max_lots))
    return {
        "entry_price": round(float(entry_price), 4),
        "premium_per_lot_inr": round(float(premium_per_lot), 2),
        "affordable_lots_by_capital": affordable_lots,
        "affordable_lots_by_risk": risk_lots,
        "chosen_lots": chosen_lots,
        "notional_premium_outlay_inr": round(float(chosen_lots * premium_per_lot), 2),
        "risk_budget_inr": round(float(risk_budget), 2),
        "assumed_risk_per_lot_inr": round(float(per_lot_risk), 2),
        "is_tradable_under_constraints": bool(chosen_lots >= 1),
    }


def row_to_json(row: pd.Series, lot_details: dict[str, Any]) -> dict[str, Any]:
    return {
        "identifier": str(row["identifier"]),
        "option_type": str(row["option_type"]),
        "contract_expiry": row["contract_expiry_dt"].strftime("%Y-%m-%d"),
        "strike_price": float(row["strike_price"]),
        "last_price": float(row["last_price"]),
        "best_bid_price": float(row["best_bid_price"]) if pd.notna(row["best_bid_price"]) else None,
        "best_ask_price": float(row["best_ask_price"]) if pd.notna(row["best_ask_price"]) else None,
        "spread_pct": round(float(row["spread_pct"] * 100.0), 4),
        "volume": int(row["volume"]),
        "open_interest": int(row["open_interest"]),
        **lot_details,
    }


def main() -> int:
    args = parse_args()
    signals_path = Path(args.signals_csv)
    options_path = Path(args.options_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- VIX guard --------------------------------------------------------
    vix_value: float | None = None
    if args.vix_csv is not None:
        vix_value = load_vix(Path(args.vix_csv))

    vix_ctx = vix_assessment(
        vix_value=vix_value,
        warn_above=args.vix_warn_above,
        block_above=args.vix_block_above,
    )

    if vix_ctx["vix_blocked"]:
        payload = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "vix_blocked",
            "reason": vix_ctx["vix_note"],
            "vix_context": vix_ctx,
            "inputs": {
                "signals_csv": str(signals_path.resolve()),
                "options_csv": str(options_path.resolve()),
                "vix_csv": str(Path(args.vix_csv).resolve()) if args.vix_csv else None,
            },
        }
        out_path = output_dir / f"strategy_options_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Trade BLOCKED — {vix_ctx['vix_note']}")
        print(f"Plan JSON: {out_path.resolve()}")
        return 1

    if vix_ctx["vix_warning"]:
        print(f"[VIX WARNING] {vix_ctx['vix_note']}")
    # -----------------------------------------------------------------------

    signals = load_signals(signals_path)
    signal, signal_row_idx = pick_signal(
        signals, use_latest_nonzero=args.use_latest_nonzero_signal
    )
    close_price = float(signals.loc[signal_row_idx, "Close"])

    option_side = option_side_from_signal(signal)
    if option_side is None:
        payload = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "no_trade",
            "reason": "combined_signal is HOLD (0).",
            "latest_signal_row_index": signal_row_idx,
            "close_price": close_price,
            "vix_context": vix_ctx,
            "inputs": {
                "signals_csv": str(signals_path.resolve()),
                "options_csv": str(options_path.resolve()),
                "vix_csv": str(Path(args.vix_csv).resolve()) if args.vix_csv else None,
            },
        }
        out_path = output_dir / f"strategy_options_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"No trade signal. Plan JSON: {out_path.resolve()}")
        return 0

    options = load_options(options_path)
    expiry = choose_expiry(options, rank=args.expiry_rank)

    # Prefer live underlying from options snapshot; fallback to signal close.
    uv = options["underlying_value"].replace(0, np.nan).dropna()
    snapshot_spot = float(uv.median()) if not uv.empty else close_price
    spot = snapshot_spot
    warnings: list[str] = []
    if close_price > 0:
        spot_gap_pct = abs(snapshot_spot - close_price) / close_price * 100.0
        if spot_gap_pct > 2.5:
            warnings.append(
                "Signal close and options snapshot spot differ materially; "
                "selection switched to signal close for moneyness ranking."
            )
            spot = close_price

    ranked = rank_candidates(
        frame=options,
        side=option_side,
        expiry=expiry,
        spot=spot,
    )
    if ranked.empty:
        raise ValueError("No tradable candidates found for the selected signal and expiry.")

    ranked = ranked.copy()
    tradable_flags: list[bool] = []
    for _, row in ranked.iterrows():
        row_plan = lot_plan(
            entry_price=float(row["entry_price"]),
            lot_size=args.lot_size,
            capital=args.capital,
            max_lots=args.max_lots,
            risk_per_trade_pct=args.risk_per_trade_pct,
            stop_loss_pct=args.stop_loss_pct,
        )
        tradable_flags.append(bool(row_plan["is_tradable_under_constraints"]))
    ranked["is_tradable_under_constraints"] = tradable_flags

    tradable = ranked[ranked["is_tradable_under_constraints"]]
    if tradable.empty:
        warnings.append(
            "No option candidate satisfies current capital+risk constraints; "
            "showing best-ranked contract anyway."
        )
        best = ranked.iloc[0]
    else:
        best = tradable.iloc[0]

    best_lot_plan = lot_plan(
        entry_price=float(best["entry_price"]),
        lot_size=args.lot_size,
        capital=args.capital,
        max_lots=args.max_lots,
        risk_per_trade_pct=args.risk_per_trade_pct,
        stop_loss_pct=args.stop_loss_pct,
    )

    alt_source = tradable if not tradable.empty else ranked
    alternatives: list[dict[str, Any]] = []
    for _, row in alt_source.iterrows():
        if str(row["identifier"]) == str(best["identifier"]):
            continue
        alt_lot_plan = lot_plan(
            entry_price=float(row["entry_price"]),
            lot_size=args.lot_size,
            capital=args.capital,
            max_lots=args.max_lots,
            risk_per_trade_pct=args.risk_per_trade_pct,
            stop_loss_pct=args.stop_loss_pct,
        )
        alternatives.append(row_to_json(row, alt_lot_plan))
        if len(alternatives) >= 5:
            break

    action = "BUY_CALL" if option_side == "CE" else "BUY_PUT"
    payload = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "trade_candidate_generated",
        "signal_context": {
            "selected_signal": int(signal),
            "action": action,
            "latest_signal_row_index": signal_row_idx,
            "close_price_from_signal_csv": round(close_price, 4),
            "spot_from_options_snapshot": round(snapshot_spot, 4),
            "spot_used_for_selection": round(spot, 4),
        },
        "risk_context": {
            "capital_inr": args.capital,
            "lot_size": args.lot_size,
            "max_lots": args.max_lots,
            "risk_per_trade_pct": args.risk_per_trade_pct,
            "assumed_stop_loss_pct": args.stop_loss_pct,
        },
        "vix_context": vix_ctx,
        "selected_expiry": expiry.strftime("%Y-%m-%d"),
        "primary_candidate": row_to_json(best, best_lot_plan),
        "alternatives_top5": alternatives,
        "inputs": {
            "signals_csv": str(signals_path.resolve()),
            "options_csv": str(options_path.resolve()),
            "vix_csv": str(Path(args.vix_csv).resolve()) if args.vix_csv else None,
        },
        "notes": (
            "This is a contract selection layer, not a historical options backtest. "
            "It maps direction signal -> liquid option candidate using current snapshot."
        ),
        "warnings": warnings,
    }

    out_path = output_dir / f"strategy_options_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Action: {action}")
    print(f"Selected contract: {best['identifier']}")
    print(f"Plan JSON: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
