#!/usr/bin/env python3
"""Run migrated strategies individually and as a weighted combination."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import strategies as strategies_pkg
from backtester.strategy import Strategy

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run migrated strategy signals and build a weighted combo signal."
        )
    )
    parser.add_argument(
        "--strategies",
        default="all",
        help=(
            "Comma-separated strategy class names to run, or 'all'. "
            "Use --list to see names."
        ),
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated weights matching --strategies order.",
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.2,
        help=(
            "Threshold for combined score to emit signal. "
            ">= threshold => buy, <= -threshold => sell. Default: 0.2"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any strategy throws an exception.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available migrated strategy class names and exit.",
    )
    parser.add_argument(
        "--data-csv",
        default=None,
        help=(
            "Path to OHLCV CSV. If omitted, data is fetched via yfinance "
            "using --symbol/--period/--interval."
        ),
    )
    parser.add_argument(
        "--symbol",
        default="^NSEI",
        help="yfinance symbol when --data-csv is not supplied. Default: ^NSEI",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="yfinance period. Default: 1y",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval. Default: 1d",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for CSV/JSON. Default: output",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100000.0,
        help="Initial capital for return simulation. Default: 100000",
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=0.0,
        help="Transaction cost in basis points per unit turnover. Default: 0",
    )
    return parser.parse_args()


def build_strategy_registry() -> dict[str, type[Strategy]]:
    registry: dict[str, type[Strategy]] = {}
    for symbol_name in getattr(strategies_pkg, "__all__", []):
        symbol_obj = getattr(strategies_pkg, symbol_name, None)
        if (
            inspect.isclass(symbol_obj)
            and issubclass(symbol_obj, Strategy)
            and symbol_obj is not Strategy
        ):
            registry[symbol_name] = symbol_obj
    return dict(sorted(registry.items(), key=lambda item: item[0]))


def parse_strategy_selection(
    strategies_arg: str,
    registry: dict[str, type[Strategy]],
) -> list[str]:
    if strategies_arg.strip().lower() == "all":
        return list(registry.keys())

    selected = [part.strip() for part in strategies_arg.split(",") if part.strip()]
    unknown = [name for name in selected if name not in registry]
    if unknown:
        raise ValueError(
            "Unknown strategy names: "
            + ", ".join(unknown)
            + ". Use --list to view supported names."
        )
    if not selected:
        raise ValueError("No strategies selected.")
    return selected


def parse_weights(weights_arg: str | None, strategy_count: int) -> np.ndarray:
    if strategy_count <= 0:
        raise ValueError("At least one strategy is required.")

    if not weights_arg:
        return np.full(strategy_count, 1.0 / strategy_count)

    raw = [part.strip() for part in weights_arg.split(",") if part.strip()]
    if len(raw) != strategy_count:
        raise ValueError(
            f"Expected {strategy_count} weights, got {len(raw)}."
        )

    values = np.array([float(part) for part in raw], dtype=float)
    if np.any(values < 0):
        raise ValueError("Weights must be non-negative.")

    total = float(values.sum())
    if total == 0:
        raise ValueError("At least one weight must be > 0.")

    return values / total


def load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_csv:
        csv_path = Path(args.data_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame = parse_datetime_index(frame)
    else:
        frame = yf.Ticker(args.symbol).history(
            period=args.period,
            interval=args.interval,
            auto_adjust=False,
        )

    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing)
        )

    frame = frame[REQUIRED_COLUMNS].copy()
    frame = frame.sort_index()
    if frame.empty:
        raise ValueError("No market data rows loaded.")
    return frame


def parse_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame

    date_candidates = ["Date", "Datetime", "Timestamp", "date", "datetime", "timestamp"]
    index_col = next((col for col in date_candidates if col in frame.columns), None)

    if index_col is None:
        index_col = frame.columns[0]

    parsed = pd.to_datetime(frame[index_col], errors="coerce")
    if parsed.isna().all():
        raise ValueError(
            "Could not parse a datetime index from CSV. "
            "Include a Date/Datetime/Timestamp column."
        )

    output = frame.copy()
    output.index = parsed
    if index_col in output.columns:
        output.drop(columns=[index_col], inplace=True)
    output = output[~output.index.isna()]
    return output


def run_single_strategy(
    strategy_cls: type[Strategy],
    data: pd.DataFrame,
) -> pd.Series:
    instance = strategy_cls()
    signals = instance.generate_signals(data.copy())
    if not isinstance(signals, pd.DataFrame) or "signal" not in signals.columns:
        raise ValueError(
            f"{strategy_cls.__name__} must return a DataFrame with a 'signal' column."
        )

    series = pd.to_numeric(signals["signal"], errors="coerce")
    series = series.reindex(data.index).fillna(0.0)
    return np.sign(series).astype(int)


def summarize_signal_counts(signal_series: pd.Series) -> dict[str, int]:
    counts = signal_series.value_counts().to_dict()
    return {
        "buy": int(counts.get(1, 0)),
        "sell": int(counts.get(-1, 0)),
        "hold": int(counts.get(0, 0)),
    }


def signals_to_position(signal_series: pd.Series) -> pd.Series:
    raw_position = signal_series.replace(0, np.nan).ffill().fillna(0.0)
    # Execute on next bar to avoid look-ahead bias.
    return raw_position.shift(1).fillna(0.0)


def evaluate_signal_performance(
    signal_series: pd.Series,
    close_series: pd.Series,
    initial_capital: float,
    fee_bps: float,
) -> tuple[dict[str, Any], pd.Series, pd.Series, pd.Series]:
    position = signals_to_position(signal_series)
    close_returns = close_series.pct_change().fillna(0.0)
    gross_returns = position * close_returns

    fee_rate = fee_bps / 10000.0
    turnover = position.diff().abs().fillna(position.abs())
    fee_returns = turnover * fee_rate
    net_returns = gross_returns - fee_returns

    equity_curve = initial_capital * (1.0 + net_returns).cumprod()
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)

    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    days = int((close_series.index[-1] - close_series.index[0]).days) if len(close_series) > 1 else 0
    annualized_return = None
    if days > 0 and (1.0 + total_return) > 0:
        annualized_return = float((1.0 + total_return) ** (365.25 / days) - 1.0)

    active_mask = position != 0
    active_bars = int(active_mask.sum())
    win_rate = None
    if active_bars > 0:
        win_rate = float((net_returns[active_mask] > 0).mean())

    metrics = {
        "initial_capital": float(initial_capital),
        "ending_capital": float(equity_curve.iloc[-1]),
        "total_return_pct": round(total_return * 100.0, 4),
        "annualized_return_pct": round(annualized_return * 100.0, 4) if annualized_return is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100.0, 4),
        "trade_count": int((turnover > 0).sum()),
        "active_bar_ratio_pct": round((active_bars / len(position)) * 100.0, 4) if len(position) else 0.0,
        "win_rate_on_active_bars_pct": round(win_rate * 100.0, 4) if win_rate is not None else None,
        "gross_return_pct": round(float((initial_capital * (1.0 + gross_returns).cumprod().iloc[-1] / initial_capital - 1.0) * 100.0), 4),
        "fees_paid_pct_of_capital": round(float(fee_returns.sum() * 100.0), 4),
    }
    return metrics, position, net_returns, equity_curve


def main() -> int:
    args = parse_args()
    registry = build_strategy_registry()

    if args.list:
        if not registry:
            print("No strategies found.")
            return 1
        print("Available migrated strategy classes:")
        for name in registry:
            print(f"- {name}")
        return 0

    try:
        selected_names = parse_strategy_selection(args.strategies, registry)
        weights = parse_weights(args.weights, len(selected_names))
        data = load_market_data(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_frame = pd.DataFrame(index=data.index)
    per_strategy_summary: dict[str, Any] = {}
    active_names: list[str] = []
    active_weights: list[float] = []

    for idx, name in enumerate(selected_names):
        strategy_cls = registry[name]
        try:
            signal_series = run_single_strategy(strategy_cls=strategy_cls, data=data)
        except Exception as exc:  # noqa: BLE001
            if args.strict:
                print(f"Error in strategy {name}: {exc}", file=sys.stderr)
                return 1
            print(f"Warning: skipping {name} ({exc})", file=sys.stderr)
            continue

        signal_frame[name] = signal_series
        active_names.append(name)
        active_weights.append(float(weights[idx]))
        per_strategy_summary[name] = summarize_signal_counts(signal_series)

    if signal_frame.empty:
        print("Error: no strategy outputs available after execution.", file=sys.stderr)
        return 1

    weights_vector = np.array(active_weights, dtype=float)
    weights_vector = weights_vector / weights_vector.sum()
    combined_score = signal_frame[active_names].to_numpy(dtype=float).dot(weights_vector)

    combined_signal = np.where(
        combined_score >= args.vote_threshold,
        1,
        np.where(combined_score <= -args.vote_threshold, -1, 0),
    )

    signal_frame["combined_score"] = combined_score
    signal_frame["combined_signal"] = combined_signal.astype(int)
    signal_frame["Close"] = data["Close"]

    per_strategy_performance: dict[str, Any] = {}
    close_series = data["Close"]
    for name in active_names:
        perf, position, strategy_returns, _ = evaluate_signal_performance(
            signal_series=signal_frame[name],
            close_series=close_series,
            initial_capital=args.initial_capital,
            fee_bps=args.fee_bps,
        )
        per_strategy_performance[name] = perf
        signal_frame[f"{name}_position"] = position
        signal_frame[f"{name}_ret"] = strategy_returns

    combined_perf, combined_position, combined_returns, combined_equity = evaluate_signal_performance(
        signal_series=signal_frame["combined_signal"],
        close_series=close_series,
        initial_capital=args.initial_capital,
        fee_bps=args.fee_bps,
    )
    signal_frame["combined_position"] = combined_position
    signal_frame["combined_ret"] = combined_returns
    signal_frame["combined_equity"] = combined_equity

    buy_hold_returns = close_series.pct_change().fillna(0.0)
    buy_hold_equity = args.initial_capital * (1.0 + buy_hold_returns).cumprod()
    buy_hold_return_pct = float((buy_hold_equity.iloc[-1] / args.initial_capital - 1.0) * 100.0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    signals_csv = output_dir / f"strategy_combo_signals_{timestamp}.csv"
    summary_json = output_dir / f"strategy_combo_summary_{timestamp}.json"
    performance_json = output_dir / f"strategy_combo_performance_{timestamp}.json"
    signal_frame.to_csv(signals_csv)

    summary_payload = {
        "run_timestamp": timestamp,
        "data_source": str(args.data_csv) if args.data_csv else "yfinance",
        "symbol": args.symbol if not args.data_csv else None,
        "period": args.period if not args.data_csv else None,
        "interval": args.interval if not args.data_csv else None,
        "rows": int(len(signal_frame)),
        "selected_strategies": active_names,
        "weights": {name: float(weight) for name, weight in zip(active_names, weights_vector)},
        "vote_threshold": args.vote_threshold,
        "initial_capital": args.initial_capital,
        "fee_bps": args.fee_bps,
        "combined_signal_counts": summarize_signal_counts(signal_frame["combined_signal"]),
        "per_strategy_signal_counts": per_strategy_summary,
        "signals_csv": str(signals_csv.resolve()),
        "performance_json": str(performance_json.resolve()),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    performance_payload = {
        "run_timestamp": timestamp,
        "symbol": args.symbol if not args.data_csv else None,
        "rows": int(len(signal_frame)),
        "initial_capital": args.initial_capital,
        "fee_bps": args.fee_bps,
        "combined_performance": combined_perf,
        "buy_and_hold_return_pct": round(buy_hold_return_pct, 4),
        "per_strategy_performance": per_strategy_performance,
        "notes": (
            "Returns use close-to-close bars with 1-bar signal execution delay; "
            "positions are carried forward until opposite signal."
        ),
    }
    performance_json.write_text(json.dumps(performance_payload, indent=2), encoding="utf-8")

    print(f"Ran {len(active_names)} strategies.")
    print(f"Signals CSV: {signals_csv.resolve()}")
    print(f"Summary JSON: {summary_json.resolve()}")
    print(f"Performance JSON: {performance_json.resolve()}")
    print(
        "Combined signal counts: "
        + json.dumps(summary_payload["combined_signal_counts"])
    )
    print(
        "Combined return (%): "
        + str(performance_payload["combined_performance"]["total_return_pct"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
