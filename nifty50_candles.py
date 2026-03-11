#!/usr/bin/env python3
"""Fetch NIFTY 50 index historical data and render a candlestick chart."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

NIFTY50_INDEX_TICKER = "^NSEI"


def fetch_history(period: str, interval: str) -> pd.DataFrame:
    frame = yf.Ticker(NIFTY50_INDEX_TICKER).history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )
    if frame.empty:
        raise ValueError(
            "No historical data returned for NIFTY50 index "
            f"(period={period}, interval={interval})"
        )
    return frame


def build_candlestick(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=frame.index,
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                name="NIFTY50 Index",
            )
        ]
    )
    fig.update_layout(
        title="NIFTY 50 Index (^NSEI) historical candlestick",
        xaxis_title="Date",
        yaxis_title="Index Value",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NIFTY50 index historical data and generate a candlestick chart."
    )
    parser.add_argument(
        "--period",
        default="6mo",
        help="yfinance period (1mo, 3mo, 6mo, 1y, 5y, max...). default: 6mo",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="yfinance interval (1d, 1h, 15m...). default: 1d",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files. default: output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        frame = fetch_history(args.period, args.interval)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"NIFTY50_INDEX_{args.period}_{args.interval}.csv"
    html_path = output_dir / f"NIFTY50_INDEX_{args.period}_{args.interval}_candles.html"

    frame.to_csv(csv_path)
    fig = build_candlestick(frame)
    fig.write_html(html_path, include_plotlyjs="cdn")

    print(f"Fetched {len(frame)} rows for {NIFTY50_INDEX_TICKER}")
    print(f"CSV:  {csv_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
