#!/usr/bin/env python3
"""NSE 200 swing screener with dead-zone labels and PnL walk-forward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - runtime optional dependency
    xgb = None

NSE200_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate an NSE 200 swing screener (3-5 day horizon) using OHLCV "
            "features, dead-zone labels, and walk-forward PnL."
        )
    )
    parser.add_argument("--period", default="5y", help="History period for yfinance. default: 5y")
    parser.add_argument("--interval", default="1d", help="Data interval. default: 1d")
    parser.add_argument("--hold-days", type=int, default=5, help="Forward return horizon. default: 5")
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.015,
        help="Drop labels with abs(forward_return) <= deadzone. default: 0.015",
    )
    parser.add_argument("--train-years", type=int, default=3, help="Walk-forward train years. default: 3")
    parser.add_argument("--val-months", type=int, default=12, help="Walk-forward validation months. default: 12")
    parser.add_argument(
        "--walk-forward-years",
        type=int,
        default=2,
        help="Number of recent yearly folds to test. default: 2",
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "logreg"],
        help="Classifier model. default: xgboost",
    )
    parser.add_argument("--n-estimators", type=int, default=350, help="XGBoost estimators. default: 350")
    parser.add_argument("--max-depth", type=int, default=4, help="XGBoost max depth. default: 4")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate. default: 0.05")
    parser.add_argument("--subsample", type=float, default=0.85, help="XGBoost subsample. default: 0.85")
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.85,
        help="XGBoost colsample_bytree. default: 0.85",
    )
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel jobs for model and downloads. default: 6")
    parser.add_argument("--top-k", type=int, default=5, help="Max picks per day. default: 5")
    parser.add_argument(
        "--capital-per-trade",
        type=float,
        default=50000,
        help="Capital allocated per trade (INR). default: 50000",
    )
    parser.add_argument(
        "--roundtrip-cost",
        type=float,
        default=0.002,
        help="Cost fraction per trade (e.g. 0.002 = 0.2%%). default: 0.002",
    )
    parser.add_argument("--rsi-min", type=float, default=40.0, help="RSI lower filter. default: 40")
    parser.add_argument("--rsi-max", type=float, default=60.0, help="RSI upper filter. default: 60")
    parser.add_argument("--threshold-min", type=float, default=0.55, help="Min prob threshold. default: 0.55")
    parser.add_argument("--threshold-max", type=float, default=0.80, help="Max prob threshold. default: 0.80")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Threshold step. default: 0.05")
    parser.add_argument(
        "--symbols-file",
        default="",
        help="Optional local CSV/TXT with symbols (column 'Symbol').",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Use only first N symbols (for fast tests). 0 means all. default: 0",
    )
    parser.add_argument(
        "--min-symbol-rows",
        type=int,
        default=400,
        help="Minimum rows required per symbol. default: 400",
    )
    parser.add_argument("--seed", type=int, default=42, help="default: 42")
    parser.add_argument("--output-dir", default="output", help="default: output")
    parser.add_argument("--force-download", action="store_true", help="Ignore cached data.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_index_to_ist_date(index_values: Iterable[object]) -> pd.DatetimeIndex | pd.Series:
    idx = pd.to_datetime(index_values, utc=True)
    if isinstance(idx, pd.Series):
        return idx.dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
    return idx.tz_convert("Asia/Kolkata").normalize().tz_localize(None)


def load_nse200_symbols(symbols_file: str) -> tuple[list[str], str]:
    if symbols_file:
        path = Path(symbols_file)
        if not path.exists():
            raise ValueError(f"symbols file not found: {symbols_file}")
        if path.suffix.lower() == ".txt":
            symbols = [line.strip().upper() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            frame = pd.read_csv(path)
            if "Symbol" not in frame.columns:
                raise ValueError("symbols file must include a 'Symbol' column")
            symbols = frame["Symbol"].astype(str).str.strip().str.upper().tolist()
        source = str(path.resolve())
    else:
        response = requests.get(
            NSE200_CSV_URL,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=25,
        )
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
        if "Symbol" not in frame.columns:
            raise ValueError("NSE200 constituent CSV did not include Symbol column")
        symbols = frame["Symbol"].astype(str).str.strip().str.upper().tolist()
        source = NSE200_CSV_URL

    symbols = sorted({sym for sym in symbols if sym and sym != "NAN"})
    tickers = [f"{sym}.NS" for sym in symbols]
    return tickers, source


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def load_or_download_universe_ohlcv(
    tickers: list[str],
    period: str,
    interval: str,
    output_dir: Path,
    force_download: bool,
) -> tuple[pd.DataFrame, Path]:
    signature = hashlib.md5(",".join(sorted(tickers)).encode("utf-8")).hexdigest()[:12]
    cache_path = output_dir / f"NSE200_OHLCV_{period}_{interval}_{len(tickers)}_{signature}.csv.gz"
    if cache_path.exists() and not force_download:
        frame = pd.read_csv(cache_path, parse_dates=["Date"])
        frame["Date"] = normalize_index_to_ist_date(frame["Date"])
        return frame.sort_values(["Symbol", "Date"]), cache_path

    rows: list[pd.DataFrame] = []
    for chunk in chunked(tickers, 40):
        raw = yf.download(
            tickers=" ".join(chunk),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            available = set(raw.columns.get_level_values(0))
            for ticker in chunk:
                if ticker not in available:
                    continue
                sub = raw[ticker].copy()
                if "Close" not in sub.columns:
                    continue
                sub = sub.dropna(subset=["Close"])
                if sub.empty:
                    continue
                sub = sub.reset_index()
                sub["Symbol"] = ticker
                rows.append(sub[["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]])
        else:
            # yfinance shape when only one symbol is downloaded.
            ticker = chunk[0]
            sub = raw.dropna(subset=["Close"]).reset_index()
            if sub.empty:
                continue
            sub["Symbol"] = ticker
            rows.append(sub[["Date", "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"]])

    if not rows:
        raise ValueError("No OHLCV data was downloaded for the symbol universe.")

    universe = pd.concat(rows, ignore_index=True)
    universe["Date"] = normalize_index_to_ist_date(universe["Date"])
    universe = universe.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    universe.to_csv(cache_path, index=False, compression="gzip")
    return universe, cache_path


def load_or_download_vix(
    period: str,
    interval: str,
    output_dir: Path,
    force_download: bool,
) -> tuple[pd.DataFrame, Path]:
    cache_path = output_dir / f"INDIAVIX_{period}_{interval}.csv"
    if cache_path.exists() and not force_download:
        frame = pd.read_csv(cache_path, parse_dates=["Date"])
        frame["Date"] = normalize_index_to_ist_date(frame["Date"])
        return frame.sort_values("Date"), cache_path

    raw = yf.Ticker("^INDIAVIX").history(period=period, interval=interval, auto_adjust=False)
    if raw.empty:
        raise ValueError("Could not download India VIX data.")
    raw.index = normalize_index_to_ist_date(raw.index)
    frame = raw.reset_index().rename(columns={"index": "Date"})[["Date", "Close"]]
    frame = frame.rename(columns={"Close": "VIX_Close"})
    frame["VIX_Return1"] = frame["VIX_Close"].pct_change()
    frame.to_csv(cache_path, index=False)
    return frame.sort_values("Date"), cache_path


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    tr = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def engineer_symbol_features(symbol_frame: pd.DataFrame, hold_days: int, deadzone: float) -> pd.DataFrame:
    df = symbol_frame.sort_values("Date").copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"]

    candle_range = (high - low).replace(0.0, np.nan)
    atr14 = compute_atr(df, period=14)
    ret3 = close.pct_change(3)
    atr_pct = atr14 / close

    df["VolumeSurge20"] = volume / volume.rolling(20).mean()
    df["ATRNormRet3"] = ret3 / atr_pct.replace(0.0, np.nan)
    df["RSI14"] = compute_rsi(close, period=14)
    df["BodyToRange"] = (close - open_).abs() / candle_range
    df["UpperWickToRange"] = (high - np.maximum(open_, close)) / candle_range
    df["LowerWickToRange"] = (np.minimum(open_, close) - low) / candle_range
    df["GapPct"] = (open_ - close.shift(1)) / close.shift(1)
    df["Proximity52WHigh"] = close / close.rolling(252).max()
    df["DistFrom52WHigh"] = 1.0 - df["Proximity52WHigh"]
    df["Return1"] = close.pct_change()
    df["Volatility20"] = df["Return1"].rolling(20).std()

    df["ForwardReturn"] = close.shift(-hold_days) / close - 1.0
    df["TargetLabel"] = np.where(
        df["ForwardReturn"] > deadzone,
        1.0,
        np.where(df["ForwardReturn"] < -deadzone, 0.0, np.nan),
    )

    keep_cols = [
        "Date",
        "Symbol",
        "Close",
        "Volume",
        "VolumeSurge20",
        "ATRNormRet3",
        "RSI14",
        "BodyToRange",
        "UpperWickToRange",
        "LowerWickToRange",
        "GapPct",
        "Proximity52WHigh",
        "DistFrom52WHigh",
        "Return1",
        "Volatility20",
        "ForwardReturn",
        "TargetLabel",
    ]
    return df[keep_cols]


def build_feature_dataset(
    ohlcv_frame: pd.DataFrame,
    vix_frame: pd.DataFrame,
    hold_days: int,
    deadzone: float,
    min_symbol_rows: int,
) -> pd.DataFrame:
    feature_parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in ohlcv_frame.groupby("Symbol", sort=False):
        if len(symbol_frame) < min_symbol_rows:
            continue
        part = engineer_symbol_features(symbol_frame, hold_days=hold_days, deadzone=deadzone)
        feature_parts.append(part)

    if not feature_parts:
        raise ValueError("No symbols had enough rows to build features.")

    data = pd.concat(feature_parts, ignore_index=True)
    data = data.merge(vix_frame[["Date", "VIX_Close", "VIX_Return1"]], on="Date", how="left")
    data[["VIX_Close", "VIX_Return1"]] = data[["VIX_Close", "VIX_Return1"]].ffill().bfill()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["ForwardReturn", "VolumeSurge20", "ATRNormRet3", "RSI14", "BodyToRange"])
    data = data.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return data


def feature_columns() -> list[str]:
    return [
        "VolumeSurge20",
        "ATRNormRet3",
        "RSI14",
        "BodyToRange",
        "UpperWickToRange",
        "LowerWickToRange",
        "GapPct",
        "Proximity52WHigh",
        "DistFrom52WHigh",
        "Return1",
        "Volatility20",
        "VIX_Close",
        "VIX_Return1",
    ]


def build_model(args: argparse.Namespace):
    if args.model == "xgboost":
        if xgb is None:
            print("Warning: xgboost is not installed, using logistic regression fallback.", file=sys.stderr)
        else:
            return xgb.XGBClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=args.seed,
                n_jobs=args.n_jobs,
                tree_method="hist",
            ), "xgboost"

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    return model, "logreg"


def predict_prob(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)
    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    tn = int(((y_true_i == 0) & (y_pred_i == 0)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())
    total = max(len(y_true_i), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def simulate_swing_trades(
    scored_frame: pd.DataFrame,
    min_prob: float,
    top_k: int,
    rsi_min: float,
    rsi_max: float,
    capital_per_trade: float,
    roundtrip_cost: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    candidates = scored_frame[
        (scored_frame["ProbUp"] >= min_prob)
        & (scored_frame["RSI14"] >= rsi_min)
        & (scored_frame["RSI14"] <= rsi_max)
    ].copy()

    if candidates.empty:
        empty_summary = {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
        }
        return candidates, empty_summary

    selected = (
        candidates.sort_values(["Date", "ProbUp"], ascending=[True, False])
        .groupby("Date", as_index=False)
        .head(top_k)
        .copy()
    )
    selected["GrossReturn"] = selected["ForwardReturn"]
    selected["NetReturn"] = selected["GrossReturn"] - roundtrip_cost
    selected["PnL"] = selected["NetReturn"] * capital_per_trade
    selected["Win"] = (selected["PnL"] > 0).astype(int)

    daily_pnl = selected.groupby("Date")["PnL"].sum().sort_index()
    equity = daily_pnl.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    summary = {
        "num_trades": int(len(selected)),
        "win_rate": float(selected["Win"].mean()),
        "avg_net_return": float(selected["NetReturn"].mean()),
        "total_pnl": float(selected["PnL"].sum()),
        "avg_pnl": float(selected["PnL"].mean()),
        "max_drawdown": max_drawdown,
    }
    return selected, summary


def select_prob_threshold(val_scored: pd.DataFrame, args: argparse.Namespace) -> tuple[float, dict[str, float]]:
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    best_threshold = float(args.threshold_min)
    best_summary: dict[str, float] | None = None

    for threshold in thresholds:
        _, summary = simulate_swing_trades(
            scored_frame=val_scored,
            min_prob=float(threshold),
            top_k=args.top_k,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max,
            capital_per_trade=args.capital_per_trade,
            roundtrip_cost=args.roundtrip_cost,
        )
        if best_summary is None:
            best_threshold = float(threshold)
            best_summary = summary
            continue
        if summary["total_pnl"] > best_summary["total_pnl"]:
            best_threshold = float(threshold)
            best_summary = summary
        elif math.isclose(summary["total_pnl"], best_summary["total_pnl"], abs_tol=1e-9):
            if summary["win_rate"] > best_summary["win_rate"]:
                best_threshold = float(threshold)
                best_summary = summary

    if best_summary is None:
        best_summary = {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_net_return": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "max_drawdown": 0.0,
        }
    return best_threshold, best_summary


def run_walk_forward_backtest(data: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, object]], pd.DataFrame]:
    cols = feature_columns()
    years = sorted(data["Date"].dt.year.unique().tolist())
    latest_date = data["Date"].max()
    latest_complete_year = latest_date.year if latest_date.month == 12 else latest_date.year - 1
    candidate_years = [year for year in years if year <= latest_complete_year]
    if len(candidate_years) < args.train_years + 1:
        raise ValueError("Not enough yearly data to run walk-forward backtest.")

    test_years = candidate_years[-args.walk_forward_years :]
    fold_results: list[dict[str, object]] = []
    all_trades: list[pd.DataFrame] = []

    for year in test_years:
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year + 1, month=1, day=1)
        val_start = test_start - pd.DateOffset(months=args.val_months)
        train_start = val_start - pd.DateOffset(years=args.train_years)

        fold = data[(data["Date"] >= train_start) & (data["Date"] < test_end)].copy()
        if fold.empty:
            continue

        train = fold[(fold["Date"] >= train_start) & (fold["Date"] < val_start)].copy()
        val = fold[(fold["Date"] >= val_start) & (fold["Date"] < test_start)].copy()
        test = fold[(fold["Date"] >= test_start) & (fold["Date"] < test_end)].copy()

        train_labeled = train.dropna(subset=["TargetLabel"])
        val_labeled = val.dropna(subset=["TargetLabel"])
        test_labeled = test.dropna(subset=["TargetLabel"])

        if len(train_labeled) < 1500 or len(val_labeled) < 300 or len(test) < 200:
            print(f"Skipping {year}: insufficient samples.", file=sys.stderr)
            continue

        model, model_name = build_model(args)
        model.fit(train_labeled[cols], train_labeled["TargetLabel"].astype(int))

        val_prob = predict_prob(model, val_labeled[cols])
        val_scored = val_labeled[["Date", "Symbol", "RSI14", "ForwardReturn"]].copy()
        val_scored["ProbUp"] = val_prob
        threshold, _ = select_prob_threshold(val_scored, args)

        test_prob = predict_prob(model, test[cols])
        test_scored = test[["Date", "Symbol", "RSI14", "ForwardReturn"]].copy()
        test_scored["ProbUp"] = test_prob
        trades, pnl_summary = simulate_swing_trades(
            scored_frame=test_scored,
            min_prob=threshold,
            top_k=args.top_k,
            rsi_min=args.rsi_min,
            rsi_max=args.rsi_max,
            capital_per_trade=args.capital_per_trade,
            roundtrip_cost=args.roundtrip_cost,
        )

        class_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
        if not test_labeled.empty:
            test_labeled_prob = predict_prob(model, test_labeled[cols])
            test_labeled_pred = (test_labeled_prob >= threshold).astype(int)
            class_metrics = classification_metrics(
                y_true=test_labeled["TargetLabel"].to_numpy(dtype=int),
                y_pred=test_labeled_pred,
            )

        fold_result: dict[str, object] = {
            "year": int(year),
            "model": model_name,
            "train_start": str(train_start.date()),
            "val_start": str(val_start.date()),
            "test_start": str(test_start.date()),
            "test_end": str((test_end - pd.Timedelta(days=1)).date()),
            "threshold": float(threshold),
            "train_samples": int(len(train_labeled)),
            "val_samples": int(len(val_labeled)),
            "test_samples": int(len(test)),
            "classified_test_samples": int(len(test_labeled)),
            "classification": class_metrics,
            "pnl": pnl_summary,
        }
        fold_results.append(fold_result)

        if not trades.empty:
            trades = trades.copy()
            trades["Year"] = year
            trades["Threshold"] = threshold
            all_trades.append(trades)

        print(
            f"Fold {year}: trades={pnl_summary['num_trades']} "
            f"win_rate={pnl_summary['win_rate']:.2%} total_pnl=₹{pnl_summary['total_pnl']:.2f} "
            f"acc={class_metrics['accuracy']:.2%}"
        )

    combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return fold_results, combined_trades


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tickers, symbol_source = load_nse200_symbols(args.symbols_file)
        if args.max_symbols > 0:
            tickers = tickers[: args.max_symbols]
        ohlcv, ohlcv_source = load_or_download_universe_ohlcv(
            tickers=tickers,
            period=args.period,
            interval=args.interval,
            output_dir=output_dir,
            force_download=args.force_download,
        )
        vix, vix_source = load_or_download_vix(
            period=args.period,
            interval=args.interval,
            output_dir=output_dir,
            force_download=args.force_download,
        )
        dataset = build_feature_dataset(
            ohlcv_frame=ohlcv,
            vix_frame=vix,
            hold_days=args.hold_days,
            deadzone=args.deadzone,
            min_symbol_rows=args.min_symbol_rows,
        )
        fold_results, trades = run_walk_forward_backtest(dataset, args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    metrics_path = output_dir / "NSE200_screener_metrics.json"
    trades_path = output_dir / "NSE200_screener_trades.csv"
    daily_path = output_dir / "NSE200_screener_daily_pnl.csv"
    chart_path = output_dir / "NSE200_screener_equity_curve.html"
    feature_path = output_dir / "NSE200_screener_feature_data.csv.gz"

    dataset.to_csv(feature_path, index=False, compression="gzip")
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
        daily_pnl = trades.groupby("Date")["PnL"].sum().sort_index().reset_index()
        daily_pnl["CumulativePnL"] = daily_pnl["PnL"].cumsum()
    else:
        trades = pd.DataFrame()
        trades.to_csv(trades_path, index=False)
        daily_pnl = pd.DataFrame(columns=["Date", "PnL", "CumulativePnL"])
    daily_pnl.to_csv(daily_path, index=False)

    fig = go.Figure()
    if not daily_pnl.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(daily_pnl["Date"]),
                y=daily_pnl["CumulativePnL"],
                mode="lines",
                name="Cumulative PnL",
            )
        )
    fig.update_layout(
        title="NSE 200 Screener Walk-Forward Equity Curve",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (INR)",
        template="plotly_white",
    )
    fig.write_html(chart_path, include_plotlyjs="cdn")

    total_trades = int(len(trades))
    total_pnl = float(trades["PnL"].sum()) if not trades.empty else 0.0
    win_rate = float((trades["PnL"] > 0).mean()) if not trades.empty else 0.0
    avg_trade = float(trades["PnL"].mean()) if not trades.empty else 0.0

    metrics = {
        "config": vars(args),
        "universe_size": len(tickers),
        "date_range": {
            "start": str(dataset["Date"].min().date()),
            "end": str(dataset["Date"].max().date()),
        },
        "sources": {
            "symbols": symbol_source,
            "ohlcv_cache": str(ohlcv_source.resolve()),
            "vix_cache": str(vix_source.resolve()),
        },
        "feature_columns": feature_columns(),
        "rows_after_features": int(len(dataset)),
        "folds": fold_results,
        "aggregate": {
            "total_trades": total_trades,
            "total_pnl_inr": total_pnl,
            "avg_trade_pnl_inr": avg_trade,
            "win_rate": win_rate,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Universe size used: {len(tickers)}")
    print(f"Rows after feature engineering: {len(dataset)}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total PnL: ₹{total_pnl:.2f}")
    print(f"Avg trade PnL: ₹{avg_trade:.2f}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Trades:  {trades_path.resolve()}")
    print(f"Daily:   {daily_path.resolve()}")
    print(f"Chart:   {chart_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
