#!/usr/bin/env python3
"""Train an LSTM classifier for NIFTY 50 next-day direction."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yfinance as yf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

NIFTY50_INDEX_TICKER = "^NSEI"
CONTEXT_TICKERS = {
    "BANKNIFTY": "^NSEBANK",
    "USDINR": "INR=X",
    "INDIAVIX": "^INDIAVIX",
}


@dataclass
class SequenceSplit:
    x: np.ndarray
    y: np.ndarray
    signal_date: np.ndarray
    target_date: np.ndarray
    prev_close: np.ndarray
    next_close: np.ndarray
    actual_return: np.ndarray


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an LSTM classifier on NIFTY50 index direction using technical indicators "
            "and a time-based train/validation/test split."
        )
    )
    parser.add_argument("--period", default="10y", help="Data period. default: 10y")
    parser.add_argument("--interval", default="1d", help="Data interval. default: 1d")
    parser.add_argument("--lookback", type=int, default=60, help="Sequence length. default: 60")
    parser.add_argument("--train-years", type=int, default=8, help="default: 8")
    parser.add_argument("--val-years", type=int, default=1, help="default: 1")
    parser.add_argument("--test-years", type=int, default=1, help="default: 1")
    parser.add_argument("--epochs", type=int, default=80, help="default: 80")
    parser.add_argument("--batch-size", type=int, default=64, help="default: 64")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="default: 0.001")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="default: 0.0001")
    parser.add_argument("--hidden-size", type=int, default=64, help="default: 64")
    parser.add_argument("--num-layers", type=int, default=2, help="default: 2")
    parser.add_argument("--dropout", type=float, default=0.2, help="default: 0.2")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience. default: 12")
    parser.add_argument("--seed", type=int, default=42, help="default: 42")
    parser.add_argument("--output-dir", default="output", help="default: output")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore cached CSV files and download fresh market data.",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Skip walk-forward evaluation folds.",
    )
    parser.add_argument(
        "--walk-forward-max-folds",
        type=int,
        default=3,
        help="Maximum walk-forward yearly folds. default: 3",
    )
    parser.add_argument(
        "--walk-forward-epochs",
        type=int,
        default=25,
        help="Epochs per walk-forward fold. default: 25",
    )
    parser.add_argument(
        "--walk-forward-val-months",
        type=int,
        default=12,
        help="Validation window size inside each walk-forward fold. default: 12",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    frame = yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )
    if frame.empty:
        raise ValueError(
            f"No historical data returned for {ticker} "
            f"(period={period}, interval={interval})"
        )
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_ticker_data(
    ticker: str,
    label: str,
    period: str,
    interval: str,
    output_dir: Path,
    force_download: bool,
) -> tuple[pd.DataFrame, Path]:
    csv_path = output_dir / f"{label}_{period}_{interval}.csv"
    if csv_path.exists() and not force_download:
        frame = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        idx = pd.to_datetime(frame.index, utc=True)
        frame.index = idx.tz_convert("Asia/Kolkata").normalize().tz_localize(None)
        return frame.sort_index(), csv_path

    frame = fetch_history(ticker=ticker, period=period, interval=interval)
    idx = pd.to_datetime(frame.index, utc=True)
    frame.index = idx.tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    frame.to_csv(csv_path)
    return frame, csv_path


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


def add_context_columns(
    base_frame: pd.DataFrame,
    period: str,
    interval: str,
    output_dir: Path,
    force_download: bool,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = base_frame.copy()
    source_paths: dict[str, str] = {}

    for name, ticker in CONTEXT_TICKERS.items():
        try:
            context_frame, context_path = load_ticker_data(
                ticker=ticker,
                label=name,
                period=period,
                interval=interval,
                output_dir=output_dir,
                force_download=force_download,
            )
            close_col = f"{name}_Close"
            ret_col = f"{name}_Return1"
            aligned_close = context_frame["Close"].reindex(frame.index)
            frame[close_col] = aligned_close.ffill()
            frame[ret_col] = frame[close_col].pct_change()
            source_paths[name] = str(context_path.resolve())
        except ValueError as exc:
            print(f"Warning: skipping {name} context ({exc})", file=sys.stderr)

    return frame, source_paths


def engineer_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = frame.copy()
    close = df["Close"]
    volume = df["Volume"]

    # Core price/volume features.
    df["Return1"] = close.pct_change()
    df["LogReturn1"] = np.log(close).diff()
    df["HLSpread"] = (df["High"] - df["Low"]) / close.shift(1)
    df["OCSpread"] = (close - df["Open"]) / df["Open"]

    # Momentum and trend indicators.
    df["RSI14"] = compute_rsi(close, period=14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]
    df["ROC10"] = close.pct_change(periods=10)
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    df["SMA10Ratio"] = close / sma10 - 1.0
    df["SMA20Ratio"] = close / sma20 - 1.0
    df["EMA50Ratio"] = close / ema50 - 1.0
    df["EMA200Ratio"] = close / ema200 - 1.0

    # Volatility indicators.
    df["ATR14"] = compute_atr(df, period=14)
    returns = close.pct_change()
    rolling_std20 = returns.rolling(20).std()
    df["RollingStd20"] = rolling_std20
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BBPctB"] = (close - bb_lower) / (bb_upper - bb_lower)
    df["BBWidth"] = (bb_upper - bb_lower) / bb_mid

    # Volume features.
    df["VolumeChange"] = volume.pct_change()
    volume_ma20 = volume.rolling(20).mean()
    volume_std20 = volume.rolling(20).std()
    df["VolumeMA20Ratio"] = volume / volume_ma20
    df["VolumeZScore"] = (volume - volume_ma20) / volume_std20

    # Next-day target.
    df["TargetReturn"] = close.pct_change().shift(-1)
    df["TargetDirection"] = (df["TargetReturn"] > 0).astype(float)
    df["PrevClose"] = close
    df["NextClose"] = close.shift(-1)
    df["TargetDate"] = df.index.to_series().shift(-1)

    feature_cols = [
        "Return1",
        "LogReturn1",
        "HLSpread",
        "OCSpread",
        "RSI14",
        "MACD",
        "MACDSignal",
        "MACDHist",
        "ROC10",
        "SMA10Ratio",
        "SMA20Ratio",
        "EMA50Ratio",
        "EMA200Ratio",
        "ATR14",
        "RollingStd20",
        "BBPctB",
        "BBWidth",
        "VolumeChange",
        "VolumeMA20Ratio",
        "VolumeZScore",
    ]

    for context_name in CONTEXT_TICKERS:
        ret_col = f"{context_name}_Return1"
        if ret_col in df.columns:
            feature_cols.append(ret_col)

    model_frame = df[feature_cols + ["TargetDirection", "TargetReturn", "PrevClose", "NextClose", "TargetDate"]]
    model_frame = model_frame.replace([np.inf, -np.inf], np.nan).dropna()
    return model_frame, feature_cols


def build_splits_by_time(
    frame: pd.DataFrame,
    train_years: int,
    val_years: int,
    test_years: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    latest = frame.index.max()
    test_start = latest - pd.DateOffset(years=test_years)
    val_start = test_start - pd.DateOffset(years=val_years)
    train_start = val_start - pd.DateOffset(years=train_years)

    train_rows = int(((frame.index >= train_start) & (frame.index < val_start)).sum())
    val_rows = int(((frame.index >= val_start) & (frame.index < test_start)).sum())
    test_rows = int((frame.index >= test_start).sum())
    min_required = 120
    if train_rows < min_required or val_rows < min_required // 3 or test_rows < min_required // 3:
        raise ValueError(
            "Not enough rows for train/val/test split with current settings. "
            "Increase --period or reduce split durations."
        )

    return train_start, val_start, test_start


def build_sequence_splits(
    frame: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    split_label_fn: Callable[[pd.Timestamp], str],
) -> dict[str, SequenceSplit]:
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    target = frame["TargetDirection"].to_numpy(dtype=np.float32)
    target_return = frame["TargetReturn"].to_numpy(dtype=np.float32)
    prev_close = frame["PrevClose"].to_numpy(dtype=np.float32)
    next_close = frame["NextClose"].to_numpy(dtype=np.float32)
    signal_dates = frame.index.to_numpy()
    target_dates = frame["TargetDate"].to_numpy()

    num_features = features.shape[1]
    buckets: dict[str, dict[str, list]] = {
        "train": {"x": [], "y": [], "signal_date": [], "target_date": [], "prev_close": [], "next_close": [], "actual_return": []},
        "val": {"x": [], "y": [], "signal_date": [], "target_date": [], "prev_close": [], "next_close": [], "actual_return": []},
        "test": {"x": [], "y": [], "signal_date": [], "target_date": [], "prev_close": [], "next_close": [], "actual_return": []},
    }

    for end_idx in range(lookback - 1, len(frame)):
        label = split_label_fn(pd.Timestamp(signal_dates[end_idx]))
        if label not in buckets:
            continue

        start_idx = end_idx - lookback + 1
        buckets[label]["x"].append(features[start_idx : end_idx + 1])
        buckets[label]["y"].append(float(target[end_idx]))
        buckets[label]["signal_date"].append(signal_dates[end_idx])
        buckets[label]["target_date"].append(target_dates[end_idx])
        buckets[label]["prev_close"].append(float(prev_close[end_idx]))
        buckets[label]["next_close"].append(float(next_close[end_idx]))
        buckets[label]["actual_return"].append(float(target_return[end_idx]))

    splits: dict[str, SequenceSplit] = {}
    for key, data in buckets.items():
        if data["x"]:
            x_arr = np.array(data["x"], dtype=np.float32)
        else:
            x_arr = np.empty((0, lookback, num_features), dtype=np.float32)

        splits[key] = SequenceSplit(
            x=x_arr,
            y=np.array(data["y"], dtype=np.float32),
            signal_date=np.array(data["signal_date"], dtype="datetime64[ns]"),
            target_date=np.array(data["target_date"], dtype="datetime64[ns]"),
            prev_close=np.array(data["prev_close"], dtype=np.float32),
            next_close=np.array(data["next_close"], dtype=np.float32),
            actual_return=np.array(data["actual_return"], dtype=np.float32),
        )
    return splits


def standardize_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = frame.loc[train_mask, feature_cols].mean()
    stds = frame.loc[train_mask, feature_cols].std().replace(0.0, 1.0)
    standardized = frame.copy()
    standardized[feature_cols] = (standardized[feature_cols] - means) / stds
    return standardized, means, stds


def fallback_validation_split(splits: dict[str, SequenceSplit]) -> dict[str, SequenceSplit]:
    if len(splits["val"].y) > 0:
        return splits

    train_size = len(splits["train"].y)
    if train_size < 20:
        return splits

    val_take = max(1, int(train_size * 0.15))
    val_start = train_size - val_take

    splits["val"] = SequenceSplit(
        x=splits["train"].x[val_start:],
        y=splits["train"].y[val_start:],
        signal_date=splits["train"].signal_date[val_start:],
        target_date=splits["train"].target_date[val_start:],
        prev_close=splits["train"].prev_close[val_start:],
        next_close=splits["train"].next_close[val_start:],
        actual_return=splits["train"].actual_return[val_start:],
    )

    splits["train"] = SequenceSplit(
        x=splits["train"].x[:val_start],
        y=splits["train"].y[:val_start],
        signal_date=splits["train"].signal_date[:val_start],
        target_date=splits["train"].target_date[:val_start],
        prev_close=splits["train"].prev_close[:val_start],
        next_close=splits["train"].next_close[:val_start],
        actual_return=splits["train"].actual_return[:val_start],
    )
    return splits


def classification_metrics(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true_i = y_true.astype(int)
    y_pred_i = (prob_up >= threshold).astype(int)

    tp = int(((y_true_i == 1) & (y_pred_i == 1)).sum())
    tn = int(((y_true_i == 0) & (y_pred_i == 0)).sum())
    fp = int(((y_true_i == 0) & (y_pred_i == 1)).sum())
    fn = int(((y_true_i == 1) & (y_pred_i == 0)).sum())

    total = max(len(y_true_i), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    balanced_accuracy = (recall + tnr) / 2
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    brier = float(np.mean((prob_up - y_true) ** 2))

    denom = np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denom

    eps = 1e-7
    clipped = np.clip(prob_up, eps, 1 - eps)
    log_loss = float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "mcc": float(mcc),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "actual_up_rate": float(y_true.mean()),
        "predicted_up_rate": float(y_pred_i.mean()),
    }


def select_threshold(y_true: np.ndarray, prob_up: np.ndarray) -> float:
    best_threshold = 0.5
    best_score = -1.0
    best_rate_diff = float("inf")
    for threshold in np.arange(0.30, 0.71, 0.01):
        metrics = classification_metrics(y_true, prob_up, threshold=float(threshold))
        score = metrics["balanced_accuracy"]
        rate_diff = abs(metrics["predicted_up_rate"] - metrics["actual_up_rate"])
        if score > best_score or (abs(score - best_score) < 1e-8 and rate_diff < best_rate_diff):
            best_score = score
            best_rate_diff = rate_diff
            best_threshold = float(threshold)
    return best_threshold


def tensor_dataset_from_split(split: SequenceSplit) -> TensorDataset:
    x_tensor = torch.from_numpy(split.x)
    y_tensor = torch.from_numpy(split.y)
    return TensorDataset(x_tensor, y_tensor)


def evaluate_model(
    model: nn.Module,
    split: SequenceSplit,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray, dict[str, float]]:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(split.x).to(device)
        y = torch.from_numpy(split.y).to(device)
        logits = model(x)
        loss = float(criterion(logits, y).item())
        prob_up = torch.sigmoid(logits).cpu().numpy()

    metrics = classification_metrics(split.y, prob_up, threshold=threshold)
    return loss, prob_up, metrics


def train_classifier(
    train_split: SequenceSplit,
    val_split: SequenceSplit,
    input_size: int,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int | None = None,
) -> tuple[nn.Module, float]:
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    pos = float(train_split.y.sum())
    neg = float(len(train_split.y) - pos)
    if pos <= 0:
        pos_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([neg / pos], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    dataset = tensor_dataset_from_split(train_split)
    batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    wait = 0
    max_epochs = epochs if epochs is not None else args.epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_loss = running / len(dataset)
        val_loss, _, val_metrics = evaluate_model(model, val_split, criterion, device)
        if epoch == 1 or epoch == max_epochs or epoch % max(1, max_epochs // 10) == 0:
            print(
                f"Epoch {epoch:>3}/{max_epochs}: "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f}"
            )

        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def run_walk_forward(
    frame: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    if args.walk_forward_max_folds <= 0:
        return []

    years = sorted({int(ts.year) for ts in frame.index})
    if len(years) < 6:
        return []

    fold_years = years[-(args.walk_forward_max_folds + 1) : -1]
    results: list[dict[str, float]] = []

    for test_year in fold_years:
        test_start = pd.Timestamp(year=test_year, month=1, day=1, tz=frame.index.tz)
        test_end = pd.Timestamp(year=test_year + 1, month=1, day=1, tz=frame.index.tz)
        val_start = test_start - pd.DateOffset(months=args.walk_forward_val_months)

        train_mask = frame.index < val_start
        if train_mask.sum() < 252 * 3:
            continue

        standardized, _, _ = standardize_features(
            frame=frame,
            feature_cols=feature_cols,
            train_mask=train_mask,
        )

        def fold_label_fn(date_value: pd.Timestamp) -> str:
            if date_value < val_start:
                return "train"
            if date_value < test_start:
                return "val"
            if date_value < test_end:
                return "test"
            return "skip"

        splits = build_sequence_splits(
            frame=standardized,
            feature_cols=feature_cols,
            lookback=args.lookback,
            split_label_fn=fold_label_fn,
        )
        splits = fallback_validation_split(splits)

        if len(splits["train"].y) < 100 or len(splits["test"].y) < 50 or len(splits["val"].y) < 20:
            continue

        model, _ = train_classifier(
            train_split=splits["train"],
            val_split=splits["val"],
            input_size=len(feature_cols),
            args=args,
            device=device,
            epochs=args.walk_forward_epochs,
        )
        criterion = nn.BCEWithLogitsLoss()
        _, val_prob, _ = evaluate_model(model, splits["val"], criterion, device)
        threshold = select_threshold(splits["val"].y, val_prob)
        _, prob_up, metrics = evaluate_model(
            model,
            splits["test"],
            criterion,
            device,
            threshold=threshold,
        )
        fold_result = {
            "test_year": test_year,
            "samples": int(len(splits["test"].y)),
            "threshold": threshold,
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "predicted_up_rate": float(metrics["predicted_up_rate"]),
            "actual_up_rate": float(metrics["actual_up_rate"]),
            "avg_prob_up": float(np.mean(prob_up)),
        }
        print(
            f"Walk-forward {test_year}: "
            f"acc={fold_result['accuracy']:.3f} f1={fold_result['f1']:.3f} samples={fold_result['samples']}"
        )
        results.append(fold_result)

    return results


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_frame, main_source_csv = load_ticker_data(
            ticker=NIFTY50_INDEX_TICKER,
            label="NIFTY50_INDEX",
            period=args.period,
            interval=args.interval,
            output_dir=output_dir,
            force_download=args.force_download,
        )
        merged, context_sources = add_context_columns(
            base_frame=raw_frame,
            period=args.period,
            interval=args.interval,
            output_dir=output_dir,
            force_download=args.force_download,
        )
        model_frame, feature_cols = engineer_features(merged)
        train_start, val_start, test_start = build_splits_by_time(
            frame=model_frame,
            train_years=args.train_years,
            val_years=args.val_years,
            test_years=args.test_years,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    train_mask = (model_frame.index >= train_start) & (model_frame.index < val_start)
    standardized, feature_means, feature_stds = standardize_features(
        frame=model_frame,
        feature_cols=feature_cols,
        train_mask=train_mask,
    )

    def split_label_fn(date_value: pd.Timestamp) -> str:
        if date_value < train_start:
            return "skip"
        if date_value < val_start:
            return "train"
        if date_value < test_start:
            return "val"
        return "test"

    splits = build_sequence_splits(
        frame=standardized,
        feature_cols=feature_cols,
        lookback=args.lookback,
        split_label_fn=split_label_fn,
    )
    splits = fallback_validation_split(splits)

    if len(splits["train"].y) == 0 or len(splits["test"].y) == 0 or len(splits["val"].y) == 0:
        print(
            "Error: not enough sequence samples for train/val/test. "
            "Adjust --period, --lookback, or split years.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Samples -> train: {len(splits['train'].y)}, val: {len(splits['val'].y)}, "
        f"test: {len(splits['test'].y)}"
    )
    print(
        "Split dates -> "
        f"train_start: {train_start.date()}, val_start: {val_start.date()}, test_start: {test_start.date()}"
    )
    print(f"Feature count: {len(feature_cols)}")

    device = torch.device("cpu")
    model, best_val_loss = train_classifier(
        train_split=splits["train"],
        val_split=splits["val"],
        input_size=len(feature_cols),
        args=args,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_prob, _ = evaluate_model(model, splits["val"], criterion, device)
    decision_threshold = select_threshold(splits["val"].y, val_prob)
    val_metrics = classification_metrics(splits["val"].y, val_prob, threshold=decision_threshold)
    test_loss, test_prob, test_metrics = evaluate_model(
        model,
        splits["test"],
        criterion,
        device,
        threshold=decision_threshold,
    )
    test_pred = (test_prob >= decision_threshold).astype(int)

    prediction_rows = pd.DataFrame(
        {
            "SignalDate": pd.to_datetime(splits["test"].signal_date),
            "TargetDate": pd.to_datetime(splits["test"].target_date),
            "PrevClose": splits["test"].prev_close,
            "NextClose": splits["test"].next_close,
            "ActualReturn": splits["test"].actual_return,
            "ActualDirection": splits["test"].y.astype(int),
            "ProbUp": test_prob,
            "PredictedDirection": test_pred,
            "Correct": (test_pred == splits["test"].y.astype(int)).astype(int),
        }
    )

    predictions_path = output_dir / "NIFTY50_INDEX_lstm_direction_predictions.csv"
    metrics_path = output_dir / "NIFTY50_INDEX_lstm_direction_metrics.json"
    model_path = output_dir / "NIFTY50_INDEX_lstm_direction_model.pt"
    chart_path = output_dir / "NIFTY50_INDEX_lstm_direction_test_plot.html"
    walk_forward_path = output_dir / "NIFTY50_INDEX_lstm_walk_forward.json"

    prediction_rows.to_csv(predictions_path, index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "lookback": args.lookback,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "feature_cols": feature_cols,
            },
            "feature_mean": feature_means.to_dict(),
            "feature_std": feature_stds.to_dict(),
        },
        model_path,
    )

    walk_forward_results: list[dict[str, float]] = []
    if not args.no_walk_forward:
        walk_forward_results = run_walk_forward(
            frame=model_frame,
            feature_cols=feature_cols,
            args=args,
            device=device,
        )
        walk_forward_path.write_text(
            json.dumps(walk_forward_results, indent=2),
            encoding="utf-8",
        )

    metrics = {
        "ticker": NIFTY50_INDEX_TICKER,
        "period": args.period,
        "interval": args.interval,
        "lookback": args.lookback,
        "train_years": args.train_years,
        "val_years": args.val_years,
        "test_years": args.test_years,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "rows_after_feature_engineering": int(len(model_frame)),
        "split_dates": {
            "train_start": str(train_start.date()),
            "val_start": str(val_start.date()),
            "test_start": str(test_start.date()),
        },
        "sample_counts": {
            "train": int(len(splits["train"].y)),
            "val": int(len(splits["val"].y)),
            "test": int(len(splits["test"].y)),
        },
        "loss": {
            "best_val_loss": float(best_val_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
        },
        "decision_threshold": float(decision_threshold),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "source_csv": str(main_source_csv.resolve()),
        "context_sources": context_sources,
        "walk_forward_folds": walk_forward_results,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    correct_mask = prediction_rows["Correct"] == 1
    wrong_mask = prediction_rows["Correct"] == 0

    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "NIFTY 50 Close Price (test window)",
            "Model P(UP) vs Decision Threshold",
            "Prediction Outcome (correct / wrong)",
        ),
        row_heights=[0.40, 0.35, 0.25],
    )

    # Panel 1: actual close price coloured by prediction correctness.
    fig.add_trace(
        go.Scatter(
            x=prediction_rows.loc[correct_mask, "TargetDate"],
            y=prediction_rows.loc[correct_mask, "NextClose"],
            mode="markers",
            marker={"color": "#2ecc71", "size": 5, "symbol": "circle"},
            name="Correct prediction",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_rows.loc[wrong_mask, "TargetDate"],
            y=prediction_rows.loc[wrong_mask, "NextClose"],
            mode="markers",
            marker={"color": "#e74c3c", "size": 5, "symbol": "x"},
            name="Wrong prediction",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prediction_rows["TargetDate"],
            y=prediction_rows["NextClose"],
            mode="lines",
            line={"color": "#bdc3c7", "width": 1},
            name="Close price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel 2: P(UP) probability with threshold line and actual direction shading.
    fig.add_trace(
        go.Scatter(
            x=prediction_rows["TargetDate"],
            y=prediction_rows["ProbUp"],
            mode="lines",
            line={"color": "#3498db", "width": 1.5},
            name="P(UP)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=decision_threshold,
        line_dash="dash",
        line_color="#e67e22",
        annotation_text=f"threshold={decision_threshold:.2f}",
        annotation_position="top left",
        row=2,
        col=1,
    )
    # Shade regions where the market actually went up.
    up_dates = prediction_rows.loc[prediction_rows["ActualDirection"] == 1, "TargetDate"]
    for date in up_dates:
        fig.add_vrect(
            x0=date,
            x1=date,
            fillcolor="#2ecc71",
            opacity=0.15,
            line_width=0,
            row=2,
            col=1,
        )

    # Panel 3: rolling 20-day accuracy.
    rolling_correct = prediction_rows["Correct"].rolling(20, min_periods=5).mean()
    fig.add_trace(
        go.Scatter(
            x=prediction_rows["TargetDate"],
            y=rolling_correct,
            mode="lines",
            line={"color": "#9b59b6", "width": 1.5},
            name="Rolling 20-day accuracy",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="#7f8c8d",
        annotation_text="50% baseline",
        annotation_position="bottom right",
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="Index Value", row=1, col=1)
    fig.update_yaxes(title_text="P(UP)", range=[-0.05, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Rolling Acc", range=[0, 1], row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    test_acc = test_metrics["accuracy"]
    test_f1 = test_metrics["f1"]
    fig.update_layout(
        title=(
            f"NIFTY 50 LSTM — Test Window  |  "
            f"Accuracy: {test_acc:.1%}  |  F1: {test_f1:.3f}  |  "
            f"Threshold: {decision_threshold:.2f}"
        ),
        template="plotly_white",
        height=800,
        legend={"orientation": "h", "y": -0.05},
    )
    fig.write_html(chart_path, include_plotlyjs="cdn")

    print(f"Validation accuracy: {val_metrics['accuracy']:.2%}")
    print(f"Decision threshold:  {decision_threshold:.2f}")
    print(f"Test accuracy:       {test_metrics['accuracy']:.2%}")
    print(f"Test F1:             {test_metrics['f1']:.3f}")
    print(f"Predictions:         {predictions_path.resolve()}")
    print(f"Metrics:             {metrics_path.resolve()}")
    print(f"Model:               {model_path.resolve()}")
    print(f"Chart:               {chart_path.resolve()}")
    if walk_forward_results:
        avg_wf_acc = float(np.mean([x["accuracy"] for x in walk_forward_results]))
        print(f"Walk-forward avg acc: {avg_wf_acc:.2%}")
        print(f"Walk-forward file:    {walk_forward_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
