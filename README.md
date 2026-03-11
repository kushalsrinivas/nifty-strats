# nofty

NIFTY 50 options analysis toolkit — fetch market data, run trading strategies, visualise option chains, and calculate options P&L.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
nofty/
├── backtester/              # Base strategy ABC
│   ├── __init__.py
│   └── strategy.py
├── strategies/              # 30+ signal generators
│   ├── ma_crossover.py
│   ├── momentum.py
│   ├── mean_reversion.py
│   ├── combined_strategy.py
│   ├── rsi_bb_strategy.py
│   ├── stochastic_breakout.py
│   ├── vwap_reversal.py
│   ├── supertrend_momentum.py
│   ├── keltner_squeeze.py
│   ├── williams_trend.py
│   ├── donchian_breakout.py
│   ├── trendline_strategy.py
│   ├── support_resistance.py
│   ├── sr_advanced_strategies.py
│   ├── inversion_fvg.py
│   ├── enhanced_macd.py
│   ├── nifty_trend_options.py
│   └── harmonic_patterns.py
├── output/                  # Generated data & visualisations
│
│── nifty50_candles.py           # 1. Fetch OHLCV + candlestick chart
│── fetch_nse_options_chain.py   # 2. Fetch NSE option chain
│── run_strategy_combo.py        # 3. Run strategies + weighted combo
│── combine_strategy_with_options.py  # 4. Map signals → option contracts
│── visualise_option_chain.py    # 5. Sensibull-style chain viewer
│── options_profit_calculator.py # 6. Options P&L calculator
│── train_nifty50_lstm.py        # 7. LSTM direction classifier
└── train_nse200_screener.py     # 8. NSE 200 swing screener
```

---

## Scripts

### 1. Fetch NIFTY OHLCV Data

```bash
python nifty50_candles.py --period 1y --interval 1d
```

Outputs OHLCV CSV and a Plotly candlestick chart HTML in `output/`.

### 2. Fetch NSE Option Chain

Downloads a full snapshot of all CE/PE rows across all available expiries.

```bash
python fetch_nse_options_chain.py --symbol NIFTY --segment indices
```

Outputs:

- `output/NIFTY_indices_option_chain_YYYYMMDD_HHMMSS.csv`
- `output/NIFTY_indices_expiries_YYYYMMDD_HHMMSS.csv`

Use `--segment equity` for stock options. If NSE blocks a session, rerun after a short delay.

### 3. Run Strategy Combos

```bash
# List all available strategies
python run_strategy_combo.py --list

# Run all strategies with weighted combo signal
python run_strategy_combo.py --strategies all --symbol ^NSEI --period 1y --interval 1d

# Custom mix with explicit weights
python run_strategy_combo.py \
  --strategies EnhancedMACDStrategy,InversionFVGStrategy,SRAllInOneStrategy \
  --weights 0.4,0.3,0.3 \
  --symbol ^NSEI --period 1y --interval 1d
```

Outputs signals CSV, summary JSON, and performance JSON (combined return, max drawdown, win rate, per-strategy stats).

### 4. Map Signals to Option Contracts

```bash
python combine_strategy_with_options.py \
  --signals-csv output/strategy_combo_signals_*.csv \
  --options-csv output/NIFTY_indices_option_chain_*.csv \
  --capital 30000 --lot-size 65 --max-lots 1 --use-latest-nonzero-signal
```

Outputs `strategy_options_plan_*.json` with action (`BUY_CALL`/`BUY_PUT`/`no_trade`), selected expiry, primary contract, top 5 alternatives, and premium outlay.

### 5. Option Chain Visualiser

Sensibull-style dark-theme option chain viewer with ITM/OTM colouring, OI bars, IV, breakeven, time value, max pain, and PCR.

```bash
python visualise_option_chain.py --open
python visualise_option_chain.py --expiry 17-03-2026 --strikes 40 --open
```

### 6. Options Profit Calculator

Interactive P&L calculator inspired by [optionsprofitcalculator.com](https://www.optionsprofitcalculator.com). Uses real NSE option chain data with Black-Scholes pricing.

```bash
python options_profit_calculator.py --open
python options_profit_calculator.py --expiry 17-03-2026 --open
```

Features:

- 10 strategy presets (Long Call, Bull Call Spread, Iron Condor, Butterfly, etc.)
- Up to 4 custom legs with real strikes, premiums, and IVs from NSE data
- Multi-date payoff chart (separate line per day to expiry)
- P&L heatmap table with date columns, price rows, and % change from spot
- Greeks table (Delta, Gamma, Theta, Vega) per leg and net
- Summary: max profit, max loss, breakevens, risk/reward ratio

### 7. LSTM Direction Model

Predicts next-day NIFTY direction (UP/DOWN) using technical indicators + context features (Bank Nifty, USD/INR, India VIX).

```bash
python train_nifty50_lstm.py
python train_nifty50_lstm.py --no-walk-forward  # faster
```

### 8. NSE 200 Swing Screener

Stock-level swing signals across the NSE 200 universe with walk-forward evaluation on PnL.

```bash
python train_nse200_screener.py
python train_nse200_screener.py --max-symbols 40 --period 3y  # smoke test
```

---

## Dependencies

See `requirements.txt`. Core: pandas, numpy, scipy, plotly, requests, yfinance, scikit-learn, torch, xgboost.
# nifty-strats
