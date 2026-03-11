"""
NIFTY Trend Options Strategy

A mechanical daily strategy for NIFTY 50 index options based on:
  Trend (20/50 EMA cross) → Strength (ADX) → Momentum (MACD) → Timing (RSI)

Entry Rules
-----------
Bullish (BUY CALL signal = 1):
  - 20 EMA > 50 EMA AND Close > both EMAs
  - ADX > adx_threshold (25)
  - MACD line > Signal line AND histogram increasing
  - RSI in [rsi_buy_min, rsi_buy_max] (40–60)

Bearish (BUY PUT signal = -1):
  - 20 EMA < 50 EMA AND Close < both EMAs
  - ADX > adx_threshold (25)
  - MACD line < Signal line AND histogram decreasing
  - RSI in [rsi_sell_min, rsi_sell_max] (40–60)

No-Trade Zone:
  - ADX < adx_no_trade_below (20)

Exit Rules (signal resets to 0):
  - MACD crosses against position
  - Close crosses 20 EMA against position
  - RSI hits extreme (>70 for longs, <30 for shorts)

Classes
-------
NiftyTrendOptionsStrategy   — all 4 conditions must pass (min_indicators=4)
StrictNiftyTrendStrategy    — identical alias with explicit "no compromise" label
RelaxedNiftyTrendStrategy   — 3-of-4 conditions (min_indicators=3) for sensitivity testing
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


# ---------------------------------------------------------------------------
# Indicator helpers (no external dependencies beyond pandas/numpy)
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    """Return (macd_line, signal_line, histogram) as pd.Series."""
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-smoothed ADX calculation (no external libraries required).

    Steps:
      1. True Range (TR)
      2. Directional Movement: +DM, -DM
      3. Wilder smooth all three over `period`
      4. DI+ = 100 * smoothed_+DM / smoothed_TR
         DI- = 100 * smoothed_-DM / smoothed_TR
      5. DX  = 100 * |DI+ - DI-| / (DI+ + DI-)
      6. ADX = Wilder smooth of DX over `period`
    """
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    # Wilder smoothing: first value is sum of first `period` values, then rolling
    def wilder_smooth(s: pd.Series, n: int) -> pd.Series:
        result = pd.Series(np.nan, index=s.index)
        # seed with simple sum for the first window
        first_valid = s.first_valid_index()
        if first_valid is None:
            return result
        start_idx = s.index.get_loc(first_valid)
        if start_idx + n > len(s):
            return result
        result.iloc[start_idx + n - 1] = s.iloc[start_idx: start_idx + n].sum()
        for i in range(start_idx + n, len(s)):
            result.iloc[i] = result.iloc[i - 1] - (result.iloc[i - 1] / n) + s.iloc[i]
        return result

    smoothed_tr = wilder_smooth(tr, period)
    smoothed_plus = wilder_smooth(plus_dm_s, period)
    smoothed_minus = wilder_smooth(minus_dm_s, period)

    di_plus = 100 * smoothed_plus / smoothed_tr.replace(0, np.nan)
    di_minus = 100 * smoothed_minus / smoothed_tr.replace(0, np.nan)

    di_sum = di_plus + di_minus
    dx = 100 * (di_plus - di_minus).abs() / di_sum.replace(0, np.nan)
    adx = wilder_smooth(dx.fillna(0), period)

    return adx


# ---------------------------------------------------------------------------
# Base Strategy
# ---------------------------------------------------------------------------

class NiftyTrendOptionsStrategy(Strategy):
    """
    Mechanical NIFTY 50 options strategy: EMA cross + ADX + MACD + RSI.

    Signal = 1  → buy ATM/1-ITM CALL option
    Signal = -1 → buy ATM/1-ITM PUT option
    Signal = 0  → no position / exit existing position
    """

    def __init__(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        adx_no_trade_below: float = 20.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        rsi_period: int = 14,
        rsi_buy_min: float = 40.0,
        rsi_buy_max: float = 60.0,
        rsi_sell_min: float = 40.0,
        rsi_sell_max: float = 60.0,
        rsi_exit_overbought: float = 70.0,
        rsi_exit_oversold: float = 30.0,
        min_indicators: int = 4,
    ):
        """
        Parameters
        ----------
        ema_fast : int
            Fast EMA period (default 20).
        ema_slow : int
            Slow EMA period (default 50).
        adx_period : int
            ADX Wilder-smoothing period (default 14).
        adx_threshold : float
            ADX must exceed this for a trend to be considered strong (default 25).
        adx_no_trade_below : float
            If ADX is below this, market is sideways — hard no-trade (default 20).
        macd_fast / macd_slow / macd_signal : int
            Standard MACD parameters (12/26/9).
        rsi_period : int
            RSI period (default 14).
        rsi_buy_min / rsi_buy_max : float
            RSI window that qualifies a bullish entry (default 40–60).
        rsi_sell_min / rsi_sell_max : float
            RSI window that qualifies a bearish entry (default 40–60).
        rsi_exit_overbought / rsi_exit_oversold : float
            RSI extremes that force an exit from an existing position (70 / 30).
        min_indicators : int
            How many of the 4 indicator conditions must be satisfied to enter.
            4 = all must agree (base/strict), 3 = relaxed (for sensitivity testing).
        """
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.adx_no_trade_below = adx_no_trade_below
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_buy_min = rsi_buy_min
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min
        self.rsi_sell_max = rsi_sell_max
        self.rsi_exit_overbought = rsi_exit_overbought
        self.rsi_exit_oversold = rsi_exit_oversold
        self.min_indicators = min_indicators

        self.parameters = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'adx_no_trade_below': adx_no_trade_below,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'rsi_period': rsi_period,
            'rsi_buy_min': rsi_buy_min,
            'rsi_buy_max': rsi_buy_max,
            'rsi_sell_min': rsi_sell_min,
            'rsi_sell_max': rsi_sell_max,
            'rsi_exit_overbought': rsi_exit_overbought,
            'rsi_exit_oversold': rsi_exit_oversold,
            'min_indicators': min_indicators,
        }

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return an enriched DataFrame."""
        df = data.copy()

        df['ema_fast'] = _ema(df['Close'], self.ema_fast)
        df['ema_slow'] = _ema(df['Close'], self.ema_slow)
        df['adx'] = _adx(df['High'], df['Low'], df['Close'], self.adx_period)
        df['rsi'] = _rsi(df['Close'], self.rsi_period)

        macd_line, signal_line, histogram = _macd(
            df['Close'], self.macd_fast, self.macd_slow, self.macd_signal
        )
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = histogram
        df['macd_hist_prev'] = df['macd_hist'].shift(1)

        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Returns
        -------
        pd.DataFrame with columns:
            signal      : 1 (bullish / buy call), -1 (bearish / buy put), 0 (no trade)
            adx         : current ADX value (informational)
            rsi         : current RSI value (informational)
            macd_line   : current MACD line (informational)
            ema_fast    : fast EMA value (informational)
            ema_slow    : slow EMA value (informational)
        """
        df = self._compute_indicators(data)

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['adx'] = df['adx']
        signals['rsi'] = df['rsi']
        signals['macd_line'] = df['macd_line']
        signals['ema_fast'] = df['ema_fast']
        signals['ema_slow'] = df['ema_slow']

        # Minimum warmup: need enough bars for slow EMA + ADX + MACD
        warmup = max(self.ema_slow, self.macd_slow + self.macd_signal, self.adx_period * 2)

        position = 0  # track current position for exit logic

        for i in range(warmup, len(df)):
            row = df.iloc[i]

            adx_val = row['adx']
            rsi_val = row['rsi']
            close = row['Close']
            ema_f = row['ema_fast']
            ema_s = row['ema_slow']
            macd_l = row['macd_line']
            macd_sig = row['macd_signal']
            hist = row['macd_hist']
            hist_prev = row['macd_hist_prev']

            # Skip bars with NaN indicators (ADX warmup)
            if pd.isna(adx_val) or pd.isna(rsi_val) or pd.isna(macd_l):
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                continue

            # ---- Exit logic for existing positions ----
            if position == 1:
                macd_reversed = macd_l < macd_sig
                price_below_fast_ema = close < ema_f
                rsi_overbought = rsi_val > self.rsi_exit_overbought
                if macd_reversed or price_below_fast_ema or rsi_overbought:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                    continue

            elif position == -1:
                macd_reversed = macd_l > macd_sig
                price_above_fast_ema = close > ema_f
                rsi_oversold = rsi_val < self.rsi_exit_oversold
                if macd_reversed or price_above_fast_ema or rsi_oversold:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                    continue

            # ---- Hard no-trade zone ----
            if adx_val < self.adx_no_trade_below:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                continue

            # ---- Evaluate four indicator conditions ----

            # 1. EMA trend direction
            bullish_ema = (ema_f > ema_s) and (close > ema_f) and (close > ema_s)
            bearish_ema = (ema_f < ema_s) and (close < ema_f) and (close < ema_s)

            # 2. ADX strength
            adx_ok = adx_val >= self.adx_threshold

            # 3. MACD momentum
            hist_increasing = (not pd.isna(hist_prev)) and (hist > hist_prev)
            hist_decreasing = (not pd.isna(hist_prev)) and (hist < hist_prev)
            bullish_macd = (macd_l > macd_sig) and hist_increasing
            bearish_macd = (macd_l < macd_sig) and hist_decreasing

            # 4. RSI timing (avoid extremes)
            rsi_buy_ok = self.rsi_buy_min <= rsi_val <= self.rsi_buy_max
            rsi_sell_ok = self.rsi_sell_min <= rsi_val <= self.rsi_sell_max

            # ---- Count satisfied conditions and decide ----
            bull_score = sum([bullish_ema, adx_ok, bullish_macd, rsi_buy_ok])
            bear_score = sum([bearish_ema, adx_ok, bearish_macd, rsi_sell_ok])

            if bull_score >= self.min_indicators and position != 1:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                position = 1
            elif bear_score >= self.min_indicators and position != -1:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                position = -1
            else:
                # Carry forward existing position (hold) or stay flat
                if position != 0:
                    signals.iloc[i, signals.columns.get_loc('signal')] = position
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0

        return signals[['signal', 'adx', 'rsi', 'macd_line', 'ema_fast', 'ema_slow']]


# ---------------------------------------------------------------------------
# Strict variant — identical parameters to base, no compromise on all-4 rule
# ---------------------------------------------------------------------------

class StrictNiftyTrendStrategy(NiftyTrendOptionsStrategy):
    """
    Strict variant of NiftyTrendOptionsStrategy.

    All four indicator conditions MUST pass. This is the recommended daily
    checklist mode: if even one condition fails, no trade is taken.

    Identical to NiftyTrendOptionsStrategy with min_indicators=4 (the default),
    but named explicitly to make the rule unambiguous in back-test comparisons.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('min_indicators', 4)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Relaxed variant — 3-of-4 for sensitivity / exploration testing
# ---------------------------------------------------------------------------

class RelaxedNiftyTrendStrategy(NiftyTrendOptionsStrategy):
    """
    Relaxed variant of NiftyTrendOptionsStrategy.

    Three out of four indicator conditions must agree. Useful for back-test
    sensitivity analysis to understand how much each indicator contributes.

    NOT recommended for live trading — use StrictNiftyTrendStrategy instead.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('min_indicators', 3)
        super().__init__(**kwargs)
