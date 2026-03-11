"""
Momentum Strategy

Buys when price momentum is positive and sells when momentum turns negative.
Uses Rate of Change (ROC) and Relative Strength Index (RSI).
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class MomentumStrategy(Strategy):
    """
    Momentum strategy based on Rate of Change (ROC)
    
    Buy when ROC is above threshold (strong upward momentum)
    Sell when ROC is below negative threshold (downward momentum)
    """
    
    def __init__(self, period: int = 20, buy_threshold: float = 5.0, sell_threshold: float = -5.0):
        """
        Initialize strategy
        
        Args:
            period: Lookback period for momentum calculation
            buy_threshold: ROC threshold for buy signal (percentage)
            sell_threshold: ROC threshold for sell signal (percentage)
        """
        super().__init__()
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.parameters = {
            'period': period,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on momentum
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate Rate of Change (ROC)
        signals['roc'] = ((data['Close'] - data['Close'].shift(self.period)) / 
                          data['Close'].shift(self.period) * 100)
        
        # Generate signals
        # Buy when ROC crosses above buy threshold
        signals.loc[signals['roc'] > self.buy_threshold, 'signal'] = 1
        
        # Sell when ROC crosses below sell threshold
        signals.loc[signals['roc'] < self.sell_threshold, 'signal'] = -1
        
        return signals[['signal']]


class RSIMomentumStrategy(Strategy):
    """
    Momentum strategy based on Relative Strength Index (RSI)
    
    Buy when RSI crosses above oversold level
    Sell when RSI crosses above overbought level
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Initialize strategy
        
        Args:
            period: Period for RSI calculation
            oversold: RSI level considered oversold (buy signal)
            overbought: RSI level considered overbought (sell signal)
        """
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.parameters = {
            'period': period,
            'oversold': oversold,
            'overbought': overbought
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate RSI
        signals['rsi'] = self._calculate_rsi(data['Close'], self.period)
        
        # Track previous RSI to detect crossovers
        signals['rsi_prev'] = signals['rsi'].shift(1)
        
        # Buy when RSI crosses above oversold level
        buy_condition = (signals['rsi'] > self.oversold) & (signals['rsi_prev'] <= self.oversold)
        signals.loc[buy_condition, 'signal'] = 1
        
        # Sell when RSI crosses above overbought level
        sell_condition = (signals['rsi'] > self.overbought) & (signals['rsi_prev'] <= self.overbought)
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals[['signal']]


class MACDMomentumStrategy(Strategy):
    """
    Enhanced MACD Momentum Strategy
    
    Improvements:
    - Only buys when MACD is above zero (bullish zone)
    - Considers gap between MACD and signal line for trend strength
    - Uses histogram strength to filter weak signals
    
    Buy when:
    1. MACD line crosses above signal line
    2. MACD is above zero (bullish zone)
    3. Gap between MACD and signal line indicates strong trend
    
    Sell when:
    - MACD crosses below signal line, OR
    - MACD drops below zero (bearish zone)
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        require_macd_above_zero: bool = True,
        min_histogram_strength: float = 0.0,
        use_histogram_divergence: bool = True
    ):
        """
        Initialize strategy
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            require_macd_above_zero: Only buy when MACD > 0 (bullish zone)
            min_histogram_strength: Minimum histogram value to trigger buy
            use_histogram_divergence: Consider increasing histogram as confirmation
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.require_macd_above_zero = require_macd_above_zero
        self.min_histogram_strength = min_histogram_strength
        self.use_histogram_divergence = use_histogram_divergence
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'require_macd_above_zero': require_macd_above_zero,
            'min_histogram_strength': min_histogram_strength,
            'use_histogram_divergence': use_histogram_divergence
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on enhanced MACD logic
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate MACD components
        exp1 = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        
        signals['macd'] = exp1 - exp2
        signals['signal_line'] = signals['macd'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram (gap between MACD and signal line)
        signals['histogram'] = signals['macd'] - signals['signal_line']
        
        # Previous values for crossover detection
        signals['macd_prev'] = signals['macd'].shift(1)
        signals['signal_line_prev'] = signals['signal_line'].shift(1)
        signals['histogram_prev'] = signals['histogram'].shift(1)
        
        # === BUY CONDITIONS ===
        
        # 1. MACD crosses above signal line (bullish crossover)
        bullish_crossover = (signals['macd'] > signals['signal_line']) & \
                           (signals['macd_prev'] <= signals['signal_line_prev'])
        
        # 2. MACD is above zero (bullish zone) - optional
        if self.require_macd_above_zero:
            macd_above_zero = signals['macd'] > 0
        else:
            macd_above_zero = True
        
        # 3. Histogram strength (gap indicates trend strength)
        histogram_strong = signals['histogram'] >= self.min_histogram_strength
        
        # 4. Histogram divergence (gap is widening = strengthening trend)
        if self.use_histogram_divergence:
            histogram_increasing = signals['histogram'] > signals['histogram_prev']
        else:
            histogram_increasing = True
        
        # Combine all buy conditions
        buy_condition = bullish_crossover & macd_above_zero & histogram_strong & histogram_increasing
        signals.loc[buy_condition, 'signal'] = 1
        
        # === SELL CONDITIONS ===
        
        # Sell when MACD crosses below signal line (bearish crossover)
        bearish_crossover = (signals['macd'] < signals['signal_line']) & \
                           (signals['macd_prev'] >= signals['signal_line_prev'])
        
        # OR when MACD drops below zero (exit bullish zone)
        macd_below_zero = signals['macd'] < 0
        
        # Combine sell conditions
        sell_condition = bearish_crossover | (macd_below_zero & (signals['macd_prev'] >= 0))
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals[['signal']]

