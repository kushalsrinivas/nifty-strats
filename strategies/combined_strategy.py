"""
Combined Strategy using RSI, MACD, and Bollinger Bands

This strategy combines three popular technical indicators:
- RSI (Relative Strength Index) for momentum
- MACD (Moving Average Convergence Divergence) for trend
- Bollinger Bands for volatility and mean reversion

Buy Signal: When ALL conditions are met:
  1. RSI is oversold (< 30) - indicates potential reversal
  2. MACD line crosses above signal line - confirms upward momentum
  3. Price touches lower Bollinger Band - indicates oversold condition

Sell Signal: When ANY condition is met:
  1. RSI is overbought (> 70) - indicates potential reversal
  2. MACD line crosses below signal line - confirms downward momentum
  3. Price touches upper Bollinger Band - indicates overbought condition
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class CombinedStrategy(Strategy):
    """
    Combined trading strategy using RSI, MACD, and Bollinger Bands
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        """
        Initialize combined strategy
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line period for MACD
            bb_period: Period for Bollinger Bands
            bb_std: Number of standard deviations for Bollinger Bands
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        self.parameters = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bb_period': bb_period,
            'bb_std': bb_std
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper_band = ma + (std * self.bb_std)
        lower_band = ma - (std * self.bb_std)
        
        return upper_band, ma, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on combined indicators
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate all indicators
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['macd'], signals['macd_signal'] = self._calculate_macd(data['Close'])
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = \
            self._calculate_bollinger_bands(data['Close'])
        
        # Track previous values to detect crossovers
        signals['macd_prev'] = signals['macd'].shift(1)
        signals['macd_signal_prev'] = signals['macd_signal'].shift(1)
        
        # Define conditions
        # Buy conditions (ALL must be true)
        rsi_oversold = signals['rsi'] < self.rsi_oversold
        macd_bullish_cross = (signals['macd'] > signals['macd_signal']) & \
                            (signals['macd_prev'] <= signals['macd_signal_prev'])
        price_at_lower_bb = data['Close'] <= signals['bb_lower'] * 1.01  # Within 1% of lower band
        
        buy_signal = rsi_oversold & macd_bullish_cross & price_at_lower_bb
        
        # Sell conditions (ANY can be true)
        rsi_overbought = signals['rsi'] > self.rsi_overbought
        macd_bearish_cross = (signals['macd'] < signals['macd_signal']) & \
                            (signals['macd_prev'] >= signals['macd_signal_prev'])
        price_at_upper_bb = data['Close'] >= signals['bb_upper'] * 0.99  # Within 1% of upper band
        
        sell_signal = rsi_overbought | macd_bearish_cross | price_at_upper_bb
        
        # Generate final signals
        signals.loc[buy_signal, 'signal'] = 1
        signals.loc[sell_signal, 'signal'] = -1
        
        return signals[['signal']]


class AggressiveCombinedStrategy(Strategy):
    """
    More aggressive version - requires fewer conditions for signals
    
    Buy: RSI oversold OR (MACD bullish cross AND price near lower BB)
    Sell: RSI overbought OR (MACD bearish cross AND price near upper BB)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35,
        rsi_overbought: float = 65,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        self.parameters = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'bb_period': bb_period,
            'bb_std': bb_std
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper_band = ma + (std * self.bb_std)
        lower_band = ma - (std * self.bb_std)
        
        return upper_band, ma, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate more aggressive trading signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate all indicators
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['macd'], signals['macd_signal'] = self._calculate_macd(data['Close'])
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = \
            self._calculate_bollinger_bands(data['Close'])
        
        # Track previous values
        signals['macd_prev'] = signals['macd'].shift(1)
        signals['macd_signal_prev'] = signals['macd_signal'].shift(1)
        
        # Buy conditions (any can trigger)
        rsi_oversold = signals['rsi'] < self.rsi_oversold
        macd_bullish = (signals['macd'] > signals['macd_signal']) & \
                      (signals['macd_prev'] <= signals['macd_signal_prev'])
        price_near_lower = data['Close'] <= signals['bb_lower'] * 1.02
        
        buy_signal = rsi_oversold | (macd_bullish & price_near_lower)
        
        # Sell conditions
        rsi_overbought = signals['rsi'] > self.rsi_overbought
        macd_bearish = (signals['macd'] < signals['macd_signal']) & \
                      (signals['macd_prev'] >= signals['macd_signal_prev'])
        price_near_upper = data['Close'] >= signals['bb_upper'] * 0.98
        
        sell_signal = rsi_overbought | (macd_bearish & price_near_upper)
        
        signals.loc[buy_signal, 'signal'] = 1
        signals.loc[sell_signal, 'signal'] = -1
        
        return signals[['signal']]

