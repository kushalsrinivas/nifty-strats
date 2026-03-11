"""
Stochastic RSI Breakout Strategy

Primary Indicator: Stochastic Oscillator
Confirmation Indicators: Volume Spike + ADX (Trend Strength)

Buy Signal:
  - Stochastic %K crosses above oversold level (typically 20)
  - Volume is significantly above average (volume spike)
  - ADX shows strong trend strength (> threshold)
  
Sell Signal:
  - Stochastic %K crosses below overbought level (typically 80)
  - OR ADX weakens significantly (trend exhaustion)

Strategy Type: Breakout/Momentum - Aggressive entries on momentum shifts
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class StochasticBreakoutStrategy(Strategy):
    """
    Aggressive breakout strategy using Stochastic Oscillator with Volume and ADX confirmation
    
    This strategy identifies momentum shifts using Stochastic, confirms with volume surges,
    and ensures trend strength using ADX before entering positions.
    """
    
    def __init__(
        self,
        stoch_period: int = 14,
        stoch_smooth_k: int = 3,
        stoch_smooth_d: int = 3,
        stoch_oversold: float = 20,
        stoch_overbought: float = 80,
        adx_period: int = 14,
        adx_threshold: float = 20,  # Aggressive: lower threshold for faster entries
        volume_ma_period: int = 20,
        volume_spike_multiplier: float = 1.3  # Aggressive: 30% above average
    ):
        """
        Initialize Stochastic Breakout Strategy
        
        Args:
            stoch_period: Lookback period for Stochastic calculation
            stoch_smooth_k: Smoothing period for %K line
            stoch_smooth_d: Smoothing period for %D line
            stoch_oversold: Oversold threshold for buy signal
            stoch_overbought: Overbought threshold for sell signal
            adx_period: Period for ADX calculation
            adx_threshold: Minimum ADX for trend confirmation (lower = more aggressive)
            volume_ma_period: Period for volume moving average
            volume_spike_multiplier: Volume must be X times above average
        """
        super().__init__()
        self.stoch_period = stoch_period
        self.stoch_smooth_k = stoch_smooth_k
        self.stoch_smooth_d = stoch_smooth_d
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_ma_period = volume_ma_period
        self.volume_spike_multiplier = volume_spike_multiplier
        
        self.parameters = {
            'stoch_period': stoch_period,
            'stoch_smooth_k': stoch_smooth_k,
            'stoch_smooth_d': stoch_smooth_d,
            'stoch_oversold': stoch_oversold,
            'stoch_overbought': stoch_overbought,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'volume_ma_period': volume_ma_period,
            'volume_spike_multiplier': volume_spike_multiplier
        }
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> tuple:
        """
        Calculate Stochastic Oscillator (%K and %D)
        
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = Moving Average of %K
        """
        # Calculate raw stochastic
        low_min = data['Low'].rolling(window=self.stoch_period).min()
        high_max = data['High'].rolling(window=self.stoch_period).max()
        
        stoch_raw = 100 * (data['Close'] - low_min) / (high_max - low_min)
        
        # Smooth %K
        stoch_k = stoch_raw.rolling(window=self.stoch_smooth_k).mean()
        
        # Calculate %D (signal line)
        stoch_d = stoch_k.rolling(window=self.stoch_smooth_d).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ADX (Average Directional Index) - measures trend strength
        
        ADX ranges from 0-100:
        - Below 20: Weak trend
        - 20-40: Strong trend
        - Above 40: Very strong trend
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smooth using Wilder's smoothing (exponential moving average)
        atr = tr.ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()
        
        return adx
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Stochastic, Volume, and ADX
        
        Buy: Stochastic crosses above oversold + Volume spike + Strong ADX
        Sell: Stochastic crosses below overbought OR ADX weakens
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['stoch_k'], signals['stoch_d'] = self._calculate_stochastic(data)
        signals['adx'] = self._calculate_adx(data)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        # Track previous values for crossover detection
        signals['stoch_k_prev'] = signals['stoch_k'].shift(1)
        signals['adx_prev'] = signals['adx'].shift(1)
        
        in_position = False
        
        for i in range(len(data)):
            # Skip until we have enough data
            if i < max(self.stoch_period, self.adx_period, self.volume_ma_period):
                continue
            
            stoch_k = signals['stoch_k'].iloc[i]
            stoch_k_prev = signals['stoch_k_prev'].iloc[i]
            adx = signals['adx'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(stoch_k) or pd.isna(adx) or pd.isna(volume_ma):
                continue
            
            # BUY SIGNAL: Stochastic crosses above oversold + Volume spike + Strong ADX
            if not in_position:
                stoch_cross_up = (stoch_k > self.stoch_oversold) and (stoch_k_prev <= self.stoch_oversold)
                volume_spike = volume > (volume_ma * self.volume_spike_multiplier)
                strong_trend = adx > self.adx_threshold
                
                if stoch_cross_up and volume_spike and strong_trend:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL SIGNAL: Stochastic crosses below overbought OR ADX weakens significantly
            else:
                stoch_cross_down = (stoch_k < self.stoch_overbought) and (stoch_k_prev >= self.stoch_overbought)
                trend_weakening = adx < (self.adx_threshold * 0.7)  # ADX drops below 70% of threshold
                
                if stoch_cross_down or trend_weakening:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]


class AggressiveStochasticStrategy(Strategy):
    """
    Ultra-aggressive version - looser conditions for faster entries
    
    Entry: Stochastic above oversold OR strong volume with any ADX trend
    Exit: Stochastic overbought OR momentum shift detected
    """
    
    def __init__(
        self,
        stoch_period: int = 9,  # Shorter period for faster signals
        stoch_smooth_k: int = 3,
        stoch_smooth_d: int = 3,
        stoch_oversold: float = 25,  # Less extreme levels
        stoch_overbought: float = 75,
        adx_period: int = 14,
        adx_threshold: float = 15,  # Very low threshold
        volume_ma_period: int = 10,  # Shorter MA
        volume_spike_multiplier: float = 1.2  # Just 20% above average
    ):
        super().__init__()
        self.stoch_period = stoch_period
        self.stoch_smooth_k = stoch_smooth_k
        self.stoch_smooth_d = stoch_smooth_d
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_ma_period = volume_ma_period
        self.volume_spike_multiplier = volume_spike_multiplier
        
        self.parameters = {
            'stoch_period': stoch_period,
            'stoch_smooth_k': stoch_smooth_k,
            'stoch_smooth_d': stoch_smooth_d,
            'stoch_oversold': stoch_oversold,
            'stoch_overbought': stoch_overbought,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'volume_ma_period': volume_ma_period,
            'volume_spike_multiplier': volume_spike_multiplier
        }
    
    def _calculate_stochastic(self, data: pd.DataFrame) -> tuple:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=self.stoch_period).min()
        high_max = data['High'].rolling(window=self.stoch_period).max()
        
        stoch_raw = 100 * (data['Close'] - low_min) / (high_max - low_min)
        stoch_k = stoch_raw.rolling(window=self.stoch_smooth_k).mean()
        stoch_d = stoch_k.rolling(window=self.stoch_smooth_d).mean()
        
        return stoch_k, stoch_d
    
    def _calculate_adx(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ADX"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        atr = tr.ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()
        
        return adx
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra-aggressive trading signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['stoch_k'], signals['stoch_d'] = self._calculate_stochastic(data)
        signals['adx'] = self._calculate_adx(data)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        signals['stoch_k_prev'] = signals['stoch_k'].shift(1)
        
        in_position = False
        
        for i in range(len(data)):
            if i < max(self.stoch_period, self.adx_period, self.volume_ma_period):
                continue
            
            stoch_k = signals['stoch_k'].iloc[i]
            stoch_k_prev = signals['stoch_k_prev'].iloc[i]
            adx = signals['adx'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            if pd.isna(stoch_k) or pd.isna(adx) or pd.isna(volume_ma):
                continue
            
            # BUY: Looser conditions - just need stochastic momentum OR volume
            if not in_position:
                stoch_rising = stoch_k > self.stoch_oversold
                volume_high = volume > (volume_ma * self.volume_spike_multiplier)
                any_trend = adx > self.adx_threshold
                
                if (stoch_rising and volume_high) or (stoch_rising and any_trend):
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Quick exit on momentum loss
            else:
                stoch_too_high = stoch_k > self.stoch_overbought
                momentum_shift = (stoch_k < stoch_k_prev) and (stoch_k > 60)
                
                if stoch_too_high or momentum_shift:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

