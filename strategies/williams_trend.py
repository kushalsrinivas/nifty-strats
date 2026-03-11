"""
Williams %R Trend Catcher Strategy

Primary Indicator: Williams %R (momentum oscillator)
Confirmation Indicators: ADX (trend strength) + Volume Moving Average

Buy Signal:
  - Williams %R shows extreme oversold (< -80)
  - ADX confirms strong trend is forming (> threshold)
  - Volume is above moving average (participation)
  - Price action confirms reversal momentum
  
Sell Signal:
  - Williams %R shows extreme overbought (> -20)
  - OR ADX weakens significantly (trend exhaustion)
  - OR volume dries up with weakening momentum

Strategy Type: Momentum/Trend - Aggressive reversal and continuation plays
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class WilliamsTrendStrategy(Strategy):
    """
    Aggressive momentum/trend strategy using Williams %R with ADX and Volume confirmation
    
    Williams %R is a momentum oscillator that measures overbought/oversold levels.
    Combined with ADX for trend strength and volume for confirmation, it catches
    both reversals at extremes and trend continuations with strong momentum.
    """
    
    def __init__(
        self,
        williams_period: int = 14,
        williams_oversold: float = -80,  # Oversold threshold
        williams_overbought: float = -20,  # Overbought threshold
        williams_exit_oversold: float = -50,  # Less extreme for exits
        williams_exit_overbought: float = -50,
        adx_period: int = 14,
        adx_strong_trend: float = 20,  # Aggressive: lower threshold
        adx_weak_trend: float = 15,  # Below this, trend is too weak
        volume_ma_period: int = 20,
        volume_threshold: float = 1.1,  # Volume 10% above average
        momentum_lookback: int = 3  # Bars to confirm momentum shift
    ):
        """
        Initialize Williams %R Trend Catcher Strategy
        
        Args:
            williams_period: Period for Williams %R calculation
            williams_oversold: Oversold threshold for buy signals (typically -80)
            williams_overbought: Overbought threshold for sell signals (typically -20)
            williams_exit_oversold: Less extreme level for exits
            williams_exit_overbought: Less extreme level for exits
            adx_period: Period for ADX calculation
            adx_strong_trend: ADX threshold for strong trend confirmation
            adx_weak_trend: ADX threshold below which trend is too weak
            volume_ma_period: Period for volume moving average
            volume_threshold: Volume multiplier for confirmation
            momentum_lookback: Bars to look back for momentum confirmation
        """
        super().__init__()
        self.williams_period = williams_period
        self.williams_oversold = williams_oversold
        self.williams_overbought = williams_overbought
        self.williams_exit_oversold = williams_exit_oversold
        self.williams_exit_overbought = williams_exit_overbought
        self.adx_period = adx_period
        self.adx_strong_trend = adx_strong_trend
        self.adx_weak_trend = adx_weak_trend
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.momentum_lookback = momentum_lookback
        
        self.parameters = {
            'williams_period': williams_period,
            'williams_oversold': williams_oversold,
            'williams_overbought': williams_overbought,
            'williams_exit_oversold': williams_exit_oversold,
            'williams_exit_overbought': williams_exit_overbought,
            'adx_period': adx_period,
            'adx_strong_trend': adx_strong_trend,
            'adx_weak_trend': adx_weak_trend,
            'volume_ma_period': volume_ma_period,
            'volume_threshold': volume_threshold,
            'momentum_lookback': momentum_lookback
        }
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Williams %R
        
        Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        
        Range: -100 to 0
        - Below -80: Oversold (potential buy)
        - Above -20: Overbought (potential sell)
        - -50: Midpoint
        
        Note: Unlike Stochastic (0-100), Williams %R uses negative values
        """
        highest_high = data['High'].rolling(window=self.williams_period).max()
        lowest_low = data['Low'].rolling(window=self.williams_period).min()
        
        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
    def _calculate_adx(self, data: pd.DataFrame) -> tuple:
        """
        Calculate ADX (Average Directional Index) and directional indicators
        
        Returns:
            adx: Trend strength (0-100)
            plus_di: Positive directional indicator
            minus_di: Negative directional indicator
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
        
        # Smooth using Wilder's smoothing
        atr = tr.ewm(alpha=1/self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1/self.adx_period, adjust=False).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/self.adx_period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    def _check_price_momentum(self, prices: pd.Series, current_idx: int, 
                             lookback: int, direction: str) -> bool:
        """
        Check if price momentum confirms the signal direction
        
        For bullish: Price should be making higher lows or breaking resistance
        For bearish: Price should be making lower highs or breaking support
        """
        if current_idx < lookback:
            return False
        
        recent_prices = prices.iloc[current_idx - lookback:current_idx + 1]
        
        if direction == 'bullish':
            # Check if recent prices show upward momentum
            # Price should be above the average of recent prices
            price_avg = recent_prices.mean()
            current_price = prices.iloc[current_idx]
            return current_price >= price_avg
        
        elif direction == 'bearish':
            # Check if recent prices show downward momentum
            price_avg = recent_prices.mean()
            current_price = prices.iloc[current_idx]
            return current_price <= price_avg
        
        return False
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Williams %R, ADX, and Volume
        
        Buy: Williams %R oversold + Strong ADX + Volume + Price momentum
        Sell: Williams %R overbought OR ADX weakens OR volume dries up
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['williams_r'] = self._calculate_williams_r(data)
        signals['adx'], signals['plus_di'], signals['minus_di'] = self._calculate_adx(data)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        # Track previous values for trend detection
        signals['williams_r_prev'] = signals['williams_r'].shift(1)
        signals['adx_prev'] = signals['adx'].shift(1)
        
        in_position = False
        
        for i in range(len(data)):
            # Skip until we have enough data
            if i < max(self.williams_period, self.adx_period, self.volume_ma_period, self.momentum_lookback):
                continue
            
            williams_r = signals['williams_r'].iloc[i]
            williams_r_prev = signals['williams_r_prev'].iloc[i]
            adx = signals['adx'].iloc[i]
            plus_di = signals['plus_di'].iloc[i]
            minus_di = signals['minus_di'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(williams_r) or pd.isna(adx) or pd.isna(volume_ma):
                continue
            
            # BUY SIGNAL: Williams %R oversold + Strong trend forming + Volume + Momentum
            if not in_position:
                # Williams %R in extreme oversold territory
                williams_oversold = williams_r < self.williams_oversold
                
                # Williams %R starting to turn up from oversold (momentum shift)
                williams_turning_up = williams_r > williams_r_prev
                
                # ADX shows strong trend is forming or present
                strong_trend = adx > self.adx_strong_trend
                
                # Directional indicators: +DI should be rising or > -DI for bullish
                bullish_direction = plus_di > minus_di
                
                # Volume confirmation
                volume_confirmed = volume > (volume_ma * self.volume_threshold)
                
                # Price momentum confirmation
                price_momentum_bullish = self._check_price_momentum(
                    data['Close'], i, self.momentum_lookback, 'bullish'
                )
                
                # Entry logic: Multiple confirmation levels
                # Level 1: Extreme oversold with strong trend and volume
                strong_entry = (williams_oversold and strong_trend and 
                               bullish_direction and volume_confirmed)
                
                # Level 2: Oversold turning up with any trend and momentum
                moderate_entry = (williams_oversold and williams_turning_up and 
                                 adx > self.adx_weak_trend and price_momentum_bullish)
                
                if strong_entry or moderate_entry:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL SIGNAL: Overbought OR trend exhaustion OR volume dry up
            else:
                # Exit 1: Williams %R reaches overbought (target reached)
                williams_overbought = williams_r > self.williams_overbought
                
                # Exit 2: Williams %R back to neutral but turning down
                williams_neutral_down = (williams_r < self.williams_exit_overbought and 
                                        williams_r < williams_r_prev)
                
                # Exit 3: ADX weakens significantly (trend exhaustion)
                trend_weakening = adx < self.adx_weak_trend
                
                # Exit 4: Directional indicators flip bearish
                bearish_direction = minus_di > plus_di and (minus_di - plus_di) > 5
                
                # Exit 5: Volume dries up below average
                volume_dry = volume < volume_ma
                
                # Exit logic
                if williams_overbought:
                    # Immediate exit on extreme overbought
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                elif trend_weakening and williams_neutral_down:
                    # Exit on trend exhaustion with momentum turning
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                elif bearish_direction and (williams_r < self.williams_exit_oversold):
                    # Exit on directional shift
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]


class AggressiveWilliamsStrategy(Strategy):
    """
    Ultra-aggressive version - trades any Williams %R extreme with minimal confirmation
    
    Entry: Williams %R < -70 (less extreme) with any ADX trend
    Exit: Williams %R > -30 OR quick momentum shift
    """
    
    def __init__(
        self,
        williams_period: int = 10,  # Faster period
        williams_oversold: float = -70,  # Less extreme
        williams_overbought: float = -30,
        adx_period: int = 14,
        adx_threshold: float = 15,  # Very low threshold
        volume_ma_period: int = 10
    ):
        super().__init__()
        self.williams_period = williams_period
        self.williams_oversold = williams_oversold
        self.williams_overbought = williams_overbought
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volume_ma_period = volume_ma_period
        
        self.parameters = {
            'williams_period': williams_period,
            'williams_oversold': williams_oversold,
            'williams_overbought': williams_overbought,
            'adx_period': adx_period,
            'adx_threshold': adx_threshold,
            'volume_ma_period': volume_ma_period
        }
    
    def _calculate_williams_r(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['High'].rolling(window=self.williams_period).max()
        lowest_low = data['Low'].rolling(window=self.williams_period).min()
        
        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        
        return williams_r
    
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
        """Generate ultra-aggressive Williams %R signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        signals['williams_r'] = self._calculate_williams_r(data)
        signals['adx'] = self._calculate_adx(data)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        in_position = False
        
        for i in range(len(data)):
            if i < max(self.williams_period, self.adx_period, self.volume_ma_period):
                continue
            
            williams_r = signals['williams_r'].iloc[i]
            adx = signals['adx'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            if pd.isna(williams_r) or pd.isna(adx) or pd.isna(volume_ma):
                continue
            
            # BUY: Williams oversold with any trend
            if not in_position:
                oversold = williams_r < self.williams_oversold
                any_trend = adx > self.adx_threshold
                
                if oversold and any_trend:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Quick exit on momentum shift
            else:
                overbought = williams_r > self.williams_overbought
                
                if overbought:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

