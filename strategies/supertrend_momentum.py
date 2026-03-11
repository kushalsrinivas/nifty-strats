"""
Supertrend Momentum Strategy

Primary Indicator: Supertrend (ATR-based trend following)
Confirmation Indicators: MACD Histogram + EMA Slope

Buy Signal:
  - Supertrend flips from bearish to bullish (trend change)
  - MACD histogram is positive and increasing (accelerating momentum)
  - Price is above fast EMA and EMA is sloping upward
  
Sell Signal:
  - Supertrend flips from bullish to bearish
  - OR MACD histogram turns negative
  - OR price crosses below fast EMA with weakening momentum

Strategy Type: Trend Following - Ride strong trends aggressively
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class SupertrendMomentumStrategy(Strategy):
    """
    Aggressive trend-following strategy using Supertrend with MACD and EMA confirmation
    
    Supertrend is an ATR-based indicator that identifies trend direction.
    Combined with MACD acceleration and EMA slope, we capture high-momentum trends.
    """
    
    def __init__(
        self,
        atr_period: int = 10,  # Shorter period for faster signals
        atr_multiplier: float = 2.5,  # Aggressive: lower multiplier for more signals
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        ema_period: int = 20,  # Fast EMA for trend confirmation
        ema_slope_period: int = 3,  # Periods to check EMA slope
        histogram_acceleration_threshold: float = 0.0  # Histogram must be growing
    ):
        """
        Initialize Supertrend Momentum Strategy
        
        Args:
            atr_period: Period for ATR (Average True Range) calculation
            atr_multiplier: Multiplier for ATR bands (lower = more sensitive)
            macd_fast: Fast EMA period for MACD
            macd_slow: Slow EMA period for MACD
            macd_signal: Signal line period for MACD
            ema_period: Period for trend confirmation EMA
            ema_slope_period: Bars to measure EMA slope
            histogram_acceleration_threshold: Min MACD histogram acceleration
        """
        super().__init__()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_period = ema_period
        self.ema_slope_period = ema_slope_period
        self.histogram_acceleration_threshold = histogram_acceleration_threshold
        
        self.parameters = {
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ema_period': ema_period,
            'ema_slope_period': ema_slope_period,
            'histogram_acceleration_threshold': histogram_acceleration_threshold
        }
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR (Average True Range)
        
        ATR measures market volatility
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def _calculate_supertrend(self, data: pd.DataFrame) -> tuple:
        """
        Calculate Supertrend indicator
        
        Supertrend uses ATR to create dynamic support/resistance bands
        - When price is above Supertrend: Bullish (uptrend)
        - When price is below Supertrend: Bearish (downtrend)
        
        Returns:
            supertrend: Series with Supertrend values
            direction: Series with 1 (bullish) or -1 (bearish)
        """
        atr = self._calculate_atr(data)
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate basic bands
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (self.atr_multiplier * atr)
        lower_band = hl_avg - (self.atr_multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        # Calculate supertrend values
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
                continue
            
            # Adjust bands based on previous values
            if pd.notna(lower_band.iloc[i]) and pd.notna(lower_band.iloc[i-1]):
                if lower_band.iloc[i] < lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
            
            if pd.notna(upper_band.iloc[i]) and pd.notna(upper_band.iloc[i-1]):
                if upper_band.iloc[i] > upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Determine trend direction
            if pd.notna(close.iloc[i]):
                if close.iloc[i] <= upper_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1  # Bearish
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1  # Bullish
        
        return supertrend, direction
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD and histogram"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def _calculate_ema_slope(self, ema: pd.Series, period: int) -> pd.Series:
        """
        Calculate EMA slope (rate of change)
        
        Positive slope = uptrend, Negative slope = downtrend
        """
        slope = ema.diff(period)
        return slope
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Supertrend, MACD, and EMA
        
        Buy: Supertrend flip to bullish + MACD accelerating + Price above rising EMA
        Sell: Supertrend flip to bearish OR MACD turns negative OR EMA breakdown
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['supertrend'], signals['st_direction'] = self._calculate_supertrend(data)
        signals['macd'], signals['macd_signal'], signals['macd_histogram'] = \
            self._calculate_macd(data['Close'])
        signals['ema'] = data['Close'].ewm(span=self.ema_period, adjust=False).mean()
        signals['ema_slope'] = self._calculate_ema_slope(signals['ema'], self.ema_slope_period)
        
        # Track previous direction for crossover detection
        signals['st_direction_prev'] = signals['st_direction'].shift(1)
        signals['histogram_prev'] = signals['macd_histogram'].shift(1)
        
        in_position = False
        
        for i in range(len(data)):
            # Skip until we have enough data
            if i < max(self.atr_period, self.macd_slow, self.ema_period, self.ema_slope_period):
                continue
            
            current_price = data['Close'].iloc[i]
            st_direction = signals['st_direction'].iloc[i]
            st_direction_prev = signals['st_direction_prev'].iloc[i]
            histogram = signals['macd_histogram'].iloc[i]
            histogram_prev = signals['histogram_prev'].iloc[i]
            ema = signals['ema'].iloc[i]
            ema_slope = signals['ema_slope'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(st_direction) or pd.isna(histogram) or pd.isna(ema) or pd.isna(ema_slope):
                continue
            
            # BUY SIGNAL: Supertrend flips bullish + MACD accelerating + Price above rising EMA
            if not in_position:
                # Supertrend flip from bearish to bullish
                supertrend_bullish_flip = (st_direction == 1) and (st_direction_prev == -1)
                
                # MACD histogram positive and accelerating
                histogram_positive = histogram > 0
                histogram_accelerating = histogram > histogram_prev + self.histogram_acceleration_threshold
                
                # Price above EMA and EMA sloping upward
                price_above_ema = current_price > ema
                ema_rising = ema_slope > 0
                
                # Strong confluence: all conditions met
                if supertrend_bullish_flip and histogram_positive and histogram_accelerating:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
                # Moderate confluence: trend flip with EMA support
                elif supertrend_bullish_flip and price_above_ema and ema_rising:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL SIGNAL: Supertrend flips bearish OR momentum deteriorates
            else:
                # Supertrend flip from bullish to bearish (trend reversal)
                supertrend_bearish_flip = (st_direction == -1) and (st_direction_prev == 1)
                
                # MACD histogram turns negative (momentum loss)
                histogram_negative = histogram < 0
                histogram_decelerating = histogram < histogram_prev
                
                # Price breaks below EMA with downward slope (support break)
                price_below_ema = current_price < ema
                ema_falling = ema_slope < 0
                
                # Exit conditions
                if supertrend_bearish_flip:
                    # Immediate exit on trend reversal
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                elif histogram_negative and histogram_decelerating:
                    # Exit on momentum loss
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                elif price_below_ema and ema_falling:
                    # Exit on support break
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]


class AggressiveSupertrendStrategy(Strategy):
    """
    Ultra-aggressive version - enters on any Supertrend signal with momentum
    
    Entry: Supertrend bullish + (MACD positive OR price above EMA)
    Exit: Supertrend bearish OR quick momentum shift
    """
    
    def __init__(
        self,
        atr_period: int = 7,  # Very fast
        atr_multiplier: float = 2.0,  # More sensitive
        macd_fast: int = 8,
        macd_slow: int = 21,
        macd_signal: int = 5,
        ema_period: int = 10
    ):
        super().__init__()
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_period = ema_period
        
        self.parameters = {
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'ema_period': ema_period
        }
    
    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        return atr
    
    def _calculate_supertrend(self, data: pd.DataFrame) -> tuple:
        """Calculate Supertrend"""
        atr = self._calculate_atr(data)
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        hl_avg = (high + low) / 2
        upper_band = hl_avg + (self.atr_multiplier * atr)
        lower_band = hl_avg - (self.atr_multiplier * atr)
        
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
                continue
            
            if pd.notna(lower_band.iloc[i]) and pd.notna(lower_band.iloc[i-1]):
                if lower_band.iloc[i] < lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
            
            if pd.notna(upper_band.iloc[i]) and pd.notna(upper_band.iloc[i-1]):
                if upper_band.iloc[i] > upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
            
            if pd.notna(close.iloc[i]):
                if close.iloc[i] <= upper_band.iloc[i]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
        
        return supertrend, direction
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD"""
        exp1 = prices.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra-aggressive trend signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        signals['supertrend'], signals['st_direction'] = self._calculate_supertrend(data)
        signals['macd'], signals['macd_signal'], signals['macd_histogram'] = \
            self._calculate_macd(data['Close'])
        signals['ema'] = data['Close'].ewm(span=self.ema_period, adjust=False).mean()
        signals['st_direction_prev'] = signals['st_direction'].shift(1)
        
        in_position = False
        
        for i in range(len(data)):
            if i < max(self.atr_period, self.macd_slow, self.ema_period):
                continue
            
            current_price = data['Close'].iloc[i]
            st_direction = signals['st_direction'].iloc[i]
            st_direction_prev = signals['st_direction_prev'].iloc[i]
            histogram = signals['macd_histogram'].iloc[i]
            ema = signals['ema'].iloc[i]
            
            if pd.isna(st_direction) or pd.isna(histogram) or pd.isna(ema):
                continue
            
            # BUY: Supertrend bullish + any momentum confirmation
            if not in_position:
                st_bullish = (st_direction == 1) and (st_direction_prev == -1)
                macd_positive = histogram > 0
                above_ema = current_price > ema
                
                if st_bullish and (macd_positive or above_ema):
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Supertrend bearish OR momentum turns
            else:
                st_bearish = (st_direction == -1) and (st_direction_prev == 1)
                macd_negative = histogram < 0
                
                if st_bearish or macd_negative:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

