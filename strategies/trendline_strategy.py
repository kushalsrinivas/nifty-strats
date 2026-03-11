"""
Trend Line Trading Strategy

Identifies and trades based on trend lines:
- Uptrend: Buy when price bounces off ascending trend line
- Downtrend: Sell when price bounces off descending trend line
- Breakout: Trade when price breaks through trend line

Uses linear regression to identify trend lines and validates with
multiple touch points for confirmation.
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy
from typing import Tuple, Optional, List
from scipy import stats


class TrendLineStrategy(Strategy):
    """
    Trend Line Trading Strategy
    
    Identifies trend lines using swing highs/lows and trades bounces
    or breakouts from these lines.
    """
    
    def __init__(
        self,
        lookback_period: int = 50,
        min_touches: int = 2,
        trend_threshold: float = 0.0001,  # Minimum slope for valid trend
        bounce_tolerance: float = 0.02,  # 2% from trend line to consider "bounce"
        volume_confirmation: bool = True,
        volume_threshold: float = 1.2,  # 20% above average
        atr_period: int = 14,
        atr_multiplier: float = 1.5,  # For stop loss
        breakout_mode: bool = False,  # False = bounce, True = breakout
    ):
        """
        Initialize Trend Line Strategy
        
        Args:
            lookback_period: Bars to look back for trend line calculation
            min_touches: Minimum touch points to validate trend line
            trend_threshold: Minimum slope to consider valid trend
            bounce_tolerance: Distance from trend line to trigger (%)
            volume_confirmation: Require volume spike for entry
            volume_threshold: Volume multiplier vs average
            atr_period: Period for ATR calculation (stop loss)
            atr_multiplier: ATR multiplier for stop loss
            breakout_mode: Trade breakouts instead of bounces
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.min_touches = min_touches
        self.trend_threshold = trend_threshold
        self.bounce_tolerance = bounce_tolerance
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.breakout_mode = breakout_mode
        
        self.parameters = {
            'lookback_period': lookback_period,
            'min_touches': min_touches,
            'trend_threshold': trend_threshold,
            'bounce_tolerance': bounce_tolerance,
            'volume_confirmation': volume_confirmation,
            'volume_threshold': volume_threshold,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'mode': 'Breakout' if breakout_mode else 'Bounce'
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """
        Find swing highs and lows
        
        Returns:
            Tuple of (swing_highs, swing_lows) with (index, price) tuples
        """
        swing_highs = []
        swing_lows = []
        
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(window, len(data) - window):
            # Check for swing high
            is_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            
            if is_high:
                swing_highs.append((i, highs[i]))
            
            # Check for swing low
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            
            if is_low:
                swing_lows.append((i, lows[i]))
        
        return swing_highs, swing_lows
    
    def _calculate_trendline(self, points: List[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
        """
        Calculate trend line using linear regression
        
        Args:
            points: List of (index, price) tuples
            
        Returns:
            (slope, intercept) or None if insufficient points
        """
        if len(points) < self.min_touches:
            return None
        
        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Validate trend strength
        if abs(slope) < self.trend_threshold:
            return None
        
        return slope, intercept
    
    def _get_trendline_value(self, slope: float, intercept: float, index: int) -> float:
        """Calculate trend line value at given index"""
        return slope * index + intercept
    
    def _count_touches(self, points: List[Tuple[int, float]], 
                       slope: float, intercept: float, 
                       tolerance: float = 0.02) -> int:
        """
        Count how many points touch the trend line
        
        Args:
            points: List of (index, price) tuples
            slope: Trend line slope
            intercept: Trend line intercept
            tolerance: Distance tolerance (%)
            
        Returns:
            Number of touch points
        """
        touches = 0
        
        for idx, price in points:
            trendline_value = self._get_trendline_value(slope, intercept, idx)
            distance = abs(price - trendline_value) / trendline_value
            
            if distance <= tolerance:
                touches += 1
        
        return touches
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on trend line bounces/breakouts
        
        Logic:
        - Identify swing highs and lows
        - Calculate trend lines (support and resistance)
        - Bounce mode: Buy when price bounces off support, sell at resistance
        - Breakout mode: Buy when breaking resistance, sell when breaking support
        - Use volume and ATR for confirmation
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        # Calculate ATR for stop loss
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        
        # Calculate volume average
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        # Track position
        in_position = False
        current_trend = None  # 'up' or 'down'
        support_line = None
        resistance_line = None
        
        for i in range(self.lookback_period, len(data)):
            # Get lookback window
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            # Find swing points in window
            swing_highs, swing_lows = self._find_swing_points(window_data)
            
            if len(swing_highs) < self.min_touches or len(swing_lows) < self.min_touches:
                continue
            
            # Adjust indices to absolute position
            swing_highs_abs = [(window_start + idx, price) for idx, price in swing_highs]
            swing_lows_abs = [(window_start + idx, price) for idx, price in swing_lows]
            
            # Calculate resistance line (from swing highs)
            resistance = self._calculate_trendline(swing_highs_abs[-5:])
            
            # Calculate support line (from swing lows)
            support = self._calculate_trendline(swing_lows_abs[-5:])
            
            # Current price info
            current_price = data['Close'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            
            if pd.isna(current_atr) or pd.isna(avg_volume):
                continue
            
            # Volume confirmation
            volume_ok = True
            if self.volume_confirmation:
                volume_ok = current_volume >= (avg_volume * self.volume_threshold)
            
            # ========== UPTREND - SUPPORT LINE ==========
            if support is not None:
                slope_sup, intercept_sup = support
                
                # Validate support line
                touches = self._count_touches(swing_lows_abs[-10:], slope_sup, 
                                             intercept_sup, self.bounce_tolerance)
                
                if touches >= self.min_touches and slope_sup > 0:  # Uptrend
                    support_value = self._get_trendline_value(slope_sup, intercept_sup, i)
                    distance_from_support = (current_price - support_value) / support_value
                    
                    # BOUNCE MODE: Buy when bouncing off support
                    if not self.breakout_mode and not in_position:
                        # Price near support (within tolerance)
                        if abs(distance_from_support) <= self.bounce_tolerance and volume_ok:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            # Stop loss below support
                            signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                                support_value - (current_atr * self.atr_multiplier)
                            in_position = True
                            current_trend = 'up'
                            support_line = (slope_sup, intercept_sup)
                    
                    # BREAKOUT MODE: Sell when breaking below support
                    elif self.breakout_mode and in_position and current_trend == 'up':
                        # Price breaks below support
                        if distance_from_support < -self.bounce_tolerance and volume_ok:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            in_position = False
                            current_trend = None
            
            # ========== DOWNTREND - RESISTANCE LINE ==========
            if resistance is not None:
                slope_res, intercept_res = resistance
                
                # Validate resistance line
                touches = self._count_touches(swing_highs_abs[-10:], slope_res, 
                                             intercept_res, self.bounce_tolerance)
                
                if touches >= self.min_touches:
                    resistance_value = self._get_trendline_value(slope_res, intercept_res, i)
                    distance_from_resistance = (current_price - resistance_value) / resistance_value
                    
                    # BOUNCE MODE: Sell when hitting resistance
                    if not self.breakout_mode and in_position:
                        # Price near resistance (within tolerance)
                        if abs(distance_from_resistance) <= self.bounce_tolerance:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            in_position = False
                            current_trend = None
                    
                    # BREAKOUT MODE: Buy when breaking above resistance
                    elif self.breakout_mode and not in_position:
                        # Price breaks above resistance
                        if distance_from_resistance > self.bounce_tolerance and volume_ok:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            # Stop loss below resistance (now support)
                            signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                                resistance_value - (current_atr * self.atr_multiplier)
                            in_position = True
                            current_trend = 'up'
                            resistance_line = (slope_res, intercept_res)
            
            # Stop loss check for existing positions
            if in_position and current_trend == 'up':
                if support_line is not None:
                    slope_sup, intercept_sup = support_line
                    support_value = self._get_trendline_value(slope_sup, intercept_sup, i)
                    
                    # Update trailing stop if support is rising
                    if support_value > signals['stop_price'].iloc[i-1]:
                        signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                            support_value - (current_atr * self.atr_multiplier)
        
        return signals[['signal', 'stop_price']]


class TrendLineBreakoutStrategy(Strategy):
    """
    Simplified Trend Line Breakout Strategy
    
    Focuses on clean breakouts with strong volume confirmation.
    """
    
    def __init__(
        self,
        lookback_period: int = 40,
        min_touches: int = 2,
        volume_threshold: float = 1.5,  # 50% above average
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """Initialize Trend Line Breakout Strategy"""
        super().__init__()
        
        # Use base strategy in breakout mode
        self.base_strategy = TrendLineStrategy(
            lookback_period=lookback_period,
            min_touches=min_touches,
            trend_threshold=0.0001,
            bounce_tolerance=0.015,  # Tighter for breakouts
            volume_confirmation=True,
            volume_threshold=volume_threshold,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            breakout_mode=True
        )
        
        self.parameters = self.base_strategy.parameters
        self.parameters['strategy_type'] = 'Trend Line Breakout'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using breakout mode"""
        return self.base_strategy.generate_signals(data)

