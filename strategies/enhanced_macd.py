"""
Enhanced MACD Strategies

Advanced MACD strategies with improved logic:
- Only buy when MACD is above zero (bullish zone)
- Consider gap between MACD and signal line (histogram)
- Histogram strength indicates trend quality
- Optional volume confirmation
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class EnhancedMACDStrategy(Strategy):
    """
    Enhanced MACD Strategy with Zero-Line and Histogram Analysis
    
    Key Improvements over basic MACD:
    1. MACD Above Zero: Only buys in bullish zone (MACD > 0)
    2. Histogram Strength: Gap between MACD and signal line indicates trend strength
    3. Histogram Momentum: Increasing histogram = strengthening trend
    4. Volume Confirmation: Optional volume spike requirement
    
    Trading Logic:
    - BUY: MACD crosses above signal + MACD > 0 + Strong histogram + Widening gap
    - SELL: MACD crosses below signal OR drops below zero OR histogram weakens
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        require_macd_above_zero: bool = True,
        min_histogram_strength: float = 0.5,
        use_histogram_divergence: bool = True,
        use_volume_confirmation: bool = False,
        volume_threshold: float = 1.2
    ):
        """
        Initialize Enhanced MACD Strategy
        
        Args:
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line period (default 9)
            require_macd_above_zero: Only buy when MACD > 0 (bullish zone)
            min_histogram_strength: Minimum histogram value for entry
            use_histogram_divergence: Require widening histogram
            use_volume_confirmation: Require volume spike
            volume_threshold: Volume must be X times 20-day average
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.require_macd_above_zero = require_macd_above_zero
        self.min_histogram_strength = min_histogram_strength
        self.use_histogram_divergence = use_histogram_divergence
        self.use_volume_confirmation = use_volume_confirmation
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'require_macd_above_zero': require_macd_above_zero,
            'min_histogram_strength': min_histogram_strength,
            'use_histogram_divergence': use_histogram_divergence,
            'use_volume_confirmation': use_volume_confirmation,
            'volume_threshold': volume_threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with enhanced MACD logic
        
        BUY when:
        1. MACD line crosses above signal line (bullish crossover)
        2. MACD is above zero (bullish zone) - confirms uptrend
        3. Histogram (gap) is strong enough (indicates clear trend)
        4. Histogram is widening (trend is strengthening)
        5. Volume spike (optional confirmation)
        
        SELL when:
        1. MACD crosses below signal line, OR
        2. MACD drops below zero (exits bullish zone), OR
        3. Histogram turns negative (trend weakening)
        
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
        # Positive histogram = bullish (MACD above signal)
        # Larger histogram = stronger trend
        signals['histogram'] = signals['macd'] - signals['signal_line']
        
        # Previous values for crossover detection
        signals['macd_prev'] = signals['macd'].shift(1)
        signals['signal_line_prev'] = signals['signal_line'].shift(1)
        signals['histogram_prev'] = signals['histogram'].shift(1)
        
        # Volume average for confirmation
        if self.use_volume_confirmation:
            signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        # Track position state
        in_position = False
        
        for i in range(1, len(signals)):
            # === BUY CONDITIONS ===
            if not in_position:
                # 1. Bullish crossover: MACD crosses above signal line
                bullish_crossover = (
                    signals['macd'].iloc[i] > signals['signal_line'].iloc[i] and
                    signals['macd_prev'].iloc[i] <= signals['signal_line_prev'].iloc[i]
                )
                
                # 2. MACD above zero (bullish zone - confirms uptrend)
                if self.require_macd_above_zero:
                    macd_above_zero = signals['macd'].iloc[i] > 0
                else:
                    macd_above_zero = True
                
                # 3. Histogram strength (gap indicates trend strength)
                # Larger gap = stronger trend momentum
                histogram_strong = signals['histogram'].iloc[i] >= self.min_histogram_strength
                
                # 4. Histogram divergence (gap is widening = strengthening trend)
                if self.use_histogram_divergence:
                    histogram_widening = (
                        signals['histogram'].iloc[i] > signals['histogram_prev'].iloc[i]
                    )
                else:
                    histogram_widening = True
                
                # 5. Volume confirmation (optional)
                volume_ok = True
                if self.use_volume_confirmation:
                    volume_ma = signals['volume_ma'].iloc[i]
                    if not pd.isna(volume_ma):
                        volume_ok = data['Volume'].iloc[i] >= (volume_ma * self.volume_threshold)
                
                # Combine all buy conditions
                if (bullish_crossover and macd_above_zero and 
                    histogram_strong and histogram_widening and volume_ok):
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # === SELL CONDITIONS ===
            else:
                # 1. Bearish crossover: MACD crosses below signal line
                bearish_crossover = (
                    signals['macd'].iloc[i] < signals['signal_line'].iloc[i] and
                    signals['macd_prev'].iloc[i] >= signals['signal_line_prev'].iloc[i]
                )
                
                # 2. MACD drops below zero (exits bullish zone)
                macd_drops_below_zero = (
                    signals['macd'].iloc[i] < 0 and
                    signals['macd_prev'].iloc[i] >= 0
                )
                
                # 3. Histogram turns negative (trend is weakening)
                histogram_negative = signals['histogram'].iloc[i] < 0
                
                # Exit on any sell condition
                if bearish_crossover or macd_drops_below_zero or histogram_negative:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]


class AggressiveMACDStrategy(Strategy):
    """
    Aggressive MACD Strategy - Higher Trading Frequency
    
    Relaxed filters for more trading opportunities:
    - Trades in both bullish AND bearish zones (MACD above/below zero)
    - Lower histogram strength requirement
    - No histogram divergence requirement
    - No volume confirmation
    
    Best for: Active traders, volatile stocks, shorter timeframes
    """
    
    def __init__(self):
        """Initialize Aggressive MACD Strategy with relaxed parameters"""
        super().__init__()
        
        self.base_strategy = EnhancedMACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            require_macd_above_zero=False,    # Trade in any zone
            min_histogram_strength=-0.5,      # Very low threshold
            use_histogram_divergence=False,   # No divergence check
            use_volume_confirmation=False,    # No volume filter
            volume_threshold=1.0
        )
        
        self.parameters = self.base_strategy.parameters.copy()
        self.parameters['strategy_type'] = 'Aggressive MACD'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using aggressive parameters"""
        return self.base_strategy.generate_signals(data)


class ConservativeMACDStrategy(Strategy):
    """
    Conservative MACD Strategy - High Quality Signals Only
    
    Strict filters for high-probability setups:
    - MACD must be above zero (bullish zone only)
    - Strong histogram requirement (large gap = strong trend)
    - Histogram must be widening (strengthening trend)
    - Volume confirmation enabled (high volume = institutional interest)
    
    Best for: Risk-averse traders, large-cap stocks, longer timeframes
    """
    
    def __init__(self):
        """Initialize Conservative MACD Strategy with strict parameters"""
        super().__init__()
        
        self.base_strategy = EnhancedMACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            require_macd_above_zero=True,     # Bullish zone only
            min_histogram_strength=1.0,       # Strong histogram required
            use_histogram_divergence=True,    # Must be widening
            use_volume_confirmation=True,     # Volume filter enabled
            volume_threshold=1.5              # High volume requirement
        )
        
        self.parameters = self.base_strategy.parameters.copy()
        self.parameters['strategy_type'] = 'Conservative MACD'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using conservative parameters"""
        return self.base_strategy.generate_signals(data)


class MACDTrendFollowingStrategy(Strategy):
    """
    MACD Trend Following Strategy
    
    Focuses on strong trends with multiple confirmations:
    - MACD well above zero (strong bullish trend)
    - Large histogram (strong momentum)
    - Accelerating histogram (momentum building)
    - High volume (institutional participation)
    
    Best for: Trend traders, momentum traders, growth stocks
    """
    
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        macd_zero_threshold: float = 1.0,    # MACD must be > 1.0 (strong uptrend)
        histogram_strength: float = 2.0,     # Strong histogram
        volume_threshold: float = 1.3
    ):
        """
        Initialize MACD Trend Following Strategy
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            macd_zero_threshold: Minimum MACD value above zero
            histogram_strength: Minimum histogram strength
            volume_threshold: Volume multiplier requirement
        """
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.macd_zero_threshold = macd_zero_threshold
        self.histogram_strength = histogram_strength
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'macd_zero_threshold': macd_zero_threshold,
            'histogram_strength': histogram_strength,
            'volume_threshold': volume_threshold,
            'strategy_type': 'MACD Trend Following'
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for strong trending markets"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate MACD components
        exp1 = data['Close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.slow_period, adjust=False).mean()
        
        signals['macd'] = exp1 - exp2
        signals['signal_line'] = signals['macd'].ewm(span=self.signal_period, adjust=False).mean()
        signals['histogram'] = signals['macd'] - signals['signal_line']
        
        # Volume and previous values
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        signals['macd_prev'] = signals['macd'].shift(1)
        signals['signal_line_prev'] = signals['signal_line'].shift(1)
        signals['histogram_prev'] = signals['histogram'].shift(1)
        
        in_position = False
        
        for i in range(1, len(signals)):
            if not in_position:
                # BUY: Strong uptrend with momentum
                bullish_cross = (
                    signals['macd'].iloc[i] > signals['signal_line'].iloc[i] and
                    signals['macd_prev'].iloc[i] <= signals['signal_line_prev'].iloc[i]
                )
                
                # MACD well above zero (strong trend)
                strong_uptrend = signals['macd'].iloc[i] >= self.macd_zero_threshold
                
                # Large histogram (strong momentum)
                strong_momentum = signals['histogram'].iloc[i] >= self.histogram_strength
                
                # Accelerating (histogram widening)
                accelerating = signals['histogram'].iloc[i] > signals['histogram_prev'].iloc[i]
                
                # Volume confirmation
                volume_ma = signals['volume_ma'].iloc[i]
                volume_ok = True
                if not pd.isna(volume_ma):
                    volume_ok = data['Volume'].iloc[i] >= (volume_ma * self.volume_threshold)
                
                if bullish_cross and strong_uptrend and strong_momentum and accelerating and volume_ok:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            else:
                # SELL: Trend weakening
                bearish_cross = (
                    signals['macd'].iloc[i] < signals['signal_line'].iloc[i] and
                    signals['macd_prev'].iloc[i] >= signals['signal_line_prev'].iloc[i]
                )
                
                # MACD drops significantly
                trend_weakening = signals['macd'].iloc[i] < (self.macd_zero_threshold * 0.5)
                
                # Histogram turning negative
                momentum_lost = signals['histogram'].iloc[i] < 0
                
                if bearish_cross or trend_weakening or momentum_lost:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]
