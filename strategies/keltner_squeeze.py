"""
Keltner Channel Squeeze Strategy

Primary Indicator: Keltner Channels (volatility-based bands)
Confirmation Indicators: Bollinger Band Width + Momentum Oscillator

Buy Signal:
  - Volatility squeeze detected (Bollinger Bands inside Keltner Channels)
  - Price breaks out above Keltner Channel upper band
  - Momentum oscillator confirms upward direction
  - Volume surge on breakout
  
Sell Signal:
  - Price breaks below Keltner Channel lower band
  - OR momentum reverses significantly
  - OR volatility expands to extreme levels (profit taking)

Strategy Type: Breakout/Volatility - Catch explosive moves after consolidation
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class KeltnerSqueezeStrategy(Strategy):
    """
    Aggressive breakout strategy using Keltner Channel squeeze detection
    
    The "squeeze" occurs when volatility contracts (Bollinger Bands narrow and move
    inside Keltner Channels). This compression often precedes explosive moves.
    We enter aggressively on the breakout with momentum confirmation.
    """
    
    def __init__(
        self,
        kc_period: int = 20,
        kc_atr_multiplier: float = 2.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        momentum_period: int = 10,  # Rate of change period
        momentum_threshold: float = 1.0,  # Minimum % momentum
        volume_ma_period: int = 20,
        volume_threshold: float = 1.3,  # Volume 30% above average on breakout
        squeeze_threshold: float = 0.95  # BB width must be < 95% of KC width
    ):
        """
        Initialize Keltner Squeeze Strategy
        
        Args:
            kc_period: Period for Keltner Channel EMA
            kc_atr_multiplier: ATR multiplier for Keltner bands
            bb_period: Period for Bollinger Bands
            bb_std: Standard deviations for Bollinger Bands
            momentum_period: Period for momentum calculation
            momentum_threshold: Minimum momentum % for entry
            volume_ma_period: Period for volume average
            volume_threshold: Volume multiplier for breakout confirmation
            squeeze_threshold: BB width / KC width ratio for squeeze detection
        """
        super().__init__()
        self.kc_period = kc_period
        self.kc_atr_multiplier = kc_atr_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.squeeze_threshold = squeeze_threshold
        
        self.parameters = {
            'kc_period': kc_period,
            'kc_atr_multiplier': kc_atr_multiplier,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'momentum_period': momentum_period,
            'momentum_threshold': momentum_threshold,
            'volume_ma_period': volume_ma_period,
            'volume_threshold': volume_threshold,
            'squeeze_threshold': squeeze_threshold
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_keltner_channels(self, data: pd.DataFrame) -> tuple:
        """
        Calculate Keltner Channels
        
        Keltner Channels use ATR (volatility) to create bands around an EMA
        - Middle: EMA of typical price
        - Upper: EMA + (ATR * multiplier)
        - Lower: EMA - (ATR * multiplier)
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        middle = typical_price.ewm(span=self.kc_period, adjust=False).mean()
        
        atr = self._calculate_atr(data, self.kc_period)
        
        upper = middle + (atr * self.kc_atr_multiplier)
        lower = middle - (atr * self.kc_atr_multiplier)
        
        return upper, middle, lower
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> tuple:
        """
        Calculate Bollinger Bands
        
        Bollinger Bands use standard deviation to measure volatility
        - Middle: SMA
        - Upper: SMA + (std * multiplier)
        - Lower: SMA - (std * multiplier)
        """
        close = data['Close']
        middle = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return upper, middle, lower
    
    def _detect_squeeze(self, bb_upper: float, bb_lower: float, 
                       kc_upper: float, kc_lower: float) -> bool:
        """
        Detect volatility squeeze
        
        Squeeze occurs when Bollinger Bands are inside Keltner Channels
        This indicates very low volatility - a calm before the storm
        """
        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(kc_upper) or pd.isna(kc_lower):
            return False
        
        # Calculate band widths
        bb_width = bb_upper - bb_lower
        kc_width = kc_upper - kc_lower
        
        if kc_width == 0:
            return False
        
        # Squeeze: BB width is smaller than KC width
        width_ratio = bb_width / kc_width
        
        return width_ratio < self.squeeze_threshold
    
    def _calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate momentum (Rate of Change)
        
        Positive momentum = upward price movement
        Negative momentum = downward price movement
        """
        momentum = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return momentum
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Keltner Squeeze
        
        Buy: Squeeze detected + Breakout above KC + Momentum + Volume
        Sell: Breakout fails OR momentum reverses OR extreme volatility
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['kc_upper'], signals['kc_middle'], signals['kc_lower'] = \
            self._calculate_keltner_channels(data)
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = \
            self._calculate_bollinger_bands(data)
        signals['momentum'] = self._calculate_momentum(data['Close'], self.momentum_period)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        # Track squeeze state
        signals['squeeze'] = False
        for i in range(len(data)):
            if i < max(self.kc_period, self.bb_period):
                continue
            
            squeeze = self._detect_squeeze(
                signals['bb_upper'].iloc[i],
                signals['bb_lower'].iloc[i],
                signals['kc_upper'].iloc[i],
                signals['kc_lower'].iloc[i]
            )
            signals.iloc[i, signals.columns.get_loc('squeeze')] = squeeze
        
        in_position = False
        squeeze_active = False
        entry_price = None
        
        for i in range(len(data)):
            # Skip until we have enough data
            if i < max(self.kc_period, self.bb_period, self.momentum_period, self.volume_ma_period):
                continue
            
            current_price = data['Close'].iloc[i]
            kc_upper = signals['kc_upper'].iloc[i]
            kc_lower = signals['kc_lower'].iloc[i]
            kc_middle = signals['kc_middle'].iloc[i]
            momentum = signals['momentum'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            squeeze = signals['squeeze'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(kc_upper) or pd.isna(momentum) or pd.isna(volume_ma):
                continue
            
            # Track squeeze state
            if squeeze:
                squeeze_active = True
            
            # BUY SIGNAL: Breakout from squeeze with momentum and volume
            if not in_position:
                # Must have had a recent squeeze (within last 5 bars)
                recent_squeeze = squeeze_active or any(signals['squeeze'].iloc[max(0, i-5):i])
                
                # Bullish breakout: price breaks above upper Keltner Channel
                bullish_breakout = current_price > kc_upper
                
                # Momentum confirmation: strong upward momentum
                strong_momentum = momentum > self.momentum_threshold
                
                # Volume surge on breakout
                volume_surge = volume > (volume_ma * self.volume_threshold)
                
                # Entry condition: all factors align
                if recent_squeeze and bullish_breakout and strong_momentum and volume_surge:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
                    squeeze_active = False  # Reset squeeze tracking
                    entry_price = current_price
            
            # SELL SIGNAL: Breakout failure or momentum reversal
            else:
                # Exit 1: Price falls back below middle Keltner (failed breakout)
                breakout_failed = current_price < kc_middle
                
                # Exit 2: Momentum reverses significantly
                momentum_reversal = momentum < -self.momentum_threshold
                
                # Exit 3: Price drops below lower Keltner (bearish breakout)
                bearish_breakout = current_price < kc_lower
                
                # Exit 4: Take profit - volatility expands too much (optional)
                # Calculate band width for extreme volatility check
                kc_width = kc_upper - kc_lower
                bb_width = signals['bb_upper'].iloc[i] - signals['bb_lower'].iloc[i]
                if pd.notna(kc_width) and kc_width > 0:
                    extreme_volatility = (bb_width / kc_width) > 1.5  # BB much wider than KC
                else:
                    extreme_volatility = False
                
                # Exit conditions
                if breakout_failed or momentum_reversal or bearish_breakout:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_price = None
                elif extreme_volatility and entry_price is not None:
                    # Take profit if we've moved significantly
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    if profit_pct > 3.0:  # 3% profit with extreme volatility
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                        entry_price = None
        
        return signals[['signal']]


class AggressiveSqueezeStrategy(Strategy):
    """
    Ultra-aggressive version - trades any volatility contraction/expansion
    
    Entry: Narrow bands + any breakout direction with volume
    Exit: Opposite direction move or quick profit target
    """
    
    def __init__(
        self,
        kc_period: int = 15,
        kc_atr_multiplier: float = 1.5,  # Tighter bands
        bb_period: int = 15,
        bb_std: float = 1.5,  # Tighter bands
        momentum_period: int = 5,  # Faster momentum
        volume_ma_period: int = 10,
        volume_threshold: float = 1.15  # Just 15% above average
    ):
        super().__init__()
        self.kc_period = kc_period
        self.kc_atr_multiplier = kc_atr_multiplier
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.momentum_period = momentum_period
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'kc_period': kc_period,
            'kc_atr_multiplier': kc_atr_multiplier,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'momentum_period': momentum_period,
            'volume_ma_period': volume_ma_period,
            'volume_threshold': volume_threshold
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_keltner_channels(self, data: pd.DataFrame) -> tuple:
        """Calculate Keltner Channels"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        middle = typical_price.ewm(span=self.kc_period, adjust=False).mean()
        
        atr = self._calculate_atr(data, self.kc_period)
        
        upper = middle + (atr * self.kc_atr_multiplier)
        lower = middle - (atr * self.kc_atr_multiplier)
        
        return upper, middle, lower
    
    def _calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate momentum"""
        momentum = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return momentum
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra-aggressive squeeze breakout signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        signals['kc_upper'], signals['kc_middle'], signals['kc_lower'] = \
            self._calculate_keltner_channels(data)
        signals['momentum'] = self._calculate_momentum(data['Close'], self.momentum_period)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        in_position = False
        
        for i in range(len(data)):
            if i < max(self.kc_period, self.momentum_period, self.volume_ma_period):
                continue
            
            current_price = data['Close'].iloc[i]
            kc_upper = signals['kc_upper'].iloc[i]
            kc_lower = signals['kc_lower'].iloc[i]
            kc_middle = signals['kc_middle'].iloc[i]
            momentum = signals['momentum'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            if pd.isna(kc_upper) or pd.isna(momentum) or pd.isna(volume_ma):
                continue
            
            # BUY: Any breakout above upper band with volume
            if not in_position:
                breakout = current_price > kc_upper
                volume_ok = volume > (volume_ma * self.volume_threshold)
                momentum_positive = momentum > 0
                
                if breakout and volume_ok and momentum_positive:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Quick exit on reversal or target
            else:
                below_middle = current_price < kc_middle
                momentum_negative = momentum < 0
                
                if below_middle or momentum_negative:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

