"""
Inversion FVG (Fair Value Gap) Strategy

Based on ICT (Inner Circle Trader) and Smart Money Concepts (SMC).

FVG (Fair Value Gap): A 3-candle price imbalance where the middle candle moves
so fast that there's no overlapping price between candle 1 and candle 3.

Inversion: When an FVG gets filled and then acts as the opposite side's level:
- Bullish FVG filled → becomes support (demand zone)
- Bearish FVG filled → becomes resistance (supply zone)

Trading Logic:
- Detect FVGs (bullish and bearish)
- Track when FVGs get filled
- Trade inversions: price returns to filled FVG and bounces
- Use volume and momentum confirmation
- ATR-based stop loss
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy
from typing import List, Tuple, Optional, Dict


class InversionFVGStrategy(Strategy):
    """
    Inversion FVG (Fair Value Gap) Trading Strategy
    
    Identifies FVGs, tracks when they're filled, and trades inversions
    when price returns to these levels.
    """
    
    def __init__(
        self,
        fvg_min_gap: float = 0.005,  # Minimum 0.5% gap to be valid FVG
        inversion_tolerance: float = 0.01,  # 1% tolerance for inversion touch
        lookback_bars: int = 50,  # How many bars back to track FVGs
        use_volume_confirmation: bool = True,
        volume_threshold: float = 1.2,  # 20% above average
        use_momentum_confirmation: bool = True,
        rsi_period: int = 14,
        rsi_bullish_min: float = 40,  # RSI must be curling up from here
        rsi_bearish_max: float = 60,  # RSI must be curling down from here
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        max_fvg_age: int = 100,  # Maximum bars to keep tracking an FVG
    ):
        """
        Initialize Inversion FVG Strategy
        
        Args:
            fvg_min_gap: Minimum gap size (%) to qualify as FVG
            inversion_tolerance: Price tolerance for inversion level touch
            lookback_bars: Bars to look back for FVG detection
            use_volume_confirmation: Require volume spike
            volume_threshold: Volume multiplier vs average
            use_momentum_confirmation: Use RSI for momentum confirmation
            rsi_period: RSI calculation period
            rsi_bullish_min: Minimum RSI for bullish entry
            rsi_bearish_max: Maximum RSI for bearish exit
            atr_period: ATR period for stop loss
            atr_multiplier: ATR multiplier for stop loss
            max_fvg_age: Maximum age (bars) to track FVG
        """
        super().__init__()
        self.fvg_min_gap = fvg_min_gap
        self.inversion_tolerance = inversion_tolerance
        self.lookback_bars = lookback_bars
        self.use_volume_confirmation = use_volume_confirmation
        self.volume_threshold = volume_threshold
        self.use_momentum_confirmation = use_momentum_confirmation
        self.rsi_period = rsi_period
        self.rsi_bullish_min = rsi_bullish_min
        self.rsi_bearish_max = rsi_bearish_max
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.max_fvg_age = max_fvg_age
        
        self.parameters = {
            'fvg_min_gap': fvg_min_gap,
            'inversion_tolerance': inversion_tolerance,
            'lookback_bars': lookback_bars,
            'volume_confirmation': use_volume_confirmation,
            'volume_threshold': volume_threshold,
            'momentum_confirmation': use_momentum_confirmation,
            'rsi_period': rsi_period,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
            'max_fvg_age': max_fvg_age,
        }
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        close = data['Close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
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
    
    def _detect_fvg(
        self,
        candle1_high: float,
        candle1_low: float,
        candle2_high: float,
        candle2_low: float,
        candle2_close: float,
        candle3_high: float,
        candle3_low: float,
    ) -> Optional[Dict]:
        """
        Detect Fair Value Gap
        
        Bullish FVG: High of candle 1 < Low of candle 3 (gap up)
        Bearish FVG: Low of candle 1 > High of candle 3 (gap down)
        
        Returns:
            Dictionary with FVG info or None
        """
        # Bullish FVG: Gap between C1 high and C3 low
        if candle1_high < candle3_low:
            gap_size = (candle3_low - candle1_high) / candle1_high
            
            if gap_size >= self.fvg_min_gap:
                return {
                    'type': 'bullish',
                    'upper': candle3_low,
                    'lower': candle1_high,
                    'gap_size': gap_size,
                    'filled': False,
                    'inverted': False,
                    'mid': (candle3_low + candle1_high) / 2,
                }
        
        # Bearish FVG: Gap between C1 low and C3 high
        elif candle1_low > candle3_high:
            gap_size = (candle1_low - candle3_high) / candle3_high
            
            if gap_size >= self.fvg_min_gap:
                return {
                    'type': 'bearish',
                    'upper': candle1_low,
                    'lower': candle3_high,
                    'gap_size': gap_size,
                    'filled': False,
                    'inverted': False,
                    'mid': (candle1_low + candle3_high) / 2,
                }
        
        return None
    
    def _is_fvg_filled(self, fvg: Dict, candle_high: float, candle_low: float) -> bool:
        """
        Check if FVG has been filled
        
        Bullish FVG filled: Price comes back down and touches the gap
        Bearish FVG filled: Price comes back up and touches the gap
        """
        if fvg['filled']:
            return True
        
        if fvg['type'] == 'bullish':
            # Price must come back down into the gap
            if candle_low <= fvg['mid']:
                return True
        
        elif fvg['type'] == 'bearish':
            # Price must come back up into the gap
            if candle_high >= fvg['mid']:
                return True
        
        return False
    
    def _check_inversion_touch(
        self,
        fvg: Dict,
        candle_high: float,
        candle_low: float,
        candle_close: float,
    ) -> bool:
        """
        Check if price is touching the inversion level
        
        Bullish FVG (now support): Price comes down to it
        Bearish FVG (now resistance): Price comes up to it
        """
        if not fvg['filled'] or not fvg['inverted']:
            return False
        
        fvg_mid = fvg['mid']
        tolerance = fvg_mid * self.inversion_tolerance
        
        if fvg['type'] == 'bullish':
            # After filling, bullish FVG becomes support (demand zone)
            # Look for price coming down to test it
            if candle_low <= (fvg_mid + tolerance) and candle_close >= (fvg_mid - tolerance):
                return True
        
        elif fvg['type'] == 'bearish':
            # After filling, bearish FVG becomes resistance (supply zone)
            # Look for price coming up to test it
            if candle_high >= (fvg_mid - tolerance) and candle_close <= (fvg_mid + tolerance):
                return True
        
        return False
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Inversion FVGs
        
        Logic:
        1. Detect FVGs (3-candle imbalances)
        2. Track when FVGs get filled
        3. Mark filled FVGs as "inverted" (potential S/R)
        4. Buy when price bounces off bullish inversion FVG (support)
        5. Sell when price rejects at bearish inversion FVG (resistance)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        # Calculate indicators
        signals['rsi'] = self._calculate_rsi(data, self.rsi_period)
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        # Track active FVGs
        active_fvgs: List[Dict] = []
        
        # Track position
        in_position = False
        entry_fvg = None
        
        # Start from bar 3 (need 3 candles for FVG)
        for i in range(3, len(data)):
            # === STEP 1: Detect new FVGs ===
            if i >= 3:
                candle1_idx = i - 2
                candle2_idx = i - 1
                candle3_idx = i
                
                fvg = self._detect_fvg(
                    candle1_high=data['High'].iloc[candle1_idx],
                    candle1_low=data['Low'].iloc[candle1_idx],
                    candle2_high=data['High'].iloc[candle2_idx],
                    candle2_low=data['Low'].iloc[candle2_idx],
                    candle2_close=data['Close'].iloc[candle2_idx],
                    candle3_high=data['High'].iloc[candle3_idx],
                    candle3_low=data['Low'].iloc[candle3_idx],
                )
                
                if fvg is not None:
                    fvg['created_bar'] = i
                    active_fvgs.append(fvg)
            
            # === STEP 2: Update active FVGs (check if filled) ===
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_close = data['Close'].iloc[i]
            
            for fvg in active_fvgs:
                if not fvg['filled']:
                    if self._is_fvg_filled(fvg, current_high, current_low):
                        fvg['filled'] = True
                        fvg['filled_bar'] = i
                        fvg['inverted'] = True  # Now it can act as S/R
            
            # === STEP 3: Clean up old FVGs ===
            active_fvgs = [
                fvg for fvg in active_fvgs
                if i - fvg['created_bar'] <= self.max_fvg_age
            ]
            
            # === STEP 4: Check for valid indicators ===
            current_rsi = signals['rsi'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            
            if pd.isna(current_rsi) or pd.isna(current_atr) or pd.isna(avg_volume):
                continue
            
            # Volume confirmation
            volume_ok = True
            if self.use_volume_confirmation:
                volume_ok = current_volume >= (avg_volume * self.volume_threshold)
            
            # RSI momentum
            rsi_prev = signals['rsi'].iloc[i-1] if i > 0 else 50
            rsi_curling_up = current_rsi > rsi_prev and current_rsi >= self.rsi_bullish_min
            rsi_curling_down = current_rsi < rsi_prev and current_rsi <= self.rsi_bearish_max
            
            # === STEP 5: Generate signals from inversion FVGs ===
            
            # Check all inverted FVGs
            for fvg in active_fvgs:
                if not fvg['inverted']:
                    continue
                
                is_touching = self._check_inversion_touch(
                    fvg, current_high, current_low, current_close
                )
                
                if not is_touching:
                    continue
                
                # BUY: Bullish inversion FVG (support/demand zone)
                if fvg['type'] == 'bullish' and not in_position:
                    momentum_ok = True
                    if self.use_momentum_confirmation:
                        momentum_ok = rsi_curling_up
                    
                    if volume_ok and momentum_ok:
                        # Price bounced off the inverted bullish FVG (now support)
                        if current_close > fvg['lower']:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            # Stop loss below the FVG zone
                            stop_price = fvg['lower'] - (current_atr * self.atr_multiplier)
                            signals.iloc[i, signals.columns.get_loc('stop_price')] = stop_price
                            in_position = True
                            entry_fvg = fvg
                            break  # Only one signal per bar
                
                # SELL: Bearish inversion FVG (resistance/supply zone)
                elif fvg['type'] == 'bearish' and in_position:
                    momentum_ok = True
                    if self.use_momentum_confirmation:
                        momentum_ok = rsi_curling_down
                    
                    # Price rejected at inverted bearish FVG (now resistance)
                    if current_close < fvg['upper'] or momentum_ok:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                        entry_fvg = None
                        break  # Only one signal per bar
            
            # === STEP 6: Stop loss check ===
            if in_position and i > 0:
                stop_price = signals['stop_price'].iloc[i-1]
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_fvg = None
        
        return signals[['signal', 'stop_price']]


class AggressiveInversionFVGStrategy(Strategy):
    """
    Aggressive Inversion FVG Strategy
    
    More sensitive parameters for higher trading frequency.
    """
    
    def __init__(self):
        """Initialize Aggressive Inversion FVG Strategy"""
        super().__init__()
        
        self.base_strategy = InversionFVGStrategy(
            fvg_min_gap=0.003,  # Smaller 0.3% gap threshold
            inversion_tolerance=0.015,  # Wider 1.5% tolerance
            lookback_bars=30,  # Shorter lookback
            use_volume_confirmation=True,
            volume_threshold=1.1,  # Lower volume requirement
            use_momentum_confirmation=True,
            rsi_period=14,
            rsi_bullish_min=35,  # Lower RSI threshold
            rsi_bearish_max=65,  # Higher RSI threshold
            atr_period=10,
            atr_multiplier=1.5,  # Tighter stops
            max_fvg_age=50,  # Track for less time
        )
        
        self.parameters = self.base_strategy.parameters.copy()
        self.parameters['strategy_type'] = 'Aggressive Inversion FVG'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using aggressive parameters"""
        return self.base_strategy.generate_signals(data)


class ConservativeInversionFVGStrategy(Strategy):
    """
    Conservative Inversion FVG Strategy
    
    Stricter parameters for higher quality setups.
    """
    
    def __init__(self):
        """Initialize Conservative Inversion FVG Strategy"""
        super().__init__()
        
        self.base_strategy = InversionFVGStrategy(
            fvg_min_gap=0.008,  # Larger 0.8% gap threshold
            inversion_tolerance=0.008,  # Tighter tolerance
            lookback_bars=100,  # Longer lookback
            use_volume_confirmation=True,
            volume_threshold=1.5,  # Higher volume requirement
            use_momentum_confirmation=True,
            rsi_period=14,
            rsi_bullish_min=45,  # Higher RSI threshold
            rsi_bearish_max=55,  # Lower RSI threshold
            atr_period=14,
            atr_multiplier=2.5,  # Wider stops
            max_fvg_age=150,  # Track for longer
        )
        
        self.parameters = self.base_strategy.parameters.copy()
        self.parameters['strategy_type'] = 'Conservative Inversion FVG'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using conservative parameters"""
        return self.base_strategy.generate_signals(data)

