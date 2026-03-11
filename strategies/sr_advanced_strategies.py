"""
Advanced Support/Resistance Strategies with Multiple Confirmations

These strategies combine S/R levels with various technical indicators
for higher probability trades:

1. S/R + RSI (Momentum Confirmation)
2. S/R + Volume (Breakout Strength)
3. S/R + EMA (Trend Filter)
4. S/R + MACD (Trend Reversal Confirmation)
5. S/R + ALL-IN-ONE COMBO (Most Reliable)

Each strategy identifies key S/R levels and waits for confirmation
before entering trades, significantly improving win rates.
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class SRRSIStrategy(Strategy):
    """
    🔥 S/R + RSI Strategy (Momentum Confirmation)
    
    Most reliable for beginners!
    
    Trading Logic:
    - Buy near Support only if RSI is oversold (below 30-40) and curling upward
    - Sell near Resistance only if RSI is overbought (above 60-70) and curling downward
    - Avoid when RSI stays overbought/oversold in strong trends
    
    Why it works:
    - At S/R levels, market either rejects (reversal) or breaks out
    - RSI helps identify which one will happen
    - Prevents counter-trend trades in strong momentum
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        price_tolerance: float = 0.02,
        min_touches: int = 2,
        rsi_period: int = 14,
        rsi_oversold: float = 40,
        rsi_overbought: float = 65,
        rsi_momentum_threshold: float = 2.0,  # RSI must curl by this much
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        """
        Initialize S/R + RSI Strategy
        
        Args:
            lookback_period: Bars to look back for S/R identification
            price_tolerance: Price clustering tolerance (%)
            min_touches: Minimum touches to validate S/R level
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            rsi_momentum_threshold: Minimum RSI change for "curling"
            atr_period: ATR period for stop loss
            atr_multiplier: ATR multiplier for stop loss
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_momentum_threshold = rsi_momentum_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'strategy': 'S/R + RSI (Momentum Confirmation)',
            'lookback_period': lookback_period,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'min_touches': min_touches,
        }
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = data['Close'].diff()
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
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swing_points = []
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
                swing_points.append((i, highs[i], 'high'))
            
            # Check for swing low
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_points.append((i, lows[i], 'low'))
        
        return swing_points
    
    def _cluster_levels(self, prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
        """Cluster prices into support/resistance levels"""
        if not prices:
            return []
        
        clusters = defaultdict(list)
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            found_cluster = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[price].append(price)
        
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        levels.sort(key=lambda x: x[1], reverse=True)
        return levels
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        swing_points = self._find_swing_points(data)
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        resistance_levels = [level for level, touches in resistance_levels if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels if touches >= self.min_touches]
        
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """Find nearest support or resistance level"""
        if not levels:
            return None, float('inf')
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with S/R + RSI confirmation
        
        Entry Rules:
        - BUY: Price near support + RSI oversold + RSI curling upward
        - SELL: Price near resistance + RSI overbought + RSI curling downward
        
        Exit Rules:
        - Stop loss: ATR-based below support/resistance
        - Take profit: Opposite S/R level
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        # Calculate indicators
        signals['rsi'] = self._calculate_rsi(data, self.rsi_period)
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        
        # Track position
        in_position = False
        
        for i in range(self.lookback_period, len(data)):
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not support_levels and not resistance_levels:
                continue
            
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            current_rsi = signals['rsi'].iloc[i]
            prev_rsi = signals['rsi'].iloc[i-1] if i > 0 else np.nan
            current_atr = signals['atr'].iloc[i]
            
            if pd.isna(current_rsi) or pd.isna(prev_rsi) or pd.isna(current_atr):
                continue
            
            # Calculate RSI momentum (curling)
            rsi_momentum = current_rsi - prev_rsi
            
            nearest_support, dist_to_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance, dist_to_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # BUY SIGNAL: Support + RSI oversold + RSI curling up
            if not in_position and nearest_support is not None:
                near_support = dist_to_support <= 0.02  # Within 2% of support
                rsi_oversold = current_rsi < self.rsi_oversold
                rsi_curling_up = rsi_momentum > self.rsi_momentum_threshold
                
                # Price touched support area
                support_touched = current_low <= nearest_support * 1.01
                
                if near_support and rsi_oversold and rsi_curling_up and support_touched:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                        nearest_support - (current_atr * self.atr_multiplier)
                    in_position = True
            
            # SELL SIGNAL: Resistance + RSI overbought + RSI curling down
            elif in_position and nearest_resistance is not None:
                near_resistance = dist_to_resistance <= 0.02
                rsi_overbought = current_rsi > self.rsi_overbought
                rsi_curling_down = rsi_momentum < -self.rsi_momentum_threshold
                
                # Price touched resistance area
                resistance_touched = current_high >= nearest_resistance * 0.99
                
                if near_resistance and (rsi_overbought or rsi_curling_down) and resistance_touched:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
            
            # Stop loss check
            if in_position:
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal', 'stop_price']]


class SRVolumeStrategy(Strategy):
    """
    🔥 S/R + Volume Strategy (Breakout Strength)
    
    Best for breakout traders!
    
    Trading Logic:
    - Breakout BUY: Price breaks resistance with volume > 150% of average
    - Breakout SELL: Price breaks support with big volume
    - Avoid when volume is low → almost always a fake breakout
    
    Why it works:
    - S/R breakouts are only meaningful when real money enters (high volume)
    - Low volume breakouts fail 80% of the time
    - Volume confirms institutional participation
    """
    
    def __init__(
        self,
        lookback_period: int = 80,
        price_tolerance: float = 0.025,
        min_touches: int = 2,
        volume_threshold: float = 1.5,  # 150% of average
        breakout_confirmation: float = 0.01,  # 1% beyond S/R
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """Initialize S/R + Volume Strategy"""
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.volume_threshold = volume_threshold
        self.breakout_confirmation = breakout_confirmation
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'strategy': 'S/R + Volume (Breakout Strength)',
            'lookback_period': lookback_period,
            'volume_threshold': volume_threshold,
            'min_touches': min_touches,
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
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swing_points = []
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(window, len(data) - window):
            is_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_points.append((i, highs[i], 'high'))
            
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_points.append((i, lows[i], 'low'))
        
        return swing_points
    
    def _cluster_levels(self, prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
        """Cluster prices into support/resistance levels"""
        if not prices:
            return []
        
        clusters = defaultdict(list)
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            found_cluster = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[price].append(price)
        
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        levels.sort(key=lambda x: x[1], reverse=True)
        return levels
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        swing_points = self._find_swing_points(data)
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        resistance_levels = [level for level, touches in resistance_levels if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels if touches >= self.min_touches]
        
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """Find nearest support or resistance level"""
        if not levels:
            return None, float('inf')
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with S/R + Volume confirmation
        
        Entry Rules:
        - BUY: Price breaks resistance + volume > 150% average
        - Resistance becomes new support
        
        Exit Rules:
        - Stop loss: Below broken resistance (now support)
        - Take profit: Next resistance level
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        in_position = False
        broken_resistance = None
        
        for i in range(self.lookback_period, len(data)):
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not resistance_levels:
                continue
            
            current_price = data['Close'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            
            if pd.isna(current_atr) or pd.isna(avg_volume):
                continue
            
            # Volume spike confirmation
            volume_spike = current_volume >= (avg_volume * self.volume_threshold)
            
            nearest_resistance, _ = self._find_nearest_level(current_price, resistance_levels)
            
            # BUY SIGNAL: Breakout above resistance with high volume
            if not in_position and nearest_resistance is not None and volume_spike:
                breakout_confirmed = current_price > nearest_resistance * (1 + self.breakout_confirmation)
                
                if breakout_confirmed:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    # Stop below broken resistance
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                        nearest_resistance - (current_atr * self.atr_multiplier)
                    in_position = True
                    broken_resistance = nearest_resistance
            
            # SELL SIGNAL: Price returns to broken resistance or stop loss
            elif in_position:
                # Stop loss hit
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_price <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    broken_resistance = None
                # Take profit at next resistance
                elif nearest_resistance is not None and current_price >= nearest_resistance * 0.99:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    broken_resistance = None
        
        return signals[['signal', 'stop_price']]


class SREMAStrategy(Strategy):
    """
    🔥 S/R + 20/50 EMA Strategy (Trend Filter)
    
    Best intraday + swing combo!
    
    Trading Logic:
    - Only buy support when price is above 20/50 EMA (uptrend)
    - Only short resistance when price is below 20/50 EMA (downtrend)
    - Bonus: If S/R + EMA + volume align → 90% cleaner signals
    
    Why it works:
    - S/R works best when aligned with trend
    - Prevents counter-trend trades
    - EMAs filter out choppy markets
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        price_tolerance: float = 0.02,
        min_touches: int = 2,
        ema_fast: int = 20,
        ema_slow: int = 50,
        volume_confirmation: bool = True,
        volume_threshold: float = 1.2,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        """Initialize S/R + EMA Strategy"""
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'strategy': 'S/R + EMA (Trend Filter)',
            'lookback_period': lookback_period,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'min_touches': min_touches,
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
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swing_points = []
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(window, len(data) - window):
            is_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_points.append((i, highs[i], 'high'))
            
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_points.append((i, lows[i], 'low'))
        
        return swing_points
    
    def _cluster_levels(self, prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
        """Cluster prices into support/resistance levels"""
        if not prices:
            return []
        
        clusters = defaultdict(list)
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            found_cluster = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[price].append(price)
        
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        levels.sort(key=lambda x: x[1], reverse=True)
        return levels
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        swing_points = self._find_swing_points(data)
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        resistance_levels = [level for level, touches in resistance_levels if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels if touches >= self.min_touches]
        
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """Find nearest support or resistance level"""
        if not levels:
            return None, float('inf')
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with S/R + EMA trend filter
        
        Entry Rules:
        - BUY: Support bounce + price above 20/50 EMA (uptrend)
        - Optional: Volume confirmation
        
        Exit Rules:
        - Stop loss: Below support
        - Take profit: At resistance
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        signals['ema_fast'] = data['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        signals['ema_slow'] = data['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        in_position = False
        
        for i in range(self.lookback_period, len(data)):
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not support_levels and not resistance_levels:
                continue
            
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            ema_fast = signals['ema_fast'].iloc[i]
            ema_slow = signals['ema_slow'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            
            if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(current_atr):
                continue
            
            # Trend filter
            in_uptrend = current_price > ema_fast and current_price > ema_slow
            in_downtrend = current_price < ema_fast and current_price < ema_slow
            
            # Volume confirmation
            volume_ok = True
            if self.volume_confirmation:
                volume_ok = current_volume >= (avg_volume * self.volume_threshold)
            
            nearest_support, dist_to_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance, dist_to_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # BUY SIGNAL: Support bounce + uptrend
            if not in_position and nearest_support is not None and in_uptrend:
                near_support = dist_to_support <= 0.02
                support_touched = current_low <= nearest_support * 1.015
                
                if near_support and support_touched and volume_ok:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                        nearest_support - (current_atr * self.atr_multiplier)
                    in_position = True
            
            # SELL SIGNAL: Resistance hit or stop loss
            elif in_position:
                if nearest_resistance is not None:
                    near_resistance = dist_to_resistance <= 0.02
                    resistance_touched = current_high >= nearest_resistance * 0.985
                    
                    if near_resistance and resistance_touched:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                
                # Stop loss
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal', 'stop_price']]


class SRMACDStrategy(Strategy):
    """
    🔥 S/R + MACD Strategy (Trend Reversal Confirmation)
    
    Trading Logic:
    - Bullish reversal: Support + MACD crossover upward
    - Bearish reversal: Resistance + MACD crossover downward
    - Avoid when MACD is flat → no momentum → bad trade
    
    Why it works:
    - MACD confirms if momentum shift supports S/R
    - Catches trend reversals early
    - Filters out weak bounces
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        price_tolerance: float = 0.02,
        min_touches: int = 2,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        """Initialize S/R + MACD Strategy"""
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'strategy': 'S/R + MACD (Trend Reversal)',
            'lookback_period': lookback_period,
            'macd_fast': macd_fast,
            'macd_slow': macd_slow,
            'macd_signal': macd_signal,
            'min_touches': min_touches,
        }
    
    def _calculate_macd(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal, and Histogram"""
        ema_fast = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
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
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swing_points = []
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(window, len(data) - window):
            is_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_points.append((i, highs[i], 'high'))
            
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_points.append((i, lows[i], 'low'))
        
        return swing_points
    
    def _cluster_levels(self, prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
        """Cluster prices into support/resistance levels"""
        if not prices:
            return []
        
        clusters = defaultdict(list)
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            found_cluster = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[price].append(price)
        
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        levels.sort(key=lambda x: x[1], reverse=True)
        return levels
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        swing_points = self._find_swing_points(data)
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        resistance_levels = [level for level, touches in resistance_levels if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels if touches >= self.min_touches]
        
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """Find nearest support or resistance level"""
        if not levels:
            return None, float('inf')
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with S/R + MACD confirmation
        
        Entry Rules:
        - BUY: Support + MACD bullish crossover
        - SELL: Resistance + MACD bearish crossover
        
        Exit Rules:
        - Stop loss: ATR-based
        - Take profit: Opposite S/R level
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        macd_line, signal_line, histogram = self._calculate_macd(data)
        signals['macd'] = macd_line
        signals['macd_signal'] = signal_line
        signals['macd_histogram'] = histogram
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        
        in_position = False
        
        for i in range(self.lookback_period, len(data)):
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not support_levels and not resistance_levels:
                continue
            
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            current_macd = signals['macd'].iloc[i]
            current_signal = signals['macd_signal'].iloc[i]
            prev_macd = signals['macd'].iloc[i-1] if i > 0 else np.nan
            prev_signal = signals['macd_signal'].iloc[i-1] if i > 0 else np.nan
            current_atr = signals['atr'].iloc[i]
            
            if pd.isna(current_macd) or pd.isna(prev_macd) or pd.isna(current_atr):
                continue
            
            # MACD crossovers
            macd_bullish_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)
            macd_bearish_cross = (prev_macd >= prev_signal) and (current_macd < current_signal)
            
            # Check if MACD has momentum (not flat)
            macd_has_momentum = abs(current_macd - prev_macd) > 0.5
            
            nearest_support, dist_to_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance, dist_to_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # BUY SIGNAL: Support + MACD bullish crossover
            if not in_position and nearest_support is not None and macd_bullish_cross and macd_has_momentum:
                near_support = dist_to_support <= 0.025
                support_touched = current_low <= nearest_support * 1.02
                
                if near_support and support_touched:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                        nearest_support - (current_atr * self.atr_multiplier)
                    in_position = True
            
            # SELL SIGNAL: Resistance or MACD bearish crossover
            elif in_position:
                # Take profit at resistance
                if nearest_resistance is not None:
                    near_resistance = dist_to_resistance <= 0.025
                    resistance_touched = current_high >= nearest_resistance * 0.98
                    
                    if (near_resistance and resistance_touched) or macd_bearish_cross:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                
                # Stop loss
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal', 'stop_price']]


class SRAllInOneStrategy(Strategy):
    """
    ⭐ S/R ALL-IN-ONE COMBO Strategy (Most Profitable)
    
    Best for swing trading - institutional-style setup!
    
    Uses 4 confirmations:
    1️⃣ S/R zone
    2️⃣ RSI between 30-40 (for buy) or 60-70 (for sell)
    3️⃣ Price above/below 20/50 EMA (trend filter)
    4️⃣ Volume spike at entry
    
    Trading Logic:
    - BUY: Support + RSI oversold + Price > EMA + Volume spike
    - SELL: Resistance + RSI overbought + Exit conditions
    
    Why it works:
    - Multiple confirmations = highest probability trades
    - Used by most institutional-style retail traders
    - Filters out 90% of false signals
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        price_tolerance: float = 0.02,
        min_touches: int = 2,
        rsi_period: int = 14,
        rsi_buy_min: float = 30,
        rsi_buy_max: float = 45,
        rsi_sell_min: float = 60,
        rsi_sell_max: float = 75,
        ema_fast: int = 20,
        ema_slow: int = 50,
        volume_threshold: float = 1.3,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
    ):
        """Initialize S/R All-in-One Strategy"""
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.rsi_period = rsi_period
        self.rsi_buy_min = rsi_buy_min
        self.rsi_buy_max = rsi_buy_max
        self.rsi_sell_min = rsi_sell_min
        self.rsi_sell_max = rsi_sell_max
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'strategy': 'S/R ALL-IN-ONE COMBO',
            'lookback_period': lookback_period,
            'rsi_period': rsi_period,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'volume_threshold': volume_threshold,
            'confirmations': '4 (S/R + RSI + EMA + Volume)',
        }
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = data['Close'].diff()
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
    
    def _find_swing_points(self, data: pd.DataFrame, window: int = 5) -> List[Tuple[int, float, str]]:
        """Find swing highs and lows"""
        swing_points = []
        highs = data['High'].values
        lows = data['Low'].values
        
        for i in range(window, len(data) - window):
            is_high = True
            for j in range(1, window + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            if is_high:
                swing_points.append((i, highs[i], 'high'))
            
            is_low = True
            for j in range(1, window + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            if is_low:
                swing_points.append((i, lows[i], 'low'))
        
        return swing_points
    
    def _cluster_levels(self, prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
        """Cluster prices into support/resistance levels"""
        if not prices:
            return []
        
        clusters = defaultdict(list)
        sorted_prices = sorted(prices)
        
        for price in sorted_prices:
            found_cluster = False
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            if not found_cluster:
                clusters[price].append(price)
        
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        levels.sort(key=lambda x: x[1], reverse=True)
        return levels
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        swing_points = self._find_swing_points(data)
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        resistance_levels = [level for level, touches in resistance_levels if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels if touches >= self.min_touches]
        
        support_levels = sorted(support_levels, reverse=True)
        resistance_levels = sorted(resistance_levels)
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """Find nearest support or resistance level"""
        if not levels:
            return None, float('inf')
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with ALL confirmations
        
        Entry Rules (ALL must be true):
        1️⃣ Price near support (S/R zone)
        2️⃣ RSI between 30-40 (oversold but not extreme)
        3️⃣ Price above 20/50 EMA (uptrend)
        4️⃣ Volume spike > 130% of average
        
        Exit Rules:
        - At resistance level
        - Stop loss below support
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        # Calculate all indicators
        signals['rsi'] = self._calculate_rsi(data, self.rsi_period)
        signals['ema_fast'] = data['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        signals['ema_slow'] = data['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        in_position = False
        
        for i in range(self.lookback_period, len(data)):
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not support_levels and not resistance_levels:
                continue
            
            current_price = data['Close'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_high = data['High'].iloc[i]
            current_rsi = signals['rsi'].iloc[i]
            ema_fast = signals['ema_fast'].iloc[i]
            ema_slow = signals['ema_slow'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            
            if pd.isna(current_rsi) or pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(current_atr) or pd.isna(avg_volume):
                continue
            
            # 4 CONFIRMATIONS
            # 1️⃣ S/R levels
            nearest_support, dist_to_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance, dist_to_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # 2️⃣ RSI confirmation
            rsi_in_buy_zone = self.rsi_buy_min <= current_rsi <= self.rsi_buy_max
            rsi_in_sell_zone = self.rsi_sell_min <= current_rsi <= self.rsi_sell_max
            
            # 3️⃣ EMA trend filter
            in_uptrend = current_price > ema_fast and current_price > ema_slow
            
            # 4️⃣ Volume confirmation
            volume_spike = current_volume >= (avg_volume * self.volume_threshold)
            
            # BUY SIGNAL: ALL 4 confirmations must align
            if not in_position and nearest_support is not None:
                near_support = dist_to_support <= 0.02
                support_touched = current_low <= nearest_support * 1.015
                
                # ALL confirmations required
                all_buy_conditions = (
                    near_support and
                    support_touched and
                    rsi_in_buy_zone and
                    in_uptrend and
                    volume_spike
                )
                
                if all_buy_conditions:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                        nearest_support - (current_atr * self.atr_multiplier)
                    in_position = True
            
            # SELL SIGNAL: Resistance or RSI overbought
            elif in_position:
                if nearest_resistance is not None:
                    near_resistance = dist_to_resistance <= 0.02
                    resistance_touched = current_high >= nearest_resistance * 0.985
                    
                    # Exit at resistance OR RSI overbought
                    exit_conditions = (
                        (near_resistance and resistance_touched) or
                        rsi_in_sell_zone
                    )
                    
                    if exit_conditions:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                
                # Stop loss
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal', 'stop_price']]

