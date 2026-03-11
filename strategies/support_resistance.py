"""
Support and Resistance Trading Strategy

Identifies key support and resistance levels based on:
- Historical swing highs and lows
- Price clustering (levels where price consolidated)
- Volume profile (high volume areas)

Trading Logic:
- Buy when price bounces off support
- Sell when price hits resistance
- Optionally trade breakouts through key levels
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class SupportResistanceStrategy(Strategy):
    """
    Support and Resistance Trading Strategy
    
    Identifies horizontal S/R levels and trades bounces or breakouts.
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        price_tolerance: float = 0.02,  # 2% price clustering tolerance
        min_touches: int = 2,  # Minimum touches to validate level
        bounce_distance: float = 0.015,  # 1.5% from level to trigger
        volume_confirmation: bool = True,
        volume_threshold: float = 1.2,
        use_volume_profile: bool = True,
        breakout_mode: bool = False,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
    ):
        """
        Initialize Support and Resistance Strategy
        
        Args:
            lookback_period: Bars to look back for S/R identification
            price_tolerance: Price clustering tolerance (%)
            min_touches: Minimum touches to validate S/R level
            bounce_distance: Distance from level to trigger trade (%)
            volume_confirmation: Require volume spike
            volume_threshold: Volume multiplier vs average
            use_volume_profile: Consider volume at price levels
            breakout_mode: Trade breakouts instead of bounces
            atr_period: ATR period for stop loss
            atr_multiplier: ATR multiplier for stop loss
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.price_tolerance = price_tolerance
        self.min_touches = min_touches
        self.bounce_distance = bounce_distance
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
        self.use_volume_profile = use_volume_profile
        self.breakout_mode = breakout_mode
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        self.parameters = {
            'lookback_period': lookback_period,
            'price_tolerance': price_tolerance,
            'min_touches': min_touches,
            'bounce_distance': bounce_distance,
            'volume_confirmation': volume_confirmation,
            'volume_threshold': volume_threshold,
            'use_volume_profile': use_volume_profile,
            'mode': 'Breakout' if breakout_mode else 'Bounce',
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier,
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
        """
        Find swing highs and lows
        
        Returns:
            List of (index, price, type) where type is 'high' or 'low'
        """
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
        """
        Cluster prices into support/resistance levels
        
        Args:
            prices: List of prices to cluster
            tolerance: Price clustering tolerance (%)
            
        Returns:
            List of (level_price, touch_count) sorted by touch count
        """
        if not prices:
            return []
        
        clusters = defaultdict(list)
        
        # Sort prices
        sorted_prices = sorted(prices)
        
        # Cluster nearby prices
        for price in sorted_prices:
            found_cluster = False
            
            for cluster_price in list(clusters.keys()):
                if abs(price - cluster_price) / cluster_price <= tolerance:
                    clusters[cluster_price].append(price)
                    found_cluster = True
                    break
            
            if not found_cluster:
                clusters[price].append(price)
        
        # Calculate average price for each cluster and count touches
        levels = []
        for cluster_prices in clusters.values():
            avg_price = np.mean(cluster_prices)
            touch_count = len(cluster_prices)
            levels.append((avg_price, touch_count))
        
        # Sort by touch count (descending)
        levels.sort(key=lambda x: x[1], reverse=True)
        
        return levels
    
    def _calculate_volume_profile(self, data: pd.DataFrame, num_bins: int = 50) -> Dict[float, float]:
        """
        Calculate volume profile (volume at price levels)
        
        Args:
            data: OHLCV data
            num_bins: Number of price bins
            
        Returns:
            Dictionary mapping price level to volume
        """
        # Create price bins
        price_min = data['Low'].min()
        price_max = data['High'].max()
        bins = np.linspace(price_min, price_max, num_bins)
        
        volume_profile = defaultdict(float)
        
        for idx, row in data.iterrows():
            # Typical price for the bar
            typical_price = (row['High'] + row['Low'] + row['Close']) / 3
            
            # Find which bin this price falls into
            bin_idx = np.digitize(typical_price, bins) - 1
            bin_idx = max(0, min(bin_idx, len(bins) - 2))
            
            # Add volume to this price level
            bin_price = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            volume_profile[bin_price] += row['Volume']
        
        return volume_profile
    
    def _identify_sr_levels(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        Identify support and resistance levels
        
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find swing points
        swing_points = self._find_swing_points(data)
        
        # Separate highs and lows
        swing_highs = [price for idx, price, typ in swing_points if typ == 'high']
        swing_lows = [price for idx, price, typ in swing_points if typ == 'low']
        
        # Cluster into resistance and support levels
        resistance_levels = self._cluster_levels(swing_highs, self.price_tolerance)
        support_levels = self._cluster_levels(swing_lows, self.price_tolerance)
        
        # Filter by minimum touches
        resistance_levels = [level for level, touches in resistance_levels 
                            if touches >= self.min_touches]
        support_levels = [level for level, touches in support_levels 
                         if touches >= self.min_touches]
        
        # Enhance with volume profile if enabled
        if self.use_volume_profile:
            volume_profile = self._calculate_volume_profile(data)
            
            # Find high volume nodes (potential S/R)
            sorted_volumes = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            high_volume_levels = [price for price, vol in sorted_volumes[:5]]
            
            # Add high volume levels that aren't already identified
            current_price = data['Close'].iloc[-1]
            
            for hvl in high_volume_levels:
                # Check if this level is close to existing levels
                is_new = True
                for level in support_levels + resistance_levels:
                    if abs(hvl - level) / level <= self.price_tolerance:
                        is_new = False
                        break
                
                if is_new:
                    if hvl < current_price:
                        support_levels.append(hvl)
                    else:
                        resistance_levels.append(hvl)
        
        # Sort levels
        support_levels = sorted(support_levels, reverse=True)  # Highest first
        resistance_levels = sorted(resistance_levels)  # Lowest first
        
        return support_levels, resistance_levels
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Tuple[Optional[float], float]:
        """
        Find nearest support or resistance level
        
        Returns:
            (nearest_level, distance_pct) or (None, inf) if no levels
        """
        if not levels:
            return None, float('inf')
        
        distances = [(level, abs(price - level) / price) for level in levels]
        nearest = min(distances, key=lambda x: x[1])
        
        return nearest[0], nearest[1]
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on support/resistance bounces or breakouts
        
        Logic:
        - Bounce Mode:
            • Buy when price bounces off support
            • Sell when price hits resistance
        - Breakout Mode:
            • Buy when price breaks above resistance
            • Sell when price breaks below support
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        
        # Calculate ATR
        signals['atr'] = self._calculate_atr(data, self.atr_period)
        
        # Calculate volume average
        signals['volume_ma'] = data['Volume'].rolling(window=20).mean()
        
        # Track position
        in_position = False
        entry_level = None
        
        for i in range(self.lookback_period, len(data)):
            # Get window for S/R calculation
            window_start = max(0, i - self.lookback_period)
            window_data = data.iloc[window_start:i]
            
            # Identify S/R levels
            support_levels, resistance_levels = self._identify_sr_levels(window_data)
            
            if not support_levels and not resistance_levels:
                continue
            
            # Current price info
            current_price = data['Close'].iloc[i]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_volume = data['Volume'].iloc[i]
            avg_volume = signals['volume_ma'].iloc[i]
            current_atr = signals['atr'].iloc[i]
            
            if pd.isna(current_atr) or pd.isna(avg_volume):
                continue
            
            # Volume confirmation
            volume_ok = True
            if self.volume_confirmation:
                volume_ok = current_volume >= (avg_volume * self.volume_threshold)
            
            # Find nearest support and resistance
            nearest_support, dist_to_support = self._find_nearest_level(current_price, support_levels)
            nearest_resistance, dist_to_resistance = self._find_nearest_level(current_price, resistance_levels)
            
            # ========== BOUNCE MODE ==========
            if not self.breakout_mode:
                # BUY: Bouncing off support
                if not in_position and nearest_support is not None:
                    # Price is near support
                    if dist_to_support <= self.bounce_distance and volume_ok:
                        # Confirm bounce: low touched or went below support, but closed above
                        if current_low <= nearest_support * (1 + self.bounce_distance):
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            # Stop loss below support
                            signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                                nearest_support - (current_atr * self.atr_multiplier)
                            in_position = True
                            entry_level = nearest_support
                
                # SELL: Hitting resistance
                elif in_position and nearest_resistance is not None:
                    # Price is near resistance
                    if dist_to_resistance <= self.bounce_distance:
                        # Confirm hit: high touched or went above resistance
                        if current_high >= nearest_resistance * (1 - self.bounce_distance):
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            in_position = False
                            entry_level = None
            
            # ========== BREAKOUT MODE ==========
            else:
                # BUY: Breaking above resistance
                if not in_position and nearest_resistance is not None:
                    # Price breaks above resistance with volume
                    if current_price > nearest_resistance * (1 + self.bounce_distance) and volume_ok:
                        # Confirm close above resistance
                        if current_price > nearest_resistance:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            # Stop loss below broken resistance (now support)
                            signals.iloc[i, signals.columns.get_loc('stop_price')] = \
                                nearest_resistance - (current_atr * self.atr_multiplier)
                            in_position = True
                            entry_level = nearest_resistance
                
                # SELL: Breaking below support
                elif in_position and nearest_support is not None:
                    # Price breaks below support
                    if current_price < nearest_support * (1 - self.bounce_distance):
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        in_position = False
                        entry_level = None
            
            # Stop loss check
            if in_position:
                stop_price = signals['stop_price'].iloc[i-1] if i > 0 else np.nan
                if not pd.isna(stop_price) and current_low <= stop_price:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_level = None
        
        return signals[['signal', 'stop_price']]


class SupportResistanceBounceStrategy(Strategy):
    """
    Simplified Support/Resistance Bounce Strategy
    
    Focuses on clean bounces with strong validation.
    """
    
    def __init__(
        self,
        lookback_period: int = 80,
        min_touches: int = 3,
        volume_threshold: float = 1.3,
    ):
        """Initialize S/R Bounce Strategy"""
        super().__init__()
        
        self.base_strategy = SupportResistanceStrategy(
            lookback_period=lookback_period,
            price_tolerance=0.02,
            min_touches=min_touches,
            bounce_distance=0.015,
            volume_confirmation=True,
            volume_threshold=volume_threshold,
            use_volume_profile=True,
            breakout_mode=False,
            atr_period=14,
            atr_multiplier=1.5,
        )
        
        self.parameters = self.base_strategy.parameters
        self.parameters['strategy_type'] = 'Support/Resistance Bounce'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using bounce mode"""
        return self.base_strategy.generate_signals(data)


class SupportResistanceBreakoutStrategy(Strategy):
    """
    Simplified Support/Resistance Breakout Strategy
    
    Trades clean breakouts through key levels.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        min_touches: int = 2,
        volume_threshold: float = 1.5,
    ):
        """Initialize S/R Breakout Strategy"""
        super().__init__()
        
        self.base_strategy = SupportResistanceStrategy(
            lookback_period=lookback_period,
            price_tolerance=0.025,
            min_touches=min_touches,
            bounce_distance=0.02,
            volume_confirmation=True,
            volume_threshold=volume_threshold,
            use_volume_profile=True,
            breakout_mode=True,
            atr_period=14,
            atr_multiplier=2.0,
        )
        
        self.parameters = self.base_strategy.parameters
        self.parameters['strategy_type'] = 'Support/Resistance Breakout'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using breakout mode"""
        return self.base_strategy.generate_signals(data)

