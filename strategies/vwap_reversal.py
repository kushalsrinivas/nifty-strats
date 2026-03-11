"""
VWAP Reversal Strategy

Primary Indicator: VWAP (Volume Weighted Average Price)
Confirmation Indicators: RSI Divergence + Volume

Buy Signal:
  - Price deviates significantly below VWAP (oversold area)
  - Bullish RSI divergence detected (price making lower lows, RSI making higher lows)
  - Volume confirmation (above average)
  
Sell Signal:
  - Price reaches or exceeds VWAP (mean reversion target)
  - Bearish RSI divergence detected
  - OR RSI shows overbought conditions

Strategy Type: Mean Reversion - Buy dips with divergence confirmation
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class VWAPReversalStrategy(Strategy):
    """
    Mean reversion strategy using VWAP with RSI divergence detection
    
    VWAP acts as a fair value anchor. When price deviates significantly below VWAP
    with bullish RSI divergence, it signals a high-probability reversal opportunity.
    """
    
    def __init__(
        self,
        vwap_deviation_threshold: float = 1.5,  # Aggressive: 1.5% below VWAP triggers
        rsi_period: int = 14,
        rsi_overbought: float = 65,  # Aggressive: lower thresholds
        rsi_oversold: float = 35,
        divergence_lookback: int = 10,  # Bars to look back for divergence
        volume_ma_period: int = 20,
        volume_threshold: float = 1.1,  # Volume 10% above average
        profit_target_pct: float = 1.0  # Target 1% profit to VWAP
    ):
        """
        Initialize VWAP Reversal Strategy
        
        Args:
            vwap_deviation_threshold: % below VWAP to trigger buy consideration
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI threshold for overbought
            rsi_oversold: RSI threshold for oversold
            divergence_lookback: Periods to scan for divergence patterns
            volume_ma_period: Period for volume moving average
            volume_threshold: Minimum volume multiplier for confirmation
            profit_target_pct: Profit target as % move toward VWAP
        """
        super().__init__()
        self.vwap_deviation_threshold = vwap_deviation_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.divergence_lookback = divergence_lookback
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        self.profit_target_pct = profit_target_pct
        
        self.parameters = {
            'vwap_deviation_threshold': vwap_deviation_threshold,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'divergence_lookback': divergence_lookback,
            'volume_ma_period': volume_ma_period,
            'volume_threshold': volume_threshold,
            'profit_target_pct': profit_target_pct
        }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price)
        
        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        Typical Price = (High + Low + Close) / 3
        
        Note: In real trading, VWAP resets each day. For backtesting across multiple days,
        we use a rolling VWAP or session-based calculation.
        """
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        
        # Calculate cumulative VWAP (simplified version for continuous data)
        # For intraday data, you'd reset this at the start of each session
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        return vwap
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _detect_bullish_divergence(self, prices: pd.Series, rsi: pd.Series, 
                                   current_idx: int, lookback: int) -> bool:
        """
        Detect bullish RSI divergence
        
        Bullish divergence occurs when:
        - Price makes lower lows
        - RSI makes higher lows
        
        This suggests weakening downward momentum and potential reversal
        """
        if current_idx < lookback:
            return False
        
        # Get recent price and RSI data
        recent_prices = prices.iloc[current_idx - lookback:current_idx + 1]
        recent_rsi = rsi.iloc[current_idx - lookback:current_idx + 1]
        
        if len(recent_prices) < 2 or recent_rsi.isna().any():
            return False
        
        # Find local minima (valleys) in price and RSI
        price_min_idx = recent_prices.idxmin()
        price_min_pos = recent_prices.index.get_loc(price_min_idx)
        
        # Check if we have a lower low in price
        if price_min_pos == 0:  # Minimum at start of window
            return False
        
        # Compare: is current price lower but RSI higher than previous low?
        current_price = prices.iloc[current_idx]
        current_rsi = rsi.iloc[current_idx]
        prev_low_rsi = recent_rsi.iloc[price_min_pos]
        
        # Bullish divergence: price making new lows but RSI not confirming
        price_lower = current_price <= recent_prices.min()
        rsi_higher = current_rsi > prev_low_rsi
        
        return price_lower and rsi_higher
    
    def _detect_bearish_divergence(self, prices: pd.Series, rsi: pd.Series,
                                   current_idx: int, lookback: int) -> bool:
        """
        Detect bearish RSI divergence
        
        Bearish divergence occurs when:
        - Price makes higher highs
        - RSI makes lower highs
        
        This suggests weakening upward momentum
        """
        if current_idx < lookback:
            return False
        
        recent_prices = prices.iloc[current_idx - lookback:current_idx + 1]
        recent_rsi = rsi.iloc[current_idx - lookback:current_idx + 1]
        
        if len(recent_prices) < 2 or recent_rsi.isna().any():
            return False
        
        price_max_idx = recent_prices.idxmax()
        price_max_pos = recent_prices.index.get_loc(price_max_idx)
        
        if price_max_pos == 0:
            return False
        
        current_price = prices.iloc[current_idx]
        current_rsi = rsi.iloc[current_idx]
        prev_high_rsi = recent_rsi.iloc[price_max_pos]
        
        # Bearish divergence: price making new highs but RSI not confirming
        price_higher = current_price >= recent_prices.max()
        rsi_lower = current_rsi < prev_high_rsi
        
        return price_higher and rsi_lower
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on VWAP, RSI divergence, and Volume
        
        Buy: Price below VWAP + Bullish RSI divergence + Volume confirmation
        Sell: Price at/above VWAP OR Bearish divergence OR RSI overbought
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['vwap'] = self._calculate_vwap(data)
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        # Track entry price for exit logic
        entry_price = None
        in_position = False
        
        for i in range(len(data)):
            # Skip until we have enough data
            if i < max(self.rsi_period, self.volume_ma_period, self.divergence_lookback):
                continue
            
            current_price = data['Close'].iloc[i]
            vwap = signals['vwap'].iloc[i]
            rsi = signals['rsi'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(vwap) or pd.isna(rsi) or pd.isna(volume_ma):
                continue
            
            # BUY SIGNAL: Price below VWAP + Bullish divergence + Volume
            if not in_position:
                # Price significantly below VWAP (oversold vs fair value)
                deviation_pct = ((vwap - current_price) / vwap) * 100
                price_below_vwap = deviation_pct >= self.vwap_deviation_threshold
                
                # Bullish RSI divergence detection
                bullish_divergence = self._detect_bullish_divergence(
                    data['Close'], signals['rsi'], i, self.divergence_lookback
                )
                
                # Volume confirmation
                volume_confirmed = volume > (volume_ma * self.volume_threshold)
                
                # Additional confirmation: RSI in oversold territory
                rsi_oversold = rsi < self.rsi_oversold
                
                # Enter if we have strong confluence
                if price_below_vwap and (bullish_divergence or rsi_oversold) and volume_confirmed:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
                    entry_price = current_price
            
            # SELL SIGNAL: Mean reversion complete OR divergence shifts bearish
            else:
                # Target 1: Price reaches VWAP (mean reversion complete)
                reached_vwap = current_price >= (vwap * (1 - self.profit_target_pct / 100))
                
                # Target 2: Bearish divergence appears
                bearish_divergence = self._detect_bearish_divergence(
                    data['Close'], signals['rsi'], i, self.divergence_lookback
                )
                
                # Target 3: RSI overbought
                rsi_overbought = rsi > self.rsi_overbought
                
                # Exit conditions
                if reached_vwap or bearish_divergence or rsi_overbought:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
                    entry_price = None
        
        return signals[['signal']]


class AggressiveVWAPStrategy(Strategy):
    """
    Ultra-aggressive version - enters on any VWAP deviation with volume
    
    Entry: Price below VWAP with volume OR RSI oversold near VWAP
    Exit: Quick profit taking at VWAP or small loss
    """
    
    def __init__(
        self,
        vwap_deviation_threshold: float = 0.8,  # Just 0.8% deviation
        rsi_period: int = 9,  # Faster RSI
        rsi_overbought: float = 60,
        rsi_oversold: float = 40,
        volume_ma_period: int = 10,
        volume_threshold: float = 1.05  # Just 5% above average
    ):
        super().__init__()
        self.vwap_deviation_threshold = vwap_deviation_threshold
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.volume_threshold = volume_threshold
        
        self.parameters = {
            'vwap_deviation_threshold': vwap_deviation_threshold,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'volume_ma_period': volume_ma_period,
            'volume_threshold': volume_threshold
        }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ultra-aggressive VWAP reversal signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        signals['vwap'] = self._calculate_vwap(data)
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['volume_ma'] = data['Volume'].rolling(window=self.volume_ma_period).mean()
        
        in_position = False
        
        for i in range(len(data)):
            if i < max(self.rsi_period, self.volume_ma_period):
                continue
            
            current_price = data['Close'].iloc[i]
            vwap = signals['vwap'].iloc[i]
            rsi = signals['rsi'].iloc[i]
            volume = data['Volume'].iloc[i]
            volume_ma = signals['volume_ma'].iloc[i]
            
            if pd.isna(vwap) or pd.isna(rsi) or pd.isna(volume_ma):
                continue
            
            # BUY: Any deviation below VWAP with decent volume
            if not in_position:
                deviation_pct = ((vwap - current_price) / vwap) * 100
                below_vwap = deviation_pct >= self.vwap_deviation_threshold
                volume_ok = volume > (volume_ma * self.volume_threshold)
                
                if below_vwap and volume_ok:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Quick exit at VWAP or RSI signal
            else:
                at_vwap = current_price >= vwap
                rsi_high = rsi > self.rsi_overbought
                
                if at_vwap or rsi_high:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

