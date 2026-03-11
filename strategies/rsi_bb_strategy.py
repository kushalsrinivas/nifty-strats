"""
RSI + Bollinger Bands Strategy

Buy Signal:
  - Price hits lower Bollinger Band (oversold)
  - RSI confirms oversold condition (< 30 or 40)

Sell Signal:
  - Price reaches middle Bollinger Band (mean reversion target)
  - OR RSI shows overbought (> 70)
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class RSIBollingerStrategy(Strategy):
    """
    Mean Reversion Strategy using RSI and Bollinger Bands
    
    Entry: Buy when price is at lower BB and RSI confirms oversold
    Exit: Sell when price reaches middle BB (take profit) or RSI overbought
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 40,
        rsi_overbought: float = 70,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_threshold: float = 1.02  # Price within 2% of lower BB
    ):
        """
        Initialize RSI + Bollinger Bands strategy
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold (buy confirmation)
            rsi_overbought: RSI threshold for overbought (sell signal)
            bb_period: Period for Bollinger Bands moving average
            bb_std: Number of standard deviations for bands
            bb_threshold: How close to lower BB to trigger (1.0 = exact, 1.02 = within 2%)
        """
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_threshold = bb_threshold
        
        self.parameters = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'bb_threshold': bb_threshold
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper_band = ma + (std * self.bb_std)
        lower_band = ma - (std * self.bb_std)
        
        return upper_band, ma, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI and Bollinger Bands
        
        Buy: Price at lower BB AND RSI oversold
        Sell: Price at middle BB OR RSI overbought
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = \
            self._calculate_bollinger_bands(data['Close'])
        
        # Track if we're in a position (to implement proper exit logic)
        in_position = False
        
        for i in range(len(data)):
            if i < self.bb_period:  # Skip until we have enough data
                continue
            
            current_price = data['Close'].iloc[i]
            rsi = signals['rsi'].iloc[i]
            lower_bb = signals['bb_lower'].iloc[i]
            middle_bb = signals['bb_middle'].iloc[i]
            upper_bb = signals['bb_upper'].iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(rsi) or pd.isna(lower_bb) or pd.isna(middle_bb):
                continue
            
            # BUY SIGNAL: Price touches lower BB and RSI confirms oversold
            if not in_position:
                price_at_lower_bb = current_price <= (lower_bb * self.bb_threshold)
                rsi_oversold = rsi < self.rsi_oversold
                
                if price_at_lower_bb and rsi_oversold:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL SIGNAL: Price reaches middle BB (take profit) OR RSI overbought
            else:
                price_at_middle = current_price >= middle_bb
                rsi_overbought = rsi > self.rsi_overbought
                
                if price_at_middle or rsi_overbought:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]


class AggressiveRSIBBStrategy(Strategy):
    """
    More aggressive version - looser conditions
    
    Entry: Price near lower BB (within 5%) OR RSI very oversold
    Exit: Price above middle BB OR RSI overbought
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35,
        rsi_very_oversold: float = 25,
        rsi_overbought: float = 65,
        bb_period: int = 20,
        bb_std: float = 2.0
    ):
        super().__init__()
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_very_oversold = rsi_very_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        
        self.parameters = {
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_very_oversold': rsi_very_oversold,
            'rsi_overbought': rsi_overbought,
            'bb_period': bb_period,
            'bb_std': bb_std
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=self.bb_period).mean()
        std = prices.rolling(window=self.bb_period).std()
        
        upper_band = ma + (std * self.bb_std)
        lower_band = ma - (std * self.bb_std)
        
        return upper_band, ma, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate more aggressive trading signals"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate indicators
        signals['rsi'] = self._calculate_rsi(data['Close'], self.rsi_period)
        signals['bb_upper'], signals['bb_middle'], signals['bb_lower'] = \
            self._calculate_bollinger_bands(data['Close'])
        
        in_position = False
        
        for i in range(len(data)):
            if i < self.bb_period:
                continue
            
            current_price = data['Close'].iloc[i]
            rsi = signals['rsi'].iloc[i]
            lower_bb = signals['bb_lower'].iloc[i]
            middle_bb = signals['bb_middle'].iloc[i]
            
            if pd.isna(rsi) or pd.isna(lower_bb) or pd.isna(middle_bb):
                continue
            
            # BUY: Near lower BB with oversold OR very oversold RSI
            if not in_position:
                near_lower_bb = current_price <= (lower_bb * 1.05)
                rsi_condition = (near_lower_bb and rsi < self.rsi_oversold) or \
                               (rsi < self.rsi_very_oversold)
                
                if rsi_condition:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    in_position = True
            
            # SELL: Above middle BB OR overbought
            else:
                if current_price >= middle_bb or rsi > self.rsi_overbought:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    in_position = False
        
        return signals[['signal']]

