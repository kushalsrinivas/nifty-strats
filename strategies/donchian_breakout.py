"""
Donchian Breakout Strategy

The Donchian Channel breakout strategy is a pure trend-following approach
created by Richard Donchian and famously used by the Turtle Traders.

Concept:
- Upper Band = Highest High over X periods
- Lower Band = Lowest Low over X periods
- Middle Band = (Upper + Lower) / 2
- Buy when price breaks above upper band (new X-day high)
- Sell when price breaks below lower band (new X-day low)

This strategy works best in trending markets and requires strict risk management.
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class DonchianBreakoutStrategy(Strategy):
    """
    Donchian Channel Breakout Strategy
    
    Entry Signals:
    - Buy: Price closes above the upper Donchian band (breakout to new high)
    - Sell: Price closes below the lower Donchian band (breakout to new low)
    
    Exit Signals:
    - Exit long when price closes below the exit channel (shorter period)
    - Exit short when price closes above the exit channel
    
    This is a trend-following strategy that captures major moves.
    """
    
    def __init__(self, 
                 entry_period: int = 55,
                 exit_period: int = 20,
                 use_middle_band: bool = True,
                 atr_period: int = 14):
        """
        Initialize Donchian Breakout Strategy
        
        Args:
            entry_period: Period for entry Donchian channel (55 = Turtle Traders default)
            exit_period: Period for exit Donchian channel (20 = typical exit)
            use_middle_band: Whether to use middle band for additional signals
            atr_period: Period for ATR calculation (for volatility-based stops)
        """
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.use_middle_band = use_middle_band
        self.atr_period = atr_period
        self.parameters = {
            'entry_period': entry_period,
            'exit_period': exit_period,
            'use_middle_band': use_middle_band,
            'atr_period': atr_period
        }
    
    def calculate_donchian(self, data: pd.DataFrame, period: int) -> tuple:
        """
        Calculate Donchian Channel bands
        
        Args:
            data: DataFrame with OHLCV data
            period: Lookback period
            
        Returns:
            (upper_band, lower_band, middle_band)
        """
        upper = data['High'].rolling(window=period, min_periods=1).max()
        lower = data['Low'].rolling(window=period, min_periods=1).min()
        middle = (upper + lower) / 2
        
        return upper, lower, middle
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement
        
        Args:
            data: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            ATR series
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Donchian Channel breakouts
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate entry Donchian channel (wider)
        entry_upper, entry_lower, entry_middle = self.calculate_donchian(data, self.entry_period)
        
        # Calculate exit Donchian channel (narrower)
        exit_upper, exit_lower, exit_middle = self.calculate_donchian(data, self.exit_period)
        
        # Calculate ATR for volatility context
        atr = self.calculate_atr(data, self.atr_period)
        
        # Store channels for reference
        signals['entry_upper'] = entry_upper
        signals['entry_lower'] = entry_lower
        signals['entry_middle'] = entry_middle
        signals['exit_upper'] = exit_upper
        signals['exit_lower'] = exit_lower
        signals['atr'] = atr
        
        # Track position state
        position = 0  # 0 = no position, 1 = long, -1 = short
        
        for i in range(1, len(data)):
            current_close = data['Close'].iloc[i]
            prev_close = data['Close'].iloc[i-1]
            
            # ENTRY SIGNALS
            
            # Bullish Breakout: Price closes above upper band (new high)
            if position <= 0 and current_close > entry_upper.iloc[i]:
                signals.loc[data.index[i], 'signal'] = 1
                position = 1
            
            # Bearish Breakout: Price closes below lower band (new low)
            elif position >= 0 and current_close < entry_lower.iloc[i]:
                signals.loc[data.index[i], 'signal'] = -1
                position = -1
            
            # EXIT SIGNALS (using shorter period channel)
            
            # Exit long position: Price closes below exit lower band
            elif position == 1 and current_close < exit_lower.iloc[i]:
                signals.loc[data.index[i], 'signal'] = -1
                position = 0
            
            # Exit short position: Price closes above exit upper band
            elif position == -1 and current_close > exit_upper.iloc[i]:
                signals.loc[data.index[i], 'signal'] = 1
                position = 0
            
            # Optional: Middle band crossover for early exits in ranging markets
            elif self.use_middle_band:
                # Exit long if price crosses below middle band
                if position == 1 and current_close < entry_middle.iloc[i] and prev_close >= entry_middle.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = -1
                    position = 0
                
                # Exit short if price crosses above middle band
                elif position == -1 and current_close > entry_middle.iloc[i] and prev_close <= entry_middle.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = 1
                    position = 0
        
        return signals[['signal']]


class AggressiveDonchianStrategy(Strategy):
    """
    Aggressive Donchian Breakout Strategy
    
    Uses shorter periods for faster entries and tighter stops.
    Better for volatile, trending markets.
    Higher frequency trading with more signals.
    """
    
    def __init__(self, 
                 entry_period: int = 20,
                 exit_period: int = 10,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        """
        Initialize Aggressive Donchian Strategy
        
        Args:
            entry_period: Shorter entry period (20 for swing trading)
            exit_period: Shorter exit period (10 for quick exits)
            atr_period: Period for ATR calculation
            atr_multiplier: ATR multiplier for stop loss (2.0 = 2x ATR)
        """
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.parameters = {
            'entry_period': entry_period,
            'exit_period': exit_period,
            'atr_period': atr_period,
            'atr_multiplier': atr_multiplier
        }
    
    def calculate_donchian(self, data: pd.DataFrame, period: int) -> tuple:
        """Calculate Donchian Channel bands"""
        upper = data['High'].rolling(window=period, min_periods=1).max()
        lower = data['Low'].rolling(window=period, min_periods=1).min()
        middle = (upper + lower) / 2
        
        return upper, lower, middle
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate aggressive trading signals with ATR-based stops
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate entry and exit channels
        entry_upper, entry_lower, entry_middle = self.calculate_donchian(data, self.entry_period)
        exit_upper, exit_lower, exit_middle = self.calculate_donchian(data, self.exit_period)
        
        # Calculate ATR for volatility-based stops
        atr = self.calculate_atr(data, self.atr_period)
        
        # Store indicators
        signals['entry_upper'] = entry_upper
        signals['entry_lower'] = entry_lower
        signals['entry_middle'] = entry_middle
        signals['exit_upper'] = exit_upper
        signals['exit_lower'] = exit_lower
        signals['atr'] = atr
        
        # Calculate channel width (for volatility filtering)
        channel_width = (entry_upper - entry_lower) / entry_middle
        signals['channel_width'] = channel_width
        
        # Track position
        position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            current_close = data['Close'].iloc[i]
            current_high = data['High'].iloc[i]
            current_low = data['Low'].iloc[i]
            current_atr = atr.iloc[i]
            
            # ENTRY SIGNALS (more aggressive)
            
            # Bullish breakout: Price breaks above upper band
            if position <= 0 and current_high > entry_upper.iloc[i]:
                signals.loc[data.index[i], 'signal'] = 1
                position = 1
                entry_price = current_close
            
            # Bearish breakout: Price breaks below lower band
            elif position >= 0 and current_low < entry_lower.iloc[i]:
                signals.loc[data.index[i], 'signal'] = -1
                position = -1
                entry_price = current_close
            
            # EXIT SIGNALS (ATR-based stops)
            
            # Exit long with ATR stop or exit channel
            elif position == 1:
                stop_loss = entry_price - (self.atr_multiplier * current_atr)
                
                # Stop loss hit or exit channel breached
                if current_close < stop_loss or current_low < exit_lower.iloc[i]:
                    signals.loc[data.index[i], 'signal'] = -1
                    position = 0
                    entry_price = 0
            
            # Exit short with ATR stop or exit channel
            elif position == -1:
                stop_loss = entry_price + (self.atr_multiplier * current_atr)
                
                # Stop loss hit or exit channel breached
                if current_close > stop_loss or current_high > exit_upper.iloc[i]:
                    signals.loc[data.index[i], 'signal'] = 1
                    position = 0
                    entry_price = 0
        
        return signals[['signal']]


class TurtleTradersStrategy(Strategy):
    """
    Classic Turtle Traders Donchian Strategy
    
    This is the famous strategy taught to the Turtle Traders by Richard Dennis.
    
    System 1 (Fast): 20-day breakout, 10-day exit
    System 2 (Slow): 55-day breakout, 20-day exit (implemented here)
    
    Rules:
    - Enter long on 55-day high
    - Exit on 20-day low
    - Position sizing based on volatility (ATR)
    - Risk 1-2% of capital per trade
    """
    
    def __init__(self, 
                 entry_period: int = 55,
                 exit_period: int = 20,
                 atr_period: int = 20,
                 risk_per_trade: float = 0.02):
        """
        Initialize Turtle Traders Strategy
        
        Args:
            entry_period: Entry breakout period (55 days)
            exit_period: Exit period (20 days)
            atr_period: ATR period for position sizing (20 days)
            risk_per_trade: Risk percentage per trade (0.02 = 2%)
        """
        super().__init__()
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade
        self.parameters = {
            'entry_period': entry_period,
            'exit_period': exit_period,
            'atr_period': atr_period,
            'risk_per_trade': risk_per_trade
        }
    
    def calculate_donchian(self, data: pd.DataFrame, period: int) -> tuple:
        """Calculate Donchian Channel"""
        upper = data['High'].rolling(window=period, min_periods=1).max()
        lower = data['Low'].rolling(window=period, min_periods=1).min()
        middle = (upper + lower) / 2
        
        return upper, lower, middle
    
    def calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR"""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Turtle Traders signals
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate channels
        entry_upper, entry_lower, entry_middle = self.calculate_donchian(data, self.entry_period)
        exit_upper, exit_lower, exit_middle = self.calculate_donchian(data, self.exit_period)
        
        # Calculate ATR for position sizing
        atr = self.calculate_atr(data, self.atr_period)
        
        # Store indicators
        signals['entry_upper'] = entry_upper
        signals['entry_lower'] = entry_lower
        signals['exit_upper'] = exit_upper
        signals['exit_lower'] = exit_lower
        signals['atr'] = atr
        
        # Calculate position sizing based on ATR (N-value in Turtle speak)
        # Turtle's risked 1N (1 ATR) per position, which was 2% of capital
        signals['position_size_multiplier'] = self.risk_per_trade / (atr / data['Close'])
        
        # Track position
        position = 0
        
        for i in range(1, len(data)):
            current_close = data['Close'].iloc[i]
            prev_high = data['High'].iloc[i-1]
            prev_low = data['Low'].iloc[i-1]
            
            # Entry: 55-day breakout
            if position == 0:
                # Long entry: New 55-day high
                if current_close > entry_upper.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = 1
                    position = 1
                
                # Short entry: New 55-day low
                elif current_close < entry_lower.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = -1
                    position = -1
            
            # Exit long: 20-day low
            elif position == 1:
                if current_close < exit_lower.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = -1
                    position = 0
            
            # Exit short: 20-day high
            elif position == -1:
                if current_close > exit_upper.iloc[i-1]:
                    signals.loc[data.index[i], 'signal'] = 1
                    position = 0
        
        return signals[['signal']]

