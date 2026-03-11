"""
Moving Average Crossover Strategy

Generates buy signals when short-term MA crosses above long-term MA
and sell signals when short-term MA crosses below long-term MA.
"""

import pandas as pd
from backtester.strategy import Strategy


class MovingAverageCrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy
    
    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """
    
    def __init__(self, short_window: int = 50, long_window: int = 200):
        """
        Initialize strategy
        
        Args:
            short_window: Period for short-term moving average
            long_window: Period for long-term moving average
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate moving averages
        signals['short_ma'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        signals['long_ma'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()
        
        # Generate signals
        # Buy when short MA crosses above long MA
        signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 1
        
        # Sell when short MA crosses below long MA
        signals.loc[signals['short_ma'] < signals['long_ma'], 'signal'] = -1
        
        # Only trigger on crossovers (change in signal)
        signals['position'] = signals['signal'].diff()
        
        # Keep only crossover points
        signals.loc[signals['position'] == 0, 'signal'] = 0
        
        return signals[['signal']]


class ExponentialMovingAverageCrossover(Strategy):
    """
    Exponential Moving Average Crossover Strategy
    
    Similar to simple MA crossover but uses exponential moving averages
    which give more weight to recent prices.
    """
    
    def __init__(self, short_window: int = 12, long_window: int = 26):
        """
        Initialize strategy
        
        Args:
            short_window: Period for short-term EMA
            long_window: Period for long-term EMA
        """
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.parameters = {
            'short_window': short_window,
            'long_window': long_window
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on EMA crossover
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate exponential moving averages
        signals['short_ema'] = data['Close'].ewm(span=self.short_window, adjust=False).mean()
        signals['long_ema'] = data['Close'].ewm(span=self.long_window, adjust=False).mean()
        
        # Generate signals
        signals.loc[signals['short_ema'] > signals['long_ema'], 'signal'] = 1
        signals.loc[signals['short_ema'] < signals['long_ema'], 'signal'] = -1
        
        # Only trigger on crossovers
        signals['position'] = signals['signal'].diff()
        signals.loc[signals['position'] == 0, 'signal'] = 0
        
        return signals[['signal']]

