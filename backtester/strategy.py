"""
Base Strategy Class
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional


class Strategy(ABC):
    """
    Abstract base class for trading strategies
    
    All strategies must implement the generate_signals method.
    """
    
    def __init__(self):
        """Initialize strategy"""
        self.name = self.__class__.__name__
        self.parameters = {}
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            data: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
                  Index should be DatetimeIndex
        
        Returns:
            DataFrame with same index as input data and a 'signal' column where:
                1 = Buy signal
                -1 = Sell signal
                0 = Hold/No signal
        
        The returned DataFrame can also include additional columns for:
            - 'limit_price': Price for limit orders
            - 'stop_price': Price for stop loss orders
            - 'quantity': Number of shares (if variable position sizing)
        """
        pass
    
    def on_bar(self, timestamp: pd.Timestamp, data: pd.Series) -> Dict[str, Any]:
        """
        Optional: Called for each bar during backtesting
        Can be overridden for real-time signal generation or complex logic
        
        Args:
            timestamp: Current bar timestamp
            data: Current bar data (Open, High, Low, Close, Volume)
        
        Returns:
            Dictionary with signal information:
            {
                'signal': 1/-1/0,
                'limit_price': Optional[float],
                'stop_price': Optional[float],
                'quantity': Optional[int]
            }
        """
        return {'signal': 0}
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.parameters
    
    def set_parameters(self, **kwargs):
        """Update strategy parameters"""
        self.parameters.update(kwargs)
    
    def __repr__(self):
        params_str = ', '.join(f'{k}={v}' for k, v in self.parameters.items())
        return f"{self.name}({params_str})"


class SimpleStrategy(Strategy):
    """
    A simple example strategy that buys when price crosses above a threshold
    and sells when it crosses below
    """
    
    def __init__(self, buy_threshold: float = 100, sell_threshold: float = 95):
        super().__init__()
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals based on price thresholds"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Buy when price crosses above buy threshold
        signals.loc[data['Close'] > self.buy_threshold, 'signal'] = 1
        
        # Sell when price crosses below sell threshold
        signals.loc[data['Close'] < self.sell_threshold, 'signal'] = -1
        
        return signals

