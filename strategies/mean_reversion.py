"""
Mean Reversion Strategy

Buys when price is significantly below average (expecting reversion to mean)
and sells when price is significantly above average.
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands
    
    Buy when price touches lower Bollinger Band (oversold)
    Sell when price touches upper Bollinger Band (overbought)
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        """
        Initialize strategy
        
        Args:
            period: Period for moving average calculation
            num_std: Number of standard deviations for bands
        """
        super().__init__()
        self.period = period
        self.num_std = num_std
        self.parameters = {
            'period': period,
            'num_std': num_std
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate Bollinger Bands
        signals['ma'] = data['Close'].rolling(window=self.period).mean()
        signals['std'] = data['Close'].rolling(window=self.period).std()
        
        signals['upper_band'] = signals['ma'] + (signals['std'] * self.num_std)
        signals['lower_band'] = signals['ma'] - (signals['std'] * self.num_std)
        
        # Generate signals
        # Buy when price touches or crosses below lower band
        signals.loc[data['Close'] <= signals['lower_band'], 'signal'] = 1
        
        # Sell when price touches or crosses above upper band
        signals.loc[data['Close'] >= signals['upper_band'], 'signal'] = -1
        
        return signals[['signal']]


class ZScoreMeanReversion(Strategy):
    """
    Mean reversion strategy using Z-Score
    
    Buy when z-score is below threshold (price is significantly below mean)
    Sell when z-score is above threshold (price is significantly above mean)
    """
    
    def __init__(self, period: int = 20, buy_threshold: float = -2.0, sell_threshold: float = 2.0):
        """
        Initialize strategy
        
        Args:
            period: Lookback period for mean and std calculation
            buy_threshold: Z-score threshold for buy signal (negative)
            sell_threshold: Z-score threshold for sell signal (positive)
        """
        super().__init__()
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.parameters = {
            'period': period,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Z-Score
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate Z-Score
        signals['ma'] = data['Close'].rolling(window=self.period).mean()
        signals['std'] = data['Close'].rolling(window=self.period).std()
        
        signals['zscore'] = (data['Close'] - signals['ma']) / signals['std']
        
        # Generate signals
        # Buy when z-score is below buy threshold (oversold)
        signals.loc[signals['zscore'] < self.buy_threshold, 'signal'] = 1
        
        # Sell when z-score is above sell threshold (overbought)
        signals.loc[signals['zscore'] > self.sell_threshold, 'signal'] = -1
        
        return signals[['signal']]


class PercentileReversion(Strategy):
    """
    Mean reversion using percentile ranks
    
    Buy when price is in the bottom percentile
    Sell when price is in the top percentile
    """
    
    def __init__(self, period: int = 20, buy_percentile: float = 20, sell_percentile: float = 80):
        """
        Initialize strategy
        
        Args:
            period: Lookback period
            buy_percentile: Percentile threshold for buy (e.g., 20 = bottom 20%)
            sell_percentile: Percentile threshold for sell (e.g., 80 = top 20%)
        """
        super().__init__()
        self.period = period
        self.buy_percentile = buy_percentile
        self.sell_percentile = sell_percentile
        self.parameters = {
            'period': period,
            'buy_percentile': buy_percentile,
            'sell_percentile': sell_percentile
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on percentile ranks
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Calculate rolling percentile rank
        def percentile_rank(series):
            return series.rank(pct=True).iloc[-1] * 100
        
        signals['percentile'] = data['Close'].rolling(window=self.period).apply(percentile_rank, raw=False)
        
        # Generate signals
        # Buy when price is in bottom percentile
        signals.loc[signals['percentile'] <= self.buy_percentile, 'signal'] = 1
        
        # Sell when price is in top percentile
        signals.loc[signals['percentile'] >= self.sell_percentile, 'signal'] = -1
        
        return signals[['signal']]

