"""
Harmonic Pattern Recognition Strategy

Identifies and trades based on harmonic patterns using Fibonacci ratios.
Supported patterns: Gartley, Bat, Butterfly, Crab, and Cypher.

Pattern Structure (5 points): X → A → B → C → D
- D point is the Potential Reversal Zone (PRZ)

Each pattern has specific Fibonacci ratio requirements:
1. Gartley: Classic harmonic pattern
2. Bat: Deep retracement to 88.6%
3. Butterfly: Extended D beyond X
4. Crab: Very sharp CD leg
5. Cypher: Modified ratios

References:
- Scott Carney's "Harmonic Trading" methodology
- Fibonacci retracement and extension levels
"""

import pandas as pd
import numpy as np
from backtester.strategy import Strategy
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class PatternPoint:
    """Represents a swing point in the pattern"""
    index: int
    price: float
    timestamp: pd.Timestamp


@dataclass
class HarmonicPattern:
    """Represents a detected harmonic pattern"""
    pattern_type: str  # 'Gartley', 'Bat', 'Butterfly', 'Crab', 'Cypher'
    direction: str  # 'bullish' or 'bearish'
    X: PatternPoint
    A: PatternPoint
    B: PatternPoint
    C: PatternPoint
    D: PatternPoint
    prz_price: float  # Potential Reversal Zone price
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float  # 0-100, how well ratios match


class HarmonicPatternStrategy(Strategy):
    """
    Harmonic Pattern Recognition and Trading Strategy
    
    Identifies geometric price patterns based on Fibonacci ratios.
    Trades at the D point (PRZ) with multiple profit targets.
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 80,
        zigzag_threshold: float = 0.03,  # 3% minimum swing
        ratio_tolerance: float = 0.05,  # 5% tolerance for Fibonacci ratios
        min_confidence: float = 70.0,  # Minimum pattern confidence to trade
        use_gartley: bool = True,
        use_bat: bool = True,
        use_butterfly: bool = True,
        use_crab: bool = True,
        use_cypher: bool = True,
        stop_loss_pct: float = 0.02,  # 2% stop loss beyond D
        take_profit_1_pct: float = 0.382,  # First target at 38.2% retracement
        take_profit_2_pct: float = 0.618,  # Second target at 61.8% retracement
    ):
        """
        Initialize Harmonic Pattern Recognition Strategy
        
        Args:
            lookback_period: Number of bars to look back for pattern detection
            min_pattern_bars: Minimum bars for a complete pattern
            max_pattern_bars: Maximum bars for a complete pattern
            zigzag_threshold: Minimum price move % to identify swing points
            ratio_tolerance: Tolerance for Fibonacci ratio matching
            min_confidence: Minimum confidence score to trade pattern
            use_gartley: Enable Gartley pattern detection
            use_bat: Enable Bat pattern detection
            use_butterfly: Enable Butterfly pattern detection
            use_crab: Enable Crab pattern detection
            use_cypher: Enable Cypher pattern detection
            stop_loss_pct: Stop loss percentage beyond D point
            take_profit_1_pct: First profit target (Fibonacci level)
            take_profit_2_pct: Second profit target (Fibonacci level)
        """
        super().__init__()
        self.lookback_period = lookback_period
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        self.zigzag_threshold = zigzag_threshold
        self.ratio_tolerance = ratio_tolerance
        self.min_confidence = min_confidence
        self.use_gartley = use_gartley
        self.use_bat = use_bat
        self.use_butterfly = use_butterfly
        self.use_crab = use_crab
        self.use_cypher = use_cypher
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_1_pct = take_profit_1_pct
        self.take_profit_2_pct = take_profit_2_pct
        
        self.parameters = {
            'lookback_period': lookback_period,
            'min_pattern_bars': min_pattern_bars,
            'max_pattern_bars': max_pattern_bars,
            'zigzag_threshold': zigzag_threshold,
            'ratio_tolerance': ratio_tolerance,
            'min_confidence': min_confidence,
            'patterns_enabled': self._get_enabled_patterns(),
            'stop_loss_pct': stop_loss_pct,
            'take_profit_1_pct': take_profit_1_pct,
            'take_profit_2_pct': take_profit_2_pct,
        }
    
    def _get_enabled_patterns(self) -> List[str]:
        """Get list of enabled patterns"""
        patterns = []
        if self.use_gartley:
            patterns.append('Gartley')
        if self.use_bat:
            patterns.append('Bat')
        if self.use_butterfly:
            patterns.append('Butterfly')
        if self.use_crab:
            patterns.append('Crab')
        if self.use_cypher:
            patterns.append('Cypher')
        return patterns
    
    def _identify_swing_points(self, data: pd.DataFrame) -> List[PatternPoint]:
        """
        Identify swing highs and lows using zigzag method
        
        Returns list of swing points (alternating highs and lows)
        """
        swing_points = []
        
        # Need at least 5 bars to identify a swing
        if len(data) < 5:
            return swing_points
        
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Start with first significant point
        last_swing_idx = 0
        last_swing_price = closes[0]
        last_swing_type = None  # 'high' or 'low'
        
        for i in range(2, len(data) - 2):
            # Check for swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                # Calculate move size
                move_pct = abs(highs[i] - last_swing_price) / last_swing_price
                
                # Only add if move is significant and not same type as last
                if move_pct >= self.zigzag_threshold and last_swing_type != 'high':
                    point = PatternPoint(
                        index=i,
                        price=highs[i],
                        timestamp=data.index[i]
                    )
                    swing_points.append(point)
                    last_swing_idx = i
                    last_swing_price = highs[i]
                    last_swing_type = 'high'
            
            # Check for swing low
            elif (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                  lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                move_pct = abs(lows[i] - last_swing_price) / last_swing_price
                
                if move_pct >= self.zigzag_threshold and last_swing_type != 'low':
                    point = PatternPoint(
                        index=i,
                        price=lows[i],
                        timestamp=data.index[i]
                    )
                    swing_points.append(point)
                    last_swing_idx = i
                    last_swing_price = lows[i]
                    last_swing_type = 'low'
        
        return swing_points
    
    def _calculate_ratio(self, point1: float, point2: float, point3: float) -> float:
        """
        Calculate Fibonacci ratio between price moves
        
        Returns: ratio of (point2 - point3) / (point1 - point2)
        """
        move1 = abs(point1 - point2)
        move2 = abs(point2 - point3)
        
        if move1 == 0:
            return 0
        
        return move2 / move1
    
    def _check_ratio(self, actual: float, expected: float) -> bool:
        """Check if ratio is within tolerance"""
        lower = expected * (1 - self.ratio_tolerance)
        upper = expected * (1 + self.ratio_tolerance)
        return lower <= actual <= upper
    
    def _calculate_confidence(self, ratios: Dict[str, float], 
                            expected: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate pattern confidence based on how well ratios match
        
        Returns: confidence score 0-100
        """
        scores = []
        
        for name, actual in ratios.items():
            if name in expected:
                exp_min, exp_max = expected[name]
                
                # Calculate how close actual is to expected range
                if exp_min <= actual <= exp_max:
                    # Perfect match
                    scores.append(100)
                else:
                    # Calculate deviation
                    if actual < exp_min:
                        deviation = (exp_min - actual) / exp_min
                    else:
                        deviation = (actual - exp_max) / exp_max
                    
                    # Score decreases with deviation
                    score = max(0, 100 - (deviation * 100 / self.ratio_tolerance))
                    scores.append(score)
        
        return np.mean(scores) if scores else 0
    
    def _identify_gartley(self, X: PatternPoint, A: PatternPoint, 
                         B: PatternPoint, C: PatternPoint, 
                         D: PatternPoint) -> Optional[HarmonicPattern]:
        """
        Identify Gartley pattern
        
        Ratios:
        - AB = 61.8% of XA
        - BC = 38.2% to 88.6% of AB
        - CD = 127.2% to 161.8% of BC
        - AD = 78.6% of XA
        """
        # Calculate ratios
        AB_XA = self._calculate_ratio(X.price, A.price, B.price)
        BC_AB = self._calculate_ratio(A.price, B.price, C.price)
        CD_BC = self._calculate_ratio(B.price, C.price, D.price)
        AD_XA = self._calculate_ratio(X.price, A.price, D.price)
        
        # Define expected ratios with ranges
        expected = {
            'AB_XA': (0.586, 0.650),      # 61.8% ± tolerance
            'BC_AB': (0.382, 0.886),       # 38.2% to 88.6%
            'CD_BC': (1.272, 1.618),       # 127.2% to 161.8%
            'AD_XA': (0.746, 0.826),       # 78.6% ± tolerance
        }
        
        ratios = {
            'AB_XA': AB_XA,
            'BC_AB': BC_AB,
            'CD_BC': CD_BC,
            'AD_XA': AD_XA,
        }
        
        # Check if ratios match
        confidence = self._calculate_confidence(ratios, expected)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine direction
        direction = 'bullish' if A.price < X.price else 'bearish'
        
        # Calculate PRZ and targets
        XA_range = abs(X.price - A.price)
        if direction == 'bullish':
            prz = X.price - (XA_range * 0.786)
            stop_loss = D.price * (1 - self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price + (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price + (CD_range * self.take_profit_2_pct)
        else:
            prz = X.price + (XA_range * 0.786)
            stop_loss = D.price * (1 + self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price - (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price - (CD_range * self.take_profit_2_pct)
        
        return HarmonicPattern(
            pattern_type='Gartley',
            direction=direction,
            X=X, A=A, B=B, C=C, D=D,
            prz_price=prz,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence
        )
    
    def _identify_bat(self, X: PatternPoint, A: PatternPoint, 
                     B: PatternPoint, C: PatternPoint, 
                     D: PatternPoint) -> Optional[HarmonicPattern]:
        """
        Identify Bat pattern
        
        Ratios:
        - AB = 38.2% to 50% of XA
        - BC = 38.2% to 88.6% of AB
        - CD = 161.8% to 261.8% of BC
        - AD = 88.6% of XA (key characteristic)
        """
        AB_XA = self._calculate_ratio(X.price, A.price, B.price)
        BC_AB = self._calculate_ratio(A.price, B.price, C.price)
        CD_BC = self._calculate_ratio(B.price, C.price, D.price)
        AD_XA = self._calculate_ratio(X.price, A.price, D.price)
        
        expected = {
            'AB_XA': (0.382, 0.500),
            'BC_AB': (0.382, 0.886),
            'CD_BC': (1.618, 2.618),
            'AD_XA': (0.841, 0.931),  # 88.6% ± tolerance
        }
        
        ratios = {
            'AB_XA': AB_XA,
            'BC_AB': BC_AB,
            'CD_BC': CD_BC,
            'AD_XA': AD_XA,
        }
        
        confidence = self._calculate_confidence(ratios, expected)
        
        if confidence < self.min_confidence:
            return None
        
        direction = 'bullish' if A.price < X.price else 'bearish'
        
        XA_range = abs(X.price - A.price)
        if direction == 'bullish':
            prz = X.price - (XA_range * 0.886)
            stop_loss = D.price * (1 - self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price + (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price + (CD_range * self.take_profit_2_pct)
        else:
            prz = X.price + (XA_range * 0.886)
            stop_loss = D.price * (1 + self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price - (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price - (CD_range * self.take_profit_2_pct)
        
        return HarmonicPattern(
            pattern_type='Bat',
            direction=direction,
            X=X, A=A, B=B, C=C, D=D,
            prz_price=prz,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence
        )
    
    def _identify_butterfly(self, X: PatternPoint, A: PatternPoint, 
                           B: PatternPoint, C: PatternPoint, 
                           D: PatternPoint) -> Optional[HarmonicPattern]:
        """
        Identify Butterfly pattern
        
        Ratios:
        - AB = 78.6% of XA
        - BC = 38.2% to 88.6% of AB
        - CD = 161.8% to 261.8% of BC
        - AD = 127% to 161.8% of XA (D extends beyond X)
        """
        AB_XA = self._calculate_ratio(X.price, A.price, B.price)
        BC_AB = self._calculate_ratio(A.price, B.price, C.price)
        CD_BC = self._calculate_ratio(B.price, C.price, D.price)
        AD_XA = self._calculate_ratio(X.price, A.price, D.price)
        
        expected = {
            'AB_XA': (0.746, 0.826),
            'BC_AB': (0.382, 0.886),
            'CD_BC': (1.618, 2.618),
            'AD_XA': (1.270, 1.618),  # Key: extends beyond X
        }
        
        ratios = {
            'AB_XA': AB_XA,
            'BC_AB': BC_AB,
            'CD_BC': CD_BC,
            'AD_XA': AD_XA,
        }
        
        confidence = self._calculate_confidence(ratios, expected)
        
        if confidence < self.min_confidence:
            return None
        
        direction = 'bullish' if A.price < X.price else 'bearish'
        
        XA_range = abs(X.price - A.price)
        if direction == 'bullish':
            prz = X.price - (XA_range * 1.272)
            stop_loss = D.price * (1 - self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price + (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price + (CD_range * self.take_profit_2_pct)
        else:
            prz = X.price + (XA_range * 1.272)
            stop_loss = D.price * (1 + self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price - (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price - (CD_range * self.take_profit_2_pct)
        
        return HarmonicPattern(
            pattern_type='Butterfly',
            direction=direction,
            X=X, A=A, B=B, C=C, D=D,
            prz_price=prz,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence
        )
    
    def _identify_crab(self, X: PatternPoint, A: PatternPoint, 
                      B: PatternPoint, C: PatternPoint, 
                      D: PatternPoint) -> Optional[HarmonicPattern]:
        """
        Identify Crab pattern
        
        Ratios:
        - AB = 38.2% to 61.8% of XA
        - BC = 38.2% to 88.6% of AB
        - CD = 261.8% to 361.8% of BC (very sharp)
        - AD = 161.8% of XA (most precise pattern)
        """
        AB_XA = self._calculate_ratio(X.price, A.price, B.price)
        BC_AB = self._calculate_ratio(A.price, B.price, C.price)
        CD_BC = self._calculate_ratio(B.price, C.price, D.price)
        AD_XA = self._calculate_ratio(X.price, A.price, D.price)
        
        expected = {
            'AB_XA': (0.382, 0.618),
            'BC_AB': (0.382, 0.886),
            'CD_BC': (2.618, 3.618),  # Very sharp CD leg
            'AD_XA': (1.533, 1.703),  # 161.8% ± tolerance
        }
        
        ratios = {
            'AB_XA': AB_XA,
            'BC_AB': BC_AB,
            'CD_BC': CD_BC,
            'AD_XA': AD_XA,
        }
        
        confidence = self._calculate_confidence(ratios, expected)
        
        if confidence < self.min_confidence:
            return None
        
        direction = 'bullish' if A.price < X.price else 'bearish'
        
        XA_range = abs(X.price - A.price)
        if direction == 'bullish':
            prz = X.price - (XA_range * 1.618)
            stop_loss = D.price * (1 - self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price + (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price + (CD_range * self.take_profit_2_pct)
        else:
            prz = X.price + (XA_range * 1.618)
            stop_loss = D.price * (1 + self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price - (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price - (CD_range * self.take_profit_2_pct)
        
        return HarmonicPattern(
            pattern_type='Crab',
            direction=direction,
            X=X, A=A, B=B, C=C, D=D,
            prz_price=prz,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence
        )
    
    def _identify_cypher(self, X: PatternPoint, A: PatternPoint, 
                        B: PatternPoint, C: PatternPoint, 
                        D: PatternPoint) -> Optional[HarmonicPattern]:
        """
        Identify Cypher pattern
        
        Ratios:
        - AB = 38.2% to 61.8% of XA
        - BC = 113% to 141.4% of XA (extends beyond A)
        - CD = 78.6% of XC
        """
        AB_XA = self._calculate_ratio(X.price, A.price, B.price)
        BC_XA = abs(B.price - C.price) / abs(X.price - A.price)
        XC = abs(X.price - C.price)
        CD = abs(C.price - D.price)
        CD_XC = CD / XC if XC != 0 else 0
        
        expected = {
            'AB_XA': (0.382, 0.618),
            'BC_XA': (1.130, 1.414),  # BC extends beyond A
            'CD_XC': (0.746, 0.826),  # 78.6% ± tolerance
        }
        
        ratios = {
            'AB_XA': AB_XA,
            'BC_XA': BC_XA,
            'CD_XC': CD_XC,
        }
        
        confidence = self._calculate_confidence(ratios, expected)
        
        if confidence < self.min_confidence:
            return None
        
        direction = 'bullish' if A.price < X.price else 'bearish'
        
        if direction == 'bullish':
            prz = X.price - (XC * 0.786)
            stop_loss = D.price * (1 - self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price + (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price + (CD_range * self.take_profit_2_pct)
        else:
            prz = X.price + (XC * 0.786)
            stop_loss = D.price * (1 + self.stop_loss_pct)
            CD_range = abs(C.price - D.price)
            take_profit_1 = D.price - (CD_range * self.take_profit_1_pct)
            take_profit_2 = D.price - (CD_range * self.take_profit_2_pct)
        
        return HarmonicPattern(
            pattern_type='Cypher',
            direction=direction,
            X=X, A=A, B=B, C=C, D=D,
            prz_price=prz,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            confidence=confidence
        )
    
    def _detect_patterns(self, swing_points: List[PatternPoint]) -> List[HarmonicPattern]:
        """
        Detect all harmonic patterns from swing points
        
        Returns list of detected patterns
        """
        patterns = []
        
        # Need at least 5 swing points for a pattern
        if len(swing_points) < 5:
            return patterns
        
        # Check last N swing point combinations
        # Look at the most recent 5 points (X, A, B, C, D)
        for i in range(len(swing_points) - 4):
            X = swing_points[i]
            A = swing_points[i + 1]
            B = swing_points[i + 2]
            C = swing_points[i + 3]
            D = swing_points[i + 4]
            
            # Check pattern duration
            pattern_bars = D.index - X.index
            if pattern_bars < self.min_pattern_bars or pattern_bars > self.max_pattern_bars:
                continue
            
            # Try to identify each pattern type
            if self.use_gartley:
                pattern = self._identify_gartley(X, A, B, C, D)
                if pattern:
                    patterns.append(pattern)
            
            if self.use_bat:
                pattern = self._identify_bat(X, A, B, C, D)
                if pattern:
                    patterns.append(pattern)
            
            if self.use_butterfly:
                pattern = self._identify_butterfly(X, A, B, C, D)
                if pattern:
                    patterns.append(pattern)
            
            if self.use_crab:
                pattern = self._identify_crab(X, A, B, C, D)
                if pattern:
                    patterns.append(pattern)
            
            if self.use_cypher:
                pattern = self._identify_cypher(X, A, B, C, D)
                if pattern:
                    patterns.append(pattern)
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return patterns
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on harmonic pattern detection
        
        Signal logic:
        - Buy at D point (PRZ) for bullish patterns
        - Sell at D point (PRZ) for bearish patterns
        - Use stop loss and take profit levels
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals and price levels
        """
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        signals['stop_price'] = np.nan
        signals['limit_price'] = np.nan
        signals['pattern_type'] = ''
        signals['confidence'] = 0.0
        
        # Track current position
        in_position = False
        current_pattern = None
        entry_price = 0
        
        for i in range(self.lookback_period, len(data)):
            # Get lookback window
            window_data = data.iloc[max(0, i - self.lookback_period):i + 1]
            
            # Identify swing points
            swing_points = self._identify_swing_points(window_data)
            
            if len(swing_points) < 5:
                continue
            
            # Detect patterns
            patterns = self._detect_patterns(swing_points)
            
            if not patterns:
                continue
            
            # Use highest confidence pattern
            best_pattern = patterns[0]
            
            # Check if we're at or near the D point (PRZ)
            current_price = data['Close'].iloc[i]
            current_idx = i
            
            # We're at D point if this is close to the last swing point
            at_d_point = (current_idx >= best_pattern.D.index - 2 and 
                         current_idx <= best_pattern.D.index + 2)
            
            if not in_position and at_d_point:
                # Price should be near PRZ
                prz_tolerance = 0.02  # 2% tolerance
                price_near_prz = abs(current_price - best_pattern.prz_price) / best_pattern.prz_price < prz_tolerance
                
                if price_near_prz:
                    # Generate signal based on pattern direction
                    if best_pattern.direction == 'bullish':
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    
                    signals.iloc[i, signals.columns.get_loc('stop_price')] = best_pattern.stop_loss
                    signals.iloc[i, signals.columns.get_loc('limit_price')] = best_pattern.take_profit_1
                    signals.iloc[i, signals.columns.get_loc('pattern_type')] = best_pattern.pattern_type
                    signals.iloc[i, signals.columns.get_loc('confidence')] = best_pattern.confidence
                    
                    in_position = True
                    current_pattern = best_pattern
                    entry_price = current_price
            
            elif in_position and current_pattern:
                # Check exit conditions
                hit_stop = False
                hit_target = False
                
                if current_pattern.direction == 'bullish':
                    hit_stop = current_price <= current_pattern.stop_loss
                    hit_target = current_price >= current_pattern.take_profit_1
                else:
                    hit_stop = current_price >= current_pattern.stop_loss
                    hit_target = current_price <= current_pattern.take_profit_1
                
                if hit_stop or hit_target:
                    # Exit position
                    if current_pattern.direction == 'bullish':
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    
                    in_position = False
                    current_pattern = None
        
        return signals[['signal', 'stop_price', 'limit_price']]


class SimpleHarmonicStrategy(Strategy):
    """
    Simplified Harmonic Pattern Strategy
    
    Focuses only on Gartley and Bat patterns with relaxed parameters.
    Good for beginners to understand harmonic trading.
    """
    
    def __init__(
        self,
        lookback_period: int = 80,
        zigzag_threshold: float = 0.04,
        min_confidence: float = 60.0,
    ):
        """
        Initialize Simple Harmonic Strategy
        
        Args:
            lookback_period: Bars to look back
            zigzag_threshold: Minimum swing size
            min_confidence: Minimum pattern confidence
        """
        super().__init__()
        
        # Use base strategy with simplified settings
        self.base_strategy = HarmonicPatternStrategy(
            lookback_period=lookback_period,
            min_pattern_bars=15,
            max_pattern_bars=60,
            zigzag_threshold=zigzag_threshold,
            ratio_tolerance=0.08,  # More lenient
            min_confidence=min_confidence,
            use_gartley=True,
            use_bat=True,
            use_butterfly=False,
            use_crab=False,
            use_cypher=False,
            stop_loss_pct=0.03,
            take_profit_1_pct=0.382,
            take_profit_2_pct=0.618,
        )
        
        self.parameters = self.base_strategy.parameters
        self.parameters['strategy_type'] = 'Simple (Gartley & Bat only)'
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using simplified harmonic detection"""
        return self.base_strategy.generate_signals(data)

