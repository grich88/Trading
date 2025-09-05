import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RSIAnalysis:
    current_value: float
    historical_avg: float
    trend_strength: float
    divergence: bool
    support_level: float
    resistance_level: float
    range_status: str  # 'overbought', 'oversold', 'neutral'
    
class TechnicalAnalyzer:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        
    def analyze_rsi(self, data: pd.DataFrame, period: int = 14) -> RSIAnalysis:
        """
        Comprehensive RSI analysis including historical comparisons and pattern detection.
        """
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate historical average
        hist_avg = rsi.rolling(window=self.lookback_period).mean()
        
        # Calculate trend strength (how far RSI is from its moving average)
        trend_strength = abs(rsi - hist_avg) / hist_avg * 100
        
        # Detect divergence
        price_higher_highs = data['close'].diff().rolling(window=5).apply(lambda x: all(x > 0))
        rsi_lower_highs = rsi.diff().rolling(window=5).apply(lambda x: all(x < 0))
        divergence = any(price_higher_highs & rsi_lower_highs)
        
        # Calculate support and resistance levels
        rsi_sorted = sorted(rsi.dropna())
        support_level = np.percentile(rsi_sorted, 25)
        resistance_level = np.percentile(rsi_sorted, 75)
        
        # Determine range status
        current_rsi = rsi.iloc[-1]
        if current_rsi > 70:
            range_status = 'overbought'
        elif current_rsi < 30:
            range_status = 'oversold'
        else:
            range_status = 'neutral'
            
        return RSIAnalysis(
            current_value=current_rsi,
            historical_avg=hist_avg.iloc[-1],
            trend_strength=trend_strength.iloc[-1],
            divergence=divergence,
            support_level=support_level,
            resistance_level=resistance_level,
            range_status=range_status
        )
    
    def detect_candlestick_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect various candlestick patterns.
        """
        patterns = {}
        
        # Bullish Engulfing
        patterns['bullish_engulfing'] = (
            (data['open'].iloc[-2] > data['close'].iloc[-2]) &  # Previous red candle
            (data['close'].iloc[-1] > data['open'].iloc[-1]) &  # Current green candle
            (data['open'].iloc[-1] < data['close'].iloc[-2]) &  # Opens below previous close
            (data['close'].iloc[-1] > data['open'].iloc[-2])    # Closes above previous open
        )
        
        # Bearish Engulfing
        patterns['bearish_engulfing'] = (
            (data['close'].iloc[-2] > data['open'].iloc[-2]) &  # Previous green candle
            (data['open'].iloc[-1] > data['close'].iloc[-1]) &  # Current red candle
            (data['close'].iloc[-1] < data['open'].iloc[-2]) &  # Closes below previous open
            (data['open'].iloc[-1] > data['close'].iloc[-2])    # Opens above previous close
        )
        
        # Doji
        body_size = abs(data['close'] - data['open'])
        wick_size = data['high'] - data['low']
        patterns['doji'] = body_size.iloc[-1] < (wick_size.iloc[-1] * 0.1)
        
        # Hammer
        if data['close'].iloc[-1] > data['open'].iloc[-1]:
            body_top = data['close'].iloc[-1]
            body_bottom = data['open'].iloc[-1]
        else:
            body_top = data['open'].iloc[-1]
            body_bottom = data['close'].iloc[-1]
        
        upper_wick = data['high'].iloc[-1] - body_top
        lower_wick = body_bottom - data['low'].iloc[-1]
        body = abs(data['close'].iloc[-1] - data['open'].iloc[-1])
        
        patterns['hammer'] = (
            (lower_wick > (body * 2)) &  # Lower wick at least 2x body
            (upper_wick < (body * 0.5))  # Upper wick less than half body
        )
        
        return patterns
    
    def analyze_volume_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volume patterns and trends.
        """
        volume_analysis = {}
        
        # Volume moving averages
        volume_analysis['vol_sma_5'] = data['volume'].rolling(window=5).mean().iloc[-1]
        volume_analysis['vol_sma_20'] = data['volume'].rolling(window=20).mean().iloc[-1]
        
        # Volume trend (comparing to historical average)
        hist_vol_avg = data['volume'].rolling(window=self.lookback_period).mean().iloc[-1]
        volume_analysis['vol_trend'] = (data['volume'].iloc[-1] / hist_vol_avg - 1) * 100
        
        # Volume acceleration (rate of change)
        vol_roc = data['volume'].pct_change()
        volume_analysis['vol_acceleration'] = vol_roc.rolling(window=5).mean().iloc[-1] * 100
        
        # Volume support/resistance levels
        vol_sorted = sorted(data['volume'].dropna())
        volume_analysis['vol_support'] = np.percentile(vol_sorted, 25)
        volume_analysis['vol_resistance'] = np.percentile(vol_sorted, 75)
        
        return volume_analysis
    
    def detect_chart_patterns(self, data: pd.DataFrame) -> Dict[str, Tuple[bool, float]]:
        """
        Detect chart patterns and return their reliability scores.
        """
        patterns = {}
        
        # Head and Shoulders
        patterns['head_and_shoulders'] = self._detect_head_and_shoulders(data)
        
        # Double Top
        patterns['double_top'] = self._detect_double_top(data)
        
        # Double Bottom
        patterns['double_bottom'] = self._detect_double_bottom(data)
        
        # Triangle Patterns
        patterns['ascending_triangle'] = self._detect_ascending_triangle(data)
        patterns['descending_triangle'] = self._detect_descending_triangle(data)
        
        return patterns
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect Head and Shoulders pattern.
        Returns (pattern_exists, reliability_score)
        """
        # Implementation here
        return (False, 0.0)  # Placeholder
    
    def _detect_double_top(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect Double Top pattern.
        Returns (pattern_exists, reliability_score)
        """
        # Implementation here
        return (False, 0.0)  # Placeholder
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect Double Bottom pattern.
        Returns (pattern_exists, reliability_score)
        """
        # Implementation here
        return (False, 0.0)  # Placeholder
    
    def _detect_ascending_triangle(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect Ascending Triangle pattern.
        Returns (pattern_exists, reliability_score)
        """
        # Implementation here
        return (False, 0.0)  # Placeholder
    
    def _detect_descending_triangle(self, data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Detect Descending Triangle pattern.
        Returns (pattern_exists, reliability_score)
        """
        # Implementation here
        return (False, 0.0)  # Placeholder
