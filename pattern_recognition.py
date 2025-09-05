import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PatternMatch:
    pattern_type: str
    confidence: float  # 0 to 1
    target_price: float
    stop_loss: float
    formation_length: int
    volume_confirmation: bool
    supporting_indicators: Dict[str, float]

class PatternRecognizer:
    def __init__(self, min_pattern_bars: int = 5, max_pattern_bars: int = 50):
        self.min_pattern_bars = min_pattern_bars
        self.max_pattern_bars = max_pattern_bars
        
    def find_swing_points(self, data: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows in the price data."""
        highs = []
        lows = []
        
        for i in range(window, len(data) - window):
            if all(data['high'].iloc[i] > data['high'].iloc[i-window:i]) and \
               all(data['high'].iloc[i] > data['high'].iloc[i+1:i+window+1]):
                highs.append(i)
            if all(data['low'].iloc[i] < data['low'].iloc[i-window:i]) and \
               all(data['low'].iloc[i] < data['low'].iloc[i+1:i+window+1]):
                lows.append(i)
                
        return highs, lows

    def detect_head_and_shoulders(self, data: pd.DataFrame) -> Optional[PatternMatch]:
        """Detect head and shoulders pattern with precise measurements."""
        highs, lows = self.find_swing_points(data)
        
        for i in range(len(highs)-2):
            # Check for three peaks with middle one highest
            if data['high'].iloc[highs[i+1]] > data['high'].iloc[highs[i]] and \
               data['high'].iloc[highs[i+1]] > data['high'].iloc[highs[i+2]] and \
               abs(data['high'].iloc[highs[i]] - data['high'].iloc[highs[i+2]]) / \
               data['high'].iloc[highs[i]] < 0.03:  # Shoulders within 3%
                
                # Find neckline
                neckline_lows = [data['low'].iloc[lows[j]] for j in range(len(lows)) 
                               if lows[j] > highs[i] and lows[j] < highs[i+2]]
                
                if len(neckline_lows) >= 2:
                    neckline = np.mean(neckline_lows)
                    height = data['high'].iloc[highs[i+1]] - neckline
                    target = neckline - height
                    
                    # Calculate confidence based on various factors
                    confidence = 0.7  # Base confidence
                    
                    # Volume confirmation
                    vol_confirms = data['volume'].iloc[highs[i+1]] < data['volume'].iloc[highs[i]] and \
                                 data['volume'].iloc[highs[i+1]] < data['volume'].iloc[highs[i+2]]
                    
                    if vol_confirms:
                        confidence += 0.2
                    
                    return PatternMatch(
                        pattern_type="head_and_shoulders",
                        confidence=confidence,
                        target_price=target,
                        stop_loss=data['high'].iloc[highs[i+1]],
                        formation_length=highs[i+2] - highs[i],
                        volume_confirmation=vol_confirms,
                        supporting_indicators={}
                    )
        
        return None

    def detect_double_top_bottom(self, data: pd.DataFrame) -> Optional[PatternMatch]:
        """Detect double top or bottom patterns."""
        highs, lows = self.find_swing_points(data)
        
        # Check for double top
        for i in range(len(highs)-1):
            price_diff = abs(data['high'].iloc[highs[i]] - data['high'].iloc[highs[i+1]]) / \
                        data['high'].iloc[highs[i]]
            
            if price_diff < 0.02:  # Peaks within 2%
                # Find confirmation level (lowest point between peaks)
                between_lows = [data['low'].iloc[lows[j]] for j in range(len(lows))
                              if lows[j] > highs[i] and lows[j] < highs[i+1]]
                
                if between_lows:
                    neckline = min(between_lows)
                    height = data['high'].iloc[highs[i]] - neckline
                    target = neckline - height
                    
                    # Volume analysis
                    vol_confirms = data['volume'].iloc[highs[i+1]] < data['volume'].iloc[highs[i]]
                    
                    confidence = 0.7 + (0.2 if vol_confirms else 0)
                    
                    return PatternMatch(
                        pattern_type="double_top",
                        confidence=confidence,
                        target_price=target,
                        stop_loss=max(data['high'].iloc[highs[i]], data['high'].iloc[highs[i+1]]),
                        formation_length=highs[i+1] - highs[i],
                        volume_confirmation=vol_confirms,
                        supporting_indicators={}
                    )
        
        # Check for double bottom (similar logic, reversed)
        for i in range(len(lows)-1):
            price_diff = abs(data['low'].iloc[lows[i]] - data['low'].iloc[lows[i+1]]) / \
                        data['low'].iloc[lows[i]]
            
            if price_diff < 0.02:
                between_highs = [data['high'].iloc[highs[j]] for j in range(len(highs))
                               if highs[j] > lows[i] and highs[j] < lows[i+1]]
                
                if between_highs:
                    neckline = max(between_highs)
                    height = neckline - data['low'].iloc[lows[i]]
                    target = neckline + height
                    
                    vol_confirms = data['volume'].iloc[lows[i+1]] < data['volume'].iloc[lows[i]]
                    
                    confidence = 0.7 + (0.2 if vol_confirms else 0)
                    
                    return PatternMatch(
                        pattern_type="double_bottom",
                        confidence=confidence,
                        target_price=target,
                        stop_loss=min(data['low'].iloc[lows[i]], data['low'].iloc[lows[i+1]]),
                        formation_length=lows[i+1] - lows[i],
                        volume_confirmation=vol_confirms,
                        supporting_indicators={}
                    )
        
        return None

    def detect_triangle(self, data: pd.DataFrame) -> Optional[PatternMatch]:
        """Detect ascending, descending, and symmetric triangles."""
        highs, lows = self.find_swing_points(data)
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Calculate trend lines
        high_line = np.polyfit([highs[i] for i in range(min(3, len(highs)))],
                             [data['high'].iloc[highs[i]] for i in range(min(3, len(highs)))], 1)
        low_line = np.polyfit([lows[i] for i in range(min(3, len(lows)))],
                            [data['low'].iloc[lows[i]] for i in range(min(3, len(lows)))], 1)
        
        high_slope = high_line[0]
        low_slope = low_line[0]
        
        # Determine triangle type
        if abs(high_slope) < 0.0001 and low_slope > 0:
            pattern_type = "ascending_triangle"
            confidence = 0.75
        elif high_slope < 0 and abs(low_slope) < 0.0001:
            pattern_type = "descending_triangle"
            confidence = 0.75
        elif abs(high_slope + low_slope) < 0.0001:
            pattern_type = "symmetric_triangle"
            confidence = 0.65
        else:
            return None
        
        # Calculate target and stop loss
        formation_height = data['high'].iloc[highs[0]] - data['low'].iloc[lows[0]]
        current_price = data['close'].iloc[-1]
        
        if pattern_type in ["ascending_triangle", "symmetric_triangle"]:
            target_price = current_price + formation_height
            stop_loss = current_price - (formation_height * 0.5)
        else:
            target_price = current_price - formation_height
            stop_loss = current_price + (formation_height * 0.5)
        
        # Volume analysis
        recent_vol_avg = data['volume'].tail(5).mean()
        vol_trend = recent_vol_avg / data['volume'].tail(20).mean()
        vol_confirms = vol_trend < 0.8  # Decreasing volume in triangle formation
        
        if vol_confirms:
            confidence += 0.15
        
        return PatternMatch(
            pattern_type=pattern_type,
            confidence=min(confidence, 1.0),
            target_price=target_price,
            stop_loss=stop_loss,
            formation_length=len(data),
            volume_confirmation=vol_confirms,
            supporting_indicators={'volume_trend': vol_trend}
        )

    def detect_all_patterns(self, data: pd.DataFrame) -> List[PatternMatch]:
        """Detect all possible patterns in the data."""
        patterns = []
        
        # Head and Shoulders
        hs_pattern = self.detect_head_and_shoulders(data)
        if hs_pattern:
            patterns.append(hs_pattern)
        
        # Double Top/Bottom
        double_pattern = self.detect_double_top_bottom(data)
        if double_pattern:
            patterns.append(double_pattern)
        
        # Triangles
        triangle_pattern = self.detect_triangle(data)
        if triangle_pattern:
            patterns.append(triangle_pattern)
        
        return patterns
