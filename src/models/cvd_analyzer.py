"""
Spot vs Perpetual CVD (Cumulative Volume Delta) Analysis module.

This module provides advanced analysis of Spot vs Perpetual CVD patterns,
including divergence detection, trend analysis, and combined signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import math

from src.utils import (
    get_logger,
    performance_monitor,
    ModelError
)

# Import configuration
from src.config import (
    CVD_LOOKBACK_PERIOD,
    CVD_DIVERGENCE_THRESHOLD,
    CVD_SIGNAL_THRESHOLD,
    CVD_TREND_PERIOD
)

logger = get_logger("CVDAnalyzer")


class CVDAnalyzer:
    """
    Advanced Spot vs Perpetual CVD analyzer for market data.
    
    This class provides methods for:
    - CVD calculation and normalization
    - Spot vs Perp CVD divergence detection
    - CVD trend analysis
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = CVD_LOOKBACK_PERIOD,
                 divergence_threshold: float = CVD_DIVERGENCE_THRESHOLD,
                 signal_threshold: float = CVD_SIGNAL_THRESHOLD,
                 trend_period: int = CVD_TREND_PERIOD):
        """
        Initialize the CVD analyzer.
        
        Args:
            lookback_period: Lookback period for divergence detection
            divergence_threshold: Threshold for divergence detection
            signal_threshold: Threshold for signal generation
            trend_period: Period for trend analysis
        """
        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold
        self.signal_threshold = signal_threshold
        self.trend_period = trend_period
        
        logger.info(f"CVDAnalyzer initialized with lookback period: {lookback_period}, "
                   f"divergence threshold: {divergence_threshold}")
    
    @performance_monitor()
    def calculate_cvd(self, 
                    buy_volume: np.ndarray, 
                    sell_volume: np.ndarray) -> np.ndarray:
        """
        Calculate Cumulative Volume Delta (CVD).
        
        Args:
            buy_volume: Array of buy volume data
            sell_volume: Array of sell volume data
            
        Returns:
            Array of CVD values
        """
        if len(buy_volume) != len(sell_volume):
            raise ModelError("Buy and sell volume arrays must have the same length")
        
        if len(buy_volume) < 2:
            raise ModelError("Not enough data to calculate CVD")
        
        # Calculate delta volume (buy - sell)
        delta_volume = buy_volume - sell_volume
        
        # Calculate cumulative sum
        cvd = np.cumsum(delta_volume)
        
        return cvd
    
    @performance_monitor()
    def normalize_cvd(self, 
                     cvd: np.ndarray,
                     method: str = 'z-score') -> np.ndarray:
        """
        Normalize CVD for comparison.
        
        Args:
            cvd: Array of CVD values
            method: Normalization method ('z-score', 'min-max', or 'percent-change')
            
        Returns:
            Array of normalized CVD values
        """
        if len(cvd) < 2:
            raise ModelError("Not enough data to normalize CVD")
        
        if method == 'z-score':
            # Z-score normalization
            mean = np.mean(cvd)
            std = np.std(cvd)
            if std == 0:
                return np.zeros_like(cvd)
            return (cvd - mean) / std
        
        elif method == 'min-max':
            # Min-max normalization
            min_val = np.min(cvd)
            max_val = np.max(cvd)
            if max_val == min_val:
                return np.zeros_like(cvd)
            return (cvd - min_val) / (max_val - min_val)
        
        elif method == 'percent-change':
            # Percent change from first value
            first_val = cvd[0]
            if first_val == 0:
                # Avoid division by zero
                first_val = 1e-8
            return (cvd - first_val) / abs(first_val) * 100
        
        else:
            raise ModelError(f"Unknown normalization method: {method}")
    
    @performance_monitor()
    def detect_divergence(self, 
                        spot_cvd: np.ndarray, 
                        perp_cvd: np.ndarray,
                        lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect divergence between spot and perpetual CVD.
        
        Args:
            spot_cvd: Array of spot CVD data
            perp_cvd: Array of perpetual CVD data
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(spot_cvd) < lookback or len(perp_cvd) < lookback:
            raise ModelError(f"Not enough data to detect divergence. Need at least {lookback} data points.")
        
        if len(spot_cvd) != len(perp_cvd):
            raise ModelError("Spot and perpetual CVD arrays must have the same length")
        
        # Get the last lookback periods
        recent_spot_cvd = spot_cvd[-lookback:]
        recent_perp_cvd = perp_cvd[-lookback:]
        
        # Calculate CVD trends
        spot_trend = np.polyfit(np.arange(lookback), recent_spot_cvd, 1)[0]
        perp_trend = np.polyfit(np.arange(lookback), recent_perp_cvd, 1)[0]
        
        # Normalize trends
        spot_trend_norm = spot_trend / np.mean(np.abs(recent_spot_cvd)) * 100
        perp_trend_norm = perp_trend / np.mean(np.abs(recent_perp_cvd)) * 100
        
        # Calculate trend divergence
        trend_divergence = spot_trend_norm - perp_trend_norm
        
        # Detect bullish divergence (spot up, perp down or flat)
        bullish_divergence = (spot_trend_norm > self.divergence_threshold and 
                              perp_trend_norm < self.divergence_threshold / 2)
        
        # Detect bearish divergence (spot down, perp up or flat)
        bearish_divergence = (spot_trend_norm < -self.divergence_threshold and 
                              perp_trend_norm > -self.divergence_threshold / 2)
        
        # Detect extreme divergence
        extreme_divergence = abs(trend_divergence) > self.divergence_threshold * 2
        
        # Calculate correlation
        correlation = np.corrcoef(recent_spot_cvd, recent_perp_cvd)[0, 1]
        
        # Calculate divergence strength
        divergence_strength = 0.0
        if bullish_divergence:
            divergence_strength = abs(trend_divergence) / 100.0
        elif bearish_divergence:
            divergence_strength = -abs(trend_divergence) / 100.0
        
        # Adjust for correlation
        if correlation < 0:
            # Negative correlation strengthens divergence
            divergence_strength *= (1.0 - correlation)
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
            "extreme_divergence": extreme_divergence,
            "spot_trend": float(spot_trend_norm),
            "perp_trend": float(perp_trend_norm),
            "trend_divergence": float(trend_divergence),
            "correlation": float(correlation),
            "divergence_strength": float(divergence_strength)
        }
    
    @performance_monitor()
    def analyze_cvd_trend(self, 
                        cvd: np.ndarray, 
                        period: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze CVD trend.
        
        Args:
            cvd: Array of CVD data
            period: Period for trend analysis
            
        Returns:
            Dictionary with CVD trend analysis
        """
        if period is None:
            period = self.trend_period
        
        if len(cvd) < period + 1:
            raise ModelError(f"Not enough data to analyze CVD trend. Need at least {period + 1} data points.")
        
        # Calculate short-term trend (last 'period' periods)
        short_term_cvd = cvd[-period:]
        short_term_trend = np.polyfit(np.arange(period), short_term_cvd, 1)[0]
        
        # Calculate medium-term trend (last 'period*2' periods)
        medium_term_periods = min(period * 2, len(cvd))
        medium_term_cvd = cvd[-medium_term_periods:]
        medium_term_trend = np.polyfit(np.arange(medium_term_periods), medium_term_cvd, 1)[0]
        
        # Calculate long-term trend (last 'period*4' periods)
        long_term_periods = min(period * 4, len(cvd))
        long_term_cvd = cvd[-long_term_periods:]
        long_term_trend = np.polyfit(np.arange(long_term_periods), long_term_cvd, 1)[0]
        
        # Normalize trends
        short_term_trend_norm = short_term_trend / np.mean(np.abs(short_term_cvd)) * 100
        medium_term_trend_norm = medium_term_trend / np.mean(np.abs(medium_term_cvd)) * 100
        long_term_trend_norm = long_term_trend / np.mean(np.abs(long_term_cvd)) * 100
        
        # Determine trend direction
        short_term_direction = "up" if short_term_trend > 0 else "down"
        medium_term_direction = "up" if medium_term_trend > 0 else "down"
        long_term_direction = "up" if long_term_trend > 0 else "down"
        
        # Determine trend strength
        short_term_strength = abs(short_term_trend_norm)
        medium_term_strength = abs(medium_term_trend_norm)
        long_term_strength = abs(long_term_trend_norm)
        
        # Determine trend consistency
        consistent_trend = (
            (short_term_direction == medium_term_direction) and 
            (medium_term_direction == long_term_direction)
        )
        
        # Determine trend acceleration
        accelerating_trend = abs(short_term_trend_norm) > abs(medium_term_trend_norm)
        
        # Calculate overall trend score
        trend_score = 0.0
        if short_term_direction == "up":
            trend_score += short_term_strength * 0.5
        else:
            trend_score -= short_term_strength * 0.5
        
        if medium_term_direction == "up":
            trend_score += medium_term_strength * 0.3
        else:
            trend_score -= medium_term_strength * 0.3
        
        if long_term_direction == "up":
            trend_score += long_term_strength * 0.2
        else:
            trend_score -= long_term_strength * 0.2
        
        # Cap score
        trend_score = max(-100.0, min(100.0, trend_score))
        
        return {
            "short_term_trend": float(short_term_trend_norm),
            "medium_term_trend": float(medium_term_trend_norm),
            "long_term_trend": float(long_term_trend_norm),
            "short_term_direction": short_term_direction,
            "medium_term_direction": medium_term_direction,
            "long_term_direction": long_term_direction,
            "short_term_strength": float(short_term_strength),
            "medium_term_strength": float(medium_term_strength),
            "long_term_strength": float(long_term_strength),
            "consistent_trend": consistent_trend,
            "accelerating_trend": accelerating_trend,
            "trend_score": float(trend_score)
        }
    
    @performance_monitor()
    def analyze_cvd_momentum(self, 
                           spot_cvd: np.ndarray, 
                           perp_cvd: np.ndarray) -> Dict[str, Any]:
        """
        Analyze CVD momentum and relative strength.
        
        Args:
            spot_cvd: Array of spot CVD data
            perp_cvd: Array of perpetual CVD data
            
        Returns:
            Dictionary with CVD momentum analysis
        """
        if len(spot_cvd) < 5 or len(perp_cvd) < 5:
            raise ModelError("Not enough data to analyze CVD momentum. Need at least 5 data points.")
        
        if len(spot_cvd) != len(perp_cvd):
            raise ModelError("Spot and perpetual CVD arrays must have the same length")
        
        # Calculate spot CVD momentum (rate of change)
        spot_momentum_1 = spot_cvd[-1] - spot_cvd[-2]
        spot_momentum_5 = spot_cvd[-1] - spot_cvd[-5]
        
        # Calculate perp CVD momentum (rate of change)
        perp_momentum_1 = perp_cvd[-1] - perp_cvd[-2]
        perp_momentum_5 = perp_cvd[-1] - perp_cvd[-5]
        
        # Calculate relative momentum
        relative_momentum_1 = spot_momentum_1 - perp_momentum_1
        relative_momentum_5 = spot_momentum_5 - perp_momentum_5
        
        # Calculate spot vs perp ratio
        spot_perp_ratio = spot_cvd[-1] / perp_cvd[-1] if perp_cvd[-1] != 0 else float('inf')
        
        # Calculate spot vs perp ratio change
        prev_ratio = spot_cvd[-2] / perp_cvd[-2] if perp_cvd[-2] != 0 else float('inf')
        ratio_change = spot_perp_ratio - prev_ratio if prev_ratio != float('inf') and spot_perp_ratio != float('inf') else 0
        
        # Determine momentum strength
        momentum_strength = 0.0
        if spot_momentum_1 > 0 and perp_momentum_1 < 0:
            # Spot up, perp down - strong bullish
            momentum_strength = 0.8
        elif spot_momentum_1 > 0 and perp_momentum_1 > 0 and spot_momentum_1 > perp_momentum_1:
            # Both up, but spot stronger - bullish
            momentum_strength = 0.4
        elif spot_momentum_1 < 0 and perp_momentum_1 > 0:
            # Spot down, perp up - strong bearish
            momentum_strength = -0.8
        elif spot_momentum_1 < 0 and perp_momentum_1 < 0 and spot_momentum_1 < perp_momentum_1:
            # Both down, but spot weaker - bearish
            momentum_strength = -0.4
        
        # Adjust for 5-period momentum
        if (spot_momentum_5 > 0 and perp_momentum_5 < 0) or (spot_momentum_5 > 0 and perp_momentum_5 > 0 and spot_momentum_5 > perp_momentum_5):
            momentum_strength += 0.2
        elif (spot_momentum_5 < 0 and perp_momentum_5 > 0) or (spot_momentum_5 < 0 and perp_momentum_5 < 0 and spot_momentum_5 < perp_momentum_5):
            momentum_strength -= 0.2
        
        # Cap momentum strength
        momentum_strength = max(-1.0, min(1.0, momentum_strength))
        
        return {
            "spot_momentum_1": float(spot_momentum_1),
            "spot_momentum_5": float(spot_momentum_5),
            "perp_momentum_1": float(perp_momentum_1),
            "perp_momentum_5": float(perp_momentum_5),
            "relative_momentum_1": float(relative_momentum_1),
            "relative_momentum_5": float(relative_momentum_5),
            "spot_perp_ratio": float(spot_perp_ratio) if spot_perp_ratio != float('inf') else None,
            "ratio_change": float(ratio_change),
            "momentum_strength": float(momentum_strength)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      divergence_analysis: Dict[str, Any],
                      spot_trend_analysis: Dict[str, Any],
                      perp_trend_analysis: Dict[str, Any],
                      momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined CVD signal.
        
        Args:
            divergence_analysis: Divergence analysis results
            spot_trend_analysis: Spot CVD trend analysis results
            perp_trend_analysis: Perp CVD trend analysis results
            momentum_analysis: Momentum analysis results
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Divergence Component (range: -0.4 to 0.4)
        divergence_score = 0.0
        if divergence_analysis["bullish_divergence"]:
            divergence_score = 0.3
        elif divergence_analysis["bearish_divergence"]:
            divergence_score = -0.3
        
        # Adjust by strength
        divergence_score *= abs(divergence_analysis["divergence_strength"]) * 2
        
        # Adjust for extreme divergence
        if divergence_analysis["extreme_divergence"]:
            divergence_score *= 1.3
        
        # Adjust for correlation
        correlation = divergence_analysis["correlation"]
        if correlation < 0:
            # Negative correlation strengthens the divergence signal
            divergence_score *= (1.0 - correlation)
        
        # Cap score
        divergence_score = max(-0.4, min(0.4, divergence_score))
        signal_components["divergence"] = divergence_score
        
        # 2. Trend Component (range: -0.3 to 0.3)
        # Combine spot and perp trends with spot having higher weight
        spot_trend_score = spot_trend_analysis["trend_score"] / 100.0 * 0.2
        perp_trend_score = perp_trend_analysis["trend_score"] / 100.0 * 0.1
        
        trend_score = spot_trend_score + perp_trend_score
        
        # Adjust for trend consistency
        if spot_trend_analysis["consistent_trend"]:
            trend_score *= 1.2
        
        # Adjust for trend acceleration
        if spot_trend_analysis["accelerating_trend"]:
            if trend_score > 0:
                trend_score *= 1.1
            else:
                trend_score *= 1.1
        
        # Cap score
        trend_score = max(-0.3, min(0.3, trend_score))
        signal_components["trend"] = trend_score
        
        # 3. Momentum Component (range: -0.3 to 0.3)
        momentum_score = momentum_analysis["momentum_strength"] * 0.3
        
        # Cap score
        momentum_score = max(-0.3, min(0.3, momentum_score))
        signal_components["momentum"] = momentum_score
        
        # Calculate final score
        final_score = sum(signal_components.values())
        
        # Cap final score
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determine signal
        signal = "NEUTRAL"
        if final_score > self.signal_threshold:
            signal = "BUY"
        elif final_score > self.signal_threshold * 2:
            signal = "STRONG BUY"
        elif final_score < -self.signal_threshold:
            signal = "SELL"
        elif final_score < -self.signal_threshold * 2:
            signal = "STRONG SELL"
        
        return {
            "signal": signal,
            "score": float(final_score),
            "components": signal_components,
            "confidence": min(1.0, abs(final_score) * 1.5)
        }
    
    @performance_monitor()
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete Spot vs Perp CVD analysis on a DataFrame.
        
        Args:
            df: DataFrame with spot and perpetual market data
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["spot_buy_volume", "spot_sell_volume", "perp_buy_volume", "perp_sell_volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        spot_buy_volume = df["spot_buy_volume"].values
        spot_sell_volume = df["spot_sell_volume"].values
        perp_buy_volume = df["perp_buy_volume"].values
        perp_sell_volume = df["perp_sell_volume"].values
        
        # Calculate CVD
        spot_cvd = self.calculate_cvd(spot_buy_volume, spot_sell_volume)
        perp_cvd = self.calculate_cvd(perp_buy_volume, perp_sell_volume)
        
        # Normalize CVD for comparison
        spot_cvd_norm = self.normalize_cvd(spot_cvd, method='z-score')
        perp_cvd_norm = self.normalize_cvd(perp_cvd, method='z-score')
        
        # Perform analyses
        divergence_analysis = self.detect_divergence(spot_cvd_norm, perp_cvd_norm)
        spot_trend_analysis = self.analyze_cvd_trend(spot_cvd_norm)
        perp_trend_analysis = self.analyze_cvd_trend(perp_cvd_norm)
        momentum_analysis = self.analyze_cvd_momentum(spot_cvd_norm, perp_cvd_norm)
        
        # Generate signal
        signal = self.generate_signal(
            divergence_analysis,
            spot_trend_analysis,
            perp_trend_analysis,
            momentum_analysis
        )
        
        # Combine all results
        result = {
            "metrics": {
                "spot_cvd": spot_cvd.tolist(),
                "perp_cvd": perp_cvd.tolist(),
                "spot_cvd_norm": spot_cvd_norm.tolist(),
                "perp_cvd_norm": perp_cvd_norm.tolist()
            },
            "divergence_analysis": divergence_analysis,
            "spot_trend_analysis": spot_trend_analysis,
            "perp_trend_analysis": perp_trend_analysis,
            "momentum_analysis": momentum_analysis,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def calculate_cvd(buy_volume: np.ndarray, sell_volume: np.ndarray) -> np.ndarray:
    """
    Standalone function to calculate CVD.
    
    Args:
        buy_volume: Array of buy volume data
        sell_volume: Array of sell volume data
        
    Returns:
        Array of CVD values
    """
    analyzer = CVDAnalyzer()
    return analyzer.calculate_cvd(buy_volume, sell_volume)


def analyze_spot_perp_cvd(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze spot vs perp CVD.
    
    Args:
        df: DataFrame with spot and perpetual market data
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = CVDAnalyzer()
    return analyzer.analyze(df)


# Test function
def test_cvd_analyzer():
    """Test the CVD analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    periods = 100
    np.random.seed(42)
    
    # Generate buy/sell volume data with trends
    # Spot: Initially balanced, then more buying
    # Perp: Initially balanced, then more selling
    spot_buy_base = np.random.normal(100, 20, periods)
    spot_sell_base = np.random.normal(100, 20, periods)
    perp_buy_base = np.random.normal(100, 20, periods)
    perp_sell_base = np.random.normal(100, 20, periods)
    
    # Add trend
    spot_buy_trend = np.linspace(0, 50, periods)
    spot_sell_trend = np.linspace(0, 0, periods)
    perp_buy_trend = np.linspace(0, 0, periods)
    perp_sell_trend = np.linspace(0, 50, periods)
    
    spot_buy_volume = spot_buy_base + spot_buy_trend
    spot_sell_volume = spot_sell_base + spot_sell_trend
    perp_buy_volume = perp_buy_base + perp_buy_trend
    perp_sell_volume = perp_sell_base + perp_sell_trend
    
    # Create DataFrame
    df = pd.DataFrame({
        "spot_buy_volume": spot_buy_volume,
        "spot_sell_volume": spot_sell_volume,
        "perp_buy_volume": perp_buy_volume,
        "perp_sell_volume": perp_sell_volume
    })
    
    # Create analyzer
    analyzer = CVDAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(df)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nDivergence Analysis:")
    print(f"  Bullish Divergence: {result['divergence_analysis']['bullish_divergence']}")
    print(f"  Bearish Divergence: {result['divergence_analysis']['bearish_divergence']}")
    print(f"  Extreme Divergence: {result['divergence_analysis']['extreme_divergence']}")
    print(f"  Spot Trend: {result['divergence_analysis']['spot_trend']:.3f}")
    print(f"  Perp Trend: {result['divergence_analysis']['perp_trend']:.3f}")
    print(f"  Correlation: {result['divergence_analysis']['correlation']:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Spot and Perp CVD
    plt.subplot(2, 1, 1)
    plt.plot(result["metrics"]["spot_cvd"], label="Spot CVD")
    plt.plot(result["metrics"]["perp_cvd"], label="Perp CVD")
    plt.title("Spot and Perp CVD")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Normalized CVD
    plt.subplot(2, 1, 2)
    plt.plot(result["metrics"]["spot_cvd_norm"], label="Spot CVD (Normalized)")
    plt.plot(result["metrics"]["perp_cvd_norm"], label="Perp CVD (Normalized)")
    plt.title("Normalized CVD")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_cvd_analyzer()
