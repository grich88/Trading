"""
Open Interest vs Price Divergence Analysis module.

This module provides advanced analysis of Open Interest (OI) and price movements,
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
    OI_CHANGE_THRESHOLD,
    OI_LOOKBACK_PERIOD,
    OI_DIVERGENCE_THRESHOLD,
    OI_SIGNAL_THRESHOLD,
    OI_TREND_PERIOD
)

logger = get_logger("OpenInterestAnalyzer")


class OpenInterestAnalyzer:
    """
    Advanced Open Interest analyzer for market data.
    
    This class provides methods for:
    - Open Interest trend analysis
    - Price vs OI divergence detection
    - Combined signal generation
    - Open Interest change rate analysis
    """
    
    def __init__(self, 
                 change_threshold: float = OI_CHANGE_THRESHOLD,
                 lookback_period: int = OI_LOOKBACK_PERIOD,
                 divergence_threshold: float = OI_DIVERGENCE_THRESHOLD,
                 signal_threshold: float = OI_SIGNAL_THRESHOLD,
                 trend_period: int = OI_TREND_PERIOD):
        """
        Initialize the Open Interest analyzer.
        
        Args:
            change_threshold: Threshold for significant OI change (percentage)
            lookback_period: Lookback period for divergence detection
            divergence_threshold: Threshold for divergence detection
            signal_threshold: Threshold for signal generation
            trend_period: Period for trend analysis
        """
        self.change_threshold = change_threshold
        self.lookback_period = lookback_period
        self.divergence_threshold = divergence_threshold
        self.signal_threshold = signal_threshold
        self.trend_period = trend_period
        
        logger.info(f"OpenInterestAnalyzer initialized with lookback period: {lookback_period}, "
                   f"change threshold: {change_threshold}%")
    
    @performance_monitor()
    def calculate_oi_metrics(self, 
                           open_interest: np.ndarray, 
                           prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Open Interest metrics.
        
        Args:
            open_interest: Array of open interest data
            prices: Array of price data
            
        Returns:
            Dictionary with OI metrics
        """
        if len(open_interest) != len(prices):
            raise ModelError("Open interest and price arrays must have the same length")
        
        if len(open_interest) < 2:
            raise ModelError("Not enough data to calculate OI metrics")
        
        # Calculate OI change
        oi_change = np.zeros_like(open_interest)
        oi_change[1:] = np.diff(open_interest)
        
        # Calculate OI change percentage
        oi_change_pct = np.zeros_like(open_interest, dtype=float)
        oi_change_pct[1:] = (np.diff(open_interest) / open_interest[:-1]) * 100
        
        # Calculate OI rate of change (ROC)
        oi_roc = np.zeros_like(open_interest, dtype=float)
        if len(open_interest) >= self.trend_period:
            oi_roc[self.trend_period:] = (
                (open_interest[self.trend_period:] - open_interest[:-self.trend_period]) / 
                open_interest[:-self.trend_period]
            ) * 100
        
        # Calculate price change
        price_change = np.zeros_like(prices)
        price_change[1:] = np.diff(prices)
        
        # Calculate price change percentage
        price_change_pct = np.zeros_like(prices, dtype=float)
        price_change_pct[1:] = (np.diff(prices) / prices[:-1]) * 100
        
        # Calculate price rate of change (ROC)
        price_roc = np.zeros_like(prices, dtype=float)
        if len(prices) >= self.trend_period:
            price_roc[self.trend_period:] = (
                (prices[self.trend_period:] - prices[:-self.trend_period]) / 
                prices[:-self.trend_period]
            ) * 100
        
        # Calculate OI to price ratio
        oi_price_ratio = open_interest / prices
        
        # Calculate OI to price ratio change
        oi_price_ratio_change = np.zeros_like(oi_price_ratio)
        oi_price_ratio_change[1:] = np.diff(oi_price_ratio)
        
        return {
            "oi_change": oi_change,
            "oi_change_pct": oi_change_pct,
            "oi_roc": oi_roc,
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "price_roc": price_roc,
            "oi_price_ratio": oi_price_ratio,
            "oi_price_ratio_change": oi_price_ratio_change
        }
    
    @performance_monitor()
    def detect_divergence(self, 
                        open_interest: np.ndarray, 
                        prices: np.ndarray,
                        lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect divergence between price and open interest.
        
        Args:
            open_interest: Array of open interest data
            prices: Array of price data
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(open_interest) < lookback or len(prices) < lookback:
            raise ModelError(f"Not enough data to detect divergence. Need at least {lookback} data points.")
        
        # Get the last lookback periods
        recent_oi = open_interest[-lookback:]
        recent_prices = prices[-lookback:]
        
        # Find local maxima and minima
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        oi_max_idx = np.argmax(recent_oi)
        oi_min_idx = np.argmin(recent_oi)
        
        # Calculate price and OI trends
        price_trend = np.polyfit(np.arange(lookback), recent_prices, 1)[0]
        oi_trend = np.polyfit(np.arange(lookback), recent_oi, 1)[0]
        
        # Normalize trends
        price_trend_norm = price_trend / np.mean(recent_prices) * 100
        oi_trend_norm = oi_trend / np.mean(recent_oi) * 100
        
        # Calculate trend divergence
        trend_divergence = price_trend_norm - oi_trend_norm
        
        # Detect bullish divergence (price down, OI up)
        bullish_divergence = (price_trend_norm < -self.divergence_threshold and 
                              oi_trend_norm > self.divergence_threshold)
        
        # Detect bearish divergence (price up, OI down)
        bearish_divergence = (price_trend_norm > self.divergence_threshold and 
                              oi_trend_norm < -self.divergence_threshold)
        
        # Detect extreme divergence
        extreme_divergence = abs(trend_divergence) > self.divergence_threshold * 2
        
        # Calculate divergence strength
        divergence_strength = 0.0
        if bullish_divergence:
            divergence_strength = abs(trend_divergence) / 100.0
        elif bearish_divergence:
            divergence_strength = -abs(trend_divergence) / 100.0
        
        # Check for local extrema divergence
        local_extrema_divergence = False
        if price_max_idx > 0 and oi_max_idx > 0:
            # Price makes higher high but OI makes lower high
            if (recent_prices[price_max_idx] > recent_prices[price_max_idx-1] and
                recent_oi[oi_max_idx] < recent_oi[oi_max_idx-1]):
                local_extrema_divergence = True
                divergence_strength -= 0.2
        
        if price_min_idx > 0 and oi_min_idx > 0:
            # Price makes lower low but OI makes higher low
            if (recent_prices[price_min_idx] < recent_prices[price_min_idx-1] and
                recent_oi[oi_min_idx] > recent_oi[oi_min_idx-1]):
                local_extrema_divergence = True
                divergence_strength += 0.2
        
        return {
            "bullish_divergence": bullish_divergence,
            "bearish_divergence": bearish_divergence,
            "extreme_divergence": extreme_divergence,
            "local_extrema_divergence": local_extrema_divergence,
            "price_trend": float(price_trend_norm),
            "oi_trend": float(oi_trend_norm),
            "trend_divergence": float(trend_divergence),
            "divergence_strength": float(divergence_strength)
        }
    
    @performance_monitor()
    def analyze_oi_trend(self, 
                       open_interest: np.ndarray, 
                       period: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze Open Interest trend.
        
        Args:
            open_interest: Array of open interest data
            period: Period for trend analysis
            
        Returns:
            Dictionary with OI trend analysis
        """
        if period is None:
            period = self.trend_period
        
        if len(open_interest) < period + 1:
            raise ModelError(f"Not enough data to analyze OI trend. Need at least {period + 1} data points.")
        
        # Calculate short-term trend (last 'period' periods)
        short_term_oi = open_interest[-period:]
        short_term_trend = np.polyfit(np.arange(period), short_term_oi, 1)[0]
        
        # Calculate medium-term trend (last 'period*2' periods)
        medium_term_periods = min(period * 2, len(open_interest))
        medium_term_oi = open_interest[-medium_term_periods:]
        medium_term_trend = np.polyfit(np.arange(medium_term_periods), medium_term_oi, 1)[0]
        
        # Calculate long-term trend (last 'period*4' periods)
        long_term_periods = min(period * 4, len(open_interest))
        long_term_oi = open_interest[-long_term_periods:]
        long_term_trend = np.polyfit(np.arange(long_term_periods), long_term_oi, 1)[0]
        
        # Normalize trends
        short_term_trend_norm = short_term_trend / np.mean(short_term_oi) * 100
        medium_term_trend_norm = medium_term_trend / np.mean(medium_term_oi) * 100
        long_term_trend_norm = long_term_trend / np.mean(long_term_oi) * 100
        
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
    def analyze_significant_changes(self, 
                                  open_interest: np.ndarray,
                                  prices: np.ndarray,
                                  threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Analyze significant changes in Open Interest.
        
        Args:
            open_interest: Array of open interest data
            prices: Array of price data
            threshold: Threshold for significant change
            
        Returns:
            Dictionary with significant change analysis
        """
        if threshold is None:
            threshold = self.change_threshold
        
        if len(open_interest) < 5 or len(prices) < 5:
            raise ModelError("Not enough data to analyze significant changes. Need at least 5 data points.")
        
        # Calculate OI change percentage
        oi_change_pct = np.zeros_like(open_interest, dtype=float)
        oi_change_pct[1:] = (np.diff(open_interest) / open_interest[:-1]) * 100
        
        # Calculate price change percentage
        price_change_pct = np.zeros_like(prices, dtype=float)
        price_change_pct[1:] = (np.diff(prices) / prices[:-1]) * 100
        
        # Detect significant OI increases
        significant_increases = oi_change_pct > threshold
        
        # Detect significant OI decreases
        significant_decreases = oi_change_pct < -threshold
        
        # Count recent significant changes
        recent_increases = np.sum(significant_increases[-5:])
        recent_decreases = np.sum(significant_decreases[-5:])
        
        # Detect significant OI increase with price increase
        oi_up_price_up = np.logical_and(oi_change_pct > threshold, price_change_pct > 0)
        
        # Detect significant OI increase with price decrease
        oi_up_price_down = np.logical_and(oi_change_pct > threshold, price_change_pct < 0)
        
        # Detect significant OI decrease with price increase
        oi_down_price_up = np.logical_and(oi_change_pct < -threshold, price_change_pct > 0)
        
        # Detect significant OI decrease with price decrease
        oi_down_price_down = np.logical_and(oi_change_pct < -threshold, price_change_pct < 0)
        
        # Count recent combined signals
        recent_oi_up_price_up = np.sum(oi_up_price_up[-5:])
        recent_oi_up_price_down = np.sum(oi_up_price_down[-5:])
        recent_oi_down_price_up = np.sum(oi_down_price_up[-5:])
        recent_oi_down_price_down = np.sum(oi_down_price_down[-5:])
        
        # Calculate signal strength
        signal_strength = 0.0
        
        # OI up, price up: Bullish continuation
        signal_strength += recent_oi_up_price_up * 0.1
        
        # OI up, price down: Potential reversal (bearish continuation)
        signal_strength -= recent_oi_up_price_down * 0.2
        
        # OI down, price up: Weak rally, potential reversal
        signal_strength -= recent_oi_down_price_up * 0.15
        
        # OI down, price down: Bearish continuation
        signal_strength -= recent_oi_down_price_down * 0.1
        
        return {
            "significant_increases": significant_increases.tolist(),
            "significant_decreases": significant_decreases.tolist(),
            "recent_increases": int(recent_increases),
            "recent_decreases": int(recent_decreases),
            "recent_oi_up_price_up": int(recent_oi_up_price_up),
            "recent_oi_up_price_down": int(recent_oi_up_price_down),
            "recent_oi_down_price_up": int(recent_oi_down_price_up),
            "recent_oi_down_price_down": int(recent_oi_down_price_down),
            "signal_strength": float(signal_strength)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      divergence_analysis: Dict[str, Any],
                      trend_analysis: Dict[str, Any],
                      change_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined Open Interest signal.
        
        Args:
            divergence_analysis: Divergence analysis results
            trend_analysis: Trend analysis results
            change_analysis: Significant change analysis results
            
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
        
        # Adjust for local extrema divergence
        if divergence_analysis["local_extrema_divergence"]:
            if divergence_score > 0:
                divergence_score += 0.1
            elif divergence_score < 0:
                divergence_score -= 0.1
        
        # Cap score
        divergence_score = max(-0.4, min(0.4, divergence_score))
        signal_components["divergence"] = divergence_score
        
        # 2. Trend Component (range: -0.3 to 0.3)
        trend_score = trend_analysis["trend_score"] / 100.0 * 0.3
        
        # Adjust for trend consistency
        if trend_analysis["consistent_trend"]:
            trend_score *= 1.2
        
        # Adjust for trend acceleration
        if trend_analysis["accelerating_trend"]:
            if trend_score > 0:
                trend_score *= 1.1
            else:
                trend_score *= 1.1
        
        # Cap score
        trend_score = max(-0.3, min(0.3, trend_score))
        signal_components["trend"] = trend_score
        
        # 3. Change Component (range: -0.3 to 0.3)
        change_score = change_analysis["signal_strength"]
        
        # Cap score
        change_score = max(-0.3, min(0.3, change_score))
        signal_components["change"] = change_score
        
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
        Perform complete Open Interest analysis on a DataFrame.
        
        Args:
            df: DataFrame with price and open interest data
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["close", "open_interest"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        prices = df["close"].values
        open_interest = df["open_interest"].values
        
        # Calculate metrics
        metrics = self.calculate_oi_metrics(open_interest, prices)
        
        # Perform analyses
        divergence_analysis = self.detect_divergence(open_interest, prices)
        trend_analysis = self.analyze_oi_trend(open_interest)
        change_analysis = self.analyze_significant_changes(open_interest, prices)
        
        # Generate signal
        signal = self.generate_signal(
            divergence_analysis,
            trend_analysis,
            change_analysis
        )
        
        # Combine all results
        result = {
            "metrics": {
                "oi_change": metrics["oi_change"].tolist(),
                "oi_change_pct": metrics["oi_change_pct"].tolist(),
                "oi_roc": metrics["oi_roc"].tolist(),
                "price_change": metrics["price_change"].tolist(),
                "price_change_pct": metrics["price_change_pct"].tolist(),
                "price_roc": metrics["price_roc"].tolist(),
                "oi_price_ratio": metrics["oi_price_ratio"].tolist(),
                "oi_price_ratio_change": metrics["oi_price_ratio_change"].tolist()
            },
            "divergence_analysis": divergence_analysis,
            "trend_analysis": trend_analysis,
            "change_analysis": change_analysis,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def analyze_open_interest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze open interest.
    
    Args:
        df: DataFrame with price and open interest data
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = OpenInterestAnalyzer()
    return analyzer.analyze(df)


def detect_oi_divergence(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to detect open interest divergence.
    
    Args:
        df: DataFrame with price and open interest data
        
    Returns:
        Dictionary with divergence analysis
    """
    if "close" not in df.columns or "open_interest" not in df.columns:
        raise ModelError("DataFrame must contain 'close' and 'open_interest' columns")
    
    analyzer = OpenInterestAnalyzer()
    return analyzer.detect_divergence(df["open_interest"].values, df["close"].values)


# Test function
def test_open_interest_analyzer():
    """Test the Open Interest analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    periods = 100
    np.random.seed(42)
    
    # Generate price data with trend and noise
    prices = np.cumsum(np.random.normal(0, 1, periods)) + 100
    
    # Generate open interest data with some divergence from price
    # First half: OI follows price
    # Second half: OI diverges from price
    oi_first_half = prices[:periods//2] * (1 + np.random.normal(0, 0.05, periods//2))
    oi_second_half = prices[periods//2:] * (1 - np.random.normal(0, 0.05, periods - periods//2))
    open_interest = np.concatenate([oi_first_half, oi_second_half])
    
    # Create DataFrame
    df = pd.DataFrame({
        "close": prices,
        "open_interest": open_interest
    })
    
    # Create analyzer
    analyzer = OpenInterestAnalyzer()
    
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
    print(f"  Price Trend: {result['divergence_analysis']['price_trend']:.3f}")
    print(f"  OI Trend: {result['divergence_analysis']['oi_trend']:.3f}")
    
    print("\nTrend Analysis:")
    print(f"  Short-term Trend: {result['trend_analysis']['short_term_trend']:.3f}")
    print(f"  Medium-term Trend: {result['trend_analysis']['medium_term_trend']:.3f}")
    print(f"  Long-term Trend: {result['trend_analysis']['long_term_trend']:.3f}")
    print(f"  Consistent Trend: {result['trend_analysis']['consistent_trend']}")
    print(f"  Accelerating Trend: {result['trend_analysis']['accelerating_trend']}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price and Open Interest
    plt.subplot(2, 1, 1)
    plt.plot(prices, label="Price")
    plt.plot(open_interest, label="Open Interest")
    plt.title("Price and Open Interest")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: OI to Price Ratio
    plt.subplot(2, 1, 2)
    plt.plot(result["metrics"]["oi_price_ratio"], label="OI/Price Ratio")
    plt.title("OI to Price Ratio")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_open_interest_analyzer()
