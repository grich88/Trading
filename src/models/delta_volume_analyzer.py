"""
Delta (Aggressor) Volume Imbalance Analysis module.

This module provides advanced analysis of Delta Volume (Aggressor Volume) patterns,
including imbalance detection, trend analysis, and combined signal generation.
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
    DELTA_LOOKBACK_PERIOD,
    DELTA_IMBALANCE_THRESHOLD,
    DELTA_SIGNAL_THRESHOLD,
    DELTA_TREND_PERIOD
)

logger = get_logger("DeltaVolumeAnalyzer")


class DeltaVolumeAnalyzer:
    """
    Advanced Delta (Aggressor) Volume analyzer for market data.
    
    This class provides methods for:
    - Delta volume calculation and analysis
    - Volume imbalance detection
    - Delta trend analysis
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = DELTA_LOOKBACK_PERIOD,
                 imbalance_threshold: float = DELTA_IMBALANCE_THRESHOLD,
                 signal_threshold: float = DELTA_SIGNAL_THRESHOLD,
                 trend_period: int = DELTA_TREND_PERIOD):
        """
        Initialize the Delta Volume analyzer.
        
        Args:
            lookback_period: Lookback period for imbalance detection
            imbalance_threshold: Threshold for imbalance detection
            signal_threshold: Threshold for signal generation
            trend_period: Period for trend analysis
        """
        self.lookback_period = lookback_period
        self.imbalance_threshold = imbalance_threshold
        self.signal_threshold = signal_threshold
        self.trend_period = trend_period
        
        logger.info(f"DeltaVolumeAnalyzer initialized with lookback period: {lookback_period}, "
                   f"imbalance threshold: {imbalance_threshold}")
    
    @performance_monitor()
    def calculate_delta_volume(self, 
                             buy_market_volume: np.ndarray, 
                             sell_market_volume: np.ndarray) -> np.ndarray:
        """
        Calculate Delta Volume (Aggressor Volume).
        
        Args:
            buy_market_volume: Array of buy market order volume data
            sell_market_volume: Array of sell market order volume data
            
        Returns:
            Array of delta volume values
        """
        if len(buy_market_volume) != len(sell_market_volume):
            raise ModelError("Buy and sell market volume arrays must have the same length")
        
        if len(buy_market_volume) < 2:
            raise ModelError("Not enough data to calculate delta volume")
        
        # Calculate delta volume (buy market - sell market)
        delta_volume = buy_market_volume - sell_market_volume
        
        return delta_volume
    
    @performance_monitor()
    def calculate_cumulative_delta(self, delta_volume: np.ndarray) -> np.ndarray:
        """
        Calculate Cumulative Delta Volume.
        
        Args:
            delta_volume: Array of delta volume values
            
        Returns:
            Array of cumulative delta values
        """
        if len(delta_volume) < 2:
            raise ModelError("Not enough data to calculate cumulative delta")
        
        # Calculate cumulative sum
        cumulative_delta = np.cumsum(delta_volume)
        
        return cumulative_delta
    
    @performance_monitor()
    def detect_imbalance(self, 
                       delta_volume: np.ndarray,
                       lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect volume imbalance in delta volume.
        
        Args:
            delta_volume: Array of delta volume data
            lookback: Lookback period for imbalance detection
            
        Returns:
            Dictionary with imbalance information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(delta_volume) < lookback:
            raise ModelError(f"Not enough data to detect imbalance. Need at least {lookback} data points.")
        
        # Get the last lookback periods
        recent_delta = delta_volume[-lookback:]
        
        # Calculate total buy and sell market volume
        total_buy_market = np.sum(np.where(recent_delta > 0, recent_delta, 0))
        total_sell_market = np.sum(np.where(recent_delta < 0, -recent_delta, 0))
        
        # Calculate buy/sell ratio
        if total_sell_market == 0:
            buy_sell_ratio = float('inf')
        else:
            buy_sell_ratio = total_buy_market / total_sell_market
        
        # Calculate sell/buy ratio
        if total_buy_market == 0:
            sell_buy_ratio = float('inf')
        else:
            sell_buy_ratio = total_sell_market / total_buy_market
        
        # Determine imbalance direction
        if buy_sell_ratio > 1 + self.imbalance_threshold:
            imbalance_direction = "buy"
            imbalance_ratio = buy_sell_ratio
        elif sell_buy_ratio > 1 + self.imbalance_threshold:
            imbalance_direction = "sell"
            imbalance_ratio = sell_buy_ratio
        else:
            imbalance_direction = "neutral"
            imbalance_ratio = 1.0
        
        # Calculate net delta volume
        net_delta = np.sum(recent_delta)
        
        # Calculate delta volume standard deviation
        delta_std = np.std(recent_delta)
        
        # Calculate delta volume Z-score (current delta vs. historical)
        if delta_std == 0:
            current_z_score = 0
        else:
            current_z_score = (recent_delta[-1] - np.mean(recent_delta)) / delta_std
        
        # Calculate imbalance strength
        if imbalance_direction == "buy":
            imbalance_strength = min(1.0, (buy_sell_ratio - 1) / 10)
        elif imbalance_direction == "sell":
            imbalance_strength = -min(1.0, (sell_buy_ratio - 1) / 10)
        else:
            imbalance_strength = 0.0
        
        return {
            "imbalance_direction": imbalance_direction,
            "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio != float('inf') else None,
            "total_buy_market": float(total_buy_market),
            "total_sell_market": float(total_sell_market),
            "net_delta": float(net_delta),
            "delta_std": float(delta_std),
            "current_z_score": float(current_z_score),
            "imbalance_strength": float(imbalance_strength)
        }
    
    @performance_monitor()
    def analyze_delta_trend(self, 
                          delta_volume: np.ndarray, 
                          period: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze Delta Volume trend.
        
        Args:
            delta_volume: Array of delta volume data
            period: Period for trend analysis
            
        Returns:
            Dictionary with delta trend analysis
        """
        if period is None:
            period = self.trend_period
        
        if len(delta_volume) < period + 1:
            raise ModelError(f"Not enough data to analyze delta trend. Need at least {period + 1} data points.")
        
        # Calculate cumulative delta
        cumulative_delta = self.calculate_cumulative_delta(delta_volume)
        
        # Calculate short-term trend (last 'period' periods)
        short_term_delta = cumulative_delta[-period:]
        short_term_trend = np.polyfit(np.arange(period), short_term_delta, 1)[0]
        
        # Calculate medium-term trend (last 'period*2' periods)
        medium_term_periods = min(period * 2, len(cumulative_delta))
        medium_term_delta = cumulative_delta[-medium_term_periods:]
        medium_term_trend = np.polyfit(np.arange(medium_term_periods), medium_term_delta, 1)[0]
        
        # Calculate long-term trend (last 'period*4' periods)
        long_term_periods = min(period * 4, len(cumulative_delta))
        long_term_delta = cumulative_delta[-long_term_periods:]
        long_term_trend = np.polyfit(np.arange(long_term_periods), long_term_delta, 1)[0]
        
        # Normalize trends
        short_term_trend_norm = short_term_trend / np.mean(np.abs(short_term_delta)) * 100 if np.mean(np.abs(short_term_delta)) > 0 else 0
        medium_term_trend_norm = medium_term_trend / np.mean(np.abs(medium_term_delta)) * 100 if np.mean(np.abs(medium_term_delta)) > 0 else 0
        long_term_trend_norm = long_term_trend / np.mean(np.abs(long_term_delta)) * 100 if np.mean(np.abs(long_term_delta)) > 0 else 0
        
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
    def detect_divergence(self, 
                        delta_volume: np.ndarray, 
                        prices: np.ndarray,
                        lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect divergence between delta volume and price.
        
        Args:
            delta_volume: Array of delta volume data
            prices: Array of price data
            lookback: Lookback period for divergence detection
            
        Returns:
            Dictionary with divergence information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(delta_volume) < lookback or len(prices) < lookback:
            raise ModelError(f"Not enough data to detect divergence. Need at least {lookback} data points.")
        
        if len(delta_volume) != len(prices):
            raise ModelError("Delta volume and price arrays must have the same length")
        
        # Get the last lookback periods
        recent_delta = delta_volume[-lookback:]
        recent_prices = prices[-lookback:]
        
        # Calculate cumulative delta
        cumulative_delta = np.cumsum(recent_delta)
        
        # Calculate delta and price trends
        delta_trend = np.polyfit(np.arange(lookback), cumulative_delta, 1)[0]
        price_trend = np.polyfit(np.arange(lookback), recent_prices, 1)[0]
        
        # Normalize trends
        delta_trend_norm = delta_trend / np.mean(np.abs(cumulative_delta)) * 100 if np.mean(np.abs(cumulative_delta)) > 0 else 0
        price_trend_norm = price_trend / np.mean(np.abs(recent_prices)) * 100
        
        # Calculate trend divergence
        trend_divergence = delta_trend_norm - price_trend_norm
        
        # Detect bullish divergence (price down, delta up)
        bullish_divergence = (price_trend_norm < -self.imbalance_threshold and 
                              delta_trend_norm > self.imbalance_threshold)
        
        # Detect bearish divergence (price up, delta down)
        bearish_divergence = (price_trend_norm > self.imbalance_threshold and 
                              delta_trend_norm < -self.imbalance_threshold)
        
        # Detect extreme divergence
        extreme_divergence = abs(trend_divergence) > self.imbalance_threshold * 2
        
        # Calculate correlation
        correlation = np.corrcoef(cumulative_delta, recent_prices)[0, 1]
        
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
            "delta_trend": float(delta_trend_norm),
            "price_trend": float(price_trend_norm),
            "trend_divergence": float(trend_divergence),
            "correlation": float(correlation),
            "divergence_strength": float(divergence_strength)
        }
    
    @performance_monitor()
    def analyze_vwap_delta(self, 
                         delta_volume: np.ndarray, 
                         prices: np.ndarray,
                         volumes: np.ndarray) -> Dict[str, Any]:
        """
        Analyze VWAP (Volume-Weighted Average Price) delta.
        
        Args:
            delta_volume: Array of delta volume data
            prices: Array of price data
            volumes: Array of total volume data
            
        Returns:
            Dictionary with VWAP delta analysis
        """
        if len(delta_volume) != len(prices) or len(prices) != len(volumes):
            raise ModelError("Delta volume, price, and volume arrays must have the same length")
        
        if len(delta_volume) < 5:
            raise ModelError("Not enough data to analyze VWAP delta")
        
        # Calculate VWAP
        vwap = np.sum(prices * volumes) / np.sum(volumes)
        
        # Calculate buy VWAP and sell VWAP
        buy_volume = np.where(delta_volume > 0, delta_volume, 0)
        sell_volume = np.where(delta_volume < 0, -delta_volume, 0)
        
        # Avoid division by zero
        if np.sum(buy_volume) == 0:
            buy_vwap = vwap
        else:
            buy_vwap = np.sum(prices * buy_volume) / np.sum(buy_volume)
        
        if np.sum(sell_volume) == 0:
            sell_vwap = vwap
        else:
            sell_vwap = np.sum(prices * sell_volume) / np.sum(sell_volume)
        
        # Calculate VWAP delta
        vwap_delta = buy_vwap - sell_vwap
        
        # Calculate normalized VWAP delta
        normalized_vwap_delta = vwap_delta / vwap * 100
        
        # Determine VWAP delta strength
        if normalized_vwap_delta > self.imbalance_threshold:
            vwap_delta_direction = "buy"
            vwap_delta_strength = min(1.0, normalized_vwap_delta / 10)
        elif normalized_vwap_delta < -self.imbalance_threshold:
            vwap_delta_direction = "sell"
            vwap_delta_strength = -min(1.0, -normalized_vwap_delta / 10)
        else:
            vwap_delta_direction = "neutral"
            vwap_delta_strength = 0.0
        
        return {
            "vwap": float(vwap),
            "buy_vwap": float(buy_vwap),
            "sell_vwap": float(sell_vwap),
            "vwap_delta": float(vwap_delta),
            "normalized_vwap_delta": float(normalized_vwap_delta),
            "vwap_delta_direction": vwap_delta_direction,
            "vwap_delta_strength": float(vwap_delta_strength)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      imbalance_analysis: Dict[str, Any],
                      trend_analysis: Dict[str, Any],
                      divergence_analysis: Dict[str, Any],
                      vwap_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined Delta Volume signal.
        
        Args:
            imbalance_analysis: Imbalance analysis results
            trend_analysis: Delta trend analysis results
            divergence_analysis: Divergence analysis results
            vwap_analysis: VWAP delta analysis results
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Imbalance Component (range: -0.3 to 0.3)
        imbalance_score = imbalance_analysis["imbalance_strength"] * 0.3
        
        # Adjust for Z-score
        z_score = imbalance_analysis["current_z_score"]
        if abs(z_score) > 2:
            # Strengthen the signal for significant Z-scores
            imbalance_score *= (1 + min(1.0, (abs(z_score) - 2) / 3))
        
        # Cap score
        imbalance_score = max(-0.3, min(0.3, imbalance_score))
        signal_components["imbalance"] = imbalance_score
        
        # 2. Trend Component (range: -0.25 to 0.25)
        trend_score = trend_analysis["trend_score"] / 100.0 * 0.25
        
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
        trend_score = max(-0.25, min(0.25, trend_score))
        signal_components["trend"] = trend_score
        
        # 3. Divergence Component (range: -0.25 to 0.25)
        divergence_score = 0.0
        if divergence_analysis["bullish_divergence"]:
            divergence_score = 0.2
        elif divergence_analysis["bearish_divergence"]:
            divergence_score = -0.2
        
        # Adjust by strength
        divergence_score *= abs(divergence_analysis["divergence_strength"]) * 2
        
        # Adjust for extreme divergence
        if divergence_analysis["extreme_divergence"]:
            divergence_score *= 1.25
        
        # Adjust for correlation
        correlation = divergence_analysis["correlation"]
        if correlation < 0:
            # Negative correlation strengthens the divergence signal
            divergence_score *= (1.0 - correlation)
        
        # Cap score
        divergence_score = max(-0.25, min(0.25, divergence_score))
        signal_components["divergence"] = divergence_score
        
        # 4. VWAP Component (range: -0.2 to 0.2)
        vwap_score = vwap_analysis["vwap_delta_strength"] * 0.2
        
        # Cap score
        vwap_score = max(-0.2, min(0.2, vwap_score))
        signal_components["vwap"] = vwap_score
        
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
        Perform complete Delta Volume analysis on a DataFrame.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["close", "volume", "buy_market_volume", "sell_market_volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        prices = df["close"].values
        volumes = df["volume"].values
        buy_market_volume = df["buy_market_volume"].values
        sell_market_volume = df["sell_market_volume"].values
        
        # Calculate delta volume
        delta_volume = self.calculate_delta_volume(buy_market_volume, sell_market_volume)
        
        # Calculate cumulative delta
        cumulative_delta = self.calculate_cumulative_delta(delta_volume)
        
        # Perform analyses
        imbalance_analysis = self.detect_imbalance(delta_volume)
        trend_analysis = self.analyze_delta_trend(delta_volume)
        divergence_analysis = self.detect_divergence(delta_volume, prices)
        vwap_analysis = self.analyze_vwap_delta(delta_volume, prices, volumes)
        
        # Generate signal
        signal = self.generate_signal(
            imbalance_analysis,
            trend_analysis,
            divergence_analysis,
            vwap_analysis
        )
        
        # Combine all results
        result = {
            "metrics": {
                "delta_volume": delta_volume.tolist(),
                "cumulative_delta": cumulative_delta.tolist()
            },
            "imbalance_analysis": imbalance_analysis,
            "trend_analysis": trend_analysis,
            "divergence_analysis": divergence_analysis,
            "vwap_analysis": vwap_analysis,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def calculate_delta_volume(buy_market_volume: np.ndarray, sell_market_volume: np.ndarray) -> np.ndarray:
    """
    Standalone function to calculate delta volume.
    
    Args:
        buy_market_volume: Array of buy market order volume data
        sell_market_volume: Array of sell market order volume data
        
    Returns:
        Array of delta volume values
    """
    analyzer = DeltaVolumeAnalyzer()
    return analyzer.calculate_delta_volume(buy_market_volume, sell_market_volume)


def analyze_delta_volume(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze delta volume.
    
    Args:
        df: DataFrame with market data
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = DeltaVolumeAnalyzer()
    return analyzer.analyze(df)


# Test function
def test_delta_volume_analyzer():
    """Test the Delta Volume analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    periods = 100
    np.random.seed(42)
    
    # Generate price data with trend and noise
    prices = np.cumsum(np.random.normal(0, 1, periods)) + 100
    
    # Generate volume data
    volumes = np.random.lognormal(0, 1, periods) * 1000
    
    # Generate buy and sell market volume data
    # First half: balanced
    # Second half: more buy market volume
    buy_market_base = np.random.normal(100, 20, periods)
    sell_market_base = np.random.normal(100, 20, periods)
    
    # Add trend
    buy_market_trend = np.concatenate([np.zeros(periods//2), np.linspace(0, 50, periods - periods//2)])
    sell_market_trend = np.zeros(periods)
    
    buy_market_volume = buy_market_base + buy_market_trend
    sell_market_volume = sell_market_base + sell_market_trend
    
    # Create DataFrame
    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "buy_market_volume": buy_market_volume,
        "sell_market_volume": sell_market_volume
    })
    
    # Create analyzer
    analyzer = DeltaVolumeAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(df)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nImbalance Analysis:")
    print(f"  Imbalance Direction: {result['imbalance_analysis']['imbalance_direction']}")
    print(f"  Imbalance Ratio: {result['imbalance_analysis']['imbalance_ratio']:.3f}")
    print(f"  Imbalance Strength: {result['imbalance_analysis']['imbalance_strength']:.3f}")
    
    print("\nDivergence Analysis:")
    print(f"  Bullish Divergence: {result['divergence_analysis']['bullish_divergence']}")
    print(f"  Bearish Divergence: {result['divergence_analysis']['bearish_divergence']}")
    print(f"  Extreme Divergence: {result['divergence_analysis']['extreme_divergence']}")
    print(f"  Delta Trend: {result['divergence_analysis']['delta_trend']:.3f}")
    print(f"  Price Trend: {result['divergence_analysis']['price_trend']:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price and Cumulative Delta
    plt.subplot(2, 1, 1)
    plt.plot(prices, label="Price")
    plt.title("Price")
    plt.grid(True)
    
    # Plot 2: Delta Volume and Cumulative Delta
    plt.subplot(2, 1, 2)
    plt.plot(result["metrics"]["delta_volume"], label="Delta Volume", alpha=0.5)
    plt.plot(result["metrics"]["cumulative_delta"], label="Cumulative Delta")
    plt.title("Delta Volume and Cumulative Delta")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_delta_volume_analyzer()
