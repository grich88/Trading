"""
Liquidation Map Analysis module.

This module provides advanced analysis of liquidation data,
including liquidation clustering, cascade detection, and combined signal generation.
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
    LIQUIDATION_LOOKBACK_PERIOD,
    LIQUIDATION_CLUSTER_THRESHOLD,
    LIQUIDATION_SIGNAL_THRESHOLD,
    LIQUIDATION_CASCADE_THRESHOLD
)

logger = get_logger("LiquidationMapAnalyzer")


class LiquidationMapAnalyzer:
    """
    Advanced Liquidation Map analyzer for market data.
    
    This class provides methods for:
    - Liquidation data processing and normalization
    - Liquidation cluster detection
    - Liquidation cascade detection
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = LIQUIDATION_LOOKBACK_PERIOD,
                 cluster_threshold: float = LIQUIDATION_CLUSTER_THRESHOLD,
                 signal_threshold: float = LIQUIDATION_SIGNAL_THRESHOLD,
                 cascade_threshold: float = LIQUIDATION_CASCADE_THRESHOLD):
        """
        Initialize the Liquidation Map analyzer.
        
        Args:
            lookback_period: Lookback period for analysis
            cluster_threshold: Threshold for cluster detection
            signal_threshold: Threshold for signal generation
            cascade_threshold: Threshold for cascade detection
        """
        self.lookback_period = lookback_period
        self.cluster_threshold = cluster_threshold
        self.signal_threshold = signal_threshold
        self.cascade_threshold = cascade_threshold
        
        logger.info(f"LiquidationMapAnalyzer initialized with lookback period: {lookback_period}, "
                   f"cluster threshold: {cluster_threshold}")
    
    @performance_monitor()
    def normalize_liquidations(self, 
                             liquidations: np.ndarray, 
                             prices: np.ndarray,
                             method: str = 'z-score') -> np.ndarray:
        """
        Normalize liquidation data for analysis.
        
        Args:
            liquidations: Array of liquidation amounts
            prices: Array of price data
            method: Normalization method ('z-score', 'percent-of-volume', or 'price-relative')
            
        Returns:
            Array of normalized liquidation values
        """
        if len(liquidations) != len(prices):
            raise ModelError("Liquidation and price arrays must have the same length")
        
        if len(liquidations) < 2:
            raise ModelError("Not enough data to normalize liquidations")
        
        if method == 'z-score':
            # Z-score normalization
            mean = np.mean(liquidations)
            std = np.std(liquidations)
            if std == 0:
                return np.zeros_like(liquidations)
            return (liquidations - mean) / std
        
        elif method == 'percent-of-volume':
            # Normalize by total volume
            total_volume = np.sum(liquidations)
            if total_volume == 0:
                return np.zeros_like(liquidations)
            return liquidations / total_volume * 100
        
        elif method == 'price-relative':
            # Normalize relative to price
            return liquidations / prices
        
        else:
            raise ModelError(f"Unknown normalization method: {method}")
    
    @performance_monitor()
    def detect_liquidation_clusters(self, 
                                  liquidations: np.ndarray,
                                  prices: np.ndarray,
                                  price_ranges: np.ndarray) -> Dict[str, Any]:
        """
        Detect liquidation clusters at specific price levels.
        
        Args:
            liquidations: Array of liquidation amounts
            prices: Array of current price data
            price_ranges: Array of price ranges for liquidation data
            
        Returns:
            Dictionary with cluster information
        """
        if len(liquidations) != len(price_ranges):
            raise ModelError("Liquidation and price range arrays must have the same length")
        
        if len(liquidations) < 5:
            raise ModelError("Not enough data to detect liquidation clusters")
        
        # Find the current price
        current_price = prices[-1]
        
        # Calculate distance from current price to each price range
        price_distances = np.abs(price_ranges - current_price) / current_price * 100
        
        # Identify clusters (areas with high liquidation concentration)
        clusters = []
        for i in range(len(liquidations)):
            if liquidations[i] > self.cluster_threshold:
                clusters.append({
                    "price_level": float(price_ranges[i]),
                    "distance_percent": float(price_distances[i]),
                    "liquidation_amount": float(liquidations[i]),
                    "is_support": price_ranges[i] < current_price,
                    "is_resistance": price_ranges[i] > current_price,
                    "strength": float(liquidations[i] / self.cluster_threshold)
                })
        
        # Sort clusters by liquidation amount
        clusters.sort(key=lambda x: x["liquidation_amount"], reverse=True)
        
        # Identify nearest support and resistance clusters
        support_clusters = [c for c in clusters if c["is_support"]]
        resistance_clusters = [c for c in clusters if c["is_resistance"]]
        
        nearest_support = min(support_clusters, key=lambda x: x["distance_percent"]) if support_clusters else None
        nearest_resistance = min(resistance_clusters, key=lambda x: x["distance_percent"]) if resistance_clusters else None
        
        # Calculate total liquidations above and below current price
        total_liquidations = np.sum(liquidations)
        liquidations_below = np.sum(liquidations[price_ranges < current_price])
        liquidations_above = np.sum(liquidations[price_ranges > current_price])
        
        # Calculate imbalance ratio
        if liquidations_below == 0:
            above_below_ratio = float('inf')
        else:
            above_below_ratio = liquidations_above / liquidations_below
        
        if liquidations_above == 0:
            below_above_ratio = float('inf')
        else:
            below_above_ratio = liquidations_below / liquidations_above
        
        # Determine imbalance direction
        if above_below_ratio > 1 + self.cluster_threshold:
            imbalance_direction = "resistance"
            imbalance_ratio = above_below_ratio
        elif below_above_ratio > 1 + self.cluster_threshold:
            imbalance_direction = "support"
            imbalance_ratio = below_above_ratio
        else:
            imbalance_direction = "neutral"
            imbalance_ratio = 1.0
        
        return {
            "clusters": clusters,
            "total_clusters": len(clusters),
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "total_liquidations": float(total_liquidations),
            "liquidations_below": float(liquidations_below),
            "liquidations_above": float(liquidations_above),
            "above_below_ratio": float(above_below_ratio) if above_below_ratio != float('inf') else None,
            "below_above_ratio": float(below_above_ratio) if below_above_ratio != float('inf') else None,
            "imbalance_direction": imbalance_direction,
            "imbalance_ratio": float(imbalance_ratio) if imbalance_ratio != float('inf') else None
        }
    
    @performance_monitor()
    def detect_liquidation_cascade(self, 
                                 liquidation_history: np.ndarray,
                                 price_history: np.ndarray,
                                 lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect potential liquidation cascades based on historical data.
        
        Args:
            liquidation_history: Array of historical liquidation amounts
            price_history: Array of historical price data
            lookback: Lookback period for cascade detection
            
        Returns:
            Dictionary with cascade information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(liquidation_history) < lookback or len(price_history) < lookback:
            raise ModelError(f"Not enough data to detect liquidation cascade. Need at least {lookback} data points.")
        
        if len(liquidation_history) != len(price_history):
            raise ModelError("Liquidation and price arrays must have the same length")
        
        # Get the last lookback periods
        recent_liquidations = liquidation_history[-lookback:]
        recent_prices = price_history[-lookback:]
        
        # Calculate liquidation momentum (rate of change)
        liquidation_momentum = np.diff(recent_liquidations)
        price_momentum = np.diff(recent_prices)
        
        # Calculate correlation between liquidation and price changes
        correlation = np.corrcoef(liquidation_momentum, price_momentum)[0, 1]
        
        # Calculate liquidation acceleration (second derivative)
        liquidation_acceleration = np.diff(liquidation_momentum)
        
        # Detect cascade conditions
        # 1. High liquidation volume
        high_liquidation = np.max(recent_liquidations) > self.cascade_threshold * np.mean(recent_liquidations)
        
        # 2. Increasing liquidation momentum
        increasing_momentum = np.sum(liquidation_momentum > 0) > lookback * 0.6
        
        # 3. Positive liquidation acceleration
        positive_acceleration = np.sum(liquidation_acceleration > 0) > len(liquidation_acceleration) * 0.6
        
        # 4. Negative correlation with price (more liquidations as price moves against positions)
        negative_correlation = correlation < -0.3
        
        # Determine cascade probability
        cascade_score = 0.0
        if high_liquidation:
            cascade_score += 0.3
        if increasing_momentum:
            cascade_score += 0.3
        if positive_acceleration:
            cascade_score += 0.2
        if negative_correlation:
            cascade_score += 0.2
        
        # Determine cascade direction
        if cascade_score > self.cascade_threshold:
            if np.mean(price_momentum) < 0:
                cascade_direction = "down"
            else:
                cascade_direction = "up"
        else:
            cascade_direction = "neutral"
        
        # Calculate liquidation volatility
        liquidation_volatility = np.std(recent_liquidations) / np.mean(recent_liquidations) if np.mean(recent_liquidations) > 0 else 0
        
        return {
            "cascade_score": float(cascade_score),
            "cascade_direction": cascade_direction,
            "high_liquidation": high_liquidation,
            "increasing_momentum": increasing_momentum,
            "positive_acceleration": positive_acceleration,
            "negative_correlation": negative_correlation,
            "correlation": float(correlation),
            "liquidation_volatility": float(liquidation_volatility),
            "max_liquidation": float(np.max(recent_liquidations)),
            "avg_liquidation": float(np.mean(recent_liquidations))
        }
    
    @performance_monitor()
    def analyze_liquidation_impact(self, 
                                 liquidation_history: np.ndarray,
                                 price_history: np.ndarray,
                                 volume_history: np.ndarray,
                                 lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze the impact of liquidations on price and volume.
        
        Args:
            liquidation_history: Array of historical liquidation amounts
            price_history: Array of historical price data
            volume_history: Array of historical volume data
            lookback: Lookback period for impact analysis
            
        Returns:
            Dictionary with impact analysis
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(liquidation_history) < lookback or len(price_history) < lookback or len(volume_history) < lookback:
            raise ModelError(f"Not enough data to analyze liquidation impact. Need at least {lookback} data points.")
        
        if len(liquidation_history) != len(price_history) or len(price_history) != len(volume_history):
            raise ModelError("Liquidation, price, and volume arrays must have the same length")
        
        # Get the last lookback periods
        recent_liquidations = liquidation_history[-lookback:]
        recent_prices = price_history[-lookback:]
        recent_volumes = volume_history[-lookback:]
        
        # Calculate liquidation to volume ratio
        liquidation_volume_ratio = recent_liquidations / recent_volumes
        avg_liquidation_volume_ratio = np.mean(liquidation_volume_ratio)
        
        # Identify high-impact periods (where liquidations are a significant portion of volume)
        high_impact_periods = np.where(liquidation_volume_ratio > avg_liquidation_volume_ratio * 2)[0]
        
        # Calculate price changes following high-impact periods
        price_impacts = []
        for i in high_impact_periods:
            if i + 3 < len(recent_prices):
                price_change_1 = (recent_prices[i+1] - recent_prices[i]) / recent_prices[i] * 100
                price_change_3 = (recent_prices[i+3] - recent_prices[i]) / recent_prices[i] * 100
                price_impacts.append({
                    "period": int(i),
                    "liquidation_volume_ratio": float(liquidation_volume_ratio[i]),
                    "price_change_1": float(price_change_1),
                    "price_change_3": float(price_change_3)
                })
        
        # Calculate average price impact
        if price_impacts:
            avg_price_change_1 = np.mean([impact["price_change_1"] for impact in price_impacts])
            avg_price_change_3 = np.mean([impact["price_change_3"] for impact in price_impacts])
        else:
            avg_price_change_1 = 0.0
            avg_price_change_3 = 0.0
        
        # Determine if liquidations lead to mean reversion or trend continuation
        if abs(avg_price_change_1) > 0.1 and abs(avg_price_change_3) > 0.1:
            if (avg_price_change_1 < 0 and avg_price_change_3 > 0) or (avg_price_change_1 > 0 and avg_price_change_3 < 0):
                impact_pattern = "mean_reversion"
            elif (avg_price_change_1 < 0 and avg_price_change_3 < 0) or (avg_price_change_1 > 0 and avg_price_change_3 > 0):
                impact_pattern = "trend_continuation"
            else:
                impact_pattern = "neutral"
        else:
            impact_pattern = "neutral"
        
        # Calculate correlation between liquidations and subsequent price changes
        price_changes = np.diff(recent_prices) / recent_prices[:-1] * 100
        lagged_correlation = np.corrcoef(recent_liquidations[:-1], price_changes)[0, 1]
        
        return {
            "avg_liquidation_volume_ratio": float(avg_liquidation_volume_ratio),
            "high_impact_periods": len(high_impact_periods),
            "price_impacts": price_impacts,
            "avg_price_change_1": float(avg_price_change_1),
            "avg_price_change_3": float(avg_price_change_3),
            "impact_pattern": impact_pattern,
            "lagged_correlation": float(lagged_correlation)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      cluster_analysis: Dict[str, Any],
                      cascade_analysis: Dict[str, Any],
                      impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined Liquidation Map signal.
        
        Args:
            cluster_analysis: Liquidation cluster analysis results
            cascade_analysis: Liquidation cascade analysis results
            impact_analysis: Liquidation impact analysis results
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Cluster Component (range: -0.4 to 0.4)
        cluster_score = 0.0
        
        # Check imbalance direction
        if cluster_analysis["imbalance_direction"] == "support":
            cluster_score = 0.2
        elif cluster_analysis["imbalance_direction"] == "resistance":
            cluster_score = -0.2
        
        # Adjust by imbalance ratio
        if cluster_analysis["imbalance_ratio"] is not None:
            cluster_score *= min(2.0, cluster_analysis["imbalance_ratio"] / 2)
        
        # Adjust for nearest support/resistance strength
        if cluster_analysis["nearest_support"] and cluster_analysis["nearest_resistance"]:
            support_strength = cluster_analysis["nearest_support"]["strength"]
            resistance_strength = cluster_analysis["nearest_resistance"]["strength"]
            
            if support_strength > resistance_strength * 1.5:
                cluster_score += 0.1
            elif resistance_strength > support_strength * 1.5:
                cluster_score -= 0.1
        
        # Cap score
        cluster_score = max(-0.4, min(0.4, cluster_score))
        signal_components["cluster"] = cluster_score
        
        # 2. Cascade Component (range: -0.3 to 0.3)
        cascade_score = 0.0
        
        if cascade_analysis["cascade_score"] > self.cascade_threshold:
            if cascade_analysis["cascade_direction"] == "down":
                cascade_score = -0.3 * min(1.0, cascade_analysis["cascade_score"] / self.cascade_threshold)
            elif cascade_analysis["cascade_direction"] == "up":
                cascade_score = 0.3 * min(1.0, cascade_analysis["cascade_score"] / self.cascade_threshold)
        
        # Cap score
        cascade_score = max(-0.3, min(0.3, cascade_score))
        signal_components["cascade"] = cascade_score
        
        # 3. Impact Component (range: -0.3 to 0.3)
        impact_score = 0.0
        
        # Check impact pattern
        if impact_analysis["impact_pattern"] == "mean_reversion":
            # For mean reversion, recent price drops might lead to bounces
            if impact_analysis["avg_price_change_1"] < 0 and impact_analysis["avg_price_change_3"] > 0:
                impact_score = 0.2
            elif impact_analysis["avg_price_change_1"] > 0 and impact_analysis["avg_price_change_3"] < 0:
                impact_score = -0.2
        elif impact_analysis["impact_pattern"] == "trend_continuation":
            # For trend continuation, recent price movements continue
            if impact_analysis["avg_price_change_1"] < 0 and impact_analysis["avg_price_change_3"] < 0:
                impact_score = -0.2
            elif impact_analysis["avg_price_change_1"] > 0 and impact_analysis["avg_price_change_3"] > 0:
                impact_score = 0.2
        
        # Adjust by lagged correlation
        impact_score *= (1.0 + abs(impact_analysis["lagged_correlation"]))
        
        # Cap score
        impact_score = max(-0.3, min(0.3, impact_score))
        signal_components["impact"] = impact_score
        
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
        Perform complete Liquidation Map analysis on a DataFrame.
        
        Args:
            df: DataFrame with market data including liquidations
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["close", "volume", "liquidation_amount", "liquidation_price"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        prices = df["close"].values
        volumes = df["volume"].values
        liquidation_amounts = df["liquidation_amount"].values
        liquidation_prices = df["liquidation_price"].values
        
        # Normalize liquidations
        normalized_liquidations = self.normalize_liquidations(liquidation_amounts, prices)
        
        # Perform analyses
        cluster_analysis = self.detect_liquidation_clusters(liquidation_amounts, prices, liquidation_prices)
        cascade_analysis = self.detect_liquidation_cascade(liquidation_amounts, prices)
        impact_analysis = self.analyze_liquidation_impact(liquidation_amounts, prices, volumes)
        
        # Generate signal
        signal = self.generate_signal(
            cluster_analysis,
            cascade_analysis,
            impact_analysis
        )
        
        # Combine all results
        result = {
            "metrics": {
                "liquidation_amounts": liquidation_amounts.tolist(),
                "normalized_liquidations": normalized_liquidations.tolist()
            },
            "cluster_analysis": cluster_analysis,
            "cascade_analysis": cascade_analysis,
            "impact_analysis": impact_analysis,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def normalize_liquidations(liquidations: np.ndarray, prices: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """
    Standalone function to normalize liquidation data.
    
    Args:
        liquidations: Array of liquidation amounts
        prices: Array of price data
        method: Normalization method
        
    Returns:
        Array of normalized liquidation values
    """
    analyzer = LiquidationMapAnalyzer()
    return analyzer.normalize_liquidations(liquidations, prices, method)


def analyze_liquidation_map(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze liquidation map data.
    
    Args:
        df: DataFrame with market data including liquidations
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = LiquidationMapAnalyzer()
    return analyzer.analyze(df)


# Test function
def test_liquidation_map_analyzer():
    """Test the Liquidation Map analyzer."""
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
    
    # Generate liquidation data
    # Create clusters of liquidations at certain price levels
    liquidation_prices = np.linspace(prices.min() * 0.9, prices.max() * 1.1, periods)
    
    # Create liquidation amounts with clusters
    base_liquidations = np.random.lognormal(0, 1, periods) * 500
    
    # Add liquidation clusters
    for i in range(5):
        cluster_center = np.random.randint(0, periods)
        cluster_width = np.random.randint(3, 10)
        cluster_start = max(0, cluster_center - cluster_width // 2)
        cluster_end = min(periods, cluster_center + cluster_width // 2)
        base_liquidations[cluster_start:cluster_end] *= np.random.randint(3, 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        "close": prices,
        "volume": volumes,
        "liquidation_amount": base_liquidations,
        "liquidation_price": liquidation_prices
    })
    
    # Create analyzer
    analyzer = LiquidationMapAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(df)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nCluster Analysis:")
    print(f"  Total Clusters: {result['cluster_analysis']['total_clusters']}")
    print(f"  Imbalance Direction: {result['cluster_analysis']['imbalance_direction']}")
    if result['cluster_analysis']['imbalance_ratio'] is not None:
        print(f"  Imbalance Ratio: {result['cluster_analysis']['imbalance_ratio']:.3f}")
    
    print("\nCascade Analysis:")
    print(f"  Cascade Score: {result['cascade_analysis']['cascade_score']:.3f}")
    print(f"  Cascade Direction: {result['cascade_analysis']['cascade_direction']}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price and Liquidation Prices
    plt.subplot(2, 1, 1)
    plt.plot(prices, label="Price")
    plt.scatter(range(len(liquidation_prices)), liquidation_prices, 
                c=base_liquidations, cmap='hot', alpha=0.5, s=base_liquidations/50)
    plt.colorbar(label="Liquidation Amount")
    plt.title("Price and Liquidation Map")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Liquidation Amounts
    plt.subplot(2, 1, 2)
    plt.bar(range(len(base_liquidations)), base_liquidations, alpha=0.7)
    plt.title("Liquidation Amounts")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_liquidation_map_analyzer()
