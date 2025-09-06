"""
Funding Rate & Bias Tracking Analysis module.

This module provides advanced analysis of funding rates and market bias,
including rate anomaly detection, bias tracking, and combined signal generation.
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
    FUNDING_LOOKBACK_PERIOD,
    FUNDING_ANOMALY_THRESHOLD,
    FUNDING_SIGNAL_THRESHOLD,
    FUNDING_BIAS_THRESHOLD
)

logger = get_logger("FundingRateAnalyzer")


class FundingRateAnalyzer:
    """
    Advanced Funding Rate & Bias analyzer for market data.
    
    This class provides methods for:
    - Funding rate analysis and anomaly detection
    - Market bias tracking based on funding rates
    - Historical funding rate pattern recognition
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = FUNDING_LOOKBACK_PERIOD,
                 anomaly_threshold: float = FUNDING_ANOMALY_THRESHOLD,
                 signal_threshold: float = FUNDING_SIGNAL_THRESHOLD,
                 bias_threshold: float = FUNDING_BIAS_THRESHOLD):
        """
        Initialize the Funding Rate analyzer.
        
        Args:
            lookback_period: Lookback period for analysis
            anomaly_threshold: Threshold for anomaly detection
            signal_threshold: Threshold for signal generation
            bias_threshold: Threshold for bias detection
        """
        self.lookback_period = lookback_period
        self.anomaly_threshold = anomaly_threshold
        self.signal_threshold = signal_threshold
        self.bias_threshold = bias_threshold
        
        logger.info(f"FundingRateAnalyzer initialized with lookback period: {lookback_period}, "
                   f"anomaly threshold: {anomaly_threshold}")
    
    @performance_monitor()
    def detect_funding_anomalies(self, 
                              funding_rates: np.ndarray,
                              timestamps: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in funding rates.
        
        Args:
            funding_rates: Array of funding rate data
            timestamps: Array of timestamp data
            
        Returns:
            Dictionary with anomaly information
        """
        if len(funding_rates) < self.lookback_period:
            raise ModelError(f"Not enough data to detect funding anomalies. Need at least {self.lookback_period} data points.")
        
        if len(funding_rates) != len(timestamps):
            raise ModelError("Funding rates and timestamps arrays must have the same length")
        
        # Calculate statistics
        mean_rate = np.mean(funding_rates)
        std_rate = np.std(funding_rates)
        max_rate = np.max(funding_rates)
        min_rate = np.min(funding_rates)
        current_rate = funding_rates[-1]
        
        # Calculate Z-score of current rate
        if std_rate == 0:
            current_z_score = 0
        else:
            current_z_score = (current_rate - mean_rate) / std_rate
        
        # Detect anomalies
        is_anomaly = abs(current_z_score) > self.anomaly_threshold
        is_extreme_anomaly = abs(current_z_score) > self.anomaly_threshold * 2
        
        # Determine direction
        if current_rate > 0:
            direction = "positive"  # Longs pay shorts
        elif current_rate < 0:
            direction = "negative"  # Shorts pay longs
        else:
            direction = "neutral"
        
        # Find historical anomalies
        historical_anomalies = []
        for i in range(len(funding_rates)):
            if std_rate == 0:
                z_score = 0
            else:
                z_score = (funding_rates[i] - mean_rate) / std_rate
            
            if abs(z_score) > self.anomaly_threshold:
                historical_anomalies.append({
                    "timestamp": timestamps[i],
                    "funding_rate": float(funding_rates[i]),
                    "z_score": float(z_score),
                    "is_extreme": abs(z_score) > self.anomaly_threshold * 2
                })
        
        # Sort anomalies by absolute Z-score
        historical_anomalies.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        
        return {
            "mean_rate": float(mean_rate),
            "std_rate": float(std_rate),
            "max_rate": float(max_rate),
            "min_rate": float(min_rate),
            "current_rate": float(current_rate),
            "current_z_score": float(current_z_score),
            "is_anomaly": is_anomaly,
            "is_extreme_anomaly": is_extreme_anomaly,
            "direction": direction,
            "historical_anomalies": historical_anomalies[:10]  # Return top 10 anomalies
        }
    
    @performance_monitor()
    def analyze_funding_bias(self, 
                           funding_rates: np.ndarray,
                           lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze market bias based on funding rates.
        
        Args:
            funding_rates: Array of funding rate data
            lookback: Lookback period for bias analysis
            
        Returns:
            Dictionary with bias information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(funding_rates) < lookback:
            raise ModelError(f"Not enough data to analyze funding bias. Need at least {lookback} data points.")
        
        # Get the last lookback periods
        recent_rates = funding_rates[-lookback:]
        
        # Calculate bias metrics
        positive_count = np.sum(recent_rates > 0)
        negative_count = np.sum(recent_rates < 0)
        neutral_count = np.sum(recent_rates == 0)
        
        total_count = positive_count + negative_count + neutral_count
        
        positive_percentage = positive_count / total_count * 100
        negative_percentage = negative_count / total_count * 100
        
        # Calculate average rates
        positive_avg = np.mean(recent_rates[recent_rates > 0]) if positive_count > 0 else 0
        negative_avg = np.mean(recent_rates[recent_rates < 0]) if negative_count > 0 else 0
        
        # Calculate net bias
        net_bias = np.sum(recent_rates) / len(recent_rates)
        
        # Determine bias direction
        if positive_percentage > 50 + self.bias_threshold:
            bias_direction = "bullish"  # Consistent positive funding (longs paying shorts)
        elif negative_percentage > 50 + self.bias_threshold:
            bias_direction = "bearish"  # Consistent negative funding (shorts paying longs)
        else:
            bias_direction = "neutral"
        
        # Calculate bias strength
        bias_strength = abs(positive_percentage - negative_percentage) / 100
        
        return {
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "neutral_count": int(neutral_count),
            "positive_percentage": float(positive_percentage),
            "negative_percentage": float(negative_percentage),
            "positive_avg": float(positive_avg),
            "negative_avg": float(negative_avg),
            "net_bias": float(net_bias),
            "bias_direction": bias_direction,
            "bias_strength": float(bias_strength)
        }
    
    @performance_monitor()
    def analyze_funding_patterns(self, 
                               funding_rates: np.ndarray,
                               prices: np.ndarray,
                               lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze historical patterns in funding rates and their correlation with price.
        
        Args:
            funding_rates: Array of funding rate data
            prices: Array of price data
            lookback: Lookback period for pattern analysis
            
        Returns:
            Dictionary with pattern information
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(funding_rates) < lookback or len(prices) < lookback:
            raise ModelError(f"Not enough data to analyze funding patterns. Need at least {lookback} data points.")
        
        if len(funding_rates) != len(prices):
            raise ModelError("Funding rates and prices arrays must have the same length")
        
        # Get the last lookback periods
        recent_rates = funding_rates[-lookback:]
        recent_prices = prices[-lookback:]
        
        # Calculate correlation between funding rates and price
        correlation = np.corrcoef(recent_rates, recent_prices)[0, 1]
        
        # Calculate funding rate momentum (rate of change)
        funding_momentum = np.diff(recent_rates)
        avg_momentum = np.mean(funding_momentum)
        
        # Detect pattern types
        # 1. Funding rate sign changes (crossovers)
        sign_changes = np.sum(np.diff(np.signbit(recent_rates)))
        
        # 2. Funding rate peaks and troughs
        peaks = 0
        troughs = 0
        for i in range(1, len(recent_rates) - 1):
            if recent_rates[i] > recent_rates[i-1] and recent_rates[i] > recent_rates[i+1]:
                peaks += 1
            elif recent_rates[i] < recent_rates[i-1] and recent_rates[i] < recent_rates[i+1]:
                troughs += 1
        
        # 3. Detect convergence/divergence with price
        # Calculate normalized rates and prices for comparison
        norm_rates = (recent_rates - np.mean(recent_rates)) / np.std(recent_rates) if np.std(recent_rates) > 0 else recent_rates
        norm_prices = (recent_prices - np.mean(recent_prices)) / np.std(recent_prices) if np.std(recent_prices) > 0 else recent_prices
        
        # Calculate the difference between normalized rates and prices
        diff = norm_rates - norm_prices
        
        # Calculate the trend of the difference
        diff_trend = np.polyfit(np.arange(len(diff)), diff, 1)[0]
        
        # Determine if converging or diverging
        if diff_trend > 0.1:
            convergence_pattern = "diverging"
        elif diff_trend < -0.1:
            convergence_pattern = "converging"
        else:
            convergence_pattern = "neutral"
        
        # Identify common patterns
        patterns = []
        
        # Pattern 1: Sustained positive funding
        if np.sum(recent_rates > 0) > lookback * 0.8:
            patterns.append({
                "name": "sustained_positive_funding",
                "description": "Sustained positive funding rates (longs paying shorts)",
                "strength": float(np.mean(recent_rates[recent_rates > 0]) / np.std(recent_rates) if np.std(recent_rates) > 0 else 0)
            })
        
        # Pattern 2: Sustained negative funding
        if np.sum(recent_rates < 0) > lookback * 0.8:
            patterns.append({
                "name": "sustained_negative_funding",
                "description": "Sustained negative funding rates (shorts paying longs)",
                "strength": float(abs(np.mean(recent_rates[recent_rates < 0])) / np.std(recent_rates) if np.std(recent_rates) > 0 else 0)
            })
        
        # Pattern 3: Funding rate reversal
        if len(recent_rates) > 5:
            if np.all(recent_rates[-5:-1] > 0) and recent_rates[-1] < 0:
                patterns.append({
                    "name": "positive_to_negative_reversal",
                    "description": "Funding rate reversed from positive to negative",
                    "strength": float(abs(recent_rates[-1]) / np.std(recent_rates) if np.std(recent_rates) > 0 else 0)
                })
            elif np.all(recent_rates[-5:-1] < 0) and recent_rates[-1] > 0:
                patterns.append({
                    "name": "negative_to_positive_reversal",
                    "description": "Funding rate reversed from negative to positive",
                    "strength": float(abs(recent_rates[-1]) / np.std(recent_rates) if np.std(recent_rates) > 0 else 0)
                })
        
        # Pattern 4: Funding rate spike
        if abs(recent_rates[-1]) > np.mean(abs(recent_rates)) * 2:
            patterns.append({
                "name": "funding_spike",
                "description": f"Sudden spike in funding rate ({'positive' if recent_rates[-1] > 0 else 'negative'})",
                "strength": float(abs(recent_rates[-1]) / np.mean(abs(recent_rates)))
            })
        
        return {
            "correlation": float(correlation),
            "avg_momentum": float(avg_momentum),
            "sign_changes": int(sign_changes),
            "peaks": int(peaks),
            "troughs": int(troughs),
            "convergence_pattern": convergence_pattern,
            "patterns": patterns
        }
    
    @performance_monitor()
    def analyze_funding_impact(self, 
                             funding_rates: np.ndarray,
                             prices: np.ndarray,
                             lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze the impact of funding rates on subsequent price movements.
        
        Args:
            funding_rates: Array of funding rate data
            prices: Array of price data
            lookback: Lookback period for impact analysis
            
        Returns:
            Dictionary with impact analysis
        """
        if lookback is None:
            lookback = self.lookback_period
        
        if len(funding_rates) < lookback + 5 or len(prices) < lookback + 5:
            raise ModelError(f"Not enough data to analyze funding impact. Need at least {lookback + 5} data points.")
        
        if len(funding_rates) != len(prices):
            raise ModelError("Funding rates and prices arrays must have the same length")
        
        # Get data for analysis
        analysis_rates = funding_rates[-(lookback+5):-5]
        analysis_prices = prices[-(lookback+5):-5]
        
        # Calculate price changes following different funding conditions
        price_after_positive = []
        price_after_negative = []
        price_after_extreme_positive = []
        price_after_extreme_negative = []
        
        for i in range(len(analysis_rates)):
            if i + 5 >= len(analysis_prices):
                break
                
            rate = analysis_rates[i]
            price = analysis_prices[i]
            future_price = analysis_prices[i+5]
            price_change = (future_price - price) / price * 100
            
            if rate > 0:
                price_after_positive.append(price_change)
                if rate > np.mean(analysis_rates) + np.std(analysis_rates):
                    price_after_extreme_positive.append(price_change)
            elif rate < 0:
                price_after_negative.append(price_change)
                if rate < np.mean(analysis_rates) - np.std(analysis_rates):
                    price_after_extreme_negative.append(price_change)
        
        # Calculate average price changes
        avg_after_positive = np.mean(price_after_positive) if price_after_positive else 0
        avg_after_negative = np.mean(price_after_negative) if price_after_negative else 0
        avg_after_extreme_positive = np.mean(price_after_extreme_positive) if price_after_extreme_positive else 0
        avg_after_extreme_negative = np.mean(price_after_extreme_negative) if price_after_extreme_negative else 0
        
        # Determine if there's a significant impact pattern
        significant_impact = False
        impact_pattern = "neutral"
        
        if len(price_after_positive) > 5 and len(price_after_negative) > 5:
            if avg_after_positive < -0.5 and avg_after_negative > 0.5:
                significant_impact = True
                impact_pattern = "mean_reverting"  # High funding leads to price drops, low funding leads to price increases
            elif avg_after_positive > 0.5 and avg_after_negative < -0.5:
                significant_impact = True
                impact_pattern = "trend_following"  # High funding leads to further price increases, low funding leads to further drops
        
        # Calculate correlation between funding and future returns
        future_returns = np.zeros(len(analysis_rates))
        for i in range(len(analysis_rates)):
            if i + 5 < len(analysis_prices):
                future_returns[i] = (analysis_prices[i+5] - analysis_prices[i]) / analysis_prices[i] * 100
        
        predictive_correlation = np.corrcoef(analysis_rates, future_returns)[0, 1]
        
        return {
            "avg_after_positive": float(avg_after_positive),
            "avg_after_negative": float(avg_after_negative),
            "avg_after_extreme_positive": float(avg_after_extreme_positive),
            "avg_after_extreme_negative": float(avg_after_extreme_negative),
            "significant_impact": significant_impact,
            "impact_pattern": impact_pattern,
            "predictive_correlation": float(predictive_correlation),
            "positive_samples": len(price_after_positive),
            "negative_samples": len(price_after_negative),
            "extreme_positive_samples": len(price_after_extreme_positive),
            "extreme_negative_samples": len(price_after_extreme_negative)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      anomaly_analysis: Dict[str, Any],
                      bias_analysis: Dict[str, Any],
                      pattern_analysis: Dict[str, Any],
                      impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined Funding Rate & Bias signal.
        
        Args:
            anomaly_analysis: Funding anomaly analysis results
            bias_analysis: Funding bias analysis results
            pattern_analysis: Funding pattern analysis results
            impact_analysis: Funding impact analysis results
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Anomaly Component (range: -0.3 to 0.3)
        anomaly_score = 0.0
        
        # Check for anomalies
        if anomaly_analysis["is_anomaly"]:
            if anomaly_analysis["direction"] == "positive":
                anomaly_score = -0.2  # Positive funding anomaly is bearish (longs paying too much)
            elif anomaly_analysis["direction"] == "negative":
                anomaly_score = 0.2  # Negative funding anomaly is bullish (shorts paying too much)
            
            # Adjust for extreme anomalies
            if anomaly_analysis["is_extreme_anomaly"]:
                anomaly_score *= 1.5
        
        # Cap score
        anomaly_score = max(-0.3, min(0.3, anomaly_score))
        signal_components["anomaly"] = anomaly_score
        
        # 2. Bias Component (range: -0.3 to 0.3)
        bias_score = 0.0
        
        # Check bias direction
        if bias_analysis["bias_direction"] == "bullish":
            bias_score = -0.2  # Consistent positive funding is bearish (market is too long)
        elif bias_analysis["bias_direction"] == "bearish":
            bias_score = 0.2  # Consistent negative funding is bullish (market is too short)
        
        # Adjust by bias strength
        bias_score *= bias_analysis["bias_strength"] * 1.5
        
        # Cap score
        bias_score = max(-0.3, min(0.3, bias_score))
        signal_components["bias"] = bias_score
        
        # 3. Pattern Component (range: -0.2 to 0.2)
        pattern_score = 0.0
        
        # Check for specific patterns
        for pattern in pattern_analysis["patterns"]:
            if pattern["name"] == "sustained_positive_funding":
                pattern_score -= 0.1 * min(1.0, pattern["strength"])
            elif pattern["name"] == "sustained_negative_funding":
                pattern_score += 0.1 * min(1.0, pattern["strength"])
            elif pattern["name"] == "positive_to_negative_reversal":
                pattern_score += 0.15 * min(1.0, pattern["strength"])
            elif pattern["name"] == "negative_to_positive_reversal":
                pattern_score -= 0.15 * min(1.0, pattern["strength"])
            elif pattern["name"] == "funding_spike":
                if "positive" in pattern["description"]:
                    pattern_score -= 0.1 * min(1.0, pattern["strength"])
                else:
                    pattern_score += 0.1 * min(1.0, pattern["strength"])
        
        # Adjust for correlation with price
        if abs(pattern_analysis["correlation"]) > 0.5:
            pattern_score *= (1.0 - abs(pattern_analysis["correlation"]) * 0.5)
        
        # Cap score
        pattern_score = max(-0.2, min(0.2, pattern_score))
        signal_components["pattern"] = pattern_score
        
        # 4. Impact Component (range: -0.2 to 0.2)
        impact_score = 0.0
        
        # Check impact pattern
        if impact_analysis["significant_impact"]:
            if impact_analysis["impact_pattern"] == "mean_reverting":
                # In mean-reverting pattern, positive funding leads to price drops
                if anomaly_analysis["direction"] == "positive":
                    impact_score -= 0.15
                elif anomaly_analysis["direction"] == "negative":
                    impact_score += 0.15
            elif impact_analysis["impact_pattern"] == "trend_following":
                # In trend-following pattern, positive funding leads to further price increases
                if anomaly_analysis["direction"] == "positive":
                    impact_score += 0.15
                elif anomaly_analysis["direction"] == "negative":
                    impact_score -= 0.15
        
        # Adjust by predictive correlation
        if abs(impact_analysis["predictive_correlation"]) > 0.3:
            impact_score *= (1.0 + abs(impact_analysis["predictive_correlation"]))
        
        # Cap score
        impact_score = max(-0.2, min(0.2, impact_score))
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
        Perform complete Funding Rate & Bias analysis on a DataFrame.
        
        Args:
            df: DataFrame with market data including funding rates
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["close", "funding_rate", "timestamp"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        prices = df["close"].values
        funding_rates = df["funding_rate"].values
        timestamps = df["timestamp"].values
        
        # Perform analyses
        anomaly_analysis = self.detect_funding_anomalies(funding_rates, timestamps)
        bias_analysis = self.analyze_funding_bias(funding_rates)
        pattern_analysis = self.analyze_funding_patterns(funding_rates, prices)
        impact_analysis = self.analyze_funding_impact(funding_rates, prices)
        
        # Generate signal
        signal = self.generate_signal(
            anomaly_analysis,
            bias_analysis,
            pattern_analysis,
            impact_analysis
        )
        
        # Combine all results
        result = {
            "metrics": {
                "funding_rates": funding_rates.tolist(),
                "current_rate": float(funding_rates[-1]),
                "mean_rate": float(np.mean(funding_rates)),
                "std_rate": float(np.std(funding_rates))
            },
            "anomaly_analysis": anomaly_analysis,
            "bias_analysis": bias_analysis,
            "pattern_analysis": pattern_analysis,
            "impact_analysis": impact_analysis,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def detect_funding_anomalies(funding_rates: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
    """
    Standalone function to detect funding rate anomalies.
    
    Args:
        funding_rates: Array of funding rate data
        timestamps: Array of timestamp data
        
    Returns:
        Dictionary with anomaly information
    """
    analyzer = FundingRateAnalyzer()
    return analyzer.detect_funding_anomalies(funding_rates, timestamps)


def analyze_funding_rates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze funding rates.
    
    Args:
        df: DataFrame with market data including funding rates
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = FundingRateAnalyzer()
    return analyzer.analyze(df)


# Test function
def test_funding_rate_analyzer():
    """Test the Funding Rate analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data
    periods = 100
    np.random.seed(42)
    
    # Generate timestamps
    base_timestamp = datetime.now() - timedelta(days=periods)
    timestamps = np.array([base_timestamp + timedelta(hours=8*i) for i in range(periods)])
    
    # Generate price data with trend and noise
    prices = np.cumsum(np.random.normal(0, 1, periods)) + 100
    
    # Generate funding rate data with some patterns
    # First half: mostly negative funding
    # Second half: mostly positive funding
    base_funding = np.random.normal(0, 0.001, periods)
    trend_funding = np.concatenate([
        np.ones(periods//2) * -0.001,
        np.ones(periods - periods//2) * 0.001
    ])
    
    # Add some spikes
    for i in range(5):
        spike_idx = np.random.randint(0, periods)
        base_funding[spike_idx] = np.random.choice([0.005, -0.005])
    
    funding_rates = base_funding + trend_funding
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "funding_rate": funding_rates
    })
    
    # Create analyzer
    analyzer = FundingRateAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(df)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nAnomaly Analysis:")
    print(f"  Current Rate: {result['anomaly_analysis']['current_rate']:.6f}")
    print(f"  Z-Score: {result['anomaly_analysis']['current_z_score']:.3f}")
    print(f"  Is Anomaly: {result['anomaly_analysis']['is_anomaly']}")
    print(f"  Direction: {result['anomaly_analysis']['direction']}")
    
    print("\nBias Analysis:")
    print(f"  Bias Direction: {result['bias_analysis']['bias_direction']}")
    print(f"  Positive %: {result['bias_analysis']['positive_percentage']:.1f}%")
    print(f"  Negative %: {result['bias_analysis']['negative_percentage']:.1f}%")
    print(f"  Net Bias: {result['bias_analysis']['net_bias']:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price and Funding Rate
    plt.subplot(2, 1, 1)
    plt.plot(prices, label="Price")
    plt.title("Price")
    plt.grid(True)
    
    # Plot 2: Funding Rates
    plt.subplot(2, 1, 2)
    plt.plot(funding_rates, label="Funding Rate")
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title("Funding Rate")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_funding_rate_analyzer()
