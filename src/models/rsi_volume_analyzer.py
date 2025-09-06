"""
RSI and Volume Analysis module.

This module provides advanced analysis of RSI (Relative Strength Index) and volume patterns,
including divergence detection, volume confirmation, and combined signal generation.
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
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    RSI_PERIOD,
    VOLUME_MA_PERIOD,
    RSI_DIVERGENCE_LOOKBACK,
    RSI_SIGNAL_THRESHOLD
)

logger = get_logger("RSIVolumeAnalyzer")


class RSIVolumeAnalyzer:
    """
    Advanced RSI and Volume analyzer for market data.
    
    This class provides methods for:
    - RSI calculation and analysis
    - Volume pattern detection
    - RSI-volume divergence detection
    - Combined signal generation
    """
    
    def __init__(self, 
                 rsi_period: int = RSI_PERIOD,
                 rsi_overbought: float = RSI_OVERBOUGHT,
                 rsi_oversold: float = RSI_OVERSOLD,
                 volume_ma_period: int = VOLUME_MA_PERIOD,
                 divergence_lookback: int = RSI_DIVERGENCE_LOOKBACK,
                 signal_threshold: float = RSI_SIGNAL_THRESHOLD):
        """
        Initialize the RSI and Volume analyzer.
        
        Args:
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI level considered overbought
            rsi_oversold: RSI level considered oversold
            volume_ma_period: Period for volume moving average
            divergence_lookback: Lookback period for divergence detection
            signal_threshold: Threshold for signal generation
        """
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.divergence_lookback = divergence_lookback
        self.signal_threshold = signal_threshold
        
        logger.info(f"RSIVolumeAnalyzer initialized with RSI period: {rsi_period}, "
                   f"Volume MA period: {volume_ma_period}")
    
    @performance_monitor()
    def calculate_rsi(self, prices: np.ndarray, period: Optional[int] = None) -> np.ndarray:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Array of price data
            period: Period for RSI calculation (default: self.rsi_period)
            
        Returns:
            Array of RSI values
        """
        if period is None:
            period = self.rsi_period
        
        if len(prices) < period + 1:
            raise ModelError(f"Not enough data to calculate RSI. Need at least {period + 1} data points.")
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Initialize arrays
        gains = np.zeros_like(deltas)
        losses = np.zeros_like(deltas)
        
        # Separate gains and losses
        gains[deltas > 0] = deltas[deltas > 0]
        losses[deltas < 0] = -deltas[deltas < 0]
        
        # Calculate average gains and losses
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])
        
        # Calculate subsequent averages
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
        
        # Calculate RS and RSI
        rs = np.zeros_like(prices)
        rsi = np.zeros_like(prices)
        
        # Avoid division by zero
        nonzero_indices = avg_loss > 0
        rs[nonzero_indices] = avg_gain[nonzero_indices] / avg_loss[nonzero_indices]
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Set initial values to NaN
        rsi[:period] = np.nan
        
        return rsi
    
    @performance_monitor()
    def calculate_volume_ma(self, volume: np.ndarray, period: Optional[int] = None) -> np.ndarray:
        """
        Calculate volume moving average.
        
        Args:
            volume: Array of volume data
            period: Period for moving average (default: self.volume_ma_period)
            
        Returns:
            Array of volume moving average values
        """
        if period is None:
            period = self.volume_ma_period
        
        if len(volume) < period:
            raise ModelError(f"Not enough data to calculate volume MA. Need at least {period} data points.")
        
        # Calculate simple moving average
        volume_ma = np.zeros_like(volume, dtype=float)
        
        for i in range(period - 1, len(volume)):
            volume_ma[i] = np.mean(volume[i-period+1:i+1])
        
        # Set initial values to NaN
        volume_ma[:period-1] = np.nan
        
        return volume_ma
    
    @performance_monitor()
    def detect_volume_anomalies(self, volume: np.ndarray, volume_ma: np.ndarray, threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect volume anomalies.
        
        Args:
            volume: Array of volume data
            volume_ma: Array of volume moving average values
            threshold: Threshold for anomaly detection (default: 1.5)
            
        Returns:
            Dictionary with volume anomaly information
        """
        # Calculate volume relative to moving average
        volume_ratio = volume / volume_ma
        
        # Detect anomalies
        high_volume = volume_ratio > threshold
        very_high_volume = volume_ratio > threshold * 2
        extremely_high_volume = volume_ratio > threshold * 3
        
        # Calculate recent anomalies (last 5 periods)
        recent_high = np.sum(high_volume[-5:])
        recent_very_high = np.sum(very_high_volume[-5:])
        recent_extremely_high = np.sum(extremely_high_volume[-5:])
        
        # Calculate volume trend (last 10 periods)
        if len(volume) >= 10:
            volume_trend = np.polyfit(np.arange(10), volume[-10:], 1)[0]
        else:
            volume_trend = 0
        
        # Normalize volume trend
        if abs(volume_trend) > 0:
            volume_trend_normalized = volume_trend / np.mean(volume[-10:])
        else:
            volume_trend_normalized = 0
        
        return {
            "high_volume": high_volume,
            "very_high_volume": very_high_volume,
            "extremely_high_volume": extremely_high_volume,
            "recent_high": int(recent_high),
            "recent_very_high": int(recent_very_high),
            "recent_extremely_high": int(recent_extremely_high),
            "volume_trend": float(volume_trend_normalized),
            "volume_ratio": volume_ratio.tolist()
        }
    
    @performance_monitor()
    def detect_rsi_divergence(self, 
                            prices: np.ndarray, 
                            rsi: np.ndarray, 
                            lookback: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect RSI divergence.
        
        Args:
            prices: Array of price data
            rsi: Array of RSI values
            lookback: Lookback period for divergence detection (default: self.divergence_lookback)
            
        Returns:
            Dictionary with divergence information
        """
        if lookback is None:
            lookback = self.divergence_lookback
        
        if len(prices) < lookback or len(rsi) < lookback:
            raise ModelError(f"Not enough data to detect divergence. Need at least {lookback} data points.")
        
        # Get the last lookback periods
        recent_prices = prices[-lookback:]
        recent_rsi = rsi[-lookback:]
        
        # Find local maxima and minima
        price_max_idx = np.argmax(recent_prices)
        price_min_idx = np.argmin(recent_prices)
        rsi_max_idx = np.argmax(recent_rsi)
        rsi_min_idx = np.argmin(recent_rsi)
        
        # Detect bearish divergence (price makes higher high, RSI makes lower high)
        bearish_divergence = False
        if price_max_idx > 0 and rsi_max_idx > 0:
            # Find previous high
            prev_price_max_idx = np.argmax(recent_prices[:price_max_idx])
            prev_rsi_max_idx = np.argmax(recent_rsi[:rsi_max_idx])
            
            # Check if price made higher high but RSI made lower high
            if (recent_prices[price_max_idx] > recent_prices[prev_price_max_idx] and
                recent_rsi[rsi_max_idx] < recent_rsi[prev_rsi_max_idx]):
                bearish_divergence = True
        
        # Detect bullish divergence (price makes lower low, RSI makes higher low)
        bullish_divergence = False
        if price_min_idx > 0 and rsi_min_idx > 0:
            # Find previous low
            prev_price_min_idx = np.argmin(recent_prices[:price_min_idx])
            prev_rsi_min_idx = np.argmin(recent_rsi[:rsi_min_idx])
            
            # Check if price made lower low but RSI made higher low
            if (recent_prices[price_min_idx] < recent_prices[prev_price_min_idx] and
                recent_rsi[rsi_min_idx] > recent_rsi[prev_rsi_min_idx]):
                bullish_divergence = True
        
        # Calculate divergence strength
        divergence_strength = 0.0
        if bearish_divergence:
            # Calculate strength based on difference between RSI highs
            max_rsi = np.max(recent_rsi)
            prev_max_rsi = np.max(recent_rsi[:rsi_max_idx])
            divergence_strength = -abs(max_rsi - prev_max_rsi) / 100.0
        elif bullish_divergence:
            # Calculate strength based on difference between RSI lows
            min_rsi = np.min(recent_rsi)
            prev_min_rsi = np.min(recent_rsi[:rsi_min_idx])
            divergence_strength = abs(min_rsi - prev_min_rsi) / 100.0
        
        return {
            "bearish_divergence": bearish_divergence,
            "bullish_divergence": bullish_divergence,
            "divergence_strength": float(divergence_strength)
        }
    
    @performance_monitor()
    def analyze_rsi_zones(self, rsi: np.ndarray) -> Dict[str, Any]:
        """
        Analyze RSI zones (overbought/oversold).
        
        Args:
            rsi: Array of RSI values
            
        Returns:
            Dictionary with RSI zone analysis
        """
        if len(rsi) < 5:
            raise ModelError("Not enough data to analyze RSI zones. Need at least 5 data points.")
        
        # Check current and recent RSI values
        current_rsi = rsi[-1]
        
        # Check overbought/oversold conditions
        is_overbought = current_rsi > self.rsi_overbought
        is_oversold = current_rsi < self.rsi_oversold
        
        # Check if RSI is exiting overbought/oversold zones
        exiting_overbought = False
        exiting_oversold = False
        
        if len(rsi) >= 3:
            prev_overbought = rsi[-2] > self.rsi_overbought
            prev_oversold = rsi[-2] < self.rsi_oversold
            
            exiting_overbought = prev_overbought and not is_overbought
            exiting_oversold = prev_oversold and not is_oversold
        
        # Calculate time spent in zones
        periods_in_overbought = 0
        periods_in_oversold = 0
        
        for i in range(min(10, len(rsi))):
            if rsi[-(i+1)] > self.rsi_overbought:
                periods_in_overbought += 1
            elif rsi[-(i+1)] < self.rsi_oversold:
                periods_in_oversold += 1
        
        # Calculate RSI momentum
        rsi_momentum = 0.0
        if len(rsi) >= 5:
            rsi_5_period_change = current_rsi - rsi[-5]
            rsi_momentum = rsi_5_period_change / 5.0
        
        return {
            "current_rsi": float(current_rsi),
            "is_overbought": is_overbought,
            "is_oversold": is_oversold,
            "exiting_overbought": exiting_overbought,
            "exiting_oversold": exiting_oversold,
            "periods_in_overbought": periods_in_overbought,
            "periods_in_oversold": periods_in_oversold,
            "rsi_momentum": float(rsi_momentum)
        }
    
    @performance_monitor()
    def analyze_volume_confirmation(self, 
                                  prices: np.ndarray, 
                                  volume: np.ndarray, 
                                  volume_ma: np.ndarray) -> Dict[str, Any]:
        """
        Analyze volume confirmation of price movements.
        
        Args:
            prices: Array of price data
            volume: Array of volume data
            volume_ma: Array of volume moving average values
            
        Returns:
            Dictionary with volume confirmation analysis
        """
        if len(prices) < 5 or len(volume) < 5 or len(volume_ma) < 5:
            raise ModelError("Not enough data to analyze volume confirmation. Need at least 5 data points.")
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Calculate volume relative to moving average
        volume_ratio = volume[1:] / volume_ma[1:]
        
        # Check if volume confirms price movement
        confirms_up = np.logical_and(price_changes > 0, volume_ratio > 1.0)
        confirms_down = np.logical_and(price_changes < 0, volume_ratio > 1.0)
        
        # Calculate recent confirmation (last 5 periods)
        recent_confirms_up = np.sum(confirms_up[-5:])
        recent_confirms_down = np.sum(confirms_down[-5:])
        
        # Calculate confirmation score
        confirmation_score = 0.0
        
        if len(price_changes) >= 5:
            # Positive score for up confirmation, negative for down confirmation
            for i in range(5):
                if confirms_up[-(i+1)]:
                    confirmation_score += (5 - i) * 0.1
                elif confirms_down[-(i+1)]:
                    confirmation_score -= (5 - i) * 0.1
        
        # Check for climax volume
        climax_volume = False
        if len(volume) >= 10:
            max_volume_idx = np.argmax(volume[-10:])
            if max_volume_idx >= 8:  # In the last 2 periods
                climax_volume = True
        
        return {
            "confirms_up": confirms_up.tolist(),
            "confirms_down": confirms_down.tolist(),
            "recent_confirms_up": int(recent_confirms_up),
            "recent_confirms_down": int(recent_confirms_down),
            "confirmation_score": float(confirmation_score),
            "climax_volume": climax_volume
        }
    
    @performance_monitor()
    def generate_rsi_volume_signal(self, 
                                 rsi_analysis: Dict[str, Any],
                                 divergence_analysis: Dict[str, Any],
                                 volume_analysis: Dict[str, Any],
                                 confirmation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined RSI and volume signal.
        
        Args:
            rsi_analysis: RSI zone analysis
            divergence_analysis: RSI divergence analysis
            volume_analysis: Volume anomaly analysis
            confirmation_analysis: Volume confirmation analysis
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. RSI Zone Component (range: -0.4 to 0.4)
        rsi_zone_score = 0.0
        if rsi_analysis["is_overbought"]:
            rsi_zone_score = -0.3
        elif rsi_analysis["is_oversold"]:
            rsi_zone_score = 0.3
        elif rsi_analysis["exiting_overbought"]:
            rsi_zone_score = -0.4
        elif rsi_analysis["exiting_oversold"]:
            rsi_zone_score = 0.4
        
        # Add momentum influence
        rsi_zone_score += rsi_analysis["rsi_momentum"] * 0.5
        
        # Cap score
        rsi_zone_score = max(-0.4, min(0.4, rsi_zone_score))
        signal_components["rsi_zone"] = rsi_zone_score
        
        # 2. Divergence Component (range: -0.3 to 0.3)
        divergence_score = 0.0
        if divergence_analysis["bearish_divergence"]:
            divergence_score = -0.3
        elif divergence_analysis["bullish_divergence"]:
            divergence_score = 0.3
        
        # Adjust by strength
        divergence_score *= abs(divergence_analysis["divergence_strength"]) * 3
        
        # Cap score
        divergence_score = max(-0.3, min(0.3, divergence_score))
        signal_components["divergence"] = divergence_score
        
        # 3. Volume Anomaly Component (range: -0.15 to 0.15)
        volume_score = 0.0
        
        # Volume trend influence
        volume_score += volume_analysis["volume_trend"] * 0.1
        
        # Recent high volume influence
        if volume_analysis["recent_high"] > 0:
            if confirmation_analysis["recent_confirms_up"] > confirmation_analysis["recent_confirms_down"]:
                volume_score += 0.05 * volume_analysis["recent_high"]
            elif confirmation_analysis["recent_confirms_down"] > confirmation_analysis["recent_confirms_up"]:
                volume_score -= 0.05 * volume_analysis["recent_high"]
        
        # Extremely high volume influence
        if volume_analysis["recent_extremely_high"] > 0:
            if confirmation_analysis["climax_volume"]:
                # Climax volume can be a reversal signal
                volume_score = -volume_score
        
        # Cap score
        volume_score = max(-0.15, min(0.15, volume_score))
        signal_components["volume"] = volume_score
        
        # 4. Confirmation Component (range: -0.15 to 0.15)
        confirmation_score = confirmation_analysis["confirmation_score"]
        
        # Cap score
        confirmation_score = max(-0.15, min(0.15, confirmation_score))
        signal_components["confirmation"] = confirmation_score
        
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
        Perform complete RSI and volume analysis on a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns
        required_columns = ["close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ModelError(f"Required column not found in DataFrame: {col}")
        
        # Extract data
        prices = df["close"].values
        volume = df["volume"].values
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        volume_ma = self.calculate_volume_ma(volume)
        
        # Perform analyses
        rsi_analysis = self.analyze_rsi_zones(rsi)
        divergence_analysis = self.detect_rsi_divergence(prices, rsi)
        volume_anomalies = self.detect_volume_anomalies(volume, volume_ma)
        volume_confirmation = self.analyze_volume_confirmation(prices, volume, volume_ma)
        
        # Generate signal
        signal = self.generate_rsi_volume_signal(
            rsi_analysis,
            divergence_analysis,
            volume_anomalies,
            volume_confirmation
        )
        
        # Combine all results
        result = {
            "rsi": rsi.tolist(),
            "volume_ma": volume_ma.tolist(),
            "rsi_analysis": rsi_analysis,
            "divergence_analysis": divergence_analysis,
            "volume_anomalies": volume_anomalies,
            "volume_confirmation": volume_confirmation,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def calculate_rsi(prices: np.ndarray, period: int = RSI_PERIOD) -> np.ndarray:
    """
    Standalone function to calculate RSI.
    
    Args:
        prices: Array of price data
        period: Period for RSI calculation
        
    Returns:
        Array of RSI values
    """
    analyzer = RSIVolumeAnalyzer(rsi_period=period)
    return analyzer.calculate_rsi(prices)


def analyze_rsi_volume(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze RSI and volume.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = RSIVolumeAnalyzer()
    return analyzer.analyze(df)


# Test function
def test_rsi_volume_analyzer():
    """Test the RSI and Volume analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    periods = 100
    np.random.seed(42)
    prices = np.cumsum(np.random.normal(0, 1, periods)) + 100
    volume = np.random.lognormal(0, 1, periods) * 1000
    
    # Create DataFrame
    df = pd.DataFrame({
        "close": prices,
        "volume": volume
    })
    
    # Create analyzer
    analyzer = RSIVolumeAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(df)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Price and Volume
    plt.subplot(3, 1, 1)
    plt.plot(prices, label="Price")
    plt.title("Price")
    plt.grid(True)
    
    # Plot 2: RSI
    plt.subplot(3, 1, 2)
    plt.plot(result["rsi"], label="RSI")
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title("RSI")
    plt.grid(True)
    
    # Plot 3: Volume
    plt.subplot(3, 1, 3)
    plt.bar(range(len(volume)), volume, alpha=0.5, label="Volume")
    plt.plot(result["volume_ma"], color='r', label="Volume MA")
    plt.title("Volume")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_rsi_volume_analyzer()
