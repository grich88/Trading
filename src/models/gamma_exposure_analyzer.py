"""
Gamma Exposure / Option Flow Analysis module.

This module provides advanced analysis of gamma exposure and option flow data,
including gamma levels, dealer positioning, and combined signal generation.
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
    GAMMA_LOOKBACK_PERIOD,
    GAMMA_SIGNAL_THRESHOLD,
    GAMMA_FLIP_THRESHOLD,
    OPTION_FLOW_THRESHOLD
)

logger = get_logger("GammaExposureAnalyzer")


class GammaExposureAnalyzer:
    """
    Advanced Gamma Exposure / Option Flow analyzer for market data.
    
    This class provides methods for:
    - Gamma exposure calculation and analysis
    - Dealer positioning analysis
    - Option flow analysis
    - Gamma flip detection
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = GAMMA_LOOKBACK_PERIOD,
                 signal_threshold: float = GAMMA_SIGNAL_THRESHOLD,
                 flip_threshold: float = GAMMA_FLIP_THRESHOLD,
                 flow_threshold: float = OPTION_FLOW_THRESHOLD):
        """
        Initialize the Gamma Exposure analyzer.
        
        Args:
            lookback_period: Lookback period for analysis
            signal_threshold: Threshold for signal generation
            flip_threshold: Threshold for gamma flip detection
            flow_threshold: Threshold for significant option flow
        """
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
        self.flip_threshold = flip_threshold
        self.flow_threshold = flow_threshold
        
        logger.info(f"GammaExposureAnalyzer initialized with lookback period: {lookback_period}, "
                   f"flip threshold: {flip_threshold}")
    
    @performance_monitor()
    def calculate_gamma_exposure(self, 
                               options_data: pd.DataFrame,
                               spot_price: float) -> Dict[str, Any]:
        """
        Calculate gamma exposure from options data.
        
        Args:
            options_data: DataFrame with option chain data
            spot_price: Current spot price
            
        Returns:
            Dictionary with gamma exposure information
        """
        required_columns = ["strike", "gamma", "open_interest", "option_type"]
        for col in required_columns:
            if col not in options_data.columns:
                raise ModelError(f"Required column not found in options data: {col}")
        
        # Separate calls and puts
        calls = options_data[options_data["option_type"] == "CALL"].copy()
        puts = options_data[options_data["option_type"] == "PUT"].copy()
        
        # Calculate gamma exposure for calls and puts
        # For market makers: Call gamma is negative, Put gamma is positive
        calls["dealer_gamma"] = -calls["gamma"] * calls["open_interest"] * 100  # Each contract = 100 shares
        puts["dealer_gamma"] = puts["gamma"] * puts["open_interest"] * 100
        
        # Calculate total gamma by strike
        gamma_by_strike = {}
        
        for _, call in calls.iterrows():
            strike = call["strike"]
            if strike not in gamma_by_strike:
                gamma_by_strike[strike] = 0
            gamma_by_strike[strike] += call["dealer_gamma"]
        
        for _, put in puts.iterrows():
            strike = put["strike"]
            if strike not in gamma_by_strike:
                gamma_by_strike[strike] = 0
            gamma_by_strike[strike] += put["dealer_gamma"]
        
        # Convert to sorted lists
        strikes = sorted(gamma_by_strike.keys())
        gammas = [gamma_by_strike[strike] for strike in strikes]
        
        # Calculate cumulative gamma
        cumulative_gamma = np.cumsum(gammas)
        
        # Find gamma flip point (where cumulative gamma crosses zero)
        gamma_flip_point = None
        for i in range(1, len(cumulative_gamma)):
            if cumulative_gamma[i-1] < 0 and cumulative_gamma[i] >= 0:
                # Interpolate to find exact flip point
                ratio = -cumulative_gamma[i-1] / (cumulative_gamma[i] - cumulative_gamma[i-1])
                gamma_flip_point = strikes[i-1] + ratio * (strikes[i] - strikes[i-1])
                break
            elif cumulative_gamma[i-1] >= 0 and cumulative_gamma[i] < 0:
                # Interpolate to find exact flip point
                ratio = cumulative_gamma[i-1] / (cumulative_gamma[i-1] - cumulative_gamma[i])
                gamma_flip_point = strikes[i-1] + ratio * (strikes[i] - strikes[i-1])
                break
        
        # Calculate total gamma exposure
        total_gamma = sum(gammas)
        
        # Calculate gamma at current spot
        spot_gamma = 0
        if strikes:
            # Find nearest strikes
            nearest_idx = np.searchsorted(strikes, spot_price)
            if nearest_idx > 0 and nearest_idx < len(strikes):
                # Interpolate gamma at spot
                lower_strike = strikes[nearest_idx - 1]
                upper_strike = strikes[nearest_idx]
                lower_gamma = gammas[nearest_idx - 1]
                upper_gamma = gammas[nearest_idx]
                
                ratio = (spot_price - lower_strike) / (upper_strike - lower_strike)
                spot_gamma = lower_gamma + ratio * (upper_gamma - lower_gamma)
        
        # Determine dealer positioning
        if total_gamma > 0:
            dealer_position = "long_gamma"  # Dealers are long gamma, will hedge by selling rallies and buying dips
        else:
            dealer_position = "short_gamma"  # Dealers are short gamma, will hedge by buying rallies and selling dips
        
        return {
            "strikes": strikes,
            "gammas": gammas,
            "cumulative_gamma": cumulative_gamma.tolist(),
            "total_gamma": float(total_gamma),
            "spot_gamma": float(spot_gamma),
            "gamma_flip_point": float(gamma_flip_point) if gamma_flip_point else None,
            "dealer_position": dealer_position,
            "distance_to_flip": float(abs(spot_price - gamma_flip_point)) if gamma_flip_point else None,
            "flip_percentage": float(abs(spot_price - gamma_flip_point) / spot_price * 100) if gamma_flip_point else None
        }
    
    @performance_monitor()
    def analyze_option_flow(self, 
                          option_trades: pd.DataFrame,
                          lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze option flow to determine institutional positioning.
        
        Args:
            option_trades: DataFrame with option trade data
            lookback_hours: Number of hours to look back
            
        Returns:
            Dictionary with option flow analysis
        """
        required_columns = ["timestamp", "volume", "premium", "option_type", "strike", "trade_type"]
        for col in required_columns:
            if col not in option_trades.columns:
                raise ModelError(f"Required column not found in option trades: {col}")
        
        # Filter trades to lookback period
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        recent_trades = option_trades[pd.to_datetime(option_trades["timestamp"]) >= cutoff_time].copy()
        
        if len(recent_trades) == 0:
            return {
                "total_call_volume": 0,
                "total_put_volume": 0,
                "call_put_ratio": 1.0,
                "total_call_premium": 0,
                "total_put_premium": 0,
                "premium_ratio": 1.0,
                "buy_sell_ratio": 1.0,
                "large_trades": [],
                "unusual_activity": False,
                "flow_sentiment": "neutral"
            }
        
        # Separate calls and puts
        call_trades = recent_trades[recent_trades["option_type"] == "CALL"]
        put_trades = recent_trades[recent_trades["option_type"] == "PUT"]
        
        # Calculate volumes
        total_call_volume = call_trades["volume"].sum()
        total_put_volume = put_trades["volume"].sum()
        
        # Calculate call/put ratio
        if total_put_volume > 0:
            call_put_ratio = total_call_volume / total_put_volume
        else:
            call_put_ratio = float('inf') if total_call_volume > 0 else 1.0
        
        # Calculate premiums
        total_call_premium = call_trades["premium"].sum()
        total_put_premium = put_trades["premium"].sum()
        
        # Calculate premium ratio
        if total_put_premium > 0:
            premium_ratio = total_call_premium / total_put_premium
        else:
            premium_ratio = float('inf') if total_call_premium > 0 else 1.0
        
        # Calculate buy/sell ratio
        buy_trades = recent_trades[recent_trades["trade_type"] == "BUY"]
        sell_trades = recent_trades[recent_trades["trade_type"] == "SELL"]
        
        buy_volume = buy_trades["volume"].sum()
        sell_volume = sell_trades["volume"].sum()
        
        if sell_volume > 0:
            buy_sell_ratio = buy_volume / sell_volume
        else:
            buy_sell_ratio = float('inf') if buy_volume > 0 else 1.0
        
        # Identify large trades (top 10% by premium)
        premium_threshold = recent_trades["premium"].quantile(0.9)
        large_trades = recent_trades[recent_trades["premium"] >= premium_threshold].to_dict('records')
        
        # Sort by premium
        large_trades.sort(key=lambda x: x["premium"], reverse=True)
        
        # Detect unusual activity
        unusual_activity = False
        if call_put_ratio > 2.0 or call_put_ratio < 0.5:
            unusual_activity = True
        if premium_ratio > 2.5 or premium_ratio < 0.4:
            unusual_activity = True
        if buy_sell_ratio > 3.0 or buy_sell_ratio < 0.33:
            unusual_activity = True
        
        # Determine flow sentiment
        flow_sentiment = "neutral"
        bullish_score = 0
        bearish_score = 0
        
        # Call/put ratio sentiment
        if call_put_ratio > 1.5:
            bullish_score += 1
        elif call_put_ratio < 0.67:
            bearish_score += 1
        
        # Premium ratio sentiment
        if premium_ratio > 1.5:
            bullish_score += 1
        elif premium_ratio < 0.67:
            bearish_score += 1
        
        # Buy/sell ratio sentiment
        if buy_sell_ratio > 1.5:
            bullish_score += 1
        elif buy_sell_ratio < 0.67:
            bearish_score += 1
        
        # Determine overall sentiment
        if bullish_score > bearish_score:
            flow_sentiment = "bullish"
        elif bearish_score > bullish_score:
            flow_sentiment = "bearish"
        
        return {
            "total_call_volume": int(total_call_volume),
            "total_put_volume": int(total_put_volume),
            "call_put_ratio": float(call_put_ratio) if call_put_ratio != float('inf') else None,
            "total_call_premium": float(total_call_premium),
            "total_put_premium": float(total_put_premium),
            "premium_ratio": float(premium_ratio) if premium_ratio != float('inf') else None,
            "buy_sell_ratio": float(buy_sell_ratio) if buy_sell_ratio != float('inf') else None,
            "large_trades": large_trades[:10],  # Top 10 large trades
            "unusual_activity": unusual_activity,
            "flow_sentiment": flow_sentiment
        }
    
    @performance_monitor()
    def analyze_gamma_dynamics(self, 
                             gamma_history: List[Dict[str, Any]],
                             price_history: np.ndarray) -> Dict[str, Any]:
        """
        Analyze gamma dynamics over time.
        
        Args:
            gamma_history: List of historical gamma exposure calculations
            price_history: Array of historical prices
            
        Returns:
            Dictionary with gamma dynamics analysis
        """
        if len(gamma_history) < 2:
            raise ModelError("Not enough historical data to analyze gamma dynamics")
        
        if len(gamma_history) != len(price_history):
            raise ModelError("Gamma history and price history must have the same length")
        
        # Extract time series data
        total_gammas = [g["total_gamma"] for g in gamma_history]
        flip_points = [g["gamma_flip_point"] for g in gamma_history if g["gamma_flip_point"] is not None]
        
        # Calculate gamma momentum
        gamma_changes = np.diff(total_gammas)
        gamma_momentum = np.mean(gamma_changes) if len(gamma_changes) > 0 else 0
        
        # Calculate flip point stability
        flip_stability = 0
        if len(flip_points) > 1:
            flip_changes = np.diff(flip_points)
            flip_volatility = np.std(flip_changes) / np.mean(flip_points) if np.mean(flip_points) > 0 else 0
            flip_stability = 1 - min(1, flip_volatility)
        
        # Analyze gamma vs price relationship
        price_changes = np.diff(price_history)
        
        # Check if gamma is acting as support/resistance
        gamma_effectiveness = 0
        flip_tests = 0
        successful_tests = 0
        
        for i in range(1, len(gamma_history)):
            if gamma_history[i]["gamma_flip_point"] is not None:
                flip = gamma_history[i]["gamma_flip_point"]
                price = price_history[i]
                prev_price = price_history[i-1]
                
                # Check if price tested the flip point
                if min(price, prev_price) <= flip <= max(price, prev_price):
                    flip_tests += 1
                    # Check if flip point held
                    if (prev_price > flip and price > flip * 0.995) or \
                       (prev_price < flip and price < flip * 1.005):
                        successful_tests += 1
        
        if flip_tests > 0:
            gamma_effectiveness = successful_tests / flip_tests
        
        # Determine current gamma regime
        current_gamma = total_gammas[-1]
        avg_gamma = np.mean(total_gammas)
        
        if current_gamma > avg_gamma * 1.5:
            gamma_regime = "high_gamma"
        elif current_gamma < avg_gamma * 0.5:
            gamma_regime = "low_gamma"
        else:
            gamma_regime = "normal_gamma"
        
        # Check for gamma squeeze conditions
        gamma_squeeze = False
        if len(total_gammas) > 3:
            recent_gammas = total_gammas[-3:]
            if all(g < 0 for g in recent_gammas) and abs(recent_gammas[-1]) > abs(recent_gammas[-3]) * 1.5:
                gamma_squeeze = True
        
        return {
            "gamma_momentum": float(gamma_momentum),
            "flip_stability": float(flip_stability),
            "gamma_effectiveness": float(gamma_effectiveness),
            "flip_tests": int(flip_tests),
            "successful_tests": int(successful_tests),
            "gamma_regime": gamma_regime,
            "gamma_squeeze": gamma_squeeze,
            "current_gamma": float(current_gamma),
            "avg_gamma": float(avg_gamma)
        }
    
    @performance_monitor()
    def generate_signal(self, 
                      gamma_exposure: Dict[str, Any],
                      option_flow: Dict[str, Any],
                      gamma_dynamics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined Gamma Exposure / Option Flow signal.
        
        Args:
            gamma_exposure: Current gamma exposure analysis
            option_flow: Option flow analysis
            gamma_dynamics: Gamma dynamics analysis
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Gamma Exposure Component (range: -0.4 to 0.4)
        gamma_score = 0.0
        
        # Check dealer positioning
        if gamma_exposure["dealer_position"] == "short_gamma":
            # Short gamma means increased volatility, trend following
            gamma_score = 0.1
            
            # Adjust for distance to flip
            if gamma_exposure["gamma_flip_point"] is not None:
                if gamma_exposure["flip_percentage"] < 2.0:
                    # Very close to flip point, expect volatility
                    gamma_score += 0.1
                elif gamma_exposure["flip_percentage"] > 5.0:
                    # Far from flip point, less impact
                    gamma_score *= 0.5
        else:
            # Long gamma means decreased volatility, mean reversion
            gamma_score = -0.1
            
            # Adjust for gamma magnitude
            if abs(gamma_exposure["total_gamma"]) > abs(gamma_dynamics["avg_gamma"]) * 2:
                gamma_score *= 1.5
        
        # Check for gamma squeeze
        if gamma_dynamics["gamma_squeeze"]:
            gamma_score += 0.2  # Gamma squeeze is bullish
        
        # Cap score
        gamma_score = max(-0.4, min(0.4, gamma_score))
        signal_components["gamma_exposure"] = gamma_score
        
        # 2. Option Flow Component (range: -0.3 to 0.3)
        flow_score = 0.0
        
        # Check flow sentiment
        if option_flow["flow_sentiment"] == "bullish":
            flow_score = 0.2
        elif option_flow["flow_sentiment"] == "bearish":
            flow_score = -0.2
        
        # Adjust for unusual activity
        if option_flow["unusual_activity"]:
            flow_score *= 1.5
        
        # Adjust for call/put ratio
        if option_flow["call_put_ratio"] is not None:
            if option_flow["call_put_ratio"] > 2.0:
                flow_score += 0.1
            elif option_flow["call_put_ratio"] < 0.5:
                flow_score -= 0.1
        
        # Cap score
        flow_score = max(-0.3, min(0.3, flow_score))
        signal_components["option_flow"] = flow_score
        
        # 3. Gamma Dynamics Component (range: -0.3 to 0.3)
        dynamics_score = 0.0
        
        # Check gamma regime
        if gamma_dynamics["gamma_regime"] == "high_gamma":
            dynamics_score = -0.1  # High gamma = mean reversion
        elif gamma_dynamics["gamma_regime"] == "low_gamma":
            dynamics_score = 0.1   # Low gamma = trending
        
        # Adjust for gamma momentum
        if gamma_dynamics["gamma_momentum"] > 0:
            dynamics_score -= 0.1  # Increasing gamma = decreasing volatility
        else:
            dynamics_score += 0.1  # Decreasing gamma = increasing volatility
        
        # Adjust for gamma effectiveness
        dynamics_score *= (0.5 + gamma_dynamics["gamma_effectiveness"])
        
        # Adjust for flip stability
        if gamma_dynamics["flip_stability"] > 0.8:
            # Stable flip points are more reliable
            dynamics_score *= 1.2
        
        # Cap score
        dynamics_score = max(-0.3, min(0.3, dynamics_score))
        signal_components["gamma_dynamics"] = dynamics_score
        
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
    def analyze(self, 
               options_data: pd.DataFrame, 
               option_trades: pd.DataFrame,
               price_history: pd.DataFrame,
               gamma_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Perform complete Gamma Exposure / Option Flow analysis.
        
        Args:
            options_data: DataFrame with current option chain data
            option_trades: DataFrame with recent option trades
            price_history: DataFrame with price history
            gamma_history: Optional list of historical gamma calculations
            
        Returns:
            Dictionary with complete analysis results
        """
        # Check required columns in price_history
        if "close" not in price_history.columns:
            raise ModelError("Price history must contain 'close' column")
        
        # Get current spot price
        spot_price = price_history["close"].iloc[-1]
        
        # Calculate gamma exposure
        gamma_exposure = self.calculate_gamma_exposure(options_data, spot_price)
        
        # Analyze option flow
        option_flow = self.analyze_option_flow(option_trades)
        
        # Analyze gamma dynamics if history is provided
        if gamma_history is not None and len(gamma_history) >= 2:
            gamma_dynamics = self.analyze_gamma_dynamics(
                gamma_history,
                price_history["close"].values[-len(gamma_history):]
            )
        else:
            # Create dummy gamma history from current data
            gamma_history = [gamma_exposure]
            gamma_dynamics = {
                "gamma_momentum": 0.0,
                "flip_stability": 0.5,
                "gamma_effectiveness": 0.5,
                "flip_tests": 0,
                "successful_tests": 0,
                "gamma_regime": "normal_gamma",
                "gamma_squeeze": False,
                "current_gamma": gamma_exposure["total_gamma"],
                "avg_gamma": gamma_exposure["total_gamma"]
            }
        
        # Generate signal
        signal = self.generate_signal(
            gamma_exposure,
            option_flow,
            gamma_dynamics
        )
        
        # Combine all results
        result = {
            "spot_price": float(spot_price),
            "gamma_exposure": gamma_exposure,
            "option_flow": option_flow,
            "gamma_dynamics": gamma_dynamics,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def calculate_gamma_exposure(options_data: pd.DataFrame, spot_price: float) -> Dict[str, Any]:
    """
    Standalone function to calculate gamma exposure.
    
    Args:
        options_data: DataFrame with option chain data
        spot_price: Current spot price
        
    Returns:
        Dictionary with gamma exposure information
    """
    analyzer = GammaExposureAnalyzer()
    return analyzer.calculate_gamma_exposure(options_data, spot_price)


def analyze_gamma_exposure(options_data: pd.DataFrame, 
                         option_trades: pd.DataFrame,
                         price_history: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze gamma exposure and option flow.
    
    Args:
        options_data: DataFrame with option chain data
        option_trades: DataFrame with recent option trades
        price_history: DataFrame with price history
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = GammaExposureAnalyzer()
    return analyzer.analyze(options_data, option_trades, price_history)


# Test function
def test_gamma_exposure_analyzer():
    """Test the Gamma Exposure analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample options data
    np.random.seed(42)
    spot_price = 100
    
    # Generate strikes around spot
    strikes = np.arange(80, 121, 2.5)
    
    options_data = []
    for strike in strikes:
        # Generate call data
        moneyness = spot_price / strike
        gamma = np.exp(-0.5 * ((moneyness - 1) / 0.1) ** 2) * np.random.uniform(0.01, 0.05)
        oi = np.random.randint(100, 10000) * (1 + np.exp(-abs(strike - spot_price) / 5))
        
        options_data.append({
            "strike": strike,
            "gamma": gamma,
            "open_interest": oi,
            "option_type": "CALL"
        })
        
        # Generate put data
        gamma = np.exp(-0.5 * ((1 / moneyness - 1) / 0.1) ** 2) * np.random.uniform(0.01, 0.05)
        oi = np.random.randint(100, 10000) * (1 + np.exp(-abs(strike - spot_price) / 5))
        
        options_data.append({
            "strike": strike,
            "gamma": gamma,
            "open_interest": oi,
            "option_type": "PUT"
        })
    
    options_df = pd.DataFrame(options_data)
    
    # Generate sample option trades
    trade_times = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
    
    option_trades = []
    for t in trade_times:
        n_trades = np.random.randint(5, 20)
        for _ in range(n_trades):
            option_trades.append({
                "timestamp": t,
                "volume": np.random.randint(10, 1000),
                "premium": np.random.uniform(100, 10000),
                "option_type": np.random.choice(["CALL", "PUT"]),
                "strike": np.random.choice(strikes),
                "trade_type": np.random.choice(["BUY", "SELL"], p=[0.6, 0.4])
            })
    
    trades_df = pd.DataFrame(option_trades)
    
    # Generate price history
    price_history = pd.DataFrame({
        "close": spot_price + np.cumsum(np.random.normal(0, 0.5, 100))
    })
    
    # Create analyzer
    analyzer = GammaExposureAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(options_df, trades_df, price_history)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nGamma Exposure:")
    print(f"  Total Gamma: {result['gamma_exposure']['total_gamma']:.0f}")
    print(f"  Dealer Position: {result['gamma_exposure']['dealer_position']}")
    print(f"  Gamma Flip Point: {result['gamma_exposure']['gamma_flip_point']:.2f}")
    print(f"  Distance to Flip: {result['gamma_exposure']['flip_percentage']:.1f}%")
    
    print("\nOption Flow:")
    print(f"  Call/Put Ratio: {result['option_flow']['call_put_ratio']:.2f}")
    print(f"  Flow Sentiment: {result['option_flow']['flow_sentiment']}")
    print(f"  Unusual Activity: {result['option_flow']['unusual_activity']}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Gamma Profile
    plt.subplot(2, 1, 1)
    strikes = result['gamma_exposure']['strikes']
    gammas = result['gamma_exposure']['gammas']
    plt.bar(strikes, gammas, width=1.5, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=spot_price, color='g', linestyle='--', label=f'Spot: {spot_price:.2f}')
    if result['gamma_exposure']['gamma_flip_point']:
        plt.axvline(x=result['gamma_exposure']['gamma_flip_point'], 
                   color='r', linestyle='--', 
                   label=f"Flip: {result['gamma_exposure']['gamma_flip_point']:.2f}")
    plt.title("Dealer Gamma Exposure by Strike")
    plt.xlabel("Strike Price")
    plt.ylabel("Gamma Exposure")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Gamma
    plt.subplot(2, 1, 2)
    cumulative = result['gamma_exposure']['cumulative_gamma']
    plt.plot(strikes, cumulative, 'b-', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=spot_price, color='g', linestyle='--', label=f'Spot: {spot_price:.2f}')
    if result['gamma_exposure']['gamma_flip_point']:
        plt.axvline(x=result['gamma_exposure']['gamma_flip_point'], 
                   color='r', linestyle='--', 
                   label=f"Flip: {result['gamma_exposure']['gamma_flip_point']:.2f}")
    plt.title("Cumulative Gamma Exposure")
    plt.xlabel("Strike Price")
    plt.ylabel("Cumulative Gamma")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_gamma_exposure_analyzer()
