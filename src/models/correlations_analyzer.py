"""
Correlations Analysis module for BTC vs SOL vs BONK.

This module provides advanced correlation analysis between crypto assets,
including rolling correlations, correlation breakdowns, divergence detection,
and combined signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from scipy import stats

from src.utils import (
    get_logger,
    performance_monitor,
    ModelError
)

# Import configuration
from src.config import (
    CORRELATION_LOOKBACK_PERIOD,
    CORRELATION_SIGNAL_THRESHOLD,
    CORRELATION_WINDOW_SHORT,
    CORRELATION_WINDOW_LONG
)

logger = get_logger("CorrelationsAnalyzer")


class CorrelationsAnalyzer:
    """
    Advanced Correlations analyzer for BTC vs SOL vs BONK.
    
    This class provides methods for:
    - Rolling correlation calculation
    - Correlation breakdown detection
    - Divergence analysis
    - Lead-lag relationship analysis
    - Combined signal generation
    """
    
    def __init__(self, 
                 lookback_period: int = CORRELATION_LOOKBACK_PERIOD,
                 signal_threshold: float = CORRELATION_SIGNAL_THRESHOLD,
                 window_short: int = CORRELATION_WINDOW_SHORT,
                 window_long: int = CORRELATION_WINDOW_LONG):
        """
        Initialize the Correlations analyzer.
        
        Args:
            lookback_period: Lookback period for analysis
            signal_threshold: Threshold for signal generation
            window_short: Short window for correlation calculation
            window_long: Long window for correlation calculation
        """
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
        self.window_short = window_short
        self.window_long = window_long
        
        logger.info(f"CorrelationsAnalyzer initialized with lookback period: {lookback_period}, "
                   f"windows: {window_short}/{window_long}")
    
    @performance_monitor()
    def calculate_rolling_correlations(self, 
                                     btc_data: pd.DataFrame,
                                     sol_data: pd.DataFrame,
                                     bonk_data: pd.DataFrame,
                                     window: int) -> Dict[str, pd.Series]:
        """
        Calculate rolling correlations between assets.
        
        Args:
            btc_data: DataFrame with BTC price data
            sol_data: DataFrame with SOL price data
            bonk_data: DataFrame with BONK price data
            window: Window size for rolling correlation
            
        Returns:
            Dictionary with correlation time series
        """
        # Check required columns
        for df, name in [(btc_data, "BTC"), (sol_data, "SOL"), (bonk_data, "BONK")]:
            if "close" not in df.columns:
                raise ModelError(f"{name} data must contain 'close' column")
        
        # Calculate returns
        btc_returns = btc_data["close"].pct_change()
        sol_returns = sol_data["close"].pct_change()
        bonk_returns = bonk_data["close"].pct_change()
        
        # Calculate rolling correlations
        correlations = {
            "btc_sol": btc_returns.rolling(window).corr(sol_returns),
            "btc_bonk": btc_returns.rolling(window).corr(bonk_returns),
            "sol_bonk": sol_returns.rolling(window).corr(bonk_returns)
        }
        
        return correlations
    
    @performance_monitor()
    def detect_correlation_breakdown(self, 
                                   correlations_short: Dict[str, pd.Series],
                                   correlations_long: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Detect correlation breakdowns between short and long term.
        
        Args:
            correlations_short: Short-term correlations
            correlations_long: Long-term correlations
            
        Returns:
            Dictionary with breakdown analysis
        """
        breakdowns = {}
        
        for pair in ["btc_sol", "btc_bonk", "sol_bonk"]:
            if pair not in correlations_short or pair not in correlations_long:
                continue
            
            short_corr = correlations_short[pair].iloc[-1] if len(correlations_short[pair]) > 0 else np.nan
            long_corr = correlations_long[pair].iloc[-1] if len(correlations_long[pair]) > 0 else np.nan
            
            if pd.isna(short_corr) or pd.isna(long_corr):
                breakdowns[pair] = {
                    "short_correlation": float(short_corr) if not pd.isna(short_corr) else None,
                    "long_correlation": float(long_corr) if not pd.isna(long_corr) else None,
                    "divergence": None,
                    "breakdown": False,
                    "breakdown_type": None
                }
                continue
            
            divergence = short_corr - long_corr
            
            # Detect breakdown types
            breakdown = False
            breakdown_type = None
            
            # Strong positive to negative
            if long_corr > 0.5 and short_corr < -0.2:
                breakdown = True
                breakdown_type = "positive_to_negative"
            # Strong negative to positive
            elif long_corr < -0.5 and short_corr > 0.2:
                breakdown = True
                breakdown_type = "negative_to_positive"
            # Decorrelation
            elif abs(long_corr) > 0.6 and abs(short_corr) < 0.2:
                breakdown = True
                breakdown_type = "decorrelation"
            # Significant divergence
            elif abs(divergence) > 0.5:
                breakdown = True
                breakdown_type = "divergence"
            
            breakdowns[pair] = {
                "short_correlation": float(short_corr),
                "long_correlation": float(long_corr),
                "divergence": float(divergence),
                "breakdown": breakdown,
                "breakdown_type": breakdown_type
            }
        
        return breakdowns
    
    @performance_monitor()
    def analyze_divergence(self, 
                         btc_data: pd.DataFrame,
                         sol_data: pd.DataFrame,
                         bonk_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price divergences between correlated assets.
        
        Args:
            btc_data: DataFrame with BTC price data
            sol_data: DataFrame with SOL price data
            bonk_data: DataFrame with BONK price data
            
        Returns:
            Dictionary with divergence analysis
        """
        # Calculate normalized returns over different periods
        periods = [1, 7, 30]
        divergences = {}
        
        for period in periods:
            if len(btc_data) < period or len(sol_data) < period or len(bonk_data) < period:
                continue
            
            # Calculate returns
            btc_return = (btc_data["close"].iloc[-1] / btc_data["close"].iloc[-period] - 1) * 100
            sol_return = (sol_data["close"].iloc[-1] / sol_data["close"].iloc[-period] - 1) * 100
            bonk_return = (bonk_data["close"].iloc[-1] / bonk_data["close"].iloc[-period] - 1) * 100
            
            # Calculate divergences
            divergences[f"{period}d"] = {
                "btc_return": float(btc_return),
                "sol_return": float(sol_return),
                "bonk_return": float(bonk_return),
                "btc_sol_divergence": float(btc_return - sol_return),
                "btc_bonk_divergence": float(btc_return - bonk_return),
                "sol_bonk_divergence": float(sol_return - bonk_return)
            }
        
        # Identify significant divergences
        significant_divergences = []
        
        if "7d" in divergences:
            # BTC vs SOL divergence
            if abs(divergences["7d"]["btc_sol_divergence"]) > 10:
                significant_divergences.append({
                    "pair": "BTC_SOL",
                    "divergence": divergences["7d"]["btc_sol_divergence"],
                    "leader": "BTC" if divergences["7d"]["btc_sol_divergence"] > 0 else "SOL"
                })
            
            # BTC vs BONK divergence
            if abs(divergences["7d"]["btc_bonk_divergence"]) > 15:
                significant_divergences.append({
                    "pair": "BTC_BONK",
                    "divergence": divergences["7d"]["btc_bonk_divergence"],
                    "leader": "BTC" if divergences["7d"]["btc_bonk_divergence"] > 0 else "BONK"
                })
            
            # SOL vs BONK divergence
            if abs(divergences["7d"]["sol_bonk_divergence"]) > 12:
                significant_divergences.append({
                    "pair": "SOL_BONK",
                    "divergence": divergences["7d"]["sol_bonk_divergence"],
                    "leader": "SOL" if divergences["7d"]["sol_bonk_divergence"] > 0 else "BONK"
                })
        
        return {
            "period_divergences": divergences,
            "significant_divergences": significant_divergences,
            "divergence_count": len(significant_divergences)
        }
    
    @performance_monitor()
    def analyze_lead_lag(self, 
                        btc_data: pd.DataFrame,
                        sol_data: pd.DataFrame,
                        bonk_data: pd.DataFrame,
                        max_lag: int = 24) -> Dict[str, Any]:
        """
        Analyze lead-lag relationships between assets.
        
        Args:
            btc_data: DataFrame with BTC price data
            sol_data: DataFrame with SOL price data
            bonk_data: DataFrame with BONK price data
            max_lag: Maximum lag periods to test
            
        Returns:
            Dictionary with lead-lag analysis
        """
        # Calculate returns
        btc_returns = btc_data["close"].pct_change().dropna()
        sol_returns = sol_data["close"].pct_change().dropna()
        bonk_returns = bonk_data["close"].pct_change().dropna()
        
        # Ensure equal length
        min_len = min(len(btc_returns), len(sol_returns), len(bonk_returns))
        btc_returns = btc_returns.iloc[-min_len:]
        sol_returns = sol_returns.iloc[-min_len:]
        bonk_returns = bonk_returns.iloc[-min_len:]
        
        lead_lag_results = {}
        
        # Test different lag combinations
        pairs = [
            ("BTC_SOL", btc_returns, sol_returns),
            ("BTC_BONK", btc_returns, bonk_returns),
            ("SOL_BONK", sol_returns, bonk_returns)
        ]
        
        for pair_name, series1, series2 in pairs:
            correlations = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Series1 leads series2
                    corr = series1.iloc[:lag].corr(series2.iloc[-lag:])
                elif lag > 0:
                    # Series2 leads series1
                    corr = series1.iloc[lag:].corr(series2.iloc[:-lag])
                else:
                    # No lag
                    corr = series1.corr(series2)
                
                correlations.append({
                    "lag": lag,
                    "correlation": float(corr) if not pd.isna(corr) else 0.0
                })
            
            # Find optimal lag
            max_corr_item = max(correlations, key=lambda x: abs(x["correlation"]))
            
            lead_lag_results[pair_name] = {
                "optimal_lag": max_corr_item["lag"],
                "max_correlation": max_corr_item["correlation"],
                "correlations": correlations,
                "leader": pair_name.split("_")[0] if max_corr_item["lag"] < 0 else pair_name.split("_")[1],
                "lag_hours": abs(max_corr_item["lag"])
            }
        
        return lead_lag_results
    
    @performance_monitor()
    def calculate_beta_relationships(self, 
                                   btc_data: pd.DataFrame,
                                   sol_data: pd.DataFrame,
                                   bonk_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate beta relationships between assets.
        
        Args:
            btc_data: DataFrame with BTC price data
            sol_data: DataFrame with SOL price data
            bonk_data: DataFrame with BONK price data
            
        Returns:
            Dictionary with beta calculations
        """
        # Calculate returns
        btc_returns = btc_data["close"].pct_change().dropna()
        sol_returns = sol_data["close"].pct_change().dropna()
        bonk_returns = bonk_data["close"].pct_change().dropna()
        
        # Ensure equal length
        min_len = min(len(btc_returns), len(sol_returns), len(bonk_returns))
        btc_returns = btc_returns.iloc[-min_len:].values
        sol_returns = sol_returns.iloc[-min_len:].values
        bonk_returns = bonk_returns.iloc[-min_len:].values
        
        betas = {}
        
        # SOL beta to BTC
        if len(btc_returns) > 1:
            sol_btc_cov = np.cov(sol_returns, btc_returns)[0, 1]
            btc_var = np.var(btc_returns)
            sol_beta_to_btc = sol_btc_cov / btc_var if btc_var > 0 else 0
            
            # R-squared
            correlation = np.corrcoef(sol_returns, btc_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            betas["SOL_to_BTC"] = {
                "beta": float(sol_beta_to_btc),
                "r_squared": float(r_squared),
                "interpretation": self._interpret_beta(sol_beta_to_btc)
            }
        
        # BONK beta to BTC
        if len(btc_returns) > 1:
            bonk_btc_cov = np.cov(bonk_returns, btc_returns)[0, 1]
            bonk_beta_to_btc = bonk_btc_cov / btc_var if btc_var > 0 else 0
            
            # R-squared
            correlation = np.corrcoef(bonk_returns, btc_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            betas["BONK_to_BTC"] = {
                "beta": float(bonk_beta_to_btc),
                "r_squared": float(r_squared),
                "interpretation": self._interpret_beta(bonk_beta_to_btc)
            }
        
        # BONK beta to SOL
        if len(sol_returns) > 1:
            bonk_sol_cov = np.cov(bonk_returns, sol_returns)[0, 1]
            sol_var = np.var(sol_returns)
            bonk_beta_to_sol = bonk_sol_cov / sol_var if sol_var > 0 else 0
            
            # R-squared
            correlation = np.corrcoef(bonk_returns, sol_returns)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            betas["BONK_to_SOL"] = {
                "beta": float(bonk_beta_to_sol),
                "r_squared": float(r_squared),
                "interpretation": self._interpret_beta(bonk_beta_to_sol)
            }
        
        return betas
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value."""
        if beta > 1.5:
            return "high_beta"
        elif beta > 1.0:
            return "amplified"
        elif beta > 0.5:
            return "correlated"
        elif beta > 0:
            return "weakly_correlated"
        elif beta > -0.5:
            return "weakly_inverse"
        else:
            return "strongly_inverse"
    
    @performance_monitor()
    def generate_signal(self, 
                      correlation_breakdown: Dict[str, Any],
                      divergence_analysis: Dict[str, Any],
                      lead_lag_analysis: Dict[str, Any],
                      beta_relationships: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate combined correlation signal.
        
        Args:
            correlation_breakdown: Correlation breakdown analysis
            divergence_analysis: Divergence analysis results
            lead_lag_analysis: Lead-lag relationship analysis
            beta_relationships: Beta calculations
            
        Returns:
            Dictionary with combined signal information
        """
        # Initialize signal components
        signal_components = {}
        
        # 1. Correlation Breakdown Component (range: -0.3 to 0.3)
        breakdown_score = 0.0
        breakdown_count = sum(1 for v in correlation_breakdown.values() if v.get("breakdown", False))
        
        if breakdown_count > 0:
            # Breakdowns suggest regime change
            breakdown_score = -0.1 * breakdown_count
            
            # Check for specific patterns
            for pair, data in correlation_breakdown.items():
                if data.get("breakdown_type") == "positive_to_negative":
                    breakdown_score -= 0.05  # Bearish signal
                elif data.get("breakdown_type") == "negative_to_positive":
                    breakdown_score += 0.05  # Bullish signal
                elif data.get("breakdown_type") == "decorrelation":
                    breakdown_score -= 0.03  # Uncertainty
        
        signal_components["correlation_breakdown"] = max(-0.3, min(0.3, breakdown_score))
        
        # 2. Divergence Component (range: -0.3 to 0.3)
        divergence_score = 0.0
        
        if divergence_analysis["divergence_count"] > 0:
            for div in divergence_analysis["significant_divergences"]:
                if div["pair"] == "BTC_SOL":
                    # BTC leading SOL is normal, SOL leading BTC is significant
                    if div["leader"] == "SOL" and div["divergence"] < -10:
                        divergence_score += 0.1  # SOL strength
                    elif div["leader"] == "BTC" and div["divergence"] > 15:
                        divergence_score -= 0.05  # BTC overextended
                
                elif div["pair"] == "SOL_BONK":
                    # BONK typically follows SOL
                    if div["leader"] == "BONK" and div["divergence"] < -12:
                        divergence_score += 0.15  # BONK strength unusual
                    elif div["leader"] == "SOL" and div["divergence"] > 20:
                        divergence_score -= 0.05  # SOL overextended
        
        signal_components["divergence"] = max(-0.3, min(0.3, divergence_score))
        
        # 3. Lead-Lag Component (range: -0.2 to 0.2)
        leadlag_score = 0.0
        
        # Check if correlations are breaking down
        btc_sol_lag = lead_lag_analysis.get("BTC_SOL", {})
        if btc_sol_lag.get("leader") == "SOL" and abs(btc_sol_lag.get("max_correlation", 0)) > 0.6:
            leadlag_score += 0.1  # SOL leading is bullish for alts
        
        sol_bonk_lag = lead_lag_analysis.get("SOL_BONK", {})
        if sol_bonk_lag.get("leader") == "BONK" and abs(sol_bonk_lag.get("max_correlation", 0)) > 0.5:
            leadlag_score += 0.1  # BONK leading is very bullish
        
        signal_components["lead_lag"] = max(-0.2, min(0.2, leadlag_score))
        
        # 4. Beta Component (range: -0.2 to 0.2)
        beta_score = 0.0
        
        # High beta suggests risk-on environment
        sol_beta = beta_relationships.get("SOL_to_BTC", {}).get("beta", 1.0)
        bonk_beta = beta_relationships.get("BONK_to_SOL", {}).get("beta", 1.0)
        
        if sol_beta > 1.5:
            beta_score += 0.1  # High beta environment
        elif sol_beta < 0.5:
            beta_score -= 0.1  # Low beta, risk-off
        
        if bonk_beta > 2.0:
            beta_score += 0.1  # Very high beta, strong risk-on
        elif bonk_beta < 0:
            beta_score -= 0.1  # Negative beta, decorrelation
        
        signal_components["beta"] = max(-0.2, min(0.2, beta_score))
        
        # Calculate final score
        final_score = sum(signal_components.values())
        
        # Adjust for market regime
        # If many correlations are breaking down, reduce signal confidence
        if breakdown_count >= 2:
            final_score *= 0.7
        
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
            "confidence": min(1.0, abs(final_score) * 1.5),
            "regime": self._determine_regime(correlation_breakdown, beta_relationships)
        }
    
    def _determine_regime(self, 
                         correlation_breakdown: Dict[str, Any],
                         beta_relationships: Dict[str, Any]) -> str:
        """Determine market regime based on correlations and betas."""
        breakdown_count = sum(1 for v in correlation_breakdown.values() if v.get("breakdown", False))
        avg_beta = np.mean([v.get("beta", 1.0) for v in beta_relationships.values()])
        
        if breakdown_count >= 2:
            return "decorrelated"
        elif avg_beta > 1.5:
            return "high_beta_risk_on"
        elif avg_beta < 0.5:
            return "low_beta_risk_off"
        else:
            return "normal_correlation"
    
    @performance_monitor()
    def analyze(self, 
               btc_data: pd.DataFrame,
               sol_data: pd.DataFrame,
               bonk_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete correlation analysis.
        
        Args:
            btc_data: DataFrame with BTC price data
            sol_data: DataFrame with SOL price data
            bonk_data: DataFrame with BONK price data
            
        Returns:
            Dictionary with complete analysis results
        """
        # Calculate correlations
        correlations_short = self.calculate_rolling_correlations(
            btc_data, sol_data, bonk_data, self.window_short
        )
        correlations_long = self.calculate_rolling_correlations(
            btc_data, sol_data, bonk_data, self.window_long
        )
        
        # Detect correlation breakdowns
        correlation_breakdown = self.detect_correlation_breakdown(
            correlations_short, correlations_long
        )
        
        # Analyze divergences
        divergence_analysis = self.analyze_divergence(
            btc_data, sol_data, bonk_data
        )
        
        # Analyze lead-lag relationships
        lead_lag_analysis = self.analyze_lead_lag(
            btc_data, sol_data, bonk_data
        )
        
        # Calculate beta relationships
        beta_relationships = self.calculate_beta_relationships(
            btc_data, sol_data, bonk_data
        )
        
        # Generate signal
        signal = self.generate_signal(
            correlation_breakdown,
            divergence_analysis,
            lead_lag_analysis,
            beta_relationships
        )
        
        # Get current correlations
        current_correlations = {
            pair: {
                "short": float(correlations_short[pair].iloc[-1]) if len(correlations_short[pair]) > 0 and not pd.isna(correlations_short[pair].iloc[-1]) else None,
                "long": float(correlations_long[pair].iloc[-1]) if len(correlations_long[pair]) > 0 and not pd.isna(correlations_long[pair].iloc[-1]) else None
            }
            for pair in ["btc_sol", "btc_bonk", "sol_bonk"]
        }
        
        # Combine all results
        result = {
            "current_correlations": current_correlations,
            "correlation_breakdown": correlation_breakdown,
            "divergence_analysis": divergence_analysis,
            "lead_lag_analysis": lead_lag_analysis,
            "beta_relationships": beta_relationships,
            "signal": signal["signal"],
            "score": signal["score"],
            "components": signal["components"],
            "confidence": signal["confidence"],
            "regime": signal["regime"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result


# Helper functions
def calculate_correlations(btc_data: pd.DataFrame,
                         sol_data: pd.DataFrame,
                         bonk_data: pd.DataFrame,
                         window: int = 24) -> Dict[str, float]:
    """
    Standalone function to calculate current correlations.
    
    Args:
        btc_data: DataFrame with BTC price data
        sol_data: DataFrame with SOL price data
        bonk_data: DataFrame with BONK price data
        window: Window for correlation calculation
        
    Returns:
        Dictionary with current correlations
    """
    analyzer = CorrelationsAnalyzer()
    correlations = analyzer.calculate_rolling_correlations(
        btc_data, sol_data, bonk_data, window
    )
    
    return {
        pair: float(corr.iloc[-1]) if len(corr) > 0 and not pd.isna(corr.iloc[-1]) else None
        for pair, corr in correlations.items()
    }


def analyze_correlations(btc_data: pd.DataFrame,
                        sol_data: pd.DataFrame,
                        bonk_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Standalone function to analyze correlations between assets.
    
    Args:
        btc_data: DataFrame with BTC price data
        sol_data: DataFrame with SOL price data
        bonk_data: DataFrame with BONK price data
        
    Returns:
        Dictionary with complete correlation analysis
    """
    analyzer = CorrelationsAnalyzer()
    return analyzer.analyze(btc_data, sol_data, bonk_data)


# Test function
def test_correlations_analyzer():
    """Test the Correlations analyzer."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate sample data with correlations
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='H')
    
    # Generate BTC data (base asset)
    btc_returns = np.random.normal(0.001, 0.02, 500)
    btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
    btc_data = pd.DataFrame({
        "date": dates,
        "close": btc_prices
    })
    
    # Generate SOL data (correlated with BTC, beta ~1.5)
    sol_base_returns = btc_returns * 1.5 + np.random.normal(0, 0.01, 500)
    # Add decorrelation period
    sol_base_returns[300:350] = np.random.normal(0.002, 0.03, 50)
    sol_prices = 100 * np.exp(np.cumsum(sol_base_returns))
    sol_data = pd.DataFrame({
        "date": dates,
        "close": sol_prices
    })
    
    # Generate BONK data (correlated with SOL, beta ~2.0)
    bonk_base_returns = sol_base_returns * 2.0 + np.random.normal(0, 0.02, 500)
    # Add leading period
    bonk_base_returns[400:450] = np.random.normal(0.005, 0.04, 50)
    bonk_prices = 0.00001 * np.exp(np.cumsum(bonk_base_returns))
    bonk_data = pd.DataFrame({
        "date": dates,
        "close": bonk_prices
    })
    
    # Create analyzer
    analyzer = CorrelationsAnalyzer()
    
    # Analyze data
    result = analyzer.analyze(btc_data, sol_data, bonk_data)
    
    # Print results
    print(f"Signal: {result['signal']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Regime: {result['regime']}")
    
    print("\nSignal Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.3f}")
    
    print("\nCurrent Correlations:")
    for pair, corrs in result['current_correlations'].items():
        print(f"  {pair}: Short={corrs['short']:.3f}, Long={corrs['long']:.3f}")
    
    print("\nCorrelation Breakdowns:")
    for pair, breakdown in result['correlation_breakdown'].items():
        if breakdown['breakdown']:
            print(f"  {pair}: {breakdown['breakdown_type']} (divergence={breakdown['divergence']:.3f})")
    
    print("\nBeta Relationships:")
    for pair, beta_data in result['beta_relationships'].items():
        print(f"  {pair}: β={beta_data['beta']:.2f}, R²={beta_data['r_squared']:.3f}, {beta_data['interpretation']}")
    
    print("\nLead-Lag Analysis:")
    for pair, lag_data in result['lead_lag_analysis'].items():
        print(f"  {pair}: {lag_data['leader']} leads by {lag_data['lag_hours']}h (corr={lag_data['max_correlation']:.3f})")
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Price Series (normalized)
    plt.subplot(3, 1, 1)
    btc_norm = btc_data["close"] / btc_data["close"].iloc[0]
    sol_norm = sol_data["close"] / sol_data["close"].iloc[0]
    bonk_norm = bonk_data["close"] / bonk_data["close"].iloc[0]
    
    plt.plot(dates, btc_norm, 'b-', label='BTC', linewidth=2)
    plt.plot(dates, sol_norm, 'g-', label='SOL', linewidth=2)
    plt.plot(dates, bonk_norm, 'r-', label='BONK', linewidth=2)
    plt.title("Normalized Price Series")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Correlations
    plt.subplot(3, 1, 2)
    
    # Calculate correlations for plotting
    window = 24
    btc_returns = btc_data["close"].pct_change()
    sol_returns = sol_data["close"].pct_change()
    bonk_returns = bonk_data["close"].pct_change()
    
    btc_sol_corr = btc_returns.rolling(window).corr(sol_returns)
    btc_bonk_corr = btc_returns.rolling(window).corr(bonk_returns)
    sol_bonk_corr = sol_returns.rolling(window).corr(bonk_returns)
    
    plt.plot(dates, btc_sol_corr, 'b-', label='BTC-SOL', linewidth=2)
    plt.plot(dates, btc_bonk_corr, 'g-', label='BTC-BONK', linewidth=2)
    plt.plot(dates, sol_bonk_corr, 'r-', label='SOL-BONK', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title(f"Rolling {window}h Correlations")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.ylim(-1, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Lead-Lag Correlations
    plt.subplot(3, 1, 3)
    
    # Plot BTC-SOL lead-lag
    btc_sol_lags = result['lead_lag_analysis']['BTC_SOL']['correlations']
    lags = [item['lag'] for item in btc_sol_lags]
    corrs = [item['correlation'] for item in btc_sol_lags]
    
    plt.plot(lags, corrs, 'b-', linewidth=2)
    optimal_lag = result['lead_lag_analysis']['BTC_SOL']['optimal_lag']
    optimal_corr = result['lead_lag_analysis']['BTC_SOL']['max_correlation']
    plt.scatter([optimal_lag], [optimal_corr], color='red', s=100, zorder=5)
    
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.title("BTC-SOL Lead-Lag Analysis")
    plt.xlabel("Lag (hours, negative = BTC leads)")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_correlations_analyzer()
