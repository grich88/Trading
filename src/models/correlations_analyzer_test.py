"""
Tests for the Correlations Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.correlations_analyzer import (
    CorrelationsAnalyzer,
    calculate_correlations,
    analyze_correlations
)


class TestCorrelationsAnalyzer(unittest.TestCase):
    """Test cases for the CorrelationsAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = CorrelationsAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='H')
        
        # Generate BTC data (base asset)
        btc_returns = np.random.normal(0.001, 0.02, 500)
        btc_prices = 40000 * np.exp(np.cumsum(btc_returns))
        self.btc_data = pd.DataFrame({
            "date": dates,
            "close": btc_prices
        })
        
        # Generate SOL data (correlated with BTC, beta ~1.5)
        sol_base_returns = btc_returns * 1.5 + np.random.normal(0, 0.01, 500)
        # Add decorrelation period
        sol_base_returns[300:350] = np.random.normal(0.002, 0.03, 50)
        sol_prices = 100 * np.exp(np.cumsum(sol_base_returns))
        self.sol_data = pd.DataFrame({
            "date": dates,
            "close": sol_prices
        })
        
        # Generate BONK data (correlated with SOL, beta ~2.0)
        bonk_base_returns = sol_base_returns * 2.0 + np.random.normal(0, 0.02, 500)
        # Add leading period
        bonk_base_returns[400:450] = np.random.normal(0.005, 0.04, 50)
        bonk_prices = 0.00001 * np.exp(np.cumsum(bonk_base_returns))
        self.bonk_data = pd.DataFrame({
            "date": dates,
            "close": bonk_prices
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = CorrelationsAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = CorrelationsAnalyzer(
            lookback_period=30,
            signal_threshold=0.4,
            window_short=12,
            window_long=48
        )
        self.assertEqual(custom_analyzer.lookback_period, 30)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.window_short, 12)
        self.assertEqual(custom_analyzer.window_long, 48)
    
    def test_calculate_rolling_correlations(self):
        """Test rolling correlation calculation."""
        # Calculate correlations
        window = 24
        correlations = self.analyzer.calculate_rolling_correlations(
            self.btc_data, self.sol_data, self.bonk_data, window
        )
        
        # Check result structure
        self.assertIn("btc_sol", correlations)
        self.assertIn("btc_bonk", correlations)
        self.assertIn("sol_bonk", correlations)
        
        # Check result types
        for pair, corr in correlations.items():
            self.assertIsInstance(corr, pd.Series)
            self.assertEqual(len(corr), len(self.btc_data))
        
        # Check correlation values are in valid range
        for pair, corr in correlations.items():
            valid_corr = corr.dropna()
            if len(valid_corr) > 0:
                self.assertTrue(all(valid_corr >= -1))
                self.assertTrue(all(valid_corr <= 1))
        
        # Test with missing columns
        bad_df = self.btc_data.drop(columns=["close"])
        with self.assertRaises(ModelError):
            self.analyzer.calculate_rolling_correlations(
                bad_df, self.sol_data, self.bonk_data, window
            )
        
        # Test standalone function
        standalone_result = calculate_correlations(
            self.btc_data, self.sol_data, self.bonk_data, window
        )
        self.assertIsInstance(standalone_result, dict)
        for pair in ["btc_sol", "btc_bonk", "sol_bonk"]:
            self.assertIn(pair, standalone_result)
    
    def test_detect_correlation_breakdown(self):
        """Test correlation breakdown detection."""
        # Calculate correlations
        correlations_short = self.analyzer.calculate_rolling_correlations(
            self.btc_data, self.sol_data, self.bonk_data, self.analyzer.window_short
        )
        correlations_long = self.analyzer.calculate_rolling_correlations(
            self.btc_data, self.sol_data, self.bonk_data, self.analyzer.window_long
        )
        
        # Detect breakdowns
        breakdowns = self.analyzer.detect_correlation_breakdown(
            correlations_short, correlations_long
        )
        
        # Check result structure
        for pair in ["btc_sol", "btc_bonk", "sol_bonk"]:
            self.assertIn(pair, breakdowns)
            breakdown_data = breakdowns[pair]
            
            self.assertIn("short_correlation", breakdown_data)
            self.assertIn("long_correlation", breakdown_data)
            self.assertIn("divergence", breakdown_data)
            self.assertIn("breakdown", breakdown_data)
            self.assertIn("breakdown_type", breakdown_data)
        
        # Test specific breakdown scenarios
        # Create artificial breakdown data
        short_corr = pd.Series([-0.3] * 100)
        long_corr = pd.Series([0.6] * 100)
        
        test_breakdowns = self.analyzer.detect_correlation_breakdown(
            {"btc_sol": short_corr},
            {"btc_sol": long_corr}
        )
        
        # Should detect positive to negative breakdown
        self.assertTrue(test_breakdowns["btc_sol"]["breakdown"])
        self.assertEqual(test_breakdowns["btc_sol"]["breakdown_type"], "positive_to_negative")
    
    def test_analyze_divergence(self):
        """Test divergence analysis."""
        # Analyze divergence
        divergence = self.analyzer.analyze_divergence(
            self.btc_data, self.sol_data, self.bonk_data
        )
        
        # Check result structure
        self.assertIn("period_divergences", divergence)
        self.assertIn("significant_divergences", divergence)
        self.assertIn("divergence_count", divergence)
        
        # Check period divergences
        for period in ["1d", "7d", "30d"]:
            if period in divergence["period_divergences"]:
                period_data = divergence["period_divergences"][period]
                self.assertIn("btc_return", period_data)
                self.assertIn("sol_return", period_data)
                self.assertIn("bonk_return", period_data)
                self.assertIn("btc_sol_divergence", period_data)
                self.assertIn("btc_bonk_divergence", period_data)
                self.assertIn("sol_bonk_divergence", period_data)
        
        # Check significant divergences
        self.assertIsInstance(divergence["significant_divergences"], list)
        for div in divergence["significant_divergences"]:
            self.assertIn("pair", div)
            self.assertIn("divergence", div)
            self.assertIn("leader", div)
            self.assertIn(div["pair"], ["BTC_SOL", "BTC_BONK", "SOL_BONK"])
        
        # Test with insufficient data
        short_btc = self.btc_data.iloc[-5:]
        short_sol = self.sol_data.iloc[-5:]
        short_bonk = self.bonk_data.iloc[-5:]
        
        short_divergence = self.analyzer.analyze_divergence(
            short_btc, short_sol, short_bonk
        )
        # Should still work but with limited periods
        self.assertIn("1d", short_divergence["period_divergences"])
        self.assertNotIn("30d", short_divergence["period_divergences"])
    
    def test_analyze_lead_lag(self):
        """Test lead-lag analysis."""
        # Analyze lead-lag relationships
        lead_lag = self.analyzer.analyze_lead_lag(
            self.btc_data, self.sol_data, self.bonk_data, max_lag=12
        )
        
        # Check result structure
        for pair in ["BTC_SOL", "BTC_BONK", "SOL_BONK"]:
            self.assertIn(pair, lead_lag)
            pair_data = lead_lag[pair]
            
            self.assertIn("optimal_lag", pair_data)
            self.assertIn("max_correlation", pair_data)
            self.assertIn("correlations", pair_data)
            self.assertIn("leader", pair_data)
            self.assertIn("lag_hours", pair_data)
        
        # Check correlations list
        for pair, data in lead_lag.items():
            correlations = data["correlations"]
            self.assertIsInstance(correlations, list)
            self.assertTrue(len(correlations) > 0)
            
            for corr_item in correlations:
                self.assertIn("lag", corr_item)
                self.assertIn("correlation", corr_item)
        
        # Check optimal lag is within tested range
        for pair, data in lead_lag.items():
            self.assertTrue(abs(data["optimal_lag"]) <= 12)
            self.assertEqual(data["lag_hours"], abs(data["optimal_lag"]))
    
    def test_calculate_beta_relationships(self):
        """Test beta calculation."""
        # Calculate betas
        betas = self.analyzer.calculate_beta_relationships(
            self.btc_data, self.sol_data, self.bonk_data
        )
        
        # Check result structure
        expected_pairs = ["SOL_to_BTC", "BONK_to_BTC", "BONK_to_SOL"]
        for pair in expected_pairs:
            self.assertIn(pair, betas)
            beta_data = betas[pair]
            
            self.assertIn("beta", beta_data)
            self.assertIn("r_squared", beta_data)
            self.assertIn("interpretation", beta_data)
        
        # Check value ranges
        for pair, data in betas.items():
            self.assertIsInstance(data["beta"], float)
            self.assertIsInstance(data["r_squared"], float)
            self.assertTrue(0 <= data["r_squared"] <= 1)
            self.assertIn(data["interpretation"], [
                "high_beta", "amplified", "correlated", 
                "weakly_correlated", "weakly_inverse", "strongly_inverse"
            ])
        
        # Test with known beta relationship
        # Create perfectly correlated data with beta = 2
        test_btc = pd.DataFrame({
            "close": np.arange(100) + 100
        })
        test_sol = pd.DataFrame({
            "close": 2 * np.arange(100) + 50
        })
        test_bonk = pd.DataFrame({
            "close": 4 * np.arange(100) + 10
        })
        
        test_betas = self.analyzer.calculate_beta_relationships(
            test_btc, test_sol, test_bonk
        )
        
        # SOL should have beta ~2 to BTC
        self.assertAlmostEqual(test_betas["SOL_to_BTC"]["beta"], 2.0, places=1)
        self.assertAlmostEqual(test_betas["SOL_to_BTC"]["r_squared"], 1.0, places=1)
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        correlation_breakdown = {
            "btc_sol": {
                "short_correlation": 0.2,
                "long_correlation": 0.7,
                "divergence": -0.5,
                "breakdown": True,
                "breakdown_type": "divergence"
            },
            "btc_bonk": {
                "short_correlation": 0.6,
                "long_correlation": 0.5,
                "divergence": 0.1,
                "breakdown": False,
                "breakdown_type": None
            },
            "sol_bonk": {
                "short_correlation": 0.8,
                "long_correlation": 0.7,
                "divergence": 0.1,
                "breakdown": False,
                "breakdown_type": None
            }
        }
        
        divergence_analysis = {
            "period_divergences": {
                "7d": {
                    "btc_return": 5.0,
                    "sol_return": 20.0,
                    "bonk_return": 10.0,
                    "btc_sol_divergence": -15.0,
                    "btc_bonk_divergence": -5.0,
                    "sol_bonk_divergence": 10.0
                }
            },
            "significant_divergences": [
                {
                    "pair": "BTC_SOL",
                    "divergence": -15.0,
                    "leader": "SOL"
                }
            ],
            "divergence_count": 1
        }
        
        lead_lag_analysis = {
            "BTC_SOL": {
                "optimal_lag": 2,
                "max_correlation": 0.7,
                "leader": "SOL",
                "lag_hours": 2
            },
            "SOL_BONK": {
                "optimal_lag": -1,
                "max_correlation": 0.8,
                "leader": "SOL",
                "lag_hours": 1
            }
        }
        
        beta_relationships = {
            "SOL_to_BTC": {
                "beta": 1.8,
                "r_squared": 0.7,
                "interpretation": "high_beta"
            },
            "BONK_to_SOL": {
                "beta": 2.5,
                "r_squared": 0.6,
                "interpretation": "high_beta"
            }
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            correlation_breakdown,
            divergence_analysis,
            lead_lag_analysis,
            beta_relationships
        )
        
        # Check result structure
        self.assertIn("signal", signal)
        self.assertIn("score", signal)
        self.assertIn("components", signal)
        self.assertIn("confidence", signal)
        self.assertIn("regime", signal)
        
        # Check result types
        self.assertIsInstance(signal["signal"], str)
        self.assertIsInstance(signal["score"], float)
        self.assertIsInstance(signal["components"], dict)
        self.assertIsInstance(signal["confidence"], float)
        self.assertIsInstance(signal["regime"], str)
        
        # Check component structure
        expected_components = [
            "correlation_breakdown", "divergence", "lead_lag", "beta"
        ]
        for component in expected_components:
            self.assertIn(component, signal["components"])
            self.assertIsInstance(signal["components"][component], float)
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Check regime
        self.assertIn(signal["regime"], [
            "decorrelated", "high_beta_risk_on", 
            "low_beta_risk_off", "normal_correlation"
        ])
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(
            self.btc_data, self.sol_data, self.bonk_data
        )
        
        # Check result structure
        self.assertIn("current_correlations", result)
        self.assertIn("correlation_breakdown", result)
        self.assertIn("divergence_analysis", result)
        self.assertIn("lead_lag_analysis", result)
        self.assertIn("beta_relationships", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("regime", result)
        self.assertIn("timestamp", result)
        
        # Check result types
        self.assertIsInstance(result["current_correlations"], dict)
        self.assertIsInstance(result["correlation_breakdown"], dict)
        self.assertIsInstance(result["divergence_analysis"], dict)
        self.assertIsInstance(result["lead_lag_analysis"], dict)
        self.assertIsInstance(result["beta_relationships"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["regime"], str)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check current correlations
        for pair in ["btc_sol", "btc_bonk", "sol_bonk"]:
            self.assertIn(pair, result["current_correlations"])
            self.assertIn("short", result["current_correlations"][pair])
            self.assertIn("long", result["current_correlations"][pair])
        
        # Test standalone function
        standalone_result = analyze_correlations(
            self.btc_data, self.sol_data, self.bonk_data
        )
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        bad_df = pd.DataFrame({"price": [100, 101, 102]})
        with self.assertRaises(ModelError):
            self.analyzer.calculate_rolling_correlations(
                bad_df, self.sol_data, self.bonk_data, 24
            )
        
        # Test with mismatched lengths (should handle gracefully)
        short_btc = self.btc_data.iloc[:100]
        result = self.analyzer.analyze(
            short_btc, self.sol_data, self.bonk_data
        )
        # Should still produce valid results
        self.assertIn("signal", result)
        self.assertIsInstance(result["score"], float)
    
    def test_regime_determination(self):
        """Test regime determination logic."""
        # Test decorrelated regime
        correlation_breakdown = {
            f"pair_{i}": {"breakdown": True} for i in range(3)
        }
        beta_relationships = {
            f"pair_{i}": {"beta": 1.0} for i in range(3)
        }
        
        regime = self.analyzer._determine_regime(
            correlation_breakdown, beta_relationships
        )
        self.assertEqual(regime, "decorrelated")
        
        # Test high beta regime
        correlation_breakdown = {
            f"pair_{i}": {"breakdown": False} for i in range(3)
        }
        beta_relationships = {
            f"pair_{i}": {"beta": 2.0} for i in range(3)
        }
        
        regime = self.analyzer._determine_regime(
            correlation_breakdown, beta_relationships
        )
        self.assertEqual(regime, "high_beta_risk_on")
        
        # Test low beta regime
        beta_relationships = {
            f"pair_{i}": {"beta": 0.3} for i in range(3)
        }
        
        regime = self.analyzer._determine_regime(
            correlation_breakdown, beta_relationships
        )
        self.assertEqual(regime, "low_beta_risk_off")


if __name__ == "__main__":
    unittest.main()
