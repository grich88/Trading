"""
Tests for the Funding Rate & Bias Tracking Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.funding_rate_analyzer import (
    FundingRateAnalyzer,
    detect_funding_anomalies,
    analyze_funding_rates
)


class TestFundingRateAnalyzer(unittest.TestCase):
    """Test cases for the FundingRateAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = FundingRateAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.periods = 100
        
        # Generate timestamps
        self.base_timestamp = datetime.now() - timedelta(days=self.periods)
        self.timestamps = np.array([self.base_timestamp + timedelta(hours=8*i) for i in range(self.periods)])
        
        # Generate price data with trend and noise
        self.prices = np.cumsum(np.random.normal(0, 1, self.periods)) + 100
        
        # Generate funding rate data with some patterns
        # First half: mostly negative funding
        # Second half: mostly positive funding
        base_funding = np.random.normal(0, 0.001, self.periods)
        trend_funding = np.concatenate([
            np.ones(self.periods//2) * -0.001,
            np.ones(self.periods - self.periods//2) * 0.001
        ])
        
        # Add some spikes
        for i in range(5):
            spike_idx = np.random.randint(0, self.periods)
            base_funding[spike_idx] = np.random.choice([0.005, -0.005])
        
        self.funding_rates = base_funding + trend_funding
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "timestamp": self.timestamps,
            "close": self.prices,
            "funding_rate": self.funding_rates
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = FundingRateAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = FundingRateAnalyzer(
            lookback_period=15,
            anomaly_threshold=3.0,
            signal_threshold=0.4,
            bias_threshold=15.0
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.anomaly_threshold, 3.0)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.bias_threshold, 15.0)
    
    def test_detect_funding_anomalies(self):
        """Test funding anomaly detection."""
        # Detect anomalies
        anomalies = self.analyzer.detect_funding_anomalies(self.funding_rates, self.timestamps)
        
        # Check result structure
        self.assertIn("mean_rate", anomalies)
        self.assertIn("std_rate", anomalies)
        self.assertIn("max_rate", anomalies)
        self.assertIn("min_rate", anomalies)
        self.assertIn("current_rate", anomalies)
        self.assertIn("current_z_score", anomalies)
        self.assertIn("is_anomaly", anomalies)
        self.assertIn("is_extreme_anomaly", anomalies)
        self.assertIn("direction", anomalies)
        self.assertIn("historical_anomalies", anomalies)
        
        # Check result types
        self.assertIsInstance(anomalies["mean_rate"], float)
        self.assertIsInstance(anomalies["std_rate"], float)
        self.assertIsInstance(anomalies["max_rate"], float)
        self.assertIsInstance(anomalies["min_rate"], float)
        self.assertIsInstance(anomalies["current_rate"], float)
        self.assertIsInstance(anomalies["current_z_score"], float)
        self.assertIsInstance(anomalies["is_anomaly"], bool)
        self.assertIsInstance(anomalies["is_extreme_anomaly"], bool)
        self.assertIn(anomalies["direction"], ["positive", "negative", "neutral"])
        self.assertIsInstance(anomalies["historical_anomalies"], list)
        
        # Check historical anomalies structure
        if anomalies["historical_anomalies"]:
            anomaly = anomalies["historical_anomalies"][0]
            self.assertIn("timestamp", anomaly)
            self.assertIn("funding_rate", anomaly)
            self.assertIn("z_score", anomaly)
            self.assertIn("is_extreme", anomaly)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.detect_funding_anomalies(self.funding_rates[:50], self.timestamps)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_funding_anomalies(
                np.array([0.001, 0.002]), 
                np.array([self.base_timestamp, self.base_timestamp + timedelta(hours=8)])
            )
        
        # Test standalone function
        standalone_anomalies = detect_funding_anomalies(self.funding_rates, self.timestamps)
        self.assertEqual(standalone_anomalies["current_rate"], anomalies["current_rate"])
    
    def test_analyze_funding_bias(self):
        """Test funding bias analysis."""
        # Analyze bias
        bias = self.analyzer.analyze_funding_bias(self.funding_rates)
        
        # Check result structure
        self.assertIn("positive_count", bias)
        self.assertIn("negative_count", bias)
        self.assertIn("neutral_count", bias)
        self.assertIn("positive_percentage", bias)
        self.assertIn("negative_percentage", bias)
        self.assertIn("positive_avg", bias)
        self.assertIn("negative_avg", bias)
        self.assertIn("net_bias", bias)
        self.assertIn("bias_direction", bias)
        self.assertIn("bias_strength", bias)
        
        # Check result types
        self.assertIsInstance(bias["positive_count"], int)
        self.assertIsInstance(bias["negative_count"], int)
        self.assertIsInstance(bias["neutral_count"], int)
        self.assertIsInstance(bias["positive_percentage"], float)
        self.assertIsInstance(bias["negative_percentage"], float)
        self.assertIsInstance(bias["positive_avg"], float)
        self.assertIsInstance(bias["negative_avg"], float)
        self.assertIsInstance(bias["net_bias"], float)
        self.assertIn(bias["bias_direction"], ["bullish", "bearish", "neutral"])
        self.assertIsInstance(bias["bias_strength"], float)
        
        # Check calculations
        self.assertEqual(bias["positive_count"] + bias["negative_count"] + bias["neutral_count"], self.analyzer.lookback_period)
        self.assertAlmostEqual(bias["positive_percentage"] + bias["negative_percentage"] + (bias["neutral_count"] / self.analyzer.lookback_period * 100), 100.0)
        
        # Test with custom lookback
        custom_bias = self.analyzer.analyze_funding_bias(self.funding_rates, lookback=10)
        self.assertEqual(custom_bias["positive_count"] + custom_bias["negative_count"] + custom_bias["neutral_count"], 10)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_funding_bias(np.array([0.001, 0.002]))
        
        # Test extreme cases
        all_positive = np.ones(30) * 0.001
        all_positive_bias = self.analyzer.analyze_funding_bias(all_positive, lookback=30)
        self.assertEqual(all_positive_bias["bias_direction"], "bullish")
        self.assertEqual(all_positive_bias["positive_count"], 30)
        self.assertEqual(all_positive_bias["negative_count"], 0)
        
        all_negative = np.ones(30) * -0.001
        all_negative_bias = self.analyzer.analyze_funding_bias(all_negative, lookback=30)
        self.assertEqual(all_negative_bias["bias_direction"], "bearish")
        self.assertEqual(all_negative_bias["positive_count"], 0)
        self.assertEqual(all_negative_bias["negative_count"], 30)
    
    def test_analyze_funding_patterns(self):
        """Test funding pattern analysis."""
        # Analyze patterns
        patterns = self.analyzer.analyze_funding_patterns(self.funding_rates, self.prices)
        
        # Check result structure
        self.assertIn("correlation", patterns)
        self.assertIn("avg_momentum", patterns)
        self.assertIn("sign_changes", patterns)
        self.assertIn("peaks", patterns)
        self.assertIn("troughs", patterns)
        self.assertIn("convergence_pattern", patterns)
        self.assertIn("patterns", patterns)
        
        # Check result types
        self.assertIsInstance(patterns["correlation"], float)
        self.assertIsInstance(patterns["avg_momentum"], float)
        self.assertIsInstance(patterns["sign_changes"], int)
        self.assertIsInstance(patterns["peaks"], int)
        self.assertIsInstance(patterns["troughs"], int)
        self.assertIn(patterns["convergence_pattern"], ["converging", "diverging", "neutral"])
        self.assertIsInstance(patterns["patterns"], list)
        
        # Check patterns structure
        if patterns["patterns"]:
            pattern = patterns["patterns"][0]
            self.assertIn("name", pattern)
            self.assertIn("description", pattern)
            self.assertIn("strength", pattern)
        
        # Test with custom lookback
        custom_patterns = self.analyzer.analyze_funding_patterns(self.funding_rates, self.prices, lookback=10)
        self.assertIsInstance(custom_patterns["correlation"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.analyze_funding_patterns(self.funding_rates[:50], self.prices)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_funding_patterns(
                np.array([0.001, 0.002]), 
                np.array([100.0, 101.0])
            )
    
    def test_analyze_funding_impact(self):
        """Test funding impact analysis."""
        # Analyze impact
        impact = self.analyzer.analyze_funding_impact(self.funding_rates, self.prices)
        
        # Check result structure
        self.assertIn("avg_after_positive", impact)
        self.assertIn("avg_after_negative", impact)
        self.assertIn("avg_after_extreme_positive", impact)
        self.assertIn("avg_after_extreme_negative", impact)
        self.assertIn("significant_impact", impact)
        self.assertIn("impact_pattern", impact)
        self.assertIn("predictive_correlation", impact)
        self.assertIn("positive_samples", impact)
        self.assertIn("negative_samples", impact)
        self.assertIn("extreme_positive_samples", impact)
        self.assertIn("extreme_negative_samples", impact)
        
        # Check result types
        self.assertIsInstance(impact["avg_after_positive"], float)
        self.assertIsInstance(impact["avg_after_negative"], float)
        self.assertIsInstance(impact["avg_after_extreme_positive"], float)
        self.assertIsInstance(impact["avg_after_extreme_negative"], float)
        self.assertIsInstance(impact["significant_impact"], bool)
        self.assertIn(impact["impact_pattern"], ["mean_reverting", "trend_following", "neutral"])
        self.assertIsInstance(impact["predictive_correlation"], float)
        self.assertIsInstance(impact["positive_samples"], int)
        self.assertIsInstance(impact["negative_samples"], int)
        self.assertIsInstance(impact["extreme_positive_samples"], int)
        self.assertIsInstance(impact["extreme_negative_samples"], int)
        
        # Test with custom lookback
        custom_impact = self.analyzer.analyze_funding_impact(self.funding_rates, self.prices, lookback=10)
        self.assertIsInstance(custom_impact["avg_after_positive"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.analyze_funding_impact(self.funding_rates[:50], self.prices)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_funding_impact(
                np.array([0.001] * 10), 
                np.array([100.0] * 10)
            )
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        anomaly_analysis = {
            "mean_rate": 0.0001,
            "std_rate": 0.001,
            "max_rate": 0.005,
            "min_rate": -0.005,
            "current_rate": 0.003,
            "current_z_score": 2.9,
            "is_anomaly": True,
            "is_extreme_anomaly": False,
            "direction": "positive",
            "historical_anomalies": []
        }
        
        bias_analysis = {
            "positive_count": 15,
            "negative_count": 5,
            "neutral_count": 0,
            "positive_percentage": 75.0,
            "negative_percentage": 25.0,
            "positive_avg": 0.002,
            "negative_avg": -0.001,
            "net_bias": 0.0015,
            "bias_direction": "bullish",
            "bias_strength": 0.5
        }
        
        pattern_analysis = {
            "correlation": 0.3,
            "avg_momentum": 0.0001,
            "sign_changes": 2,
            "peaks": 3,
            "troughs": 2,
            "convergence_pattern": "neutral",
            "patterns": [
                {
                    "name": "sustained_positive_funding",
                    "description": "Sustained positive funding rates (longs paying shorts)",
                    "strength": 0.8
                }
            ]
        }
        
        impact_analysis = {
            "avg_after_positive": -1.2,
            "avg_after_negative": 0.8,
            "avg_after_extreme_positive": -2.5,
            "avg_after_extreme_negative": 1.5,
            "significant_impact": True,
            "impact_pattern": "mean_reverting",
            "predictive_correlation": -0.4,
            "positive_samples": 10,
            "negative_samples": 5,
            "extreme_positive_samples": 3,
            "extreme_negative_samples": 2
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            anomaly_analysis,
            bias_analysis,
            pattern_analysis,
            impact_analysis
        )
        
        # Check result structure
        self.assertIn("signal", signal)
        self.assertIn("score", signal)
        self.assertIn("components", signal)
        self.assertIn("confidence", signal)
        
        # Check result types
        self.assertIsInstance(signal["signal"], str)
        self.assertIsInstance(signal["score"], float)
        self.assertIsInstance(signal["components"], dict)
        self.assertIsInstance(signal["confidence"], float)
        
        # Check component structure
        self.assertIn("anomaly", signal["components"])
        self.assertIn("bias", signal["components"])
        self.assertIn("pattern", signal["components"])
        self.assertIn("impact", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish signals
        bearish_anomaly = anomaly_analysis.copy()
        bearish_anomaly["direction"] = "negative"
        bearish_anomaly["current_rate"] = -0.003
        
        bearish_bias = bias_analysis.copy()
        bearish_bias["bias_direction"] = "bearish"
        bearish_bias["positive_percentage"] = 25.0
        bearish_bias["negative_percentage"] = 75.0
        
        bearish_pattern = pattern_analysis.copy()
        bearish_pattern["patterns"] = [
            {
                "name": "sustained_negative_funding",
                "description": "Sustained negative funding rates (shorts paying longs)",
                "strength": 0.8
            }
        ]
        
        bearish_impact = impact_analysis.copy()
        bearish_impact["impact_pattern"] = "trend_following"
        
        bearish_signal = self.analyzer.generate_signal(
            bearish_anomaly,
            bearish_bias,
            bearish_pattern,
            bearish_impact
        )
        
        # In this case, negative funding is actually bullish (shorts paying longs)
        self.assertGreater(bearish_signal["score"], 0)
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("metrics", result)
        self.assertIn("anomaly_analysis", result)
        self.assertIn("bias_analysis", result)
        self.assertIn("pattern_analysis", result)
        self.assertIn("impact_analysis", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check metrics structure
        self.assertIn("funding_rates", result["metrics"])
        self.assertIn("current_rate", result["metrics"])
        self.assertIn("mean_rate", result["metrics"])
        self.assertIn("std_rate", result["metrics"])
        
        # Check result types
        self.assertIsInstance(result["metrics"]["funding_rates"], list)
        self.assertIsInstance(result["metrics"]["current_rate"], float)
        self.assertIsInstance(result["metrics"]["mean_rate"], float)
        self.assertIsInstance(result["metrics"]["std_rate"], float)
        self.assertIsInstance(result["anomaly_analysis"], dict)
        self.assertIsInstance(result["bias_analysis"], dict)
        self.assertIsInstance(result["pattern_analysis"], dict)
        self.assertIsInstance(result["impact_analysis"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["metrics"]["funding_rates"]), len(self.funding_rates))
        
        # Test standalone function
        standalone_result = analyze_funding_rates(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        df_missing = pd.DataFrame({
            "close": self.prices,
            "funding_rate": self.funding_rates
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(df_missing)
        
        # Test with insufficient data
        short_df = pd.DataFrame({
            "timestamp": self.timestamps[:5],
            "close": self.prices[:5],
            "funding_rate": self.funding_rates[:5]
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(short_df)


if __name__ == "__main__":
    unittest.main()
