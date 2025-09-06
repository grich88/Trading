"""
Tests for the Open Interest vs Price Divergence Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.open_interest_analyzer import (
    OpenInterestAnalyzer,
    analyze_open_interest,
    detect_oi_divergence
)


class TestOpenInterestAnalyzer(unittest.TestCase):
    """Test cases for the OpenInterestAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = OpenInterestAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.periods = 100
        
        # Generate price data with trend and noise
        self.prices = np.cumsum(np.random.normal(0, 1, self.periods)) + 100
        
        # Generate open interest data with some divergence from price
        # First half: OI follows price
        # Second half: OI diverges from price
        oi_first_half = self.prices[:self.periods//2] * (1 + np.random.normal(0, 0.05, self.periods//2))
        oi_second_half = self.prices[self.periods//2:] * (1 - np.random.normal(0, 0.05, self.periods - self.periods//2))
        self.open_interest = np.concatenate([oi_first_half, oi_second_half])
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "close": self.prices,
            "open_interest": self.open_interest
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = OpenInterestAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = OpenInterestAnalyzer(
            change_threshold=10.0,
            lookback_period=15,
            divergence_threshold=5.0,
            signal_threshold=0.4,
            trend_period=10
        )
        self.assertEqual(custom_analyzer.change_threshold, 10.0)
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.divergence_threshold, 5.0)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.trend_period, 10)
    
    def test_calculate_oi_metrics(self):
        """Test OI metrics calculation."""
        # Calculate metrics
        metrics = self.analyzer.calculate_oi_metrics(self.open_interest, self.prices)
        
        # Check result structure
        self.assertIn("oi_change", metrics)
        self.assertIn("oi_change_pct", metrics)
        self.assertIn("oi_roc", metrics)
        self.assertIn("price_change", metrics)
        self.assertIn("price_change_pct", metrics)
        self.assertIn("price_roc", metrics)
        self.assertIn("oi_price_ratio", metrics)
        self.assertIn("oi_price_ratio_change", metrics)
        
        # Check result shapes
        self.assertEqual(len(metrics["oi_change"]), len(self.open_interest))
        self.assertEqual(len(metrics["price_change"]), len(self.prices))
        
        # Check first change values
        self.assertEqual(metrics["oi_change"][0], 0)
        self.assertEqual(metrics["price_change"][0], 0)
        
        # Check that later values are calculated correctly
        self.assertNotEqual(metrics["oi_change"][1], 0)
        self.assertNotEqual(metrics["price_change"][1], 0)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.calculate_oi_metrics(self.open_interest[:50], self.prices)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.calculate_oi_metrics(np.array([1]), np.array([1]))
    
    def test_detect_divergence(self):
        """Test divergence detection."""
        # Detect divergence
        divergence = self.analyzer.detect_divergence(self.open_interest, self.prices)
        
        # Check result structure
        self.assertIn("bullish_divergence", divergence)
        self.assertIn("bearish_divergence", divergence)
        self.assertIn("extreme_divergence", divergence)
        self.assertIn("local_extrema_divergence", divergence)
        self.assertIn("price_trend", divergence)
        self.assertIn("oi_trend", divergence)
        self.assertIn("trend_divergence", divergence)
        self.assertIn("divergence_strength", divergence)
        
        # Check result types
        self.assertIsInstance(divergence["bullish_divergence"], bool)
        self.assertIsInstance(divergence["bearish_divergence"], bool)
        self.assertIsInstance(divergence["price_trend"], float)
        self.assertIsInstance(divergence["divergence_strength"], float)
        
        # Test with custom lookback
        custom_divergence = self.analyzer.detect_divergence(self.open_interest, self.prices, lookback=10)
        self.assertIsInstance(custom_divergence["bullish_divergence"], bool)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_divergence(self.open_interest[:5], self.prices[:5])
    
    def test_analyze_oi_trend(self):
        """Test OI trend analysis."""
        # Analyze trend
        trend = self.analyzer.analyze_oi_trend(self.open_interest)
        
        # Check result structure
        self.assertIn("short_term_trend", trend)
        self.assertIn("medium_term_trend", trend)
        self.assertIn("long_term_trend", trend)
        self.assertIn("short_term_direction", trend)
        self.assertIn("medium_term_direction", trend)
        self.assertIn("long_term_direction", trend)
        self.assertIn("short_term_strength", trend)
        self.assertIn("medium_term_strength", trend)
        self.assertIn("long_term_strength", trend)
        self.assertIn("consistent_trend", trend)
        self.assertIn("accelerating_trend", trend)
        self.assertIn("trend_score", trend)
        
        # Check result types
        self.assertIsInstance(trend["short_term_trend"], float)
        self.assertIn(trend["short_term_direction"], ["up", "down"])
        self.assertIsInstance(trend["short_term_strength"], float)
        self.assertIsInstance(trend["consistent_trend"], bool)
        self.assertIsInstance(trend["trend_score"], float)
        
        # Test with custom period
        custom_trend = self.analyzer.analyze_oi_trend(self.open_interest, period=5)
        self.assertIsInstance(custom_trend["short_term_trend"], float)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_oi_trend(self.open_interest[:2])
    
    def test_analyze_significant_changes(self):
        """Test significant change analysis."""
        # Analyze significant changes
        changes = self.analyzer.analyze_significant_changes(self.open_interest, self.prices)
        
        # Check result structure
        self.assertIn("significant_increases", changes)
        self.assertIn("significant_decreases", changes)
        self.assertIn("recent_increases", changes)
        self.assertIn("recent_decreases", changes)
        self.assertIn("recent_oi_up_price_up", changes)
        self.assertIn("recent_oi_up_price_down", changes)
        self.assertIn("recent_oi_down_price_up", changes)
        self.assertIn("recent_oi_down_price_down", changes)
        self.assertIn("signal_strength", changes)
        
        # Check result types
        self.assertIsInstance(changes["significant_increases"], list)
        self.assertIsInstance(changes["recent_increases"], int)
        self.assertIsInstance(changes["recent_oi_up_price_up"], int)
        self.assertIsInstance(changes["signal_strength"], float)
        
        # Check result shapes
        self.assertEqual(len(changes["significant_increases"]), len(self.open_interest))
        
        # Test with custom threshold
        custom_changes = self.analyzer.analyze_significant_changes(self.open_interest, self.prices, threshold=10.0)
        self.assertIsInstance(custom_changes["significant_increases"], list)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_significant_changes(self.open_interest[:2], self.prices[:2])
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        divergence_analysis = {
            "bullish_divergence": True,
            "bearish_divergence": False,
            "extreme_divergence": False,
            "local_extrema_divergence": True,
            "price_trend": 1.5,
            "oi_trend": -2.0,
            "trend_divergence": 3.5,
            "divergence_strength": 0.7
        }
        
        trend_analysis = {
            "short_term_trend": 2.0,
            "medium_term_trend": 1.5,
            "long_term_trend": 1.0,
            "short_term_direction": "up",
            "medium_term_direction": "up",
            "long_term_direction": "up",
            "short_term_strength": 2.0,
            "medium_term_strength": 1.5,
            "long_term_strength": 1.0,
            "consistent_trend": True,
            "accelerating_trend": True,
            "trend_score": 50.0
        }
        
        change_analysis = {
            "significant_increases": [False, True, False],
            "significant_decreases": [False, False, True],
            "recent_increases": 1,
            "recent_decreases": 1,
            "recent_oi_up_price_up": 1,
            "recent_oi_up_price_down": 0,
            "recent_oi_down_price_up": 0,
            "recent_oi_down_price_down": 1,
            "signal_strength": 0.1
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            divergence_analysis,
            trend_analysis,
            change_analysis
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
        self.assertIn("divergence", signal["components"])
        self.assertIn("trend", signal["components"])
        self.assertIn("change", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish divergence
        bearish_divergence = divergence_analysis.copy()
        bearish_divergence["bullish_divergence"] = False
        bearish_divergence["bearish_divergence"] = True
        bearish_divergence["divergence_strength"] = -0.7
        
        bearish_signal = self.analyzer.generate_signal(
            bearish_divergence,
            trend_analysis,
            change_analysis
        )
        
        self.assertLess(bearish_signal["components"]["divergence"], 0)
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("metrics", result)
        self.assertIn("divergence_analysis", result)
        self.assertIn("trend_analysis", result)
        self.assertIn("change_analysis", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check metrics structure
        self.assertIn("oi_change", result["metrics"])
        self.assertIn("oi_change_pct", result["metrics"])
        self.assertIn("oi_roc", result["metrics"])
        self.assertIn("price_change", result["metrics"])
        self.assertIn("price_change_pct", result["metrics"])
        self.assertIn("price_roc", result["metrics"])
        self.assertIn("oi_price_ratio", result["metrics"])
        self.assertIn("oi_price_ratio_change", result["metrics"])
        
        # Check result types
        self.assertIsInstance(result["metrics"]["oi_change"], list)
        self.assertIsInstance(result["divergence_analysis"], dict)
        self.assertIsInstance(result["trend_analysis"], dict)
        self.assertIsInstance(result["change_analysis"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["metrics"]["oi_change"]), len(self.open_interest))
        self.assertEqual(len(result["metrics"]["price_change"]), len(self.prices))
        
        # Test standalone function
        standalone_result = analyze_open_interest(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
        
        # Test divergence standalone function
        divergence_result = detect_oi_divergence(self.df)
        self.assertIsInstance(divergence_result["bullish_divergence"], bool)
        self.assertIsInstance(divergence_result["bearish_divergence"], bool)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        df_missing = pd.DataFrame({
            "open": self.prices,
            "high": self.prices,
            "low": self.prices
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(df_missing)
        
        # Test with insufficient data
        short_prices = np.array([100, 101, 102])
        short_oi = np.array([1000, 1100, 1200])
        short_df = pd.DataFrame({
            "close": short_prices,
            "open_interest": short_oi
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze_oi_trend(short_oi)
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze_significant_changes(short_oi, short_prices)
        
        # Test standalone function with missing columns
        with self.assertRaises(ModelError):
            detect_oi_divergence(df_missing)


if __name__ == "__main__":
    unittest.main()
