"""
Tests for the Spot vs Perpetual CVD (Cumulative Volume Delta) Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.cvd_analyzer import (
    CVDAnalyzer,
    calculate_cvd,
    analyze_spot_perp_cvd
)


class TestCVDAnalyzer(unittest.TestCase):
    """Test cases for the CVDAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = CVDAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.periods = 100
        
        # Generate buy/sell volume data with trends
        # Spot: Initially balanced, then more buying
        # Perp: Initially balanced, then more selling
        spot_buy_base = np.random.normal(100, 20, self.periods)
        spot_sell_base = np.random.normal(100, 20, self.periods)
        perp_buy_base = np.random.normal(100, 20, self.periods)
        perp_sell_base = np.random.normal(100, 20, self.periods)
        
        # Add trend
        spot_buy_trend = np.linspace(0, 50, self.periods)
        spot_sell_trend = np.linspace(0, 0, self.periods)
        perp_buy_trend = np.linspace(0, 0, self.periods)
        perp_sell_trend = np.linspace(0, 50, self.periods)
        
        self.spot_buy_volume = spot_buy_base + spot_buy_trend
        self.spot_sell_volume = spot_sell_base + spot_sell_trend
        self.perp_buy_volume = perp_buy_base + perp_buy_trend
        self.perp_sell_volume = perp_sell_base + perp_sell_trend
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "spot_buy_volume": self.spot_buy_volume,
            "spot_sell_volume": self.spot_sell_volume,
            "perp_buy_volume": self.perp_buy_volume,
            "perp_sell_volume": self.perp_sell_volume
        })
        
        # Calculate CVD
        self.spot_cvd = self.analyzer.calculate_cvd(self.spot_buy_volume, self.spot_sell_volume)
        self.perp_cvd = self.analyzer.calculate_cvd(self.perp_buy_volume, self.perp_sell_volume)
        
        # Normalize CVD
        self.spot_cvd_norm = self.analyzer.normalize_cvd(self.spot_cvd, method='z-score')
        self.perp_cvd_norm = self.analyzer.normalize_cvd(self.perp_cvd, method='z-score')
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = CVDAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = CVDAnalyzer(
            lookback_period=15,
            divergence_threshold=5.0,
            signal_threshold=0.4,
            trend_period=10
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.divergence_threshold, 5.0)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.trend_period, 10)
    
    def test_calculate_cvd(self):
        """Test CVD calculation."""
        # Calculate CVD
        cvd = self.analyzer.calculate_cvd(self.spot_buy_volume, self.spot_sell_volume)
        
        # Check result shape
        self.assertEqual(len(cvd), len(self.spot_buy_volume))
        
        # Check that first value is correct (buy - sell)
        expected_first_value = self.spot_buy_volume[0] - self.spot_sell_volume[0]
        self.assertAlmostEqual(cvd[0], expected_first_value)
        
        # Check that values are cumulative
        delta1 = self.spot_buy_volume[1] - self.spot_sell_volume[1]
        expected_second_value = expected_first_value + delta1
        self.assertAlmostEqual(cvd[1], expected_second_value)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.calculate_cvd(self.spot_buy_volume[:50], self.spot_sell_volume)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.calculate_cvd(np.array([1]), np.array([1]))
        
        # Test standalone function
        standalone_cvd = calculate_cvd(self.spot_buy_volume, self.spot_sell_volume)
        self.assertTrue(np.allclose(cvd, standalone_cvd))
    
    def test_normalize_cvd(self):
        """Test CVD normalization."""
        # Test z-score normalization
        z_score_norm = self.analyzer.normalize_cvd(self.spot_cvd, method='z-score')
        self.assertEqual(len(z_score_norm), len(self.spot_cvd))
        self.assertAlmostEqual(np.mean(z_score_norm), 0, delta=1e-10)
        self.assertAlmostEqual(np.std(z_score_norm), 1, delta=1e-10)
        
        # Test min-max normalization
        min_max_norm = self.analyzer.normalize_cvd(self.spot_cvd, method='min-max')
        self.assertEqual(len(min_max_norm), len(self.spot_cvd))
        self.assertGreaterEqual(np.min(min_max_norm), 0)
        self.assertLessEqual(np.max(min_max_norm), 1)
        
        # Test percent-change normalization
        pct_change_norm = self.analyzer.normalize_cvd(self.spot_cvd, method='percent-change')
        self.assertEqual(len(pct_change_norm), len(self.spot_cvd))
        self.assertAlmostEqual(pct_change_norm[0], 0, delta=1e-10)
        
        # Test with invalid method
        with self.assertRaises(ModelError):
            self.analyzer.normalize_cvd(self.spot_cvd, method='invalid')
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.normalize_cvd(np.array([1]))
        
        # Test with zero standard deviation
        zero_std_cvd = np.ones(10)
        zero_std_norm = self.analyzer.normalize_cvd(zero_std_cvd, method='z-score')
        self.assertTrue(np.all(zero_std_norm == 0))
        
        # Test with zero range
        zero_range_cvd = np.ones(10)
        zero_range_norm = self.analyzer.normalize_cvd(zero_range_cvd, method='min-max')
        self.assertTrue(np.all(zero_range_norm == 0))
        
        # Test with zero first value
        zero_first_cvd = np.ones(10)
        zero_first_cvd[0] = 0
        zero_first_norm = self.analyzer.normalize_cvd(zero_first_cvd, method='percent-change')
        self.assertNotEqual(zero_first_norm[1], float('inf'))
    
    def test_detect_divergence(self):
        """Test divergence detection."""
        # Detect divergence
        divergence = self.analyzer.detect_divergence(self.spot_cvd_norm, self.perp_cvd_norm)
        
        # Check result structure
        self.assertIn("bullish_divergence", divergence)
        self.assertIn("bearish_divergence", divergence)
        self.assertIn("extreme_divergence", divergence)
        self.assertIn("spot_trend", divergence)
        self.assertIn("perp_trend", divergence)
        self.assertIn("trend_divergence", divergence)
        self.assertIn("correlation", divergence)
        self.assertIn("divergence_strength", divergence)
        
        # Check result types
        self.assertIsInstance(divergence["bullish_divergence"], bool)
        self.assertIsInstance(divergence["bearish_divergence"], bool)
        self.assertIsInstance(divergence["spot_trend"], float)
        self.assertIsInstance(divergence["correlation"], float)
        self.assertIsInstance(divergence["divergence_strength"], float)
        
        # Test with custom lookback
        custom_divergence = self.analyzer.detect_divergence(self.spot_cvd_norm, self.perp_cvd_norm, lookback=10)
        self.assertIsInstance(custom_divergence["bullish_divergence"], bool)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.detect_divergence(self.spot_cvd_norm[:50], self.perp_cvd_norm)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_divergence(self.spot_cvd_norm[:5], self.perp_cvd_norm[:5])
    
    def test_analyze_cvd_trend(self):
        """Test CVD trend analysis."""
        # Analyze trend
        trend = self.analyzer.analyze_cvd_trend(self.spot_cvd_norm)
        
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
        custom_trend = self.analyzer.analyze_cvd_trend(self.spot_cvd_norm, period=5)
        self.assertIsInstance(custom_trend["short_term_trend"], float)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cvd_trend(self.spot_cvd_norm[:2])
    
    def test_analyze_cvd_momentum(self):
        """Test CVD momentum analysis."""
        # Analyze momentum
        momentum = self.analyzer.analyze_cvd_momentum(self.spot_cvd_norm, self.perp_cvd_norm)
        
        # Check result structure
        self.assertIn("spot_momentum_1", momentum)
        self.assertIn("spot_momentum_5", momentum)
        self.assertIn("perp_momentum_1", momentum)
        self.assertIn("perp_momentum_5", momentum)
        self.assertIn("relative_momentum_1", momentum)
        self.assertIn("relative_momentum_5", momentum)
        self.assertIn("spot_perp_ratio", momentum)
        self.assertIn("ratio_change", momentum)
        self.assertIn("momentum_strength", momentum)
        
        # Check result types
        self.assertIsInstance(momentum["spot_momentum_1"], float)
        self.assertIsInstance(momentum["relative_momentum_1"], float)
        self.assertIsInstance(momentum["momentum_strength"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cvd_momentum(self.spot_cvd_norm[:50], self.perp_cvd_norm)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cvd_momentum(self.spot_cvd_norm[:2], self.perp_cvd_norm[:2])
        
        # Test with zero perp_cvd values
        zero_perp_cvd = np.zeros_like(self.perp_cvd_norm)
        zero_momentum = self.analyzer.analyze_cvd_momentum(self.spot_cvd_norm, zero_perp_cvd)
        self.assertIsNone(zero_momentum["spot_perp_ratio"])
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        divergence_analysis = {
            "bullish_divergence": True,
            "bearish_divergence": False,
            "extreme_divergence": False,
            "spot_trend": 2.5,
            "perp_trend": -1.5,
            "trend_divergence": 4.0,
            "correlation": -0.2,
            "divergence_strength": 0.6
        }
        
        spot_trend_analysis = {
            "short_term_trend": 3.0,
            "medium_term_trend": 2.0,
            "long_term_trend": 1.0,
            "short_term_direction": "up",
            "medium_term_direction": "up",
            "long_term_direction": "up",
            "short_term_strength": 3.0,
            "medium_term_strength": 2.0,
            "long_term_strength": 1.0,
            "consistent_trend": True,
            "accelerating_trend": True,
            "trend_score": 60.0
        }
        
        perp_trend_analysis = {
            "short_term_trend": -2.0,
            "medium_term_trend": -1.5,
            "long_term_trend": -1.0,
            "short_term_direction": "down",
            "medium_term_direction": "down",
            "long_term_direction": "down",
            "short_term_strength": 2.0,
            "medium_term_strength": 1.5,
            "long_term_strength": 1.0,
            "consistent_trend": True,
            "accelerating_trend": True,
            "trend_score": -40.0
        }
        
        momentum_analysis = {
            "spot_momentum_1": 1.0,
            "spot_momentum_5": 5.0,
            "perp_momentum_1": -1.0,
            "perp_momentum_5": -5.0,
            "relative_momentum_1": 2.0,
            "relative_momentum_5": 10.0,
            "spot_perp_ratio": 1.5,
            "ratio_change": 0.1,
            "momentum_strength": 0.8
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            divergence_analysis,
            spot_trend_analysis,
            perp_trend_analysis,
            momentum_analysis
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
        self.assertIn("momentum", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish divergence
        bearish_divergence = divergence_analysis.copy()
        bearish_divergence["bullish_divergence"] = False
        bearish_divergence["bearish_divergence"] = True
        bearish_divergence["divergence_strength"] = -0.6
        
        bearish_signal = self.analyzer.generate_signal(
            bearish_divergence,
            spot_trend_analysis,
            perp_trend_analysis,
            momentum_analysis
        )
        
        self.assertLess(bearish_signal["components"]["divergence"], 0)
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("metrics", result)
        self.assertIn("divergence_analysis", result)
        self.assertIn("spot_trend_analysis", result)
        self.assertIn("perp_trend_analysis", result)
        self.assertIn("momentum_analysis", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check metrics structure
        self.assertIn("spot_cvd", result["metrics"])
        self.assertIn("perp_cvd", result["metrics"])
        self.assertIn("spot_cvd_norm", result["metrics"])
        self.assertIn("perp_cvd_norm", result["metrics"])
        
        # Check result types
        self.assertIsInstance(result["metrics"]["spot_cvd"], list)
        self.assertIsInstance(result["divergence_analysis"], dict)
        self.assertIsInstance(result["spot_trend_analysis"], dict)
        self.assertIsInstance(result["perp_trend_analysis"], dict)
        self.assertIsInstance(result["momentum_analysis"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["metrics"]["spot_cvd"]), len(self.spot_buy_volume))
        self.assertEqual(len(result["metrics"]["perp_cvd"]), len(self.perp_buy_volume))
        
        # Test standalone function
        standalone_result = analyze_spot_perp_cvd(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        df_missing = pd.DataFrame({
            "spot_buy_volume": self.spot_buy_volume,
            "spot_sell_volume": self.spot_sell_volume,
            "perp_buy_volume": self.perp_buy_volume
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(df_missing)
        
        # Test with insufficient data
        short_df = pd.DataFrame({
            "spot_buy_volume": self.spot_buy_volume[:2],
            "spot_sell_volume": self.spot_sell_volume[:2],
            "perp_buy_volume": self.perp_buy_volume[:2],
            "perp_sell_volume": self.perp_sell_volume[:2]
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cvd_trend(np.array([1, 2]))
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze_cvd_momentum(np.array([1, 2]), np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
