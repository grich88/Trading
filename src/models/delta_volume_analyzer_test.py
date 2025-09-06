"""
Tests for the Delta (Aggressor) Volume Imbalance Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.delta_volume_analyzer import (
    DeltaVolumeAnalyzer,
    calculate_delta_volume,
    analyze_delta_volume
)


class TestDeltaVolumeAnalyzer(unittest.TestCase):
    """Test cases for the DeltaVolumeAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = DeltaVolumeAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.periods = 100
        
        # Generate price data with trend and noise
        self.prices = np.cumsum(np.random.normal(0, 1, self.periods)) + 100
        
        # Generate volume data
        self.volumes = np.random.lognormal(0, 1, self.periods) * 1000
        
        # Generate buy and sell market volume data
        # First half: balanced
        # Second half: more buy market volume
        buy_market_base = np.random.normal(100, 20, self.periods)
        sell_market_base = np.random.normal(100, 20, self.periods)
        
        # Add trend
        buy_market_trend = np.concatenate([np.zeros(self.periods//2), np.linspace(0, 50, self.periods - self.periods//2)])
        sell_market_trend = np.zeros(self.periods)
        
        self.buy_market_volume = buy_market_base + buy_market_trend
        self.sell_market_volume = sell_market_base + sell_market_trend
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "close": self.prices,
            "volume": self.volumes,
            "buy_market_volume": self.buy_market_volume,
            "sell_market_volume": self.sell_market_volume
        })
        
        # Calculate delta volume
        self.delta_volume = self.analyzer.calculate_delta_volume(self.buy_market_volume, self.sell_market_volume)
        
        # Calculate cumulative delta
        self.cumulative_delta = self.analyzer.calculate_cumulative_delta(self.delta_volume)
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = DeltaVolumeAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = DeltaVolumeAnalyzer(
            lookback_period=15,
            imbalance_threshold=5.0,
            signal_threshold=0.4,
            trend_period=10
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.imbalance_threshold, 5.0)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.trend_period, 10)
    
    def test_calculate_delta_volume(self):
        """Test delta volume calculation."""
        # Calculate delta volume
        delta_volume = self.analyzer.calculate_delta_volume(self.buy_market_volume, self.sell_market_volume)
        
        # Check result shape
        self.assertEqual(len(delta_volume), len(self.buy_market_volume))
        
        # Check that values are correctly calculated
        for i in range(len(delta_volume)):
            expected_delta = self.buy_market_volume[i] - self.sell_market_volume[i]
            self.assertAlmostEqual(delta_volume[i], expected_delta)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.calculate_delta_volume(self.buy_market_volume[:50], self.sell_market_volume)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.calculate_delta_volume(np.array([1]), np.array([1]))
        
        # Test standalone function
        standalone_delta = calculate_delta_volume(self.buy_market_volume, self.sell_market_volume)
        self.assertTrue(np.allclose(delta_volume, standalone_delta))
    
    def test_calculate_cumulative_delta(self):
        """Test cumulative delta calculation."""
        # Calculate cumulative delta
        cumulative_delta = self.analyzer.calculate_cumulative_delta(self.delta_volume)
        
        # Check result shape
        self.assertEqual(len(cumulative_delta), len(self.delta_volume))
        
        # Check that values are correctly calculated
        expected_cumulative = np.cumsum(self.delta_volume)
        self.assertTrue(np.allclose(cumulative_delta, expected_cumulative))
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.calculate_cumulative_delta(np.array([1]))
    
    def test_detect_imbalance(self):
        """Test imbalance detection."""
        # Detect imbalance
        imbalance = self.analyzer.detect_imbalance(self.delta_volume)
        
        # Check result structure
        self.assertIn("imbalance_direction", imbalance)
        self.assertIn("imbalance_ratio", imbalance)
        self.assertIn("total_buy_market", imbalance)
        self.assertIn("total_sell_market", imbalance)
        self.assertIn("net_delta", imbalance)
        self.assertIn("delta_std", imbalance)
        self.assertIn("current_z_score", imbalance)
        self.assertIn("imbalance_strength", imbalance)
        
        # Check result types
        self.assertIn(imbalance["imbalance_direction"], ["buy", "sell", "neutral"])
        self.assertIsInstance(imbalance["total_buy_market"], float)
        self.assertIsInstance(imbalance["total_sell_market"], float)
        self.assertIsInstance(imbalance["net_delta"], float)
        self.assertIsInstance(imbalance["delta_std"], float)
        self.assertIsInstance(imbalance["current_z_score"], float)
        self.assertIsInstance(imbalance["imbalance_strength"], float)
        
        # Test with custom lookback
        custom_imbalance = self.analyzer.detect_imbalance(self.delta_volume, lookback=10)
        self.assertIn(custom_imbalance["imbalance_direction"], ["buy", "sell", "neutral"])
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_imbalance(self.delta_volume[:5])
        
        # Test with extreme cases
        # All buy market volume
        all_buy = np.ones(30) * 100
        all_sell = np.zeros(30)
        delta_all_buy = all_buy - all_sell
        imbalance_all_buy = self.analyzer.detect_imbalance(delta_all_buy)
        self.assertEqual(imbalance_all_buy["imbalance_direction"], "buy")
        self.assertIsNone(imbalance_all_buy["imbalance_ratio"])  # Should be inf
        
        # All sell market volume
        all_buy = np.zeros(30)
        all_sell = np.ones(30) * 100
        delta_all_sell = all_buy - all_sell
        imbalance_all_sell = self.analyzer.detect_imbalance(delta_all_sell)
        self.assertEqual(imbalance_all_sell["imbalance_direction"], "sell")
        self.assertIsNone(imbalance_all_sell["imbalance_ratio"])  # Should be inf
    
    def test_analyze_delta_trend(self):
        """Test delta trend analysis."""
        # Analyze trend
        trend = self.analyzer.analyze_delta_trend(self.delta_volume)
        
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
        self.assertIsInstance(trend["accelerating_trend"], bool)
        self.assertIsInstance(trend["trend_score"], float)
        
        # Test with custom period
        custom_trend = self.analyzer.analyze_delta_trend(self.delta_volume, period=5)
        self.assertIsInstance(custom_trend["short_term_trend"], float)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_delta_trend(self.delta_volume[:2])
        
        # Test with zero mean delta
        zero_delta = np.zeros(30)
        zero_trend = self.analyzer.analyze_delta_trend(zero_delta)
        self.assertEqual(zero_trend["short_term_trend"], 0.0)
    
    def test_detect_divergence(self):
        """Test divergence detection."""
        # Detect divergence
        divergence = self.analyzer.detect_divergence(self.delta_volume, self.prices)
        
        # Check result structure
        self.assertIn("bullish_divergence", divergence)
        self.assertIn("bearish_divergence", divergence)
        self.assertIn("extreme_divergence", divergence)
        self.assertIn("delta_trend", divergence)
        self.assertIn("price_trend", divergence)
        self.assertIn("trend_divergence", divergence)
        self.assertIn("correlation", divergence)
        self.assertIn("divergence_strength", divergence)
        
        # Check result types
        self.assertIsInstance(divergence["bullish_divergence"], bool)
        self.assertIsInstance(divergence["bearish_divergence"], bool)
        self.assertIsInstance(divergence["extreme_divergence"], bool)
        self.assertIsInstance(divergence["delta_trend"], float)
        self.assertIsInstance(divergence["price_trend"], float)
        self.assertIsInstance(divergence["trend_divergence"], float)
        self.assertIsInstance(divergence["correlation"], float)
        self.assertIsInstance(divergence["divergence_strength"], float)
        
        # Test with custom lookback
        custom_divergence = self.analyzer.detect_divergence(self.delta_volume, self.prices, lookback=10)
        self.assertIsInstance(custom_divergence["bullish_divergence"], bool)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.detect_divergence(self.delta_volume[:50], self.prices)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_divergence(self.delta_volume[:5], self.prices[:5])
        
        # Test with zero cumulative delta
        zero_delta = np.zeros(30)
        zero_prices = np.ones(30) * 100
        zero_divergence = self.analyzer.detect_divergence(zero_delta, zero_prices)
        self.assertEqual(zero_divergence["delta_trend"], 0.0)
    
    def test_analyze_vwap_delta(self):
        """Test VWAP delta analysis."""
        # Analyze VWAP delta
        vwap = self.analyzer.analyze_vwap_delta(self.delta_volume, self.prices, self.volumes)
        
        # Check result structure
        self.assertIn("vwap", vwap)
        self.assertIn("buy_vwap", vwap)
        self.assertIn("sell_vwap", vwap)
        self.assertIn("vwap_delta", vwap)
        self.assertIn("normalized_vwap_delta", vwap)
        self.assertIn("vwap_delta_direction", vwap)
        self.assertIn("vwap_delta_strength", vwap)
        
        # Check result types
        self.assertIsInstance(vwap["vwap"], float)
        self.assertIsInstance(vwap["buy_vwap"], float)
        self.assertIsInstance(vwap["sell_vwap"], float)
        self.assertIsInstance(vwap["vwap_delta"], float)
        self.assertIsInstance(vwap["normalized_vwap_delta"], float)
        self.assertIn(vwap["vwap_delta_direction"], ["buy", "sell", "neutral"])
        self.assertIsInstance(vwap["vwap_delta_strength"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.analyze_vwap_delta(self.delta_volume[:50], self.prices, self.volumes)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_vwap_delta(self.delta_volume[:2], self.prices[:2], self.volumes[:2])
        
        # Test with all buy volume
        all_buy_delta = np.ones(30) * 100
        all_buy_vwap = self.analyzer.analyze_vwap_delta(all_buy_delta, self.prices[:30], self.volumes[:30])
        self.assertEqual(all_buy_vwap["vwap_delta_direction"], "buy")
        
        # Test with all sell volume
        all_sell_delta = np.ones(30) * -100
        all_sell_vwap = self.analyzer.analyze_vwap_delta(all_sell_delta, self.prices[:30], self.volumes[:30])
        self.assertEqual(all_sell_vwap["vwap_delta_direction"], "sell")
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        imbalance_analysis = {
            "imbalance_direction": "buy",
            "imbalance_ratio": 2.5,
            "total_buy_market": 1000.0,
            "total_sell_market": 400.0,
            "net_delta": 600.0,
            "delta_std": 50.0,
            "current_z_score": 2.5,
            "imbalance_strength": 0.6
        }
        
        trend_analysis = {
            "short_term_trend": 5.0,
            "medium_term_trend": 3.0,
            "long_term_trend": 2.0,
            "short_term_direction": "up",
            "medium_term_direction": "up",
            "long_term_direction": "up",
            "short_term_strength": 5.0,
            "medium_term_strength": 3.0,
            "long_term_strength": 2.0,
            "consistent_trend": True,
            "accelerating_trend": True,
            "trend_score": 60.0
        }
        
        divergence_analysis = {
            "bullish_divergence": True,
            "bearish_divergence": False,
            "extreme_divergence": False,
            "delta_trend": 5.0,
            "price_trend": -2.0,
            "trend_divergence": 7.0,
            "correlation": -0.3,
            "divergence_strength": 0.7
        }
        
        vwap_analysis = {
            "vwap": 100.0,
            "buy_vwap": 102.0,
            "sell_vwap": 98.0,
            "vwap_delta": 4.0,
            "normalized_vwap_delta": 4.0,
            "vwap_delta_direction": "buy",
            "vwap_delta_strength": 0.4
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            imbalance_analysis,
            trend_analysis,
            divergence_analysis,
            vwap_analysis
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
        self.assertIn("imbalance", signal["components"])
        self.assertIn("trend", signal["components"])
        self.assertIn("divergence", signal["components"])
        self.assertIn("vwap", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish signals
        bearish_imbalance = imbalance_analysis.copy()
        bearish_imbalance["imbalance_direction"] = "sell"
        bearish_imbalance["imbalance_strength"] = -0.6
        
        bearish_trend = trend_analysis.copy()
        bearish_trend["short_term_direction"] = "down"
        bearish_trend["medium_term_direction"] = "down"
        bearish_trend["long_term_direction"] = "down"
        bearish_trend["trend_score"] = -60.0
        
        bearish_divergence = divergence_analysis.copy()
        bearish_divergence["bullish_divergence"] = False
        bearish_divergence["bearish_divergence"] = True
        bearish_divergence["divergence_strength"] = -0.7
        
        bearish_vwap = vwap_analysis.copy()
        bearish_vwap["vwap_delta_direction"] = "sell"
        bearish_vwap["vwap_delta_strength"] = -0.4
        
        bearish_signal = self.analyzer.generate_signal(
            bearish_imbalance,
            bearish_trend,
            bearish_divergence,
            bearish_vwap
        )
        
        self.assertLess(bearish_signal["score"], 0)
        self.assertIn(bearish_signal["signal"], ["SELL", "STRONG SELL"])
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("metrics", result)
        self.assertIn("imbalance_analysis", result)
        self.assertIn("trend_analysis", result)
        self.assertIn("divergence_analysis", result)
        self.assertIn("vwap_analysis", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check metrics structure
        self.assertIn("delta_volume", result["metrics"])
        self.assertIn("cumulative_delta", result["metrics"])
        
        # Check result types
        self.assertIsInstance(result["metrics"]["delta_volume"], list)
        self.assertIsInstance(result["metrics"]["cumulative_delta"], list)
        self.assertIsInstance(result["imbalance_analysis"], dict)
        self.assertIsInstance(result["trend_analysis"], dict)
        self.assertIsInstance(result["divergence_analysis"], dict)
        self.assertIsInstance(result["vwap_analysis"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["metrics"]["delta_volume"]), len(self.delta_volume))
        self.assertEqual(len(result["metrics"]["cumulative_delta"]), len(self.cumulative_delta))
        
        # Test standalone function
        standalone_result = analyze_delta_volume(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        df_missing = pd.DataFrame({
            "close": self.prices,
            "volume": self.volumes,
            "buy_market_volume": self.buy_market_volume
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(df_missing)
        
        # Test with insufficient data
        short_df = pd.DataFrame({
            "close": self.prices[:2],
            "volume": self.volumes[:2],
            "buy_market_volume": self.buy_market_volume[:2],
            "sell_market_volume": self.sell_market_volume[:2]
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(short_df)


if __name__ == "__main__":
    unittest.main()
