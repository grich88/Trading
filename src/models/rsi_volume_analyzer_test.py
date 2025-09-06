"""
Tests for the RSI and Volume Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.rsi_volume_analyzer import (
    RSIVolumeAnalyzer,
    calculate_rsi,
    analyze_rsi_volume
)


class TestRSIVolumeAnalyzer(unittest.TestCase):
    """Test cases for the RSIVolumeAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = RSIVolumeAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.prices = np.cumsum(np.random.normal(0, 1, 100)) + 100
        self.volume = np.random.lognormal(0, 1, 100) * 1000
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "close": self.prices,
            "volume": self.volume
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = RSIVolumeAnalyzer()
        self.assertEqual(analyzer.rsi_period, 14)  # Default RSI period
        
        # Test custom initialization
        custom_analyzer = RSIVolumeAnalyzer(
            rsi_period=10,
            rsi_overbought=75,
            rsi_oversold=25,
            volume_ma_period=20,
            divergence_lookback=15,
            signal_threshold=0.4
        )
        self.assertEqual(custom_analyzer.rsi_period, 10)
        self.assertEqual(custom_analyzer.rsi_overbought, 75)
        self.assertEqual(custom_analyzer.rsi_oversold, 25)
        self.assertEqual(custom_analyzer.volume_ma_period, 20)
        self.assertEqual(custom_analyzer.divergence_lookback, 15)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Calculate RSI
        rsi = self.analyzer.calculate_rsi(self.prices)
        
        # Check result shape
        self.assertEqual(len(rsi), len(self.prices))
        
        # Check that initial values are NaN
        self.assertTrue(np.isnan(rsi[0]))
        
        # Check that later values are in the correct range
        self.assertTrue(np.all((rsi[15:] >= 0) & (rsi[15:] <= 100)))
        
        # Test with custom period
        custom_rsi = self.analyzer.calculate_rsi(self.prices, period=7)
        self.assertEqual(len(custom_rsi), len(self.prices))
        self.assertTrue(np.all((custom_rsi[8:] >= 0) & (custom_rsi[8:] <= 100)))
        
        # Test standalone function
        standalone_rsi = calculate_rsi(self.prices)
        self.assertTrue(np.allclose(rsi[15:], standalone_rsi[15:], equal_nan=True))
    
    def test_calculate_volume_ma(self):
        """Test volume moving average calculation."""
        # Calculate volume MA
        volume_ma = self.analyzer.calculate_volume_ma(self.volume)
        
        # Check result shape
        self.assertEqual(len(volume_ma), len(self.volume))
        
        # Check that initial values are NaN
        self.assertTrue(np.isnan(volume_ma[0]))
        
        # Check that later values are positive
        self.assertTrue(np.all(volume_ma[self.analyzer.volume_ma_period:] > 0))
        
        # Test with custom period
        custom_ma = self.analyzer.calculate_volume_ma(self.volume, period=5)
        self.assertEqual(len(custom_ma), len(self.volume))
        self.assertTrue(np.all(custom_ma[5:] > 0))
    
    def test_detect_volume_anomalies(self):
        """Test volume anomaly detection."""
        # Calculate volume MA
        volume_ma = self.analyzer.calculate_volume_ma(self.volume)
        
        # Detect anomalies
        anomalies = self.analyzer.detect_volume_anomalies(self.volume, volume_ma)
        
        # Check result structure
        self.assertIn("high_volume", anomalies)
        self.assertIn("very_high_volume", anomalies)
        self.assertIn("extremely_high_volume", anomalies)
        self.assertIn("recent_high", anomalies)
        self.assertIn("recent_very_high", anomalies)
        self.assertIn("recent_extremely_high", anomalies)
        self.assertIn("volume_trend", anomalies)
        self.assertIn("volume_ratio", anomalies)
        
        # Check result types
        self.assertIsInstance(anomalies["high_volume"], np.ndarray)
        self.assertIsInstance(anomalies["recent_high"], int)
        self.assertIsInstance(anomalies["volume_trend"], float)
        self.assertIsInstance(anomalies["volume_ratio"], list)
        
        # Check result shapes
        self.assertEqual(len(anomalies["high_volume"]), len(self.volume))
        self.assertEqual(len(anomalies["volume_ratio"]), len(self.volume))
        
        # Test with custom threshold
        custom_anomalies = self.analyzer.detect_volume_anomalies(self.volume, volume_ma, threshold=2.0)
        self.assertLessEqual(sum(custom_anomalies["high_volume"]), sum(anomalies["high_volume"]))
    
    def test_detect_rsi_divergence(self):
        """Test RSI divergence detection."""
        # Calculate RSI
        rsi = self.analyzer.calculate_rsi(self.prices)
        
        # Detect divergence
        divergence = self.analyzer.detect_rsi_divergence(self.prices, rsi)
        
        # Check result structure
        self.assertIn("bearish_divergence", divergence)
        self.assertIn("bullish_divergence", divergence)
        self.assertIn("divergence_strength", divergence)
        
        # Check result types
        self.assertIsInstance(divergence["bearish_divergence"], bool)
        self.assertIsInstance(divergence["bullish_divergence"], bool)
        self.assertIsInstance(divergence["divergence_strength"], float)
        
        # Test with custom lookback
        custom_divergence = self.analyzer.detect_rsi_divergence(self.prices, rsi, lookback=10)
        self.assertIsInstance(custom_divergence["bearish_divergence"], bool)
    
    def test_analyze_rsi_zones(self):
        """Test RSI zone analysis."""
        # Calculate RSI
        rsi = self.analyzer.calculate_rsi(self.prices)
        
        # Analyze RSI zones
        zones = self.analyzer.analyze_rsi_zones(rsi)
        
        # Check result structure
        self.assertIn("current_rsi", zones)
        self.assertIn("is_overbought", zones)
        self.assertIn("is_oversold", zones)
        self.assertIn("exiting_overbought", zones)
        self.assertIn("exiting_oversold", zones)
        self.assertIn("periods_in_overbought", zones)
        self.assertIn("periods_in_oversold", zones)
        self.assertIn("rsi_momentum", zones)
        
        # Check result types
        self.assertIsInstance(zones["current_rsi"], float)
        self.assertIsInstance(zones["is_overbought"], bool)
        self.assertIsInstance(zones["periods_in_overbought"], int)
        self.assertIsInstance(zones["rsi_momentum"], float)
        
        # Check value ranges
        self.assertTrue(0 <= zones["current_rsi"] <= 100)
        self.assertTrue(zones["periods_in_overbought"] >= 0)
        self.assertTrue(zones["periods_in_oversold"] >= 0)
    
    def test_analyze_volume_confirmation(self):
        """Test volume confirmation analysis."""
        # Calculate volume MA
        volume_ma = self.analyzer.calculate_volume_ma(self.volume)
        
        # Analyze volume confirmation
        confirmation = self.analyzer.analyze_volume_confirmation(self.prices, self.volume, volume_ma)
        
        # Check result structure
        self.assertIn("confirms_up", confirmation)
        self.assertIn("confirms_down", confirmation)
        self.assertIn("recent_confirms_up", confirmation)
        self.assertIn("recent_confirms_down", confirmation)
        self.assertIn("confirmation_score", confirmation)
        self.assertIn("climax_volume", confirmation)
        
        # Check result types
        self.assertIsInstance(confirmation["confirms_up"], list)
        self.assertIsInstance(confirmation["recent_confirms_up"], int)
        self.assertIsInstance(confirmation["confirmation_score"], float)
        self.assertIsInstance(confirmation["climax_volume"], bool)
        
        # Check result shapes
        self.assertEqual(len(confirmation["confirms_up"]), len(self.prices) - 1)
        
        # Check value ranges
        self.assertTrue(confirmation["recent_confirms_up"] >= 0)
        self.assertTrue(confirmation["recent_confirms_down"] >= 0)
        self.assertTrue(-1.0 <= confirmation["confirmation_score"] <= 1.0)
    
    def test_generate_rsi_volume_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        rsi_analysis = {
            "current_rsi": 65.0,
            "is_overbought": False,
            "is_oversold": False,
            "exiting_overbought": False,
            "exiting_oversold": False,
            "periods_in_overbought": 0,
            "periods_in_oversold": 0,
            "rsi_momentum": 0.5
        }
        
        divergence_analysis = {
            "bearish_divergence": False,
            "bullish_divergence": True,
            "divergence_strength": 0.7
        }
        
        volume_analysis = {
            "high_volume": np.array([False, True, False]),
            "very_high_volume": np.array([False, False, False]),
            "extremely_high_volume": np.array([False, False, False]),
            "recent_high": 1,
            "recent_very_high": 0,
            "recent_extremely_high": 0,
            "volume_trend": 0.2,
            "volume_ratio": [0.8, 1.2, 0.9]
        }
        
        confirmation_analysis = {
            "confirms_up": [True, False],
            "confirms_down": [False, True],
            "recent_confirms_up": 1,
            "recent_confirms_down": 1,
            "confirmation_score": 0.1,
            "climax_volume": False
        }
        
        # Generate signal
        signal = self.analyzer.generate_rsi_volume_signal(
            rsi_analysis,
            divergence_analysis,
            volume_analysis,
            confirmation_analysis
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
        self.assertIn("rsi_zone", signal["components"])
        self.assertIn("divergence", signal["components"])
        self.assertIn("volume", signal["components"])
        self.assertIn("confirmation", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish divergence
        bearish_divergence = divergence_analysis.copy()
        bearish_divergence["bearish_divergence"] = True
        bearish_divergence["bullish_divergence"] = False
        
        bearish_signal = self.analyzer.generate_rsi_volume_signal(
            rsi_analysis,
            bearish_divergence,
            volume_analysis,
            confirmation_analysis
        )
        
        self.assertLess(bearish_signal["score"], signal["score"])
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("rsi", result)
        self.assertIn("volume_ma", result)
        self.assertIn("rsi_analysis", result)
        self.assertIn("divergence_analysis", result)
        self.assertIn("volume_anomalies", result)
        self.assertIn("volume_confirmation", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check result types
        self.assertIsInstance(result["rsi"], list)
        self.assertIsInstance(result["volume_ma"], list)
        self.assertIsInstance(result["rsi_analysis"], dict)
        self.assertIsInstance(result["divergence_analysis"], dict)
        self.assertIsInstance(result["volume_anomalies"], dict)
        self.assertIsInstance(result["volume_confirmation"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["rsi"]), len(self.prices))
        self.assertEqual(len(result["volume_ma"]), len(self.volume))
        
        # Test standalone function
        standalone_result = analyze_rsi_volume(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
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
        short_volume = np.array([1000, 1100, 1200])
        
        with self.assertRaises(ModelError):
            self.analyzer.calculate_rsi(short_prices)
        
        with self.assertRaises(ModelError):
            self.analyzer.calculate_volume_ma(short_volume)


if __name__ == "__main__":
    unittest.main()
