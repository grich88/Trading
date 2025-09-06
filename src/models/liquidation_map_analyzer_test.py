"""
Tests for the Liquidation Map Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.liquidation_map_analyzer import (
    LiquidationMapAnalyzer,
    normalize_liquidations,
    analyze_liquidation_map
)


class TestLiquidationMapAnalyzer(unittest.TestCase):
    """Test cases for the LiquidationMapAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = LiquidationMapAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.periods = 100
        
        # Generate price data with trend and noise
        self.prices = np.cumsum(np.random.normal(0, 1, self.periods)) + 100
        
        # Generate volume data
        self.volumes = np.random.lognormal(0, 1, self.periods) * 1000
        
        # Generate liquidation data
        # Create clusters of liquidations at certain price levels
        self.liquidation_prices = np.linspace(self.prices.min() * 0.9, self.prices.max() * 1.1, self.periods)
        
        # Create liquidation amounts with clusters
        self.base_liquidations = np.random.lognormal(0, 1, self.periods) * 500
        
        # Add liquidation clusters
        for i in range(5):
            cluster_center = np.random.randint(0, self.periods)
            cluster_width = np.random.randint(3, 10)
            cluster_start = max(0, cluster_center - cluster_width // 2)
            cluster_end = min(self.periods, cluster_center + cluster_width // 2)
            self.base_liquidations[cluster_start:cluster_end] *= np.random.randint(3, 10)
        
        # Create DataFrame
        self.df = pd.DataFrame({
            "close": self.prices,
            "volume": self.volumes,
            "liquidation_amount": self.base_liquidations,
            "liquidation_price": self.liquidation_prices
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = LiquidationMapAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = LiquidationMapAnalyzer(
            lookback_period=15,
            cluster_threshold=5.0,
            signal_threshold=0.4,
            cascade_threshold=0.6
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.cluster_threshold, 5.0)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.cascade_threshold, 0.6)
    
    def test_normalize_liquidations(self):
        """Test liquidation normalization."""
        # Test z-score normalization
        z_score_norm = self.analyzer.normalize_liquidations(self.base_liquidations, self.prices, method='z-score')
        self.assertEqual(len(z_score_norm), len(self.base_liquidations))
        self.assertAlmostEqual(np.mean(z_score_norm), 0, delta=1e-10)
        self.assertAlmostEqual(np.std(z_score_norm), 1, delta=1e-10)
        
        # Test percent-of-volume normalization
        pct_volume_norm = self.analyzer.normalize_liquidations(self.base_liquidations, self.prices, method='percent-of-volume')
        self.assertEqual(len(pct_volume_norm), len(self.base_liquidations))
        self.assertAlmostEqual(np.sum(pct_volume_norm), 100, delta=1e-10)
        
        # Test price-relative normalization
        price_rel_norm = self.analyzer.normalize_liquidations(self.base_liquidations, self.prices, method='price-relative')
        self.assertEqual(len(price_rel_norm), len(self.base_liquidations))
        
        # Test with invalid method
        with self.assertRaises(ModelError):
            self.analyzer.normalize_liquidations(self.base_liquidations, self.prices, method='invalid')
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.normalize_liquidations(self.base_liquidations[:50], self.prices)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.normalize_liquidations(np.array([1]), np.array([1]))
        
        # Test with zero standard deviation
        zero_std_liquidations = np.ones(10)
        zero_std_norm = self.analyzer.normalize_liquidations(zero_std_liquidations, np.ones(10), method='z-score')
        self.assertTrue(np.all(zero_std_norm == 0))
        
        # Test with zero total volume
        zero_volume_norm = self.analyzer.normalize_liquidations(np.zeros(10), np.ones(10), method='percent-of-volume')
        self.assertTrue(np.all(zero_volume_norm == 0))
        
        # Test standalone function
        standalone_norm = normalize_liquidations(self.base_liquidations, self.prices)
        self.assertTrue(np.allclose(z_score_norm, standalone_norm))
    
    def test_detect_liquidation_clusters(self):
        """Test liquidation cluster detection."""
        # Detect clusters
        clusters = self.analyzer.detect_liquidation_clusters(
            self.base_liquidations, 
            self.prices, 
            self.liquidation_prices
        )
        
        # Check result structure
        self.assertIn("clusters", clusters)
        self.assertIn("total_clusters", clusters)
        self.assertIn("nearest_support", clusters)
        self.assertIn("nearest_resistance", clusters)
        self.assertIn("total_liquidations", clusters)
        self.assertIn("liquidations_below", clusters)
        self.assertIn("liquidations_above", clusters)
        self.assertIn("above_below_ratio", clusters)
        self.assertIn("below_above_ratio", clusters)
        self.assertIn("imbalance_direction", clusters)
        self.assertIn("imbalance_ratio", clusters)
        
        # Check result types
        self.assertIsInstance(clusters["clusters"], list)
        self.assertIsInstance(clusters["total_clusters"], int)
        self.assertIn(clusters["imbalance_direction"], ["support", "resistance", "neutral"])
        
        # Check cluster structure
        if clusters["clusters"]:
            cluster = clusters["clusters"][0]
            self.assertIn("price_level", cluster)
            self.assertIn("distance_percent", cluster)
            self.assertIn("liquidation_amount", cluster)
            self.assertIn("is_support", cluster)
            self.assertIn("is_resistance", cluster)
            self.assertIn("strength", cluster)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.detect_liquidation_clusters(
                self.base_liquidations[:50], 
                self.prices, 
                self.liquidation_prices
            )
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_liquidation_clusters(
                np.array([1, 2, 3, 4]), 
                np.array([1, 2, 3, 4]), 
                np.array([1, 2, 3, 4])
            )
        
        # Test extreme cases
        # All liquidations below current price
        current_price = 100
        liquidation_prices = np.ones(30) * 50
        liquidation_amounts = np.ones(30) * 100
        below_clusters = self.analyzer.detect_liquidation_clusters(
            liquidation_amounts, 
            np.array([current_price]), 
            liquidation_prices
        )
        self.assertEqual(below_clusters["imbalance_direction"], "support")
        self.assertIsNone(below_clusters["above_below_ratio"])  # Should be inf
        
        # All liquidations above current price
        liquidation_prices = np.ones(30) * 150
        above_clusters = self.analyzer.detect_liquidation_clusters(
            liquidation_amounts, 
            np.array([current_price]), 
            liquidation_prices
        )
        self.assertEqual(above_clusters["imbalance_direction"], "resistance")
        self.assertIsNone(above_clusters["below_above_ratio"])  # Should be inf
    
    def test_detect_liquidation_cascade(self):
        """Test liquidation cascade detection."""
        # Detect cascade
        cascade = self.analyzer.detect_liquidation_cascade(
            self.base_liquidations, 
            self.prices
        )
        
        # Check result structure
        self.assertIn("cascade_score", cascade)
        self.assertIn("cascade_direction", cascade)
        self.assertIn("high_liquidation", cascade)
        self.assertIn("increasing_momentum", cascade)
        self.assertIn("positive_acceleration", cascade)
        self.assertIn("negative_correlation", cascade)
        self.assertIn("correlation", cascade)
        self.assertIn("liquidation_volatility", cascade)
        self.assertIn("max_liquidation", cascade)
        self.assertIn("avg_liquidation", cascade)
        
        # Check result types
        self.assertIsInstance(cascade["cascade_score"], float)
        self.assertIn(cascade["cascade_direction"], ["up", "down", "neutral"])
        self.assertIsInstance(cascade["high_liquidation"], bool)
        self.assertIsInstance(cascade["increasing_momentum"], bool)
        self.assertIsInstance(cascade["positive_acceleration"], bool)
        self.assertIsInstance(cascade["negative_correlation"], bool)
        self.assertIsInstance(cascade["correlation"], float)
        self.assertIsInstance(cascade["liquidation_volatility"], float)
        self.assertIsInstance(cascade["max_liquidation"], float)
        self.assertIsInstance(cascade["avg_liquidation"], float)
        
        # Test with custom lookback
        custom_cascade = self.analyzer.detect_liquidation_cascade(
            self.base_liquidations, 
            self.prices, 
            lookback=10
        )
        self.assertIsInstance(custom_cascade["cascade_score"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.detect_liquidation_cascade(
                self.base_liquidations[:50], 
                self.prices
            )
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.detect_liquidation_cascade(
                np.array([1, 2, 3, 4]), 
                np.array([1, 2, 3, 4])
            )
    
    def test_analyze_liquidation_impact(self):
        """Test liquidation impact analysis."""
        # Analyze impact
        impact = self.analyzer.analyze_liquidation_impact(
            self.base_liquidations, 
            self.prices, 
            self.volumes
        )
        
        # Check result structure
        self.assertIn("avg_liquidation_volume_ratio", impact)
        self.assertIn("high_impact_periods", impact)
        self.assertIn("price_impacts", impact)
        self.assertIn("avg_price_change_1", impact)
        self.assertIn("avg_price_change_3", impact)
        self.assertIn("impact_pattern", impact)
        self.assertIn("lagged_correlation", impact)
        
        # Check result types
        self.assertIsInstance(impact["avg_liquidation_volume_ratio"], float)
        self.assertIsInstance(impact["high_impact_periods"], int)
        self.assertIsInstance(impact["price_impacts"], list)
        self.assertIsInstance(impact["avg_price_change_1"], float)
        self.assertIsInstance(impact["avg_price_change_3"], float)
        self.assertIn(impact["impact_pattern"], ["mean_reversion", "trend_continuation", "neutral"])
        self.assertIsInstance(impact["lagged_correlation"], float)
        
        # Check price_impacts structure
        if impact["price_impacts"]:
            price_impact = impact["price_impacts"][0]
            self.assertIn("period", price_impact)
            self.assertIn("liquidation_volume_ratio", price_impact)
            self.assertIn("price_change_1", price_impact)
            self.assertIn("price_change_3", price_impact)
        
        # Test with custom lookback
        custom_impact = self.analyzer.analyze_liquidation_impact(
            self.base_liquidations, 
            self.prices, 
            self.volumes, 
            lookback=10
        )
        self.assertIsInstance(custom_impact["avg_liquidation_volume_ratio"], float)
        
        # Test with mismatched arrays
        with self.assertRaises(ModelError):
            self.analyzer.analyze_liquidation_impact(
                self.base_liquidations[:50], 
                self.prices, 
                self.volumes
            )
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_liquidation_impact(
                np.array([1, 2, 3, 4]), 
                np.array([1, 2, 3, 4]), 
                np.array([1, 2, 3, 4])
            )
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        cluster_analysis = {
            "clusters": [
                {
                    "price_level": 95.0,
                    "distance_percent": 5.0,
                    "liquidation_amount": 5000.0,
                    "is_support": True,
                    "is_resistance": False,
                    "strength": 2.5
                },
                {
                    "price_level": 105.0,
                    "distance_percent": 5.0,
                    "liquidation_amount": 3000.0,
                    "is_support": False,
                    "is_resistance": True,
                    "strength": 1.5
                }
            ],
            "total_clusters": 2,
            "nearest_support": {
                "price_level": 95.0,
                "distance_percent": 5.0,
                "liquidation_amount": 5000.0,
                "is_support": True,
                "is_resistance": False,
                "strength": 2.5
            },
            "nearest_resistance": {
                "price_level": 105.0,
                "distance_percent": 5.0,
                "liquidation_amount": 3000.0,
                "is_support": False,
                "is_resistance": True,
                "strength": 1.5
            },
            "total_liquidations": 10000.0,
            "liquidations_below": 6000.0,
            "liquidations_above": 4000.0,
            "above_below_ratio": 0.67,
            "below_above_ratio": 1.5,
            "imbalance_direction": "support",
            "imbalance_ratio": 1.5
        }
        
        cascade_analysis = {
            "cascade_score": 0.6,
            "cascade_direction": "down",
            "high_liquidation": True,
            "increasing_momentum": True,
            "positive_acceleration": False,
            "negative_correlation": True,
            "correlation": -0.5,
            "liquidation_volatility": 0.8,
            "max_liquidation": 5000.0,
            "avg_liquidation": 1000.0
        }
        
        impact_analysis = {
            "avg_liquidation_volume_ratio": 0.2,
            "high_impact_periods": 3,
            "price_impacts": [
                {
                    "period": 10,
                    "liquidation_volume_ratio": 0.5,
                    "price_change_1": -2.0,
                    "price_change_3": 1.0
                }
            ],
            "avg_price_change_1": -2.0,
            "avg_price_change_3": 1.0,
            "impact_pattern": "mean_reversion",
            "lagged_correlation": -0.3
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            cluster_analysis,
            cascade_analysis,
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
        self.assertIn("cluster", signal["components"])
        self.assertIn("cascade", signal["components"])
        self.assertIn("impact", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with bearish signals
        bearish_cluster = cluster_analysis.copy()
        bearish_cluster["imbalance_direction"] = "resistance"
        bearish_cluster["imbalance_ratio"] = 1.5
        
        bearish_cascade = cascade_analysis.copy()
        bearish_cascade["cascade_direction"] = "down"
        
        bearish_impact = impact_analysis.copy()
        bearish_impact["avg_price_change_1"] = 2.0
        bearish_impact["avg_price_change_3"] = -1.0
        bearish_impact["impact_pattern"] = "mean_reversion"
        
        bearish_signal = self.analyzer.generate_signal(
            bearish_cluster,
            bearish_cascade,
            bearish_impact
        )
        
        self.assertLess(bearish_signal["score"], 0)
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(self.df)
        
        # Check result structure
        self.assertIn("metrics", result)
        self.assertIn("cluster_analysis", result)
        self.assertIn("cascade_analysis", result)
        self.assertIn("impact_analysis", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check metrics structure
        self.assertIn("liquidation_amounts", result["metrics"])
        self.assertIn("normalized_liquidations", result["metrics"])
        
        # Check result types
        self.assertIsInstance(result["metrics"]["liquidation_amounts"], list)
        self.assertIsInstance(result["metrics"]["normalized_liquidations"], list)
        self.assertIsInstance(result["cluster_analysis"], dict)
        self.assertIsInstance(result["cascade_analysis"], dict)
        self.assertIsInstance(result["impact_analysis"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Check result shapes
        self.assertEqual(len(result["metrics"]["liquidation_amounts"]), len(self.base_liquidations))
        self.assertEqual(len(result["metrics"]["normalized_liquidations"]), len(self.base_liquidations))
        
        # Test standalone function
        standalone_result = analyze_liquidation_map(self.df)
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns
        df_missing = pd.DataFrame({
            "close": self.prices,
            "volume": self.volumes,
            "liquidation_amount": self.base_liquidations
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(df_missing)
        
        # Test with insufficient data
        short_df = pd.DataFrame({
            "close": self.prices[:2],
            "volume": self.volumes[:2],
            "liquidation_amount": self.base_liquidations[:2],
            "liquidation_price": self.liquidation_prices[:2]
        })
        
        with self.assertRaises(ModelError):
            self.analyzer.analyze(short_df)


if __name__ == "__main__":
    unittest.main()
