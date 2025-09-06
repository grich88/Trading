"""
Tests for the Gamma Exposure / Option Flow Analysis module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.gamma_exposure_analyzer import (
    GammaExposureAnalyzer,
    calculate_gamma_exposure,
    analyze_gamma_exposure
)


class TestGammaExposureAnalyzer(unittest.TestCase):
    """Test cases for the GammaExposureAnalyzer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create analyzer with default parameters
        self.analyzer = GammaExposureAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.spot_price = 100
        
        # Generate strikes around spot
        self.strikes = np.arange(80, 121, 2.5)
        
        # Generate options data
        options_data = []
        for strike in self.strikes:
            # Generate call data
            moneyness = self.spot_price / strike
            gamma = np.exp(-0.5 * ((moneyness - 1) / 0.1) ** 2) * np.random.uniform(0.01, 0.05)
            oi = np.random.randint(100, 10000) * (1 + np.exp(-abs(strike - self.spot_price) / 5))
            
            options_data.append({
                "strike": strike,
                "gamma": gamma,
                "open_interest": oi,
                "option_type": "CALL"
            })
            
            # Generate put data
            gamma = np.exp(-0.5 * ((1 / moneyness - 1) / 0.1) ** 2) * np.random.uniform(0.01, 0.05)
            oi = np.random.randint(100, 10000) * (1 + np.exp(-abs(strike - self.spot_price) / 5))
            
            options_data.append({
                "strike": strike,
                "gamma": gamma,
                "open_interest": oi,
                "option_type": "PUT"
            })
        
        self.options_df = pd.DataFrame(options_data)
        
        # Generate option trades
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
                    "strike": np.random.choice(self.strikes),
                    "trade_type": np.random.choice(["BUY", "SELL"], p=[0.6, 0.4])
                })
        
        self.trades_df = pd.DataFrame(option_trades)
        
        # Generate price history
        self.price_history = pd.DataFrame({
            "close": self.spot_price + np.cumsum(np.random.normal(0, 0.5, 100))
        })
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer = GammaExposureAnalyzer()
        self.assertEqual(analyzer.lookback_period, 20)  # Default lookback period
        
        # Test custom initialization
        custom_analyzer = GammaExposureAnalyzer(
            lookback_period=15,
            signal_threshold=0.4,
            flip_threshold=100.0,
            flow_threshold=10000.0
        )
        self.assertEqual(custom_analyzer.lookback_period, 15)
        self.assertEqual(custom_analyzer.signal_threshold, 0.4)
        self.assertEqual(custom_analyzer.flip_threshold, 100.0)
        self.assertEqual(custom_analyzer.flow_threshold, 10000.0)
    
    def test_calculate_gamma_exposure(self):
        """Test gamma exposure calculation."""
        # Calculate gamma exposure
        gamma_exposure = self.analyzer.calculate_gamma_exposure(self.options_df, self.spot_price)
        
        # Check result structure
        self.assertIn("strikes", gamma_exposure)
        self.assertIn("gammas", gamma_exposure)
        self.assertIn("cumulative_gamma", gamma_exposure)
        self.assertIn("total_gamma", gamma_exposure)
        self.assertIn("spot_gamma", gamma_exposure)
        self.assertIn("gamma_flip_point", gamma_exposure)
        self.assertIn("dealer_position", gamma_exposure)
        self.assertIn("distance_to_flip", gamma_exposure)
        self.assertIn("flip_percentage", gamma_exposure)
        
        # Check result types
        self.assertIsInstance(gamma_exposure["strikes"], list)
        self.assertIsInstance(gamma_exposure["gammas"], list)
        self.assertIsInstance(gamma_exposure["cumulative_gamma"], list)
        self.assertIsInstance(gamma_exposure["total_gamma"], float)
        self.assertIsInstance(gamma_exposure["spot_gamma"], float)
        self.assertIn(gamma_exposure["dealer_position"], ["long_gamma", "short_gamma"])
        
        # Check result consistency
        self.assertEqual(len(gamma_exposure["strikes"]), len(gamma_exposure["gammas"]))
        self.assertEqual(len(gamma_exposure["strikes"]), len(gamma_exposure["cumulative_gamma"]))
        
        # Test with missing columns
        bad_df = self.options_df.drop(columns=["gamma"])
        with self.assertRaises(ModelError):
            self.analyzer.calculate_gamma_exposure(bad_df, self.spot_price)
        
        # Test standalone function
        standalone_result = calculate_gamma_exposure(self.options_df, self.spot_price)
        self.assertEqual(standalone_result["total_gamma"], gamma_exposure["total_gamma"])
    
    def test_analyze_option_flow(self):
        """Test option flow analysis."""
        # Analyze option flow
        option_flow = self.analyzer.analyze_option_flow(self.trades_df)
        
        # Check result structure
        self.assertIn("total_call_volume", option_flow)
        self.assertIn("total_put_volume", option_flow)
        self.assertIn("call_put_ratio", option_flow)
        self.assertIn("total_call_premium", option_flow)
        self.assertIn("total_put_premium", option_flow)
        self.assertIn("premium_ratio", option_flow)
        self.assertIn("buy_sell_ratio", option_flow)
        self.assertIn("large_trades", option_flow)
        self.assertIn("unusual_activity", option_flow)
        self.assertIn("flow_sentiment", option_flow)
        
        # Check result types
        self.assertIsInstance(option_flow["total_call_volume"], int)
        self.assertIsInstance(option_flow["total_put_volume"], int)
        self.assertIsInstance(option_flow["total_call_premium"], float)
        self.assertIsInstance(option_flow["total_put_premium"], float)
        self.assertIsInstance(option_flow["large_trades"], list)
        self.assertIsInstance(option_flow["unusual_activity"], bool)
        self.assertIn(option_flow["flow_sentiment"], ["bullish", "bearish", "neutral"])
        
        # Test with empty data
        empty_df = pd.DataFrame(columns=self.trades_df.columns)
        empty_flow = self.analyzer.analyze_option_flow(empty_df)
        self.assertEqual(empty_flow["total_call_volume"], 0)
        self.assertEqual(empty_flow["total_put_volume"], 0)
        self.assertEqual(empty_flow["call_put_ratio"], 1.0)
        
        # Test with missing columns
        bad_df = self.trades_df.drop(columns=["volume"])
        with self.assertRaises(ModelError):
            self.analyzer.analyze_option_flow(bad_df)
        
        # Test extreme cases
        # All calls
        all_calls = self.trades_df.copy()
        all_calls["option_type"] = "CALL"
        calls_flow = self.analyzer.analyze_option_flow(all_calls)
        self.assertIsNone(calls_flow["call_put_ratio"])  # Should be inf
        
        # All puts
        all_puts = self.trades_df.copy()
        all_puts["option_type"] = "PUT"
        puts_flow = self.analyzer.analyze_option_flow(all_puts)
        self.assertEqual(puts_flow["call_put_ratio"], 0.0)
    
    def test_analyze_gamma_dynamics(self):
        """Test gamma dynamics analysis."""
        # Create gamma history
        gamma_history = []
        prices = self.price_history["close"].values[-10:]
        
        for i, price in enumerate(prices):
            # Simulate changing gamma exposure
            gamma_result = self.analyzer.calculate_gamma_exposure(self.options_df, price)
            gamma_history.append(gamma_result)
        
        # Analyze gamma dynamics
        gamma_dynamics = self.analyzer.analyze_gamma_dynamics(gamma_history, prices)
        
        # Check result structure
        self.assertIn("gamma_momentum", gamma_dynamics)
        self.assertIn("flip_stability", gamma_dynamics)
        self.assertIn("gamma_effectiveness", gamma_dynamics)
        self.assertIn("flip_tests", gamma_dynamics)
        self.assertIn("successful_tests", gamma_dynamics)
        self.assertIn("gamma_regime", gamma_dynamics)
        self.assertIn("gamma_squeeze", gamma_dynamics)
        self.assertIn("current_gamma", gamma_dynamics)
        self.assertIn("avg_gamma", gamma_dynamics)
        
        # Check result types
        self.assertIsInstance(gamma_dynamics["gamma_momentum"], float)
        self.assertIsInstance(gamma_dynamics["flip_stability"], float)
        self.assertIsInstance(gamma_dynamics["gamma_effectiveness"], float)
        self.assertIsInstance(gamma_dynamics["flip_tests"], int)
        self.assertIsInstance(gamma_dynamics["successful_tests"], int)
        self.assertIn(gamma_dynamics["gamma_regime"], ["high_gamma", "low_gamma", "normal_gamma"])
        self.assertIsInstance(gamma_dynamics["gamma_squeeze"], bool)
        self.assertIsInstance(gamma_dynamics["current_gamma"], float)
        self.assertIsInstance(gamma_dynamics["avg_gamma"], float)
        
        # Check value ranges
        self.assertTrue(0 <= gamma_dynamics["flip_stability"] <= 1)
        self.assertTrue(0 <= gamma_dynamics["gamma_effectiveness"] <= 1)
        
        # Test with insufficient data
        with self.assertRaises(ModelError):
            self.analyzer.analyze_gamma_dynamics([gamma_history[0]], prices[:1])
        
        # Test with mismatched lengths
        with self.assertRaises(ModelError):
            self.analyzer.analyze_gamma_dynamics(gamma_history, prices[:5])
    
    def test_generate_signal(self):
        """Test signal generation."""
        # Create sample analysis results
        gamma_exposure = {
            "total_gamma": -50000,
            "spot_gamma": -1000,
            "gamma_flip_point": 98.5,
            "dealer_position": "short_gamma",
            "distance_to_flip": 1.5,
            "flip_percentage": 1.5
        }
        
        option_flow = {
            "total_call_volume": 10000,
            "total_put_volume": 5000,
            "call_put_ratio": 2.0,
            "flow_sentiment": "bullish",
            "unusual_activity": True
        }
        
        gamma_dynamics = {
            "gamma_momentum": -1000,
            "flip_stability": 0.8,
            "gamma_effectiveness": 0.7,
            "gamma_regime": "low_gamma",
            "gamma_squeeze": False,
            "current_gamma": -50000,
            "avg_gamma": -30000
        }
        
        # Generate signal
        signal = self.analyzer.generate_signal(
            gamma_exposure,
            option_flow,
            gamma_dynamics
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
        self.assertIn("gamma_exposure", signal["components"])
        self.assertIn("option_flow", signal["components"])
        self.assertIn("gamma_dynamics", signal["components"])
        
        # Check value ranges
        self.assertTrue(-1.0 <= signal["score"] <= 1.0)
        self.assertTrue(0.0 <= signal["confidence"] <= 1.0)
        
        # Test with gamma squeeze
        squeeze_dynamics = gamma_dynamics.copy()
        squeeze_dynamics["gamma_squeeze"] = True
        squeeze_signal = self.analyzer.generate_signal(
            gamma_exposure,
            option_flow,
            squeeze_dynamics
        )
        # Gamma squeeze should increase bullish signal
        self.assertGreater(squeeze_signal["components"]["gamma_exposure"], 
                          signal["components"]["gamma_exposure"])
    
    def test_analyze(self):
        """Test complete analysis."""
        # Perform analysis
        result = self.analyzer.analyze(
            self.options_df,
            self.trades_df,
            self.price_history
        )
        
        # Check result structure
        self.assertIn("spot_price", result)
        self.assertIn("gamma_exposure", result)
        self.assertIn("option_flow", result)
        self.assertIn("gamma_dynamics", result)
        self.assertIn("signal", result)
        self.assertIn("score", result)
        self.assertIn("components", result)
        self.assertIn("confidence", result)
        self.assertIn("timestamp", result)
        
        # Check result types
        self.assertIsInstance(result["spot_price"], float)
        self.assertIsInstance(result["gamma_exposure"], dict)
        self.assertIsInstance(result["option_flow"], dict)
        self.assertIsInstance(result["gamma_dynamics"], dict)
        self.assertIsInstance(result["signal"], str)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["components"], dict)
        self.assertIsInstance(result["confidence"], float)
        self.assertIsInstance(result["timestamp"], str)
        
        # Test with gamma history
        gamma_history = []
        for i in range(5):
            gamma_result = self.analyzer.calculate_gamma_exposure(
                self.options_df, 
                self.price_history["close"].iloc[-(i+1)]
            )
            gamma_history.append(gamma_result)
        
        result_with_history = self.analyzer.analyze(
            self.options_df,
            self.trades_df,
            self.price_history,
            gamma_history=gamma_history[::-1]  # Reverse to chronological order
        )
        
        # Should have more meaningful gamma dynamics with history
        self.assertNotEqual(result_with_history["gamma_dynamics"]["gamma_momentum"], 0.0)
        
        # Test standalone function
        standalone_result = analyze_gamma_exposure(
            self.options_df,
            self.trades_df,
            self.price_history
        )
        self.assertEqual(standalone_result["signal"], result["signal"])
        self.assertEqual(standalone_result["score"], result["score"])
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with missing columns in price history
        bad_price_history = pd.DataFrame({"price": [100, 101, 102]})
        with self.assertRaises(ModelError):
            self.analyzer.analyze(
                self.options_df,
                self.trades_df,
                bad_price_history
            )
        
        # Test with empty options data
        empty_options = pd.DataFrame(columns=self.options_df.columns)
        result = self.analyzer.analyze(
            empty_options,
            self.trades_df,
            self.price_history
        )
        # Should still work but with empty gamma exposure
        self.assertEqual(len(result["gamma_exposure"]["strikes"]), 0)


if __name__ == "__main__":
    unittest.main()
