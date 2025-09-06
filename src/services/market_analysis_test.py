"""
Tests for the market analysis service module.
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil
import json

from src.services.market_analysis import MarketAnalysisService


class TestMarketAnalysisService(unittest.TestCase):
    """Test cases for the MarketAnalysisService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for results
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_analysis_service = MagicMock()
        self.mock_data_service = MagicMock()
        
        # Create patches
        self.analysis_service_patch = patch('src.services.market_analysis.AnalysisService')
        self.data_service_patch = patch('src.services.market_analysis.DataService')
        
        # Start patches
        self.mock_analysis_service_class = self.analysis_service_patch.start()
        self.mock_data_service_class = self.data_service_patch.start()
        
        # Configure mocks
        self.mock_analysis_service_class.return_value = self.mock_analysis_service
        self.mock_data_service_class.return_value = self.mock_data_service
        
        # Create service with mocked dependencies
        self.service = MarketAnalysisService(
            analysis_results_dir=self.temp_dir,
            worker_count=1
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop service if running
        if self.service.running:
            self.service.stop()
        
        # Stop patches
        self.analysis_service_patch.stop()
        self.data_service_patch.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.name, "MarketAnalysisService")
        self.assertIsNotNone(self.service.analysis_service)
        self.assertIsNotNone(self.service.data_service)
        self.assertEqual(self.service.analysis_results_dir, self.temp_dir)
        self.assertEqual(len(self.service.signal_processors), 0)  # Not registered until start
    
    def test_start_stop(self):
        """Test service start and stop."""
        # Test start
        self.assertTrue(self.service.start())
        self.assertTrue(self.service.running)
        self.mock_analysis_service.start.assert_called_once()
        self.mock_data_service.start.assert_called_once()
        self.assertGreater(len(self.service.signal_processors), 0)  # Signal processors registered
        
        # Test stop
        self.assertTrue(self.service.stop())
        self.assertFalse(self.service.running)
        self.mock_analysis_service.stop.assert_called_once()
        self.mock_data_service.stop.assert_called_once()
    
    def test_register_signal_processors(self):
        """Test registering signal processors."""
        self.service._register_signal_processors()
        
        # Check that all expected processors are registered
        expected_processors = [
            "rsi_volume", "divergence", "webtrend", "cross_asset",
            "liquidity", "funding", "open_interest", "volume_delta"
        ]
        
        for processor in expected_processors:
            self.assertIn(processor, self.service.signal_processors)
    
    @patch('src.services.market_analysis.pd.DataFrame')
    def test_run_analysis_for_asset(self, mock_df_class):
        """Test running analysis for an asset."""
        # Set up mock data
        mock_df = MagicMock()
        mock_df_class.return_value = mock_df
        
        # Configure mock data service
        self.mock_data_service.fetch_historical_data.return_value = mock_df
        self.mock_data_service.calculate_indicators.return_value = mock_df
        
        # Configure mock analysis service
        self.mock_analysis_service.analyze_rsi_volume.return_value = {
            "signal": "BUY",
            "final_score": 0.5,
            "targets": {"TP1": 100, "TP2": 110, "SL": 90}
        }
        self.mock_analysis_service.detect_divergence.return_value = {
            "bearish_divergence": False,
            "bullish_divergence": True,
            "score": 0.5
        }
        self.mock_analysis_service.calculate_webtrend.return_value = {
            "lines": {
                "status": True,
                "mid": 100,
                "upper": 110,
                "lower": 90
            }
        }
        self.mock_analysis_service.analyze_cross_asset.return_value = {
            "SOL": {
                "bias": 0.3,
                "correlation": 0.8
            }
        }
        
        # Register signal processors
        self.service._register_signal_processors()
        
        # Run analysis
        results = self.service._run_analysis_for_asset("BTC", "1h")
        
        # Check that data was fetched and indicators calculated
        self.mock_data_service.fetch_historical_data.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1h",
            use_cache=False
        )
        self.mock_data_service.calculate_indicators.assert_called_once()
        
        # Check that analyses were run
        self.mock_analysis_service.analyze_rsi_volume.assert_called_once()
        self.mock_analysis_service.detect_divergence.assert_called_once()
        self.mock_analysis_service.calculate_webtrend.assert_called_once()
        
        # Check results
        self.assertIn("rsi_volume", results)
        self.assertIn("divergence", results)
        self.assertIn("webtrend", results)
        self.assertIn("integrated", results)
    
    def test_process_task_analysis(self):
        """Test processing an analysis task."""
        # Mock _run_analysis_for_asset
        self.service._run_analysis_for_asset = MagicMock()
        
        # Create task
        task = {
            "type": "analysis",
            "asset": "BTC",
            "timeframe": "1h",
            "scheduled_time": datetime.now()
        }
        
        # Process task
        self.service._process_task(task)
        
        # Check that analysis was run
        self.service._run_analysis_for_asset.assert_called_once_with("BTC", "1h")
    
    def test_process_task_custom_analysis(self):
        """Test processing a custom analysis task."""
        # Create mock function
        mock_func = MagicMock()
        
        # Create task
        task = {
            "type": "custom_analysis",
            "analysis_func": mock_func,
            "args": [1, 2],
            "kwargs": {"a": 3},
            "scheduled_time": datetime.now()
        }
        
        # Process task
        self.service._process_task(task)
        
        # Check that function was called
        mock_func.assert_called_once_with(1, 2, a=3)
    
    def test_score_to_signal(self):
        """Test converting score to signal."""
        # Test various scores
        self.assertEqual(self.service._score_to_signal(0.7), "STRONG BUY")
        self.assertEqual(self.service._score_to_signal(0.5), "BUY")
        self.assertEqual(self.service._score_to_signal(0.0), "NEUTRAL")
        self.assertEqual(self.service._score_to_signal(-0.5), "SELL")
        self.assertEqual(self.service._score_to_signal(-0.7), "STRONG SELL")
    
    def test_integrate_signals(self):
        """Test integrating signals."""
        # Set up signal processors
        self.service._register_signal_processors()
        
        # Mock processors
        for name in self.service.signal_processors:
            setattr(self.service, f"_process_{name}_signal", MagicMock(return_value={
                "score": 0.5,
                "signal": "BUY",
                "confidence": 0.8
            }))
        
        # Set up results
        results = {
            "rsi_volume": {"signal": "BUY", "final_score": 0.5},
            "divergence": {"bullish_divergence": True, "score": 0.5},
            "webtrend": {"status": True, "score": 0.5},
            "cross_asset": {"SOL": {"bias": 0.3}},
            "liquidity": {"score": 0.0, "signal": "NEUTRAL"},
            "funding": {"score": 0.0, "signal": "NEUTRAL"},
            "open_interest": {"score": 0.0, "signal": "NEUTRAL"},
            "volume_delta": {"score": 0.3, "signal": "BUY"}
        }
        
        # Integrate signals
        integrated = self.service._integrate_signals(results, "BTC")
        
        # Check result
        self.assertIn("score", integrated)
        self.assertIn("signal", integrated)
        self.assertIn("components", integrated)
        self.assertEqual(len(integrated["components"]), len(results))
    
    def test_save_analysis_results(self):
        """Test saving analysis results."""
        # Set up results
        results = {
            "integrated": {
                "score": 0.5,
                "signal": "BUY",
                "components": {
                    "rsi_volume": {"score": 0.5, "signal": "BUY"}
                }
            }
        }
        
        # Save results
        self.service._save_analysis_results("BTC", "1h", results)
        
        # Check that file was created
        file_path = os.path.join(self.temp_dir, "BTC", "1h_latest.json")
        self.assertTrue(os.path.exists(file_path))
        
        # Check file contents
        with open(file_path, 'r') as f:
            saved_results = json.load(f)
        
        self.assertIn("integrated", saved_results)
        self.assertEqual(saved_results["integrated"]["score"], 0.5)
        self.assertEqual(saved_results["integrated"]["signal"], "BUY")
    
    def test_prepare_results_for_json(self):
        """Test preparing results for JSON serialization."""
        # Create test data with various types
        test_data = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "date": datetime(2023, 1, 1),
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2},
            "dataframe": pd.DataFrame({"a": [1, 2, 3]}),
            "series": pd.Series([1, 2, 3])
        }
        
        # Prepare for JSON
        result = self.service._prepare_results_for_json(test_data)
        
        # Check result
        self.assertEqual(result["int"], 42)
        self.assertEqual(result["float"], 3.14)
        self.assertEqual(result["date"], "2023-01-01T00:00:00")
        self.assertEqual(result["list"], [1, 2, 3])
        self.assertEqual(result["dict"], {"a": 1, "b": 2})
        self.assertEqual(result["dataframe"], "DataFrame object (not serialized)")
        self.assertEqual(result["series"], "Series object (not serialized)")
    
    def test_get_latest_analysis(self):
        """Test getting latest analysis."""
        # Set up analysis results
        self.service.analysis_results = {
            "BTC": {
                "1h": {"signal": "BUY"},
                "4h": {"signal": "SELL"}
            },
            "SOL": {
                "1h": {"signal": "NEUTRAL"}
            }
        }
        
        # Test getting specific timeframe
        result = self.service.get_latest_analysis("BTC", "1h")
        self.assertEqual(result, {"signal": "BUY"})
        
        # Test getting all timeframes
        result = self.service.get_latest_analysis("BTC")
        self.assertEqual(result, {"1h": {"signal": "BUY"}, "4h": {"signal": "SELL"}})
        
        # Test getting non-existent asset
        result = self.service.get_latest_analysis("ETH")
        self.assertEqual(result, {})
        
        # Test getting non-existent timeframe
        result = self.service.get_latest_analysis("BTC", "1d")
        self.assertEqual(result, {})
    
    def test_run_custom_analysis(self):
        """Test running custom analysis."""
        # Create mock function
        mock_func = MagicMock(return_value=42)
        
        # Run immediately
        result = self.service.run_custom_analysis(mock_func, 1, 2, a=3)
        self.assertEqual(result, 42)
        mock_func.assert_called_once_with(1, 2, a=3)
        
        # Schedule
        result = self.service.run_custom_analysis(mock_func, 1, 2, schedule=True, a=3)
        self.assertIsNone(result)
        self.assertEqual(len(self.service.task_queue), 1)
    
    def test_update_signal_weights(self):
        """Test updating signal weights."""
        # Set initial weights
        self.service.signal_weights = {"rsi_volume": 1.0, "divergence": 1.0}
        
        # Update weights
        self.service.update_signal_weights({"rsi_volume": 2.0, "webtrend": 1.5})
        
        # Check weights
        self.assertEqual(self.service.signal_weights["rsi_volume"], 2.0)
        self.assertEqual(self.service.signal_weights["divergence"], 1.0)
        self.assertEqual(self.service.signal_weights["webtrend"], 1.5)
    
    def test_check_service_health(self):
        """Test service health check."""
        # Mock health of dependent services
        self.mock_analysis_service.get_health.return_value = {"status": "running"}
        self.mock_data_service.get_health.return_value = {"status": "running"}
        
        # Check health
        health = self.service._check_service_health()
        
        # Check result
        self.assertIn("analysis_service_health", health)
        self.assertIn("data_service_health", health)
        self.assertIn("assets_analyzed", health)
        self.assertIn("signal_processors", health)
        self.assertIn("analysis_tasks", health)


if __name__ == "__main__":
    unittest.main()
