"""
Tests for the data collection service module.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
import os
import tempfile
import shutil

from src.services.data_collection import DataCollectionService


class TestDataCollectionService(unittest.TestCase):
    """Test cases for the DataCollectionService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            self.service = DataCollectionService(
                data_dir=self.temp_dir,
                default_timeframe='1h',
                collection_interval=60,
                max_concurrent_requests=5,
                rate_limit_per_minute=60
            )
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop service if running
        if self.service.running:
            self.service.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.name, "DataCollectionService")
        self.assertEqual(self.service.data_dir, self.temp_dir)
        self.assertEqual(self.service.default_timeframe, '1h')
        self.assertEqual(self.service.collection_interval, 60)
        self.assertEqual(self.service.max_concurrent_requests, 5)
        self.assertEqual(self.service.rate_limit_per_minute, 60)
        self.assertIsNotNone(self.service.data_service)
        self.assertEqual(len(self.service.collection_targets), 0)
    
    @patch('src.services.data_collection.DataService')
    def test_start_stop(self, mock_data_service):
        """Test service start and stop."""
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Test start
        self.assertTrue(service.start())
        self.assertTrue(service.running)
        mock_data_service_instance.start.assert_called_once()
        
        # Test stop
        self.assertTrue(service.stop())
        self.assertFalse(service.running)
        mock_data_service_instance.stop.assert_called_once()
    
    def test_add_remove_collection_target(self):
        """Test adding and removing collection targets."""
        # Add a target
        self.service.add_collection_target("BTC/USDT", ["1h", "4h"])
        self.assertEqual(len(self.service.collection_targets), 1)
        self.assertIn("BTC/USDT", self.service.collection_targets)
        self.assertEqual(self.service.collection_targets["BTC/USDT"]["timeframes"], {"1h", "4h"})
        
        # Add another timeframe to the same target
        self.service.add_collection_target("BTC/USDT", ["1d"])
        self.assertEqual(len(self.service.collection_targets), 1)
        self.assertEqual(self.service.collection_targets["BTC/USDT"]["timeframes"], {"1h", "4h", "1d"})
        
        # Add another target
        self.service.add_collection_target("ETH/USDT", ["1h"])
        self.assertEqual(len(self.service.collection_targets), 2)
        self.assertIn("ETH/USDT", self.service.collection_targets)
        
        # Remove a timeframe
        self.service.remove_collection_target("BTC/USDT", ["4h"])
        self.assertEqual(self.service.collection_targets["BTC/USDT"]["timeframes"], {"1h", "1d"})
        
        # Remove a target completely
        self.service.remove_collection_target("ETH/USDT")
        self.assertEqual(len(self.service.collection_targets), 1)
        self.assertNotIn("ETH/USDT", self.service.collection_targets)
    
    @patch('src.services.data_collection.asyncio.run')
    def test_collect_all_data(self, mock_asyncio_run):
        """Test collecting all data."""
        # Set up mock data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='H'))
        
        # Set up mock result
        mock_result = {
            "BTC/USDT": {
                "1h": mock_df,
                "4h": mock_df
            },
            "ETH/USDT": {
                "1h": mock_df
            }
        }
        mock_asyncio_run.return_value = mock_result
        
        # Add collection targets
        self.service.add_collection_target("BTC/USDT", ["1h", "4h"])
        self.service.add_collection_target("ETH/USDT", ["1h"])
        
        # Collect data
        results = self.service.collect_all_data()
        
        # Check results
        self.assertIn("BTC/USDT", results)
        self.assertIn("ETH/USDT", results)
        self.assertIn("1h", results["BTC/USDT"])
        self.assertIn("4h", results["BTC/USDT"])
        self.assertIn("1h", results["ETH/USDT"])
        self.assertTrue(results["BTC/USDT"]["1h"]["success"])
        self.assertEqual(results["BTC/USDT"]["1h"]["data_points"], 3)
        
        # Check statistics
        self.assertEqual(self.service.collection_stats["total_collections"], 1)
        self.assertEqual(self.service.collection_stats["successful_collections"], 1)
        self.assertEqual(self.service.collection_stats["total_data_points"], 9)  # 3 points * 3 dataframes
    
    @patch('src.services.data_collection.DataService')
    def test_get_latest_data(self, mock_data_service):
        """Test getting latest data."""
        # Set up mock data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='H'))
        
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        mock_data_service_instance.load_data.return_value = mock_df
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Get latest data
        result = service.get_latest_data("BTC/USDT", "1h", 2)
        
        # Check result
        self.assertEqual(len(result), 2)  # Should return only 2 rows
        mock_data_service_instance.load_data.assert_called_once_with("BTC_USDT_1h", "historical")
    
    @patch('src.services.data_collection.DataService')
    def test_get_latest_data_error(self, mock_data_service):
        """Test getting latest data with error."""
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        mock_data_service_instance.load_data.side_effect = Exception("File not found")
        
        # Set up mock for collect_historical_data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='H'))
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Mock collect_historical_data
        service.collect_historical_data = MagicMock(return_value=mock_df)
        
        # Get latest data
        result = service.get_latest_data("BTC/USDT", "1h", 2)
        
        # Check result
        self.assertEqual(len(result), 3)  # Should return all rows from mock_df
        mock_data_service_instance.load_data.assert_called_once()
        service.collect_historical_data.assert_called_once()
    
    @patch('src.services.data_collection.DataService')
    def test_collect_historical_data(self, mock_data_service):
        """Test collecting historical data."""
        # Set up mock data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='H'))
        
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        mock_data_service_instance.fetch_historical_data.return_value = mock_df
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Collect historical data
        result = service.collect_historical_data(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-01-02",
            save_data=True
        )
        
        # Check result
        self.assertEqual(len(result), 3)
        mock_data_service_instance.fetch_historical_data.assert_called_once_with(
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-01-02",
            use_cache=False
        )
        mock_data_service_instance.save_data.assert_called_once()
    
    def test_check_service_health(self):
        """Test service health check."""
        # Add collection targets
        self.service.add_collection_target("BTC/USDT", ["1h", "4h"])
        self.service.add_collection_target("ETH/USDT", ["1h"])
        
        # Check health
        health = self.service._check_service_health()
        
        # Check result
        self.assertEqual(health["collection_targets"], 2)
        self.assertIn("collection_stats", health)
        self.assertIn("data_service_health", health)


class TestDataCollectionServiceAsync(unittest.IsolatedAsyncioTestCase):
    """Test cases for async methods in DataCollectionService."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Create a temporary directory for data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            self.service = DataCollectionService(
                data_dir=self.temp_dir,
                default_timeframe='1h',
                collection_interval=60,
                max_concurrent_requests=5,
                rate_limit_per_minute=60
            )
        
        # Set up request semaphore
        self.service.request_semaphore = asyncio.Semaphore(5)
        self.service.request_timestamps = []
    
    async def asyncTearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_apply_rate_limit(self):
        """Test rate limiting."""
        # Add timestamps to simulate requests
        now = time.time()
        for i in range(self.service.rate_limit_per_minute - 1):
            self.service.request_timestamps.append(now - i)
        
        # Apply rate limit (should not wait)
        start_time = time.time()
        await self.service._apply_rate_limit()
        duration = time.time() - start_time
        
        # Should be quick
        self.assertLess(duration, 0.1)
        
        # Add one more timestamp to hit the limit
        self.service.request_timestamps.append(now)
        
        # Apply rate limit with patched sleep
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await self.service._apply_rate_limit()
            mock_sleep.assert_called_once()
    
    @patch('src.services.data_collection.DataService')
    async def test_collect_symbol_data_async(self, mock_data_service):
        """Test collecting symbol data asynchronously."""
        # Set up mock data
        mock_ohlcv = [
            [1609459200000, 100, 105, 95, 102, 1000],
            [1609462800000, 101, 106, 96, 103, 1100],
            [1609466400000, 102, 107, 97, 104, 1200]
        ]
        
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        
        # Mock async exchange client
        mock_client = AsyncMock()
        mock_client.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv)
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Set up async exchange clients
        service.async_exchange_clients = {"binance": mock_client}
        service.request_semaphore = asyncio.Semaphore(5)
        service.data_service = mock_data_service_instance
        
        # Collect symbol data
        symbol, timeframe, df = await service._collect_symbol_data_async(
            symbol="BTC/USDT",
            timeframe="1h",
            exchange="binance",
            collect_indicators=True
        )
        
        # Check result
        self.assertEqual(symbol, "BTC/USDT")
        self.assertEqual(timeframe, "1h")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        mock_client.fetch_ohlcv.assert_called_once()
        mock_data_service_instance.calculate_indicators.assert_called_once()
        mock_data_service_instance.save_data.assert_called_once()
    
    @patch('src.services.data_collection.DataService')
    async def test_collect_all_data_async(self, mock_data_service):
        """Test collecting all data asynchronously."""
        # Set up mock data
        mock_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='H'))
        
        # Set up mocks
        mock_data_service_instance = MagicMock()
        mock_data_service.return_value = mock_data_service_instance
        
        # Create service with mocked dependencies
        with patch('src.services.data_collection.ccxt', create=True), \
             patch('src.services.data_collection.ccxt_async', create=True):
            service = DataCollectionService(data_dir=self.temp_dir)
        
        # Mock _collect_symbol_data_async
        service._collect_symbol_data_async = AsyncMock(
            return_value=("BTC/USDT", "1h", mock_df)
        )
        
        # Add collection targets
        service.add_collection_target("BTC/USDT", ["1h", "4h"])
        service.add_collection_target("ETH/USDT", ["1h"])
        
        # Collect all data
        results = await service._collect_all_data_async()
        
        # Check result
        self.assertIn("BTC/USDT", results)
        self.assertIn("ETH/USDT", results)
        self.assertEqual(service._collect_symbol_data_async.call_count, 3)  # 2 for BTC/USDT, 1 for ETH/USDT


if __name__ == "__main__":
    unittest.main()
