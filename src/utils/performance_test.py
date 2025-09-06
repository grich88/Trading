"""
Tests for the performance monitoring module.
"""

import time
import unittest
from unittest.mock import patch, MagicMock, call
import threading
import gc

from src.utils.performance import (
    performance_monitor,
    MemoryMonitor,
    get_memory_monitor,
    adaptive_batch_processing
)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for the performance_monitor decorator."""
    
    @patch('src.utils.performance.time.time')
    @patch('src.utils.performance.psutil.Process')
    @patch('src.utils.performance.logger')
    def test_performance_monitor(self, mock_logger, mock_process, mock_time):
        """Test performance_monitor decorator."""
        # Mock time.time() to return predictable values
        mock_time.side_effect = [100.0, 105.0]  # Start time, end time
        
        # Mock memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Create a decorated function
        @performance_monitor("test_operation")
        def test_func(x, y):
            return x + y
        
        # Call the function
        result = test_func(10, 20)
        
        # Check that the function returned the correct result
        self.assertEqual(result, 30)
        
        # Check that time.time() was called twice
        self.assertEqual(mock_time.call_count, 2)
        
        # Check that memory_info() was called
        self.assertTrue(mock_process.return_value.memory_info.called)
        
        # Check that logger.info() was called with performance metrics
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("test_operation", log_message)
        self.assertIn("5.0000 seconds", log_message)
    
    @patch('src.utils.performance.time.time')
    @patch('src.utils.performance.psutil.Process')
    @patch('src.utils.performance.logger')
    def test_performance_monitor_default_name(self, mock_logger, mock_process, mock_time):
        """Test performance_monitor decorator with default operation name."""
        # Mock time.time() to return predictable values
        mock_time.side_effect = [100.0, 105.0]  # Start time, end time
        
        # Mock memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Create a decorated function without specifying operation name
        @performance_monitor()
        def test_func(x, y):
            return x + y
        
        # Call the function
        result = test_func(10, 20)
        
        # Check that the function returned the correct result
        self.assertEqual(result, 30)
        
        # Check that logger.info() was called with the function name
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        self.assertIn("test_func", log_message)


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for the MemoryMonitor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Save original gc.collect to restore it later
        self.original_gc_collect = gc.collect
        
        # Mock gc.collect to avoid actual garbage collection during tests
        gc.collect = MagicMock()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original gc.collect
        gc.collect = self.original_gc_collect
    
    def test_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(
            threshold_percent=75,
            check_interval_seconds=30,
            initial_batch_size=50,
            min_batch_size=5,
            max_batch_size=500,
            gc_frequency=10
        )
        
        # Check that parameters are set correctly
        self.assertEqual(monitor.threshold_percent, 75)
        self.assertEqual(monitor.check_interval_seconds, 30)
        self.assertEqual(monitor.initial_batch_size, 50)
        self.assertEqual(monitor.min_batch_size, 5)
        self.assertEqual(monitor.max_batch_size, 500)
        self.assertEqual(monitor.gc_frequency, 10)
        
        # Check that current_batch_size is initialized to initial_batch_size
        self.assertEqual(monitor.current_batch_size, 50)
        
        # Check that monitoring is initially False
        self.assertFalse(monitor.monitoring)
    
    @patch('src.utils.performance.threading.Thread')
    @patch('src.utils.performance.logger')
    def test_start_stop_monitoring(self, mock_logger, mock_thread):
        """Test starting and stopping monitoring."""
        monitor = MemoryMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Check that monitoring is True
        self.assertTrue(monitor.monitoring)
        
        # Check that Thread was created and started
        mock_thread.assert_called_once_with(target=monitor._monitor_memory, daemon=True)
        mock_thread.return_value.start.assert_called_once()
        
        # Check that logger.info() was called
        mock_logger.info.assert_called_once_with("Memory monitoring started")
        
        # Reset mock_logger
        mock_logger.reset_mock()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Check that monitoring is False
        self.assertFalse(monitor.monitoring)
        
        # Check that logger.info() was called
        mock_logger.info.assert_called_once_with("Memory monitoring stopped")
    
    @patch('src.utils.performance.psutil.Process')
    @patch('src.utils.performance.psutil.virtual_memory')
    @patch('src.utils.performance.logger')
    def test_check_memory_below_threshold(self, mock_logger, mock_virtual_memory, mock_process):
        """Test _check_memory when memory usage is below threshold."""
        # Mock memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory
        mock_virtual_memory.return_value.percent = 50  # 50% usage
        
        # Create monitor with threshold of 80%
        monitor = MemoryMonitor(threshold_percent=80, initial_batch_size=100)
        
        # Call _check_memory
        metrics = monitor._check_memory()
        
        # Check that metrics were updated
        self.assertEqual(metrics["current_memory_mb"], 100)
        self.assertEqual(metrics["memory_percent"], 50)
        self.assertEqual(metrics["peak_memory_mb"], 100)
        
        # Check that batch size was not changed
        self.assertEqual(monitor.current_batch_size, 100)
        
        # Check that logger.debug() was called
        mock_logger.debug.assert_called_once()
    
    @patch('src.utils.performance.psutil.Process')
    @patch('src.utils.performance.psutil.virtual_memory')
    @patch('src.utils.performance.logger')
    def test_check_memory_above_threshold(self, mock_logger, mock_virtual_memory, mock_process):
        """Test _check_memory when memory usage is above threshold."""
        # Mock memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory
        mock_virtual_memory.return_value.percent = 90  # 90% usage
        
        # Create monitor with threshold of 80%
        monitor = MemoryMonitor(threshold_percent=80, initial_batch_size=100)
        
        # Call _check_memory
        metrics = monitor._check_memory()
        
        # Check that batch size was reduced
        self.assertEqual(monitor.current_batch_size, 75)  # 100 * 0.75 = 75
        
        # Check that batch_size_adjustments was incremented
        self.assertEqual(metrics["batch_size_adjustments"], 1)
        
        # Check that gc.collect() was called
        gc.collect.assert_called_once()
        
        # Check that gc_collections was incremented
        self.assertEqual(metrics["gc_collections"], 1)
        
        # Check that logger.warning() was called
        mock_logger.warning.assert_called_once()
    
    @patch('src.utils.performance.psutil.Process')
    @patch('src.utils.performance.psutil.virtual_memory')
    @patch('src.utils.performance.logger')
    def test_check_memory_well_below_threshold(self, mock_logger, mock_virtual_memory, mock_process):
        """Test _check_memory when memory usage is well below threshold."""
        # Mock memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory
        mock_virtual_memory.return_value.percent = 40  # 40% usage
        
        # Create monitor with threshold of 80%
        monitor = MemoryMonitor(threshold_percent=80, initial_batch_size=100)
        
        # Call _check_memory
        metrics = monitor._check_memory()
        
        # Check that batch size was increased
        self.assertEqual(monitor.current_batch_size, 125)  # 100 * 1.25 = 125
        
        # Check that batch_size_adjustments was incremented
        self.assertEqual(metrics["batch_size_adjustments"], 1)
        
        # Check that logger.debug() was called
        self.assertEqual(mock_logger.debug.call_count, 2)
    
    def test_get_batch_size(self):
        """Test get_batch_size method."""
        monitor = MemoryMonitor(initial_batch_size=100)
        self.assertEqual(monitor.get_batch_size(), 100)
    
    @patch('src.utils.performance.MemoryMonitor._check_memory')
    def test_get_metrics(self, mock_check_memory):
        """Test get_metrics method."""
        monitor = MemoryMonitor()
        mock_check_memory.return_value = {"test": "metrics"}
        
        metrics = monitor.get_metrics()
        
        # Check that _check_memory was called
        mock_check_memory.assert_called_once()
        
        # Check that metrics were returned
        self.assertEqual(metrics, {"test": "metrics"})
    
    def test_maybe_collect_garbage(self):
        """Test maybe_collect_garbage method."""
        monitor = MemoryMonitor(gc_frequency=3)
        
        # Call maybe_collect_garbage multiple times
        monitor.maybe_collect_garbage()  # operation_count = 1
        self.assertEqual(monitor.operation_count, 1)
        self.assertEqual(gc.collect.call_count, 0)
        
        monitor.maybe_collect_garbage()  # operation_count = 2
        self.assertEqual(monitor.operation_count, 2)
        self.assertEqual(gc.collect.call_count, 0)
        
        monitor.maybe_collect_garbage()  # operation_count = 3
        self.assertEqual(monitor.operation_count, 3)
        self.assertEqual(gc.collect.call_count, 1)
        
        monitor.maybe_collect_garbage()  # operation_count = 4
        self.assertEqual(monitor.operation_count, 4)
        self.assertEqual(gc.collect.call_count, 1)


class TestAdaptiveBatchProcessing(unittest.TestCase):
    """Test cases for the adaptive_batch_processing function."""
    
    def test_adaptive_batch_processing(self):
        """Test adaptive_batch_processing function."""
        # Create a mock memory monitor
        mock_monitor = MagicMock()
        mock_monitor.get_batch_size.return_value = 10
        
        # Create a mock process function
        def process_func(batch):
            return [x * 2 for x in batch]
        
        # Create test items
        items = list(range(25))
        
        # Call adaptive_batch_processing
        results = adaptive_batch_processing(items, process_func, mock_monitor)
        
        # Check that results are correct
        self.assertEqual(results, [x * 2 for x in items])
        
        # Check that get_batch_size was called multiple times
        self.assertEqual(mock_monitor.get_batch_size.call_count, 3)
        
        # Check that maybe_collect_garbage was called multiple times
        self.assertEqual(mock_monitor.maybe_collect_garbage.call_count, 3)


class TestGlobalMemoryMonitor(unittest.TestCase):
    """Test cases for the global memory monitor instance."""
    
    def test_get_memory_monitor(self):
        """Test get_memory_monitor function."""
        # Get the global memory monitor instance
        monitor1 = get_memory_monitor()
        monitor2 = get_memory_monitor()
        
        # Check that both calls return the same instance
        self.assertIs(monitor1, monitor2)
        
        # Check that it's a MemoryMonitor instance
        self.assertIsInstance(monitor1, MemoryMonitor)


if __name__ == "__main__":
    unittest.main()
