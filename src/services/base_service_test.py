"""
Tests for the base service module.
"""

import unittest
from unittest.mock import patch, MagicMock
import threading
import time

from src.services.base_service import BaseService, LongRunningService


class TestBaseService(unittest.TestCase):
    """Test cases for the BaseService class."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = BaseService("TestService")
        self.assertEqual(service.name, "TestService")
        self.assertFalse(service.running)
        self.assertIsNone(service.start_time)
    
    def test_start_stop(self):
        """Test service start and stop."""
        service = BaseService("TestService")
        
        # Test start
        self.assertTrue(service.start())
        self.assertTrue(service.running)
        self.assertIsNotNone(service.start_time)
        
        # Test stop
        self.assertTrue(service.stop())
        self.assertFalse(service.running)
    
    def test_health_check(self):
        """Test health check."""
        service = BaseService("TestService")
        service.start()
        
        # Check health
        health = service.check_health()
        self.assertEqual(health["status"], "running")
        self.assertGreaterEqual(health["uptime_seconds"], 0)
        
        service.stop()
    
    def test_process_batch(self):
        """Test batch processing."""
        service = BaseService("TestService", enable_memory_monitoring=True)
        service.start()
        
        # Define process function
        def process_func(items):
            return [item * 2 for item in items]
        
        # Process items
        items = list(range(10))
        results = service.process_batch(items, process_func)
        
        # Check results
        self.assertEqual(results, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        
        service.stop()
    
    @patch("src.services.base_service.performance_monitor")
    def test_execute_with_monitoring(self, mock_monitor):
        """Test execute with monitoring."""
        service = BaseService("TestService")
        
        # Define test function
        def test_func(a, b):
            return a + b
        
        # Set up mock
        mock_decorator = MagicMock()
        mock_monitor.return_value = mock_decorator
        mock_wrapped = MagicMock()
        mock_decorator.return_value = mock_wrapped
        mock_wrapped.return_value = 42
        
        # Execute function
        result = service.execute_with_monitoring(test_func, 10, 32)
        
        # Check result
        self.assertEqual(result, 42)
        mock_monitor.assert_called_once()
        mock_decorator.assert_called_once()
        mock_wrapped.assert_called_once()
    
    @patch("src.services.base_service.exception_handler")
    def test_execute_with_error_handling(self, mock_handler):
        """Test execute with error handling."""
        service = BaseService("TestService")
        
        # Define test function
        def test_func(a, b):
            return a / b
        
        # Set up mock
        mock_decorator = MagicMock()
        mock_handler.return_value = mock_decorator
        mock_wrapped = MagicMock()
        mock_decorator.return_value = mock_wrapped
        mock_wrapped.return_value = 5
        
        # Execute function
        result = service.execute_with_error_handling(test_func, 10, 2)
        
        # Check result
        self.assertEqual(result, 5)
        mock_handler.assert_called_once_with(log_exception=True)
        mock_decorator.assert_called_once()
        mock_wrapped.assert_called_once()


class TestLongRunningService(unittest.TestCase):
    """Test cases for the LongRunningService class."""
    
    def test_initialization(self):
        """Test service initialization."""
        service = LongRunningService("TestService", worker_count=2)
        self.assertEqual(service.name, "TestService")
        self.assertEqual(service.worker_count, 2)
        self.assertEqual(len(service.worker_threads), 0)
        self.assertEqual(len(service.task_queue), 0)
    
    def test_start_stop(self):
        """Test service start and stop."""
        service = LongRunningService("TestService", worker_count=2)
        
        # Test start
        self.assertTrue(service.start())
        self.assertTrue(service.running)
        self.assertEqual(len(service.worker_threads), 2)
        
        # Test stop
        self.assertTrue(service.stop())
        self.assertFalse(service.running)
        self.assertEqual(len(service.worker_threads), 0)
    
    def test_add_task(self):
        """Test adding tasks to the queue."""
        service = LongRunningService("TestService", worker_count=1)
        
        # Add tasks
        service.add_task("Task 1")
        service.add_task("Task 2")
        
        # Check queue size
        self.assertEqual(service.get_queue_size(), 2)
    
    def test_process_task(self):
        """Test task processing."""
        # Create a subclass that implements _process_task
        class TestService(LongRunningService):
            def __init__(self):
                super().__init__("TestService", worker_count=1)
                self.processed_tasks = []
            
            def _process_task(self, task):
                self.processed_tasks.append(task)
        
        # Create service
        service = TestService()
        service.start()
        
        # Add tasks
        service.add_task("Task 1")
        service.add_task("Task 2")
        
        # Wait for tasks to be processed
        time.sleep(0.5)
        
        # Check processed tasks
        self.assertIn("Task 1", service.processed_tasks)
        self.assertIn("Task 2", service.processed_tasks)
        
        # Check queue size
        self.assertEqual(service.get_queue_size(), 0)
        
        service.stop()


if __name__ == "__main__":
    unittest.main()
