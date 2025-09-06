"""
Tests for the logging service module.
"""

import os
import sys
import json
import unittest
import logging
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import io

from src.utils.logging_service import (
    LoggingService, 
    get_logger, 
    JsonFormatter, 
    log_function_call,
    configure_root_logger
)


class TestLoggingService(unittest.TestCase):
    """Test cases for the LoggingService class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.test_dir, "test.log")
        
        # Reset singleton instance
        LoggingService._instance = None
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_singleton_pattern(self):
        """Test that LoggingService follows the singleton pattern."""
        logger1 = LoggingService(name="TestLogger1")
        logger2 = LoggingService(name="TestLogger2")
        
        # Both instances should be the same object
        self.assertIs(logger1, logger2)
        
        # The name should be from the first initialization
        self.assertEqual(logger1.name, "TestLogger1")
    
    def test_log_levels(self):
        """Test setting different log levels."""
        for level_name, level_value in [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL)
        ]:
            # Reset singleton instance
            LoggingService._instance = None
            
            logger = LoggingService(name="TestLogger", log_level=level_name)
            self.assertEqual(logger.log_level, level_value)
    
    def test_console_logging(self):
        """Test logging to console."""
        with patch("logging.StreamHandler") as mock_handler:
            mock_handler.return_value = MagicMock()
            
            # Reset singleton instance
            LoggingService._instance = None
            
            logger = LoggingService(
                name="TestLogger",
                log_to_console=True,
                log_to_file=False
            )
            
            # Check that StreamHandler was called
            mock_handler.assert_called_once()
            
            # Check that handler was added to logger
            self.assertEqual(len(logger.logger.handlers), 1)
    
    def test_file_logging(self):
        """Test logging to file."""
        # Reset singleton instance
        LoggingService._instance = None
        
        logger = LoggingService(
            name="TestLogger",
            log_to_console=False,
            log_to_file=True,
            log_file_path=self.log_file_path
        )
        
        # Log a message
        test_message = "Test log message"
        logger.info(test_message)
        
        # Check that file was created
        self.assertTrue(os.path.exists(self.log_file_path))
        
        # Check that message was written to file
        with open(self.log_file_path, "r") as f:
            log_content = f.read()
            self.assertIn(test_message, log_content)
    
    def test_log_methods(self):
        """Test all logging methods."""
        # Reset singleton instance
        LoggingService._instance = None
        
        logger = LoggingService(
            name="TestLogger",
            log_to_console=False,
            log_to_file=True,
            log_file_path=self.log_file_path
        )
        
        # Test all log methods
        test_messages = {
            "debug": "Debug message",
            "info": "Info message",
            "warning": "Warning message",
            "error": "Error message",
            "critical": "Critical message"
        }
        
        for method, message in test_messages.items():
            getattr(logger, method)(message)
        
        # Check that all messages were written to file
        with open(self.log_file_path, "r") as f:
            log_content = f.read()
            for message in test_messages.values():
                self.assertIn(message, log_content)
    
    def test_exception_logging(self):
        """Test exception logging."""
        # Reset singleton instance
        LoggingService._instance = None
        
        logger = LoggingService(
            name="TestLogger",
            log_to_console=False,
            log_to_file=True,
            log_file_path=self.log_file_path
        )
        
        # Log an exception
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.exception(f"An error occurred: {e}")
        
        # Check that exception was written to file
        with open(self.log_file_path, "r") as f:
            log_content = f.read()
            self.assertIn("An error occurred: Test exception", log_content)
            self.assertIn("Traceback", log_content)
    
    def test_get_logger(self):
        """Test get_logger function."""
        # Reset singleton instance
        LoggingService._instance = None
        
        logger1 = get_logger("TestLogger1")
        logger2 = get_logger("TestLogger1")
        
        # Both should return LoggingService instances
        self.assertIsInstance(logger1, LoggingService)
        self.assertIsInstance(logger2, LoggingService)
        
        # Both should be the same instance due to singleton pattern
        self.assertIs(logger1, logger2)


class TestJsonFormatter(unittest.TestCase):
    """Test cases for the JsonFormatter class."""
    
    def setUp(self):
        """Set up test environment."""
        self.formatter = JsonFormatter()
        self.record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_path",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
    
    def test_format_basic(self):
        """Test basic formatting."""
        formatted = self.formatter.format(self.record)
        
        # Parse the JSON string
        log_data = json.loads(formatted)
        
        # Check basic fields
        self.assertEqual(log_data["level"], "INFO")
        self.assertEqual(log_data["name"], "test_logger")
        self.assertEqual(log_data["message"], "Test message")
        self.assertIn("timestamp", log_data)
    
    def test_format_with_exception(self):
        """Test formatting with exception info."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.record.exc_info = sys.exc_info()
            
        formatted = self.formatter.format(self.record)
        
        # Parse the JSON string
        log_data = json.loads(formatted)
        
        # Check exception fields
        self.assertIn("exception", log_data)
        self.assertEqual(log_data["exception"]["type"], "ValueError")
        self.assertEqual(log_data["exception"]["message"], "Test exception")
        self.assertIsInstance(log_data["exception"]["traceback"], list)
    
    def test_format_with_extra(self):
        """Test formatting with extra data."""
        self.record.extra = {"user_id": 123, "action": "login"}
        
        formatted = self.formatter.format(self.record)
        
        # Parse the JSON string
        log_data = json.loads(formatted)
        
        # Check extra fields
        self.assertIn("extra", log_data)
        self.assertEqual(log_data["extra"]["user_id"], 123)
        self.assertEqual(log_data["extra"]["action"], "login")


class TestTimedRotation(unittest.TestCase):
    """Test cases for timed rotation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.test_dir, "test.log")
        
        # Reset singleton instance
        LoggingService._instance = None
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    @patch("src.utils.logging_service.TimedRotatingFileHandler")
    def test_timed_rotation(self, mock_handler):
        """Test timed rotation setup."""
        mock_handler.return_value = MagicMock()
        
        # Create logger with timed rotation
        logger = LoggingService(
            name="TestLogger",
            log_to_console=False,
            log_to_file=True,
            log_file_path=self.log_file_path,
            use_timed_rotation=True,
            rotation_interval="H"
        )
        
        # Check that TimedRotatingFileHandler was called with correct parameters
        mock_handler.assert_called_once_with(
            self.log_file_path,
            when="H",
            backupCount=5
        )


class TestLoggingDecorator(unittest.TestCase):
    """Test cases for the logging decorator."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a string IO to capture logs
        self.log_output = io.StringIO()
        self.handler = logging.StreamHandler(self.log_output)
        self.handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
        
        # Create a logger
        self.logger = logging.getLogger("test_decorator")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.handler)
        self.logger.propagate = False
        
        # Create a LoggingService mock
        self.mock_logging_service = MagicMock()
        self.mock_logging_service.debug = self.logger.debug
        self.mock_logging_service.info = self.logger.info
        self.mock_logging_service.warning = self.logger.warning
        self.mock_logging_service.error = self.logger.error
        self.mock_logging_service.exception = self.logger.exception
    
    def test_decorator_logs_call(self):
        """Test that decorator logs function calls."""
        # Create a decorated function
        @log_function_call(logger=self.mock_logging_service, level="DEBUG")
        def test_func(a, b, c=None):
            return a + b + (c or 0)
        
        # Call the function
        result = test_func(1, 2, c=3)
        
        # Check the log output
        log_content = self.log_output.getvalue()
        self.assertIn("DEBUG:Calling test_func(1, 2, c=3)", log_content)
        self.assertIn(f"DEBUG:test_func returned {result}", log_content)
    
    def test_decorator_logs_exception(self):
        """Test that decorator logs exceptions."""
        # Create a decorated function that raises an exception
        @log_function_call(logger=self.mock_logging_service, level="DEBUG")
        def failing_func():
            raise ValueError("Test exception")
        
        # Call the function and catch the exception
        with self.assertRaises(ValueError):
            failing_func()
        
        # Check the log output
        log_content = self.log_output.getvalue()
        self.assertIn("DEBUG:Calling failing_func()", log_content)
        self.assertIn("Exception in failing_func: Test exception", log_content)


class TestRootLoggerConfiguration(unittest.TestCase):
    """Test cases for root logger configuration."""
    
    def setUp(self):
        """Set up test environment."""
        # Save the original root logger handlers
        self.original_handlers = logging.getLogger().handlers.copy()
        self.original_level = logging.getLogger().level
        
        # Create a temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.log_file_path = os.path.join(self.test_dir, "root.log")
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore the original root logger handlers
        root_logger = logging.getLogger()
        root_logger.handlers = self.original_handlers
        root_logger.setLevel(self.original_level)
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_configure_root_logger(self):
        """Test configuring the root logger."""
        # Configure the root logger
        configure_root_logger(
            log_level="DEBUG",
            log_to_file=True,
            log_file_path=self.log_file_path,
            use_json_format=False
        )
        
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Check the log level
        self.assertEqual(root_logger.level, logging.DEBUG)
        
        # Check that handlers were added
        self.assertGreaterEqual(len(root_logger.handlers), 1)
        
        # Log a message
        root_logger.info("Test root logger message")
        
        # Check that message was written to file
        with open(self.log_file_path, "r") as f:
            log_content = f.read()
            self.assertIn("Test root logger message", log_content)


if __name__ == "__main__":
    unittest.main()
