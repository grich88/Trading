"""
Tests for the error handling module.
"""

import unittest
import time
from unittest.mock import patch, MagicMock
import logging

from src.utils.error_handling import (
    ApplicationError,
    ConfigurationError,
    DataError,
    APIError,
    ModelError,
    handle_exception,
    exception_handler,
    retry
)


class TestExceptionClasses(unittest.TestCase):
    """Test cases for the exception classes."""
    
    def test_application_error(self):
        """Test ApplicationError class."""
        # Test with basic message
        error = ApplicationError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertEqual(error.message, "Test error")
        self.assertIsNone(error.error_code)
        self.assertEqual(error.details, {})
        
        # Test with error code
        error = ApplicationError("Test error", "E001")
        self.assertEqual(error.error_code, "E001")
        
        # Test with details
        details = {"param": "value", "source": "test"}
        error = ApplicationError("Test error", "E001", details)
        self.assertEqual(error.details, details)
    
    def test_specific_error_classes(self):
        """Test specific error subclasses."""
        # Test ConfigurationError
        error = ConfigurationError("Config error")
        self.assertIsInstance(error, ApplicationError)
        self.assertEqual(error.message, "Config error")
        
        # Test DataError
        error = DataError("Data error")
        self.assertIsInstance(error, ApplicationError)
        self.assertEqual(error.message, "Data error")
        
        # Test APIError
        error = APIError("API error")
        self.assertIsInstance(error, ApplicationError)
        self.assertEqual(error.message, "API error")
        
        # Test ModelError
        error = ModelError("Model error")
        self.assertIsInstance(error, ApplicationError)
        self.assertEqual(error.message, "Model error")


class TestHandleException(unittest.TestCase):
    """Test cases for the handle_exception function."""
    
    @patch('src.utils.error_handling.logger')
    def test_handle_standard_exception(self, mock_logger):
        """Test handling a standard exception."""
        e = ValueError("Test error")
        result = handle_exception(e)
        
        # Check result structure
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Test error")
        self.assertEqual(result["error_type"], "ValueError")
        
        # Check that logger was called
        mock_logger.error.assert_called()
    
    @patch('src.utils.error_handling.logger')
    def test_handle_application_error(self, mock_logger):
        """Test handling an ApplicationError with code and details."""
        details = {"param": "value"}
        e = ApplicationError("App error", "E001", details)
        result = handle_exception(e)
        
        # Check result structure
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "App error")
        self.assertEqual(result["error_type"], "ApplicationError")
        self.assertEqual(result["error_code"], "E001")
        self.assertEqual(result["details"], details)
        
        # Check that logger was called
        mock_logger.error.assert_called()
    
    @patch('src.utils.error_handling.logger')
    def test_no_logging(self, mock_logger):
        """Test handling an exception without logging."""
        e = ValueError("Test error")
        handle_exception(e, log_exception=False)
        
        # Check that logger was not called
        mock_logger.error.assert_not_called()


class TestExceptionHandlerDecorator(unittest.TestCase):
    """Test cases for the exception_handler decorator."""
    
    def test_no_exception(self):
        """Test when no exception occurs."""
        @exception_handler()
        def test_func():
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
    
    def test_with_exception(self):
        """Test when an exception occurs."""
        fallback = {"status": "error"}
        
        @exception_handler(fallback_return=fallback)
        def test_func():
            raise ValueError("Test error")
        
        result = test_func()
        self.assertEqual(result, fallback)
    
    def test_expected_exceptions(self):
        """Test with expected exception types."""
        @exception_handler(fallback_return="fallback", expected_exceptions=ValueError)
        def test_func(error_type):
            if error_type == "value":
                raise ValueError("Value error")
            else:
                raise TypeError("Type error")
        
        # Should handle ValueError and return fallback
        result = test_func("value")
        self.assertEqual(result, "fallback")
        
        # Should re-raise TypeError
        with self.assertRaises(TypeError):
            test_func("type")
    
    @patch('src.utils.error_handling.logger')
    def test_no_logging(self, mock_logger):
        """Test with logging disabled."""
        @exception_handler(log_exception=False)
        def test_func():
            raise ValueError("Test error")
        
        test_func()
        mock_logger.exception.assert_not_called()


class TestRetryDecorator(unittest.TestCase):
    """Test cases for the retry decorator."""
    
    def test_no_exception(self):
        """Test when no exception occurs."""
        @retry(max_attempts=3)
        def test_func():
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
    
    def test_retry_success(self):
        """Test when retry succeeds."""
        attempts = 0
        
        @retry(max_attempts=3, delay_seconds=0.01)
        def test_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Test error")
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
        self.assertEqual(attempts, 3)
    
    def test_retry_failure(self):
        """Test when all retries fail."""
        @retry(max_attempts=3, delay_seconds=0.01)
        def test_func():
            raise ConnectionError("Test error")
        
        with self.assertRaises(ConnectionError):
            test_func()
    
    def test_retry_specific_exceptions(self):
        """Test retrying only specific exceptions."""
        @retry(max_attempts=3, retry_exceptions=ValueError, delay_seconds=0.01)
        def test_func(error_type):
            if error_type == "value":
                raise ValueError("Value error")
            else:
                raise TypeError("Type error")
        
        # Should retry ValueError
        with self.assertRaises(ValueError):
            test_func("value")
        
        # Should not retry TypeError
        with self.assertRaises(TypeError):
            test_func("type")
    
    @patch('src.utils.error_handling.logger')
    def test_no_logging(self, mock_logger):
        """Test with logging disabled."""
        @retry(max_attempts=2, delay_seconds=0.01, log_retries=False)
        def test_func():
            raise ConnectionError("Test error")
        
        with self.assertRaises(ConnectionError):
            test_func()
        
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()


class TestIntegration(unittest.TestCase):
    """Integration tests for error handling components."""
    
    def test_exception_handler_with_application_error(self):
        """Test exception_handler with ApplicationError."""
        @exception_handler(fallback_return={"status": "error"})
        def test_func():
            raise ConfigurationError("Config error", "E001", {"param": "value"})
        
        result = test_func()
        self.assertEqual(result, {"status": "error"})
    
    def test_retry_with_exception_handler(self):
        """Test retry and exception_handler together."""
        attempts = 0
        
        @exception_handler(fallback_return="fallback")
        @retry(max_attempts=3, delay_seconds=0.01)
        def test_func():
            nonlocal attempts
            attempts += 1
            raise ConnectionError("Test error")
        
        # Should retry 3 times, then return fallback
        result = test_func()
        self.assertEqual(result, "fallback")
        self.assertEqual(attempts, 3)


if __name__ == "__main__":
    unittest.main()
