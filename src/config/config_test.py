"""
Tests for the configuration module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from src.config.config import Config, validate_all_config
from src.config.defaults import get_default
from src.config.validators import validate_email, validate_url, validate_date, validate_timeframe


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear environment variables before each test
        for key in list(os.environ.keys()):
            if key.startswith("TEST_"):
                del os.environ[key]
    
    def test_get_str(self):
        """Test get_str method."""
        # Test with environment variable set
        os.environ["TEST_STR"] = "test_value"
        self.assertEqual(Config.get_str("TEST_STR"), "test_value")
        
        # Test with default value
        self.assertEqual(Config.get_str("TEST_MISSING", "default"), "default")
        
        # Test with no default value
        self.assertEqual(Config.get_str("TEST_MISSING"), "")
    
    def test_get_int(self):
        """Test get_int method."""
        # Test with environment variable set
        os.environ["TEST_INT"] = "42"
        self.assertEqual(Config.get_int("TEST_INT"), 42)
        
        # Test with default value
        self.assertEqual(Config.get_int("TEST_MISSING", 24), 24)
        
        # Test with no default value
        self.assertEqual(Config.get_int("TEST_MISSING"), 0)
        
        # Test with invalid value
        os.environ["TEST_INVALID_INT"] = "not_an_int"
        self.assertEqual(Config.get_int("TEST_INVALID_INT", 99), 99)
    
    def test_get_float(self):
        """Test get_float method."""
        # Test with environment variable set
        os.environ["TEST_FLOAT"] = "3.14"
        self.assertEqual(Config.get_float("TEST_FLOAT"), 3.14)
        
        # Test with default value
        self.assertEqual(Config.get_float("TEST_MISSING", 2.71), 2.71)
        
        # Test with no default value
        self.assertEqual(Config.get_float("TEST_MISSING"), 0.0)
        
        # Test with invalid value
        os.environ["TEST_INVALID_FLOAT"] = "not_a_float"
        self.assertEqual(Config.get_float("TEST_INVALID_FLOAT", 9.9), 9.9)
    
    def test_get_bool(self):
        """Test get_bool method."""
        # Test with environment variable set to true values
        for true_value in ["true", "True", "TRUE", "yes", "Yes", "YES", "1", "t", "y"]:
            os.environ["TEST_BOOL"] = true_value
            self.assertTrue(Config.get_bool("TEST_BOOL"))
        
        # Test with environment variable set to false values
        for false_value in ["false", "False", "FALSE", "no", "No", "NO", "0", "f", "n"]:
            os.environ["TEST_BOOL"] = false_value
            self.assertFalse(Config.get_bool("TEST_BOOL"))
        
        # Test with default value
        self.assertTrue(Config.get_bool("TEST_MISSING", True))
        self.assertFalse(Config.get_bool("TEST_MISSING", False))
        
        # Test with no default value
        self.assertFalse(Config.get_bool("TEST_MISSING"))
    
    def test_get_list(self):
        """Test get_list method."""
        # Test with environment variable set
        os.environ["TEST_LIST"] = "item1,item2,item3"
        self.assertEqual(Config.get_list("TEST_LIST"), ["item1", "item2", "item3"])
        
        # Test with custom separator
        os.environ["TEST_LIST_SEMICOLON"] = "item1;item2;item3"
        self.assertEqual(Config.get_list("TEST_LIST_SEMICOLON", separator=";"), ["item1", "item2", "item3"])
        
        # Test with default value
        self.assertEqual(Config.get_list("TEST_MISSING", ["default"]), ["default"])
        
        # Test with no default value
        self.assertEqual(Config.get_list("TEST_MISSING"), [])
        
        # Test with empty string
        os.environ["TEST_LIST_EMPTY"] = ""
        self.assertEqual(Config.get_list("TEST_LIST_EMPTY"), [""])


class TestValidators(unittest.TestCase):
    """Test cases for the validator functions."""
    
    def test_validate_email(self):
        """Test email validation."""
        self.assertTrue(validate_email("user@example.com"))
        self.assertTrue(validate_email("user.name+tag@example.co.uk"))
        self.assertFalse(validate_email("not_an_email"))
        self.assertFalse(validate_email("missing@domain"))
        self.assertFalse(validate_email("@example.com"))
        # Empty string should be valid (optional email)
        self.assertTrue(validate_email(""))
    
    def test_validate_url(self):
        """Test URL validation."""
        self.assertTrue(validate_url("http://example.com"))
        self.assertTrue(validate_url("https://api.example.com/v1"))
        self.assertFalse(validate_url("not_a_url"))
        self.assertFalse(validate_url("ftp://example.com"))
        # Empty string should be valid (optional URL)
        self.assertTrue(validate_url(""))
    
    def test_validate_date(self):
        """Test date validation."""
        self.assertTrue(validate_date("2023-01-01"))
        self.assertTrue(validate_date("2023-12-31"))
        self.assertFalse(validate_date("2023/01/01"))
        self.assertFalse(validate_date("01-01-2023"))
        self.assertFalse(validate_date("2023-13-01"))  # Invalid month
        self.assertFalse(validate_date("2023-01-32"))  # Invalid day
        # Empty string should be valid (optional date)
        self.assertTrue(validate_date(""))
    
    def test_validate_timeframe(self):
        """Test timeframe validation."""
        self.assertTrue(validate_timeframe("1m"))
        self.assertTrue(validate_timeframe("5m"))
        self.assertTrue(validate_timeframe("1h"))
        self.assertTrue(validate_timeframe("4h"))
        self.assertTrue(validate_timeframe("1d"))
        self.assertTrue(validate_timeframe("1w"))
        self.assertTrue(validate_timeframe("1M"))
        self.assertFalse(validate_timeframe("m"))
        self.assertFalse(validate_timeframe("5"))
        self.assertFalse(validate_timeframe("5x"))
        self.assertFalse(validate_timeframe("0m"))


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for the configuration module."""
    
    @patch('src.config.config.validate_config')
    def test_validate_all_config(self, mock_validate_config):
        """Test validate_all_config function."""
        # Test with valid configuration
        mock_validate_config.return_value = []
        self.assertTrue(validate_all_config())
        
        # Test with invalid configuration
        mock_validate_config.return_value = ["Error 1", "Error 2"]
        self.assertFalse(validate_all_config())


if __name__ == "__main__":
    unittest.main()
