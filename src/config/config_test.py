"""
Tests for the configuration module.
"""

import os
import unittest
from unittest.mock import patch

from src.config.config import Config


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


if __name__ == "__main__":
    unittest.main()
