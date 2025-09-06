"""
Tests for the configuration module.
"""

import os
from unittest import mock

import pytest

from src.config import Config
from src.utils.error_handling import ValidationError
from src.config.validators import (
    validate_required_config,
    validate_directory,
    validate_api_credentials,
    validate_numeric_range,
    validate_enum,
)


def test_config_get():
    """Test getting configuration values."""
    # Test getting a value from environment
    with mock.patch.dict(os.environ, {"TEST_KEY": "test_value"}):
        assert Config.get("TEST_KEY") == "test_value"
    
    # Test getting a default value
    assert Config.get("NON_EXISTENT_KEY", "default") == "default"
    
    # Test getting a value from defaults
    assert Config.get("APP_MODE") == "development"


def test_config_set():
    """Test setting configuration values."""
    # Set a value
    Config.set("TEST_KEY", "test_value")
    
    # Check that the value was set
    assert Config.get("TEST_KEY") == "test_value"
    
    # Override the value
    Config.set("TEST_KEY", "new_value")
    
    # Check that the value was updated
    assert Config.get("TEST_KEY") == "new_value"


def test_config_environment_modes():
    """Test environment mode checks."""
    # Test development mode
    Config.set("APP_MODE", "development")
    assert Config.is_development() is True
    assert Config.is_production() is False
    assert Config.is_testing() is False
    
    # Test production mode
    Config.set("APP_MODE", "production")
    assert Config.is_development() is False
    assert Config.is_production() is True
    assert Config.is_testing() is False
    
    # Test testing mode
    Config.set("APP_MODE", "testing")
    assert Config.is_development() is False
    assert Config.is_production() is False
    assert Config.is_testing() is True


def test_validate_required_config():
    """Test validation of required configuration keys."""
    # Test valid configuration
    config = {"key1": "value1", "key2": "value2"}
    validate_required_config(config, ["key1", "key2"])
    
    # Test missing key
    with pytest.raises(ValidationError):
        validate_required_config(config, ["key1", "key3"])
    
    # Test empty value
    config["key2"] = ""
    with pytest.raises(ValidationError):
        validate_required_config(config, ["key1", "key2"])
    
    # Test boolean value (should not be considered empty)
    config["key2"] = False
    validate_required_config(config, ["key1", "key2"])


def test_validate_directory(temp_dir):
    """Test validation of directories."""
    # Test existing directory
    validate_directory(temp_dir)
    
    # Test non-existent directory
    non_existent_dir = os.path.join(temp_dir, "non_existent")
    with pytest.raises(ValidationError):
        validate_directory(non_existent_dir)
    
    # Test creating non-existent directory
    validate_directory(non_existent_dir, create=True)
    assert os.path.exists(non_existent_dir)
    
    # Test file as directory
    test_file = os.path.join(temp_dir, "test_file")
    with open(test_file, "w") as f:
        f.write("test")
    
    with pytest.raises(ValidationError):
        validate_directory(test_file)


def test_validate_api_credentials():
    """Test validation of API credentials."""
    # Test valid credentials
    validate_api_credentials("api_key", "api_secret")
    
    # Test missing API key
    with pytest.raises(ValidationError):
        validate_api_credentials("", "api_secret")
    
    # Test missing API secret
    with pytest.raises(ValidationError):
        validate_api_credentials("api_key", "")


def test_validate_numeric_range():
    """Test validation of numeric ranges."""
    # Test valid value
    validate_numeric_range(5, 0, 10)
    
    # Test value below minimum
    with pytest.raises(ValidationError):
        validate_numeric_range(5, 6, 10)
    
    # Test value above maximum
    with pytest.raises(ValidationError):
        validate_numeric_range(5, 0, 4)
    
    # Test without minimum
    validate_numeric_range(5, None, 10)
    
    # Test without maximum
    validate_numeric_range(5, 0, None)


def test_validate_enum():
    """Test validation of enum values."""
    # Test valid value
    validate_enum("a", {"a", "b", "c"})
    
    # Test invalid value
    with pytest.raises(ValidationError):
        validate_enum("d", {"a", "b", "c"})
    
    # Test with custom name
    with pytest.raises(ValidationError) as excinfo:
        validate_enum("d", {"a", "b", "c"}, name="Letter")
    
    assert "Letter" in str(excinfo.value)