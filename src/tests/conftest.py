"""
Pytest Configuration

This module contains pytest fixtures and configuration for the Trading Algorithm System.
"""

import os
import tempfile
from typing import Dict, Generator, Any

import pandas as pd
import pytest

from src.config import Config


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture that provides sample data for testing.

    Returns:
        A DataFrame containing sample data.
    """
    # Create sample data
    data = {
        "timestamp": pd.date_range(start="2023-01-01", periods=100, freq="4H"),
        "open": [100 + i * 0.1 for i in range(100)],
        "high": [105 + i * 0.1 for i in range(100)],
        "low": [95 + i * 0.1 for i in range(100)],
        "close": [102 + i * 0.1 for i in range(100)],
        "volume": [1000 + i * 10 for i in range(100)],
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Fixture that provides a temporary directory for testing.

    Yields:
        The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_config() -> Generator[Dict[str, Any], None, None]:
    """
    Fixture that provides a test configuration.

    This fixture sets up a test configuration and restores the original
    configuration after the test.

    Yields:
        The test configuration.
    """
    # Save original cache
    original_cache = Config._cache.copy()
    
    # Set up test configuration
    test_config = {
        "APP_MODE": "testing",
        "LOG_LEVEL": "DEBUG",
        "DATA_DIR": "test_data",
        "MODEL_WEIGHTS_DIR": "test_weights",
        "ENABLE_MEMORY_MONITORING": False,
    }
    
    # Update configuration
    for key, value in test_config.items():
        Config.set(key, value)
    
    yield test_config
    
    # Restore original cache
    Config._cache = original_cache
