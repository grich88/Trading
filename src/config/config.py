"""
Configuration Management

This module provides a centralized way to manage application settings.
"""

import os
from typing import Any, Dict, Optional, Union, cast

from dotenv import load_dotenv

from src.utils.logging_service import setup_logger

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = setup_logger(__name__)


class Config:
    """
    Configuration manager for the application.

    This class provides a centralized way to access configuration values from
    environment variables and other sources.
    """

    # Default configuration values
    _defaults: Dict[str, Any] = {
        # Application settings
        "APP_MODE": "development",
        "LOG_LEVEL": "INFO",
        
        # Data settings
        "DEFAULT_TIMEFRAME": "4h",
        "DATA_DIR": "data",
        
        # Model settings
        "MODEL_WEIGHTS_DIR": "data/weights",
        "DEFAULT_WINDOW_SIZE": 60,
        "DEFAULT_LOOKAHEAD_CANDLES": 10,
        "DEFAULT_MIN_ABS_SCORE": 0.425,
        
        # Memory management
        "MEMORY_THRESHOLD_PERCENT": 80,
        "INITIAL_BATCH_SIZE": 100,
        "MIN_BATCH_SIZE": 10,
        "MAX_BATCH_SIZE": 1000,
        "GC_FREQUENCY": 5,
        "MEMORY_CHECK_INTERVAL_SECONDS": 60,
        "ENABLE_MEMORY_MONITORING": True,
        
        # Web application
        "WEB_APP_PORT": 8501,
        "WEB_APP_HOST": "localhost",
        "WEB_APP_TITLE": "Trading Algorithm Dashboard",
    }

    # Cache for configuration values
    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: The configuration key.
            default: The default value to return if the key is not found.
                    If None, the value from _defaults will be used.

        Returns:
            The configuration value.
        """
        # Check cache first
        if key in cls._cache:
            return cls._cache[key]

        # Check environment variables
        env_value = os.environ.get(key)
        if env_value is not None:
            # Convert the value to the appropriate type
            value = cls._convert_value(env_value)
            cls._cache[key] = value
            return value

        # Check defaults
        if key in cls._defaults:
            cls._cache[key] = cls._defaults[key]
            return cls._defaults[key]

        # Use provided default or None
        return default

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: The configuration key.
            value: The configuration value.
        """
        cls._cache[key] = value

    @classmethod
    def _convert_value(cls, value: str) -> Any:
        """
        Convert a string value to the appropriate type.

        Args:
            value: The string value to convert.

        Returns:
            The converted value.
        """
        # Try to convert to int
        try:
            return int(value)
        except ValueError:
            pass

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            pass

        # Convert to boolean
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False

        # Return as string
        return value

    @classmethod
    def is_development(cls) -> bool:
        """
        Check if the application is running in development mode.

        Returns:
            True if in development mode, False otherwise.
        """
        return cls.get("APP_MODE") == "development"

    @classmethod
    def is_production(cls) -> bool:
        """
        Check if the application is running in production mode.

        Returns:
            True if in production mode, False otherwise.
        """
        return cls.get("APP_MODE") == "production"

    @classmethod
    def is_testing(cls) -> bool:
        """
        Check if the application is running in testing mode.

        Returns:
            True if in testing mode, False otherwise.
        """
        return cls.get("APP_MODE") == "testing"