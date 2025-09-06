"""
Configuration management module.

This module provides a centralized configuration management system
that loads settings from environment variables with sensible defaults.
"""

import os
import sys
from typing import Any, Dict, Optional, Union, List
from dotenv import load_dotenv
import logging

from .defaults import get_default, get_all_defaults
from .validators import validate_config

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Config")


class Config:
    """
    Centralized configuration management class.
    
    This class provides access to all configuration settings with appropriate
    type conversion and default values.
    """
    
    @staticmethod
    def get_str(key: str, default: Optional[str] = None) -> str:
        """
        Get a string configuration value.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            
        Returns:
            The string value
        """
        # Use default from defaults module if not provided
        if default is None:
            default = get_default(key)
            
        value = os.environ.get(key, default)
        if value is None:
            logger.warning(f"Configuration key '{key}' not found and no default provided")
            return ""
        return value
    
    @staticmethod
    def get_int(key: str, default: Optional[int] = None) -> int:
        """
        Get an integer configuration value.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            
        Returns:
            The integer value
        """
        # Use default from defaults module if not provided
        if default is None:
            default_val = get_default(key)
            if default_val is not None:
                try:
                    default = int(default_val)
                except (ValueError, TypeError):
                    pass
                    
        value = os.environ.get(key)
        if value is None:
            if default is None:
                logger.warning(f"Configuration key '{key}' not found and no default provided")
                return 0
            return default
        
        try:
            return int(value)
        except ValueError:
            logger.error(f"Configuration key '{key}' value '{value}' is not a valid integer")
            return default if default is not None else 0
    
    @staticmethod
    def get_float(key: str, default: Optional[float] = None) -> float:
        """
        Get a float configuration value.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            
        Returns:
            The float value
        """
        # Use default from defaults module if not provided
        if default is None:
            default_val = get_default(key)
            if default_val is not None:
                try:
                    default = float(default_val)
                except (ValueError, TypeError):
                    pass
                    
        value = os.environ.get(key)
        if value is None:
            if default is None:
                logger.warning(f"Configuration key '{key}' not found and no default provided")
                return 0.0
            return default
        
        try:
            return float(value)
        except ValueError:
            logger.error(f"Configuration key '{key}' value '{value}' is not a valid float")
            return default if default is not None else 0.0
    
    @staticmethod
    def get_bool(key: str, default: Optional[bool] = None) -> bool:
        """
        Get a boolean configuration value.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            
        Returns:
            The boolean value
        """
        # Use default from defaults module if not provided
        if default is None:
            default_val = get_default(key)
            if default_val is not None:
                if isinstance(default_val, bool):
                    default = default_val
                elif isinstance(default_val, str):
                    default = default_val.lower() in ('true', 'yes', '1', 't', 'y')
                    
        value = os.environ.get(key)
        if value is None:
            if default is None:
                logger.warning(f"Configuration key '{key}' not found and no default provided")
                return False
            return default
        
        return value.lower() in ('true', 'yes', '1', 't', 'y')
    
    @staticmethod
    def get_list(key: str, default: Optional[List[str]] = None, separator: str = ',') -> List[str]:
        """
        Get a list configuration value.
        
        Args:
            key: The environment variable name
            default: Default value if not found
            separator: The separator character for the list
            
        Returns:
            The list value
        """
        # Use default from defaults module if not provided
        if default is None:
            default_val = get_default(key)
            if default_val is not None and isinstance(default_val, list):
                default = default_val
                    
        value = os.environ.get(key)
        if value is None:
            if default is None:
                logger.warning(f"Configuration key '{key}' not found and no default provided")
                return []
            return default
        
        return [item.strip() for item in value.split(separator)]


# Core configuration
APP_MODE = Config.get_str("APP_MODE", "development")
LOG_LEVEL = Config.get_str("LOG_LEVEL", "INFO")

# API credentials
COINGLASS_API_KEY = Config.get_str("COINGLASS_API_KEY", "")
COINGLASS_API_BASE_URL = Config.get_str("COINGLASS_API_BASE_URL", "https://open-api.coinglass.com/api/v3")
EXCHANGE_API_KEY = Config.get_str("EXCHANGE_API_KEY", "")
EXCHANGE_API_SECRET = Config.get_str("EXCHANGE_API_SECRET", "")

# Data collection
DEFAULT_TIMEFRAME = Config.get_str("DEFAULT_TIMEFRAME", "4h")
DATA_DIR = Config.get_str("DATA_DIR", "data")
HISTORICAL_DATA_START_DATE = Config.get_str("HISTORICAL_DATA_START_DATE", "2023-01-01")
HISTORICAL_DATA_END_DATE = Config.get_str("HISTORICAL_DATA_END_DATE", "")

# Model configuration
MODEL_WEIGHTS_DIR = Config.get_str("MODEL_WEIGHTS_DIR", "data/weights")
DEFAULT_WINDOW_SIZE = Config.get_int("DEFAULT_WINDOW_SIZE", 60)
DEFAULT_LOOKAHEAD_CANDLES = Config.get_int("DEFAULT_LOOKAHEAD_CANDLES", 10)
DEFAULT_MIN_ABS_SCORE = Config.get_float("DEFAULT_MIN_ABS_SCORE", 0.425)

# Memory management
MEMORY_THRESHOLD_PERCENT = Config.get_int("MEMORY_THRESHOLD_PERCENT", 80)
INITIAL_BATCH_SIZE = Config.get_int("INITIAL_BATCH_SIZE", 100)
MIN_BATCH_SIZE = Config.get_int("MIN_BATCH_SIZE", 10)
MAX_BATCH_SIZE = Config.get_int("MAX_BATCH_SIZE", 1000)
GC_FREQUENCY = Config.get_int("GC_FREQUENCY", 5)
MEMORY_CHECK_INTERVAL_SECONDS = Config.get_int("MEMORY_CHECK_INTERVAL_SECONDS", 60)
ENABLE_MEMORY_MONITORING = Config.get_bool("ENABLE_MEMORY_MONITORING", True)

# Web application
WEB_APP_PORT = Config.get_int("WEB_APP_PORT", 8501)
WEB_APP_HOST = Config.get_str("WEB_APP_HOST", "localhost")
WEB_APP_TITLE = Config.get_str("WEB_APP_TITLE", "Trading Algorithm Dashboard")

# Notification settings
ENABLE_EMAIL_NOTIFICATIONS = Config.get_bool("ENABLE_EMAIL_NOTIFICATIONS", False)
SMTP_SERVER = Config.get_str("SMTP_SERVER", "smtp.example.com")
SMTP_PORT = Config.get_int("SMTP_PORT", 587)
SMTP_USERNAME = Config.get_str("SMTP_USERNAME", "")
SMTP_PASSWORD = Config.get_str("SMTP_PASSWORD", "")
NOTIFICATION_EMAIL_FROM = Config.get_str("NOTIFICATION_EMAIL_FROM", "alerts@example.com")
NOTIFICATION_EMAIL_TO = Config.get_str("NOTIFICATION_EMAIL_TO", "")


def get_config_dict() -> Dict[str, Any]:
    """
    Get all configuration values as a dictionary.
    
    Returns:
        Dictionary of all configuration values
    """
    return {
        "APP_MODE": APP_MODE,
        "LOG_LEVEL": LOG_LEVEL,
        "COINGLASS_API_KEY": COINGLASS_API_KEY,
        "COINGLASS_API_BASE_URL": COINGLASS_API_BASE_URL,
        "EXCHANGE_API_KEY": EXCHANGE_API_KEY,
        "EXCHANGE_API_SECRET": EXCHANGE_API_SECRET,
        "DEFAULT_TIMEFRAME": DEFAULT_TIMEFRAME,
        "DATA_DIR": DATA_DIR,
        "HISTORICAL_DATA_START_DATE": HISTORICAL_DATA_START_DATE,
        "HISTORICAL_DATA_END_DATE": HISTORICAL_DATA_END_DATE,
        "MODEL_WEIGHTS_DIR": MODEL_WEIGHTS_DIR,
        "DEFAULT_WINDOW_SIZE": DEFAULT_WINDOW_SIZE,
        "DEFAULT_LOOKAHEAD_CANDLES": DEFAULT_LOOKAHEAD_CANDLES,
        "DEFAULT_MIN_ABS_SCORE": DEFAULT_MIN_ABS_SCORE,
        "MEMORY_THRESHOLD_PERCENT": MEMORY_THRESHOLD_PERCENT,
        "INITIAL_BATCH_SIZE": INITIAL_BATCH_SIZE,
        "MIN_BATCH_SIZE": MIN_BATCH_SIZE,
        "MAX_BATCH_SIZE": MAX_BATCH_SIZE,
        "GC_FREQUENCY": GC_FREQUENCY,
        "MEMORY_CHECK_INTERVAL_SECONDS": MEMORY_CHECK_INTERVAL_SECONDS,
        "ENABLE_MEMORY_MONITORING": ENABLE_MEMORY_MONITORING,
        "WEB_APP_PORT": WEB_APP_PORT,
        "WEB_APP_HOST": WEB_APP_HOST,
        "WEB_APP_TITLE": WEB_APP_TITLE,
        "ENABLE_EMAIL_NOTIFICATIONS": ENABLE_EMAIL_NOTIFICATIONS,
        "SMTP_SERVER": SMTP_SERVER,
        "SMTP_PORT": SMTP_PORT,
        "SMTP_USERNAME": SMTP_USERNAME,
        "SMTP_PASSWORD": "******" if SMTP_PASSWORD else "",
        "NOTIFICATION_EMAIL_FROM": NOTIFICATION_EMAIL_FROM,
        "NOTIFICATION_EMAIL_TO": NOTIFICATION_EMAIL_TO
    }


def print_config() -> None:
    """Print the current configuration (with sensitive values masked)."""
    config = get_config_dict()
    logger.info("Current configuration:")
    for key, value in config.items():
        if key in ("EXCHANGE_API_SECRET", "SMTP_PASSWORD"):
            logger.info(f"  {key}: {'*' * 8}")
        else:
            logger.info(f"  {key}: {value}")


def validate_all_config() -> bool:
    """
    Validate the entire configuration.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    config = get_config_dict()
    errors = validate_config(config)
    
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("Configuration validation successful")
    return True


if __name__ == "__main__":
    # Example usage
    print_config()
    validate_all_config()
