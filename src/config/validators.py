"""
Configuration validation module.

This module provides validation functions for configuration values
to ensure they meet required constraints.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

# Regular expression patterns for validation
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
URL_PATTERN = re.compile(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
TIMEFRAME_PATTERN = re.compile(r'^[1-9]\d*[mhdwM]$')
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]+/[A-Z0-9]+$')


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""
    pass


def validate_email(value: str) -> bool:
    """
    Validate an email address.
    
    Args:
        value: The email address to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(EMAIL_PATTERN.match(value)) if value else True


def validate_url(value: str) -> bool:
    """
    Validate a URL.
    
    Args:
        value: The URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(URL_PATTERN.match(value)) if value else True


def validate_date(value: str) -> bool:
    """
    Validate a date string in YYYY-MM-DD format.
    
    Args:
        value: The date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not value:
        return True
    
    if not DATE_PATTERN.match(value):
        return False
    
    try:
        datetime.strptime(value, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def validate_timeframe(value: str) -> bool:
    """
    Validate a timeframe string (e.g., 1m, 5m, 1h, 4h, 1d, 1w, 1M).
    
    Args:
        value: The timeframe string to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(TIMEFRAME_PATTERN.match(value))


def validate_symbol(value: str) -> bool:
    """
    Validate a trading symbol (e.g., BTC/USDT).
    
    Args:
        value: The symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    return bool(SYMBOL_PATTERN.match(value))


def validate_port(value: int) -> bool:
    """
    Validate a network port number.
    
    Args:
        value: The port number to validate
        
    Returns:
        True if valid, False otherwise
    """
    return 1 <= value <= 65535


def validate_percentage(value: float) -> bool:
    """
    Validate a percentage value (0-100).
    
    Args:
        value: The percentage value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return 0 <= value <= 100


def validate_positive(value: Union[int, float]) -> bool:
    """
    Validate that a value is positive.
    
    Args:
        value: The value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return value > 0


def validate_non_negative(value: Union[int, float]) -> bool:
    """
    Validate that a value is non-negative.
    
    Args:
        value: The value to validate
        
    Returns:
        True if valid, False otherwise
    """
    return value >= 0


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate the entire configuration.
    
    Args:
        config: Dictionary of configuration values
        
    Returns:
        List of validation error messages, empty if all valid
    """
    errors = []
    
    # Validate email addresses
    for key in ["NOTIFICATION_EMAIL_FROM", "NOTIFICATION_EMAIL_TO"]:
        if config.get(key) and not validate_email(config[key]):
            errors.append(f"Invalid email address for {key}: {config[key]}")
    
    # Validate URLs
    for key in ["COINGLASS_API_BASE_URL"]:
        if config.get(key) and not validate_url(config[key]):
            errors.append(f"Invalid URL for {key}: {config[key]}")
    
    # Validate dates
    for key in ["HISTORICAL_DATA_START_DATE", "HISTORICAL_DATA_END_DATE"]:
        if config.get(key) and not validate_date(config[key]):
            errors.append(f"Invalid date format for {key}: {config[key]}")
    
    # Validate timeframe
    if not validate_timeframe(config.get("DEFAULT_TIMEFRAME", "")):
        errors.append(f"Invalid timeframe format for DEFAULT_TIMEFRAME: {config.get('DEFAULT_TIMEFRAME')}")
    
    # Validate symbol
    if not validate_symbol(config.get("DEFAULT_SYMBOL", "")):
        errors.append(f"Invalid symbol format for DEFAULT_SYMBOL: {config.get('DEFAULT_SYMBOL')}")
    
    # Validate port
    if not validate_port(config.get("WEB_APP_PORT", 0)):
        errors.append(f"Invalid port number for WEB_APP_PORT: {config.get('WEB_APP_PORT')}")
    
    if not validate_port(config.get("SMTP_PORT", 0)):
        errors.append(f"Invalid port number for SMTP_PORT: {config.get('SMTP_PORT')}")
    
    # Validate percentages
    for key in ["MEMORY_THRESHOLD_PERCENT", "RISK_PER_TRADE_PERCENT", 
                "STOP_LOSS_PERCENT", "TAKE_PROFIT_PERCENT"]:
        if not validate_percentage(config.get(key, 0)):
            errors.append(f"Invalid percentage value for {key}: {config.get(key)}")
    
    # Validate positive values
    for key in ["DEFAULT_WINDOW_SIZE", "DEFAULT_LOOKAHEAD_CANDLES", 
                "INITIAL_BATCH_SIZE", "MIN_BATCH_SIZE", "MAX_BATCH_SIZE",
                "GC_FREQUENCY", "MEMORY_CHECK_INTERVAL_SECONDS", 
                "MAX_POSITION_SIZE_USD", "MAX_OPEN_POSITIONS"]:
        if not validate_positive(config.get(key, 0)):
            errors.append(f"Value must be positive for {key}: {config.get(key)}")
    
    # Validate non-negative values
    for key in ["DEFAULT_MIN_ABS_SCORE", "VOLUME_SURGE_THRESHOLD", 
                "LIQUIDATION_THRESHOLD_USD", "FUNDING_RATE_THRESHOLD",
                "PRICE_DIVERGENCE_THRESHOLD", "CVD_DIVERGENCE_THRESHOLD",
                "AGGRESSOR_VOLUME_THRESHOLD"]:
        if not validate_non_negative(config.get(key, -1)):
            errors.append(f"Value must be non-negative for {key}: {config.get(key)}")
    
    return errors
