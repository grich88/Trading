"""
Configuration package.

This package provides centralized configuration management for the application.
"""

from src.config.config import (
    Config, 
    get_config_dict, 
    print_config,
    # Core configuration
    APP_MODE,
    LOG_LEVEL,
    # API credentials
    COINGLASS_API_KEY,
    COINGLASS_API_BASE_URL,
    EXCHANGE_API_KEY,
    EXCHANGE_API_SECRET,
    # Data collection
    DEFAULT_TIMEFRAME,
    DATA_DIR,
    HISTORICAL_DATA_START_DATE,
    HISTORICAL_DATA_END_DATE,
    # Model configuration
    MODEL_WEIGHTS_DIR,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_LOOKAHEAD_CANDLES,
    DEFAULT_MIN_ABS_SCORE,
    # Memory management
    MEMORY_THRESHOLD_PERCENT,
    INITIAL_BATCH_SIZE,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    GC_FREQUENCY,
    MEMORY_CHECK_INTERVAL_SECONDS,
    ENABLE_MEMORY_MONITORING,
    # Web application
    WEB_APP_PORT,
    WEB_APP_HOST,
    WEB_APP_TITLE,
    # Notification settings
    ENABLE_EMAIL_NOTIFICATIONS,
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    NOTIFICATION_EMAIL_FROM,
    NOTIFICATION_EMAIL_TO
)

__all__ = [
    'Config',
    'get_config_dict',
    'print_config',
    'APP_MODE',
    'LOG_LEVEL',
    'COINGLASS_API_KEY',
    'COINGLASS_API_BASE_URL',
    'EXCHANGE_API_KEY',
    'EXCHANGE_API_SECRET',
    'DEFAULT_TIMEFRAME',
    'DATA_DIR',
    'HISTORICAL_DATA_START_DATE',
    'HISTORICAL_DATA_END_DATE',
    'MODEL_WEIGHTS_DIR',
    'DEFAULT_WINDOW_SIZE',
    'DEFAULT_LOOKAHEAD_CANDLES',
    'DEFAULT_MIN_ABS_SCORE',
    'MEMORY_THRESHOLD_PERCENT',
    'INITIAL_BATCH_SIZE',
    'MIN_BATCH_SIZE',
    'MAX_BATCH_SIZE',
    'GC_FREQUENCY',
    'MEMORY_CHECK_INTERVAL_SECONDS',
    'ENABLE_MEMORY_MONITORING',
    'WEB_APP_PORT',
    'WEB_APP_HOST',
    'WEB_APP_TITLE',
    'ENABLE_EMAIL_NOTIFICATIONS',
    'SMTP_SERVER',
    'SMTP_PORT',
    'SMTP_USERNAME',
    'SMTP_PASSWORD',
    'NOTIFICATION_EMAIL_FROM',
    'NOTIFICATION_EMAIL_TO'
]
