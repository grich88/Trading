"""
Default configuration values.

This module contains default values for all configuration settings.
These are used when environment variables are not set.
"""

from typing import Dict, Any

# Core configuration defaults
DEFAULT_CONFIG = {
    # Application settings
    "APP_MODE": "development",
    "LOG_LEVEL": "INFO",
    
    # API credentials
    "COINGLASS_API_KEY": "",
    "COINGLASS_API_BASE_URL": "https://open-api.coinglass.com/api/v3",
    "EXCHANGE_API_KEY": "",
    "EXCHANGE_API_SECRET": "",
    
    # Data collection
    "DEFAULT_TIMEFRAME": "4h",
    "DATA_DIR": "data",
    "HISTORICAL_DATA_START_DATE": "2023-01-01",
    "HISTORICAL_DATA_END_DATE": "",
    
    # Model configuration
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
    
    # Notification settings
    "ENABLE_EMAIL_NOTIFICATIONS": False,
    "SMTP_SERVER": "smtp.example.com",
    "SMTP_PORT": 587,
    "SMTP_USERNAME": "",
    "SMTP_PASSWORD": "",
    "NOTIFICATION_EMAIL_FROM": "alerts@example.com",
    "NOTIFICATION_EMAIL_TO": "",
    
    # Trading parameters
    "DEFAULT_SYMBOL": "BTC/USDT",
    "TRADING_ENABLED": False,
    "MAX_POSITION_SIZE_USD": 1000.0,
    "RISK_PER_TRADE_PERCENT": 1.0,
    "STOP_LOSS_PERCENT": 2.0,
    "TAKE_PROFIT_PERCENT": 4.0,
    "MAX_OPEN_POSITIONS": 3,
    
    # Signal thresholds
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "VOLUME_SURGE_THRESHOLD": 2.0,
    "LIQUIDATION_THRESHOLD_USD": 5000000,
    "FUNDING_RATE_THRESHOLD": 0.01,
    "PRICE_DIVERGENCE_THRESHOLD": 0.02,
    "CVD_DIVERGENCE_THRESHOLD": 0.05,
    "AGGRESSOR_VOLUME_THRESHOLD": 0.7
}

def get_default(key: str) -> Any:
    """
    Get a default configuration value.
    
    Args:
        key: The configuration key
        
    Returns:
        The default value or None if not found
    """
    return DEFAULT_CONFIG.get(key)


def get_all_defaults() -> Dict[str, Any]:
    """
    Get all default configuration values.
    
    Returns:
        Dictionary of all default values
    """
    return DEFAULT_CONFIG.copy()
