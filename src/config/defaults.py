"""
Default Configuration Values

This module defines default configuration values for the application.
"""

from typing import Dict, Any

# Default configuration values
DEFAULT_CONFIG: Dict[str, Any] = {
    # Application settings
    "APP_MODE": "development",
    "LOG_LEVEL": "INFO",
    
    # Data settings
    "DEFAULT_TIMEFRAME": "4h",
    "DATA_DIR": "data",
    "HISTORICAL_DATA_START_DATE": "2023-01-01",
    "HISTORICAL_DATA_END_DATE": "2023-12-31",
    
    # API credentials
    "COINGLASS_API_KEY": "",
    "COINGLASS_API_BASE_URL": "https://open-api.coinglass.com/api/v3",
    "EXCHANGE_API_KEY": "",
    "EXCHANGE_API_SECRET": "",
    
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
    
    # Notification settings
    "ENABLE_EMAIL_NOTIFICATIONS": False,
    "SMTP_SERVER": "smtp.example.com",
    "SMTP_PORT": 587,
    "SMTP_USERNAME": "",
    "SMTP_PASSWORD": "",
    "NOTIFICATION_EMAIL_FROM": "alerts@example.com",
    "NOTIFICATION_EMAIL_TO": "",
}

# Asset-specific default parameters
ASSET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "window_size": 60,
        "lookahead_candles": 10,
        "min_abs_score": 0.425,
        "regime_filter": True,
        "fee_bps": 5.0,
        "adaptive_cutoff": True,
        "min_agree_features": 1,
    },
    "SOL": {
        "window_size": 40,
        "lookahead_candles": 10,
        "min_abs_score": 0.520,
        "regime_filter": True,
        "fee_bps": 5.0,
        "adaptive_cutoff": True,
        "min_agree_features": 1,
    },
    "BONK": {
        "window_size": 40,
        "lookahead_candles": 10,
        "min_abs_score": 0.550,
        "regime_filter": True,
        "fee_bps": 5.0,
        "adaptive_cutoff": True,
        "min_agree_features": 1,
    },
}
