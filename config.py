import yaml
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class TradingConfig:
    # Exchange settings
    exchange_id: str
    api_key: Optional[str]
    api_secret: Optional[str]
    testnet: bool
    
    # Trading parameters
    trading_timeframe: str
    base_assets: list
    quote_asset: str
    min_volume_usd: float
    
    # Risk management
    max_position_size: float
    max_risk_per_trade: float
    max_open_trades: int
    max_daily_drawdown: float
    
    # Technical analysis
    rsi_period: int
    rsi_overbought: float
    rsi_oversold: float
    macd_fast: int
    macd_slow: int
    macd_signal: int
    
    # Pattern recognition
    min_pattern_bars: int
    max_pattern_bars: int
    pattern_confidence_threshold: float
    
    # Cross-asset analysis
    correlation_threshold: float
    min_correlation_window: int
    lead_lag_max_offset: int
    
    # Execution
    use_limit_orders: bool
    slippage_tolerance: float
    max_retries: int
    retry_delay: int

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.default_config = {
            "exchange": {
                "id": "binance",
                "api_key": None,
                "api_secret": None,
                "testnet": True
            },
            "trading": {
                "timeframe": "1h",
                "base_assets": ["BTC", "ETH", "SOL"],
                "quote_asset": "USDT",
                "min_volume_usd": 1000000
            },
            "risk": {
                "max_position_size": 0.1,
                "max_risk_per_trade": 0.02,
                "max_open_trades": 3,
                "max_daily_drawdown": 0.05
            },
            "technical": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "patterns": {
                "min_pattern_bars": 5,
                "max_pattern_bars": 50,
                "pattern_confidence_threshold": 0.7
            },
            "cross_asset": {
                "correlation_threshold": 0.7,
                "min_correlation_window": 100,
                "lead_lag_max_offset": 10
            },
            "execution": {
                "use_limit_orders": True,
                "slippage_tolerance": 0.001,
                "max_retries": 3,
                "retry_delay": 1
            }
        }
    
    def load_config(self, filename: str = "config.yaml") -> TradingConfig:
        """Load configuration from file or create default."""
        config_path = self.config_dir / filename
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if filename.endswith('.yaml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self.logger.info(f"Loaded configuration from {filename}")
            else:
                config_data = self.default_config
                self.save_config(config_data, filename)
                self.logger.info(f"Created default configuration in {filename}")
            
            # Convert to TradingConfig
            return TradingConfig(
                # Exchange settings
                exchange_id=config_data["exchange"]["id"],
                api_key=config_data["exchange"]["api_key"],
                api_secret=config_data["exchange"]["api_secret"],
                testnet=config_data["exchange"]["testnet"],
                
                # Trading parameters
                trading_timeframe=config_data["trading"]["timeframe"],
                base_assets=config_data["trading"]["base_assets"],
                quote_asset=config_data["trading"]["quote_asset"],
                min_volume_usd=config_data["trading"]["min_volume_usd"],
                
                # Risk management
                max_position_size=config_data["risk"]["max_position_size"],
                max_risk_per_trade=config_data["risk"]["max_risk_per_trade"],
                max_open_trades=config_data["risk"]["max_open_trades"],
                max_daily_drawdown=config_data["risk"]["max_daily_drawdown"],
                
                # Technical analysis
                rsi_period=config_data["technical"]["rsi_period"],
                rsi_overbought=config_data["technical"]["rsi_overbought"],
                rsi_oversold=config_data["technical"]["rsi_oversold"],
                macd_fast=config_data["technical"]["macd_fast"],
                macd_slow=config_data["technical"]["macd_slow"],
                macd_signal=config_data["technical"]["macd_signal"],
                
                # Pattern recognition
                min_pattern_bars=config_data["patterns"]["min_pattern_bars"],
                max_pattern_bars=config_data["patterns"]["max_pattern_bars"],
                pattern_confidence_threshold=config_data["patterns"]["pattern_confidence_threshold"],
                
                # Cross-asset analysis
                correlation_threshold=config_data["cross_asset"]["correlation_threshold"],
                min_correlation_window=config_data["cross_asset"]["min_correlation_window"],
                lead_lag_max_offset=config_data["cross_asset"]["lead_lag_max_offset"],
                
                # Execution
                use_limit_orders=config_data["execution"]["use_limit_orders"],
                slippage_tolerance=config_data["execution"]["slippage_tolerance"],
                max_retries=config_data["execution"]["max_retries"],
                retry_delay=config_data["execution"]["retry_delay"]
            )
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            self.logger.info("Using default configuration")
            return self.load_config("default_config.yaml")
    
    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml") -> None:
        """Save configuration to file."""
        config_path = self.config_dir / filename
        
        try:
            with open(config_path, 'w') as f:
                if filename.endswith('.yaml'):
                    yaml.dump(config, f, default_flow_style=False)
                else:
                    json.dump(config, f, indent=4)
            
            self.logger.info(f"Saved configuration to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any], filename: str = "config.yaml") -> TradingConfig:
        """Update specific configuration values."""
        current_config = self.load_config(filename)
        config_dict = self.default_config.copy()
        
        # Update nested dictionaries
        for section, values in updates.items():
            if section in config_dict and isinstance(values, dict):
                config_dict[section].update(values)
            else:
                config_dict[section] = values
        
        self.save_config(config_dict, filename)
        return self.load_config(filename)
    
    def validate_config(self, config: TradingConfig) -> bool:
        """Validate configuration values."""
        try:
            # Validate exchange settings
            if not config.exchange_id:
                raise ValueError("Exchange ID is required")
            
            # Validate trading parameters
            if not config.base_assets:
                raise ValueError("At least one base asset is required")
            if config.min_volume_usd <= 0:
                raise ValueError("Minimum volume must be positive")
            
            # Validate risk parameters
            if not (0 < config.max_position_size <= 1):
                raise ValueError("Max position size must be between 0 and 1")
            if not (0 < config.max_risk_per_trade <= 0.1):
                raise ValueError("Max risk per trade must be between 0 and 0.1")
            if config.max_open_trades < 1:
                raise ValueError("Max open trades must be positive")
            
            # Validate technical parameters
            if config.rsi_period < 2:
                raise ValueError("RSI period must be at least 2")
            if not (0 < config.rsi_oversold < config.rsi_overbought < 100):
                raise ValueError("Invalid RSI overbought/oversold levels")
            
            # Validate pattern parameters
            if not (0 < config.min_pattern_bars < config.max_pattern_bars):
                raise ValueError("Invalid pattern bar range")
            if not (0 < config.pattern_confidence_threshold <= 1):
                raise ValueError("Pattern confidence threshold must be between 0 and 1")
            
            # Validate cross-asset parameters
            if not (0 < config.correlation_threshold <= 1):
                raise ValueError("Correlation threshold must be between 0 and 1")
            if config.min_correlation_window < 10:
                raise ValueError("Minimum correlation window must be at least 10")
            
            # Validate execution parameters
            if config.slippage_tolerance < 0:
                raise ValueError("Slippage tolerance must be non-negative")
            if config.max_retries < 0:
                raise ValueError("Max retries must be non-negative")
            if config.retry_delay < 0:
                raise ValueError("Retry delay must be non-negative")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False
