"""
Data service module.

This module provides a unified service for data operations, including
fetching, processing, and storing market data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

# Import base service
from src.services.base_service import BaseService

# Import utilities
from src.utils import exception_handler, performance_monitor, DataError

# Import configuration
from src.config import (
    DATA_DIR,
    DEFAULT_TIMEFRAME,
    HISTORICAL_DATA_START_DATE,
    HISTORICAL_DATA_END_DATE
)

# Try to import ccxt for exchange data
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


class DataService(BaseService):
    """
    Unified service for data operations.
    
    This service provides centralized functionality for:
    - Fetching historical market data
    - Calculating technical indicators
    - Storing and retrieving data
    - Managing data caching
    """
    
    def __init__(self, 
                 data_dir: str = DATA_DIR,
                 default_timeframe: str = DEFAULT_TIMEFRAME,
                 default_start_date: str = HISTORICAL_DATA_START_DATE,
                 default_end_date: Optional[str] = None,
                 **kwargs: Any):
        """
        Initialize the data service.
        
        Args:
            data_dir: Directory for storing data
            default_timeframe: Default timeframe for data
            default_start_date: Default start date for historical data
            default_end_date: Default end date for historical data
            **kwargs: Additional arguments for BaseService
        """
        super().__init__("DataService", **kwargs)
        
        self.data_dir = data_dir
        self.default_timeframe = default_timeframe
        self.default_start_date = default_start_date
        self.default_end_date = default_end_date or datetime.now().strftime("%Y-%m-%d")
        
        # Create data directories
        self._create_data_directories()
        
        # Initialize exchange API if available
        self.exchange = None
        if HAS_CCXT:
            try:
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                })
                self.logger.info("CCXT initialized with Binance exchange")
            except Exception as e:
                self.logger.error(f"Error initializing CCXT: {str(e)}")
        else:
            self.logger.warning("CCXT not available. Install it with 'pip install ccxt'")
        
        # Data cache
        self.data_cache = {}
        
        self.logger.info("Data service initialized")
    
    def _create_data_directories(self) -> None:
        """Create necessary data directories."""
        # Main data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Subdirectories
        subdirs = [
            "historical",
            "indicators",
            "models",
            "backtest_results"
        ]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.data_dir, subdir), exist_ok=True)
        
        self.logger.debug(f"Created data directories in {self.data_dir}")
    
    @performance_monitor()
    def fetch_historical_data(self, 
                            symbol: str, 
                            timeframe: Optional[str] = None,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch historical market data.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with historical data
        """
        # Set default values
        timeframe = timeframe or self.default_timeframe
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
        
        # Check cache
        if use_cache and cache_key in self.data_cache:
            self.logger.debug(f"Using cached data for {cache_key}")
            return self.data_cache[cache_key].copy()
        
        # Check if we have CCXT available
        if not HAS_CCXT or not self.exchange:
            raise DataError("CCXT not available for fetching historical data")
        
        try:
            self.logger.info(f"Fetching {timeframe} data for {symbol} ({start_date} to {end_date})")
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=start_timestamp,
                limit=1000  # Maximum limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Cache the result
            self.data_cache[cache_key] = df.copy()
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise DataError(f"Failed to fetch historical data for {symbol}: {str(e)}")
    
    @performance_monitor()
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        try:
            # RSI
            delta = result['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            result['rsi_raw'] = 100 - (100 / (1 + rs))
            result['rsi_sma'] = result['rsi_raw'].rolling(window=10).mean()
            
            # Moving averages
            result['ema20'] = result['close'].ewm(span=20, adjust=False).mean()
            result['ema50'] = result['close'].ewm(span=50, adjust=False).mean()
            result['ema100'] = result['close'].ewm(span=100, adjust=False).mean()
            result['sma200'] = result['close'].rolling(window=200).mean()
            
            # Bollinger Bands
            result['bb_middle'] = result['close'].rolling(window=20).mean()
            result['bb_std'] = result['close'].rolling(window=20).std()
            result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * 2)
            result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * 2)
            
            # MACD
            result['ema12'] = result['close'].ewm(span=12, adjust=False).mean()
            result['ema26'] = result['close'].ewm(span=26, adjust=False).mean()
            result['macd'] = result['ema12'] - result['ema26']
            result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
            result['macd_hist'] = result['macd'] - result['macd_signal']
            
            # ATR (Average True Range)
            high_low = result['high'] - result['low']
            high_close = (result['high'] - result['close'].shift()).abs()
            low_close = (result['low'] - result['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            result['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            result['volume_sma'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma']
            
            # Clean up NaN values
            result.fillna(0, inplace=True)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            raise DataError(f"Failed to calculate indicators: {str(e)}")
    
    def save_data(self, 
                df: pd.DataFrame, 
                name: str, 
                subdir: str = "historical") -> str:
        """
        Save data to a file.
        
        Args:
            df: DataFrame to save
            name: File name (without extension)
            subdir: Subdirectory within data_dir
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        directory = os.path.join(self.data_dir, subdir)
        os.makedirs(directory, exist_ok=True)
        
        # Create file path
        file_path = os.path.join(directory, f"{name}.csv")
        
        try:
            # Save to CSV
            df.to_csv(file_path)
            self.logger.info(f"Saved data to {file_path}")
            return file_path
        
        except Exception as e:
            self.logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise DataError(f"Failed to save data: {str(e)}")
    
    def load_data(self, 
                name: str, 
                subdir: str = "historical") -> pd.DataFrame:
        """
        Load data from a file.
        
        Args:
            name: File name (without extension)
            subdir: Subdirectory within data_dir
            
        Returns:
            DataFrame with loaded data
        """
        # Create file path
        file_path = os.path.join(self.data_dir, subdir, f"{name}.csv")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise DataError(f"Data file not found: {file_path}")
            
            # Load from CSV
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded data from {file_path}")
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise DataError(f"Failed to load data: {str(e)}")
    
    def get_data_for_assets(self, 
                          assets: List[str], 
                          timeframe: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple assets.
        
        Args:
            assets: List of asset symbols (e.g., ['BTC', 'SOL', 'BONK'])
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary of DataFrames for each asset
        """
        result = {}
        
        for asset in assets:
            try:
                # Create full symbol
                symbol = f"{asset}/USDT"
                
                # Fetch data
                df = self.fetch_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache
                )
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Add to result
                result[asset] = df
            
            except Exception as e:
                self.logger.error(f"Error getting data for {asset}: {str(e)}")
                # Continue with other assets
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "cache_size": len(self.data_cache),
            "exchange_available": HAS_CCXT and self.exchange is not None
        }


# Test function
def test_data_service():
    """Test the data service functionality."""
    # Create data service
    service = DataService()
    
    # Start the service
    service.start()
    
    try:
        # Fetch data for BTC
        df = service.fetch_historical_data(
            symbol="BTC/USDT",
            timeframe="4h",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        print(f"Fetched {len(df)} candles for BTC/USDT")
        print(df.head())
        
        # Calculate indicators
        df_with_indicators = service.calculate_indicators(df)
        print("\nIndicators:")
        print(df_with_indicators.columns.tolist())
        
        # Save data
        file_path = service.save_data(df_with_indicators, "btc_4h_sample", "historical")
        print(f"\nSaved data to {file_path}")
        
        # Load data
        loaded_df = service.load_data("btc_4h_sample", "historical")
        print(f"\nLoaded data with {len(loaded_df)} rows")
        
        # Get data for multiple assets
        assets_data = service.get_data_for_assets(
            assets=["BTC", "SOL"],
            timeframe="4h",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        for asset, data in assets_data.items():
            print(f"\nFetched {len(data)} candles for {asset}")
        
        # Check health
        health = service.check_health()
        print(f"\nService health: {health}")
    
    finally:
        # Stop the service
        service.stop()


if __name__ == "__main__":
    test_data_service()
