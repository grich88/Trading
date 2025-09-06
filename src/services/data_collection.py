"""
Data collection service module.

This module provides a specialized service for collecting market data from
various sources, including exchanges, APIs, and on-chain data providers.
"""

import os
import time
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from functools import partial

# Import base service
from src.services.base_service import LongRunningService

# Import data service
from src.services.data_service import DataService

# Import utilities
from src.utils import (
    exception_handler, 
    performance_monitor, 
    async_performance_monitor,
    DataError, 
    timer, 
    memory_usage,
    adaptive_batch_processing,
    async_adaptive_batch_processing
)

# Import configuration
from src.config import (
    DATA_DIR,
    DEFAULT_TIMEFRAME,
    HISTORICAL_DATA_START_DATE,
    HISTORICAL_DATA_END_DATE,
    DATA_COLLECTION_INTERVAL_SECONDS,
    MAX_CONCURRENT_REQUESTS,
    RATE_LIMIT_REQUESTS_PER_MINUTE
)

# Try to import ccxt for exchange data
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


class DataCollectionService(LongRunningService):
    """
    Specialized service for collecting market data.
    
    This service extends LongRunningService to provide:
    - Scheduled data collection
    - Real-time data streaming
    - Multi-exchange support
    - Rate limiting
    - Concurrent data fetching
    - Data validation and normalization
    """
    
    def __init__(self, 
                 data_dir: str = DATA_DIR,
                 default_timeframe: str = DEFAULT_TIMEFRAME,
                 collection_interval: int = DATA_COLLECTION_INTERVAL_SECONDS,
                 max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
                 rate_limit_per_minute: int = RATE_LIMIT_REQUESTS_PER_MINUTE,
                 **kwargs: Any):
        """
        Initialize the data collection service.
        
        Args:
            data_dir: Directory for storing data
            default_timeframe: Default timeframe for data
            collection_interval: Interval between data collections in seconds
            max_concurrent_requests: Maximum number of concurrent requests
            rate_limit_per_minute: Maximum requests per minute
            **kwargs: Additional arguments for LongRunningService
        """
        super().__init__("DataCollectionService", **kwargs)
        
        self.data_dir = data_dir
        self.default_timeframe = default_timeframe
        self.collection_interval = collection_interval
        self.max_concurrent_requests = max_concurrent_requests
        self.rate_limit_per_minute = rate_limit_per_minute
        
        # Create data service for storage and processing
        self.data_service = DataService(data_dir=data_dir, default_timeframe=default_timeframe)
        
        # Collection configuration
        self.collection_targets = {}
        self.collection_thread = None
        self.last_collection_time = None
        
        # Rate limiting
        self.request_timestamps = []
        self.request_semaphore = None
        
        # Exchange clients
        self.exchange_clients = {}
        self.async_exchange_clients = {}
        
        # Collection statistics
        self.collection_stats = {
            "total_collections": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "last_collection_duration": 0,
            "total_data_points": 0,
            "last_collection_time": None
        }
        
        self.logger.info("Data collection service initialized")
    
    def _start_service(self) -> None:
        """Start the data collection service."""
        super()._start_service()
        
        # Initialize rate limiting
        self.request_timestamps = []
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Initialize exchange clients
        self._initialize_exchange_clients()
        
        # Start the data service
        self.data_service.start()
        
        # Start collection thread if we have targets
        if self.collection_targets:
            self._start_collection_thread()
        
        self.logger.info("Data collection service started")
    
    def _stop_service(self) -> None:
        """Stop the data collection service."""
        # Stop collection thread
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=1.0)
        
        # Close async exchange clients
        asyncio.run(self._close_async_exchange_clients())
        
        # Stop the data service
        self.data_service.stop()
        
        super()._stop_service()
        self.logger.info("Data collection service stopped")
    
    def _initialize_exchange_clients(self) -> None:
        """Initialize exchange API clients."""
        if not HAS_CCXT:
            self.logger.warning("CCXT not available. Install it with 'pip install ccxt'")
            return
        
        try:
            # Initialize Binance client
            self.exchange_clients["binance"] = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # Initialize async Binance client
            self.async_exchange_clients["binance"] = ccxt_async.binance({
                'enableRateLimit': True,
            })
            
            self.logger.info("Exchange clients initialized")
        except Exception as e:
            self.logger.error(f"Error initializing exchange clients: {str(e)}")
    
    async def _close_async_exchange_clients(self) -> None:
        """Close async exchange clients."""
        for name, client in self.async_exchange_clients.items():
            try:
                await client.close()
                self.logger.debug(f"Closed async {name} client")
            except Exception as e:
                self.logger.error(f"Error closing async {name} client: {str(e)}")
    
    def _start_collection_thread(self) -> None:
        """Start the data collection thread."""
        if self.collection_thread and self.collection_thread.is_alive():
            self.logger.warning("Collection thread is already running")
            return
        
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            name="DataCollectionThread",
            daemon=True
        )
        self.collection_thread.start()
        self.logger.info("Data collection thread started")
    
    def _collection_loop(self) -> None:
        """Data collection thread function."""
        self.logger.info("Data collection loop started")
        
        while self.running:
            try:
                # Run data collection
                self.collect_all_data()
                
                # Update last collection time
                self.last_collection_time = datetime.now()
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
            
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")
                self.health_metrics["error_count"] += 1
                self.collection_stats["failed_collections"] += 1
                
                # Sleep for a shorter interval on error
                time.sleep(min(self.collection_interval, 10))
        
        self.logger.info("Data collection loop stopped")
    
    def add_collection_target(self, 
                             symbol: str, 
                             timeframes: List[str],
                             exchange: str = "binance",
                             collect_indicators: bool = True) -> None:
        """
        Add a target for data collection.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframes: List of timeframes to collect (e.g., ['1h', '4h', '1d'])
            exchange: Exchange name
            collect_indicators: Whether to calculate indicators
        """
        if symbol not in self.collection_targets:
            self.collection_targets[symbol] = {
                "timeframes": set(timeframes),
                "exchange": exchange,
                "collect_indicators": collect_indicators
            }
        else:
            # Update existing target
            self.collection_targets[symbol]["timeframes"].update(timeframes)
            self.collection_targets[symbol]["exchange"] = exchange
            self.collection_targets[symbol]["collect_indicators"] = collect_indicators
        
        self.logger.info(f"Added collection target: {symbol} ({', '.join(timeframes)})")
        
        # Start collection thread if not already running
        if self.running and not (self.collection_thread and self.collection_thread.is_alive()):
            self._start_collection_thread()
    
    def remove_collection_target(self, symbol: str, timeframes: Optional[List[str]] = None) -> None:
        """
        Remove a target from data collection.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframes: List of timeframes to remove (if None, removes all)
        """
        if symbol not in self.collection_targets:
            self.logger.warning(f"Collection target not found: {symbol}")
            return
        
        if timeframes is None:
            # Remove entire symbol
            del self.collection_targets[symbol]
            self.logger.info(f"Removed collection target: {symbol}")
        else:
            # Remove specific timeframes
            for tf in timeframes:
                if tf in self.collection_targets[symbol]["timeframes"]:
                    self.collection_targets[symbol]["timeframes"].remove(tf)
            
            # If no timeframes left, remove the symbol
            if not self.collection_targets[symbol]["timeframes"]:
                del self.collection_targets[symbol]
            
            self.logger.info(f"Removed timeframes {', '.join(timeframes)} from {symbol}")
    
    @performance_monitor()
    def collect_all_data(self) -> Dict[str, Any]:
        """
        Collect data for all targets.
        
        Returns:
            Dictionary with collection results
        """
        start_time = time.time()
        results = {}
        total_data_points = 0
        
        try:
            # Run async collection
            collection_results = asyncio.run(self._collect_all_data_async())
            
            # Process results
            for symbol, symbol_results in collection_results.items():
                results[symbol] = {}
                for timeframe, data in symbol_results.items():
                    if data is not None:
                        results[symbol][timeframe] = {
                            "success": True,
                            "data_points": len(data),
                            "start_date": data.index[0] if not data.empty else None,
                            "end_date": data.index[-1] if not data.empty else None
                        }
                        total_data_points += len(data)
                    else:
                        results[symbol][timeframe] = {
                            "success": False,
                            "data_points": 0,
                            "start_date": None,
                            "end_date": None
                        }
            
            # Update statistics
            duration = time.time() - start_time
            self.collection_stats["total_collections"] += 1
            self.collection_stats["successful_collections"] += 1
            self.collection_stats["last_collection_duration"] = duration
            self.collection_stats["total_data_points"] += total_data_points
            self.collection_stats["last_collection_time"] = datetime.now()
            
            self.logger.info(
                f"Collected data for {len(results)} symbols, "
                f"{total_data_points} data points in {duration:.2f} seconds"
            )
            
            return results
        
        except Exception as e:
            # Update statistics
            duration = time.time() - start_time
            self.collection_stats["total_collections"] += 1
            self.collection_stats["failed_collections"] += 1
            self.collection_stats["last_collection_duration"] = duration
            self.collection_stats["last_collection_time"] = datetime.now()
            
            self.logger.error(f"Error collecting data: {str(e)}")
            raise DataError(f"Failed to collect data: {str(e)}")
    
    async def _collect_all_data_async(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Collect data for all targets asynchronously.
        
        Returns:
            Dictionary with collected data
        """
        results = {}
        collection_tasks = []
        
        # Create tasks for each symbol and timeframe
        for symbol, config in self.collection_targets.items():
            results[symbol] = {}
            exchange = config["exchange"]
            
            for timeframe in config["timeframes"]:
                collection_tasks.append(
                    self._collect_symbol_data_async(
                        symbol=symbol,
                        timeframe=timeframe,
                        exchange=exchange,
                        collect_indicators=config["collect_indicators"]
                    )
                )
        
        # Run all tasks concurrently
        if collection_tasks:
            completed_tasks = await asyncio.gather(*collection_tasks, return_exceptions=True)
            
            # Process results
            for task_result in completed_tasks:
                if isinstance(task_result, Exception):
                    self.logger.error(f"Task error: {str(task_result)}")
                    continue
                
                symbol, timeframe, data = task_result
                results.setdefault(symbol, {})[timeframe] = data
        
        return results
    
    @async_performance_monitor()
    async def _collect_symbol_data_async(self, 
                                       symbol: str, 
                                       timeframe: str,
                                       exchange: str,
                                       collect_indicators: bool) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """
        Collect data for a symbol and timeframe asynchronously.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            exchange: Exchange name
            collect_indicators: Whether to calculate indicators
            
        Returns:
            Tuple of (symbol, timeframe, DataFrame or None)
        """
        # Apply rate limiting
        await self._apply_rate_limit()
        
        # Use semaphore for concurrency control
        async with self.request_semaphore:
            try:
                # Get exchange client
                if exchange not in self.async_exchange_clients:
                    self.logger.error(f"Exchange not supported: {exchange}")
                    return symbol, timeframe, None
                
                client = self.async_exchange_clients[exchange]
                
                # Get current time
                end_date = datetime.now()
                
                # Calculate start date (1 day ago for higher timeframes, 4 hours for lower)
                if timeframe in ['1d', '4h']:
                    start_date = end_date - timedelta(days=1)
                else:
                    start_date = end_date - timedelta(hours=4)
                
                # Convert dates to timestamps
                start_timestamp = int(start_date.timestamp() * 1000)
                
                # Fetch data
                self.logger.debug(f"Fetching {timeframe} data for {symbol}")
                ohlcv = await client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=start_timestamp,
                    limit=1000  # Maximum limit
                )
                
                if not ohlcv:
                    self.logger.warning(f"No data returned for {symbol} {timeframe}")
                    return symbol, timeframe, None
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate indicators if requested
                if collect_indicators:
                    # Load existing data to ensure enough history for indicators
                    try:
                        # Try to load existing data
                        existing_df = self.data_service.load_data(
                            f"{symbol.replace('/', '_')}_{timeframe}",
                            "historical"
                        )
                        
                        # Combine with new data
                        combined_df = pd.concat([existing_df, df])
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        combined_df.sort_index(inplace=True)
                        
                        # Calculate indicators
                        df_with_indicators = self.data_service.calculate_indicators(combined_df)
                        
                        # Keep only the new data points
                        df = df_with_indicators.loc[df.index]
                    
                    except Exception as e:
                        self.logger.warning(
                            f"Could not load existing data for {symbol} {timeframe}, "
                            f"calculating indicators on new data only: {str(e)}"
                        )
                        df = self.data_service.calculate_indicators(df)
                
                # Save data
                file_path = self.data_service.save_data(
                    df,
                    f"{symbol.replace('/', '_')}_{timeframe}",
                    "historical"
                )
                
                self.logger.info(f"Collected and saved {len(df)} candles for {symbol} {timeframe}")
                return symbol, timeframe, df
            
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol} {timeframe}: {str(e)}")
                return symbol, timeframe, None
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to API requests."""
        now = time.time()
        
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.rate_limit_per_minute:
            # Calculate time to wait
            oldest = min(self.request_timestamps)
            wait_time = 60 - (now - oldest)
            
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add current timestamp
        self.request_timestamps.append(time.time())
    
    def collect_historical_data(self, 
                              symbol: str, 
                              timeframe: str,
                              start_date: str,
                              end_date: Optional[str] = None,
                              exchange: str = "binance",
                              save_data: bool = True) -> pd.DataFrame:
        """
        Collect historical data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            exchange: Exchange name
            save_data: Whether to save the data
            
        Returns:
            DataFrame with historical data
        """
        # Use the data service to fetch historical data
        df = self.data_service.fetch_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=False  # Force fetch from exchange
        )
        
        # Save data if requested
        if save_data and not df.empty:
            self.data_service.save_data(
                df,
                f"{symbol.replace('/', '_')}_{timeframe}",
                "historical"
            )
        
        return df
    
    def get_latest_data(self, 
                      symbol: str, 
                      timeframe: str,
                      limit: int = 100) -> pd.DataFrame:
        """
        Get the latest data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Maximum number of candles to return
            
        Returns:
            DataFrame with latest data
        """
        try:
            # Try to load from file
            df = self.data_service.load_data(
                f"{symbol.replace('/', '_')}_{timeframe}",
                "historical"
            )
            
            # Return the latest candles
            return df.iloc[-limit:] if len(df) > limit else df
        
        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol} {timeframe}: {str(e)}")
            
            # Try to fetch from exchange
            self.logger.info(f"Fetching latest data for {symbol} {timeframe} from exchange")
            
            # Calculate start date based on timeframe and limit
            end_date = datetime.now()
            
            if timeframe == '1m':
                start_date = end_date - timedelta(minutes=limit)
            elif timeframe == '5m':
                start_date = end_date - timedelta(minutes=5 * limit)
            elif timeframe == '15m':
                start_date = end_date - timedelta(minutes=15 * limit)
            elif timeframe == '30m':
                start_date = end_date - timedelta(minutes=30 * limit)
            elif timeframe == '1h':
                start_date = end_date - timedelta(hours=limit)
            elif timeframe == '4h':
                start_date = end_date - timedelta(hours=4 * limit)
            elif timeframe == '1d':
                start_date = end_date - timedelta(days=limit)
            else:
                start_date = end_date - timedelta(days=30)  # Default
            
            # Format dates
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Fetch data
            return self.collect_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date_str,
                end_date=end_date_str,
                save_data=True
            )
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "collection_targets": len(self.collection_targets),
            "exchange_clients": len(self.exchange_clients),
            "collection_stats": self.collection_stats,
            "data_service_health": self.data_service.get_health()
        }


# Test function
def test_data_collection_service():
    """Test the data collection service functionality."""
    # Create data collection service
    service = DataCollectionService()
    
    # Start the service
    service.start()
    
    try:
        # Add collection targets
        service.add_collection_target("BTC/USDT", ["1h", "4h"])
        service.add_collection_target("SOL/USDT", ["1h", "4h"])
        
        # Collect data manually
        results = service.collect_all_data()
        print(f"Collection results: {results}")
        
        # Get latest data
        btc_data = service.get_latest_data("BTC/USDT", "1h", 10)
        print(f"Latest BTC data:\n{btc_data}")
        
        # Collect historical data
        historical_data = service.collect_historical_data(
            symbol="ETH/USDT",
            timeframe="1d",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        print(f"Historical ETH data: {len(historical_data)} candles")
        
        # Check health
        health = service.check_health()
        print(f"Service health: {health}")
        
        # Wait for some scheduled collections
        print("Waiting for scheduled collections...")
        time.sleep(10)
        
        # Check collection stats
        print(f"Collection stats: {service.collection_stats}")
    
    finally:
        # Stop the service
        service.stop()


if __name__ == "__main__":
    test_data_collection_service()
