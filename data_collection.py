import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import requests
from concurrent.futures import ThreadPoolExecutor

class DataCollector:
    def __init__(self, exchange_id: str = 'binance', rate_limit: int = 10):
        self.exchange = getattr(ccxt, exchange_id)()
        self.rate_limit = rate_limit  # requests per second
        self.last_request_time = 0
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _rate_limit_request(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1.0/self.rate_limit:
            time.sleep(1.0/self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def fetch_ohlcv(self, 
                    symbol: str,
                    timeframe: str = '1h',
                    since: Optional[int] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch OHLCV data with rate limiting and error handling."""
        try:
            self._rate_limit_request()
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_order_book(self, symbol: str, limit: Optional[int] = None) -> Dict:
        """Fetch order book data."""
        try:
            self._rate_limit_request()
            order_book = self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': pd.DataFrame(order_book['bids'], columns=['price', 'amount']),
                'asks': pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
            }
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {'bids': pd.DataFrame(), 'asks': pd.DataFrame()}
    
    def fetch_trades(self, symbol: str, since: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetch recent trades."""
        try:
            self._rate_limit_request()
            trades = self.exchange.fetch_trades(symbol, since, limit)
            
            df = pd.DataFrame(trades)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            return pd.DataFrame()

class MarketDataAggregator:
    def __init__(self, collectors: Dict[str, DataCollector]):
        self.collectors = collectors
        self.logger = logging.getLogger(__name__)
    
    def fetch_multi_exchange_data(self,
                                symbol: str,
                                timeframe: str = '1h',
                                lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Fetch data from multiple exchanges."""
        since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.collectors)) as executor:
            future_to_exchange = {
                executor.submit(collector.fetch_ohlcv, symbol, timeframe, since): exchange_id
                for exchange_id, collector in self.collectors.items()
            }
            
            for future in future_to_exchange:
                exchange_id = future_to_exchange[future]
                try:
                    data = future.result()
                    if not data.empty:
                        results[exchange_id] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data from {exchange_id}: {str(e)}")
        
        return results
    
    def calculate_arbitrage_opportunities(self,
                                       symbol: str,
                                       min_profit_threshold: float = 0.001) -> List[Dict]:
        """Find arbitrage opportunities across exchanges."""
        opportunities = []
        
        try:
            # Fetch order books from all exchanges
            order_books = {
                exchange_id: collector.fetch_order_book(symbol)
                for exchange_id, collector in self.collectors.items()
            }
            
            # Find best bid and ask across exchanges
            best_bids = {
                exchange: ob['bids'].iloc[0] if not ob['bids'].empty else pd.Series([0, 0])
                for exchange, ob in order_books.items()
            }
            best_asks = {
                exchange: ob['asks'].iloc[0] if not ob['asks'].empty else pd.Series([float('inf'), 0])
                for exchange, ob in order_books.items()
            }
            
            # Find opportunities
            for buy_exchange, buy_price in best_asks.items():
                for sell_exchange, sell_price in best_bids.items():
                    if buy_exchange != sell_exchange:
                        profit = (sell_price['price'] - buy_price['price']) / buy_price['price']
                        if profit > min_profit_threshold:
                            opportunities.append({
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': float(buy_price['price']),
                                'sell_price': float(sell_price['price']),
                                'profit_percentage': float(profit),
                                'max_volume': min(float(buy_price['amount']), float(sell_price['amount']))
                            })
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage opportunities: {str(e)}")
        
        return opportunities

class LiquidationDataCollector:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://open-api.coinglass.com/api/pro/v1"
        self.logger = logging.getLogger(__name__)
    
    def fetch_liquidation_data(self, symbol: str) -> Dict:
        """Fetch liquidation data from CoinGlass."""
        try:
            headers = {'coinglassSecret': self.api_key} if self.api_key else {}
            url = f"{self.base_url}/futures/liquidation?symbol={symbol}"
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return {
                'liquidations': pd.DataFrame(data['data']['list']),
                'total_long_liquidation': data['data']['totalLongLiquidation'],
                'total_short_liquidation': data['data']['totalShortLiquidation'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Error fetching liquidation data: {str(e)}")
            return {
                'liquidations': pd.DataFrame(),
                'total_long_liquidation': 0,
                'total_short_liquidation': 0,
                'timestamp': datetime.now()
            }
    
    def fetch_liquidation_distribution(self, symbol: str) -> pd.DataFrame:
        """Fetch liquidation price distribution."""
        try:
            headers = {'coinglassSecret': self.api_key} if self.api_key else {}
            url = f"{self.base_url}/futures/liquidation_distribution?symbol={symbol}"
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return pd.DataFrame(data['data'])
        except Exception as e:
            self.logger.error(f"Error fetching liquidation distribution: {str(e)}")
            return pd.DataFrame()

class TechnicalDataCalculator:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df

    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20, num_points: int = 3) -> Tuple[List[float], List[float]]:
        """Calculate support and resistance levels."""
        highs = df['high'].rolling(window=window, center=True).apply(
            lambda x: x[len(x)//2] == max(x)
        )
        lows = df['low'].rolling(window=window, center=True).apply(
            lambda x: x[len(x)//2] == min(x)
        )
        
        resistance_points = df[highs]['high'].tolist()[-num_points:]
        support_points = df[lows]['low'].tolist()[-num_points:]
        
        return support_points, resistance_points
