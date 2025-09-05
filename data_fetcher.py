import pandas as pd
import numpy as np
import ccxt
import talib

class CryptoDataFetcher:
    def __init__(self, exchange='binance'):
        self.exchange = getattr(ccxt, exchange)()
        
    def fetch_ohlcv(self, symbol, timeframe='4h', limit=200):
        """
        Fetch OHLCV data for a cryptocurrency pair
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTC/USDT')
        timeframe : str
            Timeframe for candlestick data (e.g., '4h', '1d')
        limit : int
            Number of candles to fetch
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing OHLCV data with calculated RSI
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate raw RSI
            df['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
            
            # Calculate smoothed RSI (SMA of RSI)
            df['rsi_sma'] = talib.SMA(df['rsi_raw'], timeperiod=3)
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def prepare_model_inputs(self, df):
        """
        Prepare inputs for the RsiVolumePredictor model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing OHLCV data with calculated RSI
            
        Returns:
        --------
        tuple
            (rsi_sma_series, rsi_raw_series, volume_series, price_series)
        """
        if df is None or df.empty:
            return None
        
        # Drop NaN values that might be present after calculating indicators
        df = df.dropna()
        
        rsi_sma_series = df['rsi_sma'].values
        rsi_raw_series = df['rsi_raw'].values
        volume_series = df['volume'].values
        price_series = df['close'].values
        
        return rsi_sma_series, rsi_raw_series, volume_series, price_series
