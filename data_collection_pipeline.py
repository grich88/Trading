"""
Data Collection Pipeline for RSI + Volume Predictive Scoring Model

This script fetches all the necessary data for the Enhanced RSI + Volume Predictive Scoring Model:
1. Price and volume data from cryptocurrency exchanges
2. RSI and other technical indicators calculated locally
3. WebTrend status calculated from moving averages
4. Liquidation data (simulated - would need to be replaced with actual data from Coinglass)

Usage:
    python data_collection_pipeline.py

Requirements:
    - ccxt
    - pandas
    - pandas_ta
    - numpy
    - matplotlib
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self, exchange_id='binance', timeframe='4h', limit=100):
        """
        Initialize the data collector
        
        Parameters:
        -----------
        exchange_id : str
            ID of the exchange to use (default: 'binance')
        timeframe : str
            Timeframe to fetch data for (default: '4h')
        limit : int
            Number of candles to fetch (default: 100)
        """
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.limit = limit
        
        # Initialize exchange
        try:
            self.exchange = getattr(ccxt, exchange_id)()
            logger.info(f"Successfully initialized {exchange_id} exchange")
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            raise
    
    def fetch_ohlcv(self, symbol):
        """
        Fetch OHLCV data for a symbol
        
        Parameters:
        -----------
        symbol : str
            Symbol to fetch data for (e.g., 'BTC/USDT')
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing OHLCV data
        """
        try:
            logger.info(f"Fetching {self.timeframe} data for {symbol} from {self.exchange_id}")
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """
        Calculate technical indicators
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing OHLCV data
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicators
        """
        try:
            logger.info("Calculating technical indicators")
            
            # Calculate RSI
            df['rsi_raw'] = ta.rsi(df['close'], length=14)
            
            # Calculate RSI SMA
            df['rsi_sma'] = ta.sma(df['rsi_raw'], length=3)
            
            # Calculate EMAs for WebTrend
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            df['ema100'] = ta.ema(df['close'], length=100)
            
            # Calculate WebTrend status
            df['webtrend_status'] = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema100'])
            
            # Calculate ATR for volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            logger.info("Successfully calculated technical indicators")
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def simulate_liquidation_data(self, symbol, current_price):
        """
        Simulate liquidation data (in a real implementation, this would be replaced with actual data from Coinglass)
        
        Parameters:
        -----------
        symbol : str
            Symbol to simulate data for
        current_price : float
            Current price of the asset
        
        Returns:
        --------
        dict
            Dictionary containing simulated liquidation data
        """
        logger.info(f"Simulating liquidation data for {symbol}")
        
        asset = symbol.split('/')[0]
        
        # Simulate clusters based on asset
        if asset == "BTC":
            # Support levels at -3% and -5% from current price
            support1 = round(current_price * 0.97, 0)
            support2 = round(current_price * 0.95, 0)
            
            # Resistance levels at +3% and +5% from current price
            resistance1 = round(current_price * 1.03, 0)
            resistance2 = round(current_price * 1.05, 0)
            
            clusters = [
                (support2, 0.7),    # Strong support
                (support1, 0.5),    # Moderate support
                (resistance1, 0.6), # Moderate resistance
                (resistance2, 0.8)  # Strong resistance
            ]
            cleared_zone = False
            
        elif asset == "SOL":
            # Support levels at -3% and -5% from current price
            support1 = round(current_price * 0.97, 2)
            support2 = round(current_price * 0.95, 2)
            
            # Resistance levels at +3% and +5% from current price
            resistance1 = round(current_price * 1.03, 2)
            resistance2 = round(current_price * 1.05, 2)
            
            clusters = [
                (support2, 0.8),    # Strong support
                (support1, 0.6),    # Moderate support
                (resistance1, 0.6), # Moderate resistance
                (resistance2, 0.7)  # Strong resistance
            ]
            cleared_zone = True
            
        elif asset == "BONK":
            # Support levels at -3% and -5% from current price
            support1 = round(current_price * 0.97, 8)
            support2 = round(current_price * 0.95, 8)
            
            # Resistance levels at +3% and +5% from current price
            resistance1 = round(current_price * 1.03, 8)
            resistance2 = round(current_price * 1.05, 8)
            
            clusters = [
                (support2, 0.6),    # Moderate support
                (support1, 0.5),    # Weak support
                (resistance1, 0.5), # Weak resistance
                (resistance2, 0.8)  # Strong resistance
            ]
            cleared_zone = False
            
        else:
            clusters = []
            cleared_zone = False
        
        return {
            'clusters': clusters,
            'cleared_zone': cleared_zone
        }
    
    def collect_data(self, symbols):
        """
        Collect all necessary data for the model
        
        Parameters:
        -----------
        symbols : list
            List of symbols to collect data for
        
        Returns:
        --------
        dict
            Dictionary containing all collected data
        """
        assets_data = {}
        
        for symbol in symbols:
            try:
                # Fetch OHLCV data
                df = self.fetch_ohlcv(symbol)
                if df is None:
                    continue
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Extract asset name
                asset_name = symbol.split('/')[0]
                
                # Get current price
                current_price = df['close'].iloc[-1]
                
                # Simulate liquidation data
                liquidation_data = self.simulate_liquidation_data(symbol, current_price)
                
                # Store data
                assets_data[asset_name] = {
                    'price': df['close'].values.tolist(),
                    'rsi_raw': df['rsi_raw'].values.tolist(),
                    'rsi_sma': df['rsi_sma'].values.tolist(),
                    'volume': df['volume'].values.tolist(),
                    'liquidation_data': liquidation_data,
                    'webtrend_status': bool(df['webtrend_status'].iloc[-1])
                }
                
                # Log summary
                logger.info(f"Summary for {asset_name}:")
                logger.info(f"  Current price: {current_price}")
                logger.info(f"  Current RSI: {df['rsi_raw'].iloc[-1]:.2f}")
                logger.info(f"  Current RSI SMA: {df['rsi_sma'].iloc[-1]:.2f}")
                logger.info(f"  WebTrend status: {'Uptrend' if df['webtrend_status'].iloc[-1] else 'Downtrend'}")
                logger.info(f"  Liquidation clusters: {len(liquidation_data['clusters'])}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return assets_data
    
    def save_data(self, assets_data, filename='model_input_data.json'):
        """
        Save collected data to a file
        
        Parameters:
        -----------
        assets_data : dict
            Dictionary containing collected data
        filename : str
            Name of the file to save data to
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save data to file
            filepath = os.path.join('data', filename)
            with open(filepath, 'w') as f:
                json.dump(assets_data, f, indent=2)
            
            logger.info(f"Successfully saved data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def visualize_data(self, assets_data):
        """
        Visualize collected data
        
        Parameters:
        -----------
        assets_data : dict
            Dictionary containing collected data
        """
        try:
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            for asset, data in assets_data.items():
                # Create figure
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
                
                # Get data
                price = data['price']
                rsi_raw = data['rsi_raw']
                rsi_sma = data['rsi_sma']
                volume = data['volume']
                
                # Create x-axis (timestamps)
                x = range(len(price))
                
                # Plot price
                axes[0].plot(x, price, label='Price')
                axes[0].set_title(f'{asset} Price')
                axes[0].grid(True)
                axes[0].legend()
                
                # Plot RSI
                axes[1].plot(x, rsi_raw, label='RSI')
                axes[1].plot(x, rsi_sma, label='RSI SMA')
                axes[1].axhline(y=70, color='r', linestyle='-', alpha=0.3)
                axes[1].axhline(y=30, color='g', linestyle='-', alpha=0.3)
                axes[1].axhline(y=50, color='k', linestyle='-', alpha=0.3)
                axes[1].set_title('RSI')
                axes[1].set_ylim(0, 100)
                axes[1].grid(True)
                axes[1].legend()
                
                # Plot volume
                axes[2].bar(x, volume, label='Volume')
                axes[2].set_title('Volume')
                axes[2].grid(True)
                
                # Add liquidation clusters
                for price_level, intensity in data['liquidation_data']['clusters']:
                    axes[0].axhline(y=price_level, color='orange', linestyle='--', alpha=intensity)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(f'plots/{asset}_analysis.png')
                plt.close()
                
                logger.info(f"Successfully created visualization for {asset}")
        except Exception as e:
            logger.error(f"Error visualizing data: {e}")


def main():
    """Main function"""
    try:
        logger.info("Starting data collection pipeline")
        
        # Initialize data collector
        collector = DataCollector(exchange_id='binance', timeframe='4h', limit=100)
        
        # Define symbols to collect data for
        symbols = ['BTC/USDT', 'SOL/USDT']
        
        # BONK may not be available on all exchanges
        try:
            collector.exchange.fetch_ticker('BONK/USDT')
            symbols.append('BONK/USDT')
        except:
            logger.warning("BONK/USDT not available on Binance, trying BONK/USDC")
            try:
                collector.exchange.fetch_ticker('BONK/USDC')
                symbols.append('BONK/USDC')
            except:
                logger.warning("BONK not available, skipping")
        
        # Collect data
        assets_data = collector.collect_data(symbols)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        collector.save_data(assets_data, f'model_input_data_{timestamp}.json')
        
        # Visualize data
        collector.visualize_data(assets_data)
        
        logger.info("Data collection pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data collection pipeline: {e}")


if __name__ == "__main__":
    main()
