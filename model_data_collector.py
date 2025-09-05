"""
Model Data Collector - Implements free and available data sources for the
Enhanced RSI + Volume Predictive Scoring Model

This script:
1. Fetches OHLCV data using CCXT (free)
2. Calculates technical indicators locally using pandas_ta (free)
3. Provides a placeholder for liquidation data (requires manual input or scraping)
4. Calculates WebTrend status using moving averages (free)

Usage:
    python model_data_collector.py [--exchange binance] [--save-data] [--visualize]
"""

import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import logging
from datetime import datetime

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

class ModelDataCollector:
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
    
    def fetch_ohlcv_data(self, symbol):
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
            logger.info(f"Successfully fetched data for {symbol}: {len(df)} candles")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators using pandas_ta
        
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
            
            # Calculate RSI (14-period)
            df['rsi_raw'] = ta.rsi(df['close'], length=14)
            
            # Calculate RSI SMA (3-period)
            df['rsi_sma'] = ta.sma(df['rsi_raw'], length=3)
            
            # Calculate moving averages for WebTrend
            df['ema20'] = ta.ema(df['close'], length=20)
            df['ema50'] = ta.ema(df['close'], length=50)
            df['ema100'] = ta.ema(df['close'], length=100)
            
            # Calculate WebTrend status (simplified version)
            df['webtrend_status'] = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema100'])
            
            # Calculate ATR for volatility
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Calculate volume metrics
            df['volume_sma5'] = ta.sma(df['volume'], length=5)
            df['volume_sma20'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma20']
            
            # Calculate price change for green/red candles
            df['price_change'] = df['close'] - df['close'].shift(1)
            df['is_green'] = df['price_change'] > 0
            
            # Calculate streaks
            df['green_streak'] = 0
            df['red_streak'] = 0
            
            # Calculate green and red streaks
            streak = 0
            for i in range(1, len(df)):
                if df['is_green'].iloc[i]:
                    if df['is_green'].iloc[i-1]:
                        streak += 1
                    else:
                        streak = 1
                    df.loc[df.index[i], 'green_streak'] = streak
                    df.loc[df.index[i], 'red_streak'] = 0
                else:
                    if not df['is_green'].iloc[i-1]:
                        streak += 1
                    else:
                        streak = 1
                    df.loc[df.index[i], 'red_streak'] = streak
                    df.loc[df.index[i], 'green_streak'] = 0
            
            logger.info("Successfully calculated technical indicators")
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def process_asset(self, symbol):
        """
        Process a single asset
        
        Parameters:
        -----------
        symbol : str
            Symbol to process (e.g., 'BTC/USDT')
        
        Returns:
        --------
        dict
            Dictionary containing processed data
        """
        try:
            # Fetch OHLCV data
            df = self.fetch_ohlcv_data(symbol)
            if df is None:
                return None
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Extract asset name
            asset_name = symbol.split('/')[0]
            
            # Create data dictionary
            asset_data = {
                'asset': asset_name,
                'symbol': symbol,
                'timestamp': df.index[-1].isoformat(),
                'price': df['close'].iloc[-1],
                'price_series': df['close'].tolist(),
                'volume': df['volume'].iloc[-1],
                'volume_series': df['volume'].tolist(),
                'rsi_raw': df['rsi_raw'].iloc[-1],
                'rsi_raw_series': df['rsi_raw'].tolist(),
                'rsi_sma': df['rsi_sma'].iloc[-1],
                'rsi_sma_series': df['rsi_sma'].tolist(),
                'atr': df['atr'].iloc[-1],
                'webtrend_status': bool(df['webtrend_status'].iloc[-1]),
                'volume_ratio': df['volume_ratio'].iloc[-1],
                'green_streak': int(df['green_streak'].iloc[-1]),
                'red_streak': int(df['red_streak'].iloc[-1]),
                'technical_data': {
                    'ema20': df['ema20'].iloc[-1],
                    'ema50': df['ema50'].iloc[-1],
                    'ema100': df['ema100'].iloc[-1],
                    'volume_sma5': df['volume_sma5'].iloc[-1],
                    'volume_sma20': df['volume_sma20'].iloc[-1],
                }
            }
            
            # Log summary
            logger.info(f"Summary for {asset_name}:")
            logger.info(f"  Current price: {asset_data['price']}")
            logger.info(f"  Current RSI: {asset_data['rsi_raw']:.2f}")
            logger.info(f"  Current RSI SMA: {asset_data['rsi_sma']:.2f}")
            logger.info(f"  WebTrend status: {'Uptrend' if asset_data['webtrend_status'] else 'Downtrend'}")
            logger.info(f"  Volume ratio: {asset_data['volume_ratio']:.2f}")
            logger.info(f"  Green streak: {asset_data['green_streak']}")
            logger.info(f"  Red streak: {asset_data['red_streak']}")
            
            return asset_data
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None
    
    def collect_data(self, symbols):
        """
        Collect data for multiple symbols
        
        Parameters:
        -----------
        symbols : list
            List of symbols to collect data for
        
        Returns:
        --------
        dict
            Dictionary containing data for all symbols
        """
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'exchange': self.exchange_id,
                'timeframe': self.timeframe,
                'limit': self.limit
            },
            'assets': {}
        }
        
        for symbol in symbols:
            asset_data = self.process_asset(symbol)
            if asset_data:
                asset_name = asset_data['asset']
                data['assets'][asset_name] = asset_data
        
        return data
    
    def save_data(self, data, filename=None):
        """
        Save data to a JSON file
        
        Parameters:
        -----------
        data : dict
            Data to save
        filename : str, optional
            Filename to save data to (default: model_data_{timestamp}.json)
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_data_{timestamp}.json"
            
            # Save data to file
            filepath = os.path.join('data', filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Successfully saved data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return None
    
    def visualize_data(self, data):
        """
        Visualize data
        
        Parameters:
        -----------
        data : dict
            Data to visualize
        """
        try:
            # Create plots directory if it doesn't exist
            os.makedirs('plots', exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for asset_name, asset_data in data['assets'].items():
                # Create figure
                fig = plt.figure(figsize=(12, 12))
                
                # Create subplots
                gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
                
                # Plot price
                ax1 = plt.subplot(gs[0])
                ax1.plot(asset_data['price_series'], label='Price')
                ax1.set_title(f"{asset_name} Price ({data['metadata']['timeframe']} timeframe)")
                ax1.grid(True)
                ax1.legend()
                
                # Plot RSI
                ax2 = plt.subplot(gs[1])
                ax2.plot(asset_data['rsi_raw_series'], label='RSI', color='blue')
                ax2.plot(asset_data['rsi_sma_series'], label='RSI SMA', color='orange')
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                ax2.axhline(y=50, color='k', linestyle='--', alpha=0.3)
                ax2.set_title('RSI')
                ax2.set_ylim(0, 100)
                ax2.grid(True)
                ax2.legend()
                
                # Plot volume
                ax3 = plt.subplot(gs[2])
                ax3.bar(range(len(asset_data['volume_series'])), asset_data['volume_series'], label='Volume')
                ax3.set_title('Volume')
                ax3.grid(True)
                
                # Plot WebTrend status
                ax4 = plt.subplot(gs[3])
                ax4.plot(asset_data['price_series'], label='Price', alpha=0.3)
                
                # Add technical indicators
                ema20 = asset_data['technical_data']['ema20']
                ema50 = asset_data['technical_data']['ema50']
                ema100 = asset_data['technical_data']['ema100']
                
                # Add horizontal lines for latest EMAs
                ax4.axhline(y=ema20, color='green', linestyle='--', label=f'EMA20: {ema20:.2f}')
                ax4.axhline(y=ema50, color='blue', linestyle='--', label=f'EMA50: {ema50:.2f}')
                ax4.axhline(y=ema100, color='red', linestyle='--', label=f'EMA100: {ema100:.2f}')
                
                ax4.set_title('WebTrend Components')
                ax4.grid(True)
                ax4.legend()
                
                # Add summary text
                plt.figtext(0.5, 0.01, 
                           f"Current Price: {asset_data['price']:.2f} | RSI: {asset_data['rsi_raw']:.2f} | "
                           f"WebTrend: {'Uptrend' if asset_data['webtrend_status'] else 'Downtrend'} | "
                           f"Volume Ratio: {asset_data['volume_ratio']:.2f}",
                           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
                
                # Save figure
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(f"plots/{asset_name}_analysis_{timestamp}.png")
                plt.close()
                
                logger.info(f"Successfully created visualization for {asset_name}")
        except Exception as e:
            logger.error(f"Error visualizing data: {e}")
    
    def prepare_model_input(self, data):
        """
        Prepare input for the Enhanced RSI + Volume Predictive Scoring Model
        
        Parameters:
        -----------
        data : dict
            Collected data
        
        Returns:
        --------
        dict
            Dictionary formatted for model input
        """
        model_input = {}
        
        for asset_name, asset_data in data['assets'].items():
            # Create model input for asset
            model_input[asset_name] = {
                'price': asset_data['price_series'],
                'rsi_raw': asset_data['rsi_raw_series'],
                'rsi_sma': asset_data['rsi_sma_series'],
                'volume': asset_data['volume_series'],
                'webtrend_status': asset_data['webtrend_status'],
                'liquidation_data': {
                    'clusters': [],  # Placeholder for liquidation data
                    'cleared_zone': False  # Placeholder for liquidation data
                }
            }
        
        return model_input


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Collect data for the Enhanced RSI + Volume Predictive Scoring Model')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange to use (default: binance)')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe to use (default: 4h)')
    parser.add_argument('--limit', type=int, default=100, help='Number of candles to fetch (default: 100)')
    parser.add_argument('--save-data', action='store_true', help='Save data to file')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    try:
        logger.info("Starting data collection")
        
        # Initialize data collector
        collector = ModelDataCollector(
            exchange_id=args.exchange,
            timeframe=args.timeframe,
            limit=args.limit
        )
        
        # Define symbols to collect data for
        symbols = ['BTC/USDT', 'SOL/USDT']
        
        # Try to add BONK
        try:
            collector.exchange.fetch_ticker('BONK/USDT')
            symbols.append('BONK/USDT')
        except:
            logger.warning("BONK/USDT not available on this exchange, trying BONK/USDC")
            try:
                collector.exchange.fetch_ticker('BONK/USDC')
                symbols.append('BONK/USDC')
            except:
                logger.warning("BONK not available, skipping")
        
        # Collect data
        data = collector.collect_data(symbols)
        
        # Save data if requested
        if args.save_data:
            collector.save_data(data)
        
        # Visualize data if requested
        if args.visualize:
            collector.visualize_data(data)
        
        # Prepare model input
        model_input = collector.prepare_model_input(data)
        
        # Print missing data
        logger.info("\nMissing data for model input:")
        for asset_name in model_input:
            logger.info(f"  {asset_name} - Liquidation data: Requires manual input or scraping from Coinglass")
        
        logger.info("Data collection completed successfully")
        
        # Return data for further processing
        return data, model_input
        
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        return None, None


if __name__ == "__main__":
    main()
