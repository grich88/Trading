"""
Simple Data Collector for RSI + Volume Predictive Scoring Model

This script collects current cryptocurrency data and calculates technical indicators
without relying on pandas_ta, which seems to have compatibility issues.
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index)
    
    Parameters:
    -----------
    prices : array-like
        Price series
    period : int
        RSI period
        
    Returns:
    --------
    array-like
        RSI values
    """
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Create seed values
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    # Calculate RSI
    rs = up / down if down != 0 else float('inf')
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    # Calculate RSI for the rest of the data
    for i in range(period, len(prices)):
        delta = deltas[i-1]
        
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        
        rs = up / down if down != 0 else float('inf')
        rsi[i] = 100. - 100. / (1. + rs)
        
    return rsi

def calculate_sma(values, period):
    """
    Calculate Simple Moving Average
    
    Parameters:
    -----------
    values : array-like
        Values to calculate SMA for
    period : int
        SMA period
        
    Returns:
    --------
    array-like
        SMA values
    """
    sma = np.zeros_like(values)
    for i in range(period, len(values)):
        sma[i] = np.mean(values[i-period+1:i+1])
    return sma

def calculate_ema(values, period):
    """
    Calculate Exponential Moving Average
    
    Parameters:
    -----------
    values : array-like
        Values to calculate EMA for
    period : int
        EMA period
        
    Returns:
    --------
    array-like
        EMA values
    """
    ema = np.zeros_like(values)
    # Start with SMA
    ema[:period] = np.mean(values[:period])
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA
    for i in range(period, len(values)):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
        
    return ema

def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range
    
    Parameters:
    -----------
    high : array-like
        High prices
    low : array-like
        Low prices
    close : array-like
        Close prices
    period : int
        ATR period
        
    Returns:
    --------
    array-like
        ATR values
    """
    # Calculate True Range
    tr = np.zeros(len(high))
    
    # First TR is high - low
    tr[0] = high[0] - low[0]
    
    # Calculate TR for the rest
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATR
    atr = np.zeros_like(tr)
    atr[:period] = np.mean(tr[:period])
    
    # Calculate ATR for the rest
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr

def fetch_and_process_data(symbol, exchange_id='binance', timeframe='4h', limit=100):
    """
    Fetch and process cryptocurrency data
    
    Parameters:
    -----------
    symbol : str
        Symbol to fetch data for (e.g., 'BTC/USDT')
    exchange_id : str
        Exchange ID (default: 'binance')
    timeframe : str
        Timeframe (default: '4h')
    limit : int
        Number of candles to fetch (default: 100)
        
    Returns:
    --------
    dict
        Processed data
    """
    try:
        logger.info(f"Fetching {timeframe} data for {symbol} from {exchange_id}")
        
        # Initialize exchange
        exchange = getattr(ccxt, exchange_id)()
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Extract data as numpy arrays
        timestamps = df['timestamp'].values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Calculate indicators
        rsi_raw = calculate_rsi(closes)
        rsi_sma = calculate_sma(rsi_raw, 3)
        
        ema20 = calculate_ema(closes, 20)
        ema50 = calculate_ema(closes, 50)
        ema100 = calculate_ema(closes, 100)
        
        # Calculate WebTrend status
        webtrend_status = (closes[-1] > ema20[-1]) and (ema20[-1] > ema50[-1]) and (ema50[-1] > ema100[-1])
        
        # Calculate volume metrics
        volume_sma5 = calculate_sma(volumes, 5)
        volume_sma20 = calculate_sma(volumes, 20)
        
        # Calculate price changes
        price_changes = np.zeros_like(closes)
        price_changes[1:] = closes[1:] - closes[:-1]
        
        # Calculate green/red candles
        is_green = price_changes > 0
        
        # Calculate streaks
        green_streak = np.zeros_like(closes, dtype=int)
        red_streak = np.zeros_like(closes, dtype=int)
        
        streak = 0
        for i in range(1, len(closes)):
            if is_green[i]:
                if is_green[i-1]:
                    streak += 1
                else:
                    streak = 1
                green_streak[i] = streak
                red_streak[i] = 0
            else:
                if not is_green[i-1]:
                    streak += 1
                else:
                    streak = 1
                red_streak[i] = streak
                green_streak[i] = 0
        
        # Extract asset name
        asset_name = symbol.split('/')[0]
        
        # Create data dictionary
        asset_data = {
            'asset': asset_name,
            'symbol': symbol,
            'timestamp': timestamps[-1].astype(str),
            'price': float(closes[-1]),
            'price_series': closes.tolist(),
            'volume': float(volumes[-1]),
            'volume_series': volumes.tolist(),
            'rsi_raw': float(rsi_raw[-1]),
            'rsi_raw_series': rsi_raw.tolist(),
            'rsi_sma': float(rsi_sma[-1]),
            'rsi_sma_series': rsi_sma.tolist(),
            'webtrend_status': bool(webtrend_status),
            'green_streak': int(green_streak[-1]),
            'red_streak': int(red_streak[-1]),
            'technical_data': {
                'ema20': float(ema20[-1]),
                'ema50': float(ema50[-1]),
                'ema100': float(ema100[-1]),
                'volume_sma5': float(volume_sma5[-1]),
                'volume_sma20': float(volume_sma20[-1])
            }
        }
        
        # Log summary
        logger.info(f"Summary for {asset_name}:")
        logger.info(f"  Current price: {asset_data['price']}")
        logger.info(f"  Current RSI: {asset_data['rsi_raw']:.2f}")
        logger.info(f"  Current RSI SMA: {asset_data['rsi_sma']:.2f}")
        logger.info(f"  WebTrend status: {'Uptrend' if asset_data['webtrend_status'] else 'Downtrend'}")
        logger.info(f"  Green streak: {asset_data['green_streak']}")
        logger.info(f"  Red streak: {asset_data['red_streak']}")
        
        return asset_data
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

def visualize_data(asset_data):
    """
    Visualize data for an asset
    
    Parameters:
    -----------
    asset_data : dict
        Asset data
    """
    try:
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        
        # Create subplots
        gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        
        # Plot price
        ax1 = plt.subplot(gs[0])
        ax1.plot(asset_data['price_series'], label='Price')
        ax1.set_title(f"{asset_data['asset']} Price")
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
                   f"Green Streak: {asset_data['green_streak']}",
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"plots/{asset_data['asset']}_analysis_{timestamp}.png")
        plt.close()
        
        logger.info(f"Successfully created visualization for {asset_data['asset']}")
    except Exception as e:
        logger.error(f"Error visualizing data: {e}")

def save_data(data, filename):
    """
    Save data to a JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    filename : str
        Filename to save data to
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save data to file
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Successfully saved data to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return None

def main():
    """Main function"""
    try:
        logger.info("Starting data collection")
        
        # Define symbols to collect data for
        symbols = ['BTC/USDT', 'SOL/USDT']
        
        # Try to add BONK
        try:
            exchange = ccxt.binance()
            exchange.fetch_ticker('BONK/USDT')
            symbols.append('BONK/USDT')
        except:
            logger.warning("BONK/USDT not available on Binance, trying BONK/USDC")
            try:
                exchange.fetch_ticker('BONK/USDC')
                symbols.append('BONK/USDC')
            except:
                logger.warning("BONK not available, skipping")
        
        # Collect data
        data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'exchange': 'binance',
                'timeframe': '4h'
            },
            'assets': {}
        }
        
        for symbol in symbols:
            asset_data = fetch_and_process_data(symbol)
            if asset_data:
                asset_name = asset_data['asset']
                data['assets'][asset_name] = asset_data
                
                # Visualize data
                visualize_data(asset_data)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_data(data, f"model_data_{timestamp}.json")
        
        logger.info("Data collection completed successfully")
        
        # Prepare model input
        model_input = {}
        for asset_name, asset_data in data['assets'].items():
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
        
        # Save model input
        save_data(model_input, f"model_input_{timestamp}.json")
        
        logger.info("\nNow we need liquidation data from Coinglass for each asset.")
        logger.info("Please visit https://www.coinglass.com/LiquidationData to get this information.")
        
        return data, model_input
        
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
