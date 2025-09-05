"""
Historical Data Collector for RSI + Volume Predictive Scoring Model Backtesting

This script fetches historical OHLCV data for cryptocurrency assets and calculates
all necessary indicators for the Enhanced RSI + Volume Predictive Scoring Model.

Usage:
    python historical_data_collector.py --start-date 2023-01-01 --end-date 2025-08-01
"""

import ccxt
import pandas as pd
import numpy as np
import os
import argparse
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backtest.log"),
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
    deltas = np.append(deltas, 0)  # Add 0 to make the array the same length as prices
    
    # Create seed values
    up = np.zeros_like(deltas)
    down = np.zeros_like(deltas)
    
    # Separate upward and downward movements
    up[deltas > 0] = deltas[deltas > 0]
    down[deltas < 0] = -deltas[deltas < 0]
    
    # Calculate rolling averages
    roll_up = np.zeros_like(up)
    roll_down = np.zeros_like(down)
    
    # First period values
    roll_up[:period] = np.nan
    roll_down[:period] = np.nan
    roll_up[period] = np.mean(up[1:period+1])
    roll_down[period] = np.mean(down[1:period+1])
    
    # Calculate remaining values
    for i in range(period+1, len(prices)):
        roll_up[i] = (roll_up[i-1] * (period-1) + up[i]) / period
        roll_down[i] = (roll_down[i-1] * (period-1) + down[i]) / period
    
    # Calculate RS and RSI
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
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
    sma[:period] = np.nan
    
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
    ema[:period] = np.nan
    
    # Start with SMA
    ema[period] = np.mean(values[1:period+1])
    
    # Calculate multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA
    for i in range(period+1, len(values)):
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
    atr[:period] = np.nan
    atr[period] = np.mean(tr[1:period+1])
    
    # Calculate ATR for the rest
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
    return atr

def calculate_indicators(df):
    """
    Calculate all indicators needed for the model
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate RSI
    df["rsi_raw"] = calculate_rsi(df["close"].values)
    
    # Calculate RSI SMA
    df["rsi_sma"] = calculate_sma(df["rsi_raw"].values, 3)
    
    # Calculate EMAs for WebTrend
    df["ema20"] = calculate_ema(df["close"].values, 20)
    df["ema50"] = calculate_ema(df["close"].values, 50)
    df["ema100"] = calculate_ema(df["close"].values, 100)
    
    # Calculate WebTrend status
    df["webtrend_status"] = ((df["close"] > df["ema20"]) & 
                              (df["ema20"] > df["ema50"]) & 
                              (df["ema50"] > df["ema100"])) .astype(int)

    # Approximate WebTrend overlay band
    try:
        wt_mid = (df["ema20"] + df["ema50"]) / 2.0
        band_width = (df["ema50"] - df["ema100"]).abs().rolling(5).mean()
        df["wt_mid"] = wt_mid
        df["wt_upper"] = wt_mid + 0.6 * band_width
        df["wt_lower"] = wt_mid - 0.6 * band_width
    except Exception:
        pass

    # Calculate ATR
    df["atr"] = calculate_atr(df["high"].values, df["low"].values, df["close"].values)
    
    # Calculate volume metrics
    df["volume_sma5"] = calculate_sma(df["volume"].values, 5)
    df["volume_sma20"] = calculate_sma(df["volume"].values, 20)
    df["volume_ratio"] = df["volume"] / df["volume_sma20"]
    
    # Calculate price changes
    df["price_change"] = df["close"].diff()
    df["is_green"] = df["price_change"] > 0
    
    # Calculate green and red streaks
    df["green_streak"] = 0
    df["red_streak"] = 0
    
    streak = 0
    for i in range(1, len(df)):
        if df["is_green"].iloc[i]:
            if df["is_green"].iloc[i-1]:
                streak += 1
            else:
                streak = 1
            df.loc[df.index[i], "green_streak"] = streak
            df.loc[df.index[i], "red_streak"] = 0
        else:
            if not df["is_green"].iloc[i-1]:
                streak += 1
            else:
                streak = 1
            df.loc[df.index[i], "red_streak"] = streak
            df.loc[df.index[i], "green_streak"] = 0
    
    return df

def fetch_historical_data(symbol, exchange_id="binance", timeframe="4h", start_date=None, end_date=None):
    """
    Fetch historical OHLCV data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol to fetch data for (e.g., "BTC/USDT")
    exchange_id : str
        Exchange ID (default: "binance")
    timeframe : str
        Timeframe (default: "4h")
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str
        End date in "YYYY-MM-DD" format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing OHLCV data
    """
    try:
        logger.info(f"Fetching {timeframe} data for {symbol} from {exchange_id}")
        
        # Initialize exchange
        exchange = getattr(ccxt, exchange_id)()
        
        # Convert dates to timestamps
        since = exchange.parse8601(start_date + "T00:00:00Z") if start_date else None
        until = exchange.parse8601(end_date + "T23:59:59Z") if end_date else None
        
        # Fetch data in chunks (most exchanges limit results per request)
        all_ohlcv = []
        
        if since:
            current_since = since
            while True:
                logger.info(f"Fetching chunk starting from {datetime.fromtimestamp(current_since/1000)}")
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Break if we"ve reached the end date
                if until and ohlcv[-1][0] >= until:
                    break
                    
                # Move to next chunk
                current_since = ohlcv[-1][0] + 1
                
                # Add a small delay to avoid rate limits
                import time
                time.sleep(exchange.rateLimit / 1000)  # rateLimit is in milliseconds
        else:
            # If no start date, just fetch the latest data
            all_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Filter by date range if provided
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]
        
        logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def simulate_liquidation_data(df, num_clusters=5, intensity_range=(0.4, 0.9)):
    """
    Simulate liquidation data based on price action and volume
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLCV data and indicators
    num_clusters : int
        Number of liquidation clusters to generate
    intensity_range : tuple
        Range of intensity values (min, max)
        
    Returns:
    --------
    dict
        Simulated liquidation data
    """
    # Find support and resistance levels
    price_series = df["close"].values
    volume_series = df["volume"].values
    
    # Calculate price percentiles
    price_min = np.nanmin(price_series)
    price_max = np.nanmax(price_series)
    price_range = price_max - price_min
    
    # Find high volume nodes
    volume_threshold = np.nanpercentile(volume_series, 80)  # Top 20% volume
    high_volume_indices = np.where(volume_series > volume_threshold)[0]
    
    # Get prices at high volume
    high_volume_prices = price_series[high_volume_indices]
    
    # Create clusters around high volume prices
    clusters = []
    
    if len(high_volume_prices) > 0:
        # Use KMeans to find clusters
        from sklearn.cluster import KMeans
        
        # Reshape for KMeans
        X = high_volume_prices.reshape(-1, 1)
        
        # Determine number of clusters (min of requested or available data points)
        n_clusters = min(num_clusters, len(X))
        
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_.flatten()
            
            # Create clusters with intensity based on volume
            for center in centers:
                # Find points in this cluster
                cluster_points = high_volume_indices[np.abs(high_volume_prices - center) < price_range * 0.01]
                
                if len(cluster_points) > 0:
                    # Calculate intensity based on volume
                    avg_volume = np.mean(volume_series[cluster_points])
                    max_volume = np.nanmax(volume_series)
                    
                    # Scale intensity between min and max of intensity_range
                    intensity = intensity_range[0] + (intensity_range[1] - intensity_range[0]) * (avg_volume / max_volume)
                    
                    clusters.append((float(center), float(intensity)))
        else:
            # If only one cluster, use the mean price
            center = np.mean(high_volume_prices)
            intensity = (intensity_range[0] + intensity_range[1]) / 2
            clusters.append((float(center), float(intensity)))
    
    # If no clusters were created, create some based on price levels
    if not clusters:
        price_levels = np.linspace(price_min, price_max, num_clusters)
        intensities = np.linspace(intensity_range[0], intensity_range[1], num_clusters)
        
        clusters = [(float(price), float(intensity)) for price, intensity in zip(price_levels, intensities)]
    
    # Determine if price has cleared major liquidation zones
    current_price = price_series[-1]
    cleared_zone = current_price > np.mean([c[0] for c in clusters])
    
    return {
        "clusters": clusters,
        "cleared_zone": cleared_zone
    }

def process_historical_data(symbol, start_date, end_date, output_dir="data/historical"):
    """
    Process historical data for a symbol
    
    Parameters:
    -----------
    symbol : str
        Symbol to process (e.g., "BTC/USDT")
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str
        End date in "YYYY-MM-DD" format
    output_dir : str
        Directory to save processed data
        
    Returns:
    --------
    str
        Path to saved data file
    """
    # Fetch historical data
    df = fetch_historical_data(symbol, start_date=start_date, end_date=end_date)
    
    if df is None or len(df) == 0:
        logger.error(f"No data fetched for {symbol}")
        return None
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Create output directory if it doesn"t exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    asset_name = symbol.split("/")[0]
    output_file = os.path.join(output_dir, f"{asset_name}_historical_data.csv")
    df.to_csv(output_file)
    
    logger.info(f"Saved processed data to {output_file}")
    
    return output_file

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Collect historical data for backtesting")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTC/USDT", "SOL/USDT"], help="Symbols to collect data for")
    parser.add_argument("--output-dir", type=str, default="data/historical", help="Output directory")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    logger.info(f"Starting historical data collection from {args.start_date} to {args.end_date}")
    
    # Process each symbol
    for symbol in args.symbols:
        try:
            logger.info(f"Processing {symbol}")
            output_file = process_historical_data(symbol, args.start_date, args.end_date, args.output_dir)
            
            if output_file:
                logger.info(f"Successfully processed {symbol}")
            else:
                logger.warning(f"Failed to process {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Historical data collection completed")

if __name__ == "__main__":
    main()
