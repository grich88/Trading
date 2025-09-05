# Data Sources for RSI + Volume Predictive Scoring Model

This document outlines the key data sources and tools needed to gather all required inputs for our Enhanced RSI + Volume Predictive Scoring Model.

## Required Data Types

Our model requires the following data types:
1. Price data (OHLCV - Open, High, Low, Close, Volume)
2. RSI values (raw and SMA)
3. Liquidation heatmap data
4. WebTrend indicator status

## Primary Data Sources

### 1. Price & Volume Data + RSI Calculation

#### API Options:

1. **TAAPI.IO**
   - Provides 200+ technical indicators including RSI
   - Supports real-time and historical data
   - Offers customizable RSI periods
   - Website: [https://taapi.io](https://taapi.io)
   - Documentation: [https://taapi.io/documentation/](https://taapi.io/documentation/)
   - Pricing: Starts at $29/month for 1 request/second

2. **CoinAPI**
   - Comprehensive market data for 380+ exchanges
   - Supports REST, WebSocket, and FIX protocols
   - Website: [https://www.coinapi.io](https://www.coinapi.io)
   - Documentation: [https://docs.coinapi.io](https://docs.coinapi.io)
   - Pricing: Free tier available with 100 daily requests

3. **Polygon.io**
   - Provides RSI and other technical indicators
   - Supports crypto, stocks, and forex
   - Website: [https://polygon.io](https://polygon.io)
   - Documentation: [https://polygon.io/docs](https://polygon.io/docs)
   - Pricing: Starts at $29/month

### 2. Liquidation Data

#### Options:

1. **Coinglass**
   - Provides liquidation heatmaps and data
   - No official API, but data can be accessed through:
     - Web scraping with Python libraries like BeautifulSoup or Selenium
     - Using the Coinglass website directly for manual input
   - Website: [https://www.coinglass.com](https://www.coinglass.com)

2. **ByBit API**
   - Provides some liquidation data (though not as comprehensive as Coinglass)
   - Website: [https://bybit-exchange.github.io/docs/](https://bybit-exchange.github.io/docs/)
   - Documentation: [https://bybit-exchange.github.io/docs/v5/market/liquidation](https://bybit-exchange.github.io/docs/v5/market/liquidation)

### 3. WebTrend Indicator

#### Options:

1. **TradingView**
   - WebTrend 3.0 is a custom indicator available on TradingView
   - Can be accessed through:
     - Manual observation from TradingView charts
     - TradingView Pine Script API (for premium users)
   - Website: [https://www.tradingview.com](https://www.tradingview.com)

2. **Custom Implementation**
   - WebTrend can be implemented using standard technical indicators
   - Typically uses multiple moving averages and trend detection algorithms
   - Can be built using Python libraries like TA-Lib or pandas_ta

## Python Libraries for Technical Analysis

For calculating indicators locally:

1. **TA-Lib**
   - Comprehensive technical analysis library
   - Provides RSI, moving averages, and 200+ indicators
   - Installation: `pip install ta-lib`
   - GitHub: [https://github.com/mrjbq7/ta-lib](https://github.com/mrjbq7/ta-lib)

2. **pandas_ta**
   - Python library built on pandas for technical analysis
   - Easier to install than TA-Lib
   - Provides 130+ indicators including RSI
   - Installation: `pip install pandas_ta`
   - GitHub: [https://github.com/twopirllc/pandas-ta](https://github.com/twopirllc/pandas-ta)

3. **ccxt**
   - Library for cryptocurrency exchange trading
   - Provides unified API for 100+ exchanges
   - Can fetch OHLCV data needed for indicator calculation
   - Installation: `pip install ccxt`
   - GitHub: [https://github.com/ccxt/ccxt](https://github.com/ccxt/ccxt)

## Implementation Approach

### Option 1: API-First Approach
1. Use TAAPI.IO or similar for RSI and other indicators
2. Use ccxt to fetch price and volume data
3. Manually input liquidation data from Coinglass
4. Manually input WebTrend status from TradingView

### Option 2: Calculation-First Approach
1. Use ccxt to fetch OHLCV data from exchanges
2. Calculate RSI and other indicators using pandas_ta or TA-Lib
3. Manually input liquidation data from Coinglass
4. Manually input WebTrend status from TradingView

### Option 3: Hybrid Approach (Recommended)
1. Use ccxt to fetch OHLCV data
2. Calculate basic indicators locally using pandas_ta
3. Use specialized APIs for complex indicators when needed
4. Create a simple web scraper for Coinglass liquidation data
5. Implement a simplified version of WebTrend using moving averages

## Data Collection Script Example

```python
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize exchange
exchange = ccxt.binance()

# Fetch OHLCV data
def fetch_ohlcv(symbol, timeframe='4h', limit=100):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Calculate indicators
def calculate_indicators(df):
    # Calculate RSI
    df['rsi_raw'] = ta.rsi(df['close'], length=14)
    
    # Calculate RSI SMA
    df['rsi_sma'] = ta.sma(df['rsi_raw'], length=3)
    
    # Calculate simple WebTrend (simplified version)
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema100'] = ta.ema(df['close'], length=100)
    
    # Simple WebTrend status (true if price > all EMAs)
    df['webtrend_status'] = (df['close'] > df['ema20']) & (df['close'] > df['ema50']) & (df['close'] > df['ema100'])
    
    return df

# Main function
def get_data_for_model(symbols=['BTC/USDT', 'SOL/USDT', 'BONK/USDT']):
    assets_data = {}
    
    for symbol in symbols:
        # Fetch data
        df = fetch_ohlcv(symbol, timeframe='4h', limit=100)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Extract latest values
        asset_name = symbol.split('/')[0]
        assets_data[asset_name] = {
            'price': df['close'].values.tolist(),
            'rsi_raw': df['rsi_raw'].values.tolist(),
            'rsi_sma': df['rsi_sma'].values.tolist(),
            'volume': df['volume'].values.tolist(),
            'webtrend_status': bool(df['webtrend_status'].iloc[-1])
        }
        
        print(f"Fetched data for {asset_name}")
        print(f"Current price: {df['close'].iloc[-1]}")
        print(f"Current RSI: {df['rsi_raw'].iloc[-1]:.2f}")
        print(f"Current RSI SMA: {df['rsi_sma'].iloc[-1]:.2f}")
        print(f"WebTrend status: {'Uptrend' if df['webtrend_status'].iloc[-1] else 'Downtrend'}")
        print("-" * 50)
    
    return assets_data

# Note: Liquidation data needs to be added manually from Coinglass
```

## Next Steps

1. **Set up data collection pipeline**:
   - Implement the data collection script
   - Add error handling and logging
   - Set up scheduled runs (e.g., every 4 hours)

2. **Create a simple UI for manual inputs**:
   - Form for entering liquidation data from Coinglass
   - Toggle for WebTrend status if using manual observation

3. **Integrate with the Enhanced RSI + Volume Predictive Model**:
   - Pass collected data to the model
   - Generate and store predictions
   - Track performance over time

4. **Consider advanced options**:
   - Develop a Coinglass scraper for automated liquidation data
   - Implement a more sophisticated WebTrend indicator
   - Create alerts for significant changes in model predictions
