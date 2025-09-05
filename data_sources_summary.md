# Data Sources for RSI + Volume Predictive Model

## Overview

To effectively implement the Enhanced RSI + Volume Predictive Scoring Model, we need to gather several types of data from various sources. This document provides a summary of the available data sources and recommended approaches.

## Required Data Types

1. **Price & Volume Data**: OHLCV (Open, High, Low, Close, Volume) data for BTC, SOL, and BONK
2. **Technical Indicators**: RSI (raw and SMA), moving averages, ATR
3. **Liquidation Data**: Support and resistance levels from liquidation heatmaps
4. **WebTrend Status**: Trend direction indicator

## Recommended Data Sources

### 1. Price & Volume + Technical Indicators

**Best Option**: Use CCXT library with local calculation
- **CCXT Library**: Provides unified API for 100+ exchanges
- **Local Calculation**: Use pandas_ta or TA-Lib to calculate indicators
- **Advantages**: Free, reliable, customizable
- **Implementation**: See `data_collection_pipeline.py`

**Alternative APIs**:
- **TAAPI.IO**: 200+ indicators including RSI (paid service)
- **CoinAPI**: Comprehensive market data (free tier available)
- **Polygon.io**: Technical indicators API (paid service)

### 2. Liquidation Data

**Best Option**: Coinglass with web scraping
- **Coinglass**: Provides detailed liquidation heatmaps
- **Web Scraping**: Use Selenium to extract data
- **Advantages**: Most comprehensive liquidation data available
- **Implementation**: See `coinglass_scraper_example.py`

**Alternative**:
- **Manual Input**: Observe Coinglass heatmaps and input key levels
- **ByBit API**: Limited liquidation data (not as comprehensive)

### 3. WebTrend Indicator

**Best Option**: Local implementation
- **Custom Implementation**: Use multiple moving averages
- **Advantages**: Free, customizable, no external dependencies
- **Implementation**: Included in `data_collection_pipeline.py`

**Alternative**:
- **TradingView**: Manual observation of WebTrend 3.0 indicator
- **TradingView Pine Script**: For premium users

## Implementation Approach

We've created a complete data collection pipeline that:

1. Fetches OHLCV data using CCXT
2. Calculates technical indicators locally
3. Simulates liquidation data (to be replaced with actual Coinglass data)
4. Generates WebTrend status based on moving averages
5. Saves and visualizes the collected data

### Files Provided:

1. `data_collection_pipeline.py`: Main data collection script
2. `coinglass_scraper_example.py`: Example scraper for liquidation data
3. `requirements.txt`: Required Python packages

### Usage:

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Run data collection:
   ```
   python data_collection_pipeline.py
   ```

3. Use collected data with the model:
   ```
   python apply_model_to_latest_data.py
   ```

## Next Steps

1. **Integrate Coinglass Scraper**: Replace simulated liquidation data with actual data
2. **Automate Data Collection**: Set up scheduled runs (e.g., every 4 hours)
3. **Create Dashboard**: Develop a simple UI to visualize model inputs and outputs
4. **Implement Alerts**: Set up notifications for significant changes in model predictions
