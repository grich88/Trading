# Cursor Prompt for Enhanced RSI + Volume Predictive Scoring Model

## Model Overview

This prompt helps you apply the Enhanced RSI + Volume Predictive Scoring Model to real-time cryptocurrency data. The model integrates RSI streak analysis, volume trends, divergence detection, liquidation heatmap data, and WebTrend indicators to generate predictive scores for short-term momentum on 4H charts.

## Instructions

1. Copy the code from `updated_rsi_volume_model.py` into Cursor
2. Provide the following inputs for each asset (BTC, SOL, BONK) from TradingView and Coinglass:

### Required Inputs

For each asset (BTC, SOL, BONK), provide:

```python
# Asset data (replace with your values)
asset_price = [current_price]  # Current price from TradingView
asset_rsi_raw = [current_rsi]  # Current RSI value (raw)
asset_rsi_sma = [current_rsi_ma]  # Current RSI SMA value
asset_volume = [current_volume]  # Current volume

# Liquidation data from Coinglass heatmap
asset_liquidation_data = {
    'clusters': [
        (price_level_1, intensity_1),  # e.g., (183.63, 0.8) - Support with 0.8 intensity
        (price_level_2, intensity_2),  # e.g., (199.31, 0.6) - Resistance with 0.6 intensity
        # Add more clusters as needed
    ],
    'cleared_zone': True/False  # True if price is above major liquidation clusters
}

# WebTrend status from TradingView
asset_webtrend_status = True/False  # True if WebTrend shows uptrend, False otherwise
```

### Example Usage

```python
from updated_rsi_volume_model import EnhancedRsiVolumePredictor, analyze_market_data, get_market_assessment

# Latest data from TradingView and Coinglass
assets_data = {
    'SOL': {
        'price': [198.94],
        'rsi_raw': [67.01],
        'rsi_sma': [51.37],
        'volume': [233.91e3],
        'liquidation_data': {
            'clusters': [(183.63, 0.8), (199.31, 0.6), (204.00, 0.7), (210.00, 0.5)],
            'cleared_zone': True
        },
        'webtrend_status': True
    },
    'BTC': {
        'price': [116736.00],
        'rsi_raw': [62.38],
        'rsi_sma': [40.25],
        'volume': [1.05e3],
        'liquidation_data': {
            'clusters': [(113456, 0.7), (120336, 0.8), (122000, 0.6)],
            'cleared_zone': False
        },
        'webtrend_status': True
    },
    'BONK': {
        'price': [0.00002344],
        'rsi_raw': [61.83],
        'rsi_sma': [41.65],
        'volume': [152.82e9],
        'liquidation_data': {
            'clusters': [(0.00002167, 0.6), (0.00002360, 0.5), (0.00002424, 0.7), (0.00002751, 0.8)],
            'cleared_zone': False
        },
        'webtrend_status': True
    }
}

# Analyze market data
results = analyze_market_data(assets_data)

# Print results for each asset
for asset, analysis in results.items():
    print(f"\n{'=' * 50}")
    print(f"{asset} ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Current Price: {analysis['price']}")
    print(f"Current RSI: {analysis['rsi']:.2f}")
    print(f"Current RSI SMA: {analysis['rsi_sma']:.2f}")
    
    print(f"\nModel Component Scores:")
    for component, score in analysis['components'].items():
        print(f"{component.replace('_', ' ').title()}: {score:.3f}")
    
    print(f"\nFinal Momentum Score: {analysis['final_score']:.3f}")
    print(f"Signal: {analysis['signal']}")
    
    print(f"\nTarget Prices:")
    print(f"TP1: {analysis['targets']['TP1']}")
    print(f"TP2: {analysis['targets']['TP2']}")
    print(f"SL: {analysis['targets']['SL']}")

# Get overall market assessment
assessment = get_market_assessment(results)
print(f"\n{'=' * 50}")
print("MARKET ASSESSMENT")
print(f"{'=' * 50}")
print(f"Market Condition: {assessment['market_condition']}")
print(f"Average Score: {assessment['average_score']}")
print(f"\nBest Opportunity: {assessment['best_opportunity']['asset']} ({assessment['best_opportunity']['signal']})")
print(f"Weakest Asset: {assessment['weakest_asset']['asset']} ({assessment['weakest_asset']['signal']})")

print(f"\nRotation Strategy:")
for strategy in assessment['rotation_strategy']:
    print(f"- {strategy}")
```

## How to Read the Liquidation Heatmap

1. Open Coinglass.com and navigate to the Liquidation chart for your asset
2. Identify major clusters (yellow/orange areas) on the chart
3. For each cluster, note:
   - Price level (y-axis)
   - Intensity (0.0-1.0, where 1.0 is brightest yellow, 0.5 is medium, 0.0 is dark)
4. Determine if price has cleared major liquidation zones (is price above most yellow clusters?)

## How to Determine WebTrend Status

1. On TradingView, apply the WebTrend 3.0 indicator to your chart
2. Check if the indicator shows green (uptrend) or red (downtrend)
3. Set `webtrend_status` to `True` for green/uptrend, `False` for red/downtrend

## Interpreting Results

### Score Interpretation
- Above +0.6: Strong bullish continuation
- +0.3 to +0.6: Moderate bullish trend
- -0.3 to +0.3: Sideways/neutral
- -0.3 to -0.6: Moderate bearish trend
- Below -0.6: Strong bearish reversal warning

### Trading Signals
- STRONG BUY: Enter long position with high confidence
- BUY: Enter long position with moderate confidence
- NEUTRAL: Wait for clearer signals
- SELL: Enter short position with moderate confidence
- STRONG SELL: Enter short position with high confidence

### Target Prices
- TP1: First take profit target (conservative)
- TP2: Second take profit target (aggressive)
- SL: Recommended stop loss level

## Model Components

The model integrates five key components:

1. **RSI Trend Analysis**: Evaluates RSI patterns, including streaks above/below key levels and the relationship between raw RSI and its moving average.

2. **Volume Trend Analysis**: Assesses recent volume patterns, comparing recent volume to earlier periods and identifying volume spikes or fades in relation to price movement.

3. **Divergence Detection**: Identifies misalignments between price action and indicators, particularly when price makes new highs/lows while RSI or volume fails to confirm.

4. **Liquidation Analysis**: Incorporates liquidation heatmap data to identify potential support/resistance levels based on liquidation clusters.

5. **WebTrend Confirmation**: Uses WebTrend indicator to confirm trend direction.

The final score is calculated as a weighted average of these components, with adjustments based on asset volatility.

## Notes

- For best results, use 4H charts from major exchanges (Binance, Coinbase)
- Update liquidation data regularly as market conditions change
- Consider inter-asset correlations when planning trades (BTC often leads, followed by SOL, then BONK)
- Use the rotation strategy to maximize returns by moving between assets based on their signals
