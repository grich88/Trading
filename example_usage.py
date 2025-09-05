import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data_fetcher import CryptoDataFetcher
from rsi_volume_model import RsiVolumePredictor
from backtester import Backtester
from visualizer import SignalVisualizer

def run_live_example():
    """
    Run a live example using recent data
    """
    print("=== RSI + Volume Predictive Scoring Model - Live Example ===")
    
    # Fetch data
    print("\nFetching recent data for BTC/USDT...")
    fetcher = CryptoDataFetcher()
    btc_data = fetcher.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=100)
    
    if btc_data is None:
        print("Error fetching data. Please check your internet connection.")
        return
    
    # Prepare model inputs
    print("\nPreparing model inputs...")
    rsi_sma, rsi_raw, volume, price = fetcher.prepare_model_inputs(btc_data)
    
    # Create predictor and compute score
    predictor = RsiVolumePredictor(rsi_sma, rsi_raw, volume, price)
    score = predictor.compute_score()
    
    # Display results
    print(f"\n📊 Current Momentum Score: {score}")
    print("\nScore Interpretation:")
    print("> +0.6\t\tStrong bullish continuation")
    print("+0.3 to +0.6\tModerate bullish trend")
    print("-0.3 to +0.3\tSideways / neutral")
    print("-0.3 to -0.6\tModerate bearish trend")
    print("< -0.6\t\tStrong bearish reversal warning")
    
    # Get current signal
    signal = "STRONG BUY" if score > 0.6 else \
             "BUY" if score > 0.3 else \
             "NEUTRAL" if score >= -0.3 else \
             "SELL" if score >= -0.6 else \
             "STRONG SELL"
    
    print(f"\nCurrent Signal: {signal}")
    
    # Display component scores
    rsi_score = predictor.get_rsi_trend_score()
    volume_score = predictor.get_volume_trend_score()
    divergence = predictor.detect_divergence()
    
    print("\nComponent Scores:")
    print(f"RSI Trend Score: {rsi_score}")
    print(f"Volume Trend Score: {volume_score}")
    print(f"Divergence Score: {divergence}")
    
    return btc_data

def run_backtest_example():
    """
    Run a backtest example using historical data
    """
    print("\n=== RSI + Volume Predictive Scoring Model - Backtest Example ===")
    
    # Fetch data
    print("\nFetching historical data for BTC/USDT...")
    fetcher = CryptoDataFetcher()
    btc_data = fetcher.fetch_ohlcv('BTC/USDT', timeframe='4h', limit=500)
    
    if btc_data is None:
        print("Error fetching data. Please check your internet connection.")
        return
    
    # Run backtest
    print("\nRunning backtest...")
    backtester = Backtester(btc_data)
    results = backtester.run_backtest()
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics = backtester.calculate_performance()
    
    # Display metrics
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = SignalVisualizer(results)
    
    # Plot signals for the last 30 days
    end_date = results.index[-1]
    start_date = end_date - timedelta(days=30)
    fig1 = visualizer.plot_signals(start_date=start_date.strftime('%Y-%m-%d'))
    
    # Plot performance
    fig2 = visualizer.plot_performance()
    
    plt.show()
    
    return results

def compare_assets():
    """
    Compare model performance across different assets
    """
    print("\n=== RSI + Volume Predictive Scoring Model - Asset Comparison ===")
    
    assets = ['BTC/USDT', 'SOL/USDT', 'ETH/USDT']
    results = {}
    
    fetcher = CryptoDataFetcher()
    
    for asset in assets:
        print(f"\nFetching data for {asset}...")
        data = fetcher.fetch_ohlcv(asset, timeframe='4h', limit=100)
        
        if data is None:
            print(f"Error fetching data for {asset}. Skipping...")
            continue
        
        # Prepare model inputs
        rsi_sma, rsi_raw, volume, price = fetcher.prepare_model_inputs(data)
        
        # Create predictor and compute score
        predictor = RsiVolumePredictor(rsi_sma, rsi_raw, volume, price)
        score = predictor.compute_score()
        
        results[asset] = {
            'score': score,
            'rsi_score': predictor.get_rsi_trend_score(),
            'volume_score': predictor.get_volume_trend_score(),
            'divergence': predictor.detect_divergence()
        }
    
    # Display comparison
    print("\nAsset Comparison:")
    print("-" * 60)
    print(f"{'Asset':<10} {'Score':<10} {'Signal':<12} {'RSI':<8} {'Volume':<8} {'Div':<8}")
    print("-" * 60)
    
    for asset, data in results.items():
        signal = "STRONG BUY" if data['score'] > 0.6 else \
                 "BUY" if data['score'] > 0.3 else \
                 "NEUTRAL" if data['score'] >= -0.3 else \
                 "SELL" if data['score'] >= -0.6 else \
                 "STRONG SELL"
        
        print(f"{asset.split('/')[0]:<10} {data['score']:<10.3f} {signal:<12} {data['rsi_score']:<8.2f} {data['volume_score']:<8.2f} {data['divergence']:<8.2f}")
    
    return results

if __name__ == "__main__":
    print("RSI + Volume Predictive Scoring Model for 4H Charts")
    print("=" * 50)
    print("\nSelect an option:")
    print("1. Run live example with current data")
    print("2. Run backtest with historical data")
    print("3. Compare multiple assets")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        data = run_live_example()
    elif choice == '2':
        data = run_backtest_example()
    elif choice == '3':
        data = compare_assets()
    else:
        print("Invalid choice. Please run the script again and select a valid option.")
