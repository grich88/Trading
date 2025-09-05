from rsi_volume_model import RsiVolumePredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Latest data from charts
# SOL data
sol_current_price = 196.62
sol_current_rsi_raw = 65.15
sol_current_rsi_sma = 51.37
sol_current_volume = 230.67e3  # 230.67K

# BTC data
btc_current_price = 116756.98
btc_current_rsi_raw = 62.44
btc_current_rsi_sma = 40.25
btc_current_volume = 1.03e3  # 1.03K

# BONK data
bonk_current_price = 0.00002341
bonk_current_rsi_raw = 61.61
bonk_current_rsi_sma = 41.65
bonk_current_volume = 149.42e9  # 149.42B

# Generate synthetic historical data for model input
# For SOL - uptrend with recent acceleration
sol_price = []
sol_rsi_raw = []
sol_rsi_sma = []
sol_volume = []

# Last 50 candles for SOL
for i in range(50):
    # Price trend - upward with recent acceleration
    if i < 30:
        sol_price.append(170 + i * 0.5 + np.random.normal(0, 1))
    else:
        sol_price.append(185 + (i-30) * 0.8 + np.random.normal(0, 1))
    
    # RSI trend - rising from neutral to bullish
    if i < 25:
        sol_rsi_raw.append(45 + i * 0.4 + np.random.normal(0, 2))
    else:
        sol_rsi_raw.append(55 + (i-25) * 0.5 + np.random.normal(0, 2))
    
    # RSI SMA trend - lagging but rising
    if i < 30:
        sol_rsi_sma.append(40 + i * 0.2 + np.random.normal(0, 1))
    else:
        sol_rsi_sma.append(46 + (i-30) * 0.3 + np.random.normal(0, 1))
    
    # Volume trend - recent spike
    if i < 45:
        sol_volume.append(150e3 + np.random.normal(0, 10e3))
    else:
        sol_volume.append((180e3 + (i-45) * 10e3) + np.random.normal(0, 10e3))

# Ensure last values match current data
sol_price[-1] = sol_current_price
sol_rsi_raw[-1] = sol_current_rsi_raw
sol_rsi_sma[-1] = sol_current_rsi_sma
sol_volume[-1] = sol_current_volume

# For BTC - steady rise
btc_price = []
btc_rsi_raw = []
btc_rsi_sma = []
btc_volume = []

# Last 50 candles for BTC
for i in range(50):
    # Price trend - steady rise
    btc_price.append(110000 + i * 150 + np.random.normal(0, 200))
    
    # RSI trend - oscillating in bullish territory
    if i < 25:
        btc_rsi_raw.append(55 + i * 0.2 + np.random.normal(0, 3))
    else:
        btc_rsi_raw.append(60 + np.sin(i/5) * 5 + np.random.normal(0, 2))
    
    # RSI SMA trend - rising slowly
    btc_rsi_sma.append(35 + i * 0.15 + np.random.normal(0, 1))
    
    # Volume trend - consistent with recent uptick
    if i < 45:
        btc_volume.append(0.8e3 + np.random.normal(0, 0.05e3))
    else:
        btc_volume.append((0.9e3 + (i-45) * 0.03e3) + np.random.normal(0, 0.05e3))

# Ensure last values match current data
btc_price[-1] = btc_current_price
btc_rsi_raw[-1] = btc_current_rsi_raw
btc_rsi_sma[-1] = btc_current_rsi_sma
btc_volume[-1] = btc_current_volume

# For BONK - volatile with recent uptrend
bonk_price = []
bonk_rsi_raw = []
bonk_rsi_sma = []
bonk_volume = []

# Last 50 candles for BONK
for i in range(50):
    # Price trend - volatile with recent uptrend
    if i < 30:
        bonk_price.append(0.00002 + np.sin(i/5) * 0.000002 + np.random.normal(0, 0.0000003))
    else:
        bonk_price.append(0.000021 + (i-30) * 0.0000001 + np.random.normal(0, 0.0000002))
    
    # RSI trend - volatile but trending up
    if i < 30:
        bonk_rsi_raw.append(40 + np.sin(i/4) * 10 + np.random.normal(0, 3))
    else:
        bonk_rsi_raw.append(50 + (i-30) * 0.5 + np.random.normal(0, 2))
    
    # RSI SMA trend - smoothed uptrend
    if i < 35:
        bonk_rsi_sma.append(35 + i * 0.1 + np.random.normal(0, 1))
    else:
        bonk_rsi_sma.append(38.5 + (i-35) * 0.2 + np.random.normal(0, 1))
    
    # Volume trend - significant recent spike
    if i < 40:
        bonk_volume.append(100e9 + np.random.normal(0, 5e9))
    else:
        bonk_volume.append((110e9 + (i-40) * 10e9) + np.random.normal(0, 5e9))

# Ensure last values match current data
bonk_price[-1] = bonk_current_price
bonk_rsi_raw[-1] = bonk_current_rsi_raw
bonk_rsi_sma[-1] = bonk_current_rsi_sma
bonk_volume[-1] = bonk_current_volume

def analyze_asset(name, price_data, rsi_raw, rsi_sma, volume_data):
    """
    Analyze an asset using the RSI + Volume Predictive Model
    """
    # Create predictor instance
    predictor = RsiVolumePredictor(rsi_sma, rsi_raw, volume_data, price_data)
    
    # Calculate score components
    rsi_score = predictor.get_rsi_trend_score()
    volume_score = predictor.get_volume_trend_score()
    divergence = predictor.detect_divergence()
    
    # Calculate final score
    final_score = predictor.compute_score()
    
    # Determine signal strength
    if final_score > 0.6:
        signal = "STRONG BUY"
    elif final_score > 0.3:
        signal = "BUY"
    elif final_score >= -0.3:
        signal = "NEUTRAL"
    elif final_score >= -0.6:
        signal = "SELL"
    else:
        signal = "STRONG SELL"
    
    # Print analysis
    print(f"\n{'=' * 50}")
    print(f"{name} ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Current Price: {price_data[-1]}")
    print(f"Current RSI: {rsi_raw[-1]:.2f}")
    print(f"Current RSI SMA: {rsi_sma[-1]:.2f}")
    
    print(f"\nModel Component Scores:")
    print(f"RSI Trend Score: {rsi_score:.3f}")
    print(f"Volume Trend Score: {volume_score:.3f}")
    print(f"Divergence Score: {divergence:.3f}")
    
    print(f"\nFinal Momentum Score: {final_score:.3f}")
    print(f"Signal: {signal}")
    
    # Provide recommendations
    if final_score > 0.6:
        print("\nRecommendation: Strong bullish continuation expected.")
        print(f"Target 1: {price_data[-1] * 1.04:.6f}")
        print(f"Target 2: {price_data[-1] * 1.08:.6f}")
        print(f"Stop Loss: {price_data[-1] * 0.97:.6f}")
    elif final_score > 0.3:
        print("\nRecommendation: Moderate bullish trend likely.")
        print(f"Target 1: {price_data[-1] * 1.02:.6f}")
        print(f"Target 2: {price_data[-1] * 1.05:.6f}")
        print(f"Stop Loss: {price_data[-1] * 0.98:.6f}")
    elif final_score >= -0.3:
        print("\nRecommendation: Sideways/neutral movement expected.")
        print("Consider waiting for a clearer signal.")
    elif final_score >= -0.6:
        print("\nRecommendation: Moderate bearish trend likely.")
        print(f"Target 1: {price_data[-1] * 0.98:.6f}")
        print(f"Target 2: {price_data[-1] * 0.95:.6f}")
        print(f"Stop Loss: {price_data[-1] * 1.02:.6f}")
    else:
        print("\nRecommendation: Strong bearish reversal warning.")
        print(f"Target 1: {price_data[-1] * 0.96:.6f}")
        print(f"Target 2: {price_data[-1] * 0.92:.6f}")
        print(f"Stop Loss: {price_data[-1] * 1.03:.6f}")
    
    return {
        "asset": name,
        "price": price_data[-1],
        "rsi": rsi_raw[-1],
        "rsi_sma": rsi_sma[-1],
        "score": final_score,
        "signal": signal,
        "rsi_score": rsi_score,
        "volume_score": volume_score,
        "divergence": divergence
    }

# Run analysis for each asset
sol_result = analyze_asset("SOLANA (SOL)", sol_price, sol_rsi_raw, sol_rsi_sma, sol_volume)
btc_result = analyze_asset("BITCOIN (BTC)", btc_price, btc_rsi_raw, btc_rsi_sma, btc_volume)
bonk_result = analyze_asset("BONK", bonk_price, bonk_rsi_raw, bonk_rsi_sma, bonk_volume)

# Compare assets
print("\n" + "=" * 50)
print("COMPARATIVE ANALYSIS")
print("=" * 50)
print(f"{'Asset':<10} {'Price':<15} {'RSI':<8} {'RSI SMA':<8} {'Score':<8} {'Signal':<12}")
print("-" * 65)
print(f"{sol_result['asset'].split()[0]:<10} {sol_result['price']:<15.2f} {sol_result['rsi']:<8.2f} {sol_result['rsi_sma']:<8.2f} {sol_result['score']:<8.3f} {sol_result['signal']:<12}")
print(f"{btc_result['asset'].split()[0]:<10} {btc_result['price']:<15.2f} {btc_result['rsi']:<8.2f} {btc_result['rsi_sma']:<8.2f} {btc_result['score']:<8.3f} {btc_result['signal']:<12}")
print(f"{bonk_result['asset'].split()[0]:<10} {bonk_result['price']:<15.6f} {bonk_result['rsi']:<8.2f} {bonk_result['rsi_sma']:<8.2f} {bonk_result['score']:<8.3f} {bonk_result['signal']:<12}")

# Provide overall market assessment
print("\n" + "=" * 50)
print("OVERALL MARKET ASSESSMENT")
print("=" * 50)

avg_score = (sol_result['score'] + btc_result['score'] + bonk_result['score']) / 3
if avg_score > 0.3:
    print("Market shows bullish momentum across assets.")
    print(f"Best opportunity: {max([sol_result, btc_result, bonk_result], key=lambda x: x['score'])['asset']}")
elif avg_score < -0.3:
    print("Market shows bearish pressure across assets.")
    print(f"Most vulnerable: {min([sol_result, btc_result, bonk_result], key=lambda x: x['score'])['asset']}")
else:
    print("Mixed market conditions with no clear direction.")
    print("Consider selective entries based on individual asset signals.")

# Provide trading strategy recommendations
print("\n" + "=" * 50)
print("TRADING STRATEGY RECOMMENDATIONS")
print("=" * 50)

# Sort assets by score
assets_by_score = sorted([sol_result, btc_result, bonk_result], key=lambda x: x['score'], reverse=True)

print("Recommended allocation based on model scores:")
for asset in assets_by_score:
    if asset['score'] > 0.3:
        print(f"{asset['asset']}: Strong position (60-80% of allocation)")
    elif asset['score'] > 0:
        print(f"{asset['asset']}: Moderate position (30-50% of allocation)")
    elif asset['score'] > -0.3:
        print(f"{asset['asset']}: Small position or wait (0-20% of allocation)")
    else:
        print(f"{asset['asset']}: Avoid or consider short position")

print("\nRotation Strategy:")
if assets_by_score[0]['score'] > 0.3:
    print(f"1. Start with {assets_by_score[0]['asset']} as primary position")
    if assets_by_score[1]['score'] > 0:
        print(f"2. Rotate profits to {assets_by_score[1]['asset']} after first target hit")
    if assets_by_score[2]['score'] > -0.3:
        print(f"3. Consider {assets_by_score[2]['asset']} only after confirmation of trend change")
else:
    print("Current market conditions do not favor a clear rotation strategy.")
    print("Focus on individual setups and strict risk management.")

# Create summary table for export
summary_data = {
    'Asset': [sol_result['asset'].split()[0], btc_result['asset'].split()[0], bonk_result['asset'].split()[0]],
    'Price': [sol_result['price'], btc_result['price'], bonk_result['price']],
    'RSI': [sol_result['rsi'], btc_result['rsi'], bonk_result['rsi']],
    'RSI SMA': [sol_result['rsi_sma'], btc_result['rsi_sma'], bonk_result['rsi_sma']],
    'RSI Score': [sol_result['rsi_score'], btc_result['rsi_score'], bonk_result['rsi_score']],
    'Volume Score': [sol_result['volume_score'], btc_result['volume_score'], bonk_result['volume_score']],
    'Divergence': [sol_result['divergence'], btc_result['divergence'], bonk_result['divergence']],
    'Final Score': [sol_result['score'], btc_result['score'], bonk_result['score']],
    'Signal': [sol_result['signal'], btc_result['signal'], bonk_result['signal']]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 50)
print("MODEL SUMMARY TABLE")
print("=" * 50)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if x < 0.01 else f"{x:.3f}"))
