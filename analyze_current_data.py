from rsi_volume_model import RsiVolumePredictor
import numpy as np

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
        print(f"Target 1: {price_data[-1] * 1.04:.2f}")
        print(f"Target 2: {price_data[-1] * 1.08:.2f}")
        print(f"Stop Loss: {price_data[-1] * 0.97:.2f}")
    elif final_score > 0.3:
        print("\nRecommendation: Moderate bullish trend likely.")
        print(f"Target 1: {price_data[-1] * 1.02:.2f}")
        print(f"Target 2: {price_data[-1] * 1.05:.2f}")
        print(f"Stop Loss: {price_data[-1] * 0.98:.2f}")
    elif final_score >= -0.3:
        print("\nRecommendation: Sideways/neutral movement expected.")
        print("Consider waiting for a clearer signal.")
    elif final_score >= -0.6:
        print("\nRecommendation: Moderate bearish trend likely.")
        print(f"Target 1: {price_data[-1] * 0.98:.2f}")
        print(f"Target 2: {price_data[-1] * 0.95:.2f}")
        print(f"Stop Loss: {price_data[-1] * 1.02:.2f}")
    else:
        print("\nRecommendation: Strong bearish reversal warning.")
        print(f"Target 1: {price_data[-1] * 0.96:.2f}")
        print(f"Target 2: {price_data[-1] * 0.92:.2f}")
        print(f"Stop Loss: {price_data[-1] * 1.03:.2f}")
    
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

# Latest data from charts
# For a proper analysis, we need historical data, not just the current values
# We'll simulate some historical data based on the current values

# SOL data
sol_current_price = 196.62
sol_current_rsi_raw = 65.15
sol_current_rsi_sma = 51.37
sol_current_volume = 230.67e3  # 230.67K

# Generate synthetic historical data (last 50 candles)
# This is a simplification - in reality, you would use actual historical data
sol_price = np.linspace(170, sol_current_price, 50) + np.random.normal(0, 2, 50)
sol_price[-1] = sol_current_price

sol_rsi_raw = np.linspace(45, sol_current_rsi_raw, 50) + np.random.normal(0, 3, 50)
sol_rsi_raw[-1] = sol_current_rsi_raw
sol_rsi_raw = np.clip(sol_rsi_raw, 0, 100)

sol_rsi_sma = np.linspace(40, sol_current_rsi_sma, 50) + np.random.normal(0, 2, 50)
sol_rsi_sma[-1] = sol_current_rsi_sma
sol_rsi_sma = np.clip(sol_rsi_sma, 0, 100)

# Create a volume pattern with a recent spike
sol_volume = np.ones(50) * 150e3 + np.random.normal(0, 20e3, 50)
sol_volume[-3:] = [180e3, 200e3, sol_current_volume]

# BTC data
btc_current_price = 116756.98
btc_current_rsi_raw = 62.44
btc_current_rsi_sma = 40.25
btc_current_volume = 1.03e3  # 1.03K

btc_price = np.linspace(110000, btc_current_price, 50) + np.random.normal(0, 500, 50)
btc_price[-1] = btc_current_price

btc_rsi_raw = np.linspace(50, btc_current_rsi_raw, 50) + np.random.normal(0, 3, 50)
btc_rsi_raw[-1] = btc_current_rsi_raw
btc_rsi_raw = np.clip(btc_rsi_raw, 0, 100)

btc_rsi_sma = np.linspace(35, btc_current_rsi_sma, 50) + np.random.normal(0, 2, 50)
btc_rsi_sma[-1] = btc_current_rsi_sma
btc_rsi_sma = np.clip(btc_rsi_sma, 0, 100)

btc_volume = np.ones(50) * 0.8e3 + np.random.normal(0, 0.1e3, 50)
btc_volume[-3:] = [0.9e3, 1.0e3, btc_current_volume]

# BONK data
bonk_current_price = 0.00002341
bonk_current_rsi_raw = 61.61
bonk_current_rsi_sma = 41.65
bonk_current_volume = 149.42e9  # 149.42B

bonk_price = np.linspace(0.00002, bonk_current_price, 50) + np.random.normal(0, 0.0000005, 50)
bonk_price[-1] = bonk_current_price

bonk_rsi_raw = np.linspace(40, bonk_current_rsi_raw, 50) + np.random.normal(0, 3, 50)
bonk_rsi_raw[-1] = bonk_current_rsi_raw
bonk_rsi_raw = np.clip(bonk_rsi_raw, 0, 100)

bonk_rsi_sma = np.linspace(30, bonk_current_rsi_sma, 50) + np.random.normal(0, 2, 50)
bonk_rsi_sma[-1] = bonk_current_rsi_sma
bonk_rsi_sma = np.clip(bonk_rsi_sma, 0, 100)

bonk_volume = np.ones(50) * 100e9 + np.random.normal(0, 10e9, 50)
bonk_volume[-3:] = [120e9, 135e9, bonk_current_volume]

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
