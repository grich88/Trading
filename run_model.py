"""
Run Enhanced RSI + Volume Predictive Scoring Model with Collected Data

This script:
1. Loads model input data with liquidation information
2. Runs the Enhanced RSI + Volume Predictive Scoring Model
3. Displays the results
"""

import json
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_latest_model_input_with_liquidation():
    """
    Load the latest model input file with liquidation data
    
    Returns:
    --------
    tuple
        (filepath, data)
    """
    # Find the most recent model_input_with_liquidation_*.json file
    data_dir = 'data'
    if not os.path.exists(data_dir):
        logger.error("Data directory not found")
        return None, None
    
    files = [f for f in os.listdir(data_dir) if f.startswith('model_input_with_liquidation_') and f.endswith('.json')]
    if not files:
        logger.error("No model input files with liquidation data found")
        return None, None
    
    # Sort by timestamp (newest first)
    files.sort(reverse=True)
    latest_file = os.path.join(data_dir, files[0])
    
    # Load data
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        return latest_file, data
    except Exception as e:
        logger.error(f"Error loading data from {latest_file}: {e}")
        return None, None

def calculate_rsi_trend_score(rsi_sma):
    """
    Calculate score based on RSI trend patterns
    
    Parameters:
    -----------
    rsi_sma : list
        RSI SMA values
        
    Returns:
    --------
    float
        Score between -1.0 and 1.0
    """
    # Use the last 50 values for analysis
    rsi_sma_recent = rsi_sma[-50:]
    
    # Calculate metrics
    candles_above_neutral = sum(1 for x in rsi_sma_recent if x > 50)
    candles_above_overbought = sum(1 for x in rsi_sma_recent[-12:] if x > 70)
    candles_below_oversold = sum(1 for x in rsi_sma_recent[-10:] if x < 30)
    
    # RSI crossing above neutral is bullish
    if rsi_sma_recent[-1] > 50 and rsi_sma_recent[-2] <= 50:
        return 0.7
    
    # RSI staying above neutral is bullish
    if rsi_sma_recent[-1] > 50 and candles_above_neutral >= 6:
        return min(0.8, 0.1 * candles_above_neutral)
    
    # RSI in overbought territory for extended period suggests caution
    if rsi_sma_recent[-1] > 70 and candles_above_overbought >= 3:
        return max(0.0, 0.9 - 0.1 * (candles_above_overbought - 3))
    
    # RSI in oversold territory suggests potential reversal
    if rsi_sma_recent[-1] < 30 and candles_below_oversold >= 3:
        return min(-0.1 * candles_below_oversold, -0.3)
    
    # RSI crossing below neutral is bearish
    if rsi_sma_recent[-1] < 50 and rsi_sma_recent[-2] >= 50:
        return -0.5
    
    return 0

def calculate_volume_trend_score(volume, price):
    """
    Calculate score based on volume trend patterns
    
    Parameters:
    -----------
    volume : list
        Volume values
    price : list
        Price values
        
    Returns:
    --------
    float
        Score between -1.0 and 1.0
    """
    # Use the last 10 values for analysis
    last_vol = volume[-10:]
    avg_early = sum(last_vol[:5]) / 5
    avg_late = sum(last_vol[5:]) / 5
    momentum = avg_late / avg_early if avg_early != 0 else 1
    
    # Calculate green/red streaks
    green_streak = 0
    red_streak = 0
    
    for i in range(-5, 0):
        if price[i] > price[i - 1]:
            green_streak += 1
        elif price[i] < price[i - 1]:
            red_streak += 1
    
    # Volume spike with green candles is bullish
    if green_streak >= 3 and momentum > 1.05:
        return min(1.0, 0.25 * green_streak + 0.5 * (momentum - 1))
    
    # Volume spike with red candles is bearish
    elif red_streak >= 3 and momentum > 1.05:
        return max(-1.0, -0.25 * red_streak - 0.5 * (momentum - 1))
    
    return 0

def detect_divergence(price, rsi_raw, volume):
    """
    Detect divergence between price and indicators
    
    Parameters:
    -----------
    price : list
        Price values
    rsi_raw : list
        RSI values
    volume : list
        Volume values
        
    Returns:
    --------
    float
        Score between -1.0 and 1.0
    """
    # Use the last 10 values for analysis
    price_highs = price[-10:]
    rsi_highs = rsi_raw[-10:]
    volume_highs = volume[-10:]
    
    # Bearish divergence: price makes new high but RSI doesn't
    price_div = price_highs[-1] > max(price_highs[:-1])
    rsi_div = rsi_highs[-1] < max(rsi_highs[:-1])
    volume_div = volume_highs[-1] < max(volume_highs[:-1])
    
    if price_div and (rsi_div or volume_div):
        return -0.5
    
    # Bullish divergence: price makes new low but RSI doesn't
    price_low = price_highs[-1] < min(price_highs[:-1])
    rsi_up = rsi_highs[-1] > min(rsi_highs[:-1])
    volume_fade = volume_highs[-1] < max(volume_highs[:-1])
    
    if price_low and (rsi_up or volume_fade):
        return 0.5
    
    return 0

def get_liquidation_score(liquidation_data, current_price):
    """
    Calculate score based on liquidation heatmap data
    
    Parameters:
    -----------
    liquidation_data : dict
        Liquidation data
    current_price : float
        Current price
        
    Returns:
    --------
    float
        Score between -1.0 and 1.0
    """
    if not liquidation_data or 'clusters' not in liquidation_data:
        return 0
    
    clusters = liquidation_data['clusters']
    
    # Find nearest clusters above and below current price
    clusters_above = [(p, i) for p, i in clusters if p > current_price]
    clusters_below = [(p, i) for p, i in clusters if p < current_price]
    
    if not clusters_above and not clusters_below:
        return 0
    
    # Calculate distance to nearest clusters
    nearest_above = min(clusters_above, key=lambda x: x[0] - current_price) if clusters_above else (float('inf'), 0)
    nearest_below = max(clusters_below, key=lambda x: current_price - x[0]) if clusters_below else (0, 0)
    
    distance_above = nearest_above[0] - current_price if nearest_above[0] != float('inf') else float('inf')
    distance_below = current_price - nearest_below[0] if nearest_below[0] != 0 else float('inf')
    
    # Calculate intensity of nearest clusters
    intensity_above = nearest_above[1] if nearest_above[0] != float('inf') else 0
    intensity_below = nearest_below[1] if nearest_below[0] != 0 else 0
    
    # Calculate score based on distance and intensity
    if distance_above < distance_below and intensity_above > 0.5:
        # Strong resistance above - bearish
        return max(-0.8, -0.4 - 0.4 * intensity_above)
    elif distance_below < distance_above and intensity_below > 0.5:
        # Strong support below - bullish
        return min(0.8, 0.4 + 0.4 * intensity_below)
    elif liquidation_data.get('cleared_zone', False):
        # Cleared liquidation zone - bullish
        return 0.3
    
    return 0

def get_webtrend_score(webtrend_status):
    """
    Calculate score based on WebTrend indicator
    
    Parameters:
    -----------
    webtrend_status : bool
        WebTrend status
        
    Returns:
    --------
    float
        Score between -1.0 and 1.0
    """
    return 0.3 if webtrend_status else -0.3

def compute_score(asset_data, asset_type='BTC'):
    """
    Compute final predictive score
    
    Parameters:
    -----------
    asset_data : dict
        Asset data
    asset_type : str
        Asset type ('BTC', 'SOL', or 'BONK')
        
    Returns:
    --------
    dict
        Score components and final score
    """
    # Set volatility coefficient based on asset type
    if asset_type == 'BONK':
        volatility_coef = 1.2  # Higher volatility for BONK
    elif asset_type == 'SOL':
        volatility_coef = 1.0  # Base volatility for SOL
    else:  # BTC
        volatility_coef = 0.8  # Lower volatility for BTC
    
    # Base weights
    w_rsi = 0.35
    w_volume = 0.30
    w_divergence = 0.15
    w_liquidation = 0.15
    w_webtrend = 0.05
    
    # Adjust weights based on asset type
    if asset_type == 'BONK':
        # BONK is more volatile and sensitive to volume
        w_volume += 0.05
        w_liquidation -= 0.05
    elif asset_type == 'BTC':
        # BTC is more stable and less sensitive to short-term fluctuations
        w_rsi += 0.05
        w_volume -= 0.05
    
    # Calculate component scores
    rsi_score = calculate_rsi_trend_score(asset_data['rsi_sma'])
    volume_score = calculate_volume_trend_score(asset_data['volume'], asset_data['price'])
    divergence = detect_divergence(asset_data['price'], asset_data['rsi_raw'], asset_data['volume'])
    liquidation_score = get_liquidation_score(asset_data['liquidation_data'], asset_data['price'][-1])
    webtrend_score = get_webtrend_score(asset_data['webtrend_status'])
    
    # Calculate final score
    final_score = (
        w_rsi * rsi_score + 
        w_volume * volume_score + 
        w_divergence * divergence + 
        w_liquidation * liquidation_score + 
        w_webtrend * webtrend_score
    )
    
    # Apply volatility coefficient
    final_score *= volatility_coef
    
    # Ensure score is within bounds
    final_score = max(-1.0, min(1.0, final_score))
    
    # Get signal
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
    
    # Calculate target prices
    current_price = asset_data['price'][-1]
    
    # Calculate ATR (Average True Range) for volatility-based targets
    high_prices = [max(asset_data['price'][i], asset_data['price'][i-1]) for i in range(1, len(asset_data['price']))]
    low_prices = [min(asset_data['price'][i], asset_data['price'][i-1]) for i in range(1, len(asset_data['price']))]
    tr_values = [high_prices[i] - low_prices[i] for i in range(len(high_prices))]
    atr = sum(tr_values[-14:]) / 14  # 14-period ATR
    
    # Adjust ATR based on volatility coefficient
    atr *= volatility_coef
    
    # Calculate target prices based on signal
    if signal in ("STRONG BUY", "BUY"):
        tp1 = current_price * 1.04  # 4% target
        tp2 = current_price * 1.08  # 8% target
        sl = current_price - 1.5 * atr  # Stop loss based on ATR
    elif signal == "NEUTRAL":
        tp1 = current_price * 1.02  # 2% target
        tp2 = current_price * 1.04  # 4% target
        sl = current_price * 0.98  # 2% stop loss
    else:  # SELL or STRONG SELL
        tp1 = current_price * 0.96  # 4% target (downside)
        tp2 = current_price * 0.92  # 8% target (downside)
        sl = current_price + 1.5 * atr  # Stop loss based on ATR
    
    # Adjust targets based on liquidation clusters if available
    if asset_data['liquidation_data'] and asset_data['liquidation_data'].get('clusters'):
        clusters = asset_data['liquidation_data']['clusters']
        
        # Find clusters above and below current price
        clusters_above = [(p, i) for p, i in clusters if p > current_price]
        clusters_below = [(p, i) for p, i in clusters if p < current_price]
        
        if signal in ("STRONG BUY", "BUY") and clusters_above:
            # Adjust TP1 to nearest significant cluster above
            for price_level, intensity in sorted(clusters_above, key=lambda x: x[0]):
                if intensity > 0.4 and price_level > current_price * 1.02:
                    tp1 = price_level * 0.995  # Just below the cluster
                    break
            
            # Adjust TP2 to next significant cluster above
            for price_level, intensity in sorted(clusters_above, key=lambda x: x[0])[1:]:
                if intensity > 0.4 and price_level > tp1 * 1.02:
                    tp2 = price_level * 0.995  # Just below the cluster
                    break
        
        elif signal in ("SELL", "STRONG SELL") and clusters_below:
            # Adjust TP1 to nearest significant cluster below
            for price_level, intensity in sorted(clusters_below, key=lambda x: x[0], reverse=True):
                if intensity > 0.4 and price_level < current_price * 0.98:
                    tp1 = price_level * 1.005  # Just above the cluster
                    break
            
            # Adjust TP2 to next significant cluster below
            for price_level, intensity in sorted(clusters_below, key=lambda x: x[0], reverse=True)[1:]:
                if intensity > 0.4 and price_level < tp1 * 0.98:
                    tp2 = price_level * 1.005  # Just above the cluster
                    break
    
    return {
        "components": {
            "rsi_score": rsi_score,
            "volume_score": volume_score,
            "divergence": divergence,
            "liquidation_score": liquidation_score,
            "webtrend_score": webtrend_score
        },
        "final_score": round(final_score, 3),
        "signal": signal,
        "targets": {
            "TP1": round(tp1, 6 if asset_type == 'BONK' else 2),
            "TP2": round(tp2, 6 if asset_type == 'BONK' else 2),
            "SL": round(sl, 6 if asset_type == 'BONK' else 2)
        }
    }

def analyze_market_data(model_input):
    """
    Analyze market data for multiple assets
    
    Parameters:
    -----------
    model_input : dict
        Model input data
        
    Returns:
    --------
    dict
        Analysis results
    """
    results = {}
    
    for asset_name, asset_data in model_input.items():
        # Compute score
        score_data = compute_score(asset_data, asset_name)
        
        # Add asset data
        results[asset_name] = {
            "price": asset_data['price'][-1],
            "rsi": asset_data['rsi_raw'][-1],
            "rsi_sma": asset_data['rsi_sma'][-1],
            "webtrend_status": asset_data['webtrend_status'],
            **score_data
        }
    
    return results

def get_market_assessment(results):
    """
    Get overall market assessment based on analysis results
    
    Parameters:
    -----------
    results : dict
        Analysis results
        
    Returns:
    --------
    dict
        Market assessment
    """
    # Calculate average score
    scores = [result['final_score'] for result in results.values()]
    avg_score = sum(scores) / len(scores)
    
    # Find best and worst assets
    best_asset = max(results.items(), key=lambda x: x[1]['final_score'])
    worst_asset = min(results.items(), key=lambda x: x[1]['final_score'])
    
    # Determine market condition
    if avg_score > 0.3:
        market_condition = "BULLISH"
    elif avg_score < -0.3:
        market_condition = "BEARISH"
    else:
        market_condition = "NEUTRAL"
    
    # Generate rotation strategy
    assets_by_score = sorted(results.items(), key=lambda x: x[1]['final_score'], reverse=True)
    rotation_strategy = []
    
    for i, (asset, result) in enumerate(assets_by_score):
        if result['final_score'] > 0.3:
            position = "Strong" if i == 0 else "Moderate" if i == 1 else "Small"
            rotation_strategy.append(f"{position} position in {asset}")
    
    if not rotation_strategy:
        rotation_strategy = ["No clear rotation strategy. Consider waiting for better setups."]
    
    return {
        "market_condition": market_condition,
        "average_score": round(avg_score, 3),
        "best_opportunity": {
            "asset": best_asset[0],
            "score": best_asset[1]['final_score'],
            "signal": best_asset[1]['signal']
        },
        "weakest_asset": {
            "asset": worst_asset[0],
            "score": worst_asset[1]['final_score'],
            "signal": worst_asset[1]['signal']
        },
        "rotation_strategy": rotation_strategy
    }

def display_results(results, assessment):
    """
    Display analysis results
    
    Parameters:
    -----------
    results : dict
        Analysis results
    assessment : dict
        Market assessment
    """
    print("\n" + "=" * 80)
    print("ENHANCED RSI + VOLUME PREDICTIVE SCORING MODEL RESULTS")
    print("=" * 80)
    
    # Display individual asset results
    for asset, analysis in results.items():
        print(f"\n{'-' * 50}")
        print(f"{asset} ANALYSIS")
        print(f"{'-' * 50}")
        print(f"Current Price: {analysis['price']}")
        print(f"Current RSI: {analysis['rsi']:.2f}")
        print(f"Current RSI SMA: {analysis['rsi_sma']:.2f}")
        print(f"WebTrend Status: {'Uptrend' if analysis['webtrend_status'] else 'Downtrend'}")
        
        print(f"\nModel Component Scores:")
        for component, score in analysis['components'].items():
            print(f"{component.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nFinal Momentum Score: {analysis['final_score']:.3f}")
        print(f"Signal: {analysis['signal']}")
        
        print(f"\nTarget Prices:")
        print(f"TP1: {analysis['targets']['TP1']}")
        print(f"TP2: {analysis['targets']['TP2']}")
        print(f"SL: {analysis['targets']['SL']}")
    
    # Display market assessment
    print("\n" + "=" * 50)
    print("MARKET ASSESSMENT")
    print("=" * 50)
    print(f"Market Condition: {assessment['market_condition']}")
    print(f"Average Score: {assessment['average_score']}")
    print(f"\nBest Opportunity: {assessment['best_opportunity']['asset']} ({assessment['best_opportunity']['signal']})")
    print(f"Weakest Asset: {assessment['weakest_asset']['asset']} ({assessment['weakest_asset']['signal']})")
    
    print(f"\nRotation Strategy:")
    for strategy in assessment['rotation_strategy']:
        print(f"- {strategy}")

def main():
    """Main function"""
    # Load latest model input with liquidation data
    input_filepath, model_input = load_latest_model_input_with_liquidation()
    if not model_input:
        return
    
    logger.info(f"Loaded model input from {input_filepath}")
    
    # Analyze market data
    results = analyze_market_data(model_input)
    
    # Get market assessment
    assessment = get_market_assessment(results)
    
    # Display results
    display_results(results, assessment)
    
    # Save results
    output = {
        "timestamp": input_filepath.split('_')[-1].split('.')[0],
        "results": results,
        "assessment": assessment
    }
    
    output_filepath = os.path.join('data', f"model_results_{output['timestamp']}.json")
    try:
        with open(output_filepath, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    return results, assessment

if __name__ == "__main__":
    main()
