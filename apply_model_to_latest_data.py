from updated_rsi_volume_model import EnhancedRsiVolumePredictor, analyze_market_data, get_market_assessment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Latest data from charts (as of the most recent update)
# SOL data
sol_current_price = 198.94
sol_current_rsi_raw = 67.01
sol_current_rsi_sma = 51.37
sol_current_volume = 233.91e3  # 233.91K

# BTC data
btc_current_price = 116736.00
btc_current_rsi_raw = 62.38
btc_current_rsi_sma = 40.25
btc_current_volume = 1.05e3  # 1.05K

# BONK data
bonk_current_price = 0.00002344
bonk_current_rsi_raw = 61.83
bonk_current_rsi_sma = 41.65
bonk_current_volume = 152.82e9  # 152.82B

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

# Define liquidation data based on chart analysis
sol_liquidation_data = {
    'clusters': [
        (183.63, 0.8),  # Strong support
        (199.31, 0.6),  # Resistance
        (204.00, 0.7),  # Resistance
        (210.00, 0.5)   # Resistance
    ],
    'cleared_zone': True  # Cleared liquidation zone
}

btc_liquidation_data = {
    'clusters': [
        (113456, 0.7),  # Support
        (120336, 0.8),  # Resistance
        (122000, 0.6)   # Resistance
    ],
    'cleared_zone': False  # Not cleared liquidation zone
}

bonk_liquidation_data = {
    'clusters': [
        (0.00002167, 0.6),  # Support
        (0.00002360, 0.5),  # Resistance
        (0.00002424, 0.7),  # Resistance
        (0.00002751, 0.8)   # Strong resistance
    ],
    'cleared_zone': False  # Not cleared liquidation zone
}

# Create assets data dictionary
assets_data = {
    'SOL': {
        'price': sol_price,
        'rsi_raw': sol_rsi_raw,
        'rsi_sma': sol_rsi_sma,
        'volume': sol_volume,
        'liquidation_data': sol_liquidation_data,
        'webtrend_status': True  # WebTrend shows uptrend
    },
    'BTC': {
        'price': btc_price,
        'rsi_raw': btc_rsi_raw,
        'rsi_sma': btc_rsi_sma,
        'volume': btc_volume,
        'liquidation_data': btc_liquidation_data,
        'webtrend_status': True  # WebTrend shows uptrend
    },
    'BONK': {
        'price': bonk_price,
        'rsi_raw': bonk_rsi_raw,
        'rsi_sma': bonk_rsi_sma,
        'volume': bonk_volume,
        'liquidation_data': bonk_liquidation_data,
        'webtrend_status': True  # WebTrend shows uptrend
    }
}

# Analyze market data
results = analyze_market_data(assets_data)

# Print results
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

# Get market assessment
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

# Create visualization
plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1])

# Create score gauge chart
ax1 = plt.subplot(gs[0, 0])
ax1.set_title('RSI + Volume Model Scores', fontsize=16)
ax1.set_xlim(-1, 1)
ax1.set_ylim(0, 3)
ax1.set_yticks([0.5, 1.5, 2.5])
ax1.set_yticklabels(['SOL', 'BTC', 'BONK'])
ax1.set_xticks([-1, -0.6, -0.3, 0, 0.3, 0.6, 1])
ax1.set_xticklabels(['Strong\nBearish', 'Bearish', 'Weak\nBearish', 'Neutral', 'Weak\nBullish', 'Bullish', 'Strong\nBullish'])
ax1.grid(True, alpha=0.3)
ax1.axvspan(-1, -0.6, alpha=0.2, color='red')
ax1.axvspan(-0.6, -0.3, alpha=0.1, color='red')
ax1.axvspan(-0.3, 0.3, alpha=0.1, color='gray')
ax1.axvspan(0.3, 0.6, alpha=0.1, color='green')
ax1.axvspan(0.6, 1, alpha=0.2, color='green')
ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Plot scores as horizontal bars
sol_score = results['SOL']['final_score']
btc_score = results['BTC']['final_score']
bonk_score = results['BONK']['final_score']

ax1.barh(0.5, sol_score, height=0.5, color='green' if sol_score > 0 else 'red', alpha=0.7)
ax1.barh(1.5, btc_score, height=0.5, color='green' if btc_score > 0 else 'red', alpha=0.7)
ax1.barh(2.5, bonk_score, height=0.5, color='green' if bonk_score > 0 else 'red', alpha=0.7)

# Add score values
ax1.text(sol_score + 0.05 if sol_score > 0 else sol_score - 0.15, 0.5, f'{sol_score:.3f}', 
         va='center', ha='left' if sol_score > 0 else 'right', fontweight='bold')
ax1.text(btc_score + 0.05 if btc_score > 0 else btc_score - 0.15, 1.5, f'{btc_score:.3f}', 
         va='center', ha='left' if btc_score > 0 else 'right', fontweight='bold')
ax1.text(bonk_score + 0.05 if bonk_score > 0 else bonk_score - 0.15, 2.5, f'{bonk_score:.3f}', 
         va='center', ha='left' if bonk_score > 0 else 'right', fontweight='bold')

# Create RSI comparison chart
ax2 = plt.subplot(gs[0, 1])
ax2.set_title('RSI Analysis', fontsize=16)
ax2.set_xlim(0, 100)
ax2.set_ylim(0, 3)
ax2.set_yticks([0.5, 1.5, 2.5])
ax2.set_yticklabels(['SOL', 'BTC', 'BONK'])
ax2.set_xticks([0, 30, 50, 70, 100])
ax2.set_xticklabels(['0', '30\nOversold', '50\nNeutral', '70\nOverbought', '100'])
ax2.grid(True, alpha=0.3)
ax2.axvspan(0, 30, alpha=0.2, color='green')
ax2.axvspan(30, 50, alpha=0.1, color='gray')
ax2.axvspan(50, 70, alpha=0.1, color='gray')
ax2.axvspan(70, 100, alpha=0.2, color='red')
ax2.axvline(x=50, color='black', linestyle='-', alpha=0.3)

# Plot RSI values
ax2.barh(0.5, sol_current_rsi_raw, height=0.3, color='blue', alpha=0.7, label='RSI Raw')
ax2.barh(0.5, sol_current_rsi_sma, height=0.15, color='orange', alpha=0.7, label='RSI SMA')
ax2.barh(1.5, btc_current_rsi_raw, height=0.3, color='blue', alpha=0.7)
ax2.barh(1.5, btc_current_rsi_sma, height=0.15, color='orange', alpha=0.7)
ax2.barh(2.5, bonk_current_rsi_raw, height=0.3, color='blue', alpha=0.7)
ax2.barh(2.5, bonk_current_rsi_sma, height=0.15, color='orange', alpha=0.7)

# Add RSI values
ax2.text(sol_current_rsi_raw + 2, 0.5, f'{sol_current_rsi_raw:.1f}', va='center', fontsize=9)
ax2.text(sol_current_rsi_sma - 8, 0.5, f'{sol_current_rsi_sma:.1f}', va='center', fontsize=9, color='white')
ax2.text(btc_current_rsi_raw + 2, 1.5, f'{btc_current_rsi_raw:.1f}', va='center', fontsize=9)
ax2.text(btc_current_rsi_sma - 8, 1.5, f'{btc_current_rsi_sma:.1f}', va='center', fontsize=9, color='white')
ax2.text(bonk_current_rsi_raw + 2, 2.5, f'{bonk_current_rsi_raw:.1f}', va='center', fontsize=9)
ax2.text(bonk_current_rsi_sma - 8, 2.5, f'{bonk_current_rsi_sma:.1f}', va='center', fontsize=9, color='white')

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

# Create component score breakdown for SOL
ax3 = plt.subplot(gs[1, 0])
ax3.set_title('SOL Component Score Breakdown', fontsize=16)
ax3.set_xlim(-1, 1)
ax3.set_ylim(0, 5)
ax3.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax3.set_yticklabels(['RSI\nTrend', 'Volume\nTrend', 'Divergence', 'Liquidation', 'WebTrend'])
ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# SOL component scores
sol_components = results['SOL']['components']

# Plot SOL component scores
component_colors = {
    'positive': 'royalblue',
    'negative': 'darkred'
}

for i, (component, score) in enumerate(sol_components.items()):
    ax3.barh(i + 0.5, score, height=0.6, 
             color=component_colors['positive'] if score >= 0 else component_colors['negative'], 
             alpha=0.7)
    ax3.text(score + 0.05 if score > 0 else score - 0.15, i + 0.5, f'{score:.3f}', 
             va='center', ha='left' if score > 0 else 'right', fontsize=9)

# Create component score breakdown for BTC
ax4 = plt.subplot(gs[1, 1])
ax4.set_title('BTC Component Score Breakdown', fontsize=16)
ax4.set_xlim(-1, 1)
ax4.set_ylim(0, 5)
ax4.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax4.set_yticklabels(['RSI\nTrend', 'Volume\nTrend', 'Divergence', 'Liquidation', 'WebTrend'])
ax4.set_xticks([-1, -0.5, 0, 0.5, 1])
ax4.grid(True, alpha=0.3)
ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# BTC component scores
btc_components = results['BTC']['components']

# Plot BTC component scores
for i, (component, score) in enumerate(btc_components.items()):
    ax4.barh(i + 0.5, score, height=0.6, 
             color=component_colors['positive'] if score >= 0 else component_colors['negative'], 
             alpha=0.7)
    ax4.text(score + 0.05 if score > 0 else score - 0.15, i + 0.5, f'{score:.3f}', 
             va='center', ha='left' if score > 0 else 'right', fontsize=9)

# Create component score breakdown for BONK
ax5 = plt.subplot(gs[2, 0])
ax5.set_title('BONK Component Score Breakdown', fontsize=16)
ax5.set_xlim(-1, 1)
ax5.set_ylim(0, 5)
ax5.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
ax5.set_yticklabels(['RSI\nTrend', 'Volume\nTrend', 'Divergence', 'Liquidation', 'WebTrend'])
ax5.set_xticks([-1, -0.5, 0, 0.5, 1])
ax5.grid(True, alpha=0.3)
ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# BONK component scores
bonk_components = results['BONK']['components']

# Plot BONK component scores
for i, (component, score) in enumerate(bonk_components.items()):
    ax5.barh(i + 0.5, score, height=0.6, 
             color=component_colors['positive'] if score >= 0 else component_colors['negative'], 
             alpha=0.7)
    ax5.text(score + 0.05 if score > 0 else score - 0.15, i + 0.5, f'{score:.3f}', 
             va='center', ha='left' if score > 0 else 'right', fontsize=9)

# Create summary table
ax6 = plt.subplot(gs[2, 1])
ax6.set_title('Trading Strategy Summary', fontsize=16)
ax6.axis('off')

# Create table data
table_data = [
    ['Asset', 'Signal', 'TP1', 'TP2', 'SL'],
    ['SOL', results['SOL']['signal'], str(results['SOL']['targets']['TP1']), 
     str(results['SOL']['targets']['TP2']), str(results['SOL']['targets']['SL'])],
    ['BTC', results['BTC']['signal'], str(results['BTC']['targets']['TP1']), 
     str(results['BTC']['targets']['TP2']), str(results['BTC']['targets']['SL'])],
    ['BONK', results['BONK']['signal'], str(results['BONK']['targets']['TP1']), 
     str(results['BONK']['targets']['TP2']), str(results['BONK']['targets']['SL'])]
]

# Create table
table = ax6.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Add cell colors based on signal
for i in range(1, 4):
    signal = table_data[i][1]
    if signal == 'STRONG BUY':
        table[(i, 1)].set_facecolor('limegreen')
    elif signal == 'BUY':
        table[(i, 1)].set_facecolor('lightgreen')
    elif signal == 'NEUTRAL':
        table[(i, 1)].set_facecolor('lightgray')
    elif signal == 'SELL':
        table[(i, 1)].set_facecolor('lightcoral')
    else:  # STRONG SELL
        table[(i, 1)].set_facecolor('red')

# Add title and subtitle
plt.figtext(0.5, 0.01, 'Enhanced RSI + Volume Predictive Scoring Model Analysis', 
            ha='center', fontsize=14, fontweight='bold')
plt.figtext(0.5, 0.005, 'Based on 4H Charts for BTC, SOL, and BONK', 
            ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary table for export
summary_data = {
    'Asset': ['SOL', 'BTC', 'BONK'],
    'Price': [sol_current_price, btc_current_price, bonk_current_price],
    'RSI Raw': [sol_current_rsi_raw, btc_current_rsi_raw, bonk_current_rsi_raw],
    'RSI SMA': [sol_current_rsi_sma, btc_current_rsi_sma, bonk_current_rsi_sma],
    'RSI Score': [sol_components['rsi_score'], btc_components['rsi_score'], bonk_components['rsi_score']],
    'Volume Score': [sol_components['volume_score'], btc_components['volume_score'], bonk_components['volume_score']],
    'Divergence': [sol_components['divergence'], btc_components['divergence'], bonk_components['divergence']],
    'Liquidation Score': [sol_components['liquidation_score'], btc_components['liquidation_score'], bonk_components['liquidation_score']],
    'WebTrend Score': [sol_components['webtrend_score'], btc_components['webtrend_score'], bonk_components['webtrend_score']],
    'Final Score': [sol_score, btc_score, bonk_score],
    'Signal': [results['SOL']['signal'], results['BTC']['signal'], results['BONK']['signal']],
    'TP1': [results['SOL']['targets']['TP1'], results['BTC']['targets']['TP1'], results['BONK']['targets']['TP1']],
    'TP2': [results['SOL']['targets']['TP2'], results['BTC']['targets']['TP2'], results['BONK']['targets']['TP2']],
    'SL': [results['SOL']['targets']['SL'], results['BTC']['targets']['SL'], results['BONK']['targets']['SL']]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + "=" * 50)
print("MODEL SUMMARY TABLE")
print("=" * 50)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if x < 0.01 else f"{x:.3f}"))
