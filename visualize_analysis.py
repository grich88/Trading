import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Latest data from charts
# SOL data
sol_current_price = 196.62
sol_current_rsi_raw = 65.15
sol_current_rsi_sma = 51.37
sol_current_volume = 230.67e3  # 230.67K
sol_score = 0.632

# BTC data
btc_current_price = 116756.98
btc_current_rsi_raw = 62.44
btc_current_rsi_sma = 40.25
btc_current_volume = 1.03e3  # 1.03K
btc_score = -0.340

# BONK data
bonk_current_price = 0.00002341
bonk_current_rsi_raw = 61.61
bonk_current_rsi_sma = 41.65
bonk_current_volume = 149.42e9  # 149.42B
bonk_score = 0.300

# Create figure
plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, height_ratios=[1, 1])

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

# Create component score breakdown
ax3 = plt.subplot(gs[1, :])
ax3.set_title('Model Component Score Breakdown', fontsize=16)
ax3.set_xlim(-1, 1)
ax3.set_ylim(0, 10)
ax3.set_yticks([1, 3, 5, 7, 9])
ax3.set_yticklabels(['SOL\nRSI', 'SOL\nVolume', 'BTC\nRSI', 'BTC\nVolume', 'BONK\nRSI'])
ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Component scores (from model output)
sol_rsi_score = 0.700
sol_volume_score = 0.880
sol_divergence = 0.000

btc_rsi_score = 0.000
btc_volume_score = -0.849
btc_divergence = 0.000

bonk_rsi_score = 0.000
bonk_volume_score = 1.000
bonk_divergence = -0.500

# Plot component scores
component_colors = {
    'rsi_positive': 'royalblue',
    'rsi_negative': 'darkblue',
    'volume_positive': 'forestgreen',
    'volume_negative': 'darkgreen',
    'divergence_positive': 'purple',
    'divergence_negative': 'darkviolet'
}

# SOL components
ax3.barh(1, sol_rsi_score, height=0.6, 
         color=component_colors['rsi_positive'] if sol_rsi_score >= 0 else component_colors['rsi_negative'], 
         alpha=0.7, label='RSI Score')
ax3.barh(3, sol_volume_score, height=0.6, 
         color=component_colors['volume_positive'] if sol_volume_score >= 0 else component_colors['volume_negative'], 
         alpha=0.7, label='Volume Score')
ax3.barh(2, sol_divergence, height=0.6, 
         color=component_colors['divergence_positive'] if sol_divergence >= 0 else component_colors['divergence_negative'], 
         alpha=0.7, label='Divergence')

# BTC components
ax3.barh(5, btc_rsi_score, height=0.6, 
         color=component_colors['rsi_positive'] if btc_rsi_score >= 0 else component_colors['rsi_negative'], 
         alpha=0.7)
ax3.barh(7, btc_volume_score, height=0.6, 
         color=component_colors['volume_positive'] if btc_volume_score >= 0 else component_colors['volume_negative'], 
         alpha=0.7)
ax3.barh(6, btc_divergence, height=0.6, 
         color=component_colors['divergence_positive'] if btc_divergence >= 0 else component_colors['divergence_negative'], 
         alpha=0.7)

# BONK components
ax3.barh(9, bonk_rsi_score, height=0.6, 
         color=component_colors['rsi_positive'] if bonk_rsi_score >= 0 else component_colors['rsi_negative'], 
         alpha=0.7)
ax3.barh(8, bonk_volume_score, height=0.6, 
         color=component_colors['volume_positive'] if bonk_volume_score >= 0 else component_colors['volume_negative'], 
         alpha=0.7)
ax3.barh(8.5, bonk_divergence, height=0.6, 
         color=component_colors['divergence_positive'] if bonk_divergence >= 0 else component_colors['divergence_negative'], 
         alpha=0.7)

# Add component values
for i, (pos, val) in enumerate([
    (1, sol_rsi_score), (3, sol_volume_score), (2, sol_divergence),
    (5, btc_rsi_score), (7, btc_volume_score), (6, btc_divergence),
    (9, bonk_rsi_score), (8, bonk_volume_score), (8.5, bonk_divergence)
]):
    if val != 0:
        ax3.text(val + 0.05 if val > 0 else val - 0.15, pos, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=8)

# Add legend
handles, labels = ax3.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax3.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

# Add summary text
plt.figtext(0.5, 0.01, 'RSI + Volume Predictive Scoring Model Analysis', 
            ha='center', fontsize=14, fontweight='bold')
plt.figtext(0.5, 0.005, 'Based on 4H Charts for BTC, SOL, and BONK', 
            ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Create summary table
summary_data = {
    'Asset': ['SOL', 'BTC', 'BONK'],
    'Price': [sol_current_price, btc_current_price, bonk_current_price],
    'RSI Raw': [sol_current_rsi_raw, btc_current_rsi_raw, bonk_current_rsi_raw],
    'RSI SMA': [sol_current_rsi_sma, btc_current_rsi_sma, bonk_current_rsi_sma],
    'RSI Score': [sol_rsi_score, btc_rsi_score, bonk_rsi_score],
    'Volume Score': [sol_volume_score, btc_volume_score, bonk_volume_score],
    'Divergence': [sol_divergence, btc_divergence, bonk_divergence],
    'Final Score': [sol_score, btc_score, bonk_score],
    'Signal': ['STRONG BUY', 'SELL', 'NEUTRAL']
}

# Create DataFrame
summary_df = pd.DataFrame(summary_data)

# Print summary table
print("\n" + "=" * 50)
print("RSI + VOLUME MODEL SUMMARY")
print("=" * 50)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if x < 0.01 else f"{x:.3f}"))
