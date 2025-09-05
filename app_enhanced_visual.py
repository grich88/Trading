import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

# Import the enhanced trading system
from enhanced_trading_system import EnhancedTradingSystem
from historical_data_collector import fetch_historical_data, calculate_indicators
from image_extractors import (
    compute_tv_score_from_text,
    HAS_TESS,
    ocr_image_to_text,
    compute_weighted_bias_from_texts,
    extract_liq_clusters_from_image,
    auto_detect_heatmap_scale,
)

# Try to import Coinglass API
try:
    from coinglass_api import get_liquidation_heatmap, extract_liquidation_clusters
    HAS_COINGLASS = True
except Exception:
    HAS_COINGLASS = False


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


@st.cache_data(show_spinner=False)
def get_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch and cache historical data."""
    df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return calculate_indicators(df)


def plot_price_signals_enhanced(df: pd.DataFrame, title: str, tp1=None, tp2=None, sl=None) -> plt.Figure:
    """Plot price and signals with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot price line
    ax.plot(df.index, df["close"], label="Price", color="black", linewidth=1.5)
    
    # Overlay WebTrend band if present
    if {"wt_upper", "wt_lower"}.issubset(df.columns):
        ax.fill_between(df.index, df["wt_lower"], df["wt_upper"], color="#6fcf97", alpha=0.15, label="WebTrend Zone")
    if "wt_mid" in df.columns:
        ax.plot(df.index, df["wt_mid"], color="#2ecc71", linewidth=1.0, label="WebTrend Mid")
    
    # Mark signals with larger, more visible markers
    if "signal" in df.columns:
        buy_mask = df["signal"].isin(["BUY", "STRONG BUY"]) & df["signal"].shift(1).ne(df["signal"])
        sell_mask = df["signal"].isin(["SELL", "STRONG SELL"]) & df["signal"].shift(1).ne(df["signal"])
        
        # Strong Buy signals - larger green triangles
        strong_buy_mask = df["signal"].eq("STRONG BUY") & df["signal"].shift(1).ne(df["signal"])
        if strong_buy_mask.any():
            ax.scatter(df.index[strong_buy_mask], df["close"][strong_buy_mask], 
                      marker="^", color="darkgreen", s=150, label="STRONG BUY", zorder=5)
            
            # Add annotations for Strong Buy signals
            for idx in df.index[strong_buy_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("STRONG BUY", (idx, price*1.02), 
                           color="darkgreen", fontweight="bold", ha="center")
        
        # Buy signals - green triangles
        regular_buy_mask = df["signal"].eq("BUY") & df["signal"].shift(1).ne(df["signal"])
        if regular_buy_mask.any():
            ax.scatter(df.index[regular_buy_mask], df["close"][regular_buy_mask], 
                      marker="^", color="green", s=100, label="BUY", zorder=4)
            
            # Add annotations for Buy signals
            for idx in df.index[regular_buy_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("BUY", (idx, price*1.02), 
                           color="green", fontweight="bold", ha="center")
        
        # Strong Sell signals - larger red triangles
        strong_sell_mask = df["signal"].eq("STRONG SELL") & df["signal"].shift(1).ne(df["signal"])
        if strong_sell_mask.any():
            ax.scatter(df.index[strong_sell_mask], df["close"][strong_sell_mask], 
                      marker="v", color="darkred", s=150, label="STRONG SELL", zorder=5)
            
            # Add annotations for Strong Sell signals
            for idx in df.index[strong_sell_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("STRONG SELL", (idx, price*0.98), 
                           color="darkred", fontweight="bold", ha="center")
        
        # Sell signals - red triangles
        regular_sell_mask = df["signal"].eq("SELL") & df["signal"].shift(1).ne(df["signal"])
        if regular_sell_mask.any():
            ax.scatter(df.index[regular_sell_mask], df["close"][regular_sell_mask], 
                      marker="v", color="red", s=100, label="SELL", zorder=4)
            
            # Add annotations for Sell signals
            for idx in df.index[regular_sell_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("SELL", (idx, price*0.98), 
                           color="red", fontweight="bold", ha="center")
    
    # Add current TP/SL levels if provided
    last_price = df["close"].iloc[-1]
    if tp1 is not None and tp2 is not None and sl is not None:
        # Add horizontal lines for TP1, TP2, SL
        ax.axhline(y=tp1, color="green", linestyle="--", linewidth=1.5, label=f"TP1: {tp1}")
        ax.axhline(y=tp2, color="darkgreen", linestyle="-.", linewidth=1.5, label=f"TP2: {tp2}")
        ax.axhline(y=sl, color="red", linestyle="--", linewidth=1.5, label=f"SL: {sl}")
        
        # Add shaded regions
        if tp1 > last_price:  # Long position
            # Profit zone (above TP1)
            ax.axhspan(tp1, tp1*1.1, alpha=0.1, color="green", label="Profit Zone")
            # Loss zone (below SL)
            ax.axhspan(sl*0.9, sl, alpha=0.1, color="red", label="Loss Zone")
        elif tp1 < last_price:  # Short position
            # Profit zone (below TP1)
            ax.axhspan(tp1*0.9, tp1, alpha=0.1, color="green", label="Profit Zone")
            # Loss zone (above SL)
            ax.axhspan(sl, sl*1.1, alpha=0.1, color="red", label="Loss Zone")
    
    # Add current price marker
    ax.axhline(y=last_price, color="blue", linestyle="-", linewidth=1.0, alpha=0.5)
    ax.scatter([df.index[-1]], [last_price], marker="o", color="blue", s=100, zorder=6)
    ax.annotate(f"Current: {last_price:.2f}", (df.index[-1], last_price), 
               xytext=(10, 0), textcoords="offset points", 
               color="blue", fontweight="bold")
    
    # Enhance the plot appearance
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    
    # Add explanation text
    fig.text(0.02, 0.02, 
             "Green triangles (▲) show buy signals, red triangles (▼) show sell signals.\n"
             "WebTrend Zone (light green) shows market trend support/resistance area.\n"
             "Dotted lines show target profit (TP1, TP2) and stop loss (SL) levels.", 
             fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_cumulative_enhanced(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot cumulative returns with enhanced visuals."""
    d = df.copy()
    d["returns"] = d["close"].pct_change()
    d["position"] = d["signal_value"].shift(1).fillna(0).clip(-1, 1)
    d["strategy_returns"] = d["position"] * d["returns"]
    d["cum_bh"] = (1 + d["returns"]).cumprod() - 1
    d["cum_strat"] = (1 + d["strategy_returns"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot buy & hold vs strategy
    ax.plot(d.index, d["cum_bh"]*100, label="Buy & Hold", color="#8888ff", linewidth=2)
    ax.plot(d.index, d["cum_strat"]*100, label="Strategy", color="#22aa22", linewidth=2.5)
    
    # Add markers for position changes
    position_changes = d["position"] != d["position"].shift(1)
    if position_changes.any():
        change_idx = d.index[position_changes]
        change_vals = d["cum_strat"][position_changes] * 100
        ax.scatter(change_idx, change_vals, color="purple", s=50, zorder=5, 
                  label="Position Change")
    
    # Add final return values
    final_bh = d["cum_bh"].iloc[-1] * 100
    final_strat = d["cum_strat"].iloc[-1] * 100
    
    ax.annotate(f"Buy & Hold: {final_bh:.2f}%", 
               xy=(d.index[-1], final_bh), 
               xytext=(10, 0), textcoords="offset points",
               color="#8888ff", fontweight="bold")
    
    ax.annotate(f"Strategy: {final_strat:.2f}%", 
               xy=(d.index[-1], final_strat), 
               xytext=(10, 20), textcoords="offset points",
               color="#22aa22", fontweight="bold")
    
    # Add outperformance
    outperf = final_strat - final_bh
    color = "green" if outperf > 0 else "red"
    ax.text(0.02, 0.02, f"Outperformance: {outperf:+.2f}%", 
           transform=ax.transAxes, color=color, fontweight="bold",
           bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"))
    
    # Enhance the plot appearance
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    # Add explanation text
    fig.text(0.02, 0.95, 
             "This chart shows how much money you would make following the strategy vs. simply holding the asset.\n"
             "Green line above blue line means the strategy is outperforming.", 
             fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_signal_gauge(score: float, signal: str) -> plt.Figure:
    """Create a gauge chart for the signal strength."""
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"projection": "polar"})
    
    # Gauge settings
    gauge_min, gauge_max = -1, 1
    gauge_range = gauge_max - gauge_min
    
    # Normalize score to 0-180 degrees (half circle)
    angle = (score - gauge_min) / gauge_range * 180
    
    # Create the gauge background
    ax.set_theta_zero_location("N")  # 0 at the top
    ax.set_theta_direction(-1)  # clockwise
    ax.set_thetagrids([])  # remove theta grids
    
    # Set limits for half circle
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    # Create colored sections
    theta = np.linspace(0, 180, 100)
    radii = np.ones_like(theta)
    
    # Strong Sell section (red)
    mask = theta <= 45
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="red", alpha=0.6)
    
    # Sell section (light red)
    mask = (theta > 45) & (theta <= 67.5)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="red", alpha=0.3)
    
    # Neutral section (gray)
    mask = (theta > 67.5) & (theta < 112.5)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="gray", alpha=0.3)
    
    # Buy section (light green)
    mask = (theta >= 112.5) & (theta < 135)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="green", alpha=0.3)
    
    # Strong Buy section (green)
    mask = theta >= 135
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="green", alpha=0.6)
    
    # Add labels
    ax.text(np.radians(22.5), 0.7, "STRONG\nSELL", ha="center", va="center", fontweight="bold", color="white")
    ax.text(np.radians(56.25), 0.7, "SELL", ha="center", va="center", fontweight="bold")
    ax.text(np.radians(90), 0.7, "NEUTRAL", ha="center", va="center", fontweight="bold")
    ax.text(np.radians(123.75), 0.7, "BUY", ha="center", va="center", fontweight="bold")
    ax.text(np.radians(157.5), 0.7, "STRONG\nBUY", ha="center", va="center", fontweight="bold", color="white")
    
    # Add the needle
    needle_angle = np.radians(angle)
    ax.plot([0, needle_angle], [0, 0.8], color="black", linewidth=4)
    ax.scatter(0, 0, color="black", s=50, zorder=10)
    
    # Add score text
    ax.text(np.radians(90), 0.4, f"{score:+.3f}", ha="center", va="center", 
           fontsize=16, fontweight="bold", color="black",
           bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
    
    # Add signal text
    signal_color = "darkgreen" if "BUY" in signal else ("darkred" if "SELL" in signal else "gray")
    ax.text(np.radians(90), 0.2, signal, ha="center", va="center", 
           fontsize=14, fontweight="bold", color=signal_color,
           bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
    
    # Remove radial ticks and labels
    ax.set_rticks([])
    
    return fig


def plot_component_contributions(components: dict) -> plt.Figure:
    """Create a horizontal bar chart showing component contributions."""
    # Extract component scores
    components = {k: v for k, v in components.items() if isinstance(v, (int, float))}
    names = []
    values = []
    colors = []
    
    for name, value in components.items():
        names.append(name.replace("_score", "").replace("_", " ").title())
        values.append(value)
        colors.append("green" if value > 0 else ("red" if value < 0 else "gray"))
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
               f"{width:+.2f}", va="center", 
               ha="left" if width > 0 else "right",
               color="black", fontweight="bold")
    
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_title("Component Contributions to Signal", fontweight="bold")
    ax.set_xlabel("Score Impact")
    
    # Add explanation
    fig.text(0.02, 0.02, 
             "Green bars push toward BUY signals, red bars push toward SELL signals.\n"
             "Longer bars have stronger influence on the final decision.",
             fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_tp_sl_visualization(current_price: float, tp1: float, tp2: float, sl: float, 
                           direction: str) -> plt.Figure:
    """Create a visual representation of TP/SL levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Determine if long or short
    is_long = direction in ("Long", "BUY", "STRONG BUY")
    
    # Calculate price range and padding
    if is_long:
        price_min = min(current_price, sl) * 0.995
        price_max = max(current_price, tp2) * 1.005
    else:
        price_min = min(current_price, tp2) * 0.995
        price_max = max(current_price, sl) * 1.005
    
    price_range = price_max - price_min
    
    # Create a horizontal price line
    ax.axhline(y=0, color="black", linestyle="-", linewidth=2)
    
    # Plot current price, TP1, TP2, and SL
    ax.scatter([0], [0], s=300, color="blue", zorder=5, marker="o")
    
    # Normalize prices to the current price
    if is_long:
        tp1_norm = (tp1 - current_price) / price_range * 100
        tp2_norm = (tp2 - current_price) / price_range * 100
        sl_norm = (sl - current_price) / price_range * 100
        
        # Add markers
        ax.scatter([tp1_norm], [0], s=200, color="green", zorder=4, marker="o")
        ax.scatter([tp2_norm], [0], s=200, color="darkgreen", zorder=4, marker="o")
        ax.scatter([sl_norm], [0], s=200, color="red", zorder=4, marker="o")
        
        # Add labels
        ax.annotate(f"Current\n{current_price:.2f}", (0, 0), xytext=(0, 20), 
                   textcoords="offset points", ha="center", fontweight="bold", color="blue")
        ax.annotate(f"TP1\n{tp1:.2f}\n+{(tp1/current_price-1)*100:.1f}%", (tp1_norm, 0), 
                   xytext=(0, 20), textcoords="offset points", ha="center", 
                   fontweight="bold", color="green")
        ax.annotate(f"TP2\n{tp2:.2f}\n+{(tp2/current_price-1)*100:.1f}%", (tp2_norm, 0), 
                   xytext=(0, 20), textcoords="offset points", ha="center", 
                   fontweight="bold", color="darkgreen")
        ax.annotate(f"SL\n{sl:.2f}\n{(sl/current_price-1)*100:.1f}%", (sl_norm, 0), 
                   xytext=(0, -40), textcoords="offset points", ha="center", 
                   fontweight="bold", color="red")
        
        # Add profit/loss zones
        ax.axvspan(0, tp1_norm, alpha=0.1, color="green", label="Profit Zone 1")
        ax.axvspan(tp1_norm, tp2_norm, alpha=0.2, color="green", label="Profit Zone 2")
        ax.axvspan(sl_norm, 0, alpha=0.1, color="red", label="Loss Zone")
        
    else:  # Short position
        tp1_norm = (current_price - tp1) / price_range * 100
        tp2_norm = (current_price - tp2) / price_range * 100
        sl_norm = (sl - current_price) / price_range * 100
        
        # Add markers
        ax.scatter([-tp1_norm], [0], s=200, color="green", zorder=4, marker="o")
        ax.scatter([-tp2_norm], [0], s=200, color="darkgreen", zorder=4, marker="o")
        ax.scatter([sl_norm], [0], s=200, color="red", zorder=4, marker="o")
        
        # Add labels
        ax.annotate(f"Current\n{current_price:.2f}", (0, 0), xytext=(0, 20), 
                   textcoords="offset points", ha="center", fontweight="bold", color="blue")
        ax.annotate(f"TP1\n{tp1:.2f}\n{(tp1/current_price-1)*100:.1f}%", (-tp1_norm, 0), 
                   xytext=(0, 20), textcoords="offset points", ha="center", 
                   fontweight="bold", color="green")
        ax.annotate(f"TP2\n{tp2:.2f}\n{(tp2/current_price-1)*100:.1f}%", (-tp2_norm, 0), 
                   xytext=(0, 20), textcoords="offset points", ha="center", 
                   fontweight="bold", color="darkgreen")
        ax.annotate(f"SL\n{sl:.2f}\n+{(sl/current_price-1)*100:.1f}%", (sl_norm, 0), 
                   xytext=(0, -40), textcoords="offset points", ha="center", 
                   fontweight="bold", color="red")
        
        # Add profit/loss zones
        ax.axvspan(0, -tp1_norm, alpha=0.1, color="green", label="Profit Zone 1")
        ax.axvspan(-tp1_norm, -tp2_norm, alpha=0.2, color="green", label="Profit Zone 2")
        ax.axvspan(0, sl_norm, alpha=0.1, color="red", label="Loss Zone")
    
    # Set axis limits
    ax.set_xlim(-100, 100)
    ax.set_ylim(-1, 1)
    
    # Remove y-axis ticks and labels
    ax.set_yticks([])
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    
    # Add title
    title = f"{direction.upper()} Position Target Levels"
    ax.set_title(title, fontsize=16, fontweight="bold")
    
    # Add risk-reward ratio
    if is_long:
        risk = current_price - sl
        reward1 = tp1 - current_price
        reward2 = tp2 - current_price
    else:
        risk = sl - current_price
        reward1 = current_price - tp1
        reward2 = current_price - tp2
    
    rr1 = reward1 / risk if risk != 0 else float('inf')
    rr2 = reward2 / risk if risk != 0 else float('inf')
    
    # Add risk-reward text
    fig.text(0.02, 0.02, 
             f"Risk-Reward Ratio: TP1 = {rr1:.2f}:1, TP2 = {rr2:.2f}:1\n"
             f"Direction: {direction}, Suggested action: {'ENTER' if direction != 'Flat' else 'WAIT'}",
             fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    return fig


def plot_volume(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot volume with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Calculate moving average of volume
    volume_ma = df["volume"].rolling(window=20).mean()
    
    # Plot volume bars with green/red coloring based on price movement
    for i in range(len(df)):
        if i > 0:
            color = "green" if df["close"].iloc[i] >= df["close"].iloc[i-1] else "red"
            alpha = 0.7 if df["volume"].iloc[i] > volume_ma.iloc[i] else 0.4
            ax.bar(df.index[i], df["volume"].iloc[i], width=0.7, color=color, alpha=alpha)
    
    # Plot volume moving average
    ax.plot(df.index, volume_ma, color="blue", linewidth=1.5, label="20-period MA")
    
    # Add high volume markers
    high_vol = df["volume"] > volume_ma * 2
    if high_vol.any():
        ax.scatter(df.index[high_vol], df["volume"][high_vol], color="purple", s=50, 
                  marker="*", label="High Volume")
    
    # Add annotations for recent volume
    recent_vol = df["volume"].iloc[-1]
    recent_ma = volume_ma.iloc[-1]
    vol_ratio = recent_vol / recent_ma if recent_ma > 0 else 1
    
    vol_status = "HIGH" if vol_ratio > 1.5 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
    vol_color = "green" if vol_ratio > 1.5 else ("red" if vol_ratio < 0.7 else "black")
    
    ax.text(0.02, 0.92, f"Current Volume: {vol_status} ({vol_ratio:.1f}x average)", 
           transform=ax.transAxes, color=vol_color, fontweight="bold",
           bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    # Add explanation text
    fig.text(0.02, 0.02, 
             "Green bars show volume on up days, red bars show volume on down days.\n"
             "Higher volume often signals stronger price moves. Volume above the blue line is above average.",
             fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_rsi_vs_sma(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot RSI vs RSI SMA with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    if "rsi_raw" in df.columns:
        ax.plot(df.index, df["rsi_raw"], label="RSI", color="#6a3d9a", linewidth=1.5)
    if "rsi_sma" in df.columns:
        ax.plot(df.index, df["rsi_sma"], label="RSI SMA", color="#ff9900", linewidth=1.5)
    
    # Add overbought/oversold zones
    ax.axhspan(70, 100, alpha=0.1, color="red", label="Overbought")
    ax.axhspan(0, 30, alpha=0.1, color="green", label="Oversold")
    
    # Add horizontal lines
    ax.axhline(70, color="#cc6666", linestyle="--", linewidth=1.0)
    ax.axhline(30, color="#66cc66", linestyle="--", linewidth=1.0)
    ax.axhline(50, color="gray", linestyle="-.", linewidth=0.8)
    
    # Add RSI crossover markers
    if "rsi_raw" in df.columns and "rsi_sma" in df.columns:
        # RSI crosses above SMA (bullish)
        cross_above = (df["rsi_raw"] > df["rsi_sma"]) & (df["rsi_raw"].shift(1) <= df["rsi_sma"].shift(1))
        if cross_above.any():
            ax.scatter(df.index[cross_above], df["rsi_raw"][cross_above], color="green", s=80, 
                      marker="^", label="Bullish Cross")
        
        # RSI crosses below SMA (bearish)
        cross_below = (df["rsi_raw"] < df["rsi_sma"]) & (df["rsi_raw"].shift(1) >= df["rsi_sma"].shift(1))
        if cross_below.any():
            ax.scatter(df.index[cross_below], df["rsi_raw"][cross_below], color="red", s=80, 
                      marker="v", label="Bearish Cross")
    
    # Add current RSI status
    if "rsi_raw" in df.columns and "rsi_sma" in df.columns:
        current_rsi = df["rsi_raw"].iloc[-1]
        current_sma = df["rsi_sma"].iloc[-1]
        
        # Determine RSI status
        if current_rsi > 70:
            rsi_status = "OVERBOUGHT"
            rsi_color = "red"
        elif current_rsi < 30:
            rsi_status = "OVERSOLD"
            rsi_color = "green"
        else:
            rsi_status = "NEUTRAL"
            rsi_color = "black"
        
        # Determine RSI vs SMA
        if current_rsi > current_sma:
            cross_status = "BULLISH (RSI > SMA)"
            cross_color = "green"
        else:
            cross_status = "BEARISH (RSI < SMA)"
            cross_color = "red"
        
        # Add status text
        ax.text(0.02, 0.92, f"RSI Status: {rsi_status}", 
               transform=ax.transAxes, color=rsi_color, fontweight="bold",
               bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
        
        ax.text(0.02, 0.85, f"Cross Status: {cross_status}", 
               transform=ax.transAxes, color=cross_color, fontweight="bold",
               bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    
    # Add explanation text
    fig.text(0.02, 0.02, 
             "RSI above 70 is overbought (potential sell), below 30 is oversold (potential buy).\n"
             "When purple RSI line crosses above orange SMA line, it's a bullish signal (and vice versa).",
             fontsize=9)
    
    plt.tight_layout()
    return fig


def _compute_atr_percent(df: pd.DataFrame, periods: int = 14) -> float:
    """Calculate ATR as percentage of price."""
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (hi - lo),
        (hi - prev_close).abs(),
        (lo - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(periods).mean().iloc[-1]
    price = float(close.iloc[-1])
    return float(atr / price) if price else 0.0


def _suggest_leverage(asset: str, signal: str, score: float, atr_pct: float) -> int:
    """Suggest leverage based on signal strength and volatility."""
    # Base cap by asset
    base_cap = 3 if asset == "BTC" else 4
    if asset not in ("BTC", "SOL"):
        base_cap = 2
    
    # Signal strength tier
    if signal in ("STRONG BUY", "STRONG SELL"):
        lev = min(base_cap, 1 + int(round(2 + abs(score) * 2)))
    elif signal in ("BUY", "SELL"):
        lev = min(base_cap, 1 + int(round(1 + abs(score) * 2)))
    else:
        lev = 1
    
    # Volatility guardrails
    if atr_pct > 0.06:
        lev = max(1, lev - 2)
    elif atr_pct > 0.03:
        lev = max(1, lev - 1)
    
    return int(max(1, min(base_cap, lev)))


def _fmt_pct(x: float) -> str:
    """Format number as percentage."""
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"


def main():
    st.set_page_config(page_title="Enhanced Trading Model UI", layout="wide")
    
    # Custom CSS for better visuals
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:18px !important;
    }
    .highlight-green {
        background-color: rgba(0, 255, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid green;
    }
    .highlight-red {
        background-color: rgba(255, 0, 0, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid red;
    }
    .highlight-neutral {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid gray;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Enhanced Trading Model – Visual Dashboard")
    
    st.markdown("""
    ### What This Tool Does
    This dashboard analyzes crypto price data using multiple technical indicators to generate trading signals.
    It shows you when to buy or sell, with specific price targets and stop-loss levels.
    
    The model combines RSI, volume analysis, trend detection, and liquidation data to find high-probability trades.
    """)

    with st.sidebar:
        st.header("📊 Controls")
        asset = st.selectbox(
            "Choose Cryptocurrency",
            ["BTC", "SOL", "BONK"],
            index=0,
            help="Select which cryptocurrency to analyze"
        )
        symbol = f"{asset}/USDT"
        opt_window = 60 if asset == "BTC" else 40
        opt_lookahead = 10
        opt_cutoff = 0.425 if asset == "BTC" else (0.520 if asset == "SOL" else 0.550)
        
        st.markdown("### 📈 Analysis Settings")
        window = st.slider(
            "Window Size (History Length)",
            20,
            120,
            opt_window,
            step=5,
            help="How many 4-hour candles to look back. Larger = smoother signals, smaller = more responsive"
        )
        lookahead = st.slider(
            "Forward-Looking Period",
            5,
            40,
            opt_lookahead,
            step=1,
            help="How many candles to look ahead when testing if targets were hit"
        )
        cutoff = st.slider(
            "Signal Strength Threshold",
            0.0,
            1.0,
            float(opt_cutoff),
            step=0.005,
            format="%.3f",
            help="Minimum score needed to generate a signal. Higher = fewer but stronger signals"
        )
        
        st.markdown("### 🔍 Advanced Features")
        use_coinglass = st.checkbox(
            "Use Liquidation Data",
            value=False and HAS_COINGLASS,
            help="Include liquidation clusters to identify key price levels"
        )
        date_mode = st.radio(
            "Time Range",
            ["Last year", "Full 2+ years"],
            index=0,
            help="How much historical data to analyze"
        )
        regime_filter = st.checkbox(
            "Only Trade With Trend",
            value=True,
            help="Only go long in uptrends and short in downtrends"
        )
        fee_bps = st.number_input(
            "Trading Fee (basis points)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Trading fee in basis points (1 bp = 0.01%)"
        )
        adaptive_cutoff = st.checkbox(
            "Adjust Threshold in High Volatility",
            value=True,
            help="Automatically increase signal threshold when market is volatile"
        )
        conflict_min = st.slider(
            "Minimum Agreeing Indicators",
            0,
            4,
            1,
            step=1,
            help="How many indicators must agree before generating a signal"
        )
        
        st.markdown("### 🎯 Profit Targets & Stop Loss")
        tp1_long = st.number_input("Long Take Profit 1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5, 
                                  help="First profit target for long positions")
        tp2_long = st.number_input("Long Take Profit 2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5,
                                  help="Second profit target for long positions")
        tp1_short = st.number_input("Short Take Profit 1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5,
                                   help="First profit target for short positions")
        tp2_short = st.number_input("Short Take Profit 2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5,
                                   help="Second profit target for short positions")
        atr_mult = st.number_input("Stop Loss ATR Multiple", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                                  help="Stop loss distance as a multiple of Average True Range")
        
        st.markdown("### ⚙️ Optimization Options")
        auto_cutoff = st.checkbox(
            "Auto-Optimize Threshold",
            value=False,
            help="Automatically find the best signal threshold for maximum returns"
        )
        w_override = st.checkbox(
            "Optimize Indicator Weights", 
            value=False, 
            help="Find the best combination of indicator weights for this asset"
        )
        use_calibration = st.checkbox(
            "Use Expected Value Filter",
            value=True,
            help="Only take trades with positive expected value based on historical performance"
        )
        use_adaptive_timeframe = st.checkbox(
            "Adapt to Market Volatility",
            value=False,
            help="Automatically switch timeframes based on market conditions"
        )
        use_cross_asset = st.checkbox(
            "Consider Other Assets",
            value=True,
            help="Include correlations between BTC, SOL and BONK in the analysis"
        )
        
        st.markdown("### 📷 External Data")
        tv_text = st.text_area(
            "TradingView Panel Text (Optional)",
            value="",
            help="Paste text from TradingView technical panels"
        )
        uploads = st.file_uploader(
            "Upload TradingView Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload Technical Summary, Oscillators, Moving Averages, or Pivots screenshots"
        )
        chart_imgs = st.file_uploader(
            "Upload Chart Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload candlestick chart screenshots for visual reference"
        )
        wt_text = st.text_area(
            "WebTrend Values (Optional)",
            value="",
            help="Format: lines=171.93,191.74,171.93; trend=165.62"
        )
        liq_imgs = st.file_uploader(
            "Upload Liquidation Heatmaps",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload Coinglass liquidation heatmap screenshots"
        )
        
        if liq_imgs:
            autodetect_scale = st.checkbox("Auto-Detect Heatmap Scale", value=HAS_TESS)
            col1, col2 = st.columns(2)
            with col1:
                liq_top = st.number_input("Top Price", min_value=0.0, value=120000.0 if asset == "BTC" else 230.0, 
                                         step=1.0, disabled=autodetect_scale)
            with col2:
                liq_bottom = st.number_input("Bottom Price", min_value=0.0, value=100000.0 if asset == "BTC" else 180.0, 
                                            step=1.0, disabled=autodetect_scale)
        
        run_button = st.button("🔍 Run Analysis", use_container_width=True)
        
        with st.expander("❓ How to Use This Tool", expanded=False):
            st.markdown("""
            1. **Select your cryptocurrency** (BTC, SOL, or BONK)
            2. **Adjust analysis settings** if needed (or keep defaults)
            3. **Click "Run Analysis"** to generate signals and visualizations
            4. **Review the trading recommendation** and supporting charts
            5. **Check the Target Price visualization** to understand entry/exit points
            
            **Advanced users:** Upload TradingView screenshots or liquidation heatmaps for enhanced analysis.
            """)

    if not run_button:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to get started.")
        
        # Show sample visuals when no analysis is running
        st.markdown("### 📊 Sample Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://i.imgur.com/JQzXiVQ.png", caption="Sample Price Chart with Signals")
        with col2:
            st.image("https://i.imgur.com/8YzTGZJ.png", caption="Sample Target Price Visualization")
        
        return

    # Date range
    now = datetime.utcnow().date()
    if date_mode == "Last year":
        start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
    else:
        start = "2023-01-01"
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # Initialize the enhanced trading system
    with st.spinner("🔄 Initializing trading system..."):
        trading_system = EnhancedTradingSystem(assets=["BTC", "SOL", "BONK"])

    # Fetch data for all assets
    with st.spinner(f"📥 Fetching data for {asset} and related assets [{start} → {end}]..."):
        # Use adaptive timeframe if selected
        timeframe = "4h"  # Default
        if use_adaptive_timeframe:
            # Fetch multi-timeframe data
            multi_tf_data = trading_system.fetch_multi_timeframe_data(start, end)
            optimal_timeframes = trading_system.select_optimal_timeframes(multi_tf_data)
            timeframe = optimal_timeframes.get(asset, "4h")
            st.success(f"✅ Selected optimal timeframe for {asset}: {timeframe} based on current volatility")
        
        # Fetch data for all assets
        data = trading_system.fetch_data(start, end, timeframe=timeframe)
        
        if asset not in data or data[asset].empty:
            st.error(f"❌ No data fetched for {asset}. Try another range or asset.")
            return

    # Optional liquidation clusters
    liq = None
    if use_coinglass and HAS_COINGLASS:
        with st.spinner("🔍 Querying liquidation data..."):
            try:
                heatmap = get_liquidation_heatmap(asset)
                if heatmap:
                    liq = extract_liquidation_clusters(heatmap, data[asset]["close"].values)
                    st.success("✅ Liquidation data retrieved successfully")
            except Exception as e:
                st.warning(f"⚠️ Coinglass API unavailable: {e}")
    
    # Fallback: extract clusters from uploaded heatmap image
    if liq is None and liq_imgs:
        with st.spinner("🔍 Extracting liquidation clusters from images..."):
            per_image = []
            for f in liq_imgs:
                data_bytes = f.read()
                top_val, bot_val = float(liq_top), float(liq_bottom)
                if autodetect_scale:
                    guess = auto_detect_heatmap_scale(data_bytes)
                    if guess:
                        top_val, bot_val = guess
                clusters = extract_liq_clusters_from_image(data_bytes, price_top=top_val, price_bottom=bot_val)
                if clusters:
                    per_image.append({"file": f.name, "top": top_val, "bottom": bot_val, "clusters": clusters})
            
            if per_image:
                # Merge clusters across images (price proximity within 0.5%)
                merged = []
                for entry in per_image:
                    for p, s in entry["clusters"]:
                        matched = False
                        for i, (mp, ms) in enumerate(merged):
                            if abs(p - mp) / max(1e-9, mp) < 0.005:
                                merged[i] = ((mp + p) / 2.0, max(ms, s))
                                matched = True
                                break
                        if not matched:
                            merged.append((p, s))
                
                # Sort and cap
                merged.sort(key=lambda x: x[1], reverse=True)
                merged = merged[:8]
                liq = {"clusters": merged, "cleared_zone": False}
                st.success(f"✅ Found {len(merged)} liquidation clusters from {len(per_image)} image(s)")
                
                # Show comparison table
                with st.expander("📊 Liquidation Clusters", expanded=False):
                    try:
                        rows = []
                        for entry in per_image:
                            for p, s in entry["clusters"]:
                                rows.append({"File": entry["file"], "Price Level": round(p, 2), "Strength": round(s, 3)})
                        if rows:
                            st.table(pd.DataFrame(rows))
                    except Exception:
                        pass

    # Calculate WebTrend
    with st.spinner("📊 Calculating trend indicators..."):
        webtrend_data = trading_system.calculate_webtrend(data)

    # Analyze cross-asset relationships if selected
    cross_asset_biases = {}
    if use_cross_asset:
        with st.spinner("🔄 Analyzing relationships between assets..."):
            cross_asset_results = trading_system.analyze_cross_asset_relationships(data)
            cross_asset_biases = trading_system.get_cross_asset_biases(data)
            
            if asset in cross_asset_biases and abs(cross_asset_biases[asset].get("bias", 0)) > 0.01:
                bias = cross_asset_biases[asset].get("bias", 0)
                confidence = cross_asset_biases[asset].get("confidence", 0)
                direction = "bullish" if bias > 0 else "bearish"
                st.info(f"ℹ️ BTC is showing a {direction} influence on {asset} (bias: {bias:.2f}, confidence: {confidence:.2f})")

    # Set up parameters for backtest
    params = {
        asset: {
            "window_size": window,
            "lookahead_candles": lookahead,
            "min_abs_score": cutoff,
            "regime_filter": regime_filter,
            "fee_bps": fee_bps,
            "adaptive_cutoff": adaptive_cutoff,
            "min_agree_features": conflict_min,
            "tp_rules": {
                "long": {"tp1_pct": tp1_long/100.0, "tp2_pct": tp2_long/100.0},
                "short": {"tp1_pct": tp1_short/100.0, "tp2_pct": tp2_short/100.0},
                "atr_mult": atr_mult
            }
        }
    }

    # Run backtest
    with st.spinner("⚙️ Running backtest and generating signals..."):
        backtest_results = trading_system.run_backtest(data, params)
        results_df = backtest_results[asset]["dataframe"]
        metrics = backtest_results[asset]["metrics"]

    # Optional TV OCR bias
    tv_bias = None
    tv_details = None
    sources = []
    if tv_text and isinstance(tv_text, str) and tv_text.strip():
        sources.append((tv_text, "text"))
    if uploads:
        texts = []
        for f in uploads:
            data_bytes = f.read()
            text = ocr_image_to_text(data_bytes) if HAS_TESS else None
            if text:
                texts.append((text, f.name))
        if texts:
            agg = compute_weighted_bias_from_texts(texts)
            tv_bias = float(agg.get("bias", 0.0))
            tv_details = agg
    if tv_bias is None and sources:
        tv_parse = compute_tv_score_from_text(sources[0][0])
        tv_bias = float(tv_parse["score"]) if isinstance(tv_parse, dict) else None

    # Generate signals
    with st.spinner("🔄 Generating trading signals..."):
        # Parse optional WebTrend text
        wt_lines = []
        wt_trend = None
        if wt_text:
            try:
                # Simple parser: lines=a,b,c; trend=x
                parts = wt_text.replace(" ", "").split(";")
                for p in parts:
                    if p.startswith("lines="):
                        wt_lines = [float(x) for x in p.split("=",1)[1].split(",") if x]
                    if p.startswith("trend="):
                        wt_trend = float(p.split("=",1)[1])
            except Exception:
                pass
        
        # Generate signals
        signals = trading_system.generate_signals(
            data, 
            webtrend_data=webtrend_data if webtrend_data else None,
            cross_asset_biases=cross_asset_biases if use_cross_asset else None
        )
        
        # Get current signal
        signal_data = signals[asset]
        
        # Apply TV bias if available
        if tv_bias is not None:
            score = signal_data["final_score"]
            adjusted_score = score + 0.15 * tv_bias  # 15% blend
            adjusted_score = max(-1.0, min(1.0, adjusted_score))
            signal_data["tv_bias"] = tv_bias
            signal_data["tv_adjusted_score"] = adjusted_score
            
            # Recalculate signal if score changed significantly
            if abs(adjusted_score - score) > 0.1:
                if adjusted_score > 0.6:
                    signal_data["tv_adjusted_signal"] = "STRONG BUY"
                elif adjusted_score > 0.3:
                    signal_data["tv_adjusted_signal"] = "BUY"
                elif adjusted_score >= -0.3:
                    signal_data["tv_adjusted_signal"] = "NEUTRAL"
                elif adjusted_score >= -0.6:
                    signal_data["tv_adjusted_signal"] = "SELL"
                else:
                    signal_data["tv_adjusted_signal"] = "STRONG SELL"

    # Calculate ATR%
    atr_pct = _compute_atr_percent(results_df)
    
    # Determine signal and direction
    signal = signal_data.get("signal")
    if use_calibration and "calibrated_signal" in signal_data:
        signal = signal_data["calibrated_signal"]
    elif tv_bias is not None and "tv_adjusted_signal" in signal_data:
        signal = signal_data["tv_adjusted_signal"]
    
    score = signal_data.get("final_score", 0.0)
    if tv_bias is not None and "tv_adjusted_score" in signal_data:
        score = signal_data["tv_adjusted_score"]
    
    direction = "Long" if signal in ("BUY", "STRONG BUY") else ("Short" if signal in ("SELL", "STRONG SELL") else "Flat")
    lev = _suggest_leverage(asset, signal, score, atr_pct)

    # Display Trading Recommendation
    st.header("📊 Trading Recommendation")
    
    # Create a visually appealing signal box
    signal_color = "green" if direction == "Long" else ("red" if direction == "Short" else "gray")
    signal_class = "highlight-green" if direction == "Long" else ("highlight-red" if direction == "Short" else "highlight-neutral")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"<div class='{signal_class}'>"
                   f"<p class='big-font'>{asset} Signal: {signal}</p>"
                   f"<p class='medium-font'>Direction: {direction} | Leverage: {lev}x | Current Price: ${float(results_df['close'].iloc[-1]):.2f}</p>"
                   f"</div>", unsafe_allow_html=True)
    
    with col2:
        # Display the signal gauge
        st.pyplot(plot_signal_gauge(score, signal))
    
    with col3:
        # Display component contributions
        st.pyplot(plot_component_contributions(signal_data.get('components', {})))

    # Display target prices
    st.subheader("🎯 Target Prices & Stop Loss")
    tp1 = float(signal_data.get("targets", {}).get("TP1", 0.0))
    tp2 = float(signal_data.get("targets", {}).get("TP2", 0.0))
    sl = float(signal_data.get("targets", {}).get("SL", 0.0))
    current_price = float(results_df["close"].iloc[-1])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Display target price visualization
        st.pyplot(plot_tp_sl_visualization(current_price, tp1, tp2, sl, direction))
    
    with col2:
        # Display targets in a more readable format
        if direction == "Long":
            profit1_pct = (tp1/current_price - 1) * 100
            profit2_pct = (tp2/current_price - 1) * 100
            loss_pct = (sl/current_price - 1) * 100
            
            st.markdown(f"""
            ### Entry & Exit Plan
            
            **Entry Price**: ${current_price:.2f}
            
            **Take Profit 1**: ${tp1:.2f} (+{profit1_pct:.2f}%)
            
            **Take Profit 2**: ${tp2:.2f} (+{profit2_pct:.2f}%)
            
            **Stop Loss**: ${sl:.2f} ({loss_pct:.2f}%)
            
            **Risk/Reward Ratio**: 1:{abs(profit1_pct/loss_pct):.2f}
            """)
            
            # Add action steps
            st.markdown("""
            ### Action Steps:
            1. **Enter Long** at current price
            2. **Set Stop Loss** at the indicated level
            3. **Take partial profits** at TP1
            4. **Move stop loss to breakeven** after TP1 is hit
            5. **Let remaining position run** to TP2
            """)
            
        elif direction == "Short":
            profit1_pct = (1 - tp1/current_price) * 100
            profit2_pct = (1 - tp2/current_price) * 100
            loss_pct = (sl/current_price - 1) * 100
            
            st.markdown(f"""
            ### Entry & Exit Plan
            
            **Entry Price**: ${current_price:.2f}
            
            **Take Profit 1**: ${tp1:.2f} (+{profit1_pct:.2f}%)
            
            **Take Profit 2**: ${tp2:.2f} (+{profit2_pct:.2f}%)
            
            **Stop Loss**: ${sl:.2f} ({loss_pct:.2f}%)
            
            **Risk/Reward Ratio**: 1:{abs(profit1_pct/loss_pct):.2f}
            """)
            
            # Add action steps
            st.markdown("""
            ### Action Steps:
            1. **Enter Short** at current price
            2. **Set Stop Loss** at the indicated level
            3. **Take partial profits** at TP1
            4. **Move stop loss to breakeven** after TP1 is hit
            5. **Let remaining position run** to TP2
            """)
        else:
            st.markdown("""
            ### No Trade Recommended
            
            The current market conditions don't meet our criteria for a high-probability trade.
            
            **Recommendation**: Wait for a stronger signal before entering a position.
            """)

    # Display enhanced price chart
    st.header("📈 Price Chart & Signals")
    st.pyplot(plot_price_signals_enhanced(
        results_df.tail(500), 
        f"{asset} – Price Action with Signals", 
        tp1=tp1, tp2=tp2, sl=sl
    ))
    
    # Add chart explanation
    with st.expander("📊 How to Read This Chart", expanded=False):
        st.markdown("""
        - **Green triangles (▲)** mark buy signals
        - **Red triangles (▼)** mark sell signals
        - **Green shaded area** shows the WebTrend zone (trend support/resistance)
        - **Horizontal lines** show target profits (green) and stop loss (red)
        - **Current price** is marked with a blue dot
        """)

    # Display performance metrics
    st.header("📊 Strategy Performance")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display cumulative returns chart
        st.pyplot(plot_cumulative_enhanced(results_df.tail(500), f"{asset} – Strategy vs Buy & Hold"))
    
    with col2:
        # Display key metrics in a more readable format
        st.markdown(f"""
        ### Key Metrics
        
        **Total Return**: {metrics.get('total_return', 0.0)*100:.2f}%
        
        **Buy & Hold**: {metrics.get('buy_hold_return', 0.0)*100:.2f}%
        
        **Win Rate**: {metrics.get('win_rate', 0.0)*100:.2f}%
        
        **Sharpe Ratio**: {metrics.get('sharpe', 0.0):.2f}
        
        **Total Trades**: {metrics.get('total_trades', 0)}
        """)