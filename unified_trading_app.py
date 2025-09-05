import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

# Import the trading system components
from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
from updated_rsi_volume_model import EnhancedRsiVolumePredictor
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

# Import necessary modules
from model_data_collector import ModelDataCollector
from liquidation_data_input import input_liquidation_data
from run_model_with_data import run_model
from visualizer import visualize_results
from backtester import run_backtest


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


def compute_cross_asset_correlations(btc: pd.DataFrame, sol: pd.DataFrame, bonk: pd.DataFrame) -> dict:
    """Compute cross-asset correlations and lead-lag relationships."""
    def prep(df: pd.DataFrame) -> pd.Series:
        x = df["close"].pct_change().dropna()
        return x
    
    res: dict = {}
    series = {"BTC": prep(btc), "SOL": prep(sol), "BONK": prep(bonk)}
    
    # Align indexes
    idx = series["BTC"].index.intersection(series["SOL"].index).intersection(series["BONK"].index)
    for k in series:
        series[k] = series[k].reindex(idx).fillna(0)
    
    windows = [30, 60, 120]
    table = []
    for w in windows:
        if len(idx) < w + 5:
            continue
        c_bs = series["BTC"].rolling(w).corr(series["SOL"]).iloc[-1]
        c_bb = series["BTC"].rolling(w).corr(series["BONK"]).iloc[-1]
        
        # Lead/lag: BTC leads by 1-2 candles
        lag1_sol = series["BTC"].shift(1).rolling(w).corr(series["SOL"]).iloc[-1]
        lag2_sol = series["BTC"].shift(2).rolling(w).corr(series["SOL"]).iloc[-1]
        lag1_bonk = series["BTC"].shift(1).rolling(w).corr(series["BONK"]).iloc[-1]
        lag2_bonk = series["BTC"].shift(2).rolling(w).corr(series["BONK"]).iloc[-1]
        
        table.append({
            "window": w,
            "BTC↔SOL": float(c_bs) if pd.notna(c_bs) else None,
            "BTC→SOL(1)": float(lag1_sol) if pd.notna(lag1_sol) else None,
            "BTC→SOL(2)": float(lag2_sol) if pd.notna(lag2_sol) else None,
            "BTC↔BONK": float(c_bb) if pd.notna(c_bb) else None,
            "BTC→BONK(1)": float(lag1_bonk) if pd.notna(lag1_bonk) else None,
            "BTC→BONK(2)": float(lag2_bonk) if pd.notna(lag2_bonk) else None,
        })
    
    res["table"] = table
    
    # Asymmetry flag: last 3 returns magnitudes
    last_moves = {
        "BTC": series["BTC"].iloc[-3:].abs().sum(),
        "SOL": series["SOL"].iloc[-3:].abs().sum(),
        "BONK": series["BONK"].iloc[-3:].abs().sum(),
    }
    res["asymmetry"] = last_moves
    return res


def nearest_cluster_distance(price: float, clusters: list[tuple[float, float]] | None) -> float | None:
    """Calculate distance to nearest liquidation cluster."""
    if not clusters:
        return None
    try:
        dists = [abs(price - p)/max(1e-9, price) for p, _ in clusters]
        return float(min(dists))
    except Exception:
        return None


def _lead_betas(btc: pd.DataFrame, target: pd.DataFrame) -> dict:
    """OLS of target returns on BTC lag1/lag2 returns over entire selected range."""
    try:
        rb = btc["close"].pct_change().dropna()
        rt = target["close"].pct_change().dropna()
        
        # Align
        idx = rb.index.intersection(rt.index)
        rb = rb.reindex(idx)
        rt = rt.reindex(idx)
        
        # Lags
        X1 = rb.shift(1).fillna(0).values
        X2 = rb.shift(2).fillna(0).values
        Y = rt.values
        
        import numpy as _np
        X = _np.column_stack([_np.ones_like(X1), X1, X2])
        beta = _np.linalg.lstsq(X, Y, rcond=None)[0]
        pred = X @ beta
        ss_res = _np.sum((Y - pred) ** 2)
        ss_tot = _np.sum((Y - _np.mean(Y)) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        
        return {"alpha": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2]), "r2": float(r2)}
    except Exception:
        return {"alpha": 0.0, "b1": 0.0, "b2": 0.0, "r2": 0.0}


def plot_price_signals_enhanced(df: pd.DataFrame, title: str, tp1=None, tp2=None, sl=None) -> plt.Figure:
    """Plot price and signals with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot price line
    ax.plot(df.index, df["close"], label="Price", color="black", linewidth=2)
    
    # Overlay WebTrend band if present
    if {"wt_upper", "wt_lower"}.issubset(df.columns):
        ax.fill_between(df.index, df["wt_lower"], df["wt_upper"], 
                       color="#6fcf97", alpha=0.15, label="WebTrend Zone")
    if "wt_mid" in df.columns:
        ax.plot(df.index, df["wt_mid"], color="#2ecc71", linewidth=1.2, label="WebTrend Mid")
    
    # Mark signals with enhanced markers
    if "signal" in df.columns:
        # Strong Buy signals
        strong_buy_mask = df["signal"].eq("STRONG BUY") & df["signal"].shift(1).ne(df["signal"])
        if strong_buy_mask.any():
            ax.scatter(df.index[strong_buy_mask], df["close"][strong_buy_mask], 
                      marker="^", color="darkgreen", s=200, label="STRONG BUY", zorder=5, edgecolors='white', linewidth=2)
            
            # Add annotations
            for idx in df.index[strong_buy_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("STRONG BUY", (idx, price*1.02), 
                           color="darkgreen", fontweight="bold", ha="center", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Buy signals
        regular_buy_mask = df["signal"].eq("BUY") & df["signal"].shift(1).ne(df["signal"])
        if regular_buy_mask.any():
            ax.scatter(df.index[regular_buy_mask], df["close"][regular_buy_mask], 
                      marker="^", color="green", s=150, label="BUY", zorder=4, edgecolors='white', linewidth=1)
        
        # Strong Sell signals
        strong_sell_mask = df["signal"].eq("STRONG SELL") & df["signal"].shift(1).ne(df["signal"])
        if strong_sell_mask.any():
            ax.scatter(df.index[strong_sell_mask], df["close"][strong_sell_mask], 
                      marker="v", color="darkred", s=200, label="STRONG SELL", zorder=5, edgecolors='white', linewidth=2)
            
            # Add annotations
            for idx in df.index[strong_sell_mask]:
                price = df.loc[idx, "close"]
                ax.annotate("STRONG SELL", (idx, price*0.98), 
                           color="darkred", fontweight="bold", ha="center", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
        
        # Sell signals
        regular_sell_mask = df["signal"].eq("SELL") & df["signal"].shift(1).ne(df["signal"])
        if regular_sell_mask.any():
            ax.scatter(df.index[regular_sell_mask], df["close"][regular_sell_mask], 
                      marker="v", color="red", s=150, label="SELL", zorder=4, edgecolors='white', linewidth=1)
    
    # Add current TP/SL levels if provided
    last_price = df["close"].iloc[-1]
    if tp1 is not None and tp2 is not None and sl is not None:
        # Add horizontal lines for TP1, TP2, SL
        ax.axhline(y=tp1, color="green", linestyle="--", linewidth=2, label=f"TP1: {tp1:.2f}", alpha=0.8)
        ax.axhline(y=tp2, color="darkgreen", linestyle="-.", linewidth=2, label=f"TP2: {tp2:.2f}", alpha=0.8)
        ax.axhline(y=sl, color="red", linestyle="--", linewidth=2, label=f"SL: {sl:.2f}", alpha=0.8)
        
        # Add shaded regions
        if tp1 > last_price:  # Long position
            ax.axhspan(tp1, tp1*1.05, alpha=0.1, color="green", label="Profit Zone")
            ax.axhspan(sl*0.95, sl, alpha=0.1, color="red", label="Loss Zone")
        elif tp1 < last_price:  # Short position
            ax.axhspan(tp1*0.95, tp1, alpha=0.1, color="green", label="Profit Zone")
            ax.axhspan(sl, sl*1.05, alpha=0.1, color="red", label="Loss Zone")
    
    # Add current price marker
    ax.axhline(y=last_price, color="blue", linestyle="-", linewidth=1.5, alpha=0.7)
    ax.scatter([df.index[-1]], [last_price], marker="o", color="blue", s=150, zorder=6, edgecolors='white', linewidth=2)
    ax.annotate(f"Current: ${last_price:.2f}", (df.index[-1], last_price), 
               xytext=(10, 10), textcoords="offset points", 
               color="blue", fontweight="bold", fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Enhance the plot appearance
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price (USDT)", fontsize=12)
    
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

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot buy & hold vs strategy
    ax.plot(d.index, d["cum_bh"]*100, label="Buy & Hold", color="#8888ff", linewidth=3, alpha=0.8)
    ax.plot(d.index, d["cum_strat"]*100, label="Strategy", color="#22aa22", linewidth=3)
    
    # Fill area between curves
    ax.fill_between(d.index, d["cum_bh"]*100, d["cum_strat"]*100, 
                   where=(d["cum_strat"] >= d["cum_bh"]), color="green", alpha=0.1, label="Outperformance")
    ax.fill_between(d.index, d["cum_bh"]*100, d["cum_strat"]*100, 
                   where=(d["cum_strat"] < d["cum_bh"]), color="red", alpha=0.1, label="Underperformance")
    
    # Add markers for significant position changes
    position_changes = d["position"] != d["position"].shift(1)
    if position_changes.any():
        change_idx = d.index[position_changes]
        change_vals = d["cum_strat"][position_changes] * 100
        ax.scatter(change_idx, change_vals, color="purple", s=60, zorder=5, 
                  label="Position Change", alpha=0.7)
    
    # Add final return values
    final_bh = d["cum_bh"].iloc[-1] * 100
    final_strat = d["cum_strat"].iloc[-1] * 100
    
    # Add performance annotations
    ax.annotate(f"Buy & Hold: {final_bh:.2f}%", 
               xy=(d.index[-1], final_bh), 
               xytext=(10, 0), textcoords="offset points",
               color="#8888ff", fontweight="bold", fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    ax.annotate(f"Strategy: {final_strat:.2f}%", 
               xy=(d.index[-1], final_strat), 
               xytext=(10, 20), textcoords="offset points",
               color="#22aa22", fontweight="bold", fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Add outperformance box
    outperf = final_strat - final_bh
    color = "green" if outperf > 0 else "red"
    ax.text(0.02, 0.95, f"Outperformance: {outperf:+.2f}%", 
           transform=ax.transAxes, color=color, fontweight="bold", fontsize=14,
           bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5", edgecolor=color))
    
    # Enhance the plot appearance
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    return fig


def plot_signal_gauge(score: float, signal: str) -> plt.Figure:
    """Create a gauge chart for the signal strength."""
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={"projection": "polar"})
    
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
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="darkred", alpha=0.8)
    
    # Sell section (light red)
    mask = (theta > 45) & (theta <= 67.5)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="red", alpha=0.6)
    
    # Neutral section (gray)
    mask = (theta > 67.5) & (theta < 112.5)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="gray", alpha=0.4)
    
    # Buy section (light green)
    mask = (theta >= 112.5) & (theta < 135)
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="green", alpha=0.6)
    
    # Strong Buy section (green)
    mask = theta >= 135
    ax.fill_between(np.radians(theta[mask]), 0, radii[mask], color="darkgreen", alpha=0.8)
    
    # Add labels
    ax.text(np.radians(22.5), 0.7, "STRONG\nSELL", ha="center", va="center", 
           fontweight="bold", color="white", fontsize=10)
    ax.text(np.radians(56.25), 0.7, "SELL", ha="center", va="center", 
           fontweight="bold", fontsize=10)
    ax.text(np.radians(90), 0.7, "NEUTRAL", ha="center", va="center", 
           fontweight="bold", fontsize=10)
    ax.text(np.radians(123.75), 0.7, "BUY", ha="center", va="center", 
           fontweight="bold", fontsize=10)
    ax.text(np.radians(157.5), 0.7, "STRONG\nBUY", ha="center", va="center", 
           fontweight="bold", color="white", fontsize=10)
    
    # Add the needle
    needle_angle = np.radians(angle)
    ax.plot([0, needle_angle], [0, 0.9], color="black", linewidth=6)
    ax.scatter(0, 0, color="black", s=100, zorder=10)
    
    # Add score text
    ax.text(np.radians(90), 0.4, f"{score:+.3f}", ha="center", va="center", 
           fontsize=20, fontweight="bold", color="black",
           bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"))
    
    # Add signal text
    signal_color = "darkgreen" if "BUY" in signal else ("darkred" if "SELL" in signal else "gray")
    ax.text(np.radians(90), 0.15, signal, ha="center", va="center", 
           fontsize=16, fontweight="bold", color=signal_color,
           bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3"))
    
    # Remove radial ticks and labels
    ax.set_rticks([])
    ax.set_title("Signal Strength Gauge", fontsize=16, fontweight="bold", pad=20)
    
    return fig


def plot_component_contributions(components: dict) -> plt.Figure:
    """Create a horizontal bar chart showing component contributions."""
    # Extract component scores
    components = {k: v for k, v in components.items() if isinstance(v, (int, float))}
    names = []
    values = []
    colors = []
    
    for name, value in components.items():
        display_name = name.replace("_score", "").replace("_", " ").title()
        names.append(display_name)
        values.append(value)
        colors.append("darkgreen" if value > 0.1 else ("darkred" if value < -0.1 else ("green" if value > 0 else ("red" if value < 0 else "gray"))))
    
    # Sort by absolute value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    names = [names[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
               f"{width:+.3f}", va="center", 
               ha="left" if width > 0 else "right",
               color="black", fontweight="bold", fontsize=11)
    
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.set_title("Component Contributions to Signal", fontweight="bold", fontsize=16, pad=20)
    ax.set_xlabel("Score Impact", fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


def plot_volume_enhanced(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot volume with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate moving average of volume
    volume_ma = df["volume"].rolling(window=20).mean()
    
    # Plot volume bars with green/red coloring based on price movement
    colors = []
    alphas = []
    for i in range(len(df)):
        if i > 0:
            color = "green" if df["close"].iloc[i] >= df["close"].iloc[i-1] else "red"
            alpha = 0.8 if df["volume"].iloc[i] > volume_ma.iloc[i] else 0.5
        else:
            color = "gray"
            alpha = 0.5
        colors.append(color)
        alphas.append(alpha)
    
    # Create bars with individual colors and alphas
    for i in range(len(df)):
        ax.bar(df.index[i], df["volume"].iloc[i], width=0.8, 
               color=colors[i], alpha=alphas[i])
    
    # Plot volume moving average
    ax.plot(df.index, volume_ma, color="blue", linewidth=2, label="20-period MA", alpha=0.8)
    
    # Add high volume markers
    high_vol = df["volume"] > volume_ma * 2
    if high_vol.any():
        ax.scatter(df.index[high_vol], df["volume"][high_vol], color="purple", s=80, 
                  marker="*", label="High Volume", zorder=5)
    
    # Add volume status annotation
    recent_vol = df["volume"].iloc[-1]
    recent_ma = volume_ma.iloc[-1]
    vol_ratio = recent_vol / recent_ma if recent_ma > 0 else 1
    
    vol_status = "HIGH" if vol_ratio > 1.5 else ("LOW" if vol_ratio < 0.7 else "NORMAL")
    vol_color = "green" if vol_ratio > 1.5 else ("red" if vol_ratio < 0.7 else "black")
    
    ax.text(0.02, 0.95, f"Current Volume: {vol_status} ({vol_ratio:.1f}x average)", 
           transform=ax.transAxes, color=vol_color, fontweight="bold", fontsize=12,
           bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5", edgecolor=vol_color))
    
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("Volume", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_rsi_vs_sma_enhanced(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot RSI vs RSI SMA with enhanced visuals."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if "rsi_raw" in df.columns:
        ax.plot(df.index, df["rsi_raw"], label="RSI", color="#6a3d9a", linewidth=2)
    if "rsi_sma" in df.columns:
        ax.plot(df.index, df["rsi_sma"], label="RSI SMA", color="#ff9900", linewidth=2)
    
    # Add overbought/oversold zones
    ax.axhspan(70, 100, alpha=0.15, color="red", label="Overbought Zone")
    ax.axhspan(0, 30, alpha=0.15, color="green", label="Oversold Zone")
    
    # Add horizontal lines
    ax.axhline(70, color="#cc6666", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(30, color="#66cc66", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axhline(50, color="gray", linestyle="-.", linewidth=1, alpha=0.6)
    
    # Add RSI crossover markers
    if "rsi_raw" in df.columns and "rsi_sma" in df.columns:
        # RSI crosses above SMA (bullish)
        cross_above = (df["rsi_raw"] > df["rsi_sma"]) & (df["rsi_raw"].shift(1) <= df["rsi_sma"].shift(1))
        if cross_above.any():
            ax.scatter(df.index[cross_above], df["rsi_raw"][cross_above], color="green", s=100, 
                      marker="^", label="Bullish Cross", zorder=5, edgecolors='white', linewidth=1)
        
        # RSI crosses below SMA (bearish)
        cross_below = (df["rsi_raw"] < df["rsi_sma"]) & (df["rsi_raw"].shift(1) >= df["rsi_sma"].shift(1))
        if cross_below.any():
            ax.scatter(df.index[cross_below], df["rsi_raw"][cross_below], color="red", s=100, 
                      marker="v", label="Bearish Cross", zorder=5, edgecolors='white', linewidth=1)
    
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
        ax.text(0.02, 0.95, f"RSI: {rsi_status} ({current_rsi:.1f})", 
               transform=ax.transAxes, color=rsi_color, fontweight="bold", fontsize=12,
               bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3", edgecolor=rsi_color))
        
        ax.text(0.02, 0.85, f"Trend: {cross_status}", 
               transform=ax.transAxes, color=cross_color, fontweight="bold", fontsize=12,
               bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.3", edgecolor=cross_color))
    
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_ylabel("RSI Value", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    
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


def _infer_webtrend_from_df(df: pd.DataFrame) -> bool:
    """Infer webtrend status from EMA alignment."""
    try:
        ema20 = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else np.nan
        ema50 = float(df["ema50"].iloc[-1]) if "ema50" in df.columns else np.nan
        ema100 = float(df["ema100"].iloc[-1]) if "ema100" in df.columns else np.nan
        last = float(df["close"].iloc[-1])
        if np.isnan(ema20) or np.isnan(ema50) or np.isnan(ema100):
            return False
        return (last > ema20) and (ema20 > ema50) and (ema50 > ema100)
    except Exception:
        return False


def _fmt_pct(x: float) -> str:
    """Format number as percentage."""
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"


def main():
    st.set_page_config(page_title="🚀 Unified Trading Model", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better visuals
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem !important;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size: 18px !important;
    }
    .highlight-green {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 200, 0, 0.05));
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00cc00;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-red {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.1), rgba(200, 0, 0, 0.05));
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #cc0000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .highlight-neutral {
        background: linear-gradient(135deg, rgba(128, 128, 128, 0.1), rgba(100, 100, 100, 0.05));
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #808080;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🚀 Unified Trading Model Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding: 20px; background: linear-gradient(135deg, #f5f7fa, #c3cfe2); border-radius: 15px;">
        <h3>🎯 Advanced Crypto Trading Analysis</h3>
        <p style="font-size: 16px; margin: 0;">
            Combines RSI, Volume, WebTrend, Liquidation Data, and Cross-Asset Analysis for optimal trading signals
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📊 Trading Controls")
        
        # Asset Selection
        st.markdown("### 🪙 Asset Selection")
        asset = st.selectbox(
            "Choose Cryptocurrency",
            ["BTC", "SOL", "BONK"],
            index=0,
            help="Select which cryptocurrency to analyze. Each asset has optimized parameters."
        )
        symbol = f"{asset}/USDT"
        
        # Set optimized defaults based on asset
        opt_window = 60 if asset == "BTC" else 40
        opt_lookahead = 10
        opt_cutoff = 0.425 if asset == "BTC" else (0.520 if asset == "SOL" else 0.550)
        
        st.markdown("### ⚙️ Analysis Parameters")
        window = st.slider(
            "Window Size (History Length)",
            20, 120, opt_window, step=5,
            help="Number of 4-hour candles to analyze. Larger = smoother signals, smaller = more responsive"
        )
        
        lookahead = st.slider(
            "Forward-Looking Period",
            5, 40, opt_lookahead, step=1,
            help="How many candles ahead to check if targets were hit (for backtesting)"
        )
        
        cutoff = st.slider(
            "Signal Strength Threshold",
            0.0, 1.0, float(opt_cutoff), step=0.005, format="%.3f",
            help="Minimum score needed to generate a signal. Higher = fewer but stronger signals"
        )
        
        st.markdown("### 🔍 Advanced Features")
        use_coinglass = st.checkbox(
            "Use Coinglass Liquidation API",
            value=False,  # Default to False due to API issues
            help="Fetch live liquidation data (may be slow or unavailable)"
        )
        
        date_mode = st.radio(
            "Historical Data Range",
            ["Last year", "Full 2+ years"],
            index=0,
            help="Amount of historical data to analyze"
        )
        
        regime_filter = st.checkbox(
            "Trend-Following Mode",
            value=True,
            help="Only trade in the direction of the trend (EMA alignment)"
        )
        
        fee_bps = st.number_input(
            "Trading Fee (basis points)",
            min_value=0.0, max_value=50.0, value=5.0, step=0.5,
            help="Trading fee in basis points (1 bp = 0.01%)"
        )
        
        adaptive_cutoff = st.checkbox(
            "Adaptive Threshold",
            value=True,
            help="Automatically adjust signal threshold based on market volatility"
        )
        
        conflict_min = st.slider(
            "Minimum Agreeing Indicators",
            0, 4, 1, step=1,
            help="How many indicators must agree before generating a signal"
        )
        
        st.markdown("### 🎯 Risk Management")
        col1, col2 = st.columns(2)
        with col1:
            tp1_long = st.number_input("Long TP1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
            tp1_short = st.number_input("Short TP1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
        with col2:
            tp2_long = st.number_input("Long TP2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
            tp2_short = st.number_input("Short TP2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
        
        atr_mult = st.number_input(
            "Stop Loss ATR Multiple", 
            min_value=0.5, max_value=5.0, value=1.5, step=0.1,
            help="Stop loss distance as a multiple of Average True Range"
        )
        
        st.markdown("### 🚀 Optimization")
        auto_cutoff = st.checkbox(
            "Auto-Optimize Threshold",
            value=False,
            help="Automatically find the best signal threshold"
        )
        
        w_override = st.checkbox(
            "Optimize Feature Weights", 
            value=False, 
            help="Fine-tune the importance of each indicator"
        )
        
        st.markdown("### 📷 External Data Sources")
        tv_text = st.text_area(
            "TradingView Panel Text",
            value="",
            help="Paste text from TradingView technical analysis panels"
        )
        
        uploads = st.file_uploader(
            "Upload TradingView Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload Technical Summary, Oscillators, Moving Averages screenshots"
        )
        
        chart_imgs = st.file_uploader(
            "Upload Chart Screenshots",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload candlestick chart screenshots for reference"
        )
        
        wt_text = st.text_area(
            "WebTrend Values",
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
                liq_top = st.number_input("Top Price", min_value=0.0, 
                                         value=120000.0 if asset == "BTC" else 230.0, 
                                         step=1.0, disabled=autodetect_scale)
            with col2:
                liq_bottom = st.number_input("Bottom Price", min_value=0.0, 
                                            value=100000.0 if asset == "BTC" else 180.0, 
                                            step=1.0, disabled=autodetect_scale)
        
        st.markdown("---")
        run_button = st.button("🔍 **RUN ANALYSIS**", use_container_width=True)
        
        with st.expander("❓ Quick Start Guide", expanded=False):
            st.markdown("""
            ### 🚀 How to Use:
            1. **Select your asset** (BTC, SOL, or BONK)
            2. **Adjust parameters** if needed (defaults are optimized)
            3. **Upload external data** for enhanced analysis (optional)
            4. **Click "RUN ANALYSIS"** to generate signals
            
            ### 📊 What You'll Get:
            - **Trading signals** with confidence scores
            - **Target prices** and stop-loss levels
            - **Performance metrics** and backtesting results
            - **Enhanced visualizations** of price action
            - **Risk management** recommendations
            """)

    if not run_button:
        # Show welcome screen with sample visualizations
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="metric-card" style="text-align: center;">
                <h3>👈 Configure Settings & Click "RUN ANALYSIS"</h3>
                <p>Select your cryptocurrency and adjust parameters in the sidebar to get started.</p>
                <p><strong>💡 Tip:</strong> Upload TradingView screenshots or liquidation heatmaps for enhanced analysis!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show feature highlights
        st.markdown("## 🌟 Key Features")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Multi-Indicator Analysis</h4>
                <p>RSI, Volume, WebTrend, Divergence, and Liquidation data combined</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Smart Risk Management</h4>
                <p>ATR-based stop losses and optimized take-profit levels</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🔄 Cross-Asset Analysis</h4>
                <p>BTC influence on altcoins with lead-lag relationships</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Enhanced Visuals</h4>
                <p>Interactive charts with signal annotations and performance metrics</p>
            </div>
            """, unsafe_allow_html=True)
        
        return

    # Main analysis starts here
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Date range setup
    status_text.text("📅 Setting up date range...")
    progress_bar.progress(10)
    
    now = datetime.now().date()  # Fixed deprecation warning
    if date_mode == "Last year":
        start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
    else:
        start = "2023-01-01"
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch historical data
    status_text.text(f"📥 Fetching historical data for {asset} and related assets...")
    progress_bar.progress(20)
    
    df = get_history(symbol, start, end)
    df_btc = get_history("BTC/USDT", start, end)
    df_sol = get_history("SOL/USDT", start, end)
    df_bonk = get_history("BONK/USDT", start, end)
    
    if df.empty:
        st.error(f"❌ No data fetched for {asset}. Please try another range or asset.")
        return
    
    progress_bar.progress(40)

    # Handle liquidation data
    status_text.text("🔍 Processing liquidation data...")
    liq = None
    
    if use_coinglass and HAS_COINGLASS:
        try:
            heatmap = get_liquidation_heatmap(asset)
            if heatmap:
                liq = extract_liquidation_clusters(heatmap, df["close"].values)
                st.success("✅ Coinglass liquidation data retrieved")
        except Exception as e:
            st.warning(f"⚠️ Coinglass API unavailable: {e}")
    
    # Process uploaded liquidation heatmaps
    if liq is None and liq_imgs:
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
            # Merge clusters across images
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
            
            merged.sort(key=lambda x: x[1], reverse=True)
            merged = merged[:8]
            liq = {"clusters": merged, "cleared_zone": False}
            st.success(f"✅ Extracted {len(merged)} liquidation clusters from {len(per_image)} image(s)")

    progress_bar.progress(60)

    # Cross-asset analysis
    status_text.text("🔄 Analyzing cross-asset relationships...")
    cross_bias = 0.0
    cutoff_adj = cutoff
    
    if asset in ("SOL", "BONK") and liq is not None and not df_btc.empty:
        tmp_bt = EnhancedBacktester(df_btc, asset_type="BTC", static_liquidation_data=liq)
        _tmp = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
        btc_sig = _tmp["signal"].iloc[-1]
        btc_price = float(_tmp["close"].iloc[-1])
        dist = nearest_cluster_distance(btc_price, liq.get("clusters"))
        
        if btc_sig in ("STRONG BUY", "STRONG SELL") and (dist is not None and dist < 0.02):
            beta = _lead_betas(df_btc, df_sol if asset == "SOL" else df_bonk)
            scale = min(0.15, 0.10 + 0.05 * max(0.0, beta.get('r2', 0.0)))
            cross_bias = scale if btc_sig == "STRONG BUY" else -scale
            cutoff_adj = max(0.0, cutoff - 0.05)

    progress_bar.progress(80)

    # Run backtest
    status_text.text("⚙️ Running backtest and optimization...")
    
    bt = EnhancedBacktester(
        df,
        asset_type=asset,
        static_liquidation_data=liq,
        regime_filter=regime_filter,
        fee_bps=float(fee_bps),
        adaptive_cutoff=adaptive_cutoff,
        min_agree_features=conflict_min,
    )
    
    # Auto-optimize cutoff if requested
    if auto_cutoff:
        best = {"cutoff": cutoff, "sharpe": -1e9, "win": 0.0}
        for c in [round(x, 3) for x in [i/100 for i in range(20, 66, 5)]]:
            tmp_bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq,
                                      regime_filter=regime_filter, fee_bps=float(fee_bps), 
                                      adaptive_cutoff=adaptive_cutoff, min_agree_features=conflict_min)
            _ = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=c)
            m = tmp_bt.performance()
            sh = float(m.get("sharpe", 0.0))
            wr = float(m.get("win_rate", 0.0))
            if sh > best["sharpe"] or (abs(sh - best["sharpe"]) < 1e-6 and wr > best["win"]):
                best = {"cutoff": c, "sharpe": sh, "win": wr}
        cutoff = best["cutoff"]
        st.success(f"🎯 Auto-optimized threshold: {cutoff:.3f} (Sharpe: {best['sharpe']:.2f}, Win Rate: {best['win']:.2%})")
    
    # Weight optimization if requested
    weight_overrides = None
    if w_override:
        best = {"w": None, "sh": -1e9}
        deltas = [-0.05, 0.0, 0.05]
        base = {"w_rsi":0.30, "w_volume":0.25, "w_divergence":0.15, "w_liquidation":0.10, "w_webtrend":0.05, "w_features":0.15, "w_sentiment":0.05}
        import itertools
        for dr, dv, dd, dl, dw, dfeat in itertools.product(deltas, repeat=6):
            w = base.copy()
            w["w_rsi"] += dr; w["w_volume"] += dv; w["w_divergence"] += dd; 
            w["w_liquidation"] += dl; w["w_webtrend"] += dw; w["w_features"] += dfeat
            tmp_bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq, 
                                      regime_filter=regime_filter, fee_bps=float(fee_bps), 
                                      adaptive_cutoff=adaptive_cutoff, min_agree_features=conflict_min, 
                                      weight_overrides=w)
            _ = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff_adj)
            m = tmp_bt.performance()
            if m.get("sharpe", 0.0) > best["sh"]:
                best = {"w": w, "sh": m["sharpe"]}
        weight_overrides = best["w"]
        if weight_overrides:
            st.success(f"🔧 Optimized feature weights (Sharpe: {best['sh']:.2f})")
        bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq, 
                              regime_filter=regime_filter, fee_bps=float(fee_bps), 
                              adaptive_cutoff=adaptive_cutoff, min_agree_features=conflict_min, 
                              weight_overrides=weight_overrides)
    
    # Run final backtest
    results = bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff_adj)
    metrics = bt.performance()
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Generate current signal
    last = results.iloc[-1]
    last_window = df.iloc[-window:]
    webtrend_status = bool(last_window.get("webtrend_status", pd.Series([_infer_webtrend_from_df(df)])).iloc[-1])
    
    # Parse WebTrend text
    wt_lines = []
    wt_trend = None
    if wt_text:
        try:
            parts = wt_text.replace(" ", "").split(";")
            for p in parts:
                if p.startswith("lines="):
                    wt_lines = [float(x) for x in p.split("=",1)[1].split(",") if x]
                if p.startswith("trend="):
                    wt_trend = float(p.split("=",1)[1])
        except Exception:
            pass

    predictor = EnhancedRsiVolumePredictor(
        rsi_sma_series=last_window.get("rsi_sma", pd.Series(np.zeros(len(last_window)))).values,
        rsi_raw_series=last_window.get("rsi_raw", pd.Series(np.zeros(len(last_window)))).values,
        volume_series=last_window.get("volume", pd.Series(np.zeros(len(last_window)))).values,
        price_series=last_window.get("close", pd.Series(np.zeros(len(last_window)))).values,
        liquidation_data=liq,
        webtrend_status=webtrend_status,
        asset_type=asset,
        webtrend_lines=wt_lines,
        webtrend_trend_value=wt_trend,
        tp_rules={"long": {"tp1_pct": tp1_long/100.0, "tp2_pct": tp2_long/100.0},
                  "short": {"tp1_pct": tp1_short/100.0, "tp2_pct": tp2_short/100.0},
                  "atr_mult": atr_mult},
    )
    
    full = predictor.get_full_analysis()
    comp = full.get("components", {})

    # Process TradingView bias
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

    # Calculate final metrics
    atr_pct = _compute_atr_percent(df)
    signal = full.get("signal")
    score = float(full.get("final_score", 0.0))
    
    # Apply biases
    combined_bias = (tv_bias if tv_bias is not None else 0.0) + cross_bias
    adjusted_score = predictor.apply_external_biases(combined_bias) if (tv_bias is not None or cross_bias != 0.0) else score
    direction = "Long" if signal in ("BUY", "STRONG BUY") else ("Short" if signal in ("SELL", "STRONG SELL") else "Flat")
    lev = _suggest_leverage(asset, signal, adjusted_score, atr_pct)

    # Display results
    st.markdown("## 🎯 Trading Recommendation")
    
    # Main signal display
    signal_color = "green" if direction == "Long" else ("red" if direction == "Short" else "gray")
    signal_class = "highlight-green" if direction == "Long" else ("highlight-red" if direction == "Short" else "highlight-neutral")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        current_price = float(last["close"])
        st.markdown(f"""
        <div class='{signal_class}'>
            <p class='big-font'>{asset} Signal: {signal}</p>
            <p class='medium-font'>Direction: {direction} | Leverage: {lev}x | Price: ${current_price:.2f}</p>
            <p>Score: {adjusted_score:+.3f} | ATR: {atr_pct:.2%} | Confidence: {'High' if abs(adjusted_score) > 0.6 else 'Medium' if abs(adjusted_score) > 0.3 else 'Low'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.pyplot(plot_signal_gauge(adjusted_score, signal))

    # Target prices and risk management
    st.markdown("## 🎯 Target Prices & Risk Management")
    
    tp1 = float(last.get("TP1", 0.0))
    tp2 = float(last.get("TP2", 0.0))
    sl = float(last.get("SL", 0.0))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if direction == "Long":
            profit1_pct = (tp1/current_price - 1) * 100
            profit2_pct = (tp2/current_price - 1) * 100
            loss_pct = (sl/current_price - 1) * 100
        elif direction == "Short":
            profit1_pct = (1 - tp1/current_price) * 100
            profit2_pct = (1 - tp2/current_price) * 100
            loss_pct = (sl/current_price - 1) * 100
        else:
            profit1_pct = profit2_pct = loss_pct = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Entry & Targets</h4>
            <p><strong>Entry:</strong> ${current_price:.2f}</p>
            <p><strong>TP1:</strong> ${tp1:.2f} ({profit1_pct:+.1f}%)</p>
            <p><strong>TP2:</strong> ${tp2:.2f} ({profit2_pct:+.1f}%)</p>
            <p><strong>Stop Loss:</strong> ${sl:.2f} ({loss_pct:+.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rr_ratio = abs(profit1_pct / loss_pct) if loss_pct != 0 else float('inf')
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚖️ Risk Management</h4>
            <p><strong>Risk/Reward:</strong> 1:{rr_ratio:.2f}</p>
            <p><strong>Position Size:</strong> {lev}x leverage</p>
            <p><strong>Max Risk:</strong> {abs(loss_pct):.1f}%</p>
            <p><strong>Volatility:</strong> {atr_pct:.2%} ATR</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.pyplot(plot_component_contributions(comp))

    # Enhanced visualizations
    st.markdown("## 📈 Enhanced Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_price_signals_enhanced(
            results.tail(500), 
            f"{asset} – Price Action & Signals", 
            tp1=tp1, tp2=tp2, sl=sl
        ))
    
    with col2:
        st.pyplot(plot_cumulative_enhanced(
            results.tail(500), 
            f"{asset} – Strategy Performance"
        ))

    # Technical indicators
    st.markdown("## 📊 Technical Indicators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_volume_enhanced(results.tail(500), f"{asset} – Volume Analysis"))
    
    with col2:
        st.pyplot(plot_rsi_vs_sma_enhanced(results.tail(500), f"{asset} – RSI Analysis"))

    # Performance metrics
    st.markdown("## 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💹 Returns</h4>
            <p><strong>Strategy:</strong> {_fmt_pct(metrics.get('total_return', 0.0))}</p>
            <p><strong>Buy & Hold:</strong> {_fmt_pct(metrics.get('buy_hold_return', 0.0))}</p>
            <p><strong>Outperformance:</strong> {_fmt_pct(metrics.get('total_return', 0.0) - metrics.get('buy_hold_return', 0.0))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Accuracy</h4>
            <p><strong>Win Rate:</strong> {_fmt_pct(metrics.get('win_rate', 0.0))}</p>
            <p><strong>TP1 Hit:</strong> {_fmt_pct(metrics.get('tp1_accuracy', 0.0))}</p>
            <p><strong>TP2 Hit:</strong> {_fmt_pct(metrics.get('tp2_accuracy', 0.0))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Risk Metrics</h4>
            <p><strong>Sharpe Ratio:</strong> {metrics.get('sharpe', 0.0):.2f}</p>
            <p><strong>Max Drawdown:</strong> {_fmt_pct(metrics.get('max_drawdown', 0.0))}</p>
            <p><strong>SL Hit Rate:</strong> {_fmt_pct(metrics.get('sl_hit_rate', 0.0))}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Trading Activity</h4>
            <p><strong>Total Trades:</strong> {metrics.get('total_trades', 0)}</p>
            <p><strong>Avg Trade:</strong> {_fmt_pct(metrics.get('avg_trade_return', 0.0))}</p>
            <p><strong>Trade Frequency:</strong> {metrics.get('total_trades', 0) / len(results) * 100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Cross-asset analysis
    if not df_btc.empty and not df_sol.empty and not df_bonk.empty:
        st.markdown("## 🔄 Cross-Asset Analysis")
        cors = compute_cross_asset_correlations(df_btc, df_sol, df_bonk)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Correlation Matrix")
            try:
                corr_df = pd.DataFrame(cors["table"])
                st.dataframe(corr_df, use_container_width=True)
            except Exception:
                st.write(cors["table"])
        
        with col2:
            asym = cors.get("asymmetry", {})
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Recent Activity</h4>
                <p><strong>BTC Movement:</strong> {asym.get('BTC', 0):.3f}</p>
                <p><strong>SOL Movement:</strong> {asym.get('SOL', 0):.3f}</p>
                <p><strong>BONK Movement:</strong> {asym.get('BONK', 0):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Lead-lag analysis
        bet_sol = _lead_betas(df_btc, df_sol)
        bet_bonk = _lead_betas(df_btc, df_bonk)
        
        st.markdown(f"""
        **Lead-Lag Relationships:** BTC→SOL (β1={bet_sol['b1']:.2f}, β2={bet_sol['b2']:.2f}, R²={bet_sol['r2']:.2f}) | 
        BTC→BONK (β1={bet_bonk['b1']:.2f}, β2={bet_bonk['b2']:.2f}, R²={bet_bonk['r2']:.2f})
        """)

    # Bias information
    if tv_bias is not None or cross_bias != 0.0:
        st.markdown("## 🔍 External Bias Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if tv_bias is not None:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📺 TradingView Bias</h4>
                    <p><strong>Bias Score:</strong> {tv_bias:+.3f}</p>
                    <p><strong>Influence:</strong> 15% weight in final score</p>
                </div>
                """, unsafe_allow_html=True)
                
                if tv_details and isinstance(tv_details, dict):
                    st.markdown("**Panel Breakdown:**")
                    panel_df = pd.DataFrame(tv_details.get("panels", []))
                    if not panel_df.empty:
                        st.dataframe(panel_df, use_container_width=True)
        
        with col2:
            if cross_bias != 0.0:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🔄 Cross-Asset Bias</h4>
                    <p><strong>BTC Influence:</strong> {cross_bias:+.3f}</p>
                    <p><strong>Reason:</strong> BTC near liquidation cluster</p>
                </div>
                """, unsafe_allow_html=True)

    # Action recommendations
    st.markdown("## 💡 Action Recommendations")
    
    def build_recommendations() -> list[str]:
        recs = []
        
        # Signal-based recommendations
        if signal in ("STRONG BUY", "STRONG SELL"):
            recs.append(f"🎯 **Strong {direction.upper()} signal detected** - High confidence trade setup")
            recs.append(f"💰 Consider {lev}x leverage with proper risk management")
        elif signal in ("BUY", "SELL"):
            recs.append(f"📈 **{direction.upper()} signal** - Moderate confidence setup")
            recs.append(f"⚖️ Use conservative position sizing ({min(lev, 2)}x leverage)")
        else:
            recs.append("⏳ **No clear signal** - Wait for better setup")
            recs.append("👀 Monitor for trend changes or stronger momentum")
        
        # Risk management
        if atr_pct > 0.05:
            recs.append("⚠️ **High volatility detected** - Reduce position size and widen stops")
        
        # Performance-based
        win_rate = metrics.get('win_rate', 0.0)
        if win_rate > 0.6:
            recs.append("✅ **High win rate strategy** - Good risk-adjusted returns expected")
        elif win_rate < 0.4:
            recs.append("⚠️ **Lower win rate** - Focus on risk management and position sizing")
        
        # Market conditions
        if regime_filter and not webtrend_status and direction == "Long":
            recs.append("🔄 **Trend filter active** - Long signals filtered in downtrend")
        elif regime_filter and webtrend_status and direction == "Short":
            recs.append("🔄 **Trend filter active** - Short signals filtered in uptrend")
        
        return recs
    
    recommendations = build_recommendations()
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

    # Export functionality
    st.markdown("## 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Save results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("data", "ui_exports", ts, asset)
        ensure_dir(out_dir)
        
        results_path = os.path.join(out_dir, f"{asset}_results.csv")
        results.to_csv(results_path)
        
        st.success(f"📊 Results saved to: `{results_path}`")
    
    with col2:
        # Create downloadable report
        report_data = {
            "asset": asset,
            "signal": signal,
            "score": adjusted_score,
            "direction": direction,
            "leverage": lev,
            "current_price": current_price,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "metrics": metrics,
            "timestamp": ts
        }
        
        report_json = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            "📄 Download Report (JSON)",
            data=report_json,
            file_name=f"{asset}_analysis_{ts}.json",
            mime="application/json"
        )
    
    with col3:
        # Save images
        try:
            price_fig = plot_price_signals_enhanced(results.tail(500), f"{asset} – Signals")
            perf_fig = plot_cumulative_enhanced(results.tail(500), f"{asset} – Performance")
            
            price_fig.savefig(os.path.join(out_dir, f"{asset}_price_signals.png"), 
                             dpi=150, bbox_inches="tight")
            perf_fig.savefig(os.path.join(out_dir, f"{asset}_performance.png"), 
                            dpi=150, bbox_inches="tight")
            
            st.success("🖼️ Charts saved to export folder")
        except Exception as e:
            st.warning(f"⚠️ Could not save charts: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🚀 <strong>Unified Trading Model Dashboard</strong> | 
        Combining technical analysis, risk management, and cross-asset intelligence</p>
        <p><em>Remember: This is for educational purposes. Always do your own research and manage risk appropriately.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Step 2: Model Execution
    model_results = execute_model(data, liquidation_data)

    # Step 3: Visualization
    visualize_results(model_results)

    # Step 4: Backtesting
    backtest_results = run_backtest(model_results)
    print("Backtest Results:", backtest_results)


if __name__ == "__main__":
    # Initialize data collector
    collector = ModelDataCollector()
    
    # Step 1: Data Collection
    symbols = ['BTC/USDT', 'SOL/USDT']  # Add more symbols as needed
    data = collector.collect_data(symbols)
    
    main()
