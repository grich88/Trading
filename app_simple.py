import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# Import basic components
from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
from updated_rsi_volume_model import EnhancedRsiVolumePredictor

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

def plot_price_signals(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot price and signals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["close"], label="Close", color="black", linewidth=1.0)
    
    # overlay WebTrend band if present
    if {"wt_upper", "wt_lower"}.issubset(df.columns):
        ax.fill_between(df.index, df["wt_lower"], df["wt_upper"], color="#6fcf97", alpha=0.15, label="WebTrend band")
    if "wt_mid" in df.columns:
        ax.plot(df.index, df["wt_mid"], color="#2ecc71", linewidth=0.8, label="WebTrend mid")
    
    # mark signals
    if "signal" in df.columns:
        buy_mask = df["signal"].isin(["BUY", "STRONG BUY"]) & df["signal"].shift(1).ne(df["signal"])
        sell_mask = df["signal"].isin(["SELL", "STRONG SELL"]) & df["signal"].shift(1).ne(df["signal"])
        ax.scatter(df.index[buy_mask], df["close"][buy_mask], marker="^", color="green", s=30, label="Buy")
        ax.scatter(df.index[sell_mask], df["close"][sell_mask], marker="v", color="red", s=30, label="Sell")
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig

def main():
    st.set_page_config(page_title="Simple Trading Model UI", layout="wide")
    st.title("Simple Trading Model – Visual Dashboard")
    
    st.markdown("""
    ### What This Tool Does
    This dashboard analyzes crypto price data using technical indicators to generate trading signals.
    It shows you when to buy or sell, with specific price targets and stop-loss levels.
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
        
        date_mode = st.radio(
            "Time Range",
            ["Last year", "Full 2+ years"],
            index=0,
            help="How much historical data to analyze"
        )
        
        run_button = st.button("🔍 Run Analysis", use_container_width=True)

    if not run_button:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to get started.")
        return

    # Date range
    now = datetime.utcnow().date()
    if date_mode == "Last year":
        start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
    else:
        start = "2023-01-01"
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    # Fetch data
    with st.spinner(f"📥 Fetching data for {asset} [{start} → {end}]..."):
        df = get_history(symbol, start, end)
        
        if df.empty:
            st.error(f"❌ No data fetched for {asset}. Try another range or asset.")
            return

    # Run backtest
    with st.spinner("⚙️ Running backtest..."):
        bt = EnhancedBacktester(df, asset_type=asset)
        results_df = bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
        metrics = bt.performance()

    # Display results
    st.header("📈 Price Chart & Signals")
    st.pyplot(plot_price_signals(results_df.tail(500), f"{asset} – Last 500 candles"))
    
    # Display metrics
    st.header("📊 Performance Metrics")
    st.write({
        "Total Return": f"{metrics.get('total_return', 0.0)*100:.2f}%",
        "Buy & Hold Return": f"{metrics.get('buy_hold_return', 0.0)*100:.2f}%",
        "Win Rate": f"{metrics.get('win_rate', 0.0)*100:.2f}%",
        "Sharpe Ratio": f"{metrics.get('sharpe', 0.0):.2f}",
        "Total Trades": metrics.get('total_trades', 0)
    })
    
    # Current signal
    last_row = results_df.iloc[-1]
    signal = last_row.get("signal", "NEUTRAL")
    score = float(last_row.get("score", 0.0))
    
    st.header("🎯 Current Signal")
    signal_color = "green" if signal in ("BUY", "STRONG BUY") else ("red" if signal in ("SELL", "STRONG SELL") else "gray")
    
    st.markdown(f"""
    <div style='padding: 10px; border-left: 5px solid {signal_color}; background-color: rgba({','.join(['0,255,0,.1' if signal_color == 'green' else '255,0,0,.1' if signal_color == 'red' else '128,128,128,.1'])})'>
    <h3 style='color: {signal_color};'>{signal}</h3>
    <p>Score: {score:+.3f}</p>
    <p>Price: ${float(last_row['close']):.2f}</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
