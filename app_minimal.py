import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

@st.cache_data(show_spinner=False)
def get_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return calculate_indicators(df)

def plot_price_signals(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["close"], label="Close", color="black", linewidth=1.0)
    
    # Mark signals
    if "signal" in df.columns:
        buy_mask = df["signal"].isin(["BUY", "STRONG BUY"]) & df["signal"].shift(1).ne(df["signal"])
        sell_mask = df["signal"].isin(["SELL", "STRONG SELL"]) & df["signal"].shift(1).ne(df["signal"])
        ax.scatter(df.index[buy_mask], df["close"][buy_mask], marker="^", color="green", s=30, label="Buy")
        ax.scatter(df.index[sell_mask], df["close"][sell_mask], marker="v", color="red", s=30, label="Sell")
    
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig

def plot_cumulative(df: pd.DataFrame, title: str) -> plt.Figure:
    d = df.copy()
    d["returns"] = d["close"].pct_change()
    d["position"] = d["signal_value"].shift(1).fillna(0).clip(-1, 1)
    d["strategy_returns"] = d["position"] * d["returns"]
    d["cum_bh"] = (1 + d["returns"]).cumprod() - 1
    d["cum_strat"] = (1 + d["strategy_returns"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(d.index, d["cum_bh"], label="Buy & Hold", color="#8888ff")
    ax.plot(d.index, d["cum_strat"], label="Strategy", color="#22aa22")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"

def main():
    st.set_page_config(page_title="Minimal Trading Model", layout="wide")
    st.title("RSI + Volume + Features Model – Minimal Version")

    with st.sidebar:
        st.header("Controls")
        asset = st.selectbox(
            "Asset",
            ["BTC", "SOL", "BONK"],
            index=0,
        )
        symbol = f"{asset}/USDT"
        
        # Simplified parameters
        window = st.slider("Window Size", 20, 120, 60, step=5)
        lookahead = st.slider("Lookahead Candles", 5, 40, 10, step=1)
        cutoff = st.slider("Score Cutoff (abs)", 0.0, 1.0, 0.425, step=0.005, format="%.3f")
        
        date_mode = st.radio("Data Range", ["Last year", "Full 2+ years"], index=0)
        regime_filter = st.checkbox("Regime filter (trade with trend only)", value=True)
        fee_bps = st.number_input("Transaction cost (bps)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        
        run_button = st.button("Run Analysis")

    if not run_button:
        st.info("Configure in the sidebar and click Run Analysis.")
        return

    # Date range
    now = datetime.now().date()  # Fixed deprecation warning
    if date_mode == "Last year":
        start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
    else:
        start = "2023-01-01"
        end = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    with st.spinner(f"Fetching data for {symbol} [{start} → {end}]…"):
        df = get_history(symbol, start, end)
        if df.empty:
            st.error("No data fetched. Try another range or asset.")
            return

    # Run backtest with minimal parameters
    with st.spinner("Running backtest…"):
        bt = EnhancedBacktester(
            df,
            asset_type=asset,
            static_liquidation_data=None,  # Skip liquidation data
            regime_filter=regime_filter,
            fee_bps=float(fee_bps),
            adaptive_cutoff=True,
            min_agree_features=0,  # Minimal constraints
        )
        
        results = bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
        metrics = bt.performance()

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price & Signals")
        st.pyplot(plot_price_signals(results.tail(500), f"{asset} – Last 500 candles"))
    with col2:
        st.subheader("Cumulative Returns")
        st.pyplot(plot_cumulative(results.tail(500), f"{asset} – Strategy vs Buy&Hold"))

    # Current signal snapshot
    last = results.iloc[-1]
    st.subheader("Current Signal Snapshot")
    st.write({
        "asset": asset,
        "price": float(last["close"]),
        "signal": last.get("signal"),
        "score": float(last.get("score", 0.0)),
        "TP1": float(last.get("TP1", 0.0)),
        "TP2": float(last.get("TP2", 0.0)),
        "SL": float(last.get("SL", 0.0)),
    })

    # Performance metrics
    st.subheader("Performance Metrics")
    st.table(pd.DataFrame([metrics]))
    st.markdown(
        "- **total_return**: Strategy cumulative return using long/short positions.\n"
        "- **buy_hold_return**: Passive return from holding the asset.\n"
        "- **sharpe**: Risk-adjusted return (daily approx).\n"
        "- **max_drawdown**: Worst strategy peak-to-trough decline.\n"
        "- **win_rate**: Fraction of profitable trade segments.\n"
    )

    # Export results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "ui_exports", ts, asset)
    ensure_dir(out_dir)
    results_path = os.path.join(out_dir, f"{asset}_results.csv")
    results.to_csv(results_path)
    st.success(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
