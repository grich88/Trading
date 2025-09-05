import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# Import only essential modules first
from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
from updated_rsi_volume_model import EnhancedRsiVolumePredictor

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_enhanced_minimal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app_enhanced_minimal")

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

@st.cache_data(show_spinner=False)
def get_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch and cache historical data."""
    try:
        logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
        logger.info(f"Calculating indicators for {symbol}")
        return calculate_indicators(df)
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return pd.DataFrame()

def plot_price_signals_enhanced(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot price and signals with enhanced visuals."""
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot price
        ax.plot(df.index, df["close"], label="Price", color="black", linewidth=1.0)
        
        # Add WebTrend band if available
        if {"wt_upper", "wt_lower"}.issubset(df.columns):
            ax.fill_between(df.index, df["wt_lower"], df["wt_upper"], 
                           color="#6fcf97", alpha=0.15, label="WebTrend band")
        
        if "wt_mid" in df.columns:
            ax.plot(df.index, df["wt_mid"], color="#2ecc71", linewidth=0.8, 
                   label="WebTrend mid", alpha=0.7)
        
        # Mark signals
        if "signal" in df.columns:
            buy_mask = df["signal"].isin(["BUY", "STRONG BUY"]) & df["signal"].shift(1).ne(df["signal"])
            sell_mask = df["signal"].isin(["SELL", "STRONG SELL"]) & df["signal"].shift(1).ne(df["signal"])
            
            # Larger markers for better visibility
            ax.scatter(df.index[buy_mask], df["close"][buy_mask], marker="^", color="green", 
                      s=100, label="Buy Signal", zorder=5)
            ax.scatter(df.index[sell_mask], df["close"][sell_mask], marker="v", color="red", 
                      s=100, label="Sell Signal", zorder=5)
            
            # Add annotations for TP and SL if available
            for idx in df.index[buy_mask]:
                row = df.loc[idx]
                if "tp1" in row and not pd.isna(row["tp1"]):
                    ax.axhline(y=row["tp1"], color="green", linestyle="--", alpha=0.5)
                    ax.text(idx, row["tp1"], f"TP1: ${row['tp1']:.2f}", color="green")
                if "sl" in row and not pd.isna(row["sl"]):
                    ax.axhline(y=row["sl"], color="red", linestyle="--", alpha=0.5)
                    ax.text(idx, row["sl"], f"SL: ${row['sl']:.2f}", color="red")
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error plotting price signals: {e}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", 
               horizontalalignment='center', verticalalignment='center')
        return fig

def plot_cumulative_enhanced(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot cumulative returns with enhanced visuals."""
    try:
        # Calculate cumulative returns if not already present
        if "cumulative_return" not in df.columns and "strategy_return" in df.columns:
            df["cumulative_return"] = (1 + df["strategy_return"]).cumprod() - 1
        
        if "buy_hold_return" not in df.columns:
            df["buy_hold_return"] = df["close"].pct_change()
            df["buy_hold_return"].fillna(0, inplace=True)
            df["buy_hold_cumulative"] = (1 + df["buy_hold_return"]).cumprod() - 1
        elif "buy_hold_cumulative" not in df.columns:
            df["buy_hold_cumulative"] = (1 + df["buy_hold_return"]).cumprod() - 1
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot strategy returns
        ax.plot(df.index, df["cumulative_return"] * 100, 
               label="Strategy", color="#2ecc71", linewidth=2)
        
        # Plot buy & hold returns
        ax.plot(df.index, df["buy_hold_cumulative"] * 100, 
               label="Buy & Hold", color="#3498db", linewidth=1.5, alpha=0.7)
        
        # Mark position changes
        if "position" in df.columns:
            position_changes = df["position"].diff().fillna(0)
            entries = df[position_changes > 0]
            exits = df[position_changes < 0]
            
            ax.scatter(entries.index, entries["cumulative_return"] * 100, 
                      marker="^", color="green", s=80, label="Entry")
            ax.scatter(exits.index, exits["cumulative_return"] * 100, 
                      marker="v", color="red", s=80, label="Exit")
        
        # Add final return annotations
        final_strategy = df["cumulative_return"].iloc[-1] * 100
        final_buyhold = df["buy_hold_cumulative"].iloc[-1] * 100
        
        ax.annotate(f"{final_strategy:.2f}%", 
                   xy=(df.index[-1], final_strategy),
                   xytext=(10, 0), textcoords="offset points",
                   color="#2ecc71", fontweight="bold")
        
        ax.annotate(f"{final_buyhold:.2f}%", 
                   xy=(df.index[-1], final_buyhold),
                   xytext=(10, -15), textcoords="offset points",
                   color="#3498db", fontweight="bold")
        
        # Formatting
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        ax.set_ylabel("Return (%)")
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error plotting cumulative returns: {e}")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error creating chart: {str(e)}", 
               horizontalalignment='center', verticalalignment='center')
        return fig

def main():
    st.set_page_config(page_title="Enhanced Trading Model", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #333;
        margin-top: 2rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #555;
    }
    .highlight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .signal-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .buy-signal {
        background-color: rgba(46, 204, 113, 0.15);
        border-left: 4px solid #2ecc71;
    }
    .sell-signal {
        background-color: rgba(231, 76, 60, 0.15);
        border-left: 4px solid #e74c3c;
    }
    .neutral-signal {
        background-color: rgba(149, 165, 166, 0.15);
        border-left: 4px solid #95a5a6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Enhanced Trading Model – Visual Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This dashboard analyzes cryptocurrency price data using technical indicators to generate trading signals.
    It shows you when to buy or sell, with specific price targets and stop-loss levels.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("📊 Controls")
        asset = st.selectbox(
            "Choose Cryptocurrency",
            ["BTC", "SOL", "BONK"],
            index=0,
            help="Select which cryptocurrency to analyze"
        )
        symbol = f"{asset}/USDT"
        
        # Optimal parameters based on asset
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
            ["Last 3 months", "Last year", "Full 2+ years"],
            index=0,
            help="How much historical data to analyze"
        )
        
        run_button = st.button("🔍 Run Analysis", use_container_width=True)

    if not run_button:
        st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to get started.")
        return

    try:
        # Date range
        now = datetime.utcnow().date()
        if date_mode == "Last 3 months":
            start = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        elif date_mode == "Last year":
            start = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            start = "2023-01-01"
        end = now.strftime("%Y-%m-%d")

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
        st.markdown('<h2 class="sub-header">📈 Price Chart & Signals</h2>', unsafe_allow_html=True)
        st.pyplot(plot_price_signals_enhanced(results_df.tail(500), f"{asset} – Last 500 candles"))
        
        # Display performance metrics
        st.markdown('<h2 class="sub-header">📊 Performance Metrics</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.pyplot(plot_cumulative_enhanced(results_df.tail(500), f"{asset} – Strategy vs Buy & Hold"))
        
        with col2:
            st.markdown(f"""
            ### Key Metrics
            
            **Total Return**: {metrics.get('total_return', 0.0)*100:.2f}%
            
            **Buy & Hold**: {metrics.get('buy_hold_return', 0.0)*100:.2f}%
            
            **Win Rate**: {metrics.get('win_rate', 0.0)*100:.2f}%
            
            **Sharpe Ratio**: {metrics.get('sharpe', 0.0):.2f}
            
            **Total Trades**: {metrics.get('total_trades', 0)}
            """)
        
        # Current signal
        last_row = results_df.iloc[-1]
        signal = last_row.get("signal", "NEUTRAL")
        score = float(last_row.get("score", 0.0))
        
        st.markdown('<h2 class="sub-header">🎯 Current Signal</h2>', unsafe_allow_html=True)
        
        signal_class = ""
        if signal in ("BUY", "STRONG BUY"):
            signal_class = "buy-signal"
        elif signal in ("SELL", "STRONG SELL"):
            signal_class = "sell-signal"
        else:
            signal_class = "neutral-signal"
        
        st.markdown(f"""
        <div class="signal-box {signal_class}">
        <h3>{signal}</h3>
        <p>Score: {score:+.3f}</p>
        <p>Price: ${float(last_row['close']):.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add explanation for current signal
        st.markdown('<h3>Signal Explanation</h3>', unsafe_allow_html=True)
        
        # Basic explanation based on RSI and volume
        rsi_value = last_row.get("rsi_raw", 0)
        rsi_sma = last_row.get("rsi_sma", 0)
        volume = last_row.get("volume", 0)
        avg_volume = results_df["volume"].rolling(14).mean().iloc[-1]
        
        explanations = []
        
        if rsi_value > 70:
            explanations.append("RSI is overbought (above 70), suggesting potential reversal.")
        elif rsi_value < 30:
            explanations.append("RSI is oversold (below 30), suggesting potential upward movement.")
            
        if rsi_value > rsi_sma:
            explanations.append("RSI is above its moving average, showing bullish momentum.")
        else:
            explanations.append("RSI is below its moving average, showing bearish pressure.")
            
        if volume > avg_volume * 1.5:
            explanations.append("Volume is significantly above average, confirming the price movement.")
        elif volume < avg_volume * 0.5:
            explanations.append("Volume is below average, suggesting weak conviction in the price movement.")
        
        for explanation in explanations:
            st.markdown(f"- {explanation}")
            
        # Show trade parameters if signal is not neutral
        if signal != "NEUTRAL":
            st.markdown('<h3>Trade Parameters</h3>', unsafe_allow_html=True)
            
            current_price = last_row["close"]
            tp1 = last_row.get("tp1", current_price * 1.05 if signal in ("BUY", "STRONG BUY") else current_price * 0.95)
            sl = last_row.get("sl", current_price * 0.95 if signal in ("BUY", "STRONG BUY") else current_price * 1.05)
            
            tp_pct = abs((tp1 / current_price - 1) * 100)
            sl_pct = abs((sl / current_price - 1) * 100)
            rr_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
            
            st.markdown(f"""
            - **Entry Price**: ${current_price:.2f}
            - **Take Profit**: ${tp1:.2f} ({tp_pct:.2f}%)
            - **Stop Loss**: ${sl:.2f} ({sl_pct:.2f}%)
            - **Risk-Reward Ratio**: {rr_ratio:.2f}
            """)
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error(f"An error occurred: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        st.error(f"Unhandled exception: {e}")
        st.code(traceback.format_exc())
