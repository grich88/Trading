import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st

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


def plot_cumulative(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot cumulative returns."""
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


def plot_volume(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot volume."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(df.index, df["volume"], width=0.02, color="#77aaff")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return fig


def plot_rsi_vs_sma(df: pd.DataFrame, title: str) -> plt.Figure:
    """Plot RSI vs RSI SMA."""
    fig, ax = plt.subplots(figsize=(10, 3))
    if "rsi_raw" in df.columns:
        ax.plot(df.index, df["rsi_raw"], label="RSI", color="#6a3d9a", linewidth=1.0)
    if "rsi_sma" in df.columns:
        ax.plot(df.index, df["rsi_sma"], label="RSI SMA", color="#ff9900", linewidth=1.0)
    ax.axhline(70, color="#cc6666", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="#66cc66", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left")
    return fig


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


def _fmt_pct(x: float) -> str:
    """Format number as percentage."""
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"


def main():
    st.set_page_config(page_title="Enhanced Trading Model UI", layout="wide")
    st.title("Enhanced RSI + Volume + Features Model – Local UI")

    with st.sidebar:
        st.header("Controls")
        asset = st.selectbox(
            "Asset",
            ["BTC", "SOL", "BONK"],
            index=0,
            help=(
                "Choose the market to evaluate. This sets asset-specific defaults, volatility"
                " coefficients, and optimized parameters used by the model."
            ),
        )
        symbol = f"{asset}/USDT"
        opt_window = 60 if asset == "BTC" else 40
        opt_lookahead = 10
        opt_cutoff = 0.425 if asset == "BTC" else (0.520 if asset == "SOL" else 0.550)
        window = st.slider(
            "Window Size",
            20,
            120,
            opt_window,
            step=5,
            help=(
                "Minimum number of past 4h candles used as context before generating"
                " signals. Larger window = smoother, slower signals; smaller window ="
                " more sensitive, more trades."
            ),
        )
        lookahead = st.slider(
            "Lookahead Candles",
            5,
            40,
            opt_lookahead,
            step=1,
            help=(
                "Backtest evaluation horizon. After a signal, we look up to this many"
                " candles to see if TP1/TP2/SL were hit. Affects metrics only, not the"
                " realtime signal logic."
            ),
        )
        cutoff = st.slider(
            "Score Cutoff (abs)",
            0.0,
            1.0,
            float(opt_cutoff),
            step=0.005,
            format="%.3f",
            help=(
                "Minimum absolute model score required to take a trade in backtests."
                " Higher cutoff = fewer but higher-conviction trades; lower cutoff ="
                " more trades with lower selectivity."
            ),
        )
        use_coinglass = st.checkbox(
            "Use Coinglass Heatmap (if available)",
            value=False and HAS_COINGLASS,
            help=(
                "When enabled, the model will pull liquidation clusters and bias TP/SL"
                " toward those levels if the endpoint responds for your API tier."
            ),
        )
        date_mode = st.radio(
            "Data Range",
            ["Last year", "Full 2+ years"],
            index=0,
            help=(
                "Historical span used for analysis and plots. Longer ranges provide more"
                " context but take a bit longer to fetch and process."
            ),
        )
        regime_filter = st.checkbox(
            "Regime filter (trade with trend only)",
            value=True,
            help="Longs only in uptrend (EMA stack), shorts only in downtrend.",
        )
        fee_bps = st.number_input(
            "Transaction cost (bps)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Applied when position changes; 1 bp = 0.01%.",
        )
        adaptive_cutoff = st.checkbox(
            "Adaptive cutoff (raise in high ATR)",
            value=True,
            help="Increases score threshold when volatility (ATR%) is high.",
        )
        conflict_min = st.slider(
            "Minimum agreeing features",
            0,
            4,
            1,
            step=1,
            help="Require this many of {RSI, WebTrend, Liquidity, Divergence} to agree with the trade direction.",
        )
        st.markdown("---")
        st.header("Targets")
        tp1_long = st.number_input("Long TP1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5, help="Percent for TP1 on longs")
        tp2_long = st.number_input("Long TP2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
        tp1_short = st.number_input("Short TP1 %", min_value=0.5, max_value=20.0, value=4.0, step=0.5)
        tp2_short = st.number_input("Short TP2 %", min_value=0.5, max_value=50.0, value=8.0, step=0.5)
        atr_mult = st.number_input("SL ATR multiple", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
        auto_cutoff = st.checkbox(
            "Auto-optimize cutoff on selected range",
            value=False,
            help="Search cutoffs and pick the one that maximizes Sharpe (tie-breaker: win rate).",
        )
        w_override = st.checkbox(
            "Enable weight tuning (quick)", 
            value=False, 
            help="Search small adjustments to feature weights for the selected range."
        )
        use_calibration = st.checkbox(
            "Use signal calibration",
            value=True,
            help="Apply expected value filter to signals based on historical success probability.",
        )
        use_adaptive_timeframe = st.checkbox(
            "Use adaptive timeframe",
            value=False,
            help="Dynamically select optimal timeframe based on volatility.",
        )
        use_cross_asset = st.checkbox(
            "Apply cross-asset bias",
            value=True,
            help="Incorporate lead-lag relationships between assets.",
        )
        tv_text = st.text_area(
            "Paste OCR text from TradingView panels (images 1 & 2)",
            value="",
            help=(
                "Optional: paste text extracted from TV technical summary/MA/oscillator panels."
                " If Tesseract is installed locally, you can OCR images and paste here."
            ),
        )
        uploads = st.file_uploader(
            "Or upload screenshots (Technical Summary, Oscillators, Moving Averages, Pivots)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help=(
                "If Tesseract is available, we will OCR each image and compute a weighted bias"
                " from Buy/Sell/Neutral counts."
            ),
        )
        st.markdown("— or —")
        chart_imgs = st.file_uploader(
            "Upload candlestick chart screenshot(s) (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Charts are saved to the report and shown alongside our computed volume/RSI comparison.",
        )
        st.caption("We compute volume and RSI/RSI-SMA directly from data; charts are for visual context.")
        st.markdown("— or —")
        wt_text = st.text_area(
            "Paste WebTrend lines/trend (optional)",
            value="",
            help="From the TV data window: provide comma-separated WebTrend Plot values and an optional Trend value like: lines=171.93,191.74,171.93; trend=165.62",
        )
        liq_imgs = st.file_uploader(
            "Upload Coinglass liquidation heatmap image(s) (optional)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="If API fails, we can approximate clusters from one or more images and compare/merge clusters.",
        )
        colp1, colp2 = st.columns(2)
        autodetect_scale = st.checkbox("Auto-detect heatmap scale (OCR axis)", value=HAS_TESS)
        with colp1:
            liq_top = st.number_input("Heatmap top price", min_value=0.0, value=120000.0 if asset == "BTC" else 230.0, step=1.0, disabled=autodetect_scale)
        with colp2:
            liq_bottom = st.number_input("Heatmap bottom price", min_value=0.0, value=100000.0 if asset == "BTC" else 180.0, step=1.0, disabled=autodetect_scale)
        
        run_button = st.button("Run Analysis")
        
        with st.expander("What do these controls do?", expanded=False):
            st.markdown(
                "- **Asset**: Selects market and asset-specific defaults.\n"
                "- **Window Size**: Past candles required before generating signals. Larger = smoother/less trades.\n"
                "- **Lookahead Candles**: Backtest horizon to check if TP/SL hit; affects metrics only.\n"
                "- **Score Cutoff (abs)**: Minimum |score| to accept a trade. Higher = fewer, higher-conviction.\n"
                "- **Use Coinglass Heatmap**: Incorporates liquidation clusters into TP/SL when the API responds.\n"
                "- **Data Range**: Amount of history to analyze and display.\n"
                "- **TV OCR text**: Optional bias derived from TV's Buy/Sell counts (images 1 & 2).\n"
                "- **Use signal calibration**: Apply expected value filter based on historical success probability.\n"
                "- **Use adaptive timeframe**: Dynamically select optimal timeframe based on volatility.\n"
                "- **Apply cross-asset bias**: Incorporate lead-lag relationships between assets.\n"
            )

    if not run_button:
        st.info("Configure in the sidebar and click Run Analysis.")
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
    trading_system = EnhancedTradingSystem(assets=["BTC", "SOL", "BONK"])

    # Fetch data for all assets
    st.write(f"Fetching data for all assets [{start} → {end}]…")
    
    # Use adaptive timeframe if selected
    timeframe = "4h"  # Default
    if use_adaptive_timeframe:
        # Fetch multi-timeframe data
        multi_tf_data = trading_system.fetch_multi_timeframe_data(start, end)
        optimal_timeframes = trading_system.select_optimal_timeframes(multi_tf_data)
        timeframe = optimal_timeframes.get(asset, "4h")
        st.info(f"Selected optimal timeframe for {asset}: {timeframe}")
    
    # Fetch data for all assets
    data = trading_system.fetch_data(start, end, timeframe=timeframe)
    
    if asset not in data or data[asset].empty:
        st.error(f"No data fetched for {asset}. Try another range or asset.")
        return

    # Optional liquidation clusters
    liq = None
    if use_coinglass and HAS_COINGLASS:
        with st.spinner("Querying Coinglass heatmap…"):
            try:
                heatmap = get_liquidation_heatmap(asset)
                if heatmap:
                    liq = extract_liquidation_clusters(heatmap, data[asset]["close"].values)
            except Exception as e:
                st.warning(f"Coinglass unavailable: {e}")
    
    # Fallback: extract clusters from uploaded heatmap image
    if liq is None and liq_imgs:
        with st.spinner("Extracting clusters from uploaded heatmap image(s)…"):
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
                st.info(f"Using {len(merged)} image-derived clusters (merged across {len(per_image)} image(s)).")
                
                # Show comparison table
                try:
                    rows = []
                    for entry in per_image:
                        for p, s in entry["clusters"]:
                            rows.append({"file": entry["file"], "price": round(p, 2), "strength": round(s, 3)})
                    if rows:
                        st.subheader("Heatmap clusters (per image)")
                        st.table(pd.DataFrame(rows))
                except Exception:
                    pass

    # Calculate WebTrend
    with st.spinner("Calculating WebTrend indicators…"):
        webtrend_data = trading_system.calculate_webtrend(data)

    # Analyze cross-asset relationships if selected
    cross_asset_biases = {}
    if use_cross_asset:
        with st.spinner("Analyzing cross-asset relationships…"):
            cross_asset_results = trading_system.analyze_cross_asset_relationships(data)
            cross_asset_biases = trading_system.get_cross_asset_biases(data)

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
    with st.spinner("Running backtest…"):
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
    with st.spinner("Generating signals…"):
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

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price & Signals")
        st.pyplot(plot_price_signals(results_df.tail(500), f"{asset} – Last 500 candles"))
    with col2:
        st.subheader("Cumulative Returns")
        st.pyplot(plot_cumulative(results_df.tail(500), f"{asset} – Strategy vs Buy&Hold"))

    # Additional analytics plots
    st.subheader("Volume and RSI Comparison")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.pyplot(plot_volume(results_df.tail(500), f"{asset} – Volume (last 500)"))
    with vc2:
        st.pyplot(plot_rsi_vs_sma(results_df.tail(500), f"{asset} – RSI vs RSI SMA (last 500)"))

    # Current signal snapshot
    st.subheader("Current Signal Snapshot")
    st.write({
        "asset": asset,
        "price": float(results_df["close"].iloc[-1]) if asset != "BONK" else float(results_df["close"].iloc[-1]),
        "signal": signal_data.get("signal"),
        "score": float(signal_data.get("final_score", 0.0)),
        "calibrated_signal": signal_data.get("calibrated_signal") if use_calibration and "calibrated_signal" in signal_data else None,
        "success_probability": float(signal_data.get("success_probability", 0.0)) if use_calibration and "success_probability" in signal_data else None,
        "expected_value": float(signal_data.get("expected_value", 0.0)) if use_calibration and "expected_value" in signal_data else None,
        "TP1": float(signal_data.get("targets", {}).get("TP1", 0.0)),
        "TP2": float(signal_data.get("targets", {}).get("TP2", 0.0)),
        "SL": float(signal_data.get("targets", {}).get("SL", 0.0)),
        "RSI": float(results_df.get("rsi_raw", pd.Series([0])).iloc[-1]),
    })

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

    st.subheader("Position Advice")
    st.markdown(
        f"- **Direction**: {direction}  |  **Recommended leverage**: {lev}x  |  **ATR%**: {atr_pct:.2%}\n"
        f"- **TP1/TP2/SL**: {signal_data['targets']['TP1']}, {signal_data['targets']['TP2']}, {signal_data['targets']['SL']}\n"
        f"- **Rationale**: RSI {signal_data.get('components', {}).get('rsi_score', 0):+.2f}, Volume {signal_data.get('components', {}).get('volume_score', 0):+.2f}, "
        f"Divergence {signal_data.get('components', {}).get('divergence', 0):+.2f}, WebTrend {signal_data.get('components', {}).get('webtrend_score', 0):+.2f}, "
        f"Liquidity {signal_data.get('components', {}).get('liquidation_score', 0):+.2f}."
    )
    
    # Display biases
    bias_texts = []
    if tv_bias is not None:
        bias_texts.append(f"TV {tv_bias:+.2f}")
    if use_cross_asset and asset in cross_asset_biases and abs(cross_asset_biases[asset].get("bias", 0)) > 0.01:
        cross_bias = cross_asset_biases[
