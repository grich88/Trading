import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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

try:
    from coinglass_api import get_liquidation_heatmap, extract_liquidation_clusters
    HAS_COINGLASS = True
except Exception:
    HAS_COINGLASS = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@st.cache_data(show_spinner=False)
def get_history(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return calculate_indicators(df)


def compute_cross_asset_correlations(btc: pd.DataFrame, sol: pd.DataFrame, bonk: pd.DataFrame) -> dict:
    def prep(df: pd.DataFrame) -> pd.Series:
        x = df["close"].pct_change().dropna()
        return x
    res: dict = {}
    series = {"BTC": prep(btc), "SOL": prep(sol), "BONK": prep(bonk)}
    # align indexes
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
        # lead/lag: BTC leads by 1-2 candles
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
    if not clusters:
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
    try:
        dists = [abs(price - p)/max(1e-9, price) for p, _ in clusters]
        return float(min(dists))
    except Exception:
        return None


def plot_price_signals(df: pd.DataFrame, title: str) -> plt.Figure:
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
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(df.index, df["volume"], width=0.02, color="#77aaff")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    return fig


def plot_rsi_vs_sma(df: pd.DataFrame, title: str) -> plt.Figure:
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

def _infer_webtrend_from_df(df: pd.DataFrame) -> bool:
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


def _compute_atr_percent(df: pd.DataFrame, periods: int = 14) -> float:
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


def main():
    st.set_page_config(page_title="Trading Model UI", layout="wide")
    st.title("RSI + Volume + Features Model – Local UI")

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
            0,
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
        w_override = st.checkbox("Enable weight tuning (quick)", value=False, help="Search small adjustments to feature weights for the selected range.")
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
                "- **TV OCR text**: Optional bias derived from TV’s Buy/Sell counts (images 1 & 2)."
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

    st.write(f"Fetching data for {symbol} [{start} → {end}]…")
    df = get_history(symbol, start, end)
    # Also fetch BTC/SOL/BONK for cross-asset panel regardless of chosen asset
    df_btc = get_history("BTC/USDT", start, end)
    df_sol = get_history("SOL/USDT", start, end)
    df_bonk = get_history("BONK/USDT", start, end)
    if df.empty:
        st.error("No data fetched. Try another range or asset.")
        return

    # Optional liquidation clusters
    liq = None
    if use_coinglass and HAS_COINGLASS:
        with st.spinner("Querying Coinglass heatmap…"):
            try:
                heatmap = get_liquidation_heatmap(asset)
                if heatmap:
                    liq = extract_liquidation_clusters(heatmap, df["close"].values)
            except Exception as e:
                st.warning(f"Coinglass unavailable: {e}")
    # Fallback: extract clusters from uploaded heatmap image
    if liq is None and liq_imgs:
        with st.spinner("Extracting clusters from uploaded heatmap image(s)…"):
            per_image = []
            for f in liq_imgs:
                data = f.read()
                top_val, bot_val = float(liq_top), float(liq_bottom)
                if autodetect_scale:
                    guess = auto_detect_heatmap_scale(data)
                    if guess:
                        top_val, bot_val = guess
                clusters = extract_liq_clusters_from_image(data, price_top=top_val, price_bottom=bot_val)
                if clusters:
                    per_image.append({"file": f.name, "top": top_val, "bottom": bot_val, "clusters": clusters})
            if per_image:
                # Merge clusters across images (price proximity within 0.5%)
                merged: list[tuple[float, float]] = []
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
                    import pandas as _pd  # lazy
                    rows = []
                    for entry in per_image:
                        for p, s in entry["clusters"]:
                            rows.append({"file": entry["file"], "price": round(p, 2), "strength": round(s, 3)})
                    if rows:
                        st.subheader("Heatmap clusters (per image)")
                        st.table(_pd.DataFrame(rows))
                except Exception:
                    pass

    # Cross-asset bias pre-computation (BTC influence)
    cross_bias = 0.0
    cutoff_adj = cutoff
    if asset in ("SOL", "BONK") and liq is not None and not df_btc.empty:
        tmp_bt = EnhancedBacktester(df_btc, asset_type="BTC", static_liquidation_data=liq)
        _tmp = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
        btc_sig = _tmp["signal"].iloc[-1]
        btc_price = float(_tmp["close"].iloc[-1])
        dist = nearest_cluster_distance(btc_price, liq.get("clusters"))
        if btc_sig in ("STRONG BUY", "STRONG SELL") and (dist is not None and dist < 0.02):
            # Add a small score bias scaled by historical lead R^2
            beta = _lead_betas(df_btc, df_sol if asset == "SOL" else df_bonk)
            scale = min(0.15, 0.10 + 0.05 * max(0.0, beta.get('r2', 0.0)))
            cross_bias = scale if btc_sig == "STRONG BUY" else -scale
            cutoff_adj = max(0.0, cutoff - 0.05)

    # Backtest with parameters
    bt = EnhancedBacktester(
        df,
        asset_type=asset,
        static_liquidation_data=liq,
        regime_filter=regime_filter,
        fee_bps=float(fee_bps),
        adaptive_cutoff=adaptive_cutoff,
        min_agree_features=conflict_min,
    )
    # Optionally auto-optimize cutoff
    if auto_cutoff:
        with st.spinner("Optimizing cutoff…"):
            best = {"cutoff": cutoff, "sharpe": -1e9, "win": 0.0}
            for c in [round(x, 3) for x in [i/100 for i in range(20, 66, 5)]]:
                tmp_bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq,
                                            regime_filter=regime_filter, fee_bps=float(fee_bps), adaptive_cutoff=adaptive_cutoff,
                                            min_agree_features=conflict_min)
                _ = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=c)
                m = tmp_bt.performance()
                sh = float(m.get("sharpe", 0.0)); wr = float(m.get("win_rate", 0.0))
                if sh > best["sharpe"] or (abs(sh - best["sharpe"]) < 1e-6 and wr > best["win"]):
                    best = {"cutoff": c, "sharpe": sh, "win": wr}
            cutoff = best["cutoff"]
            st.info(f"Auto-optimized cutoff → {cutoff:.3f} (Sharpe {best['sharpe']:.2f}, Win {best['win']:.2%})")
    # Optional simple weight search (grid over a few deltas)
    weight_overrides = None
    if w_override:
        with st.spinner("Tuning feature weights…"):
            best = {"w": None, "sh": -1e9}
            deltas = [-0.05, 0.0, 0.05]
            base = {"w_rsi":0.30, "w_volume":0.25, "w_divergence":0.15, "w_liquidation":0.10, "w_webtrend":0.05, "w_features":0.15, "w_sentiment":0.05}
            import itertools
            for dr, dv, dd, dl, dw, dfeat in itertools.product(deltas, repeat=6):
                w = base.copy()
                w["w_rsi"] += dr; w["w_volume"] += dv; w["w_divergence"] += dd; w["w_liquidation"] += dl; w["w_webtrend"] += dw; w["w_features"] += dfeat
                tmp_bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq, regime_filter=regime_filter, fee_bps=float(fee_bps), adaptive_cutoff=adaptive_cutoff, min_agree_features=conflict_min, weight_overrides=w)
                _ = tmp_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff_adj)
                m = tmp_bt.performance()
                if m.get("sharpe", 0.0) > best["sh"]:
                    best = {"w": w, "sh": m["sharpe"]}
            weight_overrides = best["w"]
            if weight_overrides:
                st.info(f"Weight tuning selected (Sharpe {best['sh']:.2f}).")
        bt = EnhancedBacktester(df, asset_type=asset, static_liquidation_data=liq, regime_filter=regime_filter, fee_bps=float(fee_bps), adaptive_cutoff=adaptive_cutoff, min_agree_features=conflict_min, weight_overrides=weight_overrides)
    with st.spinner("Running backtest…"):
        results = bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff_adj)
        metrics = bt.performance()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price & Signals")
        st.pyplot(plot_price_signals(results.tail(500), f"{asset} – Last 500 candles"))
    with col2:
        st.subheader("Cumulative Returns")
        st.pyplot(plot_cumulative(results.tail(500), f"{asset} – Strategy vs Buy&Hold"))

    # Additional analytics plots
    st.subheader("Volume and RSI Comparison")
    vc1, vc2 = st.columns(2)
    with vc1:
        st.pyplot(plot_volume(results.tail(500), f"{asset} – Volume (last 500)"))
    with vc2:
        st.pyplot(plot_rsi_vs_sma(results.tail(500), f"{asset} – RSI vs RSI SMA (last 500)"))

    # Current signal snapshot
    last = results.iloc[-1]
    st.subheader("Current Signal Snapshot")
    st.write({
        "asset": asset,
        "price": float(last["close"]) if asset != "BONK" else float(last["close"]),
        "signal": last.get("signal"),
        "score": float(last.get("score", 0.0)),
        "TP1": float(last.get("TP1", 0.0)),
        "TP2": float(last.get("TP2", 0.0)),
        "SL": float(last.get("SL", 0.0)),
        "RSI": float(last.get("rsi_raw", 0.0)),
    })

    # Derive rationale from latest window using the predictor
    last_window = df.iloc[-window:]
    webtrend_status = bool(last_window.get("webtrend_status", pd.Series([_infer_webtrend_from_df(df)])).iloc[-1])
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

    # Optional TradingView bias from OCR text
    tv_bias = None
    tv_details = None
    sources = []
    if tv_text and isinstance(tv_text, str) and tv_text.strip():
        sources.append((tv_text, "text"))
    if uploads:
        texts = []
        for f in uploads:
            data = f.read()
            text = ocr_image_to_text(data) if HAS_TESS else None
            if text:
                texts.append((text, f.name))
        if texts:
            agg = compute_weighted_bias_from_texts(texts)
            tv_bias = float(agg.get("bias", 0.0))
            tv_details = agg
    if tv_bias is None and sources:
        tv_parse = compute_tv_score_from_text(sources[0][0])
        tv_bias = float(tv_parse["score"]) if isinstance(tv_parse, dict) else None

    atr_pct = _compute_atr_percent(df)
    signal = full.get("signal")
    score = float(full.get("final_score", 0.0))
    # Combine TV OCR bias with cross-asset bias
    combined_bias = (tv_bias if tv_bias is not None else 0.0) + cross_bias
    adjusted_score = predictor.apply_external_biases(combined_bias) if (tv_bias is not None or cross_bias != 0.0) else score
    direction = "Long" if signal in ("BUY", "STRONG BUY") else ("Short" if signal in ("SELL", "STRONG SELL") else "Flat")
    lev = _suggest_leverage(asset, signal, adjusted_score, atr_pct)

    st.subheader("Position Advice")
    st.markdown(
        f"- **Direction**: {direction}  |  **Recommended leverage**: {lev}x  |  **ATR%**: {atr_pct:.2%}\n"
        f"- **TP1/TP2/SL**: {last.get('TP1')}, {last.get('TP2')}, {last.get('SL')}\n"
        f"- **Rationale**: RSI {comp.get('rsi_score', 0):+.2f}, Volume {comp.get('volume_score', 0):+.2f}, "
        f"Divergence {comp.get('divergence', 0):+.2f}, WebTrend {comp.get('webtrend_score', 0):+.2f}, "
        f"Liquidity {comp.get('liquidation_score', 0):+.2f}."
    )
    if tv_bias is not None or cross_bias != 0.0:
        btxt = []
        if tv_bias is not None:
            btxt.append(f"TV {tv_bias:+.2f}")
        if cross_bias != 0.0:
            btxt.append(f"Cross-asset {cross_bias:+.2f}")
        st.caption(f"Bias applied: {' + '.join(btxt)}. Adjusted score: {adjusted_score:+.3f} (blended). Cutoff used: {cutoff_adj:.3f}.")
        if tv_details and isinstance(tv_details, dict):
            st.markdown("TV panel contributions:")
            st.table(pd.DataFrame(tv_details.get("panels", [])))
    st.caption(
        "Signals can be long or short. Backtests use a signed position (+1 long, -1 short). "
        "Leverage suggestion is conservative, capped per asset and reduced in high volatility (ATR%)."
    )

    st.subheader("Performance Metrics")
    st.table(pd.DataFrame([metrics]))
    st.markdown(
        "- **total_return**: Strategy cumulative return using long/short positions.\n"
        "- **buy_hold_return**: Passive return from holding the asset.\n"
        "- **sharpe**: Risk-adjusted return (daily approx).\n"
        "- **max_drawdown**: Worst strategy peak-to-trough decline.\n"
        "- **win_rate**: Fraction of profitable trade segments.\n"
        "- **tp1/tp2_accuracy**: Share of signals that hit TP1/TP2 within the lookahead.\n"
        "- **sl_hit_rate**: Share of signals that hit stop within the lookahead."
    )

    # Cross-asset correlation / lead-lag panel
    st.subheader("Cross-Asset Correlations & Lead/Lag (4h)")
    cors = compute_cross_asset_correlations(df_btc, df_sol, df_bonk)
    try:
        st.table(pd.DataFrame(cors["table"]))
    except Exception:
        st.write(cors["table"])  # fallback
    asym = cors.get("asymmetry", {})
    st.caption(f"Recent 3-candle move sums: BTC {asym.get('BTC', 0):.3f}, SOL {asym.get('SOL',0):.3f}, BONK {asym.get('BONK',0):.3f}")

    # Cross-asset bias: tilt SOL/BONK if BTC signal strong and near cluster
    cross_bias_note = None
    # Historical lead betas (BTC → SOL/BONK) inform present bias magnitude
    bet_sol = _lead_betas(df_btc, df_sol)
    bet_bonk = _lead_betas(df_btc, df_bonk)
    st.caption(f"Lead betas BTC→SOL: b1={bet_sol['b1']:.2f}, b2={bet_sol['b2']:.2f}, R²={bet_sol['r2']:.2f}; BTC→BONK: b1={bet_bonk['b1']:.2f}, b2={bet_bonk['b2']:.2f}, R²={bet_bonk['r2']:.2f}")
    if asset in ("SOL", "BONK") and liq is not None:
        btc_bt = EnhancedBacktester(df_btc, asset_type="BTC", static_liquidation_data=liq)
        tmp = btc_bt.run_backtest(window_size=window, lookahead_candles=lookahead, min_abs_score=cutoff)
        btc_sig = tmp["signal"].iloc[-1]
        btc_price = float(tmp["close"].iloc[-1])
        dist = nearest_cluster_distance(btc_price, liq.get("clusters"))
        if btc_sig in ("STRONG BUY", "STRONG SELL") and (dist is not None and dist < 0.02):
            beta = bet_sol if asset == "SOL" else bet_bonk
            mag = min(0.15, 0.10 + 0.05 * max(0.0, beta.get('r2', 0.0)))
            direction = 'upside' if btc_sig.startswith('STRONG BUY') else 'downside'
            cross_bias_note = f"BTC {btc_sig} near heatmap cluster (distance {dist:.2%}); historical lead strength R²={beta.get('r2',0.0):.2f}. Favor {asset} {direction}; bias magnitude ≈ {mag:.2f}."
            st.info(cross_bias_note)

    with st.expander("How to read this dashboard", expanded=False):
        st.markdown(
            "- **Score** in [-1,1] blends RSI/Volume/Divergence/MA/WebTrend/Liquidity; the cutoff filters weak setups.\n"
            "- **Signals**: BUY/SELL when score crosses tiered thresholds; NEUTRAL when below cutoff or regime filter blocks.\n"
            "- **TP/SL**: ATR-based, nudged toward liquidation clusters and pivots.\n"
            "- **Regime filter**: Longs only in EMA uptrend; shorts only in downtrend.\n"
            "- **Adaptive cutoff**: Raises threshold when ATR% is high to avoid whipsaws.\n"
            "- **Fees**: Applied on position changes using the bps set in the sidebar.\n"
        )

    # Advisory section
    def build_advisory() -> list[str]:
        adv: list[str] = []
        # Entry sensitivity
        trades = metrics.get("total_trades", 0) or 0
        sharpe_v = metrics.get("sharpe", 0.0) or 0.0
        win_v = metrics.get("win_rate", 0.0) or 0.0
        if trades < 30:
            adv.append("Few trades detected. Consider lowering cutoff by 0.05 or reducing window by 5–10 to increase sensitivity.")
        if win_v < 0.5 and sharpe_v < 0:
            adv.append("Negative edge: raise cutoff by 0.05, enable regime filter, or increase window to smooth noise.")
        if atr_pct > 0.04:
            adv.append("High volatility: use conservative leverage (≤1–2x), widen SL slightly, and keep adaptive cutoff on.")
        if tv_bias is None:
            adv.append("Upload TradingView panels or paste OCR text to add external bias and context.")
        if not (use_coinglass and HAS_COINGLASS and liq is not None):
            adv.append("Liquidation clusters not used. Enable Coinglass (when endpoint responds) to refine TP/SL levels.")
        # Target behavior
        tp1 = metrics.get("tp1_accuracy", 0.0) or 0.0
        if tp1 < 0.05:
            adv.append("TP1 hit rate is low within lookahead. Increase lookahead by 5 or move TP1 closer (e.g., 3–3.5% for longs).")
        # Direction advice
        if signal in ("BUY", "STRONG BUY") and adjusted_score < cutoff:
            adv.append("Score below cutoff—lower cutoff slightly or wait for a stronger impulse before entry.")
        if signal in ("SELL", "STRONG SELL") and adjusted_score > -cutoff:
            adv.append("Bearish score not strong enough—raise selectivity or wait for rejection near resistance/pivot.")
        return adv

    advice = build_advisory()
    st.subheader("Advisory")
    if advice:
        st.markdown("\n".join([f"- {a}" for a in advice]))
    else:
        st.markdown("- No changes recommended under current settings.")

    # Export images and data
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("data", "ui_exports", ts, asset)
    ensure_dir(out_dir)
    results_path = os.path.join(out_dir, f"{asset}_results.csv")
    results.to_csv(results_path)
    st.success(f"Saved per-candle results to {results_path}")

    # Save images
    price_fig = plot_price_signals(results.tail(500), f"{asset} – Last 500 candles")
    perf_fig = plot_cumulative(results.tail(500), f"{asset} – Strategy vs Buy&Hold")
    price_fig.savefig(os.path.join(out_dir, f"{asset}_price_signals.png"), dpi=150, bbox_inches="tight")
    perf_fig.savefig(os.path.join(out_dir, f"{asset}_cumulative.png"), dpi=150, bbox_inches="tight")
    st.info("Images exported in data/ui_exports/")

    # Compose markdown report
    def _fmt_pct(x: float) -> str:
        try:
            return f"{float(x)*100:.2f}%"
        except Exception:
            return "-"

    metric_lines = [
        f"- total_return: {_fmt_pct(metrics.get('total_return', 0.0))}",
        f"- buy_hold_return: {_fmt_pct(metrics.get('buy_hold_return', 0.0))}",
        f"- sharpe: {metrics.get('sharpe', 0.0):.3f}",
        f"- max_drawdown: {_fmt_pct(metrics.get('max_drawdown', 0.0))}",
        f"- win_rate: {_fmt_pct(metrics.get('win_rate', 0.0))}",
        f"- total_trades: {metrics.get('total_trades', 0)}",
        f"- tp1_accuracy: {_fmt_pct(metrics.get('tp1_accuracy', 0.0))}",
        f"- tp2_accuracy: {_fmt_pct(metrics.get('tp2_accuracy', 0.0))}",
        f"- sl_hit_rate: {_fmt_pct(metrics.get('sl_hit_rate', 0.0))}",
    ]

    tv_section = ""
    if tv_bias is not None:
        tv_section += f"\n### TradingView OCR\n\n- Bias applied: {tv_bias:+.2f} (15% blend)\n\n"
        if tv_details and isinstance(tv_details, dict):
            tv_section += "| Panel | Weight | Score | Buy | Sell | Neutral |\n|---|---:|---:|---:|---:|---:|\n"
            for row in tv_details.get("panels", []):
                tv_section += f"| {row.get('panel','')} | {row.get('weight',0):.2f} | {row.get('score',0):+.2f} | {row.get('counts',{}).get('buy','')} | {row.get('counts',{}).get('sell','')} | {row.get('counts',{}).get('neutral','')} |\n"

    report = f"""
# Analysis Report – {asset}

## Parameters
- Window: {window}
- Lookahead candles: {lookahead}
- Score cutoff (abs): {cutoff:.3f}
- Data range: {start} → {end}
- Coinglass used: {'Yes' if (use_coinglass and HAS_COINGLASS and liq is not None) else 'No'}

## Current Snapshot
- Price: {float(last['close']):.2f}
- Signal: {signal}
- Score: {score:+.3f}{' → adjusted ' + str(adjusted_score) if tv_bias is not None else ''}
- RSI: {float(last.get('rsi_raw', 0.0)):.2f}
- ATR%: {_fmt_pct(atr_pct)}
- Targets: TP1 {last.get('TP1')}, TP2 {last.get('TP2')}, SL {last.get('SL')}
- Leverage suggestion: {lev}x

## Rationale
- RSI {comp.get('rsi_score', 0):+0.2f}
- Volume {comp.get('volume_score', 0):+0.2f}
- Divergence {comp.get('divergence', 0):+0.2f}
- WebTrend {comp.get('webtrend_score', 0):+0.2f}
- Liquidity {comp.get('liquidation_score', 0):+0.2f}
{tv_section}
## How to interpret
- Score combines features into [-1,1]; entries are taken when |score| ≥ cutoff (and pass the regime filter).\n
- Regime filter trades with the trend (EMA stack). If a signal conflicts with trend, it will be set to NEUTRAL.\n
- Adaptive cutoff raises selectivity in high ATR to avoid low‑quality trades.\n
- TP/SL are ATR‑based and adjusted toward liquidation clusters and pivots to respect likely reaction zones.\n
- Fees are charged at position changes using the configured bps.

## Tuning guidance
- Too few trades: decrease cutoff by ~0.05 or reduce window by 5–10.\n
- Too noisy/negative Sharpe: raise cutoff by ~0.05, increase window, and keep regime filter on.\n
- Low TP hits: increase lookahead or tighten TP1 slightly.\n
- High ATR: reduce leverage, keep adaptive cutoff on.

## Performance (backtest on selected range)
{os.linesep.join(metric_lines)}

## Assets
- Results CSV: {results_path}
- Price & Signals: {os.path.join(out_dir, f'{asset}_price_signals.png')}
- Cumulative: {os.path.join(out_dir, f'{asset}_cumulative.png')}

## Advisory
{os.linesep.join(['- ' + a for a in advice]) if advice else '- No changes recommended under current settings.'}
"""

    report_path = os.path.join(out_dir, f"{asset}_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    st.success(f"Report saved: {report_path}")
    st.download_button("Download report.md", data=report, file_name=f"{asset}_report.md")
    with st.expander("View report content"):
        st.markdown(report)

    # Save and display uploaded chart images
    if chart_imgs:
        st.subheader("Uploaded Charts")
        for fobj in chart_imgs:
            try:
                fname = os.path.join(out_dir, f"chart_{fobj.name}")
                with open(fname, "wb") as wf:
                    wf.write(fobj.read())
                st.image(fname, caption=fobj.name, use_column_width=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()


