# Enhanced Backtester Implementation
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from updated_rsi_volume_model import EnhancedRsiVolumePredictor

class EnhancedBacktester:
    """
    Walk-forward backtester for the Enhanced RSI + Volume Predictive Model.
    Expected columns: open, high, low, close, volume, rsi_raw, rsi_sma
    Optional      : webtrend_status, ema20, ema50, ema100
    """

    def __init__(self, data_df: pd.DataFrame, asset_type: str = "BTC",
                 static_liquidation_data: Optional[Dict[str, Any]] = None,
                 regime_filter: bool = True,
                 fee_bps: float = 5.0,
                 adaptive_cutoff: bool = True,
                 min_agree_features: int = 0,
                 weight_overrides: Optional[Dict[str, float]] = None) -> None:
        self.asset_type = asset_type
        self.data = data_df.copy()
        self.static_liquidation_data = static_liquidation_data
        self.regime_filter = regime_filter
        self.fee_rate = float(fee_bps) / 10000.0
        self.adaptive_cutoff = adaptive_cutoff
        self.min_agree_features = int(min_agree_features or 0)
        self.weight_overrides = weight_overrides or None
        self.results: Optional[pd.DataFrame] = None

    def _infer_webtrend(self, window_df: pd.DataFrame, price_last: float) -> bool:
        ema20 = window_df["ema20"].iloc[-1] if "ema20" in window_df.columns else np.nan
        ema50 = window_df["ema50"].iloc[-1] if "ema50" in window_df.columns else np.nan
        ema100 = window_df["ema100"].iloc[-1] if "ema100" in window_df.columns else np.nan
        if np.isnan(ema20) or np.isnan(ema50) or np.isnan(ema100):
            return False
        return (price_last > ema20) and (ema20 > ema50) and (ema50 > ema100)

    def _window_to_predictor(self, window_df: pd.DataFrame) -> EnhancedRsiVolumePredictor:
        rsi_sma = window_df["rsi_sma"].values
        rsi_raw = window_df["rsi_raw"].values
        volume = window_df["volume"].values
        price = window_df["close"].values

        if "webtrend_status" in window_df.columns:
            webtrend_status = bool(window_df["webtrend_status"].iloc[-1])
        else:
            webtrend_status = self._infer_webtrend(window_df, price[-1])

        liquidation_data = self.static_liquidation_data or None

        return EnhancedRsiVolumePredictor(
            rsi_sma_series=rsi_sma,
            rsi_raw_series=rsi_raw,
            volume_series=volume,
            price_series=price,
            liquidation_data=liquidation_data,
            webtrend_status=webtrend_status,
            asset_type=self.asset_type,
            weight_overrides=self.weight_overrides,
        )

    def run_backtest(self, window_size: int = 50, lookahead_candles: int = 20,
                     min_abs_score: float = 0.0) -> pd.DataFrame:
        df = self.data.copy()
        for col in ["score", "signal", "TP1", "TP2", "SL"]:
            if col not in df.columns:
                df[col] = np.nan

        signals: list[str] = [None] * len(df)
        def _atr_pct(win: pd.DataFrame, periods: int = 14) -> float:
            if not {"high", "low", "close"}.issubset(win.columns):
                return 0.0
            hi = win["high"].astype(float)
            lo = win["low"].astype(float)
            cl = win["close"].astype(float)
            prev = cl.shift(1)
            tr = pd.concat([(hi - lo), (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
            atr = tr.rolling(min(14, len(tr))).mean().iloc[-1]
            price = float(cl.iloc[-1]) if len(cl) else 0.0
            return float(atr / price) if price else 0.0

        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size : i + 1]
            predictor = self._window_to_predictor(window)
            analysis = predictor.get_full_analysis()
            df.iloc[i, df.columns.get_loc("score")] = analysis["final_score"]
            df.iloc[i, df.columns.get_loc("TP1")] = analysis["targets"]["TP1"]
            df.iloc[i, df.columns.get_loc("TP2")] = analysis["targets"]["TP2"]
            df.iloc[i, df.columns.get_loc("SL")] = analysis["targets"]["SL"]
            # apply score cutoff if requested
            sig = analysis["signal"]
            # adaptive cutoff: raise threshold in high volatility (ATR%)
            dynamic_cutoff = min_abs_score
            if self.adaptive_cutoff:
                atrp = _atr_pct(window)
                if atrp > 0.05:
                    dynamic_cutoff += 0.10
                elif atrp > 0.03:
                    dynamic_cutoff += 0.05
            if dynamic_cutoff > 0 and abs(analysis["final_score"]) < dynamic_cutoff:
                sig = "NEUTRAL"
            # regime filter: only take longs in uptrend, shorts in downtrend
            if self.regime_filter and sig != "NEUTRAL":
                up = self._infer_webtrend(window, window["close"].iloc[-1])
                if sig in ("BUY", "STRONG BUY") and not up:
                    sig = "NEUTRAL"
                if sig in ("SELL", "STRONG SELL") and up:
                    sig = "NEUTRAL"
            # conflict filter: require N agreeing features
            if self.min_agree_features and sig != "NEUTRAL":
                intended = 1 if sig in ("BUY", "STRONG BUY") else -1
                comp = analysis.get("components", {})
                keys = ["rsi_score", "webtrend_score", "liquidation_score", "divergence"]
                agree = 0
                for k in keys:
                    v = comp.get(k, 0.0)
                    try:
                        if float(v) * intended > 0:
                            agree += 1
                    except Exception:
                        pass
                if agree < self.min_agree_features:
                    sig = "NEUTRAL"
            signals[i] = sig

        df["signal"] = signals
        signal_map = {"STRONG BUY": 2, "BUY": 1, "NEUTRAL": 0, "SELL": -1, "STRONG SELL": -2}
        df["signal_value"] = df["signal"].map(signal_map)

        hit_tp1 = np.zeros(len(df), dtype=bool)
        hit_tp2 = np.zeros(len(df), dtype=bool)
        hit_sl = np.zeros(len(df), dtype=bool)

        for i in range(window_size, len(df) - 1):
            sig = df["signal"].iloc[i]
            if not sig or sig == "NEUTRAL":
                continue
            tp1, tp2, sl = df["TP1"].iloc[i], df["TP2"].iloc[i], df["SL"].iloc[i]
            hi = df["high"].iloc[i + 1 : i + 1 + lookahead_candles]
            lo = df["low"].iloc[i + 1 : i + 1 + lookahead_candles]
            if sig in ("BUY", "STRONG BUY"):
                hit_tp1[i] = (hi >= tp1).any()
                hit_tp2[i] = (hi >= tp2).any()
                hit_sl[i] = (lo <= sl).any()
            else:
                hit_tp1[i] = (lo <= tp1).any()
                hit_tp2[i] = (lo <= tp2).any()
                hit_sl[i] = (hi >= sl).any()

        df["hit_tp1"] = hit_tp1
        df["hit_tp2"] = hit_tp2
        df["hit_sl"] = hit_sl

        self.results = df
        return df

    def performance(self) -> Dict[str, float | int]:
        if self.results is None:
            return {"error": "Backtest not run"}
        r = self.results.copy()
        r["returns"] = r["close"].pct_change()
        r["position"] = r["signal_value"].shift(1).fillna(0).clip(-1, 1)
        r["strategy_returns"] = r["position"] * r["returns"]
        # transaction costs at position changes
        changes = (r["position"] != r["position"].shift(1)).fillna(False)
        cost = changes.astype(float) * self.fee_rate
        r["strategy_returns"] = r["strategy_returns"] - cost
        r["cum_bh"] = (1 + r["returns"]).cumprod() - 1
        r["cum_strat"] = (1 + r["strategy_returns"]).cumprod() - 1
        # Identify trade entries (when position changes to non-zero)
        entries_mask = (r["position"] != r["position"].shift(1)) & (r["position"] != 0)
        entry_idxs = list(r.index[entries_mask])
        total_trades = len(entry_idxs)

        # Compute per-trade PnL by segmenting until next change
        wins = 0
        if total_trades:
            for j, start_idx in enumerate(entry_idxs):
                # end of segment is next entry - 1, or last row
                end_idx = entry_idxs[j + 1] if j + 1 < total_trades else r.index[-1]
                seg = r.loc[start_idx:end_idx]
                seg_ret = (1 + seg["strategy_returns"]).prod() - 1
                if seg_ret > 0:
                    wins += 1
        win_rate = (wins / total_trades) if total_trades else 0.0
        max_dd = (r["cum_strat"].cummax() - r["cum_strat"]).max()
        sharpe = (
            r["strategy_returns"].mean() / r["strategy_returns"].std() * np.sqrt(365)
            if r["strategy_returns"].std() not in (0, np.nan) else 0.0
        )
        return {
            "total_return": float(r["cum_strat"].iloc[-1]) if len(r) else 0.0,
            "buy_hold_return": float(r["cum_bh"].iloc[-1]) if len(r) else 0.0,
            "sharpe": float(sharpe) if not np.isnan(sharpe) else 0.0,
            "max_drawdown": float(max_dd) if pd.notna(max_dd) else 0.0,
            "win_rate": float(win_rate),
            "total_trades": int(total_trades),
            "tp1_accuracy": float(r["hit_tp1"].mean()) if len(r) else 0.0,
            "tp2_accuracy": float(r["hit_tp2"].mean()) if len(r) else 0.0,
            "sl_hit_rate": float(r["hit_sl"].mean()) if len(r) else 0.0,
        }