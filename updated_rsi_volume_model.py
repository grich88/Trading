import numpy as np
import pandas as pd

class EnhancedRsiVolumePredictor:
    """
    Enhanced RSI + Volume Predictive Scoring Model for 4H Charts (BTC, SOL & BONK)
    
    This model integrates RSI streak analysis, volume streaks, divergence detection,
    liquidation heatmap data, and inter-asset correlation to output a predictive 
    score for short-term momentum.
    
    Score ranges from -1.0 (strong bearish) to +1.0 (strong bullish)
    """
    
    def __init__(self, rsi_sma_series, rsi_raw_series, volume_series, price_series, 
                 liquidation_data=None, webtrend_status=None, asset_type='SOL',
                 sentiment_series=None, event_series=None,
                 webtrend_lines: list[float] | None = None,
                 webtrend_trend_value: float | None = None,
                 tp_rules: dict | None = None,
                 weight_overrides: dict | None = None):
        """
        Initialize the predictor with time series data
        
        Parameters:
        -----------
        rsi_sma_series : array-like
            Series of RSI SMA values
        rsi_raw_series : array-like
            Series of raw RSI values
        volume_series : array-like
            Series of volume values
        price_series : array-like
            Series of price values
        liquidation_data : dict, optional
            Dictionary containing liquidation heatmap data
            Format: {'clusters': [(price_level, intensity)], 'cleared_zone': bool}
        webtrend_status : bool, optional
            True if WebTrend indicator shows uptrend, False otherwise
        asset_type : str
            Type of asset ('BTC', 'SOL', or 'BONK')
        """
        self.rsi_sma = rsi_sma_series
        self.rsi_raw = rsi_raw_series
        self.volume = volume_series
        self.price = price_series
        self.liquidation_data = liquidation_data or {'clusters': [], 'cleared_zone': False}
        self.webtrend_status = webtrend_status if webtrend_status is not None else False
        self.asset_type = asset_type
        # optional external signals
        self.sentiment_series = sentiment_series
        self.event_series = event_series
        # Optional explicit WebTrend inputs (from TV panel)
        self.webtrend_lines = list(webtrend_lines) if webtrend_lines else []
        self.webtrend_trend_value = webtrend_trend_value
        # Optional TP/SL configuration
        # tp_rules example: {"long": {"tp1_pct":0.04, "tp2_pct":0.08}, "short": {"tp1_pct":0.04, "tp2_pct":0.08}, "atr_mult":1.5}
        self.tp_rules = tp_rules or {}
        self.weight_overrides = weight_overrides or {}
        
        # Set volatility coefficient based on asset type
        if self.asset_type == 'BONK':
            self.volatility_coef = 1.2  # Higher volatility for BONK
        elif self.asset_type == 'SOL':
            self.volatility_coef = 1.0  # Base volatility for SOL
        else:  # BTC
            self.volatility_coef = 0.8  # Lower volatility for BTC
    
    def get_rsi_trend_score(self):
        """
        Calculate score based on RSI trend patterns
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        # Adjust RSI thresholds based on liquidation context
        if self.liquidation_data.get('cleared_zone', False):
            # Relaxed thresholds in cleared zones
            overbought_threshold = 68
            neutral_threshold = 55
            oversold_threshold = 32
        else:
            # Tighter thresholds in liquidation-heavy zones
            overbought_threshold = 65
            neutral_threshold = 50
            oversold_threshold = 35
        
        candles_above_neutral = sum(1 for x in self.rsi_sma[-50:] if x > neutral_threshold)
        candles_above_overbought = sum(1 for x in self.rsi_sma[-12:] if x > overbought_threshold)
        candles_below_oversold = sum(1 for x in self.rsi_sma[-10:] if x < oversold_threshold)
        
        # RSI crossing above neutral is bullish
        if self.rsi_sma[-1] > neutral_threshold and self.rsi_sma[-2] <= neutral_threshold:
            return 0.7
        
        # RSI staying above neutral is bullish
        if self.rsi_sma[-1] > neutral_threshold and candles_above_neutral >= 6:
            return min(0.8, 0.1 * candles_above_neutral)
        
        # RSI in overbought territory for extended period suggests caution
        if self.rsi_sma[-1] > overbought_threshold and candles_above_overbought >= 3:
            return max(0.0, 0.9 - 0.1 * (candles_above_overbought - 3))
        
        # RSI in oversold territory suggests potential reversal
        if self.rsi_sma[-1] < oversold_threshold and candles_below_oversold >= 3:
            return min(-0.1 * candles_below_oversold, -0.3)
        
        # RSI crossing below neutral is bearish
        if self.rsi_sma[-1] < neutral_threshold and self.rsi_sma[-2] >= neutral_threshold:
            return -0.5
        
        return 0
    
    def get_volume_trend_score(self):
        """
        Calculate score based on volume trend patterns
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        last_vol = self.volume[-10:]
        avg_early = sum(last_vol[:5]) / 5
        avg_late = sum(last_vol[5:]) / 5
        momentum = avg_late / avg_early if avg_early != 0 else 1
        
        green_streak = sum(1 for i in range(-5, 0) if self.price[i] > self.price[i - 1])
        red_streak = sum(1 for i in range(-5, 0) if self.price[i] < self.price[i - 1])
        
        # Volume spike with green candles is bullish
        if green_streak >= 3 and momentum > 1.05:
            return min(1.0, 0.25 * green_streak + 0.5 * (momentum - 1))
        
        # Volume spike with red candles is bearish
        elif red_streak >= 3 and momentum > 1.05:
            return max(-1.0, -0.25 * red_streak - 0.5 * (momentum - 1))
        
        # Volume spike at support/resistance is significant
        if momentum > 1.2 and self.is_near_key_level():
            return 0.6 if self.price[-1] > self.price[-2] else -0.6
        
        return 0
    
    def detect_divergence(self):
        """
        Detect divergence between price and indicators
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        price_highs = self.price[-10:]
        rsi_highs = self.rsi_raw[-10:]
        volume_highs = self.volume[-10:]
        
        # Bearish divergence: price makes new high but RSI doesn't
        price_div = price_highs[-1] > max(price_highs[:-1])
        rsi_div = rsi_highs[-1] < max(rsi_highs[:-1])
        volume_div = volume_highs[-1] < max(volume_highs[:-1])
        
        if price_div and (rsi_div or volume_div):
            return -0.5
        
        # Bullish divergence: price makes new low but RSI doesn't
        price_low = price_highs[-1] < min(price_highs[:-1])
        rsi_up = rsi_highs[-1] > min(rsi_highs[:-1])
        volume_fade = volume_highs[-1] < max(volume_highs[:-1])
        
        if price_low and (rsi_up or volume_fade):
            return 0.5
        
        return 0
    
    def get_liquidation_score(self):
        """
        Calculate score based on liquidation heatmap data
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        if not self.liquidation_data or not self.liquidation_data.get('clusters'):
            return 0
        
        clusters = self.liquidation_data['clusters']
        current_price = self.price[-1]
        
        # Find nearest clusters above and below current price
        # Handle both 2-tuple (p, i) and 3-tuple (p, i, metadata) formats
        clusters_above = []
        clusters_below = []
        
        for cluster in clusters:
            # Extract price and intensity, handling both formats
            if len(cluster) > 2:
                p, i, _ = cluster  # Unpack 3-tuple (price, intensity, metadata)
            else:
                p, i = cluster     # Unpack 2-tuple (price, intensity)
                
            # Sort into above/below current price
            if p > current_price:
                clusters_above.append((p, i))
            elif p < current_price:
                clusters_below.append((p, i))
        
        if not clusters_above and not clusters_below:
            return 0
        
        # Calculate distance to nearest clusters
        nearest_above = min(clusters_above, key=lambda x: x[0] - current_price) if clusters_above else (float('inf'), 0)
        nearest_below = max(clusters_below, key=lambda x: current_price - x[0]) if clusters_below else (0, 0)
        
        distance_above = nearest_above[0] - current_price if nearest_above[0] != float('inf') else float('inf')
        distance_below = current_price - nearest_below[0] if nearest_below[0] != 0 else float('inf')
        
        # Calculate intensity of nearest clusters
        intensity_above = nearest_above[1] if nearest_above[0] != float('inf') else 0
        intensity_below = nearest_below[1] if nearest_below[0] != 0 else 0
        
        # Calculate score based on distance and intensity
        if distance_above < distance_below and intensity_above > 0.5:
            # Strong resistance above - bearish
            return max(-0.8, -0.4 - 0.4 * intensity_above)
        elif distance_below < distance_above and intensity_below > 0.5:
            # Strong support below - bullish
            return min(0.8, 0.4 + 0.4 * intensity_below)
        elif self.liquidation_data.get('cleared_zone', False):
            # Cleared liquidation zone - bullish
            return 0.3
        
        return 0
    
    def get_webtrend_score(self):
        """
        Calculate score based on WebTrend indicator
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        base = 0.0
        if self.webtrend_status is not None:
            base = 0.3 if self.webtrend_status else -0.3
        # If explicit lines provided, refine with distance to lines
        if self.webtrend_lines:
            last = float(self.price[-1])
            above = sum(1 for v in self.webtrend_lines if last > v)
            below = sum(1 for v in self.webtrend_lines if last < v)
            # normalize count contribution
            n = len(self.webtrend_lines)
            if n > 0:
                base += max(-0.4, min(0.4, (above - below) / n * 0.6))
        # Trend value can give a small tilt
        if self.webtrend_trend_value is not None and len(self.webtrend_lines) >= 1:
            ref = sum(self.webtrend_lines) / len(self.webtrend_lines)
            diff = self.webtrend_trend_value - ref
            base += max(-0.2, min(0.2, 0.002 * diff))
        return max(-0.6, min(0.6, base))

    # ----------------- extra feature scores -----------------
    def _ema(self, values, period):
        if len(values) < period:
            return values[-1]
        k = 2 / (period + 1)
        ema_val = sum(values[-period:]) / period
        for v in values[-period+1:]:
            ema_val = (v - ema_val) * k + ema_val
        return ema_val

    def get_ma_trend_score(self):
        # Compute rough EMAs from price
        p = self.price
        if len(p) < 120:
            return 0.0
        ema20 = self._ema(p, 20)
        ema50 = self._ema(p, 50)
        ema100 = self._ema(p, 100)
        last = p[-1]
        if last > ema20 > ema50 > ema100:
            return 0.4
        if last < ema20 < ema50 < ema100:
            return -0.4
        # partial stacking
        score = 0.0
        if last > ema20: score += 0.1
        if ema20 > ema50: score += 0.1
        if ema50 > ema100: score += 0.1
        if last < ema20: score -= 0.1
        if ema20 < ema50: score -= 0.1
        if ema50 < ema100: score -= 0.1
        return max(-0.3, min(0.3, score))

    def get_pivot_score(self):
        # Approximate daily pivots using last 6 4h candles
        if len(self.price) < 6:
            return 0.0
        recent_high = max(self.price[-6:])
        recent_low = min(self.price[-6:])
        prev_close = self.price[-7] if len(self.price) >= 7 else self.price[-6]
        P = (recent_high + recent_low + prev_close) / 3
        R1 = 2 * P - recent_low
        S1 = 2 * P - recent_high
        last = self.price[-1]
        # breakout strength
        if last > R1 * 1.002:
            return 0.2
        if last < S1 * 0.998:
            return -0.2
        # near pivot congestion
        if abs(last - P) / P < 0.002:
            return -0.05
        return 0.0

    def get_oscillator_score(self):
        # Stoch %K/D, MACD, Williams %R simplified
        if len(self.price) < 26:
            return 0.0
        p = self.price
        # Williams %R
        h14 = max(p[-14:]); l14 = min(p[-14:])
        wr = -100 * (h14 - p[-1]) / max(1e-9, (h14 - l14))
        wr_score = 0.2 if wr < -80 else (-0.2 if wr > -20 else 0.0)
        # MACD
        ema12 = self._ema(p, 12)
        ema26 = self._ema(p, 26)
        macd = ema12 - ema26
        # use last two values for cross approx
        ema12_prev = self._ema(p[:-1], 12)
        ema26_prev = self._ema(p[:-1], 26)
        macd_prev = ema12_prev - ema26_prev
        macd_score = 0.15 if macd > 0 and macd_prev <= 0 else (-0.15 if macd < 0 and macd_prev >= 0 else 0.0)
        # Stoch K
        h = max(p[-14:]); l = min(p[-14:])
        k = 100 * (p[-1] - l) / max(1e-9, (h - l))
        stoch_score = 0.1 if (k < 20) else (-0.1 if (k > 80) else 0.0)
        return max(-0.4, min(0.4, wr_score + macd_score + stoch_score))

    def get_sentiment_event_score(self):
        score = 0.0
        if self.sentiment_series is not None and len(self.sentiment_series) > 0:
            s = self.sentiment_series[-1]
            score += max(-1.0, min(1.0, float(s))) * 0.5
        if self.event_series is not None and len(self.event_series) > 0:
            e = self.event_series[-1]
            score += max(-1.0, min(1.0, float(e))) * 0.5
        return max(-0.5, min(0.5, score))

    def apply_external_biases(self, tv_score: float | None = None) -> float:
        """
        Optional external bias from TradingView OCR (images 1 & 2).
        tv_score in [-1, 1]: negative = bearish tilt, positive = bullish tilt.
        We blend lightly so it cannot dominate core signals.
        """
        base = self.compute_score()
        if tv_score is None:
            return base
        blended = 0.85 * base + 0.15 * float(tv_score)
        return round(max(-1.0, min(1.0, blended)), 3)
    
    def is_near_key_level(self):
        """
        Check if price is near a key support/resistance level
        
        Returns:
        --------
        bool
            True if price is near a key level, False otherwise
        """
        if not self.liquidation_data or not self.liquidation_data.get('clusters'):
            return False
        
        clusters = self.liquidation_data['clusters']
        current_price = self.price[-1]
        
        # Calculate average price to determine threshold
        avg_price = sum(self.price[-20:]) / 20
        threshold = 0.01 * avg_price  # 1% of average price
        
        # Check if any cluster is within threshold
        for price_level, _ in clusters:
            if abs(current_price - price_level) < threshold:
                return True
        
        return False
    
    def compute_score(self):
        """
        Compute final predictive score
        
        Returns:
        --------
        float
            Score between -1.0 and 1.0
        """
        # Base weights (sum=1.0)
        w_rsi = 0.30
        w_volume = 0.25
        w_divergence = 0.15
        w_liquidation = 0.10
        w_webtrend = 0.05
        w_features = 0.15
        w_sentiment = 0.05
        
        # Adjust weights based on asset type
        if self.asset_type == 'BONK':
            # BONK is more volatile and sensitive to volume
            w_volume += 0.05
            w_liquidation -= 0.05
        elif self.asset_type == 'BTC':
            # BTC is more stable and less sensitive to short-term fluctuations
            w_rsi += 0.05
            w_volume -= 0.05
        
        # Apply optional overrides
        if self.weight_overrides:
            try:
                w_rsi = float(self.weight_overrides.get('w_rsi', w_rsi))
                w_volume = float(self.weight_overrides.get('w_volume', w_volume))
                w_divergence = float(self.weight_overrides.get('w_divergence', w_divergence))
                w_liquidation = float(self.weight_overrides.get('w_liquidation', w_liquidation))
                w_webtrend = float(self.weight_overrides.get('w_webtrend', w_webtrend))
                w_features = float(self.weight_overrides.get('w_features', w_features))
                w_sentiment = float(self.weight_overrides.get('w_sentiment', w_sentiment))
                total_w = max(1e-9, (w_rsi + w_volume + w_divergence + w_liquidation + w_webtrend + w_features + w_sentiment))
                w_rsi /= total_w; w_volume /= total_w; w_divergence /= total_w; w_liquidation /= total_w; w_webtrend /= total_w; w_features /= total_w; w_sentiment /= total_w
            except Exception:
                pass

        # Calculate component scores
        rsi_score = self.get_rsi_trend_score()
        volume_score = self.get_volume_trend_score()
        divergence = self.detect_divergence()
        liquidation_score = self.get_liquidation_score()
        webtrend_score = self.get_webtrend_score()
        feature_score = self.get_ma_trend_score() + self.get_pivot_score() + self.get_oscillator_score()
        feature_score = max(-1.0, min(1.0, feature_score))
        sentiment_score = self.get_sentiment_event_score()
        
        # Calculate final score
        final_score = (
            w_rsi * rsi_score + 
            w_volume * volume_score + 
            w_divergence * divergence + 
            w_liquidation * liquidation_score + 
            w_webtrend * webtrend_score +
            w_features * feature_score +
            w_sentiment * sentiment_score
        )
        
        # Apply volatility coefficient
        final_score *= self.volatility_coef
        
        # Ensure score is within bounds
        final_score = max(-1.0, min(1.0, final_score))
        
        return round(final_score, 3)
    
    def get_signal(self):
        """
        Get trading signal based on final score
        
        Returns:
        --------
        str
            Trading signal ('STRONG BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG SELL')
        """
        score = self.compute_score()
        
        if score > 0.6:
            return "STRONG BUY"
        elif score > 0.3:
            return "BUY"
        elif score >= -0.3:
            return "NEUTRAL"
        elif score >= -0.6:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def get_target_prices(self):
        """
        Calculate target prices for take profit and stop loss
        
        Returns:
        --------
        dict
            Dictionary containing TP1, TP2, and SL prices
        """
        current_price = self.price[-1]
        score = self.compute_score()
        signal = self.get_signal()
        
        # Calculate ATR (Average True Range) for volatility-based targets
        high_prices = [max(self.price[i], self.price[i-1]) for i in range(1, len(self.price))]
        low_prices = [min(self.price[i], self.price[i-1]) for i in range(1, len(self.price))]
        tr_values = [high_prices[i] - low_prices[i] for i in range(len(high_prices))]
        atr = sum(tr_values[-14:]) / 14  # 14-period ATR
        
        # Adjust ATR based on volatility coefficient
        atr *= self.volatility_coef
        
        # Calculate target prices based on signal
        long_cfg = self.tp_rules.get("long", {}) if isinstance(self.tp_rules, dict) else {}
        short_cfg = self.tp_rules.get("short", {}) if isinstance(self.tp_rules, dict) else {}
        atr_mult = float(self.tp_rules.get("atr_mult", 1.5)) if isinstance(self.tp_rules, dict) else 1.5
        l_tp1 = float(long_cfg.get("tp1_pct", 0.04)); l_tp2 = float(long_cfg.get("tp2_pct", 0.08))
        s_tp1 = float(short_cfg.get("tp1_pct", 0.04)); s_tp2 = float(short_cfg.get("tp2_pct", 0.08))
        if signal in ("STRONG BUY", "BUY"):
            tp1 = current_price * (1.0 + l_tp1)
            tp2 = current_price * (1.0 + l_tp2)
            sl = current_price - atr_mult * atr
        elif signal == "NEUTRAL":
            tp1 = current_price * 1.02  # 2% target
            tp2 = current_price * 1.04  # 4% target
            sl = current_price * 0.98  # 2% stop loss
        else:  # SELL or STRONG SELL
            tp1 = current_price * (1.0 - s_tp1)
            tp2 = current_price * (1.0 - s_tp2)
            sl = current_price + atr_mult * atr
        
        # Adjust targets based on liquidation clusters and pivot proximity
        if self.liquidation_data and self.liquidation_data.get('clusters'):
            clusters = self.liquidation_data['clusters']
            current_price = self.price[-1]
            
            # Find clusters above and below current price
            # Handle both 2-tuple (p, i) and 3-tuple (p, i, metadata) formats
            clusters_above = []
            clusters_below = []
            
            for cluster in clusters:
                # Extract price and intensity, handling both formats
                if len(cluster) > 2:
                    p, i, _ = cluster  # Unpack 3-tuple (price, intensity, metadata)
                else:
                    p, i = cluster     # Unpack 2-tuple (price, intensity)
                
                if p > current_price:
                    clusters_above.append((p, i))
                elif p < current_price:
                    clusters_below.append((p, i))
            
            if signal in ("STRONG BUY", "BUY") and clusters_above:
                # Adjust TP1 to nearest significant cluster above
                for price_level, intensity in sorted(clusters_above, key=lambda x: x[0]):
                    if intensity > 0.4 and price_level > current_price * 1.01:
                        tp1 = price_level * 0.995  # Just below the cluster
                        break
                
                # Adjust TP2 to next significant cluster above
                for price_level, intensity in sorted(clusters_above, key=lambda x: x[0])[1:]:
                    if intensity > 0.4 and price_level > tp1 * 1.01:
                        tp2 = price_level * 0.995  # Just below the cluster
                        break
            
            elif signal in ("SELL", "STRONG SELL") and clusters_below:
                # Adjust TP1 to nearest significant cluster below
                for price_level, intensity in sorted(clusters_below, key=lambda x: x[0], reverse=True):
                    if intensity > 0.4 and price_level < current_price * 0.99:
                        tp1 = price_level * 1.005  # Just above the cluster
                        break
                
                # Adjust TP2 to next significant cluster below
                for price_level, intensity in sorted(clusters_below, key=lambda x: x[0], reverse=True)[1:]:
                    if intensity > 0.4 and price_level < tp1 * 0.99:
                        tp2 = price_level * 1.005  # Just above the cluster
                        break

        # Blend with approximate pivots to tighten near strong levels
        try:
            recent_high = max(self.price[-6:]); recent_low = min(self.price[-6:])
            prev_close = self.price[-7] if len(self.price) >= 7 else self.price[-6]
            P = (recent_high + recent_low + prev_close) / 3
            R1 = 2 * P - recent_low; S1 = 2 * P - recent_high
            if signal in ("STRONG BUY", "BUY"):
                if abs(tp1 - R1) / R1 < 0.01: tp1 = R1 * 0.995
                if abs(tp2 - (R1 + (R1 - P))) / R1 < 0.02: tp2 = (R1 + (R1 - P)) * 0.995
            elif signal in ("SELL", "STRONG SELL"):
                if abs(tp1 - S1) / S1 < 0.01: tp1 = S1 * 1.005
                if abs(tp2 - (S1 - (P - S1))) / max(1e-9, S1) < 0.02: tp2 = (S1 - (P - S1)) * 1.005
        except Exception:
            pass
        
        return {
            "TP1": round(tp1, 6 if self.asset_type == 'BONK' else 2),
            "TP2": round(tp2, 6 if self.asset_type == 'BONK' else 2),
            "SL": round(sl, 6 if self.asset_type == 'BONK' else 2)
        }
    
    def get_component_scores(self):
        """
        Get individual component scores
        
        Returns:
        --------
        dict
            Dictionary containing component scores
        """
        return {
            "rsi_score": self.get_rsi_trend_score(),
            "volume_score": self.get_volume_trend_score(),
            "divergence": self.detect_divergence(),
            "liquidation_score": self.get_liquidation_score(),
            "webtrend_score": self.get_webtrend_score()
        }
    
    def get_full_analysis(self):
        """
        Get full analysis including all scores and targets
        
        Returns:
        --------
        dict
            Dictionary containing all analysis data
        """
        score = self.compute_score()
        signal = self.get_signal()
        targets = self.get_target_prices()
        components = self.get_component_scores()
        
        return {
            "asset": self.asset_type,
            "price": self.price[-1],
            "rsi": self.rsi_raw[-1],
            "rsi_sma": self.rsi_sma[-1],
            "final_score": score,
            "signal": signal,
            "components": components,
            "targets": targets,
            "webtrend_status": self.webtrend_status,
            "webtrend_lines": self.webtrend_lines,
            "webtrend_trend_value": self.webtrend_trend_value,
        }


def analyze_market_data(assets_data):
    """
    Analyze market data for multiple assets
    
    Parameters:
    -----------
    assets_data : dict
        Dictionary containing data for each asset
        Format: {
            'BTC': {
                'price': [...],
                'rsi_raw': [...],
                'rsi_sma': [...],
                'volume': [...],
                'liquidation_data': {...},
                'webtrend_status': bool
            },
            'SOL': {...},
            'BONK': {...}
        }
    
    Returns:
    --------
    dict
        Dictionary containing analysis results for each asset
    """
    results = {}
    
    for asset, data in assets_data.items():
        predictor = EnhancedRsiVolumePredictor(
            data['rsi_sma'],
            data['rsi_raw'],
            data['volume'],
            data['price'],
            data.get('liquidation_data'),
            data.get('webtrend_status'),
            asset
        )
        
        results[asset] = predictor.get_full_analysis()
    
    return results


def get_market_assessment(analysis_results):
    """
    Get overall market assessment based on analysis results
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary containing analysis results for each asset
    
    Returns:
    --------
    dict
        Dictionary containing market assessment
    """
    # Calculate average score
    scores = [result['final_score'] for result in analysis_results.values()]
    avg_score = sum(scores) / len(scores)
    
    # Find best and worst assets
    best_asset = max(analysis_results.items(), key=lambda x: x[1]['final_score'])
    worst_asset = min(analysis_results.items(), key=lambda x: x[1]['final_score'])
    
    # Determine market condition
    if avg_score > 0.3:
        market_condition = "BULLISH"
    elif avg_score < -0.3:
        market_condition = "BEARISH"
    else:
        market_condition = "NEUTRAL"
    
    # Generate rotation strategy
    assets_by_score = sorted(analysis_results.items(), key=lambda x: x[1]['final_score'], reverse=True)
    rotation_strategy = []
    
    for i, (asset, result) in enumerate(assets_by_score):
        if result['final_score'] > 0.3:
            position = "Strong" if i == 0 else "Moderate" if i == 1 else "Small"
            rotation_strategy.append(f"{position} position in {asset}")
    
    if not rotation_strategy:
        rotation_strategy = ["No clear rotation strategy. Consider waiting for better setups."]
    
    return {
        "market_condition": market_condition,
        "average_score": round(avg_score, 3),
        "best_opportunity": {
            "asset": best_asset[0],
            "score": best_asset[1]['final_score'],
            "signal": best_asset[1]['signal']
        },
        "weakest_asset": {
            "asset": worst_asset[0],
            "score": worst_asset[1]['final_score'],
            "signal": worst_asset[1]['signal']
        },
        "rotation_strategy": rotation_strategy
    }


# Example usage with the latest chart data:
if __name__ == "__main__":
    # Latest data from charts
    assets_data = {
        'SOL': {
            'price': [196.62],
            'rsi_raw': [65.15],
            'rsi_sma': [51.37],
            'volume': [230.67e3],
            'liquidation_data': {
                'clusters': [(183, 0.8), (198, 0.6), (205, 0.7)],
                'cleared_zone': True
            },
            'webtrend_status': True
        },
        'BTC': {
            'price': [116756.98],
            'rsi_raw': [62.44],
            'rsi_sma': [40.25],
            'volume': [1.03e3],
            'liquidation_data': {
                'clusters': [(114000, 0.7), (112000, 0.5), (120336, 0.8)],
                'cleared_zone': False
            },
            'webtrend_status': True
        },
        'BONK': {
            'price': [0.00002341],
            'rsi_raw': [61.61],
            'rsi_sma': [41.65],
            'volume': [149.42e9],
            'liquidation_data': {
                'clusters': [(0.00002100, 0.6), (0.00002750, 0.7)],
                'cleared_zone': False
            },
            'webtrend_status': True
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
