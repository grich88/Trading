# RSI + Volume Predictive Scoring Model for 4H Charts (BTC & SOL)
#
# This model integrates RSI streak analysis, volume streaks, and divergence detection
# to output a predictive score for short-term momentum.
# Score ranges from -1.0 (strong bearish) to +1.0 (strong bullish)

class RsiVolumePredictor:
    def __init__(self, rsi_sma_series, rsi_raw_series, volume_series, price_series):
        self.rsi_sma = rsi_sma_series
        self.rsi_raw = rsi_raw_series
        self.volume = volume_series
        self.price = price_series

    def get_rsi_trend_score(self):
        candles_above_50 = sum(1 for x in self.rsi_sma[-50:] if x > 50)
        candles_above_70 = sum(1 for x in self.rsi_sma[-12:] if x > 70)
        candles_below_30 = sum(1 for x in self.rsi_sma[-10:] if x < 30)

        if self.rsi_sma[-1] > 70 and candles_above_70 >= 3:
            return max(0.0, 0.9 - 0.1 * (candles_above_70 - 3))
        elif self.rsi_sma[-1] > 50 and candles_above_50 >= 6:
            return min(0.8, 0.1 * candles_above_50)
        elif self.rsi_sma[-1] < 30 and candles_below_30 >= 3:
            return min(-0.1 * candles_below_30, -0.3)
        else:
            return 0

    def get_volume_trend_score(self):
        last_vol = self.volume[-10:]
        avg_early = sum(last_vol[:5]) / 5
        avg_late = sum(last_vol[5:]) / 5
        momentum = avg_late / avg_early if avg_early != 0 else 1

        green_streak = sum(1 for i in range(-5, 0) if self.price[i] > self.price[i - 1])
        red_streak = sum(1 for i in range(-5, 0) if self.price[i] < self.price[i - 1])

        if green_streak >= 3 and momentum > 1.05:
            return min(1.0, 0.25 * green_streak + 0.5 * (momentum - 1))
        elif red_streak >= 3 and momentum > 1.05:
            return max(-1.0, -0.25 * red_streak - 0.5 * (momentum - 1))
        else:
            return 0

    def detect_divergence(self):
        price_highs = self.price[-10:]
        rsi_highs = self.rsi_raw[-10:]
        volume_highs = self.volume[-10:]

        price_div = price_highs[-1] > max(price_highs[:-1])
        rsi_div = rsi_highs[-1] < max(rsi_highs[:-1])
        volume_div = volume_highs[-1] < max(volume_highs[:-1])

        if price_div and (rsi_div or volume_div):
            return -0.5

        price_low = price_highs[-1] < min(price_highs[:-1])
        rsi_up = rsi_highs[-1] > min(rsi_highs[:-1])
        volume_fade = volume_highs[-1] < max(volume_highs[:-1])

        if price_low and (rsi_up or volume_fade):
            return 0.5

        return 0

    def compute_score(self):
        w1, w2, w3 = 0.4, 0.4, 0.2
        rsi_score = self.get_rsi_trend_score()
        volume_score = self.get_volume_trend_score()
        divergence = self.detect_divergence()

        return round(w1 * rsi_score + w2 * volume_score + w3 * divergence, 3)
