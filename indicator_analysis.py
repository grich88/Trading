import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class IndicatorSignal:
    value: float
    signal: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0 to 1
    confidence: float  # 0 to 1
    supporting_factors: List[str]

class IndicatorAnalyzer:
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
    
    def analyze_macd(self, data: pd.DataFrame, 
                    fast_period: int = 12,
                    slow_period: int = 26,
                    signal_period: int = 9) -> IndicatorSignal:
        """
        Analyze MACD with advanced signal detection.
        """
        # Calculate MACD
        exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        
        # Current values
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        current_hist = hist.iloc[-1]
        
        # Determine trend strength
        hist_abs_mean = abs(hist).mean()
        strength = min(abs(current_hist) / hist_abs_mean, 1.0)
        
        # Analyze histogram pattern
        hist_pattern = hist.tail(5)
        increasing_momentum = all(hist_pattern.diff().dropna() > 0)
        decreasing_momentum = all(hist_pattern.diff().dropna() < 0)
        
        # Determine signal
        supporting_factors = []
        if current_macd > current_signal:
            if increasing_momentum:
                signal_type = 'buy'
                supporting_factors.append('Increasing positive momentum')
                confidence = 0.8
            else:
                signal_type = 'buy'
                confidence = 0.6
                supporting_factors.append('MACD crossover but momentum unclear')
        elif current_macd < current_signal:
            if decreasing_momentum:
                signal_type = 'sell'
                supporting_factors.append('Increasing negative momentum')
                confidence = 0.8
            else:
                signal_type = 'sell'
                confidence = 0.6
                supporting_factors.append('MACD crossunder but momentum unclear')
        else:
            signal_type = 'neutral'
            confidence = 0.5
            supporting_factors.append('No clear direction')
        
        # Add volume confirmation
        if 'volume' in data.columns:
            vol_sma = data['volume'].rolling(window=20).mean()
            if data['volume'].iloc[-1] > vol_sma.iloc[-1]:
                confidence *= 1.2
                supporting_factors.append('Above average volume')
            else:
                confidence *= 0.8
                supporting_factors.append('Below average volume')
        
        return IndicatorSignal(
            value=current_macd,
            signal=signal_type,
            strength=strength,
            confidence=min(confidence, 1.0),
            supporting_factors=supporting_factors
        )
    
    def analyze_stochastic(self, data: pd.DataFrame,
                          k_period: int = 14,
                          d_period: int = 3,
                          smooth_k: int = 3) -> IndicatorSignal:
        """
        Analyze Stochastic Oscillator with advanced signal detection.
        """
        # Calculate Stochastic
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        
        k = 100 * ((data['close'] - low_min) / (high_max - low_min))
        k = k.rolling(window=smooth_k).mean()  # Apply smoothing
        d = k.rolling(window=d_period).mean()
        
        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        
        # Determine trend strength
        strength = min(abs(current_k - 50) / 50, 1.0)
        
        # Analyze pattern
        k_pattern = k.tail(5)
        d_pattern = d.tail(5)
        
        supporting_factors = []
        
        # Determine signal
        if current_k > current_d:
            if current_k < 20:
                signal_type = 'buy'
                confidence = 0.9
                supporting_factors.append('Oversold with bullish crossover')
            elif current_k > 80:
                signal_type = 'sell'
                confidence = 0.7
                supporting_factors.append('Overbought despite bullish crossover')
            else:
                signal_type = 'buy'
                confidence = 0.6
                supporting_factors.append('Bullish crossover in neutral zone')
        elif current_k < current_d:
            if current_k > 80:
                signal_type = 'sell'
                confidence = 0.9
                supporting_factors.append('Overbought with bearish crossover')
            elif current_k < 20:
                signal_type = 'buy'
                confidence = 0.7
                supporting_factors.append('Oversold despite bearish crossover')
            else:
                signal_type = 'sell'
                confidence = 0.6
                supporting_factors.append('Bearish crossover in neutral zone')
        else:
            signal_type = 'neutral'
            confidence = 0.5
            supporting_factors.append('No clear direction')
        
        # Check for divergence
        price_trend = data['close'].diff().rolling(window=5).mean().iloc[-1]
        k_trend = k.diff().rolling(window=5).mean().iloc[-1]
        
        if price_trend > 0 and k_trend < 0:
            supporting_factors.append('Bearish divergence')
            confidence *= 0.8
        elif price_trend < 0 and k_trend > 0:
            supporting_factors.append('Bullish divergence')
            confidence *= 1.2
        
        return IndicatorSignal(
            value=current_k,
            signal=signal_type,
            strength=strength,
            confidence=min(confidence, 1.0),
            supporting_factors=supporting_factors
        )
    
    def analyze_webtrend(self, data: pd.DataFrame,
                        ema_short: int = 20,
                        ema_med: int = 50,
                        ema_long: int = 100) -> IndicatorSignal:
        """
        Analyze WebTrend (EMA-based trend detection) with advanced signal detection.
        """
        # Calculate EMAs
        ema_s = data['close'].ewm(span=ema_short, adjust=False).mean()
        ema_m = data['close'].ewm(span=ema_med, adjust=False).mean()
        ema_l = data['close'].ewm(span=ema_long, adjust=False).mean()
        
        current_ema_s = ema_s.iloc[-1]
        current_ema_m = ema_m.iloc[-1]
        current_ema_l = ema_l.iloc[-1]
        
        supporting_factors = []
        
        # Determine trend alignment
        if current_ema_s > current_ema_m > current_ema_l:
            signal_type = 'buy'
            strength = (current_ema_s - current_ema_l) / current_ema_l
            confidence = 0.8
            supporting_factors.append('All EMAs aligned bullish')
        elif current_ema_s < current_ema_m < current_ema_l:
            signal_type = 'sell'
            strength = (current_ema_l - current_ema_s) / current_ema_l
            confidence = 0.8
            supporting_factors.append('All EMAs aligned bearish')
        else:
            # Check if transitioning
            if current_ema_s > current_ema_m:
                signal_type = 'buy'
                strength = (current_ema_s - current_ema_m) / current_ema_m
                confidence = 0.6
                supporting_factors.append('Short-term bullish transition')
            elif current_ema_s < current_ema_m:
                signal_type = 'sell'
                strength = (current_ema_m - current_ema_s) / current_ema_m
                confidence = 0.6
                supporting_factors.append('Short-term bearish transition')
            else:
                signal_type = 'neutral'
                strength = 0.0
                confidence = 0.5
                supporting_factors.append('No clear trend')
        
        # Check trend momentum
        ema_s_slope = (ema_s.iloc[-1] - ema_s.iloc[-5]) / ema_s.iloc[-5]
        if abs(ema_s_slope) > 0.02:  # 2% change
            confidence *= 1.2
            supporting_factors.append(f"Strong {'upward' if ema_s_slope > 0 else 'downward'} momentum")
        
        # Volume confirmation
        if 'volume' in data.columns:
            vol_sma = data['volume'].rolling(window=20).mean()
            if data['volume'].iloc[-1] > vol_sma.iloc[-1]:
                confidence *= 1.1
                supporting_factors.append('Above average volume')
        
        return IndicatorSignal(
            value=current_ema_s,
            signal=signal_type,
            strength=min(strength, 1.0),
            confidence=min(confidence, 1.0),
            supporting_factors=supporting_factors
        )
