import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from technical_analysis import TechnicalAnalyzer
from indicator_analysis import IndicatorAnalyzer
from pattern_recognition import PatternRecognizer
from cross_asset_analysis import CrossAssetAnalyzer

@dataclass
class TradeSignal:
    asset: str
    signal_type: str  # 'buy', 'sell', 'close_long', 'close_short'
    confidence: float  # 0 to 1
    entry_price: float
    target_price: float
    stop_loss: float
    timeframe: str  # '1m', '5m', '15m', '1h', '4h', '1d'
    expected_duration: str  # 'short_term', 'medium_term', 'long_term'
    supporting_factors: List[str]
    risk_reward_ratio: float
    volume_profile: Dict[str, float]
    pattern_matches: List[str]
    indicator_signals: Dict[str, float]
    cross_asset_confirmation: Dict[str, float]

class SignalGenerator:
    def __init__(self, 
                 min_confidence: float = 0.7,
                 min_risk_reward: float = 2.0,
                 required_confirmations: int = 2):
        self.technical_analyzer = TechnicalAnalyzer()
        self.indicator_analyzer = IndicatorAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.cross_asset_analyzer = CrossAssetAnalyzer()
        
        self.min_confidence = min_confidence
        self.min_risk_reward = min_risk_reward
        self.required_confirmations = required_confirmations
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze volume characteristics."""
        volume = data['volume']
        
        return {
            'current_vs_avg': volume.iloc[-1] / volume.mean(),
            'trend_strength': volume.diff().rolling(5).mean().iloc[-1] / volume.mean(),
            'volume_breakout': volume.iloc[-1] > volume.rolling(20).max().iloc[-1],
            'volume_consistency': volume.std() / volume.mean()
        }
    
    def calculate_risk_reward(self, entry: float, target: float, stop: float) -> float:
        """Calculate risk/reward ratio."""
        if entry == stop:  # Avoid division by zero
            return 0
        
        if target > entry:  # Long position
            reward = target - entry
            risk = entry - stop
        else:  # Short position
            reward = entry - target
            risk = stop - entry
        
        return reward / risk if risk > 0 else 0
    
    def estimate_signal_duration(self, pattern_type: str, timeframe: str) -> str:
        """Estimate expected duration of the trade."""
        # Map timeframes to minutes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '1h': 60,
            '4h': 240, '1d': 1440
        }
        
        # Map patterns to typical duration multipliers
        pattern_multipliers = {
            'head_and_shoulders': 5,
            'double_top': 3,
            'double_bottom': 3,
            'triangle': 2,
            'trend_continuation': 2,
            'breakout': 1.5
        }
        
        base_minutes = timeframe_minutes.get(timeframe, 60)
        multiplier = pattern_multipliers.get(pattern_type, 2)
        total_minutes = base_minutes * multiplier
        
        if total_minutes < 60:
            return 'short_term'
        elif total_minutes < 240:
            return 'medium_term'
        else:
            return 'long_term'
    
    def generate_trade_signal(self, 
                            asset: str,
                            data: pd.DataFrame,
                            timeframe: str,
                            cross_asset_data: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[TradeSignal]:
        """
        Generate comprehensive trade signal with multiple confirmations.
        """
        current_price = data['close'].iloc[-1]
        
        # Technical Analysis
        rsi_analysis = self.technical_analyzer.analyze_rsi(data)
        
        # Pattern Recognition
        patterns = self.pattern_recognizer.detect_all_patterns(data)
        
        # Indicator Analysis
        macd_signal = self.indicator_analyzer.analyze_macd(data)
        stoch_signal = self.indicator_analyzer.analyze_stochastic(data)
        webtrend_signal = self.indicator_analyzer.analyze_webtrend(data)
        
        # Cross-asset Analysis if data provided
        cross_asset_signals = {}
        if cross_asset_data:
            cross_asset_summary = self.cross_asset_analyzer.generate_cross_asset_summary(
                cross_asset_data, asset
            )
            cross_asset_signals = {
                signal.correlated_asset: signal.correlation 
                for signal in cross_asset_summary['signals']
            }
        
        # Collect all buy/sell signals
        buy_signals = 0
        sell_signals = 0
        signal_confidence = 0.0
        supporting_factors = []
        
        # Check RSI
        if rsi_analysis.current_value < 30:
            buy_signals += 1
            signal_confidence += 0.6
            supporting_factors.append(f"RSI oversold: {rsi_analysis.current_value:.2f}")
        elif rsi_analysis.current_value > 70:
            sell_signals += 1
            signal_confidence += 0.6
            supporting_factors.append(f"RSI overbought: {rsi_analysis.current_value:.2f}")
        
        # Check MACD
        if macd_signal.signal == 'buy':
            buy_signals += 1
            signal_confidence += macd_signal.confidence
            supporting_factors.extend(macd_signal.supporting_factors)
        elif macd_signal.signal == 'sell':
            sell_signals += 1
            signal_confidence += macd_signal.confidence
            supporting_factors.extend(macd_signal.supporting_factors)
        
        # Check Stochastic
        if stoch_signal.signal == 'buy':
            buy_signals += 1
            signal_confidence += stoch_signal.confidence
            supporting_factors.extend(stoch_signal.supporting_factors)
        elif stoch_signal.signal == 'sell':
            sell_signals += 1
            signal_confidence += stoch_signal.confidence
            supporting_factors.extend(stoch_signal.supporting_factors)
        
        # Check WebTrend
        if webtrend_signal.signal == 'buy':
            buy_signals += 1
            signal_confidence += webtrend_signal.confidence
            supporting_factors.extend(webtrend_signal.supporting_factors)
        elif webtrend_signal.signal == 'sell':
            sell_signals += 1
            signal_confidence += webtrend_signal.confidence
            supporting_factors.extend(webtrend_signal.supporting_factors)
        
        # Check Patterns
        pattern_matches = []
        for pattern in patterns:
            pattern_matches.append(pattern.pattern_type)
            if pattern.target_price > current_price:
                buy_signals += 1
                signal_confidence += pattern.confidence
                supporting_factors.append(f"Pattern: {pattern.pattern_type}")
            else:
                sell_signals += 1
                signal_confidence += pattern.confidence
                supporting_factors.append(f"Pattern: {pattern.pattern_type}")
        
        # Normalize confidence
        total_signals = max(buy_signals, sell_signals)
        if total_signals > 0:
            signal_confidence = signal_confidence / total_signals
        
        # Generate signal only if we have enough confirmations
        if max(buy_signals, sell_signals) >= self.required_confirmations and \
           signal_confidence >= self.min_confidence:
            
            # Determine signal type
            if buy_signals > sell_signals:
                signal_type = 'buy'
                # Calculate target and stop based on patterns and indicators
                target_prices = [p.target_price for p in patterns if p.target_price > current_price]
                stop_prices = [p.stop_loss for p in patterns if p.stop_loss < current_price]
                
                target_price = max(target_prices) if target_prices else current_price * 1.02
                stop_loss = min(stop_prices) if stop_prices else current_price * 0.98
            else:
                signal_type = 'sell'
                target_prices = [p.target_price for p in patterns if p.target_price < current_price]
                stop_prices = [p.stop_loss for p in patterns if p.stop_loss > current_price]
                
                target_price = min(target_prices) if target_prices else current_price * 0.98
                stop_loss = max(stop_prices) if stop_prices else current_price * 1.02
            
            # Calculate risk/reward
            risk_reward = self.calculate_risk_reward(current_price, target_price, stop_loss)
            
            # Only generate signal if risk/reward is acceptable
            if risk_reward >= self.min_risk_reward:
                return TradeSignal(
                    asset=asset,
                    signal_type=signal_type,
                    confidence=signal_confidence,
                    entry_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    timeframe=timeframe,
                    expected_duration=self.estimate_signal_duration(
                        pattern_matches[0] if pattern_matches else 'trend_continuation',
                        timeframe
                    ),
                    supporting_factors=supporting_factors,
                    risk_reward_ratio=risk_reward,
                    volume_profile=self.analyze_volume_profile(data),
                    pattern_matches=pattern_matches,
                    indicator_signals={
                        'rsi': rsi_analysis.current_value,
                        'macd': macd_signal.value,
                        'stochastic': stoch_signal.value,
                        'webtrend': webtrend_signal.value
                    },
                    cross_asset_confirmation=cross_asset_signals
                )
        
        return None
