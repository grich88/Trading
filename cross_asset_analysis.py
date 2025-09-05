import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression

@dataclass
class CrossAssetSignal:
    primary_asset: str
    correlated_asset: str
    correlation: float
    lead_lag_minutes: int  # Positive means primary leads, negative means primary lags
    strength: float  # 0 to 1
    signal_type: str  # 'confirmation', 'divergence', 'leading_indicator'
    confidence: float  # 0 to 1
    supporting_factors: List[str]

class CrossAssetAnalyzer:
    def __init__(self, lookback_period: int = 100, correlation_threshold: float = 0.7):
        self.lookback_period = lookback_period
        self.correlation_threshold = correlation_threshold
    
    def calculate_correlation_matrix(self, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation matrix between assets."""
        returns = {}
        for asset, data in asset_data.items():
            returns[asset] = data['close'].pct_change()
        
        return pd.DataFrame(returns).corr()
    
    def find_lead_lag_relationship(self, series1: pd.Series, series2: pd.Series, 
                                 max_lag: int = 10) -> Tuple[int, float]:
        """
        Find the lead/lag relationship between two price series.
        Returns (lag, correlation) where positive lag means series1 leads series2.
        """
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = series1.iloc[abs(lag):].corr(series2.iloc[:lag])
            elif lag > 0:
                corr = series1.iloc[:-lag].corr(series2.iloc[lag:])
            else:
                corr = series1.corr(series2)
            correlations.append((lag, corr))
        
        # Return lag with highest correlation
        return max(correlations, key=lambda x: abs(x[1]))
    
    def detect_correlation_breakdown(self, series1: pd.Series, series2: pd.Series, 
                                  window: int = 20) -> bool:
        """Detect if normal correlation pattern is breaking down."""
        rolling_corr = series1.rolling(window).corr(series2)
        recent_corr = rolling_corr.iloc[-5:].mean()
        historical_corr = rolling_corr.iloc[:-5].mean()
        
        return abs(recent_corr - historical_corr) > 0.3  # Significant correlation change
    
    def analyze_cross_asset_signals(self, asset_data: Dict[str, pd.DataFrame], 
                                  primary_asset: str) -> List[CrossAssetSignal]:
        """
        Analyze relationships between primary asset and other assets.
        Returns list of significant cross-asset signals.
        """
        signals = []
        primary_returns = asset_data[primary_asset]['close'].pct_change()
        
        for asset, data in asset_data.items():
            if asset == primary_asset:
                continue
            
            asset_returns = data['close'].pct_change()
            
            # Calculate correlation and lead/lag
            lag, correlation = self.find_lead_lag_relationship(primary_returns, asset_returns)
            
            if abs(correlation) >= self.correlation_threshold:
                # Determine if correlation pattern is stable
                correlation_breakdown = self.detect_correlation_breakdown(primary_returns, asset_returns)
                
                # Calculate recent trend alignment
                primary_trend = primary_returns.tail(5).mean()
                asset_trend = asset_returns.tail(5).mean()
                trends_aligned = (primary_trend > 0 and asset_trend > 0) or \
                               (primary_trend < 0 and asset_trend < 0)
                
                supporting_factors = []
                
                if correlation > 0:
                    if trends_aligned:
                        signal_type = 'confirmation'
                        confidence = 0.8
                        supporting_factors.append('Trends aligned')
                    else:
                        signal_type = 'divergence'
                        confidence = 0.7
                        supporting_factors.append('Trend divergence detected')
                else:  # Negative correlation
                    if not trends_aligned:
                        signal_type = 'confirmation'
                        confidence = 0.8
                        supporting_factors.append('Inverse relationship confirmed')
                    else:
                        signal_type = 'divergence'
                        confidence = 0.7
                        supporting_factors.append('Inverse relationship breaking down')
                
                # Adjust confidence based on correlation stability
                if correlation_breakdown:
                    confidence *= 0.8
                    supporting_factors.append('Correlation pattern changing')
                
                # Calculate signal strength based on correlation and trend alignment
                strength = abs(correlation) * (1.2 if trends_aligned else 0.8)
                
                signals.append(CrossAssetSignal(
                    primary_asset=primary_asset,
                    correlated_asset=asset,
                    correlation=correlation,
                    lead_lag_minutes=lag,
                    strength=min(strength, 1.0),
                    signal_type=signal_type,
                    confidence=min(confidence, 1.0),
                    supporting_factors=supporting_factors
                ))
        
        return signals
    
    def analyze_market_regime(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Analyze overall market regime based on cross-asset relationships.
        Returns regime characteristics and confidence levels.
        """
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(asset_data)
        
        # Calculate average correlation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
        
        # Calculate volatility regime
        volatilities = {asset: data['close'].pct_change().std() 
                       for asset, data in asset_data.items()}
        avg_vol = np.mean(list(volatilities.values()))
        historical_vol = np.mean([data['close'].pct_change().std() 
                                for data in asset_data.values()])
        
        # Determine market regime
        regime = {
            'correlation_regime': avg_corr,  # Higher means more correlated market
            'volatility_regime': avg_vol / historical_vol,  # > 1 means high volatility
            'risk_on': avg_corr > 0.7 and avg_vol < historical_vol,  # Classic risk-on
            'risk_off': avg_corr > 0.7 and avg_vol > historical_vol,  # Classic risk-off
            'regime_confidence': min(abs(avg_corr) + abs(1 - avg_vol/historical_vol), 1.0)
        }
        
        return regime
    
    def generate_cross_asset_summary(self, asset_data: Dict[str, pd.DataFrame], 
                                   primary_asset: str) -> Dict[str, Any]:
        """
        Generate comprehensive cross-asset analysis summary.
        """
        signals = self.analyze_cross_asset_signals(asset_data, primary_asset)
        regime = self.analyze_market_regime(asset_data)
        
        # Analyze correlation stability
        correlation_stability = {}
        for asset in asset_data.keys():
            if asset != primary_asset:
                breakdown = self.detect_correlation_breakdown(
                    asset_data[primary_asset]['close'].pct_change(),
                    asset_data[asset]['close'].pct_change()
                )
                correlation_stability[asset] = not breakdown
        
        return {
            'signals': signals,
            'market_regime': regime,
            'correlation_stability': correlation_stability,
            'analysis_timestamp': pd.Timestamp.now()
        }
