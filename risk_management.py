import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from signal_generation import TradeSignal

@dataclass
class PositionSize:
    asset: str
    base_size: float  # Base position size in quote currency
    adjusted_size: float  # Size after all adjustments
    max_size: float  # Maximum allowed position size
    risk_amount: float  # Amount at risk in quote currency
    risk_percentage: float  # Percentage of account at risk
    adjustment_factors: Dict[str, float]  # Factors that influenced the position size

@dataclass
class RiskMetrics:
    volatility: float
    correlation_impact: float
    liquidity_score: float
    market_impact: float
    overall_risk_score: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR / Expected Shortfall

class RiskManager:
    def __init__(self,
                 account_size: float,
                 max_risk_per_trade: float = 0.02,  # 2% max risk per trade
                 max_total_risk: float = 0.05,  # 5% max total portfolio risk
                 max_correlation: float = 0.7,
                 volatility_lookback: int = 20):
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.max_correlation = max_correlation
        self.volatility_lookback = volatility_lookback
        self.open_positions: Dict[str, Dict] = {}
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> RiskMetrics:
        """Calculate comprehensive risk metrics for an asset."""
        returns = data['close'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Downside returns for Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        
        # Sharpe and Sortino ratios (assuming 0% risk-free rate for simplicity)
        sharpe_ratio = (returns.mean() * 252) / volatility if volatility != 0 else 0
        sortino_ratio = (returns.mean() * 252) / downside_std if downside_std != 0 else 0
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / running_max - 1
        max_drawdown = drawdowns.min()
        
        # Market impact score (based on volume)
        avg_volume = data['volume'].mean()
        recent_volume = data['volume'].tail(5).mean()
        liquidity_score = recent_volume / avg_volume
        
        # Correlation impact (placeholder - would need market data)
        correlation_impact = 0.5  # Default middle value
        
        # Market impact (simplified model)
        market_impact = 1 - (liquidity_score / 2)  # Higher liquidity = lower impact
        
        # Overall risk score (0 to 1, higher = riskier)
        overall_risk_score = np.mean([
            volatility / 2,  # Normalize volatility
            abs(correlation_impact),
            1 - liquidity_score,
            market_impact,
            abs(max_drawdown)
        ])
        
        return RiskMetrics(
            volatility=volatility,
            correlation_impact=correlation_impact,
            liquidity_score=liquidity_score,
            market_impact=market_impact,
            overall_risk_score=overall_risk_score,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def calculate_position_size(self, 
                              signal: TradeSignal,
                              risk_metrics: RiskMetrics,
                              current_positions: Dict[str, float]) -> PositionSize:
        """Calculate optimal position size based on multiple factors."""
        # Calculate base position size based on account risk
        risk_distance = abs(signal.entry_price - signal.stop_loss)
        max_risk_amount = self.account_size * self.max_risk_per_trade
        base_size = max_risk_amount / risk_distance
        
        adjustment_factors = {}
        
        # Adjust for volatility
        vol_factor = 1.0
        if risk_metrics.volatility > 0.5:  # High volatility
            vol_factor = 0.7
        elif risk_metrics.volatility < 0.2:  # Low volatility
            vol_factor = 1.2
        adjustment_factors['volatility'] = vol_factor
        
        # Adjust for correlation
        corr_factor = 1.0
        total_correlation = sum(abs(corr) for corr in signal.cross_asset_confirmation.values())
        if total_correlation > self.max_correlation:
            corr_factor = 0.8
        adjustment_factors['correlation'] = corr_factor
        
        # Adjust for signal confidence
        conf_factor = signal.confidence
        adjustment_factors['confidence'] = conf_factor
        
        # Adjust for liquidity
        liq_factor = min(risk_metrics.liquidity_score, 1.0)
        adjustment_factors['liquidity'] = liq_factor
        
        # Adjust for existing positions
        total_exposure = sum(current_positions.values())
        exposure_factor = max(0, 1 - (total_exposure / (self.account_size * self.max_total_risk)))
        adjustment_factors['exposure'] = exposure_factor
        
        # Calculate final adjusted size
        adjusted_size = base_size
        for factor in adjustment_factors.values():
            adjusted_size *= factor
        
        # Calculate maximum allowed position size
        max_size = self.account_size * self.max_risk_per_trade * 2  # 2x max risk for leverage
        
        # Calculate actual risk amount and percentage
        risk_amount = adjusted_size * risk_distance
        risk_percentage = risk_amount / self.account_size
        
        return PositionSize(
            asset=signal.asset,
            base_size=base_size,
            adjusted_size=min(adjusted_size, max_size),
            max_size=max_size,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            adjustment_factors=adjustment_factors
        )
    
    def validate_trade(self, 
                      signal: TradeSignal,
                      risk_metrics: RiskMetrics,
                      position_size: PositionSize) -> Tuple[bool, List[str]]:
        """Validate if a trade meets all risk management criteria."""
        reasons = []
        is_valid = True
        
        # Check risk percentage
        if position_size.risk_percentage > self.max_risk_per_trade:
            is_valid = False
            reasons.append(f"Risk percentage {position_size.risk_percentage:.1%} exceeds maximum {self.max_risk_per_trade:.1%}")
        
        # Check risk/reward ratio
        if signal.risk_reward_ratio < 2.0:
            is_valid = False
            reasons.append(f"Risk/reward ratio {signal.risk_reward_ratio:.1f} below minimum 2.0")
        
        # Check volatility
        if risk_metrics.volatility > 0.8:  # 80% annualized volatility
            is_valid = False
            reasons.append(f"Volatility {risk_metrics.volatility:.1%} too high")
        
        # Check liquidity
        if risk_metrics.liquidity_score < 0.5:
            is_valid = False
            reasons.append(f"Liquidity score {risk_metrics.liquidity_score:.2f} too low")
        
        # Check correlation
        high_correlations = sum(1 for corr in signal.cross_asset_confirmation.values() 
                              if abs(corr) > self.max_correlation)
        if high_correlations > 2:  # More than 2 highly correlated assets
            is_valid = False
            reasons.append("Too many highly correlated assets")
        
        # Check overall risk score
        if risk_metrics.overall_risk_score > 0.8:
            is_valid = False
            reasons.append(f"Overall risk score {risk_metrics.overall_risk_score:.2f} too high")
        
        return is_valid, reasons
    
    def update_position(self,
                       signal: TradeSignal,
                       position_size: PositionSize,
                       risk_metrics: RiskMetrics) -> None:
        """Update or add a new position to the risk manager's tracking."""
        self.open_positions[signal.asset] = {
            'entry_price': signal.entry_price,
            'position_size': position_size.adjusted_size,
            'risk_amount': position_size.risk_amount,
            'stop_loss': signal.stop_loss,
            'target_price': signal.target_price,
            'risk_metrics': risk_metrics,
            'timestamp': pd.Timestamp.now()
        }
    
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """Calculate aggregate portfolio risk metrics."""
        if not self.open_positions:
            return {
                'total_risk_amount': 0.0,
                'total_risk_percentage': 0.0,
                'largest_position_risk': 0.0,
                'risk_concentration': 0.0,
                'portfolio_heat': 0.0
            }
        
        total_risk = sum(pos['risk_amount'] for pos in self.open_positions.values())
        max_risk = max(pos['risk_amount'] for pos in self.open_positions.values())
        risk_concentration = max_risk / total_risk if total_risk > 0 else 0
        
        return {
            'total_risk_amount': total_risk,
            'total_risk_percentage': total_risk / self.account_size,
            'largest_position_risk': max_risk,
            'risk_concentration': risk_concentration,
            'portfolio_heat': total_risk / (self.account_size * self.max_total_risk)
        }
