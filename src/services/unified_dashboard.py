"""
Unified Dashboard Service.

This module provides the UnifiedDashboardService class that integrates all
signal analyzers into a comprehensive trading dashboard with real-time
monitoring and alert capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import logging
from enum import Enum

from src.services.data_service import DataService
from src.services.market_analysis import MarketAnalysisService
from src.services.signal_integration import SignalIntegrationService
from src.utils.performance import timer, async_timer
from src.utils.error_handling import with_retry, safe_execute
from src.config.config import Config

logger = logging.getLogger(__name__)


class SignalCategory(Enum):
    """Signal category enumeration."""
    TECHNICAL = "technical"
    VOLUME = "volume"
    DERIVATIVES = "derivatives"
    ONCHAIN = "onchain"
    MACRO = "macro"
    SENTIMENT = "sentiment"


class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DashboardSignal:
    """Data class for dashboard signals."""
    category: SignalCategory
    source: str
    signal_type: str  # bullish, bearish, neutral
    strength: float
    confidence: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardAlert:
    """Data class for dashboard alerts."""
    level: AlertLevel
    category: SignalCategory
    title: str
    message: str
    timestamp: datetime
    action_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketOverview:
    """Data class for market overview."""
    symbol: str
    price: float
    price_change_24h: float
    volume_24h: float
    market_cap: float
    dominance: float
    trend: str  # uptrend, downtrend, sideways
    volatility: float
    sentiment_score: float


class UnifiedDashboardService:
    """
    Service for unified trading dashboard.
    
    This service integrates all signal analyzers and provides:
    - Real-time signal aggregation
    - Multi-timeframe analysis
    - Alert generation and management
    - Signal correlation analysis
    - Trading recommendations
    - Performance tracking
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Unified Dashboard Service.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize services
        self.data_service = DataService(config)
        self.market_analysis = MarketAnalysisService(config)
        self.signal_integration = SignalIntegrationService(config)
        
        # Dashboard state
        self.active_signals: List[DashboardSignal] = []
        self.active_alerts: List[DashboardAlert] = []
        self.signal_history: List[DashboardSignal] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configuration
        self.alert_thresholds = {
            'signal_strength': 70,  # Minimum strength for alerts
            'confidence': 60,       # Minimum confidence for alerts
            'correlation': 0.7      # Minimum correlation for related signals
        }
        
        # Signal weights by category
        self.category_weights = {
            SignalCategory.TECHNICAL: 0.20,
            SignalCategory.VOLUME: 0.20,
            SignalCategory.DERIVATIVES: 0.25,
            SignalCategory.ONCHAIN: 0.20,
            SignalCategory.MACRO: 0.10,
            SignalCategory.SENTIMENT: 0.05
        }
        
        # Monitoring
        self.is_running = False
        self.update_interval = 60  # seconds
    
    @timer
    def get_dashboard_data(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for specified symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for analysis
        
        Returns:
            Dictionary containing dashboard data
        """
        try:
            dashboard_data = {
                'timestamp': datetime.now(),
                'market_overview': self._get_market_overview(symbols),
                'active_signals': self._get_active_signals(symbols, timeframe),
                'alerts': self._get_current_alerts(),
                'signal_summary': self._get_signal_summary(),
                'recommendations': self._generate_recommendations(symbols),
                'performance': self._get_performance_metrics(),
                'correlations': self._analyze_signal_correlations()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {str(e)}")
            raise
    
    def _get_market_overview(self, symbols: List[str]) -> List[MarketOverview]:
        """Get market overview for symbols."""
        overviews = []
        
        for symbol in symbols:
            try:
                # Get market data
                market_data = self.data_service.get_latest_data(symbol, limit=100)
                
                if market_data is None or len(market_data) == 0:
                    continue
                
                # Calculate metrics
                current_price = market_data['close'].iloc[-1]
                price_24h_ago = market_data['close'].iloc[0] if len(market_data) >= 24 else current_price
                price_change = ((current_price - price_24h_ago) / price_24h_ago) * 100
                
                volume_24h = market_data['volume'].sum()
                volatility = market_data['close'].pct_change().std() * 100
                
                # Determine trend
                sma_20 = market_data['close'].rolling(20).mean().iloc[-1]
                sma_50 = market_data['close'].rolling(50).mean().iloc[-1] if len(market_data) >= 50 else sma_20
                
                if current_price > sma_20 > sma_50:
                    trend = 'uptrend'
                elif current_price < sma_20 < sma_50:
                    trend = 'downtrend'
                else:
                    trend = 'sideways'
                
                # Get sentiment (would come from sentiment analysis in real implementation)
                sentiment_score = np.random.uniform(0.3, 0.7)  # Placeholder
                
                overview = MarketOverview(
                    symbol=symbol,
                    price=current_price,
                    price_change_24h=price_change,
                    volume_24h=volume_24h,
                    market_cap=0,  # Would need external data
                    dominance=0,   # Would need external data
                    trend=trend,
                    volatility=volatility,
                    sentiment_score=sentiment_score
                )
                
                overviews.append(overview)
                
            except Exception as e:
                logger.error(f"Error getting overview for {symbol}: {str(e)}")
                continue
        
        return overviews
    
    def _get_active_signals(self, symbols: List[str], timeframe: str) -> List[DashboardSignal]:
        """Get active signals from all analyzers."""
        all_signals = []
        
        for symbol in symbols:
            try:
                # Get signals from signal integration service
                analysis_result = self.signal_integration.get_integrated_signals(
                    symbol=symbol,
                    timeframe=timeframe
                )
                
                if not analysis_result:
                    continue
                
                # Convert to dashboard signals
                signals = self._convert_to_dashboard_signals(analysis_result, symbol)
                all_signals.extend(signals)
                
            except Exception as e:
                logger.error(f"Error getting signals for {symbol}: {str(e)}")
                continue
        
        # Filter and sort by strength
        self.active_signals = sorted(
            all_signals,
            key=lambda x: x.strength * x.confidence,
            reverse=True
        )
        
        return self.active_signals
    
    def _convert_to_dashboard_signals(self, analysis_result: Dict[str, Any], symbol: str) -> List[DashboardSignal]:
        """Convert analysis results to dashboard signals."""
        dashboard_signals = []
        
        # Extract individual signals
        if 'signals' in analysis_result and 'individual_signals' in analysis_result['signals']:
            for source, signal_data in analysis_result['signals']['individual_signals'].items():
                if isinstance(signal_data, dict) and 'type' in signal_data:
                    # Determine category
                    category = self._categorize_signal_source(source)
                    
                    signal = DashboardSignal(
                        category=category,
                        source=source,
                        signal_type=signal_data['type'],
                        strength=signal_data.get('strength', 0),
                        confidence=signal_data.get('confidence', 0),
                        message=signal_data.get('message', f"{source} signal"),
                        timestamp=datetime.now(),
                        metadata={
                            'symbol': symbol,
                            'raw_data': signal_data
                        }
                    )
                    
                    dashboard_signals.append(signal)
        
        # Add combined signal
        if 'combined_signal' in analysis_result:
            combined = analysis_result['combined_signal']
            signal = DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='combined',
                signal_type=combined.get('type', 'neutral'),
                strength=combined.get('strength', 0),
                confidence=combined.get('confidence', 0),
                message=f"Combined signal for {symbol}",
                timestamp=datetime.now(),
                metadata={
                    'symbol': symbol,
                    'components': len(dashboard_signals)
                }
            )
            dashboard_signals.append(signal)
        
        return dashboard_signals
    
    def _categorize_signal_source(self, source: str) -> SignalCategory:
        """Categorize signal source into signal category."""
        source_lower = source.lower()
        
        if any(term in source_lower for term in ['rsi', 'sma', 'ema', 'macd', 'bollinger']):
            return SignalCategory.TECHNICAL
        elif any(term in source_lower for term in ['volume', 'cvd', 'delta']):
            return SignalCategory.VOLUME
        elif any(term in source_lower for term in ['oi', 'liquidation', 'funding', 'gamma']):
            return SignalCategory.DERIVATIVES
        elif any(term in source_lower for term in ['onchain', 'whale', 'exchange_flow']):
            return SignalCategory.ONCHAIN
        elif any(term in source_lower for term in ['cpi', 'macro', 'fomc']):
            return SignalCategory.MACRO
        elif any(term in source_lower for term in ['sentiment', 'social']):
            return SignalCategory.SENTIMENT
        else:
            return SignalCategory.TECHNICAL
    
    def _get_current_alerts(self) -> List[DashboardAlert]:
        """Get current active alerts."""
        alerts = []
        
        # Generate alerts based on signals
        for signal in self.active_signals:
            # High strength signals
            if signal.strength >= self.alert_thresholds['signal_strength']:
                level = AlertLevel.CRITICAL if signal.strength >= 85 else AlertLevel.WARNING
                
                alert = DashboardAlert(
                    level=level,
                    category=signal.category,
                    title=f"Strong {signal.signal_type.upper()} Signal",
                    message=f"{signal.source}: {signal.message} (Strength: {signal.strength:.1f}%)",
                    timestamp=signal.timestamp,
                    action_required=level == AlertLevel.CRITICAL,
                    metadata=signal.metadata
                )
                alerts.append(alert)
        
        # Check for signal convergence
        convergence_alerts = self._check_signal_convergence()
        alerts.extend(convergence_alerts)
        
        # Check for divergence warnings
        divergence_alerts = self._check_signal_divergence()
        alerts.extend(divergence_alerts)
        
        self.active_alerts = sorted(alerts, key=lambda x: (x.level.value, x.timestamp), reverse=True)
        return self.active_alerts
    
    def _check_signal_convergence(self) -> List[DashboardAlert]:
        """Check for signal convergence across categories."""
        alerts = []
        
        # Group signals by type
        signal_groups = defaultdict(list)
        for signal in self.active_signals:
            if signal.strength >= 50:  # Only consider meaningful signals
                signal_groups[signal.signal_type].append(signal)
        
        # Check for convergence
        for signal_type, signals in signal_groups.items():
            if len(signals) >= 3:  # At least 3 agreeing signals
                categories = set(s.category for s in signals)
                avg_strength = np.mean([s.strength for s in signals])
                
                if len(categories) >= 2:  # From different categories
                    alert = DashboardAlert(
                        level=AlertLevel.CRITICAL,
                        category=SignalCategory.TECHNICAL,
                        title=f"Signal Convergence Detected",
                        message=f"Multiple {signal_type} signals across {len(categories)} categories (Avg strength: {avg_strength:.1f}%)",
                        timestamp=datetime.now(),
                        action_required=True,
                        metadata={
                            'signal_count': len(signals),
                            'categories': list(categories),
                            'sources': [s.source for s in signals]
                        }
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _check_signal_divergence(self) -> List[DashboardAlert]:
        """Check for signal divergence warnings."""
        alerts = []
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for s in self.active_signals if s.signal_type == 'bullish' and s.strength >= 50)
        bearish_count = sum(1 for s in self.active_signals if s.signal_type == 'bearish' and s.strength >= 50)
        
        if bullish_count > 0 and bearish_count > 0:
            total_signals = bullish_count + bearish_count
            if min(bullish_count, bearish_count) / total_signals >= 0.3:  # Significant divergence
                alert = DashboardAlert(
                    level=AlertLevel.WARNING,
                    category=SignalCategory.TECHNICAL,
                    title="Signal Divergence Warning",
                    message=f"Mixed signals detected: {bullish_count} bullish vs {bearish_count} bearish",
                    timestamp=datetime.now(),
                    action_required=False,
                    metadata={
                        'bullish_count': bullish_count,
                        'bearish_count': bearish_count
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _get_signal_summary(self) -> Dict[str, Any]:
        """Get summary of current signals."""
        summary = {
            'total_signals': len(self.active_signals),
            'by_type': defaultdict(int),
            'by_category': defaultdict(int),
            'average_strength': 0,
            'average_confidence': 0,
            'strongest_signal': None,
            'distribution': {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0
            }
        }
        
        if not self.active_signals:
            return summary
        
        # Calculate statistics
        strengths = []
        confidences = []
        
        for signal in self.active_signals:
            summary['by_type'][signal.signal_type] += 1
            summary['by_category'][signal.category.value] += 1
            summary['distribution'][signal.signal_type] += 1
            
            strengths.append(signal.strength)
            confidences.append(signal.confidence)
        
        summary['average_strength'] = np.mean(strengths)
        summary['average_confidence'] = np.mean(confidences)
        
        # Find strongest signal
        strongest = max(self.active_signals, key=lambda x: x.strength * x.confidence)
        summary['strongest_signal'] = {
            'source': strongest.source,
            'type': strongest.signal_type,
            'strength': strongest.strength,
            'message': strongest.message
        }
        
        return summary
    
    def _generate_recommendations(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on signals."""
        recommendations = []
        
        # Group signals by symbol
        symbol_signals = defaultdict(list)
        for signal in self.active_signals:
            if 'symbol' in signal.metadata:
                symbol_signals[signal.metadata['symbol']].append(signal)
        
        for symbol in symbols:
            signals = symbol_signals.get(symbol, [])
            if not signals:
                continue
            
            # Calculate weighted signal
            weighted_score = self._calculate_weighted_signal_score(signals)
            
            # Generate recommendation
            if weighted_score > 0.6:
                action = 'STRONG BUY'
                confidence = 'HIGH'
            elif weighted_score > 0.3:
                action = 'BUY'
                confidence = 'MEDIUM'
            elif weighted_score > -0.3:
                action = 'HOLD'
                confidence = 'LOW'
            elif weighted_score > -0.6:
                action = 'SELL'
                confidence = 'MEDIUM'
            else:
                action = 'STRONG SELL'
                confidence = 'HIGH'
            
            # Risk assessment
            risk_level = self._assess_risk_level(signals)
            
            # Position sizing suggestion
            position_size = self._suggest_position_size(weighted_score, risk_level)
            
            recommendation = {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'weighted_score': weighted_score,
                'risk_level': risk_level,
                'position_size': position_size,
                'key_factors': self._get_key_factors(signals),
                'timestamp': datetime.now()
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_weighted_signal_score(self, signals: List[DashboardSignal]) -> float:
        """Calculate weighted signal score."""
        if not signals:
            return 0
        
        total_weight = 0
        weighted_sum = 0
        
        for signal in signals:
            # Get category weight
            weight = self.category_weights.get(signal.category, 0.1)
            
            # Adjust weight by confidence
            weight *= (signal.confidence / 100)
            
            # Calculate signal value (-1 to 1)
            if signal.signal_type == 'bullish':
                value = signal.strength / 100
            elif signal.signal_type == 'bearish':
                value = -signal.strength / 100
            else:
                value = 0
            
            weighted_sum += weight * value
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _assess_risk_level(self, signals: List[DashboardSignal]) -> str:
        """Assess risk level based on signals."""
        # Check for divergence
        signal_types = [s.signal_type for s in signals if s.strength >= 50]
        has_divergence = 'bullish' in signal_types and 'bearish' in signal_types
        
        # Check volatility signals
        volatility_signals = [s for s in signals if 'volatility' in s.source.lower()]
        high_volatility = any(s.strength > 70 for s in volatility_signals)
        
        # Check confidence levels
        avg_confidence = np.mean([s.confidence for s in signals]) if signals else 0
        
        if has_divergence or high_volatility:
            return 'HIGH'
        elif avg_confidence < 50:
            return 'MEDIUM-HIGH'
        elif avg_confidence > 70:
            return 'LOW'
        else:
            return 'MEDIUM'
    
    def _suggest_position_size(self, weighted_score: float, risk_level: str) -> str:
        """Suggest position size based on signal and risk."""
        base_size = abs(weighted_score)
        
        # Adjust for risk
        risk_multipliers = {
            'LOW': 1.2,
            'MEDIUM': 1.0,
            'MEDIUM-HIGH': 0.7,
            'HIGH': 0.5
        }
        
        adjusted_size = base_size * risk_multipliers.get(risk_level, 0.8)
        
        if adjusted_size > 0.8:
            return 'FULL'
        elif adjusted_size > 0.6:
            return 'LARGE (75%)'
        elif adjusted_size > 0.4:
            return 'MEDIUM (50%)'
        elif adjusted_size > 0.2:
            return 'SMALL (25%)'
        else:
            return 'MINIMAL (10%)'
    
    def _get_key_factors(self, signals: List[DashboardSignal]) -> List[str]:
        """Get key factors from signals."""
        # Sort by importance (strength * confidence)
        sorted_signals = sorted(
            signals,
            key=lambda x: x.strength * x.confidence,
            reverse=True
        )
        
        # Get top factors
        factors = []
        for signal in sorted_signals[:5]:  # Top 5
            factor = f"{signal.source}: {signal.message} ({signal.strength:.0f}%)"
            factors.append(factor)
        
        return factors
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the dashboard."""
        # This would track actual trading performance in production
        return {
            'signal_accuracy': self._calculate_signal_accuracy(),
            'alert_accuracy': self._calculate_alert_accuracy(),
            'recommendation_performance': self._calculate_recommendation_performance(),
            'system_uptime': self._calculate_system_uptime(),
            'last_updated': datetime.now()
        }
    
    def _calculate_signal_accuracy(self) -> float:
        """Calculate historical signal accuracy."""
        # Placeholder - would analyze historical signals vs outcomes
        return np.random.uniform(0.6, 0.8)
    
    def _calculate_alert_accuracy(self) -> float:
        """Calculate alert accuracy."""
        # Placeholder - would analyze historical alerts
        return np.random.uniform(0.7, 0.9)
    
    def _calculate_recommendation_performance(self) -> Dict[str, float]:
        """Calculate recommendation performance."""
        # Placeholder - would track actual recommendations
        return {
            'win_rate': np.random.uniform(0.5, 0.7),
            'avg_return': np.random.uniform(-0.02, 0.05),
            'sharpe_ratio': np.random.uniform(0.5, 2.0)
        }
    
    def _calculate_system_uptime(self) -> float:
        """Calculate system uptime percentage."""
        # Placeholder
        return 99.9
    
    def _analyze_signal_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different signals."""
        if len(self.active_signals) < 2:
            return {'correlations': [], 'insights': []}
        
        # Group signals by source
        signal_groups = defaultdict(list)
        for signal in self.active_signals:
            signal_groups[signal.source].append(signal)
        
        correlations = []
        insights = []
        
        # Analyze pairwise correlations
        sources = list(signal_groups.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                
                # Check if signals agree
                signals1 = signal_groups[source1]
                signals2 = signal_groups[source2]
                
                agreement = self._calculate_signal_agreement(signals1, signals2)
                
                if agreement > self.alert_thresholds['correlation']:
                    correlations.append({
                        'source1': source1,
                        'source2': source2,
                        'correlation': agreement,
                        'interpretation': 'Strong Agreement'
                    })
                    
                    insights.append(
                        f"{source1} and {source2} show strong agreement ({agreement:.1%})"
                    )
                elif agreement < -self.alert_thresholds['correlation']:
                    correlations.append({
                        'source1': source1,
                        'source2': source2,
                        'correlation': agreement,
                        'interpretation': 'Strong Disagreement'
                    })
                    
                    insights.append(
                        f"Warning: {source1} and {source2} show conflicting signals"
                    )
        
        return {
            'correlations': correlations,
            'insights': insights
        }
    
    def _calculate_signal_agreement(self, signals1: List[DashboardSignal], 
                                  signals2: List[DashboardSignal]) -> float:
        """Calculate agreement between two sets of signals."""
        if not signals1 or not signals2:
            return 0
        
        # Compare predominant signal types
        type1 = max(set(s.signal_type for s in signals1), 
                   key=lambda x: sum(s.strength for s in signals1 if s.signal_type == x))
        type2 = max(set(s.signal_type for s in signals2), 
                   key=lambda x: sum(s.strength for s in signals2 if s.signal_type == x))
        
        if type1 == type2:
            # Calculate strength correlation
            avg_strength1 = np.mean([s.strength for s in signals1])
            avg_strength2 = np.mean([s.strength for s in signals2])
            
            # Normalize to -1 to 1
            strength_diff = abs(avg_strength1 - avg_strength2) / 100
            return 1 - strength_diff
        elif (type1 == 'bullish' and type2 == 'bearish') or \
             (type1 == 'bearish' and type2 == 'bullish'):
            return -1
        else:
            return 0
    
    @async_timer
    async def start_monitoring(self, symbols: List[str], 
                             callback: Optional[Callable] = None) -> None:
        """
        Start real-time monitoring of signals.
        
        Args:
            symbols: List of symbols to monitor
            callback: Optional callback function for updates
        """
        self.is_running = True
        logger.info(f"Starting dashboard monitoring for {symbols}")
        
        while self.is_running:
            try:
                # Get latest dashboard data
                dashboard_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.get_dashboard_data, symbols
                )
                
                # Update internal state
                self._update_signal_history()
                
                # Call callback if provided
                if callback:
                    await callback(dashboard_data)
                
                # Log summary
                summary = dashboard_data['signal_summary']
                logger.info(
                    f"Dashboard update: {summary['total_signals']} signals, "
                    f"{len(dashboard_data['alerts'])} alerts"
                )
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_running = False
        logger.info("Stopping dashboard monitoring")
    
    def _update_signal_history(self) -> None:
        """Update signal history for tracking."""
        # Add current signals to history
        self.signal_history.extend(self.active_signals)
        
        # Keep only recent history (e.g., last 1000 signals)
        max_history = 1000
        if len(self.signal_history) > max_history:
            self.signal_history = self.signal_history[-max_history:]
    
    def export_dashboard_data(self, dashboard_data: Dict[str, Any], 
                            format: str = 'json') -> str:
        """
        Export dashboard data in specified format.
        
        Args:
            dashboard_data: Dashboard data to export
            format: Export format ('json', 'csv', 'html')
        
        Returns:
            Exported data as string
        """
        if format == 'json':
            import json
            return json.dumps(dashboard_data, default=str, indent=2)
        
        elif format == 'csv':
            # Convert to CSV format
            rows = []
            
            # Add signals
            for signal in dashboard_data.get('active_signals', []):
                rows.append({
                    'type': 'signal',
                    'category': signal.category.value,
                    'source': signal.source,
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'confidence': signal.confidence,
                    'message': signal.message,
                    'timestamp': signal.timestamp
                })
            
            df = pd.DataFrame(rows)
            return df.to_csv(index=False)
        
        elif format == 'html':
            # Generate HTML report
            return self._generate_html_report(dashboard_data)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_html_report(self, dashboard_data: Dict[str, Any]) -> str:
        """Generate HTML report from dashboard data."""
        html = f"""
        <html>
        <head>
            <title>Trading Dashboard Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .alert-critical {{ background-color: #ffcccc; }}
                .alert-warning {{ background-color: #ffffcc; }}
                .signal-bullish {{ color: green; }}
                .signal-bearish {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Trading Dashboard Report</h1>
            <p>Generated at: {dashboard_data['timestamp']}</p>
            
            <h2>Market Overview</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>24h Change</th>
                    <th>Volume</th>
                    <th>Trend</th>
                </tr>
        """
        
        for overview in dashboard_data.get('market_overview', []):
            html += f"""
                <tr>
                    <td>{overview.symbol}</td>
                    <td>${overview.price:.2f}</td>
                    <td>{overview.price_change_24h:+.2f}%</td>
                    <td>${overview.volume_24h:,.0f}</td>
                    <td>{overview.trend}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Active Alerts</h2>
            <table>
                <tr>
                    <th>Level</th>
                    <th>Title</th>
                    <th>Message</th>
                </tr>
        """
        
        for alert in dashboard_data.get('alerts', []):
            css_class = f"alert-{alert.level.value}"
            html += f"""
                <tr class="{css_class}">
                    <td>{alert.level.value.upper()}</td>
                    <td>{alert.title}</td>
                    <td>{alert.message}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Trading Recommendations</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Confidence</th>
                    <th>Risk Level</th>
                    <th>Position Size</th>
                </tr>
        """
        
        for rec in dashboard_data.get('recommendations', []):
            html += f"""
                <tr>
                    <td>{rec['symbol']}</td>
                    <td>{rec['action']}</td>
                    <td>{rec['confidence']}</td>
                    <td>{rec['risk_level']}</td>
                    <td>{rec['position_size']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
