"""
Tests for Unified Dashboard Service.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import json

from src.services.unified_dashboard import (
    UnifiedDashboardService,
    SignalCategory,
    AlertLevel,
    DashboardSignal,
    DashboardAlert,
    MarketOverview
)
from src.config.config import Config


class TestUnifiedDashboardService:
    """Test cases for UnifiedDashboardService class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def dashboard_service(self, config):
        """Create dashboard service instance."""
        return UnifiedDashboardService(config)
    
    @pytest.fixture
    def mock_data_service(self):
        """Create mock data service."""
        mock = Mock()
        
        # Mock market data
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
            'open': np.random.uniform(40000, 42000, 100),
            'high': np.random.uniform(41000, 43000, 100),
            'low': np.random.uniform(39000, 41000, 100),
            'close': np.random.uniform(40000, 42000, 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        mock.get_latest_data.return_value = sample_data
        return mock
    
    @pytest.fixture
    def mock_signal_integration(self):
        """Create mock signal integration service."""
        mock = Mock()
        
        # Mock integrated signals
        mock.get_integrated_signals.return_value = {
            'signals': {
                'individual_signals': {
                    'rsi': {
                        'type': 'bullish',
                        'strength': 75,
                        'confidence': 80,
                        'message': 'RSI oversold bounce'
                    },
                    'volume': {
                        'type': 'bullish',
                        'strength': 65,
                        'confidence': 70,
                        'message': 'Volume surge detected'
                    },
                    'funding_rate': {
                        'type': 'bearish',
                        'strength': 60,
                        'confidence': 65,
                        'message': 'Extreme positive funding'
                    }
                }
            },
            'combined_signal': {
                'type': 'bullish',
                'strength': 70,
                'confidence': 75
            }
        }
        
        return mock
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample dashboard signals."""
        return [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='rsi',
                signal_type='bullish',
                strength=75,
                confidence=80,
                message='RSI oversold bounce',
                timestamp=datetime.now(),
                metadata={'symbol': 'BTC/USDT'}
            ),
            DashboardSignal(
                category=SignalCategory.VOLUME,
                source='volume',
                signal_type='bullish',
                strength=65,
                confidence=70,
                message='Volume surge detected',
                timestamp=datetime.now(),
                metadata={'symbol': 'BTC/USDT'}
            ),
            DashboardSignal(
                category=SignalCategory.DERIVATIVES,
                source='funding_rate',
                signal_type='bearish',
                strength=60,
                confidence=65,
                message='Extreme positive funding',
                timestamp=datetime.now(),
                metadata={'symbol': 'BTC/USDT'}
            )
        ]
    
    def test_initialization(self, dashboard_service):
        """Test dashboard service initialization."""
        assert dashboard_service.config is not None
        assert dashboard_service.data_service is not None
        assert dashboard_service.market_analysis is not None
        assert dashboard_service.signal_integration is not None
        assert dashboard_service.active_signals == []
        assert dashboard_service.active_alerts == []
        assert dashboard_service.is_running is False
    
    def test_get_dashboard_data(self, dashboard_service, mock_data_service, mock_signal_integration):
        """Test getting dashboard data."""
        # Patch services
        dashboard_service.data_service = mock_data_service
        dashboard_service.signal_integration = mock_signal_integration
        
        # Get dashboard data
        symbols = ['BTC/USDT', 'ETH/USDT']
        data = dashboard_service.get_dashboard_data(symbols)
        
        # Verify structure
        assert 'timestamp' in data
        assert 'market_overview' in data
        assert 'active_signals' in data
        assert 'alerts' in data
        assert 'signal_summary' in data
        assert 'recommendations' in data
        assert 'performance' in data
        assert 'correlations' in data
        
        # Verify data was fetched
        assert mock_data_service.get_latest_data.called
        assert mock_signal_integration.get_integrated_signals.called
    
    def test_market_overview(self, dashboard_service, mock_data_service):
        """Test market overview generation."""
        dashboard_service.data_service = mock_data_service
        
        symbols = ['BTC/USDT']
        overviews = dashboard_service._get_market_overview(symbols)
        
        assert len(overviews) == 1
        overview = overviews[0]
        
        assert isinstance(overview, MarketOverview)
        assert overview.symbol == 'BTC/USDT'
        assert overview.price > 0
        assert overview.trend in ['uptrend', 'downtrend', 'sideways']
        assert 0 <= overview.volatility <= 100
        assert 0 <= overview.sentiment_score <= 1
    
    def test_get_active_signals(self, dashboard_service, mock_signal_integration):
        """Test getting active signals."""
        dashboard_service.signal_integration = mock_signal_integration
        
        symbols = ['BTC/USDT']
        signals = dashboard_service._get_active_signals(symbols, '1h')
        
        assert len(signals) > 0
        assert all(isinstance(s, DashboardSignal) for s in signals)
        
        # Check signals are sorted by strength * confidence
        strengths = [s.strength * s.confidence for s in signals]
        assert strengths == sorted(strengths, reverse=True)
    
    def test_convert_to_dashboard_signals(self, dashboard_service):
        """Test signal conversion."""
        analysis_result = {
            'signals': {
                'individual_signals': {
                    'rsi': {
                        'type': 'bullish',
                        'strength': 75,
                        'confidence': 80,
                        'message': 'RSI oversold'
                    }
                }
            },
            'combined_signal': {
                'type': 'bullish',
                'strength': 70,
                'confidence': 75
            }
        }
        
        signals = dashboard_service._convert_to_dashboard_signals(analysis_result, 'BTC/USDT')
        
        assert len(signals) == 2  # Individual + combined
        assert signals[0].source == 'rsi'
        assert signals[0].signal_type == 'bullish'
        assert signals[1].source == 'combined'
    
    def test_categorize_signal_source(self, dashboard_service):
        """Test signal source categorization."""
        assert dashboard_service._categorize_signal_source('rsi') == SignalCategory.TECHNICAL
        assert dashboard_service._categorize_signal_source('volume_analysis') == SignalCategory.VOLUME
        assert dashboard_service._categorize_signal_source('funding_rate') == SignalCategory.DERIVATIVES
        assert dashboard_service._categorize_signal_source('whale_movements') == SignalCategory.ONCHAIN
        assert dashboard_service._categorize_signal_source('cpi_impact') == SignalCategory.MACRO
        assert dashboard_service._categorize_signal_source('sentiment_score') == SignalCategory.SENTIMENT
        assert dashboard_service._categorize_signal_source('unknown') == SignalCategory.TECHNICAL
    
    def test_get_current_alerts(self, dashboard_service, sample_signals):
        """Test alert generation."""
        dashboard_service.active_signals = sample_signals
        
        alerts = dashboard_service._get_current_alerts()
        
        assert len(alerts) > 0
        assert all(isinstance(a, DashboardAlert) for a in alerts)
        
        # Check for high strength signal alert
        strong_signal_alerts = [a for a in alerts if 'Strong' in a.title]
        assert len(strong_signal_alerts) > 0
    
    def test_check_signal_convergence(self, dashboard_service):
        """Test signal convergence detection."""
        # Create converging signals
        signals = [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='rsi',
                signal_type='bullish',
                strength=75,
                confidence=80,
                message='',
                timestamp=datetime.now()
            ),
            DashboardSignal(
                category=SignalCategory.VOLUME,
                source='volume',
                signal_type='bullish',
                strength=70,
                confidence=75,
                message='',
                timestamp=datetime.now()
            ),
            DashboardSignal(
                category=SignalCategory.DERIVATIVES,
                source='oi',
                signal_type='bullish',
                strength=65,
                confidence=70,
                message='',
                timestamp=datetime.now()
            )
        ]
        
        dashboard_service.active_signals = signals
        alerts = dashboard_service._check_signal_convergence()
        
        assert len(alerts) > 0
        assert alerts[0].level == AlertLevel.CRITICAL
        assert 'Convergence' in alerts[0].title
    
    def test_check_signal_divergence(self, dashboard_service):
        """Test signal divergence detection."""
        # Create diverging signals
        signals = [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='rsi',
                signal_type='bullish',
                strength=75,
                confidence=80,
                message='',
                timestamp=datetime.now()
            ),
            DashboardSignal(
                category=SignalCategory.VOLUME,
                source='volume',
                signal_type='bearish',
                strength=70,
                confidence=75,
                message='',
                timestamp=datetime.now()
            )
        ]
        
        dashboard_service.active_signals = signals
        alerts = dashboard_service._check_signal_divergence()
        
        assert len(alerts) > 0
        assert alerts[0].level == AlertLevel.WARNING
        assert 'Divergence' in alerts[0].title
    
    def test_get_signal_summary(self, dashboard_service, sample_signals):
        """Test signal summary generation."""
        dashboard_service.active_signals = sample_signals
        
        summary = dashboard_service._get_signal_summary()
        
        assert summary['total_signals'] == len(sample_signals)
        assert 'by_type' in summary
        assert 'by_category' in summary
        assert summary['average_strength'] > 0
        assert summary['average_confidence'] > 0
        assert summary['strongest_signal'] is not None
        assert summary['distribution']['bullish'] == 2
        assert summary['distribution']['bearish'] == 1
    
    def test_generate_recommendations(self, dashboard_service, sample_signals):
        """Test recommendation generation."""
        dashboard_service.active_signals = sample_signals
        
        recommendations = dashboard_service._generate_recommendations(['BTC/USDT'])
        
        assert len(recommendations) == 1
        rec = recommendations[0]
        
        assert rec['symbol'] == 'BTC/USDT'
        assert rec['action'] in ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL']
        assert rec['confidence'] in ['HIGH', 'MEDIUM', 'LOW']
        assert rec['risk_level'] in ['LOW', 'MEDIUM', 'MEDIUM-HIGH', 'HIGH']
        assert 'position_size' in rec
        assert 'key_factors' in rec
    
    def test_calculate_weighted_signal_score(self, dashboard_service, sample_signals):
        """Test weighted signal score calculation."""
        score = dashboard_service._calculate_weighted_signal_score(sample_signals)
        
        assert -1 <= score <= 1
        
        # Test with all bullish signals
        bullish_signals = [s for s in sample_signals if s.signal_type == 'bullish']
        bullish_score = dashboard_service._calculate_weighted_signal_score(bullish_signals)
        assert bullish_score > 0
        
        # Test with all bearish signals
        bearish_signals = [s for s in sample_signals if s.signal_type == 'bearish']
        bearish_score = dashboard_service._calculate_weighted_signal_score(bearish_signals)
        assert bearish_score < 0
    
    def test_assess_risk_level(self, dashboard_service, sample_signals):
        """Test risk level assessment."""
        risk = dashboard_service._assess_risk_level(sample_signals)
        assert risk in ['LOW', 'MEDIUM', 'MEDIUM-HIGH', 'HIGH']
        
        # Test with high divergence
        divergent_signals = sample_signals + [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='macd',
                signal_type='bearish',
                strength=80,
                confidence=85,
                message='',
                timestamp=datetime.now()
            )
        ]
        
        high_risk = dashboard_service._assess_risk_level(divergent_signals)
        assert high_risk == 'HIGH'
    
    def test_suggest_position_size(self, dashboard_service):
        """Test position size suggestion."""
        # Test various scenarios
        assert dashboard_service._suggest_position_size(0.8, 'LOW') == 'FULL'
        assert dashboard_service._suggest_position_size(0.5, 'MEDIUM') in ['MEDIUM (50%)', 'LARGE (75%)']
        assert dashboard_service._suggest_position_size(0.3, 'HIGH') in ['MINIMAL (10%)', 'SMALL (25%)']
    
    def test_analyze_signal_correlations(self, dashboard_service, sample_signals):
        """Test signal correlation analysis."""
        dashboard_service.active_signals = sample_signals
        
        correlations = dashboard_service._analyze_signal_correlations()
        
        assert 'correlations' in correlations
        assert 'insights' in correlations
        assert isinstance(correlations['correlations'], list)
        assert isinstance(correlations['insights'], list)
    
    def test_calculate_signal_agreement(self, dashboard_service):
        """Test signal agreement calculation."""
        # Same type signals
        bullish_signals = [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='rsi',
                signal_type='bullish',
                strength=75,
                confidence=80,
                message='',
                timestamp=datetime.now()
            )
        ]
        
        agreement = dashboard_service._calculate_signal_agreement(bullish_signals, bullish_signals)
        assert agreement > 0
        
        # Opposite signals
        bearish_signals = [
            DashboardSignal(
                category=SignalCategory.TECHNICAL,
                source='macd',
                signal_type='bearish',
                strength=75,
                confidence=80,
                message='',
                timestamp=datetime.now()
            )
        ]
        
        disagreement = dashboard_service._calculate_signal_agreement(bullish_signals, bearish_signals)
        assert disagreement == -1
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, dashboard_service):
        """Test monitoring start/stop."""
        # Mock get_dashboard_data
        dashboard_service.get_dashboard_data = Mock(return_value={
            'signal_summary': {'total_signals': 5},
            'alerts': []
        })
        
        # Set short update interval for testing
        dashboard_service.update_interval = 0.1
        
        # Start monitoring
        monitor_task = asyncio.create_task(
            dashboard_service.start_monitoring(['BTC/USDT'])
        )
        
        # Let it run briefly
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        dashboard_service.stop_monitoring()
        
        # Wait for task to complete
        await asyncio.sleep(0.2)
        
        assert dashboard_service.is_running is False
        assert dashboard_service.get_dashboard_data.called
    
    def test_export_dashboard_data_json(self, dashboard_service, sample_signals):
        """Test JSON export."""
        dashboard_data = {
            'timestamp': datetime.now(),
            'active_signals': sample_signals,
            'alerts': [],
            'market_overview': []
        }
        
        json_export = dashboard_service.export_dashboard_data(dashboard_data, 'json')
        
        # Verify it's valid JSON
        parsed = json.loads(json_export)
        assert 'timestamp' in parsed
        assert 'active_signals' in parsed
    
    def test_export_dashboard_data_csv(self, dashboard_service, sample_signals):
        """Test CSV export."""
        dashboard_data = {
            'active_signals': sample_signals
        }
        
        csv_export = dashboard_service.export_dashboard_data(dashboard_data, 'csv')
        
        # Verify CSV format
        assert 'type,category,source,signal_type' in csv_export
        assert 'signal,technical,rsi,bullish' in csv_export
    
    def test_export_dashboard_data_html(self, dashboard_service):
        """Test HTML export."""
        dashboard_data = {
            'timestamp': datetime.now(),
            'market_overview': [
                MarketOverview(
                    symbol='BTC/USDT',
                    price=42000,
                    price_change_24h=2.5,
                    volume_24h=1000000,
                    market_cap=0,
                    dominance=0,
                    trend='uptrend',
                    volatility=2.5,
                    sentiment_score=0.65
                )
            ],
            'alerts': [
                DashboardAlert(
                    level=AlertLevel.WARNING,
                    category=SignalCategory.TECHNICAL,
                    title='Test Alert',
                    message='Test message',
                    timestamp=datetime.now()
                )
            ],
            'recommendations': [
                {
                    'symbol': 'BTC/USDT',
                    'action': 'BUY',
                    'confidence': 'HIGH',
                    'risk_level': 'MEDIUM',
                    'position_size': 'MEDIUM (50%)'
                }
            ]
        }
        
        html_export = dashboard_service.export_dashboard_data(dashboard_data, 'html')
        
        # Verify HTML structure
        assert '<html>' in html_export
        assert '<table>' in html_export
        assert 'BTC/USDT' in html_export
        assert 'Test Alert' in html_export
        assert 'BUY' in html_export
    
    def test_export_invalid_format(self, dashboard_service):
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            dashboard_service.export_dashboard_data({}, 'invalid')
    
    def test_update_signal_history(self, dashboard_service, sample_signals):
        """Test signal history update."""
        dashboard_service.active_signals = sample_signals
        
        # Update history
        dashboard_service._update_signal_history()
        
        assert len(dashboard_service.signal_history) == len(sample_signals)
        
        # Test max history limit
        dashboard_service.signal_history = [Mock()] * 1000
        dashboard_service._update_signal_history()
        
        assert len(dashboard_service.signal_history) == 1000
    
    def test_performance_metrics(self, dashboard_service):
        """Test performance metrics generation."""
        metrics = dashboard_service._get_performance_metrics()
        
        assert 'signal_accuracy' in metrics
        assert 'alert_accuracy' in metrics
        assert 'recommendation_performance' in metrics
        assert 'system_uptime' in metrics
        assert 'last_updated' in metrics
        
        assert 0 <= metrics['signal_accuracy'] <= 1
        assert 0 <= metrics['alert_accuracy'] <= 1
        assert metrics['system_uptime'] > 0
    
    def test_edge_cases(self, dashboard_service):
        """Test edge cases."""
        # Empty signals
        assert dashboard_service._calculate_weighted_signal_score([]) == 0
        assert dashboard_service._assess_risk_level([]) == 'MEDIUM'
        
        # Empty market data
        dashboard_service.data_service = Mock()
        dashboard_service.data_service.get_latest_data.return_value = pd.DataFrame()
        
        overviews = dashboard_service._get_market_overview(['BTC/USDT'])
        assert len(overviews) == 0
        
        # No active signals
        dashboard_service.active_signals = []
        summary = dashboard_service._get_signal_summary()
        assert summary['total_signals'] == 0
        assert summary['strongest_signal'] is None
