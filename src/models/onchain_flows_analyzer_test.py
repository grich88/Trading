"""
Tests for On-Chain Flows Analyzer Model.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.models.onchain_flows_analyzer import (
    OnChainFlowsAnalyzer,
    OnChainMetrics,
    analyze_onchain_flows,
    track_whale_movements
)


class TestOnChainFlowsAnalyzer:
    """Test cases for OnChainFlowsAnalyzer class."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data."""
        # Create diverse transaction data
        now = datetime.now()
        transactions = []
        
        # Regular transactions
        for i in range(50):
            transactions.append({
                'timestamp': now - timedelta(hours=i),
                'from_address': f'addr_{i % 10}',
                'to_address': f'addr_{(i + 5) % 15}',
                'amount': np.random.uniform(0.1, 10),
                'tx_hash': f'hash_{i}',
                'block_number': 1000 + i
            })
        
        # Whale transactions
        whale_addresses = ['whale_1', 'whale_2', 'whale_3']
        for i in range(10):
            transactions.append({
                'timestamp': now - timedelta(hours=i * 2),
                'from_address': whale_addresses[i % 3],
                'to_address': f'addr_{i + 20}',
                'amount': np.random.uniform(100, 2000),
                'tx_hash': f'whale_hash_{i}',
                'block_number': 1050 + i
            })
        
        # Exchange transactions
        exchange_addresses = ['exchange_1', 'exchange_2']
        for i in range(20):
            if i % 2 == 0:
                # Inflow to exchange
                transactions.append({
                    'timestamp': now - timedelta(hours=i),
                    'from_address': f'addr_{i + 30}',
                    'to_address': exchange_addresses[i % 2],
                    'amount': np.random.uniform(5, 50),
                    'tx_hash': f'exchange_in_hash_{i}',
                    'block_number': 1070 + i
                })
            else:
                # Outflow from exchange
                transactions.append({
                    'timestamp': now - timedelta(hours=i),
                    'from_address': exchange_addresses[i % 2],
                    'to_address': f'addr_{i + 40}',
                    'amount': np.random.uniform(5, 50),
                    'tx_hash': f'exchange_out_hash_{i}',
                    'block_number': 1070 + i
                })
        
        # Smart money transactions
        smart_money_addresses = ['smart_1', 'smart_2']
        for i in range(5):
            transactions.append({
                'timestamp': now - timedelta(hours=i * 4),
                'from_address': smart_money_addresses[i % 2],
                'to_address': f'addr_{i + 50}',
                'amount': np.random.uniform(50, 200),
                'tx_hash': f'smart_hash_{i}',
                'block_number': 1090 + i
            })
        
        return pd.DataFrame(transactions)
    
    @pytest.fixture
    def analyzer_config(self):
        """Create analyzer configuration."""
        return {
            'whale_threshold': 1000,
            'smart_money_addresses': ['smart_1', 'smart_2'],
            'exchange_addresses': ['exchange_1', 'exchange_2'],
            'flow_threshold': 100,
            'lookback_periods': 30
        }
    
    def test_initialization(self):
        """Test analyzer initialization."""
        # Default initialization
        analyzer = OnChainFlowsAnalyzer()
        assert analyzer.whale_threshold == 1000
        assert analyzer.flow_threshold == 100
        assert analyzer.lookback_periods == 30
        assert len(analyzer.smart_money_addresses) == 0
        assert len(analyzer.exchange_addresses) == 0
        
        # Custom configuration
        config = {
            'whale_threshold': 500,
            'smart_money_addresses': ['addr1', 'addr2'],
            'exchange_addresses': ['ex1', 'ex2'],
            'flow_threshold': 50,
            'lookback_periods': 20
        }
        analyzer = OnChainFlowsAnalyzer(config)
        assert analyzer.whale_threshold == 500
        assert analyzer.flow_threshold == 50
        assert analyzer.lookback_periods == 20
        assert 'addr1' in analyzer.smart_money_addresses
        assert 'ex1' in analyzer.exchange_addresses
    
    def test_validate_data(self, sample_transactions):
        """Test data validation."""
        analyzer = OnChainFlowsAnalyzer()
        
        # Valid data should pass
        analyzer._validate_data(sample_transactions)
        
        # Missing columns
        invalid_data = sample_transactions.drop(columns=['amount'])
        with pytest.raises(ValueError, match="Missing required columns"):
            analyzer._validate_data(invalid_data)
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty dataframe"):
            analyzer._validate_data(empty_df)
        
        # Negative amounts
        invalid_data = sample_transactions.copy()
        invalid_data.loc[0, 'amount'] = -10
        with pytest.raises(ValueError, match="Negative transaction amounts"):
            analyzer._validate_data(invalid_data)
    
    def test_label_addresses(self, sample_transactions, analyzer_config):
        """Test address labeling."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        
        # Check exchange labels
        exchange_txs = labeled_data[
            labeled_data['to_address'].isin(['exchange_1', 'exchange_2'])
        ]
        assert exchange_txs['to_is_exchange'].all()
        
        exchange_txs = labeled_data[
            labeled_data['from_address'].isin(['exchange_1', 'exchange_2'])
        ]
        assert exchange_txs['from_is_exchange'].all()
        
        # Check smart money labels
        smart_txs = labeled_data[
            labeled_data['from_address'].isin(['smart_1', 'smart_2'])
        ]
        assert smart_txs['from_is_smart_money'].all()
        
        # Check whale labels exist
        assert 'from_is_whale' in labeled_data.columns
        assert 'to_is_whale' in labeled_data.columns
    
    def test_calculate_address_balances(self, sample_transactions):
        """Test address balance calculation."""
        analyzer = OnChainFlowsAnalyzer()
        balances = analyzer._calculate_address_balances(sample_transactions)
        
        # Verify balances
        assert isinstance(balances, dict)
        
        # Check specific address balance
        test_addr = 'addr_0'
        received = sample_transactions[
            sample_transactions['to_address'] == test_addr
        ]['amount'].sum()
        sent = sample_transactions[
            sample_transactions['from_address'] == test_addr
        ]['amount'].sum()
        expected_balance = received - sent
        
        assert abs(balances.get(test_addr, 0) - expected_balance) < 0.01
    
    def test_calculate_metrics(self, sample_transactions, analyzer_config):
        """Test metrics calculation."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        metrics = analyzer._calculate_metrics(labeled_data)
        
        assert isinstance(metrics, OnChainMetrics)
        assert metrics.exchange_inflow >= 0
        assert metrics.exchange_outflow >= 0
        assert isinstance(metrics.net_flow, float)
        assert metrics.whale_movements >= 0
        assert isinstance(metrics.smart_money_flow, float)
        assert isinstance(metrics.network_activity_score, float)
    
    def test_analyze_exchange_flows(self, sample_transactions, analyzer_config):
        """Test exchange flow analysis."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        exchange_flows = analyzer._analyze_exchange_flows(labeled_data)
        
        assert 'trend' in exchange_flows
        assert exchange_flows['trend'] in ['accumulation', 'distribution', 'neutral']
        assert 0 <= exchange_flows['strength'] <= 100
        assert 'inflow_volume' in exchange_flows
        assert 'outflow_volume' in exchange_flows
        assert 'net_flow' in exchange_flows
        assert isinstance(exchange_flows['large_deposits'], list)
        assert isinstance(exchange_flows['large_withdrawals'], list)
    
    def test_track_whale_activity(self, sample_transactions, analyzer_config):
        """Test whale activity tracking."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        whale_activity = analyzer._track_whale_activity(labeled_data)
        
        assert 'activity_level' in whale_activity
        assert whale_activity['activity_level'] in ['very_high', 'high', 'moderate', 'low']
        assert whale_activity['accumulation_score'] >= 0
        assert whale_activity['distribution_score'] >= 0
        assert 'net_position_change' in whale_activity
        assert whale_activity['whale_count'] >= 0
        assert 0 <= whale_activity['whale_dominance'] <= 100
        assert whale_activity['average_transaction_size'] >= 0
    
    def test_analyze_smart_money(self, sample_transactions, analyzer_config):
        """Test smart money analysis."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        smart_money = analyzer._analyze_smart_money(labeled_data)
        
        assert 'sentiment' in smart_money
        assert smart_money['sentiment'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= smart_money['conviction'] <= 100
        assert isinstance(smart_money['follow_signal'], bool)
        assert smart_money['buy_volume'] >= 0
        assert smart_money['sell_volume'] >= 0
        assert 'net_flow' in smart_money
        assert smart_money['transaction_count'] >= 0
        assert smart_money['average_position_size'] >= 0
    
    def test_analyze_lth_behavior(self, sample_transactions, analyzer_config):
        """Test long-term holder behavior analysis."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        lth_behavior = analyzer._analyze_lth_behavior(labeled_data)
        
        assert 0 <= lth_behavior['hodl_strength'] <= 100
        assert lth_behavior['distribution_level'] >= 0
        assert lth_behavior['accumulation_level'] >= 0
        assert 'net_lth_change' in lth_behavior
        assert lth_behavior['active_lth_count'] >= 0
    
    def test_calculate_network_health(self, sample_transactions, analyzer_config):
        """Test network health calculation."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        network_health = analyzer._calculate_network_health(labeled_data)
        
        assert 'health_status' in network_health
        assert network_health['health_status'] in ['excellent', 'good', 'moderate', 'low', 'very_low']
        assert 0 <= network_health['activity_score'] <= 100
        assert network_health['transaction_count'] == len(labeled_data)
        assert network_health['unique_addresses'] > 0
        assert network_health['average_tx_size'] > 0
        assert network_health['median_tx_size'] > 0
        assert 0 <= network_health['tx_distribution'] <= 1
        assert 0 <= network_health['network_utilization'] <= 100
    
    def test_calculate_tx_distribution(self, sample_transactions):
        """Test transaction distribution calculation."""
        analyzer = OnChainFlowsAnalyzer()
        
        # Uniform distribution
        uniform_data = pd.DataFrame({
            'amount': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })
        dist_score = analyzer._calculate_tx_distribution(uniform_data)
        assert 0 <= dist_score <= 1
        
        # Highly skewed distribution
        skewed_data = pd.DataFrame({
            'amount': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1000]
        })
        skewed_score = analyzer._calculate_tx_distribution(skewed_data)
        assert skewed_score < dist_score  # Skewed should have lower score
        
        # Empty data
        empty_data = pd.DataFrame({'amount': []})
        assert analyzer._calculate_tx_distribution(empty_data) == 0
    
    def test_generate_signals(self, sample_transactions, analyzer_config):
        """Test signal generation."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        
        # Calculate required inputs
        metrics = analyzer._calculate_metrics(labeled_data)
        exchange_flows = analyzer._analyze_exchange_flows(labeled_data)
        whale_activity = analyzer._track_whale_activity(labeled_data)
        smart_money = analyzer._analyze_smart_money(labeled_data)
        lth_behavior = analyzer._analyze_lth_behavior(labeled_data)
        network_health = analyzer._calculate_network_health(labeled_data)
        
        signals = analyzer._generate_signals(
            metrics, exchange_flows, whale_activity,
            smart_money, lth_behavior, network_health
        )
        
        assert 'signals' in signals
        assert isinstance(signals['signals'], list)
        assert 'combined_signal' in signals
        
        combined = signals['combined_signal']
        assert combined['type'] in ['bullish', 'bearish', 'neutral']
        assert combined['strength'] >= 0
        assert 0 <= combined['confidence'] <= 100
    
    def test_calculate_signal_confidence(self):
        """Test signal confidence calculation."""
        analyzer = OnChainFlowsAnalyzer()
        
        # High confidence - all bullish
        bullish_signals = [
            {'type': 'bullish', 'strength': 80},
            {'type': 'bullish', 'strength': 70},
            {'type': 'bullish', 'strength': 90}
        ]
        confidence = analyzer._calculate_signal_confidence(bullish_signals)
        assert confidence > 70  # High agreement and strength
        
        # Low confidence - mixed signals
        mixed_signals = [
            {'type': 'bullish', 'strength': 60},
            {'type': 'bearish', 'strength': 65},
            {'type': 'neutral', 'strength': 50}
        ]
        mixed_confidence = analyzer._calculate_signal_confidence(mixed_signals)
        assert mixed_confidence < confidence  # Lower due to disagreement
        
        # No signals
        assert analyzer._calculate_signal_confidence([]) == 0
    
    def test_update_state(self, sample_transactions, analyzer_config):
        """Test state update functionality."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        labeled_data = analyzer._label_addresses(sample_transactions)
        metrics = analyzer._calculate_metrics(labeled_data)
        
        # Initial state
        assert len(analyzer.historical_flows) == 0
        
        # Update state
        analyzer._update_state(labeled_data, metrics)
        assert len(analyzer.historical_flows) == 1
        assert analyzer.historical_flows[0]['metrics'] == metrics
        assert analyzer.historical_flows[0]['transaction_count'] == len(labeled_data)
        
        # Update multiple times
        for _ in range(150):
            analyzer._update_state(labeled_data, metrics)
        
        # Should keep only last 100
        assert len(analyzer.historical_flows) == 100
    
    def test_full_analysis(self, sample_transactions, analyzer_config):
        """Test complete analysis flow."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        result = analyzer.analyze(sample_transactions)
        
        # Check all expected keys
        expected_keys = [
            'metrics', 'exchange_flows', 'whale_activity',
            'smart_money', 'lth_behavior', 'network_health',
            'signals', 'metadata'
        ]
        for key in expected_keys:
            assert key in result
        
        # Verify metrics
        assert isinstance(result['metrics'], OnChainMetrics)
        
        # Verify metadata
        metadata = result['metadata']
        assert 'timestamp' in metadata
        assert metadata['data_points'] == len(sample_transactions)
        assert metadata['total_volume'] > 0
        assert metadata['unique_addresses'] > 0
        assert 'config' in metadata
    
    def test_analyze_onchain_flows_function(self, sample_transactions):
        """Test convenience function."""
        result = analyze_onchain_flows(
            sample_transactions,
            whale_threshold=500,
            smart_money_addresses=['smart_1'],
            exchange_addresses=['exchange_1']
        )
        
        assert 'metrics' in result
        assert 'signals' in result
        assert isinstance(result['metrics'], OnChainMetrics)
    
    def test_track_whale_movements_function(self, sample_transactions):
        """Test whale tracking convenience function."""
        result = track_whale_movements(sample_transactions, whale_threshold=500)
        
        assert 'activity_level' in result
        assert 'accumulation_score' in result
        assert 'distribution_score' in result
        assert 'whale_count' in result
    
    def test_edge_cases(self, analyzer_config):
        """Test edge cases and error conditions."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        
        # Single transaction
        single_tx = pd.DataFrame([{
            'timestamp': datetime.now(),
            'from_address': 'addr1',
            'to_address': 'addr2',
            'amount': 10,
            'tx_hash': 'hash1',
            'block_number': 1000
        }])
        result = analyzer.analyze(single_tx)
        assert result is not None
        assert result['metrics'].whale_movements >= 0
        
        # All transactions to exchanges
        exchange_only = pd.DataFrame([{
            'timestamp': datetime.now(),
            'from_address': f'addr_{i}',
            'to_address': 'exchange_1',
            'amount': 10,
            'tx_hash': f'hash_{i}',
            'block_number': 1000 + i
        } for i in range(10)])
        
        result = analyzer.analyze(exchange_only)
        assert result['exchange_flows']['trend'] == 'distribution'
        
        # Very large whale transactions
        whale_dominant = pd.DataFrame([{
            'timestamp': datetime.now(),
            'from_address': 'whale_addr',
            'to_address': f'addr_{i}',
            'amount': 10000,  # Very large
            'tx_hash': f'hash_{i}',
            'block_number': 1000 + i
        } for i in range(5)])
        
        result = analyzer.analyze(whale_dominant)
        assert result['whale_activity']['activity_level'] in ['high', 'very_high']
    
    def test_stablecoin_flow_with_token_data(self):
        """Test stablecoin flow calculation with token data."""
        analyzer = OnChainFlowsAnalyzer()
        
        # Create data with token symbols
        data = pd.DataFrame([
            {
                'timestamp': datetime.now(),
                'from_address': 'addr1',
                'to_address': 'exchange_1',
                'amount': 1000,
                'token_symbol': 'USDT',
                'to_is_exchange': True,
                'from_is_exchange': False
            },
            {
                'timestamp': datetime.now(),
                'from_address': 'exchange_1',
                'to_address': 'addr2',
                'amount': 500,
                'token_symbol': 'USDC',
                'to_is_exchange': False,
                'from_is_exchange': True
            }
        ])
        
        flow = analyzer._estimate_stablecoin_flow(data)
        assert flow == -500  # 500 outflow - 1000 inflow
    
    def test_performance_monitoring(self, sample_transactions, analyzer_config):
        """Test that performance monitoring is applied."""
        analyzer = OnChainFlowsAnalyzer(analyzer_config)
        
        # The analyze method should be decorated with @timer
        # This test verifies it completes without error
        result = analyzer.analyze(sample_transactions)
        assert result is not None
        
        # Test with large dataset
        large_data = pd.concat([sample_transactions] * 10, ignore_index=True)
        result = analyzer.analyze(large_data)
        assert result is not None
