"""
On-Chain Flows Analyzer Model.

This module provides the OnChainFlowsAnalyzer class for analyzing blockchain
on-chain flows including exchange inflows/outflows, whale movements, and
smart money tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from src.models.base_model import BaseModel
from src.utils.performance import timer
from src.utils.error_handling import with_retry, safe_execute

logger = logging.getLogger(__name__)


@dataclass
class OnChainMetrics:
    """Data class for on-chain flow metrics."""
    
    exchange_inflow: float
    exchange_outflow: float
    net_flow: float
    whale_movements: int
    smart_money_flow: float
    long_term_holder_change: float
    miner_to_exchange_flow: float
    stablecoin_flow: float
    defi_tvl_change: float
    network_activity_score: float


class OnChainFlowsAnalyzer(BaseModel):
    """
    Analyzer for on-chain blockchain flows.
    
    This analyzer tracks:
    - Exchange inflows and outflows
    - Whale wallet movements
    - Smart money tracking
    - Long-term holder behavior
    - Miner flows
    - Stablecoin movements
    - DeFi activity
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the On-Chain Flows Analyzer.
        
        Args:
            config: Configuration dictionary with the following optional keys:
                - whale_threshold: Minimum balance to consider as whale (default: 1000)
                - smart_money_addresses: List of known smart money addresses
                - exchange_addresses: List of known exchange addresses
                - flow_threshold: Minimum flow to consider significant (default: 100)
                - lookback_periods: Number of periods for trend analysis (default: 30)
        """
        super().__init__(config)
        
        # Configuration
        self.whale_threshold = self.config.get('whale_threshold', 1000)
        self.smart_money_addresses = set(self.config.get('smart_money_addresses', []))
        self.exchange_addresses = set(self.config.get('exchange_addresses', []))
        self.flow_threshold = self.config.get('flow_threshold', 100)
        self.lookback_periods = self.config.get('lookback_periods', 30)
        
        # State tracking
        self.address_labels = {}
        self.historical_flows = []
        
    @timer
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Analyze on-chain flows data.
        
        Args:
            data: DataFrame with columns:
                - timestamp: Transaction timestamp
                - from_address: Sending address
                - to_address: Receiving address
                - amount: Transaction amount
                - tx_hash: Transaction hash
                - block_number: Block number
                - gas_price: Gas price (optional)
                - token_symbol: Token symbol (optional)
            **kwargs: Additional parameters
        
        Returns:
            Dictionary containing:
                - metrics: OnChainMetrics object
                - exchange_flows: Exchange flow analysis
                - whale_activity: Whale movement analysis
                - smart_money: Smart money tracking
                - ltg_behavior: Long-term holder behavior
                - network_health: Network activity metrics
                - signals: Trading signals
                - metadata: Analysis metadata
        """
        try:
            # Validate input data
            self._validate_data(data)
            
            # Label addresses
            labeled_data = self._label_addresses(data)
            
            # Calculate metrics
            metrics = self._calculate_metrics(labeled_data)
            
            # Analyze exchange flows
            exchange_flows = self._analyze_exchange_flows(labeled_data)
            
            # Track whale activity
            whale_activity = self._track_whale_activity(labeled_data)
            
            # Analyze smart money
            smart_money = self._analyze_smart_money(labeled_data)
            
            # Analyze long-term holder behavior
            lth_behavior = self._analyze_lth_behavior(labeled_data)
            
            # Calculate network health
            network_health = self._calculate_network_health(labeled_data)
            
            # Generate signals
            signals = self._generate_signals(
                metrics,
                exchange_flows,
                whale_activity,
                smart_money,
                lth_behavior,
                network_health
            )
            
            # Update state
            self._update_state(labeled_data, metrics)
            
            return {
                'metrics': metrics,
                'exchange_flows': exchange_flows,
                'whale_activity': whale_activity,
                'smart_money': smart_money,
                'lth_behavior': lth_behavior,
                'network_health': network_health,
                'signals': signals,
                'metadata': self._generate_metadata(data)
            }
            
        except Exception as e:
            logger.error(f"Error in on-chain flows analysis: {str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format and content."""
        required_columns = ['timestamp', 'from_address', 'to_address', 'amount']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(data) == 0:
            raise ValueError("Empty dataframe provided")
        
        # Check for negative amounts
        if (data['amount'] < 0).any():
            raise ValueError("Negative transaction amounts found")
    
    def _label_addresses(self, data: pd.DataFrame) -> pd.DataFrame:
        """Label addresses based on known patterns and lists."""
        df = data.copy()
        
        # Label exchange addresses
        df['from_is_exchange'] = df['from_address'].isin(self.exchange_addresses)
        df['to_is_exchange'] = df['to_address'].isin(self.exchange_addresses)
        
        # Label smart money addresses
        df['from_is_smart_money'] = df['from_address'].isin(self.smart_money_addresses)
        df['to_is_smart_money'] = df['to_address'].isin(self.smart_money_addresses)
        
        # Calculate address balances for whale detection
        address_balances = self._calculate_address_balances(df)
        
        # Label whale addresses
        whale_addresses = set(
            addr for addr, balance in address_balances.items()
            if balance >= self.whale_threshold
        )
        df['from_is_whale'] = df['from_address'].isin(whale_addresses)
        df['to_is_whale'] = df['to_address'].isin(whale_addresses)
        
        return df
    
    def _calculate_address_balances(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate current balance for each address."""
        balances = {}
        
        # Add received amounts
        for _, row in data.iterrows():
            to_addr = row['to_address']
            balances[to_addr] = balances.get(to_addr, 0) + row['amount']
        
        # Subtract sent amounts
        for _, row in data.iterrows():
            from_addr = row['from_address']
            balances[from_addr] = balances.get(from_addr, 0) - row['amount']
        
        return balances
    
    def _calculate_metrics(self, data: pd.DataFrame) -> OnChainMetrics:
        """Calculate on-chain flow metrics."""
        # Exchange flows
        exchange_inflow = data[data['to_is_exchange']]['amount'].sum()
        exchange_outflow = data[data['from_is_exchange']]['amount'].sum()
        net_flow = exchange_outflow - exchange_inflow
        
        # Whale movements
        whale_txs = data[data['from_is_whale'] | data['to_is_whale']]
        whale_movements = len(whale_txs[whale_txs['amount'] >= self.flow_threshold])
        
        # Smart money flow
        smart_money_inflow = data[data['to_is_smart_money']]['amount'].sum()
        smart_money_outflow = data[data['from_is_smart_money']]['amount'].sum()
        smart_money_flow = smart_money_inflow - smart_money_outflow
        
        # Long-term holder change (simplified - would need historical data)
        lth_change = self._estimate_lth_change(data)
        
        # Miner to exchange flow (would need miner address labels)
        miner_flow = self._estimate_miner_flow(data)
        
        # Stablecoin flow (would need token type data)
        stablecoin_flow = self._estimate_stablecoin_flow(data)
        
        # DeFi TVL change (would need DeFi protocol data)
        defi_tvl_change = self._estimate_defi_tvl_change(data)
        
        # Network activity score
        network_activity = self._calculate_network_activity(data)
        
        return OnChainMetrics(
            exchange_inflow=exchange_inflow,
            exchange_outflow=exchange_outflow,
            net_flow=net_flow,
            whale_movements=whale_movements,
            smart_money_flow=smart_money_flow,
            long_term_holder_change=lth_change,
            miner_to_exchange_flow=miner_flow,
            stablecoin_flow=stablecoin_flow,
            defi_tvl_change=defi_tvl_change,
            network_activity_score=network_activity
        )
    
    def _analyze_exchange_flows(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exchange inflow and outflow patterns."""
        exchange_txs = data[data['to_is_exchange'] | data['from_is_exchange']]
        
        if len(exchange_txs) == 0:
            return {
                'trend': 'neutral',
                'strength': 0,
                'large_deposits': [],
                'large_withdrawals': []
            }
        
        # Separate inflows and outflows
        inflows = exchange_txs[exchange_txs['to_is_exchange']]
        outflows = exchange_txs[exchange_txs['from_is_exchange']]
        
        # Calculate flow rates
        inflow_rate = len(inflows) / max(1, len(exchange_txs))
        outflow_rate = len(outflows) / max(1, len(exchange_txs))
        
        # Determine trend
        if outflow_rate > inflow_rate * 1.2:
            trend = 'accumulation'
            strength = (outflow_rate - inflow_rate) * 100
        elif inflow_rate > outflow_rate * 1.2:
            trend = 'distribution'
            strength = (inflow_rate - outflow_rate) * 100
        else:
            trend = 'neutral'
            strength = abs(outflow_rate - inflow_rate) * 100
        
        # Find large movements
        large_threshold = self.flow_threshold * 10
        large_deposits = inflows[inflows['amount'] >= large_threshold].to_dict('records')
        large_withdrawals = outflows[outflows['amount'] >= large_threshold].to_dict('records')
        
        return {
            'trend': trend,
            'strength': min(100, strength),
            'inflow_volume': inflows['amount'].sum(),
            'outflow_volume': outflows['amount'].sum(),
            'net_flow': outflows['amount'].sum() - inflows['amount'].sum(),
            'large_deposits': large_deposits[:5],  # Top 5
            'large_withdrawals': large_withdrawals[:5]  # Top 5
        }
    
    def _track_whale_activity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Track whale wallet movements and patterns."""
        whale_txs = data[data['from_is_whale'] | data['to_is_whale']]
        
        if len(whale_txs) == 0:
            return {
                'activity_level': 'low',
                'accumulation_score': 0,
                'distribution_score': 0,
                'whale_count': 0
            }
        
        # Count unique whales
        whale_addresses = set()
        whale_addresses.update(whale_txs[whale_txs['from_is_whale']]['from_address'])
        whale_addresses.update(whale_txs[whale_txs['to_is_whale']]['to_address'])
        
        # Calculate accumulation/distribution
        whale_buys = whale_txs[whale_txs['to_is_whale'] & ~whale_txs['from_is_whale']]
        whale_sells = whale_txs[whale_txs['from_is_whale'] & ~whale_txs['to_is_whale']]
        
        accumulation_score = whale_buys['amount'].sum()
        distribution_score = whale_sells['amount'].sum()
        
        # Determine activity level
        total_whale_volume = whale_txs['amount'].sum()
        total_volume = data['amount'].sum()
        whale_dominance = total_whale_volume / max(1, total_volume)
        
        if whale_dominance > 0.5:
            activity_level = 'very_high'
        elif whale_dominance > 0.3:
            activity_level = 'high'
        elif whale_dominance > 0.1:
            activity_level = 'moderate'
        else:
            activity_level = 'low'
        
        return {
            'activity_level': activity_level,
            'accumulation_score': accumulation_score,
            'distribution_score': distribution_score,
            'net_position_change': accumulation_score - distribution_score,
            'whale_count': len(whale_addresses),
            'whale_dominance': whale_dominance * 100,
            'average_transaction_size': whale_txs['amount'].mean()
        }
    
    def _analyze_smart_money(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze smart money movements and positioning."""
        smart_txs = data[data['from_is_smart_money'] | data['to_is_smart_money']]
        
        if len(smart_txs) == 0:
            return {
                'sentiment': 'neutral',
                'conviction': 0,
                'follow_signal': False
            }
        
        # Calculate net smart money flow
        smart_buys = smart_txs[smart_txs['to_is_smart_money'] & ~smart_txs['from_is_smart_money']]
        smart_sells = smart_txs[smart_txs['from_is_smart_money'] & ~smart_txs['to_is_smart_money']]
        
        buy_volume = smart_buys['amount'].sum()
        sell_volume = smart_sells['amount'].sum()
        net_flow = buy_volume - sell_volume
        
        # Determine sentiment
        total_smart_volume = buy_volume + sell_volume
        if total_smart_volume == 0:
            sentiment = 'neutral'
            conviction = 0
        else:
            flow_ratio = net_flow / total_smart_volume
            
            if flow_ratio > 0.3:
                sentiment = 'bullish'
                conviction = min(100, flow_ratio * 100)
            elif flow_ratio < -0.3:
                sentiment = 'bearish'
                conviction = min(100, abs(flow_ratio) * 100)
            else:
                sentiment = 'neutral'
                conviction = abs(flow_ratio) * 100
        
        # Generate follow signal
        follow_signal = conviction > 50 and len(smart_txs) > 5
        
        return {
            'sentiment': sentiment,
            'conviction': conviction,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_flow': net_flow,
            'transaction_count': len(smart_txs),
            'follow_signal': follow_signal,
            'average_position_size': smart_txs['amount'].mean()
        }
    
    def _analyze_lth_behavior(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze long-term holder behavior patterns."""
        # This is simplified - real implementation would need address age data
        # For now, we'll use large addresses as proxy for LTH
        
        large_threshold = self.whale_threshold * 0.5
        address_balances = self._calculate_address_balances(data)
        large_addresses = {
            addr for addr, balance in address_balances.items()
            if balance >= large_threshold
        }
        
        # Find transactions from large addresses
        lth_txs = data[data['from_address'].isin(large_addresses)]
        
        if len(lth_txs) == 0:
            return {
                'hodl_strength': 100,  # No selling = strong hodl
                'distribution_level': 0,
                'accumulation_level': 0
            }
        
        # Calculate distribution level
        total_lth_outflow = lth_txs['amount'].sum()
        total_volume = data['amount'].sum()
        distribution_level = (total_lth_outflow / max(1, total_volume)) * 100
        
        # Calculate accumulation level
        lth_buys = data[data['to_address'].isin(large_addresses)]
        accumulation_level = (lth_buys['amount'].sum() / max(1, total_volume)) * 100
        
        # Calculate hodl strength (inverse of distribution)
        hodl_strength = max(0, 100 - distribution_level)
        
        return {
            'hodl_strength': hodl_strength,
            'distribution_level': distribution_level,
            'accumulation_level': accumulation_level,
            'net_lth_change': accumulation_level - distribution_level,
            'active_lth_count': len(set(lth_txs['from_address']))
        }
    
    def _calculate_network_health(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate network health and activity metrics."""
        # Transaction metrics
        tx_count = len(data)
        unique_addresses = len(set(data['from_address']) | set(data['to_address']))
        avg_tx_size = data['amount'].mean()
        median_tx_size = data['amount'].median()
        
        # Calculate transaction distribution
        tx_distribution = self._calculate_tx_distribution(data)
        
        # Network activity score (0-100)
        # Based on transaction count, unique addresses, and distribution
        base_score = min(100, (tx_count / 1000) * 30)  # Up to 30 points for volume
        address_score = min(100, (unique_addresses / 100) * 30)  # Up to 30 points for participation
        distribution_score = tx_distribution * 40  # Up to 40 points for healthy distribution
        
        activity_score = base_score + address_score + distribution_score
        
        # Determine health status
        if activity_score > 80:
            health_status = 'excellent'
        elif activity_score > 60:
            health_status = 'good'
        elif activity_score > 40:
            health_status = 'moderate'
        elif activity_score > 20:
            health_status = 'low'
        else:
            health_status = 'very_low'
        
        return {
            'health_status': health_status,
            'activity_score': activity_score,
            'transaction_count': tx_count,
            'unique_addresses': unique_addresses,
            'average_tx_size': avg_tx_size,
            'median_tx_size': median_tx_size,
            'tx_distribution': tx_distribution,
            'network_utilization': min(100, (tx_count / 10000) * 100)  # Assuming 10k tx/period is full utilization
        }
    
    def _calculate_tx_distribution(self, data: pd.DataFrame) -> float:
        """Calculate transaction size distribution score (0-1)."""
        if len(data) == 0:
            return 0
        
        # Calculate percentiles
        p10 = data['amount'].quantile(0.1)
        p50 = data['amount'].quantile(0.5)
        p90 = data['amount'].quantile(0.9)
        
        # Good distribution has reasonable spread
        if p90 == 0 or p10 == 0:
            return 0
        
        # Calculate distribution metrics
        spread_ratio = p90 / max(1, p10)
        median_position = (p50 - p10) / max(1, (p90 - p10))
        
        # Score based on healthy distribution characteristics
        # Ideal: moderate spread (not too concentrated) and median near center
        spread_score = 1 - abs(np.log10(spread_ratio) - 2) / 2  # Peak at 100x spread
        median_score = 1 - abs(median_position - 0.5) * 2  # Peak at 0.5 (centered)
        
        return max(0, min(1, (spread_score + median_score) / 2))
    
    def _estimate_lth_change(self, data: pd.DataFrame) -> float:
        """Estimate long-term holder supply change."""
        # Simplified estimation based on large address movements
        large_addresses = self._identify_large_addresses(data)
        
        net_change = 0
        for addr in large_addresses:
            received = data[data['to_address'] == addr]['amount'].sum()
            sent = data[data['from_address'] == addr]['amount'].sum()
            net_change += received - sent
        
        return net_change
    
    def _estimate_miner_flow(self, data: pd.DataFrame) -> float:
        """Estimate miner to exchange flow."""
        # Simplified - would need miner address labels
        # Look for transactions from addresses with no incoming transactions (potential miners)
        
        addresses_with_only_outgoing = set()
        for addr in data['from_address'].unique():
            if addr not in data['to_address'].values:
                addresses_with_only_outgoing.add(addr)
        
        # Calculate flow from these addresses to exchanges
        miner_to_exchange = data[
            data['from_address'].isin(addresses_with_only_outgoing) &
            data['to_is_exchange']
        ]['amount'].sum()
        
        return miner_to_exchange
    
    def _estimate_stablecoin_flow(self, data: pd.DataFrame) -> float:
        """Estimate stablecoin flow."""
        # Simplified - would need token type data
        # For now, return 0 as we don't have token information
        if 'token_symbol' in data.columns:
            stablecoin_symbols = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD']
            stablecoin_txs = data[data['token_symbol'].isin(stablecoin_symbols)]
            
            # Net flow calculation
            inflows = stablecoin_txs[stablecoin_txs['to_is_exchange']]['amount'].sum()
            outflows = stablecoin_txs[stablecoin_txs['from_is_exchange']]['amount'].sum()
            return outflows - inflows
        
        return 0
    
    def _estimate_defi_tvl_change(self, data: pd.DataFrame) -> float:
        """Estimate DeFi TVL change."""
        # Simplified - would need DeFi protocol address labels
        # For now, return 0
        return 0
    
    def _calculate_network_activity(self, data: pd.DataFrame) -> float:
        """Calculate overall network activity score."""
        # Factors: transaction count, unique addresses, total volume
        tx_count_score = min(100, len(data) / 100)
        
        unique_addresses = len(set(data['from_address']) | set(data['to_address']))
        address_score = min(100, unique_addresses / 50)
        
        volume_score = min(100, data['amount'].sum() / 10000)
        
        # Weighted average
        return (tx_count_score * 0.3 + address_score * 0.4 + volume_score * 0.3)
    
    def _identify_large_addresses(self, data: pd.DataFrame) -> set:
        """Identify addresses with large balances."""
        balances = self._calculate_address_balances(data)
        threshold = self.whale_threshold * 0.1  # 10% of whale threshold
        
        return {addr for addr, balance in balances.items() if balance >= threshold}
    
    def _generate_signals(
        self,
        metrics: OnChainMetrics,
        exchange_flows: Dict[str, Any],
        whale_activity: Dict[str, Any],
        smart_money: Dict[str, Any],
        lth_behavior: Dict[str, Any],
        network_health: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate trading signals based on on-chain analysis."""
        signals = []
        
        # Exchange flow signals
        if exchange_flows['trend'] == 'accumulation' and exchange_flows['strength'] > 50:
            signals.append({
                'type': 'bullish',
                'source': 'exchange_flows',
                'strength': exchange_flows['strength'],
                'message': 'Strong exchange outflows indicate accumulation'
            })
        elif exchange_flows['trend'] == 'distribution' and exchange_flows['strength'] > 50:
            signals.append({
                'type': 'bearish',
                'source': 'exchange_flows',
                'strength': exchange_flows['strength'],
                'message': 'High exchange inflows suggest distribution'
            })
        
        # Whale activity signals
        if whale_activity['activity_level'] in ['high', 'very_high']:
            if whale_activity['net_position_change'] > 0:
                signals.append({
                    'type': 'bullish',
                    'source': 'whale_activity',
                    'strength': min(100, whale_activity['whale_dominance']),
                    'message': 'Whales are accumulating'
                })
            elif whale_activity['net_position_change'] < 0:
                signals.append({
                    'type': 'bearish',
                    'source': 'whale_activity',
                    'strength': min(100, whale_activity['whale_dominance']),
                    'message': 'Whales are distributing'
                })
        
        # Smart money signals
        if smart_money['follow_signal']:
            signals.append({
                'type': 'bullish' if smart_money['sentiment'] == 'bullish' else 'bearish',
                'source': 'smart_money',
                'strength': smart_money['conviction'],
                'message': f"Smart money is {smart_money['sentiment']}"
            })
        
        # Long-term holder signals
        if lth_behavior['distribution_level'] > 20:
            signals.append({
                'type': 'bearish',
                'source': 'lth_behavior',
                'strength': lth_behavior['distribution_level'],
                'message': 'Long-term holders are distributing'
            })
        elif lth_behavior['accumulation_level'] > 20:
            signals.append({
                'type': 'bullish',
                'source': 'lth_behavior',
                'strength': lth_behavior['accumulation_level'],
                'message': 'Long-term holders are accumulating'
            })
        
        # Network health signals
        if network_health['health_status'] == 'excellent':
            signals.append({
                'type': 'bullish',
                'source': 'network_health',
                'strength': network_health['activity_score'],
                'message': 'Network activity is very healthy'
            })
        elif network_health['health_status'] in ['low', 'very_low']:
            signals.append({
                'type': 'warning',
                'source': 'network_health',
                'strength': 100 - network_health['activity_score'],
                'message': 'Low network activity detected'
            })
        
        # Calculate combined signal
        bullish_signals = [s for s in signals if s['type'] == 'bullish']
        bearish_signals = [s for s in signals if s['type'] == 'bearish']
        
        if bullish_signals and bearish_signals:
            # Conflicting signals
            bullish_strength = np.mean([s['strength'] for s in bullish_signals])
            bearish_strength = np.mean([s['strength'] for s in bearish_signals])
            
            if bullish_strength > bearish_strength * 1.2:
                combined_type = 'bullish'
                combined_strength = bullish_strength - bearish_strength
            elif bearish_strength > bullish_strength * 1.2:
                combined_type = 'bearish'
                combined_strength = bearish_strength - bullish_strength
            else:
                combined_type = 'neutral'
                combined_strength = 0
        elif bullish_signals:
            combined_type = 'bullish'
            combined_strength = np.mean([s['strength'] for s in bullish_signals])
        elif bearish_signals:
            combined_type = 'bearish'
            combined_strength = np.mean([s['strength'] for s in bearish_signals])
        else:
            combined_type = 'neutral'
            combined_strength = 0
        
        return {
            'signals': signals,
            'combined_signal': {
                'type': combined_type,
                'strength': combined_strength,
                'confidence': self._calculate_signal_confidence(signals)
            }
        }
    
    def _calculate_signal_confidence(self, signals: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the signals based on agreement and strength."""
        if not signals:
            return 0
        
        # Group by type
        signal_types = {}
        for signal in signals:
            signal_type = signal['type']
            if signal_type not in signal_types:
                signal_types[signal_type] = []
            signal_types[signal_type].append(signal['strength'])
        
        # Calculate agreement score
        total_signals = len(signals)
        max_agreement = max(len(sigs) for sigs in signal_types.values())
        agreement_score = max_agreement / total_signals
        
        # Calculate average strength
        all_strengths = [s['strength'] for s in signals]
        avg_strength = np.mean(all_strengths) / 100
        
        # Confidence is combination of agreement and strength
        confidence = (agreement_score * 0.6 + avg_strength * 0.4) * 100
        
        return min(100, confidence)
    
    def _update_state(self, data: pd.DataFrame, metrics: OnChainMetrics) -> None:
        """Update internal state with latest data."""
        # Add to historical flows
        self.historical_flows.append({
            'timestamp': data['timestamp'].max() if 'timestamp' in data.columns else datetime.now(),
            'metrics': metrics,
            'transaction_count': len(data),
            'total_volume': data['amount'].sum()
        })
        
        # Keep only recent history
        max_history = 100
        if len(self.historical_flows) > max_history:
            self.historical_flows = self.historical_flows[-max_history:]
    
    def _generate_metadata(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate metadata for the analysis."""
        return {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data),
            'time_range': {
                'start': data['timestamp'].min().isoformat() if 'timestamp' in data.columns else None,
                'end': data['timestamp'].max().isoformat() if 'timestamp' in data.columns else None
            },
            'total_volume': float(data['amount'].sum()),
            'unique_addresses': len(set(data['from_address']) | set(data['to_address'])),
            'config': {
                'whale_threshold': self.whale_threshold,
                'flow_threshold': self.flow_threshold,
                'lookback_periods': self.lookback_periods
            }
        }


def analyze_onchain_flows(
    transactions: pd.DataFrame,
    whale_threshold: float = 1000,
    smart_money_addresses: Optional[List[str]] = None,
    exchange_addresses: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze on-chain flows.
    
    Args:
        transactions: DataFrame with transaction data
        whale_threshold: Minimum balance to consider as whale
        smart_money_addresses: List of known smart money addresses
        exchange_addresses: List of known exchange addresses
    
    Returns:
        Analysis results dictionary
    """
    config = {
        'whale_threshold': whale_threshold,
        'smart_money_addresses': smart_money_addresses or [],
        'exchange_addresses': exchange_addresses or []
    }
    
    analyzer = OnChainFlowsAnalyzer(config)
    return analyzer.analyze(transactions)


def track_whale_movements(
    transactions: pd.DataFrame,
    whale_threshold: float = 1000
) -> Dict[str, Any]:
    """
    Track whale wallet movements.
    
    Args:
        transactions: DataFrame with transaction data
        whale_threshold: Minimum balance to consider as whale
    
    Returns:
        Whale activity analysis
    """
    analyzer = OnChainFlowsAnalyzer({'whale_threshold': whale_threshold})
    labeled_data = analyzer._label_addresses(transactions)
    return analyzer._track_whale_activity(labeled_data)
