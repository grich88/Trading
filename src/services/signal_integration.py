"""
Signal integration service module.

This module provides a specialized service for integrating signals from multiple sources,
applying custom weighting strategies, and generating actionable trading signals.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
from datetime import datetime, timedelta
import threading
import time

# Import base service
from src.services.base_service import LongRunningService

# Import market analysis service
from src.services.market_analysis import MarketAnalysisService

# Import utilities
from src.utils import (
    exception_handler, 
    performance_monitor, 
    async_performance_monitor,
    ModelError,
    timer,
    memory_usage,
    adaptive_batch_processing
)

# Import configuration
from src.config import (
    SIGNAL_INTEGRATION_DIR,
    SIGNAL_WEIGHTS,
    SIGNAL_THRESHOLD,
    SIGNAL_HISTORY_MAX_DAYS,
    SIGNAL_UPDATE_INTERVAL_MINUTES
)


class SignalIntegrationService(LongRunningService):
    """
    Specialized service for signal integration.
    
    This service extends LongRunningService to provide:
    - Multi-source signal integration
    - Custom weighting strategies
    - Signal filtering and validation
    - Backtesting capabilities
    - Signal history tracking
    - Automated signal generation
    """
    
    def __init__(self, 
                 signal_dir: str = SIGNAL_INTEGRATION_DIR,
                 signal_weights: Optional[Dict[str, float]] = None,
                 signal_threshold: float = SIGNAL_THRESHOLD,
                 history_max_days: int = SIGNAL_HISTORY_MAX_DAYS,
                 update_interval_minutes: int = SIGNAL_UPDATE_INTERVAL_MINUTES,
                 **kwargs: Any):
        """
        Initialize the signal integration service.
        
        Args:
            signal_dir: Directory for storing signal data
            signal_weights: Weights for different signal sources
            signal_threshold: Minimum threshold for signal strength
            history_max_days: Maximum days to keep in signal history
            update_interval_minutes: Interval between signal updates
            **kwargs: Additional arguments for LongRunningService
        """
        super().__init__("SignalIntegrationService", **kwargs)
        
        self.signal_dir = signal_dir
        self.signal_weights = signal_weights or SIGNAL_WEIGHTS
        self.signal_threshold = signal_threshold
        self.history_max_days = history_max_days
        self.update_interval_minutes = update_interval_minutes
        
        # Create market analysis service for signal generation
        self.market_analysis = MarketAnalysisService()
        
        # Signal sources
        self.signal_sources = {}
        
        # Signal history
        self.signal_history = {}
        
        # Active signals
        self.active_signals = {}
        
        # Signal update thread
        self.update_thread = None
        
        # Signal processors
        self.signal_processors = {}
        
        # Create signal directory
        os.makedirs(self.signal_dir, exist_ok=True)
        
        # Signal metrics
        self.signal_metrics = {
            "total_signals_generated": 0,
            "active_signals_count": 0,
            "signal_hit_rate": 0.0,
            "average_signal_strength": 0.0,
            "last_update_time": None
        }
        
        self.logger.info("Signal integration service initialized")
    
    def _start_service(self) -> None:
        """Start the signal integration service."""
        super()._start_service()
        
        # Start market analysis service
        self.market_analysis.start()
        
        # Register signal sources
        self._register_signal_sources()
        
        # Register signal processors
        self._register_signal_processors()
        
        # Load signal history
        self._load_signal_history()
        
        # Start update thread
        self._start_update_thread()
        
        self.logger.info("Signal integration service started")
    
    def _stop_service(self) -> None:
        """Stop the signal integration service."""
        # Stop update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Save signal history
        self._save_signal_history()
        
        # Stop market analysis service
        self.market_analysis.stop()
        
        super()._stop_service()
        self.logger.info("Signal integration service stopped")
    
    def _register_signal_sources(self) -> None:
        """Register signal sources."""
        # Register standard signal sources
        self.signal_sources = {
            "market_analysis": self._get_market_analysis_signals,
            "technical": self._get_technical_signals,
            "fundamental": self._get_fundamental_signals,
            "sentiment": self._get_sentiment_signals,
            "on_chain": self._get_on_chain_signals
        }
        
        self.logger.info(f"Registered {len(self.signal_sources)} signal sources")
    
    def _register_signal_processors(self) -> None:
        """Register signal processors."""
        # Register standard signal processors
        self.signal_processors = {
            "default": self._default_signal_processor,
            "weighted_average": self._weighted_average_processor,
            "majority_vote": self._majority_vote_processor,
            "threshold_filter": self._threshold_filter_processor
        }
        
        self.logger.info(f"Registered {len(self.signal_processors)} signal processors")
    
    def _start_update_thread(self) -> None:
        """Start the signal update thread."""
        if self.update_thread and self.update_thread.is_alive():
            self.logger.warning("Update thread is already running")
            return
        
        self.update_thread = threading.Thread(
            target=self._update_loop,
            name="SignalUpdateThread",
            daemon=True
        )
        self.update_thread.start()
        self.logger.info("Signal update thread started")
    
    def _update_loop(self) -> None:
        """Signal update thread function."""
        self.logger.info("Signal update loop started")
        
        while self.running:
            try:
                # Update signals
                self.update_signals()
                
                # Update last update time
                self.signal_metrics["last_update_time"] = datetime.now()
                
                # Sleep until next update
                time.sleep(self.update_interval_minutes * 60)
            
            except Exception as e:
                self.logger.error(f"Error in update loop: {str(e)}")
                self.health_metrics["error_count"] += 1
                
                # Sleep for a shorter interval on error
                time.sleep(min(self.update_interval_minutes * 60, 60))
        
        self.logger.info("Signal update loop stopped")
    
    @performance_monitor()
    def update_signals(self) -> Dict[str, Any]:
        """
        Update signals from all sources.
        
        Returns:
            Dictionary with updated signals
        """
        self.logger.info("Updating signals from all sources")
        
        # Get signals from all sources
        signals = {}
        for source_name, source_func in self.signal_sources.items():
            try:
                source_signals = source_func()
                signals[source_name] = source_signals
                self.logger.info(f"Got {len(source_signals)} signals from {source_name}")
            except Exception as e:
                self.logger.error(f"Error getting signals from {source_name}: {str(e)}")
        
        # Integrate signals
        integrated_signals = self.integrate_signals(signals)
        
        # Update active signals
        self._update_active_signals(integrated_signals)
        
        # Update signal history
        self._update_signal_history(integrated_signals)
        
        # Update signal metrics
        self._update_signal_metrics()
        
        return integrated_signals
    
    def _get_market_analysis_signals(self) -> Dict[str, Any]:
        """
        Get signals from market analysis service.
        
        Returns:
            Dictionary with market analysis signals
        """
        signals = {}
        
        # Get signals for all assets and timeframes
        assets = ["BTC", "ETH", "SOL"]
        timeframes = ["1h", "4h", "1d"]
        
        for asset in assets:
            asset_signals = {}
            
            for timeframe in timeframes:
                try:
                    # Get latest analysis
                    analysis = self.market_analysis.get_latest_analysis(asset, timeframe)
                    
                    # If no analysis available, run analysis now
                    if not analysis:
                        analysis = self.market_analysis.run_analysis_now(asset, timeframe)
                    
                    # Extract integrated signal
                    if "integrated" in analysis:
                        signal = analysis["integrated"]
                        asset_signals[timeframe] = {
                            "signal": signal["signal"],
                            "score": signal["score"],
                            "timestamp": signal["timestamp"],
                            "components": signal.get("components", {})
                        }
                
                except Exception as e:
                    self.logger.error(f"Error getting analysis for {asset} {timeframe}: {str(e)}")
            
            if asset_signals:
                signals[asset] = asset_signals
        
        return signals
    
    def _get_technical_signals(self) -> Dict[str, Any]:
        """
        Get technical signals (placeholder).
        
        Returns:
            Dictionary with technical signals
        """
        # Placeholder for technical signals
        # In a real implementation, this would calculate technical indicators
        return {
            "BTC": {
                "1d": {
                    "signal": "NEUTRAL",
                    "score": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
    
    def _get_fundamental_signals(self) -> Dict[str, Any]:
        """
        Get fundamental signals (placeholder).
        
        Returns:
            Dictionary with fundamental signals
        """
        # Placeholder for fundamental signals
        # In a real implementation, this would analyze fundamental data
        return {
            "BTC": {
                "1d": {
                    "signal": "NEUTRAL",
                    "score": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
    
    def _get_sentiment_signals(self) -> Dict[str, Any]:
        """
        Get sentiment signals (placeholder).
        
        Returns:
            Dictionary with sentiment signals
        """
        # Placeholder for sentiment signals
        # In a real implementation, this would analyze social media sentiment
        return {
            "BTC": {
                "1d": {
                    "signal": "NEUTRAL",
                    "score": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
    
    def _get_on_chain_signals(self) -> Dict[str, Any]:
        """
        Get on-chain signals (placeholder).
        
        Returns:
            Dictionary with on-chain signals
        """
        # Placeholder for on-chain signals
        # In a real implementation, this would analyze blockchain data
        return {
            "BTC": {
                "1d": {
                    "signal": "NEUTRAL",
                    "score": 0.0,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
    
    @performance_monitor()
    def integrate_signals(self, 
                        signals: Dict[str, Dict[str, Any]],
                        processor: str = "weighted_average") -> Dict[str, Dict[str, Any]]:
        """
        Integrate signals from multiple sources.
        
        Args:
            signals: Dictionary with signals from different sources
            processor: Signal processor to use
            
        Returns:
            Dictionary with integrated signals
        """
        if processor not in self.signal_processors:
            self.logger.warning(f"Signal processor '{processor}' not found, using default")
            processor = "default"
        
        processor_func = self.signal_processors[processor]
        integrated_signals = processor_func(signals)
        
        # Update metrics
        self.signal_metrics["total_signals_generated"] += len(integrated_signals)
        
        return integrated_signals
    
    def _default_signal_processor(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Default signal processor.
        
        Args:
            signals: Dictionary with signals from different sources
            
        Returns:
            Dictionary with integrated signals
        """
        # Simple integration: use market analysis signals as the base
        result = {}
        
        if "market_analysis" in signals:
            market_signals = signals["market_analysis"]
            
            for asset, asset_signals in market_signals.items():
                result[asset] = {}
                
                for timeframe, signal_data in asset_signals.items():
                    result[asset][timeframe] = {
                        "signal": signal_data["signal"],
                        "score": signal_data["score"],
                        "timestamp": signal_data["timestamp"],
                        "sources": {"market_analysis": signal_data}
                    }
        
        return result
    
    def _weighted_average_processor(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Weighted average signal processor.
        
        Args:
            signals: Dictionary with signals from different sources
            
        Returns:
            Dictionary with integrated signals
        """
        result = {}
        
        # Get all assets and timeframes
        assets = set()
        for source in signals.values():
            assets.update(source.keys())
        
        for asset in assets:
            result[asset] = {}
            timeframes = set()
            
            # Get all timeframes for this asset
            for source in signals.values():
                if asset in source:
                    timeframes.update(source[asset].keys())
            
            for timeframe in timeframes:
                # Collect signals for this asset and timeframe
                asset_signals = {}
                for source_name, source_data in signals.items():
                    if asset in source_data and timeframe in source_data[asset]:
                        asset_signals[source_name] = source_data[asset][timeframe]
                
                if not asset_signals:
                    continue
                
                # Calculate weighted average
                total_weight = 0.0
                weighted_score = 0.0
                
                for source_name, signal_data in asset_signals.items():
                    weight = self.signal_weights.get(source_name, 1.0)
                    total_weight += weight
                    weighted_score += signal_data["score"] * weight
                
                if total_weight > 0:
                    final_score = weighted_score / total_weight
                else:
                    final_score = 0.0
                
                # Cap score
                final_score = max(-1.0, min(1.0, final_score))
                
                # Determine signal
                signal = self._score_to_signal(final_score)
                
                # Create result
                result[asset][timeframe] = {
                    "signal": signal,
                    "score": final_score,
                    "timestamp": datetime.now().isoformat(),
                    "sources": asset_signals
                }
        
        return result
    
    def _majority_vote_processor(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Majority vote signal processor.
        
        Args:
            signals: Dictionary with signals from different sources
            
        Returns:
            Dictionary with integrated signals
        """
        result = {}
        
        # Get all assets and timeframes
        assets = set()
        for source in signals.values():
            assets.update(source.keys())
        
        for asset in assets:
            result[asset] = {}
            timeframes = set()
            
            # Get all timeframes for this asset
            for source in signals.values():
                if asset in source:
                    timeframes.update(source[asset].keys())
            
            for timeframe in timeframes:
                # Collect signals for this asset and timeframe
                asset_signals = {}
                for source_name, source_data in signals.items():
                    if asset in source_data and timeframe in source_data[asset]:
                        asset_signals[source_name] = source_data[asset][timeframe]
                
                if not asset_signals:
                    continue
                
                # Count votes
                votes = {
                    "STRONG BUY": 0,
                    "BUY": 0,
                    "NEUTRAL": 0,
                    "SELL": 0,
                    "STRONG SELL": 0
                }
                
                for signal_data in asset_signals.values():
                    signal = signal_data["signal"]
                    votes[signal] = votes.get(signal, 0) + 1
                
                # Find majority
                max_votes = 0
                majority_signal = "NEUTRAL"
                
                for signal, vote_count in votes.items():
                    if vote_count > max_votes:
                        max_votes = vote_count
                        majority_signal = signal
                
                # Calculate average score
                total_score = sum(s["score"] for s in asset_signals.values())
                avg_score = total_score / len(asset_signals)
                
                # Create result
                result[asset][timeframe] = {
                    "signal": majority_signal,
                    "score": avg_score,
                    "timestamp": datetime.now().isoformat(),
                    "sources": asset_signals,
                    "votes": votes
                }
        
        return result
    
    def _threshold_filter_processor(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Threshold filter signal processor.
        
        Args:
            signals: Dictionary with signals from different sources
            
        Returns:
            Dictionary with integrated signals
        """
        # First use weighted average processor
        integrated = self._weighted_average_processor(signals)
        
        # Then filter by threshold
        for asset, asset_signals in integrated.items():
            for timeframe, signal_data in list(asset_signals.items()):
                if abs(signal_data["score"]) < self.signal_threshold:
                    signal_data["signal"] = "NEUTRAL"
                    signal_data["filtered"] = True
        
        return integrated
    
    def _score_to_signal(self, score: float) -> str:
        """
        Convert score to signal.
        
        Args:
            score: Signal score
            
        Returns:
            Signal string
        """
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
    
    def _update_active_signals(self, signals: Dict[str, Dict[str, Any]]) -> None:
        """
        Update active signals.
        
        Args:
            signals: Dictionary with integrated signals
        """
        # Update active signals
        self.active_signals = signals
        
        # Update metrics
        self.signal_metrics["active_signals_count"] = sum(
            len(timeframes) for timeframes in signals.values()
        )
    
    def _update_signal_history(self, signals: Dict[str, Dict[str, Any]]) -> None:
        """
        Update signal history.
        
        Args:
            signals: Dictionary with integrated signals
        """
        # Add signals to history
        timestamp = datetime.now().isoformat()
        
        for asset, asset_signals in signals.items():
            if asset not in self.signal_history:
                self.signal_history[asset] = {}
            
            for timeframe, signal_data in asset_signals.items():
                if timeframe not in self.signal_history[asset]:
                    self.signal_history[asset][timeframe] = []
                
                # Add to history
                self.signal_history[asset][timeframe].append({
                    "signal": signal_data["signal"],
                    "score": signal_data["score"],
                    "timestamp": timestamp
                })
                
                # Trim history
                max_history = self.history_max_days * 24  # Assuming hourly updates
                if len(self.signal_history[asset][timeframe]) > max_history:
                    self.signal_history[asset][timeframe] = self.signal_history[asset][timeframe][-max_history:]
    
    def _update_signal_metrics(self) -> None:
        """Update signal metrics."""
        # Calculate average signal strength
        total_strength = 0.0
        count = 0
        
        for asset, asset_signals in self.active_signals.items():
            for timeframe, signal_data in asset_signals.items():
                total_strength += abs(signal_data["score"])
                count += 1
        
        if count > 0:
            self.signal_metrics["average_signal_strength"] = total_strength / count
        
        # Calculate hit rate (placeholder)
        # In a real implementation, this would compare past signals to actual price movements
        self.signal_metrics["signal_hit_rate"] = 0.5  # Placeholder
    
    def _load_signal_history(self) -> None:
        """Load signal history from files."""
        try:
            history_file = os.path.join(self.signal_dir, "signal_history.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    self.signal_history = json.load(f)
                
                self.logger.info(f"Loaded signal history from {history_file}")
            else:
                self.logger.info("No signal history file found, starting fresh")
        
        except Exception as e:
            self.logger.error(f"Error loading signal history: {str(e)}")
            self.signal_history = {}
    
    def _save_signal_history(self) -> None:
        """Save signal history to files."""
        try:
            history_file = os.path.join(self.signal_dir, "signal_history.json")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.signal_history, f, indent=2)
            
            self.logger.info(f"Saved signal history to {history_file}")
        
        except Exception as e:
            self.logger.error(f"Error saving signal history: {str(e)}")
    
    def get_active_signals(self, 
                         asset: Optional[str] = None, 
                         timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Get active signals.
        
        Args:
            asset: Asset to get signals for (if None, returns all assets)
            timeframe: Timeframe to get signals for (if None, returns all timeframes)
            
        Returns:
            Dictionary with active signals
        """
        if asset:
            if asset not in self.active_signals:
                return {}
            
            if timeframe:
                if timeframe not in self.active_signals[asset]:
                    return {}
                
                return {asset: {timeframe: self.active_signals[asset][timeframe]}}
            
            return {asset: self.active_signals[asset]}
        
        if timeframe:
            result = {}
            
            for asset, asset_signals in self.active_signals.items():
                if timeframe in asset_signals:
                    result.setdefault(asset, {})[timeframe] = asset_signals[timeframe]
            
            return result
        
        return self.active_signals
    
    def get_signal_history(self, 
                         asset: str, 
                         timeframe: str,
                         days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history for an asset and timeframe.
        
        Args:
            asset: Asset to get history for
            timeframe: Timeframe to get history for
            days: Number of days to get history for (if None, returns all available)
            
        Returns:
            List of historical signals
        """
        if asset not in self.signal_history or timeframe not in self.signal_history[asset]:
            return []
        
        history = self.signal_history[asset][timeframe]
        
        if days:
            # Filter by days
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            history = [s for s in history if s["timestamp"] >= cutoff]
        
        return history
    
    def backtest_signals(self, 
                       asset: str, 
                       timeframe: str,
                       start_date: str,
                       end_date: Optional[str] = None,
                       processor: str = "weighted_average") -> Dict[str, Any]:
        """
        Backtest signals for an asset and timeframe.
        
        Args:
            asset: Asset to backtest
            timeframe: Timeframe to backtest
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            processor: Signal processor to use
            
        Returns:
            Dictionary with backtest results
        """
        # Placeholder for backtesting
        # In a real implementation, this would generate historical signals
        # and compare them to actual price movements
        return {
            "asset": asset,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date or datetime.now().strftime("%Y-%m-%d"),
            "processor": processor,
            "win_rate": 0.5,
            "profit_factor": 1.2,
            "total_trades": 10,
            "average_profit": 2.5,
            "max_drawdown": 5.0
        }
    
    def register_custom_source(self, name: str, source_func: Callable) -> None:
        """
        Register a custom signal source.
        
        Args:
            name: Source name
            source_func: Function that returns signals
        """
        self.signal_sources[name] = source_func
        self.logger.info(f"Registered custom signal source: {name}")
    
    def register_custom_processor(self, name: str, processor_func: Callable) -> None:
        """
        Register a custom signal processor.
        
        Args:
            name: Processor name
            processor_func: Function that processes signals
        """
        self.signal_processors[name] = processor_func
        self.logger.info(f"Registered custom signal processor: {name}")
    
    def update_signal_weights(self, weights: Dict[str, float]) -> None:
        """
        Update signal weights.
        
        Args:
            weights: New weights for signal sources
        """
        self.signal_weights.update(weights)
        self.logger.info(f"Updated signal weights: {self.signal_weights}")
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "market_analysis_health": self.market_analysis.get_health(),
            "signal_sources": len(self.signal_sources),
            "signal_processors": len(self.signal_processors),
            "active_signals": self.signal_metrics["active_signals_count"],
            "signal_metrics": self.signal_metrics
        }


# Test function
def test_signal_integration_service():
    """Test the signal integration service functionality."""
    # Create signal integration service
    service = SignalIntegrationService()
    
    # Start the service
    service.start()
    
    try:
        # Update signals
        signals = service.update_signals()
        
        print("Active Signals:")
        for asset, asset_signals in signals.items():
            for timeframe, signal_data in asset_signals.items():
                print(f"{asset} {timeframe}: {signal_data['signal']} (Score: {signal_data['score']:.3f})")
        
        # Test different processors
        for processor in ["default", "weighted_average", "majority_vote", "threshold_filter"]:
            print(f"\nTesting {processor} processor:")
            integrated = service.integrate_signals(service.signal_sources, processor)
            
            for asset, asset_signals in integrated.items():
                for timeframe, signal_data in asset_signals.items():
                    print(f"{asset} {timeframe}: {signal_data['signal']} (Score: {signal_data['score']:.3f})")
        
        # Check health
        health = service.check_health()
        print(f"\nService health: {health}")
    
    finally:
        # Stop the service
        service.stop()


if __name__ == "__main__":
    test_signal_integration_service()
