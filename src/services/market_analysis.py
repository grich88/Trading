"""
Market analysis service module.

This module provides specialized services for market analysis, including
advanced technical indicators, pattern recognition, and multi-signal integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
import json
import os

# Import base service
from src.services.base_service import LongRunningService

# Import analysis service
from src.services.analysis_service import AnalysisService

# Import data service
from src.services.data_service import DataService

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
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_ABS_SCORE,
    ANALYSIS_RESULTS_DIR,
    DEFAULT_TIMEFRAME,
    SIGNAL_WEIGHTS
)


class MarketAnalysisService(LongRunningService):
    """
    Specialized service for market analysis.
    
    This service extends LongRunningService to provide:
    - Advanced technical indicator analysis
    - Multi-timeframe analysis
    - Pattern recognition
    - Signal integration
    - Market regime detection
    - Automated analysis scheduling
    """
    
    def __init__(self, 
                 default_window_size: int = DEFAULT_WINDOW_SIZE,
                 default_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
                 analysis_results_dir: str = ANALYSIS_RESULTS_DIR,
                 default_timeframe: str = DEFAULT_TIMEFRAME,
                 signal_weights: Optional[Dict[str, float]] = None,
                 **kwargs: Any):
        """
        Initialize the market analysis service.
        
        Args:
            default_window_size: Default window size for analysis
            default_min_abs_score: Default minimum absolute score for signals
            analysis_results_dir: Directory for storing analysis results
            default_timeframe: Default timeframe for analysis
            signal_weights: Weights for different signal categories
            **kwargs: Additional arguments for LongRunningService
        """
        super().__init__("MarketAnalysisService", **kwargs)
        
        self.default_window_size = default_window_size
        self.default_min_abs_score = default_min_abs_score
        self.analysis_results_dir = analysis_results_dir
        self.default_timeframe = default_timeframe
        self.signal_weights = signal_weights or SIGNAL_WEIGHTS
        
        # Create analysis service for core analysis
        self.analysis_service = AnalysisService(
            default_window_size=default_window_size,
            default_min_abs_score=default_min_abs_score
        )
        
        # Create data service for data access
        self.data_service = DataService()
        
        # Analysis configuration
        self.analysis_config = {
            "timeframes": ["1h", "4h", "1d"],
            "assets": ["BTC", "ETH", "SOL"],
            "indicators": [
                "rsi_volume", "divergence", "webtrend", "cross_asset",
                "liquidity", "funding", "open_interest", "volume_delta"
            ],
            "schedule_interval_minutes": 60
        }
        
        # Analysis results cache
        self.analysis_results = {}
        self.last_analysis_time = {}
        
        # Analysis task queue
        self.analysis_tasks = []
        
        # Signal processors
        self.signal_processors = {}
        
        # Create results directory
        os.makedirs(self.analysis_results_dir, exist_ok=True)
        
        self.logger.info("Market analysis service initialized")
    
    def _start_service(self) -> None:
        """Start the market analysis service."""
        super()._start_service()
        
        # Start dependent services
        self.analysis_service.start()
        self.data_service.start()
        
        # Register signal processors
        self._register_signal_processors()
        
        # Schedule initial analysis
        self._schedule_analysis_for_all_assets()
        
        self.logger.info("Market analysis service started")
    
    def _stop_service(self) -> None:
        """Stop the market analysis service."""
        # Stop dependent services
        self.analysis_service.stop()
        self.data_service.stop()
        
        super()._stop_service()
        self.logger.info("Market analysis service stopped")
    
    def _register_signal_processors(self) -> None:
        """Register signal processors."""
        # Register standard signal processors
        self.signal_processors = {
            "rsi_volume": self._process_rsi_volume_signal,
            "divergence": self._process_divergence_signal,
            "webtrend": self._process_webtrend_signal,
            "cross_asset": self._process_cross_asset_signal,
            "liquidity": self._process_liquidity_signal,
            "funding": self._process_funding_signal,
            "open_interest": self._process_open_interest_signal,
            "volume_delta": self._process_volume_delta_signal
        }
        
        self.logger.info(f"Registered {len(self.signal_processors)} signal processors")
    
    def _schedule_analysis_for_all_assets(self) -> None:
        """Schedule analysis for all assets."""
        for asset in self.analysis_config["assets"]:
            for timeframe in self.analysis_config["timeframes"]:
                self._schedule_analysis(asset, timeframe)
    
    def _schedule_analysis(self, asset: str, timeframe: str) -> None:
        """
        Schedule analysis for an asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe
        """
        task = {
            "type": "analysis",
            "asset": asset,
            "timeframe": timeframe,
            "scheduled_time": datetime.now()
        }
        
        self.add_task(task)
        self.logger.debug(f"Scheduled analysis for {asset} {timeframe}")
    
    def _process_task(self, task: Dict[str, Any]) -> None:
        """
        Process a task from the queue.
        
        Args:
            task: Task to process
        """
        try:
            task_type = task.get("type")
            
            if task_type == "analysis":
                asset = task.get("asset")
                timeframe = task.get("timeframe")
                
                if asset and timeframe:
                    self._run_analysis_for_asset(asset, timeframe)
                    
                    # Schedule next analysis
                    interval_minutes = self.analysis_config.get("schedule_interval_minutes", 60)
                    next_task = task.copy()
                    next_task["scheduled_time"] = datetime.now() + timedelta(minutes=interval_minutes)
                    self.add_task(next_task)
            
            elif task_type == "custom_analysis":
                # Handle custom analysis tasks
                analysis_func = task.get("analysis_func")
                args = task.get("args", [])
                kwargs = task.get("kwargs", {})
                
                if analysis_func and callable(analysis_func):
                    analysis_func(*args, **kwargs)
            
            else:
                self.logger.warning(f"Unknown task type: {task_type}")
        
        except Exception as e:
            self.logger.error(f"Error processing task: {str(e)}")
    
    @performance_monitor()
    def _run_analysis_for_asset(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Run analysis for an asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Analysis results
        """
        try:
            self.logger.info(f"Running analysis for {asset} {timeframe}")
            
            # Create full symbol
            symbol = f"{asset}/USDT"
            
            # Get data
            df = self.data_service.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                use_cache=False  # Force fresh data
            )
            
            # Calculate indicators
            df = self.data_service.calculate_indicators(df)
            
            # Run analyses
            results = {}
            
            # 1. RSI Volume Analysis
            results["rsi_volume"] = self.analysis_service.analyze_rsi_volume(
                df=df,
                asset_type=asset
            )
            
            # 2. Divergence Analysis
            results["divergence"] = self.analysis_service.detect_divergence(df)
            
            # 3. WebTrend Analysis
            webtrend_result = self.analysis_service.calculate_webtrend(df)
            results["webtrend"] = {
                "status": webtrend_result["lines"]["status"],
                "lines": webtrend_result["lines"],
                "score": 0.5 if webtrend_result["lines"]["status"] else -0.5
            }
            
            # 4. Get data for cross-asset analysis if this is the reference asset
            if asset == "BTC":
                # Get data for all assets for cross-asset analysis
                assets_data = {asset: df}
                
                for other_asset in self.analysis_config["assets"]:
                    if other_asset != asset:
                        try:
                            other_symbol = f"{other_asset}/USDT"
                            other_df = self.data_service.fetch_historical_data(
                                symbol=other_symbol,
                                timeframe=timeframe
                            )
                            other_df = self.data_service.calculate_indicators(other_df)
                            assets_data[other_asset] = other_df
                        except Exception as e:
                            self.logger.error(f"Error fetching data for {other_asset}: {str(e)}")
                
                # Cross-asset analysis
                cross_asset_biases = self.analysis_service.analyze_cross_asset(
                    data=assets_data,
                    reference_asset=asset
                )
                
                results["cross_asset"] = cross_asset_biases
            
            # 5. Additional analyses (placeholders for now)
            results["liquidity"] = self._analyze_liquidity(df, asset)
            results["funding"] = self._analyze_funding(df, asset)
            results["open_interest"] = self._analyze_open_interest(df, asset)
            results["volume_delta"] = self._analyze_volume_delta(df, asset)
            
            # Integrate signals
            integrated_signal = self._integrate_signals(results, asset)
            results["integrated"] = integrated_signal
            
            # Store results
            self.analysis_results.setdefault(asset, {})[timeframe] = results
            self.last_analysis_time.setdefault(asset, {})[timeframe] = datetime.now()
            
            # Save results to file
            self._save_analysis_results(asset, timeframe, results)
            
            self.logger.info(
                f"Analysis for {asset} {timeframe} completed. "
                f"Signal: {integrated_signal['signal']}, "
                f"Score: {integrated_signal['score']:.3f}"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error running analysis for {asset} {timeframe}: {str(e)}")
            raise ModelError(f"Failed to analyze {asset} {timeframe}: {str(e)}")
    
    def _analyze_liquidity(self, df: pd.DataFrame, asset: str) -> Dict[str, Any]:
        """
        Analyze liquidity (placeholder).
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol
            
        Returns:
            Liquidity analysis results
        """
        # Placeholder for liquidity analysis
        # In a real implementation, this would analyze liquidation levels
        return {
            "score": 0.0,
            "signal": "NEUTRAL",
            "levels": {
                "support": float(df["low"].iloc[-10:].min()),
                "resistance": float(df["high"].iloc[-10:].max())
            }
        }
    
    def _analyze_funding(self, df: pd.DataFrame, asset: str) -> Dict[str, Any]:
        """
        Analyze funding rates (placeholder).
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol
            
        Returns:
            Funding analysis results
        """
        # Placeholder for funding rate analysis
        # In a real implementation, this would fetch and analyze funding rates
        return {
            "score": 0.0,
            "signal": "NEUTRAL",
            "current_rate": 0.0,
            "average_rate": 0.0
        }
    
    def _analyze_open_interest(self, df: pd.DataFrame, asset: str) -> Dict[str, Any]:
        """
        Analyze open interest (placeholder).
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol
            
        Returns:
            Open interest analysis results
        """
        # Placeholder for open interest analysis
        # In a real implementation, this would fetch and analyze OI data
        return {
            "score": 0.0,
            "signal": "NEUTRAL",
            "oi_change": 0.0,
            "price_oi_divergence": False
        }
    
    def _analyze_volume_delta(self, df: pd.DataFrame, asset: str) -> Dict[str, Any]:
        """
        Analyze volume delta (placeholder).
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset symbol
            
        Returns:
            Volume delta analysis results
        """
        # Placeholder for volume delta analysis
        # In a real implementation, this would calculate buy/sell volume imbalance
        
        # Simple volume change calculation
        volume_change = df["volume"].pct_change(5).iloc[-1]
        price_change = df["close"].pct_change(5).iloc[-1]
        
        # Volume-price divergence
        divergence = (volume_change > 0.1 and price_change < 0) or (volume_change < -0.1 and price_change > 0)
        
        # Score based on volume change
        score = 0.0
        if volume_change > 0.2 and price_change > 0:
            score = 0.3
        elif volume_change < -0.2 and price_change < 0:
            score = -0.3
        
        return {
            "score": score,
            "signal": "BUY" if score > 0 else ("SELL" if score < 0 else "NEUTRAL"),
            "volume_change": float(volume_change),
            "price_change": float(price_change),
            "divergence": divergence
        }
    
    def _integrate_signals(self, results: Dict[str, Any], asset: str) -> Dict[str, Any]:
        """
        Integrate signals from different analyses.
        
        Args:
            results: Analysis results
            asset: Asset symbol
            
        Returns:
            Integrated signal
        """
        # Process each signal type
        processed_signals = {}
        for signal_type, processor in self.signal_processors.items():
            if signal_type in results:
                processed_signals[signal_type] = processor(results[signal_type], asset)
        
        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0
        
        for signal_type, signal in processed_signals.items():
            weight = self.signal_weights.get(signal_type, 1.0)
            total_weight += weight
            weighted_score += signal["score"] * weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.0
        
        # Cap score
        final_score = max(-1.0, min(1.0, final_score))
        
        # Determine signal
        signal = self._score_to_signal(final_score)
        
        # Create result
        integrated_signal = {
            "score": final_score,
            "signal": signal,
            "components": processed_signals,
            "timestamp": datetime.now().isoformat()
        }
        
        return integrated_signal
    
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
    
    def _process_rsi_volume_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process RSI volume signal.
        
        Args:
            analysis: RSI volume analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("final_score", 0.0),
            "signal": analysis.get("signal", "NEUTRAL"),
            "confidence": analysis.get("confidence", 0.5)
        }
    
    def _process_divergence_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process divergence signal.
        
        Args:
            analysis: Divergence analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": "SELL" if analysis.get("bearish_divergence", False) else (
                "BUY" if analysis.get("bullish_divergence", False) else "NEUTRAL"
            ),
            "confidence": 0.7 if (analysis.get("bearish_divergence", False) or 
                                analysis.get("bullish_divergence", False)) else 0.3
        }
    
    def _process_webtrend_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process WebTrend signal.
        
        Args:
            analysis: WebTrend analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": "BUY" if analysis.get("status", False) else "SELL",
            "confidence": 0.6
        }
    
    def _process_cross_asset_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process cross-asset signal.
        
        Args:
            analysis: Cross-asset analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        # For BTC, use average of biases for other assets
        if asset == "BTC":
            biases = [data.get("bias", 0.0) for other_asset, data in analysis.items()]
            avg_bias = sum(biases) / len(biases) if biases else 0.0
            
            return {
                "score": avg_bias,
                "signal": self._score_to_signal(avg_bias),
                "confidence": 0.5
            }
        
        # For other assets, use their bias relative to BTC
        elif asset in analysis:
            bias = analysis[asset].get("bias", 0.0)
            confidence = abs(analysis[asset].get("correlation", 0.0))
            
            return {
                "score": bias,
                "signal": self._score_to_signal(bias),
                "confidence": confidence
            }
        
        # Default
        return {
            "score": 0.0,
            "signal": "NEUTRAL",
            "confidence": 0.0
        }
    
    def _process_liquidity_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process liquidity signal.
        
        Args:
            analysis: Liquidity analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": analysis.get("signal", "NEUTRAL"),
            "confidence": 0.4
        }
    
    def _process_funding_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process funding signal.
        
        Args:
            analysis: Funding analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": analysis.get("signal", "NEUTRAL"),
            "confidence": 0.5
        }
    
    def _process_open_interest_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process open interest signal.
        
        Args:
            analysis: Open interest analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": analysis.get("signal", "NEUTRAL"),
            "confidence": 0.6
        }
    
    def _process_volume_delta_signal(self, analysis: Dict[str, Any], asset: str) -> Dict[str, float]:
        """
        Process volume delta signal.
        
        Args:
            analysis: Volume delta analysis
            asset: Asset symbol
            
        Returns:
            Processed signal
        """
        return {
            "score": analysis.get("score", 0.0),
            "signal": analysis.get("signal", "NEUTRAL"),
            "confidence": 0.5
        }
    
    def _save_analysis_results(self, asset: str, timeframe: str, results: Dict[str, Any]) -> None:
        """
        Save analysis results to file.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe
            results: Analysis results
        """
        try:
            # Create directory if it doesn't exist
            asset_dir = os.path.join(self.analysis_results_dir, asset)
            os.makedirs(asset_dir, exist_ok=True)
            
            # Create file path
            file_path = os.path.join(asset_dir, f"{timeframe}_latest.json")
            
            # Convert datetime objects to strings
            results_copy = self._prepare_results_for_json(results)
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            self.logger.debug(f"Saved analysis results to {file_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving analysis results: {str(e)}")
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare results for JSON serialization.
        
        Args:
            results: Analysis results
            
        Returns:
            JSON-serializable results
        """
        if isinstance(results, dict):
            return {k: self._prepare_results_for_json(v) for k, v in results.items()}
        elif isinstance(results, list):
            return [self._prepare_results_for_json(item) for item in results]
        elif isinstance(results, (np.int64, np.int32, np.int16, np.int8)):
            return int(results)
        elif isinstance(results, (np.float64, np.float32, np.float16)):
            return float(results)
        elif isinstance(results, (datetime, pd.Timestamp)):
            return results.isoformat()
        elif isinstance(results, pd.DataFrame):
            return "DataFrame object (not serialized)"
        elif isinstance(results, pd.Series):
            return "Series object (not serialized)"
        else:
            return results
    
    def get_latest_analysis(self, asset: str, timeframe: Optional[str] = None) -> Dict[str, Any]:
        """
        Get latest analysis for an asset.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe (if None, returns all timeframes)
            
        Returns:
            Latest analysis results
        """
        if asset not in self.analysis_results:
            self.logger.warning(f"No analysis results for {asset}")
            return {}
        
        if timeframe:
            if timeframe not in self.analysis_results[asset]:
                self.logger.warning(f"No analysis results for {asset} {timeframe}")
                return {}
            
            return self.analysis_results[asset][timeframe]
        
        return self.analysis_results[asset]
    
    def run_analysis_now(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Run analysis immediately for an asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe
            
        Returns:
            Analysis results
        """
        return self._run_analysis_for_asset(asset, timeframe)
    
    def run_custom_analysis(self, 
                          analysis_func: Callable, 
                          *args: Any, 
                          schedule: bool = False, 
                          **kwargs: Any) -> Any:
        """
        Run a custom analysis function.
        
        Args:
            analysis_func: Analysis function to run
            *args: Arguments for the analysis function
            schedule: Whether to schedule the analysis or run immediately
            **kwargs: Keyword arguments for the analysis function
            
        Returns:
            Analysis results if run immediately, None if scheduled
        """
        if schedule:
            task = {
                "type": "custom_analysis",
                "analysis_func": analysis_func,
                "args": args,
                "kwargs": kwargs,
                "scheduled_time": datetime.now()
            }
            
            self.add_task(task)
            self.logger.debug(f"Scheduled custom analysis: {analysis_func.__name__}")
            return None
        
        return analysis_func(*args, **kwargs)
    
    def update_signal_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update signal weights.
        
        Args:
            new_weights: New weights for signal categories
        """
        self.signal_weights.update(new_weights)
        self.logger.info(f"Updated signal weights: {self.signal_weights}")
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "analysis_service_health": self.analysis_service.get_health(),
            "data_service_health": self.data_service.get_health(),
            "assets_analyzed": len(self.analysis_results),
            "signal_processors": len(self.signal_processors),
            "analysis_tasks": len(self.analysis_tasks)
        }


# Test function
def test_market_analysis_service():
    """Test the market analysis service functionality."""
    # Create market analysis service
    service = MarketAnalysisService()
    
    # Start the service
    service.start()
    
    try:
        # Run analysis for BTC
        results = service.run_analysis_now("BTC", "4h")
        
        print("BTC Analysis Results:")
        print(f"Integrated Signal: {results['integrated']['signal']}")
        print(f"Score: {results['integrated']['score']:.3f}")
        
        # Print component signals
        print("\nComponent Signals:")
        for component, signal in results["integrated"]["components"].items():
            print(f"{component}: {signal['signal']} (Score: {signal['score']:.3f})")
        
        # Check health
        health = service.check_health()
        print(f"\nService health: {health}")
    
    finally:
        # Stop the service
        service.stop()


if __name__ == "__main__":
    test_market_analysis_service()
