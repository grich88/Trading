"""
Analysis service module.

This module provides a unified service for market analysis, including
technical indicators, pattern recognition, and signal generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

# Import base service
from src.services.base_service import BaseService

# Import utilities
from src.utils import exception_handler, performance_monitor, ModelError

# Import configuration
from src.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_MIN_ABS_SCORE
)


class AnalysisService(BaseService):
    """
    Unified service for market analysis.
    
    This service provides centralized functionality for:
    - Technical indicator analysis
    - Pattern recognition
    - Signal generation
    - Cross-asset analysis
    """
    
    def __init__(self, 
                 default_window_size: int = DEFAULT_WINDOW_SIZE,
                 default_min_abs_score: float = DEFAULT_MIN_ABS_SCORE,
                 **kwargs: Any):
        """
        Initialize the analysis service.
        
        Args:
            default_window_size: Default window size for analysis
            default_min_abs_score: Default minimum absolute score for signals
            **kwargs: Additional arguments for BaseService
        """
        super().__init__("AnalysisService", **kwargs)
        
        self.default_window_size = default_window_size
        self.default_min_abs_score = default_min_abs_score
        
        # Initialize models and analyzers
        self.models = {}
        self.analyzers = {}
        
        self.logger.info("Analysis service initialized")
    
    def register_model(self, name: str, model_instance: Any) -> None:
        """
        Register a model with the analysis service.
        
        Args:
            name: Model name
            model_instance: Model instance
        """
        self.models[name] = model_instance
        self.logger.info(f"Registered model: {name}")
    
    def register_analyzer(self, name: str, analyzer_instance: Any) -> None:
        """
        Register an analyzer with the analysis service.
        
        Args:
            name: Analyzer name
            analyzer_instance: Analyzer instance
        """
        self.analyzers[name] = analyzer_instance
        self.logger.info(f"Registered analyzer: {name}")
    
    @performance_monitor()
    def analyze_rsi_volume(self, 
                         df: pd.DataFrame, 
                         window_size: Optional[int] = None,
                         webtrend_status: Optional[bool] = None,
                         liquidation_data: Optional[Dict[str, Any]] = None,
                         asset_type: str = "BTC") -> Dict[str, Any]:
        """
        Analyze RSI and volume patterns.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            window_size: Window size for analysis
            webtrend_status: WebTrend status (True for uptrend, False for downtrend)
            liquidation_data: Liquidation heatmap data
            asset_type: Asset type (BTC, SOL, BONK)
            
        Returns:
            Analysis results
        """
        try:
            # Import the model dynamically to avoid circular imports
            from src.models.rsi_volume_model import EnhancedRsiVolumePredictor
            
            # Set default window size
            window_size = window_size or self.default_window_size
            
            # Get the last window of data
            last_window = df.iloc[-window_size:]
            
            # Infer WebTrend status if not provided
            if webtrend_status is None:
                if "ema20" in df.columns and "ema50" in df.columns and "ema100" in df.columns:
                    ema20 = float(df["ema20"].iloc[-1])
                    ema50 = float(df["ema50"].iloc[-1])
                    ema100 = float(df["ema100"].iloc[-1])
                    last_price = float(df["close"].iloc[-1])
                    webtrend_status = (last_price > ema20) and (ema20 > ema50) and (ema50 > ema100)
                else:
                    webtrend_status = False
            
            # Create predictor
            predictor = EnhancedRsiVolumePredictor(
                rsi_sma_series=last_window["rsi_sma"].values,
                rsi_raw_series=last_window["rsi_raw"].values,
                volume_series=last_window["volume"].values,
                price_series=last_window["close"].values,
                liquidation_data=liquidation_data,
                webtrend_status=webtrend_status,
                asset_type=asset_type
            )
            
            # Get analysis
            analysis = predictor.get_full_analysis()
            
            return analysis
        
        except Exception as e:
            self.logger.error(f"Error in RSI volume analysis: {str(e)}")
            raise ModelError(f"Failed to analyze RSI and volume: {str(e)}")
    
    @performance_monitor()
    def detect_divergence(self, df: pd.DataFrame, lookback_period: int = 10) -> Dict[str, Any]:
        """
        Detect divergence between price and indicators.
        
        Args:
            df: DataFrame with OHLCV and indicator data
            lookback_period: Period to look back for divergence
            
        Returns:
            Divergence analysis results
        """
        try:
            # Get the last window of data
            last_window = df.iloc[-lookback_period:]
            
            # Check for required columns
            required_columns = ["close", "rsi_raw", "volume"]
            for col in required_columns:
                if col not in last_window.columns:
                    raise ModelError(f"Required column not found: {col}")
            
            # Extract data
            price_highs = last_window["close"].values
            rsi_highs = last_window["rsi_raw"].values
            volume_highs = last_window["volume"].values
            
            # Bearish divergence: price makes new high but RSI doesn't
            price_div = price_highs[-1] > max(price_highs[:-1])
            rsi_div = rsi_highs[-1] < max(rsi_highs[:-1])
            volume_div = volume_highs[-1] < max(volume_highs[:-1])
            
            bearish_divergence = price_div and (rsi_div or volume_div)
            
            # Bullish divergence: price makes new low but RSI doesn't
            price_low = price_highs[-1] < min(price_highs[:-1])
            rsi_up = rsi_highs[-1] > min(rsi_highs[:-1])
            volume_fade = volume_highs[-1] < max(volume_highs[:-1])
            
            bullish_divergence = price_low and (rsi_up or volume_fade)
            
            # Create result
            result = {
                "bearish_divergence": bearish_divergence,
                "bullish_divergence": bullish_divergence,
                "score": -0.5 if bearish_divergence else (0.5 if bullish_divergence else 0.0)
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error detecting divergence: {str(e)}")
            raise ModelError(f"Failed to detect divergence: {str(e)}")
    
    @performance_monitor()
    def calculate_webtrend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate WebTrend indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            WebTrend analysis results
        """
        try:
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate WebTrend
            # WebTrend is a proprietary indicator, this is a simplified version
            # based on moving averages and volatility
            
            # Calculate ATR
            high_low = result_df['high'] - result_df['low']
            high_close = (result_df['high'] - result_df['close'].shift()).abs()
            low_close = (result_df['low'] - result_df['close'].shift()).abs()
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Calculate WebTrend bands
            ema = result_df['close'].ewm(span=21, adjust=False).mean()
            result_df['wt_mid'] = ema
            result_df['wt_upper'] = ema + (atr * 1.5)
            result_df['wt_lower'] = ema - (atr * 1.5)
            
            # Determine WebTrend status
            close = result_df['close'].iloc[-1]
            mid = result_df['wt_mid'].iloc[-1]
            upper = result_df['wt_upper'].iloc[-1]
            lower = result_df['wt_lower'].iloc[-1]
            
            status = close > mid
            
            # Get WebTrend lines
            lines = {
                "mid": mid,
                "upper": upper,
                "lower": lower,
                "status": status
            }
            
            return {
                "dataframe": result_df,
                "lines": lines
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating WebTrend: {str(e)}")
            raise ModelError(f"Failed to calculate WebTrend: {str(e)}")
    
    @performance_monitor()
    def analyze_cross_asset(self, 
                          data: Dict[str, pd.DataFrame],
                          reference_asset: str = "BTC") -> Dict[str, Dict[str, float]]:
        """
        Analyze cross-asset relationships.
        
        Args:
            data: Dictionary of DataFrames for each asset
            reference_asset: Reference asset for comparison
            
        Returns:
            Dictionary of cross-asset biases
        """
        try:
            biases = {}
            
            # Check if reference asset is in data
            if reference_asset not in data:
                self.logger.warning(f"Reference asset {reference_asset} not found in data")
                return biases
            
            # Get reference asset data
            ref_data = data[reference_asset]
            
            for asset, df in data.items():
                if asset == reference_asset:
                    continue
                
                # Calculate correlation
                correlation = df['close'].corr(ref_data['close'])
                
                # Calculate lead-lag relationship
                # Simple approach: check if reference asset leads or lags
                lead_lag = 0.0
                
                # Calculate momentum difference
                ref_momentum = ref_data['close'].pct_change(5).iloc[-1]
                asset_momentum = df['close'].pct_change(5).iloc[-1]
                momentum_diff = asset_momentum - ref_momentum
                
                # Calculate bias
                bias = 0.0
                if abs(correlation) > 0.5:
                    # Strong correlation
                    if correlation > 0:
                        # Positive correlation
                        bias = ref_momentum * 0.5  # 50% weight
                    else:
                        # Negative correlation
                        bias = -ref_momentum * 0.5  # 50% weight
                
                # Add momentum difference with less weight
                bias += momentum_diff * 0.2  # 20% weight
                
                # Cap bias
                bias = max(-0.5, min(0.5, bias))
                
                # Add to results
                biases[asset] = {
                    "correlation": correlation,
                    "lead_lag": lead_lag,
                    "momentum_diff": momentum_diff,
                    "bias": bias,
                    "confidence": abs(correlation)
                }
            
            return biases
        
        except Exception as e:
            self.logger.error(f"Error analyzing cross-asset relationships: {str(e)}")
            raise ModelError(f"Failed to analyze cross-asset relationships: {str(e)}")
    
    def generate_signals(self, 
                       data: Dict[str, pd.DataFrame],
                       webtrend_data: Optional[Dict[str, Dict[str, Any]]] = None,
                       cross_asset_biases: Optional[Dict[str, Dict[str, float]]] = None,
                       min_abs_score: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all assets.
        
        Args:
            data: Dictionary of DataFrames for each asset
            webtrend_data: WebTrend data for each asset
            cross_asset_biases: Cross-asset biases
            min_abs_score: Minimum absolute score for signals
            
        Returns:
            Dictionary of signals for each asset
        """
        signals = {}
        min_abs_score = min_abs_score or self.default_min_abs_score
        
        for asset, df in data.items():
            try:
                self.logger.info(f"Generating signal for {asset}")
                
                # Get WebTrend status
                webtrend_status = False
                if webtrend_data and asset in webtrend_data:
                    webtrend_status = webtrend_data[asset]["lines"].get("status", False)
                
                # Get analysis
                analysis = self.analyze_rsi_volume(
                    df=df,
                    webtrend_status=webtrend_status,
                    asset_type=asset
                )
                
                # Apply cross-asset bias if available
                if cross_asset_biases and asset in cross_asset_biases:
                    bias = cross_asset_biases[asset].get("bias", 0.0)
                    if abs(bias) > 0.01:  # Only apply significant bias
                        score = analysis["final_score"]
                        adjusted_score = score + (bias * 0.3)  # 30% weight
                        adjusted_score = max(-1.0, min(1.0, adjusted_score))
                        analysis["final_score"] = adjusted_score
                        analysis["cross_asset_bias"] = bias
                        
                        # Recalculate signal if score changed significantly
                        if abs(adjusted_score - score) > 0.1:
                            if adjusted_score > 0.6:
                                analysis["signal"] = "STRONG BUY"
                            elif adjusted_score > 0.3:
                                analysis["signal"] = "BUY"
                            elif adjusted_score >= -0.3:
                                analysis["signal"] = "NEUTRAL"
                            elif adjusted_score >= -0.6:
                                analysis["signal"] = "SELL"
                            else:
                                analysis["signal"] = "STRONG SELL"
                
                # Apply minimum absolute score filter
                if abs(analysis["final_score"]) < min_abs_score:
                    analysis["signal"] = "NEUTRAL"
                    analysis["filtered"] = True
                
                signals[asset] = analysis
                
                # Log signal
                self.logger.info(f"Signal for {asset}: {analysis['signal']} (Score: {analysis['final_score']:.3f})")
            
            except Exception as e:
                self.logger.error(f"Error generating signal for {asset}: {str(e)}")
        
        return signals
    
    def _check_service_health(self) -> Dict[str, Any]:
        """
        Check service-specific health.
        
        Returns:
            Dictionary with service-specific health metrics
        """
        return {
            "registered_models": len(self.models),
            "registered_analyzers": len(self.analyzers)
        }


# Test function
def test_analysis_service():
    """Test the analysis service functionality."""
    # Import necessary modules for testing
    import os
    import pandas as pd
    from src.services.data_service import DataService
    
    # Create data service
    data_service = DataService()
    
    # Create analysis service
    analysis_service = AnalysisService()
    
    # Start the services
    data_service.start()
    analysis_service.start()
    
    try:
        # Fetch data for BTC
        df = data_service.fetch_historical_data(
            symbol="BTC/USDT",
            timeframe="4h",
            start_date="2023-01-01",
            end_date="2023-01-31"
        )
        
        # Calculate indicators
        df = data_service.calculate_indicators(df)
        
        # Analyze RSI and volume
        analysis = analysis_service.analyze_rsi_volume(
            df=df,
            asset_type="BTC"
        )
        
        print("RSI Volume Analysis:")
        print(f"Signal: {analysis['signal']}")
        print(f"Score: {analysis['final_score']}")
        print(f"Targets: TP1={analysis['targets']['TP1']}, TP2={analysis['targets']['TP2']}, SL={analysis['targets']['SL']}")
        
        # Detect divergence
        divergence = analysis_service.detect_divergence(df)
        print("\nDivergence Analysis:")
        print(f"Bearish Divergence: {divergence['bearish_divergence']}")
        print(f"Bullish Divergence: {divergence['bullish_divergence']}")
        print(f"Score: {divergence['score']}")
        
        # Calculate WebTrend
        webtrend = analysis_service.calculate_webtrend(df)
        print("\nWebTrend Analysis:")
        print(f"Status: {'Bullish' if webtrend['lines']['status'] else 'Bearish'}")
        print(f"Mid: {webtrend['lines']['mid']}")
        print(f"Upper: {webtrend['lines']['upper']}")
        print(f"Lower: {webtrend['lines']['lower']}")
        
        # Fetch data for multiple assets
        assets_data = {
            "BTC": df,
            "SOL": data_service.fetch_historical_data(
                symbol="SOL/USDT",
                timeframe="4h",
                start_date="2023-01-01",
                end_date="2023-01-31"
            )
        }
        
        # Calculate indicators for SOL
        assets_data["SOL"] = data_service.calculate_indicators(assets_data["SOL"])
        
        # Analyze cross-asset relationships
        cross_asset_biases = analysis_service.analyze_cross_asset(assets_data)
        print("\nCross-Asset Analysis:")
        for asset, bias in cross_asset_biases.items():
            print(f"{asset}: Bias={bias['bias']:.3f}, Correlation={bias['correlation']:.3f}")
        
        # Generate signals
        signals = analysis_service.generate_signals(
            data=assets_data,
            webtrend_data={"BTC": webtrend, "SOL": analysis_service.calculate_webtrend(assets_data["SOL"])},
            cross_asset_biases=cross_asset_biases
        )
        
        print("\nSignals:")
        for asset, signal in signals.items():
            print(f"{asset}: {signal['signal']} (Score: {signal['final_score']:.3f})")
        
        # Check health
        health = analysis_service.check_health()
        print(f"\nService health: {health}")
    
    finally:
        # Stop the services
        analysis_service.stop()
        data_service.stop()


if __name__ == "__main__":
    test_analysis_service()
