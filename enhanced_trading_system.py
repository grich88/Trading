import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

# Import all enhancement modules
from weight_optimizer import WeightOptimizer, load_optimized_weights
from signal_calibration import SignalCalibrator, get_calibrated_signal
from feature_analyzer import FeatureAnalyzer
from adaptive_timeframe import AdaptiveTimeframeSelector
from cross_asset_analyzer import CrossAssetAnalyzer
from webtrend_derivation import WebTrendDerivation

# Import existing modules
from historical_data_collector import fetch_historical_data, calculate_indicators
from enhanced_backtester import EnhancedBacktester
from updated_rsi_volume_model import EnhancedRsiVolumePredictor

# Configuration path
CONFIG_PATH = os.path.join("data", "configs")
SYSTEM_CONFIG_FILE = os.path.join(CONFIG_PATH, "system_config.json")

class EnhancedTradingSystem:
    """
    Integrated trading system that combines all enhancements.
    
    This class orchestrates the interaction between all enhancement modules
    to provide a comprehensive trading system with optimal predictive power.
    """
    
    def __init__(self, assets: List[str] = None):
        """
        Initialize the enhanced trading system.
        
        Parameters:
        -----------
        assets : list
            List of assets to analyze
        """
        self.assets = assets or ["BTC", "SOL", "BONK"]
        
        # Default parameters
        self.default_params = {
            "BTC": {
                "window_size": 60,
                "lookahead_candles": 10,
                "min_abs_score": 0.425,
                "regime_filter": True,
                "fee_bps": 5.0,
                "adaptive_cutoff": True,
                "min_agree_features": 1
            },
            "SOL": {
                "window_size": 40,
                "lookahead_candles": 10,
                "min_abs_score": 0.520,
                "regime_filter": True,
                "fee_bps": 5.0,
                "adaptive_cutoff": True,
                "min_agree_features": 1
            },
            "BONK": {
                "window_size": 40,
                "lookahead_candles": 10,
                "min_abs_score": 0.550,
                "regime_filter": True,
                "fee_bps": 5.0,
                "adaptive_cutoff": True,
                "min_agree_features": 1
            }
        }
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
        
        # Load custom config if available
        self._load_config()
        
        # Initialize components
        self.timeframe_selectors = {}
        self.calibrators = {}
        self.feature_analyzers = {}
        self.webtrend_derivers = {}
        
        for asset in self.assets:
            self.timeframe_selectors[asset] = AdaptiveTimeframeSelector(asset_type=asset)
            self.calibrators[asset] = SignalCalibrator(asset_type=asset)
            self.calibrators[asset].load_model()  # Load model if available
            self.feature_analyzers[asset] = FeatureAnalyzer(asset_type=asset)
            self.webtrend_derivers[asset] = WebTrendDerivation(asset_type=asset)
        
        # Initialize cross-asset analyzer
        self.cross_asset_analyzer = CrossAssetAnalyzer(assets=self.assets)
    
    def _load_config(self) -> None:
        """
        Load custom configuration from file.
        """
        if not os.path.exists(SYSTEM_CONFIG_FILE):
            return
        
        try:
            with open(SYSTEM_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Load asset parameters
            for asset, params in config.get("asset_params", {}).items():
                if asset in self.default_params:
                    self.default_params[asset].update(params)
        except Exception:
            pass
    
    def _save_config(self) -> None:
        """
        Save configuration to file.
        """
        config = {
            "asset_params": self.default_params,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(SYSTEM_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def fetch_data(self, 
                 start_date: str = None, 
                 end_date: str = None,
                 timeframe: str = "4h") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all assets.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        timeframe : str
            Timeframe for data
        
        Returns:
        --------
        dict
            Dictionary of DataFrames for each asset
        """
        # Set default date range if not provided
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        data = {}
        for asset in self.assets:
            symbol = f"{asset}/USDT"
            print(f"Fetching {timeframe} data for {symbol} ({start_date} to {end_date})...")
            
            df = fetch_historical_data(symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                data[asset] = calculate_indicators(df)
                print(f"  - Fetched {len(data[asset])} candles")
            else:
                print(f"  - Failed to fetch data")
        
        return data
    
    def fetch_multi_timeframe_data(self, 
                                 start_date: str = None, 
                                 end_date: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for multiple timeframes.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        
        Returns:
        --------
        dict
            Dictionary of DataFrames for each asset and timeframe
            Format: {asset: {timeframe: df}}
        """
        timeframes = ["1h", "4h", "1d"]
        
        data = {}
        for asset in self.assets:
            data[asset] = {}
            for tf in timeframes:
                df = fetch_historical_data(f"{asset}/USDT", timeframe=tf, start_date=start_date, end_date=end_date)
                if df is not None and not df.empty:
                    data[asset][tf] = calculate_indicators(df)
        
        return data
    
    def select_optimal_timeframes(self, multi_tf_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, str]:
        """
        Select optimal timeframe for each asset based on volatility.
        
        Parameters:
        -----------
        multi_tf_data : dict
            Dictionary of DataFrames for each asset and timeframe
        
        Returns:
        --------
        dict
            Dictionary of optimal timeframes for each asset
        """
        optimal_timeframes = {}
        
        for asset, timeframes in multi_tf_data.items():
            if "4h" in timeframes and not timeframes["4h"].empty:
                selector = self.timeframe_selectors[asset]
                optimal_tf = selector.get_optimal_timeframe(timeframes["4h"])
                optimal_timeframes[asset] = optimal_tf
                print(f"Optimal timeframe for {asset}: {optimal_tf}")
            else:
                optimal_timeframes[asset] = "4h"  # Default
        
        return optimal_timeframes
    
    def optimize_weights(self, data: Dict[str, pd.DataFrame], n_trials: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Optimize model weights for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        n_trials : int
            Number of optimization trials
        
        Returns:
        --------
        dict
            Dictionary of optimized weights for each asset
        """
        results = {}
        
        for asset, df in data.items():
            print(f"Optimizing weights for {asset}...")
            
            params = self.default_params[asset]
            
            optimizer = WeightOptimizer(
                data_df=df,
                asset_type=asset,
                window_size=params["window_size"],
                lookahead_candles=params["lookahead_candles"],
                min_abs_score=params["min_abs_score"],
                n_trials=n_trials,
                regime_filter=params["regime_filter"],
                fee_bps=params["fee_bps"],
                adaptive_cutoff=params["adaptive_cutoff"],
                min_agree_features=params["min_agree_features"]
            )
            
            weights_data = optimizer.run_and_save()
            results[asset] = weights_data
            
            print(f"  - Optimized weights saved (Sharpe: {weights_data['metadata']['sharpe']:.3f})")
        
        return results
    
    def train_calibration_models(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Train signal calibration models for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        """
        for asset, df in data.items():
            print(f"Training calibration model for {asset}...")
            
            params = self.default_params[asset]
            
            # Run backtest to get signals
            bt = EnhancedBacktester(
                df,
                asset_type=asset,
                regime_filter=params["regime_filter"],
                fee_bps=params["fee_bps"],
                adaptive_cutoff=params["adaptive_cutoff"],
                min_agree_features=params["min_agree_features"]
            )
            
            results = bt.run_backtest(
                window_size=params["window_size"],
                lookahead_candles=params["lookahead_candles"],
                min_abs_score=params["min_abs_score"]
            )
            
            # Train calibrator
            calibrator = self.calibrators[asset]
            
            try:
                X, y = calibrator.prepare_training_data(results)
                calibrator.train(X, y)
                calibrator.save_model()
                print(f"  - Calibration model trained with {len(X)} samples")
            except Exception as e:
                print(f"  - Error training calibration model: {e}")
    
    def analyze_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze feature importance for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        
        Returns:
        --------
        dict
            Dictionary of feature analysis results for each asset
        """
        results = {}
        
        for asset, df in data.items():
            print(f"Analyzing feature importance for {asset}...")
            
            params = self.default_params[asset]
            
            # Run backtest to get signals
            bt = EnhancedBacktester(
                df,
                asset_type=asset,
                regime_filter=params["regime_filter"],
                fee_bps=params["fee_bps"],
                adaptive_cutoff=params["adaptive_cutoff"],
                min_agree_features=params["min_agree_features"]
            )
            
            results_df = bt.run_backtest(
                window_size=params["window_size"],
                lookahead_candles=params["lookahead_candles"],
                min_abs_score=params["min_abs_score"]
            )
            
            # Analyze features
            analyzer = self.feature_analyzers[asset]
            
            try:
                analysis_results = analyzer.analyze_and_save(results_df)
                results[asset] = analysis_results
                
                print(f"  - Feature analysis completed")
                print(f"  - Top features:")
                for name, value in sorted(
                    analysis_results["permutation_importance"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]:
                    print(f"    - {name}: {value:.4f}")
            except Exception as e:
                print(f"  - Error analyzing features: {e}")
        
        return results
    
    def calculate_webtrend(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate WebTrend indicators for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        
        Returns:
        --------
        dict
            Dictionary of WebTrend results for each asset
        """
        results = {}
        
        for asset, df in data.items():
            print(f"Calculating WebTrend for {asset}...")
            
            webtrend = self.webtrend_derivers[asset]
            
            try:
                # Calculate WebTrend
                wt_df = webtrend.calculate_webtrend(df)
                
                # Get latest WebTrend lines
                lines = webtrend.get_webtrend_lines(wt_df)
                
                results[asset] = {
                    "dataframe": wt_df,
                    "lines": lines
                }
                
                print(f"  - WebTrend calculated: Status = {'Bullish' if lines.get('status', False) else 'Bearish'}")
            except Exception as e:
                print(f"  - Error calculating WebTrend: {e}")
        
        return results
    
    def analyze_cross_asset_relationships(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze cross-asset relationships.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        
        Returns:
        --------
        dict
            Cross-asset analysis results
        """
        print("Analyzing cross-asset relationships...")
        
        try:
            results = self.cross_asset_analyzer.analyze(data)
            
            # Print key findings
            print("\nKey Cross-Asset Relationships:")
            
            # Granger causality
            significant_gc = []
            for pair, res in results["granger_causality"].items():
                if res.get("significant", False):
                    significant_gc.append((pair, res["min_p_value"], res["best_lag"]))
            
            if significant_gc:
                print("\nSignificant Granger Causality:")
                for pair, p_value, lag in sorted(significant_gc, key=lambda x: x[1])[:3]:
                    print(f"  - {pair} (lag {lag}): p-value = {p_value:.4f}")
            
            # Lead-lag relationships
            strong_leadlag = []
            for pair, res in results["lead_lag_relationships"].items():
                corr = abs(res.get("max_correlation", 0))
                if corr > 0.3:
                    strong_leadlag.append((pair, res["max_correlation"], res["max_lag"]))
            
            if strong_leadlag:
                print("\nStrong Lead-Lag Relationships:")
                for pair, corr, lag in sorted(strong_leadlag, key=lambda x: abs(x[1]), reverse=True)[:3]:
                    print(f"  - {pair} (lag {lag}): correlation = {corr:.4f}")
            
            return results
        except Exception as e:
            print(f"Error in cross-asset analysis: {e}")
            return {}
    
    def get_cross_asset_biases(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Calculate cross-asset biases.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        
        Returns:
        --------
        dict
            Dictionary of cross-asset biases
        """
        biases = {}
        
        # BTC is the primary reference asset
        if "BTC" not in data:
            return biases
        
        for asset in self.assets:
            if asset == "BTC":
                continue
            
            bias_info = self.cross_asset_analyzer.get_cross_asset_bias(data, asset, "BTC")
            biases[asset] = bias_info
        
        return biases
    
    def run_backtest(self, 
                   data: Dict[str, pd.DataFrame], 
                   params: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run backtest for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        params : dict
            Custom parameters for each asset
        
        Returns:
        --------
        dict
            Dictionary of backtest results for each asset
        """
        results = {}
        
        for asset, df in data.items():
            print(f"Running backtest for {asset}...")
            
            # Get parameters
            asset_params = params.get(asset, {}) if params else {}
            default_params = self.default_params[asset]
            
            # Merge parameters
            p = default_params.copy()
            p.update(asset_params)
            
            # Load optimized weights
            weights = load_optimized_weights(asset)
            
            # Create backtester
            bt = EnhancedBacktester(
                df,
                asset_type=asset,
                regime_filter=p["regime_filter"],
                fee_bps=p["fee_bps"],
                adaptive_cutoff=p["adaptive_cutoff"],
                min_agree_features=p["min_agree_features"],
                weight_overrides=weights
            )
            
            # Run backtest
            results_df = bt.run_backtest(
                window_size=p["window_size"],
                lookahead_candles=p["lookahead_candles"],
                min_abs_score=p["min_abs_score"]
            )
            
            # Get performance metrics
            metrics = bt.performance()
            
            results[asset] = {
                "dataframe": results_df,
                "metrics": metrics,
                "parameters": p
            }
            
            # Print key metrics
            print(f"  - Total return: {metrics.get('total_return', 0.0)*100:.2f}%")
            print(f"  - Sharpe ratio: {metrics.get('sharpe', 0.0):.3f}")
            print(f"  - Win rate: {metrics.get('win_rate', 0.0)*100:.2f}%")
            print(f"  - Total trades: {metrics.get('total_trades', 0)}")
        
        return results
    
    def generate_signals(self, 
                       data: Dict[str, pd.DataFrame], 
                       webtrend_data: Dict[str, Dict[str, Any]] = None,
                       cross_asset_biases: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals for all assets.
        
        Parameters:
        -----------
        data : dict
            Dictionary of DataFrames for each asset
        webtrend_data : dict
            WebTrend data for each asset
        cross_asset_biases : dict
            Cross-asset biases
        
        Returns:
        --------
        dict
            Dictionary of signals for each asset
        """
        signals = {}
        
        for asset, df in data.items():
            print(f"Generating signal for {asset}...")
            
            params = self.default_params[asset]
            
            # Load optimized weights
            weights = load_optimized_weights(asset)
            
            # Get latest data window
            window_size = params["window_size"]
            last_window = df.iloc[-window_size:]
            
            # Get WebTrend status
            webtrend_status = False
            if webtrend_data and asset in webtrend_data:
                webtrend_status = webtrend_data[asset]["lines"].get("status", False)
            else:
                # Infer from EMAs
                ema20 = float(df["ema20"].iloc[-1]) if "ema20" in df.columns else np.nan
                ema50 = float(df["ema50"].iloc[-1]) if "ema50" in df.columns else np.nan
                ema100 = float(df["ema100"].iloc[-1]) if "ema100" in df.columns else np.nan
                last_price = float(df["close"].iloc[-1])
                
                if not np.isnan(ema20) and not np.isnan(ema50) and not np.isnan(ema100):
                    webtrend_status = (last_price > ema20) and (ema20 > ema50) and (ema50 > ema100)
            
            # Create predictor
            predictor = EnhancedRsiVolumePredictor(
                rsi_sma_series=last_window["rsi_sma"].values,
                rsi_raw_series=last_window["rsi_raw"].values,
                volume_series=last_window["volume"].values,
                price_series=last_window["close"].values,
                webtrend_status=webtrend_status,
                asset_type=asset,
                weight_overrides=weights
            )
            
            # Get analysis
            analysis = predictor.get_full_analysis()
            
            # Apply cross-asset bias if available
            if cross_asset_biases and asset in cross_asset_biases:
                bias = cross_asset_biases[asset].get("bias", 0.0)
                if abs(bias) > 0.01:  # Only apply significant bias
                    score = analysis["final_score"]
                    adjusted_score = predictor.apply_external_biases(bias)
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
            
            # Apply signal calibration if model is available
            calibrator = self.calibrators[asset]
            if hasattr(calibrator, 'model') and calibrator.model is not None:
                # Prepare features
                features = analysis["components"].copy()
                features["final_score"] = analysis["final_score"]
                features["signal"] = analysis["signal"]
                
                # Get TP/SL percentages
                tp_pct = 0.04  # Default 4%
                sl_pct = 0.02  # Default 2%
                
                # Predict success probability
                probability = calibrator.predict_success_probability(features)
                
                # Calculate expected value
                expected_value = calibrator.calculate_expected_value(probability, tp_pct, sl_pct)
                
                # Apply expected value filter
                calibrated_signal = analysis["signal"]
                if expected_value < 0.005 and analysis["signal"] != "NEUTRAL":  # 0.5% minimum expected value
                    calibrated_signal = "NEUTRAL"
                
                analysis["calibrated_signal"] = calibrated_signal
                analysis["success_probability"] = probability
                analysis["expected_value"] = expected_value
            
            signals[asset] = analysis
            
            # Print signal
            print(f"  - Signal: {analysis['signal']}")
            print(f"  - Score: {analysis['final_score']:.3f}")
            if "calibrated_signal" in analysis:
                print(f"  - Calibrated signal: {analysis['calibrated_signal']}")
                print(f"  - Success probability: {analysis['success_probability']:.2%}")
                print(f"  - Expected value: {analysis['expected_value']:.2%}")
            print(f"  - Targets: TP1={analysis['targets']['TP1']}, TP2={analysis['targets']['TP2']}, SL={analysis['targets']['SL']}")
        
        return signals
    
    def run_full_analysis(self, 
                        start_date: str = None, 
                        end_date: str = None,
                        optimize: bool = False) -> Dict[str, Any]:
        """
        Run full analysis pipeline.
        
        Parameters:
        -----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        optimize : bool
            Whether to run optimization steps
        
        Returns:
        --------
        dict
            Comprehensive analysis results
        """
        print("Running full analysis pipeline...")
        
        # Fetch data
        data = self.fetch_data(start_date, end_date)
        
        # Fetch multi-timeframe data if needed
        multi_tf_data = None
        if optimize:
            multi_tf_data = self.fetch_multi_timeframe_data(start_date, end_date)
            
            # Select optimal timeframes
            optimal_timeframes = self.select_optimal_timeframes(multi_tf_data)
            print(f"Optimal timeframes: {optimal_timeframes}")
        
        # Optimize weights if requested
        if optimize:
            self.optimize_weights(data)
        
        # Calculate WebTrend
        webtrend_data = self.calculate_webtrend(data)
        
        # Analyze cross-asset relationships
        cross_asset_results = self.analyze_cross_asset_relationships(data)
        
        # Calculate cross-asset biases
        cross_asset_biases = self.get_cross_asset_biases(data)
        
        # Run backtest
        backtest_results = self.run_backtest(data)
        
        # Train calibration models if optimizing
        if optimize:
            self.train_calibration_models(data)
        
        # Analyze features if optimizing
        feature_analysis = None
        if optimize:
            feature_analysis = self.analyze_features(data)
        
        # Generate signals
        signals = self.generate_signals(data, webtrend_data, cross_asset_biases)
        
        # Compile results
        results = {
            "signals": signals,
            "backtest_results": backtest_results,
            "webtrend_data": webtrend_data,
            "cross_asset_results": cross_asset_results,
            "cross_asset_biases": cross_asset_biases,
            "feature_analysis": feature_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
        return results


if __name__ == "__main__":
    # Example usage
    system = EnhancedTradingSystem()
    
    # Run full analysis
    results = system.run_full_analysis(
        start_date="2023-01-01",
        end_date=None,  # Current date
        optimize=True
    )
    
    # Print summary
    print("\nSummary of Signals:")
    for asset, signal in results["signals"].items():
        print(f"\n{asset}:")
        print(f"  Signal: {signal['signal']}")
        if "calibrated_signal" in signal:
            print(f"  Calibrated Signal: {signal['calibrated_signal']}")
            print(f"  Success Probability: {signal['success_probability']:.2%}")
            print(f"  Expected Value: {signal['expected_value']:.2%}")
        print(f"  Score: {signal['final_score']:.3f}")
        if "cross_asset_bias" in signal:
            print(f"  Cross-Asset Bias: {signal['cross_asset_bias']:.3f}")
        print(f"  Targets: TP1={signal['targets']['TP1']}, TP2={signal['targets']['TP2']}, SL={signal['targets']['SL']}")
    
    # Print cross-asset insights
    print("\nCross-Asset Insights:")
    for asset, bias in results["cross_asset_biases"].items():
        if abs(bias.get("bias", 0)) > 0.01:
            print(f"  BTC → {asset}: {bias['bias']:.3f} (confidence: {bias['confidence']:.2f})")
    
    # Print backtest performance
    print("\nBacktest Performance:")
    for asset, result in results["backtest_results"].items():
        metrics = result["metrics"]
        print(f"\n{asset}:")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  TP1 Accuracy: {metrics['tp1_accuracy']*100:.2f}%")
        print(f"  SL Hit Rate: {metrics['sl_hit_rate']*100:.2f}%")
