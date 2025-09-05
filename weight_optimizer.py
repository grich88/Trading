import os
import json
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from enhanced_backtester import EnhancedBacktester

# Configuration file path
CONFIG_PATH = os.path.join("data", "configs")
WEIGHTS_FILE = os.path.join(CONFIG_PATH, "optimized_weights.json")

class WeightOptimizer:
    """
    Optimizes model weights using Optuna for maximum predictive power.
    
    This class performs Bayesian optimization to find the optimal feature weights
    for each asset type, maximizing metrics like Sharpe ratio and win rate.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        asset_type: str = "BTC",
        window_size: int = 60,
        lookahead_candles: int = 10,
        min_abs_score: float = 0.425,
        n_trials: int = 50,
        static_liquidation_data: Optional[Dict[str, Any]] = None,
        regime_filter: bool = True,
        fee_bps: float = 5.0,
        adaptive_cutoff: bool = True,
        min_agree_features: int = 0,
    ):
        """
        Initialize the weight optimizer.
        
        Parameters:
        -----------
        data_df : pd.DataFrame
            Historical price data with indicators
        asset_type : str
            Asset to optimize for ("BTC", "SOL", "BONK")
        window_size : int
            Lookback window size for signals
        lookahead_candles : int
            Forward-looking window for TP/SL evaluation
        min_abs_score : float
            Minimum absolute score for signal generation
        n_trials : int
            Number of Optuna optimization trials
        static_liquidation_data : dict, optional
            Liquidation heatmap data
        regime_filter : bool
            Whether to apply trend regime filter
        fee_bps : float
            Transaction fee in basis points
        adaptive_cutoff : bool
            Whether to adjust cutoff based on volatility
        min_agree_features : int
            Minimum number of features that must agree on signal direction
        """
        self.data = data_df
        self.asset_type = asset_type
        self.window_size = window_size
        self.lookahead_candles = lookahead_candles
        self.min_abs_score = min_abs_score
        self.n_trials = n_trials
        self.static_liquidation_data = static_liquidation_data
        self.regime_filter = regime_filter
        self.fee_bps = fee_bps
        self.adaptive_cutoff = adaptive_cutoff
        self.min_agree_features = min_agree_features
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
        
        # Default weight ranges
        self.weight_ranges = {
            "w_rsi": (0.20, 0.40),
            "w_volume": (0.15, 0.35),
            "w_divergence": (0.05, 0.25),
            "w_liquidation": (0.05, 0.20),
            "w_webtrend": (0.02, 0.15),
            "w_features": (0.10, 0.25),
            "w_sentiment": (0.02, 0.15)
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Optuna trial object
        
        Returns:
        --------
        float
            Optimization metric (higher is better)
        """
        # Sample weights from the search space
        weights = {}
        for name, (low, high) in self.weight_ranges.items():
            weights[name] = trial.suggest_float(name, low, high)
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        
        # Create backtester with sampled weights
        bt = EnhancedBacktester(
            self.data,
            asset_type=self.asset_type,
            static_liquidation_data=self.static_liquidation_data,
            regime_filter=self.regime_filter,
            fee_bps=self.fee_bps,
            adaptive_cutoff=self.adaptive_cutoff,
            min_agree_features=self.min_agree_features,
            weight_overrides=weights
        )
        
        # Run backtest
        _ = bt.run_backtest(
            window_size=self.window_size,
            lookahead_candles=self.lookahead_candles,
            min_abs_score=self.min_abs_score
        )
        
        # Get performance metrics
        metrics = bt.performance()
        
        # Primary optimization objective: Sharpe ratio
        sharpe = metrics.get("sharpe", 0.0)
        
        # Secondary objectives as penalties/bonuses
        win_rate = metrics.get("win_rate", 0.0)
        total_trades = metrics.get("total_trades", 0)
        tp1_accuracy = metrics.get("tp1_accuracy", 0.0)
        
        # Penalize if too few trades
        if total_trades < 20:
            sharpe *= 0.8
        
        # Bonus for high win rate
        if win_rate > 0.6:
            sharpe *= 1.1
        
        # Bonus for high TP1 hit rate
        if tp1_accuracy > 0.5:
            sharpe *= 1.05
        
        # Penalize negative Sharpe more heavily
        if sharpe < 0:
            sharpe *= 1.2
        
        # Store additional metrics for pruning and analysis
        trial.set_user_attr("win_rate", float(win_rate))
        trial.set_user_attr("total_trades", int(total_trades))
        trial.set_user_attr("tp1_accuracy", float(tp1_accuracy))
        trial.set_user_attr("total_return", float(metrics.get("total_return", 0.0)))
        
        return float(sharpe)
    
    def optimize(self) -> Dict[str, float]:
        """
        Run the optimization process.
        
        Returns:
        --------
        dict
            Optimized weights
        """
        # Create study with maximization direction
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        
        # Normalize weights to sum to 1.0
        total = sum(best_params.values())
        for k in best_params:
            best_params[k] /= total
        
        # Round to 3 decimal places
        for k in best_params:
            best_params[k] = round(best_params[k], 3)
        
        # Add metadata
        best_trial = study.best_trial
        metadata = {
            "sharpe": float(best_trial.value),
            "win_rate": float(best_trial.user_attrs.get("win_rate", 0.0)),
            "total_trades": int(best_trial.user_attrs.get("total_trades", 0)),
            "tp1_accuracy": float(best_trial.user_attrs.get("tp1_accuracy", 0.0)),
            "total_return": float(best_trial.user_attrs.get("total_return", 0.0))
        }
        
        # Return optimized weights with metadata
        return {
            "weights": best_params,
            "metadata": metadata
        }
    
    def save_weights(self, weights_data: Dict[str, Any]) -> None:
        """
        Save optimized weights to config file.
        
        Parameters:
        -----------
        weights_data : dict
            Optimized weights with metadata
        """
        # Load existing weights if file exists
        if os.path.exists(WEIGHTS_FILE):
            try:
                with open(WEIGHTS_FILE, 'r') as f:
                    all_weights = json.load(f)
            except Exception:
                all_weights = {}
        else:
            all_weights = {}
        
        # Update weights for this asset
        all_weights[self.asset_type] = weights_data
        
        # Save to file
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(all_weights, f, indent=2)
    
    def run_and_save(self) -> Dict[str, Any]:
        """
        Run optimization and save results.
        
        Returns:
        --------
        dict
            Optimized weights with metadata
        """
        weights_data = self.optimize()
        self.save_weights(weights_data)
        return weights_data


def load_optimized_weights(asset_type: str) -> Dict[str, float]:
    """
    Load optimized weights for a specific asset.
    
    Parameters:
    -----------
    asset_type : str
        Asset type ("BTC", "SOL", "BONK")
    
    Returns:
    --------
    dict
        Optimized weights or None if not found
    """
    if not os.path.exists(WEIGHTS_FILE):
        return {}
    
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            all_weights = json.load(f)
        
        asset_weights = all_weights.get(asset_type, {}).get("weights", {})
        return asset_weights
    except Exception:
        return {}


def optimize_all_assets(
    data_dict: Dict[str, pd.DataFrame],
    n_trials: int = 50,
    liquidation_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Optimize weights for all assets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames for each asset
    n_trials : int
        Number of trials per asset
    liquidation_data : dict, optional
        Liquidation heatmap data
    
    Returns:
    --------
    dict
        Optimized weights for all assets
    """
    results = {}
    
    for asset, df in data_dict.items():
        print(f"Optimizing weights for {asset}...")
        
        # Set asset-specific parameters
        if asset == "BTC":
            window_size = 60
            min_abs_score = 0.425
        elif asset == "SOL":
            window_size = 40
            min_abs_score = 0.520
        else:  # BONK
            window_size = 40
            min_abs_score = 0.550
        
        # Create optimizer
        optimizer = WeightOptimizer(
            data_df=df,
            asset_type=asset,
            window_size=window_size,
            lookahead_candles=10,
            min_abs_score=min_abs_score,
            n_trials=n_trials,
            static_liquidation_data=liquidation_data,
            regime_filter=True,
            fee_bps=5.0,
            adaptive_cutoff=True,
            min_agree_features=1,
        )
        
        # Run optimization
        weights_data = optimizer.run_and_save()
        results[asset] = weights_data
    
    return results


if __name__ == "__main__":
    # Example usage
    from historical_data_collector import fetch_historical_data, calculate_indicators
    
    # Fetch data for all assets
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    data = {}
    for asset in ["BTC", "SOL", "BONK"]:
        symbol = f"{asset}/USDT"
        df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            data[asset] = calculate_indicators(df)
    
    # Run optimization for all assets
    results = optimize_all_assets(data, n_trials=30)
    
    # Print results
    for asset, result in results.items():
        print(f"\n{asset} Optimized Weights:")
        for k, v in result["weights"].items():
            print(f"  {k}: {v:.3f}")
        print(f"Sharpe: {result['metadata']['sharpe']:.3f}")
        print(f"Win Rate: {result['metadata']['win_rate']:.2%}")
        print(f"Total Trades: {result['metadata']['total_trades']}")
