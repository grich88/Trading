import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Configuration file path
CONFIG_PATH = os.path.join("data", "configs")
FEATURE_IMPORTANCE_FILE = os.path.join(CONFIG_PATH, "feature_importance.json")

class FeatureAnalyzer:
    """
    Analyzes feature importance and predictive power of model components.
    
    This class uses machine learning techniques to identify which components
    of the trading model contribute most to successful predictions.
    """
    
    def __init__(self, asset_type: str = "BTC"):
        """
        Initialize the feature analyzer.
        
        Parameters:
        -----------
        asset_type : str
            Asset to analyze ("BTC", "SOL", "BONK")
        """
        self.asset_type = asset_type
        self.model = None
        self.feature_names = [
            "rsi_score", "volume_score", "divergence", "liquidation_score", 
            "webtrend_score", "ma_trend_score", "pivot_score", "oscillator_score",
            "atr_pct", "rsi_raw", "rsi_sma"
        ]
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
        
        # Output directory for plots
        self.plots_dir = os.path.join("data", "analysis", asset_type)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def prepare_data(self, backtest_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for feature importance analysis.
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Results from backtester with signals and component scores
        
        Returns:
        --------
        tuple
            X (features), y (labels), and feature names
        """
        # Filter for actual signals (non-NEUTRAL)
        signals = backtest_results[backtest_results["signal"].isin(["BUY", "STRONG BUY", "SELL", "STRONG SELL"])].copy()
        
        if len(signals) == 0:
            raise ValueError("No signals found in backtest results")
        
        # Collect feature names and values
        features = []
        feature_names = []
        
        # Check which features are available
        for col in signals.columns:
            if col in self.feature_names or col.endswith("_score"):
                feature_names.append(col)
        
        # Add derived features
        if "score" in signals.columns:
            feature_names.append("abs_score")
        
        # Extract features
        for idx, row in signals.iterrows():
            feature_vector = []
            
            for feat in feature_names:
                if feat == "abs_score":
                    feature_vector.append(abs(row["score"]))
                else:
                    feature_vector.append(row[feat])
            
            features.append(feature_vector)
        
        # Prepare target variable (success/failure)
        targets = []
        for idx, row in signals.iterrows():
            # Determine success (hit TP1 before SL)
            success = False
            if row["signal"] in ("BUY", "STRONG BUY"):
                success = row.get("hit_tp1", False) and not row.get("hit_sl", False)
            else:  # SELL or STRONG SELL
                success = row.get("hit_tp1", False) and not row.get("hit_sl", False)
            
            targets.append(1 if success else 0)
        
        return np.array(features), np.array(targets), feature_names
    
    def analyze_importance(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze feature importance using Random Forest.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        feature_names : list
            Names of features
        
        Returns:
        --------
        dict
            Feature importance scores
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Get feature importance from model
        importance_dict = {}
        for i, name in enumerate(feature_names):
            importance_dict[name] = float(self.model.feature_importances_[i])
        
        # Calculate permutation importance (more robust)
        perm_importance = permutation_importance(
            self.model, X_test, y_test, n_repeats=10, random_state=42
        )
        
        perm_importance_dict = {}
        for i, name in enumerate(feature_names):
            perm_importance_dict[name] = float(perm_importance.importances_mean[i])
        
        # Evaluate model performance
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate PR-AUC (better for imbalanced data)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Return combined results
        return {
            "feature_importance": importance_dict,
            "permutation_importance": perm_importance_dict,
            "model_performance": {
                "auc": float(auc_score),
                "pr_auc": float(pr_auc),
                "n_samples": int(len(X)),
                "success_rate": float(y.mean())
            }
        }
    
    def plot_importance(self, importance_dict: Dict[str, float], title: str) -> str:
        """
        Plot feature importance.
        
        Parameters:
        -----------
        importance_dict : dict
            Feature importance scores
        title : str
            Plot title
        
        Returns:
        --------
        str
            Path to saved plot
        """
        # Sort features by importance
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot bars
        names = [item[0] for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        plt.barh(names, values, color='#3498db')
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        # Save plot
        filename = f"{self.asset_type}_{title.replace(' ', '_').lower()}.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save analysis results to file.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        """
        # Load existing results if file exists
        if os.path.exists(FEATURE_IMPORTANCE_FILE):
            try:
                with open(FEATURE_IMPORTANCE_FILE, 'r') as f:
                    all_results = json.load(f)
            except Exception:
                all_results = {}
        else:
            all_results = {}
        
        # Update results for this asset
        all_results[self.asset_type] = {
            "feature_importance": results["feature_importance"],
            "permutation_importance": results["permutation_importance"],
            "model_performance": results["model_performance"],
            "plots": results.get("plots", {}),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        with open(FEATURE_IMPORTANCE_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def analyze_and_save(self, backtest_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Run analysis and save results.
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Results from backtester
        
        Returns:
        --------
        dict
            Analysis results
        """
        # Prepare data
        X, y, feature_names = self.prepare_data(backtest_results)
        
        # Analyze importance
        results = self.analyze_importance(X, y, feature_names)
        
        # Generate plots
        plots = {}
        plots["feature_importance"] = self.plot_importance(
            results["feature_importance"],
            "Feature Importance"
        )
        plots["permutation_importance"] = self.plot_importance(
            results["permutation_importance"],
            "Permutation Importance"
        )
        
        # Add plots to results
        results["plots"] = plots
        
        # Save results
        self.save_results(results)
        
        return results
    
    def get_recommended_weights(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Get recommended weights based on feature importance.
        
        Parameters:
        -----------
        results : dict
            Analysis results
        
        Returns:
        --------
        dict
            Recommended weights
        """
        # Use permutation importance (more robust)
        importance = results.get("permutation_importance", {})
        
        if not importance:
            return {}
        
        # Filter for core model components
        core_components = [
            "rsi_score", "volume_score", "divergence", "liquidation_score", 
            "webtrend_score", "ma_trend_score", "pivot_score", "oscillator_score"
        ]
        
        filtered_importance = {}
        for name, value in importance.items():
            for component in core_components:
                if component in name:
                    filtered_importance[component] = value
                    break
        
        # Normalize to sum to 1.0
        total = sum(filtered_importance.values())
        if total <= 0:
            return {}
        
        weights = {}
        for name, value in filtered_importance.items():
            weights[name] = round(value / total, 3)
        
        return weights


def analyze_all_assets(backtest_results_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze feature importance for all assets.
    
    Parameters:
    -----------
    backtest_results_dict : dict
        Dictionary of backtest results for each asset
    
    Returns:
    --------
    dict
        Analysis results for all assets
    """
    results = {}
    
    for asset, df in backtest_results_dict.items():
        print(f"Analyzing feature importance for {asset}...")
        
        analyzer = FeatureAnalyzer(asset_type=asset)
        
        try:
            asset_results = analyzer.analyze_and_save(df)
            results[asset] = asset_results
            
            # Print key findings
            print(f"  - Top features by importance:")
            for name, value in sorted(
                asset_results["permutation_importance"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                print(f"    - {name}: {value:.4f}")
            
            print(f"  - Model performance: AUC = {asset_results['model_performance']['auc']:.3f}, PR-AUC = {asset_results['model_performance']['pr_auc']:.3f}")
            
            # Get recommended weights
            weights = analyzer.get_recommended_weights(asset_results)
            if weights:
                print(f"  - Recommended weights:")
                for name, value in weights.items():
                    print(f"    - {name}: {value:.3f}")
        
        except Exception as e:
            print(f"  - Error analyzing feature importance: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from enhanced_backtester import EnhancedBacktester
    from historical_data_collector import fetch_historical_data, calculate_indicators
    
    # Fetch data
    symbol = "BTC/USDT"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
    if df is not None and not df.empty:
        df = calculate_indicators(df)
        
        # Run backtest
        bt = EnhancedBacktester(df, asset_type="BTC")
        results = bt.run_backtest(window_size=60, lookahead_candles=10, min_abs_score=0.425)
        
        # Analyze feature importance
        analyzer = FeatureAnalyzer(asset_type="BTC")
        analysis_results = analyzer.analyze_and_save(results)
        
        # Print top features
        print("\nTop Features by Importance:")
        for name, value in sorted(
            analysis_results["permutation_importance"].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{name}: {value:.4f}")
        
        # Print model performance
        print(f"\nModel Performance:")
        print(f"AUC: {analysis_results['model_performance']['auc']:.3f}")
        print(f"PR-AUC: {analysis_results['model_performance']['pr_auc']:.3f}")
        print(f"Success Rate: {analysis_results['model_performance']['success_rate']:.2%}")
        
        # Get recommended weights
        weights = analyzer.get_recommended_weights(analysis_results)
        print("\nRecommended Weights:")
        for name, value in weights.items():
            print(f"{name}: {value:.3f}")
