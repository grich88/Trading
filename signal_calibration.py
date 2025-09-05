import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Configuration file path
CONFIG_PATH = os.path.join("data", "configs")
CALIBRATION_FILE = os.path.join(CONFIG_PATH, "signal_calibration.json")

class SignalCalibrator:
    """
    Calibrates model signals using logistic regression to estimate
    probability of success based on historical performance.
    
    This class builds a logistic model that predicts the probability of a
    successful trade based on component scores and market conditions.
    """
    
    def __init__(self, asset_type: str = "BTC"):
        """
        Initialize the signal calibrator.
        
        Parameters:
        -----------
        asset_type : str
            Asset to calibrate for ("BTC", "SOL", "BONK")
        """
        self.asset_type = asset_type
        self.model = None
        self.feature_names = [
            "rsi_score", "volume_score", "divergence", "liquidation_score", 
            "webtrend_score", "ma_trend", "pivot_score", "oscillator_score",
            "final_score", "atr_pct", "rsi_raw", "rsi_sma"
        ]
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
    
    def prepare_training_data(self, backtest_results: pd.DataFrame, lookahead_candles: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from backtest results.
        
        Parameters:
        -----------
        backtest_results : pd.DataFrame
            Results from backtester with signals and component scores
        lookahead_candles : int
            Number of candles to look ahead for success/failure
        
        Returns:
        --------
        tuple
            X (features) and y (success/failure labels)
        """
        # Filter for actual signals (non-NEUTRAL)
        signals = backtest_results[backtest_results["signal"].isin(["BUY", "STRONG BUY", "SELL", "STRONG SELL"])].copy()
        
        if len(signals) == 0:
            raise ValueError("No signals found in backtest results")
        
        # Prepare features
        X_list = []
        y_list = []
        
        for idx, row in signals.iterrows():
            features = {}
            
            # Extract component scores if available
            for feat in self.feature_names:
                if feat in row:
                    features[feat] = row[feat]
                else:
                    features[feat] = 0.0
            
            # Add derived features
            features["signal_strength"] = abs(row.get("score", 0.0))
            features["is_strong"] = 1.0 if row["signal"] in ("STRONG BUY", "STRONG SELL") else 0.0
            features["is_buy"] = 1.0 if row["signal"] in ("BUY", "STRONG BUY") else 0.0
            
            # Calculate ATR% if not available
            if "atr_pct" not in features:
                try:
                    high = backtest_results["high"].iloc[idx-14:idx]
                    low = backtest_results["low"].iloc[idx-14:idx]
                    close = backtest_results["close"].iloc[idx-14:idx]
                    prev_close = close.shift(1)
                    tr = pd.concat([
                        (high - low),
                        (high - prev_close).abs(),
                        (low - prev_close).abs()
                    ], axis=1).max(axis=1)
                    atr = tr.mean()
                    features["atr_pct"] = atr / close.iloc[-1]
                except Exception:
                    features["atr_pct"] = 0.02  # fallback
            
            # Determine success (hit TP1 before SL)
            success = False
            if row["signal"] in ("BUY", "STRONG BUY"):
                success = row.get("hit_tp1", False) and not row.get("hit_sl", False)
            else:  # SELL or STRONG SELL
                success = row.get("hit_tp1", False) and not row.get("hit_sl", False)
            
            # Add to dataset
            X_list.append(list(features.values()))
            y_list.append(1 if success else 0)
        
        return np.array(X_list), np.array(y_list)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels (1 for success, 0 for failure)
        """
        # Create pipeline with scaling and logistic regression
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.1,  # Regularization strength
                class_weight='balanced',  # Handle class imbalance
                max_iter=1000,
                random_state=42
            ))
        ])
        
        # Train model
        self.model.fit(X, y)
    
    def predict_success_probability(self, features: Dict[str, float]) -> float:
        """
        Predict probability of trade success.
        
        Parameters:
        -----------
        features : dict
            Feature dictionary with component scores
        
        Returns:
        --------
        float
            Probability of success (0.0 to 1.0)
        """
        if self.model is None:
            return 0.5  # Default if model not trained
        
        # Prepare feature vector
        X = []
        for feat in self.feature_names:
            X.append(features.get(feat, 0.0))
        
        # Add derived features
        X.append(abs(features.get("final_score", 0.0)))  # signal_strength
        X.append(1.0 if features.get("signal") in ("STRONG BUY", "STRONG SELL") else 0.0)  # is_strong
        X.append(1.0 if features.get("signal") in ("BUY", "STRONG BUY") else 0.0)  # is_buy
        
        # Predict probability
        try:
            proba = self.model.predict_proba(np.array([X]))[0, 1]
            return float(proba)
        except Exception:
            return 0.5
    
    def calculate_expected_value(self, 
                               probability: float, 
                               tp_pct: float = 0.04, 
                               sl_pct: float = 0.02) -> float:
        """
        Calculate expected value of a trade.
        
        Parameters:
        -----------
        probability : float
            Probability of hitting TP
        tp_pct : float
            Take profit percentage
        sl_pct : float
            Stop loss percentage
        
        Returns:
        --------
        float
            Expected value as percentage
        """
        # Expected value = (win_prob * win_amount) - (lose_prob * lose_amount)
        win_prob = probability
        lose_prob = 1.0 - probability
        
        expected_value = (win_prob * tp_pct) - (lose_prob * sl_pct)
        return expected_value
    
    def save_model(self) -> None:
        """
        Save calibration model to file.
        """
        if self.model is None:
            return
        
        import pickle
        
        # Create model directory
        model_dir = os.path.join(CONFIG_PATH, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, f"{self.asset_type}_calibration.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        metadata = {
            "asset": self.asset_type,
            "features": self.feature_names,
            "model_path": model_path,
            "version": "1.0"
        }
        
        # Update calibration config
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    config = json.load(f)
            except Exception:
                config = {}
        else:
            config = {}
        
        config[self.asset_type] = metadata
        
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self) -> bool:
        """
        Load calibration model from file.
        
        Returns:
        --------
        bool
            True if model loaded successfully, False otherwise
        """
        import pickle
        
        # Check if config exists
        if not os.path.exists(CALIBRATION_FILE):
            return False
        
        try:
            # Load config
            with open(CALIBRATION_FILE, 'r') as f:
                config = json.load(f)
            
            # Get asset config
            asset_config = config.get(self.asset_type)
            if asset_config is None:
                return False
            
            # Load model
            model_path = asset_config.get("model_path")
            if not model_path or not os.path.exists(model_path):
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            return True
        except Exception:
            return False


def train_calibration_models(backtest_results_dict: Dict[str, pd.DataFrame]) -> Dict[str, SignalCalibrator]:
    """
    Train calibration models for all assets.
    
    Parameters:
    -----------
    backtest_results_dict : dict
        Dictionary of backtest results for each asset
    
    Returns:
    --------
    dict
        Dictionary of trained calibrators
    """
    calibrators = {}
    
    for asset, results in backtest_results_dict.items():
        print(f"Training calibration model for {asset}...")
        
        calibrator = SignalCalibrator(asset_type=asset)
        
        try:
            # Prepare training data
            X, y = calibrator.prepare_training_data(results)
            
            # Train model
            calibrator.train(X, y)
            
            # Save model
            calibrator.save_model()
            
            calibrators[asset] = calibrator
            
            print(f"  - Model trained with {len(X)} samples")
        except Exception as e:
            print(f"  - Error training model: {e}")
    
    return calibrators


def get_calibrated_signal(
    calibrator: SignalCalibrator,
    component_scores: Dict[str, float],
    signal: str,
    final_score: float,
    min_ev_threshold: float = 0.005,  # 0.5% minimum expected value
    tp_pct: float = 0.04,
    sl_pct: float = 0.02
) -> Tuple[str, float, float]:
    """
    Get calibrated signal based on expected value.
    
    Parameters:
    -----------
    calibrator : SignalCalibrator
        Trained calibrator
    component_scores : dict
        Component scores
    signal : str
        Original signal
    final_score : float
        Final score
    min_ev_threshold : float
        Minimum expected value to take a trade
    tp_pct : float
        Take profit percentage
    sl_pct : float
        Stop loss percentage
    
    Returns:
    --------
    tuple
        Calibrated signal, success probability, expected value
    """
    # If signal is NEUTRAL, return as is
    if signal == "NEUTRAL":
        return signal, 0.5, 0.0
    
    # Prepare features
    features = component_scores.copy()
    features["final_score"] = final_score
    features["signal"] = signal
    
    # Predict success probability
    probability = calibrator.predict_success_probability(features)
    
    # Calculate expected value
    expected_value = calibrator.calculate_expected_value(probability, tp_pct, sl_pct)
    
    # Apply expected value filter
    if expected_value < min_ev_threshold:
        return "NEUTRAL", probability, expected_value
    
    return signal, probability, expected_value


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
        
        # Train calibrator
        calibrator = SignalCalibrator(asset_type="BTC")
        X, y = calibrator.prepare_training_data(results)
        calibrator.train(X, y)
        
        # Save model
        calibrator.save_model()
        
        # Test on recent signals
        recent = results.iloc[-10:]
        for idx, row in recent.iterrows():
            if row["signal"] != "NEUTRAL":
                features = {
                    "rsi_score": row.get("rsi_score", 0.0),
                    "volume_score": row.get("volume_score", 0.0),
                    "divergence": row.get("divergence", 0.0),
                    "liquidation_score": row.get("liquidation_score", 0.0),
                    "webtrend_score": row.get("webtrend_score", 0.0),
                    "final_score": row.get("score", 0.0),
                    "signal": row["signal"]
                }
                
                prob = calibrator.predict_success_probability(features)
                ev = calibrator.calculate_expected_value(prob)
                
                print(f"Date: {idx}")
                print(f"Signal: {row['signal']}")
                print(f"Score: {row.get('score', 0.0):.3f}")
                print(f"Success Probability: {prob:.2%}")
                print(f"Expected Value: {ev:.2%}")
                print(f"Calibrated Signal: {'NEUTRAL' if ev < 0.005 else row['signal']}")
                print("---")
