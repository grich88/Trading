import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration file path
CONFIG_PATH = os.path.join("data", "configs")
TIMEFRAME_CONFIG_FILE = os.path.join(CONFIG_PATH, "adaptive_timeframe.json")

class AdaptiveTimeframeSelector:
    """
    Dynamically selects optimal timeframes based on market volatility.
    
    This class analyzes market volatility patterns to determine the most 
    appropriate timeframe for analysis, switching between shorter timeframes
    in low-volatility periods and longer timeframes during high volatility.
    """
    
    def __init__(self, asset_type: str = "BTC"):
        """
        Initialize the timeframe selector.
        
        Parameters:
        -----------
        asset_type : str
            Asset to analyze ("BTC", "SOL", "BONK")
        """
        self.asset_type = asset_type
        self.timeframes = ["1h", "4h", "1d"]
        self.default_timeframe = "4h"
        
        # Volatility thresholds (ATR% values)
        self.low_volatility_threshold = 0.015  # 1.5%
        self.high_volatility_threshold = 0.04  # 4.0%
        
        # Timeframe mapping based on volatility
        self.volatility_mapping = {
            "low": "1h",      # Use shorter timeframe in low volatility
            "medium": "4h",   # Use default timeframe in medium volatility
            "high": "1d"      # Use longer timeframe in high volatility
        }
        
        # Asset-specific adjustments
        if asset_type == "BONK":
            self.low_volatility_threshold = 0.03   # 3.0%
            self.high_volatility_threshold = 0.08  # 8.0%
        elif asset_type == "SOL":
            self.low_volatility_threshold = 0.02   # 2.0%
            self.high_volatility_threshold = 0.05  # 5.0%
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
        
        # Load custom config if available
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load custom configuration from file.
        """
        if not os.path.exists(TIMEFRAME_CONFIG_FILE):
            return
        
        try:
            with open(TIMEFRAME_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            asset_config = config.get(self.asset_type)
            if asset_config:
                self.low_volatility_threshold = asset_config.get("low_threshold", self.low_volatility_threshold)
                self.high_volatility_threshold = asset_config.get("high_threshold", self.high_volatility_threshold)
                
                mapping = asset_config.get("timeframe_mapping")
                if mapping:
                    self.volatility_mapping.update(mapping)
        except Exception:
            pass
    
    def _save_config(self) -> None:
        """
        Save configuration to file.
        """
        # Load existing config if available
        if os.path.exists(TIMEFRAME_CONFIG_FILE):
            try:
                with open(TIMEFRAME_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except Exception:
                config = {}
        else:
            config = {}
        
        # Update config for this asset
        config[self.asset_type] = {
            "low_threshold": self.low_volatility_threshold,
            "high_threshold": self.high_volatility_threshold,
            "timeframe_mapping": self.volatility_mapping,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save to file
        with open(TIMEFRAME_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def calculate_atr_percent(self, df: pd.DataFrame, periods: int = 14) -> float:
        """
        Calculate ATR as percentage of price.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with high, low, close columns
        periods : int
            ATR period
        
        Returns:
        --------
        float
            ATR as percentage of price
        """
        if not {"high", "low", "close"}.issubset(df.columns):
            return 0.02  # Default if columns missing
        
        try:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)
            
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(min(periods, len(tr))).mean().iloc[-1]
            price = float(close.iloc[-1])
            
            return float(atr / price) if price else 0.02
        except Exception:
            return 0.02  # Default on error
    
    def calculate_volatility_regime(self, df: pd.DataFrame, window: int = 14) -> str:
        """
        Determine volatility regime.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        window : int
            Lookback window for volatility calculation
        
        Returns:
        --------
        str
            Volatility regime ("low", "medium", or "high")
        """
        # Calculate recent ATR%
        recent_df = df.iloc[-window:]
        atr_pct = self.calculate_atr_percent(recent_df, window)
        
        # Determine regime
        if atr_pct < self.low_volatility_threshold:
            return "low"
        elif atr_pct > self.high_volatility_threshold:
            return "high"
        else:
            return "medium"
    
    def get_optimal_timeframe(self, df: pd.DataFrame) -> str:
        """
        Get optimal timeframe based on current volatility.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        
        Returns:
        --------
        str
            Optimal timeframe
        """
        regime = self.calculate_volatility_regime(df)
        return self.volatility_mapping.get(regime, self.default_timeframe)
    
    def analyze_volatility_patterns(self, df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze volatility patterns across different timeframes.
        
        Parameters:
        -----------
        df_dict : dict
            Dictionary of DataFrames for different timeframes
        
        Returns:
        --------
        dict
            Volatility analysis results
        """
        results = {}
        
        for tf, df in df_dict.items():
            if df is None or df.empty:
                continue
            
            # Calculate ATR% over time
            window = 14
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            close = df["close"].astype(float)
            prev_close = close.shift(1)
            
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            
            atr = tr.rolling(window).mean()
            atr_pct = atr / close
            
            # Calculate volatility statistics
            results[tf] = {
                "mean_atr_pct": float(atr_pct.mean()),
                "median_atr_pct": float(atr_pct.median()),
                "max_atr_pct": float(atr_pct.max()),
                "min_atr_pct": float(atr_pct.min()),
                "current_atr_pct": float(atr_pct.iloc[-1]),
                "low_threshold": self.low_volatility_threshold,
                "high_threshold": self.high_volatility_threshold
            }
            
            # Calculate percentage of time in each regime
            low_pct = (atr_pct < self.low_volatility_threshold).mean()
            high_pct = (atr_pct > self.high_volatility_threshold).mean()
            medium_pct = 1.0 - low_pct - high_pct
            
            results[tf]["regime_distribution"] = {
                "low": float(low_pct),
                "medium": float(medium_pct),
                "high": float(high_pct)
            }
        
        return results
    
    def optimize_thresholds(self, df_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Optimize volatility thresholds based on historical data.
        
        Parameters:
        -----------
        df_dict : dict
            Dictionary of DataFrames for different timeframes
        """
        # Use 4h timeframe as reference
        df = df_dict.get("4h")
        if df is None or df.empty:
            return
        
        # Calculate ATR% series
        window = 14
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window).mean()
        atr_pct = atr / close
        
        # Find percentiles for thresholds
        # Low threshold: 33rd percentile
        # High threshold: 67th percentile
        low_threshold = float(atr_pct.quantile(0.33))
        high_threshold = float(atr_pct.quantile(0.67))
        
        # Update thresholds
        self.low_volatility_threshold = low_threshold
        self.high_volatility_threshold = high_threshold
        
        # Save updated config
        self._save_config()
    
    def plot_volatility_regimes(self, df: pd.DataFrame, window: int = 14) -> str:
        """
        Plot volatility regimes over time.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data
        window : int
            ATR window
        
        Returns:
        --------
        str
            Path to saved plot
        """
        # Calculate ATR%
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window).mean()
        atr_pct = atr / close
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price plot
        ax1.plot(df.index, df["close"], color="black", linewidth=1.0)
        ax1.set_title(f"{self.asset_type} Price")
        ax1.grid(True, alpha=0.3)
        
        # ATR% plot
        ax2.plot(df.index, atr_pct, color="blue", linewidth=1.0)
        ax2.axhline(self.low_volatility_threshold, color="green", linestyle="--", label=f"Low ({self.low_volatility_threshold:.2%})")
        ax2.axhline(self.high_volatility_threshold, color="red", linestyle="--", label=f"High ({self.high_volatility_threshold:.2%})")
        ax2.set_title("ATR% (Volatility)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Color volatility regimes
        ax2.fill_between(df.index, 0, self.low_volatility_threshold, 
                         where=atr_pct < self.low_volatility_threshold, 
                         color="green", alpha=0.2)
        ax2.fill_between(df.index, self.low_volatility_threshold, self.high_volatility_threshold, 
                         where=(atr_pct >= self.low_volatility_threshold) & (atr_pct <= self.high_volatility_threshold), 
                         color="yellow", alpha=0.2)
        ax2.fill_between(df.index, self.high_volatility_threshold, atr_pct.max()*1.1, 
                         where=atr_pct > self.high_volatility_threshold, 
                         color="red", alpha=0.2)
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join("data", "analysis", self.asset_type)
        os.makedirs(plots_dir, exist_ok=True)
        
        filepath = os.path.join(plots_dir, f"{self.asset_type}_volatility_regimes.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        return filepath


def analyze_all_assets(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze volatility patterns for all assets.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of DataFrames for each asset and timeframe
        Format: {asset: {timeframe: df}}
    
    Returns:
    --------
    dict
        Analysis results for all assets
    """
    results = {}
    
    for asset, timeframes in data_dict.items():
        print(f"Analyzing volatility patterns for {asset}...")
        
        selector = AdaptiveTimeframeSelector(asset_type=asset)
        
        try:
            # Optimize thresholds
            selector.optimize_thresholds(timeframes)
            
            # Analyze volatility patterns
            analysis = selector.analyze_volatility_patterns(timeframes)
            
            # Plot volatility regimes
            if "4h" in timeframes and timeframes["4h"] is not None and not timeframes["4h"].empty:
                plot_path = selector.plot_volatility_regimes(timeframes["4h"])
                analysis["plot"] = plot_path
            
            results[asset] = analysis
            
            # Print key findings
            print(f"  - Optimized thresholds: Low={selector.low_volatility_threshold:.2%}, High={selector.high_volatility_threshold:.2%}")
            
            if "4h" in analysis:
                regime_dist = analysis["4h"]["regime_distribution"]
                print(f"  - Regime distribution (4h): Low={regime_dist['low']:.1%}, Medium={regime_dist['medium']:.1%}, High={regime_dist['high']:.1%}")
                print(f"  - Current ATR%: {analysis['4h']['current_atr_pct']:.2%}")
                print(f"  - Recommended timeframe: {selector.get_optimal_timeframe(timeframes['4h'])}")
        
        except Exception as e:
            print(f"  - Error analyzing volatility patterns: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from historical_data_collector import fetch_historical_data, calculate_indicators
    
    # Fetch data for multiple timeframes
    asset = "BTC"
    symbol = f"{asset}/USDT"
    start_date = "2023-01-01"
    end_date = None  # Current date
    
    data = {}
    for tf in ["1h", "4h", "1d"]:
        df = fetch_historical_data(symbol, timeframe=tf, start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            data[tf] = calculate_indicators(df)
    
    # Create selector
    selector = AdaptiveTimeframeSelector(asset_type=asset)
    
    # Optimize thresholds
    selector.optimize_thresholds(data)
    
    # Get optimal timeframe
    if "4h" in data:
        optimal_tf = selector.get_optimal_timeframe(data["4h"])
        print(f"Optimal timeframe for {asset}: {optimal_tf}")
        
        # Plot volatility regimes
        plot_path = selector.plot_volatility_regimes(data["4h"])
        print(f"Volatility regime plot saved to: {plot_path}")
        
        # Analyze volatility patterns
        analysis = selector.analyze_volatility_patterns(data)
        
        # Print results
        for tf, results in analysis.items():
            print(f"\n{tf} Timeframe:")
            print(f"  Mean ATR%: {results['mean_atr_pct']:.2%}")
            print(f"  Current ATR%: {results['current_atr_pct']:.2%}")
            
            regime_dist = results["regime_distribution"]
            print(f"  Regime distribution: Low={regime_dist['low']:.1%}, Medium={regime_dist['medium']:.1%}, High={regime_dist['high']:.1%}")
