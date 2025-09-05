import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from scipy.signal import savgol_filter

# Configuration file path
CONFIG_PATH = os.path.join("data", "configs")
WEBTREND_CONFIG_FILE = os.path.join(CONFIG_PATH, "webtrend_config.json")

class WebTrendDerivation:
    """
    Advanced WebTrend indicator derivation from EMAs with dynamic thresholds.
    
    This class implements an improved approach to approximate the WebTrend 3.0
    indicator using combinations of EMAs with dynamic volatility-adjusted bands.
    """
    
    def __init__(self, asset_type: str = "BTC"):
        """
        Initialize the WebTrend derivation.
        
        Parameters:
        -----------
        asset_type : str
            Asset type ("BTC", "SOL", "BONK")
        """
        self.asset_type = asset_type
        
        # Base EMA periods
        self.ema_periods = {
            "fast": 20,
            "mid": 50,
            "slow": 100
        }
        
        # Band multipliers
        self.band_multipliers = {
            "upper": 1.005,  # 0.5% above mid
            "lower": 0.995   # 0.5% below mid
        }
        
        # Volatility adjustment factors
        self.volatility_adjustment = {
            "low": 0.8,    # Tighter bands in low volatility
            "medium": 1.0, # Normal bands in medium volatility
            "high": 1.2    # Wider bands in high volatility
        }
        
        # Volatility thresholds (ATR%)
        self.volatility_thresholds = {
            "low": 0.015,  # 1.5% ATR/price
            "high": 0.04   # 4.0% ATR/price
        }
        
        # Asset-specific adjustments
        if asset_type == "BONK":
            self.band_multipliers = {
                "upper": 1.01,  # 1.0% above mid
                "lower": 0.99   # 1.0% below mid
            }
            self.volatility_thresholds = {
                "low": 0.03,  # 3.0% ATR/price
                "high": 0.08  # 8.0% ATR/price
            }
        elif asset_type == "SOL":
            self.band_multipliers = {
                "upper": 1.008,  # 0.8% above mid
                "lower": 0.992   # 0.8% below mid
            }
            self.volatility_thresholds = {
                "low": 0.02,  # 2.0% ATR/price
                "high": 0.05  # 5.0% ATR/price
            }
        
        # Ensure config directory exists
        os.makedirs(CONFIG_PATH, exist_ok=True)
        
        # Load custom config if available
        self._load_config()
        
        # Output directory for plots
        self.plots_dir = os.path.join("data", "analysis", asset_type)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def _load_config(self) -> None:
        """
        Load custom configuration from file.
        """
        if not os.path.exists(WEBTREND_CONFIG_FILE):
            return
        
        try:
            with open(WEBTREND_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            asset_config = config.get(self.asset_type)
            if asset_config:
                # Load EMA periods
                ema_periods = asset_config.get("ema_periods")
                if ema_periods:
                    self.ema_periods.update(ema_periods)
                
                # Load band multipliers
                band_multipliers = asset_config.get("band_multipliers")
                if band_multipliers:
                    self.band_multipliers.update(band_multipliers)
                
                # Load volatility adjustment
                volatility_adjustment = asset_config.get("volatility_adjustment")
                if volatility_adjustment:
                    self.volatility_adjustment.update(volatility_adjustment)
                
                # Load volatility thresholds
                volatility_thresholds = asset_config.get("volatility_thresholds")
                if volatility_thresholds:
                    self.volatility_thresholds.update(volatility_thresholds)
        except Exception:
            pass
    
    def _save_config(self) -> None:
        """
        Save configuration to file.
        """
        # Load existing config if available
        if os.path.exists(WEBTREND_CONFIG_FILE):
            try:
                with open(WEBTREND_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            except Exception:
                config = {}
        else:
            config = {}
        
        # Update config for this asset
        config[self.asset_type] = {
            "ema_periods": self.ema_periods,
            "band_multipliers": self.band_multipliers,
            "volatility_adjustment": self.volatility_adjustment,
            "volatility_thresholds": self.volatility_thresholds,
            "last_updated": pd.Timestamp.now().isoformat()
        }
        
        # Save to file
        with open(WEBTREND_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Parameters:
        -----------
        prices : np.ndarray
            Price series
        period : int
            EMA period
        
        Returns:
        --------
        np.ndarray
            EMA values
        """
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average True Range.
        
        Parameters:
        -----------
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
        period : int
            ATR period
        
        Returns:
        --------
        np.ndarray
            ATR values
        """
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        
        for i in range(1, len(tr)):
            atr[i] = ((period - 1) * atr[i-1] + tr[i]) / period
        
        return atr
    
    def calculate_atr_percent(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate ATR as percentage of price.
        
        Parameters:
        -----------
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
        period : int
            ATR period
        
        Returns:
        --------
        np.ndarray
            ATR% values
        """
        atr = self.calculate_atr(high, low, close, period)
        return atr / close
    
    def determine_volatility_regime(self, atr_pct: float) -> str:
        """
        Determine volatility regime based on ATR%.
        
        Parameters:
        -----------
        atr_pct : float
            ATR as percentage of price
        
        Returns:
        --------
        str
            Volatility regime ("low", "medium", or "high")
        """
        if atr_pct < self.volatility_thresholds["low"]:
            return "low"
        elif atr_pct > self.volatility_thresholds["high"]:
            return "high"
        else:
            return "medium"
    
    def get_adjusted_band_multipliers(self, atr_pct: float) -> Dict[str, float]:
        """
        Get volatility-adjusted band multipliers.
        
        Parameters:
        -----------
        atr_pct : float
            ATR as percentage of price
        
        Returns:
        --------
        dict
            Adjusted band multipliers
        """
        regime = self.determine_volatility_regime(atr_pct)
        adjustment = self.volatility_adjustment[regime]
        
        # Calculate deviation from base multiplier
        upper_dev = self.band_multipliers["upper"] - 1.0
        lower_dev = 1.0 - self.band_multipliers["lower"]
        
        # Apply adjustment
        adjusted_upper = 1.0 + upper_dev * adjustment
        adjusted_lower = 1.0 - lower_dev * adjustment
        
        return {
            "upper": adjusted_upper,
            "lower": adjusted_lower
        }
    
    def calculate_webtrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WebTrend indicator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with OHLC
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with WebTrend indicators
        """
        result_df = df.copy()
        
        # Extract price data
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        
        # Calculate EMAs
        fast_ema = self.calculate_ema(close, self.ema_periods["fast"])
        mid_ema = self.calculate_ema(close, self.ema_periods["mid"])
        slow_ema = self.calculate_ema(close, self.ema_periods["slow"])
        
        # Calculate ATR%
        atr_pct = self.calculate_atr_percent(high, low, close)
        
        # Initialize WebTrend arrays
        wt_mid = np.zeros_like(close)
        wt_upper = np.zeros_like(close)
        wt_lower = np.zeros_like(close)
        wt_trend = np.zeros_like(close)
        wt_status = np.zeros_like(close, dtype=bool)
        
        # Calculate WebTrend for each point
        for i in range(len(close)):
            # Get current ATR%
            current_atr_pct = atr_pct[i]
            
            # Get adjusted band multipliers
            multipliers = self.get_adjusted_band_multipliers(current_atr_pct)
            
            # Calculate mid line (weighted average of EMAs)
            wt_mid[i] = (0.5 * fast_ema[i] + 0.3 * mid_ema[i] + 0.2 * slow_ema[i])
            
            # Calculate bands
            wt_upper[i] = wt_mid[i] * multipliers["upper"]
            wt_lower[i] = wt_mid[i] * multipliers["lower"]
            
            # Calculate trend value (distance-weighted)
            if i > 0:
                # Base trend on slow EMA
                wt_trend[i] = slow_ema[i]
                
                # Determine trend status (bullish/bearish)
                if close[i] > wt_upper[i]:
                    wt_status[i] = True  # Bullish
                elif close[i] < wt_lower[i]:
                    wt_status[i] = False  # Bearish
                else:
                    # Inside the band, maintain previous status
                    wt_status[i] = wt_status[i-1]
            else:
                wt_trend[i] = slow_ema[i]
                wt_status[i] = close[i] > wt_mid[i]
        
        # Add WebTrend to result DataFrame
        result_df["wt_mid"] = wt_mid
        result_df["wt_upper"] = wt_upper
        result_df["wt_lower"] = wt_lower
        result_df["wt_trend"] = wt_trend
        result_df["webtrend_status"] = wt_status
        
        # Add volatility regime
        regimes = np.array([self.determine_volatility_regime(x) for x in atr_pct])
        result_df["volatility_regime"] = regimes
        
        return result_df
    
    def optimize_parameters(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize WebTrend parameters based on historical data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with OHLC
        
        Returns:
        --------
        dict
            Optimized parameters
        """
        # Extract price data
        close = df["close"].values
        high = df["high"].values if "high" in df.columns else close
        low = df["low"].values if "low" in df.columns else close
        
        # Calculate ATR%
        atr_pct = self.calculate_atr_percent(high, low, close)
        
        # Optimize volatility thresholds based on percentiles
        self.volatility_thresholds = {
            "low": float(np.percentile(atr_pct, 33)),
            "high": float(np.percentile(atr_pct, 67))
        }
        
        # Find optimal EMA periods
        # This is a simplified approach; a more comprehensive optimization
        # would involve testing multiple combinations and evaluating performance
        
        # For now, we'll adjust based on volatility
        mean_atr_pct = float(np.mean(atr_pct))
        
        if mean_atr_pct > self.volatility_thresholds["high"]:
            # Higher volatility: use longer periods
            self.ema_periods = {
                "fast": 25,
                "mid": 60,
                "slow": 120
            }
        elif mean_atr_pct < self.volatility_thresholds["low"]:
            # Lower volatility: use shorter periods
            self.ema_periods = {
                "fast": 15,
                "mid": 40,
                "slow": 80
            }
        
        # Adjust band multipliers based on historical price movements
        typical_move = float(np.median(np.abs(np.diff(close) / close[:-1])))
        
        self.band_multipliers = {
            "upper": 1.0 + typical_move * 2.0,
            "lower": 1.0 - typical_move * 2.0
        }
        
        # Save optimized config
        self._save_config()
        
        return {
            "ema_periods": self.ema_periods,
            "band_multipliers": self.band_multipliers,
            "volatility_thresholds": self.volatility_thresholds,
            "typical_move": typical_move,
            "mean_atr_pct": mean_atr_pct
        }
    
    def plot_webtrend(self, df: pd.DataFrame) -> str:
        """
        Plot WebTrend indicator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with WebTrend indicators
        
        Returns:
        --------
        str
            Path to saved plot
        """
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot price
        ax.plot(df.index, df["close"], label="Price", color="black", linewidth=1.0)
        
        # Plot WebTrend
        ax.plot(df.index, df["wt_mid"], label="WebTrend Mid", color="blue", linewidth=1.5)
        ax.plot(df.index, df["wt_upper"], label="WebTrend Upper", color="green", linewidth=1.0, linestyle="--")
        ax.plot(df.index, df["wt_lower"], label="WebTrend Lower", color="red", linewidth=1.0, linestyle="--")
        ax.plot(df.index, df["wt_trend"], label="WebTrend Trend", color="purple", linewidth=1.0)
        
        # Highlight bullish/bearish regions
        bullish = df["webtrend_status"]
        bearish = ~bullish
        
        if bullish.any():
            bullish_idx = df.index[bullish]
            ax.scatter(bullish_idx, df.loc[bullish_idx, "close"], marker="^", color="green", s=30, label="Bullish")
        
        if bearish.any():
            bearish_idx = df.index[bearish]
            ax.scatter(bearish_idx, df.loc[bearish_idx, "close"], marker="v", color="red", s=30, label="Bearish")
        
        # Set title and labels
        ax.set_title(f"{self.asset_type} WebTrend Indicator")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        
        # Save plot
        filepath = os.path.join(self.plots_dir, f"{self.asset_type}_webtrend.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        return filepath
    
    def get_webtrend_lines(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get WebTrend lines for the latest data point.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Price data with WebTrend indicators
        
        Returns:
        --------
        dict
            WebTrend line values
        """
        if df.empty or not all(col in df.columns for col in ["wt_mid", "wt_upper", "wt_lower", "wt_trend"]):
            return {}
        
        last_row = df.iloc[-1]
        
        return {
            "mid": float(last_row["wt_mid"]),
            "upper": float(last_row["wt_upper"]),
            "lower": float(last_row["wt_lower"]),
            "trend": float(last_row["wt_trend"]),
            "status": bool(last_row["webtrend_status"])
        }


def calculate_webtrend_for_all(df_dict: Dict[str, pd.DataFrame], optimize: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Calculate WebTrend indicators for all assets.
    
    Parameters:
    -----------
    df_dict : dict
        Dictionary of DataFrames for each asset
    optimize : bool
        Whether to optimize parameters
    
    Returns:
    --------
    dict
        Dictionary of WebTrend results for each asset
    """
    results = {}
    
    for asset, df in df_dict.items():
        print(f"Calculating WebTrend for {asset}...")
        
        webtrend = WebTrendDerivation(asset_type=asset)
        
        try:
            # Optimize parameters if requested
            if optimize:
                opt_params = webtrend.optimize_parameters(df)
                print(f"  - Optimized parameters: EMA periods = {opt_params['ema_periods']}")
                print(f"  - Volatility thresholds: Low = {opt_params['volatility_thresholds']['low']:.2%}, High = {opt_params['volatility_thresholds']['high']:.2%}")
            
            # Calculate WebTrend
            wt_df = webtrend.calculate_webtrend(df)
            
            # Plot WebTrend
            plot_path = webtrend.plot_webtrend(wt_df)
            
            # Get latest WebTrend lines
            lines = webtrend.get_webtrend_lines(wt_df)
            
            results[asset] = {
                "dataframe": wt_df,
                "plot": plot_path,
                "lines": lines
            }
            
            print(f"  - Current WebTrend: Status = {'Bullish' if lines.get('status', False) else 'Bearish'}")
            print(f"  - Lines: Mid = {lines.get('mid', 0):.2f}, Upper = {lines.get('upper', 0):.2f}, Lower = {lines.get('lower', 0):.2f}")
        
        except Exception as e:
            print(f"  - Error calculating WebTrend: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from historical_data_collector import fetch_historical_data, calculate_indicators
    
    # Fetch data
    asset = "BTC"
    symbol = f"{asset}/USDT"
    start_date = "2023-01-01"
    end_date = None  # Current date
    
    df = fetch_historical_data(symbol, timeframe="4h", start_date=start_date, end_date=end_date)
    if df is not None and not df.empty:
        df = calculate_indicators(df)
        
        # Create WebTrend derivation
        webtrend = WebTrendDerivation(asset_type=asset)
        
        # Optimize parameters
        opt_params = webtrend.optimize_parameters(df)
        print(f"Optimized parameters for {asset}:")
        print(f"  EMA periods: {opt_params['ema_periods']}")
        print(f"  Band multipliers: Upper = {opt_params['band_multipliers']['upper']:.4f}, Lower = {opt_params['band_multipliers']['lower']:.4f}")
        print(f"  Volatility thresholds: Low = {opt_params['volatility_thresholds']['low']:.2%}, High = {opt_params['volatility_thresholds']['high']:.2%}")
        
        # Calculate WebTrend
        wt_df = webtrend.calculate_webtrend(df)
        
        # Plot WebTrend
        plot_path = webtrend.plot_webtrend(wt_df)
        print(f"WebTrend plot saved to: {plot_path}")
        
        # Get latest WebTrend lines
        lines = webtrend.get_webtrend_lines(wt_df)
        print("\nLatest WebTrend lines:")
        print(f"  Mid: {lines['mid']:.2f}")
        print(f"  Upper: {lines['upper']:.2f}")
        print(f"  Lower: {lines['lower']:.2f}")
        print(f"  Trend: {lines['trend']:.2f}")
        print(f"  Status: {'Bullish' if lines['status'] else 'Bearish'}")
