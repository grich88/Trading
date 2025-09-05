import pandas as pd
import numpy as np
from rsi_volume_model import RsiVolumePredictor

class Backtester:
    def __init__(self, data_df):
        """
        Initialize backtester with historical data
        
        Parameters:
        -----------
        data_df : pandas.DataFrame
            DataFrame containing OHLCV data with calculated RSI
        """
        self.data = data_df
        self.results = None
        
    def run_backtest(self, window_size=50):
        """
        Run backtest of the RSI + Volume model
        
        Parameters:
        -----------
        window_size : int
            Minimum number of candles required for prediction
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with original data plus prediction scores and signals
        """
        results = self.data.copy()
        results['score'] = np.nan
        
        # We need at least window_size candles to make predictions
        for i in range(window_size, len(results)):
            # Get the data window
            window = results.iloc[i-window_size:i+1]
            
            # Prepare inputs for the model
            rsi_sma = window['rsi_sma'].values
            rsi_raw = window['rsi_raw'].values
            volume = window['volume'].values
            price = window['close'].values
            
            # Create predictor and compute score
            predictor = RsiVolumePredictor(rsi_sma, rsi_raw, volume, price)
            score = predictor.compute_score()
            
            # Store the score
            results.iloc[i, results.columns.get_loc('score')] = score
        
        # Generate trading signals based on score thresholds
        results['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        results.loc[results['score'] > 0.6, 'signal'] = 1
        results.loc[results['score'] < -0.6, 'signal'] = -1
        
        self.results = results
        return results
    
    def calculate_performance(self):
        """
        Calculate performance metrics for the backtest
        
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if self.results is None or 'signal' not in self.results.columns:
            return {"error": "Backtest not run yet"}
        
        results = self.results.copy()
        
        # Calculate returns
        results['returns'] = results['close'].pct_change()
        
        # Calculate strategy returns (assuming we act on the next candle after signal)
        results['strategy_returns'] = results['signal'].shift(1) * results['returns']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['returns']).cumprod() - 1
        results['strategy_cumulative_returns'] = (1 + results['strategy_returns']).cumprod() - 1
        
        # Calculate metrics
        total_trades = (results['signal'] != results['signal'].shift(1)).sum()
        profitable_trades = ((results['signal'] != results['signal'].shift(1)) & 
                             (results['strategy_returns'] > 0)).sum()
        
        if total_trades > 0:
            win_rate = profitable_trades / total_trades
        else:
            win_rate = 0
            
        max_drawdown = (results['strategy_cumulative_returns'].cummax() - 
                        results['strategy_cumulative_returns']).max()
        
        metrics = {
            'total_return': results['strategy_cumulative_returns'].iloc[-1],
            'annualized_return': (1 + results['strategy_cumulative_returns'].iloc[-1]) ** (365 / len(results)) - 1,
            'sharpe_ratio': results['strategy_returns'].mean() / results['strategy_returns'].std() * np.sqrt(365),
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        return metrics
