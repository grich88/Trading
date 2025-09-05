import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import mplfinance as mpf

class SignalVisualizer:
    def __init__(self, data_df):
        """
        Initialize visualizer with data
        
        Parameters:
        -----------
        data_df : pandas.DataFrame
            DataFrame containing OHLCV data with calculated indicators and signals
        """
        self.data = data_df
        
    def plot_signals(self, start_date=None, end_date=None, save_path=None):
        """
        Plot price chart with buy/sell signals
        
        Parameters:
        -----------
        start_date : str, optional
            Start date for plotting (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for plotting (format: 'YYYY-MM-DD')
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        # Filter data by date range if provided
        data = self.data.copy()
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Create OHLCV data for mplfinance
        ohlc_data = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Create buy/sell signals for plotting
        buy_signals = data[data['signal'] == 1].index
        sell_signals = data[data['signal'] == -1].index
        
        # Create custom style
        mc = mpf.make_marketcolors(up='green', down='red', volume='gray')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--')
        
        # Create subplots
        fig, axes = mpf.plot(ohlc_data, type='candle', style=s, volume=True, 
                             returnfig=True, figsize=(12, 10))
        
        # Add buy/sell markers
        ax = axes[0]
        for date in buy_signals:
            ax.plot(mdates.date2num(date), data.loc[date, 'low'] * 0.99, 
                    '^', color='green', markersize=10)
        for date in sell_signals:
            ax.plot(mdates.date2num(date), data.loc[date, 'high'] * 1.01, 
                    'v', color='red', markersize=10)
        
        # Add RSI subplot
        ax2 = axes[2]
        ax2.clear()
        ax2.set_ylabel('RSI')
        ax2.plot(data.index, data['rsi_raw'], color='green', linewidth=1, label='RSI Raw')
        ax2.plot(data.index, data['rsi_sma'], color='orange', linewidth=1.5, label='RSI SMA')
        ax2.axhline(70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(50, color='k', linestyle='--', alpha=0.3)
        ax2.axhline(30, color='g', linestyle='--', alpha=0.5)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='best')
        
        # Add score subplot
        ax3 = axes[3]
        ax3.clear()
        ax3.set_ylabel('Score')
        ax3.plot(data.index, data['score'], color='blue', linewidth=1.5)
        ax3.axhline(0.6, color='g', linestyle='--', alpha=0.5, label='Buy threshold')
        ax3.axhline(-0.6, color='r', linestyle='--', alpha=0.5, label='Sell threshold')
        ax3.axhline(0, color='k', linestyle='--', alpha=0.3, label='Neutral')
        ax3.set_ylim(-1, 1)
        ax3.legend(loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_performance(self, save_path=None):
        """
        Plot performance metrics
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        data = self.data.copy()
        
        if 'strategy_cumulative_returns' not in data.columns:
            # Calculate returns
            data['returns'] = data['close'].pct_change()
            
            # Calculate strategy returns
            data['strategy_returns'] = data['signal'].shift(1) * data['returns']
            
            # Calculate cumulative returns
            data['cumulative_returns'] = (1 + data['returns']).cumprod() - 1
            data['strategy_cumulative_returns'] = (1 + data['strategy_returns']).cumprod() - 1
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot cumulative returns
        axes[0].plot(data.index, data['cumulative_returns'], label='Buy & Hold', color='blue', alpha=0.7)
        axes[0].plot(data.index, data['strategy_cumulative_returns'], label='Strategy', color='green')
        axes[0].set_ylabel('Cumulative Returns')
        axes[0].set_title('Performance Comparison')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Plot drawdowns
        drawdown = data['strategy_cumulative_returns'].cummax() - data['strategy_cumulative_returns']
        axes[1].fill_between(data.index, 0, drawdown, color='red', alpha=0.3)
        axes[1].set_ylabel('Drawdown')
        axes[1].grid(True, alpha=0.3)
        
        # Plot daily returns
        axes[2].bar(data.index, data['strategy_returns'], color='gray', alpha=0.7)
        axes[2].set_ylabel('Daily Returns')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
