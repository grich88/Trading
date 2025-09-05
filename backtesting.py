import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from technical_analysis import TechnicalAnalyzer
from indicator_analysis import IndicatorAnalyzer
from pattern_recognition import PatternRecognizer
from risk_management import RiskManager, RiskMetrics
from signal_generation import SignalGenerator

@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: Optional[datetime]
    asset: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: float
    pnl_percentage: float
    duration: timedelta
    exit_reason: str
    risk_metrics: RiskMetrics
    signal_confidence: float
    indicators_at_entry: Dict[str, float]
    patterns_at_entry: List[str]

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_return: float
    annualized_return: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_trade_duration: timedelta
    best_trade: BacktestTrade
    worst_trade: BacktestTrade
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    monthly_returns: pd.Series
    drawdown_curve: pd.Series

class Backtester:
    def __init__(self,
                 initial_capital: float = 100000,
                 risk_per_trade: float = 0.02,
                 trading_fee: float = 0.001):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.trading_fee = trading_fee
        
        # Initialize analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.indicator_analyzer = IndicatorAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager(
            account_size=initial_capital,
            max_risk_per_trade=risk_per_trade
        )
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _calculate_position_size(self,
                               capital: float,
                               entry_price: float,
                               stop_loss: float,
                               risk_amount: float) -> float:
        """Calculate position size based on risk parameters."""
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            return 0
        
        max_risk_amount = capital * self.risk_per_trade
        risk_amount = min(risk_amount, max_risk_amount)
        
        return risk_amount / risk_per_unit
    
    def _calculate_trade_metrics(self, trades: List[BacktestTrade]) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        if not trades:
            return BacktestResult(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                total_return=0.0,
                annualized_return=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_trade_duration=timedelta(0),
                best_trade=None,
                worst_trade=None,
                trades=[],
                equity_curve=pd.Series(),
                monthly_returns=pd.Series(),
                drawdown_curve=pd.Series()
            )
        
        # Basic metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Build equity curve
        equity_curve = pd.Series(index=[t.entry_time for t in trades])
        equity_curve.iloc[0] = self.initial_capital
        for i, trade in enumerate(trades):
            if i > 0:
                equity_curve.iloc[i] = equity_curve.iloc[i-1] + trade.pnl
        
        # Calculate drawdown
        rolling_max = equity_curve.expanding().max()
        drawdown_curve = equity_curve / rolling_max - 1
        max_drawdown = drawdown_curve.min()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized metrics
        days = (trades[-1].entry_time - trades[0].entry_time).days
        annualized_return = ((1 + total_return) ** (365/days)) - 1 if days > 0 else 0
        
        # Risk metrics
        excess_returns = returns - 0.0  # Assuming 0% risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(returns) > 1 else 0
        
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
        
        # Consecutive wins/losses
        current_streak = 1
        max_wins = 0
        max_losses = 0
        current_is_win = trades[0].pnl > 0
        
        for i in range(1, len(trades)):
            is_win = trades[i].pnl > 0
            if is_win == current_is_win:
                current_streak += 1
            else:
                if current_is_win:
                    max_wins = max(max_wins, current_streak)
                else:
                    max_losses = max(max_losses, current_streak)
                current_streak = 1
                current_is_win = is_win
        
        # Final streak
        if current_is_win:
            max_wins = max(max_wins, current_streak)
        else:
            max_losses = max(max_losses, current_streak)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        # Average duration
        durations = [t.duration for t in trades if t.duration is not None]
        avg_duration = sum(durations, timedelta()) / len(durations) if durations else timedelta(0)
        
        # Monthly returns
        monthly_returns = equity_curve.resample('M').last().pct_change()
        
        return BacktestResult(
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            total_return=total_return,
            annualized_return=annualized_return,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            avg_trade_duration=avg_duration,
            best_trade=max(trades, key=lambda x: x.pnl),
            worst_trade=min(trades, key=lambda x: x.pnl),
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            drawdown_curve=drawdown_curve
        )
    
    def backtest(self,
                data: Dict[str, pd.DataFrame],
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> BacktestResult:
        """
        Run backtest on multiple assets with cross-asset analysis.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary of asset price data frames
        start_date : datetime, optional
            Start date for backtest
        end_date : datetime, optional
            End date for backtest
        """
        trades: List[BacktestTrade] = []
        current_positions: Dict[str, Dict] = {}
        capital = self.initial_capital
        
        # Align all data to same timeframe
        aligned_data = {}
        for asset, df in data.items():
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            aligned_data[asset] = df
        
        # Get common dates
        common_dates = sorted(set.intersection(*[set(df.index) for df in aligned_data.values()]))
        
        for current_time in common_dates:
            # Update current data
            current_data = {
                asset: df[df.index <= current_time]
                for asset, df in aligned_data.items()
            }
            
            # Check for exit signals on open positions
            for asset, position in list(current_positions.items()):
                current_price = aligned_data[asset].loc[current_time, 'close']
                
                # Check stop loss and take profit
                if position['direction'] == 'long':
                    if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                        # Close position
                        pnl = (current_price - position['entry_price']) * position['size']
                        pnl -= (2 * self.trading_fee * position['size'] * current_price)  # Entry and exit fees
                        
                        trades.append(BacktestTrade(
                            entry_time=position['entry_time'],
                            exit_time=current_time,
                            asset=asset,
                            direction=position['direction'],
                            entry_price=position['entry_price'],
                            exit_price=current_price,
                            position_size=position['size'],
                            pnl=pnl,
                            pnl_percentage=pnl / capital,
                            duration=current_time - position['entry_time'],
                            exit_reason='stop_loss' if current_price <= position['stop_loss'] else 'take_profit',
                            risk_metrics=position['risk_metrics'],
                            signal_confidence=position['signal_confidence'],
                            indicators_at_entry=position['indicators'],
                            patterns_at_entry=position['patterns']
                        ))
                        
                        capital += pnl
                        del current_positions[asset]
                
                elif position['direction'] == 'short':
                    if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                        # Close position
                        pnl = (position['entry_price'] - current_price) * position['size']
                        pnl -= (2 * self.trading_fee * position['size'] * current_price)  # Entry and exit fees
                        
                        trades.append(BacktestTrade(
                            entry_time=position['entry_time'],
                            exit_time=current_time,
                            asset=asset,
                            direction=position['direction'],
                            entry_price=position['entry_price'],
                            exit_price=current_price,
                            position_size=position['size'],
                            pnl=pnl,
                            pnl_percentage=pnl / capital,
                            duration=current_time - position['entry_time'],
                            exit_reason='stop_loss' if current_price >= position['stop_loss'] else 'take_profit',
                            risk_metrics=position['risk_metrics'],
                            signal_confidence=position['signal_confidence'],
                            indicators_at_entry=position['indicators'],
                            patterns_at_entry=position['patterns']
                        ))
                        
                        capital += pnl
                        del current_positions[asset]
            
            # Look for new entry signals
            for asset, df in current_data.items():
                if asset in current_positions:
                    continue
                
                # Generate signals
                signal = self.signal_generator.generate_trade_signal(
                    asset=asset,
                    data=df,
                    timeframe='1h',  # Assuming hourly data
                    cross_asset_data=current_data
                )
                
                if signal:
                    # Calculate risk metrics
                    risk_metrics = self.risk_manager.calculate_risk_metrics(df)
                    
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        capital=capital,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        risk_amount=capital * self.risk_per_trade
                    )
                    
                    # Validate trade
                    is_valid, reasons = self.risk_manager.validate_trade(
                        signal=signal,
                        risk_metrics=risk_metrics,
                        position_size=position_size
                    )
                    
                    if is_valid:
                        # Open new position
                        current_positions[asset] = {
                            'direction': 'long' if signal.signal_type == 'buy' else 'short',
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.target_price,
                            'size': position_size,
                            'entry_time': current_time,
                            'risk_metrics': risk_metrics,
                            'signal_confidence': signal.confidence,
                            'indicators': signal.indicator_signals,
                            'patterns': signal.pattern_matches
                        }
        
        # Close any remaining positions at the end
        final_time = common_dates[-1]
        for asset, position in current_positions.items():
            final_price = aligned_data[asset].loc[final_time, 'close']
            
            if position['direction'] == 'long':
                pnl = (final_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - final_price) * position['size']
            
            pnl -= (2 * self.trading_fee * position['size'] * final_price)
            
            trades.append(BacktestTrade(
                entry_time=position['entry_time'],
                exit_time=final_time,
                asset=asset,
                direction=position['direction'],
                entry_price=position['entry_price'],
                exit_price=final_price,
                position_size=position['size'],
                pnl=pnl,
                pnl_percentage=pnl / capital,
                duration=final_time - position['entry_time'],
                exit_reason='end_of_period',
                risk_metrics=position['risk_metrics'],
                signal_confidence=position['signal_confidence'],
                indicators_at_entry=position['indicators'],
                patterns_at_entry=position['patterns']
            ))
            
            capital += pnl
        
        return self._calculate_trade_metrics(trades)
