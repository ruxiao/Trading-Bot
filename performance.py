import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(returns: np.ndarray) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annual_return = np.mean(returns) * 252
        annual_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        # Calculate drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        # Calculate win rate and other stats
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
        avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
        avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
        profit_factor = -np.sum(positive_returns) / np.sum(negative_returns) if np.sum(negative_returns) != 0 else np.inf

        # Calculate recovery periods
        underwater = drawdowns < 0
        recovery_periods = []
        current_period = 0
        for is_underwater in underwater:
            if is_underwater:
                current_period += 1
            else:
                if current_period > 0:
                    recovery_periods.append(current_period)
                current_period = 0
        avg_recovery_days = np.mean(recovery_periods) if recovery_periods else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_recovery_days': avg_recovery_days,
            'returns': returns,
            'drawdowns': drawdowns
        }

    @staticmethod
    def create_performance_charts(returns: np.ndarray, drawdowns: np.ndarray) -> go.Figure:
        """
        Create comprehensive performance visualization
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cumulative Returns',
                'Drawdown',
                'Monthly Returns Heatmap',
                'Return Distribution',
                'Rolling Sharpe Ratio',
                'Rolling Volatility'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(y=cum_returns, name='Cumulative Returns', mode='lines'),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(y=drawdowns, name='Drawdown', fill='tozeroy', mode='lines',
                      line=dict(color='red')),
            row=1, col=2
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns, name='Return Distribution',
                        nbinsx=50, histnorm='probability'),
            row=2, col=2
        )

        # Rolling metrics (252-day window)
        window = min(252, len(returns))
        rolling_returns = pd.Series(returns).rolling(window=window)
        rolling_sharpe = (
            rolling_returns.mean() * 252 / 
            (rolling_returns.std() * np.sqrt(252))
        )
        rolling_vol = rolling_returns.std() * np.sqrt(252)

        fig.add_trace(
            go.Scatter(y=rolling_sharpe, name='Rolling Sharpe',
                      mode='lines', line=dict(color='green')),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(y=rolling_vol, name='Rolling Volatility',
                      mode='lines', line=dict(color='purple')),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            template='plotly_white',
            title_text='Strategy Performance Analysis'
        )

        return fig