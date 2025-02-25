import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List
import calendar

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

        # Risk-adjusted metrics
        sortino_ratio = annual_return / (np.std(returns[returns < 0]) * np.sqrt(252)) if len(returns[returns < 0]) > 0 and np.std(returns[returns < 0]) != 0 else 0
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
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

        # Calculate max consecutive wins/losses
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_streak = 0
        
        for ret in returns:
            if ret > 0:
                if current_streak > 0:
                    current_streak += 1
                else:
                    current_streak = 1
                max_win_streak = max(max_win_streak, current_streak)
            elif ret < 0:
                if current_streak < 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
            else:
                current_streak = 0
        
        # Calculate returns by time frame
        if isinstance(returns, pd.Series) and returns.index.dtype.kind == 'M':  # datetime index
            # Monthly returns
            monthly_returns = returns.groupby(pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1)
            # Yearly returns
            yearly_returns = returns.groupby(pd.Grouper(freq='Y')).apply(lambda x: (1 + x).prod() - 1)
            
            # Best/worst periods
            best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
            worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
            best_year = yearly_returns.max() if len(yearly_returns) > 0 else 0
            worst_year = yearly_returns.min() if len(yearly_returns) > 0 else 0
        else:
            best_month = 0
            worst_month = 0
            best_year = 0
            worst_year = 0
            monthly_returns = pd.Series()
            yearly_returns = pd.Series()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_recovery_days': avg_recovery_days,
            'max_win_streak': int(max_win_streak),
            'max_loss_streak': int(max_loss_streak),
            'best_month': best_month,
            'worst_month': worst_month,
            'best_year': best_year,
            'worst_year': worst_year,
            'monthly_returns': monthly_returns,
            'yearly_returns': yearly_returns,
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
                'Return Distribution',
                'Rolling Returns',
                'Rolling Sharpe Ratio',
                'Rolling Volatility'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )

        # Handle input type
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)
        if not isinstance(drawdowns, pd.Series):
            drawdowns = pd.Series(drawdowns)

        # Dates for x-axis if available
        has_dates = returns.index.dtype.kind == 'M'  # Check if datetime index
        x_values = returns.index if has_dates else None

        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(x=x_values, y=cum_returns, name='Cumulative Returns', mode='lines'),
            row=1, col=1
        )

        # Drawdown
        fig.add_trace(
            go.Scatter(x=x_values, y=drawdowns, name='Drawdown', fill='tozeroy', mode='lines',
                      line=dict(color='red')),
            row=1, col=2
        )

        # Return distribution
        fig.add_trace(
            go.Histogram(x=returns, name='Return Distribution',
                        nbinsx=50, histnorm='probability density',
                        marker_color='blue'),
            row=2, col=1
        )
        
        # Add normal distribution curve
        x = np.linspace(returns.min(), returns.max(), 1000)
        mean = returns.mean()
        std = returns.std()
        y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', name='Normal Distribution',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )

        # Daily returns
        fig.add_trace(
            go.Scatter(x=x_values, y=returns, name='Daily Returns', mode='lines',
                      line=dict(color='green')),
            row=2, col=2
        )

        # Rolling metrics (63-day window - approximately 3 months)
        window = min(63, len(returns))
        rolling_returns = returns.rolling(window=window)
        rolling_ann_returns = rolling_returns.mean() * 252
        
        rolling_sharpe = (
            rolling_returns.mean() * 252 / 
            (rolling_returns.std() * np.sqrt(252))
        ).fillna(0)
        
        rolling_vol = (rolling_returns.std() * np.sqrt(252)).fillna(0)

        fig.add_trace(
            go.Scatter(x=x_values, y=rolling_ann_returns, name='Rolling Returns (Ann.)',
                      mode='lines', line=dict(color='purple')),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=x_values, y=rolling_sharpe, name='Rolling Sharpe',
                      mode='lines', line=dict(color='blue')),
            row=3, col=1
        )

        fig.add_trace(
            go.Scatter(x=x_values, y=rolling_vol, name='Rolling Volatility (Ann.)',
                      mode='lines', line=dict(color='orange')),
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
        
    @staticmethod
    def create_monthly_returns_heatmap(returns: pd.Series) -> go.Figure:
        """
        Create a heatmap of monthly returns
        """
        if not isinstance(returns, pd.Series) or returns.index.dtype.kind != 'M':
            # If not a Series with datetime index, return empty figure
            return go.Figure()
            
        # Calculate monthly returns
        monthly_returns = returns.groupby([returns.index.year, returns.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        ).unstack()
        
        # Rename columns to month names
        monthly_returns.columns = [calendar.month_abbr[m] for m in monthly_returns.columns]
        
        # Fill missing values with NaN
        monthly_returns = monthly_returns.fillna(0)
        
        # Convert to percentage
        monthly_returns_pct = monthly_returns * 100
        
        # Create heatmap
        fig = px.imshow(
            monthly_returns_pct,
            labels=dict(x="Month", y="Year", color="Return (%)"),
            x=monthly_returns_pct.columns,
            y=monthly_returns_pct.index,
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        # Add text annotations
        annotations = []
        for i, year in enumerate(monthly_returns_pct.index):
            for j, month in enumerate(monthly_returns_pct.columns):
                value = monthly_returns_pct.iloc[i, j]
                if not np.isnan(value):
                    text_color = "white" if abs(value) > 5 else "black"
                    annotations.append(dict(
                        x=month, 
                        y=year,
                        text=f"{value:.1f}%",
                        showarrow=False,
                        font=dict(color=text_color)
                    ))
        
        fig.update_layout(
            title="Monthly Returns (%)",
            height=400,
            annotations=annotations
        )
        
        return fig
        
    @staticmethod
    def create_trade_analysis(trade_log: List[Dict]) -> Dict:
        """
        Analyze trade performance and create visualizations
        """
        if not trade_log:
            return {"error": "No trade data provided"}
            
        # Convert to DataFrame
        trades_df = pd.DataFrame(trade_log)
        
        # Handle date column
        if 'date' in trades_df.columns:
            trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Calculate trade outcomes if we have sufficient data
        if 'price' in trades_df.columns and 'size' in trades_df.columns and 'action' in trades_df.columns:
            # Group by symbol to track positions
            symbols = trades_df['symbol'].unique()
            trade_outcomes = []
            
            for symbol in symbols:
                symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('date')
                
                # Track position and cost basis
                position = 0
                cost_basis = 0
                entry_date = None
                
                for _, trade in symbol_trades.iterrows():
                    if trade['action'] in ['BUY', 'ENTER'] and position == 0:
                        # Opening a long position
                        position = trade['size']
                        cost_basis = trade['price']
                        entry_date = trade['date']
                    elif trade['action'] in ['SELL'] and position == 0:
                        # Opening a short position
                        position = trade['size']  # This should be negative
                        cost_basis = trade['price']
                        entry_date = trade['date']
                    elif trade['action'] in ['CLOSE', 'STOP_LOSS', 'TAKE_PROFIT'] and position != 0:
                        # Closing a position
                        exit_price = trade['price']
                        exit_date = trade['date']
                        
                        # Calculate P&L
                        if position > 0:  # Long position
                            pnl_pct = (exit_price / cost_basis) - 1
                        else:  # Short position
                            pnl_pct = 1 - (exit_price / cost_basis)
                            
                        duration = (exit_date - entry_date).days
                        
                        # Record trade outcome
                        trade_outcomes.append({
                            'symbol': symbol,
                            'entry_date': entry_date,
                            'exit_date': exit_date,
                            'duration': duration,
                            'direction': 'LONG' if position > 0 else 'SHORT',
                            'entry_price': cost_basis,
                            'exit_price': exit_price,
                            'pnl_pct': pnl_pct,
                            'exit_reason': trade['action']
                        })
                        
                        # Reset position tracking
                        position = 0
                        cost_basis = 0
                        entry_date = None
            
            # Convert to DataFrame
            outcomes_df = pd.DataFrame(trade_outcomes)
            
            if len(outcomes_df) > 0:
                # Calculate trade statistics
                win_rate = len(outcomes_df[outcomes_df['pnl_pct'] > 0]) / len(outcomes_df)
                avg_win = outcomes_df[outcomes_df['pnl_pct'] > 0]['pnl_pct'].mean() if len(outcomes_df[outcomes_df['pnl_pct'] > 0]) > 0 else 0
                avg_loss = outcomes_df[outcomes_df['pnl_pct'] < 0]['pnl_pct'].mean() if len(outcomes_df[outcomes_df['pnl_pct'] < 0]) > 0 else 0
                avg_duration = outcomes_df['duration'].mean()
                
                # Profit factor
                gross_profit = outcomes_df[outcomes_df['pnl_pct'] > 0]['pnl_pct'].sum()
                gross_loss = abs(outcomes_df[outcomes_df['pnl_pct'] < 0]['pnl_pct'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
                
                # Create visualizations
                
                # Trade outcomes by exit reason
                exit_reasons = outcomes_df.groupby('exit_reason')['pnl_pct'].agg(['mean', 'count'])
                
                # Trade P&L distribution
                fig_pnl_dist = px.histogram(
                    outcomes_df, 
                    x='pnl_pct',
                    color='direction',
                    nbins=50,
                    title='Trade P&L Distribution',
                    labels={'pnl_pct': 'P&L (%)', 'count': 'Number of Trades'},
                    color_discrete_map={'LONG': 'blue', 'SHORT': 'red'}
                )
                
                # Cumulative P&L
                outcomes_df = outcomes_df.sort_values('exit_date')
                outcomes_df['cumulative_pnl'] = outcomes_df['pnl_pct'].cumsum()
                
                fig_cum_pnl = px.line(
                    outcomes_df,
                    x='exit_date',
                    y='cumulative_pnl',
                    title='Cumulative P&L (%)',
                    labels={'cumulative_pnl': 'Cumulative P&L (%)', 'exit_date': 'Date'}
                )
                
                # P&L by symbol
                symbol_pnl = outcomes_df.groupby('symbol')['pnl_pct'].mean().sort_values()
                
                fig_symbol_pnl = px.bar(
                    x=symbol_pnl.index,
                    y=symbol_pnl.values,
                    title='Average P&L by Symbol',
                    labels={'x': 'Symbol', 'y': 'Average P&L (%)'}
                )
                
                return {
                    'trades': outcomes_df,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'avg_duration': avg_duration,
                    'exit_reasons': exit_reasons,
                    'fig_pnl_dist': fig_pnl_dist,
                    'fig_cum_pnl': fig_cum_pnl,
                    'fig_symbol_pnl': fig_symbol_pnl
                }
                
        return {"error": "Insufficient trade data"}
        
    @staticmethod
    def create_stress_test_report(returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict:
        """
        Create a stress test report comparing strategy to benchmark during stress periods
        """
        if not isinstance(returns, pd.Series) or returns.index.dtype.kind != 'M':
            return {"error": "Returns must be a Series with datetime index"}
            
        # Define historical stress periods
        stress_periods = {
            'COVID-19 Crash': ('2020-02-19', '2020-03-23'),
            '2018 Q4 Sell-off': ('2018-10-01', '2018-12-24'),
            'China Slowdown 2015-16': ('2015-08-17', '2016-02-11'),
            'Flash Crash 2010': ('2010-05-06', '2010-05-07'),
            'Financial Crisis': ('2008-09-01', '2009-03-09'),
            'Dot-com Crash': ('2000-03-10', '2002-10-09'),
        }
        
        # Calculate returns during stress periods
        stress_results = {}
        
        for period_name, (start_date, end_date) in stress_periods.items():
            try:
                period_returns = returns.loc[start_date:end_date]
                if len(period_returns) > 0:
                    total_return = (1 + period_returns).prod() - 1
                    annualized_vol = period_returns.std() * np.sqrt(252)
                    max_drawdown = (period_returns.cumsum() - period_returns.cumsum().cummax()).min()
                    
                    if benchmark_returns is not None:
                        benchmark_period = benchmark_returns.loc[start_date:end_date]
                        if len(benchmark_period) > 0:
                            benchmark_return = (1 + benchmark_period).prod() - 1
                            outperformance = total_return - benchmark_return
                        else:
                            benchmark_return = None
                            outperformance = None
                    else:
                        benchmark_return = None
                        outperformance = None
                    
                    stress_results[period_name] = {
                        'total_return': total_return,
                        'annualized_vol': annualized_vol,
                        'max_drawdown': max_drawdown,
                        'benchmark_return': benchmark_return,
                        'outperformance': outperformance
                    }
            except:
                # Period may be outside data range
                pass
                
        return stress_results