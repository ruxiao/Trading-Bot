import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from genetic_algorithm import GeneticAlgorithm
from trading_strategy import TradingStrategy
from data_processor import DataProcessor
from performance import PerformanceAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import zipfile
import io
import os
import time
import json

st.set_page_config(page_title="Advanced Trading System", layout="wide")

# Initialize session state
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'best_performance' not in st.session_state:
    st.session_state.best_performance = None
if 'live_trading' not in st.session_state:
    st.session_state.live_trading = False
if 'live_strategy' not in st.session_state:
    st.session_state.live_strategy = None
if 'live_signals' not in st.session_state:
    st.session_state.live_signals = []
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 100000
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def create_source_code_zip():
    """Create a zip file containing all source code files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        source_files = [
            "app.py",
            "data_processor.py",
            "genetic_algorithm.py",
            "performance.py",
            "trading_strategy.py",
            ".streamlit/config.toml"
        ]
        for file_name in source_files:
            if os.path.exists(file_name):
                zf.write(file_name)
    return zip_buffer.getvalue()

def plot_correlation_matrix(correlation_matrix):
    """Create a heatmap of the correlation matrix"""
    fig = px.imshow(
        correlation_matrix,
        labels=dict(color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=600
    )
    return fig

def optimize_strategy(data_dict: dict, 
                     correlation_matrix: pd.DataFrame,
                     population_size: int,
                     generations: int,
                     mutation_rate: float,
                     risk_params: dict) -> tuple:
    """
    Optimize trading strategy using genetic algorithm
    """
    param_ranges = [
        (5, 50),     # Short MA window
        (20, 200),   # Long MA window
        (0, 2),      # MA signal weight
        (0, 2),      # RSI oversold weight
        (0, 2),      # RSI overbought weight
        (-1, 1)      # Correlation weight
    ]

    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate
    )

    population = ga.initialize_population(param_ranges)

    def fitness_function(params):
        strategy = TradingStrategy(
            params,
            transaction_cost=risk_params['transaction_cost'],
            slippage=risk_params['slippage'],
            risk_per_trade=risk_params['risk_per_trade'],
            max_position_size=risk_params['max_position_size'],
            stop_loss_pct=risk_params['stop_loss'],
            take_profit_pct=risk_params['take_profit'],
            volatility_scaling=risk_params['vol_scaling']
        )
        results = strategy.backtest_portfolio(
            data_dict, 
            correlation_matrix,
            initial_capital=risk_params['initial_capital']
        )
        return results['sharpe_ratio']

    best_fitness = float('-inf')
    best_params = None

    for gen in range(generations):
        population, gen_best_fitness = ga.evolve(
            population, 
            fitness_function,
            param_ranges
        )

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_params = population[0]

        progress_bar.progress((gen + 1) / generations)

    return best_params, best_fitness

def fetch_latest_data(symbols, lookback_days=60):
    """Fetch the latest market data for live trading"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    data_dict = DataProcessor.fetch_multiple_data(symbols, start_date_str, end_date_str)
    processed_data, correlation_matrix, volatility, returns = DataProcessor.prepare_data(data_dict)
    
    return processed_data, correlation_matrix, volatility, returns

def simulate_live_trading():
    """Simulates live trading with the current strategy"""
    if st.session_state.live_strategy is None:
        return
    
    # Fetch latest data
    symbols = [pos for pos in st.session_state.positions.keys()]
    
    try:
        current_data, correlation_matrix, _, _ = fetch_latest_data(symbols)
        
        # Process signals
        signals = st.session_state.live_strategy.process_live_data(current_data, correlation_matrix)
        
        # Update session state
        st.session_state.live_signals = signals
        st.session_state.last_update = datetime.now(pytz.timezone('US/Eastern')).strftime("%Y-%m-%d %H:%M:%S")
        
        # In a real system, this would execute trades
        # For simulation, we'll just log the signals
        
        return signals
    except Exception as e:
        st.error(f"Error in live trading: {str(e)}")
        return None

def format_position_sizing_params(risk_params):
    """Format position sizing parameters for display"""
    return f"""
    - Initial Capital: ${risk_params['initial_capital']:,.2f}
    - Risk Per Trade: {risk_params['risk_per_trade'] * 100:.1f}%
    - Max Position Size: {risk_params['max_position_size'] * 100:.1f}%
    - Stop Loss: {risk_params['stop_loss'] * 100:.1f}%
    - Take Profit: {risk_params['take_profit'] * 100:.1f}%
    - Transaction Cost: {risk_params['transaction_cost'] * 10000:.1f} bps
    - Slippage: {risk_params['slippage'] * 10000:.1f} bps
    - Volatility Scaling: {"Enabled" if risk_params['vol_scaling'] else "Disabled"}
    """

# App layout
st.title("Advanced Trading System")

# Tabs for different functions
tab1, tab2, tab3 = st.tabs(["Strategy Optimization", "Live Trading Simulator", "Market Analysis"])

with tab1:
    # Sidebar for Optimization
    st.sidebar.header("Optimization Parameters")
    
    # Add download button in sidebar
    st.sidebar.markdown("### Download Source Code")
    if st.sidebar.download_button(
        label="Download Project Files",
        data=create_source_code_zip(),
        file_name="trading_strategy_project.zip",
        mime="application/zip"
    ):
        st.sidebar.success("Download started!")
    
    # Multiple stock selection
    st.sidebar.subheader("Stock Selection")
    default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    symbols = st.sidebar.text_area(
        "Enter stock symbols (one per line)",
        value="\n".join(default_symbols)
    ).split()
    
    lookback_days = st.sidebar.slider("Lookback Period (days)", 100, 1000, 252)
    population_size = st.sidebar.slider("Population Size", 10, 100, 50)
    generations = st.sidebar.slider("Generations", 10, 100, 30)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
    
    # Risk management parameters
    st.sidebar.subheader("Risk Management")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 
                                            min_value=10000, 
                                            max_value=10000000, 
                                            value=100000,
                                            step=10000)
    
    risk_per_trade = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0) / 100
    max_position = st.sidebar.slider("Max Position Size (%)", 5.0, 50.0, 20.0) / 100
    stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 2.0) / 100
    take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 20.0, 3.0) / 100
    
    # Transaction costs
    transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 0.0, 50.0, 5.0) / 10000
    slippage = st.sidebar.slider("Slippage (bps)", 0.0, 30.0, 3.0) / 10000
    
    # Advanced settings
    vol_scaling = st.sidebar.checkbox("Enable Volatility Scaling", value=True)
    
    # Collect risk parameters
    risk_params = {
        'initial_capital': initial_capital,
        'risk_per_trade': risk_per_trade,
        'max_position_size': max_position,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'transaction_cost': transaction_cost,
        'slippage': slippage,
        'vol_scaling': vol_scaling
    }
    
    # Main optimization content
    if st.button("Optimize Strategy"):
        try:
            # Fetch and prepare data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
    
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
    
            with st.spinner("Fetching market data..."):
                data_dict = DataProcessor.fetch_multiple_data(symbols, start_date_str, end_date_str)
                processed_data, correlation_matrix, volatility, returns = DataProcessor.prepare_data(data_dict)
    
            # Display correlation matrix
            st.subheader("Asset Correlation Analysis")
            fig_corr = plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(fig_corr, use_container_width=True)
    
            # Display asset volatilities
            st.subheader("Asset Volatilities (Annualized)")
            vol_df = pd.DataFrame({
                'Symbol': volatility.index, 
                'Volatility': volatility.values * 100
            })
            vol_df['Volatility'] = vol_df['Volatility'].round(2).astype(str) + '%'
            st.dataframe(vol_df)
    
            # Optimize strategy
            st.write("Optimizing strategy...")
            progress_bar = st.progress(0)
    
            best_params, best_fitness = optimize_strategy(
                processed_data,
                correlation_matrix,
                population_size,
                generations,
                mutation_rate,
                risk_params
            )
    
            # Store results in session state
            st.session_state.best_params = best_params
    
            # Create strategy with best parameters
            strategy = TradingStrategy(
                best_params,
                transaction_cost=risk_params['transaction_cost'],
                slippage=risk_params['slippage'],
                risk_per_trade=risk_params['risk_per_trade'],
                max_position_size=risk_params['max_position_size'],
                stop_loss_pct=risk_params['stop_loss'],
                take_profit_pct=risk_params['take_profit'],
                volatility_scaling=risk_params['vol_scaling']
            )
            
            # Run backtest with realistic conditions
            performance = strategy.backtest_portfolio(
                processed_data, 
                correlation_matrix,
                initial_capital=risk_params['initial_capital']
            )
            
            st.session_state.best_performance = performance
            st.session_state.live_strategy = strategy
    
            # Display results
            st.success("Optimization completed!")
    
            # Strategy & Risk Management
            st.subheader("Strategy Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Alpha Formula Parameters**")
                st.markdown("""
                ```
                Alpha = w1 * (Short MA - Long MA)/Long MA 
                      + w2 * (30 - RSI)/30 
                      + w3 * (RSI - 70)/30 
                      + w4 * correlation_signal
                      + 0.2 * MACD signal
                      + 0.15 * breakout signal
                ```
                
                **Where:**
                - w1 = {:.2f} (MA signal weight)
                - w2 = {:.2f} (RSI oversold weight)
                - w3 = {:.2f} (RSI overbought weight)
                - w4 = {:.2f} (Correlation weight)
                - Short MA window = {}
                - Long MA window = {}
                """.format(
                    best_params[2],
                    best_params[3],
                    best_params[4],
                    best_params[5],
                    int(best_params[0]),
                    int(best_params[1])
                ))
            
            with col2:
                st.write("**Risk Management Parameters**")
                st.markdown(format_position_sizing_params(risk_params))
    
            # Performance metrics
            st.subheader("Portfolio Performance Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Starting Capital", f"${risk_params['initial_capital']:,.2f}")
                st.metric("Final Capital", f"${performance['final_capital']:,.2f}")
                st.metric("Total Return", f"{performance['total_return']:.2%}")
                
            with col2:
                st.metric("Annual Return", f"{performance['annual_return']:.2%}")
                st.metric("Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")
                
            with col3:
                st.metric("Win Rate", f"{performance['win_rate']:.2%}")
                st.metric("Profit Factor", f"{performance['profit_factor']:.2f}")
                st.metric("Avg Exposure", f"{performance['avg_exposure']:.2%}")
    
            # Equity curve
            st.subheader("Equity Curve")
            equity_curve = performance['equity_curve']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_curve.index, 
                y=equity_curve.values,
                mode='lines',
                name='Portfolio Value'
            ))
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
            # Trade Log
            st.subheader("Trade Log")
            trade_log = pd.DataFrame(performance['trade_log'])
            if not trade_log.empty:
                trade_log['date'] = pd.to_datetime(trade_log['date'])
                
                # Format the trade log for display
                display_log = trade_log.copy()
                display_log['price'] = display_log['price'].round(2)
                display_log['size'] = display_log['size'].round(2)
                display_log['cost'] = display_log['cost'].round(2)
                display_log['alpha_value'] = display_log['alpha_value'].round(4)
                
                # Add profit/loss column if we have the necessary data
                if 'stop_price' in display_log.columns:
                    # Filter to only completed trades
                    closed_trades = display_log[display_log['action'].isin(['TAKE_PROFIT', 'STOP_LOSS', 'CLOSE'])]
                    
                    # Display most recent trades first
                    closed_trades = closed_trades.sort_values('date', ascending=False)
                    
                    st.dataframe(closed_trades.head(20))
                else:
                    st.dataframe(display_log.head(20))
    
                # Export option
                csv = display_log.to_csv()
                st.download_button(
                    label="Download Complete Trade Log",
                    data=csv,
                    file_name="trade_log_portfolio.csv",
                    mime='text/csv',
                )
    
            # Performance charts
            st.subheader("Detailed Performance Analysis")
            fig = PerformanceAnalyzer.create_performance_charts(
                performance['returns'],
                performance['drawdowns']
            )
            st.plotly_chart(fig, use_container_width=True)
    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Display saved results if available
    elif st.session_state.best_params is not None:
        st.subheader("Previous Optimization Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alpha Formula Parameters**")
            st.markdown("""
            ```
            Alpha Formula Parameters:
            - Short MA window = {}
            - Long MA window = {}
            - MA signal weight = {:.2f}
            - RSI oversold weight = {:.2f}
            - RSI overbought weight = {:.2f}
            - Correlation weight = {:.2f}
            ```
            """.format(
                int(st.session_state.best_params[0]),
                int(st.session_state.best_params[1]),
                st.session_state.best_params[2],
                st.session_state.best_params[3],
                st.session_state.best_params[4],
                st.session_state.best_params[5]
            ))
        
        if st.session_state.best_performance is not None:
            # Performance metrics
            metrics = PerformanceAnalyzer.calculate_metrics(
                st.session_state.best_performance['returns']
            )
    
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics['total_return']:.2%}")
            col2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
            col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    
            # Equity curve if available
            if 'equity_curve' in st.session_state.best_performance:
                st.subheader("Equity Curve")
                equity_curve = st.session_state.best_performance['equity_curve']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_curve.index, 
                    y=equity_curve.values,
                    mode='lines',
                    name='Portfolio Value'
                ))
                
                fig.update_layout(
                    title='Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Legacy chart
                fig = PerformanceAnalyzer.create_performance_charts(
                    metrics['returns'],
                    metrics['drawdowns']
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Live Trading Simulator")
    
    # Check if strategy is available
    if st.session_state.live_strategy is None:
        st.warning("Please optimize a strategy first in the Strategy Optimization tab")
    else:
        # Portfolio setup
        st.subheader("Current Portfolio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Allow user to set initial capital if not already trading
            if not st.session_state.live_trading:
                initial_capital = st.number_input(
                    "Initial Capital ($)", 
                    min_value=10000, 
                    value=100000, 
                    step=10000
                )
                st.session_state.portfolio_value = initial_capital
                
                # Initialize positions with symbols from optimization
                if len(st.session_state.positions) == 0 and st.session_state.best_performance is not None:
                    trade_log = pd.DataFrame(st.session_state.best_performance['trade_log'])
                    if not trade_log.empty:
                        symbols = trade_log['symbol'].unique()
                        for symbol in symbols:
                            st.session_state.positions[symbol] = {
                                'size': 0,
                                'cost_basis': 0,
                                'market_value': 0
                            }
            else:
                st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
        
        with col2:
            if st.session_state.last_update:
                st.write(f"Last Update: {st.session_state.last_update}")
            
            if st.session_state.live_trading:
                if st.button("Stop Live Trading"):
                    st.session_state.live_trading = False
                    st.experimental_rerun()
            else:
                if st.button("Start Live Trading"):
                    # Initialize live trading
                    st.session_state.live_trading = True
                    st.session_state.live_strategy.start_live_trading(st.session_state.portfolio_value)
                    st.experimental_rerun()
        
        with col3:
            if st.session_state.live_trading:
                if st.button("Update Signals"):
                    simulate_live_trading()
                    st.experimental_rerun()
        
        # Display current positions
        if len(st.session_state.positions) > 0:
            positions_df = []
            for symbol, position in st.session_state.positions.items():
                positions_df.append({
                    'Symbol': symbol,
                    'Size': position.get('size', 0),
                    'Cost Basis': position.get('cost_basis', 0),
                    'Market Value': position.get('market_value', 0),
                    'Profit/Loss': position.get('pnl', 0) if 'pnl' in position else 0
                })
            
            positions_df = pd.DataFrame(positions_df)
            st.dataframe(positions_df)
        
        # Live trading signals
        if st.session_state.live_trading and st.session_state.live_signals:
            st.subheader("Current Trading Signals")
            
            signals_df = []
            for symbol, signal in st.session_state.live_signals.items():
                signals_df.append({
                    'Symbol': symbol,
                    'Action': signal.get('action', ''),
                    'Signal': signal.get('signal', 0),
                    'Alpha': signal.get('alpha', 0),
                    'Price': signal.get('price', 0),
                    'Size': signal.get('size', 0) if 'size' in signal else 0,
                    'Stop Price': signal.get('stop_price', 0) if 'stop_price' in signal else 0,
                    'Take Profit': signal.get('take_profit', 0) if 'take_profit' in signal else 0
                })
            
            signals_df = pd.DataFrame(signals_df)
            st.dataframe(signals_df)
            
            # Visualize signals if any new ones
            buy_signals = signals_df[signals_df['Action'].isin(['BUY', 'ENTER'])]
            sell_signals = signals_df[signals_df['Action'].isin(['SELL', 'CLOSE', 'STOP_LOSS', 'TAKE_PROFIT'])]
            
            if not buy_signals.empty or not sell_signals.empty:
                st.subheader("Signal Visualization")
                
                # Fetch chart data for visualization
                if not buy_signals.empty:
                    symbol = buy_signals.iloc[0]['Symbol']
                elif not sell_signals.empty:
                    symbol = sell_signals.iloc[0]['Symbol']
                else:
                    symbol = None
                
                if symbol:
                    try:
                        data, _, _, _ = fetch_latest_data([symbol], lookback_days=30)
                        prices = data[symbol]['Close']
                        
                        fig = go.Figure()
                        
                        # Price chart
                        fig.add_trace(go.Scatter(
                            x=prices.index,
                            y=prices.values,
                            mode='lines',
                            name='Price'
                        ))
                        
                        # Add buy signals
                        if not buy_signals.empty:
                            buy_signal = buy_signals[buy_signals['Symbol'] == symbol]
                            if not buy_signal.empty:
                                price = buy_signal.iloc[0]['Price']
                                stop = buy_signal.iloc[0]['Stop Price']
                                take_profit = buy_signal.iloc[0]['Take Profit']
                                
                                fig.add_trace(go.Scatter(
                                    x=[prices.index[-1]],
                                    y=[price],
                                    mode='markers',
                                    marker=dict(color='green', size=10),
                                    name='Buy Signal'
                                ))
                                
                                if stop > 0:
                                    fig.add_trace(go.Scatter(
                                        x=[prices.index[-1]],
                                        y=[stop],
                                        mode='markers',
                                        marker=dict(color='red', size=8),
                                        name='Stop Loss'
                                    ))
                                
                                if take_profit > 0:
                                    fig.add_trace(go.Scatter(
                                        x=[prices.index[-1]],
                                        y=[take_profit],
                                        mode='markers',
                                        marker=dict(color='blue', size=8),
                                        name='Take Profit'
                                    ))
                        
                        # Add sell signals
                        if not sell_signals.empty:
                            sell_signal = sell_signals[sell_signals['Symbol'] == symbol]
                            if not sell_signal.empty:
                                price = sell_signal.iloc[0]['Price']
                                
                                fig.add_trace(go.Scatter(
                                    x=[prices.index[-1]],
                                    y=[price],
                                    mode='markers',
                                    marker=dict(color='red', size=10),
                                    name='Sell Signal'
                                ))
                        
                        fig.update_layout(
                            title=f'{symbol} Price Chart with Signals',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error visualizing chart: {str(e)}")
        
        # If not live trading, show instructions
        if not st.session_state.live_trading:
            st.info("""
            **Live Trading Simulator Instructions**
            
            1. Optimize a strategy in the Strategy Optimization tab
            2. Set your initial capital
            3. Click "Start Live Trading" to begin the simulation
            4. Use "Update Signals" to generate new trading signals
            
            The simulator will use the optimized strategy to generate realistic trading signals based on current market data.
            """)
        
        # Strategy parameters display
        if st.session_state.best_params is not None:
            with st.expander("Current Strategy Parameters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Alpha Formula Parameters**")
                    st.markdown("""
                    ```
                    - Short MA window = {}
                    - Long MA window = {}
                    - MA signal weight = {:.2f}
                    - RSI oversold weight = {:.2f}
                    - RSI overbought weight = {:.2f}
                    - Correlation weight = {:.2f}
                    ```
                    """.format(
                        int(st.session_state.best_params[0]),
                        int(st.session_state.best_params[1]),
                        st.session_state.best_params[2],
                        st.session_state.best_params[3],
                        st.session_state.best_params[4],
                        st.session_state.best_params[5]
                    ))
                
                with col2:
                    st.write("**Risk Management Parameters**")
                    
                    # Extract risk params from the live strategy
                    if st.session_state.live_strategy:
                        risk_params = {
                            'initial_capital': st.session_state.portfolio_value,
                            'risk_per_trade': st.session_state.live_strategy.risk_per_trade,
                            'max_position_size': st.session_state.live_strategy.max_position_size,
                            'stop_loss': st.session_state.live_strategy.stop_loss_pct,
                            'take_profit': st.session_state.live_strategy.take_profit_pct,
                            'transaction_cost': st.session_state.live_strategy.transaction_cost,
                            'slippage': st.session_state.live_strategy.slippage,
                            'vol_scaling': st.session_state.live_strategy.volatility_scaling
                        }
                        
                        st.markdown(format_position_sizing_params(risk_params))

with tab3:
    st.header("Market Analysis")
    
    # Stock selection for analysis
    st.sidebar.header("Market Analysis")
    analysis_symbols = st.sidebar.text_input(
        "Enter symbols for analysis (comma separated)",
        value="AAPL,MSFT,GOOGL,AMZN,SPY"
    ).split(',')
    
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"],
        index=2
    )
    
    # Convert period to days
    period_dict = {
        "1 Week": 7,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365
    }
    analysis_days = period_dict[analysis_period]
    
    # Analysis tools
    analysis_options = st.multiselect(
        "Select Analysis Tools",
        ["Price Charts", "Correlation Analysis", "Volatility Analysis", "Return Distribution"],
        default=["Price Charts", "Correlation Analysis"]
    )
    
    if st.button("Run Analysis"):
        try:
            with st.spinner("Fetching market data..."):
                # Fetch data for analysis
                data_dict, correlation_matrix, volatility, returns = fetch_latest_data(
                    analysis_symbols, 
                    lookback_days=analysis_days
                )
            
            if "Price Charts" in analysis_options:
                st.subheader("Price Charts")
                
                # Normalize prices for comparison
                normalized_prices = pd.DataFrame()
                
                for symbol, data in data_dict.items():
                    prices = data['Close']
                    normalized_prices[symbol] = prices / prices.iloc[0]
                
                # Plot normalized prices
                fig = go.Figure()
                
                for symbol in normalized_prices.columns:
                    fig.add_trace(go.Scatter(
                        x=normalized_prices.index,
                        y=normalized_prices[symbol],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title='Normalized Price Comparison',
                    xaxis_title='Date',
                    yaxis_title='Normalized Price (Start=1)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual price charts
                col1, col2 = st.columns(2)
                
                for i, symbol in enumerate(data_dict.keys()):
                    # Alternate between columns
                    with col1 if i % 2 == 0 else col2:
                        prices = data_dict[symbol]['Close']
                        
                        # Calculate moving averages
                        ma20 = prices.rolling(window=20).mean()
                        ma50 = prices.rolling(window=50).mean()
                        
                        # Create chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=prices.index,
                            y=prices.values,
                            mode='lines',
                            name='Price'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=ma20.index,
                            y=ma20.values,
                            mode='lines',
                            name='20-day MA',
                            line=dict(color='orange')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=ma50.index,
                            y=ma50.values,
                            mode='lines',
                            name='50-day MA',
                            line=dict(color='green')
                        ))
                        
                        fig.update_layout(
                            title=f'{symbol} Price Chart',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            if "Correlation Analysis" in analysis_options:
                st.subheader("Correlation Analysis")
                
                # Correlation heatmap
                fig_corr = plot_correlation_matrix(correlation_matrix)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Correlation statistics
                st.write("**Correlation Statistics**")
                
                # Average correlation for each asset
                avg_correlations = correlation_matrix.mean().sort_values()
                
                # Format as dataframe
                corr_df = pd.DataFrame({
                    'Symbol': avg_correlations.index,
                    'Avg Correlation': avg_correlations.values
                })
                
                # Format correlation values
                corr_df['Avg Correlation'] = corr_df['Avg Correlation'].round(3)
                
                # Color-code correlations
                def color_correlation(val):
                    if val < 0.3:
                        return 'background-color: green; color: white'
                    elif val > 0.7:
                        return 'background-color: red; color: white'
                    else:
                        return ''
                
                st.dataframe(corr_df.style.applymap(
                    color_correlation, subset=['Avg Correlation']
                ))
            
            if "Volatility Analysis" in analysis_options:
                st.subheader("Volatility Analysis")
                
                # Format volatility data
                vol_df = pd.DataFrame({
                    'Symbol': volatility.index,
                    'Annualized Volatility': volatility.values * 100
                })
                
                vol_df = vol_df.sort_values('Annualized Volatility')
                
                # Create bar chart of volatilities
                fig = px.bar(
                    vol_df,
                    x='Symbol',
                    y='Annualized Volatility',
                    title='Annualized Volatility (%)',
                    labels={'Annualized Volatility': 'Volatility (%)'},
                    color='Annualized Volatility',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility over time
                st.write("**Volatility Over Time**")
                
                # Calculate rolling volatility
                rolling_vol = pd.DataFrame()
                
                for symbol, data in data_dict.items():
                    rolling_vol[symbol] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
                
                # Plot rolling volatility
                fig = go.Figure()
                
                for symbol in rolling_vol.columns:
                    fig.add_trace(go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol[symbol],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title='20-Day Rolling Volatility',
                    xaxis_title='Date',
                    yaxis_title='Annualized Volatility (%)',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if "Return Distribution" in analysis_options:
                st.subheader("Return Distribution Analysis")
                
                # Calculate daily returns
                daily_returns = pd.DataFrame()
                
                for symbol, data in data_dict.items():
                    daily_returns[symbol] = data['Close'].pct_change().dropna()
                
                # Plot return distribution
                fig = go.Figure()
                
                for symbol in daily_returns.columns:
                    fig.add_trace(go.Histogram(
                        x=daily_returns[symbol] * 100,
                        name=symbol,
                        opacity=0.7,
                        nbinsx=50
                    ))
                
                fig.update_layout(
                    title='Daily Return Distribution',
                    xaxis_title='Daily Return (%)',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Return statistics
                st.write("**Return Statistics**")
                
                stats_df = pd.DataFrame(index=daily_returns.columns)
                stats_df['Mean Return (%)'] = daily_returns.mean() * 100
                stats_df['Std Dev (%)'] = daily_returns.std() * 100
                stats_df['Min (%)'] = daily_returns.min() * 100
                stats_df['Max (%)'] = daily_returns.max() * 100
                stats_df['Skewness'] = daily_returns.skew()
                stats_df['Kurtosis'] = daily_returns.kurtosis()
                
                # Round values
                stats_df = stats_df.round(3)
                
                # Reset index
                stats_df = stats_df.reset_index()
                stats_df = stats_df.rename(columns={'index': 'Symbol'})
                
                st.dataframe(stats_df)
        
        except Exception as e:
            st.error(f"Error in market analysis: {str(e)}")
    else:
        st.info("""
        **Market Analysis Tools**
        
        Select one or more analysis tools and click "Run Analysis" to:
        
        - View price charts with moving averages
        - Analyze correlations between assets
        - Examine volatility patterns
        - Analyze return distributions
        
        This information can help you understand market conditions and refine your trading strategy.
        """)
        
# Auto-update live trading (if active)
if st.session_state.live_trading:
    # Only update if it's been more than 60 seconds since last update
    if (st.session_state.last_update is None or
        (datetime.now() - datetime.strptime(st.session_state.last_update, '%Y-%m-%d %H:%M:%S')).total_seconds() > 60):
        simulate_live_trading()