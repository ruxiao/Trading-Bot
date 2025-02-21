import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from genetic_algorithm import GeneticAlgorithm
from trading_strategy import TradingStrategy
from data_processor import DataProcessor
from performance import PerformanceAnalyzer
import plotly.graph_objects as go
import plotly.express as px
import zipfile
import io
import os

st.set_page_config(page_title="Multi-Asset Trading Optimizer", layout="wide")

# Initialize session state
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'best_performance' not in st.session_state:
    st.session_state.best_performance = None

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
                     mutation_rate: float) -> tuple:
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
        strategy = TradingStrategy(params)
        results = strategy.backtest_portfolio(data_dict, correlation_matrix)
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

# App layout
st.title("Multi-Asset Trading Optimizer")

# Sidebar
st.sidebar.header("Parameters")

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

# Main content
if st.button("Optimize Strategy"):
    try:
        # Fetch and prepare data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        with st.spinner("Fetching data..."):
            data_dict = DataProcessor.fetch_multiple_data(symbols, start_date_str, end_date_str)
            processed_data, correlation_matrix, volatility, returns = DataProcessor.prepare_data(data_dict)

        # Display correlation matrix
        st.subheader("Asset Correlation Analysis")
        fig_corr = plot_correlation_matrix(correlation_matrix)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Display asset volatilities
        st.subheader("Asset Volatilities (Annualized)")
        vol_df = pd.DataFrame({'Symbol': volatility.index, 'Volatility': volatility.values})
        st.dataframe(vol_df)

        # Optimize strategy
        st.write("Optimizing strategy...")
        progress_bar = st.progress(0)

        best_params, best_fitness = optimize_strategy(
            processed_data,
            correlation_matrix,
            population_size,
            generations,
            mutation_rate
        )

        # Store results in session state
        st.session_state.best_params = best_params

        # Test strategy with best parameters
        strategy = TradingStrategy(best_params)
        performance = strategy.backtest_portfolio(processed_data, correlation_matrix)
        st.session_state.best_performance = performance

        # Display results
        st.success("Optimization completed!")

        # Alpha Formula
        st.subheader("Alpha Formula")
        st.markdown("""
        The trading signals are generated based on the following alpha formula:
        ```
        Alpha = w1 * (Short MA - Long MA)/Long MA + w2 * (30 - RSI)/30 + w3 * (RSI - 70)/30 + w4 * correlation_signal

        where:
        - w1 = {:.2f} (MA signal weight)
        - w2 = {:.2f} (RSI oversold weight)
        - w3 = {:.2f} (RSI overbought weight)
        - w4 = {:.2f} (Correlation weight)
        - Short MA window = {}
        - Long MA window = {}
        ```
        """.format(
            best_params[2],
            best_params[3],
            best_params[4],
            best_params[5],
            int(best_params[0]),
            int(best_params[1])
        ))

        # Trade Log
        st.subheader("Trade Log")
        trade_log = pd.DataFrame(performance['trade_log'])
        if not trade_log.empty:
            trade_log['date'] = pd.to_datetime(trade_log['date'])
            trade_log = trade_log.set_index('date')

            # Format the trade log for display
            display_log = trade_log.copy()
            display_log['price'] = display_log['price'].round(2)
            display_log['alpha_value'] = display_log['alpha_value'].round(4)
            display_log['ma_short'] = display_log['ma_short'].round(2)
            display_log['ma_long'] = display_log['ma_long'].round(2)
            display_log['rsi'] = display_log['rsi'].round(1)
            display_log['correlation'] = display_log['correlation'].round(3)

            st.dataframe(display_log)

            # Export option
            csv = display_log.to_csv()
            st.download_button(
                label="Download Trade Log",
                data=csv,
                file_name="trade_log_portfolio.csv",
                mime='text/csv',
            )

        # Performance metrics
        st.subheader("Portfolio Performance Report")
        metrics = PerformanceAnalyzer.calculate_metrics(performance['returns'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics['total_return']:.2%}")
        col2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

        # Performance charts
        st.subheader("Performance Analysis")
        fig = PerformanceAnalyzer.create_performance_charts(
            metrics['returns'],
            metrics['drawdowns']
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display saved results if available
elif st.session_state.best_params is not None:
    st.subheader("Previous Optimization Results")
    st.write("Best Parameters:", st.session_state.best_params)

    if st.session_state.best_performance is not None:
        metrics = PerformanceAnalyzer.calculate_metrics(
            st.session_state.best_performance['returns']
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics['total_return']:.2%}")
        col2.metric("Annual Return", f"{metrics['annual_return']:.2%}")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        col4.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

        fig = PerformanceAnalyzer.create_performance_charts(
            metrics['returns'],
            metrics['drawdowns']
        )
        st.plotly_chart(fig, use_container_width=True)