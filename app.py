import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from genetic_algorithm import GeneticAlgorithm
from trading_strategy import TradingStrategy
from data_processor import DataProcessor
from performance import PerformanceAnalyzer
import zipfile
import io
import os

st.set_page_config(page_title="Trading Agent Optimizer", layout="wide")

# Initialize session state
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'best_performance' not in st.session_state:
    st.session_state.best_performance = None

def create_source_code_zip():
    """Create a zip file containing all source code files"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # List of source files to include
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

def optimize_strategy(data: pd.DataFrame, 
                     population_size: int,
                     generations: int,
                     mutation_rate: float) -> tuple:
    """
    Optimize trading strategy using genetic algorithm
    """
    param_ranges = [
        (5, 50),    # Short MA window
        (20, 200),  # Long MA window
        (0, 2),     # MA signal weight
        (0, 2),     # RSI oversold weight
        (0, 2)      # RSI overbought weight
    ]

    ga = GeneticAlgorithm(
        population_size=population_size,
        mutation_rate=mutation_rate
    )

    population = ga.initialize_population(param_ranges)

    def fitness_function(params):
        strategy = TradingStrategy(params)
        results = strategy.backtest(data)
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
st.title("Trading Agent Optimizer")

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

symbol = st.sidebar.text_input("Stock Symbol", value="AAPL")
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
            data = DataProcessor.fetch_data(symbol, start_date_str, end_date_str)
            data = DataProcessor.prepare_data(data)

        # Optimize strategy
        st.write("Optimizing strategy...")
        progress_bar = st.progress(0)

        best_params, best_fitness = optimize_strategy(
            data,
            population_size,
            generations,
            mutation_rate
        )

        # Store results in session state
        st.session_state.best_params = best_params

        # Test strategy with best parameters
        strategy = TradingStrategy(best_params)
        performance = strategy.backtest(data)
        st.session_state.best_performance = performance

        # Display results
        st.success("Optimization completed!")

        # Alpha Formula
        st.subheader("Alpha Formula")
        st.markdown("""
        The trading signals are generated based on the following alpha formula:
        ```
        Alpha = w1 * (Short MA - Long MA)/Long MA + w2 * (30 - RSI)/30 + w3 * (RSI - 70)/30

        where:
        - w1 = {:.2f} (MA signal weight)
        - w2 = {:.2f} (RSI oversold weight)
        - w3 = {:.2f} (RSI overbought weight)
        - Short MA window = {}
        - Long MA window = {}
        ```
        """.format(
            best_params[2],
            best_params[3],
            best_params[4],
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

            st.dataframe(display_log)

            # Export option
            csv = display_log.to_csv()
            st.download_button(
                label="Download Trade Log",
                data=csv,
                file_name=f'trade_log_{symbol}.csv',
                mime='text/csv',
            )

        # Performance metrics
        st.subheader("Strategy Performance Report")
        metrics = PerformanceAnalyzer.calculate_metrics(performance['returns'])

        # Display summary metrics in columns
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