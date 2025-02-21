import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    @staticmethod
    def fetch_multiple_data(symbols: list, start_date: str, end_date: str) -> dict:
        """
        Fetch historical data for multiple symbols from Yahoo Finance
        """
        data_dict = {}
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                data_dict[symbol] = data
            return data_dict
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    @staticmethod
    def prepare_data(data_dict: dict) -> tuple:
        """
        Prepare data for strategy testing and calculate correlation matrix
        """
        # Create a combined DataFrame with adjusted close prices
        combined_prices = pd.DataFrame()

        for symbol, data in data_dict.items():
            # Ensure data is sorted by date
            data = data.sort_index()
            # Remove any missing values
            data = data.dropna()
            # Add adjusted close price to combined DataFrame
            combined_prices[symbol] = data['Close']

        # Calculate returns
        returns = combined_prices.pct_change().dropna()

        # Calculate correlation matrix
        correlation_matrix = returns.corr()

        # Calculate volatility for each asset
        volatility = returns.std() * np.sqrt(252)

        # Prepare individual DataFrames with technical indicators
        processed_data = {}
        for symbol, data in data_dict.items():
            processed_data[symbol] = data

        return processed_data, correlation_matrix, volatility, returns

    @staticmethod
    def get_portfolio_metrics(returns: pd.DataFrame, weights: np.ndarray) -> dict:
        """
        Calculate portfolio-level metrics
        """
        # Portfolio returns
        portfolio_returns = returns.dot(weights)

        # Portfolio volatility
        portfolio_vol = np.sqrt(weights.T.dot(returns.cov().dot(weights))) * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = np.mean(portfolio_returns) * 252 / (np.std(portfolio_returns) * np.sqrt(252))

        return {
            'returns': portfolio_returns,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio
        }