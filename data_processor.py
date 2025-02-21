import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataProcessor:
    @staticmethod
    def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    @staticmethod
    def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for strategy testing
        """
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Remove any missing values
        data = data.dropna()
        
        return data
