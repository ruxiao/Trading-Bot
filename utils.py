import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def download_data(ticker, start_date, end_date, interval='1m'):
    """
    Downloads historical market data using the yfinance library.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'QQQ').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        interval (str): The data interval (e.g., '1m' for 1-minute data).

    Returns:
        pandas.DataFrame: A DataFrame containing the historical data.
    """
    print(f"Downloading {ticker} data from {start_date} to {end_date} with {interval} interval...")
    # Note: 1-minute data is only available for the last 30 days from Yahoo Finance for free users.
    # For longer history or more reliable intraday data, a dedicated data provider is recommended.
    # If interval is '1d', it fetches daily data.
    # For intraday, yfinance has limitations on how far back it can go (e.g. '1m' is typically last 7 days, '5m' last 60 days)
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError(f"No data downloaded for {ticker} from {start_date} to {end_date} with interval {interval}. "
                         "Check ticker symbol, date range, and interval. "
                         "For 1m data, ensure the period is within the last 7 days. "
                         "For 5m/15m, ensure it's within the last 60 days.")
    print("Data download complete.")
    # Ensure column names are consistent and lowercase for easier access
    data.columns = [col.lower() for col in data.columns]
    return data

def preprocess_data(df):
    """
    Preprocesses the raw data by calculating technical indicators and scaling features.

    Args:
        df (pandas.DataFrame): The raw market data with 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pandas.DataFrame: The preprocessed data with technical indicators and scaled features.
    """
    print("Preprocessing data...")
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Make a copy of the original close prices before calculating indicators that might use it
    # or before it gets scaled. This will be used for trade execution.
    df['close_unscaled'] = df['close']

    # Calculate basic technical indicators using the original 'close' column
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['volatility'] = df['close'].rolling(window=10).std()

    # Drop rows with NaN values created by the rolling windows
    # This must be done BEFORE scaling, and 'close_unscaled' will also be affected (rows dropped).
    df.dropna(inplace=True)
    if df.empty:
        raise ValueError("DataFrame became empty after dropping NaNs from technical indicator calculation. "
                         "Ensure you have enough data for the lookback periods.")


    # Feature Scaling
    # We will scale the features to have zero mean and unit variance.
    # Note: It's crucial to fit the scaler on training data only and transform validation/test data.
    # For simplicity here, we scale the whole dataframe. In a proper setup, this would be handled carefully.
    scaler = StandardScaler()
    # Ensure 'volume' is present, if not, don't scale it.
    feature_columns = ['open', 'high', 'low', 'close', 'sma_10', 'sma_30', 'rsi', 'macd', 'volatility']
    if 'volume' in df.columns:
        feature_columns.append('volume')

    # Ensure all feature columns are present before scaling
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for scaling: {missing_cols}")

    df_scaled = df.copy() # Avoid SettingWithCopyWarning
    df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])

    print("Data preprocessing complete.")
    return df_scaled

def compute_rsi(series, period=14):
    """
    Computes the Relative Strength Index (RSI).

    Args:
        series (pandas.Series): A series of price data (e.g., 'close' prices).
        period (int): The lookback period for calculating RSI.

    Returns:
        pandas.Series: The RSI values.
    """
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Avoid division by zero if loss is 0
    rs = gain / loss
    rs[loss == 0] = np.inf # If loss is zero, RS is infinite (or very high)

    rsi = 100 - (100 / (1 + rs))
    rsi[loss == 0] = 100 # If loss is zero, RSI is 100
    rsi[gain == 0] = 0   # If gain is zero (and loss is not), RSI is 0

    return rsi
