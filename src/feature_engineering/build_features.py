import logging
import os
import pandas as pd
import pandas_ta as ta
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Assuming the script is run from the repository root or paths are adjusted accordingly
DATA_DIR = "data"
RAW_DATA_FILE = os.path.join(DATA_DIR, "qqq_1min_1month.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "qqq_features.parquet") # Using Parquet for efficiency

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a broad set of technical indicators on the input OHLCV DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
                           Must have a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with original data and added features.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    logger.info(f"Original data shape: {df.shape}")

    # Use pandas_ta to create a custom strategy for multiple indicators
    # This is a flexible way to add many indicators at once.
    # Refer to pandas-ta documentation for available indicators and their parameters.
    custom_strategy = ta.Strategy(
        name="Comprehensive Indicators",
        description="A collection of common technical indicators for RL state representation.",
        ta=[
            # Trend Indicators
            {"kind": "sma", "length": 10, "col_names": "SMA_10"},
            {"kind": "sma", "length": 20, "col_names": "SMA_20"},
            {"kind": "sma", "length": 50, "col_names": "SMA_50"},
            {"kind": "ema", "length": 10, "col_names": "EMA_10"},
            {"kind": "ema", "length": 20, "col_names": "EMA_20"},
            {"kind": "ema", "length": 50, "col_names": "EMA_50"},
            {"kind": "macd", "fast": 12, "slow": 26, "signal": 9, "col_names": ("MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9")},
            {"kind": "adx", "length": 14, "col_names": ("ADX_14", "DMP_14", "DMN_14")},
            {"kind": "aroon", "length": 14, "col_names": ("AROOND_14", "AROONU_14", "AROONOSC_14")},
            {"kind": "psar", "col_names": ("PSARl", "PSARs", "PSARaf", "PSARr")}, # Parabolic SAR

            # Momentum Indicators
            {"kind": "rsi", "length": 14, "col_names": "RSI_14"},
            {"kind": "rsi", "length": 7, "col_names": "RSI_7"},
            {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3, "col_names": ("STOCHk_14_3_3", "STOCHd_14_3_3")},
            {"kind": "stochrsi", "length": 14, "rsi_length": 14, "k": 3, "d": 3, "col_names": ("STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3")},
            {"kind": "willr", "length": 14, "col_names": "WILLR_14"}, # Williams %R
            {"kind": "cci", "length": 20, "col_names": "CCI_20_0.015"},

            # Volatility Indicators
            {"kind": "bbands", "length": 20, "std": 2, "col_names": ("BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "BBB_20_2.0", "BBP_20_2.0")},
            {"kind": "atr", "length": 14, "col_names": "ATR_14"},
            {"kind": "natr", "length": 14, "col_names": "NATR_14"}, # Normalized ATR
            {"kind": "donchian", "lower_length": 20, "upper_length": 20, "col_names": ("DONCHIANl_20_20", "DONCHIANm_20_20", "DONCHIANu_20_20")},

            # Volume Indicators (ensure 'volume' column exists and is named correctly)
            {"kind": "obv", "col_names": "OBV"},
            {"kind": "mfi", "length": 14, "col_names": "MFI_14"}, # Money Flow Index
            {"kind": "cmf", "length": 20, "col_names": "CMF_20"}, # Chaikin Money Flow
            # {"kind": "efi", "length": 13, "col_names": "EFI_13"}, # Elder's Force Index - might need adjustment if issues
            {"kind": "vwap", "col_names": "VWAP"}, # Needs typical price (HLC/3) - pandas-ta handles this if columns are standard

            # Other / Pattern-based (less common for direct RL state but can be informative)
            # {"kind": "cdl_doji", "col_names": "CDL_DOJI"}, # Example Candlestick pattern
            # {"kind": "ha", "col_names": ("HA_open", "HA_high", "HA_low", "HA_close")}, # Heikin Ashi
        ]
    )

    # Apply the strategy. Ensure the DataFrame has 'open', 'high', 'low', 'close', 'volume' (lowercase)
    # The qqq_1min_1month.csv currently saves with title case column names.
    df.columns = [col.lower() for col in df.columns] # Convert to lowercase

    # Ensure standard column names pandas_ta expects if they differ
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        logger.error(f"DataFrame is missing one or more required columns. Found: {df.columns.tolist()}. Expected: {list(required_cols)}")
        # Attempt to rename if common alternatives exist (e.g. 'Date' to 'date' was handled by index)
        # This part might need more robust handling based on actual CSV column names from IBKR
        # For now, we assume the fetch_qqq_data script provides 'Open', 'High', 'Low', 'Close', 'Volume'
        # which are then lowercased here.
        raise ValueError("Missing required OHLCV columns for feature calculation.")

    df.ta.strategy(custom_strategy)
    logger.info(f"Shape after adding TA features: {df.shape}")
    logger.info(f"Columns after adding TA features: {df.columns.tolist()}")

    # --- Post-processing Features ---
    # 1. Handle NaNs:
    #    - Indicators have a warm-up period, creating NaNs at the beginning.
    #    - Some indicators might produce NaNs if inputs are all zero, etc.
    initial_nans = df.isna().sum().sum()
    logger.info(f"Total NaN values before handling: {initial_nans}")

    # Option 1: Forward-fill NaNs. This is common but can propagate stale data.
    # df.ffill(inplace=True)
    # Option 2: Backward-fill NaNs. Can use future data, less common for trading systems.
    # df.bfill(inplace=True)
    # Option 3: Fill with a specific value (e.g., 0 or mean). Mean can be problematic with non-stationary data.
    # df.fillna(0, inplace=True)
    # Option 4: Drop rows with any NaNs. This is the cleanest for model training if enough data remains.
    #           Given we have 1-min data for a month, dropping initial rows is usually fine.

    # Let's find the first valid index for most indicators (after warm-up)
    # Most indicators with length N will have N-1 NaNs. Longest length here is SMA_50 or ADX/AROON (14*2 for internal calcs).
    # psar also has warmup. CMF_20, etc.
    # Safest to drop any row that still has a NaN after all calculations.
    df.fillna(0, inplace=True)
    logger.info(f"Shape after filling NaNs with 0: {df.shape}")
    final_nans = df.isna().sum().sum()
    if final_nans > 0: # Should be 0
        logger.warning(f"Still {final_nans} NaN values after filling with 0. Check indicator calculations.")


    # 2. Feature Scaling (Optional here, often done just before model training)
    #    If scaling here, save scalers. For now, let's assume scaling will be part of the VAE/Transformer preprocessing.
    #    Example:
    #    from sklearn.preprocessing import StandardScaler
    #    scaler = StandardScaler()
    #    # Select only feature columns, not OHLCV if you want to keep them raw or scale differently
    #    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    #    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 3. Remove highly correlated features (Optional, can sometimes help models)
    #    Example:
    #    corr_matrix = df[feature_cols].corr().abs()
    #    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    #    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    #    df.drop(columns=to_drop, inplace=True)
    #    logger.info(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")

    # 4. Add time-based features (if not inherently captured by sequences in Transformer/VAE)
    #    df['minute_of_hour'] = df.index.minute
    #    df['hour_of_day'] = df.index.hour
    #    df['day_of_week'] = df.index.dayofweek

    # For now, the raw indicators + OHLCV will be the feature set.
    # The VAE/Transformer will learn to extract useful patterns from this high-dimensional space.

    logger.info(f"Final feature set shape: {df.shape}")
    logger.info(f"Final columns: {df.columns.tolist()}")
    return df

def main():
    logger.info("Starting feature engineering process...")

    if not os.path.exists(RAW_DATA_FILE):
        logger.error(f"Raw data file not found: {RAW_DATA_FILE}. Please run the data collection script first.")
        return

    # Load raw data
    # The CSV from IBKR has 'date' as the first column, which pandas can parse as index.
    # It also has columns like 'Open', 'High', 'Low', 'Close', 'Volume', 'Average', 'BarCount'
    # We need to make sure 'date' is parsed correctly as DatetimeIndex.
    try:
        raw_df = pd.read_csv(RAW_DATA_FILE)
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'])
            raw_df.set_index('date', inplace=True)
        elif raw_df.index.name == 'date' and pd.api.types.is_datetime64_any_dtype(raw_df.index):
            # Already has datetime index, common if read_csv infers it
            pass
        elif raw_df.columns[0].lower() == 'date' and 'Unnamed: 0' not in raw_df.columns[0]: # Check if first column is date-like
             raw_df.rename(columns={raw_df.columns[0]: 'date'}, inplace=True)
             raw_df['date'] = pd.to_datetime(raw_df['date'])
             raw_df.set_index('date', inplace=True)
        else:
            logger.error("Could not find a 'date' column or a DatetimeIndex in the raw CSV. Please check the file format.")
            return

    except Exception as e:
        logger.error(f"Error loading or processing raw data from {RAW_DATA_FILE}: {e}", exc_info=True)
        return

    logger.info(f"Loaded raw data. Shape: {raw_df.shape}, Index type: {type(raw_df.index)}")

    # Keep only OHLCV for feature calculation, pandas-ta typically uses these.
    # The CSV from fetch_qqq_data.py should have: date,open,high,low,close,volume,average,barCount
    # Let's ensure we use the correct column names. 'average' is WAP, 'barCount' is trades.
    # pandas-ta uses 'open', 'high', 'low', 'close', 'volume'.
    # The fetch_qqq_data script saves them as 'open', 'high', 'low', 'close', 'volume'.
    # The to_lower() in calculate_features should handle it if they are capitalized.

    # Select necessary columns (assuming they are already named correctly or will be lowercased)
    ohlcv_df = raw_df[['open', 'high', 'low', 'close', 'volume']].copy()
    # Some indicators might use WAP (average in our CSV) if 'close' is not representative.
    # For now, standard OHLCV.

    # Calculate features
    features_df = calculate_features(ohlcv_df)

    if features_df is not None and not features_df.empty:
        # Save features
        try:
            features_df.to_parquet(FEATURES_FILE, index=True)
            logger.info(f"Features saved to {FEATURES_FILE}")
            logger.info(f"Feature DataFrame head:\n{features_df.head()}")
        except Exception as e:
            logger.error(f"Error saving features to {FEATURES_FILE}: {e}", exc_info=True)
    else:
        logger.error("Feature calculation resulted in an empty DataFrame. Not saving.")

    logger.info("Feature engineering process finished.")

if __name__ == "__main__":
    main()
