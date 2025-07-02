import logging
import os
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

from ib_insync import IB, Stock, util

# Assuming config.py is in the parent directory of src, or adjust path as needed
# For simplicity, let's assume config.py is accessible in thePYTHONPATH or same level for now
# Or, more robustly:
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) # Add repo root to path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = "data"
FILE_NAME = "qqq_1min_1month.csv"

def fetch_and_save_qqq_data(host: str, port: int, client_id: int, symbol: str = "QQQ",
                             duration_days: int = 30, bar_size: str = "1 min",
                             what_to_show: str = "TRADES", use_rth: bool = True) -> Optional[pd.DataFrame]:
    """
    Connects to IBKR, fetches historical 1-minute data for QQQ for the last month,
    and saves it to a CSV file.

    Args:
        host (str): IBKR host.
        port (int): IBKR port.
        client_id (int): IBKR client ID.
        symbol (str): Stock symbol to fetch (default: "QQQ").
        duration_days (int): How many past days of data to fetch (default: 30 for approx 1 month).
        bar_size (str): Bar size for historical data (default: "1 min").
        what_to_show (str): Data type (default: "TRADES").
        use_rth (bool): Use regular trading hours only (default: True).

    Returns:
        Optional[pd.DataFrame]: DataFrame of the fetched data, or None if an error occurs.
    """
    ib = IB()
    try:
        logger.info(f"Connecting to IBKR at {host}:{port} with ClientID {client_id}...")
        ib.connect(host, port, clientId=client_id, timeout=10)
        logger.info("Successfully connected to IBKR.")

        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        logger.info(f"Qualified contract: {contract}")

        # Calculate duration string for IBKR (e.g., "30 D")
        # IBKR max duration for 1-min bars is typically around 1 month in one go.
        # If longer is needed, multiple requests would be necessary.
        # For 30 days, "30 D" is fine. Let's use a more flexible approach for duration.
        # reqHistoricalData 'durationStr' expects format like 'X S/D/W/M/Y'
        # For 1-min bars, IBKR has limitations. A safe bet for '1 min' bars is up to 1 month ('1 M') or '30 D'.
        # Let's use days for precision.
        duration_str = f"{duration_days} D"

        logger.info(f"Fetching historical data for {symbol} for the past {duration_days} days ({duration_str}), Bar size: {bar_size}...")

        # endDateTime='' means up to the present.
        # UseRTH=True is important for stocks to avoid pre/post market if not desired.
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1  # Format as yyyyMMdd HH:mm:ss
        )

        if bars:
            df = util.df(bars)
            logger.info(f"Fetched {len(df)} bars of data for {symbol}.")

            if df is not None and not df.empty:
                # Ensure 'date' column is datetime and set as index
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)

                # Create data directory if it doesn't exist
                if not os.path.exists(DATA_DIR):
                    os.makedirs(DATA_DIR)

                file_path = os.path.join(DATA_DIR, FILE_NAME)
                df.to_csv(file_path)
                logger.info(f"Data saved to {file_path}")
                return df
            else:
                logger.warning("No data returned from IBKR.")
                return None
        else:
            logger.warning("No bars returned from IBKR reqHistoricalData call.")
            return None

    except ConnectionRefusedError:
        logger.error(f"Connection to IBKR refused. Ensure TWS/Gateway is running on {host}:{port} and API access is enabled.")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return None
    finally:
        if ib.isConnected():
            logger.info("Disconnecting from IBKR.")
            ib.disconnect()

if __name__ == "__main__":
    logger.info("Starting QQQ data fetch process...")

    # Use connection parameters from config.py
    ibkr_host = config.IBKR_HOST
    ibkr_port = config.IBKR_PORT
    # For data fetching, it's good to use a unique client ID,
    # different from any live trading or other app instance.
    ibkr_client_id_fetch = config.IBKR_CLIENT_ID + 50 # Example: offset for fetcher

    # Fetch data for the last 30 calendar days.
    # This should give roughly 20-22 trading days of 1-minute data.
    # If a full month of *trading days* is strictly needed, logic would be more complex.
    # For "1 month" as per request, 30 days is a reasonable interpretation.
    fetched_data = fetch_and_save_qqq_data(
        host=ibkr_host,
        port=ibkr_port,
        client_id=ibkr_client_id_fetch,
        symbol="QQQ",
        duration_days=30, # Approx 1 month
        bar_size="1 min"
    )

    if fetched_data is not None:
        logger.info(f"Successfully fetched and saved QQQ data. Shape: {fetched_data.shape}")
        logger.info(f"Columns: {fetched_data.columns.tolist()}")
        logger.info(f"Head:\n{fetched_data.head()}")
        logger.info(f"Tail:\n{fetched_data.tail()}")
    else:
        logger.error("Failed to fetch QQQ data.")

    logger.info("QQQ data fetch process finished.")
