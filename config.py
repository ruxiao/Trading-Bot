# IBKR Connection Parameters

# IBKR_HOST: The hostname or IP address of the machine running IBKR Trader Workstation (TWS) or IBKR Gateway.
# Default: '127.0.0.1' (localhost)
IBKR_HOST = '127.0.0.1'

# IBKR_PORT: The socket port number configured in TWS/Gateway for API connections.
# Common defaults:
# - 7496: Live TWS account
# - 7497: Paper TWS account
# - 4001: Live Gateway account
# - 4002: Paper Gateway account
IBKR_PORT = 7497

# IBKR_CLIENT_ID: A unique integer ID for this API client connection.
# Each concurrent API connection to TWS/Gateway must use a unique client ID.
# Valid range is typically 1-255, but can be higher depending on TWS/Gateway settings.
IBKR_CLIENT_ID = 1

# Add other configurations here as needed
# For example:
# LOG_LEVEL = 'INFO' # Logging level for the application
# DEFAULT_ACCOUNT = 'YourPaperAccountNumber' # Specify a default account for trading if managing multiple
# LIVE_TRADING_SYMBOLS = ['AAPL', 'MSFT', 'EURUSD'] # Default symbols for live trading
# BACKTEST_DEFAULT_START_DATE = '2022-01-01' # Default start date for backtests
# BACKTEST_DEFAULT_END_DATE = '2023-01-01'   # Default end date for backtests
