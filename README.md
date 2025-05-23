# Advanced Event-Driven Trading Bot with IBKR Integration

This project implements an advanced trading bot that connects to Interactive Brokers (IBKR) for live paper trading and provides an event-driven backtesting engine using historical data from IBKR. The bot employs a customizable trading strategy based on technical indicators and risk management rules.

## Key Features

*   **Event-Driven Architecture**: The bot is designed around an event-driven model, allowing it to react to real-time market data for live trading and simulate this behavior accurately in backtests.
*   **Interactive Brokers (IBKR) Integration**:
    *   Connects to IBKR Trader Workstation (TWS) or IBKR Gateway.
    *   Supports live paper trading with orders placed via the IBKR API.
    *   Fetches historical market data from IBKR for backtesting.
    *   Uses the `ib_insync` library for asynchronous communication with IBKR.
*   **Trading Strategy**:
    *   Core strategy logic based on Moving Averages (MA), Relative Strength Index (RSI), and (optionally) asset correlation.
    *   Includes configurable parameters for indicators and signal weighting.
    *   Incorporates risk management:
        *   Stop-loss and take-profit levels.
        *   Position sizing based on risk per trade and maximum position size percentages.
        *   Optional volatility-based position scaling.
        *   Models transaction costs and slippage.
*   **Live Paper Trading (IBKR)**:
    *   Streamlit UI to manage IBKR connection.
    *   Start and stop live paper trading sessions for selected symbols.
    *   Real-time processing of incoming market data (5-second bars by default).
    *   Live updates on capital, positions, and trade activity.
*   **Event-Driven Backtesting (IBKR)**:
    *   Fetches historical data for specified symbols and periods directly from IBKR.
    *   Simulates the strategy's event-driven logic bar-by-bar, providing a realistic performance assessment.
    *   Applies simulated transaction costs and slippage.
    *   Generates comprehensive performance metrics (Total Return, Sharpe Ratio, Max Drawdown, etc.) and trade logs.
*   **Streamlit Web Application (`app.py`)**:
    *   User-friendly interface to:
        *   Configure IBKR connection.
        *   Manage live paper trading sessions (start/stop, select symbols, set capital & alpha parameters).
        *   Configure and run event-driven backtests (select symbols, date range, bar size, capital & alpha parameters).
        *   View live trading status (capital, positions, trade log).
        *   View backtesting results (performance metrics, equity curve, trade log).
        *   Monitor application logs.

## Project Structure

```
├── ibkr_client.py         # Handles connection and data interaction with IBKR.
├── trading_strategy.py    # Implements the trading logic, signal generation, and backtesting engine.
├── performance.py         # Analyzes performance metrics.
├── app.py                 # Main Streamlit application for UI and control.
├── config.py              # Configuration for IBKR connection parameters.
├── requirements.txt       # Python dependencies.
├── tests/                 # Unit tests for the core modules.
│   ├── __init__.py
│   ├── test_ibkr_client.py
│   └── test_trading_strategy_event.py
└── README.md              # This file.
```

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.10+
    *   Interactive Brokers Trader Workstation (TWS) or IBKR Gateway installed and running.
        *   Ensure API access is enabled: In TWS, go to `File -> Global Configuration -> API -> Settings`. Enable "Enable ActiveX and Socket Clients". Note the "Socket port" (default is 7496 for live, 7497 for paper).
        *   For paper trading, ensure you are logged into your Paper Trading Account in TWS/Gateway.

2.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    *   `streamlit`: For the web application UI.
    *   `ib_insync`: For IBKR API communication.
    *   `pandas`, `numpy`: For data manipulation and numerical operations.
    *   `plotly`: For charting (used by `PerformanceAnalyzer`).
    *   `pytest`, `pytest-mock`: For running unit tests.

5.  **Configure IBKR Connection**:
    *   Edit `config.py` to set your IBKR TWS/Gateway connection parameters:
        *   `IBKR_HOST`: Usually `'127.0.0.1'`.
        *   `IBKR_PORT`: The socket port configured in TWS/Gateway (e.g., `7497` for paper trading).
        *   `IBKR_CLIENT_ID`: A unique client ID (e.g., `1` to `100+`). Each running instance of the bot or other API connection should use a unique ID.

## Running the Application

1.  **Ensure IBKR TWS/Gateway is running and you are logged in.**
2.  **Activate your virtual environment** (if you created one).
3.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your web browser.

## Using the Application

### 1. Connect to IBKR
*   Navigate to the sidebar in the Streamlit application.
*   Verify the Host, Port, and Client ID. Adjust if necessary.
*   Click the "Connect" button. Status messages will indicate success or failure.
*   Logs in the sidebar provide more details on the connection process.

### 2. Live Paper Trading (IBKR)
*   Go to the "Live Paper Trading (IBKR)" tab.
*   Ensure you are connected to IBKR.
*   **Configure**:
    *   Enter symbols to trade (comma-separated, e.g., `AAPL,EURUSD,TSLA`).
    *   Set your initial paper trading capital.
    *   Adjust Alpha Parameters (MA windows, weights) for the trading strategy.
*   **Start Trading**: Click "Start Live Trading". The application will subscribe to real-time bar data for the specified symbols. The `TradingStrategy` instance will process these bars and may place paper orders via IBKR.
*   **Monitor**:
    *   View current strategy capital.
    *   See active positions held by the strategy.
    *   Check the live session trade log for executed orders or attempts.
*   **Stop Trading**: Click "Stop Live Trading". This will cancel active data subscriptions and stop new order generation. Positions will typically remain open unless the strategy logic dictates closing them (e.g. via a signal change when stopping).

### 3. Event-Driven Backtesting (IBKR)
*   Go to the "Event-Driven Backtesting (IBKR)" tab.
*   Ensure you are connected to IBKR (required for fetching historical data).
*   **Configure**:
    *   Enter symbols to backtest.
    *   Select the Start Date, End Date, and Bar Size for historical data.
    *   Set the Initial Capital for the backtest simulation.
    *   Adjust Alpha Parameters for the strategy used in the backtest.
*   **Run Backtest**: Click "Run Backtest". The application will:
    1.  Fetch historical data from IBKR for the selected symbols and period.
    2.  Create a chronological stream of bar events.
    3.  Process each bar through the `TradingStrategy`, simulating order execution (including slippage and transaction costs).
*   **View Results**:
    *   Key performance metrics (Total Return, Sharpe Ratio, Max Drawdown, etc.).
    *   An equity curve chart showing capital over time.
    *   A detailed trade log from the backtest.

## Development and Testing

*   **Unit Tests**: The project includes unit tests for `ibkr_client.py` and `trading_strategy.py`.
*   To run tests:
    ```bash
    pytest
    ```

## Disclaimer

This software is for educational and research purposes only. Trading financial markets involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use this software at your own risk. Paper trading is highly recommended before considering any real-money trading.
The authors and contributors are not responsible for any financial losses or other damages incurred from the use of this software.
```
