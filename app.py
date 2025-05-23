import streamlit as st
import logging
import pandas as pd
from ib_insync import Stock, Forex, Contract # For creating contract objects

# Assuming your custom modules are in the same directory or accessible via PYTHONPATH
from ibkr_client import IBKRClient
from trading_strategy import TradingStrategy
from performance import PerformanceAnalyzer # For displaying backtest results
import config # For default connection params
from datetime import datetime, timedelta

# --- Global Variables & Configuration ---
# Configure logging for the app
# For Streamlit, consider using st.text_area for a simple log display
class StreamlitLogHandler(logging.Handler):
    def __init__(self, text_area_widget_key): # Pass key instead of widget
        super().__init__()
        self.text_area_widget_key = text_area_widget_key
        if self.text_area_widget_key not in st.session_state:
            st.session_state[self.text_area_widget_key] = []

    def emit(self, record):
        log_entry = self.format(record)
        st.session_state[self.text_area_widget_key].append(log_entry)
        # Keep last N records to prevent slowdown
        if len(st.session_state[self.text_area_widget_key]) > 200: # Increased limit
            st.session_state[self.text_area_widget_key] = st.session_state[self.text_area_widget_key][-200:]
        # No direct widget update here, Streamlit handles it on rerun

logger = logging.getLogger() # Get root logger
logger.setLevel(logging.INFO)

# Remove existing handlers to prevent duplication if script reruns in some environments (less common with Streamlit)
if logger.hasHandlers():
    for handler in logger.handlers[:]: # Iterate over a copy
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, StreamlitLogHandler): # Keep only our Streamlit handler if it was added
            logger.removeHandler(handler)


# --- Streamlit Session State Initialization ---
default_states = {
    'ibkr_client': None,
    'trading_strategy': None,
    'live_trading_active': False,
    'subscribed_contracts': {}, # Stores {symbol_str: Contract}
    'active_subscriptions': {}, # Stores {symbol_str: BarDataList_obj}
    'backtest_results': None,
    'backtest_trade_log': pd.DataFrame(),
    'backtest_equity_curve': pd.DataFrame(),
    'app_logs': [] # Initialize log storage in session state
}
for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --- Helper Functions ---
def get_contract_object(symbol_input: str) -> Contract | None:
    """
    Creates an IB Contract object from a symbol string.
    Assumes Stocks on SMART in USD, or Forex pairs.

    Args:
        symbol_input (str): The symbol string (e.g., "AAPL", "EURUSD", "EUR.USD").

    Returns:
        Contract | None: An ib_insync Contract object or None if input is invalid.
    """
    symbol_input = symbol_input.strip().upper()
    if not symbol_input: return None
    # Basic check for Forex pairs (e.g., EURUSD, EUR.USD)
    if '.' in symbol_input or (len(symbol_input) == 6 and not any(char.isdigit() for char in symbol_input)):
        # ib_insync's Forex contract typically doesn't use '.', so remove it.
        return Forex(symbol_input.replace('.', ''))
    else: # Assume it's a stock
        return Stock(symbol_input, 'SMART', 'USD')

# --- UI Callbacks ---
def handle_connect():
    """
    Handles the 'Connect' button action.
    Establishes a connection to IBKR TWS/Gateway using parameters from session state
    (defaulting to config.py values) and manages the asyncio event loop.
    Updates session state with the IBKRClient instance.
    """
    logger.info("Attempting to connect to IBKR...")
    host = st.session_state.get('ibkr_host_input', config.IBKR_HOST)
    port = st.session_state.get('ibkr_port_input', config.IBKR_PORT)
    client_id = st.session_state.get('ibkr_client_id_input', config.IBKR_CLIENT_ID)

    # Prevent multiple connection attempts if already connected
    if st.session_state.ibkr_client and st.session_state.ibkr_client.ib.isConnected():
        st.sidebar.warning("Already connected.")
        logger.warning("Connection attempt while already connected.")
        return

    try:
        # Instantiate the client
        client = IBKRClient(host=host, port=port, clientId=client_id)
        client.connect() # Attempt connection
        
        # Start or patch the asyncio event loop, crucial for ib_insync callbacks
        if client.run_async_event_loop_if_needed():
            st.session_state.ibkr_client = client # Store client in session state
            st.sidebar.success(f"Connected: {host}:{port} (ID: {client_id})")
            logger.info(f"IBKR connection successful: {host}:{port}, ClientID {client_id}.")
        else:
            # If loop management fails, connection might be unstable for callbacks
            st.sidebar.error("Loop Error. Callbacks may fail.")
            logger.error("IBKR connected but event loop management failed.")
            try: client.disconnect() # Attempt to clean up
            except Exception as e_disc: logger.error(f"Error disconnecting after loop fail: {e_disc}")
    except Exception as e:
        st.sidebar.error(f"Connection Failed: {str(e)}")
        logger.error(f"IBKR connection error: {e}", exc_info=True)

def handle_disconnect():
    """
    Handles the 'Disconnect' button action.
    Stops live trading if active, disconnects from IBKR, and clears related session state.
    """
    logger.info("Attempting to disconnect from IBKR...")
    if st.session_state.live_trading_active:
        handle_stop_trading() # Ensure live trading is stopped first

    client = st.session_state.ibkr_client
    if client and client.ib.isConnected():
        try:
            client.disconnect() # IBKRClient.disconnect handles cancelling subscriptions
            st.sidebar.info("Disconnected from IBKR.")
            logger.info("Successfully disconnected from IBKR.")
        except Exception as e:
            st.sidebar.error(f"Disconnect Error: {str(e)}")
            logger.error(f"Error during IBKR disconnection: {e}", exc_info=True)
    # If not connected, no specific warning needed as it's the desired state or already handled.
    
    # Clear all IBKR and trading related session state
    st.session_state.ibkr_client = None
    st.session_state.subscribed_contracts.clear()
    st.session_state.active_subscriptions.clear()
    st.session_state.live_trading_active = False 
    st.session_state.trading_strategy = None # Clear strategy object


def handle_start_trading():
    """
    Handles the 'Start Live Trading' button action.
    Initializes the TradingStrategy, subscribes to real-time bars for selected symbols,
    and sets the application to live trading mode.
    """
    logger.info("Attempting to start live paper trading...")
    client = st.session_state.ibkr_client # IBKRClient instance from session state
    if not client or not client.ib.isConnected():
        st.error("Not connected to IBKR. Please connect first.")
        logger.warning("Start trading attempt without IBKR connection.")
        return
    if st.session_state.live_trading_active:
        st.warning("Live trading is already active.")
        return

    # Get parameters from UI via session state
    symbols_str = st.session_state.get('live_symbols_input', "AAPL,TSLA") # Default if key missing
    initial_capital = st.session_state.get('live_initial_capital_input', 100000.0)
    alpha_short_ma = st.session_state.get('live_alpha_short_ma', 10) # Default value for short MA
    alpha_long_ma = st.session_state.get('live_alpha_long_ma', 20)   # Default value for long MA
    
    if not symbols_str:
        st.error("Please enter symbols for live trading.")
        return
    
    symbols_list = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if not symbols_list:
        st.error("No valid symbols entered for live trading.")
        return

    # Construct alpha parameters list for TradingStrategy
    alpha_params = [
        alpha_short_ma, alpha_long_ma, 
        st.session_state.get('live_alpha_ma_weight', 0.6),       # Default MA weight
        st.session_state.get('live_alpha_rsi_os_weight', 0.25),  # Default RSI oversold weight
        st.session_state.get('live_alpha_rsi_ob_weight', -0.25), # Default RSI overbought weight
        st.session_state.get('live_alpha_corr_weight', 0.15)     # Default Correlation weight
    ]
    # Create a new TradingStrategy instance for this session
    strategy = TradingStrategy(alpha_params=alpha_params)
    strategy.start_live_trading(initial_capital=initial_capital, ibkr_client_instance=client)
    st.session_state.trading_strategy = strategy # Store this active strategy
    
    # Clear any previous subscription states before starting new ones
    st.session_state.subscribed_contracts.clear()
    st.session_state.active_subscriptions.clear()
    
    success_symbols, failed_symbols = [], []
    for sym_str in symbols_list:
        contract = get_contract_object(sym_str) # Create contract object
        if not contract:
            failed_symbols.append(f"{sym_str} (Invalid contract format)")
            continue
        try:
            logger.info(f"Subscribing to real-time bars for {sym_str}...")
            # The strategy's on_realtime_bar method is passed as the callback
            bars_obj = client.subscribe_realtime_bars(contract, strategy.on_realtime_bar)
            if bars_obj and hasattr(bars_obj, 'contract'): # Check if subscription was successful
                # Store the qualified contract (returned by subscribe_realtime_bars within bars_obj)
                # and the BarDataList object for potential cancellation later.
                st.session_state.subscribed_contracts[sym_str] = bars_obj.contract 
                st.session_state.active_subscriptions[sym_str] = bars_obj
                success_symbols.append(f"{sym_str} (ConID: {bars_obj.contract.conId})")
                logger.info(f"Successfully subscribed to {sym_str} (ConID: {bars_obj.contract.conId}).")
            else:
                logger.error(f"Subscription failed for {sym_str}. Contract: {contract}, Received: {bars_obj}")
                failed_symbols.append(f"{sym_str} (Subscription failed: No valid bars object returned)")
        except Exception as e:
            logger.error(f"Error subscribing to {sym_str}: {e}", exc_info=True)
            failed_symbols.append(f"{sym_str} (Error: {str(e)[:50]}...)") # Show truncated error

    if success_symbols:
        st.success(f"Live trading started for: {', '.join(success_symbols)}.")
        st.session_state.live_trading_active = True # Set global flag
    if failed_symbols:
        st.error(f"Failed to start/subscribe for: {', '.join(failed_symbols)}.")


def handle_stop_trading():
    """
    Handles the 'Stop Live Trading' button action.
    Cancels all active real-time bar subscriptions and stops the trading strategy.
    The IBKR connection remains active.
    """
    logger.info("Attempting to stop live trading...")
    client = st.session_state.ibkr_client # Get client from session state
    strategy = st.session_state.trading_strategy # Get strategy from session state

    if strategy:
        strategy.stop_live_trading() # Notify strategy to stop
    
    cancelled_symbols, failed_cancellation = [], []
    if client and st.session_state.active_subscriptions: # Check if client and subscriptions exist
        # Iterate over a copy of items because cancel_realtime_bars might modify the dict via callbacks or internal state
        for sym_str, bars_obj in list(st.session_state.active_subscriptions.items()):
            try:
                logger.info(f"Attempting to cancel subscription for {sym_str} using object: {bars_obj}")
                client.cancel_realtime_bars(bars_obj) # Pass the BarDataList object
                cancelled_symbols.append(sym_str)
                logger.info(f"Cancelled real-time bar subscription for {sym_str}.")
            except Exception as e:
                failed_cancellation.append(f"{sym_str} (Error: {str(e)})")
                logger.error(f"Error cancelling subscription for {sym_str}: {e}", exc_info=True)
    
    if cancelled_symbols: st.info(f"Successfully unsubscribed from: {', '.join(cancelled_symbols)}")
    if failed_cancellation: st.error(f"Failed to unsubscribe from: {', '.join(failed_cancellation)}")

    st.session_state.live_trading_active = False # Update global flag
    st.session_state.active_subscriptions.clear() # Clear active subscription objects
    st.session_state.subscribed_contracts.clear() # Clear stored contracts
    # Keep st.session_state.trading_strategy to allow viewing of final state (capital, logs)
    st.info("Live trading stopped. Connection to IBKR remains active unless disconnected separately.")


def handle_run_backtest():
    """
    Handles the 'Run Backtest' button action.
    Fetches historical data via IBKRClient, runs the TradingStrategy's event-driven backtest,
    and stores/displays the results.
    """
    logger.info("Attempting to run event-driven backtest...")
    client = st.session_state.ibkr_client
    if not client or not client.ib.isConnected():
        st.error("Not connected to IBKR. Please connect first to fetch historical data.")
        return

    symbols_str = st.session_state.get('backtest_symbols_input', "AAPL,MSFT")
    start_date = st.session_state.get('backtest_start_date_input', datetime.now() - timedelta(days=90))
    end_date = st.session_state.get('backtest_end_date_input', datetime.now() - timedelta(days=1))
    bar_size = st.session_state.get('backtest_bar_size_input', "1 day")
    initial_capital = st.session_state.get('backtest_initial_capital_input', 100000.0)
    alpha_short_ma_bt = st.session_state.get('bt_alpha_short_ma', 10)
    alpha_long_ma_bt = st.session_state.get('bt_alpha_long_ma', 20)

    if not symbols_str:
        st.error("Please enter symbols for backtesting."); return
    symbols_list = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
    if not symbols_list:
        st.error("No valid symbols entered for backtesting."); return
    if end_date <= start_date:
        st.error("End date must be after start date."); return

    contracts_to_backtest = {}
    for sym_str in symbols_list:
        contract = get_contract_object(sym_str)
        if contract: contracts_to_backtest[sym_str] = contract
        else: st.error(f"Invalid symbol format for backtesting: {sym_str}"); return
    
    alpha_params_bt = [
        alpha_short_ma_bt, alpha_long_ma_bt, 
        st.session_state.get('bt_alpha_ma_weight', 0.6),
        st.session_state.get('bt_alpha_rsi_os_weight', 0.25),
        st.session_state.get('bt_alpha_rsi_ob_weight', -0.25),
        st.session_state.get('bt_alpha_corr_weight', 0.15)
    ]
    backtest_strategy = TradingStrategy(alpha_params=alpha_params_bt)
    
    with st.spinner(f"Running backtest for {', '.join(symbols_list)}..."):
        try:
            results, trade_log, equity_curve = backtest_strategy.backtest_event_driven(
                ibkr_client_instance=client, symbols_contracts=contracts_to_backtest,
                start_date_str=start_date.strftime('%Y-%m-%d'), end_date_str=end_date.strftime('%Y-%m-%d'),
                bar_size=bar_size, initial_capital=initial_capital, correlation_matrix_df=None
            )
            st.session_state.backtest_results = results
            st.session_state.backtest_trade_log = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
            st.session_state.backtest_equity_curve = equity_curve if equity_curve is not None else pd.DataFrame()
            st.success("Backtest completed!")
            logger.info("Backtest completed successfully.")
        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            logger.error(f"Backtest execution error: {e}", exc_info=True)
            st.session_state.backtest_results = None
            st.session_state.backtest_trade_log = pd.DataFrame()
            st.session_state.backtest_equity_curve = pd.DataFrame()

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="IBKR Trading Bot")
st.title("IBKR Event-Driven Trading & Backtesting")

# Setup log display area (sidebar)
log_text_area_key = "app_logs_display_key" # Unique key for the text_area
log_handler = StreamlitLogHandler(log_text_area_key)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
if not any(isinstance(h, StreamlitLogHandler) for h in logger.handlers):
    logger.addHandler(log_handler)

st.sidebar.header("IBKR Connection")
st.sidebar.text_input("Host", value=config.IBKR_HOST, key='ibkr_host_input')
st.sidebar.number_input("Port", value=config.IBKR_PORT, key='ibkr_port_input', min_value=1, max_value=65535, format="%d")
st.sidebar.number_input("Client ID", value=config.IBKR_CLIENT_ID, key='ibkr_client_id_input', min_value=0, format="%d")

col_conn1, col_conn2 = st.sidebar.columns(2)
col_conn1.button("Connect", on_click=handle_connect, use_container_width=True)
col_conn2.button("Disconnect", on_click=handle_disconnect, use_container_width=True)

client_status = st.session_state.ibkr_client
if client_status and client_status.ib.isConnected():
    st.sidebar.success(f"Status: Connected ({client_status.host}:{client_status.port}, ID:{client_status.clientId})")
else:
    st.sidebar.error("Status: Disconnected")

st.sidebar.header("Logs")
st.sidebar.text_area("App Logs", value="\n".join(st.session_state.get(log_text_area_key, [])), height=200, key="log_display_final_area", disabled=True)


# --- Main Content Tabs ---
tab_live, tab_backtest = st.tabs(["Live Paper Trading (IBKR)", "Event-Driven Backtesting (IBKR)"])

with tab_live:
    st.header("Live Paper Trading Controls")
    if not (st.session_state.ibkr_client and st.session_state.ibkr_client.ib.isConnected()):
        st.warning("Connect to IBKR to enable live trading.")
    
    st.text_input("Symbols (comma-separated, e.g., AAPL,EURUSD)", value="AAPL,TSLA", key='live_symbols_input', disabled=st.session_state.live_trading_active)
    st.number_input("Initial Capital", value=100000.0, key='live_initial_capital_input', format="%.2f", disabled=st.session_state.live_trading_active)
    
    st.subheader("Alpha Parameters (Live)")
    col_alpha_live1, col_alpha_live2 = st.columns(2)
    col_alpha_live1.number_input("Short MA", value=10, key='live_alpha_short_ma', min_value=1, disabled=st.session_state.live_trading_active)
    col_alpha_live1.number_input("Long MA", value=20, key='live_alpha_long_ma', min_value=2, disabled=st.session_state.live_trading_active)
    col_alpha_live2.number_input("MA Weight", value=0.6, key='live_alpha_ma_weight', format="%.2f", disabled=st.session_state.live_trading_active)
    col_alpha_live2.number_input("RSI OS Weight", value=0.25, key='live_alpha_rsi_os_weight', format="%.2f", disabled=st.session_state.live_trading_active)
    col_alpha_live1.number_input("RSI OB Weight", value=-0.25, key='live_alpha_rsi_ob_weight', format="%.2f", disabled=st.session_state.live_trading_active)
    col_alpha_live1.number_input("Corr Weight", value=0.15, key='live_alpha_corr_weight', format="%.2f", disabled=st.session_state.live_trading_active)


    col_trade1, col_trade2 = st.columns(2)
    col_trade1.button("Start Live Trading", on_click=handle_start_trading, 
                      disabled=not (st.session_state.ibkr_client and st.session_state.ibkr_client.ib.isConnected()) or st.session_state.live_trading_active, 
                      use_container_width=True)
    col_trade2.button("Stop Live Trading", on_click=handle_stop_trading, 
                      disabled=not st.session_state.live_trading_active, 
                      use_container_width=True)

    if st.session_state.live_trading_active:
        st.info(f"Live trading is ON for: {', '.join(st.session_state.subscribed_contracts.keys())}")
    else:
        st.info("Live trading is OFF.")

    st.subheader("Live Strategy State")
    strategy_instance = st.session_state.get('trading_strategy')
    if strategy_instance: # Display even if not active, to see last state
        st.metric("Strategy Capital", f"${strategy_instance.live_capital:,.2f}")
        
        st.subheader("Current Positions (Strategy View)")
        if strategy_instance.live_positions:
            st.dataframe(pd.DataFrame.from_dict(strategy_instance.live_positions, orient='index'))
        else:
            st.write("No active positions in strategy.")

        st.subheader("Trade Log (Current Live Session)")
        if strategy_instance.trade_log:
            st.dataframe(pd.DataFrame(strategy_instance.trade_log).tail(10)) # Show last 10
        else:
            st.write("No trades yet in this session.")
    else:
        st.write("Trading strategy not initialized yet.")


with tab_backtest:
    st.header("Event-Driven Backtest Configuration")
    if not (st.session_state.ibkr_client and st.session_state.ibkr_client.ib.isConnected()):
        st.warning("Connect to IBKR to enable backtesting data fetching.")

    st.text_input("Symbols (comma-separated)", value="AAPL,MSFT", key='backtest_symbols_input')
    d_col1, d_col2 = st.columns(2)
    d_col1.date_input("Start Date", value=datetime.now() - timedelta(days=90), key='backtest_start_date_input')
    d_col2.date_input("End Date", value=datetime.now() - timedelta(days=1), key='backtest_end_date_input')
    
    st.selectbox("Bar Size", 
                 options=["1 secs", "5 secs", "10 secs", "15 secs", "30 secs", "1 min", "2 mins", "3 mins", "5 mins", "10 mins", "15 mins", "20 mins", "30 mins", "1 hour", "2 hours", "3 hours", "4 hours", "8 hours", "1 day", "1 week", "1 month"], 
                 index=13, key='backtest_bar_size_input') # Default to 1 hour
    st.number_input("Initial Capital (Backtest)", value=100000.0, key='backtest_initial_capital_input', format="%.2f")

    st.subheader("Alpha Parameters (Backtest)")
    col_alpha_bt1, col_alpha_bt2 = st.columns(2)
    col_alpha_bt1.number_input("Short MA", value=10, key='bt_alpha_short_ma', min_value=1)
    col_alpha_bt1.number_input("Long MA", value=20, key='bt_alpha_long_ma', min_value=2)
    col_alpha_bt2.number_input("MA Weight", value=0.6, key='bt_alpha_ma_weight', format="%.2f")
    col_alpha_bt2.number_input("RSI OS Weight", value=0.25, key='bt_alpha_rsi_os_weight', format="%.2f")
    col_alpha_bt1.number_input("RSI OB Weight", value=-0.25, key='bt_alpha_rsi_ob_weight', format="%.2f")
    col_alpha_bt1.number_input("Corr Weight", value=0.15, key='bt_alpha_corr_weight', format="%.2f")


    st.button("Run Backtest", on_click=handle_run_backtest, 
              disabled=not (st.session_state.ibkr_client and st.session_state.ibkr_client.ib.isConnected()))

    if st.session_state.backtest_results:
        st.subheader("Backtest Performance Metrics")
        results = st.session_state.backtest_results
        if results and "error" not in results:
            col_bt_m1, col_bt_m2, col_bt_m3 = st.columns(3)
            col_bt_m1.metric("Initial Capital", f"${results.get('initial_capital', 0):,.2f}")
            col_bt_m1.metric("Final Capital", f"${results.get('final_capital', 0):,.2f}")
            col_bt_m2.metric("Total Return", f"{results.get('total_return', 0):.2%}")
            col_bt_m2.metric("Sharpe Ratio", f"${results.get('sharpe_ratio', 0):.2f}")
            col_bt_m3.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
            col_bt_m3.metric("Number of Trades", f"{results.get('num_trades', 0)}")
            
            st.subheader("Equity Curve (Backtest)")
            if not st.session_state.backtest_equity_curve.empty:
                st.line_chart(st.session_state.backtest_equity_curve['capital'])
            else: st.write("Equity curve data not available.")

            st.subheader("Trade Log (Backtest)")
            if not st.session_state.backtest_trade_log.empty:
                st.dataframe(st.session_state.backtest_trade_log)
            else: st.write("No trades in this backtest.")
        elif results and "error" in results:
             st.error(f"Backtest Error: {results['error']}")


logger.info("Streamlit app script execution finished. UI rendered.")