import logging
from ib_insync import IB, util
from ib_insync.contract import Stock, Forex
from ib_insync.order import MarketOrder, LimitOrder
import pandas as pd

# Configure logging
# The global logger is configured in app.py now, or can be configured here if run standalone.
# For library use, it's better to get a logger instance:
logger = logging.getLogger(__name__)


class IBKRClient:
    """
    Client for interacting with Interactive Brokers (IBKR) TWS/Gateway.
    Manages connection, data fetching (historical & real-time), and order placement.
    Uses the ib_insync library for asynchronous communication.
    """
    def __init__(self, host: str, port: int, clientId: int):
        """
        Initializes the IBKRClient.

        Args:
            host (str): The hostname or IP address of the TWS/Gateway.
            port (int): The port number for the API connection.
            clientId (int): A unique client ID for this connection.
        """
        self.host = host
        self.port = port
        self.clientId = clientId
        self.ib = IB()
        # Stores active real-time bar subscriptions: {conId: {'contract': Contract, 'bars': BarDataList, 'callback': function}}
        self._active_realtime_bars: dict = {} 
        self.logger = logging.getLogger(f"{__name__}.IBKRClient") # Instance specific logger

    def connect(self, timeout: int = 10):
        """
        Connects to the IBKR TWS/Gateway.

        Args:
            timeout (int): Connection timeout in seconds.

        Raises:
            ConnectionRefusedError: If the connection is refused by TWS/Gateway.
            Exception: For other connection errors.
        """
        try:
            if not self.ib.isConnected():
                self.logger.info(f"Attempting to connect to IBKR: {self.host}:{self.port} with clientId {self.clientId}...")
                self.ib.connect(self.host, self.port, self.clientId, timeout=timeout)
                self.logger.info(f"Successfully connected to IBKR: {self.host}:{self.port} with clientId {self.clientId}")
        except ConnectionRefusedError as cre:
            self.logger.error(f"Connection refused by IBKR TWS/Gateway: {cre}. Ensure TWS/Gateway is running and API connections are enabled.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}", exc_info=True)
            raise

    def disconnect(self):
        """
        Disconnects from IBKR TWS/Gateway.
        Cancels any active real-time bar subscriptions before disconnecting.
        """
        self.logger.info("Disconnecting from IBKR...")
        # Cancel all active real-time bar subscriptions before disconnecting
        # Iterate over a copy of keys as cancel_realtime_bars modifies the dictionary
        for con_id_key in list(self._active_realtime_bars.keys()):
            # cancel_realtime_bars uses the BarDataList object for cancellation which is stored in 'bars'
            # It will find the contract symbol from the bars object itself if needed for logging.
            bars_object_to_cancel = self._active_realtime_bars[con_id_key]['bars']
            self.logger.info(f"Cancelling real-time bars for contract related to conId {con_id_key} before disconnecting.")
            try:
                # Pass the BarDataList object directly for cancellation
                self.cancel_realtime_bars(bars_object_to_cancel) 
            except Exception as e:
                # Log error but continue disconnecting other subscriptions and the client itself
                self.logger.error(f"Error cancelling real-time bars for conId {con_id_key} during disconnect: {e}", exc_info=True)
        
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Successfully disconnected from IBKR.")
        else:
            self.logger.info("Already disconnected or was never connected.")
        self._active_realtime_bars.clear() # Ensure this is cleared

    def qualify_contract(self, contract: Contract) -> Contract | None:
        """
        Qualifies a contract to resolve ambiguities and fill in details like conId.
        IBKR requires contracts to be qualified before use in most API calls.

        Args:
            contract (Contract): The ib_insync Contract object to qualify (e.g., Stock, Forex).

        Returns:
            Contract | None: The qualified contract object if successful, otherwise None.
        
        Raises:
            Exception: If there's an error during the qualification process from IBKR.
        """
        self.logger.debug(f"Qualifying contract: {contract}")
        try:
            self.connect() # Ensure connection
            qualified_contracts = self.ib.qualifyContracts(contract)
            if qualified_contracts:
                if len(qualified_contracts) == 1:
                    self.logger.info(f"Contract qualified successfully: {qualified_contracts[0]}")
                    return qualified_contracts[0]
                else:
                    # Handle ambiguity: if a SMART contract is preferred and found, use it.
                    self.logger.warning(f"Contract {contract.symbol if hasattr(contract, 'symbol') else contract} is ambiguous. Found: {qualified_contracts}")
                    if hasattr(contract, 'exchange') and contract.exchange and contract.exchange.upper() == "SMART":
                        for qc in qualified_contracts:
                            if qc.exchange == "SMART":
                                self.logger.info(f"Returning first SMART contract for ambiguous query: {qc}")
                                return qc
                    # If no specific preference or SMART not found among ambiguous, log error or return first as fallback.
                    # Current logic from previous version returns the first one if original contract.exchange was "SMART".
                    # A safer default might be to return None if truly ambiguous and not resolved.
                    self.logger.warning(f"Ambiguous contract {contract.symbol if hasattr(contract, 'symbol') else contract}. Returning the first from list as a fallback: {qualified_contracts[0]}")
                    return qualified_contracts[0] # Fallback to first, user should be specific.
            else:
                self.logger.error(f"Could not qualify contract: {contract.symbol if hasattr(contract, 'symbol') else contract}. No matching contract found by IBKR.")
                return None
        except Exception as e:
            self.logger.error(f"Error qualifying contract {contract.symbol if hasattr(contract, 'symbol') else contract}: {e}", exc_info=True)
            raise # Re-raise to signal failure to caller

    def fetch_historical_data(self, contract: Contract, endDateTime: str, durationStr: str, 
                              barSizeSetting: str, whatToShow: str, useRTH: bool, 
                              formatDate: int = 1) -> pd.DataFrame:
        """
        Fetches historical bar data from IBKR.

        Args:
            contract (Contract): The qualified ib_insync Contract object.
            endDateTime (str): The end date/time of the historical data request.
                               Format: 'yyyyMMdd HH:mm:ss [zzz]' or '' for current time.
            durationStr (str): The duration of the data request (e.g., '30 D', '1 M', '1 Y').
            barSizeSetting (str): The bar size (e.g., '1 min', '1 hour', '1 day').
            whatToShow (str): The type of data to show (e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK').
            useRTH (bool): If True, data is returned for regular trading hours only.
            formatDate (int): Date formatting for IBKR (1 for yyyyMMdd HH:mm:ss, 2 for system time zone).

        Returns:
            pd.DataFrame: A DataFrame containing the historical bar data, or an empty DataFrame if no data.
        
        Raises:
            Exception: If there's an error during data fetching from IBKR.
        """
        self.logger.info(f"Fetching historical data for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}: "
                         f"End: {endDateTime}, Duration: {durationStr}, Bar: {barSizeSetting}")
        try:
            self.connect() 
            
            qualified_contract = self.qualify_contract(contract)
            if not qualified_contract:
                self.logger.warning(f"Historical data fetch failed for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol} due to contract qualification failure.")
                return pd.DataFrame()

            bars = self.ib.reqHistoricalData(
                qualified_contract,
                contract,
                endDateTime=endDateTime,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow=whatToShow,
                useRTH=useRTH,
                endDateTime=endDateTime,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate 
            )
            if bars:
                df = util.df(bars) # Converts list of BarData to DataFrame
                self.logger.info(f"Fetched {len(bars)} bars for {qualified_contract.symbol}.")
                # Ensure 'date' column is datetime objects if present, util.df usually handles this.
                if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
                     df['date'] = pd.to_datetime(df['date']) # pragma: no cover (util.df should handle)
                return df
            else:
                self.logger.warning(f"No historical data returned for {qualified_contract.symbol}.")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}: {e}", exc_info=True)
            raise

    def place_paper_order(self, contract: Contract, order: Order) -> Trade | None:
        """
        Places a paper order.
        Note: For paper trading, orders are simulated by IB TWS/Gateway.
        """
        Places a paper trading order.
        Note: Ensure you are connected to a paper trading account in TWS/Gateway.

        Args:
            contract (Contract): The qualified ib_insync Contract object for the instrument.
            order (Order): The ib_insync Order object (e.g., MarketOrder, LimitOrder).

        Returns:
            Trade | None: The ib_insync Trade object if order placement was initiated, otherwise None.
                         The Trade object tracks the order's status and fills.
        
        Raises:
            Exception: If there's an error during order placement from IBKR.
        """
        self.logger.info(f"Attempting to place paper order for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}: "
                         f"{order.action} {order.totalQuantity} units.")
        try:
            self.connect() 

            qualified_contract = self.qualify_contract(contract)
            if not qualified_contract:
                self.logger.error(f"Cannot place order for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}, contract qualification failed.")
                return None

            # Optionally, specify paper trading account if managing multiple accounts
            # if self.ib.managedAccounts():
            #     order.account = self.ib.managedAccounts()[0] # Or a specific paper account

            trade = self.ib.placeOrder(qualified_contract, order)
            self.logger.info(f"Paper order placed for {qualified_contract.symbol}. OrderId: {trade.order.orderId}, "
                             f"PermId: {trade.order.permId}, Status: {trade.orderStatus.status}")
            
            # For more advanced handling, you can register callbacks for order status changes:
            # trade.filledEvent += self._on_order_filled_callback
            # trade.statusEvent += self._on_order_status_callback
            
            return trade
        except Exception as e:
            self.logger.error(f"Error placing paper order for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}: {e}", exc_info=True)
            raise

    def subscribe_realtime_bars(self, contract: Contract, callback_function, 
                                barSize: int = 5, whatToShow: str = 'TRADES', useRTH: bool = True) -> util.BarDataList | None:
        """
        Subscribes to real-time bars for a given contract.

        Args:
            contract (Contract): The ib_insync Contract object to subscribe to.
            callback_function (function): A function to be called when a new bar is received.
                                          It will receive (bar_data_dict, contract_object).
            barSize (int): The bar size in seconds. IBKR typically supports 5-second bars for TRADES.
                           Other values might be valid for different whatToShow types (e.g., MIDPOINT).
            whatToShow (str): Type of data for bars (e.g., 'TRADES', 'MIDPOINT', 'BID_ASK').
            useRTH (bool): If True, only bars during regular trading hours are delivered.

        Returns:
            ib_insync.util.BarDataList | None: The BarDataList object representing the subscription,
                                                or None if subscription failed. This object can be used
                                                to cancel the subscription.
        
        Raises:
            Exception: If there's an error during subscription setup from IBKR.
        """
        self.logger.info(f"Attempting to subscribe to real-time bars for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}...")
        try:
            self.connect() 
            
            qualified_contract = self.qualify_contract(contract)
            if not qualified_contract:
                self.logger.error(f"Cannot subscribe to real-time bars for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}, contract qualification failed.")
                return None

            con_id = qualified_contract.conId
            if con_id in self._active_realtime_bars:
                self.logger.warning(f"Already subscribed to real-time bars for {qualified_contract.symbol} (ConID: {con_id}). Returning existing subscription object.")
                return self._active_realtime_bars[con_id]['bars']

            self.logger.info(f"Subscribing to real-time bars for {qualified_contract.symbol} (ConID: {con_id}), BarSize: {barSize}s, WhatToShow: {whatToShow}, UseRTH: {useRTH}")
            
            # reqRealTimeBars returns a BarDataList which is an event source for bar updates.
            bars = self.ib.reqRealTimeBars(
                contract=qualified_contract,
                barSize=barSize, # Note: IB API has limitations, 5 seconds is standard for TRADES.
                whatToShow=whatToShow,
                useRTH=useRTH
            )
            
            self._active_realtime_bars[con_id] = {
                'contract': qualified_contract, 
                'bars': bars, # Store the BarDataList object
                'callback': callback_function
            }
            bars.updateEvent += self._on_bar_update # Register internal handler for new bar events

            self.logger.info(f"Successfully subscribed to real-time bars for {qualified_contract.symbol}. Listening for updates...")
            return bars
            
        except Exception as e:
            self.logger.error(f"Error subscribing to real-time bars for {contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol}: {e}", exc_info=True)
            # Clean up if partial subscription happened before error
            if 'qualified_contract' in locals() and qualified_contract and qualified_contract.conId in self._active_realtime_bars:
                 del self._active_realtime_bars[qualified_contract.conId] # pragma: no cover (hard to test this specific path)
            raise

    def _on_bar_update(self, bars: util.BarDataList, hasNewBar: bool):
        """
        Internal callback triggered by ib_insync when new bar data is received
        for any real-time bar subscription. It then calls the user-provided callback.

        Args:
            bars (ib_insync.util.BarDataList): The BarDataList object that was updated.
                                               Contains all bars for that subscription, newest is bars[-1].
            hasNewBar (bool): True if bars[-1] is a new bar, False if it's an update to the last bar.
        """
        if hasNewBar:
            # Find the subscription details (contract, user_callback) associated with this 'bars' object
            found_sub_info = None
            for con_id_iter, sub_info_iter in self._active_realtime_bars.items():
                if sub_info_iter['bars'] is bars: # Check if it's the same BarDataList object
                    found_sub_info = sub_info_iter
                    break
            
            if found_sub_info:
                latest_bar: BarData = bars[-1] # The newest bar data
                contract_info: Contract = found_sub_info['contract']
                user_callback = found_sub_info['callback']
                
                self.logger.debug(f"New bar for {contract_info.symbol}: Time={latest_bar.time}, Close={latest_bar.close}")
                
                try:
                    # Prepare a dictionary representation of the bar for the user callback
                    bar_data_dict = {
                        "time": latest_bar.time,    # datetime object (usually)
                        "open": latest_bar.open_,
                        "high": latest_bar.high,
                        "low": latest_bar.low,
                        "close": latest_bar.close,
                        "volume": latest_bar.volume,
                        "wap": latest_bar.wap,      # Weighted Average Price
                        "count": latest_bar.count,  # Number of trades in the bar
                        "symbol": contract_info.symbol, # Convenience: add symbol
                        "conId": contract_info.conId    # Convenience: add conId
                    }
                    user_callback(bar_data_dict, contract_info)
                except Exception as e:
                    self.logger.error(f"Error executing user-provided callback for {contract_info.symbol}: {e}", exc_info=True)
            else:
                # This should ideally not happen if subscription management is correct.
                self.logger.warning(f"Received bar update for an untracked or unknown BarDataList subscription. Bar time: {bars[-1].time if bars and bars[-1] else 'N/A'}")


    def cancel_realtime_bars(self, contract_or_bars_object: Contract | util.BarDataList):
        """
        Cancels an active real-time bar subscription.

        Args:
            contract_or_bars_object (Contract | util.BarDataList): 
                Either the original (or qualified) Contract object used for subscription,
                or the BarDataList object that was returned by `subscribe_realtime_bars`.
        
        Raises:
            Exception: If there's an error during cancellation from IBKR.
        """
        self.logger.info(f"Attempting to cancel real-time bars for: {contract_or_bars_object}")
        try:
            self.connect() 
            
            con_id_to_cancel = None
            bars_obj_to_cancel = None
            target_symbol_for_log = "Unknown"

            if isinstance(contract_or_bars_object, Contract):
                # If it's a contract, we need its conId.
                # It's best if it's already qualified. If not, qualify it.
                # Then find the BarDataList from our _active_realtime_bars using conId.
                q_contract = self.qualify_contract(contract_or_bars_object) # Ensure we have conId
                if q_contract:
                    con_id_to_cancel = q_contract.conId
                    target_symbol_for_log = q_contract.symbol
                    if con_id_to_cancel in self._active_realtime_bars:
                        bars_obj_to_cancel = self._active_realtime_bars[con_id_to_cancel]['bars']
                    else:
                        self.logger.warning(f"No active subscription found for conId {con_id_to_cancel} ({target_symbol_for_log}) from the provided contract.")
                        return # Nothing to cancel if not tracked
                else:
                    self.logger.error(f"Could not qualify contract for cancellation: {contract_or_bars_object.symbol if hasattr(contract_or_bars_object, 'symbol') else contract_or_bars_object}. Cannot determine conId.")
                    return
            elif isinstance(contract_or_bars_object, util.BarDataList):
                # If it's a BarDataList object, we can use it directly.
                # Find its conId from its associated contract to update our tracking.
                bars_obj_to_cancel = contract_or_bars_object
                if hasattr(bars_obj_to_cancel, 'contract') and bars_obj_to_cancel.contract:
                    con_id_to_cancel = bars_obj_to_cancel.contract.conId
                    target_symbol_for_log = bars_obj_to_cancel.contract.symbol
                else: # pragma: no cover (BarDataList should always have .contract from reqRealTimeBars)
                    self.logger.error("Invalid BarDataList object passed for cancellation: missing .contract attribute.")
                    return
            else:
                self.logger.error(f"Invalid argument type for cancel_realtime_bars: {type(contract_or_bars_object)}. "
                                  "Must be a Contract or BarDataList object.")
                return

            if con_id_to_cancel and con_id_to_cancel in self._active_realtime_bars:
                # Ensure we have the correct bars object if we started with a contract
                if not bars_obj_to_cancel: # This case should be covered by logic above, but as safeguard
                    bars_obj_to_cancel = self._active_realtime_bars[con_id_to_cancel]['bars'] # pragma: no cover
                
                self.logger.info(f"Cancelling real-time bars subscription for {target_symbol_for_log} (ConID: {con_id_to_cancel}).")
                
                # De-register our internal handler from the updateEvent
                # Check if it's actually registered to avoid errors on multiple cancel calls or race conditions.
                # This requires careful checking as ib_insync's event system might not expose handlers list directly.
                # A simple try-except can also work if removing a non-existent handler raises specific error.
                try:
                    # Assuming __isub__ ( -= ) is robust to removing non-existent handlers or doesn't error.
                    bars_obj_to_cancel.updateEvent -= self._on_bar_update
                except Exception as event_err: # pragma: no cover (specific exception type depends on ib_insync)
                    self.logger.warning(f"Could not de-register _on_bar_update for {target_symbol_for_log} (ConID: {con_id_to_cancel}). "
                                        f"It might have been already removed or not set. Error: {event_err}")

                self.ib.cancelRealTimeBars(bars_obj_to_cancel)
                del self._active_realtime_bars[con_id_to_cancel] # Remove from our tracking
                self.logger.info(f"Successfully cancelled real-time bars for {target_symbol_for_log}.")
            elif con_id_to_cancel: # conId was found but not in our active list
                self.logger.warning(f"No active real-time subscription found in internal tracking for {target_symbol_for_log} (ConID: {con_id_to_cancel}) to cancel.")
            # If con_id_to_cancel was never found (e.g. contract qualification failed), already handled.

        except Exception as e:
            self.logger.error(f"Error cancelling real-time bars for {target_symbol_for_log}: {e}", exc_info=True)
            raise # Re-raise to signal failure to caller

    def run_async_event_loop_if_needed(self) -> bool:
        """
        Manages the ib_insync asyncio event loop, crucial for callbacks (e.g., real-time data).
        
        - If an asyncio loop is already running in the current context (e.g., Jupyter, Streamlit's thread),
          it patches ib_insync to use this existing loop via `util.patchAsyncio()`.
        - If no loop is running, it starts the ib_insync event loop in a new daemon thread
          using `util.startLoop()`. This is often suitable for scripts or integrations where
          the main thread needs to remain unblocked.
        
        This method should be called after a successful connection if real-time updates are needed.

        Returns:
            bool: True if the loop is running (either pre-existing or newly started), False otherwise.
        """
        if not self.ib.isConnected():
            self.logger.warning("IBKR not connected. Cannot manage event loop related operations.")
            return False
            
        try:
            import asyncio
            try:
                # Check if an asyncio loop is already running in the current thread
                asyncio.get_running_loop()
                # If here, a loop is running. Patch ib_insync to use it.
                self.logger.info("Existing asyncio loop detected. Patching ib_insync to use it.")
                util.patchAsyncio() # Ensures ib_insync uses this loop
                
                # If ib_insync's own loop/client task isn't specifically running on this loop yet, start it.
                if not self.ib.loop.is_running(): # pragma: no cover (hard to test specific state of IB loop on existing external loop)
                    self.logger.info("Starting ib_insync client task on the existing asyncio loop.")
                    self.ib.loop.create_task(self.ib.client.run_async()) 
                self.logger.info("ib_insync is now configured to use the existing asyncio loop.")
                return True
            except RuntimeError: # No asyncio loop is running in the current thread
                self.logger.info("No existing asyncio loop detected. Starting dedicated ib_insync event loop in a new thread.")
                util.startLoop(self.ib) # This starts ib.run() in a new daemon thread.
                if self.ib.loop.is_running():
                    self.logger.info("ib_insync event loop successfully started via util.startLoop().")
                    return True
                else: # pragma: no cover (startLoop itself should raise if it fails catastrophically)
                    self.logger.error("Failed to start ib_insync event loop using util.startLoop(). Callbacks may not work.")
                    return False
        except Exception as e:
            self.logger.error(f"Exception during asyncio event loop management: {e}", exc_info=True)
            return False


if __name__ == '__main__': # pragma: no cover
    # This block is for basic manual testing or demonstration.
    # Ensure TWS or Gateway is running on port 7497 (paper) or 7496 (live) for this to work.
    
    # Setup basic logging to console for this example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    
    client = IBKRClient(host='127.0.0.1', port=7497, clientId=10) # Use a unique clientId

    def my_simple_bar_handler(bar_data_dict, contract_object):
        """User-defined callback function for real-time bars."""
        logger.info(f"BAR HANDLER: Symbol={bar_data_dict['symbol']}, Time={bar_data_dict['time']}, Close={bar_data_dict['close']}")

    try:
        client.connect()
        if client.run_async_event_loop_if_needed(): # Start event loop for callbacks
            logger.info("Event loop is running.")

            # 1. Qualify a contract (e.g., AAPL stock)
            aapl_contract_unqualified = Stock('AAPL', 'SMART', 'USD')
            aapl_contract_qualified = client.qualify_contract(aapl_contract_unqualified)

            if aapl_contract_qualified:
                logger.info(f"AAPL Contract Qualified: {aapl_contract_qualified}")

                # 2. Fetch some historical data for AAPL
                # hist_data_df = client.fetch_historical_data(
                #     aapl_contract_qualified,
                #     endDateTime='', # Current time
                #     durationStr='5 D',
                #     barSizeSetting='1 hour',
                #     whatToShow='TRADES',
                #     useRTH=True
                # )
                # if not hist_data_df.empty:
                #     logger.info(f"Historical Data for AAPL (last 5 rows):\n{hist_data_df.tail()}")

                # 3. Subscribe to real-time bars for AAPL (will run for ~20 seconds)
                logger.info("Subscribing to AAPL real-time bars...")
                subscription = client.subscribe_realtime_bars(aapl_contract_qualified, my_simple_bar_handler)
                
                if subscription:
                    logger.info("Subscription active. Waiting for bar data for ~20 seconds...")
                    client.ib.sleep(20) # Keep script alive to receive bars
                    
                    logger.info("Attempting to cancel AAPL real-time bar subscription...")
                    client.cancel_realtime_bars(subscription) # Or client.cancel_realtime_bars(aapl_contract_qualified)
                    logger.info("AAPL real-time bar subscription cancelled.")
                else:
                    logger.error("Failed to subscribe to AAPL real-time bars.")
            else:
                logger.error("Failed to qualify AAPL contract.")
            
            # Example for Forex
            # eurusd_contract_unq = Forex('EURUSD')
            # eurusd_contract_q = client.qualify_contract(eurusd_contract_unq)
            # if eurusd_contract_q:
            #    logger.info(f"EURUSD Contract Qualified: {eurusd_contract_q}")
            #    # client.subscribe_realtime_bars(eurusd_contract_q, my_simple_bar_handler)
            #    # client.ib.sleep(20)
            #    # client.cancel_realtime_bars(eurusd_contract_q)

        else:
            logger.error("Failed to start or patch the event loop. Real-time updates will not work.")

    except ConnectionRefusedError:
        logger.error("Connection to IBKR refused. Is TWS/Gateway running and API enabled on the correct port?")
    except Exception as main_e:
        logger.error(f"An error occurred in the main example: {main_e}", exc_info=True)
    finally:
        if client.ib.isConnected():
            logger.info("Disconnecting client in finally block...")
            client.disconnect()
            # If util.startLoop was used, the daemon thread should exit eventually after disconnect.
            # If ib.run() was used directly and blocking, ib.stop() would be needed here.
        logger.info("IBKRClient example finished.")
    # This is a simple example of how to use the IBKRClient
    # Replace with your actual connection details and desired contract/order

    # Configuration (ideally from a config file)
    IBKR_HOST = '127.0.0.1'
    IBKR_PORT = 7497  # 7497 for TWS Paper, 4002 for Gateway Paper
    IBKR_CLIENT_ID = 1

    client = IBKRClient(host=IBKR_HOST, port=IBKR_PORT, clientId=IBKR_CLIENT_ID)

    try:
        client.connect()

        # Example 1: Fetch historical data for AAPL stock
        # aapl_stock_contract_details = Stock(symbol='AAPL', exchange='SMART', currency='USD')
        # Note: For live examples, ensure TWS/Gateway is running and connected.
        # And that you are connected to a Paper account for testing orders.

        # qualified_aapl_contract = client.qualify_contract(aapl_stock_contract_details)
        # if qualified_aapl_contract:
        #     historical_data_df = client.fetch_historical_data(
        #         contract=qualified_aapl_contract,
        #         endDateTime='',  # Empty for current time
        #         durationStr='30 D',
        #         barSizeSetting='1 hour',
        #         whatToShow='TRADES',
        #         useRTH=True
        #     )
        #     if not historical_data_df.empty:
        #         print("\nHistorical Data for AAPL:")
        #         print(historical_data_df.head())

        # Example 2: Place a paper order for a stock
        # tsla_stock_details = Stock(symbol='TSLA', exchange='SMART', currency='USD')
        # qualified_tsla_contract = client.qualify_contract(tsla_stock_details)
        # if qualified_tsla_contract:
        #     market_order = MarketOrder(action='BUY', totalQuantity=1) # Buy 1 share of TSLA at market price
        #     # paper_trade = client.place_paper_order(qualified_tsla_contract, market_order)
        #     # if paper_trade:
        #     #     print(f"\nPaper trade placed for TSLA: {paper_trade}")
        #     #     print(f"Order Status: {paper_trade.orderStatus.status}")
        #     #     # client.ib.sleep(5) # Wait for potential status updates
        #     #     # print(f"Updated Order Status: {paper_trade.orderStatus.status}")
        # else:
        #     logging.error("Could not place order for TSLA, contract qualification failed.")


        # Example 3: Fetch historical data for EUR.USD Forex pair
        # eurusd_forex_details = Forex('EURUSD') # Forex contracts are often simpler to qualify
        # qualified_eurusd_contract = client.qualify_contract(eurusd_forex_details) # Should find 'IDEALPRO'
        # if qualified_eurusd_contract:
        #     historical_forex_data_df = client.fetch_historical_data(
        #         contract=qualified_eurusd_contract,
        #         endDateTime='',
        #         durationStr='1 M', # 1 Month
        #         barSizeSetting='4 hours',
        #         whatToShow='MIDPOINT', # or ASK, BID
        #         useRTH=True
        #     )
        #     if not historical_forex_data_df.empty:
        #         print("\nHistorical Data for EUR.USD:")
        #         print(historical_forex_data_df.head())
        # else:
        #    logging.error("Could not get historical data for EURUSD, contract qualification failed.")
        
        # Example 4: Subscribe to real-time bars for a stock (e.g., MSFT)
        # This part requires the event loop to be running.
        # def my_bar_update_handler(bar_dict, contract_object):
        #    # Note: bar_dict now contains 'symbol' and 'conId'
        #    print(f"Real-time bar for {bar_dict['symbol']} (ConID: {bar_dict['conId']}): "
        #          f"{bar_dict['time']} - O: {bar_dict['open']} H: {bar_dict['high']} L: {bar_dict['low']} C: {bar_dict['close']} V: {bar_dict['volume']}")

        # msft_stock_details = Stock(symbol='MSFT', exchange='SMART', currency='USD')
        
        # client.connect() # Ensure connected before starting loop management
        # loop_running = client.run_async_event_loop_if_needed()

        # if loop_running and client.ib.isConnected():
        #    qualified_msft_contract = client.qualify_contract(msft_stock_details)
        #    if qualified_msft_contract:
        #        logging.info(f"Attempting to subscribe to MSFT real-time bars...")
        #        active_subscription = client.subscribe_realtime_bars(qualified_msft_contract, my_bar_update_handler, barSize=5, whatToShow="TRADES", useRTH=True)
        #        if active_subscription:
        #            logging.info("Successfully initiated MSFT real-time bar subscription. Running for 30 seconds...")
        #            client.ib.sleep(30) # Keep the connection alive and processing events for 30s
        #            logging.info("Finished 30s real-time bar test. Cancelling subscription.")
        #            client.cancel_realtime_bars(active_subscription) 
        #            logging.info("MSFT real-time subscription cancelled.")
        #        else:
        #            logging.error("Failed to subscribe to MSFT real-time bars.")
        #    else:
        #        logging.error("Could not subscribe to MSFT real-time bars, contract qualification failed.")
        # else:
        #    logging.error("IBKR Client not connected or event loop not running. Cannot run real-time example.")


    except ConnectionRefusedError:
        logging.error("Connection refused. Ensure TWS/Gateway is running and API connections are enabled.")
    except Exception as e:
        logging.error(f"An error occurred in the main example: {e}", exc_info=True)
    finally:
        logging.info("Disconnecting client in finally block of __main__...")
        if 'client' in locals() and client.ib.isConnected():
             # If util.startLoop was used, it runs a daemon thread.
             # Explicitly stopping the IB client loop may or may not be needed/possible
             # depending on how it was started and if ib_insync handles it on disconnect.
             # For util.startLoop, ib.disconnect() should be enough.
             client.disconnect() # This will also cancel active subscriptions.
        
        # If a loop was started with ib.run() directly (not in a thread), you'd need ib.stop()
        # if 'client' in locals() and client.ib.loop.is_running() and not util.isBackgroundLoop():
        #    logging.info("Attempting to stop foreground IB event loop.")
        #    client.ib.stop()

        logging.info("IBKRClient example finished.")
