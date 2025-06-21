import logging
from ib_insync import IB, util, Contract # Make sure Contract is directly importable
from ib_insync.contract import Stock, Forex, Option # Option class for defining options
from ib_insync.order import MarketOrder, LimitOrder, Order # Order class for type hinting
from ib_insync.objects import BarDataList, BarData, Trade # For type hinting
import pandas as pd
from typing import List, Optional, Union # For type hinting
from datetime import datetime, date # For date operations

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

    def qualify_contract(self, contract: Contract) -> Optional[Contract]:
        """
        Qualifies a contract to resolve ambiguities and fill in details like conId.
        IBKR requires contracts to be qualified before use in most API calls.

        Args:
            contract (Contract): The ib_insync Contract object to qualify (e.g., Stock, Forex, Option).

        Returns:
            Optional[Contract]: The qualified contract object if successful, otherwise None.
        
        Raises:
            Exception: If there's an error during the qualification process from IBKR.
        """
        self.logger.debug(f"Qualifying contract: {contract}")
        try:
            self.connect() # Ensure connection

            # For options, it's good practice to set a default exchange if not provided, e.g., 'SMART'
            if isinstance(contract, Option) and not contract.exchange:
                contract.exchange = 'SMART' # Or specific options exchange like 'CBOE', 'BOX' etc.
                self.logger.debug(f"Set default exchange 'SMART' for Option contract: {contract.localSymbol if contract.localSymbol else contract.symbol}")

            qualified_contracts = self.ib.qualifyContracts(contract)

            if qualified_contracts:
                if len(qualified_contracts) == 1:
                    self.logger.info(f"Contract qualified successfully: {qualified_contracts[0]}")
                    return qualified_contracts[0]
                else:
                    # Handle ambiguity
                    self.logger.warning(f"Contract {contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol} is ambiguous. Found: {qualified_contracts}")
                    # If it's an option and multiple matches, it might be due to different primaryExch, multipliers, etc.
                    # A common strategy for options is to pick the one with the standard multiplier (usually 100) if SMART.
                    if isinstance(contract, Option):
                        for qc in qualified_contracts:
                            if hasattr(qc, 'multiplier') and qc.multiplier == '100' and qc.exchange == 'SMART': # Multiplier is string
                                self.logger.info(f"Returning SMART option with multiplier 100 for ambiguous query: {qc}")
                                return qc
                    elif hasattr(contract, 'exchange') and contract.exchange and contract.exchange.upper() == "SMART":
                        for qc in qualified_contracts: # Fallback for non-options or if option specific logic failed
                            if qc.exchange == "SMART":
                                self.logger.info(f"Returning first SMART contract for ambiguous query: {qc}")
                                return qc

                    self.logger.warning(f"Ambiguous contract {contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol}. Returning the first from list as a fallback: {qualified_contracts[0]}")
                    return qualified_contracts[0] # Fallback to first, user should be specific.
            else:
                self.logger.error(f"Could not qualify contract: {contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol}. No matching contract found by IBKR.")
                return None
        except Exception as e:
            self.logger.error(f"Error qualifying contract {contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol}: {e}", exc_info=True)
            raise # Re-raise to signal failure to caller

    def fetch_historical_data(self, contract: Contract, endDateTime: str, durationStr: str,
                              barSizeSetting: str, whatToShow: str, useRTH: bool,
                              formatDate: int = 1, keepUpToDate: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetches historical bar data from IBKR.

        Args:
            contract (Contract): The ib_insync Contract object (should be qualified if possible,
                                 but this method will attempt to qualify it if not).
            endDateTime (str): The end date/time of the historical data request.
                               Format: 'yyyyMMdd HH:mm:ss [zzz]' or '' for current time.
            durationStr (str): The duration of the data request (e.g., '30 D', '1 M', '1 Y').
            barSizeSetting (str): The bar size (e.g., '1 min', '1 hour', '1 day').
            whatToShow (str): The type of data to show (e.g., 'TRADES', 'MIDPOINT', 'BID', 'ASK', 'OPTION_IMPLIED_VOLATILITY').
            useRTH (bool): If True, data is returned for regular trading hours only.
            formatDate (int): Date formatting for IBKR (1 for yyyyMMdd HH:mm:ss, 2 for system time zone).
            keepUpToDate (bool): If True, continuously update with new bars (primarily for live data, less common for historical fetches).

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the historical bar data, or None if no data or error.
        
        Raises:
            Exception: If there's an error during data fetching from IBKR that is not handled internally.
        """
        contract_identifier = contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol
        self.logger.info(f"Fetching historical data for {contract_identifier}: "
                         f"End: {endDateTime}, Duration: {durationStr}, Bar: {barSizeSetting}, WhatToShow: {whatToShow}")
        try:
            self.connect()
            
            # Attempt to qualify the contract if it doesn't have a conId yet.
            # For already qualified contracts (e.g., from option chain fetching), this will be quick.
            if not contract.conId:
                self.logger.debug(f"Contract {contract_identifier} does not have conId, attempting to qualify.")
                qualified_contract = self.qualify_contract(contract)
                if not qualified_contract:
                    self.logger.warning(f"Historical data fetch failed for {contract_identifier} due to contract qualification failure.")
                    return None # Return None instead of empty DataFrame for clarity
            else:
                qualified_contract = contract # Assume it's already qualified

            bars = self.ib.reqHistoricalData(
                qualified_contract, # Use the qualified contract
                endDateTime=endDateTime,
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate,
                keepUpToDate=keepUpToDate
            )
            if bars:
                df = util.df(bars) # Converts list of BarData to DataFrame
                self.logger.info(f"Fetched {len(bars)} bars for {qualified_contract.localSymbol if qualified_contract.localSymbol else qualified_contract.symbol}.")
                if df is not None and 'date' in df.columns: # util.df might return None if bars is empty or malformed
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date']) # pragma: no cover
                    df.set_index('date', inplace=True) # Standardize to have 'date' as index
                return df
            else:
                self.logger.warning(f"No historical data returned for {qualified_contract.localSymbol if qualified_contract.localSymbol else qualified_contract.symbol}.")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {contract_identifier}: {e}", exc_info=True)
            raise # Re-raise to signal failure to caller

    def place_paper_order(self, contract: Contract, order: Order) -> Optional[Trade]:
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
        Places a paper trading order.
        Note: Ensure you are connected to a paper trading account in TWS/Gateway.

        Args:
            contract (Contract): The ib_insync Contract object for the instrument.
                                 Should be qualified before calling this method.
            order (Order): The ib_insync Order object (e.g., MarketOrder, LimitOrder).

        Returns:
            Optional[Trade]: The ib_insync Trade object if order placement was initiated, otherwise None.
                             The Trade object tracks the order's status and fills.

        Raises:
            Exception: If there's an error during order placement from IBKR.
        """
        contract_identifier = contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol
        self.logger.info(f"Attempting to place paper order for {contract_identifier}: "
                         f"{order.action} {order.totalQuantity} units.")
        try:
            self.connect()

            # Qualification should ideally happen before calling place_order,
            # but as a safeguard, ensure it has conId.
            if not contract.conId:
                q_contract = self.qualify_contract(contract)
                if not q_contract:
                    self.logger.error(f"Cannot place order for {contract_identifier}, contract qualification failed.")
                    return None
                final_contract = q_contract
            else:
                final_contract = contract


            # Optionally, specify paper trading account if managing multiple accounts
            # if self.ib.managedAccounts():
            #     order.account = self.ib.managedAccounts()[0] # Or a specific paper account

            trade = self.ib.placeOrder(final_contract, order) # Use the final_contract
            self.logger.info(f"Paper order placed for {final_contract.localSymbol if final_contract.localSymbol else final_contract.symbol}. OrderId: {trade.order.orderId}, "
                             f"PermId: {trade.order.permId}, Status: {trade.orderStatus.status}")
            
            # For more advanced handling, you can register callbacks for order status changes:
            # trade.filledEvent += self._on_order_filled_callback
            # trade.statusEvent += self._on_order_status_callback
            
            return trade
        except Exception as e:
            self.logger.error(f"Error placing paper order for {contract_identifier}: {e}", exc_info=True)
            raise

    def subscribe_realtime_bars(self, contract: Contract, callback_function,
                                barSize: int = 5, whatToShow: str = 'TRADES', useRTH: bool = True) -> Optional[BarDataList]:
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
        contract_identifier = contract.localSymbol if hasattr(contract, 'localSymbol') and contract.localSymbol else contract.symbol
        self.logger.info(f"Attempting to subscribe to real-time bars for {contract_identifier}...")
        try:
            self.connect()
            
            # Ensure contract is qualified
            if not contract.conId:
                q_contract = self.qualify_contract(contract)
                if not q_contract:
                    self.logger.error(f"Cannot subscribe to real-time bars for {contract_identifier}, contract qualification failed.")
                    return None
                final_contract = q_contract
            else:
                final_contract = contract

            con_id = final_contract.conId
            if con_id in self._active_realtime_bars:
                self.logger.warning(f"Already subscribed to real-time bars for {final_contract.localSymbol if final_contract.localSymbol else final_contract.symbol} (ConID: {con_id}). Returning existing subscription object.")
                return self._active_realtime_bars[con_id]['bars']

            self.logger.info(f"Subscribing to real-time bars for {final_contract.localSymbol if final_contract.localSymbol else final_contract.symbol} (ConID: {con_id}), BarSize: {barSize}s, WhatToShow: {whatToShow}, UseRTH: {useRTH}")
            
            bars_obj = self.ib.reqRealTimeBars( # Renamed variable to avoid conflict with 'bars' in _on_bar_update
                contract=final_contract,
                barSize=barSize,
                whatToShow=whatToShow,
                useRTH=useRTH
            )
            
            self._active_realtime_bars[con_id] = {
                'contract': final_contract,
                'bars': bars_obj, # Store the BarDataList object
                'callback': callback_function
            }
            bars_obj.updateEvent += self._on_bar_update # Register internal handler

            self.logger.info(f"Successfully subscribed to real-time bars for {final_contract.localSymbol if final_contract.localSymbol else final_contract.symbol}. Listening for updates...")
            return bars_obj
            
        except Exception as e:
            self.logger.error(f"Error subscribing to real-time bars for {contract_identifier}: {e}", exc_info=True)
            if 'final_contract' in locals() and final_contract and final_contract.conId in self._active_realtime_bars:
                 del self._active_realtime_bars[final_contract.conId]
            raise

    def _on_bar_update(self, updated_bars: BarDataList, hasNewBar: bool): # Parameter name changed from 'bars' to 'updated_bars'
        """
        Internal callback triggered by ib_insync when new bar data is received
        for any real-time bar subscription. It then calls the user-provided callback.

        Args:
            bars (ib_insync.util.BarDataList): The BarDataList object that was updated.
                                               Contains all bars for that subscription, newest is bars[-1].
            hasNewBar (bool): True if bars[-1] is a new bar, False if it's an update to the last bar.
        """
        if hasNewBar:
            # Find the subscription details (contract, user_callback) associated with this 'updated_bars' object
            found_sub_info = None
            for con_id_iter, sub_info_iter in self._active_realtime_bars.items():
                if sub_info_iter['bars'] is updated_bars: # Check if it's the same BarDataList object
                    found_sub_info = sub_info_iter
                    break
            
            if found_sub_info:
                latest_bar: BarData = updated_bars[-1] # The newest bar data
                contract_info: Contract = found_sub_info['contract'] # This is the qualified contract
                user_callback = found_sub_info['callback']
                
                contract_identifier = contract_info.localSymbol if contract_info.localSymbol else contract_info.symbol
                self.logger.debug(f"New bar for {contract_identifier}: Time={latest_bar.time}, Close={latest_bar.close}")
                
                try:
                    # Prepare a dictionary representation of the bar for the user callback
                    bar_data_dict = {
                        "time": latest_bar.time,
                        "open": latest_bar.open_, # Note: open_ (with underscore) for BarData
                        "high": latest_bar.high,
                        "low": latest_bar.low,
                        "close": latest_bar.close,
                        "volume": latest_bar.volume,
                        "wap": latest_bar.wap,
                        "count": latest_bar.count,
                        "symbol": contract_identifier,
                        "conId": contract_info.conId
                    }
                    user_callback(bar_data_dict, contract_info) # Pass the qualified contract_info
                except Exception as e:
                    self.logger.error(f"Error executing user-provided callback for {contract_identifier}: {e}", exc_info=True)
            else:
                self.logger.warning(f"Received bar update for an untracked subscription. Bar time: {updated_bars[-1].time if updated_bars and updated_bars[-1] else 'N/A'}")


    def cancel_realtime_bars(self, contract_or_bars_object: Union[Contract, BarDataList]):
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
                # If it's a contract object, it should ideally be qualified or have conId.
                # If not, qualify it to get conId.
                final_contract_for_cancel = contract_or_bars_object
                if not final_contract_for_cancel.conId:
                    self.logger.debug(f"Contract for cancellation needs qualification: {final_contract_for_cancel}")
                    final_contract_for_cancel = self.qualify_contract(contract_or_bars_object)

                if final_contract_for_cancel and final_contract_for_cancel.conId:
                    con_id_to_cancel = final_contract_for_cancel.conId
                    target_symbol_for_log = final_contract_for_cancel.localSymbol or final_contract_for_cancel.symbol
                    if con_id_to_cancel in self._active_realtime_bars:
                        bars_obj_to_cancel = self._active_realtime_bars[con_id_to_cancel]['bars']
                    else:
                        self.logger.warning(f"No active subscription found for conId {con_id_to_cancel} ({target_symbol_for_log}) from the provided contract to cancel.")
                        return
                else:
                    self.logger.error(f"Could not determine conId for cancellation from contract: {contract_or_bars_object}")
                    return
            elif isinstance(contract_or_bars_object, BarDataList):
                bars_obj_to_cancel = contract_or_bars_object
                if hasattr(bars_obj_to_cancel, 'contract') and bars_obj_to_cancel.contract and bars_obj_to_cancel.contract.conId:
                    con_id_to_cancel = bars_obj_to_cancel.contract.conId
                    target_symbol_for_log = bars_obj_to_cancel.contract.localSymbol or bars_obj_to_cancel.contract.symbol
                else:
                    self.logger.error("Invalid BarDataList object for cancellation: missing .contract or conId.")
                    return
            else:
                self.logger.error(f"Invalid argument type for cancel_realtime_bars: {type(contract_or_bars_object)}. Must be Contract or BarDataList.")
                return

            if con_id_to_cancel and con_id_to_cancel in self._active_realtime_bars:
                if not bars_obj_to_cancel: # Should be set if con_id_to_cancel was found via Contract object
                     bars_obj_to_cancel = self._active_realtime_bars[con_id_to_cancel]['bars']
                
                self.logger.info(f"Cancelling real-time bars subscription for {target_symbol_for_log} (ConID: {con_id_to_cancel}).")
                
                try:
                    bars_obj_to_cancel.updateEvent -= self._on_bar_update
                except Exception as event_err:
                    self.logger.warning(f"Could not de-register _on_bar_update for {target_symbol_for_log} (ConID: {con_id_to_cancel}). Error: {event_err}")

                self.ib.cancelRealTimeBars(bars_obj_to_cancel)
                del self._active_realtime_bars[con_id_to_cancel]
                self.logger.info(f"Successfully cancelled real-time bars for {target_symbol_for_log}.")
            elif con_id_to_cancel:
                self.logger.warning(f"No active real-time subscription found in internal tracking for {target_symbol_for_log} (ConID: {con_id_to_cancel}) to cancel.")
            # If con_id_to_cancel was never found, already handled.

        except Exception as e:
            self.logger.error(f"Error cancelling real-time bars for {target_symbol_for_log}: {e}", exc_info=True)
            raise

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

    # --- Methods for Options Trading ---

    def fetch_option_chain(self, underlying_symbol: str, expiration_date: Optional[str] = None, underlying_con_id: Optional[int] = None) -> List[Contract]:
        """
        Fetches the option chain for a given underlying symbol and optionally an expiration date.

        Args:
            underlying_symbol (str): The symbol of the underlying asset (e.g., 'QQQ').
            expiration_date (Optional[str]): Specific expiration date in 'YYYYMMDD' format.
                                             If None, fetches for all available expirations.
            underlying_con_id (Optional[int]): The conId of the underlying asset. If not provided,
                                               it will be fetched.

        Returns:
            List[Contract]: A list of qualified ib_insync.Option contracts.
        """
        self.logger.info(f"Fetching option chain for {underlying_symbol}, Expiration: {expiration_date or 'All'}")
        try:
            self.connect()

            if not underlying_con_id:
                # First, get the conId of the underlying stock/ETF
                underlying_contract = Stock(underlying_symbol, 'SMART', 'USD')
                qualified_underlying = self.qualify_contract(underlying_contract)
                if not qualified_underlying or not qualified_underlying.conId:
                    self.logger.error(f"Could not qualify underlying symbol {underlying_symbol} to get conId.")
                    return []
                underlying_con_id = qualified_underlying.conId
                self.logger.info(f"Underlying {underlying_symbol} conId: {underlying_con_id}")

            # Fetch option chain details
            # reqSecDefOptParams returns a list of OptionChain objects
            chains = self.ib.reqSecDefOptParams(underlyingSymbol=underlying_symbol, futFopExchange='', underlyingSecType='STK', underlyingConId=underlying_con_id)

            if not chains:
                self.logger.warning(f"No option chain data returned for {underlying_symbol} with conId {underlying_con_id}.")
                return []

            option_contracts = []
            # Iterate through exchanges and expirations if needed
            # For simplicity, assuming one primary exchange or SMART handles it.
            # The chains object contains a list of exchanges, each with expirations and strikes.
            # Example: chains[0].expirations, chains[0].strikes

            target_expirations = []
            if expiration_date: # Specific expiration
                target_expirations.append(expiration_date)
            else: # All expirations found for the primary exchange in the chain
                if chains[0].expirations: # chains[0] is usually the primary/SMART
                    target_expirations.extend(chains[0].expirations)

            if not target_expirations:
                self.logger.warning(f"No expirations found for {underlying_symbol} in the received chain data.")
                return []

            self.logger.debug(f"Target expirations for {underlying_symbol}: {target_expirations}")

            # Get all strikes for the primary exchange in the chain
            # This might fetch a very large number of strikes.
            # For 0DTE, we might want to filter strikes around the current underlying price later.
            all_strikes = chains[0].strikes
            if not all_strikes:
                self.logger.warning(f"No strikes found for {underlying_symbol} in the received chain data.")
                return []

            self.logger.debug(f"Number of strikes found for {underlying_symbol}: {len(all_strikes)}")


            # Create Option contract objects for each combination
            for exp in target_expirations:
                for strike in all_strikes:
                    for right in ['C', 'P']: # Call and Put
                        # Create an unqualified Option contract
                        opt_contract = Option(
                            symbol=underlying_symbol,
                            lastTradeDateOrContractMonth=exp, # Format YYYYMMDD
                            strike=strike,
                            right=right,
                            exchange='SMART', # Use SMART for broad routing
                            multiplier='100', # Standard multiplier
                            currency='USD'
                        )
                        option_contracts.append(opt_contract)

            self.logger.info(f"Generated {len(option_contracts)} potential option contracts for {underlying_symbol} across {len(target_expirations)} expiration(s). Qualifying them now...")

            # Qualify all generated contracts
            # This can be slow if many contracts are generated.
            # Consider qualifying them in batches or as needed.
            qualified_options = []
            for opt_c in option_contracts:
                # reqContractDetails can also be used to find specific options,
                # but qualifying a fully defined Option object is often more direct if parameters are known.
                q_opt = self.qualify_contract(opt_c) # This uses the existing qualify_contract method
                if q_opt and q_opt.conId: # Ensure it's a valid, qualified contract with a conId
                    qualified_options.append(q_opt)
                else:
                    self.logger.debug(f"Failed to qualify option: {opt_c.localSymbol if opt_c.localSymbol else opt_c}")


            self.logger.info(f"Successfully qualified {len(qualified_options)} option contracts for {underlying_symbol}.")
            return qualified_options

        except Exception as e:
            self.logger.error(f"Error fetching option chain for {underlying_symbol}: {e}", exc_info=True)
            return [] # Return empty list on error

    # The main method can be updated to test option chain fetching
    # ... (rest of the class)

if __name__ == '__main__': # pragma: no cover
    # This block is for basic manual testing or demonstration.
    # Ensure TWS or Gateway is running on port 7497 (paper) or 7496 (live) for this to work.

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    client = IBKRClient(host='127.0.0.1', port=7497, clientId=12) # Use a unique clientId

    def my_simple_bar_handler(bar_data_dict, contract_object):
        logger.info(f"BAR HANDLER: Symbol={bar_data_dict['symbol']}, Time={bar_data_dict['time']}, Close={bar_data_dict['close']}")

    try:
        client.connect()
        if client.run_async_event_loop_if_needed():
            logger.info("Event loop is running.")

            # --- Test Option Chain Fetching ---
            underlying_sym = 'QQQ'
            # today_date_str = datetime.now().strftime('%Y%m%d') # For 0DTE, use today's date
            # logger.info(f"Attempting to fetch 0DTE option chain for {underlying_sym} for expiry {today_date_str}")
            # option_chain = client.fetch_option_chain(underlying_sym, expiration_date=today_date_str)

            # For testing, let's fetch a known future expiry if 0DTE might not exist or be too volatile for simple test
            # Find a valid near-term expiration first
            chains_data = client.ib.reqSecDefOptParams(underlyingSymbol=underlying_sym, futFopExchange='', underlyingSecType='STK', underlyingConId=client.qualify_contract(Stock(underlying_sym, 'SMART', 'USD')).conId)
            test_expiry_date = None
            if chains_data and chains_data[0].expirations:
                # Try to find an expiration that is a Friday, not too far out.
                for exp_str in chains_data[0].expirations:
                    exp_dt = datetime.strptime(exp_str, '%Y%m%d').date()
                    if exp_dt.weekday() == 4 and (exp_dt - date.today()).days > 5 and (exp_dt - date.today()).days < 60 : # Friday, 5-60 days out
                        test_expiry_date = exp_str
                        break
                if not test_expiry_date: # Fallback to first available if no suitable Friday found
                    test_expiry_date = chains_data[0].expirations[0]


            if test_expiry_date:
                logger.info(f"Attempting to fetch option chain for {underlying_sym} for specific expiry {test_expiry_date}")
                option_chain = client.fetch_option_chain(underlying_sym, expiration_date=test_expiry_date)

                if option_chain:
                    logger.info(f"Fetched {len(option_chain)} contracts for {underlying_sym} expiring {test_expiry_date}.")
                    # Log details of a few contracts
                    for i, opt_contract in enumerate(option_chain[:3]): # Log first 3
                        logger.info(f"  {i+1}. {opt_contract.localSymbol}, Strike: {opt_contract.strike}, Type: {opt_contract.right}, ConID: {opt_contract.conId}")

                    # Test fetching historical data for one of these options
                    if len(option_chain) > 0:
                        sample_option_contract = option_chain[len(option_chain)//2] # Pick one from middle
                        logger.info(f"Fetching historical data for option: {sample_option_contract.localSymbol}")
                        # Fetch 1 day of 1 min bars for this option for "yesterday" effectively
                        # Note: For options, 'TRADES' might be sparse. 'BID_ASK' or 'MIDPOINT' might be better for liquidity assessment.
                        # However, whatToShow='OPTION_IMPLIED_VOLATILITY' or 'HISTORICAL_VOLATILITY' for options.
                        # For price bars, TRADES, BID, ASK, MIDPOINT are valid.

                        # To get very recent data, endDateTime='' and a short duration e.g., '1 D' or '2 D'
                        # For older data, specify endDateTime.
                        # Let's try to get data for a recent period.
                        hist_opt_data_df = client.fetch_historical_data(
                            sample_option_contract,
                            endDateTime='', # Current time
                            durationStr='1 D', # Last day
                            barSizeSetting='1 min', # 1 minute bars
                            whatToShow='TRADES', # or 'MIDPOINT'
                            useRTH=True
                        )
                        if hist_opt_data_df is not None and not hist_opt_data_df.empty:
                            logger.info(f"Historical Data for Option {sample_option_contract.localSymbol} (first 5 rows):\n{hist_opt_data_df.head()}")
                        else:
                            logger.warning(f"No historical data returned for option {sample_option_contract.localSymbol}.")
                else:
                    logger.warning(f"No option chain contracts found for {underlying_sym} expiring {test_expiry_date}.")
            else:
                logger.error(f"Could not determine a suitable test_expiry_date for {underlying_sym}.")


            # --- Test placing a paper order for an option (if chain was successful) ---
            # Be very careful with live order placement, even paper.
            # This part is commented out by default to prevent accidental order submission.
            """
            if option_chain and len(option_chain) > 0:
                # Select a near-the-money call option for testing
                underlying_price_approx = 350 # Assume QQQ price for selection
                atm_call_options = [
                    c for c in option_chain
                    if c.right == 'C' and abs(c.strike - underlying_price_approx) < 10 # Strike within $10 of assumed ATM
                ]
                if atm_call_options:
                    test_opt_to_order = atm_call_options[0] # Pick the first one
                    logger.info(f"Selected option for paper order test: {test_opt_to_order.localSymbol}")

                    option_order = MarketOrder(action='BUY', totalQuantity=1) # Buy 1 contract

                    # Ensure the contract is fully qualified with conId before placing order
                    # (fetch_option_chain should return qualified contracts with conId)
                    if test_opt_to_order.conId:
                        paper_trade_option = client.place_paper_order(test_opt_to_order, option_order)
                        if paper_trade_option:
                            logger.info(f"Paper order placed for option {test_opt_to_order.localSymbol}: {paper_trade_option}")
                            logger.info(f"Order Status: {paper_trade_option.orderStatus.status}")
                            # client.ib.sleep(5) # Wait for potential status updates
                            # logger.info(f"Updated Order Status: {paper_trade_option.orderStatus.status}")
                        else:
                            logger.error(f"Failed to place paper order for option {test_opt_to_order.localSymbol}.")
                    else:
                        logger.error(f"Test option {test_opt_to_order.localSymbol} does not have conId, cannot place order.")
                else:
                    logger.warning("No suitable ATM call option found in the chain for paper order test.")
            """

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
        logger.info("IBKRClient option example finished.")
