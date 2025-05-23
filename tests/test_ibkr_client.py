import pytest
from unittest.mock import MagicMock, patch, call
import pandas as pd
from ib_insync import Stock, Forex, BarData, Trade, OrderStatus, IB, util # Added IB, util for patching
from ibkr_client import IBKRClient # Assuming ibkr_client.py is in parent directory or PYTHONPATH

@pytest.fixture
def mock_ib_insync():
    """Fixture to mock the ib_insync.IB class."""
    with patch('ibkr_client.IB') as mock_ib:
        instance = mock_ib.return_value
        instance.isConnected.return_value = False # Default to not connected
        
        # Mock qualifyContracts to return a list with one item by default
        mock_contract = Stock('AAPL', 'SMART', 'USD')
        mock_contract.conId = 12345 # Assign a conId for qualified contract
        instance.qualifyContracts.return_value = [mock_contract]
        
        # Mock managedAccounts
        instance.managedAccounts.return_value = ["DU12345"]
        yield instance

@pytest.fixture
def ibkr_client_instance(mock_ib_insync):
    """Fixture to create an IBKRClient instance with a mocked IB."""
    client = IBKRClient(host='127.0.0.1', port=7497, clientId=1)
    # The IBKRClient's __init__ sets self.ib = IB(). So, we need to ensure our mock_ib_insync is used.
    # This is handled by the patch in mock_ib_insync fixture if IBKRClient creates IB() instance inside methods or init.
    # If IBKRClient takes ib_instance as param, then we'd pass mock_ib_insync.
    # Current IBKRClient creates IB() in __init__, so the patch works.
    return client

def test_ibkr_client_init(mock_ib_insync):
    client = IBKRClient(host='test_host', port=1234, clientId=100)
    assert client.host == 'test_host'
    assert client.port == 1234
    assert client.clientId == 100
    assert client.ib == mock_ib_insync # Check if the patched IB instance is used

def test_connect_success(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = False # Ensure it starts as not connected
    ibkr_client_instance.connect()
    mock_ib_insync.connect.assert_called_once_with('127.0.0.1', 7497, 1, timeout=10)

def test_connect_already_connected(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = True
    ibkr_client_instance.connect()
    mock_ib_insync.connect.assert_not_called()

def test_connect_failure(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = False
    mock_ib_insync.connect.side_effect = ConnectionRefusedError("Test connection error")
    with pytest.raises(ConnectionRefusedError):
        ibkr_client_instance.connect()

def test_disconnect_success(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = True
    # Simulate active subscriptions
    mock_bars = MagicMock()
    mock_bars.contract = Stock("MSFT", "SMART", "USD") # Give it a contract attribute
    ibkr_client_instance._active_realtime_bars = {
        5678: {'contract': Stock("MSFT", "SMART", "USD"), 'bars': mock_bars}
    }
    ibkr_client_instance.disconnect()
    mock_ib_insync.cancelRealTimeBars.assert_called_once_with(mock_bars)
    mock_ib_insync.disconnect.assert_called_once()
    assert not ibkr_client_instance._active_realtime_bars # Check if cleared

def test_disconnect_not_connected(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = False
    ibkr_client_instance.disconnect()
    mock_ib_insync.disconnect.assert_not_called()

def test_qualify_contract_success(ibkr_client_instance, mock_ib_insync):
    test_contract = Stock('AAPL', 'SMART', 'USD')
    qualified_contract = ibkr_client_instance.qualify_contract(test_contract)
    mock_ib_insync.qualifyContracts.assert_called_once_with(test_contract)
    assert qualified_contract is not None
    assert qualified_contract.conId == 12345 # From mock_ib_insync fixture

def test_qualify_contract_not_found(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.qualifyContracts.return_value = [] # Simulate contract not found
    test_contract = Stock('UNKNOWN', 'SMART', 'USD')
    qualified_contract = ibkr_client_instance.qualify_contract(test_contract)
    assert qualified_contract is None

def test_qualify_contract_ambiguous(ibkr_client_instance, mock_ib_insync):
    # Simulate multiple contracts returned, but one is on SMART
    mock_contract_smart = Stock('AMZN', 'SMART', 'USD')
    mock_contract_smart.conId = 6000
    mock_contract_other = Stock('AMZN', 'OTHEREX', 'USD')
    mock_contract_other.conId = 6001
    mock_ib_insync.qualifyContracts.return_value = [mock_contract_smart, mock_contract_other]
    
    test_contract_amb = Stock('AMZN', 'SMART', 'USD') # User prefers SMART
    qualified_contract = ibkr_client_instance.qualify_contract(test_contract_amb)
    assert qualified_contract is not None
    assert qualified_contract.conId == 6000 # Should pick the SMART one

    test_contract_amb_non_smart = Stock('AMZN', 'NYSE', 'USD') # User specified non-SMART, but SMART is an option
    qualified_contract_non_smart = ibkr_client_instance.qualify_contract(test_contract_amb_non_smart)
    # Current logic might still pick SMART if it's first and user specified SMART.
    # If user specifies OTHEREX and SMART is also found, current logic returns first if contract.exchange was SMART.
    # This test depends on exact implementation details of ambiguity resolution.
    # The current code: if contract.exchange == "SMART": return qualified_contracts[0]
    # So, if user asks for AMZN on NYSE, but SMART is also an option, it depends on the order.
    # Let's make the user's preference non-SMART and see it fails as per current strict logic.
    mock_ib_insync.qualifyContracts.return_value = [mock_contract_other, mock_contract_smart] # Other is first
    test_contract_amb_other_pref = Stock('AMZN', 'OTHEREX', 'USD')
    qualified_contract_other_pref = ibkr_client_instance.qualify_contract(test_contract_amb_other_pref)
    assert qualified_contract_other_pref is not None # As OTHEREX is first
    assert qualified_contract_other_pref.conId == 6001


def test_fetch_historical_data_success(ibkr_client_instance, mock_ib_insync):
    # Prepare mock BarData objects
    bars_list = [
        BarData(date='20230101', open=100, high=102, low=99, close=101, volume=1000, barCount=100, average=100.5),
        BarData(date='20230102', open=101, high=103, low=100, close=102, volume=1200, barCount=120, average=101.5)
    ]
    mock_ib_insync.reqHistoricalData.return_value = bars_list
    
    # Mock util.df to behave as expected
    with patch('ibkr_client.util.df') as mock_util_df:
        expected_df = pd.DataFrame([{'date':'20230101', 'open':100, 'close':101}, {'date':'20230102', 'open':101, 'close':102}])
        mock_util_df.return_value = expected_df

        contract = Stock('AAPL', 'SMART', 'USD')
        df = ibkr_client_instance.fetch_historical_data(contract, '', '1 M', '1 day', 'TRADES', True)
        
        mock_ib_insync.reqHistoricalData.assert_called_once()
        mock_util_df.assert_called_once_with(bars_list)
        assert not df.empty
        assert df.shape == (2,3)
        pd.testing.assert_frame_equal(df, expected_df)

def test_fetch_historical_data_no_data(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.reqHistoricalData.return_value = [] # Simulate no data
    with patch('ibkr_client.util.df') as mock_util_df: # To prevent error if it's called with []
        mock_util_df.return_value = pd.DataFrame() 
        contract = Stock('AAPL', 'SMART', 'USD')
        df = ibkr_client_instance.fetch_historical_data(contract, '', '1 M', '1 day', 'TRADES', True)
        assert df.empty

def test_place_paper_order_success(ibkr_client_instance, mock_ib_insync):
    contract = Stock('TSLA', 'SMART', 'USD')
    order = MagicMock() # Mock the order object itself
    order.action = "BUY"
    order.totalQuantity = 10
    
    # Mock the trade object returned by placeOrder
    mock_trade = MagicMock(spec=Trade)
    mock_trade.order = order
    mock_trade.orderStatus = OrderStatus(status='Submitted', permId=98765) # Mock orderStatus
    mock_ib_insync.placeOrder.return_value = mock_trade

    trade_result = ibkr_client_instance.place_paper_order(contract, order)
    
    # The client qualifies contract first.
    # mock_ib_insync.qualifyContracts called with contract
    # then placeOrder called with qualified_contract and order.
    # The qualified contract is mock_ib_insync.qualifyContracts.return_value[0]
    qualified_contract_mock = mock_ib_insync.qualifyContracts.return_value[0]
    mock_ib_insync.placeOrder.assert_called_once_with(qualified_contract_mock, order)
    assert trade_result == mock_trade

# --- Tests for Real-Time Bars ---
@pytest.fixture
def mock_callback():
    return MagicMock()

def test_subscribe_realtime_bars_success(ibkr_client_instance, mock_ib_insync, mock_callback):
    contract_to_sub = Stock('MSFT', 'SMART', 'USD')
    
    # Mock reqRealTimeBars to return a mock BarDataList object
    mock_bar_data_list = MagicMock()
    # The BarDataList object itself should have the qualified contract
    # The qualified contract is from mock_ib_insync.qualifyContracts.return_value[0]
    qualified_contract_for_sub = mock_ib_insync.qualifyContracts.return_value[0]
    mock_bar_data_list.contract = qualified_contract_for_sub

    mock_ib_insync.reqRealTimeBars.return_value = mock_bar_data_list

    bars_subscription = ibkr_client_instance.subscribe_realtime_bars(contract_to_sub, mock_callback)
    
    mock_ib_insync.reqRealTimeBars.assert_called_once_with(
        contract=qualified_contract_for_sub, # Uses the qualified contract
        barSize=5,
        whatToShow='TRADES',
        useRTH=True
    )
    assert bars_subscription == mock_bar_data_list
    # Check if callback is registered (indirectly by checking if it's stored)
    con_id = qualified_contract_for_sub.conId
    assert con_id in ibkr_client_instance._active_realtime_bars
    assert ibkr_client_instance._active_realtime_bars[con_id]['callback'] == mock_callback
    # Check if the internal handler is added to updateEvent
    mock_bar_data_list.updateEvent.__iadd__.assert_called_once_with(ibkr_client_instance._on_bar_update)


def test_on_bar_update_calls_user_callback(ibkr_client_instance, mock_ib_insync, mock_callback):
    contract = Stock('MSFT', 'SMART', 'USD')
    contract.conId = 5678 # Ensure conId for tracking
    
    mock_bar_data_list = MagicMock() # This is the BarDataList object
    mock_bar_data_list.contract = contract # Attach contract to it

    # Simulate that this subscription is active
    ibkr_client_instance._active_realtime_bars[contract.conId] = {
        'contract': contract,
        'bars': mock_bar_data_list, # The BarDataList object
        'callback': mock_callback
    }
    
    # Simulate a new bar arriving
    # The 'bars' argument to _on_bar_update is the BarDataList itself.
    # The BarDataList contains a list of actual BarData objects.
    latest_bar_data = BarData(time=pd.Timestamp.now(), open=10, high=12, low=9, close=11, volume=1000, wap=10.5, count=50)
    
    # When updateEvent fires, it passes the BarDataList and hasNewBar.
    # The BarDataList object itself is what we stored as 'bars' in _active_realtime_bars.
    # We need to make sure that when mock_bar_data_list is passed to _on_bar_update,
    # it can retrieve its [-1] element.
    type(mock_bar_data_list)._getitem_ = MagicMock(return_value=latest_bar_data) # Mocking __getitem__
    # Make the mock_bar_data_list iterable and able to be indexed like a list
    def mock_getitem(key):
        if key == -1: return latest_bar_data
        raise IndexError
    mock_bar_data_list.__getitem__ = mock_getitem


    ibkr_client_instance._on_bar_update(mock_bar_data_list, True) # True for hasNewBar
    
    expected_bar_dict = {
        "time": latest_bar_data.time, "open": latest_bar_data.open_,
        "high": latest_bar_data.high, "low": latest_bar_data.low,
        "close": latest_bar_data.close, "volume": latest_bar_data.volume,
        "wap": latest_bar_data.wap, "count": latest_bar_data.count,
        "symbol": contract.symbol, "conId": contract.conId
    }
    mock_callback.assert_called_once_with(expected_bar_dict, contract)


def test_cancel_realtime_bars_by_contract(ibkr_client_instance, mock_ib_insync):
    contract_to_cancel = Stock('MSFT', 'SMART', 'USD')
    # Assume it was qualified and has conId from the mock_ib_insync default
    qualified_contract = mock_ib_insync.qualifyContracts.return_value[0] 
    qualified_contract.symbol = 'MSFT' # Align symbol for clarity
    
    mock_bar_obj = MagicMock() # The BarDataList object
    mock_bar_obj.contract = qualified_contract
    mock_bar_obj.updateEvent = MagicMock() # Mock the updateEvent part

    ibkr_client_instance._active_realtime_bars[qualified_contract.conId] = {
        'contract': qualified_contract, 'bars': mock_bar_obj, 'callback': MagicMock()
    }
    
    ibkr_client_instance.cancel_realtime_bars(contract_to_cancel) # Pass original, it will be qualified
    
    mock_ib_insync.cancelRealTimeBars.assert_called_once_with(mock_bar_obj)
    mock_bar_obj.updateEvent.__isub__.assert_called_once_with(ibkr_client_instance._on_bar_update)
    assert qualified_contract.conId not in ibkr_client_instance._active_realtime_bars

def test_cancel_realtime_bars_by_bars_object(ibkr_client_instance, mock_ib_insync):
    qualified_contract = Stock('MSFT', 'SMART', 'USD')
    qualified_contract.conId = 12345 # Assume this conId
    
    mock_bar_obj_to_cancel = MagicMock(spec=util.BarDataList) # Simulate BarDataList
    mock_bar_obj_to_cancel.contract = qualified_contract # BarDataList has a contract attribute
    mock_bar_obj_to_cancel.updateEvent = MagicMock()


    ibkr_client_instance._active_realtime_bars[qualified_contract.conId] = {
        'contract': qualified_contract, 'bars': mock_bar_obj_to_cancel, 'callback': MagicMock()
    }
    
    ibkr_client_instance.cancel_realtime_bars(mock_bar_obj_to_cancel) # Pass the bars object directly
    
    mock_ib_insync.cancelRealTimeBars.assert_called_once_with(mock_bar_obj_to_cancel)
    mock_bar_obj_to_cancel.updateEvent.__isub__.assert_called_once_with(ibkr_client_instance._on_bar_update)
    assert qualified_contract.conId not in ibkr_client_instance._active_realtime_bars


@patch('ibkr_client.util.startLoop') # For when no loop is running
@patch('ibkr_client.util.patchAsyncio') # For when a loop is running
@patch('asyncio.get_running_loop')
def test_run_async_event_loop_if_needed_starts_loop(mock_get_running_loop, mock_patch_asyncio, mock_start_loop, ibkr_client_instance, mock_ib_insync):
    # Scenario 1: No loop is running, so startLoop should be called
    mock_get_running_loop.side_effect = RuntimeError("No running event loop")
    mock_ib_insync.isConnected.return_value = True # Assume connected
    mock_ib_insync.loop.is_running.return_value = False # IB's loop not running initially
    
    # Simulate util.startLoop effect
    def side_effect_start_loop(ib_instance):
        ib_instance.loop.is_running.return_value = True # Now it's running
    mock_start_loop.side_effect = side_effect_start_loop

    assert ibkr_client_instance.run_async_event_loop_if_needed() == True
    mock_start_loop.assert_called_once_with(mock_ib_insync)
    mock_patch_asyncio.assert_not_called() # Should not be called if new loop started

@patch('ibkr_client.util.patchAsyncio')
@patch('asyncio.get_running_loop')
def test_run_async_event_loop_if_needed_patches_existing_loop(mock_get_running_loop, mock_patch_asyncio, ibkr_client_instance, mock_ib_insync):
    # Scenario 2: A loop is already running
    mock_existing_loop = MagicMock()
    mock_get_running_loop.return_value = mock_existing_loop # A loop is found
    mock_ib_insync.isConnected.return_value = True
    mock_ib_insync.loop.is_running.return_value = False # IB's specific loop part not yet running on this external loop
    
    # Mock the client task starting
    mock_ib_insync.client.run_async = MagicMock()
    mock_ib_insync.loop.create_task = MagicMock()


    assert ibkr_client_instance.run_async_event_loop_if_needed() == True
    mock_patch_asyncio.assert_called_once()
    mock_ib_insync.loop.create_task.assert_called_once() # Should attempt to start IB's client task on existing loop
    
@patch('ibkr_client.util.patchAsyncio')
@patch('asyncio.get_running_loop')
def test_run_async_event_loop_if_needed_ib_loop_already_running(mock_get_running_loop, mock_patch_asyncio, ibkr_client_instance, mock_ib_insync):
    # Scenario 3: A loop is running AND ib_insync's loop is also marked as running
    mock_existing_loop = MagicMock()
    mock_get_running_loop.return_value = mock_existing_loop
    mock_ib_insync.isConnected.return_value = True
    mock_ib_insync.loop.is_running.return_value = True # IB's loop already running on the existing one

    assert ibkr_client_instance.run_async_event_loop_if_needed() == True
    mock_patch_asyncio.assert_called_once()
    mock_ib_insync.loop.create_task.assert_not_called() # Should not create task if ib.loop already running


def test_run_async_event_loop_if_needed_not_connected(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = False
    assert ibkr_client_instance.run_async_event_loop_if_needed() == False

# Add more tests for edge cases, error conditions in methods like place_paper_order, etc.
# For example, what happens if qualifyContract fails inside place_paper_order?
# Test if _active_realtime_bars is correctly managed (e.g. entry deleted on cancel or disconnect).
# Test the _on_bar_update logic more thoroughly (e.g. if hasNewBar is False).
# Test disconnect cleans up all subscriptions correctly.
# Test that if subscribe_realtime_bars fails (e.g. qualifyContract fails), it doesn't leave partial state.
# Test if subscribing to an already subscribed contract is handled (should warn and return existing).
# Test if cancelling a non-existent subscription is handled gracefully.
# Test the logic within qualify_contract for ambiguous contracts more deeply (e.g. non-SMART specified, multiple non-SMART found).
# Test if _on_bar_update handles errors in user callback gracefully.
# Test if _execute_trade_decision and _place_order_for_closing_position in TradingStrategy correctly use the qualified contract from IBKRClient.
# Test the interaction between run_async_event_loop_if_needed and ib.run() or util.startLoop() if deeper testing of asyncio integration is needed.
# Test the exception handling in various methods (e.g., if ib.reqHistoricalData throws an error).
# Test the logging calls to ensure important events are logged.
# Test the formatDate parameter usage in fetch_historical_data if it becomes configurable.
# Test whatToShow and useRTH parameter propagation in fetch_historical_data and subscribe_realtime_bars.
# Test the default values for parameters in methods like subscribe_realtime_bars.
# Test the behavior of _on_bar_update if the subscription is not found in _active_realtime_bars (should log a warning).
# Test the behavior of cancel_realtime_bars if the contract to cancel is not found or not active.

# Test for _on_bar_update when hasNewBar is False
def test_on_bar_update_no_new_bar(ibkr_client_instance, mock_callback):
    mock_bar_data_list = MagicMock()
    ibkr_client_instance._on_bar_update(mock_bar_data_list, False) # hasNewBar is False
    mock_callback.assert_not_called()

# Test subscribing to an already subscribed contract
def test_subscribe_realtime_bars_already_subscribed(ibkr_client_instance, mock_ib_insync, mock_callback):
    contract_to_sub = Stock('MSFT', 'SMART', 'USD')
    qualified_contract = mock_ib_insync.qualifyContracts.return_value[0]
    qualified_contract.symbol = 'MSFT'
    qualified_contract.conId = 5678

    # Simulate it's already active
    existing_bars_obj = MagicMock()
    ibkr_client_instance._active_realtime_bars[qualified_contract.conId] = {
        'contract': qualified_contract,
        'bars': existing_bars_obj,
        'callback': MagicMock() # Some other callback
    }
    
    returned_bars_obj = ibkr_client_instance.subscribe_realtime_bars(contract_to_sub, mock_callback)
    
    assert returned_bars_obj == existing_bars_obj # Should return existing
    mock_ib_insync.reqRealTimeBars.assert_not_called() # Should not call again
    # Ensure original callback wasn't overwritten if that's the design (current design overwrites/ignores new if existing)
    # The current code returns the existing bars object and logs a warning, doesn't change callback.

# Test cancelling a non-existent subscription
def test_cancel_realtime_bars_not_subscribed(ibkr_client_instance, mock_ib_insync):
    contract_not_subscribed = Stock('NONE', 'SMART', 'USD')
    # Qualify will return the default mock AAPL contract, let's assume it's qualified to something else
    mock_ib_insync.qualifyContracts.return_value = [Stock('NONE', 'SMART', 'USD', conId=999)]


    ibkr_client_instance.cancel_realtime_bars(contract_not_subscribed)
    mock_ib_insync.cancelRealTimeBars.assert_not_called() # Should not attempt to cancel

# Test error handling in _on_bar_update if user callback fails
def test_on_bar_update_user_callback_error(ibkr_client_instance, mock_ib_insync):
    contract = Stock('MSFT', 'SMART', 'USD'); contract.conId = 5678
    failing_callback = MagicMock(side_effect=ValueError("User callback failed"))
    
    mock_bar_data_list = MagicMock(); mock_bar_data_list.contract = contract
    ibkr_client_instance._active_realtime_bars[contract.conId] = {
        'contract': contract, 'bars': mock_bar_data_list, 'callback': failing_callback
    }
    latest_bar_data = BarData(time=pd.Timestamp.now(), open=10, high=12, low=9, close=11, volume=1000, wap=10.5, count=50)
    type(mock_bar_data_list)._getitem_ = MagicMock(return_value=latest_bar_data)
    mock_bar_data_list.__getitem__ = lambda self, key: latest_bar_data if key == -1 else (_ for _ in ()).throw(IndexError)


    # Expect it to log an error but not raise exception itself
    with patch.object(ibkr_client_instance.logger, 'error') as mock_log_error: # Assuming self.logger exists and is used
        ibkr_client_instance._on_bar_update(mock_bar_data_list, True)
        mock_log_error.assert_called_once()
        assert "Error in user-provided callback" in mock_log_error.call_args[0][0]
    failing_callback.assert_called_once() # Callback was indeed called

# Test disconnect with no active subscriptions
def test_disconnect_no_active_subscriptions(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.isConnected.return_value = True
    ibkr_client_instance._active_realtime_bars = {} # No subscriptions
    ibkr_client_instance.disconnect()
    mock_ib_insync.cancelRealTimeBars.assert_not_called()
    mock_ib_insync.disconnect.assert_called_once()
    assert not ibkr_client_instance._active_realtime_bars

# Test qualify_contract failure during fetch_historical_data
def test_fetch_historical_data_qualify_fails(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.qualifyContracts.return_value = [] # Simulate qualify failure
    contract = Stock('FAIL', 'SMART', 'USD')
    df = ibkr_client_instance.fetch_historical_data(contract, '', '1 M', '1 day', 'TRADES', True)
    assert df.empty # Should return empty DataFrame
    mock_ib_insync.reqHistoricalData.assert_not_called() # Should not proceed to fetch

# Test qualify_contract failure during place_paper_order
def test_place_paper_order_qualify_fails(ibkr_client_instance, mock_ib_insync):
    mock_ib_insync.qualifyContracts.return_value = [] # Simulate qualify failure
    contract = Stock('FAIL', 'SMART', 'USD')
    order = MagicMock()
    trade_result = ibkr_client_instance.place_paper_order(contract, order)
    assert trade_result is None # Should return None
    mock_ib_insync.placeOrder.assert_not_called()

# Test qualify_contract failure during subscribe_realtime_bars
def test_subscribe_realtime_bars_qualify_fails(ibkr_client_instance, mock_ib_insync, mock_callback):
    mock_ib_insync.qualifyContracts.return_value = [] # Simulate qualify failure
    contract = Stock('FAIL', 'SMART', 'USD')
    subscription_result = ibkr_client_instance.subscribe_realtime_bars(contract, mock_callback)
    assert subscription_result is None
    mock_ib_insync.reqRealTimeBars.assert_not_called()

# Test subscription failure (e.g. reqRealTimeBars throws exception)
def test_subscribe_realtime_bars_req_fails(ibkr_client_instance, mock_ib_insync, mock_callback):
    contract_to_sub = Stock('MSFT', 'SMART', 'USD')
    qualified_contract = mock_ib_insync.qualifyContracts.return_value[0] # Assume qualification works
    qualified_contract.symbol = 'MSFT'; qualified_contract.conId = 5678

    mock_ib_insync.reqRealTimeBars.side_effect = Exception("IB error on reqRealTimeBars")

    with pytest.raises(Exception, match="IB error on reqRealTimeBars"):
        ibkr_client_instance.subscribe_realtime_bars(contract_to_sub, mock_callback)
    
    # Ensure no partial subscription state is left
    assert qualified_contract.conId not in ibkr_client_instance._active_realtime_bars
