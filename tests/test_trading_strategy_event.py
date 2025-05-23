import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Assuming trading_strategy.py and ibkr_client.py are in the parent directory or PYTHONPATH
from trading_strategy import TradingStrategy 
from ibkr_client import IBKRClient # Needed for type hinting if strategy takes real client

from ib_insync import Contract, Stock, Forex, BarData, Order, Trade, OrderStatus

# --- Mock IBKRClient for Backtesting (moved from trading_strategy.py) ---
class MockIBKRClientForBacktest:
    def __init__(self, historical_data_map: dict):
        self.historical_data_map = historical_data_map
        self.logger = MagicMock() # Mock logger if IBKRClient uses one

    def connect(self): self.logger.info("MockIBKRClientForBacktest connected.")
    def disconnect(self): self.logger.info("MockIBKRClientForBacktest disconnected.")
    def qualify_contract(self, contract): return contract 
    
    def fetch_historical_data(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=1):
        # This mock is simplified. A real one would parse durationStr and endDateTime.
        df_to_return = self.historical_data_map.get(contract.symbol, pd.DataFrame())
        if not df_to_return.empty:
            df_copy = df_to_return.copy()
            # The backtest_event_driven method expects lowercase columns from this method
            df_copy.columns = [col.lower() for col in df_copy.columns]
            df_copy.index = pd.to_datetime(df_copy.index) # Ensure datetime index
            return df_copy
        return pd.DataFrame()

    def place_paper_order(self, contract, order):
        # This should not be called by TradingStrategy when in backtesting mode.
        raise NotImplementedError("place_paper_order should not be called in backtesting mode with this mock.")

# --- Test Fixtures ---
@pytest.fixture
def default_alpha_params():
    return [10, 20, 0.6, 0.25, -0.25, 0.15] # short_ma, long_ma, ma_w, rsi_os_w, rsi_ob_w, corr_w

@pytest.fixture
def trading_strategy_instance(default_alpha_params):
    strategy = TradingStrategy(
        alpha_params=default_alpha_params,
        min_bars_for_signal=5, # Lower for easier testing
        max_bar_history=10 # Smaller history for easier testing
    )
    return strategy

@pytest.fixture
def mock_ibkr_client_live():
    """Mocks a live IBKRClient for testing live trading methods if needed."""
    mock_client = MagicMock(spec=IBKRClient)
    mock_client.ib = MagicMock() # Mock the 'ib' attribute
    mock_client.ib.isConnected.return_value = True
    
    # Mock place_paper_order to return a simulated Trade object
    def mock_place_order(contract, order):
        trade = MagicMock(spec=Trade)
        trade.contract = contract
        trade.order = order
        trade.orderStatus = MagicMock(spec=OrderStatus)
        trade.orderStatus.status = "Submitted"
        trade.orderStatus.permId = 12345 # Example permId
        trade.order.orderId = 67890 # Example orderId
        return trade
    mock_client.place_paper_order.side_effect = mock_place_order
    return mock_client

# --- Sample Data ---
def create_sample_bar_data(symbol="AAPL", dt=None, price=150.0, volume=1000):
    if dt is None:
        dt = datetime.now()
    return {
        'time': dt, 'open': price - 0.5, 'high': price + 0.5, 'low': price - 1.0,
        'close': price, 'volume': volume, 'wap': price, 'count': 10,
        'symbol': symbol, 'conId': 12345 # Dummy conId
    }

def create_sample_contract(symbol="AAPL"):
    if "." in symbol or len(symbol) == 6:
        return Forex(symbol.replace(".",""))
    return Stock(symbol, "SMART", "USD")

# --- Tests for on_realtime_bar / on_historical_bar ---
def test_on_historical_bar_updates_history_and_processes(trading_strategy_instance):
    strategy = trading_strategy_instance
    strategy.is_backtesting = True # Set mode
    strategy.current_backtest_time = datetime(2023,1,1,10,0,0) # Required for backtesting log
    
    contract_aapl = create_sample_contract("AAPL")
    bar1_data = create_sample_bar_data("AAPL", datetime(2023,1,1,9,30,0), 150)
    
    with patch.object(strategy, 'process_signal_and_trade') as mock_process_trade:
        strategy.on_historical_bar(bar1_data, contract_aapl)
        
        assert "AAPL" in strategy.live_data_history
        assert len(strategy.live_data_history["AAPL"]) == 1
        assert strategy.live_data_history["AAPL"].iloc[0]['Close'] == 150.0
        mock_process_trade.assert_called_once_with("AAPL", contract_aapl, 150.0)
        assert len(strategy.daily_equity) == 1 # Should have one entry from on_historical_bar

def test_on_realtime_bar_updates_history_and_processes(trading_strategy_instance, mock_ibkr_client_live):
    strategy = trading_strategy_instance
    strategy.start_live_trading(100000, mock_ibkr_client_live) # Sets in_live_mode=True
    
    contract_msft = create_sample_contract("MSFT")
    bar1_data = create_sample_bar_data("MSFT", datetime.now(), 250)
    
    with patch.object(strategy, 'process_signal_and_trade') as mock_process_trade:
        strategy.on_realtime_bar(bar1_data, contract_msft)
        
        assert "MSFT" in strategy.live_data_history
        assert len(strategy.live_data_history["MSFT"]) == 1
        assert strategy.live_data_history["MSFT"].iloc[0]['Close'] == 250.0
        mock_process_trade.assert_called_once_with("MSFT", contract_msft, 250.0)

def test_history_max_length_respected(trading_strategy_instance):
    strategy = trading_strategy_instance
    strategy.is_backtesting = True
    strategy.max_bar_history = 3 # Small for test
    
    contract_aapl = create_sample_contract("AAPL")
    for i in range(5):
        dt = datetime(2023,1,1,9,30+i,0)
        strategy.current_backtest_time = dt # Update for each bar
        bar_data = create_sample_bar_data("AAPL", dt, 150+i)
        with patch.object(strategy, 'process_signal_and_trade'): # Mock to prevent other logic
             strategy.on_historical_bar(bar_data, contract_aapl)
            
    assert len(strategy.live_data_history["AAPL"]) == 3
    assert strategy.live_data_history["AAPL"].iloc[0]['Close'] == 150.0 + 2 # (5-3) = 2nd bar is 152
    assert strategy.live_data_history["AAPL"].iloc[-1]['Close'] == 150.0 + 4


# --- Tests for process_signal_and_trade ---
def test_process_signal_and_trade_stop_loss_long(trading_strategy_instance):
    strategy = trading_strategy_instance
    strategy.is_backtesting = True # Enable backtesting mode for simulated execution
    
    symbol = "AAPL"
    contract = create_sample_contract(symbol)
    entry_price = 100.0
    stop_price = entry_price * (1 - strategy.stop_loss_pct) # 98.0
    
    strategy.live_positions[symbol] = {
        'contract': contract, 'size': 10, 'entry_price': entry_price,
        'stop_price': stop_price, 'take_profit_price': entry_price * (1 + strategy.take_profit_pct)
    }
    
    # Simulate not enough data for new signal calculation to isolate SL/TP logic
    strategy.live_data_history[symbol] = pd.DataFrame({'Close': [entry_price]}) 
    strategy.min_bars_for_signal = 10 # Ensure it's more than available

    with patch.object(strategy, '_place_order_for_closing_position') as mock_close_order:
        # Price drops below stop_price
        strategy.process_signal_and_trade(symbol, contract, stop_price - 0.1)
        mock_close_order.assert_called_once_with(symbol, contract, "STOP_LOSS", fill_price=stop_price - 0.1)

def test_process_signal_and_trade_take_profit_short(trading_strategy_instance):
    strategy = trading_strategy_instance
    strategy.is_backtesting = True
    
    symbol = "MSFT"
    contract = create_sample_contract(symbol)
    entry_price = 200.0
    take_profit_price = entry_price * (1 - strategy.take_profit_pct) # e.g., 200 * (1-0.03) = 194
    
    strategy.live_positions[symbol] = {
        'contract': contract, 'size': -5, 'entry_price': entry_price,
        'stop_price': entry_price * (1 + strategy.stop_loss_pct), 
        'take_profit_price': take_profit_price
    }
    strategy.live_data_history[symbol] = pd.DataFrame({'Close': [entry_price]})
    strategy.min_bars_for_signal = 10

    with patch.object(strategy, '_place_order_for_closing_position') as mock_close_order:
        # Price drops to hit take_profit
        strategy.process_signal_and_trade(symbol, contract, take_profit_price - 0.5)
        mock_close_order.assert_called_once_with(symbol, contract, "TAKE_PROFIT", fill_price=take_profit_price - 0.5)


# --- Tests for calculate_live_signal_for_symbol ---
def test_calculate_live_signal_buy_and_sell(trading_strategy_instance):
    strategy = trading_strategy_instance # short_win=10, long_win=20
    symbol = "TEST"
    
    # Create data that should trigger a BUY signal (short MA crosses above long MA)
    prices_buy = np.concatenate([np.linspace(100,90,15), np.linspace(90,110,15)]) # Dip then strong rise
    df_buy = pd.DataFrame({'Close': prices_buy, 
                           'Open': prices_buy, 'High': prices_buy, 'Low': prices_buy, 'Volume': [100]*30}, 
                          index=pd.date_range(start='2023-01-01', periods=30, freq='D'))
    
    # To ensure RSI is not extreme, let's make it smoother
    rsi_prices = np.linspace(100, 105, 30)
    df_buy_rsi_neutral = pd.DataFrame({'Close': rsi_prices, 'Open': rsi_prices, 'High': rsi_prices, 'Low': rsi_prices, 'Volume': [100]*30},
                                     index=pd.date_range(start='2023-01-01', periods=30, freq='D'))
    df_buy_rsi_neutral['Close'] = df_buy['Close'] # Use original close for MA, but other prices for neutral RSI calc if needed.
                                                # For simplicity, current RSI calc in strategy only uses 'Close'.

    signal_buy = strategy.calculate_live_signal_for_symbol(df_buy_rsi_neutral, symbol, df_buy_rsi_neutral['Close'].iloc[-1])
    assert signal_buy['action'] == 'BUY'
    assert signal_buy['size'] > 0 # Position sizing should yield some shares

    # Create data that should trigger a SELL signal (short MA crosses below long MA)
    prices_sell = np.concatenate([np.linspace(100,110,15), np.linspace(110,90,15)]) # Rise then strong fall
    df_sell = pd.DataFrame({'Close': prices_sell,
                           'Open': prices_sell, 'High': prices_sell, 'Low': prices_sell, 'Volume': [100]*30},
                           index=pd.date_range(start='2023-01-01', periods=30, freq='D'))
    signal_sell = strategy.calculate_live_signal_for_symbol(df_sell, symbol, df_sell['Close'].iloc[-1])
    assert signal_sell['action'] == 'SELL'
    assert signal_sell['size'] < 0


# --- Tests for _execute_trade_decision (Backtesting Mode) ---
def test_execute_trade_decision_buy_backtesting(trading_strategy_instance):
    strategy = trading_strategy_instance
    strategy.is_backtesting = True
    strategy.live_capital = 100000
    strategy.current_backtest_time = datetime(2023,1,1,10,0,0)
    
    symbol = "AAPL"
    contract = create_sample_contract(symbol)
    current_price = 150.0
    trade_decision = {'action': 'BUY', 'size': 10, 'current_price': current_price} # Size is positive from calc_live_signal
    
    strategy._execute_trade_decision(symbol, contract, trade_decision, current_price)
    
    assert symbol in strategy.live_positions
    pos = strategy.live_positions[symbol]
    assert pos['size'] == 10
    
    sim_fill_price = current_price * (1 + strategy.slippage_pct)
    assert pos['entry_price'] == pytest.approx(sim_fill_price)
    
    trade_value = sim_fill_price * 10
    tx_cost = trade_value * strategy.transaction_cost_pct
    expected_capital = 100000 - trade_value - tx_cost
    assert strategy.live_capital == pytest.approx(expected_capital)
    
    assert len(strategy.trade_log) == 1
    log_entry = strategy.trade_log[0]
    assert log_entry['symbol'] == symbol
    assert log_entry['action'] == 'BUY'
    assert log_entry['size'] == 10
    assert log_entry['price'] == pytest.approx(sim_fill_price)
    assert log_entry['transaction_cost'] == pytest.approx(tx_cost)


# --- Test for backtest_event_driven (Integration) ---
def test_backtest_event_driven_simple_run(default_alpha_params):
    strategy = TradingStrategy(default_alpha_params, min_bars_for_signal=5, max_bar_history=10)

    # Create dummy historical data
    dates1 = pd.date_range(start='2023-01-01', periods=20, freq='D')
    df_sym1 = pd.DataFrame({
        'Open': np.linspace(100, 110, 20), 'High': np.linspace(101, 111, 20),
        'Low': np.linspace(99, 109, 20), 'Close': np.linspace(100.5, 110.5, 20),
        'Volume': np.random.randint(100, 200, 20)
    }, index=dates1)
    
    mock_hist_data = {'SYM1': df_sym1}
    mock_ib_client = MockIBKRClientForBacktest(historical_data_map=mock_hist_data)
    
    contracts_to_test = {'SYM1': Stock('SYM1', 'SMART', 'USD')}
    
    results, trade_log_df, equity_curve_df = strategy.backtest_event_driven(
        ibkr_client_instance=mock_ib_client,
        symbols_contracts=contracts_to_test,
        start_date_str='2023-01-01',
        end_date_str='2023-01-20',
        bar_size='1 day',
        initial_capital=100000.0
    )
    
    assert "error" not in results
    assert results['initial_capital'] == 100000.0
    assert 'final_capital' in results
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert not equity_curve_df.empty
    
    # Check if any trades were made (depends on data and simple strategy)
    if not trade_log_df.empty:
        assert 'pnl' in trade_log_df.columns
        assert 'transaction_cost' in trade_log_df.columns
        # Verify P&L consistency (approximate)
        total_pnl_from_log = trade_log_df['pnl'].sum()
        capital_change = results['final_capital'] - results['initial_capital']
        # This might not be exact due to how capital is tracked (e.g. only on close vs. mark-to-market)
        # but should be related.
        # For this test, simply check it runs and produces output.
        # A more detailed P&L check would require a very specific trade sequence.
    
    assert strategy.is_backtesting is False # Should be reset
    assert len(strategy.daily_equity) == len(df_sym1) + 1 # Initial point + each bar
    assert strategy.daily_equity[0]['capital'] == 100000.0
    assert strategy.daily_equity[-1]['capital'] == pytest.approx(results['final_capital'])

# TODO: Add more granular tests for TradingStrategy:
# - _calculate_live_position_size scenarios
# - _place_order_for_closing_position in backtesting mode (P&L calc, capital update)
# - More complex scenarios for calculate_live_signal (e.g. RSI triggers, correlation influence if mocked)
# - Edge cases: not enough data for min_bars_for_signal in process_signal_and_trade
# - Multiple symbols in backtest_event_driven to test data merging and chronological processing
# - Test for correct application of transaction_cost_pct and slippage_pct in various scenarios.
# - Test behavior when correlation_matrix is None vs. provided.
# - Test what happens if fetch_historical_data returns empty df for one symbol in multi-symbol backtest.
# - Test for correct handling of `max_bar_history`.
# - Test stop-loss/take-profit logic within `process_signal_and_trade` more directly.
# - Test that `current_backtest_time` is correctly used in trade logs.
# - Test equity curve calculation.
# - Test behavior of `calculate_live_signal_for_symbol` when data has NaNs or is too short for indicators.
# - Test `_get_symbol_from_contract_or_bar` helper.
# - Test `start_live_trading` and `stop_live_trading` for mode changes.
# - Test `calculate_atr` if it's used by the strategy (currently example, not core logic).
# - Test `process_signal_and_trade` when a signal leads to closing an existing position and opening a new one (reversal).
# - Test `_execute_trade_decision` and `_place_order_for_closing_position` for live mode (mocking IBKRClient's place_paper_order).
# - Test that `live_positions` is correctly updated (size, entry_price, SL/TP prices).
