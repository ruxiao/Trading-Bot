import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import random # For slippage simulation, consider removing or making deterministic for live
import logging

# Assuming ib_insync objects will be used for orders
from ib_insync.order import MarketOrder, LimitOrder, Order # For type hinting
from ib_insync.contract import Contract # For type hinting

# Logger for this module
logger = logging.getLogger(__name__)

class TradingStrategy:
    """
    Implements the trading logic, signal generation, and both live/paper trading execution
    as well as event-driven backtesting.
    
    The strategy primarily uses Moving Average (MA) crossovers and RSI for signals,
    with considerations for risk management (stop-loss, take-profit, position sizing)
    and simulated/real transaction costs and slippage.
    """
    def __init__(self, alpha_params: List[float], 
                 transaction_cost_pct: float = 0.0005, 
                 slippage_pct: float = 0.0003,        
                 risk_per_trade_pct: float = 0.01,    
                 max_position_size_pct: float = 0.1,  
                 stop_loss_pct: float = 0.02,         
                 take_profit_pct: float = 0.03,       
                 volatility_scaling: bool = True,
                 max_bar_history: int = 200,          
                 min_bars_for_signal: int = 50        
                 ):
        """
        Initializes the TradingStrategy.

        Args:
            alpha_params (List[float]): List of parameters for alpha generation.
                Expected order: [short_ma_window, long_ma_window, ma_weight, 
                                 rsi_oversold_weight, rsi_overbought_weight, correlation_weight].
            transaction_cost_pct (float): Percentage transaction cost per trade.
            slippage_pct (float): Percentage slippage per trade.
            risk_per_trade_pct (float): Max percentage of capital to risk on a single trade.
            max_position_size_pct (float): Max percentage of capital for a single position.
            stop_loss_pct (float): Percentage from entry price to set stop-loss.
            take_profit_pct (float): Percentage from entry price to set take-profit.
            volatility_scaling (bool): Whether to use volatility in position sizing.
            max_bar_history (int): Maximum number of historical bars to keep per symbol.
            min_bars_for_signal (int): Minimum bars required before generating signals for a symbol.
        """
        
        self.short_window = max(5, int(round(alpha_params[0]))) 
        self.long_window = max(10, int(round(alpha_params[1])))
        if self.short_window >= self.long_window: 
            self.long_window = self.short_window + 5 
            
        self.alpha_params_config = { 
            'short_window': self.short_window,
            'long_window': self.long_window,
            'ma_weight': alpha_params[2],
            'rsi_os_weight': alpha_params[3],
            'rsi_ob_weight': alpha_params[4], # Typically negative if overbought means sell
            'corr_weight': alpha_params[5]
        }
        
        self.position_limits = (-1, 1) # Allow short (-1) and long (1) positions.
        
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct 
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_size_pct = max_position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volatility_scaling = volatility_scaling
        
        self.max_bar_history = max_bar_history
        self.min_bars_for_signal = max(self.long_window + 10, min_bars_for_signal) 
        
        # State variables
        self.ibkr_client: Optional[IBKRClient] = None # IBKR client instance, injected
        self.in_live_mode: bool = False               # True if in live/paper trading session
        self.is_backtesting: bool = False             # True if in backtesting mode
        self.live_capital: float = 0.0                # Current capital
        
        # Stores current positions: {'SYMBOL': {'contract': Contract, 'size': float, 'entry_price': float, ...}}
        self.live_positions: Dict[str, Dict] = {} 
        # Stores historical bar data: {'SYMBOL': pd.DataFrame}
        self.live_data_history: Dict[str, pd.DataFrame] = {} 
        # Log of trades made: List of dictionaries
        self.trade_log: List[Dict] = [] 
        # Tracks equity changes during backtest: List of {'timestamp': datetime, 'capital': float}
        self.daily_equity: List[Dict] = [] 
        # Current time in backtest, used for logging and consistent decision making
        self.current_backtest_time: Optional[datetime] = None 

        # Correlation matrix for strategy (optional, can be updated externally)
        self.correlation_matrix: Optional[pd.DataFrame] = None 
        self.default_avg_correlation: float = 0.3 # Fallback if no matrix or symbol not found

        logger.info(f"TradingStrategy initialized. Short MA: {self.short_window}, Long MA: {self.long_window}, Min Bars: {self.min_bars_for_signal}")

    def start_live_trading(self, initial_capital: float, ibkr_client_instance: IBKRClient):
        """
        Initializes the strategy for a live (or paper) trading session.

        Args:
            initial_capital (float): The starting capital for this trading session.
            ibkr_client_instance (IBKRClient): An active IBKRClient instance for market interaction.
        """
        self.is_backtesting = False 
        self.in_live_mode = True
        self.live_capital = initial_capital
        self.ibkr_client = ibkr_client_instance 
        self.live_positions = {}
        self.live_data_history = {}
        self.trade_log = []
        self.daily_equity = [] # Not typically used in live mode this way, but reset for consistency
        logger.info(f"Live trading mode started with capital: {initial_capital:.2f}")

    def stop_live_trading(self):
        """
        Stops the live trading mode. Resets flags.
        Does not automatically close open positions.
        """
        self.is_backtesting = False 
        # Note: Current open positions are not automatically closed by this method.
        # This would require iterating through self.live_positions and calling
        # self._place_order_for_closing_position for each, which might be desired.
        self.in_live_mode = False
        logger.info("Live trading mode stopped.")

    def _get_symbol_from_contract_or_bar(self, contract_info: Optional[Contract] = None, bar_data: Optional[dict] = None) -> Optional[str]:
        """
        Helper to consistently extract a symbol string from either contract or bar data.
        
        Args:
            contract_info (Optional[Contract]): An ib_insync Contract object.
            bar_data (Optional[dict]): A dictionary representing bar data, expected to have a 'symbol' key.

        Returns:
            Optional[str]: The extracted symbol string, or None if not found.
        """
        if bar_data and 'symbol' in bar_data:
            return bar_data['symbol']
        if contract_info and hasattr(contract_info, 'symbol'): # Forex contracts might use localSymbol
            return contract_info.symbol if contract_info.symbol else contract_info.localSymbol
        logger.warning("Could not determine symbol from provided contract or bar data.")
        return None

    def on_realtime_bar(self, bar_data: dict, contract_info: Contract):
        """
        Callback method for processing incoming real-time bars from IBKRClient.
        This is the primary entry point for new data during live/paper trading.

        Args:
            bar_data (dict): A dictionary containing the latest bar data. Expected keys:
                             'time', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'conId'.
            contract_info (Contract): The ib_insync Contract object for the received bar.
        """
        if not self.in_live_mode or not self.ibkr_client:
            logger.warning("on_realtime_bar called but not in live mode or IBKR client not set.")
            return

        symbol = self._get_symbol_from_contract_or_bar(contract_info, bar_data)
        if not symbol: return # Error already logged by helper
        
        current_bar_time = pd.to_datetime(bar_data['time'])
        try:
            bar_series = pd.Series({
                'Open': float(bar_data['open']), 'High': float(bar_data['high']),
                'Low': float(bar_data['low']), 'Close': float(bar_data['close']),
                'Volume': float(bar_data['volume'])
            }, name=current_bar_time)
        except Exception as e:
            logger.error(f"Error converting real-time bar_data to Series for {symbol}: {e}, Data: {bar_data}")
            return

        # Initialize DataFrame for symbol if not present
        if symbol not in self.live_data_history:
            self.live_data_history[symbol] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Append new bar, handling potential duplicates by index (time)
        if bar_series.name not in self.live_data_history[symbol].index:
            self.live_data_history[symbol] = pd.concat([self.live_data_history[symbol], bar_series.to_frame().T])
        else: # Update existing bar if timestamp matches (e.g. corrections from IBKR)
            self.live_data_history[symbol].loc[bar_series.name] = bar_series # pragma: no cover (rare case)
            
        # Maintain max history length
        if len(self.live_data_history[symbol]) > self.max_bar_history:
            self.live_data_history[symbol] = self.live_data_history[symbol].iloc[-self.max_bar_history:]
        
        logger.debug(f"Live bar for {symbol}: {bar_series.to_dict()}. History size: {len(self.live_data_history[symbol])}")
        
        # Process signal and potentially trade using the close of the new bar
        self.process_signal_and_trade(symbol, contract_info, bar_series['Close'])


    def process_signal_and_trade(self, symbol: str, contract_info: Contract, current_price: float):
        """
        Core logic to process signals and manage trades for a given symbol based on its latest price.
        This is called by `on_realtime_bar` (live) or `on_historical_bar` (backtest).

        Args:
            symbol (str): The symbol string (e.g., 'AAPL').
            contract_info (Contract): The ib_insync Contract object for the symbol.
            current_price (float): The current market price for the symbol, used for evaluating
                                   SL/TP conditions and as a potential fill price.
        """
        # Ensure enough historical data is available for signal calculation
        symbol_history_df = self.live_data_history.get(symbol, pd.DataFrame())
        if len(symbol_history_df) < self.min_bars_for_signal:
            logger.info(f"Not enough data for {symbol} to generate signal (have {len(symbol_history_df)}, need {self.min_bars_for_signal}).")
            return

        # 1. Check Stop-Loss / Take-Profit for existing positions
        if symbol in self.live_positions:
            pos = self.live_positions[symbol]
            action_taken = False
            if pos['size'] > 0: # Long position
                if current_price <= pos['stop_price']:
                    logging.info(f"STOP-LOSS for long {symbol} at {current_price} (stop was {pos['stop_price']})")
                    self._place_order_for_closing_position(symbol, contract_info, "STOP_LOSS")
                    action_taken = True
                elif current_price >= pos['take_profit_price']:
                    logging.info(f"TAKE-PROFIT for long {symbol} at {current_price} (TP was {pos['take_profit_price']})")
                    self._place_order_for_closing_position(symbol, contract_info, "TAKE_PROFIT")
                    action_taken = True
            elif pos['size'] < 0: # Short position
                if current_price >= pos['stop_price']:
                    logging.info(f"STOP-LOSS for short {symbol} at {current_price} (stop was {pos['stop_price']})")
                    self._place_order_for_closing_position(symbol, contract_info, "STOP_LOSS")
                    action_taken = True
                elif current_price <= pos['take_profit_price']:
                    logging.info(f"TAKE-PROFIT for short {symbol} at {current_price} (TP was {pos['take_profit_price']})")
                    self._place_order_for_closing_position(symbol, contract_info, "TAKE_PROFIT")
                    action_taken = True
            
            if action_taken: # If SL/TP hit, no further signal processing for this bar
                return

        # 2. Calculate new trading signal
        # Pass current_price to calculate_live_signal_for_symbol for up-to-date reference if needed
        symbol_data_history = self.live_data_history[symbol] # This is the historical data up to the *previous* bar
        
        # For backtesting, the "current_price" is the close of the current event bar.
        # For live trading, it's the price from the latest tick/bar.
        signal_decision = self.calculate_live_signal_for_symbol(symbol_data_history, symbol, current_price)
        
        # 3. Act on the signal
        current_pos_size = self.live_positions.get(symbol, {}).get('size', 0)
        # The current_price here is the fill price for the new order if one is generated.
        # Or the price at which SL/TP is evaluated.

        if signal_decision['action'] == 'BUY':
            if current_pos_size < 0: # Current short, need to close and go long
                logging.info(f"Signal for {symbol}: REVERSE to LONG from SHORT at price {current_price}. Current size: {current_pos_size}")
                self._place_order_for_closing_position(symbol, contract_info, "REVERSING_TO_LONG", fill_price=current_price)
                # After closing, state is updated. Now process the BUY part.
                # Re-fetch current_pos_size as it's now 0 after closing.
                current_pos_size = self.live_positions.get(symbol, {}).get('size', 0) 
                if current_pos_size == 0: # Ensure close was processed
                     self._execute_trade_decision(symbol, contract_info, signal_decision, current_price)
                else:
                    logging.warning(f"Position for {symbol} not zero after attempting to close short. Size: {current_pos_size}. Cannot reverse to long.")

            elif current_pos_size == 0: # No position, go long
                logging.info(f"Signal for {symbol}: ENTER LONG at price {current_price}. Current size: {current_pos_size}")
                self._execute_trade_decision(symbol, contract_info, signal_decision, current_price)
            else: # Already long, potentially scale in or hold (no scaling in logic yet)
                logging.debug(f"Signal for {symbol}: HOLD LONG or already long. Action: {signal_decision['action']}")
        
        elif signal_decision['action'] == 'SELL':
            if current_pos_size > 0: # Current long, need to close and go short
                logging.info(f"Signal for {symbol}: REVERSE to SHORT from LONG at price {current_price}. Current size: {current_pos_size}")
                self._place_order_for_closing_position(symbol, contract_info, "REVERSING_TO_SHORT", fill_price=current_price)
                current_pos_size = self.live_positions.get(symbol, {}).get('size', 0)
                if current_pos_size == 0: # Ensure close was processed
                    self._execute_trade_decision(symbol, contract_info, signal_decision, current_price)
                else:
                    logging.warning(f"Position for {symbol} not zero after attempting to close long. Size: {current_pos_size}. Cannot reverse to short.")

            elif current_pos_size == 0: # No position, go short
                logging.info(f"Signal for {symbol}: ENTER SHORT at price {current_price}. Current size: {current_pos_size}")
                self._execute_trade_decision(symbol, contract_info, signal_decision, current_price)
            else: # Already short
                logging.debug(f"Signal for {symbol}: HOLD SHORT or already short. Action: {signal_decision['action']}")

        elif signal_decision['action'] == 'CLOSE':
            if current_pos_size != 0:
                logging.info(f"Signal for {symbol}: CLOSE current position of {current_pos_size} at price {current_price}.")
                self._place_order_for_closing_position(symbol, contract_info, "SIGNAL_CLOSE", fill_price=current_price)
            else:
                logging.debug(f"Signal for {symbol}: CLOSE, but no position held.")
        
        # HOLD action implies doing nothing

    def calculate_live_signal_for_symbol(self, data: pd.DataFrame, symbol: str, current_price: float) -> Dict:
        """
        Calculate trading signal based on the latest historical data for a single symbol.
        This is the core alpha generation logic.

        Args:
            data (pd.DataFrame): DataFrame of historical bars for the symbol.
                                 Index should be datetime, columns should include 'Close', 'Open', 'High', 'Low', 'Volume'.
            symbol (str): The symbol string.
            current_price (float): The current market price (used for logging context, not directly in MA/RSI from history).

        Returns:
            Dict: A dictionary detailing the trading decision. Example:
                  `{'action': 'BUY'/'SELL'/'HOLD'/'CLOSE', 'size': float, 'alpha': float, 'current_price': float}`
                  'size' is 0 if action is 'HOLD' or 'CLOSE'. For 'BUY' it's positive, for 'SELL' it's negative.
        """
        if len(data) < self.min_bars_for_signal: 
            return {'action': 'HOLD', 'alpha': 0, 'comment': f"Not enough data: {len(data)} bars, need {self.min_bars_for_signal}"}

        # Ensure data has 'Close' column and enough rows for rolling operations
        if 'Close' not in data.columns or len(data) < self.long_window : # pragma: no cover (defensive)
             logger.warning(f"Data for {symbol} missing 'Close' or too short for MAs ({len(data)} bars). Holding.")
             return {'action': 'HOLD', 'alpha': 0, 'comment': "Data missing 'Close' or too short for MAs."}

        # --- Calculate Technical Indicators (Pandas-based) ---
        ma_short = data['Close'].rolling(window=self.short_window).mean().iloc[-1]
        ma_long = data['Close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # Volatility (e.g., 20-period standard deviation of percentage returns)
        volatility_series = data['Close'].pct_change().rolling(window=20).std()
        volatility = volatility_series.iloc[-1] if not volatility_series.empty and pd.notna(volatility_series.iloc[-1]) else 0.01
        volatility = max(volatility, 0.0001) # Ensure non-zero positive volatility

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain_series = (delta.where(delta > 0, 0.0)).rolling(window=14).mean() # Ensure 0.0 for non-positive delta
        loss_series = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean() # Ensure 0.0 for non-negative delta
        
        gain = gain_series.iloc[-1] if not gain_series.empty and pd.notna(gain_series.iloc[-1]) else 0
        loss = loss_series.iloc[-1] if not loss_series.empty and pd.notna(loss_series.iloc[-1]) else 0
        
        if loss == 0: # Avoid division by zero; if no losses, RSI is 100, if no gains (and no losses), neutral 50.
            rsi = 100.0 if gain > 0 else 50.0
        else:
            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Correlation component (simplified for this example)
        # In a real system, self.correlation_matrix should be updated periodically.
        correlation_signal = 0.0
        if isinstance(self.correlation_matrix, pd.DataFrame) and \
           symbol in self.correlation_matrix.columns and \
           symbol in self.correlation_matrix.index:
            correlations = self.correlation_matrix[symbol].drop(symbol, errors='ignore')
            if not correlations.empty:
                avg_correlation = correlations.mean()
                correlation_signal = -avg_correlation if pd.notna(avg_correlation) else 0.0
        else: 
            correlation_signal = -self.default_avg_correlation # Fallback if no matrix or symbol not found

        # --- Alpha Calculation (Combine indicator signals) ---
        ma_component = (ma_short - ma_long) / ma_long if ma_long != 0 else 0.0
        
        rsi_component = 0.0
        if rsi < 30: # Oversold condition
            rsi_component = self.alpha_params_config['rsi_os_weight'] * ((30.0 - rsi) / 30.0)
        elif rsi > 70: # Overbought condition
            rsi_component = self.alpha_params_config['rsi_ob_weight'] * ((rsi - 70.0) / 30.0) # This weight is typically negative

        # Final alpha value combining components
        alpha = (self.alpha_params_config['ma_weight'] * ma_component +
                   rsi_component + 
                   self.alpha_params_config['corr_weight'] * correlation_signal)
        
        logger.debug(f"Symbol: {symbol}, Price: {current_price:.2f}, Alpha: {alpha:.4f} "
                     f"(MA: {ma_component:.4f}, RSI({rsi:.1f}): {rsi_component:.4f}, Corr: {correlation_signal:.4f})")

        # --- Decision Logic (Convert alpha to trade action) ---
        action = 'HOLD'
        # Example thresholds (these should be tuned or part of strategy parameters)
        alpha_buy_threshold = 0.15 
        alpha_sell_threshold = -0.15
        alpha_close_threshold_long = -0.05 # If long and alpha turns slightly negative, consider closing
        alpha_close_threshold_short = 0.05  # If short and alpha turns slightly positive, consider closing

        current_pos_size = self.live_positions.get(symbol, {}).get('size', 0)

        if current_pos_size == 0: # If no current position
            if alpha > alpha_buy_threshold: action = 'BUY'
            elif alpha < alpha_sell_threshold: action = 'SELL'
        elif current_pos_size > 0: # Currently Long
            if alpha < alpha_close_threshold_long: action = 'CLOSE'       # Signal weakened or reversed
            elif alpha < alpha_sell_threshold: action = 'SELL' # Strong signal to reverse
        elif current_pos_size < 0: # Currently Short
            if alpha > alpha_close_threshold_short: action = 'CLOSE'      # Signal weakened or reversed
            elif alpha > alpha_buy_threshold: action = 'BUY'  # Strong signal to reverse
        
        calculated_size = 0.0 # Default size is 0 (no action or closing)
        if action in ['BUY', 'SELL']: # Only calculate size for new/reverse positions
            calculated_size = self._calculate_live_position_size(symbol, current_price, volatility)
            if action == 'SELL': # For opening a new short position
                calculated_size = -calculated_size # Negative size indicates short

        return {'action': action, 'size': calculated_size, 'alpha': alpha, 'current_price': current_price}

    def _calculate_live_position_size(self, symbol: str, price: float, volatility: float, order_action: str = "NEW") -> float:
        """
        Calculates the number of shares for a new position based on risk management rules.

        Args:
            symbol (str): The symbol for which to calculate position size.
            price (float): The current market price of the symbol.
            volatility (float): The recent volatility of the symbol (e.g., standard deviation of returns).
            order_action (str, optional): Context of the order, default "NEW". Not currently used but for future.
        
        Returns:
            float: The number of shares to trade. Positive for long, should be made negative by caller for short.
                   Returns 0 if conditions don't permit trading (e.g., zero capital, zero price).
        """
        if price <= 0 or self.live_capital <= 0: return 0 # Cannot trade with no capital or invalid price

        # Max value for this position based on percentage of total capital
        max_value_per_position = self.live_capital * self.max_position_size_pct
        
        # Calculate risk amount per share based on stop-loss percentage
        risk_amount_per_share = price * self.stop_loss_pct
        # Note: If volatility_scaling is True, one might adjust risk_amount_per_share based on volatility here,
        # e.g., risk_amount_per_share = price * volatility * some_factor if stop is ATR based.
        # Current implementation uses a fixed stop_loss_pct, volatility scaling might influence overall risk_per_trade_pct application.
        # For this simple example, volatility scaling is more a general flag than directly modifying this part of size calc.

        if risk_amount_per_share <= 0: return 0 # Stop loss implies no risk or invalid price/stop.

        # Number of shares based on capital at risk per trade
        shares_from_risk = (self.live_capital * self.risk_per_trade_pct) / risk_amount_per_share
        # Value of position if sized by risk_per_trade
        value_from_risk = shares_from_risk * price
        
        # Final position value is the minimum of max allowed value and risk-based value
        final_value_for_position = min(max_value_per_position, value_from_risk)
        
        # Calculate number of shares from this final value
        shares = final_value_for_position / price
        
        # Return whole shares (or tradable lots, depending on asset)
        return round(shares) if shares > 0 else 0 # Ensure non-negative shares from this calculation


    def _execute_trade_decision(self, symbol: str, contract_info: Contract, trade_decision: Dict, fill_price: float):
        """
        Executes a new trade (BUY or SELL to open) based on the signal decision.
        Handles both live/paper trading and backtest simulation.

        Args:
            symbol (str): The symbol to trade.
            contract_info (Contract): The contract object for the symbol.
            trade_decision (Dict): Dictionary from `calculate_live_signal_for_symbol` containing
                                   'action' ('BUY' or 'SELL'), 'size', etc.
            fill_price (float): The price at which the trade is assumed to be filled.
                                For live, this is an estimate (current market). For backtest, it's bar close.
        """
        action = trade_decision['action'] # Should be 'BUY' or 'SELL' for opening
        target_quantity = abs(trade_decision['size']) # Order quantity is always positive
        
        if target_quantity == 0:
            logger.warning(f"Calculated trade size for {symbol} is 0. Action: {action}. No trade placed.")
            return

        order_status = 'Filled' # Default for backtesting simulation
        pnl = 0 # P&L is typically for closing trades.

        # Simulate slippage
        simulated_fill_price = fill_price
        if action == 'BUY':
            simulated_fill_price = fill_price * (1 + self.slippage_pct)
        elif action == 'SELL': # Opening a short
            simulated_fill_price = fill_price * (1 - self.slippage_pct)
        
        trade_value = simulated_fill_price * target_quantity
        transaction_cost = trade_value * self.transaction_cost_pct
        
        self.live_capital -= transaction_cost # Deduct transaction cost

        if self.is_backtesting:
            logging.info(f"BACKTEST_SIM: {action} {target_quantity} {symbol} at {simulated_fill_price:.2f} (Cost: {transaction_cost:.2f})")
            # Update capital and positions
            if action == 'BUY':
                self.live_capital -= trade_value 
                self.live_positions[symbol] = {
                    'contract': contract_info, 'size': target_quantity, 'entry_price': simulated_fill_price,
                    'stop_price': simulated_fill_price * (1 - self.stop_loss_pct),
                    'take_profit_price': simulated_fill_price * (1 + self.take_profit_pct),
                    'last_update_time': self.current_backtest_time # Needs current bar time
                }
            elif action == 'SELL': # Opening a new short
                self.live_capital += trade_value # Add proceeds from short sell (will be negative asset)
                self.live_positions[symbol] = {
                    'contract': contract_info, 'size': -target_quantity, 'entry_price': simulated_fill_price,
                    'stop_price': simulated_fill_price * (1 + self.stop_loss_pct),
                    'take_profit_price': simulated_fill_price * (1 - self.take_profit_pct),
                    'last_update_time': self.current_backtest_time
                }
            
            self.trade_log.append({
                'timestamp': self.current_backtest_time, 'symbol': symbol, 'action': action, 
                'size': target_quantity if action == 'BUY' else -target_quantity, 
                'price': simulated_fill_price, 'status': order_status, 'reason': 'SIGNAL_ENTRY',
                'transaction_cost': transaction_cost, 'pnl': pnl, # PNL is 0 for entry
                'capital_after_trade': self.live_capital
            })
        else: # Live Trading
            order = MarketOrder(action, target_quantity)
            logging.info(f"LIVE: Attempting to place order for {symbol}: {order.action} {order.totalQuantity} shares at market.")
            try:
                trade = self.ibkr_client.place_paper_order(contract_info, order)
                if trade and trade.orderStatus:
                    logging.info(f"LIVE: Paper order placed for {symbol}: {order.action} {order.totalQuantity}. Status: {trade.orderStatus.status}, OrderId: {trade.order.orderId}")
                    # Optimistic update for live mode (actual fill price/time would come from IBKR events)
                    entry_price = fill_price # Approximate with current price for live logging
                    if action == 'BUY':
                         self.live_capital -= (entry_price * target_quantity) # Cost of shares
                         self.live_positions[symbol] = {
                            'contract': contract_info, 'size': target_quantity, 'entry_price': entry_price,
                            'stop_price': entry_price * (1 - self.stop_loss_pct),
                            'take_profit_price': entry_price * (1 + self.take_profit_pct),
                            'last_update_time': datetime.now() 
                        }
                    elif action == 'SELL': # Opening short
                         self.live_capital += (entry_price * target_quantity) # Proceeds
                         self.live_positions[symbol] = {
                            'contract': contract_info, 'size': -target_quantity, 'entry_price': entry_price,
                            'stop_price': entry_price * (1 + self.stop_loss_pct),
                            'take_profit_price': entry_price * (1 - self.take_profit_pct),
                            'last_update_time': datetime.now()
                        }
                    self.trade_log.append({
                        'timestamp': datetime.now(), 'symbol': symbol, 'action': action, 
                        'size': target_quantity if action == 'BUY' else -target_quantity, 
                        'price': entry_price, 'status': trade.orderStatus.status,
                        'order_id': trade.order.orderId, 'reason': 'SIGNAL_ENTRY',
                        'transaction_cost': transaction_cost, # Note: live fill price could differ
                        'pnl': 0, 'capital_after_trade': self.live_capital
                    })
                else: logging.error(f"LIVE: Failed to place paper order for {symbol} or trade object invalid. Trade: {trade}")
            except Exception as e: logging.error(f"LIVE: Exception placing paper order for {symbol}: {e}")


    def _place_order_for_closing_position(self, symbol: str, contract_info: Contract, reason: str, fill_price: Optional[float] = None):
        if symbol not in self.live_positions or self.live_positions[symbol]['size'] == 0:
            logging.warning(f"Request to close position for {symbol}, but no position found or size is zero.")
            return

        pos_info = self.live_positions[symbol]
        position_size_to_close = pos_info['size'] # Actual signed size
        entry_price = pos_info['entry_price']
        
        close_action = 'SELL' if position_size_to_close > 0 else 'BUY'
        quantity_to_close = abs(position_size_to_close)
        order_status = 'Filled' # Default for backtesting simulation

        # Use provided fill_price for backtesting, otherwise it's a live close (price unknown until fill)
        simulated_fill_price = fill_price if self.is_backtesting and fill_price is not None else entry_price # Fallback, less accurate for live
        
        # Simulate slippage for closing trade
        if close_action == 'SELL': # Selling to close a long
            simulated_fill_price = simulated_fill_price * (1 - self.slippage_pct)
        elif close_action == 'BUY': # Buying to close a short
            simulated_fill_price = simulated_fill_price * (1 + self.slippage_pct)

        trade_value = simulated_fill_price * quantity_to_close
        transaction_cost = trade_value * self.transaction_cost_pct
        
        self.live_capital -= transaction_cost # Deduct transaction cost for the close

        # Calculate P&L
        if position_size_to_close > 0: # Closed a long position
            pnl = (simulated_fill_price - entry_price) * quantity_to_close - transaction_cost
            self.live_capital += trade_value # Add proceeds from selling shares
        else: # Closed a short position
            pnl = (entry_price - simulated_fill_price) * quantity_to_close - transaction_cost
            self.live_capital -= trade_value # Cost to buy back shares

        if self.is_backtesting:
            logging.info(f"BACKTEST_SIM: CLOSE {close_action} {quantity_to_close} {symbol} at {simulated_fill_price:.2f} (Entry: {entry_price:.2f}, P&L: {pnl:.2f}, Cost: {transaction_cost:.2f}) Reason: {reason}")
            del self.live_positions[symbol]
            log_event_time = self.current_backtest_time
        else: # Live trading
            order = MarketOrder(close_action, quantity_to_close)
            logging.info(f"LIVE: Attempting to CLOSE position for {symbol}: {order.action} {order.totalQuantity} shares. Reason: {reason}")
            try:
                trade = self.ibkr_client.place_paper_order(contract_info, order)
                if trade and trade.orderStatus:
                    logging.info(f"LIVE: Paper CLOSE order placed for {symbol}: {order.action} {order.totalQuantity}. Status: {trade.orderStatus.status}, OrderId: {trade.order.orderId}")
                    order_status = trade.orderStatus.status
                    del self.live_positions[symbol] 
                else:
                    logging.error(f"LIVE: Failed to place paper CLOSE order for {symbol} or trade object invalid.")
                    order_status = "Error" # Or some other error status
            except Exception as e:
                logging.error(f"LIVE: Exception placing paper CLOSE order for {symbol}: {e}")
                order_status = "Exception"
            log_event_time = datetime.now()
        
        self.trade_log.append({
            'timestamp': log_event_time, 'symbol': symbol, 'action': close_action, 
            'size': quantity_to_close, 'price': simulated_fill_price, 'status': order_status,
            'reason': reason, 'transaction_cost': transaction_cost, 'pnl': pnl,
            'capital_after_trade': self.live_capital, 'entry_price': entry_price
        })


    def on_historical_bar(self, bar_data: dict, contract_info: Contract):
        """
        Processes a single historical bar during backtesting.
        Similar to on_realtime_bar but uses self.current_backtest_time.
        """
        if not self.is_backtesting:
            logging.warning("on_historical_bar called when not in backtesting mode.")
            return

        symbol = self._get_symbol_from_contract_or_bar(contract_info, bar_data)
        if not symbol: return

        current_bar_time = pd.to_datetime(bar_data['time'])
        self.current_backtest_time = current_bar_time # Store current bar time for logging

        try:
            bar_series = pd.Series({
                'Open': float(bar_data['open']), 'High': float(bar_data['high']),
                'Low': float(bar_data['low']), 'Close': float(bar_data['close']),
                'Volume': float(bar_data['volume'])
            }, name=current_bar_time)
        except Exception as e:
            logging.error(f"Error converting historical bar_data to Series for {symbol}: {e}, Data: {bar_data}")
            return

        if symbol not in self.live_data_history:
            self.live_data_history[symbol] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        if bar_series.name not in self.live_data_history[symbol].index:
            self.live_data_history[symbol] = pd.concat([self.live_data_history[symbol], bar_series.to_frame().T])
        else: 
            self.live_data_history[symbol].loc[bar_series.name] = bar_series
            
        if len(self.live_data_history[symbol]) > self.max_bar_history:
            self.live_data_history[symbol] = self.live_data_history[symbol].iloc[-self.max_bar_history:]
        
        # The current_price for signal processing and potential execution is the close of this historical bar.
        self.process_signal_and_trade(symbol, contract_info, bar_series['Close'])
        
        # Record daily equity at the end of processing this bar (or could be daily if bars are intraday)
        self.daily_equity.append({'timestamp': current_bar_time, 'capital': self.live_capital})


    def backtest_event_driven(self, ibkr_client_instance, symbols_contracts: dict, 
                              start_date_str: str, end_date_str: str, 
                              bar_size: str, initial_capital: float,
                              correlation_matrix_df: Optional[pd.DataFrame] = None):
        logging.info(f"Starting event-driven backtest. Symbols: {list(symbols_contracts.keys())}, Period: {start_date_str} to {end_date_str}, Bar: {bar_size}")
        self.is_backtesting = True
        self.ibkr_client = ibkr_client_instance # Use passed client for fetching data
        self.live_capital = initial_capital
        self.live_positions = {}
        self.live_data_history = {} # Reset history for each symbol
        self.trade_log = []
        self.daily_equity = [{'timestamp': pd.to_datetime(start_date_str) - pd.Timedelta(days=1), 'capital': initial_capital}] # Initial capital point
        self.current_backtest_time = pd.to_datetime(start_date_str) # Initialize

        if correlation_matrix_df is not None:
            self.correlation_matrix = correlation_matrix_df
            logging.info("Correlation matrix provided for backtest.")


        all_bars_data = []
        for symbol, contract in symbols_contracts.items():
            logging.info(f"Fetching historical data for {symbol} from {start_date_str} to {end_date_str} ({bar_size})...")
            # Calculate duration string (approximate, IBKR's reqHistoricalData is flexible)
            # Example: '30 D', '1 M', '1 Y'
            # This needs to be calculated based on start/end date for fetch_historical_data
            # For simplicity, assuming fetch_historical_data can handle start/end dates if the client is adapted,
            # or we fetch a bit more and filter. Let's assume durationStr is needed.
            # A more robust way is to calculate days: (pd.to_datetime(end_date_str) - pd.to_datetime(start_date_str)).days
            # Max duration for smaller bars can be an issue. Fetching year by year or month by month might be needed for long periods.
            # For now, let's assume a simple duration. This is a known limitation of reqHistoricalData.
            # A duration like "1 Y" for 1 day bars is fine. For 5 sec bars, it's much shorter.
            # The IBKRClient's fetch_historical_data uses endDateTime and durationStr.
            # We'll use a large enough duration and filter, or adjust client.
            # A simple approach: if period > ~200 days, use '1 Y', if > month use 'X M', else 'X D'.
            # This part is tricky with IBKR's API. Let's assume a fixed duration for now and filter.
            # For a proper backtester, fetch_historical_data would need to be more robust, possibly paginating.
            
            # Let's assume fetch_historical_data is called with endDateTime='' (now) and appropriate duration.
            # For backtesting, endDateTime should be end_date_str.
            # And duration calculated.
            # For this example, I'll simplify and assume fetch_historical_data can take start/end or is adapted.
            # Let's assume `fetch_historical_data` is modified to take start/end or we fetch broadly.
            # For now, I'll use a placeholder duration and expect the user to ensure data covers the range.
            # This is a simplification for the current step.
            # A better `fetch_historical_data` in `ibkr_client` would accept start and end dates.
            # Let's assume it does for the purpose of this subtask, or we fetch a large chunk.
            # The current `fetch_historical_data` takes `endDateTime` and `durationStr`.
            # We'll set endDateTime to `end_date_str` and calculate duration.
            
            s_date = pd.to_datetime(start_date_str)
            e_date = pd.to_datetime(end_date_str)
            delta_days = (e_date - s_date).days + 1 # Ensure end date is included
            duration_str_for_fetch = f"{delta_days} D"
            if delta_days > 365 * 2: duration_str_for_fetch = "2 Y" # Max for some bar sizes
            elif delta_days > 365: duration_str_for_fetch = "1 Y"
            elif delta_days > 180: duration_str_for_fetch = "6 M"
            elif delta_days > 90: duration_str_for_fetch = "3 M"
            elif delta_days > 30: duration_str_for_fetch = "1 M"


            hist_df = self.ibkr_client.fetch_historical_data(
                contract, 
                endDateTime=end_date_str, # Fetch up to the end date
                durationStr=duration_str_for_fetch, # Broad duration
                barSizeSetting=bar_size, 
                whatToShow='TRADES', 
                useRTH=True # Configurable
            )

            if hist_df is not None and not hist_df.empty:
                # Ensure DataFrame index is datetime
                hist_df.index = pd.to_datetime(hist_df.index)
                # Filter for the exact date range
                hist_df = hist_df[(hist_df.index >= pd.to_datetime(start_date_str)) & (hist_df.index <= pd.to_datetime(end_date_str))]

                for index_time, row in hist_df.iterrows():
                    bar_event_data = {
                        'time': index_time, # This is the datetime index from df
                        'open': row['open'], 'high': row['high'], 
                        'low': row['low'], 'close': row['close'], 
                        'volume': row['volume'], 'symbol': symbol, # Add symbol
                        'contract_info': contract # Add contract object
                    }
                    all_bars_data.append(bar_event_data)
            else:
                logging.warning(f"No historical data fetched for {symbol} for the period.")

        if not all_bars_data:
            logging.error("No historical data found for any symbol. Aborting backtest.")
            self.is_backtesting = False
            return {"error": "No data"}, [], []

        # Sort all bars chronologically
        all_bars_data.sort(key=lambda x: x['time'])
        logging.info(f"Total historical bars to process: {len(all_bars_data)}")

        # Event loop simulation
        for bar_event in all_bars_data:
            event_time = bar_event['time']
            event_symbol = bar_event['symbol']
            event_contract = bar_event['contract_info']
            
            # The bar_event already contains symbol and contract_info,
            # but on_historical_bar expects them as separate args too.
            self.on_historical_bar(bar_event, event_contract) 
            # Note: on_historical_bar updates self.current_backtest_time and self.daily_equity

        self.is_backtesting = False
        logging.info(f"Event-driven backtest finished. Total trades: {len(self.trade_log)}")
        
        # Performance Calculation
        equity_df = pd.DataFrame(self.daily_equity).set_index('timestamp')
        equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
        
        # Use PerformanceAnalyzer
        # analyzer = PerformanceAnalyzer() # If it were a class with instance methods
        # metrics = analyzer.calculate_metrics(equity_df['returns'])
        # For static methods:
        from performance import PerformanceAnalyzer # Assuming performance.py is in path
        metrics = PerformanceAnalyzer.calculate_metrics(equity_df['returns'])
        
        # Add final capital to metrics
        metrics['final_capital'] = self.live_capital
        metrics['initial_capital'] = initial_capital
        metrics['num_trades'] = len(self.trade_log)
        
        return metrics, self.trade_log, equity_df


    # --- Methods below are largely from the old backtesting-focused version ---
    # --- They need review/adaptation if to be used for event-driven backtesting or removed ---

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series: # Keep ATR as it's a utility
        """Calculate Average True Range. Expects DataFrame with Open, High, Low, Close columns."""
        if not all(col in data.columns for col in ['High', 'Low', 'Close']):
            # Fallback if H/L not present, use Close for a rough ATR (less accurate) # pragma: no cover
            logging.warning("High/Low not in data for ATR, using Close. ATR will be less accurate.") # pragma: no cover
            close = data['Close'] # pragma: no cover
            tr = close.diff().abs() # Simplified TR # pragma: no cover
        else:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        return atr.fillna(0) # fillna for initial period

# Example Usage (for testing parts of the strategy class, including event-driven backtest)
if __name__ == '__main__': # pragma: no cover
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

    # --- Mock IBKRClient for testing ---
    class MockIBKRClientForBacktest:
        def __init__(self, historical_data_map: Dict[str, pd.DataFrame]):
            self.historical_data_map = historical_data_map # {'AAPL': df_aapl, 'GOOG': df_goog}
            self.placed_orders = []

        def connect(self): logging.info("MockIBKRClientForBacktest connected.")
        def disconnect(self): logging.info("MockIBKRClientForBacktest disconnected.")
        def qualify_contract(self, contract): return contract 
        
        def fetch_historical_data(self, contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate=1):
            logging.info(f"Mock fetch_historical_data for {contract.symbol}. End: {endDateTime}, Duration: {durationStr}, Bar: {barSizeSetting}")
            # Return data for the specific symbol if available in the map
            # This mock is simplified; a real one would need to parse durationStr and endDateTime
            # to return the correct slice of data. For this test, we assume the preloaded df is what's needed.
            df_to_return = self.historical_data_map.get(contract.symbol, pd.DataFrame())
            
            # Ensure columns are lowercase as expected by some parts of ib_insync.util.df or direct access
            # The strategy code expects 'Open', 'High', 'Low', 'Close', 'Volume' (uppercase)
            # The IBKRClient's fetch_historical_data returns df with columns: date, open, high, low, close, volume, average, barCount
            # Let's simulate the structure returned by the actual IBKRClient's processing
            if not df_to_return.empty:
                # Our strategy's on_historical_bar expects 'open', 'high', 'low', 'close', 'volume' in the bar_event_data dict
                # and the DataFrame it constructs internally uses 'Open', 'High', 'Low', 'Close', 'Volume'.
                # The actual `util.df(bars)` from `ib_insync` produces columns like: date, open, high, low, close, volume, average, barCount
                # Let's ensure our mock provides what the backtester's bar event construction needs.
                # The backtester's bar event construction in `backtest_event_driven` does:
                # 'open': row['open'], 'high': row['high'], etc. So it expects lowercase from the DataFrame rows.
                # The initial DataFrame for AAPL in the test below has uppercase.
                # So, the mock should provide lowercase column names if it's mimicking the direct output of ib_insync.IB.reqHistoricalData processing.
                # However, our `ibkr_client.fetch_historical_data` already converts to uppercase keys in the `util.df(bars)` step.
                # So the mock should return df with 'open', 'high', etc. as the strategy expects from its iteration.
                
                # The `backtest_event_driven` method iterates `hist_df.iterrows()` and expects keys like `row['open']`.
                # So this mock should return a DataFrame with lowercase column names to match that specific loop.
                # If the test data has uppercase, we'll convert here.
                df_copy = df_to_return.copy()
                df_copy.columns = [col.lower() for col in df_copy.columns]
                # Ensure index is datetime
                df_copy.index = pd.to_datetime(df_copy.index)
                return df_copy
            return pd.DataFrame()

        def place_paper_order(self, contract, order): # This shouldn't be called if is_backtesting is True
            logging.warning(f"MockIBKRClientForBacktest: place_paper_order called unexpectedly for {contract.symbol} during backtest simulation.")
            # This is only for live mode, so during backtest, this indicates an issue if called.
            return None 


    # --- Prepare data for backtest_event_driven ---
    alpha_p_test = [12, 26, 0.5, 0.3, -0.3, 0.1] 
    strategy_bt = TradingStrategy(alpha_params=alpha_p_test, 
                                  min_bars_for_signal=30, # Need enough for long_window + buffer
                                  transaction_cost_pct=0.001, 
                                  slippage_pct=0.0005)

    # Create dummy historical data for two symbols: AAPL, MSFT
    # AAPL Data (trends up then down)
    dates_aapl = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-10 16:00:00', freq='1h')
    prices_aapl1 = np.linspace(150, 160, len(dates_aapl)//2)
    prices_aapl2 = np.linspace(160, 155, len(dates_aapl) - len(dates_aapl)//2)
    prices_aapl = np.concatenate([prices_aapl1, prices_aapl2])
    volume_aapl = np.random.randint(10000, 50000, len(dates_aapl))
    df_aapl = pd.DataFrame({
        'Open': prices_aapl - 0.2, 'High': prices_aapl + 0.3, 
        'Low': prices_aapl - 0.3, 'Close': prices_aapl, 'Volume': volume_aapl
    }, index=dates_aapl)

    # MSFT Data (more choppy)
    dates_msft = pd.date_range(start='2023-01-01 09:30:00', end='2023-01-10 16:00:00', freq='1h')
    prices_msft = 250 + np.sin(np.linspace(0, 10, len(dates_msft))) * 5 + np.random.randn(len(dates_msft)) * 0.5
    volume_msft = np.random.randint(15000, 60000, len(dates_msft))
    df_msft = pd.DataFrame({
        'Open': prices_msft - 0.15, 'High': prices_msft + 0.25, 
        'Low': prices_msft - 0.25, 'Close': prices_msft, 'Volume': volume_msft
    }, index=dates_msft)

    # Mock client with this data
    # The mock's fetch_historical_data will return DFs with lowercase column names as per its logic.
    mock_hist_data = {'AAPL': df_aapl.copy(), 'MSFT': df_msft.copy()}
    mock_ib_client_bt = MockIBKRClientForBacktest(historical_data_map=mock_hist_data)

    # Define contracts for backtest
    contracts_bt = {
        'AAPL': Contract(symbol='AAPL', exchange='SMART', currency='USD', conId=101),
        'MSFT': Contract(symbol='MSFT', exchange='SMART', currency='USD', conId=102)
    }
    
    # Correlation matrix (optional, can be None)
    # For testing, ensure columns and index match symbols if provided
    corr_data = {'AAPL': {'AAPL': 1.0, 'MSFT': 0.6}, 'MSFT': {'AAPL': 0.6, 'MSFT': 1.0}}
    correlation_df_bt = pd.DataFrame(corr_data)


    logging.info("\n--- Running Event-Driven Backtest ---")
    backtest_results, trade_log_bt, equity_curve_bt = strategy_bt.backtest_event_driven(
        ibkr_client_instance=mock_ib_client_bt,
        symbols_contracts=contracts_bt,
        start_date_str='2023-01-01',
        end_date_str='2023-01-10',
        bar_size='1 hour', # This should match the frequency of data in mock_hist_data
        initial_capital=100000,
        correlation_matrix_df=correlation_df_bt 
    )

    logging.info("\n--- Backtest Results ---")
    if "error" in backtest_results:
        logging.error(f"Backtest failed: {backtest_results['error']}")
    else:
        for metric, value in backtest_results.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                logging.info(f"{metric.replace('_', ' ').title()}: (see details below)")
                # print(value.head()) # Avoid printing large series/dfs in main log
            else:
                logging.info(f"{metric.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, float) else f"{metric.replace('_', ' ').title()}: {value}")
    
    logging.info(f"\n--- Trade Log (Last 5 trades) ---")
    if trade_log_bt:
        for trade_entry in pd.DataFrame(trade_log_bt).tail(5).to_dict('records'):
            logging.info(trade_entry)
    else:
        logging.info("No trades in backtest log.")

    logging.info(f"\n--- Equity Curve (Last 5 points) ---")
    if not equity_curve_bt.empty:
        logging.info(equity_curve_bt.tail())
    else:
        logging.info("Equity curve is empty.")
    
    logging.info(f"\nFinal Capital from Strategy Object: {strategy_bt.live_capital:.2f}")
    logging.info(f"Total P&L from trade log: {pd.DataFrame(trade_log_bt)['pnl'].sum() if trade_log_bt else 0.0:.2f}")

    # --- Test live trading part briefly (original __main__ content) ---
    # This part tests the live trading path with a different mock client.
    # Note: The state (live_capital, positions) will be from the backtest if not reset.
    # For a clean test of live mode, re-initialize strategy or reset state.
    
    # Re-initialize strategy for a clean live mode test or reset state explicitly.
    # strategy_live_test = TradingStrategy(alpha_params=alpha_p, min_bars_for_signal=25)
    # mock_client_live = MockIBKRClient() # Original mock for live testing
    # strategy_live_test.start_live_trading(initial_capital=100000, ibkr_client_instance=mock_client_live)
    # ... (rest of the original __main__ for live mode testing if desired) ...
    # For now, focusing on the backtest part.