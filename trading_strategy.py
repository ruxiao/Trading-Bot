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


    def backtest_event_driven(self, ibkr_client_instance,
                              # For QQQ0DTEScalping, symbols_contracts will typically be {'QQQ': qqq_contract_object}
                              symbols_contracts: dict,
                              start_date_str: str, end_date_str: str, 
                              bar_size: str, # This bar_size is for the QQQ underlying
                              initial_capital: float,
                              # correlation_matrix_df is not used by QQQ0DTEScalping directly
                              correlation_matrix_df: Optional[pd.DataFrame] = None
                              ):

        # Determine if this is the QQQ0DTEScalping strategy
        is_qqq_0dte_strategy = isinstance(self, QQQ0DTEScalping)

        if is_qqq_0dte_strategy:
            log_prefix = "QQQ0DTE Backtest"
            if self.underlying_symbol not in symbols_contracts:
                logger.error(f"{log_prefix}: Underlying symbol {self.underlying_symbol} not found in symbols_contracts. Aborting.")
                return {"error": f"Underlying {self.underlying_symbol} not provided"}, [], pd.DataFrame()
            underlying_contract_to_fetch = symbols_contracts[self.underlying_symbol]
            logger.info(f"{log_prefix}: Using {self.underlying_symbol} as the primary underlying data feed.")
        else: # Original multi-symbol logic
            log_prefix = "Multi-Asset Backtest"
            logger.info(f"{log_prefix}: Symbols: {list(symbols_contracts.keys())}")

        logger.info(f"{log_prefix}: Period: {start_date_str} to {end_date_str}, Bar Size (Underlying): {bar_size}")

        self.is_backtesting = True
        self.ibkr_client = ibkr_client_instance
        self.live_capital = initial_capital
        self.live_positions = {} # For base strategy's stock positions
        if is_qqq_0dte_strategy:
            self.active_option_position = None # Reset specific state for QQQ0DTE
            self.daily_trade_count = 0
            self.last_trade_date = None

        self.live_data_history = {}
        self.trade_log = []
        self.daily_equity = [{'timestamp': pd.to_datetime(start_date_str) - pd.Timedelta(days=1), 'capital': initial_capital}]
        self.current_backtest_time = pd.to_datetime(start_date_str)

        if correlation_matrix_df is not None and not is_qqq_0dte_strategy: # Corr matrix for base strategy
            self.correlation_matrix = correlation_matrix_df
            logger.info(f"{log_prefix}: Correlation matrix provided.")

        all_bars_data = []

        # If QQQ0DTE strategy, only fetch data for QQQ.
        # Otherwise, fetch for all symbols in symbols_contracts.
        contracts_to_process_for_bars = {}
        if is_qqq_0dte_strategy:
            contracts_to_process_for_bars[self.underlying_symbol] = underlying_contract_to_fetch
        else:
            contracts_to_process_for_bars = symbols_contracts

        for symbol, contract_to_fetch in contracts_to_process_for_bars.items():
            logger.info(f"{log_prefix}: Fetching historical data for {symbol} from {start_date_str} to {end_date_str} ({bar_size})...")
            
            s_date = pd.to_datetime(start_date_str)
            e_date = pd.to_datetime(end_date_str)
            delta_days = (e_date - s_date).days + 1
            duration_str_for_fetch = f"{max(1, delta_days)} D" # Ensure at least 1 D

            # Adjust duration string for very long periods or specific IBKR limits if known
            # This is a simplified duration calculation.
            if delta_days > 730: duration_str_for_fetch = "2 Y"
            elif delta_days > 365: duration_str_for_fetch = "1 Y"
            # ... (add more granular duration logic if needed based on bar_size)

            hist_df = self.ibkr_client.fetch_historical_data(
                contract_to_fetch,
                endDateTime=e_date.strftime('%Y%m%d %H:%M:%S'), # Fetch up to the end date & time
                durationStr=duration_str_for_fetch,
                barSizeSetting=bar_size, 
                whatToShow='TRADES', 
                useRTH=True
            )

            if hist_df is not None and not hist_df.empty:
                # hist_df.index is already datetime from fetch_historical_data's set_index('date')
                # Filter for the exact date range (inclusive)
                hist_df = hist_df[(hist_df.index >= pd.to_datetime(start_date_str)) & (hist_df.index <= pd.to_datetime(end_date_str + " 23:59:59"))] # Ensure end_date is inclusive

                for index_time, row in hist_df.iterrows():
                    bar_event_data = {
                        'time': index_time,
                        'open': row['open'], 'high': row['high'], 
                        'low': row['low'], 'close': row['close'], 
                        'volume': row['volume'], 'symbol': symbol,
                        'contract_info': contract_to_fetch
                    }
                    all_bars_data.append(bar_event_data)
            else:
                logger.warning(f"{log_prefix}: No historical data fetched for {symbol} for the period.")

        if not all_bars_data:
            logger.error(f"{log_prefix}: No historical data found for any primary symbol(s). Aborting backtest.")
            self.is_backtesting = False
            return {"error": "No primary data"}, [], pd.DataFrame()

        all_bars_data.sort(key=lambda x: x['time'])
        logger.info(f"{log_prefix}: Total historical bars (underlying) to process: {len(all_bars_data)}")

        for bar_event in all_bars_data:
            self.on_historical_bar(bar_event, bar_event['contract_info'])

        self.is_backtesting = False
        logger.info(f"{log_prefix}: Event-driven backtest finished. Total items in trade_log: {len(self.trade_log)}")
        
        equity_df = pd.DataFrame(self.daily_equity).set_index('timestamp')
        if equity_df.empty:
            logger.warning(f"{log_prefix}: Equity DataFrame is empty. Cannot calculate performance metrics.")
            return {"error": "No equity data logged"}, self.trade_log, pd.DataFrame()

        equity_df['returns'] = equity_df['capital'].pct_change().fillna(0)
        
        from performance import PerformanceAnalyzer
        metrics = PerformanceAnalyzer.calculate_metrics(equity_df['returns'].to_numpy()) # Pass as numpy array
        
        metrics['final_capital'] = self.live_capital
        metrics['initial_capital'] = initial_capital
        # num_trades for options strategy is more complex (round trips).
        # The trade_log for QQQ0DTE contains individual option buy/sell legs.
        # For now, len(self.trade_log) gives total option order executions.
        metrics['num_option_order_executions'] = len(self.trade_log) if is_qqq_0dte_strategy else "N/A"
        metrics['num_stock_trades'] = len([t for t in self.trade_log if self.underlying_symbol in t.get('symbol','').upper() and t.get('action','').upper() in ['BUY','SELL']]) if not is_qqq_0dte_strategy else len(self.trade_log)


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

# (Keep existing TradingStrategy class and its methods)

# +++ QQQ 0DTE Scalping Strategy +++
class QQQ0DTEScalping(TradingStrategy):
    """
    Strategy for scalping QQQ 0DTE (Zero Days To Expiration) options.
    Focuses on intraday movements of QQQ using technical indicators like VWAP and EMAs.
    Trades ATM or slightly OTM options.
    Limits trades to a maximum of 5 round trips per day.
    Closes all positions before market end.
    """
    def __init__(self,
                 # Alpha parameters for QQQ underlying
                 vwap_period: int = 20, # Typical for intraday VWAP calculations from bar data
                 short_ema_period: int = 5,
                 long_ema_period: int = 12,
                 rsi_period: int = 9,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 # Option selection parameters
                 strike_offset_otm: float = 0.5, # E.g., $0.50 OTM from current QQQ price
                 # Risk management for options
                 option_stop_loss_pct: float = 0.10, # -10% from premium paid
                 option_take_profit_pct: float = 0.20, # +20% from premium paid
                 max_daily_trades: int = 5,
                 # General strategy params (inherited or overridden)
                 transaction_cost_pct: float = 0.0005, # Per stock/ETF trade value
                 option_transaction_cost_per_contract: float = 0.65, # Typical per contract fee
                 slippage_pct: float = 0.0003, # For underlying
                 option_slippage_per_contract: float = 0.02, # $0.02 slippage on premium
                 risk_per_trade_pct: float = 0.01, # For underlying, if traded
                 max_option_contracts_per_trade: int = 5, # Max number of option contracts per single trade
                 capital_per_option_trade_pct: float = 0.02 # Max % of capital for one option trade (e.g. 5 contracts * premium)
                 ):

        # Initialize base TradingStrategy with dummy alpha_params as this strategy has its own.
        # The base class's alpha_params were for a generic multi-asset strategy.
        # We are specializing here for QQQ options.
        super().__init__(alpha_params=[short_ema_period, long_ema_period, 0.6, 0.2, -0.2, 0], # Dummy MA weights etc.
                         transaction_cost_pct=transaction_cost_pct,
                         slippage_pct=slippage_pct,
                         risk_per_trade_pct=risk_per_trade_pct,
                         # stop_loss_pct and take_profit_pct from base are for underlying,
                         # we'll use option_stop_loss_pct for options.
                         stop_loss_pct=0.02, # Default for underlying if ever used by base logic
                         take_profit_pct=0.03 # Default for underlying
                         )

        self.underlying_symbol = "QQQ" # Hardcoded for this strategy

        # Indicator parameters for QQQ
        self.vwap_period = vwap_period
        self.short_ema_period = short_ema_period
        self.long_ema_period = long_ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

        # Option specific parameters
        self.strike_offset_otm = strike_offset_otm
        self.option_stop_loss_pct = option_stop_loss_pct
        self.option_take_profit_pct = option_take_profit_pct
        self.max_daily_trades = max_daily_trades
        self.option_transaction_cost_per_contract = option_transaction_cost_per_contract
        self.option_slippage_per_contract = option_slippage_per_contract
        self.max_option_contracts_per_trade = max_option_contracts_per_trade
        self.capital_per_option_trade_pct = capital_per_option_trade_pct

        # State variables for 0DTE strategy
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.active_option_position: Optional[Dict] = None # Stores info about the current option held
        # Example: {'contract_obj': QualifiedIBOptionContract, 'entry_premium': float, 'quantity': int,
        #           'stop_price': float, 'profit_target': float, 'entry_time': datetime, 'type': 'CALL'/'PUT'}

        logger.info(f"QQQ0DTEScalping strategy initialized. Underlying: {self.underlying_symbol}, Max Daily Trades: {self.max_daily_trades}")
        logger.info(f"QQQ Indicators: VWAP({vwap_period}), EMA({short_ema_period},{long_ema_period}), RSI({rsi_period},{rsi_oversold},{rsi_overbought})")
        logger.info(f"Option Params: StrikeOffsetOTM=${strike_offset_otm}, SL={option_stop_loss_pct:.0%}, TP={option_take_profit_pct:.0%}")

    def _reset_daily_counters(self, current_bar_time: datetime):
        """Resets daily counters if the date has changed."""
        current_date = current_bar_time.date()
        if self.last_trade_date is None or self.last_trade_date != current_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
            logger.info(f"New trading day: {current_date}. Daily trade count reset.")

    def _is_near_market_close(self, current_bar_time: datetime, minutes_before_close: int = 20) -> bool:
        """Checks if it's near market close (e.g., 3:40 PM ET for 4:00 PM ET close)."""
        # Assuming US Eastern Time market hours. This might need adjustment for other markets/timezones.
        market_close_hour = 16
        market_close_minute = 0

        # Create datetime objects for comparison in the current bar's timezone (if available)
        # For simplicity, let's assume current_bar_time is already in market's timezone (e.g., ET)
        # This is a common simplification in backtesters using data without explicit timezone info.
        # A robust solution would involve timezone conversions.
        market_close_time_today = current_bar_time.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
        time_to_close = market_close_time_today - current_bar_time

        return time_to_close.total_seconds() <= minutes_before_close * 60


    # Override process_signal_and_trade for option-specific logic
    def process_signal_and_trade(self, symbol: str, contract_info: Contract, current_price: float):
        """
        Core logic for QQQ 0DTE options. This overrides the base class method.
        'symbol' here refers to the underlying QQQ.
        'contract_info' is the QQQ underlying contract.
        'current_price' is the current price of QQQ.
        """
        if symbol != self.underlying_symbol: # This strategy only trades QQQ
            logger.debug(f"QQQ0DTEScalping received bar for non-QQQ symbol: {symbol}. Ignoring.")
            return

        current_bar_time = self.current_backtest_time if self.is_backtesting else datetime.now() #pd.to_datetime(self.live_data_history[symbol].index[-1])
        if current_bar_time is None: # Should not happen if processing a bar
            logger.error("current_bar_time is None in QQQ0DTEScalping.process_signal_and_trade. Skipping.")
            return

        self._reset_daily_counters(current_bar_time)

        # 1. Manage existing option position (SL/TP, EOD close)
        if self.active_option_position:
            option_contract_ib = self.active_option_position['contract_obj'] # This is an ib_insync.Option
            option_symbol_str = option_contract_ib.localSymbol # For logging

            # Fetch current premium for the active option.
            # This is the tricky part in backtesting. For live, it's a market data request.
            # For backtesting, we need to simulate or fetch its price.
            # Assuming current_price is for QQQ, we need option's current premium.
            current_option_premium = self._get_current_option_premium(option_contract_ib, current_bar_time)

            if current_option_premium is None:
                logger.warning(f"Could not get current premium for active option {option_symbol_str} at {current_bar_time}. Cannot manage position effectively.")
                # If premium becomes unavailable for a held option during backtesting, it's problematic.
                # Consider a rule: if premium is unavailable for X consecutive checks, force close.
                # For now, if it's near EOD and premium is gone, try to close.
                if self.is_backtesting and self._is_near_market_close(current_bar_time, minutes_before_close=10): # More aggressive EOD close if data lost
                     logger.warning(f"Force closing {option_symbol_str} due to unavailable premium near EOD (backtesting). Assumed closing at 50% loss from entry or small value.")
                     assumed_closing_premium = self.active_option_position.get('entry_premium', 0.02) * 0.5 # Assume 50% loss
                     assumed_closing_premium = max(0.01, assumed_closing_premium) # Ensure it's not zero
                     self._close_active_option_position(option_contract_ib, assumed_closing_premium, "UNAVAILABLE_PREMIUM_EOD_BT")
                return # Cannot proceed without current premium

            # Check SL/TP
            action_taken = False
            if self.active_option_position['type'] == 'CALL' or self.active_option_position['type'] == 'PUT': # Both are bought options
                if current_option_premium <= self.active_option_position['stop_price']:
                    logger.info(f"OPTION STOP-LOSS for {option_symbol_str} at {current_option_premium:.2f} (Stop was {self.active_option_position['stop_price']:.2f})")
                    self._close_active_option_position(option_contract_ib, current_option_premium, "STOP_LOSS")
                    action_taken = True
                elif current_option_premium >= self.active_option_position['profit_target']:
                    logger.info(f"OPTION TAKE-PROFIT for {option_symbol_str} at {current_option_premium:.2f} (TP was {self.active_option_position['profit_target']:.2f})")
                    self._close_active_option_position(option_contract_ib, current_option_premium, "TAKE_PROFIT")
                    action_taken = True

            # Check for EOD closure
            if not action_taken and self._is_near_market_close(current_bar_time):
                logger.info(f"EOD CLOSE for option {option_symbol_str} at {current_option_premium:.2f}")
                self._close_active_option_position(option_contract_ib, current_option_premium, "EOD_CLOSE")
                action_taken = True

            if action_taken:
                return # Position closed, wait for next bar

        # 2. Check if new trade can be initiated (within daily limit, no active position)
        if self.daily_trade_count >= self.max_daily_trades:
            # logger.debug(f"Max daily trades ({self.max_daily_trades}) reached for {self.last_trade_date}. No new trades.")
            return
        if self.active_option_position: # Already holding an option
            # logger.debug("Already an active option position. No new trades.")
            return
        if self._is_near_market_close(current_bar_time, minutes_before_close=30): # Too late to open new 0DTE
            # logger.debug("Too close to market EOD to open new 0DTE position.")
            return

        # 3. Calculate QQQ signals (VWAP, EMAs, RSI)
        qqq_history_df = self.live_data_history.get(self.underlying_symbol, pd.DataFrame())
        if len(qqq_history_df) < max(self.vwap_period, self.long_ema_period, self.rsi_period) + 5: # Need enough data for all indicators
            logger.info(f"Not enough QQQ data for signal generation (have {len(qqq_history_df)}, need more).")
            return

        # Calculate Indicators for QQQ (current_price is QQQ's current price)
        # VWAP: (Typical Price * Volume) / Volume, over a period.
        # For simplicity with bar data, let's use typical_price.rolling.sum / volume.rolling.sum
        # A simpler VWAP for bars: (High+Low+Close)/3. This is not a true VWAP.
        # True VWAP needs cumulative (Typical Price * Volume) / cumulative Volume from start of day/period.
        # Let's use a simpler EMA-based proxy or assume VWAP is externally provided if possible.
        # For this implementation, we'll use a rolling mean of typical price as a VWAP proxy.
        tp = (qqq_history_df['High'] + qqq_history_df['Low'] + qqq_history_df['Close']) / 3
        vwap_proxy = tp.rolling(window=self.vwap_period).mean().iloc[-1]

        ema_short_qqq = qqq_history_df['Close'].rolling(window=self.short_ema_period).mean().iloc[-1] # Using SMA as proxy for EMA for simplicity here
        ema_long_qqq = qqq_history_df['Close'].rolling(window=self.long_ema_period).mean().iloc[-1]   # Using SMA as proxy for EMA

        delta_qqq = qqq_history_df['Close'].diff()
        gain_qqq = (delta_qqq.where(delta_qqq > 0, 0.0)).rolling(window=self.rsi_period).mean().iloc[-1]
        loss_qqq = (-delta_qqq.where(delta_qqq < 0, 0.0)).rolling(window=self.rsi_period).mean().iloc[-1]
        rsi_qqq = 100.0 - (100.0 / (1.0 + (gain_qqq / loss_qqq))) if loss_qqq != 0 else (100.0 if gain_qqq > 0 else 50.0)

        logger.debug(f"QQQ Signals: Price={current_price:.2f}, VWAP_proxy={vwap_proxy:.2f}, EMA_S={ema_short_qqq:.2f}, EMA_L={ema_long_qqq:.2f}, RSI={rsi_qqq:.1f}")

        # 4. Determine trade action (Buy Call or Buy Put)
        option_type_to_trade = None
        if current_price > vwap_proxy and ema_short_qqq > ema_long_qqq and rsi_qqq < self.rsi_overbought:
            option_type_to_trade = 'CALL'
            logger.info(f"Signal to BUY CALL on QQQ. Price > VWAP_proxy, EMA_S > EMA_L, RSI not OB.")
        elif current_price < vwap_proxy and ema_short_qqq < ema_long_qqq and rsi_qqq > self.rsi_oversold:
            option_type_to_trade = 'PUT'
            logger.info(f"Signal to BUY PUT on QQQ. Price < VWAP_proxy, EMA_S < EMA_L, RSI not OS.")

        if option_type_to_trade:
            # Select 0DTE option contract (ATM or slightly OTM)
            target_strike = round(current_price / self.strike_offset_otm) * self.strike_offset_otm # Round to nearest offset
            if option_type_to_trade == 'CALL':
                target_strike = current_price + self.strike_offset_otm # Slightly OTM call
            else: # PUT
                target_strike = current_price - self.strike_offset_otm # Slightly OTM put

            # Ensure strike is reasonable (e.g., rounded to nearest $0.5 or $1)
            target_strike = round(target_strike * 2) / 2 # Example: round to nearest $0.50

            option_to_trade = self._find_0dte_option_contract(
                underlying_price=current_price, # QQQ price
                target_strike_approx=target_strike,
                option_type=option_type_to_trade,
                current_bar_time=current_bar_time
            )

            if option_to_trade and option_to_trade.conId: # Ensure we have a valid contract from IBKR
                # Calculate position size for the option
                # Max capital for this trade:
                option_trade_capital = self.live_capital * self.capital_per_option_trade_pct
                # Estimated cost of one contract:
                # Need option_to_trade.premium - this requires fetching its current market price.
                # This is a challenge for backtesting if not already part of option_to_trade.
                # For now, assume _find_0dte_option_contract might populate an estimated premium
                # or we fetch it here.

                # Let's assume _get_current_option_premium can get it.
                current_opt_premium_for_sizing = self._get_current_option_premium(option_to_trade, current_bar_time)

                if current_opt_premium_for_sizing is not None and current_opt_premium_for_sizing > 0:
                    cost_per_contract = current_opt_premium_for_sizing * 100 # Multiplier
                    num_contracts = min(self.max_option_contracts_per_trade, int(option_trade_capital / cost_per_contract))
                    num_contracts = max(1, num_contracts) # Trade at least 1 if affordable

                    if num_contracts > 0 :
                        self._execute_option_trade(option_to_trade, "BUY", num_contracts, current_opt_premium_for_sizing, current_bar_time)
                    else:
                        logger.warning(f"Cannot afford any contracts of {option_to_trade.localSymbol} at premium {current_opt_premium_for_sizing:.2f} with capital limit {option_trade_capital:.2f}")
                else:
                    logger.warning(f"Could not get valid premium for sizing {option_to_trade.localSymbol}. Skipping trade.")
            else:
                logger.warning(f"Could not find suitable 0DTE {option_type_to_trade} option for QQQ at strike ~{target_strike:.2f}.")


    def _find_0dte_option_contract(self, underlying_price: float, target_strike_approx: float, option_type: str, current_bar_time: datetime) -> Optional[Contract]:
        """
        Finds a suitable 0DTE option contract (qualified ib_insync.Option object).
        This involves fetching the option chain for today's expiration and selecting the closest strike.
        """
        if not self.ibkr_client: return None

        today_expiry_str = current_bar_time.strftime('%Y%m%d')

        try:
            # Fetch option chain for QQQ for today's expiry
            # This needs self.ibkr_client to have fetch_option_chain method
            option_chain: List[Contract] = self.ibkr_client.fetch_option_chain(
                underlying_symbol=self.underlying_symbol,
                expiration_date=today_expiry_str
            )

            if not option_chain:
                logger.warning(f"No 0DTE option chain found for {self.underlying_symbol} expiring {today_expiry_str}.")
                return None

            # Filter for the correct type (CALL/PUT) and find closest strike to target_strike_approx
            best_match_option = None
            min_strike_diff = float('inf')

            for opt_contract_ib in option_chain:
                if opt_contract_ib.right == option_type[0]: # 'C' or 'P'
                    strike_diff = abs(opt_contract_ib.strike - target_strike_approx)
                    if strike_diff < min_strike_diff:
                        min_strike_diff = strike_diff
                        best_match_option = opt_contract_ib
                    # If exact match, prefer it (though unlikely with float target_strike_approx)
                    elif strike_diff == min_strike_diff and opt_contract_ib.strike == target_strike_approx:
                         best_match_option = opt_contract_ib

            if best_match_option:
                logger.info(f"Found 0DTE {option_type} for {self.underlying_symbol}: {best_match_option.localSymbol} (Strike: {best_match_option.strike})")
                return best_match_option
            else:
                logger.warning(f"Could not find a suitable 0DTE {option_type} strike near {target_strike_approx} for {self.underlying_symbol} expiring {today_expiry_str}.")
                return None
        except Exception as e:
            logger.error(f"Error finding 0DTE option contract: {e}", exc_info=True)
            return None

    def _get_current_option_premium(self, option_contract_ib: Contract, current_bar_time: datetime) -> Optional[float]:
        """
        Gets the current market premium for a given option contract.
        For backtesting, this is complex. It might involve:
        - Looking up historical option data if available for that exact timestamp.
        - Using a pricing model (e.g., Black-Scholes) if only underlying data and IV are available.
        - For simplicity in this phase, if backtesting, we might try to fetch the 1-min bar close.
          If live, it would be a market data request for the option.
        """
        if not self.ibkr_client: return None

        if self.is_backtesting:
            # Attempt to fetch the most recent bar data for this specific option contract
            # This assumes fetch_historical_data can get very recent (e.g., last minute) data for an option.
            # The barSizeSetting should be small (e.g., '1 min').
            # endDateTime should be current_bar_time.strftime('%Y%m%d %H:%M:%S')
            # durationStr should be short, e.g., '60 S' or covering a few minutes.
            try:
                # We need to fetch a very small window around current_bar_time
                # For simplicity, let's try to get the bar that *ends* at current_bar_time
                # This requires careful handling of endDateTime and duration.
                # A robust way: fetch last N bars and pick the one matching current_bar_time.

                # Fetch last ~5 minutes of 1-min bars for the option
                hist_opt_df = self.ibkr_client.fetch_historical_data(
                    contract=option_contract_ib,
                    endDateTime=current_bar_time.strftime('%Y%m%d %H:%M:%S'), # Data up to this time
                    durationStr='300 S', # Last 5 minutes
                    barSizeSetting='1 min',
                    whatToShow='TRADES', # Or 'MIDPOINT' if trades are sparse
                    useRTH=True
                )
                if hist_opt_df is not None and not hist_opt_df.empty:
                    # Assuming hist_opt_df is indexed by datetime
                    # Get the bar closest to or at current_bar_time
                    # This might need more precise timestamp matching.
                    # For now, take the last available bar's close.
                    # Ensure index is sorted if not already.
                    hist_opt_df = hist_opt_df.sort_index()
                    # Find row closest to current_bar_time (this is simplified)
                    # A better way would be to use hist_opt_df.index.get_loc(current_bar_time, method='ffill' or 'nearest')
                    # but current_bar_time might not be in index.
                    # For 0DTE, the last price in a 1-min bar is a decent estimate.
                    # Ensure the fetched data is not empty and the timestamp is reasonable
                    if not hist_opt_df.empty:
                        # Potentially add a check here: if hist_opt_df.index[-1] is too far from current_bar_time,
                        # it might be stale. For 0DTE, this is critical.
                        # Example: if (current_bar_time - hist_opt_df.index[-1]).total_seconds() > 120: # More than 2 mins old
                        #    logger.warning(f"Fetched option premium for {option_contract_ib.localSymbol} is stale: {hist_opt_df.index[-1]} vs {current_bar_time}")
                        #    return None
                        return hist_opt_df['close'].iloc[-1]
                    else: # hist_opt_df is None or empty
                        logger.warning(f"No recent historical premium found (empty df) for {option_contract_ib.localSymbol} at {current_bar_time}.")
                        return None

                else: # hist_opt_df is None (fetch_historical_data returned None)
                    logger.warning(f"No recent historical premium found (fetch returned None) for {option_contract_ib.localSymbol} at {current_bar_time}.")
                    return None
            except Exception as e:
                logger.error(f"Exception fetching historical premium for {option_contract_ib.localSymbol}: {e}", exc_info=True)
                return None
        else: # Live trading
            # For live trading, you'd request market data (e.g., using reqMktData or reqTickers)
            # This is more complex as it involves handling live ticks.
            # For simplicity in a bar-based live strategy, one might use snapshot data or last trade price.
            # This part needs to be implemented based on how live data is fed for options.
            # Placeholder:
            logger.warning("_get_current_option_premium live mode not fully implemented. Needs live market data fetch for option.")
            # Try to get a snapshot (less ideal for fast scalping but simpler)
            try:
                # This might not be fast enough for true scalping.
                # reqMktData with streaming updates is better.
                ticker = self.ibkr_client.ib.reqMktData(option_contract_ib, '', True, False, [])
                self.ibkr_client.ib.sleep(0.5) # Allow time for snapshot to arrive (can be non-blocking with proper event handling)

                premium = np.nan
                if ticker.last != np.nan and ticker.last > 0: premium = ticker.last
                elif ticker.close != np.nan and ticker.close > 0 : premium = ticker.close # Previous day's close if last is not available
                elif ticker.bid != np.nan and ticker.ask != np.nan and ticker.bid > 0 and ticker.ask > 0:
                    premium = (ticker.bid + ticker.ask) / 2 # Midpoint

                self.ibkr_client.ib.cancelMktData(option_contract_ib) # Clean up snapshot request

                if premium != np.nan and premium > 0: return premium
                logger.warning(f"Live premium snapshot failed or invalid for {option_contract_ib.localSymbol}")
                return None
            except Exception as e:
                 logger.error(f"Error fetching live option premium snapshot for {option_contract_ib.localSymbol}: {e}")
                 return None


    def _execute_option_trade(self, option_contract_ib: Contract, action: str, quantity: int, entry_premium: float, current_bar_time: datetime):
        """Executes an option trade (simulated for backtest, real for live)."""
        if quantity <=0: return

        option_symbol_str = option_contract_ib.localSymbol
        order_action = action.upper() # 'BUY' or 'SELL' (to close)

        # Calculate SL and TP prices based on entry premium
        if order_action == 'BUY':
            stop_price = entry_premium * (1 - self.option_stop_loss_pct)
            profit_target = entry_premium * (1 + self.option_take_profit_pct)
        else: # SELL to close, no SL/TP needed for this execution step
            stop_price = 0
            profit_target = 0

        # Slippage & Transaction Costs for options
        simulated_fill_premium = entry_premium
        if self.is_backtesting:
            if order_action == 'BUY':
                simulated_fill_premium += self.option_slippage_per_contract
            else: # SELL
                simulated_fill_premium -= self.option_slippage_per_contract
            simulated_fill_premium = max(0.01, simulated_fill_premium) # Premium can't be negative or zero

        transaction_cost = self.option_transaction_cost_per_contract * quantity

        self.live_capital -= transaction_cost # Deduct transaction cost first

        trade_value = simulated_fill_premium * quantity * 100 # Option multiplier

        if self.is_backtesting:
            logger.info(f"BACKTEST_SIM: {order_action} {quantity} contract(s) of {option_symbol_str} at {simulated_fill_premium:.2f} (Slippage/Cost Adj). Cost: {transaction_cost:.2f}")

            if order_action == 'BUY':
                self.live_capital -= trade_value # Cost of buying options
                self.active_option_position = {
                    'contract_obj': option_contract_ib, # Store the ib_insync.Option object
                    'entry_premium': simulated_fill_premium,
                    'quantity': quantity,
                    'stop_price': stop_price,
                    'profit_target': profit_target,
                    'entry_time': current_bar_time,
                    'type': option_contract_ib.right # 'C' or 'P'
                }
                self.daily_trade_count +=1 # Count as one leg of a round trip

            self.trade_log.append({
                'timestamp': current_bar_time, 'symbol': option_symbol_str,
                'underlying_at_trade': self.live_data_history[self.underlying_symbol]['Close'].iloc[-1], # QQQ price at time of option trade
                'action': f"{order_action}_{option_contract_ib.right}", # e.g. BUY_CALL
                'size': quantity, 'price': simulated_fill_premium, 'status': 'Filled',
                'reason': 'SIGNAL_ENTRY_OPTION', 'transaction_cost': transaction_cost,
                'pnl': 0, # PNL is 0 for entry
                'capital_after_trade': self.live_capital
            })

        else: # Live Trading
            ib_order = MarketOrder(order_action, quantity) # Market order for options
            logger.info(f"LIVE: Attempting to {order_action} {quantity} of {option_symbol_str} at market.")
            try:
                trade = self.ibkr_client.place_paper_order(option_contract_ib, ib_order)
                if trade and trade.orderStatus:
                    # Actual fill price would come from trade.orderStatus.avgFillPrice or fills
                    # For simplicity, using entry_premium (which should be current market for live)
                    # This needs robust fill handling in a real live system.
                    actual_fill_premium = entry_premium # Placeholder for actual fill
                    logger.info(f"LIVE: Paper order for {option_symbol_str} placed. Status: {trade.orderStatus.status}, OrderId: {trade.order.orderId}. Approx Fill: {actual_fill_premium:.2f}")

                    if order_action == 'BUY':
                        self.live_capital -= (actual_fill_premium * quantity * 100)
                        self.active_option_position = {
                            'contract_obj': option_contract_ib,
                            'entry_premium': actual_fill_premium,
                            'quantity': quantity,
                            'stop_price': stop_price,
                            'profit_target': profit_target,
                            'entry_time': datetime.now(),
                            'type': option_contract_ib.right,
                            'order_id': trade.order.orderId # Store order ID
                        }
                        self.daily_trade_count +=1

                    self.trade_log.append({
                        'timestamp': datetime.now(), 'symbol': option_symbol_str,
                        'underlying_at_trade': self.live_data_history[self.underlying_symbol]['Close'].iloc[-1],
                        'action': f"{order_action}_{option_contract_ib.right}",
                        'size': quantity, 'price': actual_fill_premium, 'status': trade.orderStatus.status,
                        'order_id': trade.order.orderId, 'reason': 'SIGNAL_ENTRY_OPTION',
                        'transaction_cost': transaction_cost, 'pnl': 0,
                        'capital_after_trade': self.live_capital
                    })
                else:
                    logger.error(f"LIVE: Failed to place paper order for {option_symbol_str} or trade object invalid.")
            except Exception as e:
                logger.error(f"LIVE: Exception placing paper order for {option_symbol_str}: {e}", exc_info=True)


    def _close_active_option_position(self, option_contract_ib: Contract, closing_premium: float, reason: str):
        """Closes the currently active option position."""
        if not self.active_option_position:
            logger.warning("Request to close option position, but no active position found.")
            return

        pos_info = self.active_option_position
        option_symbol_str = option_contract_ib.localSymbol
        quantity_to_close = pos_info['quantity']
        entry_premium = pos_info['entry_premium']
        option_type = pos_info['type'] # 'C' or 'P'

        order_action = 'SELL' # Always selling to close a bought option

        # Slippage & Transaction Costs
        simulated_closing_premium = closing_premium
        if self.is_backtesting:
            simulated_closing_premium -= self.option_slippage_per_contract # Selling, so price might be worse
            simulated_closing_premium = max(0.01, simulated_closing_premium)

        transaction_cost = self.option_transaction_cost_per_contract * quantity_to_close
        self.live_capital -= transaction_cost

        # Calculate P&L for the option trade
        pnl_per_contract = (simulated_closing_premium - entry_premium)
        total_pnl = pnl_per_contract * quantity_to_close * 100 # Option multiplier

        self.live_capital += (simulated_closing_premium * quantity_to_close * 100) # Add proceeds from selling options
        self.live_capital += total_pnl # This seems redundant if capital was already adjusted for proceeds.
                                       # Let's adjust: self.live_capital was reduced by entry cost, now add exit proceeds.
                                       # P&L is (exit_value - entry_value) - costs
                                       # Entry: Capital -= entry_value + entry_cost
                                       # Exit: Capital += exit_value - exit_cost
                                       # Net PNL: exit_value - entry_value - entry_cost - exit_cost
                                       # The capital adjustment is:
                                       # self.live_capital += (simulated_closing_premium * quantity_to_close * 100) # Add back proceeds
                                       # The total_pnl is for logging.
                                       # Let's re-verify capital calculation logic.
                                       # Initial buy: capital -= (entry_premium * qty * 100) - entry_tx_cost
                                       # Close sell: capital += (simulated_closing_premium * qty * 100) - close_tx_cost
                                       # The self.live_capital -= transaction_cost (for closing) is correct.
                                       # The self.live_capital += (simulated_closing_premium * quantity_to_close * 100) is also correct for proceeds.
                                       # So total_pnl is just for the log.

        current_event_time = self.current_backtest_time if self.is_backtesting else datetime.now()

        if self.is_backtesting:
            logger.info(f"BACKTEST_SIM: CLOSE {order_action} {quantity_to_close} of {option_symbol_str} at {simulated_closing_premium:.2f}. Reason: {reason}. P&L: {total_pnl:.2f}")
            self.trade_log.append({
                'timestamp': current_event_time, 'symbol': option_symbol_str,
                'underlying_at_trade': self.live_data_history[self.underlying_symbol]['Close'].iloc[-1],
                'action': f"{order_action}_{option_type}", # e.g. SELL_CALL
                'size': quantity_to_close, 'price': simulated_closing_premium, 'status': 'Filled',
                'reason': reason, 'transaction_cost': transaction_cost, 'pnl': total_pnl,
                'capital_after_trade': self.live_capital, 'entry_premium_ref': entry_premium
            })
        else: # Live Trading
            ib_order = MarketOrder(order_action, quantity_to_close)
            logger.info(f"LIVE: Attempting to CLOSE {order_action} {quantity_to_close} of {option_symbol_str}. Reason: {reason}")
            try:
                trade = self.ibkr_client.place_paper_order(option_contract_ib, ib_order)
                if trade and trade.orderStatus:
                    actual_closing_premium = closing_premium # Placeholder for actual fill
                    logger.info(f"LIVE: Paper CLOSE order for {option_symbol_str} placed. Status: {trade.orderStatus.status}, OrderId: {trade.order.orderId}. Approx Fill: {actual_closing_premium:.2f}")
                    # P&L calculation based on actual fill would be needed here from IBKR fills.
                    self.trade_log.append({
                        'timestamp': current_event_time, 'symbol': option_symbol_str,
                        'underlying_at_trade': self.live_data_history[self.underlying_symbol]['Close'].iloc[-1],
                        'action': f"{order_action}_{option_type}",
                        'size': quantity_to_close, 'price': actual_closing_premium, 'status': trade.orderStatus.status,
                        'order_id': trade.order.orderId, 'reason': reason,
                        'transaction_cost': transaction_cost, 'pnl': total_pnl, # total_pnl here is based on simulated_closing_premium
                        'capital_after_trade': self.live_capital, 'entry_premium_ref': entry_premium
                    })
                else:
                    logger.error(f"LIVE: Failed to place paper CLOSE order for {option_symbol_str}.")
            except Exception as e:
                logger.error(f"LIVE: Exception placing paper CLOSE order for {option_symbol_str}: {e}", exc_info=True)

        self.active_option_position = None # Clear active position
        # self.daily_trade_count does not increment on close, only on open.
        # If a round trip is defined as buy AND sell, then this close completes one trade.
        # The current self.daily_trade_count increments on BUY. If it's meant for round trips,
        # it should increment here or be handled differently. For now, it counts opening trades.


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