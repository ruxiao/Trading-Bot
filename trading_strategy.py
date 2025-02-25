import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
import random

class TradingStrategy:
    def __init__(self, alpha_params: List[float], 
                 transaction_cost: float = 0.0005,      # 5 basis points
                 slippage: float = 0.0003,              # 3 basis points
                 risk_per_trade: float = 0.02,          # 2% risk per trade
                 max_position_size: float = 0.2,        # 20% max position size
                 stop_loss_pct: float = 0.02,           # 2% stop loss
                 take_profit_pct: float = 0.03,         # 3% take profit
                 volatility_scaling: bool = True):      # Use volatility-based position sizing
        # Ensure MA windows are valid integers
        self.short_window = max(2, int(round(alpha_params[0])))
        self.long_window = max(2, int(round(alpha_params[1])))
        self.alpha_params = [
            self.short_window,
            self.long_window,
            alpha_params[2],  # MA signal weight
            alpha_params[3],  # RSI oversold weight
            alpha_params[4],  # RSI overbought weight
            alpha_params[5]   # Correlation weight
        ]
        self.position_limits = (-1, 1)  # Allow both short (-1) and long (1) positions
        
        # Real trading constraints
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.volatility_scaling = volatility_scaling
        
        # Live trading support
        self.in_live_mode = False
        self.current_positions = {}  # Stores current positions for live trading

    def calculate_position_size(self, symbol: str, price: float, volatility: float, account_value: float) -> float:
        """
        Calculate position size based on risk management rules
        """
        # Base position size on risk per trade and volatility
        if self.volatility_scaling:
            # Adjust position size inversely to volatility (higher vol = smaller position)
            risk_factor = self.risk_per_trade / (volatility * 2)  # Assuming 2 std dev move
        else:
            risk_factor = self.risk_per_trade
        
        # Calculate dollar risk amount
        risk_amount = account_value * risk_factor
        
        # Calculate number of shares based on stop loss
        shares = risk_amount / (price * self.stop_loss_pct)
        
        # Calculate position value
        position_value = shares * price
        
        # Cap at maximum position size
        max_value = account_value * self.max_position_size
        if position_value > max_value:
            position_value = max_value
            shares = max_value / price
            
        return shares

    def apply_market_impact(self, price: float, direction: int, volume: float) -> float:
        """
        Apply slippage and market impact based on trade size and direction
        """
        # Simulate slippage: bigger trades have bigger impact
        impact = self.slippage * (1 + random.random())  # Add some randomness
        
        # Direction matters: buys get worse prices, sells get worse prices
        if direction > 0:  # Buying
            executed_price = price * (1 + impact)
        else:  # Selling
            executed_price = price * (1 - impact)
            
        return executed_price

    def calculate_signal(self, data: pd.DataFrame, correlation_matrix: pd.DataFrame, 
                        symbol: str, account_value: float = 100000) -> Tuple:
        """
        Calculate trading signals with risk management and realistic execution
        Returns signals, alpha values, position sizes, and trade log
        """
        signals = np.zeros(len(data))
        alpha_values = np.zeros(len(data))
        position_sizes = np.zeros(len(data))
        trade_log = []
        
        # Track current position and entry price for stop loss and take profit
        current_position = 0
        entry_price = 0
        stop_price = 0
        take_profit_price = 0

        # Calculate technical indicators with validated windows
        ma_short = data['Close'].rolling(window=self.short_window).mean()
        ma_long = data['Close'].rolling(window=self.long_window).mean()
        
        # Calculate 20-day volatility for position sizing
        volatility = data['Close'].pct_change().rolling(window=20).std()

        # MACD indicator 
        macd = ma_short - ma_long
        macd_signal = macd.rolling(window=9).mean()
        macd_hist = macd - macd_signal

        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Handle NaN values in indicators
        rsi = rsi.fillna(50)  # Fill NaN with neutral RSI value
        volatility = volatility.fillna(0.01)  # Default to 1% volatility when unknown
        
        # Calculate volume profile and liquidity
        avg_volume = data['Volume'].rolling(window=20).mean() if 'Volume' in data.columns else pd.Series(1000000, index=data.index)
        
        # Add volatility breakout indicator
        atr = self.calculate_atr(data, period=14)
        high_channel = data['Close'].rolling(window=20).max()
        low_channel = data['Close'].rolling(window=20).min()
        
        # Calculate correlation component for diversification
        correlations = correlation_matrix[symbol].drop(symbol)
        avg_correlation = correlations.mean()
        correlation_signal = -avg_correlation  # Negative correlation is good for diversification

        # Enhanced alpha formula with additional indicators
        ma_component = (ma_short - ma_long) / ma_long
        rsi_oversold = (30 - rsi) / 30
        rsi_overbought = (rsi - 70) / 30
        
        # Breakout component
        breakout_component = ((data['Close'] - low_channel) / (high_channel - low_channel) - 0.5) * 2
        
        # Volume factor - higher volume gives stronger signals
        volume_factor = avg_volume / avg_volume.rolling(window=60).mean()
        volume_factor = volume_factor.fillna(1)
        
        # Handle NaN values in alpha calculation
        ma_component = ma_component.fillna(0)
        breakout_component = breakout_component.fillna(0)
        
        # Combined alpha formula with multiple return drivers
        alpha_values = (
            self.alpha_params[2] * ma_component +
            self.alpha_params[3] * rsi_oversold +
            self.alpha_params[4] * rsi_overbought +
            self.alpha_params[5] * correlation_signal +
            (np.sign(macd_hist) * 0.2) +  # Add MACD component
            (breakout_component * 0.15)   # Add breakout component
        )
        
        # Apply volume-based signal strength adjustment
        alpha_values = alpha_values * np.clip(volume_factor, 0.5, 1.5)

        # Track positions, applying stop-loss and take-profit
        for i in range(max(self.short_window, self.long_window), len(data)):
            price = data['Close'].iloc[i]
            current_volatility = volatility.iloc[i]
            
            # Check stop loss and take profit
            if current_position != 0:
                # Check if stop loss hit
                if (current_position > 0 and price <= stop_price) or \
                   (current_position < 0 and price >= stop_price):
                    # Stop loss hit - close position
                    executed_price = self.apply_market_impact(price, -current_position, position_sizes[i-1])
                    trade_cost = executed_price * position_sizes[i-1] * self.transaction_cost
                    
                    trade_log.append({
                        'date': data.index[i],
                        'symbol': symbol,
                        'price': executed_price,
                        'size': position_sizes[i-1],
                        'cost': trade_cost,
                        'action': 'STOP_LOSS',
                        'alpha_value': alpha_values[i],
                        'ma_short': ma_short.iloc[i],
                        'ma_long': ma_long.iloc[i],
                        'rsi': rsi.iloc[i],
                        'correlation': correlation_signal,
                        'volatility': current_volatility
                    })
                    
                    signals[i] = 0
                    position_sizes[i] = 0
                    current_position = 0
                    continue
                    
                # Check if take profit hit
                elif (current_position > 0 and price >= take_profit_price) or \
                     (current_position < 0 and price <= take_profit_price):
                    # Take profit hit - close position
                    executed_price = self.apply_market_impact(price, -current_position, position_sizes[i-1])
                    trade_cost = executed_price * position_sizes[i-1] * self.transaction_cost
                    
                    trade_log.append({
                        'date': data.index[i],
                        'symbol': symbol,
                        'price': executed_price,
                        'size': position_sizes[i-1],
                        'cost': trade_cost,
                        'action': 'TAKE_PROFIT',
                        'alpha_value': alpha_values[i],
                        'ma_short': ma_short.iloc[i],
                        'ma_long': ma_long.iloc[i],
                        'rsi': rsi.iloc[i],
                        'correlation': correlation_signal,
                        'volatility': current_volatility
                    })
                    
                    signals[i] = 0
                    position_sizes[i] = 0
                    current_position = 0
                    continue
            
            # Calculate new signal based on alpha value
            new_signal = np.clip(np.sign(alpha_values[i]), self.position_limits[0], self.position_limits[1])
            
            # Position changes only on significant signal difference
            signal_changed = False
            
            # Rules for position changes:
            # 1. If no position, take new position if signal is strong enough
            if current_position == 0 and abs(alpha_values[i]) > 0.2:
                signal_changed = True
            # 2. If current position, only reverse on strong opposite signal
            elif (current_position > 0 and alpha_values[i] < -0.3) or \
                 (current_position < 0 and alpha_values[i] > 0.3):
                signal_changed = True
            # 3. Scale out if signal weakens substantially
            elif (current_position > 0 and alpha_values[i] < 0.1) or \
                 (current_position < 0 and alpha_values[i] > -0.1):
                signal_changed = True
            
            if signal_changed:
                # Calculate proper position size based on volatility and risk
                new_size = self.calculate_position_size(
                    symbol, price, current_volatility, account_value
                )
                
                # Apply weighting by alpha strength
                new_size = new_size * min(1.0, abs(alpha_values[i]) * 2)
                
                # Adjust for direction
                if new_signal < 0:
                    new_size = -new_size
                
                # Model execution with realistic price
                executed_price = self.apply_market_impact(price, int(new_signal), new_size)
                
                # Apply transaction costs
                trade_cost = abs(executed_price * new_size) * self.transaction_cost
                
                # Set stop loss and take profit levels
                if new_signal != 0:
                    entry_price = executed_price
                    if new_signal > 0:  # Long position
                        stop_price = entry_price * (1 - self.stop_loss_pct)
                        take_profit_price = entry_price * (1 + self.take_profit_pct)
                    else:  # Short position
                        stop_price = entry_price * (1 + self.stop_loss_pct)
                        take_profit_price = entry_price * (1 - self.take_profit_pct)
                
                # Log the trade
                trade_log.append({
                    'date': data.index[i],
                    'symbol': symbol,
                    'price': executed_price,
                    'size': new_size,
                    'cost': trade_cost,
                    'action': 'BUY' if new_signal > 0 else 'SELL' if new_signal < 0 else 'CLOSE',
                    'alpha_value': alpha_values[i],
                    'ma_short': ma_short.iloc[i],
                    'ma_long': ma_long.iloc[i],
                    'rsi': rsi.iloc[i],
                    'correlation': correlation_signal,
                    'volatility': current_volatility,
                    'stop_price': stop_price,
                    'take_profit': take_profit_price
                })
                
                # Update tracking variables
                signals[i] = new_signal
                position_sizes[i] = new_size
                current_position = new_signal
            else:
                # Maintain previous position
                signals[i] = signals[i-1] if i > 0 else 0
                position_sizes[i] = position_sizes[i-1] if i > 0 else 0

        return signals, alpha_values, position_sizes, trade_log

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility measurement"""
        high = data['High'] if 'High' in data.columns else data['Close'] * 1.005
        low = data['Low'] if 'Low' in data.columns else data['Close'] * 0.995
        close = data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr

    def backtest_portfolio(self, data_dict: dict, correlation_matrix: pd.DataFrame, 
                         initial_capital: float = 100000) -> Dict:
        """
        Backtest the trading strategy across multiple assets with realistic constraints
        Returns performance metrics and trade log
        """
        all_signals = {}
        all_position_sizes = {}
        all_returns = pd.DataFrame()
        combined_trade_log = []
        
        # Track portfolio value through time
        equity_curve = pd.Series(initial_capital, index=next(iter(data_dict.values())).index)
        cash = initial_capital
        positions = {}
        
        # Risk allocation to each asset - dynamic through time
        num_assets = len(data_dict)
        allocation_per_asset = 1.0 / num_assets
        
        # Calculate volatility for risk-adjusted position sizing
        asset_volatility = {}
        for symbol, data in data_dict.items():
            asset_volatility[symbol] = data['Close'].pct_change().rolling(window=20).std().fillna(0.01)
        
        # Sort by date to ensure chronological processing
        common_dates = sorted(set.intersection(*[set(data.index) for data in data_dict.values()]))
        
        # Calculate signals for each asset
        for symbol, data in data_dict.items():
            signals, alpha_values, position_sizes, trade_log = self.calculate_signal(
                data, correlation_matrix, symbol, initial_capital * allocation_per_asset
            )
            
            all_signals[symbol] = signals
            all_position_sizes[symbol] = position_sizes
            combined_trade_log.extend(trade_log)
        
        # Process trades chronologically and track portfolio performance
        daily_returns = []
        daily_positions = []
        daily_exposures = []
        
        # Group trades by date
        trade_by_date = {}
        for trade in combined_trade_log:
            date = trade['date']
            if date not in trade_by_date:
                trade_by_date[date] = []
            trade_by_date[date].append(trade)
        
        # Process each day
        for date in common_dates:
            # Process trades for this date
            if date in trade_by_date:
                for trade in trade_by_date[date]:
                    symbol = trade['symbol']
                    action = trade['action']
                    price = trade['price']
                    size = trade['size']
                    cost = trade['cost']
                    
                    # Update positions
                    if symbol not in positions:
                        positions[symbol] = {'size': 0, 'cost_basis': 0}
                    
                    # Calculate trade value
                    trade_value = price * abs(size)
                    
                    # Apply transaction costs
                    cash -= cost
                    
                    if action == 'BUY':
                        # Buying shares
                        cash -= trade_value
                        positions[symbol]['size'] += size
                        positions[symbol]['cost_basis'] = price
                    elif action == 'SELL':
                        # Selling shares
                        cash += trade_value
                        positions[symbol]['size'] += size  # size is negative for selling
                        positions[symbol]['cost_basis'] = price
                    elif action in ['CLOSE', 'STOP_LOSS', 'TAKE_PROFIT']:
                        # Close position
                        cash += trade_value * np.sign(positions[symbol]['size'])
                        positions[symbol]['size'] = 0
            
            # Calculate portfolio value at end of day
            portfolio_value = cash
            position_values = {}
            
            for symbol, position in positions.items():
                if position['size'] != 0:
                    # Get closing price for the day
                    if date in data_dict[symbol].index:
                        close_price = data_dict[symbol].loc[date, 'Close']
                        position_value = close_price * position['size']
                        portfolio_value += position_value
                        position_values[symbol] = position_value
            
            # Record daily portfolio value
            equity_curve[date] = portfolio_value
            
            # Calculate daily returns
            if date != common_dates[0]:
                prev_date = common_dates[common_dates.index(date) - 1]
                daily_return = (equity_curve[date] / equity_curve[prev_date]) - 1
                daily_returns.append(daily_return)
                
                # Record positions and exposures
                daily_positions.append(positions.copy())
                
                # Calculate exposure ratio (invested / total capital)
                invested = sum(abs(val) for val in position_values.values())
                exposure_ratio = invested / portfolio_value if portfolio_value > 0 else 0
                daily_exposures.append(exposure_ratio)
        
        # Convert to Series for analysis
        daily_returns_series = pd.Series(daily_returns, index=common_dates[1:])
        daily_exposures_series = pd.Series(daily_exposures, index=common_dates[1:])
        
        # Calculate portfolio metrics
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = ((1 + total_return) ** (252 / len(daily_returns_series))) - 1
        volatility = daily_returns_series.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility != 0 else 0  # Assuming 2% risk-free rate
        
        # Calculate drawdown
        cum_returns = (1 + daily_returns_series).cumprod()
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns / running_max) - 1
        max_drawdown = drawdowns.min()
        
        # Advanced metrics
        winning_days = len(daily_returns_series[daily_returns_series > 0])
        losing_days = len(daily_returns_series[daily_returns_series < 0])
        win_rate = winning_days / (winning_days + losing_days) if (winning_days + losing_days) > 0 else 0
        
        avg_win = daily_returns_series[daily_returns_series > 0].mean() if len(daily_returns_series[daily_returns_series > 0]) > 0 else 0
        avg_loss = daily_returns_series[daily_returns_series < 0].mean() if len(daily_returns_series[daily_returns_series < 0]) > 0 else 0
        profit_factor = abs(daily_returns_series[daily_returns_series > 0].sum() / daily_returns_series[daily_returns_series < 0].sum()) if daily_returns_series[daily_returns_series < 0].sum() != 0 else float('inf')
        
        # Calculate turnover and average exposure
        avg_exposure = daily_exposures_series.mean()

        return {
            'total_return': total_return,
            'annual_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_exposure': avg_exposure,
            'returns': daily_returns_series,
            'equity_curve': equity_curve,
            'drawdowns': drawdowns,
            'trade_log': combined_trade_log,
            'final_capital': equity_curve.iloc[-1],
            'positions': daily_positions
        }
        
    def start_live_trading(self, initial_capital: float = 100000):
        """
        Initialize for live trading
        """
        self.in_live_mode = True
        self.live_capital = initial_capital
        self.live_positions = {}
        self.trade_history = []
        
    def process_live_data(self, current_data: dict, correlation_matrix: pd.DataFrame):
        """
        Process latest market data and generate live trading signals
        """
        if not self.in_live_mode:
            raise ValueError("Must call start_live_trading before processing live data")
            
        signals = {}
        
        # Process signals for each asset
        for symbol, data in current_data.items():
            # Use the most recent data points to calculate signals
            latest_close = data['Close'].iloc[-1]
            latest_alpha = self._calculate_live_alpha(data, correlation_matrix, symbol)
            
            # Determine signal direction
            signal = np.clip(np.sign(latest_alpha), self.position_limits[0], self.position_limits[1])
            signals[symbol] = {
                'signal': signal,
                'alpha': latest_alpha,
                'price': latest_close,
                'timestamp': data.index[-1]
            }
            
            # Check for stop loss/take profit in existing positions
            if symbol in self.live_positions:
                position = self.live_positions[symbol]
                
                # Check stop loss
                if position['size'] > 0 and latest_close <= position['stop_price']:
                    signals[symbol]['action'] = 'STOP_LOSS'
                elif position['size'] < 0 and latest_close >= position['stop_price']:
                    signals[symbol]['action'] = 'STOP_LOSS'
                # Check take profit
                elif position['size'] > 0 and latest_close >= position['take_profit']:
                    signals[symbol]['action'] = 'TAKE_PROFIT'
                elif position['size'] < 0 and latest_close <= position['take_profit']:
                    signals[symbol]['action'] = 'TAKE_PROFIT'
                else:
                    # Calculate normal action based on signal change
                    current_position = np.sign(position['size'])
                    if signal != current_position:
                        if signal == 0:
                            signals[symbol]['action'] = 'CLOSE'
                        else:
                            signals[symbol]['action'] = 'REVERSE'
                    else:
                        signals[symbol]['action'] = 'HOLD'
            else:
                # New position
                if signal != 0:
                    signals[symbol]['action'] = 'ENTER'
                else:
                    signals[symbol]['action'] = 'NONE'
                    
            # Calculate position size if we're entering or reversing
            if signals[symbol]['action'] in ['ENTER', 'REVERSE']:
                # Get current volatility
                volatility = data['Close'].pct_change().rolling(window=20).std().iloc[-1]
                
                # Calculate position size
                new_size = self.calculate_position_size(
                    symbol, latest_close, volatility, self.live_capital
                )
                
                if signal < 0:
                    new_size = -new_size
                    
                signals[symbol]['size'] = new_size
                
                # Calculate stop loss and take profit prices
                if signal > 0:  # Long
                    stop_price = latest_close * (1 - self.stop_loss_pct)
                    take_profit = latest_close * (1 + self.take_profit_pct)
                else:  # Short
                    stop_price = latest_close * (1 + self.stop_loss_pct)
                    take_profit = latest_close * (1 - self.take_profit_pct)
                    
                signals[symbol]['stop_price'] = stop_price
                signals[symbol]['take_profit'] = take_profit
                
        return signals
    
    def _calculate_live_alpha(self, data: pd.DataFrame, correlation_matrix: pd.DataFrame, symbol: str) -> float:
        """Calculate alpha value for live trading based on latest data"""
        # Calculate indicators
        ma_short = data['Close'].rolling(window=self.short_window).mean().iloc[-1]
        ma_long = data['Close'].rolling(window=self.long_window).mean().iloc[-1]
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 9999  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Correlation
        correlations = correlation_matrix[symbol].drop(symbol)
        avg_correlation = correlations.mean()
        correlation_signal = -avg_correlation
        
        # Alpha components
        ma_component = (ma_short - ma_long) / ma_long if ma_long != 0 else 0
        rsi_oversold = (30 - rsi) / 30 if rsi < 30 else 0
        rsi_overbought = (rsi - 70) / 30 if rsi > 70 else 0
        
        # Combined alpha
        alpha = (
            self.alpha_params[2] * ma_component +
            self.alpha_params[3] * rsi_oversold +
            self.alpha_params[4] * rsi_overbought +
            self.alpha_params[5] * correlation_signal
        )
        
        return alpha