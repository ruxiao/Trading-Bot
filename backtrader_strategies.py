import backtrader as bt
import random # Ensure this is at the top of the file
from datetime import datetime, timedelta # Ensure timedelta is imported
import numpy as np

# Imports for the new strategy
from alternative_data_feed import AlternativeDataFeed, DarkPoolData, OptionsData
from option_contract import OptionContract


class EMA20Strategy(bt.Strategy):
    params = (
        ('ema_period', 20),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add ExponentialMovingAverage indicator
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.ema_period
        )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Log a few key values (ensure this is active for debugging if needed, otherwise comment out for cleaner run)
        # self.log(f'Date: {self.datas[0].datetime.date(0)}, Close: {self.dataclose[0]:.2f}, EMA: {self.ema[0]:.2f}, Position: {self.position.size if self.position else 0}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in the market, look for a signal to enter
            if self.dataclose[0] > self.ema[0]:
                self.log(f'BUY CREATE, Close: {self.dataclose[0]:.2f}, EMA: {self.ema[0]:.2f}')
                self.order = self.buy()
        else:
            # Already in the market, look for a signal to sell
            if self.dataclose[0] < self.ema[0]:
                self.log(f'SELL CREATE, Close: {self.dataclose[0]:.2f}, EMA: {self.ema[0]:.2f}')
                self.order = self.sell()

class MACrossoverStrategy(bt.Strategy):
    params = (
        ('short_ema_period', 50),
        ('long_ema_period', 200),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add short and long ExponentialMovingAverage indicators
        self.short_ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.short_ema_period
        )
        self.long_ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.long_ema_period
        )

        # Crossover signal
        self.crossover = bt.indicators.CrossOver(self.short_ema, self.long_ema)

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position: # Not in the market
            if self.crossover > 0: # If short EMA crosses above long EMA
                self.log(f'BUY CREATE (MA Crossover), {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else: # Already in the market
            if self.crossover < 0: # If short EMA crosses below long EMA
                self.log(f'SELL CREATE (MA Crossover), {self.dataclose[0]:.2f}')
                self.order = self.sell()

class RSIStrategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add RSI indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            period=self.params.rsi_period
        )

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position: # Not in the market
            if self.rsi < self.params.rsi_oversold:
                self.log(f'BUY CREATE (RSI Oversold), {self.dataclose[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                self.order = self.buy()
        else: # Already in the market
            if self.rsi > self.params.rsi_overbought:
                self.log(f'SELL CREATE (RSI Overbought), {self.dataclose[0]:.2f}, RSI: {self.rsi[0]:.2f}')
                self.order = self.sell()

class EMACrossoverWithSentimentFilter(bt.Strategy):
    params = (
        ('short_ema_period', 20),
        ('long_ema_period', 50),
        ('sentiment_ema_period', 10), # For smoothing random sentiment
        ('sentiment_buy_threshold', 0.15), # Buy if sentiment > this
        ('sentiment_sell_threshold', -0.15), # Sell if sentiment < this (for closing longs)
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None

        # Technical indicators
        self.short_ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.short_ema_period
        )
        self.long_ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0], period=self.params.long_ema_period
        )
        self.crossover = bt.indicators.CrossOver(self.short_ema, self.long_ema)

        # Simulated sentiment data
        class SimulatedSentiment(bt.Indicator):
            lines = ('sentiment',)
            params = (('period', 10),)

            def __init__(self):
                self.addminperiod(1)
                self.line.sentiment = 0

            def next(self):
                if len(self) == 1:
                    self.lines.sentiment[0] = random.uniform(-0.5, 0.5)
                else:
                    move = random.uniform(-0.2, 0.2)
                    self.lines.sentiment[0] = max(-1, min(1, self.lines.sentiment[-1] + move))

        self.sim_sentiment_raw = SimulatedSentiment()
        self.sentiment_ema = bt.indicators.ExponentialMovingAverage(
            self.sim_sentiment_raw, period=self.params.sentiment_ema_period
        )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
            else:  # Sell
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return
        current_sentiment = self.sentiment_ema[0]
        # self.log(f'Date: {self.datas[0].datetime.date(0)}, Close: {self.dataclose[0]:.2f}, SentEMA: {current_sentiment:.2f}, Cross: {self.crossover[0]}')

        if not self.position:
            if self.crossover > 0:
                if current_sentiment > self.params.sentiment_buy_threshold:
                    self.log(f'BUY CREATE (EMA Cross + Sentiment), Close: {self.dataclose[0]:.2f}, Sentiment: {current_sentiment:.2f}')
                    self.order = self.buy()
                else:
                    self.log(f'Buy signal (EMA Cross) ignored due to low sentiment: {current_sentiment:.2f}')
        else:
            if self.crossover < 0:
                if current_sentiment < self.params.sentiment_sell_threshold:
                    self.log(f'SELL CREATE (EMA Cross + Sentiment), Close: {self.dataclose[0]:.2f}, Sentiment: {current_sentiment:.2f}')
                    self.order = self.sell()
                else:
                    self.log(f'Sell signal (EMA Cross) ignored due to non-negative sentiment: {current_sentiment:.2f}')

# ... (Keep existing strategies: EMA20Strategy, MACrossoverStrategy, RSIStrategy, EMACrossoverWithSentimentFilter) ...

class AlternativeDataStrategy(bt.Strategy):
    '''
    实盘级另类数据驱动策略 - 支持股票和期权交易
    '''
    params = (
        ('symbol', 'TSLA'), # Default symbol, will be overridden by run_backtest.py
        ('ema_short', 20),
        ('ema_long', 50),
        ('dark_pool_threshold', 0.18),    # 基于历史数据的真实阈值
        ('pcr_buy_threshold', 1.7),       # PCR极端悲观买入阈值
        ('pcr_sell_threshold', 0.6),      # PCR极端乐观卖出阈值
        ('vol_surface_threshold', 35),     # 波动率曲面高阈值
        ('iv_skew_threshold', 1.25),      # IV偏斜阈值
        ('gamma_exposure_threshold', 0.1), # 伽玛敞口阈值
        ('max_option_position', 0.3),     # 期权最大仓位比例
    )

    def __init__(self):
        # 核心价格指标
        # For custom data feeds, access lines via self.datas[0].lines.your_line_name or self.datas[0].your_line_name
        # self.data refers to self.datas[0]
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.params.ema_short)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.params.ema_long)
        self.crossover = bt.indicators.CrossOver(self.ema_short, self.ema_long)

        # Accessing alternative data lines from the custom feed
        # These will be populated by backtrader from the dataframe passed to AlternativeDataFeed
        self.dark_pool_ratio_line = self.datas[0].lines.dark_pool_ratio
        self.vol_surface_line = self.datas[0].lines.volatility_surface
        self.put_call_ratio_line = self.datas[0].lines.put_call_ratio
        self.iv_skew_line = self.datas[0].lines.iv_skew
        self.gamma_exposure_line = self.datas[0].lines.gamma_exposure

        # 跟踪期权持仓
        self.option_positions = []  # 持有的期权合约列表

        # 记录上次数据更新时间 (These would be instance variables if fetched per bar in next)
        # For this simulated version where data is part of the feed, direct access is fine.
        # self.last_data_update = None # Not needed if data comes via lines

        # Variables to hold current values of alternative data for easier access in next()
        self.current_dark_pool_ratio = 0.0
        self.current_vol_surface = 0.0
        self.current_put_call_ratio = 0.0
        self.current_iv_skew = 0.0
        self.current_gamma_exposure = 0.0


    def log(self, txt, dt=None):
        '''日志记录'''
        dt = dt or self.data.datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        '''订单状态通知'''
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - {order.info.get("name", "STOCK")} Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell(): # Check order.issell()
                self.log(f'SELL EXECUTED - {order.info.get("name", "STOCK")} Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'Order Canceled/Margin/Rejected/Expired - {order.getstatusname()}')

        # Reset order
        # self.order = None # Only if you track orders with self.order

    # The user's strategy fetches data inside next().
    # For backtrader feeds, data is usually accessed via self.datas[0].lines.your_line_name[0]
    # The fetch_real_time_data method is more for a live scenario.
    # For backtesting with data already in the feed, we directly access the lines.
    # def fetch_real_time_data(self):
    #     '''获取实时另类数据'''
    #     # This method as defined by user is for live data fetching.
    #     # In backtesting with data passed via feed, this is not how it's typically done.
    #     # We will access the lines directly in next()
    #     pass


    def get_option_contract(self, strike_offset=0.05, days_to_expiry=30, option_type='call'):
        '''
        获取期权合约（简化模拟）
        实际应用中应从经纪商API获取真实报价
        '''
        underlying_price = self.data.close[0]
        strike = round(underlying_price * (1 + strike_offset if option_type == 'call' else 1 - strike_offset))

        # Ensure expiry is a date object, not datetime
        current_date = self.data.datetime.date(0)
        expiry_date = current_date + timedelta(days=days_to_expiry)

        # Simplified option premium calculation based on vol_surface_line
        # Ensure vol_surface_line has data; use a default if not.
        iv = self.current_vol_surface / 100 if self.current_vol_surface else 0.2 # Default IV 20%
        time_to_expiry_years = days_to_expiry / 365.0

        # Very rough Black-Scholes like approximation for premium (not accurate, for simulation only)
        # This is highly simplified. Real option pricing is complex.
        if iv <= 0 or time_to_expiry_years <=0: # Basic check for invalid inputs
            premium = 0.01
        else:
            # Simplified premium: proportional to price, IV, and sqrt of time.
            # This is not a financial model, just a placeholder for simulation.
            premium_factor = 0.1 # Adjust this factor as needed for reasonable premium values
            premium = underlying_price * iv * np.sqrt(time_to_expiry_years) * premium_factor
            premium = max(0.01, premium) # Ensure premium is at least 0.01

        return OptionContract(
            symbol=f"{self.p.symbol}{expiry_date.strftime('%y%m%d')}{strike}{'C' if option_type == 'call' else 'P'}",
            strike=strike,
            expiry=expiry_date,
            option_type=option_type,
            premium=premium
        )

    def execute_option_trade(self, option_type, size=1):
        '''执行期权交易'''
        # Note: backtrader does not natively support options trading in the same way as stocks.
        # This is a simulation of holding option contracts.
        # Orders placed here won't be processed by cerebro.broker directly unless you build a custom broker extension.
        # We are logging and tracking them manually in self.option_positions.

        contract = self.get_option_contract(option_type=option_type)
        if contract.premium == 0 : # Avoid division by zero if premium is zero
            self.log(f"Skipping option trade due to zero premium for {contract}")
            return

        # Calculate max contracts based on a portion of portfolio value
        # This is a simplified position sizing for options
        cash_for_options = self.broker.getvalue() * self.params.max_option_position

        # Ensure contract.premium is not zero to avoid division by zero error
        if contract.premium <= 0:
            self.log(f"Cannot calculate max_contracts, option premium is {contract.premium:.2f} for {contract}")
            return

        num_contracts_affordable = cash_for_options / (contract.premium * 100) # Assuming 100 shares per contract

        actual_size_to_trade = min(size, int(num_contracts_affordable))

        if actual_size_to_trade > 0:
            self.option_positions.append({
                'contract': contract,
                'entry_price': contract.premium, # Cost per share, not total
                'size': actual_size_to_trade, # Number of contracts
                'entry_date': self.data.datetime.date(0)
            })
            # Simulate cost: Reduce cash by option premium * size * 100 (multiplier)
            # This is a manual cash adjustment because Cerebro isn't managing these options.
            simulated_cost = contract.premium * actual_size_to_trade * 100
            self.broker.cash -= simulated_cost
            self.log(f"OPTION {'BUY' if option_type == 'call' else 'SELL_OPEN_PUT?? (User strategy implies buying puts)'} {actual_size_to_trade} contracts - {contract}. Simulated Cost: {simulated_cost:.2f}")
        else:
            self.log(f"Not enough capital or size is zero for option trade: {contract}")


    def manage_option_positions(self):
        '''管理期权持仓'''
        # This is a simplified management logic.
        # Real options management is much more complex (rolling, assignment, early exercise etc.)
        current_date = self.data.datetime.date(0)

        for position in self.option_positions[:]: # Iterate over a copy
            contract = position['contract']

            # Simplified P&L check (using entry premium vs a "current" premium, which is hard to simulate accurately here)
            # For this simulation, let's assume premium decays linearly to zero at expiry for simplicity (very rough)
            days_held = (current_date - position['entry_date']).days
            total_days_to_expiry_at_entry = (contract.expiry - position['entry_date']).days

            if total_days_to_expiry_at_entry <= 0:
                current_simulated_premium = contract.intrinsic_value(self.data.close[0]) # At expiry, only intrinsic
            else:
                # Rough time decay simulation + intrinsic value
                time_value_at_entry = position['entry_price'] - contract.intrinsic_value(self.data.close[-days_held] if days_held > 0 else self.data.close[0]) # Approx entry underlying
                time_value_at_entry = max(0, time_value_at_entry)
                current_time_value = time_value_at_entry * (1 - (days_held / total_days_to_expiry_at_entry))
                current_simulated_premium = contract.intrinsic_value(self.data.close[0]) + max(0, current_time_value)


            # Check for expiry
            if current_date >= contract.expiry:
                self.log(f"Option expired: {contract}. Last Sim Premium: {current_simulated_premium:.2f}. Closing based on intrinsic value.")
                # Simulate closing: Add back cash based on intrinsic value at expiry
                simulated_value_at_expiry = contract.intrinsic_value(self.data.close[0]) * position['size'] * 100
                self.broker.cash += simulated_value_at_expiry
                self.option_positions.remove(position)
                self.log(f"Closed expired option {contract}. Added {simulated_value_at_expiry:.2f} to cash.")
                continue

            # Simplified P&L for stop loss / take profit (based on simulated premium)
            pl_ratio = (current_simulated_premium - position['entry_price']) / position['entry_price'] if position['entry_price'] > 0 else 0

            if pl_ratio > 0.5:  # Stop GGU
                self.log(f"Option take profit: {contract} | Sim Profit: {pl_ratio*100:.2f}%. Current Sim Premium {current_simulated_premium:.2f}")
                self.broker.cash += current_simulated_premium * position['size'] * 100 # Add back value
                self.option_positions.remove(position)
            elif pl_ratio < -0.3:  # Stop Loss
                self.log(f"Option stop loss: {contract} | Sim Loss: {pl_ratio*100:.2f}%. Current Sim Premium {current_simulated_premium:.2f}")
                self.broker.cash += current_simulated_premium * position['size'] * 100 # Add back remaining value (could be small)
                self.option_positions.remove(position)


    def next(self):
        # Update current alternative data values from the lines
        # These lines are populated by backtrader from the DataFrame columns
        self.current_dark_pool_ratio = self.dark_pool_ratio_line[0]
        self.current_vol_surface = self.vol_surface_line[0]
        self.current_put_call_ratio = self.put_call_ratio_line[0]
        self.current_iv_skew = self.iv_skew_line[0]
        self.current_gamma_exposure = self.gamma_exposure_line[0]

        # Log current values (optional, for debugging)
        # self.log(f"Close: {self.data.close[0]:.2f}, DPR: {self.current_dark_pool_ratio:.2f}, VolSurf: {self.current_vol_surface:.2f}, PCR: {self.current_put_call_ratio:.2f}, IVSkew: {self.current_iv_skew:.2f}, GammaExp: {self.current_gamma_exposure:.2f}")

        self.manage_option_positions()

        # Core trading signals
        # Signal 1: Institutional inflow (Dark Pool) + EMA Crossover (Buy stock + Call options)
        if (self.current_dark_pool_ratio > self.params.dark_pool_threshold and
            self.crossover[0] > 0 and
            not self.position): # No stock position

            available_cash = self.broker.getcash()
            target_value = available_cash * (1 - self.params.max_option_position) # Reserve cash for options
            size = target_value / self.data.close[0]

            self.log(f"Signal 1: BUY STOCK. DarkPool: {self.current_dark_pool_ratio:.4f} > {self.params.dark_pool_threshold}, EMA Crossover. Size: {size:.0f}")
            self.buy(size=size)

            if self.current_vol_surface < self.params.vol_surface_threshold: # Low vol, good for buying options
                self.log(f"Signal 1: BUY CALL OPTION. VolSurface: {self.current_vol_surface:.2f} < {self.params.vol_surface_threshold}")
                self.execute_option_trade('call', size=1) # Simplified size for options


        # Signal 2: Extreme pessimism (PCR) + High IV Skew (Contrarian Buy stock + Call options)
        elif (self.current_put_call_ratio > self.params.pcr_buy_threshold and
              self.current_iv_skew > self.params.iv_skew_threshold and
              not self.position):

            available_cash = self.broker.getcash()
            target_value = available_cash * 0.5 * (1- self.params.max_option_position) # Use 50% of available cash for stock
            size = target_value / self.data.close[0]

            self.log(f"Signal 2: CONTRARIAN BUY STOCK. PCR: {self.current_put_call_ratio:.2f} > {self.params.pcr_buy_threshold}, IVSkew: {self.current_iv_skew:.2f} > {self.params.iv_skew_threshold}. Size {size:.0f}")
            self.buy(size=size)
            self.log(f"Signal 2: BUY CALL OPTION.")
            self.execute_option_trade('call', size=1)

        # Signal 3: Extreme optimism (PCR) + High Gamma Exposure (Sell stock + Buy Put options for hedge)
        elif (self.current_put_call_ratio < self.params.pcr_sell_threshold and
              self.current_gamma_exposure > self.params.gamma_exposure_threshold and
              self.position.size > 0): # Have a stock position to sell

            self.log(f"Signal 3: SELL STOCK. PCR: {self.current_put_call_ratio:.2f} < {self.params.pcr_sell_threshold}, GammaExp: {self.current_gamma_exposure:.4f} > {self.params.gamma_exposure_threshold}. Size: {self.position.size}")
            self.sell(size=self.position.size) # Sell entire stock position

            self.log(f"Signal 3: BUY PUT OPTION (Hedge).")
            self.execute_option_trade('put', size=1)


        # Signal 4: High Volatility Surface + EMA Death Cross (Sell stock + Buy Put options)
        elif (self.current_vol_surface > self.params.vol_surface_threshold and
              self.crossover[0] < 0 and
              self.position.size > 0): # Have a stock position to sell

            self.log(f"Signal 4: SELL STOCK (Volatile). VolSurface: {self.current_vol_surface:.2f} > {self.params.vol_surface_threshold}, EMA Death Cross. Size: {self.position.size}")
            self.sell(size=self.position.size) # Sell entire stock position

            self.log(f"Signal 4: BUY PUT OPTION.")
            self.execute_option_trade('put', size=1)


if __name__ == '__main__':
    # This part is for basic testing if the file is run directly
    print("backtrader_strategies.py executed directly.")
    print("EMA20Strategy class defined.")
    print("MACrossoverStrategy class defined.")
    print("RSIStrategy class defined.")
    print("EMACrossoverWithSentimentFilter class defined.")
    print("AlternativeDataStrategy class defined.") # Added this line
