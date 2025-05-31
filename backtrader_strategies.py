import backtrader as bt

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
        # Simply log the closing price of the series from the reference
        # self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not in the market, look for a signal to enter
            if self.dataclose[0] > self.ema[0]:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            # Already in the market, look for a signal to sell
            if self.dataclose[0] < self.ema[0]:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
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

if __name__ == '__main__':
    # This part is for basic testing if the file is run directly
    # It won't run a full backtest but can help catch syntax errors
    print("backtrader_strategies.py executed directly.")
    print("EMA20Strategy class defined.")
    print("MACrossoverStrategy class defined.")
    print("RSIStrategy class defined.")
