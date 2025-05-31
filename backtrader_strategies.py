import backtrader as bt
import random # Ensure this is at the top of the file

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

if __name__ == '__main__':
    # This part is for basic testing if the file is run directly
    print("backtrader_strategies.py executed directly.")
    print("EMA20Strategy class defined.")
    print("MACrossoverStrategy class defined.")
    print("RSIStrategy class defined.")
    print("EMACrossoverWithSentimentFilter class defined.")
