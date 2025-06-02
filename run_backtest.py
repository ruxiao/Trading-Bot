import backtrader as bt
import datetime
import yfinance as yf
import argparse
import numpy as np # Added
import pandas as pd # Added

from backtrader_strategies import EMA20Strategy, MACrossoverStrategy, RSIStrategy, EMACrossoverWithSentimentFilter, AlternativeDataStrategy # Added AlternativeDataStrategy
from alternative_data_feed import AlternativeDataFeed # Added

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest trading strategies.')
    parser.add_argument('--ticker', type=str, default='TSLA', help='Stock ticker symbol (e.g., TSLA, META)')
    parser.add_argument('--fromdate', type=str, default=None, help='Start date for backtesting (YYYY-MM-DD). Defaults to 1 year ago.')
    parser.add_argument('--todate', type=str, default=None, help='End date for backtesting (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--strategy', type=str, default='EMA20',
                        choices=['EMA20', 'MACrossover', 'RSI', 'EMASentiment', 'AlternativeData'], # Added AlternativeData
                        help='Strategy to use')

    args = parser.parse_args()

    to_date = datetime.datetime.now() if args.todate is None else datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    from_date = (to_date - datetime.timedelta(days=365)) if args.fromdate is None else datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')

    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    print(f"Fetching data for {args.ticker} from {from_date_str} to {to_date_str} for {args.strategy} strategy.")

    try:
        # Using auto_adjust=False to get unadjusted prices, which is often preferred for backtesting
        # as adjustments can be handled by the strategy or broker simulation if needed.
        # We will need to manually select 'Adj Close' as 'close' or ensure other OHLC are what we want.
        raw_data_df = yf.download(args.ticker, start=from_date_str, end=to_date_str, progress=False, auto_adjust=False)
    except Exception as e:
        print(f"Error fetching data using yfinance for ticker {args.ticker}: {e}")
        exit()

    if raw_data_df.empty:
        print(f"No data fetched for {args.ticker}. Check ticker symbol or date range ({from_date_str} to {to_date_str}).")
        exit()

    # Prepare DataFrame for backtrader
    # Use 'Adj Close' for 'close', and ensure standard OHLCV columns are present and lowercase
    data_df = pd.DataFrame(index=raw_data_df.index)
    data_df['open'] = raw_data_df['Open']
    data_df['high'] = raw_data_df['High']
    data_df['low'] = raw_data_df['Low']
    data_df['close'] = raw_data_df['Adj Close'] # Use adjusted close for 'close'
    data_df['volume'] = raw_data_df['Volume']

    # Remove any rows with NaN values that might have resulted from joins or data issues
    data_df.dropna(inplace=True)

    if data_df.empty:
        print(f"Data for {args.ticker} became empty after NaN drop. Original rows: {len(raw_data_df)}")
        exit()

    cerebro = bt.Cerebro()

    use_alternative_data_feed = False # Flag

    if args.strategy == 'EMA20':
        cerebro.addstrategy(EMA20Strategy)
        print("Using EMA20 Strategy")
    elif args.strategy == 'MACrossover':
        cerebro.addstrategy(MACrossoverStrategy)
        print("Using MACrossover Strategy")
    elif args.strategy == 'RSI':
        cerebro.addstrategy(RSIStrategy)
        print("Using RSI Strategy")
    elif args.strategy == 'EMASentiment':
        cerebro.addstrategy(EMACrossoverWithSentimentFilter)
        print("Using EMACrossoverWithSentimentFilter Strategy")
    elif args.strategy == 'AlternativeData':
        use_alternative_data_feed = True
        cerebro.addstrategy(AlternativeDataStrategy, symbol=args.ticker) # Pass ticker to strategy params
        print("Using AlternativeData Strategy")
    else:
        print(f"Unknown strategy: {args.strategy}. Defaulting to EMA20.")
        cerebro.addstrategy(EMA20Strategy)

    # Create and add data feed
    if use_alternative_data_feed:
        print("Preparing AlternativeDataFeed...")
        # Add simulated alternative data columns
        data_df['dark_pool_ratio'] = np.random.uniform(0.1, 0.25, len(data_df))
        data_df['volatility_surface'] = np.random.uniform(25, 45, len(data_df))
        data_df['put_call_ratio'] = np.random.uniform(0.7, 1.5, len(data_df))
        data_df['iv_skew'] = np.random.uniform(0.9, 1.5, len(data_df))
        data_df['gamma_exposure'] = np.random.uniform(-0.15, 0.15, len(data_df))

        # Ensure all column names are lowercase (should be already by construction above)
        data_df.columns = [str(col).lower() for col in data_df.columns]

        print("DataFrame columns for AlternativeDataFeed:", data_df.columns)
        # print(data_df.head()) # Optional: print head for debugging

        data_feed = AlternativeDataFeed(dataname=data_df)
    else:
        # For standard strategies, ensure columns are lowercase if not already
        data_df.columns = [str(col).lower() for col in data_df.columns]
        data_feed = bt.feeds.PandasData(dataname=data_df)

    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    strat = results[0]
    print('\n--- Analyzers ---')

    sharpe_analysis = strat.analyzers.sharpe_ratio.get_analysis()
    if sharpe_analysis and 'sharperatio' in sharpe_analysis and sharpe_analysis['sharperatio'] is not None:
        print(f"Sharpe Ratio: {sharpe_analysis['sharperatio']:.2f}")
    else:
        print("Sharpe Ratio: N/A")

    annual_return_analysis = strat.analyzers.annual_return.get_analysis()
    if annual_return_analysis:
        print("Annual Return:")
        for year, ret in annual_return_analysis.items():
            print(f"  {year}: {ret*100:.2f}%")
    else:
        print("Annual Return: N/A")

    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    if drawdown_analysis and 'max' in drawdown_analysis and 'drawdown' in drawdown_analysis['max'] and drawdown_analysis['max']['drawdown'] is not None:
        print(f"Max Drawdown: {drawdown_analysis['max']['drawdown']:.2f}%")
    else:
        print("Max Drawdown: N/A")

    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    if trade_analysis and hasattr(trade_analysis, 'total') and trade_analysis.total.closed > 0 :
        print("\n--- Trade Analysis ---")
        print(f"Total Trades: {trade_analysis.total.total}")
        print(f"Total Open Trades: {trade_analysis.total.open}")
        print(f"Total Closed Trades: {trade_analysis.total.closed}")
        print("-" * 20)
        win_rate_val = (trade_analysis.won.total / trade_analysis.total.closed * 100) if trade_analysis.total.closed > 0 else 0
        print(f"Win Rate: {win_rate_val:.2f}%")

        avg_win_val = trade_analysis.won.pnl.average if trade_analysis.won.total > 0 and hasattr(trade_analysis.won.pnl, 'average') else 0
        print(f"Average Win $: {avg_win_val:.2f}")

        avg_loss_val = trade_analysis.lost.pnl.average if trade_analysis.lost.total > 0 and hasattr(trade_analysis.lost.pnl, 'average') else 0
        print(f"Average Loss $: {avg_loss_val:.2f}")

        profit_factor_val = float('inf')
        if hasattr(trade_analysis.lost.pnl, 'total') and trade_analysis.lost.pnl.total != 0 and hasattr(trade_analysis.won.pnl, 'total'): # Check if total is not None
            if trade_analysis.lost.pnl.total != 0: # Ensure denominator is not zero
                 profit_factor_val = abs(trade_analysis.won.pnl.total / trade_analysis.lost.pnl.total)
            elif trade_analysis.won.pnl.total > 0 : # All wins, no losses
                 profit_factor_val = float('inf')
            else: # No wins and no losses
                 profit_factor_val = 0
        elif hasattr(trade_analysis.won.pnl, 'total') and trade_analysis.won.pnl.total > 0: # All wins
             profit_factor_val = float('inf')
        else: # No wins or no losses
            profit_factor_val = 0

        print(f"Profit Factor: {'inf' if profit_factor_val == float('inf') else f'{profit_factor_val:.2f}'}")
        print("-" * 20)
        longest_win_streak = trade_analysis.streak.won.longest if hasattr(trade_analysis.streak.won, 'longest') else 0
        print(f"Longest Winning Streak: {longest_win_streak}")
        longest_loss_streak = trade_analysis.streak.lost.longest if hasattr(trade_analysis.streak.lost, 'longest') else 0
        print(f"Longest Losing Streak: {longest_loss_streak}")
    else:
        print("\n--- Trade Analysis ---")
        print("No closed trades to analyze or trade_analysis object is not as expected.")

    try:
        print("Attempting to plot results...")
        cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False) # volume=False
    except Exception as e:
        print(f"Could not plot results. Error: {e}. Ensure matplotlib and a GUI backend like tkinter are installed and configured.")
