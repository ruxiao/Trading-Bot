import backtrader as bt
import datetime
import yfinance as yf
import argparse
import pandas as pd # Ensure pandas is imported
from backtrader_strategies import EMA20Strategy, MACrossoverStrategy, RSIStrategy, EMACrossoverWithSentimentFilter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest trading strategies.')
    parser.add_argument('--ticker', type=str, default='TSLA', help='Stock ticker symbol (e.g., TSLA, META)')
    parser.add_argument('--fromdate', type=str, default=None, help='Start date for backtesting (YYYY-MM-DD). Defaults to 1 year ago.')
    parser.add_argument('--todate', type=str, default=None, help='End date for backtesting (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--strategy', type=str, default='EMA20', choices=['EMA20', 'MACrossover', 'RSI', 'EMASentiment'], help='Strategy to use')

    args = parser.parse_args()

    # Determine date range
    to_date = datetime.datetime.now() if args.todate is None else datetime.datetime.strptime(args.todate, '%Y-%m-%d')
    from_date = (to_date - datetime.timedelta(days=365)) if args.fromdate is None else datetime.datetime.strptime(args.fromdate, '%Y-%m-%d')

    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')

    print(f"Fetching data for {args.ticker} from {from_date_str} to {to_date_str} for {args.strategy} strategy.")

    try:
        # Using auto_adjust=False to get 'Adj Close' separately
        data_df_raw = yf.download(args.ticker, start=from_date_str, end=to_date_str, progress=False, auto_adjust=False, actions=False)

        # Select and rename specific columns for backtrader
        data_df = pd.DataFrame(index=data_df_raw.index)
        data_df['open'] = data_df_raw['Open']
        data_df['high'] = data_df_raw['High']
        data_df['low'] = data_df_raw['Low']
        data_df['close'] = data_df_raw['Adj Close'] # Use Adj Close for close
        data_df['volume'] = data_df_raw['Volume']

        # Remove any rows with NaN values that might have been introduced (e.g. if 'Adj Close' had NaNs where 'Close' didn't)
        data_df.dropna(inplace=True)

        print("--- DataFrame Head ---")
        print(data_df.head())
        print("--- DataFrame Info ---")
        data_df.info()
        print("--- DataFrame Columns ---")
        print(data_df.columns)

    except Exception as e:
        print(f"Error fetching data or processing DataFrame for ticker {args.ticker}: {e}")
        exit()

    if data_df.empty:
        print(f"No data fetched for {args.ticker}. Check ticker symbol or date range ({from_date_str} to {to_date_str}).")
        exit()

    if data_df.empty:
        print(f"No data fetched for {args.ticker}. Check ticker symbol or date range ({from_date_str} to {to_date_str}).")
        exit()

    # The index is already datetime from yfinance
    # data_df column names are already lowercase 'open', 'high', 'low', 'close', 'volume'

    # Explicit column mapping for PandasData
    data_feed = bt.feeds.PandasData(
        dataname=data_df,
        datetime=None,  # Use index for datetime
        open='open',    # These are now the lowercase names in data_df
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=None # Explicitly state no open interest
    )

    cerebro = bt.Cerebro()

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
    else:
        # Should not happen due to 'choices' in argparse, but as a fallback:
        print(f"Unknown strategy: {args.strategy}. Defaulting to EMA20.")
        cerebro.addstrategy(EMA20Strategy)

    cerebro.adddata(data_feed)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001) # 0.1% commission

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days) # Specify timeframe
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

    # Use .get() for safer access, providing default empty dicts or zero values
    total_trades = trade_analysis.get('total', {}).get('total', 0)
    open_trades = trade_analysis.get('total', {}).get('open', 0)
    closed_trades = trade_analysis.get('total', {}).get('closed', 0)

    print("\n--- Trade Analysis ---")
    if closed_trades > 0:
        print(f"Total Trades: {total_trades}")
        print(f"Total Open Trades: {open_trades}")
        print(f"Total Closed Trades: {closed_trades}")
        print("-" * 20)

        won_total = trade_analysis.get('won', {}).get('total', 0)
        win_rate_val = (won_total / closed_trades * 100) if closed_trades > 0 else 0
        print(f"Win Rate: {win_rate_val:.2f}%")

        avg_win_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
        print(f"Average Win $: {avg_win_pnl:.2f}")

        lost_total = trade_analysis.get('lost', {}).get('total', 0)
        avg_loss_pnl = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
        print(f"Average Loss $: {avg_loss_pnl:.2f}")

        total_won_pnl = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        total_lost_pnl = trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0)

        profit_factor_val = 0
        if total_lost_pnl != 0:
            profit_factor_val = abs(total_won_pnl / total_lost_pnl)
        elif total_won_pnl > 0: # All wins, no losses
             profit_factor_val = float('inf')
        print(f"Profit Factor: {'inf' if profit_factor_val == float('inf') else f'{profit_factor_val:.2f}'}")
        print("-" * 20)

        longest_win_streak = trade_analysis.get('streak', {}).get('won', {}).get('longest', 0)
        print(f"Longest Winning Streak: {longest_win_streak}")

        longest_loss_streak = trade_analysis.get('streak', {}).get('lost', {}).get('longest', 0)
        print(f"Longest Losing Streak: {longest_loss_streak}")
    else:
        print("No closed trades to analyze.")

    try:
        print("Attempting to plot results...")
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    except Exception as e:
        print(f"Could not plot results. Error: {e}. Ensure matplotlib and a GUI backend like tkinter are installed and configured.")
