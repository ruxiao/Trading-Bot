import backtrader as bt
import datetime
import pandas as pd
from backtrader_strategies import EMA20Strategy # Assuming this is in backtrader_strategies.py

if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(EMA20Strategy)

    # Datas are in a subfolder called 'data'
    # Create a dummy data feed for initial setup
    # This will be replaced by actual data loading, e.g., from a CSV file
    # For now, create a sample CSV file named 'sample_data.csv'
    # with columns: Date,Open,High,Low,Close,Volume,OpenInterest
    sample_data_path = 'sample_data.csv' # Expected in the same directory

    # Create the data feed
    # Ensure the CSV has the correct date format (YYYY-MM-DD) and headers
    data_feed = bt.feeds.GenericCSVData(
        dataname=sample_data_path,
        dtformat=('%Y-%m-%d'),
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1, # No OpenInterest column in this sample
        fromdate=datetime.datetime(2022, 1, 1), # Adjust as needed
        todate=datetime.datetime(2023, 12, 31)   # Adjust as needed
    )

    cerebro.adddata(data_feed)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Add commission
    cerebro.broker.setcommission(commission=0.001) # 0.1% commission

    # Add slippage
    # cerebro.broker.set_slippage_perc(perc=0.001) # 0.1% slippage, if needed

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    results = cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Print out the analysis results
    strat = results[0] # Get the first strategy
    print('\n--- Analyzers ---')
    print(f"Sharpe Ratio: {strat.analyzers.sharpe_ratio.get_analysis()['sharperatio']:.2f}")
    # AnnualReturn is a dictionary of year:return_rate
    print("Annual Return:")
    for year, ret in strat.analyzers.annual_return.get_analysis().items():
        print(f"  {year}: {ret*100:.2f}%")
    print(f"Max Drawdown: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")

    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    if trade_analysis:
        print("\n--- Trade Analysis ---")
        print(f"Total Trades: {trade_analysis.total.total}")
        print(f"Total Open Trades: {trade_analysis.total.open}")
        print(f"Total Closed Trades: {trade_analysis.total.closed}")
        print("-" * 20)
        print(f"Win Rate: {(trade_analysis.won.total / trade_analysis.total.closed * 100) if trade_analysis.total.closed > 0 else 0:.2f}%")
        print(f"Average Win $: {trade_analysis.won.pnl.average:.2f}" if trade_analysis.won.total > 0 else "Average Win $: N/A")
        print(f"Average Loss $: {trade_analysis.lost.pnl.average:.2f}" if trade_analysis.lost.total > 0 else "Average Loss $: N/A")
        print(f"Profit Factor: {(trade_analysis.won.pnl.total / abs(trade_analysis.lost.pnl.total)) if trade_analysis.lost.pnl.total != 0 else 'inf'}")
        print("-" * 20)
        print(f"Longest Winning Streak: {trade_analysis.streak.won.longest}")
        print(f"Longest Losing Streak: {trade_analysis.streak.lost.longest}")

    # Plot the result
    # Make sure you have matplotlib installed: pip install matplotlib
    try:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    except Exception as e:
        print(f"Could not plot results. Error: {e}. Make sure matplotlib is installed.")
