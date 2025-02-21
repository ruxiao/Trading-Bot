import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime

class TradingStrategy:
    def __init__(self, alpha_params: List[float]):
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

    def calculate_signal(self, data: pd.DataFrame, correlation_matrix: pd.DataFrame, symbol: str) -> tuple:
        """
        Calculate trading signals based on alpha parameters and correlation
        Returns signals and alpha values for analysis
        """
        signals = np.zeros(len(data))
        alpha_values = np.zeros(len(data))
        trade_log = []

        # Calculate technical indicators with validated windows
        ma_short = data['Close'].rolling(window=self.short_window).mean()
        ma_long = data['Close'].rolling(window=self.long_window).mean()

        # RSI calculation
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Handle NaN values in RSI
        rsi = rsi.fillna(50)  # Fill NaN with neutral RSI value

        # Calculate correlation component
        # Average correlation with other assets
        correlations = correlation_matrix[symbol].drop(symbol)
        avg_correlation = correlations.mean()
        correlation_signal = -avg_correlation  # Negative correlation is good for diversification

        # Alpha formula calculation
        # Alpha = w1 * (Short MA - Long MA)/Long MA + w2 * (30 - RSI)/30 + w3 * (RSI - 70)/30 + w4 * correlation_signal
        ma_component = (ma_short - ma_long) / ma_long
        rsi_oversold = (30 - rsi) / 30
        rsi_overbought = (rsi - 70) / 30

        # Handle NaN values in alpha calculation
        ma_component = ma_component.fillna(0)
        alpha_values = (
            self.alpha_params[2] * ma_component +
            self.alpha_params[3] * rsi_oversold +
            self.alpha_params[4] * rsi_overbought +
            self.alpha_params[5] * correlation_signal
        )

        # Convert alpha values to position signals (-1 for short, 0 for neutral, 1 for long)
        signals = np.clip(np.sign(alpha_values), self.position_limits[0], self.position_limits[1])

        # Create trade log
        position = 0
        for i in range(max(self.short_window, self.long_window), len(signals)):
            if signals[i] != signals[i-1]:  # Position change
                date = data.index[i]
                price = data['Close'].iloc[i]
                new_position = int(signals[i])

                trade_log.append({
                    'date': date,
                    'symbol': symbol,
                    'price': price,
                    'action': 'BUY' if new_position == 1 else 'SELL' if new_position == -1 else 'CLOSE',
                    'alpha_value': alpha_values[i],
                    'ma_short': ma_short.iloc[i],
                    'ma_long': ma_long.iloc[i],
                    'rsi': rsi.iloc[i],
                    'correlation': correlation_signal
                })
                position = new_position

        return signals, alpha_values, trade_log

    def backtest_portfolio(self, data_dict: dict, correlation_matrix: pd.DataFrame) -> Dict:
        """
        Backtest the trading strategy across multiple assets
        Returns performance metrics and trade log
        """
        all_signals = {}
        all_returns = pd.DataFrame()
        combined_trade_log = []

        # Calculate signals for each asset
        for symbol, data in data_dict.items():
            signals, alpha_values, trade_log = self.calculate_signal(data, correlation_matrix, symbol)
            all_signals[symbol] = signals

            # Calculate returns
            price_returns = data['Close'].pct_change().fillna(0)
            strategy_returns = signals[:-1] * price_returns[1:]
            all_returns[symbol] = strategy_returns

            combined_trade_log.extend(trade_log)

        # Calculate portfolio metrics
        portfolio_returns = all_returns.mean(axis=1)  # Equal-weighted portfolio
        total_return = np.prod(1 + portfolio_returns) - 1
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) != 0 else 0
        max_drawdown = np.min(np.maximum.accumulate(portfolio_returns) - portfolio_returns)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns': portfolio_returns,
            'trade_log': combined_trade_log,
            'asset_returns': all_returns
        }