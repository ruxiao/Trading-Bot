import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment import TradingEnv # Assuming this is the Gym-compatible env

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculates the Sortino Ratio for a series of returns.
    """
    returns_series = pd.Series(returns)
    mean_return = returns_series.mean()

    # Calculate downside returns
    downside_returns = returns_series[returns_series < risk_free_rate] # Compare to risk_free_rate for downside

    # Calculate downside deviation (standard deviation of downside returns)
    # If there are no downside returns, downside deviation is 0.
    if downside_returns.empty:
        downside_deviation = 0.0
    else:
        downside_deviation = np.std(downside_returns) # Traditionally, this is (target - return)^2, but std of neg returns is common
                                                     # A more precise Sortino would use (return - risk_free_rate) for downside returns.
                                                     # For simplicity, using std of returns < risk_free_rate.

    if downside_deviation == 0:
        # If no downside deviation, Sortino is inf if mean return > risk_free_rate, else 0 or undefined.
        return np.inf if mean_return > risk_free_rate else 0.0

    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
    return sortino_ratio

def evaluate_sb3_agent(model, env: TradingEnv, num_episodes=10, model_name="Agent"):
    """
    Evaluates the performance of a Stable Baselines3 agent on a given environment.

    Args:
        model: The trained Stable Baselines3 model.
        env (TradingEnv): The environment for evaluation (must be a single, non-vectorized env for history).
        num_episodes (int): The number of episodes to run for evaluation.
        model_name (str): Name of the model for printing results.

    Returns:
        dict: A dictionary of performance metrics.
        pd.DataFrame: A DataFrame containing the detailed backtest history from the last episode.
    """
    all_episode_rewards = []
    all_portfolio_histories = [] # To store history from each episode if needed, though usually focus on one

    print(f"\n--- Evaluating {model_name} for {num_episodes} episodes ---")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_rewards_sum = 0

        # env.history should be cleared by env.reset()

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards_sum += reward

            if done or truncated:
                # History for this episode is now in env.get_episode_history()
                # For a multi-episode evaluation, we might average metrics.
                # For detailed backtest plot, usually one representative episode (e.g., the last one) is used.
                if episode == num_episodes -1 : # Store history of the last episode for detailed plotting
                    all_portfolio_histories.append(env.get_episode_history())
                break # Exit while loop for this episode

        all_episode_rewards.append(episode_rewards_sum)
        # Portfolio value at the end of the episode
        final_portfolio_value = info.get('portfolio_value', env.initial_balance) # Fallback to initial if not in info
        initial_balance = env.initial_balance
        episode_return_pct = (final_portfolio_value - initial_balance) / initial_balance * 100

        print(f"Episode {episode + 1}: Total Reward: {episode_rewards_sum:.4f}, Final Portfolio: {final_portfolio_value:.2f}, Return: {episode_return_pct:.2f}%")

    # Calculate overall metrics
    avg_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    # For Sortino ratio, we need returns per episode or per trade.
    # Let's use per-episode returns (percentage change in portfolio value).
    episode_returns = [(env.history[-1]['portfolio_value'] - env.initial_balance) / env.initial_balance
                       if len(env.history) > 0 else 0
                       for _ in range(num_episodes)] # This needs to be fixed; history is per episode.
                                                    # Recalculate returns based on final portfolio values.

    portfolio_final_values = []
    # Re-run episodes or get final values if they were stored
    # For simplicity, let's assume 'info' from the last step of each episode gives the final portfolio value.
    # This part needs careful implementation if we want accurate multi-episode Sortino.
    # The current 'all_episode_rewards' are sum of step rewards, not directly portfolio returns.
    # Let's make a simplification: use the sum of rewards as a proxy for returns for Sortino for now.
    # A better way is to calculate actual portfolio returns for each episode.

    # To get proper episode returns for Sortino:
    episode_portfolio_returns = []
    temp_env_for_returns = env # Use the same env, it will reset
    for _ in range(num_episodes):
        obs, _ = temp_env_for_returns.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info_ep = temp_env_for_returns.step(action)
        final_val = info_ep.get('portfolio_value', temp_env_for_returns.initial_balance)
        ep_return = (final_val - temp_env_for_returns.initial_balance) / temp_env_for_returns.initial_balance
        episode_portfolio_returns.append(ep_return)

    sortino = calculate_sortino_ratio(np.array(episode_portfolio_returns))

    print(f"\n{model_name} Evaluation Summary:")
    print(f"Average Reward per Episode: {avg_reward:.4f}")
    print(f"Std Dev of Reward: {std_reward:.4f}")
    print(f"Sortino Ratio (based on {num_episodes} episode returns): {sortino:.4f}")

    metrics = {
        'average_reward_per_episode': avg_reward,
        'reward_std_dev': std_reward,
        'sortino_ratio': sortino,
        'average_episode_portfolio_return': np.mean(episode_portfolio_returns) if episode_portfolio_returns else 0,
    }

    # Return history of the last evaluated episode
    last_episode_history_df = all_portfolio_histories[-1] if all_portfolio_histories else pd.DataFrame()

    return metrics, last_episode_history_df


def plot_backtest_results(history_df: pd.DataFrame, ticker_name="Stock", output_filename="backtest_plot.png"):
    """
    Plots the backtest results: price with trades and portfolio value over time.
    Saves the portfolio history to a CSV file.

    Args:
        history_df (pd.DataFrame): DataFrame with backtest history. Must include:
                                   'timestamp', 'price', 'action', 'portfolio_value'.
                                   'action' > 0 for buy, < 0 for sell.
        ticker_name (str): Name of the traded stock/asset for plot titles.
        output_filename (str): Filename for saving the plot.
    """
    if history_df.empty:
        print("History DataFrame is empty. Cannot generate plot or CSV.")
        return

    # Save to CSV
    csv_filename = output_filename.replace(".png", "_history.csv")
    history_df.to_csv(csv_filename, index=False)
    print(f"Backtest history saved to {csv_filename}")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot 1: Price and Trades
    ax1 = axes[0]
    ax1.plot(history_df['timestamp'], history_df['price'], label=f'{ticker_name} Price', color='skyblue')

    # Plot Buy signals
    buy_signals = history_df[history_df['action'] > 0.1] # Threshold to consider it a buy signal
    ax1.scatter(buy_signals['timestamp'], buy_signals['price'], marker='^', color='green', label='Buy', s=100, alpha=0.8)

    # Plot Sell signals
    sell_signals = history_df[history_df['action'] < -0.1] # Threshold to consider it a sell signal
    ax1.scatter(sell_signals['timestamp'], sell_signals['price'], marker='v', color='red', label='Sell', s=100, alpha=0.8)

    ax1.set_title(f'{ticker_name} Trading Activity')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Portfolio Value
    ax2 = axes[1]
    ax2.plot(history_df['timestamp'], history_df['portfolio_value'], label='Portfolio Value', color='orange')
    ax2.set_title('Portfolio Value Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True)

    fig.autofmt_xdate() # Auto-format x-axis dates for better readability
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Backtest plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optionally show plot interactively


if __name__ == '__main__':
    # This section is for example/testing if evaluation.py is run directly.
    # It would require a dummy agent and environment setup.

    print("Evaluation module ready.")
    print("To test, you would need a trained SB3 model and a compatible TradingEnv instance.")

    # Example of creating a dummy history DataFrame for testing plot_backtest_results
    num_points = 100
    start_time = pd.Timestamp('2023-01-01 09:30:00')
    time_deltas = [pd.Timedelta(minutes=i) for i in range(num_points)]
    dummy_timestamps = [start_time + delta for delta in time_deltas]

    dummy_data = {
        'timestamp': dummy_timestamps,
        'price': np.sin(np.linspace(0, 10, num_points)) * 5 + 100, # Dummy price data
        'action': np.random.choice([-1, 0, 1], size=num_points, p=[0.1, 0.8, 0.1]) * np.random.rand(num_points), # Dummy actions
        'portfolio_value': np.linspace(100000, 105000, num_points) + np.random.randn(num_points) * 500, # Dummy portfolio
        'balance': np.linspace(50000, 45000, num_points),
        'shares_held': np.linspace(50, 55, num_points)
    }
    dummy_history_df = pd.DataFrame(dummy_data)

    print("\nTesting plot_backtest_results with dummy data...")
    plot_backtest_results(dummy_history_df, ticker_name="DUMMY", output_filename="dummy_backtest_plot.png")

    # Test Sortino calculation
    test_returns_positive = np.array([0.01, 0.02, 0.005, 0.03])
    test_returns_mixed = np.array([-0.01, 0.02, -0.005, 0.015, -0.02])
    print(f"\nSortino for positive returns: {calculate_sortino_ratio(test_returns_positive):.4f}") # Should be inf or high
    print(f"Sortino for mixed returns: {calculate_sortino_ratio(test_returns_mixed):.4f}")

    # Edge case: no downside returns but mean return is negative (relative to risk_free_rate=0)
    test_returns_all_positive_but_low_mean_if_rf_high = np.array([0.001, 0.002, 0.0005])
    print(f"Sortino (rf=0): {calculate_sortino_ratio(test_returns_all_positive_but_low_mean_if_rf_high, risk_free_rate=0):.4f}") # Inf
    print(f"Sortino (rf=0.01): {calculate_sortino_ratio(test_returns_all_positive_but_low_mean_if_rf_high, risk_free_rate=0.01):.4f}") # Should be 0 or neg

    # Edge case: all returns are zero
    test_returns_zero = np.array([0.0, 0.0, 0.0])
    print(f"Sortino for zero returns: {calculate_sortino_ratio(test_returns_zero):.4f}") # Should be 0.0

    # Edge case: empty returns
    test_returns_empty = np.array([])
    # print(f"Sortino for empty returns: {calculate_sortino_ratio(test_returns_empty):.4f}") # Will likely raise error or NaN
    # Current implementation of calculate_sortino_ratio handles empty returns in downside_returns,
    # but not if the initial 'returns' series is empty (pandas mean() would be NaN).
    # Adding a check for empty returns input:
    def calculate_sortino_ratio_robust(returns, risk_free_rate=0.0):
        if not isinstance(returns, pd.Series):
            returns_series = pd.Series(returns)
        else:
            returns_series = returns

        if returns_series.empty:
            return 0.0 # Or np.nan, depending on desired behavior for no data
        return calculate_sortino_ratio(returns_series, risk_free_rate) # Call original

    print(f"Sortino for empty returns (robust): {calculate_sortino_ratio_robust(test_returns_empty):.4f}")


# Original evaluate_agent (for reference, will be removed or adapted for SB3)
# def evaluate_agent(agent, env, num_episodes=10):
#     """
#     Evaluates the performance of the original custom agent.
#     (This will be replaced by evaluate_sb3_agent)
#     """
#     total_returns = []
#     # ... (original implementation) ...
#     pass
