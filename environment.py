import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    A simulated day trading environment for Stable Baselines3.
    This environment provides the framework for a DRL agent to learn a trading strategy.

    Attributes:
        df (pd.DataFrame): The dataset for trading, indexed by datetime.
        initial_balance (float): The starting account balance for each episode.
        transaction_cost_pct (float): The percentage cost per transaction.
        observation_space (gym.spaces.Box): The observation space.
        action_space (gym.spaces.Box): The action space.
        max_steps_per_episode (int): Maximum number of steps in an episode.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, data, initial_balance=100000, transaction_cost_pct=0.001, max_steps_per_episode=390):
        super(TradingEnv, self).__init__()

        self.df = data
        if self.df.empty:
            raise ValueError("Data provided to TradingEnv is empty.")

        self.initial_balance = float(initial_balance)
        self.transaction_cost_pct = float(transaction_cost_pct)
        self.max_steps_per_episode = max_steps_per_episode


        # Define observation space: [norm_balance, norm_portfolio_value, shares_held_ratio, current_price_scaled] + technical_indicators
        # shares_held_ratio: ratio of shares held to max possible shares (e.g., if all balance was used to buy at initial price)
        # current_price_scaled: current price scaled by initial price of episode or a rolling mean. Here, we use the scaled 'close' from preprocessed data.
        # The number of technical indicators can vary.
        # Assuming 'open', 'high', 'low', 'close', 'volume' are base columns, rest are indicators.
        # The scaled features from preprocessor are used directly.
        indicator_cols = [col for col in self.df.columns if col not in ['open', 'high', 'low', 'volume']] # 'close' is part of indicators if scaled
        self.num_indicator_features = len(indicator_cols)

        # State: [norm_balance, norm_portfolio_value, shares_held_normalized] + selected_market_features
        # shares_held_normalized could be ratio of current shares to max possible shares if all balance was used
        # For simplicity, we'll use a fixed number of features from the preprocessed data.
        # The features from self.df are already scaled.
        # We need to ensure the observation space matches what _get_observation provides.
        # Let's define state components:
        # 1. Normalized balance
        # 2. Normalized portfolio value
        # 3. Fraction of portfolio in shares (shares_value / portfolio_value)
        # 4. Technical indicators and price information from self.df (already scaled)

        # The features from self.df are already scaled.
        # 'close_unscaled' should exist in self.df for transaction logic.
        if 'close_unscaled' not in self.df.columns:
            raise ValueError("'close_unscaled' column not found in the provided data. "
                             "Ensure preprocess_data adds it.")

        # Agent observes scaled features. 'close_unscaled' is for internal env use.
        self.data_feature_columns = [col for col in self.df.columns if col != 'close_unscaled']

        # Observation space: [balance_norm, portfolio_value_norm, position_norm (fraction of shares)] + all preprocessed (scaled) market data columns
        # Using low=-np.inf, high=np.inf for some values can be problematic for some SB3 policies if not normalized well.
        # Let's try to keep them bounded. For now, -np.inf, np.inf is fine as SB3 can normalize observations.

        # Number of features from the dataframe (excluding 'close_unscaled') + 3 portfolio metrics
        self.observation_space_dim = 3 + len(self.data_feature_columns)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32
        )

        # Action space: Continuous action: -1 (sell all) to +1 (buy all).
        # For SB3, this is a Box space.
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Episode internal state
        self._current_step = 0
        self._current_day_data = None # This will be the full df for now, or a segment if we sample days.
                                     # For simplicity with SB3, we'll treat the whole df as one long episode source,
                                     # and reset just picks a new starting point.
        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.current_portfolio_value = self.initial_balance
        self.start_tick = 0 # The starting index in the dataframe for the current episode
        self.end_tick = len(self.df) -1

        # For plotting and CSV
        self.history = []


    def _get_observation(self):
        """Constructs the observation (state) for the current timestep."""
        current_market_data_point = self.df.iloc[self.start_tick + self._current_step]
        market_features = current_market_data_point[self.data_feature_columns].values.astype(np.float32)

        norm_balance = self.current_balance / self.initial_balance
        norm_portfolio_value = self.current_portfolio_value / self.initial_balance

        # Position normalization: fraction of portfolio value that is stock
        # If portfolio value is zero (e.g. initial balance is zero), this can be an issue.
        # Use unscaled price for this calculation for accuracy.
        current_unscaled_price = current_market_data_point['close_unscaled']
        if self.current_portfolio_value != 0:
            position_value_fraction = (self.shares_held * current_unscaled_price) / self.current_portfolio_value
        else:
            position_value_fraction = 0.0

        # Ensure it's within a reasonable range, e.g. if shorting was allowed, it could be negative.
        # For long only, it's 0 to 1.
        position_value_fraction = np.clip(position_value_fraction, 0, 1)


        state = np.concatenate([
            np.array([norm_balance, norm_portfolio_value, position_value_fraction], dtype=np.float32),
            market_features
        ])
        return state

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new starting point in the dataset.
        Returns:
            np.array: The initial state observation.
            dict: Auxiliary information.
        """
        super().reset(seed=seed) # Important for reproducibility with SB3

        self.current_balance = self.initial_balance
        self.shares_held = 0.0
        self.current_portfolio_value = self.initial_balance

        # For simplicity, always start from a random point in the df if it's long enough,
        # ensuring there are enough steps left for an episode.
        # Otherwise, start from the beginning.
        if len(self.df) - self.max_steps_per_episode > 0 :
            self.start_tick = self.np_random.integers(0, len(self.df) - self.max_steps_per_episode)
        else:
            self.start_tick = 0
            # If df is shorter than max_steps_per_episode, update max_steps_per_episode
            # This is not ideal, environment should be consistent.
            # Consider raising error or handling it in main script by providing enough data.
            if len(self.df) < self.max_steps_per_episode :
                # This implies an episode can be shorter than max_steps_per_episode
                pass


        self._current_step = 0
        self.end_tick = self.start_tick + self.max_steps_per_episode -1
        if self.end_tick >= len(self.df):
             self.end_tick = len(self.df) -1


        self.history = [] # Reset history for new episode
        initial_observation = self._get_observation()
        info = {} # SB3 expects an info dictionary

        # Log initial state for backtesting/plotting
        self._log_step(action_taken=0) # Log initial state as a "hold"

        return initial_observation, info

    def step(self, action):
        """
        Executes a trading action and advances the environment by one timestep.
        Args:
            action (np.array): A continuous action value from the agent.
        Returns:
            tuple: (next_state, reward, done, truncated, info).
        """
        action_value = action[0] # Action is a single continuous value from -1 to 1
        # Use 'close_unscaled' for all financial calculations
        current_unscaled_price_for_step = self.df.iloc[self.start_tick + self._current_step]['close_unscaled']

        # --- Execute Trade ---
        self._take_action(action_value, current_unscaled_price_for_step)

        # --- Update Portfolio ---
        self.current_portfolio_value = self.current_balance + (self.shares_held * current_unscaled_price_for_step)

        # --- Log step for backtesting/plotting ---
        self._log_step(action_taken=action_value)

        # --- Move to the next timestep ---
        self._current_step += 1

        # --- Check if Done ---
        # Done if end of data for the episode is reached OR if max_steps_per_episode is reached
        done = (self.start_tick + self._current_step >= len(self.df)) or \
               (self._current_step >= self.max_steps_per_episode)

        # --- Calculate Reward ---
        # Reward is the change in portfolio value from the previous step.
        # Or, it can be the overall P&L for the episode so far.
        # Let's use change in portfolio value as the immediate reward.
        # previous_portfolio_value = self.history[-2]['portfolio_value'] if len(self.history) > 1 else self.initial_balance
        # reward = (self.current_portfolio_value - previous_portfolio_value) / previous_portfolio_value if previous_portfolio_value !=0 else 0

        # A common reward is the percentage change in portfolio value over the step,
        # or simply the change in portfolio value normalized by initial balance.
        reward = (self.current_portfolio_value - self.initial_balance) / self.initial_balance
        # For a step-wise reward, it might be better to use the change from the *last step's* portfolio value.
        if len(self.history) > 1: # history[-1] is current step, history[-2] is previous
            prev_val = self.history[-2]['portfolio_value']
            reward_step = (self.current_portfolio_value - prev_val) / self.initial_balance # Normalize by initial balance
        else: # First step
            reward_step = (self.current_portfolio_value - self.initial_balance) / self.initial_balance

        reward = reward_step # Using step-wise reward


        # --- Get Next State ---
        if done:
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32) # Or the last valid observation
        else:
            next_observation = self._get_observation()

        # SB3 expects `terminated` and `truncated` flags.
        # `terminated` is True if the episode ends due to an environment condition (e.g., task completion, failure).
        # `truncated` is True if the episode ends due to a time limit (e.g., max_steps_per_episode).
        terminated = (self.start_tick + self._current_step >= len(self.df)) # End of actual data
        truncated = (self._current_step >= self.max_steps_per_episode) and not terminated # Max steps reached before end of data

        if terminated or truncated: # Ensure done is true if either is true
            done = True


        info = {'balance': self.current_balance,
                'shares_held': self.shares_held,
                'portfolio_value': self.current_portfolio_value,
                'action_taken': action_value}

        return next_observation, reward, terminated, truncated, info

    def _take_action(self, action, current_price):
        """
        Implements the logic for buying or selling based on the agent's action.
        Action > 0: Buy
        Action < 0: Sell
        Amount is proportional to abs(action) * current available cash (for buy) or shares (for sell).
        """
        slippage = 0.0005 * abs(action) # Simplified dynamic slippage

        if action > 0: # Buy
            buy_price = current_price * (1 + slippage)
            # Max amount to invest is `action` fraction of current balance
            amount_to_invest = self.current_balance * action

            cost_of_investment = amount_to_invest
            transaction_fees = cost_of_investment * self.transaction_cost_pct

            if self.current_balance > transaction_fees and cost_of_investment > 0: # Can afford transaction
                actual_investment_after_fees = cost_of_investment - transaction_fees
                if actual_investment_after_fees > 0 and buy_price > 0:
                    shares_to_buy = actual_investment_after_fees / buy_price
                    self.shares_held += shares_to_buy
                    self.current_balance -= (shares_to_buy * buy_price + transaction_fees) # Total deduction

        elif action < 0: # Sell
            sell_price = current_price * (1 - slippage)
            # Max shares to sell is `abs(action)` fraction of current shares
            shares_to_sell = self.shares_held * abs(action)

            if self.shares_held > 0 and shares_to_sell > 0 and sell_price > 0:
                sale_revenue = shares_to_sell * sell_price
                transaction_fees = sale_revenue * self.transaction_cost_pct

                self.shares_held -= shares_to_sell
                self.current_balance += (sale_revenue - transaction_fees) # Net income

        # Ensure balance and shares are not negative due to float precision
        self.current_balance = max(0, self.current_balance)
        self.shares_held = max(0, self.shares_held)

    def _log_step(self, action_taken):
        """Logs data for the current step for backtesting and plotting."""
        timestamp = self.df.index[self.start_tick + self._current_step]
        # Log the unscaled price
        price_to_log = self.df.iloc[self.start_tick + self._current_step]['close_unscaled']

        log_entry = {
            'timestamp': timestamp,
            'price': price_to_log,
            'action': action_taken, # Could be mapped to Buy/Sell/Hold string later
            'portfolio_value': self.current_portfolio_value,
            'balance': self.current_balance,
            'shares_held': self.shares_held,
            'reward_for_step': 0 # This will be updated after reward calculation in step() if needed
        }
        self.history.append(log_entry)

    def get_episode_history(self):
        return pd.DataFrame(self.history)

    def render(self, mode='human'):
        # For now, just print some info. A full render would involve plotting.
        if mode == 'human':
            # Use unscaled price for rendering
            price_to_render = self.df.iloc[self.start_tick + self._current_step]['close_unscaled']
            print(f"Step: {self._current_step}")
            print(f"Price: {price_to_render:.2f}")
            print(f"Balance: {self.current_balance:.2f}")
            print(f"Shares Held: {self.shares_held:.4f}")
            print(f"Portfolio Value: {self.current_portfolio_value:.2f}")

    def close(self):
        # Any cleanup operations if needed
        pass

if __name__ == '__main__':
    # --- Example Usage (for testing the environment) ---
    from utils import download_data, preprocess_data
    import datetime

    ticker = 'QQQ'
    # For 1m data, yfinance typically provides last 7 days.
    # Let's try to get recent data.
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=6) # Fetch 6 days to have enough for indicators + test

    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    try:
        raw_data_test = download_data(ticker, start_date_str, end_date_str, interval='1m')
        if raw_data_test.empty:
            print("Failed to download 1m data, trying '5m'")
            start_date = end_date - datetime.timedelta(days=50) # 5m goes back further
            start_date_str = start_date.strftime('%Y-%m-%d')
            raw_data_test = download_data(ticker, start_date_str, end_date_str, interval='5m')
        if raw_data_test.empty:
            print("Failed to download 5m data, trying '1d' for basic test structure")
            start_date = end_date - datetime.timedelta(days=200)
            start_date_str = start_date.strftime('%Y-%m-%d')
            raw_data_test = download_data(ticker, start_date_str, end_date_str, interval='1d')

        if not raw_data_test.empty:
            processed_data_test = preprocess_data(raw_data_test.copy()) # Use .copy() to avoid warnings on original df

            if not processed_data_test.empty:
                # Ensure 'close' (unscaled) is available for transactions if needed, or pass it separately
                # For this example, TradingEnv will use the 'close' column from the processed_data for unscaled price.
                # This assumes 'close' in processed_data is the original, unscaled close price.
                # This is a flaw in the current preprocess_data which scales 'close'.
                # We need to fix preprocess_data to keep original 'close' or pass it.

                # Let's adjust preprocess_data to keep original close
                # For now, let's assume processed_data_test still has a 'close' column that can be used,
                # even if it's scaled. The transaction logic should ideally use unscaled prices.
                # A better fix: environment takes raw_data for prices and processed_data for features.
                # Or, ensure 'close_original' is in processed_data.

                # Quick fix for testing: Add original 'close' to processed_data if it's not there.
                # This is messy, proper fix in preprocess_data is better.
                if 'close' in processed_data_test.columns and 'close' not in raw_data_test.columns: # if 'close' was scaled
                     # This assumes raw_data_test.index matches processed_data_test.index after dropna
                     # This alignment is tricky.
                     # Simplest for now: Environment uses the 'close' from its self.df, which is scaled.
                     # This is not ideal for realistic backtesting but makes the structure run.
                     pass


                env = TradingEnv(processed_data_test, max_steps_per_episode=50)
                obs, info = env.reset()
                print("Observation space:", env.observation_space)
                print("Action space:", env.action_space)
                print("Initial observation shape:", obs.shape)

                done = False
                truncated = False
                total_reward = 0
                for i in range(100): # Simulate 100 steps
                    action = env.action_space.sample() # Sample a random action
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    # env.render()
                    if terminated or truncated:
                        print(f"Episode finished after {i+1} steps. Terminated: {terminated}, Truncated: {truncated}")
                        print(f"Total reward: {total_reward}")
                        obs, info = env.reset()
                        total_reward = 0

                env.close()
                print("TradingEnv basic test completed.")

                # Test history saving
                history_df = env.get_episode_history() # this will be from the last episode
                if not history_df.empty:
                    print("\nSample of episode history:")
                    print(history_df.head())
                else:
                    # This happens because get_episode_history is called after reset() clears it
                    # Let's test it within an episode
                    obs, info = env.reset()
                    for _ in range(5):
                        action = env.action_space.sample()
                        env.step(action)
                    history_df_during_episode = env.get_episode_history()
                    if not history_df_during_episode.empty:
                         print("\nSample of episode history (during episode):")
                         print(history_df_during_episode.head())


            else:
                print("Processed data is empty, cannot run TradingEnv example.")
        else:
            print("Raw data is empty, cannot run TradingEnv example.")

    except ValueError as e:
        print(f"Error in TradingEnv example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in TradingEnv example: {e}")
