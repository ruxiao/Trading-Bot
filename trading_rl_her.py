import pandas as pd
import numpy as np
import random
from collections import deque
import io

# --- 0. Sample Data Generation ---
# I'll create a string that mimics your CSV file so the script is self-contained.
# In your real code, you would replace this with: df = pd.read_csv("your_data.csv")
csv_data = """date,open,high,low,close,volume,average,barCount,SMA_20,RSI_14,MACD_12_26_9,MACDh_12_26_9,MACDs_12_26_9
2025-06-13 09:30:00-04:00,527.67,528.67,527.25,528.67,441688.0,527.858,1749,,,,
2025-06-13 09:31:00-04:00,528.65,529.66,528.65,529.65,317682.0,529.121,1886,,,,
2025-06-13 09:32:00-04:00,529.65,530.01,529.3,529.63,323640.0,529.69,2028,,,,
2025-06-13 09:33:00-04:00,529.62,529.99,529.43,529.52,304049.0,529.731,1349,,,,
2025-06-13 09:34:00-04:00,529.5,529.71,529.34,529.68,133004.0,529.525,800,,,,
2025-06-13 09:43:00-04:00,529.91,530.1,529.6,529.66,120879.0,529.889,842,,,,
2025-06-13 09:44:00-04:00,529.66,529.89,529.52,529.82,85964.0,529.697,616,,60.90302491,,,,
2025-06-13 09:45:00-04:00,529.83,529.83,529.06,529.23,148011.0,529.399,1120,,48.17856966,,,,
2025-06-13 09:46:00-04:00,529.18,529.57,529.07,529.38,95070.0,529.332,713,,50.9825501,,,,
2025-06-13 09:47:00-04:00,529.38,529.7,529.32,529.48,87161.0,529.506,627,,52.81553157,,,,
2025-06-13 09:48:00-04:00,529.47,529.62,529.28,529.49,62042.0,529.414,442,,53.00478583,,,,
2025-06-13 09:49:00-04:00,529.48,529.53,529.18,529.47,68000.0,529.328,510,529.5535,52.55801743,,,,
2025-06-13 09:50:00-04:00,529.47,530.05,529.38,529.8,170010.0,529.662,1420,529.61,59.91035286,,,,
"""

import datetime

# --- 1. The Realistic Trading Environment ---
class TradingEnv:
    def __init__(self, df, initial_balance=100000, commission=0.001,
                 intra_day_trade_max_steps=60, max_trades_per_day=5,
                 trading_start_time=datetime.time(9, 30), trading_end_time=datetime.time(16, 0)):
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance # Current balance
        self.commission = commission
        self.intra_day_trade_max_steps = intra_day_trade_max_steps
        self.max_trades_per_day = max_trades_per_day

        # Ensure trading_start_time and trading_end_time are datetime.time objects
        if not isinstance(trading_start_time, datetime.time):
            raise ValueError("trading_start_time must be a datetime.time object")
        if not isinstance(trading_end_time, datetime.time):
            raise ValueError("trading_end_time must be a datetime.time object")

        self.trading_start_time = trading_start_time
        self.trading_end_time = trading_end_time

        self.max_data_steps = len(df)
        self.current_step = 0
        self.current_day_trades = 0
        self.current_date_in_sim = None

        # Observation: [pnl_pct, drawdown_pct, pos_open, rsi, macd_diff, trades_today_norm, time_to_eod_norm]
        self.observation_space_dim = 7
        self.action_space_dim = 3  # 0: Hold, 1: Buy, 2: Sell

        self._reset_trade_info()
        # Initialize trade_history in __init__ so it persists across resets
        self.trade_history = []

    def _reset_trade_info(self):
        self.trade_info = {
            'is_open': False, 'entry_price': 0, 'position_size': 0,
            'entry_value': 0, 'peak_value': 0, 'entry_step': 0, 'entry_time': None
        }

    def _get_achieved_goal(self):
        if not self.trade_info['is_open'] or self.current_step >= self.max_data_steps:
            return np.array([0.0, 0.0])

        # Ensure current_step is valid index for df.loc
        safe_current_step = min(self.current_step, self.max_data_steps - 1)
        current_price = self.df.loc[safe_current_step, 'close']

        pnl = (current_price - self.trade_info['entry_price']) * self.trade_info['position_size']
        current_value = self.trade_info['entry_value'] + pnl

        if current_value > self.trade_info['peak_value']:
            self.trade_info['peak_value'] = current_value

        pnl_pct = (current_value / self.trade_info['entry_value']) - 1.0 if self.trade_info['entry_value'] > 0 else 0.0
        drawdown_pct = (current_value / self.trade_info['peak_value']) - 1.0 if self.trade_info['peak_value'] > 0 else 0.0

        return np.array([pnl_pct, drawdown_pct])

    def _get_observation(self):
        safe_current_step = min(self.current_step, self.max_data_steps - 1)
        market_data_row = self.df.loc[safe_current_step]

        pnl_pct, drawdown_pct = self._get_achieved_goal()
        is_holding_position = 1.0 if self.trade_info['is_open'] else 0.0

        rsi = market_data_row['RSI_14'] / 100.0
        macd_line = market_data_row.get('MACD_12_26_9', 0.0)
        macd_signal = market_data_row.get('MACDs_12_26_9', 0.0)
        macd_diff = macd_line - macd_signal
        if pd.isna(rsi): rsi = 0.5
        if pd.isna(macd_diff): macd_diff = 0.0

        trades_today_count_norm = self.current_day_trades / self.max_trades_per_day

        current_timestamp = market_data_row['date'] # This is pd.Timestamp

        # Combine date part of current_timestamp with trading_end_time
        # Ensure current_timestamp.date() is a valid date object
        current_date_obj = current_timestamp.date()
        market_close_dt_naive = datetime.datetime.combine(current_date_obj, self.trading_end_time)

        if current_timestamp.tzinfo:
            # If current_timestamp is aware, make market_close_dt_naive aware with the same timezone
            market_close_dt_aware = market_close_dt_naive.replace(tzinfo=current_timestamp.tzinfo)
            time_to_eod_seconds = (market_close_dt_aware - current_timestamp).total_seconds()
        else: # Assuming naive datetime for both if current_timestamp is naive
            time_to_eod_seconds = (market_close_dt_naive - current_timestamp).total_seconds()

        total_trading_seconds_in_day = (datetime.datetime.combine(datetime.date.min, self.trading_end_time) -
                                        datetime.datetime.combine(datetime.date.min, self.trading_start_time)).total_seconds()

        time_to_eod_norm = max(0, time_to_eod_seconds / total_trading_seconds_in_day) if total_trading_seconds_in_day > 0 else 0

        return np.array([
            pnl_pct, drawdown_pct, is_holding_position, rsi, macd_diff,
            trades_today_count_norm, time_to_eod_norm
        ])

    def reset(self, start_step=None):
        self.balance = self.initial_balance
        # self.trade_history = [] # DO NOT clear history here, keep it for the entire env lifetime for plotting

        min_required_steps_for_trading = max(20, self.intra_day_trade_max_steps) # Min steps to allow some trading

        if start_step is not None and start_step < self.max_data_steps - min_required_steps_for_trading :
             self.current_step = start_step
        else:
            # Start at a random valid point, ensuring enough data for at least one episode + indicators
            min_start_idx = 20 # For indicator warmup
            max_start_idx = self.max_data_steps - min_required_steps_for_trading -1
            if min_start_idx >= max_start_idx: # Handle very short dataframes
                self.current_step = min_start_idx if self.max_data_steps > min_start_idx else 0
            else:
                self.current_step = random.randint(min_start_idx, max_start_idx)

        self._reset_trade_info()

        initial_data_row = self.df.loc[self.current_step]
        self.current_date_in_sim = initial_data_row['date_only']
        self.current_day_trades = 0

        # Fast-forward to the first valid trading time on or after current_step if current_step is outside hours
        while self.current_step < self.max_data_steps -1:
            current_time = self.df.loc[self.current_step, 'time_only']
            current_date_val = self.df.loc[self.current_step, 'date_only']
            if current_date_val != self.current_date_in_sim: # New day
                self.current_date_in_sim = current_date_val
                self.current_day_trades = 0

            if current_time >= self.trading_start_time and current_time < self.trading_end_time:
                break # Found valid start time
            self.current_step +=1
        else: # Reached end of data while searching for start time
            # This means the initial random point was too close to the end with no valid trading window
            # Fallback: reset to a much earlier point or raise error
            if start_step is None: # only if it was a random start
                self.current_step = min_start_idx if self.max_data_steps > min_start_idx else 0
                initial_data_row = self.df.loc[self.current_step]
                self.current_date_in_sim = initial_data_row['date_only']
                self.current_day_trades = 0
                # And re-fast-forward (this could be cleaner)
                while self.current_step < self.max_data_steps -1:
                    if self.df.loc[self.current_step, 'time_only'] >= self.trading_start_time: break
                    self.current_step +=1


        if self.current_step >= self.max_data_steps -1:
            print("Warning: Could not find a valid trading start point in reset. Data might be too short or outside trading hours.")
            # Return a dummy observation if truly stuck
            return np.zeros(self.observation_space_dim), True # True for done

        return self._get_observation(), False # False for done

    def _record_trade(self, entry_time, entry_price, exit_time, exit_price, pnl):
        self.trade_history.append({
            'entry_time': entry_time, 'entry_price': entry_price,
            'exit_time': exit_time, 'exit_price': exit_price, 'pnl': pnl
        })

    def _close_position(self, price_at_closure, reason=""):
        if not self.trade_info['is_open']:
            return False

        pnl = (price_at_closure - self.trade_info['entry_price']) * self.trade_info['position_size']
        exit_value_before_commission = self.trade_info['entry_value'] + pnl
        exit_value_after_commission = exit_value_before_commission * (1 - self.commission)
        self.balance += exit_value_after_commission

        closure_time = self.df.loc[min(self.current_step, self.max_data_steps -1), 'date']
        self._record_trade(self.trade_info['entry_time'], self.trade_info['entry_price'],
                           closure_time, price_at_closure, pnl)

        # print(f"    CLOSE {reason}: Entry: {self.trade_info['entry_price']:.2f}@{self.trade_info['entry_time']}, Exit: {price_at_closure:.2f}@{closure_time}, PnL: {pnl:.2f}, Bal: {self.balance:.2f}")
        self._reset_trade_info()
        return True

    def step(self, action):
        if self.current_step >= self.max_data_steps -1:
            if self.trade_info['is_open']:
                self._close_position(self.df.loc[self.max_data_steps -1, 'close'], reason="DATA_END")
            return self._get_observation(), -1.0, True, {'achieved_goal': np.array([0.0,0.0]), 'trades_today': self.current_day_trades}

        current_data_row = self.df.loc[self.current_step]
        current_price = current_data_row['close']
        current_timestamp = current_data_row['date']
        current_date = current_data_row['date_only']
        current_time = current_data_row['time_only']

        done_for_episode = False
        reward = -0.01 # Small penalty for each step to encourage faster trades if profitable

        # --- Day/Trading Hours Logic ---
        if current_date != self.current_date_in_sim: # New Day
            if self.trade_info['is_open']:
                prev_step_price = self.df.loc[self.current_step -1, 'close']
                self._close_position(prev_step_price, reason="EOD_ROLL")
                done_for_episode = True
            self.current_date_in_sim = current_date
            self.current_day_trades = 0
            # If EOD closure makes episode done, return early. Agent gets new obs for new day.
            if done_for_episode:
                 obs = self._get_observation() # obs for start of new day after closure
                 return obs, reward, True, {'achieved_goal': np.array([0.0,0.0]), 'trades_today': self.current_day_trades}

        # Check if outside trading hours for active trading
        if current_time >= self.trading_end_time or current_time < self.trading_start_time:
            if self.trade_info['is_open']: # If caught holding outside hours (e.g. market just closed)
                self._close_position(current_price, reason="MARKET_CLOSE")
                done_for_episode = True
            # If not holding, just advance step. No new trades outside hours.
            action = 0 # Force hold/do nothing if outside hours
            if done_for_episode: # If market close forced closure and ended episode
                obs = self._get_observation()
                return obs, reward, True, {'achieved_goal': np.array([0.0,0.0]), 'trades_today': self.current_day_trades}


        # --- Action Handling (only if within trading hours) ---
        if self.trading_start_time <= current_time < self.trading_end_time:
            if action == 1 and not self.trade_info['is_open']: # Buy
                if self.current_day_trades < self.max_trades_per_day:
                    self.trade_info['is_open'] = True
                    self.trade_info['entry_price'] = current_price
                    investment = self.balance * 0.95
                    if current_price > 0: self.trade_info['position_size'] = investment / current_price
                    else: self.trade_info['is_open'] = False # Cannot buy at zero price

                    if self.trade_info['is_open']:
                        self.trade_info['entry_value'] = investment
                        self.trade_info['peak_value'] = investment
                        self.trade_info['entry_step'] = self.current_step
                        self.trade_info['entry_time'] = current_timestamp
                        self.balance -= investment
                        self.current_day_trades += 1
                else: action = 0 # Max trades for day reached, force hold

            elif action == 2 and self.trade_info['is_open']: # Sell
                self._close_position(current_price, reason="AGENT_SELL")
                done_for_episode = True
        else: # Outside trading hours, already handled above, but ensure action is passive
            action = 0

        # --- Intra-day trade duration limit ---
        if self.trade_info['is_open'] and (self.current_step - self.trade_info['entry_step']) >= self.intra_day_trade_max_steps:
            self._close_position(current_price, reason="INTRA_DAY_TIMEOUT")
            done_for_episode = True

        self.current_step += 1
        simulation_done = self.current_step >= self.max_data_steps # Is all data consumed?
        if simulation_done and self.trade_info['is_open']: # Final EOD if data ends
             self._close_position(self.df.loc[self.max_data_steps -1, 'close'], reason="DATA_END_FINAL")

        done_for_episode = done_for_episode or simulation_done

        observation = self._get_observation()
        # Achieved goal is from the perspective of the current state if a trade is open, else [0,0]
        achieved_goal_for_her = self._get_achieved_goal()

        info = {'achieved_goal': achieved_goal_for_her, 'trades_today': self.current_day_trades}
        if done_for_episode:
            # If a trade was just closed, its final PnL was part of _close_position.
            # For the purpose of 'is_success' against IDEAL_GOAL, it's tricky if IDEAL_GOAL is for a single trade.
            # Let's assume IDEAL_GOAL is about the PnL of the trade that *just ended* if one did.
            # This would require passing the PnL of the specific trade that ended.
            # For now, compute_reward will use achieved_goal_for_her which might be [0,0] if trade closed.
            # So, is_success here needs care.
             is_success_val = self.compute_reward(achieved_goal_for_her, IDEAL_GOAL) == 0.0 # Placeholder logic for now
             info['is_success'] = is_success_val

        return observation, reward, done_for_episode, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        pnl_achieved, drawdown_achieved = achieved_goal
        pnl_desired, drawdown_desired_limit = desired_goal
        if pnl_achieved >= pnl_desired and drawdown_achieved >= drawdown_desired_limit:
            return 0.0
        else:
            return -1.0

# --- 2. Hindsight Experience Replay Buffer ---
class HerReplayBuffer:
    def __init__(self, capacity, k_future_goals, env):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.k = k_future_goals
        self.env = env

    def add(self, episode_experience):
        for t, transition in enumerate(episode_experience):
            state, action, achieved_goal_in_next_state, next_state, desired_goal_episode = transition

            original_reward = self.env.compute_reward(achieved_goal_in_next_state, desired_goal_episode)
            self.buffer.append((state, action, original_reward, next_state, desired_goal_episode, False))

            future_indices = np.random.choice(np.arange(t, len(episode_experience)), size=self.k, replace=True)
            for future_index in future_indices:
                hindsight_goal = episode_experience[future_index][2] # This is info['achieved_goal'] from a future step
                hindsight_reward = self.env.compute_reward(achieved_goal_in_next_state, hindsight_goal)
                self.buffer.append((state, action, hindsight_reward, next_state, hindsight_goal, True))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)

# --- 3. Main Simulation Loop ---
if __name__ == "__main__":
    # --- Data Loading and Preprocessing ---
    try:
        # Load from the specified CSV file
        df = pd.read_csv("QQQ_1min_data_with_indicators.csv")
        print(f"Successfully loaded QQQ_1min_data_with_indicators.csv. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: QQQ_1min_data_with_indicators.csv not found. Using sample data instead.")
        df = pd.read_csv(io.StringIO(csv_data)) # Fallback to sample data
    except Exception as e:
        print(f"Error loading QQQ_1min_data_with_indicators.csv: {e}. Using sample data instead.")
        df = pd.read_csv(io.StringIO(csv_data)) # Fallback to sample data

    # **Step 1: Data Preprocessing and Date Handling**
    print("\n--- Starting Data Preprocessing and Date Handling ---")

    # Ensure 'date' column exists
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column for day trading logic.")

    # Convert 'date' column to datetime objects
    # Assuming the date format might be like 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD HH:MM:SS-ZZ:ZZ'
    # pd.to_datetime will try to infer the format. Add `utc=True` if timezone naive and you want UTC.
    # If specific format is known and causing issues, use format string e.g. format='%Y-%m-%d %H:%M:%S%z'
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    except Exception as e:
        print(f"Error converting 'date' column: {e}. Ensure it's in a recognizable format.")
        # Attempt to parse known common formats if generic fails
        try:
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce') # With timezone
        except:
             df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce') # Without timezone


    df.dropna(subset=['date'], inplace=True) # Remove rows where date conversion failed

    # Sort by date just in case it's not
    df.sort_values(by='date', inplace=True)

    # Extract 'date_only' (for EOD checks) and 'time_only'
    df['date_only'] = df['date'].dt.date
    df['time_only'] = df['date'].dt.time

    print(f"'date' column converted to datetime. 'date_only' and 'time_only' extracted.")

    # Handle missing values in 'close' and indicator columns
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if df['close'].isnull().any():
        print("Warning: 'close' column contains NaNs. Attempting to backfill then forward-fill...")
        df['close'].bfill(inplace=True) # Updated to new syntax
        df['close'].ffill(inplace=True) # Updated to new syntax
        if df['close'].isnull().any():
            # If still NaNs (e.g. whole column was NaN), raise error or fill with a constant if appropriate
            raise ValueError("Could not fill all NaNs in 'close' column. Please check data quality.")

    indicator_cols = ['RSI_14', 'MACD_12_26_9', 'MACDs_12_26_9', 'SMA_20']
    for col in indicator_cols:
        if col in df.columns:
            if df[col].isnull().any():
                print(f"Indicator column '{col}' contains NaNs. Backfilling then forward-filling.")
                df[col].bfill(inplace=True) # Updated to new syntax
                df[col].ffill(inplace=True) # Updated to new syntax
                if df[col].isnull().any():
                    print(f"Warning: Column '{col}' still has NaNs after fill; defaulting remaining to 0 or neutral.")
                    if col == 'RSI_14': df[col].fillna(50.0, inplace=True) # Neutral RSI
                    else: df[col].fillna(0.0, inplace=True)
        else:
            print(f"Warning: Indicator column '{col}' not found. Creating it with default values.")
            if col == 'RSI_14': df[col] = 50.0 # Default neutral RSI
            else: df[col] = 0.0 # Default neutral for others

    # Ensure all necessary columns for observation are present after filling
    if 'RSI_14' not in df.columns: df['RSI_14'] = 50.0
    if 'MACD_12_26_9' not in df.columns: df['MACD_12_26_9'] = 0.0
    if 'MACDs_12_26_9' not in df.columns: df['MACDs_12_26_9'] = 0.0

    # Consider dropping initial rows if indicators need warm-up and are still NaN
    # For example, if SMA_20 is used, the first 19 rows for SMA_20 will be NaN.
    # bfill/ffill handles this, but if you prefer to remove:
    # min_indicator_warmup = 20 # e.g. for SMA_20
    # df = df.iloc[min_indicator_warmup:].copy()

    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("DataFrame is empty after preprocessing. Check data and preprocessing steps.")

    print(f"Data preprocessing complete. Final Shape: {df.shape}")
    print("First 5 rows of data after full preprocessing:")
    print(df.head())
    print("\nLast 5 rows of data after full preprocessing:")
    print(df.tail())
    print("--- End of Data Preprocessing ---\n")

    # --- Setup ---
    # episode_max_steps for TradingEnv is now intra_day_trade_max_steps
    INTRA_DAY_MAX_STEPS = 60 # e.g., a trade should not last more than 60 minutes within a day
    MAX_TRADES_PER_DAY = 5

    # Define US market trading hours (can be adjusted)
    # Ensure these are datetime.time objects
    TRADING_START_TIME = datetime.time(9, 30)
    TRADING_END_TIME = datetime.time(16, 0) # Market close, positions should be closed by then

    env = TradingEnv(df, initial_balance=100000, commission=0.001,
                     intra_day_trade_max_steps=INTRA_DAY_MAX_STEPS,
                     max_trades_per_day=MAX_TRADES_PER_DAY,
                     trading_start_time=TRADING_START_TIME,
                     trading_end_time=TRADING_END_TIME)
    buffer = HerReplayBuffer(capacity=100000, k_future_goals=4, env=env)

    IDEAL_GOAL = np.array([0.02, -0.01])

    N_EPISODES = 50 # Number of "trade attempts" or "agent learning episodes"

    print(f"\n--- Starting simulation with {N_EPISODES} episodes ---")
    print(f"Data covers dates from {df['date_only'].min()} to {df['date_only'].max()}")
    print(f"Ideal Goal (per trade): PnL >= {IDEAL_GOAL[0]*100:.2f}%, Max Drawdown >= {IDEAL_GOAL[1]*100:.2f}%")
    print(f"Max {MAX_TRADES_PER_DAY} trades per day. Intra-day trade max steps: {INTRA_DAY_MAX_STEPS}.")
    print(f"Trading Hours: {TRADING_START_TIME} - {TRADING_END_TIME}")

    successful_ideal_goal_trades = 0 # Counter for trades that met the ideal goal
    total_steps_simulated = 0

    for i_episode in range(N_EPISODES):
        # Each episode starts from a potentially random point in the data, respecting day boundaries for trade counts.
        # The env.reset() now handles finding a valid start time within trading hours.
        state, episode_done_at_reset = env.reset() # Returns obs, done

        if episode_done_at_reset: # True if reset couldn't find a valid trading start point in the whole dataset
            print(f"Ep {i_episode+1:02d}/{N_EPISODES}: Skipped - No valid start point found by reset().")
            continue

        episode_experience = []
        done_for_episode = False # This 'done' is for the agent's current learning episode (trade sequence)
        episode_steps = 0

        print(f"\nEp {i_episode+1:02d}/{N_EPISODES}: Start St: {env.current_step}, Date: {env.df.loc[env.current_step, 'date_only']}, TradesToday: {env.current_day_trades}, Bal: ${env.balance:.2f}")

        while not done_for_episode:
            # Random Agent Logic
            if not env.trade_info['is_open']:
                action = 1 if random.random() < 0.5 else 0
            else:
                action = 2 if random.random() < 0.1 else 0

            next_state, reward_from_env, done_for_episode, info = env.step(action)
            achieved_goal_for_her = info['achieved_goal']

            episode_experience.append((state, action, achieved_goal_for_her, next_state, IDEAL_GOAL))
            state = next_state
            episode_steps += 1
            total_steps_simulated +=1

            # Safety break: if an episode runs too long without natural termination (should be caught by env limits)
            if episode_steps > (env.max_data_steps * 2) : # Arbitrary very large number
                 print(f"    SAFETY BREAK: Episode {i_episode+1} ran too long ({episode_steps} steps). Force ending.")
                 if env.trade_info['is_open']:
                     env._close_position(env.df.loc[min(env.current_step, env.max_data_steps-1), 'close'], "SAFETY_TIMEOUT")
                 done_for_episode = True


            if done_for_episode:
                # This 'info' dictionary might contain 'is_success' if the environment calculates it
                # based on the trade that just finished vs IDEAL_GOAL.
                # The current env.step doesn't explicitly calculate this for IDEAL_GOAL success.
                # The reward in HER buffer is what matters for learning.
                print(f"  Ep {i_episode+1:02d} ended. Steps: {episode_steps}. Sim Step: {env.current_step}/{env.max_data_steps}. TradesToday: {info.get('trades_today', env.current_day_trades)}. Bal: ${env.balance:.2f}")
                if info.get('is_success'): # If env provided this based on IDEAL_GOAL
                     successful_ideal_goal_trades += 1

        if episode_experience:
            buffer.add(episode_experience)

    print(f"\n--- Simulation finished ({N_EPISODES} episodes attempted) ---")
    print(f"Total steps simulated across all episodes: {total_steps_simulated}")
    print(f"Trades recorded in history for plotting: {len(env.trade_history)}")
    # print(f"Trades that met IDEAL_GOAL (approx): {successful_ideal_goal_trades}") # This is still approximate
    print(f"Final Balance: ${env.balance:.2f}")
    print(f"HER Buffer contains {len(buffer)} experiences (original + hindsight).")

    if len(buffer) > 0:
        print("\n--- Sampling 5 experiences from HER buffer to inspect: ---")
        sample_batch = buffer.sample(min(5, len(buffer)))
        for idx, (s, a, r, s_next, g, is_hindsight) in enumerate(sample_batch):
            print(f"\n--- Sample {idx+1} ---")
            obs_dim_names = ["PnL%", "DD%", "PosOpen", "RSI", "MACD", "TrdsToday", "TimeToEOD"]
            s_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(obs_dim_names, s)])
            s_next_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(obs_dim_names, s_next)])
            print(f"  State (s):       [{s_str}]")
            print(f"  Action (a):      {['Hold', 'Buy', 'Sell'][a]}")
            print(f"  Desired Goal (g): [PnL_tgt: {g[0]:.2%}, DD_tgt: {g[1]:.2%}]")
            print(f"  Reward (r):      {'Success (0.0)' if r == 0.0 else 'Failure (-1.0)'} (achieved in s_next vs g)")
            print(f"  Next State (s'): [{s_next_str}]")
            print(f"  Is Hindsight?    {'Yes' if is_hindsight else 'No'}")
    else:
        print("\nNo experiences in buffer to sample.")

    # --- Plotting ---
    def plot_trades_for_day(df_full, trade_history, target_date_obj):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        day_df = df_full[df_full['date_only'] == target_date_obj].copy()
        if day_df.empty:
            print(f"No data for plotting on {target_date_obj}")
            return

        day_trades = [t for t in trade_history if t['entry_time'].date() == target_date_obj or t['exit_time'].date() == target_date_obj]

        if not day_trades and day_df.empty: # Check if there's anything to plot for this day
            print(f"No trades or price data to plot for {target_date_obj}")
            return

        plt.figure(figsize=(15, 7))
        plt.plot(day_df['date'], day_df['close'], label='Close Price', alpha=0.7)

        buy_times = []
        buy_prices = []
        sell_times = []
        sell_prices = []

        for trade in day_trades:
            # Ensure entry is on the target day
            if trade['entry_time'].date() == target_date_obj:
                buy_times.append(trade['entry_time'])
                buy_prices.append(trade['entry_price'])
            # Ensure exit is on the target day
            if trade['exit_time'].date() == target_date_obj:
                sell_times.append(trade['exit_time'])
                sell_prices.append(trade['exit_price'])

        plt.scatter(buy_times, buy_prices, marker='^', color='green', s=100, label='Buy', alpha=1, zorder=5)
        plt.scatter(sell_times, sell_prices, marker='v', color='red', s=100, label='Sell', alpha=1, zorder=5)

        plt.title(f"Trading Activity on {target_date_obj}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        # Format x-axis to show time properly
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator()) # Auto tick placement
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save or show plot
        plot_filename = f"trades_{target_date_obj}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        # plt.show() # Uncomment to display plot interactively if in a suitable environment

    # Plot for each day that has trades
    if env.trade_history:
        print("\n--- Generating Trade Plots ---")
        # Get unique days from trade history (considering both entry and exit days)
        trade_days = set()
        for t in env.trade_history:
            trade_days.add(t['entry_time'].date())
            trade_days.add(t['exit_time'].date())

        sorted_trade_days = sorted(list(trade_days))

        for day_to_plot in sorted_trade_days:
            # Check if this day exists in the original DataFrame to prevent errors if trades span beyond df dates
            if not df[df['date_only'] == day_to_plot].empty:
                 plot_trades_for_day(df, env.trade_history, day_to_plot)
            else:
                print(f"Skipping plot for {day_to_plot} as it's not in the loaded price data range (e.g. trade closed on next day's open).")

    elif not df.empty: # If no trades, but we have data, maybe plot first day's prices?
        print("\nNo trades were made to plot. Plotting first day of price data as an example.")
        first_day_in_df = df['date_only'].min()
        plot_trades_for_day(df, [], first_day_in_df)


print("\n--- Script execution complete. ---")
