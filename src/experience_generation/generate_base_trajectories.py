import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
RL_MODEL_DIR = "rl_models" # VAE encoder and scaler are here
FEATURES_FILE = os.path.join(DATA_DIR, "qqq_features.parquet")
# We need raw price data for P&L calculation, assuming 'close' from qqq_1min_1month.csv
RAW_DATA_FILE = os.path.join(DATA_DIR, "qqq_1min_1month.csv")

VAE_ENCODER_FILE = os.path.join(RL_MODEL_DIR, "vae_encoder.h5")
VAE_SCALER_FILE = os.path.join(RL_MODEL_DIR, "vae_feature_scaler.joblib")

TRAJECTORIES_FILE = os.path.join(DATA_DIR, "base_trajectories.parquet")

# Action space for long-only strategy
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL_EXIT = 2 # Sell to close a long position

# Parameters for MACD Crossover Strategy
# These columns should exist in qqq_features.parquet
# MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
MACD_LINE_COL = "MACD_12_26_9"
MACD_SIGNAL_COL = "MACDs_12_26_9"
# MACD_HIST_COL = "MACDh_12_26_9" # Histogram, difference between line and signal

# Simulation parameters
INITIAL_CAPITAL = 100000 # Example starting capital
TRADE_SIZE_PERCENT = 0.10 # Use 10% of capital per trade
SLIPPAGE_PERCENT = 0.0005 # 0.05% slippage per transaction (buy or sell)
TRANSACTION_COST_PERCENT = 0.0005 # 0.05% transaction cost per transaction

class Sampling(layers.Layer): # Required for loading VAE encoder
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def load_data_and_models():
    """Loads features, raw prices, VAE encoder, and scaler."""
    logger.info("Loading data and models...")
    if not all(os.path.exists(f) for f in [FEATURES_FILE, RAW_DATA_FILE, VAE_ENCODER_FILE, VAE_SCALER_FILE]):
        logger.error("One or more required files are missing. Ensure previous steps ran successfully.")
        return None, None, None, None, None

    try:
        features_df = pd.read_parquet(FEATURES_FILE)

        # Load raw QQQ data for close prices
        raw_prices_df = pd.read_csv(RAW_DATA_FILE, index_col='date', parse_dates=True)
        # Ensure columns are lowercase, especially 'close'
        raw_prices_df.columns = [col.lower() for col in raw_prices_df.columns]

        # Align features_df and raw_prices_df by index (timestamps)
        # The features_df might have fewer rows due to NaN dropping from indicator calculation.
        # We need the close prices corresponding to the timestamps in features_df.
        aligned_idx = features_df.index.intersection(raw_prices_df.index)
        features_df = features_df.loc[aligned_idx]
        prices_for_features = raw_prices_df.loc[aligned_idx, ['close']] # Only need close price for simulation

        if features_df.empty or prices_for_features.empty:
            logger.error("Feature data or price data is empty after alignment.")
            return None, None, None, None, None

        logger.info(f"Features shape after alignment: {features_df.shape}")
        logger.info(f"Prices shape after alignment: {prices_for_features.shape}")

        scaler = joblib.load(VAE_SCALER_FILE)
        encoder = tf.keras.models.load_model(VAE_ENCODER_FILE, custom_objects={'Sampling': Sampling})

        # Transform features to state vectors 'z'
        # Ensure feature columns used for scaling/training VAE are present and in correct order
        # The scaler was fit on features_df.values, so the order is implicitly handled if
        # features_df loaded here has the same columns in the same order as during VAE training.

        # Check if columns match those used for scaler (important if subset was used)
        # For now, assume scaler was fit on all columns of features_df.
        scaled_features = scaler.transform(features_df.values)
        z_mean, _, z_sampled = encoder.predict(scaled_features)

        # Use z_mean as the deterministic state representation
        # Create a DataFrame for states 'z' with the same index as features_df
        state_z_df = pd.DataFrame(z_mean, index=features_df.index, columns=[f"z_{i}" for i in range(z_mean.shape[1])])
        logger.info(f"State vectors 'z' generated. Shape: {state_z_df.shape}")

        return features_df, prices_for_features, state_z_df, scaler, encoder
    except Exception as e:
        logger.error(f"Error loading data or models: {e}", exc_info=True)
        return None, None, None, None, None

def macd_strategy_signal(current_features: pd.Series, prev_features: pd.Series) -> int:
    """
    Determines action based on MACD crossover.
    Assumes current_features and prev_features contain MACD_LINE_COL and MACD_SIGNAL_COL.

    Returns:
        ACTION_BUY if bullish crossover.
        ACTION_SELL_EXIT if bearish crossover (interpreted as exit signal for long-only).
        ACTION_HOLD otherwise.
    """
    if prev_features is None: # Not enough history for crossover
        return ACTION_HOLD

    # Bullish crossover: MACD line crosses above Signal line
    # (MACD_t-1 < Signal_t-1) AND (MACD_t > Signal_t)
    if (prev_features[MACD_LINE_COL] < prev_features[MACD_SIGNAL_COL] and
            current_features[MACD_LINE_COL] > current_features[MACD_SIGNAL_COL]):
        return ACTION_BUY

    # Bearish crossover: MACD line crosses below Signal line
    # (MACD_t-1 > Signal_t-1) AND (MACD_t < Signal_t)
    if (prev_features[MACD_LINE_COL] > prev_features[MACD_SIGNAL_COL] and
            current_features[MACD_LINE_COL] < current_features[MACD_SIGNAL_COL]):
        return ACTION_SELL_EXIT # Signal to exit if in a long position

    return ACTION_HOLD


def run_simulation_and_collect_trajectories(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    state_z_df: pd.DataFrame):
    """
    Runs the MACD Crossover strategy and collects (s, a, r, s', done) trajectories.
    """
    logger.info("Running simulation to generate base trajectories...")
    trajectories = []

    capital = INITIAL_CAPITAL
    position_size_contracts = 0
    entry_price = 0
    in_position = False

    num_timesteps = len(features_df)
    if num_timesteps <= 1:
        logger.warning("Not enough data points to run simulation.")
        return []

    # Iterate through each timestep
    for t in tqdm(range(num_timesteps -1), desc="Simulating"): # -1 because we need s_next
        current_timestamp = features_df.index[t]
        current_features = features_df.iloc[t]
        current_price_data = prices_df.loc[current_timestamp] # Contains 'close'
        current_close = current_price_data['close']
        current_state_z = state_z_df.iloc[t].values # numpy array for s_t

        prev_features = features_df.iloc[t-1] if t > 0 else None

        # 1. Get action from behavior policy (MACD strategy)
        strategy_action = macd_strategy_signal(current_features, prev_features)

        # Refine action based on current position status
        actual_action_taken = ACTION_HOLD # Default to hold
        if strategy_action == ACTION_BUY and not in_position:
            actual_action_taken = ACTION_BUY
        elif strategy_action == ACTION_SELL_EXIT and in_position:
            actual_action_taken = ACTION_SELL_EXIT

        # Store current state
        s_t = current_state_z
        a_t = actual_action_taken

        # Initialize reward for this step
        r_t = 0.0

        # 2. Simulate trade execution and calculate reward
        if actual_action_taken == ACTION_BUY:
            # Buy to enter position
            trade_value = capital * TRADE_SIZE_PERCENT
            entry_price_slippage = current_close * (1 + SLIPPAGE_PERCENT)
            position_size_contracts = trade_value / entry_price_slippage
            cost = position_size_contracts * entry_price_slippage * TRANSACTION_COST_PERCENT

            capital -= (position_size_contracts * entry_price_slippage) + cost
            entry_price = entry_price_slippage
            in_position = True
            logger.debug(f"{current_timestamp}: BUY {position_size_contracts:.2f} at {entry_price:.2f} (Close: {current_close:.2f}). Capital: {capital:.2f}")
            # Reward for buy action itself is often 0, reward comes from holding or selling.
            # Or, a small negative reward for transaction costs.
            r_t = -cost

        elif actual_action_taken == ACTION_SELL_EXIT:
            # Sell to exit position
            exit_price_slippage = current_close * (1 - SLIPPAGE_PERCENT)
            proceeds = position_size_contracts * exit_price_slippage
            cost = proceeds * TRANSACTION_COST_PERCENT

            capital += proceeds - cost

            # Calculate P&L for this trade
            trade_pnl = (exit_price_slippage - entry_price) * position_size_contracts - (cost + (entry_price * position_size_contracts * TRANSACTION_COST_PERCENT)) # pnl after costs
            r_t = trade_pnl # Reward is the P&L of the closed trade
            logger.info(f"{current_timestamp}: SELL {position_size_contracts:.2f} at {exit_price_slippage:.2f} (Close: {current_close:.2f}). P&L: {trade_pnl:.2f}. Capital: {capital:.2f}")

            in_position = False
            position_size_contracts = 0
            entry_price = 0

        # If holding a position, reward can be shaped (e.g., change in value of open position)
        # For now, let's keep reward simple: P&L on close, cost on open, 0 on hold.
        # More sophisticated reward shaping can be added later if needed.
        # if in_position and actual_action_taken == ACTION_HOLD:
        #    # Example: reward is the change in value of the open position for this timestep
        #    # current_position_value = position_size_contracts * current_close
        #    # prev_close_for_reward_calc = prices_df['close'].iloc[t-1] if t > 0 else entry_price
        #    # prev_position_value = position_size_contracts * prev_close_for_reward_calc
        #    # r_t = current_position_value - prev_position_value
        #    pass


        # 3. Get next state
        s_next_t = state_z_df.iloc[t+1].values

        # 4. Determine 'done' flag
        # 'done' is true if the episode ends. In continuous trading, an "episode" might be a single trade.
        # Or, if it's the last data point.
        done_flag = (actual_action_taken == ACTION_SELL_EXIT) or (t == num_timesteps - 2) # -2 because loop is num_timesteps-1

        trajectories.append({
            "s_t": s_t.tolist(), # Convert numpy array to list for Parquet storage if not handled well
            "a_t": a_t,
            "r_t": r_t,
            "s_next_t": s_next_t.tolist(),
            "done": done_flag,
            "timestamp": current_timestamp # For debugging and alignment
        })

    logger.info(f"Simulation finished. Generated {len(trajectories)} trajectory points.")
    logger.info(f"Final capital: {capital:.2f}")
    return pd.DataFrame(trajectories)


def main():
    logger.info("Starting base experience generation process...")

    # Ensure model directory exists (it should, but good check)
    if not os.path.exists(RL_MODEL_DIR):
        os.makedirs(RL_MODEL_DIR, exist_ok=True)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    features_df, prices_df, state_z_df, _, _ = load_data_and_models()

    if features_df is None or prices_df is None or state_z_df is None:
        logger.error("Failed to load necessary data/models. Aborting trajectory generation.")
        return

    # Check if MACD columns exist in features_df
    if not {MACD_LINE_COL, MACD_SIGNAL_COL}.issubset(features_df.columns):
        logger.error(f"Required MACD columns ({MACD_LINE_COL}, {MACD_SIGNAL_COL}) not found in features file.")
        logger.error(f"Available columns: {features_df.columns.tolist()}")
        return

    trajectories_df = run_simulation_and_collect_trajectories(features_df, prices_df, state_z_df)

    if trajectories_df is not None and not trajectories_df.empty:
        try:
            # Convert list columns to appropriate types if Parquet has issues (e.g. nested lists for states)
            # Parquet handles lists of primitives well. For lists of lists (if states were sequences), might need care.
            # Here s_t and s_next_t are flat lists of numbers, should be fine.
            trajectories_df.to_parquet(TRAJECTORIES_FILE, index=False)
            logger.info(f"Base trajectories saved to {TRAJECTORIES_FILE}")
            logger.info(f"Trajectories DataFrame head:\n{trajectories_df.head()}")
        except Exception as e:
            logger.error(f"Error saving trajectories to {TRAJECTORIES_FILE}: {e}", exc_info=True)
    else:
        logger.error("No trajectories generated or an error occurred.")

    logger.info("Base experience generation process finished.")


if __name__ == "__main__":
    # GPU setup (less critical for this script but good practice if TF is used)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPU found, using CPU (VAE encoder prediction).")
    main()

```
A note on state representation `s_t`:
The request mentions "Run a simple 'behavior policy' (e.g., MACD crossover) on the historical data to generate a base set of trajectories (s, a, r, s')."
Here, `s` is the state vector `z` obtained from the VAE. The MACD policy itself will operate on the MACD values from the *feature* DataFrame, but the `s` recorded in the trajectory will be the compressed `z`.

Reward Calculation:
The current reward `r_t` is:
-   `-cost` if action is BUY (cost of transaction).
-   `trade_pnl` if action is SELL_EXIT (profit or loss from the trade).
-   `0` if action is HOLD.
This is a common way to define rewards. For HER, goal-based rewards will be calculated later.

This script is now quite comprehensive. It loads data, models, runs a basic simulation with a MACD strategy, and saves the experiences.
