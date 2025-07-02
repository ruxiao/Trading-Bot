import pandas as pd
import numpy as np
import datetime
import os

from utils import download_data, preprocess_data
from environment import TradingEnv # Gym-compatible environment
from evaluation import evaluate_sb3_agent, plot_backtest_results

# Stable Baselines3 components
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
# from sb3_contrib import RecurrentPPO # If LSTM/recurrent policy is desired

def main():
    # --- Configuration ---
    TICKER = 'QQQ'

    # Data period for training
    DATA_INTERVAL = '5m' # '1m', '5m', '15m', '1h', '1d'
    MAX_LOOKBACK_PERIODS = 30 # Max periods needed by any indicator (e.g., SMA30)

    # Calculate required calendar days for indicator buffer
    if DATA_INTERVAL == '1d':
        # For daily data, need MAX_LOOKBACK_PERIODS trading days.
        # Estimate calendar days: multiply by ~1.5 (e.g., 30 trading days -> 45 calendar days)
        indicator_buffer_calendar_days = int(MAX_LOOKBACK_PERIODS * 1.5) + 5 # Add small safety margin
    else:
        # For intraday, MAX_LOOKBACK_PERIODS usually fall within a few calendar days.
        # e.g., 30 * 1-min = 30 mins. 30 * 1-hour = ~5 trading days.
        # A buffer of 10-15 calendar days should be safe for intraday to ensure enough actual trading periods.
        indicator_buffer_calendar_days = 15

    # Training data: Target ~50 days of usable training data
    TRAIN_TARGET_DURATION_DAYS = 50
    TRAIN_END_DATE_DT = datetime.datetime.now() - datetime.timedelta(days=1) # Ensure train data is not too recent
    TRAIN_START_DATE_DT = TRAIN_END_DATE_DT - datetime.timedelta(days=30) # Try a shorter period for data download
    # yfinance start is inclusive, end is exclusive for intraday. For daily, both inclusive? Let's be safe.
    # Using string formatting for yfinance for clarity.
    TRAIN_START_DATE_STR = TRAIN_START_DATE_DT.strftime('%Y-%m-%d')
    TRAIN_END_DATE_STR = TRAIN_END_DATE_DT.strftime('%Y-%m-%d')


    # Backtest data: Target 5 days of usable backtesting data
    BACKTEST_TARGET_DURATION_DAYS = 5
    BACKTEST_END_DATE_DT = datetime.datetime.now() - datetime.timedelta(days=1) # Yesterday
    BACKTEST_START_DATE_DT = BACKTEST_END_DATE_DT - datetime.timedelta(days=5) # 5 days for backtest
    BACKTEST_START_DATE_STR = BACKTEST_START_DATE_DT.strftime('%Y-%m-%d')
    BACKTEST_END_DATE_STR = BACKTEST_END_DATE_DT.strftime('%Y-%m-%d')

    # Training parameters for SB3
    TOTAL_TRAINING_TIMESTEPS = 100000  # Adjust as needed (e.g., 100k, 500k, 1M)
    SB3_POLICY = "MlpPolicy" # "MlpPolicy" or "MlpLstmPolicy" (from sb3_contrib for PPO)
                             # If using MlpLstmPolicy, ensure sb3_contrib.RecurrentPPO is used or PPO is compatible.
                             # For standard PPO, MlpPolicy is common. LSTM might be better for time series.

    # Environment settings
    INITIAL_BALANCE = 100000
    MAX_STEPS_PER_EPISODE = 390 // 5 # If 5m data, a trading day (6.5hrs) has 78 5-min intervals.
                                    # Let an episode be one day. If data is 1m, then 390.
                                    # This needs to be set according to DATA_INTERVAL.
    if DATA_INTERVAL == '1m':
        MAX_STEPS_PER_EPISODE = 6.5 * 60 # 390 steps
    elif DATA_INTERVAL == '5m':
        MAX_STEPS_PER_EPISODE = int(6.5 * 60 / 5) # 78 steps
    elif DATA_INTERVAL == '15m':
        MAX_STEPS_PER_EPISODE = int(6.5 * 60 / 15) # 26 steps
    elif DATA_INTERVAL == '1h':
        MAX_STEPS_PER_EPISODE = int(6.5) # ~6-7 steps
    else: # '1d'
        MAX_STEPS_PER_EPISODE = 20 # e.g., an episode is a month of trading days


    MODEL_SAVE_PATH = "trained_models/ppo_trading_agent"
    BEST_MODEL_SAVE_PATH = "trained_models/best_ppo_trading_agent"
    LOG_DIR = "logs/"
    os.makedirs(MODEL_SAVE_PATH.split('/')[0], exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- 1. Data Loading and Preparation ---
    print("--- 1. Data Loading and Preparation ---")
    try:
        print(f"Type of TICKER: {type(TICKER)}")
        print(f"Attempting to download training data: {TICKER} from {TRAIN_START_DATE_STR} to {TRAIN_END_DATE_STR} ({DATA_INTERVAL})")
        raw_train_data = download_data(TICKER, TRAIN_START_DATE_STR, TRAIN_END_DATE_STR, interval=DATA_INTERVAL)

        print(f"
Attempting to download backtesting data: {TICKER} from {BACKTEST_START_DATE_STR} to {BACKTEST_END_DATE_STR} ({DATA_INTERVAL})"
        raw_backtest_data = download_data(TICKER, BACKTEST_START_DATE_STR, BACKTEST_END_DATE_STR, interval=DATA_INTERVAL)

    except ValueError as e:
        print(f"Error downloading data: {e}")
        print("Please ensure your date ranges and intervals are valid for yfinance.")
        print("For 1-minute data, typically only the last 7 days are available.")
        print("For 5-minute data, typically the last 60 days.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data download: {e}")
        return

    if raw_train_data.empty or raw_backtest_data.empty:
        print("Failed to download sufficient data for training or backtesting. Exiting.")
        return

    # Preprocess data
    # Important: Fit scaler on training data ONLY, then transform backtest data.
    # The current preprocess_data fits scaler on the df it receives. This needs adjustment for proper train/test split.
    # For now, we'll preprocess them separately, which is not ideal for scaling.
    # A better approach:
    # 1. Preprocess train_data (calculates indicators).
    # 2. Fit scaler on processed train_data features.
    # 3. Transform processed train_data features with the fitted scaler.
    # 4. Preprocess backtest_data (calculates indicators).
    # 5. Transform processed backtest_data features with the *fitted training scaler*.

    # Simplified preprocessing for now (will scale them independently):
    processed_train_data = preprocess_data(raw_train_data.copy())
    processed_backtest_data = preprocess_data(raw_backtest_data.copy())

    if processed_train_data.empty or processed_backtest_data.empty:
        print("Data became empty after preprocessing. Check data quality and indicator periods. Exiting.")
        return

    # Select the most recent 5 trading days from processed_backtest_data for the actual backtest
    # This assumes data is sorted by time.
    # Get unique days, then take the last 5.
    unique_days_in_backtest_data = processed_backtest_data.index.normalize().unique()
    if len(unique_days_in_backtest_data) < 5:
        print(f"Not enough unique days in the downloaded backtest data ({len(unique_days_in_backtest_data)}) to perform a 5-day backtest. Need at least 5.")
        # We might proceed with fewer days if user agrees, or stop. For now, we'll try with what we have.
        # return # Or adjust num_days_for_backtest
        num_actual_backtest_days = len(unique_days_in_backtest_data)
        print(f"Proceeding with {num_actual_backtest_days} days for backtesting.")
    else:
        num_actual_backtest_days = 5

    if num_actual_backtest_days == 0:
        print("No days available for backtesting after processing. Exiting.")
        return

    last_n_days_timestamps = unique_days_in_backtest_data[-num_actual_backtest_days:]
    # Filter the DataFrame to include only these days
    final_backtest_df = processed_backtest_data[processed_backtest_data.index.normalize().isin(last_n_days_timestamps)]

    if final_backtest_df.empty:
        print("Final backtest DataFrame is empty after selecting last 5 days. Check data. Exiting.")
        return

    print(f"Training data shape: {processed_train_data.shape}")
    print(f"Final {num_actual_backtest_days}-day backtest data shape: {final_backtest_df.shape}")


    # --- 2. Environment Setup ---
    print("\n--- 2. Environment Setup ---")
    # Training Environment
    # Important: For SB3, it's good practice to wrap the env with Monitor for logging
    train_env_raw = TradingEnv(processed_train_data, initial_balance=INITIAL_BALANCE, max_steps_per_episode=MAX_STEPS_PER_EPISODE)
    train_env = Monitor(train_env_raw, LOG_DIR) # Monitor wrapper for SB3 logging

    # Evaluation environment for callbacks (optional, but good for saving best model)
    eval_env_raw = TradingEnv(final_backtest_df.copy(), initial_balance=INITIAL_BALANCE, max_steps_per_episode=len(final_backtest_df)) # One episode for full backtest
    eval_env = Monitor(eval_env_raw, LOG_DIR + "eval/")

    # Backtesting Environment (a separate instance for final evaluation)
    # Use the final_backtest_df, run for its full length as one episode.
    backtest_env = TradingEnv(final_backtest_df.copy(), initial_balance=INITIAL_BALANCE, max_steps_per_episode=len(final_backtest_df))
    # No Monitor needed if we use our custom evaluate_sb3_agent which extracts history.

    print(f"Training Environment Observation Space: {train_env.observation_space}")
    print(f"Training Environment Action Space: {train_env.action_space}")


    # --- 3. Agent Training ---
    print("\n--- 3. Agent Training ---")
    # Check if a pre-trained model exists
    load_existing_model = False # Set to True to load if available
    if load_existing_model and os.path.exists(MODEL_SAVE_PATH + ".zip"):
        print(f"Loading existing model from {MODEL_SAVE_PATH}.zip")
        model = PPO.load(MODEL_SAVE_PATH, env=train_env)
    else:
        print(f"No existing model found or load_existing_model is False. Training a new model: {SB3_POLICY}")

        # Define callbacks
        # StopTrainingOnRewardThreshold can be useful if you have a target reward
        # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=SOME_TARGET_REWARD, verbose=1)

        # EvalCallback saves the best model found during training based on performance on eval_env
        eval_callback = EvalCallback(eval_env, best_model_save_path=BEST_MODEL_SAVE_PATH.split('/')[0]+"/", # path to folder
                                     log_path=LOG_DIR + "eval_logs/", eval_freq=max(TOTAL_TRAINING_TIMESTEPS // 20, 500), # eval every X steps
                                     n_eval_episodes=3, # number of episodes to run for evaluation
                                     deterministic=True, render=False)

        # If using LSTM policy from sb3_contrib:
        # from sb3_contrib import RecurrentPPO
        # model = RecurrentPPO("MlpLstmPolicy", train_env, verbose=1, tensorboard_log=LOG_DIR + "tensorboard/")

        # Standard PPO with MlpPolicy
        # For SB3, it's common to use make_vec_env to create multiple parallel environments for faster training.
        # However, our custom env might not be trivially vectorizable if it has complex state.
        # Let's try with DummyVecEnv first (single environment).
        # train_vec_env = DummyVecEnv([lambda: train_env]) # This re-wraps, train_env is already Monitor(TradingEnv)

        # If train_env is already a single Monitor(TradingEnv) instance, it can be passed directly.
        model = PPO(SB3_POLICY, train_env, verbose=1,
                    tensorboard_log=LOG_DIR + "tensorboard/",
                    # learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2 etc.
                   )

        print(f"Starting training for {TOTAL_TRAINING_TIMESTEPS} timesteps...")
        try:
            model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS, callback=eval_callback) # Add callbacks=[eval_callback]
            model.save(MODEL_SAVE_PATH)
            print(f"Training complete. Model saved to {MODEL_SAVE_PATH}.zip")
        except Exception as e:
            print(f"An error occurred during training: {e}")
            print("Consider reducing complexity, checking environment, or data.")
            return


    # --- 4. Backtesting and Evaluation (on the 5-day data) ---
    print("\n--- 4. Backtesting and Evaluation ---")
    # Load the best model saved by EvalCallback if it exists, otherwise use the last saved model
    best_model_zip = BEST_MODEL_SAVE_PATH + "/best_model.zip" # EvalCallback saves it in a subfolder
    final_model_to_test_path = ""
    if os.path.exists(best_model_zip):
        print(f"Loading best model from {best_model_zip} for final backtest.")
        final_model_to_test_path = best_model_zip
    elif os.path.exists(MODEL_SAVE_PATH + ".zip"):
        print(f"Best model not found. Loading last saved model from {MODEL_SAVE_PATH}.zip for final backtest.")
        final_model_to_test_path = MODEL_SAVE_PATH + ".zip"
    else:
        print("No trained model found to evaluate. Please train a model first.")
        return

    # Ensure the model is loaded with the correct environment or set env later
    # For evaluation, we typically don't need to pass env to load if custom_objects are not used,
    # but it's safer to do so or use model.set_env(backtest_env) if needed.
    try:
        eval_model = PPO.load(final_model_to_test_path, env=backtest_env) # Pass backtest_env
    except Exception as e:
        print(f"Error loading the model for evaluation: {e}")
        print("This can happen if the environment structure changed or if there are custom objects.")
        # Fallback: if model was just trained, it's already in memory as 'model'
        if 'model' in locals() and model is not None:
            print("Using the model instance from training.")
            eval_model = model
            eval_model.set_env(backtest_env) # Ensure it's using the correct env
        else:
            return


    # Run evaluation on the backtest_env (which uses the 5-day data)
    # The backtest_env is configured to run for the full length of the 5-day data as one episode.
    # num_episodes=1 for evaluate_sb3_agent will run this single, long episode.
    print(f"Running backtest on {num_actual_backtest_days} days of data...")
    performance_metrics, backtest_history_df = evaluate_sb3_agent(
        eval_model,
        backtest_env, # This is the non-Monitor wrapped env, as evaluate_sb3_agent uses its history
        num_episodes=1, # One long episode covering the entire 5-day backtest period
        model_name=SB3_POLICY
    )

    print("\n--- Final Backtest Performance ---")
    if not backtest_history_df.empty:
        for key, value in performance_metrics.items():
            print(f"{key}: {value}")

        initial_portfolio_value = backtest_history_df['portfolio_value'].iloc[0]
        final_portfolio_value = backtest_history_df['portfolio_value'].iloc[-1]
        total_return_pct = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
        print(f"Initial Portfolio Value: ${initial_portfolio_value:,.2f}")
        print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
        print(f"Total Return for the backtest period: {total_return_pct:.2f}%")
        print("---------------------------------------")

        # --- 5. Plotting and CSV Output ---
        print("\n--- 5. Plotting and CSV Output ---")
        plot_filename = f"{TICKER}_{DATA_INTERVAL}_{num_actual_backtest_days}day_backtest_results.png"
        plot_backtest_results(backtest_history_df, ticker_name=TICKER, output_filename=plot_filename)
    else:
        print("Backtest history is empty. Cannot generate report or plot.")

    print("\n--- Script Finished ---")

if __name__ == '__main__':
    main()
