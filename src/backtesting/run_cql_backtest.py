import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers # For Sampling class if directly used
import joblib
from tqdm import tqdm
import pandas_ta as ta # For ATR calculation for dynamic goals

# Need to import VAE's Sampling layer definition if vae_encoder.h5 uses it as a custom layer
# Assuming VAE encoder was saved without the custom object or it's simple enough
# Or, redefine it here:
class Sampling(layers.Layer): # Required for loading VAE encoder
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Import feature calculation logic from feature_engineering.build_features
# This assumes build_features.py is structured to allow calling calculate_features
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add src to path
from feature_engineering.build_features import calculate_features as calculate_technical_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
RL_MODEL_DIR = "rl_models"
RAW_PRICES_FILE = os.path.join(DATA_DIR, "qqq_1min_1month.csv") # Source for backtest data
BACKTEST_RESULTS_FILE = os.path.join(DATA_DIR, "cql_backtest_results.csv")
EQUITY_CURVE_FILE = os.path.join(DATA_DIR, "cql_equity_curve.csv")


VAE_ENCODER_FILE = os.path.join(RL_MODEL_DIR, "vae_encoder.h5")
VAE_SCALER_FILE = os.path.join(RL_MODEL_DIR, "vae_feature_scaler.joblib")
CQL_ACTOR_FILE = os.path.join(RL_MODEL_DIR, "cql_actor_goal_conditioned.h5")

# Backtest Configuration
BACKTEST_DATA_SPLIT_RATIO = 0.7 # Use first 70% for training features, last 30% for backtesting
INITIAL_CAPITAL = 100000
TRADE_SIZE_FIXED_contracts = 10 # Number of contracts per trade for simplicity
                               # Or make it dynamic based on capital: TRADE_RISK_PER_TRADE = 0.01 * INITIAL_CAPITAL

SLIPPAGE_PERCENT = 0.0005 # 0.05%
TRANSACTION_COST_PERCENT = 0.0005 # 0.05%

# Goal Setting Configuration
GOAL_SETTING_MODE = "dynamic_atr" # "fixed" or "dynamic_atr"
FIXED_PROFIT_TARGET_PERCENT = 0.0075  # +0.75%
FIXED_STOP_LOSS_PERCENT = -0.0040 # -0.4% (negative value)
ATR_PERIOD_FOR_GOAL = 14 # ATR period for dynamic goals
ATR_PROFIT_MULTIPLIER = 1.0 # Target profit: 1.0 * ATR
ATR_STOP_LOSS_MULTIPLIER = 0.5 # Stop loss: 0.5 * ATR

# Action space (consistent with training)
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL_EXIT = 2

def load_models_and_data_for_backtest():
    logger.info("Loading models and data for backtesting...")
    required_files = [RAW_PRICES_FILE, VAE_ENCODER_FILE, VAE_SCALER_FILE, CQL_ACTOR_FILE]
    if not all(os.path.exists(f) for f in required_files):
        missing = [f for f in required_files if not os.path.exists(f)]
        logger.error(f"Missing required files for backtest: {missing}")
        return None, None, None, None, None

    try:
        raw_prices_df_full = pd.read_csv(RAW_PRICES_FILE, index_col='date', parse_dates=True)
        raw_prices_df_full.columns = [col.lower() for col in raw_prices_df_full.columns]

        # Split data: Use last part for backtesting
        split_index = int(len(raw_prices_df_full) * BACKTEST_DATA_SPLIT_RATIO)
        backtest_raw_df = raw_prices_df_full.iloc[split_index:].copy()

        if len(backtest_raw_df) < ATR_PERIOD_FOR_GOAL + 50: # Need enough data for features and ATR
             logger.error(f"Backtest data partition too small ({len(backtest_raw_df)} bars). Adjust split or get more data.")
             return None, None, None, None, None

        logger.info(f"Full raw data len: {len(raw_prices_df_full)}, Backtest raw data len: {len(backtest_raw_df)}")

        vae_encoder = tf.keras.models.load_model(VAE_ENCODER_FILE, custom_objects={'Sampling': Sampling})
        feature_scaler = joblib.load(VAE_SCALER_FILE) # This is the scaler for VAE features
        cql_actor = tf.keras.models.load_model(CQL_ACTOR_FILE)

        return backtest_raw_df, vae_encoder, feature_scaler, cql_actor, raw_prices_df_full.columns.tolist()

    except Exception as e:
        logger.error(f"Error loading models/data for backtest: {e}", exc_info=True)
        return None, None, None, None, None

def get_dynamic_goal(current_price_data_series: pd.Series, mode: str):
    """
    Determines the goal for the next potential trade.
    Args:
        current_price_data_series (pd.Series): Series containing at least 'close' and 'atr_X' (if dynamic)
        mode (str): "fixed" or "dynamic_atr"
    Returns:
        tuple: (profit_target_percent, stop_loss_percent)
    """
    if mode == "fixed":
        return FIXED_PROFIT_TARGET_PERCENT, FIXED_STOP_LOSS_PERCENT
    elif mode == "dynamic_atr":
        atr_col_name = f"ATR_{ATR_PERIOD_FOR_GOAL}" # Consistent with pandas-ta default
        if atr_col_name not in current_price_data_series or pd.isna(current_price_data_series[atr_col_name]):
            logger.warning(f"ATR value not available or NaN for dynamic goal setting. Falling back to fixed goals for this step.")
            return FIXED_PROFIT_TARGET_PERCENT, FIXED_STOP_LOSS_PERCENT

        current_atr = current_price_data_series[atr_col_name]
        current_close = current_price_data_series['close']
        if current_close == 0: return FIXED_PROFIT_TARGET_PERCENT, FIXED_STOP_LOSS_PERCENT # Avoid division by zero

        profit_target = (ATR_PROFIT_MULTIPLIER * current_atr) / current_close
        stop_loss = -(ATR_STOP_LOSS_MULTIPLIER * current_atr) / current_close # Negative value
        return profit_target, stop_loss
    else:
        raise ValueError(f"Unknown goal setting mode: {mode}")


def run_backtest_simulation(
    backtest_raw_df: pd.DataFrame,
    vae_encoder: tf.keras.Model,
    feature_scaler, # scikit-learn scaler instance
    cql_actor: tf.keras.Model,
    original_feature_columns: list # Columns VAE was trained on
    ):
    logger.info("Starting backtest simulation...")

    # 1. Calculate technical features for the backtest period
    # Ensure 'volume' column exists if not already, and other OHLCV are present
    # calculate_technical_features expects specific column names, usually lowercase ohlcv
    # backtest_raw_df already has lowercase columns from loading.

    # Add ATR directly to backtest_raw_df for dynamic goal setting before full feature calculation
    # if GOAL_SETTING_MODE == "dynamic_atr": # pandas-ta calculates ATR based on high, low, close
    backtest_raw_df.ta.atr(length=ATR_PERIOD_FOR_GOAL, append=True) # Appends "ATR_14" (or similar) column

    # Now calculate all features needed for VAE
    # The `calculate_technical_features` function drops NaNs, so the resulting df might be shorter.
    # It also returns a df with many feature columns.
    logger.info("Calculating technical features for backtest period...")
    # We need to ensure the columns expected by the scaler are present.
    # `calculate_technical_features` should produce these.
    # The input to `calculate_technical_features` should be OHLCV.
    # The current `backtest_raw_df` contains ohlcv + ATR. We need to pass only ohlcv.
    ohlcv_cols_for_features = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in backtest_raw_df.columns for col in ohlcv_cols_for_features):
        logger.error(f"Backtest data missing one of required OHLCV columns: {ohlcv_cols_for_features}")
        return None, None

    backtest_features_df = calculate_technical_features(backtest_raw_df[ohlcv_cols_for_features].copy())

    # Align backtest_features_df with backtest_raw_df (which has ATR and prices)
    # This is important because calculate_technical_features drops initial NaNs.
    common_index = backtest_raw_df.index.intersection(backtest_features_df.index)
    if common_index.empty:
        logger.error("No common index between raw data (with ATR) and calculated features. Feature calculation might have failed or removed all data.")
        return None, None

    backtest_raw_df_aligned = backtest_raw_df.loc[common_index]
    # backtest_features_df is already aligned as it's derived from common_index effectively

    logger.info(f"Technical features calculated. Shape: {backtest_features_df.shape}")
    if backtest_features_df.empty:
        logger.error("Feature calculation for backtest period resulted in empty DataFrame.")
        return None, None

    # Ensure feature_scaler was fit on the same columns as present in backtest_features_df
    # The scaler expects features in a specific order.
    # `feature_scaler.feature_names_in_` (if sklearn scaler) can tell us this.
    # For now, assume `backtest_features_df` has the right columns in the right order.
    # This is error-prone if `calculate_technical_features` changes its output columns.
    # A robust way is to save `original_feature_columns` when scaler is fit.
    # Let's assume `feature_scaler.get_feature_names_out()` or similar gives the order.
    # For now, we will rely on the order from `calculate_technical_features` being consistent.
    try:
        # Use `feature_scaler.transform` which expects a NumPy array.
        # The columns of `backtest_features_df` must match what the scaler was trained on.
        # If `feature_scaler` is from scikit-learn and was fit on a DataFrame, it might remember column names.
        # If not, the order of columns in `backtest_features_df.values` must be correct.
        # Let's assume `feature_scaler` was fit on the output of `calculate_technical_features` directly.
        scaled_backtest_features = feature_scaler.transform(backtest_features_df.values)
    except Exception as e:
        logger.error(f"Error scaling backtest features: {e}. Ensure columns match scaler's training features.")
        logger.error(f"Scaler expected features (example from sklearn): {getattr(feature_scaler, 'feature_names_in_', 'N/A')}")
        logger.error(f"Features provided columns: {backtest_features_df.columns.tolist()}")
        return None, None

    z_mean, _, z_sampled = vae_encoder.predict(scaled_backtest_features)
    # Use z_mean for deterministic state representation during backtest
    backtest_state_z_df = pd.DataFrame(z_mean, index=backtest_features_df.index,
                                       columns=[f"z_{i}" for i in range(z_mean.shape[1])])
    logger.info(f"State vectors 'z' for backtest period generated. Shape: {backtest_state_z_df.shape}")

    # --- Backtesting Loop ---
    capital = INITIAL_CAPITAL
    equity_curve = []
    trades_log = []

    in_position = False
    current_goal = None # (profit_target_perc, stop_loss_perc)
    entry_price = 0.0
    position_contracts = 0

    # Iterate through combined data (prices, Z-states, and ATR if needed)
    # The loop should go over timestamps for which we have Z-states
    for timestamp in tqdm(backtest_state_z_df.index, desc="Backtesting"):
        if timestamp not in backtest_raw_df_aligned.index: # Should not happen if indices are aligned
            logger.warning(f"Timestamp {timestamp} from Z-states not in aligned raw data. Skipping.")
            continue

        current_market_data = backtest_raw_df_aligned.loc[timestamp] # Contains close, ATR etc.
        current_z_vector = backtest_state_z_df.loc[timestamp].values # Z-vector
        current_close_price = current_market_data['close']

        # Goal Management
        if not in_position:
            # If not in position, consider setting a new goal for a potential new trade
            # For simplicity, let's assume the policy decides to enter, and then we use the goal.
            # The goal needs to be known *before* action selection if it's part of policy input.
            # Let's set a "current prospective goal"
            prospective_profit_target, prospective_stop_loss = get_dynamic_goal(current_market_data, GOAL_SETTING_MODE)
            # For this backtest, let's use the profit target as the scalar goal input to policy.
            # The stop loss will be handled by backtest logic.
            current_scalar_goal_for_policy = prospective_profit_target
        # If in position, current_goal (and thus current_scalar_goal_for_policy) should persist.

        # Prepare input for CQL actor: [z_vector, scalar_goal]
        state_goal_input = np.concatenate([current_z_vector, np.array([current_scalar_goal_for_policy])]).astype(np.float32)
        state_goal_input = np.expand_dims(state_goal_input, axis=0) # Batch dimension

        action_probs = cql_actor.predict(state_goal_input)[0]
        chosen_action = np.argmax(action_probs) # Greedy action

        # Trade Execution Logic
        trade_executed_this_step = None
        if chosen_action == ACTION_BUY and not in_position:
            in_position = True
            current_goal = (prospective_profit_target, prospective_stop_loss) # Lock in the goal for this trade
            entry_price_slippage = current_close_price * (1 + SLIPPAGE_PERCENT)
            entry_price = entry_price_slippage # Store effective entry price

            # Position sizing - fixed contracts for now
            position_contracts = TRADE_SIZE_FIXED_contracts
            buy_cost = position_contracts * entry_price * TRANSACTION_COST_PERCENT
            capital -= buy_cost # Cost of transaction

            trade_executed_this_step = {
                "timestamp": timestamp, "type": "BUY", "price": entry_price,
                "contracts": position_contracts, "cost": buy_cost, "goal_profit_perc": current_goal[0], "goal_sl_perc": current_goal[1]
            }
            # logger.debug(f"{timestamp}: BUY {position_contracts} at {entry_price:.2f}. Goal: P {current_goal[0]*100:.2f}%, SL {current_goal[1]*100:.2f}%. Capital after cost: {capital:.2f}")

        elif chosen_action == ACTION_SELL_EXIT and in_position:
            exit_price_slippage = current_close_price * (1 - SLIPPAGE_PERCENT)
            proceeds = position_contracts * exit_price_slippage
            sell_cost = proceeds * TRANSACTION_COST_PERCENT
            capital += proceeds - sell_cost # Add proceeds, subtract cost

            pnl = (exit_price_slippage - entry_price) * position_contracts - (trades_log[-1]['cost'] + sell_cost) # trades_log[-1] should be the BUY

            trade_executed_this_step = {
                "timestamp": timestamp, "type": "SELL_EXIT_POLICY", "price": exit_price_slippage,
                "contracts": position_contracts, "pnl": pnl, "cost": sell_cost
            }
            # logger.info(f"{timestamp}: SELL_EXIT (Policy) {position_contracts} at {exit_price_slippage:.2f}. P&L: {pnl:.2f}. Capital: {capital:.2f}")
            in_position = False; entry_price = 0; position_contracts = 0; current_goal = None


        # Stop-Loss / Take-Profit logic (overrides policy if triggered)
        if in_position:
            # Check Take Profit
            if current_close_price >= entry_price * (1 + current_goal[0]):
                exit_price_slippage = current_close_price * (1 - SLIPPAGE_PERCENT) # Assume TP hits at current close for simplicity
                proceeds = position_contracts * exit_price_slippage
                sell_cost = proceeds * TRANSACTION_COST_PERCENT
                capital += proceeds - sell_cost
                pnl = (exit_price_slippage - entry_price) * position_contracts - (trades_log[-1]['cost'] + sell_cost)
                trade_executed_this_step = {
                    "timestamp": timestamp, "type": "TAKE_PROFIT", "price": exit_price_slippage,
                    "contracts": position_contracts, "pnl": pnl, "cost": sell_cost
                }
                # logger.info(f"{timestamp}: TAKE_PROFIT {position_contracts} at {exit_price_slippage:.2f}. P&L: {pnl:.2f}. Capital: {capital:.2f}")
                in_position = False; entry_price = 0; position_contracts = 0; current_goal = None

            # Check Stop Loss (only if not already exited by TP)
            elif current_close_price <= entry_price * (1 + current_goal[1]): # current_goal[1] is negative
                exit_price_slippage = current_close_price * (1 - SLIPPAGE_PERCENT) # Assume SL hits at current close
                proceeds = position_contracts * exit_price_slippage
                sell_cost = proceeds * TRANSACTION_COST_PERCENT
                capital += proceeds - sell_cost
                pnl = (exit_price_slippage - entry_price) * position_contracts - (trades_log[-1]['cost'] + sell_cost)
                trade_executed_this_step = {
                    "timestamp": timestamp, "type": "STOP_LOSS", "price": exit_price_slippage,
                    "contracts": position_contracts, "pnl": pnl, "cost": sell_cost
                }
                # logger.info(f"{timestamp}: STOP_LOSS {position_contracts} at {exit_price_slippage:.2f}. P&L: {pnl:.2f}. Capital: {capital:.2f}")
                in_position = False; entry_price = 0; position_contracts = 0; current_goal = None

        if trade_executed_this_step:
            trades_log.append(trade_executed_this_step)

        # Update equity curve
        current_portfolio_value = capital
        if in_position:
            current_portfolio_value += position_contracts * current_close_price # Mark-to-market value of open position
        equity_curve.append({"timestamp": timestamp, "equity": current_portfolio_value})

    logger.info(f"Backtest simulation finished. Final capital: {capital:.2f}")
    return pd.DataFrame(trades_log), pd.DataFrame(equity_curve)


def calculate_performance_metrics(equity_curve_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float):
    if equity_curve_df.empty:
        logger.warning("Equity curve is empty. Cannot calculate performance metrics.")
        return {}

    # Final P&L
    total_pnl = equity_curve_df['equity'].iloc[-1] - initial_capital
    total_return_percent = (total_pnl / initial_capital) * 100

    # Sharpe Ratio (simplified, assuming daily returns for now, needs risk-free rate)
    # For minute data, need to aggregate to daily or use appropriate risk-free rate.
    # For simplicity, let's calculate annualized Sharpe based on overall return and std dev of equity.
    equity_returns = equity_curve_df['equity'].pct_change().dropna()
    if len(equity_returns) > 1:
        # Assuming 252 trading days, (num_minutes_in_day * 252) minutes in a year for 1-min data
        # Number of minutes in the backtest period:
        total_minutes = (equity_curve_df['timestamp'].iloc[-1] - equity_curve_df['timestamp'].iloc[0]).total_seconds() / 60.0
        minutes_in_year = 252 * 6.5 * 60 # Approx trading minutes in a year
        annualization_factor = minutes_in_year / total_minutes if total_minutes > 0 else 0

        mean_return_annualized = equity_returns.mean() * annualization_factor
        std_dev_annualized = equity_returns.std() * np.sqrt(annualization_factor)
        sharpe_ratio = mean_return_annualized / std_dev_annualized if std_dev_annualized > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    # Max Drawdown
    equity_curve_df['peak'] = equity_curve_df['equity'].cummax()
    equity_curve_df['drawdown'] = equity_curve_df['equity'] - equity_curve_df['peak']
    equity_curve_df['drawdown_percent'] = (equity_curve_df['drawdown'] / equity_curve_df['peak']) * 100
    max_drawdown_percent = equity_curve_df['drawdown_percent'].min() # Most negative drawdown

    # Trades analysis
    num_trades = len(trades_df[trades_df['type'].str.contains("SELL|TAKE_PROFIT|STOP_LOSS")])
    winning_trades = trades_df[trades_df['pnl'] > 0]
    num_winning_trades = len(winning_trades)
    win_rate = (num_winning_trades / num_trades) * 100 if num_trades > 0 else 0
    avg_win_pnl = winning_trades['pnl'].mean() if num_winning_trades > 0 else 0
    avg_loss_pnl = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0

    # Goal Achievement (simple version)
    # How many trades achieved their profit target vs hit stop loss or policy exit?
    tp_exits = len(trades_df[trades_df['type'] == "TAKE_PROFIT"])
    sl_exits = len(trades_df[trades_df['type'] == "STOP_LOSS"])
    policy_exits = len(trades_df[trades_df['type'] == "SELL_EXIT_POLICY"])

    metrics = {
        "Total P&L": total_pnl,
        "Total Return (%)": total_return_percent,
        "Sharpe Ratio (Annualized, Approx)": sharpe_ratio,
        "Max Drawdown (%)": max_drawdown_percent,
        "Number of Trades": num_trades,
        "Win Rate (%)": win_rate,
        "Average Winning P&L": avg_win_pnl,
        "Average Losing P&L": avg_loss_pnl,
        "Take Profit Exits": tp_exits,
        "Stop Loss Exits": sl_exits,
        "Policy Exits": policy_exits
    }
    logger.info("Performance Metrics:")
    for k, v in metrics.items(): logger.info(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    return metrics

def main():
    logger.info("Starting CQL Backtest Simulation Process...")

    backtest_raw_df, vae_enc, feat_scaler, cql_act, orig_cols = load_models_and_data_for_backtest()

    if backtest_raw_df is None:
        logger.error("Failed to load data/models for backtest. Aborting.")
        return

    trades_df, equity_df = run_backtest_simulation(backtest_raw_df, vae_enc, feat_scaler, cql_act, orig_cols)

    if trades_df is not None and equity_df is not None:
        logger.info(f"Backtest generated {len(trades_df)} trade records and {len(equity_df)} equity points.")

        # Save results
        os.makedirs(DATA_DIR, exist_ok=True)
        trades_df.to_csv(BACKTEST_RESULTS_FILE, index=False)
        equity_df.to_csv(EQUITY_CURVE_FILE, index=False)
        logger.info(f"Trade results saved to {BACKTEST_RESULTS_FILE}")
        logger.info(f"Equity curve saved to {EQUITY_CURVE_FILE}")

        # Calculate and display performance metrics
        performance_metrics = calculate_performance_metrics(equity_df, trades_df, INITIAL_CAPITAL)
        # Could save metrics to a file too
    else:
        logger.error("Backtest simulation failed to produce results.")

    logger.info("CQL Backtest Simulation Process Finished.")

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPU found, using CPU.")

    main()
