import logging
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
BASE_TRAJECTORIES_FILE = os.path.join(DATA_DIR, "base_trajectories.parquet")
# We need raw price data for P&L and goal calculation
RAW_PRICES_FILE = os.path.join(DATA_DIR, "qqq_1min_1month.csv")
AUGMENTED_TRAJECTORIES_FILE = os.path.join(DATA_DIR, "her_augmented_trajectories.parquet")

# HER Configuration
# For this version, the "achieved goal" will be based on the final outcome of the trade.
# Goal representation: desired price change percentage (e.g., 0.01 for +1%)
# Reward for HER: Binary: 0 if goal achieved, -1 otherwise.

def load_data():
    """Loads base trajectories and raw QQQ prices."""
    logger.info("Loading base trajectories and raw prices...")
    if not os.path.exists(BASE_TRAJECTORIES_FILE):
        logger.error(f"Base trajectories file not found: {BASE_TRAJECTORIES_FILE}")
        return None, None
    if not os.path.exists(RAW_PRICES_FILE):
        logger.error(f"Raw prices file not found: {RAW_PRICES_FILE}")
        return None, None

    try:
        base_trajectories_df = pd.read_parquet(BASE_TRAJECTORIES_FILE)
        # s_t and s_next_t are stored as lists, convert them back to numpy arrays for consistency if needed
        # For processing, lists are fine.

        raw_prices_df = pd.read_csv(RAW_PRICES_FILE, index_col='date', parse_dates=True)
        raw_prices_df.columns = [col.lower() for col in raw_prices_df.columns]

        logger.info(f"Loaded {len(base_trajectories_df)} base trajectory points.")
        logger.info(f"Loaded {len(raw_prices_df)} raw price points.")
        return base_trajectories_df, raw_prices_df
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return None, None

def identify_trade_episodes(base_trajectories_df: pd.DataFrame) -> list:
    """
    Identifies trade episodes from the base trajectories.
    An episode is defined as a sequence of transitions from a BUY action
    until a SELL_EXIT action or the end of data.
    """
    episodes = []
    current_episode = []
    in_trade = False

    for i, row in base_trajectories_df.iterrows():
        if row['a_t'] == 1: # ACTION_BUY from previous script (generate_base_trajectories.py)
            if in_trade and current_episode: # Should not happen if BUY is only when not in_position
                logger.warning(f"New BUY action encountered while already in a trade at index {i}. Ending previous episode.")
                episodes.append(current_episode)
                current_episode = []
            in_trade = True

        if in_trade:
            current_episode.append(row.to_dict()) # Store row as dict

        if row['a_t'] == 2 and in_trade: # ACTION_SELL_EXIT
            # episodes.append(current_episode) # Already added by row.done
            # current_episode = [] # Reset for next trade
            in_trade = False
            # The 'done' flag from base trajectories should also mark end of trade episode

        if row['done'] and current_episode: # If 'done' is True, the episode (trade) ends here
            # This check ensures that even if last action wasn't SELL_EXIT (e.g. data ends), episode is stored.
            episodes.append(list(current_episode)) # Store a copy
            current_episode = []
            in_trade = False # Reset in_trade status

    # If the loop finishes and there's an unterminated episode (e.g., data ends while in position)
    if current_episode:
        logger.info(f"Found an unterminated episode at the end of trajectories with {len(current_episode)} steps.")
        episodes.append(current_episode)

    logger.info(f"Identified {len(episodes)} trade episodes.")
    return episodes

def apply_her(episodes: list, raw_prices_df: pd.DataFrame, default_goal_value: float = 0.0075) -> pd.DataFrame:
    """
    Applies Hindsight Experience Replay to the identified episodes.

    Args:
        episodes (list): List of trade episodes, where each episode is a list of trajectory dicts.
        raw_prices_df (pd.DataFrame): DataFrame of raw prices, indexed by timestamp.
        default_goal_value (float): A default intended goal for original trajectories (e.g., +0.75% profit).

    Returns:
        pd.DataFrame: DataFrame of original and augmented HER trajectories.
                      Each state 's_t' and 's_next_t' will be a tuple (original_z_vector, goal_vector).
                      Goal vector could be a single float for simplicity here.
    """
    augmented_experiences = []

    for episode in tqdm(episodes, desc="Processing HER episodes"):
        if not episode:
            continue

        # Determine the entry price for this episode
        # The first transition's timestamp and the raw_prices_df will give us this.
        # However, the base trajectories already had simulated entry/exit.
        # For HER, we need the price at s_t to define the goal relative to it.

        entry_timestamp = episode[0]['timestamp']
        # Ensure entry_timestamp is in raw_prices_df (it should be, due to alignment in previous step)
        if entry_timestamp not in raw_prices_df.index:
            logger.warning(f"Entry timestamp {entry_timestamp} not found in raw prices. Skipping episode for HER.")
            continue

        entry_price_at_s0 = raw_prices_df.loc[entry_timestamp]['close']
        if pd.isna(entry_price_at_s0):
            logger.warning(f"NaN entry price at {entry_timestamp}. Skipping episode.")
            continue

        # The "achieved outcome" for this episode, using the 'future' strategy for HER.
        # We will use the actual outcome of the trade as the hindsight goal.
        # The last state of the episode gives the exit.
        exit_timestamp = episode[-1]['timestamp'] # Timestamp of the state BEFORE the exit action's consequence
        # The actual exit price is related to s_next_t of the step where action is SELL_EXIT, or close of s_next_t if done.
        # Let's use the close price at the timestamp of s_next_t of the final step in the episode.

        # Find the timestamp of the state s_next_t for the last transition in the episode
        # The 'timestamp' in the trajectory dict is for s_t. s_next_t is one step ahead.
        # We need to find the index for s_next_t in raw_prices_df.
        # Assuming 1-minute bars, s_next_t's timestamp is roughly episode[-1]['timestamp'] + 1 min.
        # This needs careful handling if timestamps are not perfectly sequential or have gaps.

        # Let's use the actual reward `r_t` from the episode's last step if it was a SELL action,
        # as it already captured the P&L.
        # The goal is a price change. The achieved price change for the whole episode:
        # If the last action was SELL_EXIT, its 'r_t' is the P&L.
        # P&L = (exit_price_eff - entry_price_eff) * contracts - total_costs
        # Achieved % change = (exit_price_actual / entry_price_actual) - 1

        # For HER, let's define the achieved goal as the price at the time of the episode's last s_next_t
        last_s_next_t_timestamp_approx = episode[-1]['timestamp'] + pd.Timedelta(minutes=1) # Approximate

        # Find the closest available timestamp in raw_prices_df for s_next_t of the last step
        final_s_next_idx = raw_prices_df.index.get_indexer([episode[-1]['timestamp']], method='ffill')[0]
        if final_s_next_idx + 1 < len(raw_prices_df.index):
            final_s_next_timestamp = raw_prices_df.index[final_s_next_idx + 1]
        else: # It was the last data point overall
             final_s_next_timestamp = episode[-1]['timestamp'] # Use the last known timestamp

        if final_s_next_timestamp not in raw_prices_df.index:
            logger.warning(f"Final s_next_timestamp {final_s_next_timestamp} for episode ending near {episode[-1]['timestamp']} not in raw prices. Skipping HER for this episode.")
            continue

        achieved_exit_price_at_s_final_next = raw_prices_df.loc[final_s_next_timestamp]['close']
        if pd.isna(achieved_exit_price_at_s_final_next):
            logger.warning(f"NaN achieved exit price at {final_s_next_timestamp}. Skipping HER for this episode.")
            continue

        # This is the hindsight goal: achieve the price change observed over the whole trade.
        hindsight_goal_value = (achieved_exit_price_at_s_final_next / entry_price_at_s0) - 1.0

        # Original transitions (with a default/intended goal)
        for t_idx, transition in enumerate(episode):
            s_t = transition['s_t']
            a_t = transition['a_t']
            original_r_t = transition['r_t']
            s_next_t = transition['s_next_t']
            done = transition['done']
            timestamp_t = transition['timestamp'] # Timestamp of s_t

            # Augment state with goal. Goal is a single float.
            # For policy input, (s, g) will be concatenated or processed.
            # Store as tuple (list_of_z_values, goal_value) for clarity in DataFrame.

            # 1. Original experience with a default intended goal
            # The reward is kept as original from the behavior policy
            augmented_experiences.append({
                's_g_t': (s_t, default_goal_value), # State s_t, Intended Goal default_goal_value
                'a_t': a_t,
                'r_t': original_r_t,
                's_g_next_t': (s_next_t, default_goal_value), # Next State s_next_t, Intended Goal default_goal_value
                'done': done,
                'is_her': False, # Mark as not a HER sample
                'timestamp': timestamp_t
            })

            # 2. HER experience: relabel with the hindsight_goal_value for this episode
            # The hindsight goal is to achieve the final outcome of this specific trade.

            # Recompute reward for HER: binary (0 if goal achieved, -1 otherwise)
            # Goal: from price_at_s_t, achieve price_at_s_t * (1 + hindsight_goal_value)
            # This needs price at s_t (price_at_current_step) and price at s_next_t (price_at_next_step)

            price_at_current_step = raw_prices_df.loc[timestamp_t]['close']

            # Find timestamp for s_next_t
            current_s_t_idx_in_raw = raw_prices_df.index.get_indexer([timestamp_t], method='ffill')[0]
            if current_s_t_idx_in_raw + 1 < len(raw_prices_df.index):
                timestamp_next_t = raw_prices_df.index[current_s_t_idx_in_raw + 1]
            else:
                timestamp_next_t = timestamp_t # Reached end of data

            if pd.isna(price_at_current_step) or timestamp_next_t not in raw_prices_df.index:
                # logger.debug(f"NaN price or next timestamp issue at {timestamp_t} for HER reward. Skipping this HER sample.")
                continue
            price_at_next_step = raw_prices_df.loc[timestamp_next_t]['close']
            if pd.isna(price_at_next_step):
                # logger.debug(f"NaN next price at {timestamp_next_t} for HER reward. Skipping this HER sample.")
                continue

            # Did s_next_t achieve the hindsight_goal_value relative to price_at_current_step?
            # Target price for hindsight_goal_value from current step:
            target_hindsight_price = price_at_current_step * (1 + hindsight_goal_value)

            # Check if price_at_next_step met this target.
            # For positive goals (profit target): achieved if price_at_next_step >= target_hindsight_price
            # For negative goals (stop-loss): achieved if price_at_next_step <= target_hindsight_price
            # Let's use a small tolerance for floating point comparisons.
            tolerance = 0.0001 * price_at_current_step # 0.01% tolerance

            her_goal_achieved_at_s_next = False
            if hindsight_goal_value >= 0: # Profit target
                if price_at_next_step >= target_hindsight_price - tolerance:
                    her_goal_achieved_at_s_next = True
            else: # Loss target (e.g. stop-loss)
                if price_at_next_step <= target_hindsight_price + tolerance: # Price fell to or below target
                    her_goal_achieved_at_s_next = True

            her_r_t = 0.0 if her_goal_achieved_at_s_next else -1.0

            # For HER, 'done' is true if this step achieves the hindsight goal.
            her_done = her_goal_achieved_at_s_next

            augmented_experiences.append({
                's_g_t': (s_t, hindsight_goal_value), # State s_t, Hindsight Goal
                'a_t': a_t,
                'r_t': her_r_t,
                's_g_next_t': (s_next_t, hindsight_goal_value), # Next State s_next_t, Hindsight Goal
                'done': her_done,
                'is_her': True, # Mark as a HER sample
                'timestamp': timestamp_t
            })

    return pd.DataFrame(augmented_experiences)


def main():
    logger.info("Starting HER augmentation process...")

    base_trajectories_df, raw_prices_df = load_data()
    if base_trajectories_df is None or raw_prices_df is None:
        logger.error("Failed to load data. Aborting HER augmentation.")
        return

    trade_episodes = identify_trade_episodes(base_trajectories_df)
    if not trade_episodes:
        logger.warning("No trade episodes identified from base trajectories. Output will be empty.")
        # Still create an empty DataFrame with correct columns if no episodes.
        # Or, handle what to do if only original trajectories with default goals are desired.
        # For now, if no episodes, implies no trades, so HER has nothing to work on.
        # We can still process non-trade trajectories if needed, but current focus is on trades.

    # If there are episodes, apply HER
    if trade_episodes:
        augmented_df = apply_her(trade_episodes, raw_prices_df)
    else: # Create an empty df with expected columns if no episodes for HER
        # Or, if we want to include non-trade original trajectories with default goals:
        # This part would iterate base_trajectories_df again for non-episode parts if desired.
        # For now, let's assume HER focuses on relabeling trades.
        # If no trades, augmented_df will be empty.
        logger.info("No trade episodes to apply HER. If there were non-trade trajectories, they are not currently being processed for HER.")
        augmented_df = pd.DataFrame(columns=['s_g_t', 'a_t', 'r_t', 's_g_next_t', 'done', 'is_her', 'timestamp'])


    if not augmented_df.empty:
        try:
            # Ensure output directory exists
            os.makedirs(DATA_DIR, exist_ok=True)
            augmented_df.to_parquet(AUGMENTED_TRAJECTORIES_FILE, index=False)
            logger.info(f"Augmented HER trajectories saved to {AUGMENTED_TRAJECTORIES_FILE}")
            logger.info(f"Augmented DataFrame head:\n{augmented_df.head()}")
            logger.info(f"Number of original experiences: {len(augmented_df[~augmented_df['is_her']])}")
            logger.info(f"Number of HER experiences: {len(augmented_df[augmented_df['is_her']])}")
        except Exception as e:
            logger.error(f"Error saving augmented trajectories: {e}", exc_info=True)
    else:
        logger.warning("No augmented trajectories generated. File not saved.")

    logger.info("HER augmentation process finished.")

if __name__ == "__main__":
    main()
