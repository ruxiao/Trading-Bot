import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split # For splitting data if needed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
RL_MODEL_DIR = "rl_models"
AUGMENTED_TRAJECTORIES_FILE = os.path.join(DATA_DIR, "her_augmented_trajectories.parquet")

CQL_ACTOR_FILE = os.path.join(RL_MODEL_DIR, "cql_actor_goal_conditioned.h5")
CQL_CRITIC_FILE = os.path.join(RL_MODEL_DIR, "cql_critic_goal_conditioned.h5") # Assuming critic1 was saved

# OPE Configuration
TEST_SET_SIZE = 0.2 # Use 20% of the augmented data as a pseudo-test set for OPE

# --- Helper function to unpack state-goal tuples from DataFrame column ---
def unpack_s_g_column_for_ope(df_column):
    # df_column contains tuples of (list_of_z_values, goal_value)
    z_vectors = np.array([item[0] for item in df_column])
    goals = np.array([item[1] for item in df_column]).reshape(-1, 1) # GOAL_DIM is 1
    # Concatenate z and g to form the input for the models
    s_g_vectors = np.concatenate([z_vectors, goals], axis=1).astype(np.float32)
    return s_g_vectors

def load_data_and_models_for_ope():
    logger.info("Loading data and models for OPE...")
    if not all(os.path.exists(f) for f in [AUGMENTED_TRAJECTORIES_FILE, CQL_ACTOR_FILE, CQL_CRITIC_FILE]):
        logger.error("One or more required files are missing for OPE.")
        missing = [f for f in [AUGMENTED_TRAJECTORIES_FILE, CQL_ACTOR_FILE, CQL_CRITIC_FILE] if not os.path.exists(f)]
        logger.error(f"Missing files: {missing}")
        return None, None, None

    try:
        augmented_df = pd.read_parquet(AUGMENTED_TRAJECTORIES_FILE)
        logger.info(f"Loaded {len(augmented_df)} trajectory points.")

        # Split into train/test for OPE. Ideally, OPE uses a dataset not seen during training AT ALL.
        # Here, we simulate it by splitting the offline dataset used for training.
        # We are interested in the initial states of episodes.
        # For simplicity, let's just take a random sample of all (s,g) pairs as the "test set" for this OPE.

        if len(augmented_df) < 2: # Need at least 2 samples to split
            logger.error("Not enough data to create a test split for OPE.")
            return None, None, None

        _, test_df = train_test_split(augmented_df, test_size=TEST_SET_SIZE, random_state=42, shuffle=True)
        logger.info(f"Using {len(test_df)} samples for OPE test set.")

        if test_df.empty:
            logger.error("Test DataFrame is empty after split.")
            return None, None, None

        # Unpack s_g_t for the test set
        s_g_test = unpack_s_g_column_for_ope(test_df['s_g_t'])

        # Load trained models
        actor = tf.keras.models.load_model(CQL_ACTOR_FILE)
        critic = tf.keras.models.load_model(CQL_CRITIC_FILE) # Assuming critic1 was saved

        logger.info("Data and models loaded successfully for OPE.")
        return s_g_test, actor, critic

    except Exception as e:
        logger.error(f"Error during data/model loading for OPE: {e}", exc_info=True)
        return None, None, None

def run_ope(s_g_test_data, actor_model, critic_model):
    """
    Performs Offline Policy Evaluation using the FQE-like approach.
    Calculates E[Q(s, g, pi(a|s,g))] over the test set initial states.
    """
    logger.info("Running Offline Policy Evaluation...")
    if s_g_test_data is None or actor_model is None or critic_model is None:
        logger.error("Missing data or models for OPE.")
        return None

    if len(s_g_test_data) == 0:
        logger.warning("OPE test data is empty. Cannot perform evaluation.")
        return 0.0


    # 1. Get actions from the policy (actor) for each state-goal pair in the test set
    # actor_model outputs probabilities for each action
    action_probabilities = actor_model.predict(s_g_test_data)

    # For OPE, we often want the greedy action from the policy
    # chosen_actions = np.argmax(action_probabilities, axis=1) # Greedy actions

    # 2. Get Q-values from the critic for these state-goal pairs and chosen actions
    # critic_model outputs Q-values for all actions given (s,g)
    q_values_all_actions = critic_model.predict(s_g_test_data) # Shape: (num_samples, num_actions)

    # We need the Q-value of the action the policy *would take*.
    # This is E_{a ~ pi(s,g)} [Q(s,g,a)] = sum_a (pi(a|s,g) * Q(s,g,a))
    estimated_q_for_policy_actions = np.sum(action_probabilities * q_values_all_actions, axis=1)

    # 3. Calculate the average of these Q-values
    ope_score = np.mean(estimated_q_for_policy_actions)

    logger.info(f"Offline Policy Evaluation Score (Average Expected Q-value): {ope_score:.4f}")
    return ope_score

def main():
    logger.info("Starting Offline Policy Evaluation Process...")

    s_g_test_data, actor, critic = load_data_and_models_for_ope()

    if s_g_test_data is not None:
        ope_score = run_ope(s_g_test_data, actor, critic)
        if ope_score is not None:
            logger.info(f"Final OPE Score: {ope_score:.4f}")
        else:
            logger.error("OPE calculation failed.")
    else:
        logger.error("Could not proceed with OPE due to data/model loading issues.")

    logger.info("Offline Policy Evaluation Process Finished.")

if __name__ == "__main__":
    # GPU setup (less critical for inference but good practice)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU for OPE (if TF models run on it): {gpus}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPU found, using CPU for OPE.")

    main()
```

Now for `src/backtesting/run_cql_backtest.py`. This is more complex.
I'll need:
*   The VAE-related components: `Sampling` class, VAE encoder, feature scaler.
*   The feature calculation logic (or a way to call it).
*   The CQL actor model.
*   Logic for dynamic goal setting (e.g., based on ATR).
*   A backtesting loop.
*   Performance calculation.

For the "unseen data", I'll modify the script to load `qqq_1min_1month.csv` and then split it: use the first part for generating features for training (as done by `build_features.py`) and the latter part for this backtest. This requires careful indexing. The `build_features.py` script would ideally also take start/end dates to formalize this. For now, I'll assume `run_cql_backtest.py` will handle loading the full raw data and selecting the "unseen" portion for its run. It will then compute features on-the-fly for this portion.
