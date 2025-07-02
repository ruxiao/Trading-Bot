import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras import backend as K

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
AUGMENTED_TRAJECTORIES_FILE = os.path.join(DATA_DIR, "her_augmented_trajectories.parquet")
RL_MODEL_DIR = "rl_models"

# Model saving paths
CQL_ACTOR_FILE = os.path.join(RL_MODEL_DIR, "cql_actor_goal_conditioned.h5")
CQL_CRITIC_FILE = os.path.join(RL_MODEL_DIR, "cql_critic_goal_conditioned.h5")

# VAE Configuration (from train_vae.py, needed for state dimension)
LATENT_DIM = 16
# Goal dimension (e.g., price change percentage) is 1
STATE_DIM = LATENT_DIM + 1 # VAE latent state + goal
ACTION_DIM = 3 # Hold, Buy, Sell

# Hyperparameters for simplified agent
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
BATCH_SIZE = 256
TRAIN_STEPS_PER_EPOCH = 100
EPOCHS = 10
GAMMA = 0.99 # Discount factor

# Networks
def build_actor(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="softmax")(x) # Softmax for discrete actions
    model = Model(inputs, outputs, name="actor")
    return model

def build_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    # Critic predicts Q-values for all actions given a state
    x = layers.Dense(256, activation="relu")(state_input)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(action_dim)(x) # Output Q-value for each action
    model = Model(state_input, outputs, name="critic")
    return model

class SimpleRLAgent:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr):
        self.actor = build_actor(state_dim, action_dim)
        self.critic = build_critic(state_dim, action_dim)
        self.target_critic = build_critic(state_dim, action_dim)
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

        self.tau = 0.005 # Target network update rate

    def update_critic(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # Q-values for current state-action pairs
            q_values = self.critic(states)
            action_masks = tf.one_hot(tf.cast(actions, tf.int32), ACTION_DIM)
            predicted_q = tf.reduce_sum(q_values * action_masks, axis=1, keepdims=True)

            # Target Q-values
            target_q_next = tf.reduce_max(self.target_critic(next_states), axis=1, keepdims=True)
            target_q = rewards + (1 - dones) * GAMMA * target_q_next

            critic_loss = tf.reduce_mean(tf.square(predicted_q - target_q))

        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        return critic_loss

    def update_actor(self, states):
        with tf.GradientTape() as tape:
            action_probs = self.actor(states)
            # We want to maximize the Q-value of the actions chosen by the policy
            # For simplicity, we'll use the Q-values from the critic directly
            # and try to make the policy output higher probabilities for actions
            # that have higher Q-values. This is a basic policy gradient idea.
            q_values_from_critic = self.critic(states)
            # Maximize expected Q-value under the policy
            actor_loss = -tf.reduce_mean(tf.reduce_sum(action_probs * q_values_from_critic, axis=1))

        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        return actor_loss

    def update_target_networks(self):
        for target_param, param in zip(self.target_critic.trainable_variables, self.critic.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)

def train_simple_rl_agent():
    tf.config.run_functions_eagerly(True) # Enable eager execution
    logger.info("Starting simple RL agent training process...")

    if not os.path.exists(AUGMENTED_TRAJECTORIES_FILE):
        logger.error(f"Augmented trajectories file not found: {AUGMENTED_TRAJECTORIES_FILE}. Please run HER augmentation first.")
        return

    # Load augmented data
    try:
        augmented_df = pd.read_parquet(AUGMENTED_TRAJECTORIES_FILE)
        logger.info(f"Loaded {len(augmented_df)} augmented trajectory points.")
    except Exception as e:
        logger.error(f"Error loading augmented trajectories: {e}", exc_info=True)
        return

    # Prepare data for training
    # Ensure s_g_t and s_g_next_t are numpy arrays of floats
    states = np.array(augmented_df['s_g_t'].tolist(), dtype=np.float32)
    actions = np.array(augmented_df['a_t'].tolist(), dtype=np.int32)
    rewards = np.array(augmented_df['r_t'].tolist(), dtype=np.float32).reshape(-1, 1)
    next_states = np.array(augmented_df['s_g_next_t'].tolist(), dtype=np.float32)
    dones = np.array(augmented_df['done'].tolist(), dtype=np.float32).reshape(-1, 1)

    logger.info(f"States shape: {states.shape}")
    logger.info(f"Actions shape: {actions.shape}")
    logger.info(f"Rewards shape: {rewards.shape}")
    logger.info(f"Next States shape: {next_states.shape}")
    logger.info(f"Dones shape: {dones.shape}")

    agent = SimpleRLAgent(STATE_DIM, ACTION_DIM, ACTOR_LR, CRITIC_LR)

    # Training loop
    logger.info("Training simple RL agent...")
    for epoch in range(EPOCHS):
        critic_losses = []
        actor_losses = []
        for _ in range(TRAIN_STEPS_PER_EPOCH):
            # Sample a batch from the replay buffer
            indices = np.random.randint(0, len(augmented_df), size=BATCH_SIZE)
            batch_states = tf.constant(states[indices], dtype=tf.float32)
            batch_actions = tf.constant(actions[indices], dtype=tf.int32)
            batch_rewards = tf.constant(rewards[indices], dtype=tf.float32)
            batch_next_states = tf.constant(next_states[indices], dtype=tf.float32)
            batch_dones = tf.constant(dones[indices], dtype=tf.float32)

            # Update critic
            c_loss = agent.update_critic(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            critic_losses.append(c_loss.numpy())

            # Update actor
            a_loss = agent.update_actor(batch_states)
            actor_losses.append(a_loss.numpy())

            # Update target networks
            agent.update_target_networks()

        avg_critic_loss = np.mean(critic_losses)
        avg_actor_loss = np.mean(actor_losses)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Avg Critic Loss: {avg_critic_loss:.4f} | Avg Actor Loss: {avg_actor_loss:.4f}")

    logger.info("Simple RL agent training complete.")

    # Save trained models
    # Ensure model directory exists
    os.makedirs(RL_MODEL_DIR, exist_ok=True)
    agent.actor.save(CQL_ACTOR_FILE)
    agent.critic.save(CQL_CRITIC_FILE)
    logger.info(f"Trained actor saved to {CQL_ACTOR_FILE}")
    logger.info(f"Trained critic saved to {CQL_CRITIC_FILE}")

    logger.info("Simple RL agent training process finished.")

if __name__ == "__main__":
    # Ensure TensorFlow uses GPU if available, and set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Using GPU: {gpus}")
        except RuntimeError as e:
            logger.error(f"Error setting up GPU: {e}")
    else:
        logger.info("No GPU found, using CPU.")

    train_simple_rl_agent()
