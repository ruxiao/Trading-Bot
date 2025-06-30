import argparse
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
import joblib
from tqdm import tqdm

# Attempt to import local modules - adjust paths as necessary
# Assuming VAE and feature engineering are in 'src' and replay buffer is top-level
try:
    from src.feature_engineering.build_features import calculate_technical_features
    # from src.representation_learning.train_vae import Sampling # If VAE model uses custom Sampling layer
    from replay_buffer import ReplayBuffer # Assuming a ReplayBuffer class exists
    # from src.experience_generation.augment_with_her import ... # If HER is used here
except ImportError as e:
    logging.warning(f"Could not import all local modules: {e}. Ensure PYTHONPATH is set or files are accessible.")

# Define Sampling layer here if VAE model needs it and it's not easily importable
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Keras Models ---
def build_actor(state_dim, goal_dim, action_dim, actor_fc_units=(256, 256)):
    """Builds a goal-conditioned actor network."""
    state_input = layers.Input(shape=(state_dim,), name='actor_state_input')
    goal_input = layers.Input(shape=(goal_dim,), name='actor_goal_input')

    merged_input = layers.concatenate([state_input, goal_input])

    x = merged_input
    for units in actor_fc_units:
        x = layers.Dense(units, activation='relu')(x)

    action_output = layers.Dense(action_dim, activation='softmax')(x) # Assuming discrete actions, softmax for probabilities

    model = models.Model(inputs=[state_input, goal_input], outputs=action_output, name='GoalConditionedActor')
    logger.info("Actor Model Summary:")
    model.summary(print_fn=logger.info)
    return model

def build_critic(state_dim, goal_dim, action_dim, critic_fc_units=(256, 256)):
    """Builds a goal-conditioned critic network (Q-function)."""
    state_input = layers.Input(shape=(state_dim,), name='critic_state_input')
    goal_input = layers.Input(shape=(goal_dim,), name='critic_goal_input')
    action_input = layers.Input(shape=(action_dim,), name='critic_action_input') # For continuous actions
    # For discrete actions, action_input might be an integer index, or Q-values for all actions are output.
    # Here, let's assume the critic outputs Q-values for all actions if action_input is not used for discrete.
    # Or, if action_input is provided as one-hot encoded for discrete actions:

    merged_input = layers.concatenate([state_input, goal_input, action_input]) # If action is an input

    x = merged_input
    for units in critic_fc_units:
        x = layers.Dense(units, activation='relu')(x)

    q_value_output = layers.Dense(1)(x) # Outputs a single Q-value for the (s,g,a) tuple

    model = models.Model(inputs=[state_input, goal_input, action_input], outputs=q_value_output, name='GoalConditionedCritic')
    logger.info("Critic Model Summary:")
    model.summary(print_fn=logger.info)
    return model

# --- CQL Specific Components (Placeholders) ---
class CQLTrainer:
    def __init__(self, actor, critic1, critic2, target_actor, target_critic1, target_critic2,
                 actor_optimizer, critic_optimizer, alpha_optimizer,
                 state_dim, goal_dim, action_dim,
                 gamma=0.99, tau=0.005,
                 cql_alpha_initial=1.0, cql_target_action_gap=10.0, auto_tune_cql_alpha=True,
                 num_random_actions_cql=10,
                 log_std_min=-20, log_std_max=2): # For continuous actions if actor outputs log_std

        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_actor = target_actor
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer # For CQL alpha if auto-tuned

        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim

        # CQL parameters
        self.auto_tune_cql_alpha = auto_tune_cql_alpha
        if auto_tune_cql_alpha:
            self.log_cql_alpha = tf.Variable(tf.math.log(cql_alpha_initial), trainable=True)
            self.cql_alpha = tf.exp(self.log_cql_alpha)
            self.cql_target_action_gap = cql_target_action_gap
        else:
            self.cql_alpha = tf.constant(cql_alpha_initial, dtype=tf.float32)

        self.num_random_actions_cql = num_random_actions_cql
        # For continuous actions, actor might output mean and log_std.
        # For discrete, actor outputs logits/probabilities. CQL details differ slightly.
        # This skeleton assumes discrete actions based on softmax in actor.

        logger.info("CQLTrainer initialized.")
        logger.info(f"  Gamma: {self.gamma}, Tau: {self.tau}")
        logger.info(f"  CQL Alpha: {'Auto-tune from ' + str(cql_alpha_initial) if auto_tune_cql_alpha else str(cql_alpha_initial)}")
        if auto_tune_cql_alpha: logger.info(f"  CQL Target Action Gap: {self.cql_target_action_gap}")
        logger.info(f"  Num random actions for CQL: {self.num_random_actions_cql}")


    def _cql_critic_loss(self, states, goals, actions, next_states, rewards, dones,
                         q1_pred_current_actions, q2_pred_current_actions):
        """
        Calculates the CQL conservative loss for one critic.
        Args:
            states, goals, actions, next_states, rewards, dones: From the batch.
            q_pred_current_actions: Q-values for actions from the dataset, from the critic being trained.
        Returns:
            cql_term_loss: The conservative penalty term.
        """
        # This is a highly simplified placeholder. Actual implementation is more involved.
        # 1. Get Q-values for actions from the dataset (q_pred_current_actions - already provided)

        # 2. Sample random actions and get their Q-values from the current critic
        #    For discrete actions, this means sampling action indices.
        random_actions = tf.random.uniform(shape=(tf.shape(states)[0], self.num_random_actions_cql, self.action_dim))
        # Assuming actions are one-hot. If not, adjust random_actions and tile inputs.
        # This part needs careful handling of tensor shapes for batch operations.

        # Tile states and goals for random actions
        tiled_states_random = tf.tile(tf.expand_dims(states, 1), [1, self.num_random_actions_cql, 1])
        tiled_goals_random = tf.tile(tf.expand_dims(goals, 1), [1, self.num_random_actions_cql, 1])

        # Critic expecting [state, goal, action]. Reshape needed.
        # q_random_actions = self.critic1([
        #     tf.reshape(tiled_states_random, [-1, self.state_dim]),
        #     tf.reshape(tiled_goals_random, [-1, self.goal_dim]),
        #     tf.reshape(random_actions, [-1, self.action_dim]) # Assuming action is one-hot
        # ])
        # q_random_actions = tf.reshape(q_random_actions, [tf.shape(states)[0], self.num_random_actions_cql, 1])

        # Placeholder for Q-values of random actions
        q_random_actions_placeholder = tf.random.normal(shape=(tf.shape(states)[0], self.num_random_actions_cql, 1))

        # 3. Sample actions from current policy and get their Q-values
        # policy_actions, _ = self.actor([states, goals]) # Or actor.sample() if stochastic
        # q_policy_actions = self.critic1([states, goals, policy_actions]) # Or actor.sample() if stochastic

        # Placeholder for Q-values of policy actions
        q_policy_actions_placeholder = tf.random.normal(shape=(tf.shape(states)[0], self.action_dim, 1))


        # Log-sum-exp over Q-values (example for random actions, needs policy actions too)
        # For discrete actions, this would be over Q(s,g,a') for all a' or sampled a'.
        # concatenated_q_values = tf.concat([q_random_actions_placeholder, q_policy_actions_placeholder], axis=1) # And Q(s, a_dataset)
        # logsumexp_term = tf.reduce_logsumexp(concatenated_q_values, axis=1)

        logsumexp_term_placeholder = tf.reduce_mean(q_random_actions_placeholder, axis=1) # Highly simplified

        # CQL penalty: E_{s,g ~ D} [log_sum_exp Q(s,g,a') - E_{a ~ D} Q(s,g,a)]
        # cql_penalty = tf.reduce_mean(logsumexp_term - q_pred_current_actions)
        cql_penalty_placeholder = tf.reduce_mean(logsumexp_term_placeholder - q_pred_current_actions)

        return cql_penalty_placeholder


    @tf.function
    def train_step(self, states, goals, actions, next_states, rewards, dones):
        """Performs a single training step for CQL."""
        # Ensure actions are in the correct format (e.g., one-hot for discrete if critic expects that)
        # Assuming 'actions' from buffer are indices for discrete, convert to one-hot
        actions_one_hot = tf.one_hot(tf.squeeze(actions, axis=-1), self.action_dim, dtype=tf.float32)

        # --- Critic Update ---
        with tf.GradientTape(persistent=True) as tape:
            # Target Q-values (using target networks)
            next_actions_probs = self.target_actor([next_states, goals])
            # For discrete actions, Q_target often involves E_{a'~pi}[Q_target(s',g,a')] or max_{a'} Q_target(s',g,a')
            # Using expected Q-value under target policy for next state (SAC-style)
            next_q1_target = self.target_critic1([next_states, goals, next_actions_probs]) # This assumes critic takes probs; usually takes specific action
            next_q2_target = self.target_critic2([next_states, goals, next_actions_probs])

            # Placeholder: Simpler target using max over Q-values if critic output all Qs for (s,g)
            # Or, if using specific next actions from policy:
            # For discrete, usually take the action with max probability or sample.
            # This part requires careful formulation based on discrete vs continuous and SAC vs Q-learning style targets.
            # Let's assume for now a simplified target for placeholder:
            target_q_next_values = tf.minimum(next_q1_target, next_q2_target) # Clipped Double-Q

            q_targets = rewards + self.gamma * (1.0 - dones) * target_q_next_values

            # Current Q-values (from online critics)
            q1_current = self.critic1([states, goals, actions_one_hot])
            q2_current = self.critic2([states, goals, actions_one_hot])

            # Bellman loss (MSE)
            critic1_bellman_loss = tf.reduce_mean(tf.square(q1_current - q_targets))
            critic2_bellman_loss = tf.reduce_mean(tf.square(q2_current - q_targets))

            # --- CQL specific loss terms ---
            # This is the core of CQL and needs careful implementation.
            cql1_penalty = self._cql_critic_loss(states, goals, actions_one_hot, next_states, rewards, dones, q1_current)
            cql2_penalty = self._cql_critic_loss(states, goals, actions_one_hot, next_states, rewards, dones, q2_current)

            critic1_loss = critic1_bellman_loss + self.cql_alpha * cql1_penalty
            critic2_loss = critic2_bellman_loss + self.cql_alpha * cql2_penalty

            # --- CQL Alpha Update (if auto-tuning) ---
            if self.auto_tune_cql_alpha:
                # This is also a placeholder. The actual update depends on the gap.
                # alpha_loss = -tf.reduce_mean(self.log_cql_alpha * (cql1_penalty - self.cql_target_action_gap)) # Example
                alpha_loss = -tf.reduce_mean(self.log_cql_alpha * (tf.stop_gradient(cql1_penalty) - self.cql_target_action_gap)) # Using critic1's penalty
            else:
                alpha_loss = 0.0

        # Critic Gradients
        critic1_gradients = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_gradients = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic1_gradients, self.critic1.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic2_gradients, self.critic2.trainable_variables))

        if self.auto_tune_cql_alpha:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_cql_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_gradients, [self.log_cql_alpha]))
            self.cql_alpha = tf.exp(self.log_cql_alpha) # Update alpha value

        # --- Actor Update (delayed, typical in SAC/CQL) ---
        with tf.GradientTape() as tape:
            # Actor aims to maximize Q-value from one of the critics
            current_actions_probs = self.actor([states, goals])
            # Again, for discrete, this needs care. If actor outputs probs, Q usually needs specific action.
            # Let's assume Q is taken for the policy's current action distribution (expected Q) or sampled action.
            # Placeholder:
            actor_q_values = self.critic1([states, goals, current_actions_probs]) # Use critic1's estimate
            actor_loss = -tf.reduce_mean(actor_q_values)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        del tape # Important for persistent tape

        # --- Target Network Updates (Polyak averaging) ---
        self._update_target_networks(self.target_actor, self.actor, self.tau)
        self._update_target_networks(self.target_critic1, self.critic1, self.tau)
        self._update_target_networks(self.target_critic2, self.critic2, self.tau)

        return {
            "critic1_loss": critic1_loss, "critic2_loss": critic2_loss,
            "actor_loss": actor_loss, "cql_alpha": self.cql_alpha,
            "cql1_penalty": cql1_penalty, "cql2_penalty": cql2_penalty,
            "q1_current_mean": tf.reduce_mean(q1_current),
            "q_target_mean": tf.reduce_mean(q_targets)
        }

    def _update_target_networks(self, target_model, source_model, tau):
        for target_weights, source_weights in zip(target_model.weights, source_model.weights):
            target_weights.assign(tau * source_weights + (1.0 - tau) * target_weights)

    def save_models(self, path_prefix):
        self.actor.save(f"{path_prefix}_actor.h5")
        self.critic1.save(f"{path_prefix}_critic1.h5")
        self.critic2.save(f"{path_prefix}_critic2.h5")
        # Target models are typically not saved as they are derived from online models.
        # Save optimizers and alpha if needed for resuming training.
        logger.info(f"Models saved with prefix: {path_prefix}")

    def load_models(self, path_prefix):
        self.actor = models.load_model(f"{path_prefix}_actor.h5")
        self.critic1 = models.load_model(f"{path_prefix}_critic1.h5")
        self.critic2 = models.load_model(f"{path_prefix}_critic2.h5")
        # Re-initialize target networks from loaded online networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        logger.info(f"Models loaded from prefix: {path_prefix}")


# --- Data Handling ---
def load_and_preprocess_data(data_path, vae_encoder_path, vae_scaler_path, goal_dim, use_her=False):
    """
    Loads offline data, applies VAE encoding, and prepares it for the replay buffer.
    This is a placeholder and needs to be adapted to the actual data format and HER usage.
    """
    logger.info(f"Loading data from {data_path}, VAE from {vae_encoder_path}, Scaler from {vae_scaler_path}")

    # 1. Load raw trajectories (s, a, r, s', d, potentially goals if pre-computed)
    # Example: raw_df = pd.read_csv(data_path)
    # This needs to match the output of `generate_base_trajectories.py` and `augment_with_her.py`.
    # For now, creating dummy data:
    num_samples = 1000
    # Assume states are already somewhat processed (e.g. technical indicators)
    # These dimensions need to match VAE input and scaler.
    raw_state_dim = 50 # Example: dimension of features fed to VAE
    action_dim = 3     # Example: (hold, buy, sell)

    dummy_states = np.random.rand(num_samples, raw_state_dim).astype(np.float32)
    dummy_actions = np.random.randint(0, action_dim, size=(num_samples, 1)).astype(np.int32) # Action indices
    dummy_rewards = np.random.rand(num_samples, 1).astype(np.float32)
    dummy_next_states = np.random.rand(num_samples, raw_state_dim).astype(np.float32)
    dummy_dones = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)
    # Goals might be generated by HER or loaded.
    # For goal-conditioned RL, each transition (s,a,r,s') needs an associated goal 'g'.
    dummy_goals = np.random.rand(num_samples, goal_dim).astype(np.float32) # Example: target price or % change

    # 2. Load VAE encoder and feature scaler
    try:
        vae_encoder = models.load_model(vae_encoder_path, custom_objects={'Sampling': Sampling})
        feature_scaler = joblib.load(vae_scaler_path)
        logger.info("VAE encoder and feature scaler loaded.")
    except Exception as e:
        logger.error(f"Error loading VAE/scaler: {e}. Using dummy encoder/scaler for structure testing.")
        # Dummy VAE encoder (identity function for placeholder)
        state_input = layers.Input(shape=(raw_state_dim,))
        z_mean_output = layers.Dense(goal_dim, name="z_mean")(state_input) # Assuming latent_dim = goal_dim for simplicity here
        z_log_var_output = layers.Dense(goal_dim, name="z_log_var")(state_input)
        z_output = Sampling()([z_mean_output, z_log_var_output])
        vae_encoder = models.Model(inputs=state_input, outputs=[z_mean_output, z_log_var_output, z_output])
        # Dummy scaler
        class DummyScaler:
            def transform(self, x): return x
        feature_scaler = DummyScaler()


    # 3. Preprocess: Scale features and encode states with VAE
    scaled_states = feature_scaler.transform(dummy_states)
    scaled_next_states = feature_scaler.transform(dummy_next_states)

    # Use z_mean from VAE as the deterministic state representation
    z_mean_states, _, _ = vae_encoder.predict(scaled_states)
    z_mean_next_states, _, _ = vae_encoder.predict(scaled_next_states)

    vae_latent_dim = z_mean_states.shape[1]
    logger.info(f"States encoded using VAE. Latent dimension: {vae_latent_dim}")

    # 4. Populate Replay Buffer
    # Assuming ReplayBuffer class takes (state, action, reward, next_state, done, goal)
    # And can be initialized with capacity and then filled.
    # For offline, the buffer is typically filled once from the dataset.

    # This is a simplified representation. Real replay buffer would store these.
    # For CQL, we usually load the entire dataset into memory or a buffer structure.
    dataset = {
        'states': z_mean_states,      # VAE encoded states
        'goals': dummy_goals,         # Goals
        'actions': dummy_actions,     # Original actions
        'rewards': dummy_rewards,
        'next_states': z_mean_next_states, # VAE encoded next_states
        'dones': dummy_dones
    }

    # If using ReplayBuffer class:
    # buffer = ReplayBuffer(capacity=num_samples, state_dim=vae_latent_dim, action_dim=action_dim, goal_dim=goal_dim)
    # for i in range(num_samples):
    #     buffer.add(z_mean_states[i], dummy_actions[i], dummy_rewards[i], z_mean_next_states[i], dummy_dones[i], dummy_goals[i])
    # return buffer, vae_latent_dim

    return dataset, vae_latent_dim # Returning dict for direct batch sampling

# --- Main Training Script ---
def main(args):
    logger.info("--- Starting CQL Agent Training ---")

    # --- 1. Load and Preprocess Data ---
    # The goal_dim should be known or inferred. For now, let's assume it's args.goal_dim
    # The action_dim also needs to be known.
    # For this placeholder, these will be determined by the dummy data.

    # Using dummy goal_dim for now as it's part of the dummy data generation
    goal_dim_from_data = 1 # Example: if goals are scalar like target % change
    if args.goal_dim: # Override if provided
        goal_dim_from_data = args.goal_dim

    dataset, vae_latent_dim = load_and_preprocess_data(
        args.data_path, args.vae_encoder_path, args.vae_scaler_path,
        goal_dim=goal_dim_from_data
    )

    # Infer action_dim from dataset. For dummy data, it's pre-set.
    # In reality: action_dim = dataset['actions'].shape[1] or np.max(dataset['actions']) + 1 for discrete indices
    action_dim = 3 # From dummy data generation

    logger.info(f"Dataset loaded. VAE Latent Dim: {vae_latent_dim}, Action Dim: {action_dim}, Goal Dim: {goal_dim_from_data}")

    # --- 2. Initialize Networks and Optimizer ---
    actor = build_actor(vae_latent_dim, goal_dim_from_data, action_dim, args.actor_fc_units)
    critic1 = build_critic(vae_latent_dim, goal_dim_from_data, action_dim, args.critic_fc_units) # action_dim for one-hot encoded action
    critic2 = build_critic(vae_latent_dim, goal_dim_from_data, action_dim, args.critic_fc_units)

    target_actor = build_actor(vae_latent_dim, goal_dim_from_data, action_dim, args.actor_fc_units)
    target_critic1 = build_critic(vae_latent_dim, goal_dim_from_data, action_dim, args.critic_fc_units)
    target_critic2 = build_critic(vae_latent_dim, goal_dim_from_data, action_dim, args.critic_fc_units)

    # Initialize target networks with online network weights
    target_actor.set_weights(actor.get_weights())
    target_critic1.set_weights(critic1.get_weights())
    target_critic2.set_weights(critic2.get_weights())

    actor_optimizer = optimizers.Adam(learning_rate=args.actor_lr)
    critic_optimizer = optimizers.Adam(learning_rate=args.critic_lr) # Single optimizer for both critics
    alpha_optimizer = optimizers.Adam(learning_rate=args.alpha_lr) if args.auto_tune_cql_alpha else None

    cql_trainer = CQLTrainer(
        actor, critic1, critic2, target_actor, target_critic1, target_critic2,
        actor_optimizer, critic_optimizer, alpha_optimizer,
        state_dim=vae_latent_dim, goal_dim=goal_dim_from_data, action_dim=action_dim,
        gamma=args.gamma, tau=args.tau,
        cql_alpha_initial=args.cql_alpha_initial,
        cql_target_action_gap=args.cql_target_action_gap,
        auto_tune_cql_alpha=args.auto_tune_cql_alpha,
        num_random_actions_cql=args.num_random_actions_cql
    )

    # --- 3. Training Loop ---
    logger.info(f"Starting training for {args.epochs} epochs.")
    total_samples = dataset['states'].shape[0]
    num_batches = total_samples // args.batch_size

    for epoch in range(args.epochs):
        epoch_metrics = {}
        # Shuffle data each epoch (typical for offline learning)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        shuffled_dataset = {key: val[indices] for key, val in dataset.items()}

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{args.epochs}"):
            start = batch_idx * args.batch_size
            end = start + args.batch_size

            batch_states = tf.convert_to_tensor(shuffled_dataset['states'][start:end], dtype=tf.float32)
            batch_goals = tf.convert_to_tensor(shuffled_dataset['goals'][start:end], dtype=tf.float32)
            batch_actions = tf.convert_to_tensor(shuffled_dataset['actions'][start:end], dtype=tf.int32) # Action indices
            batch_rewards = tf.convert_to_tensor(shuffled_dataset['rewards'][start:end], dtype=tf.float32)
            batch_next_states = tf.convert_to_tensor(shuffled_dataset['next_states'][start:end], dtype=tf.float32)
            batch_dones = tf.convert_to_tensor(shuffled_dataset['dones'][start:end], dtype=tf.float32)

            metrics = cql_trainer.train_step(
                batch_states, batch_goals, batch_actions,
                batch_next_states, batch_rewards, batch_dones
            )

            if batch_idx == 0 and epoch == 0: # Log first batch details once
                logger.debug(f"Sample batch shapes: S:{batch_states.shape}, G:{batch_goals.shape}, A:{batch_actions.shape}, R:{batch_rewards.shape}, S':{batch_next_states.shape}, D:{batch_dones.shape}")


            for key, value in metrics.items():
                if key not in epoch_metrics: epoch_metrics[key] = []
                epoch_metrics[key].append(value.numpy())

        # Log epoch metrics (mean over batches)
        log_str = f"Epoch {epoch+1} completed. "
        for key, values in epoch_metrics.items():
            log_str += f"{key}: {np.mean(values):.4f} | "
        logger.info(log_str)

        # Save models periodically
        if (epoch + 1) % args.save_freq == 0:
            os.makedirs(args.model_save_dir, exist_ok=True)
            save_path_prefix = os.path.join(args.model_save_dir, f"cql_epoch_{epoch+1}")
            cql_trainer.save_models(save_path_prefix)

    # Save final models
    final_save_path_prefix = os.path.join(args.model_save_dir, "cql_final")
    cql_trainer.save_models(final_save_path_prefix)
    logger.info(f"--- Training finished. Final models saved to {final_save_path_prefix} ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Goal-Conditioned CQL Agent.")

    # Data and Model Paths
    parser.add_argument('--data_path', type=str, default='data/processed_trajectories_her.csv', help='Path to offline dataset.')
    parser.add_argument('--vae_encoder_path', type=str, default='rl_models/vae_encoder.h5', help='Path to pre-trained VAE encoder model.')
    parser.add_argument('--vae_scaler_path', type=str, default='rl_models/vae_feature_scaler.joblib', help='Path to VAE feature scaler.')
    parser.add_argument('--model_save_dir', type=str, default='rl_models/cql_trained', help='Directory to save trained CQL models.')

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training.')
    parser.add_argument('--actor_lr', type=float, default=3e-5, help='Actor learning rate.') # CQL often uses smaller LRs
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic learning rate.')
    parser.add_argument('--alpha_lr', type=float, default=3e-4, help='Learning rate for CQL alpha (if auto-tuning).')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--tau', type=float, default=0.005, help='Polyak averaging factor for target networks.')

    # Network Architecture
    parser.add_argument('--actor_fc_units', type=int, nargs='+', default=[256, 256], help='Hidden units for actor network.')
    parser.add_argument('--critic_fc_units', type=int, nargs='+', default=[256, 256], help='Hidden units for critic network.')
    parser.add_argument('--goal_dim', type=int, default=1, help='Dimension of the goal vector. Will be inferred if possible, this is an override.')

    # CQL Specific Parameters
    parser.add_argument('--cql_alpha_initial', type=float, default=5.0, help='Initial value for CQL alpha or fixed value if not auto-tuning.')
    parser.add_argument('--auto_tune_cql_alpha', action='store_true', help='Enable automatic tuning of CQL alpha.')
    parser.add_argument('--cql_target_action_gap', type=float, default=5.0, help='Target OOD action Q-value gap for CQL alpha auto-tuning.')
    parser.add_argument('--num_random_actions_cql', type=int, default=10, help='Number of random actions to sample for CQL loss per state.')

    parser.add_argument('--save_freq', type=int, default=10, help='Frequency (in epochs) to save models.')

    args = parser.parse_args()

    # Ensure model save directory exists
    os.makedirs(args.model_save_dir, exist_ok=True)

    # Set random seeds for reproducibility (optional)
    # tf.random.set_seed(42)
    # np.random.seed(42)

    main(args)
