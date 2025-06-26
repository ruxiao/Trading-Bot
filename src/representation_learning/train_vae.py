import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib # For saving the scaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
FEATURES_FILE = os.path.join(DATA_DIR, "qqq_features.parquet")
MODEL_DIR = "rl_models" # For saving VAE encoder and scaler
VAE_ENCODER_FILE = os.path.join(MODEL_DIR, "vae_encoder.h5")
VAE_SCALER_FILE = os.path.join(MODEL_DIR, "vae_feature_scaler.joblib")

# VAE Configuration
LATENT_DIM = 16 # Dimensionality of the latent space vector z
RECONSTRUCTION_WEIGHT = 1.0 # Weight for reconstruction loss
KL_WEIGHT = 1.0 # Weight for KL divergence loss (can be tuned)
EPOCHS = 50 # Number of training epochs
BATCH_SIZE = 64

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim, latent_dim):
    # --- Encoder ---
    encoder_inputs = layers.Input(shape=(original_dim,), name="encoder_input")
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    logger.info("Encoder summary:")
    encoder.summary(print_fn=logger.info)

    # --- Decoder ---
    latent_inputs = layers.Input(shape=(latent_dim,), name="decoder_input")
    x = layers.Dense(64, activation="relu")(latent_inputs)
    x = layers.Dense(128, activation="relu")(x)
    decoder_outputs = layers.Dense(original_dim, activation="sigmoid", name="decoder_output")(x) # Sigmoid if data is scaled to [0,1]
    # If using StandardScaler (mean 0, std 1), linear activation might be better for output.
    # Let's assume MinMaxScaler for now due to sigmoid.

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    logger.info("Decoder summary:")
    decoder.summary(print_fn=logger.info)

    # --- VAE Model ---
    # Connect encoder and decoder
    vae_outputs = decoder(encoder(encoder_inputs)[2]) # Use the sampled 'z' from encoder
    vae = Model(encoder_inputs, vae_outputs, name="vae")

    # VAE Loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.mean_squared_error(encoder_inputs, vae_outputs), axis=-1 # Sum over features
        )
    ) * RECONSTRUCTION_WEIGHT # MSE is common, can also use binary_crossentropy if inputs are binary

    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * KL_WEIGHT

    total_loss = reconstruction_loss + kl_loss
    vae.add_loss(total_loss)

    return vae, encoder, decoder

def main():
    logger.info("Starting VAE training process...")

    if not os.path.exists(FEATURES_FILE):
        logger.error(f"Features file not found: {FEATURES_FILE}. Please run feature engineering first.")
        return

    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Load features
    try:
        features_df = pd.read_parquet(FEATURES_FILE)
        logger.info(f"Loaded features. Shape: {features_df.shape}")
        # Drop any remaining NaNs just in case, though build_features should handle this
        features_df.dropna(inplace=True)
        if features_df.empty:
            logger.error("Feature DataFrame is empty after loading and NaN drop. Cannot train VAE.")
            return
        logger.info(f"Shape after final NaN check: {features_df.shape}")
    except Exception as e:
        logger.error(f"Error loading features from {FEATURES_FILE}: {e}", exc_info=True)
        return

    # 2. Preprocess data
    # Keep original column order for reconstruction and for the scaler
    original_columns = features_df.columns.tolist()
    data_values = features_df.values

    # Scale features - MinMaxScaler is often used with VAEs if output activation is sigmoid
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler() # If using StandardScaler, decoder output activation should be linear

    scaled_data = scaler.fit_transform(data_values)
    joblib.dump(scaler, VAE_SCALER_FILE)
    logger.info(f"Feature scaler saved to {VAE_SCALER_FILE}")

    original_dim = scaled_data.shape[1]
    logger.info(f"Data scaled. Original dimension: {original_dim}")

    # Split data
    # For VAE, we are learning a representation, so a simple random split is fine.
    # If we were doing time-series forecasting, a sequential split would be needed.
    x_train, x_val = train_test_split(scaled_data, test_size=0.2, random_state=42, shuffle=True)
    logger.info(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")

    # 3. Build VAE
    vae, encoder, _ = build_vae(original_dim, LATENT_DIM) # We only need to save the encoder

    # Optimizer
    # Learning rate scheduling can be beneficial for VAEs
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000, # Adjust based on dataset size and epochs
        decay_rate=0.96,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


    vae.compile(optimizer=optimizer) # Loss is added in the model definition

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10, # Number of epochs with no improvement after which training will be stopped.
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2, # Factor by which the learning rate will be reduced. new_lr = lr * factor
        patience=5, # Number of epochs with no improvement after which learning rate will be reduced.
        min_lr=1e-6,
        verbose=1
    )

    # 4. Train VAE
    logger.info("Training VAE...")
    history = vae.fit(
        x_train,
        x_train, # VAE tries to reconstruct its input
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, x_val),
        callbacks=[early_stopping, reduce_lr_on_plateau],
        verbose=1
    )

    logger.info("VAE training complete.")

    # Plot training history (optional, good for notebooks)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('VAE Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_path = os.path.join(MODEL_DIR, "vae_training_loss.png")
        plt.savefig(plot_path)
        logger.info(f"Training loss plot saved to {plot_path}")
        plt.close()
    except ImportError:
        logger.warning("Matplotlib not found. Skipping saving training loss plot.")
    except Exception as e:
        logger.error(f"Error plotting training loss: {e}", exc_info=True)


    # 5. Save the trained encoder
    # The encoder model outputs [z_mean, z_log_var, z_sampled]
    # For creating the state vector `z`, we typically use `z_mean` as the deterministic representation,
    # or `z_sampled` if stochasticity is desired during RL agent's interaction with environment.
    # Let's save the encoder that outputs all three, and decide later which to use.
    encoder.save(VAE_ENCODER_FILE)
    logger.info(f"Trained VAE encoder saved to {VAE_ENCODER_FILE}")

    # 6. (Optional) Evaluation / Sanity Check
    logger.info("Performing a quick sanity check on the VAE encoder...")
    # Take a few samples from validation set
    sample_input = x_val[:5]
    z_mean, z_log_var, z_sampled = encoder.predict(sample_input)
    logger.info(f"Sample input shape: {sample_input.shape}")
    logger.info(f"Encoded z_mean shape: {z_mean.shape}, first z_mean: {z_mean[0][:5]}") # Log first 5 dims of first sample
    logger.info(f"Encoded z_sampled shape: {z_sampled.shape}, first z_sampled: {z_sampled[0][:5]}")

    # To get the compressed state vector `z` for the entire dataset:
    # _, _, all_z_sampled = encoder.predict(scaled_data)
    # all_z_mean, _, _ = encoder.predict(scaled_data) # If using z_mean
    # This `all_z_sampled` or `all_z_mean` would be the input to the RL agent's state.

    logger.info("VAE training and encoder saving process finished.")

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

    main()

```python
# Example of how to load and use the encoder and scaler later:
# scaler = joblib.load(VAE_SCALER_FILE)
# encoder = tf.keras.models.load_model(VAE_ENCODER_FILE, custom_objects={'Sampling': Sampling})
#
# new_raw_features_df = pd.read_parquet(...) # Load new raw features
# new_scaled_features = scaler.transform(new_raw_features_df[original_columns].values) # Use original_columns from training
#
# z_mean, z_log_var, z_sampled = encoder.predict(new_scaled_features)
# state_vectors = z_mean # Or z_sampled
```
