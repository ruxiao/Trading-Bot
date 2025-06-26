# Goal-Conditioned Reinforcement Learning Trading Bot

This project implements a trading bot based on a goal-conditioned offline reinforcement learning (RL) agent. It uses Conservative Q-Learning (CQL) trained on data augmented with Hindsight Experience Replay (HER). The state representation is learned using a Variational Autoencoder (VAE).

## Project Workflow & Structure

The project follows these main steps:

1.  **Data Collection**: Fetches 1-minute QQQ OHLCV data using Interactive Brokers (IBKR).
    *   Script: `src/data_collection/fetch_qqq_data.py`
    *   Output: `data/qqq_1min_1month.csv`

2.  **Feature Engineering**: Calculates a rich set of technical indicators from the raw data.
    *   Script: `src/feature_engineering/build_features.py`
    *   Input: `data/qqq_1min_1month.csv`
    *   Output: `data/qqq_features.parquet`

3.  **Representation Learning (VAE)**: Trains a VAE on the engineered features to learn a compressed state vector `z`.
    *   Script: `src/representation_learning/train_vae.py`
    *   Input: `data/qqq_features.parquet`
    *   Outputs: `rl_models/vae_encoder.h5` (encoder model), `rl_models/vae_feature_scaler.joblib` (scaler for features)

4.  **Base Experience Generation**: Runs a simple MACD crossover strategy on historical data to generate initial trajectories `(s, a, r, s')`, where `s` is the VAE-encoded state `z`.
    *   Script: `src/experience_generation/generate_base_trajectories.py`
    *   Inputs: `data/qqq_features.parquet`, `data/qqq_1min_1month.csv` (for prices), VAE encoder & scaler.
    *   Output: `data/base_trajectories.parquet`

5.  **Hindsight Experience Replay (HER) Augmentation**: Augments the base trajectories by relabeling goals based on achieved outcomes, creating synthetic experiences.
    *   Script: `src/experience_generation/augment_with_her.py`
    *   Input: `data/base_trajectories.parquet`, `data/qqq_1min_1month.csv` (for prices).
    *   Output: `data/her_augmented_trajectories.parquet`. States in this dataset are `(z, g)` where `g` is the goal.

6.  **Goal-Conditioned Offline RL Agent Training (CQL)**: Trains a CQL agent. The Q-function and policy networks take `(state_z, goal_g)` as input.
    *   Script: `src/rl_agent/train_cql_agent.py`
    *   Input: `data/her_augmented_trajectories.parquet`
    *   Outputs: `rl_models/cql_actor_goal_conditioned.h5` (policy), `rl_models/cql_critic_goal_conditioned.h5` (critic).

7.  **Evaluation and Backtesting**:
    *   **Offline Policy Evaluation (OPE)**: Estimates the trained policy's performance using a held-out part of the dataset.
        *   Script: `src/backtesting/offline_policy_evaluation.py`
        *   Inputs: `data/her_augmented_trajectories.parquet`, trained CQL models.
    *   **Full Backtest Simulation**: Runs the trained agent on recent, unseen data, providing realistic goals (fixed or dynamic ATR-based) at the start of each trade.
        *   Script: `src/backtesting/run_cql_backtest.py`
        *   Inputs: `data/qqq_1min_1month.csv` (uses a later portion as "unseen"), VAE models, CQL actor model, feature engineering logic.
        *   Outputs: `data/cql_backtest_results.csv` (trade log), `data/cql_equity_curve.csv`.

## Directory Structure

```
.
├── data/                     # Stores raw data, features, trajectories, results
├── rl_models/                # Stores trained VAE, CQL models, scalers
├── src/
│   ├── data_collection/
│   │   └── fetch_qqq_data.py
│   ├── feature_engineering/
│   │   └── build_features.py
│   ├── representation_learning/
│   │   └── train_vae.py
│   ├── experience_generation/
│   │   ├── generate_base_trajectories.py
│   │   └── augment_with_her.py
│   ├── rl_agent/
│   │   └── train_cql_agent.py
│   ├── backtesting/
│   │   ├── offline_policy_evaluation.py
│   │   └── run_cql_backtest.py
│   └── utils/                  # Utility scripts (if any)
├── config.py                 # IBKR connection parameters
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup and Installation

1.  **Prerequisites**:
    *   Python 3.9+
    *   Interactive Brokers Trader Workstation (TWS) or IBKR Gateway installed and running (for data collection).
        *   Ensure API access is enabled. Note the "Socket port".
    *   Potentially a GPU for faster training of VAE and CQL models.

2.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

3.  **Create a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `tensorflow`, `pandas`, `pandas-ta`, `numpy`, `scikit-learn`, `ib_insync`, `pyarrow`, `joblib`, `tqdm`.

5.  **Configure IBKR Connection**:
    *   Edit `config.py` to set your IBKR TWS/Gateway connection parameters (`IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`).

## Running the Pipeline

Execute the scripts in the `src/` subdirectories in the order outlined in the "Project Workflow & Structure" section. Each script processes data or trains a model, producing artifacts used by subsequent scripts.

Example execution order:
```bash
python src/data_collection/fetch_qqq_data.py
python src/feature_engineering/build_features.py
python src/representation_learning/train_vae.py
python src/experience_generation/generate_base_trajectories.py
python src/experience_generation/augment_with_her.py
python src/rl_agent/train_cql_agent.py
python src/backtesting/offline_policy_evaluation.py
python src/backtesting/run_cql_backtest.py
```
Monitor the logs and output files from each script.

## Disclaimer

This software is for educational and research purposes only. Trading financial markets involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use this software at your own risk. Paper trading is highly recommended before considering any real-money trading. The authors and contributors are not responsible for any financial losses or other damages incurred from the use of this software.
