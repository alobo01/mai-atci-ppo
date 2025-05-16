# Refactored RL Experiments Framework

This repository provides a refactored framework for running Reinforcement Learning experiments using PPO and GRPO agents. It supports both Normal and Beta distributions for continuous action spaces in PPO, and Normal/Beta for GRPO. The framework emphasizes modularity, type safety with Pydantic, and concurrent experiment execution.

**Author:** Antonio Lobo (Refactored Version)

## Features

*   **Modular Agents:**
    *   `PPO`: Supports Normal or Beta distributions for continuous actions, configurable via `distribution_type`.
    *   `GRPO_NoCritic`: GRPO variant without a critic, also supports Normal or Beta distributions.
    *   Inheritance from a common `BaseAgent`.
*   **Flexible Configuration:** Experiments defined using JSON configuration files, validated by Pydantic models (`utils/pydantic_models.py`).
*   **Concurrent Execution:** Run multiple experiments in parallel using threading (`--workers` argument).
*   **Network Architectures:**
    *   `FeedForwardNN` (MLP) in `networks/mlp.py`.
    *   `CNNFeatureExtractor` in `networks/cnn.py` for image-based inputs, followed by MLP heads.
    *   Configurable via `network_config` in the experiment JSON.
*   **Rollout Buffer:** Centralized `RolloutBuffer` (`algorithms/buffer.py`) for PPO data collection and GAE computation. GRPO uses a simplified path through the buffer.
*   **Utilities (Refactored into `utils/` submodules):**
    *   Logging, seeding, environment creation.
    *   Tensor handling, action distribution creation (Normal, Beta, Categorical without Tanh squashing for Normal).
    *   Performance timing, checkpointing, video recording, metrics saving/loading.
*   **Logging & Checkpointing:** Robust logging to console and file, automatic checkpoint saving and resuming.
*   **Evaluation & Video:** Periodic evaluation with video recording of the best-performing episode.
*   **Plotting & Analysis Scripts:** (Assumed to be adapted or provided separately by the user, `run_experiment.py` includes a basic plotting function).

## Project Structure

```
.
├── algorithms/             # Agent implementations and buffer
│   ├── __init__.py
│   ├── base_agent.py
│   ├── buffer.py
│   ├── ppo.py
│   └── grpo.py
├── configs/                # Experiment configuration files (.json)
├── networks/               # Neural network modules
│   ├── __init__.py
│   ├── mlp.py
│   └── cnn.py
├── scripts/                # Helper scripts (e.g., config generation, plotting)
│   └── generate_configs.py # (Example, needs to be updated for Pydantic models)
│   └── (plotting scripts)
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── checkpoint_utils.py
│   ├── distribution_utils.py
│   ├── env_utils.py
│   ├── logging_utils.py
│   ├── pydantic_models.py
│   ├── reproducibility_utils.py
│   ├── timing_utils.py
│   ├── torch_utils.py
│   └── video_plot_utils.py
├── experiment_runs/        # Default output directory for all runs
│   └── <run_name>/         # Subdirectory for each experiment run
│       ├── checkpoints/
│       ├── logs/
│       ├── videos/
│       ├── metrics.json
│       └── timings.jsonl
│   └── _plots_learning_curves/ # Generated learning curve plots
├── run_experiment.py       # Main script to launch experiments
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Installation

1.  **Clone the repository.**
2.  **Create a virtual environment (recommended).**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   Ensure you have the necessary extras for `gymnasium` depending on the environments you plan to use (e.g., `gymnasium[mujoco]`, `gymnasium[box2d]`).

## Workflow

### 1. Generate Configuration Files

Use or adapt `scripts/generate_configs.py` to create JSON configuration files. Ensure they conform to the Pydantic models in `utils/pydantic_models.py`.

**Key configuration fields in JSON:**
*   `env_id`: e.g., "CartPole-v1", "Pendulum-v1", "CarRacing-v0".
*   `algo`: "ppo" or "grpo".
*   `seed`: Integer.
*   `network_config`: Dict with `network_type` ("mlp" or "cnn"), `mlp_hidden_dims`, `cnn_output_features`.
*   `ppo_config` (if `algo` is "ppo"): Dict with PPO hyperparameters like `distribution_type` ("normal" or "beta"), `lr`, `clip_eps`, etc.
*   `grpo_config` (if `algo` is "grpo"): Dict with GRPO hyperparameters like `distribution_type`, `group_size`, `lr`, etc.
*   Other general parameters like `total_steps`, `gamma`, etc.

Example `ppo_pendulum_beta.json`:
```json
{
  "env_id": "Pendulum-v1",
  "algo": "ppo",
  "seed": 0,
  "total_steps": 200000,
  "network_config": {
    "network_type": "mlp",
    "mlp_hidden_dims": [64, 64]
  },
  "ppo_config": {
    "distribution_type": "beta",
    "lr": 0.0003,
    "rollout_steps": 2048,
    "num_minibatches": 32,
    "ppo_epochs": 10,
    "clip_eps": 0.2,
    "entropy_coef": 0.005, // Beta entropy can be tricky, adjust carefully
    "value_coef": 0.5
  }
}
```

### 2. Run Experiments

```bash
python run_experiment.py --config-dir configs/ --workers 4 --device cuda
```
*   `--config-dir`: Path to your JSON config files.
*   `--workers`: Number of parallel experiments.
*   `--device`: `cpu` or `cuda`.
*   `--base-log-dir`: Where to save `experiment_runs`.

### 3. Analyze Results

The `run_experiment.py` script includes a basic plotting function `plot_all_learning_curves`. You can expand this or use separate analysis scripts (like those previously in `scripts/`) by adapting them to read from the `metrics.json` files in each run's directory.

## Notes on Action Distributions

*   **Normal Distribution:** The actor network outputs `mean` and `log_std`. Actions are sampled from `Normal(mean, std)`. **No Tanh squashing is applied.** The model is expected to learn appropriate `mean` and `std` such that samples are generally within the environment's action bounds. Actions are **clipped** to the environment's `action_space.low` and `action_space.high` before being passed to `env.step()`.
*   **Beta Distribution:** The actor network outputs `raw_alpha` and `raw_beta`, which are transformed (e.g., `softplus(raw) + 1.0`) to get `alpha` and `beta` parameters for the `Beta(alpha, beta)` distribution. This distribution naturally samples actions in the `[0, 1]` range. These samples are then linearly scaled to the environment's `[action_space.low, action_space.high]` range.
*   **`Independent` Wrapper:** For multi-dimensional continuous action spaces, `torch.distributions.Independent` is used to wrap the base per-dimension distribution (Normal or Beta). This allows treating a batch of independent uni-dimensional distributions as a single multivariate distribution with a diagonal covariance matrix, simplifying log probability calculations by summing log-probs across action dimensions. This is a standard assumption and doesn't inherently affect reproducibility if seeding is correct. If action dimensions are strongly correlated, a `MultivariateNormal` with a learned full covariance matrix would be more expressive but complex.

## Reproducibility

The framework includes `utils/reproducibility_utils.py` to seed random number generators. For full PyTorch determinism (which can be slower), use the `--deterministic` flag. Note that complete determinism in RL can be challenging due to environment stochasticity and subtle GPU behaviors.