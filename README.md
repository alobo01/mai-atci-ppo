# Refactored RL Experiments Framework

This repository provides a refactored framework for running Reinforcement Learning experiments using PPO (Gaussian and Beta policies) and GRPO agents. It emphasizes best practices like strong typing, code reuse through inheritance, modular utilities, efficient performance measurement, and concurrent experiment execution.

**Author:** Antonio Lobo

## Features

*   **Modular Agents:** PPO (Gaussian), PPO (Beta), and GRPO (no critic) implementations inheriting from a common `BaseAgent`.
*   **Flexible Configuration:** Experiments are defined using JSON configuration files, easily generated via a script.
*   **Concurrent Execution:** Run multiple experiments in parallel using threading via the `--workers` argument.
*   **Network Options:** Supports MLP (`FeedForwardNN`) by default. For image-based environments specify `network_type: "cnn"` in the config to use a `CNNFeatureExtractor` followed by separate `FeedForwardNN` heads.
*   **Utilities:** Comprehensive `utils.py` module for logging, seeding, environment creation, tensor handling, action distributions, timing, checkpointing, and video saving.
*   **Performance Measurement:** Built-in timing for key operations (rollout, update, env steps, network forwards) logged to `timings.jsonl`.
*   **Logging & Checkpointing:** Robust logging to console and file, automatic checkpoint saving and resuming.
*   **Evaluation & Video:** Periodic evaluation episodes with video recording of the best performing episode.
*   **Plotting & Analysis Scripts:** Scripts provided for generating learning curve plots (mean/std/min/max over seeds) and analyzing the GRPO `group_size` hyperparameter.
*   **Typing:** Strong type hinting used throughout the codebase.

## Project Structure

```
.
├── experiment_runs/        # Default output directory for logs, models, videos, plots
│   └── <env_id>_<algo>[_g<G>]_seed<N>/ # Subdirectory for each experiment run
│       ├── checkpoints/      # Saved network/optimizer states
│       ├── logs/             # Log files (.log)
│       ├── videos/           # Evaluation videos (.mp4)
│       ├── metrics.json      # Training metrics (rewards, lengths)
│       └── timings.jsonl     # Performance timings per step
│   └── _plots_learning_curves/ # Generated learning curve plots (.png)
│   └── _plots_grpo_g_analysis/ # Generated GRPO G analysis plots (.png)
├── configs/                # Generated configuration files
│   └── ...
├── config_examples/        # Example JSON configuration files (Optional)
│   └── ...
├── scripts/                # Scripts for generation and analysis
│   ├── generate_configs.py
│   ├── plot_learning_curves.py
│   └── plot_grpo_g_analysis.py
├── base_agent.py           # Abstract Base Class for agents
├── grpo.py                 # GRPO (no critic) implementation
├── networks.py             # Network definitions (MLP, CNN)
├── ppo_beta.py             # PPO with Beta distribution
├── ppo_revised.py          # PPO with Gaussian distribution (base PPO)
├── run_experiment.py       # Main script to launch experiments
├── utils.py                # Shared utility functions
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **MuJoCo:** If you plan to use MuJoCo environments (HalfCheetah, Hopper, etc.), make sure you have MuJoCo installed correctly and install the extra dependencies:
        ```bash
        pip install gymnasium[mujoco]
        ```
    *   **CarRacing/Box2D:** For CarRacing, install the Box2D extras:
        ```bash
        pip install gymnasium[box2d]
        # You might need system dependencies like swig:
        # sudo apt-get update && sudo apt-get install swig # Debian/Ubuntu
        # brew install swig # macOS
        ```
        Accept the Atari ROM license during installation if prompted, or run `ale-import-roms`.

## Workflow

### 1. Generate Configuration Files (Optional but Recommended)

Use the provided script to generate JSON configuration files for all desired combinations of environments, algorithms, seeds, and GRPO G-values.

```bash
python scripts/generate_configs.py
```

This will create a `configs/` directory (by default) containing the `.json` files. Review the script (`scripts/generate_configs.py`) to customize environments, seeds, hyperparameters, or the output directory.

### 2. Run Experiments

Use the `run_experiment.py` script, pointing it to the directory containing your configuration files (e.g., the `configs/` directory created in the previous step).

```bash
python run_experiment.py --config-dir configs/ --workers 4 --device cuda
```

**Key Arguments:**
*   `--config-dir` (`-cd`): **Required.** Path to the directory containing your `.json` config files.
*   `--workers` (`-w`): Number of experiments to run in parallel (default: 1). Adjust based on your CPU cores and GPU memory.
*   `--device` (`-d`): Device to use (`cpu` or `cuda`). Defaults to `cuda` if available, otherwise `cpu`.
*   `--base-log-dir` (`-ld`): Base directory for saving all outputs (default: `experiment_runs/`).
*   `--deterministic`: Use deterministic PyTorch algorithms (slower, for reproducibility checks).
*   `--skip-plots`: Disable automatic plot generation *by the main runner* (you can still run analysis scripts manually).

Experiments will run, saving logs, checkpoints, videos, metrics, and timings into subdirectories within the specified `--base-log-dir`.

### 3. Analyze Results

After experiments have finished (or partially completed), use the analysis scripts:

*   **Plot General Learning Curves:**
    ```bash
    python scripts/plot_learning_curves.py experiment_runs/
    ```
    (Replace `experiment_runs/` if you used a different `--base-log-dir`). This script aggregates results across seeds for each (environment, algorithm) pair and plots the mean reward curve with standard deviation and min/max ranges. Plots are saved to `experiment_runs/_plots_learning_curves/` by default.

*   **Plot GRPO G-Parameter Analysis:**
    ```bash
    python scripts/plot_grpo_g_analysis.py experiment_runs/
    ```
    This script focuses specifically on GRPO runs. For each environment, it compares the learning curves (mean over seeds) for different `group_size` (G) values used in the experiments. Plots are saved to `experiment_runs/_plots_grpo_g_analysis/` by default.

**Analysis Script Arguments:**
*   Both plotting scripts accept the results directory as the first positional argument.
*   `--output-dir` (`-o`): Specify a different directory to save the plots.
*   `--total-steps` (`-t`): Set the maximum x-axis value (environment steps) for the plots (default: 1,000,000).

## Code Structure Overview

*   **`run_experiment.py`:** Entry point. Parses arguments, finds configs, uses `ThreadPoolExecutor` to run `run_single_experiment` for each config concurrently. Handles overall timing.
*   **`base_agent.py`:** Defines the `BaseAgent` ABC with common initialization (`__init__`), the main `train()` loop structure, evaluation/video logic, and abstract methods (`_setup_networks_and_optimizers`, `get_action`, `_rollout`, `_update`) that specific algorithms must implement.
*   **`ppo_revised.py`:** Implements `PPO` inheriting from `BaseAgent`. Handles Gaussian policy logic, GAE calculation, and the PPO clipped surrogate objective update. Supports MLP and CNN+Head architectures.
*   **`ppo_beta.py`:** Inherits from `PPO`. Overrides distribution-specific methods (`_get_distribution`, `_update`) to handle the Beta distribution and action scaling.
*   **`grpo.py`:** Implements `GRPO_NoCritic` inheriting from `BaseAgent`. Implements the group rollout (`_rollout`) and the GRPO update rule (`_update`) involving the reference policy and KL penalty. Supports MLP and CNN+Head architectures.
*   **`networks.py`:** Contains `nn.Module` definitions for `FeedForwardNN` (MLP) and `CNNFeatureExtractor`. Includes a `NETWORK_REGISTRY`.
*   **`utils.py`:** Contains standalone helper functions for common tasks used across different modules.
*   **`scripts/`:** Contains helper scripts for configuration generation and results analysis/plotting.

## TODO / Potential Improvements

*   Implement learning rate scheduling.
*   Add support for vectorized environments (e.g., `gymnasium.vector.AsyncVectorEnv`) for faster rollouts.
*   More sophisticated advantage normalization options.
*   More detailed analysis scripts (e.g., final performance tables, hyperparameter sensitivity beyond GRPO-G).
*   Refine CNN architecture and make it more configurable.
*   Consider using Hydra or another configuration management library instead of plain JSON/script generation.