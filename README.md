# Reinforcement Learning Experiments Framework (PPO & GRPO)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/) [![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/) [![Gymnasium 0.28+](https://img.shields.io/badge/Gymnasium-0.28+-purple)](https://gymnasium.farama.org/)

*Deliverable for the **ATCI** course (Reinforcement Learning) of the Master in Artificial Intelligence (MAI) program. This repository provides a modular implementation of **Proximal Policy Optimization (PPO)** and a variant **Group Relative Policy Optimization (GRPO)** for continuous control tasks, emphasizing clarity, reproducibility, and extensibility in an academic setting.*

## Table of Contents

- [Reinforcement Learning Experiments Framework (PPO \& GRPO)](#reinforcement-learning-experiments-framework-ppo--grpo)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Preparing Experiment Configurations](#preparing-experiment-configurations)
    - [Running Training Experiments](#running-training-experiments)
    - [Monitoring and Analyzing Results](#monitoring-and-analyzing-results)
  - [Algorithmic Background](#algorithmic-background)
    - [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
    - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
  - [Action Distributions: Normal vs. Beta](#action-distributions-normal-vs-beta)
  - [Reproducibility](#reproducibility)
  - [Author and Citation](#author-and-citation)

## Features

* **Modular Agent Classes:**

  * **PPO Agent:** A classic actor-critic agent implementing Proximal Policy Optimization. Supports either **Normal (Gaussian)** or **Beta** distributions for continuous action outputs, configurable via a `distribution_type` setting.
  * **GRPO Agent (No-Critic variant):** An implementation of **Group Relative Policy Optimization** that *omits the value critic*. It also supports Normal/Beta action distributions. Both agent types share a common `BaseAgent` interface for consistency.

* **Flexible Experiment Configuration:**
  Define each experiment in a self-contained JSON file (in the `configs/` directory). Config files are validated with **Pydantic** models (`utils/pydantic_models.py`) to ensure type safety. This approach cleanly separates hyperparameters and settings from code.

* **Concurrent Experiment Runs:**
  The framework can execute multiple experiments in parallel threads (use the `--workers` argument). This is useful for running parameter sweeps or comparing variants concurrently on multi-core systems.

* **Neural Network Architectures:**
  Choose between a **Feed-Forward MLP** (`networks/mlp.py`) for low-dimensional inputs or a **CNN feature extractor** (`networks/cnn.py`) for image-based observations, followed by fully-connected layers. The network architecture is specified in the JSON config (`network_config`), allowing easy swapping of model types per experiment.

* **Centralized Rollout Buffer:**
  Collected experiences are stored in a **RolloutBuffer** (`algorithms/buffer.py`) which handles trajectory storage and **Generalized Advantage Estimation (GAE)** computation. This buffer centralizes the logic for computing advantages and returns from trajectories. (The GRPO agent uses a simplified buffering strategy since it does not use a learned critic for advantages.)

* **Utility Modules:**
  A rich set of utilities in the `utils/` package covers environment creation (`env_utils.py`), seeding and reproducibility (`reproducibility_utils.py`), logging (`logging_utils.py`), tensor operations (`torch_utils.py`), distribution handling (`distribution_utils.py` for Normal/Beta/Categorical distributions), timing (`timing_utils.py`), checkpointing (`checkpoint_utils.py`), and video rendering of episodes (`video_plot_utils.py`). These helpers ensure experiments are easier to manage and analyze.

* **Logging, Checkpointing, and Evaluation:**
  The framework provides detailed logging to both console and file, and automatically saves model checkpoints during training. Training can be resumed from checkpoints seamlessly. Periodic evaluations of the current policy are performed (frequency configurable in the config), with the highest-performing episode recorded as a video for later review. Key metrics (rewards, losses, etc.) are saved to JSON for post-hoc analysis.

* **Plotting and Analysis:**
  The main execution script includes functionality to plot learning curves across experiment runs for quick insights. Additional analysis or plotting scripts can be added (e.g., in `scripts/`) to generate comparative plots or tables (some plotting outputs, e.g. in `analysis_plots/` and `latex_tables/`, are included as examples of analyzing training outcomes).

## Project Structure

```plaintext
.
├── algorithms/           # Core reinforcement learning algorithms
│   ├── base_agent.py     # Base class for agents (common logic)
│   ├── ppo.py            # PPO agent implementation (inherits BaseAgent)
│   ├── grpo.py           # GRPO agent implementation (no critic variant)
│   └── buffer.py         # RolloutBuffer for experience storage and GAE
├── configs/              # Example experiment configuration files (.json)
├── networks/             # Neural network modules for function approximation
│   ├── mlp.py            # Feed-forward neural network (actor-critic)
│   └── cnn.py            # CNN feature extractor (for image observations)
├── utils/                # Utility modules (logging, env setup, seeding, etc.)
│   ├── pydantic_models.py    # Definitions of config schemas for validation
│   ├── env_utils.py          # Functions to create and wrap environments
│   ├── distribution_utils.py # Utilities for action distributions (Normal/Beta)
│   ├── logging_utils.py      # Configurable logging setup 
│   ├── checkpoint_utils.py   # Saving and loading model checkpoints
│   ├── reproducibility_utils.py # Seeding and deterministic settings
│   ├── timing_utils.py       # Timing context managers for performance
│   └── video_plot_utils.py   # Functions for video rendering and plotting
├── experiment_runs/      # Default output directory for experiment results
│   └── <experiment_name>/    # Subdirectory for each run (named by timestamp or config)
│       ├── checkpoints/      # Saved model checkpoints
│       ├── logs/             # Log files of training and evaluation
│       ├── videos/           # Recorded videos of episodes (evaluations)
│       ├── metrics.json      # Recorded training metrics (rewards, losses, etc.)
│       └── timings.jsonl     # Timing logs for performance analysis
├── scripts/              # Auxiliary scripts (e.g., for config generation or custom plots)
│   └── generate_configs.py   # Script to programmatically create config JSONs (template)
├── run_experiment.py     # **Main script** to launch one or multiple experiments
├── single_experiment.py  # Alternate entry-point for running a single config (if needed)
├── hyperparameter_search.py # Script for hyperparameter search (multiple configs or HPO logic)
├── requirements.txt      # Python dependencies for this project
└── README.md             # Project documentation (this file)
```

The repository emphasizes a clear separation of concerns: the `algorithms` module defines the learning algorithms (PPO, GRPO) and how they interact with the rollout buffer; the `networks` module defines the function approximators (neural networks for policy and value functions); and `utils` provide supporting functionalities. Experiment outputs (checkpoints, logs, etc.) are organized under `experiment_runs/` by run name for easy navigation.

## Installation

1. **Clone the repository** to your local machine:

   ```bash
   git clone https://github.com/alobo01/mai-atci-ppo.git
   cd mai-atci-ppo
   ```

2. **Create a virtual environment** (recommended) using Python 3.9+ and activate it. For example:

   ```bash
   python3 -m venv venv 
   source venv/bin/activate   # On Linux/Mac
   # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies** using pip:

   ```bash
   pip install -r requirements.txt
   ```

   This will install core libraries such as *PyTorch*, *Gymnasium*, *NumPy*, *Pydantic*, etc., as specified in `requirements.txt`.

4. *(Optional)* **Install environment-specific dependencies:** If you plan to run environments that require extra packages (for example, MuJoCo physics or Box2D), make sure to install the corresponding Gymnasium extras. For instance:

   * For MuJoCo-based continuous control tasks (e.g. `HalfCheetah-v4`):

     ```bash
     pip install gymnasium[mujoco] mujoco
     ```
   * For classic control tasks with Box2D (e.g. `LunarLander-v2`):

     ```bash
     pip install gymnasium[box2d]
     ```

   Gymnasium (the modern fork of OpenAI Gym) may not automatically include these components, so include them as needed.

5. **Additional requirements:** Ensure you have a suitable GPU setup if training on `cuda` device (NVIDIA CUDA drivers for PyTorch, etc.). The code can run on CPU-only, but GPU is recommended for heavier tasks.

## Usage

### Preparing Experiment Configurations

Before running training, define your experiment settings in a JSON configuration file. Several example configs are provided in the `configs/` directory (and additional templates may be found in `test_configs/` or the repository root). Each JSON file specifies one experiment (environment, agent type, hyperparameters, etc.). Key fields include:

* **`env_id`:** The environment identifier (string) from Gymnasium, e.g. `"CartPole-v1"`, `"Pendulum-v1"`, `"HalfCheetah-v4"`, etc.
* **`algo`:** Which algorithm/agent to use, either `"ppo"` or `"grpo"` (for the no-critic GRPO variant).
* **`seed`:** Random seed for reproducibility (integer).
* **`total_steps`:** Total number of timesteps to train for (integer).
* **`network_config`:** A nested object defining the neural network architecture. For example:

  * `network_type`: `"mlp"` or `"cnn"` (to use a multilayer perceptron vs. convolutional network).
  * For MLP: `mlp_hidden_dims`: list of hidden layer sizes (e.g., `[64, 64]`).
  * For CNN: `cnn_output_features`: size of the output feature vector after CNN layers (for image observations), plus possibly MLP head dimensions.
* **`ppo_config` or `grpo_config`:** An object with algorithm-specific hyperparameters. Use `ppo_config` if `algo` is `"ppo"`, otherwise `grpo_config` for `"grpo"`. These include:

  * `distribution_type`: `"normal"` or `"beta"` (choice of action distribution for continuous actions).
  * Learning rate(s) for policy (and value, if applicable), e.g. `lr`.
  * Optimization parameters: `rollout_steps` (timesteps per training batch), `ppo_epochs` (number of epochs to update on each batch), `num_minibatches` (for splitting each batch), `clip_eps` (PPO clipping epsilon), etc.
  * Advantage estimation and loss coefficients: `gamma` (discount factor), `gae_lambda` (GAE parameter, if used), `entropy_coef` (coefficient for entropy regularization), `value_coef` (coefficient for value function loss).
  * For GRPO: parameters like `group_size` (number of trajectories in a group for advantage calc) may appear in `grpo_config`.

**Example:** A configuration for training a PPO agent on Pendulum-v1 with a Beta distribution might look like:

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
    "lr": 3e-4,
    "rollout_steps": 2048,
    "num_minibatches": 32,
    "ppo_epochs": 10,
    "clip_eps": 0.2,
    "entropy_coef": 0.005,
    "value_coef": 0.5
  }
}
```

This defines a two-hidden-layer MLP policy for the Pendulum environment and configures PPO with typical hyperparameters (2048-step rollout, 10 epochs of PPO updates per rollout, 0.2 clip ratio, etc.) using a Beta distribution policy. You can modify these configs or create new ones for different tasks. **Ensure the JSON conforms to the expected schema** in `utils/pydantic_models.py` (the script will validate and error out on invalid fields/types).

*(You may use the provided `scripts/generate_configs.py` as a starting point to programmatically generate sets of config files, although it may require adaptation to the schema.)*

### Running Training Experiments

Once you have your configuration file(s) ready, use the `run_experiment.py` script to launch training. You can run a single experiment or batch multiple experiments together:

* **Single Experiment:**
  Provide a specific config file with the `--config` option. For example:

  ```bash
  python run_experiment.py --config configs/ppo_pendulum_beta.json --device cpu
  ```

  This would run the PPO experiment defined in `ppo_pendulum_beta.json` on CPU. (Use `--device cuda` to run on GPU if available.)

* **Multiple Experiments:**
  Provide a directory containing multiple config files using `--config-dir`. The script will queue up all configs in that folder to run (in parallel if workers > 1). For example, to run all configs in the `configs/` directory on GPU with 4 parallel workers:

  ```bash
  python run_experiment.py --config-dir configs/ --workers 4 --device cuda
  ```

  Here, `--workers 4` means up to 4 experiments will train concurrently (each in a separate thread). Logs for each run will be separated (see **Monitoring Results** below).

**Common CLI Arguments:**
`--config` / `--config-dir` (choose one of these to specify input), `--device` (`"cpu"` or `"cuda"`), `--workers` (for parallelism, default is 1), `--base-log-dir` (optional path to use instead of the default `experiment_runs/` for output storage), and `--deterministic` (if set, enables deterministic mode in PyTorch for potentially reproducible results, see [Reproducibility](#reproducibility)). Run `python run_experiment.py --help` to see all available options.

The script will automatically create a unique subdirectory in `experiment_runs/` for each experiment run, named after the config file (or a timestamp if not specified), and will save all outputs there.

**Hyperparameter Search:** For convenience, a `hyperparameter_search.py` script is included to facilitate running a series of experiments with different hyperparameters (e.g., grid search or random search). This script can be customized to loop over variations of config settings and launch multiple runs (potentially leveraging `run_experiment.py` under the hood or using the same internal classes). Usage of this script will depend on your specific search setup – examine and edit it to suit your needs (for example, you might define a list of config objects or modify certain values in a base config). Ensure that you do not spawn too many parallel processes to overwhelm your CPU/GPU.

### Monitoring and Analyzing Results

During training, the framework provides continuous feedback and saves data for post-analysis:

* **Console and Log File:** Each experiment outputs training progress to the console and simultaneously to a log file (`experiment_runs/<run_name>/logs/log.txt`). This includes episodic reward summaries, losses (policy loss, value loss, etc.), and other diagnostics at regular intervals (e.g., after each epoch or rollout, depending on configuration).

* **Metrics JSON:** Key metrics (such as average episodic return, policy loss, value loss, etc. per update) are saved to a `metrics.json` file in the run directory. This file can be loaded later for analysis or plotting.

* **Learning Curve Plotting:** The `run_experiment.py` script will attempt to generate a plot of the learning curve for each run (e.g., reward over time) and save it under `experiment_runs/_plots_learning_curves/` (or in each run folder). By default, it might plot the reward progression of each experiment. You can customize plotting logic or use the data in `metrics.json` to create your own charts (for instance, using Jupyter notebooks or a separate analysis script). The repository includes some sample analysis outputs in `analysis_plots/` and LaTeX tables in `latex_tables/` for illustration.

* **Evaluation and Videos:** Periodically (as configured, e.g., every N training iterations), the agent's policy is evaluated without training. During these evaluation episodes, if a new highest reward is achieved, the episode is recorded as a video (`.mp4`) in the `videos/` subfolder of that run. This is especially useful for visually inspecting how the agent behaves in the environment. For example, you might find `best_episode.mp4` showing the agent solving the task when it reaches peak performance.

* **Resuming Training:** If training is interrupted or if you want to continue improving a model, you can resume from the latest checkpoint. The training script automatically saves checkpoints (network weights and optimizer state) in the `checkpoints/` folder of the run. To resume, simply use the same output folder and config, and set the `resume` flag in the config or command (if implemented) so that the script will load the latest checkpoint instead of starting from scratch. (Under the hood, `checkpoint_utils.py` handles saving and loading of models.)

After training, you can analyze results by comparing the metrics of different runs. If multiple experiments were run in parallel, you might use the saved metrics to produce comparative plots (e.g., comparing PPO vs GRPO, or different hyperparameter settings). The output data is structured to facilitate such analysis.

## Algorithmic Background

This section provides a conceptual overview of the algorithms implemented, with an emphasis on their core components and how they are realized in this codebase.

### Proximal Policy Optimization (PPO)

**Proximal Policy Optimization (PPO)** is a state-of-the-art policy gradient method in reinforcement learning. It alternates between **sampling trajectories** from the environment using the current policy (the *actor* network) and **optimizing a surrogate objective** on those samples using stochastic gradient ascent. Unlike traditional policy gradient methods that perform a single update per batch of data, PPO enables **multiple epochs** of updates on the same batch by using a clipped surrogate objective that preserves training stability. This approach yields many of the benefits of earlier trust-region methods (like TRPO) while being simpler to implement and empirically robust.

In an actor-critic setup, PPO maintains two neural networks (which may share layers or be separate in implementation):

* **Policy Network (Actor):** This network \$\pi\_\theta(a|s)\$ outputs a probability distribution over actions for each state (for continuous actions, it parameterizes a probability distribution such as a Normal or Beta – see [Action Distributions](#action-distributions-normal-vs-beta)). The policy is updated by maximizing a **surrogate objective** function. The key term in PPO's loss is the probability ratio \$r\_t(\theta) = \frac{\pi\_\theta(a\_t|s\_t)}{\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)}\$, which measures how the new policy deviates from the old policy for the taken action. PPO clips this ratio to lie within a range $\[1-\epsilon,;1+\epsilon]\$ (where \$\epsilon\$ is a small hyperparameter like 0.2) when computing the policy loss. This **clipping mechanism** ensures the policy does not update too far in a single step, preventing excessively large policy changes that could destabilize training. Concretely, the PPO objective \$\mathcal{L}\_{\text{policy}}\$ is:
  $\mathcal{L}_{\text{policy}} = - \mathbb{E}_t\Big[ \min\big( r_t(\theta)\,\hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat{A}_t \big) \Big]$
  where \$\hat{A}\_t\$ is the advantage estimate at time \$t\$. By taking the minimum of the unclipped and clipped objectives, PPO penalizes changes that would improve the objective beyond the allowed range, thus encouraging conservative updates.

* **Value Function (Critic):** The critic network \$V\_\phi(s\_t)\$ estimates the **state value** (expected return from state \$s\_t\$). It is used to compute advantage estimates and is trained by minimizing a regression loss to fit the actual returns (or discounted returns) from the collected trajectories. In this implementation, the value function can either be a separate network or a head on a shared network with the policy. The loss for the critic is typically:
  $\mathcal{L}_{\text{value}} = \mathbb{E}_t \Big[ \frac{1}{2}\big(V_\phi(s_t) - G_t\big)^2 \Big]$
  i.e., half mean-squared-error between predicted value and the observed return \$G\_t\$. (Some PPO implementations also clip the value function updates to improve stability, though the original PPO paper did not.) The coefficient for this loss (e.g., `value_coef` in config) determines its relative importance.

* **Advantage Estimation:** PPO uses **advantage estimates** \$\hat{A}*t = G\_t - V*\phi(s\_t)\$ to weight policy updates, where \$G\_t\$ is the estimated return (cumulative reward) from time \$t\$. This code uses **Generalized Advantage Estimation (GAE)** for computing \$\hat{A}*t\$. GAE is a technique that blends multiple-step advantage estimates with an exponentially decaying weighting (parameterized by \$\lambda\$) to reduce variance while introducing minimal bias. Essentially, instead of using the raw \$n\$-step returns or TD(1) returns, GAE computes:
  $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$
  where \$\delta*{t} = r\_t + \gamma V\_\phi(s\_{t+1}) - V\_\phi(s\_t)\$ is the TD-error. The factor \$\lambda \in \[0,1]\$ controls the bias-variance trade-off (with \$\lambda=1\$ reducing to the full Monte Carlo returns, high variance; and \$\lambda=0\$ to one-step TD, more bias). GAE has been shown to substantially **reduce the variance** of policy gradient estimates by using the critic’s value to create a **baseline**, at the cost of a small bias. The RolloutBuffer in this code computes advantages in this manner once a rollout is complete.

* **Entropy Regularization:** To encourage exploration, PPO often adds an entropy bonus to the objective. An agent with higher policy entropy (i.e., more uncertainty in action selection) is less likely to get stuck in suboptimal deterministic policies early. The term \$\mathcal{L}\_{\text{entropy}} = -\mathbb{E}*t\[ \mathcal{H}\[\pi*\theta(\cdot|s\_t)] ]\$ (negative of policy entropy, so that minimizing this term maximizes entropy) may be added to the total loss with a coefficient (e.g., `entropy_coef` in the config). In this implementation, you can tune `entropy_coef` to balance exploration vs. exploitation.

**Optimization:** PPO typically uses a stochastic gradient optimizer like Adam. After each rollout of `rollout_steps` per environment, the algorithm computes advantages and then performs K epochs of minibatch updates (splitting the rollout into `num_minibatches`). This yields stable improvements on the policy . Empirically, PPO strikes a good balance between sample efficiency, stability, and ease of implementation, and is a go-to method for many continuous control problems.

### Group Relative Policy Optimization (GRPO)

**Group Relative Policy Optimization (GRPO)** is a more recent variant of PPO that was proposed to address some limitations of standard PPO, especially in contexts like large language model training with human feedback. The key innovation of GRPO is that it **eliminates the need for a learned value function critic**, instead using a group-based baseline for advantage estimation. In other words, GRPO forgoes the critic network entirely and computes advantages by comparing *groups of trajectories or rollouts*:

* **Group-Based Advantage:** In GRPO, multiple trajectories (or action sequences) are generated for the same initial state or prompt, forming a *group*. The reward outcomes of that group are compared: the average reward of the group serves as a baseline, and each trajectory’s advantage is its reward minus the group average. This technique yields a relative advantage for each trajectory within the group (hence "Group Relative"). By using the group’s mean reward as baseline, GRPO sidesteps training a separate value function to predict an absolute baseline. The intuition is that within a group of experiences, some perform better than others, and those should be given positive advantage, while those below the group mean get negative advantage.

* **No Critic Network:** Because the baseline is computed from actual sampled rewards in a group, GRPO does not require a critic network at all. This can significantly reduce complexity and potential sources of error (no value function approximation error). It also reduces the number of parameters and computations, focusing all learning on the policy itself. In this repository’s implementation (`GRPO_NoCritic` class), the training loop is adjusted accordingly: the agent still collects rollouts, but advantage computation uses grouping (with a configurable `group_size`) and no value loss is computed. Essentially, it operates more like a pure policy gradient update but using **relative** performance within groups as the signal.

* **KL Regularization:** Some formulations of GRPO also integrate a Kullback-Leibler (KL) divergence term into the loss (between the new policy and reference policy) for additional stability, similar to how PPO’s clipping is a implicit form of KL control. In the context of LLM training, this KL term keeps the policy from drifting too far from an initial model (useful for alignment). In classical RL tasks, a KL penalty could similarly be added to keep updates conservative. If implemented, this would appear as an extra term in the loss function (or one could monitor the KL divergence during training). The current codebase focuses on the no-critic aspect; if needed, one can add a KL penalty analogously to how PPO’s clipping is done.

**When to use GRPO:** GRPO is particularly useful when a well-shaped reward model is available or when *relative* performance matters more than absolute performance. By not training a critic, it avoids potential bias or poor generalization from a value function. However, it may require larger batch sizes (groups of trajectories) to get a good baseline estimate from samples. In the included implementation, GRPO is provided as an experimental alternative to PPO. For example, in **language model fine-tuning with human preference rewards**, GRPO has been shown to improve alignment by using groups of sampled responses. In control tasks, it can be seen as a variant of REINFORCE with baseline, where the baseline is the mean of a sample group rather than a learned V(s).

In summary, **GRPO\_NoCritic** in this repo demonstrates how one might implement a PPO-like training loop without a critic. It uses the same infrastructure (rollout buffer, etc.) but simplifies advantage calculation. Users are encouraged to experiment with `group_size` and see how it affects training. For certain tasks, PPO’s learned critic might be more sample-efficient, but GRPO offers an interesting perspective and can work better in scenarios where learning a value function is hard or introducing its approximation error is undesirable.

## Action Distributions: Normal vs. Beta

One distinctive feature of this codebase is the support for different action distribution types for continuous action spaces. Continuous control tasks (like MuJoCo locomotion or robotics environments) typically require the agent to output real-valued actions within a certain range \[*a*<sub>low</sub>, *a*<sub>high</sub>]. Two approaches are implemented here:

* **Gaussian (Normal) Distribution:** The policy outputs a mean (μ) and a standard deviation (σ) for each action dimension, defining a Normal distribution \$\mathcal{N}(\mu, \sigma)\$ from which to sample actions. In this implementation, no explicit squashing (like tanh) is applied to the Gaussian output; instead, an action sampled from \$\mathcal{N}(\mu, \sigma)\$ is *clipped* to the valid range of the environment’s action space before being executed. This is a common approach: the neural network can learn to mean-center within the range and keep σ small to mostly produce in-range actions, while occasional out-of-bounds samples are just clipped. One should be cautious with clipping, as it can bias the gradient if a lot of probability mass lies outside bounds, but in practice moderate σ and the penalty of out-of-range actions usually keep this in check. The Normal distribution is the default choice in many RL implementations due to its simplicity and unbounded support.

* **Beta Distribution:** The policy outputs two positive parameters α and β for each action dimension, defining a Beta distribution \$\text{Beta}(\alpha, \beta)\$. The Beta is supported on \[0,1], which aligns well with bounded action spaces. In our implementation, the raw network outputs for α and β are passed through a softplus (to ensure positivity) and then +1 (since Beta parameters > 1 yield a distribution more centered away from 0/1 extremes, and +1 avoids singularities at exactly 0). The sampled value \$x \sim \text{Beta}(\alpha,\beta)\$ is naturally between 0 and 1; we then **scale and shift** this sample to the environment’s action range \[*a*<sub>low</sub>, *a*<sub>high</sub>]. Because the Beta distribution inherently respects bounds, it can represent probabilities that concentrate near the edges without any clipping or transformation needed (other than linear scaling). Research has shown that using a Beta policy for bounded actions can improve training stability and performance, as it avoids the potential issues of Gaussian tails going out-of-bounds. In fact, empirical studies on continuous control tasks (e.g., car racing, lunar lander) found that PPO with a Beta policy converges faster and to a better policy than PPO with Gaussian, making Beta distributions a strong choice for bounded-action problems.

* **Categorical Distribution (for Discrete actions):** Although most focus is on continuous actions, the code also supports discrete action spaces using a Categorical distribution. In such cases, the policy network outputs logits for each discrete action, and the highest-likelihood action can be sampled or taken. (The distribution utilities handle this automatically if the environment’s action space is discrete.) This is standard for tasks like CartPole or Atari. Note that when using discrete actions, the Beta/Normal choice in the config is irrelevant; the code will detect the action space type and use a categorical distribution.

Under the hood, PyTorch’s `torch.distributions` module is used along with `Independent` wrappers for multi-dimensional actions. For example, if an environment has *n* continuous action dimensions, the code constructs *n* independent Beta (or Normal) distributions (one per dimension) and treats them as a single joint distribution. The log-probabilities across dimensions simply sum, which is mathematically equivalent to a multivariate diagonal Gaussian or a factored Beta distribution. This independence assumption is common and keeps the policy tractable, though it means the policy cannot natively represent correlations between action dimensions. (If action dimensions are highly coupled in an environment, one might consider a multivariate Gaussian with a full covariance or other techniques, but that is beyond our scope.)

**Practical considerations:** If you use the Beta distribution (`"distribution_type": "beta"` in config), pay attention to the entropy coefficient and learning rate. Beta distributions can have different learning dynamics (for instance, if α and β become very large, the distribution concentrates strongly; an overly high entropy bonus might interfere by keeping it too diffuse). The provided example uses a small entropy coefficient (0.005) for Beta on Pendulum, acknowledging that **tuning may be needed**. On the other hand, Gaussian policies typically use an entropy bonus to avoid premature convergence to deterministic actions. Monitor your training runs: if the policy entropy plummets quickly for Beta, you might increase `entropy_coef` or vice versa for Gaussian. The code logs entropy so you can see this behavior.

In summary, the framework allows you to easily switch between Normal and Beta distributions for continuous actions by changing one config field. This modular design lets researchers and practitioners compare performance and learning characteristics under different policy parameterizations.

## Reproducibility

Reinforcement learning results can be notoriously hard to reproduce due to their sensitivity to initialization, environment stochasticity, and asynchronous interactions. This project makes efforts to improve reproducibility:

* **Seeding:** Every experiment can be run with a specific integer seed (`"seed"` in the config). This seed is used to initialize the RNGs for NumPy, Python’s `random`, and PyTorch (and even the environment, through Gymnasium) via utilities in `utils/reproducibility_utils.py`. Using the same seed for an experiment should give identical results **on the same hardware and software environment** (barring nondeterministic operations).

* **Deterministic Mode:** If you pass the `--deterministic` flag to the run script, the code will enable PyTorch’s deterministic backend options (e.g., setting `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`). This forces certain operations to use deterministic algorithms at some potential performance cost. Note that even with this, perfect reproducibility is **not guaranteed** especially on GPU, because some low-level operations and environment dynamics might introduce nondeterminism. However, it should reduce sources of variance.

* **Environment Stochasticity:** Some environments have inherent randomness (e.g., random starting states). Even with the same seed, you might see variation if the environment’s own random seed isn’t controlled. The framework attempts to seed the environment (Gymnasium environments typically use `env.reset(seed=...)` when created via our `env_utils.py`). Users should ensure that the environment version and any external dependencies (physics engines, etc.) are consistent across runs.

* **Logging and Checkpointing:** By logging every training run and saving models, you can analyze after the fact why two runs diverged. If a run did exceptionally well, you have its checkpoint to examine or continue training from. The config JSON saved in each run folder (the exact copy) ensures you know the exact parameters used. This practice of **record-keeping** is crucial for experimental reproducibility.

Despite these measures, remember that RL has high variance. It’s good practice to run multiple seeds for each experiment and report aggregate results. The provided parallel execution functionality (`--workers`) makes it easier to launch, say, 5 runs with different seeds at once. Reproducibility is about controlling what you can, and documenting everything else.

## Author and Citation

**Author:** Antonio Lobo
This project was developed as part of the coursework for the ATCI (Advanced Topics in Computational Intelligence) Reinforcement Learning course of the Master in Artificial Intelligence program. It is an open-source implementation intended for educational and research use.

If you use or build upon this code, or if you found the README documentation helpful in your own work, **please acknowledge the author.** Appropriate citation helps support open-source contributions. You may cite this repository as:

```
@misc{Lobo2025PPO,
  author = {Lobo-Santos, Antonio},
  title = {RL Experiments Framework: Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO)},
  year = {2025},
  howpublished = {\url{https://github.com/alobo01/mai-atci-ppo}},
  note = {ATCI Course Project, M.Sc. Artificial Intelligence UPC-UB-URV}
}
```

Please include the URL and author name in academic references. For any questions or to report issues, feel free to contact the author or open a GitHub issue.

**Copyright Notice:** The contents of this repository (code and documentation) are © 2025 Antonio Lobo. Permission is granted for academic and research usage with attribution. For commercial use or integration, please contact the author.
