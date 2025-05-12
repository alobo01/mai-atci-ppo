"""
Generates JSON configuration files for RL experiments.

Creates configs for specified environments, algorithms (PPO_Gauss, PPO_Beta, GRPO),
seeds, and GRPO group sizes (G).
"""
import json
import os
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any, Tuple

# --- Configuration ---
OUTPUT_DIR = Path("configs")
ENVS: List[str] = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Swimmer-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "CarRacing-v3",
]
ALGOS: List[str] = ["ppo_gauss", "ppo_beta", "grpo"]
SEEDS: List[int] = [0, 1, 2]
GRPO_G_VALUES: List[int] = [4, 8, 16, 32, 64, 128]

# --- Base Hyperparameters ---
BASE_CONFIG: Dict[str, Any] = {
    "total_steps": 1_000_000,
    "gamma": 0.99,
    "max_episode_steps": 1000, # Default, overridden if needed
    # Logging/Saving intervals
    "log_interval": 10000,     # Log more frequently maybe
    "checkpoint_interval": 100000,
    "video_interval": 250000,
    "verbose": False,
    "network_type": "mlp",      # Default network
    # MLP structure
    "mlp_hidden_dims": [64, 64],
    # CNN structure (only used if network_type="cnn")
    "cnn_output_features": 256,
    # Default Learning Rate (can be algo/env specific)
    "lr": 3e-4,
}

# --- Algorithm Specific Parameters ---
ALGO_PARAMS: Dict[str, Dict[str, Any]] = {
    "ppo_gauss": {
        "rollout_steps": 2048, # Default, overridden by env
        "num_minibatches": 32, # rollout_steps / num_minibatches = minibatch_size (64)
        "ppo_epochs": 10,
        "lam": 0.95,          # GAE lambda
        "clip_eps": 0.2,
        "entropy_coef": 0.0,  # Typically 0 for Mujoco PPO
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,    # Optional KL early stopping
    },
    "ppo_beta": {
        "rollout_steps": 2048,
        "num_minibatches": 32,
        "ppo_epochs": 10,
        "lam": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.0,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    },
    "grpo": {
        # group_size (G) set dynamically
        "update_epochs": 10,
        "max_grad_norm": 0.5,
        "entropy_coef": 0.001, # Small entropy bonus often helps GRPO/PG
        "kl_coef": 0.01,       # KL vs reference policy weight
        "ref_update_interval": 10000, # How often to update reference policy
        "minibatch_size": 256, # For the update loop over collected steps
    }
}

# --- Environment Specific Overrides ---
ENV_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "HalfCheetah-v4": {"rollout_steps": 2048},
    "Hopper-v4": {"rollout_steps": 2048},
    "Walker2d-v4": {"rollout_steps": 2048},
    "Swimmer-v4": {"rollout_steps": 2048}, # PPO paper uses 2048, maybe adjust later
    "InvertedPendulum-v4": {"rollout_steps": 512, "max_episode_steps": 1000},
    "InvertedDoublePendulum-v4": {"rollout_steps": 2048, "max_episode_steps": 1000}, # Needs larger rollout?
    "CarRacing-v3": {
        "network_type": "cnn",
        "max_episode_steps": 1000,
        "gamma": 0.995, # Often higher gamma helps CarRacing
        # PPO params might need tuning for CarRacing (smaller batches?)
        "rollout_steps": 1024, # Smaller rollout for memory with CNN
        "num_minibatches": 16, # -> minibatch_size = 64
        "lr": 2.5e-4,
        "clip_eps": 0.1, # From original script
        "entropy_coef": 0.01, # From original script
         # GRPO params for CarRacing
        "grpo": {
             "minibatch_size": 128, # Adjust for potentially smaller rollouts
             "lr": 1e-4, # Potentially lower LR for CNN
        }
    }
}
# --- Validation Function ---
def validate_config(config: Dict[str, Any], filename: str) -> bool:
    """Basic validation for required keys and PPO batch sizes."""
    required = ["env_id", "algo", "seed", "total_steps"]
    if not all(k in config for k in required):
        print(f"Validation FAIL {filename}: Missing one of {required}")
        return False

    if "ppo" in config["algo"]:
        if "rollout_steps" not in config or "num_minibatches" not in config:
            print(f"Validation FAIL {filename}: PPO config missing rollout_steps or num_minibatches")
            return False
        if config["rollout_steps"] % config["num_minibatches"] != 0:
            print(f"Validation FAIL {filename}: PPO rollout_steps ({config['rollout_steps']}) "
                  f"not divisible by num_minibatches ({config['num_minibatches']})")
            # You could automatically adjust num_minibatches here if desired
            # e.g., config["num_minibatches"] = max(1, config["rollout_steps"] // 64) # Aim for mb size ~64
            return False
        config["minibatch_size"] = config["rollout_steps"] // config["num_minibatches"]

    if config["algo"] == "grpo":
         if "group_size" not in config:
              print(f"Validation FAIL {filename}: GRPO config missing group_size")
              return False
         if "minibatch_size" not in config:
              print(f"Validation FAIL {filename}: GRPO config missing minibatch_size (for update step)")
              return False

    return True

# --- Generation Loop ---
def generate_configs() -> None:
    """Generates all configuration files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for env_id in ENVS:
        for algo in ALGOS:
             # PPO_Beta only for continuous (Box space) environments
             if algo == "ppo_beta" and env_id in ["CarRacing-v3"]: # Add others if needed
                 # Simple check, ideally check env.action_space type
                 is_discrete = False # Assume Mujoco/CarRacing are continuous
                 if is_discrete:
                     print(f"Skipping ppo_beta for discrete env: {env_id}")
                     continue

             for seed in SEEDS:
                 # Base config for this combination
                 cfg = deepcopy(BASE_CONFIG)
                 cfg["env_id"] = env_id
                 cfg["algo"] = algo
                 cfg["seed"] = seed

                 # Apply environment overrides (including nested GRPO overrides)
                 env_cfg = ENV_OVERRIDES.get(env_id, {})
                 grpo_env_cfg = env_cfg.pop("grpo", {}) # Extract nested GRPO overrides
                 cfg.update(env_cfg)

                 # Apply base algorithm parameters
                 cfg.update(ALGO_PARAMS.get(algo, {}))

                 # Apply specific GRPO env overrides if algo is GRPO
                 if algo == "grpo":
                     cfg.update(grpo_env_cfg)

                 # Handle GRPO G-Values
                 g_values_for_run = GRPO_G_VALUES if algo == "grpo" else [None] # Use None for non-GRPO algos

                 for g_val in g_values_for_run:
                     final_cfg = deepcopy(cfg) # Copy before setting G
                     filename_parts = [env_id, algo]

                     if g_val is not None: # This is a GRPO run with a specific G
                         final_cfg["group_size"] = g_val
                         filename_parts.append(f"g{g_val}")

                     filename_parts.append(f"seed{seed}")
                     filename = "_".join(filename_parts) + ".json"
                     filepath = OUTPUT_DIR / filename

                     # Validate before saving
                     if validate_config(final_cfg, filename):
                         with open(filepath, 'w') as f:
                             json.dump(final_cfg, f, indent=4, sort_keys=True)
                         count += 1
                     else:
                         print(f"-> Config generation skipped for {filename}")


    print(f"\nGenerated {count} configuration files in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    generate_configs()