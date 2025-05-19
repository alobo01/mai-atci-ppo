"""
Generates JSON configuration files for RL experiments, compatible with Pydantic models.

Creates configs for specified environments, algorithms (PPO, GRPO),
distribution types (normal, beta), seeds, and GRPO group sizes (G).
"""
import json
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any

# --- Configuration ---
OUTPUT_DIR = Path("configs_generated")  # Avoid overwriting manual ones
ENVS: List[str] = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Swimmer-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "CarRacing-v3",
]
ALGOS: List[str] = ["ppo", "grpo"]
DISTRIBUTION_TYPES: List[str] = ["normal", "beta"]
SEEDS: List[int] = [0, 1, 2]
GRPO_G_VALUES: List[int] = [4, 16, 64]

# --- Hyperparameter grids for testing ---
ENTROPY_COEFS = [0.0, 0.001, 1e-5]
KL_COEFS    = [0.001, 0.01, 0.1]

# --- Base Experiment Hyperparameters ---
BASE_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "total_steps": 1_000_000,
    "gamma": 0.99,
    "max_episode_steps": 1000,
    "log_interval": 10000,
    "checkpoint_interval": 100000,
    "video_interval": 250000,
    "verbose": False,
}

# --- Base Network Configuration ---
BASE_NETWORK_CONFIG: Dict[str, Any] = {
    "network_type": "mlp",
    "mlp_hidden_dims": [64, 64],
    "cnn_output_features": 256,
}

# --- Base PPO Algorithm Parameters ---
BASE_PPO_PARAMS: Dict[str, Any] = {
    "lr": 3e-4,
    "rollout_steps": 2048,
    "num_minibatches": 32,
    "lam": 0.95,
    "clip_eps": 0.2,
    "ppo_epochs": 10,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": None,
}

# --- Base GRPO Algorithm Parameters ---
BASE_GRPO_PARAMS: Dict[str, Any] = {
    "lr": 1e-4,
    "update_epochs": 10,
    "max_grad_norm": 0.5,
    "ref_update_interval": 10000,
    "minibatch_size": 256,
    "rollout_steps_per_trajectory": 1000,
}

# --- Environment Specific Overrides ---
ENV_GENERAL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "InvertedPendulum-v4": {"max_episode_steps": 1000},
    "InvertedDoublePendulum-v4": {"max_episode_steps": 1000},
    "Swimmer-v4": {
        "max_episode_steps": 1000,
        "network_config": {
            "network_type": "mlp",
            "mlp_hidden_dims": [128, 128],
            "cnn_output_features": 256,
        },
    },
    "CarRacing-v3": {
        "gamma": 0.99,
        "max_episode_steps": 1000,
        "network_config": {
            "network_type": "cnn",
            "mlp_hidden_dims": [256],
            "cnn_output_features": 256,
        },
    },
}

# --- Environment Specific Algo Parameter Overrides ---
ENV_ALGO_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "InvertedPendulum-v4": {
        "ppo": {"rollout_steps": 512, "num_minibatches": 8},
    },
    "Swimmer-v4": {
        "ppo": {"lr": 2e-4, "entropy_coef": 0.01},
        "grpo": {"lr": 5e-5, "entropy_coef": 0.002},
    },
    "CarRacing-v3": {
        "ppo": {
            "lr": 2.5e-4,
            "rollout_steps": 4096,
            "num_minibatches": 64,
            "clip_eps": 0.1,
            "entropy_coef": 0.01,
        },
        "grpo": {
            "lr": 1e-4,
            "minibatch_size": 128,
            "rollout_steps_per_trajectory": 512,
            "entropy_coef": 0.005,
        },
    },
}

def generate_configs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for env_id in ENVS:
        is_discrete = "CartPole" in env_id
        for algo in ALGOS:
            for dist in DISTRIBUTION_TYPES:
                if is_discrete and dist == "beta":
                    continue
                if is_discrete and dist != "categorical":
                    continue

                # choose hyper loops
                param_combinations = []
                if algo == "ppo":
                    for ent in ENTROPY_COEFS:
                        param_combinations.append({"entropy_coef": ent})
                else:
                    for ent in ENTROPY_COEFS:
                        for kl in KL_COEFS:
                            param_combinations.append({"entropy_coef": ent, "kl_coef": kl})

                for seed in SEEDS:
                    for params in param_combinations:
                        exp_cfg = deepcopy(BASE_EXPERIMENT_CONFIG)
                        exp_cfg.update({"env_id": env_id, "algo": algo, "seed": seed})
                        exp_cfg.update(ENV_GENERAL_OVERRIDES.get(env_id, {}))

                        net_cfg = deepcopy(BASE_NETWORK_CONFIG)
                        net_cfg.update(ENV_GENERAL_OVERRIDES.get(env_id, {}).get("network_config", {}))
                        exp_cfg["network_config"] = net_cfg

                        if algo == "ppo":
                            cfg = deepcopy(BASE_PPO_PARAMS)
                            cfg["distribution_type"] = ("categorical" if is_discrete else dist)
                            cfg.update(params)
                            cfg.update(ENV_ALGO_OVERRIDES.get(env_id, {}).get("ppo", {}))
                            exp_cfg["ppo_config"] = cfg
                        else:
                            cfg = deepcopy(BASE_GRPO_PARAMS)
                            cfg["distribution_type"] = ("categorical" if is_discrete else dist)
                            cfg.update(params)
                            cfg.update(ENV_ALGO_OVERRIDES.get(env_id, {}).get("grpo", {}))
                            exp_cfg["grpo_config"] = cfg

                        g_values = GRPO_G_VALUES if algo == "grpo" else [None]
                        for g in g_values:
                            cfg_copy = deepcopy(exp_cfg)
                            fname_parts = [env_id, algo]
                            if not is_discrete:
                                fname_parts.append(dist)
                            if algo == "grpo" and g is not None:
                                cfg_copy["grpo_config"]["group_size"] = g
                                fname_parts.append(f"g{g}")
                            fname_parts.append(f"seed{seed}")
                            # include hyperparams in filename
                            fname_parts.append(f"ent{params['entropy_coef']}")
                            if algo == "grpo":
                                fname_parts.append(f"kl{params['kl_coef']}")
                            fname = "_".join(str(x) for x in fname_parts) + ".json"
                            fpath = OUTPUT_DIR / fname

                            if algo == "ppo":
                                pc = cfg_copy["ppo_config"]
                                if pc["rollout_steps"] % pc["num_minibatches"] != 0:
                                    continue

                            with open(fpath, 'w') as f:
                                json.dump(cfg_copy, f, indent=4, sort_keys=True)
                            count += 1
    print(f"Generated {count} configuration files in '{OUTPUT_DIR}' with varied entropy/kl.")

if __name__ == "__main__":
    generate_configs()
