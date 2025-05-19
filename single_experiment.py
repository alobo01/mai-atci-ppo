import argparse
import json
import sys
import time
import numpy as np  # For reward stats
import torch

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from algorithms.base_agent import BaseAgent
from algorithms.ppo import PPO
from algorithms.grpo import GRPO_NoCritic
from utils import (
    env_utils,
    logging_utils,
    reproducibility_utils,
    torch_utils,
    video_plot_utils,
)
from utils.pydantic_models import ExperimentConfig, PPOConfig, GRPOConfig
from utils.timing_utils import Timing

# Algorithm Registry
ALGO_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "ppo": PPO,
    "grpo": GRPO_NoCritic,
}

def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Loads a JSON configuration file into an ExperimentConfig Pydantic model."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config_dict['_config_file_path'] = str(config_path.resolve())
        return ExperimentConfig(**config_dict)
    except Exception as e:
        print(f"Error loading or parsing config file {config_path}: {e}")
        raise


def run_single_experiment(config: ExperimentConfig, cli_args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """
    Runs a single RL experiment and returns a summary dict.
    """
    start_time = time.perf_counter()

    algo_specific_conf = config.get_algo_specific_config()
    
    run_id_parts = [
        config.env_id,
        config.algo,
        f"seed{config.seed}"
    ]
    if hasattr(algo_specific_conf, 'entropy_coef'):
        run_id_parts.append(f"ent{getattr(algo_specific_conf, 'entropy_coef')}")
    if hasattr(algo_specific_conf, 'lr'):
        run_id_parts.append(f"lr{getattr(algo_specific_conf, 'lr')}")
    if hasattr(algo_specific_conf, 'distribution_type'):
        run_id_parts.append(f"{getattr(algo_specific_conf, 'distribution_type')}")
    if config.algo == "grpo" and hasattr(algo_specific_conf, 'group_size'): # This will pick up swept G
         run_id_parts.append(f"g{getattr(algo_specific_conf, 'group_size')}")
    
    run_id = "_".join(run_id_parts)

    print(f"--- Starting Experiment: {run_id} ---")
    config_file_name_original = getattr(config, '_config_file_path', 'N/A')


    try:
        reproducibility_utils.seed_everything(config.seed, deterministic_torch=cli_args.deterministic)

        config_for_agent = config.model_copy(deep=True)
        config_for_agent.base_log_dir = cli_args.base_log_dir


        env = env_utils.make_env(
            env_id=config.env_id,
            seed=config.seed,
            max_episode_steps=config.max_episode_steps
        )

        AgentClass = ALGO_REGISTRY[config.algo.lower()]
        agent = AgentClass(
            env=env,
            config=config_for_agent,
            device_str=cli_args.device
        )

        run_base_path = agent.log_dir.parent
        config_save_path = run_base_path / "config.json"
        with open(config_save_path, 'w') as f:
            json.dump(config.model_dump(exclude={'_config_file_path'}), f, indent=4)
        print(f"Saved run-specific config to: {config_save_path}")

        agent.train()

        final_metrics = video_plot_utils.load_metrics(agent.results_file)
        final_rewards = [r for r in final_metrics.get("avg_episodic_reward", []) if r is not None]
        last_n_rewards = final_rewards[-cli_args.reward_stats_window:] if final_rewards else []

        run_summary = {
            "run_id": run_id,
            "status": "completed",
            "duration_s": time.perf_counter() - start_time,
            "final_rewards_window": last_n_rewards,
            "metrics_file": str(agent.results_file.resolve()),
            "log_dir": str(agent.log_dir.parent.resolve()),
            "config_params": { 
                "seed": config.seed,
                "algo": config.algo,
                "env_id": config.env_id,
                "lr": getattr(algo_specific_conf, 'lr', None),
                "entropy_coef": getattr(algo_specific_conf, 'entropy_coef', None),
                "distribution_type": getattr(algo_specific_conf, 'distribution_type', None),
                "group_size": getattr(algo_specific_conf, 'group_size', None) if config.algo == "grpo" else None,
                "original_run_name_in_config": config.run_name
            }
        }
        run_summary["config_params"] = {k: v for k, v in run_summary["config_params"].items() if v is not None}


    except Exception as e:
        print(f"--- ERROR in experiment: {run_id} (Original Config: {config_file_name_original}) ---")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        run_summary = {
            "run_id": run_id,
            "status": "errored",
            "error_message": str(e),
            "duration_s": time.perf_counter() - start_time,
            "original_config_file": config_file_name_original,
            "config_params": { 
                "seed": config.seed,
                "algo": config.algo,
                "env_id": config.env_id,
                "lr": getattr(algo_specific_conf, 'lr', 'N/A_on_error'),
                "entropy_coef": getattr(algo_specific_conf, 'entropy_coef', 'N/A_on_error'),
                "group_size": getattr(algo_specific_conf, 'group_size', 'N/A_on_error') if config.algo == "grpo" else None,
                "original_run_name_in_config": config.run_name
            }
        }
    finally:
        if 'env' in locals() and env is not None:
            try:
                env.close()
            except Exception:
                pass

    print(f"--- Experiment {run_id}: {run_summary['status'].capitalize()} (Duration: {run_summary['duration_s']:.2f}s) ---")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL experiments with hyperparameter sweeping.")
    parser.add_argument(
        "--config-file", "-cf", type=str, required=True,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        "--base-log-dir", "-ld", type=str, default="experiment_desperate_runs",
        help="Base directory to store results."
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu", choices=['cpu', 'cuda'],
        help="Torch device ('cpu' or 'cuda'). Auto-detects if None."
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms in PyTorch (may impact performance)."
    )
    parser.add_argument(
        "--reward-stats-window", type=int, default=10,
        help="Number of last logged avg episodic rewards to summarize."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="List of seeds to run. Overrides seed in config. If None, uses seed from config file."
    )
    parser.add_argument(
        "--sweep-entropy-coefs", "-sec", type=float, nargs="*", default=None,
        help="List of entropy_coef values to sweep. If None, uses value from config."
    )
    parser.add_argument(
        "--sweep-lrs", "-slr", type=float, nargs="*", default=None,
        help="List of learning_rate values to sweep. If None, uses value from config."
    )
    parser.add_argument(
        "--sweep-group-sizes", "-sgs", type=int, nargs="*", default=None,
        help="List of GRPO group_size (G) values to sweep. If None, uses value from config. Only applies if algo is 'grpo'."
    )
    cli_args = parser.parse_args()

    config_path = Path(cli_args.config_file)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    try:
        base_config = load_experiment_config(config_path)
    except Exception:
        sys.exit(1)

    print(f"Loaded base config: {config_path.name}")
    print(f"Results will be stored under: {cli_args.base_log_dir}")
    device = torch_utils.get_device(cli_args.device)
    print(f"Using device: {device}")
    if cli_args.deterministic:
        print("Deterministic PyTorch algorithms enabled.")

    seeds_to_run = cli_args.seeds if cli_args.seeds is not None else [base_config.seed]
    
    base_algo_conf = base_config.get_algo_specific_config()
    
    entropy_coefs_to_sweep: List[float] = []
    if cli_args.sweep_entropy_coefs is None or len(cli_args.sweep_entropy_coefs) == 0:
        if hasattr(base_algo_conf, 'entropy_coef'):
            entropy_coefs_to_sweep = [base_algo_conf.entropy_coef]
        else:
            entropy_coefs_to_sweep = [0.0] 
            print(f"Warning: entropy_coef not found in base config for {base_config.algo}, using default 0.0 for non-sweep.")
    else:
        entropy_coefs_to_sweep = cli_args.sweep_entropy_coefs

    lrs_to_sweep: List[float] = []
    if cli_args.sweep_lrs is None or len(cli_args.sweep_lrs) == 0:
        if hasattr(base_algo_conf, 'lr'):
            lrs_to_sweep = [base_algo_conf.lr]
        else:
            lrs_to_sweep = [1e-4]
            print(f"Warning: lr not found in base config for {base_config.algo}, using default 1e-4 for non-sweep.")
    else:
        lrs_to_sweep = cli_args.sweep_lrs

    group_sizes_to_sweep: List[int] = []
    if base_config.algo == "grpo":
        if cli_args.sweep_group_sizes is None or len(cli_args.sweep_group_sizes) == 0:
            if isinstance(base_algo_conf, GRPOConfig) and hasattr(base_algo_conf, 'group_size'):
                group_sizes_to_sweep = [base_algo_conf.group_size]
            else: # Should not happen if GRPOConfig is well-defined
                group_sizes_to_sweep = [64] # A sensible default for GRPO if missing
                print(f"Warning: group_size not found in base GRPO config, using default 64 for non-sweep.")
        else:
            group_sizes_to_sweep = cli_args.sweep_group_sizes
    else: # Algo is not GRPO, so group size sweep is not applicable
        group_sizes_to_sweep = [0] # Placeholder, loop will run once for this
        if cli_args.sweep_group_sizes is not None and len(cli_args.sweep_group_sizes) > 0:
            print(f"Warning: --sweep-group-sizes provided, but algorithm is '{base_config.algo}', not 'grpo'. Group size sweep will be ignored.")


    all_summaries: List[Dict[str, Any]] = []
    total_experiments = len(seeds_to_run) * len(entropy_coefs_to_sweep) * len(lrs_to_sweep) * (len(group_sizes_to_sweep) if base_config.algo == "grpo" else 1)
    current_experiment_count = 0

    for seed_val in seeds_to_run:
        for ec_val in entropy_coefs_to_sweep:
            for lr_val in lrs_to_sweep:
                for g_val in group_sizes_to_sweep:
                    # Skip G sweep if algo is not GRPO
                    if base_config.algo != "grpo" and g_val != 0 : # g_val will be 0 if not GRPO from setup above
                        continue

                    current_experiment_count += 1
                    print(f"\n--- Starting Run {current_experiment_count}/{total_experiments} (Seed: {seed_val}, EC: {ec_val}, LR: {lr_val}"
                          f"{f', G: {g_val}' if base_config.algo == 'grpo' else ''}) ---")
                    
                    current_config = base_config.model_copy(deep=True)
                    current_config.seed = seed_val

                    algo_specific_conf_current = current_config.get_algo_specific_config()

                    can_set_ec = hasattr(algo_specific_conf_current, 'entropy_coef')
                    if can_set_ec:
                        algo_specific_conf_current.entropy_coef = ec_val
                    elif cli_args.sweep_entropy_coefs is not None and len(cli_args.sweep_entropy_coefs) > 0 : 
                        print(f"Warning: Config for algo {current_config.algo} does not have 'entropy_coef'. Cannot sweep to {ec_val}.")

                    can_set_lr = hasattr(algo_specific_conf_current, 'lr')
                    if can_set_lr:
                        algo_specific_conf_current.lr = lr_val
                    elif cli_args.sweep_lrs is not None and len(cli_args.sweep_lrs) > 0: 
                        print(f"Warning: Config for algo {current_config.algo} does not have 'lr'. Cannot sweep to {lr_val}.")

                    if current_config.algo == "grpo":
                        if isinstance(algo_specific_conf_current, GRPOConfig):
                            algo_specific_conf_current.group_size = g_val
                        # This case should ideally not be hit if pydantic models are correct
                        elif cli_args.sweep_group_sizes is not None and len(cli_args.sweep_group_sizes) > 0:
                             print(f"Warning: Algo is GRPO but config is not GRPOConfig. Cannot sweep group_size to {g_val}.")


                    summary = run_single_experiment(current_config, cli_args)
                    if summary:
                        all_summaries.append(summary)

    if all_summaries:
        summary_collection_path = Path(cli_args.base_log_dir) / f"all_experiments_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
        Path(cli_args.base_log_dir).mkdir(parents=True, exist_ok=True)
        with open(summary_collection_path, 'w') as f:
            json.dump(all_summaries, f, indent=4)
        print(f"\n--- All {len(all_summaries)} experiment summaries saved to: {summary_collection_path} ---")
    else:
        print("\n--- No experiments were run or no summaries were generated. ---")

if __name__ == "__main__":
    main()