"""
Main experiment runner for PPO (Gaussian/Beta) and GRPO agents.

Loads configurations from JSON files, handles parallel execution using threading,
manages logging, checkpointing, evaluation, and video recording.

Author: Antonio Lobo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np

# Set KMP_DUPLICATE_LIB_OK if needed (e.g., for matplotlib/torch conflicts on some systems)
# It's generally better to resolve the underlying issue if possible.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

import utils
from base_agent import BaseAgent
from grpo import GRPO_NoCritic
from ppo_beta import PPO_Beta
from ppo_revised import PPO

# --- Algorithm Registry ---
ALGO_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "ppo_gauss": PPO,
    "ppo_beta": PPO_Beta,
    "grpo": GRPO_NoCritic,
    # Add aliases if needed
    "ppo": PPO,
}

# --- Plotting Configuration ---
# Matplotlib backend configuration (important for non-interactive environments)
# Agg is good for saving figures without a display.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: Matplotlib not found or backend error. Plotting disabled.")
    PLOTTING_ENABLED = False


def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    print(f"Loading config: {config_path.name}")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        # Add config file name to config dict for reference
        config['config_file'] = config_path.name
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        raise


def run_single_experiment(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Runs a single RL experiment based on the provided config."""
    start_time = time.perf_counter()
    run_id = f"{config['env_id']}_{config['algo']}_seed{config['seed']}"
    print(f"--- Starting Experiment: {run_id} ---")

    # --- Setup ---
    utils.seed_everything(config['seed'], deterministic_torch=args.deterministic)
    env = utils.make_env(
        env_id=config['env_id'],
        seed=config['seed'],
        max_episode_steps=config.get('max_episode_steps') # Load from config if specified
    )

    # --- Agent Initialization ---
    algo_key = config.get('algo', 'ppo_gauss').lower() # Default to ppo_gauss
    if algo_key not in ALGO_REGISTRY:
        print(f"ERROR: Unknown algorithm '{algo_key}' in config '{config.get('config_file', 'N/A')}'. Skipping.")
        env.close()
        return

    AgentClass: Type[BaseAgent] = ALGO_REGISTRY[algo_key]
    try:
        agent = AgentClass(
            env=env,
            config=config,
            device=args.device # Pass device from CLI args
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize agent for {run_id}. Config: {config.get('config_file', 'N/A')}")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return # Skip this run

    # --- Training ---
    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n=== Training attempt {attempt}/{MAX_RETRIES} ===")
            agent.train()
            print(f"✅ Training succeeded on attempt {attempt}")
            break                                                   # success → leave the loop
        except Exception as e:
            print(f"\n--- ERROR during training (attempt {attempt}) for {run_id} ---")
            print(f"Config: {config.get('config_file', 'N/A')}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print(f"--- End of Error Report for {run_id} ---")

            # Clean up the environment even when training fails
            try:
                env.close()
            except Exception:
                pass                                                # swallow cleanup errors

            if attempt == MAX_RETRIES:
                print("❌ Maximum retries reached. Aborting.")
                raise                                               # re-raise to signal unrecoverable failure
            else:
                print("🔄 Retrying...\n")

    # --- Cleanup and Timing ---
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"--- Finished Experiment: {run_id} (Duration: {duration:.2f}s) ---")


def plot_learning_curves(base_log_dir: Path, output_dir: Path) -> None:
    """Plots learning curves for all runs found in subdirectories."""
    if not PLOTTING_ENABLED:
        print("Plotting disabled.")
        return

    print(f"\n--- Generating Learning Curve Plots from {base_log_dir} ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    env_algo_groups: Dict[Tuple[str, str], List[Path]] = {}

    # Group runs by (environment, algorithm) across all config subfolders
    for config_subdir in base_log_dir.iterdir():
        if not config_subdir.is_dir():
            continue
        for run_dir in config_subdir.iterdir():
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / "metrics.json"
            if not metrics_file.is_file():
                continue

        # Infer env_id and algo from directory name (e.g., CartPole-v1_ppo_gauss_seed0)
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue # Needs at least env_algo_seed
        seed_str = parts[-1]
        algo = parts[-2]
        env_id = "_".join(parts[:-2])

        if not seed_str.startswith("seed"): continue

        # Normalize common algo names if needed for grouping
        if algo == "ppo": algo = "ppo_gauss"

        key = (env_id, algo)
        env_algo_groups.setdefault(key, []).append(metrics_file)

    # Create plots for each group
    for (env_id, algo), metric_files in env_algo_groups.items():
        plt.figure(figsize=(8, 5))
        print(f"Plotting for: {env_id} - {algo} ({len(metric_files)} seeds)")

        all_rewards = []
        min_len = float('inf')

        for metrics_file in metric_files:
            try:
                metrics = utils.load_metrics(metrics_file)
                if "steps" not in metrics or "avg_episodic_reward" not in metrics:
                    print(f"  Skipping {metrics_file.parent.name}: Missing required keys.")
                    continue

                steps = metrics["steps"]
                rewards = metrics["avg_episodic_reward"]

                # Simple plot per seed
                # plt.plot(steps, rewards, alpha=0.5, linewidth=1)

                # For mean/std plot, need to align steps or interpolate
                # Store rewards and find min length for truncation
                all_rewards.append(np.array(rewards))
                min_len = min(min_len, len(rewards))

            except Exception as e:
                print(f"  Error processing {metrics_file.parent.name}: {e}")

        if not all_rewards or min_len == float('inf'):
             print(f"  No valid data found for {env_id} - {algo}. Skipping plot.")
             plt.close()
             continue

        # Truncate all reward arrays to the minimum length
        truncated_rewards = [r[:min_len] for r in all_rewards]
        rewards_arr = np.array(truncated_rewards)

        # Calculate mean and std dev across seeds
        mean_rewards = np.mean(rewards_arr, axis=0)
        std_rewards = np.std(rewards_arr, axis=0)

        # Use steps from one of the runs (assuming steps are roughly similar)
        # A more robust approach would interpolate rewards onto a common step axis.
        try:
           steps = utils.load_metrics(metric_files[0])["steps"][:min_len]
           plt.plot(steps, mean_rewards, label=f"{algo} (mean over {len(all_rewards)} seeds)")
           plt.fill_between(steps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
        except Exception as e:
           print(f"  Error getting steps or plotting mean/std for {env_id} - {algo}: {e}")
           # Fallback to simple plotting if mean/std fails
           for i, metrics_file in enumerate(metric_files):
               try:
                   metrics = utils.load_metrics(metrics_file)
                   plt.plot(metrics["steps"], metrics["avg_episodic_reward"], alpha=0.5, linewidth=1, label=f"Seed {i}")
               except Exception: pass # Ignore errors in fallback


        plt.title(f"Learning Curve: {env_id}")
        plt.xlabel("Environment Steps")
        plt.ylabel("Average Episodic Reward")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize="small")
        plt.tight_layout()

        plot_filename = output_dir / f"learning_curve_{env_id}_{algo}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"  Saved plot: {plot_filename.name}")

    print("--- Finished Plotting ---")


def main() -> None:
    """Parses arguments, loads configs, and runs experiments."""
    parser = argparse.ArgumentParser(description="Run RL experiments from config files.")
    parser.add_argument(
        "--config-dir", "-cd",
        type=str,
        required=True,
        help="Directory containing JSON configuration files for experiments."
    )
    parser.add_argument(
        "--base-log-dir", "-ld",
        type=str,
        default="experiment_runs",
        help="Base directory to store results (logs, checkpoints, videos, plots)."
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None, # Auto-detect (cuda if available, else cpu)
        choices=['cpu', 'cuda'],
        help="Torch device to use ('cpu' or 'cuda')."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of experiments to run in parallel (threads)."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms in PyTorch (can impact performance)."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating learning curve plots after experiments finish."
    )
    # Add arguments for specific hyperparameter overrides if needed, e.g.:
    # parser.add_argument("--lr", type=float, help="Override learning rate from config.")

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    base_log_dir = Path(args.base_log_dir)
    plot_output_dir = base_log_dir / "_plots" # Store plots in a subfolder

    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    # Find all JSON config files
    config_files = list(config_dir.glob("*.json"))
    if not config_files:
        print(f"Error: No JSON config files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} configuration files in {config_dir}")
    print(f"Running up to {args.workers} experiments in parallel.")
    print(f"Results will be saved under: {base_log_dir}")
    print(f"Using device: {utils.get_device(args.device)}")
    if args.deterministic:
        print("Using deterministic PyTorch algorithms.")

    # Load all configs first to catch errors early
    configs_to_run: List[Dict[str, Any]] = []
    for cfg_path in config_files:
        try:
            config = load_config(cfg_path)
            # --- Mandatory fields check ---
            required_keys = ['env_id', 'algo', 'seed']
            if not all(key in config for key in required_keys):
                print(f"WARNING: Config {cfg_path.name} missing one of required keys {required_keys}. Skipping.")
                continue
            
            # --- Create a per-config subfolder and point logs there ---
            config_subdir = base_log_dir / cfg_path.stem
            config_subdir.mkdir(parents=True, exist_ok=True)
            config['base_log_dir'] = str(config_subdir)
            # --- TODO: Add more validation as needed ---
            configs_to_run.append(config)
        except Exception:
            print(f"Skipping invalid config file: {cfg_path.name}")

    if not configs_to_run:
        print("No valid configurations to run.")
        sys.exit(0)

    print(f"\nStarting {len(configs_to_run)} experiments...")
    overall_start_time = time.perf_counter()

    # --- Run experiments using ThreadPoolExecutor ---
    # Limit torch threads per worker to avoid oversubscription on CPU
    # This might need adjustment based on the system and workload.
    try:
         torch.set_num_threads(max(1, torch.get_num_threads() // args.workers))
         print(f"Set torch num_threads per worker: {torch.get_num_threads()}")
    except Exception as e:
         print(f"Warning: Could not set torch threads: {e}")


    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single_experiment, config, args) for config in configs_to_run]
        num_completed = 0
        for future in as_completed(futures):
            num_completed += 1
            try:
                future.result() # Raise exceptions if they occurred in the thread
            except Exception as e:
                print(f"--- ERROR in experiment worker thread ---")
                print(f"Error details: {e}")
                import traceback
                traceback.print_exc()
                print(f"--- End of Worker Error Report ---")
            print(f"Progress: {num_completed}/{len(configs_to_run)} experiments completed.")


    overall_end_time = time.perf_counter()
    total_duration = overall_end_time - overall_start_time
    print(f"\n--- All experiments finished. Total duration: {total_duration:.2f}s ---")

    # --- Generate Plots ---
    if not args.skip_plots:
        plot_learning_curves(base_log_dir, plot_output_dir)

    print("--- Experiment Runner Exiting ---")


if __name__ == "__main__":
    main()
