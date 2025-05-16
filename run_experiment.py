import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np # For reward stats
import torch

from algorithms.base_agent import BaseAgent
from algorithms.ppo import PPO
from algorithms.grpo import GRPO_NoCritic
from utils import (env_utils, logging_utils, reproducibility_utils,
                   torch_utils, video_plot_utils) # video_plot_utils still needed for metrics
from utils.pydantic_models import ExperimentConfig
from utils.timing_utils import Timing

# Algorithm Registry
ALGO_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "ppo": PPO,
    "grpo": GRPO_NoCritic,
}

def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Loads a JSON configuration file into an ExperimentConfig Pydantic model."""
    # print(f"Loading config: {config_path.name}") # Less verbose for runner
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
    Runs a single RL experiment.
    Returns a dictionary with run_id and final metrics if successful, None otherwise.
    """
    start_time = time.perf_counter()
    
    algo_specific_conf = config.get_algo_specific_config()
    dist_type_suffix = ""
    if hasattr(algo_specific_conf, 'distribution_type'):
        dist_type_suffix = f"_{getattr(algo_specific_conf, 'distribution_type')}"
    
    g_value_suffix = ""
    if config.algo == "grpo" and hasattr(algo_specific_conf, 'group_size'):
        g_value_suffix = f"_g{getattr(algo_specific_conf, 'group_size')}"

    run_id = config.run_name or f"{config.env_id}_{config.algo}{dist_type_suffix}{g_value_suffix}_seed{config.seed}"
    
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

        AgentClass: Type[BaseAgent] = ALGO_REGISTRY[config.algo.lower()]
        agent: BaseAgent
        agent = AgentClass(
            env=env,
            config=config_for_agent,
            device_str=cli_args.device
        )
        
        run_base_path = agent.log_dir.parent 
        config_save_path = run_base_path / "config.json"
        with open(config_save_path, 'w') as f:
            json.dump(config.model_dump(exclude={'_config_file_path'}), f, indent=4)
        # print(f"Saved run configuration to: {config_save_path}") # Less verbose

        agent.train() # This is the main training call
        
        # Training finished successfully, load final metrics for summary
        final_metrics = video_plot_utils.load_metrics(agent.results_file)
        final_rewards = [r for r in final_metrics.get("avg_episodic_reward", []) if r is not None]
        
        # Get last N rewards if available
        last_n_rewards = final_rewards[-cli_args.reward_stats_window:] if final_rewards else []

        run_summary = {
            "run_id": run_id,
            "status": "completed",
            "duration_s": time.perf_counter() - start_time,
            "final_rewards_window": last_n_rewards,
            "metrics_file": str(agent.results_file.resolve())
        }

    except Exception as e:
        print(f"--- ERROR in experiment: {run_id} (Original Config: {config_file_name_original}) ---")
        print(f"Error details: {e}")
        # import traceback # Keep for debugging if needed locally
        # traceback.print_exc()
        run_summary = {
            "run_id": run_id,
            "status": "errored",
            "error_message": str(e),
            "duration_s": time.perf_counter() - start_time,
            "original_config_file": config_file_name_original
        }
    finally:
        # Ensure environment is closed if it was opened
        if 'env' in locals() and env is not None:
            try:
                env.close()
            except Exception: pass
        if 'agent' in locals() and hasattr(agent, 'env') and agent.env is not None:
            try:
                agent.env.close()
            except Exception: pass

    end_time_single = time.perf_counter()
    duration_single = end_time_single - start_time
    status_message = "Completed" if run_summary["status"] == "completed" else "Errored"
    print(f"--- Experiment {run_id}: {status_message} (Duration: {duration_single:.2f}s) ---")
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL experiments from config files.")
    parser.add_argument(
        "--config-dir", "-cd", type=str, required=True,
        help="Directory containing JSON configuration files."
    )
    parser.add_argument(
        "--base-log-dir", "-ld", type=str, default="experiment_runs",
        help="Base directory to store all results."
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None, choices=['cpu', 'cuda'],
        help="Torch device ('cpu' or 'cuda'). Auto-detects if None."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=1,
        help="Number of experiments to run in parallel."
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms in PyTorch (can impact performance)."
    )
    parser.add_argument(
        "--env-filter", type=str, default=None,
        help="If set, only run experiments whose env_id equals this value."
    )
    parser.add_argument(
        "--reward-stats-window", type=int, default=10,
        help="Number of last logged average episodic rewards to consider for summary statistics."
    )
    cli_args = parser.parse_args()

    config_dir = Path(cli_args.config_dir)
    master_log_dir = Path(cli_args.base_log_dir) # All runs will be subdirs of this

    if not config_dir.is_dir():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)

    all_json_configs = list(config_dir.glob("*.json"))
    if not all_json_configs:
        print(f"Error: No JSON config files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(all_json_configs)} config files in {config_dir}")
    
    configs_to_process: List[ExperimentConfig] = []
    for cfg_path in all_json_configs:
        try:
            loaded_cfg = load_experiment_config(cfg_path)
            if cli_args.env_filter and loaded_cfg.env_id != cli_args.env_filter:
                continue
            configs_to_process.append(loaded_cfg)
        except Exception as e:
            print(f"Skipping invalid config file {cfg_path.name} due to error: {e}")

    if not configs_to_process:
        print("No valid configurations to run after filtering.")
        sys.exit(0)
    
    num_total_configs = len(configs_to_process)
    print(f"Processing {num_total_configs} configurations.")
    print(f"Running up to {cli_args.workers} experiments in parallel.")
    print(f"Results will be saved under subdirectories of: {master_log_dir}")
    effective_device = torch_utils.get_device(cli_args.device)
    print(f"Using device: {effective_device}")
    if cli_args.deterministic: print("Attempting to use deterministic PyTorch algorithms.")

    overall_start_time = time.perf_counter()
    experiment_summaries: List[Optional[Dict[str, Any]]] = []


    if effective_device.type == 'cpu' and cli_args.workers > 1:
        try:
            torch.set_num_threads(1)
            # print(f"Set torch num_threads per worker (CPU): {torch.get_num_threads()}") # Less verbose
        except Exception: pass # Ignore if fails
    
    if cli_args.workers > 1:
        with ThreadPoolExecutor(max_workers=cli_args.workers) as executor:
            futures = [executor.submit(run_single_experiment, cfg, cli_args) for cfg in configs_to_process]
            num_completed_threads = 0
            for future in as_completed(futures):
                num_completed_threads += 1
                try:
                    result = future.result()
                    experiment_summaries.append(result)
                except Exception as e_thread: # Should ideally be caught within run_single_experiment
                    print(f"--- UNHANDLED ERROR in experiment worker thread: {e_thread} ---")
                    # This indicates an error *outside* the try-except in run_single_experiment
                    experiment_summaries.append({
                        "run_id": "UNKNOWN_DUE_TO_THREAD_ERROR",
                        "status": "thread_error",
                        "error_message": str(e_thread)
                    })
                print(f"Progress: {num_completed_threads}/{num_total_configs} experiments submitted to threads and result processed.")
    else:
        for i, cfg in enumerate(configs_to_process):
            # print(f"Running experiment {i+1}/{num_total_configs} sequentially...") # Less verbose
            result = run_single_experiment(cfg, cli_args)
            experiment_summaries.append(result)
            # print(f"Progress: {i+1}/{num_total_configs} experiments processed.") # Less verbose


    overall_end_time = time.perf_counter()
    print(f"\n--- All experiments processed. Total duration: {(overall_end_time - overall_start_time):.2f}s ---")

    # --- Final Summary ---
    print("\n--- Experiment Run Summary ---")
    completed_runs = [s for s in experiment_summaries if s and s["status"] == "completed"]
    errored_runs = [s for s in experiment_summaries if s and s["status"] != "completed"]

    print(f"Total configurations processed: {num_total_configs}")
    print(f"Successfully completed: {len(completed_runs)}")
    print(f"Errored or failed: {len(errored_runs)}")

    if completed_runs:
        print("\nSummary of completed runs (final reward stats for last ~{cli_args.reward_stats_window} log points):")
        for summary in completed_runs:
            rewards = summary.get("final_rewards_window", [])
            if rewards:
                mean_r = np.mean(rewards)
                std_r = np.std(rewards)
                min_r = np.min(rewards)
                max_r = np.max(rewards)
                print(f"  Run: {summary['run_id']}")
                print(f"    Mean Reward: {mean_r:.2f}, Std: {std_r:.2f}, Min: {min_r:.2f}, Max: {max_r:.2f} (over {len(rewards)} points)")
                print(f"    Metrics file: {summary.get('metrics_file', 'N/A')}")
            else:
                print(f"  Run: {summary['run_id']} - No reward data found in final metrics for summary.")
                print(f"    Metrics file: {summary.get('metrics_file', 'N/A')}")


    if errored_runs:
        print("\nSummary of errored/failed runs:")
        for summary in errored_runs:
            print(f"  Run ID (if known): {summary.get('run_id', 'N/A')}")
            print(f"    Status: {summary['status']}")
            print(f"    Error: {summary.get('error_message', 'Unknown error')}")
            if "original_config_file" in summary:
                print(f"    Original Config: {summary['original_config_file']}")
    
    master_log_dir.mkdir(parents=True, exist_ok=True)
    summary_file_path = master_log_dir / f"experiment_summary_{time.strftime('%Y%m%d-%H%M%S')}.json"
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(experiment_summaries, f, indent=4)
        print(f"\nFull summary saved to: {summary_file_path}")
    except Exception as e_json:
        print(f"\nError saving full summary JSON: {e_json}")


    print("\n--- Experiment Runner Exiting ---")

if __name__ == "__main__":
    main()