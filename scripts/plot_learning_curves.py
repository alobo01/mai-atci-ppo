"""
Plots learning curves from experiment results.

Generates plots showing mean, standard deviation, and min/max range
of average episodic rewards against environment steps for each
(environment, algorithm) combination found in the results directory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# --- Matplotlib Setup ---
try:
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend for saving figures without GUI
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: Matplotlib not found or backend error. Plotting disabled.")
    PLOTTING_ENABLED = False

# --- Constants ---
INTERPOLATION_STEPS = 200 # Number of points to interpolate onto

# --- Helper Functions ---

def load_metrics(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    """Loads steps and rewards from a metrics.json file."""
    if not filepath.is_file():
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Ensure required keys exist and are not empty
        if "steps" in data and "avg_episodic_reward" in data and \
           len(data["steps"]) > 1 and len(data["avg_episodic_reward"]) > 1:
             # Convert to numpy arrays, ensure equal length
            steps = np.array(data["steps"], dtype=np.float64)
            rewards = np.array(data["avg_episodic_reward"], dtype=np.float64)
            min_len = min(len(steps), len(rewards))
            return {"steps": steps[:min_len], "rewards": rewards[:min_len]}
        else:
             print(f"  Warning: Invalid or empty data in {filepath.name}")
             return None
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None

def interpolate_rewards(
    run_data: Dict[str, np.ndarray],
    common_step_axis: np.ndarray
) -> Optional[np.ndarray]:
    """Interpolates rewards onto a common step axis."""
    steps = run_data["steps"]
    rewards = run_data["rewards"]
    try:
        # np.interp needs increasing x-values
        if np.any(np.diff(steps) <= 0):
             print(f"  Warning: Non-increasing steps found, attempting to fix...")
             # Simple fix: remove non-increasing steps
             valid_indices = np.where(np.diff(steps, prepend=steps[0]-1) > 0)[0]
             if len(valid_indices) < 2: return None # Not enough points
             steps = steps[valid_indices]
             rewards = rewards[valid_indices]

        # Interpolate, handling potential out-of-bounds requests
        interpolated = np.interp(
            common_step_axis,
            steps,
            rewards,
            left=rewards[0], # Value for x < min(steps)
            right=rewards[-1] # Value for x > max(steps)
        )
        return interpolated
    except Exception as e:
        print(f"  Error during interpolation: {e}")
        return None

# --- Main Plotting Function ---

def plot_env_algo_curves(
    env_id: str,
    algo_name: str, # Can be 'ppo_gauss', 'ppo_beta', or 'grpo' (grouping all Gs)
    run_metrics: List[Dict[str, np.ndarray]],
    output_dir: Path,
    total_steps: int # Needed for common axis
):
    """Plots curves for a specific environment and algorithm group."""
    if not run_metrics:
        print(f"No valid run data found for {env_id} - {algo_name}. Skipping plot.")
        return

    # --- Interpolation ---
    common_step_axis = np.linspace(0, total_steps, INTERPOLATION_STEPS)
    interpolated_rewards_list: List[np.ndarray] = []

    print(f"Processing {env_id} - {algo_name} ({len(run_metrics)} seeds)...")
    for i, run_data in enumerate(run_metrics):
        interpolated = interpolate_rewards(run_data, common_step_axis)
        if interpolated is not None:
            interpolated_rewards_list.append(interpolated)
        else:
             print(f"  Skipping seed {i} due to interpolation issues.")


    if len(interpolated_rewards_list) < 2: # Need at least 2 runs for std dev/min/max
        print(f"  Insufficient valid runs ({len(interpolated_rewards_list)}) for std dev plot for {env_id} - {algo_name}.")
        # Optional: Plot individual lines if only 1 run exists
        if len(interpolated_rewards_list) == 1:
             plt.figure(figsize=(10, 6))
             plt.plot(common_step_axis, interpolated_rewards_list[0], label=f"{algo_name} (seed 0)")
        else:
             return # Skip plotting if no valid runs
    else:
        # --- Aggregation ---
        rewards_array = np.vstack(interpolated_rewards_list) # Shape (num_seeds, INTERPOLATION_STEPS)
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        min_rewards = np.min(rewards_array, axis=0)
        max_rewards = np.max(rewards_array, axis=0)
        num_seeds = rewards_array.shape[0]

        # --- Plotting ---
        plt.figure(figsize=(10, 6))
        # Plot Mean
        plt.plot(common_step_axis, mean_rewards, label=f"{algo_name.upper()} (Mean over {num_seeds} seeds)", linewidth=2)
        # Plot Std Dev Range
        plt.fill_between(common_step_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.25, label="Mean +/- Std Dev")
        # Plot Min/Max Range
        plt.fill_between(common_step_axis, min_rewards, max_rewards, alpha=0.15, label="Min/Max Range", color='gray')


    plt.title(f"Learning Curve: {env_id}")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Episodic Reward")
    plt.legend(fontsize="medium")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # Use scientific notation for x-axis if steps are large
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plot_filename = output_dir / f"curve_{env_id}_{algo_name}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {plot_filename.name}")


# --- Main Execution ---

def main():
    if not PLOTTING_ENABLED:
        return

    parser = argparse.ArgumentParser(description="Plot learning curves from RL experiment results.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Base directory containing experiment run folders (e.g., 'experiment_runs')."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None, # Default to results_dir / '_plots'
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--total-steps", "-t",
        type=int,
        default=1_000_000, # Default total steps for x-axis scaling
        help="Maximum steps for the plot's x-axis."
    )
    args = parser.parse_args()

    base_results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_results_dir / "_plots_learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_results_dir.is_dir():
        print(f"Error: Results directory not found: {base_results_dir}")
        return

    print(f"Scanning for results in: {base_results_dir}")
    print(f"Saving plots to: {output_dir}")

    # Group runs by (env_id, base_algo)
    # Treat all 'grpo_gX' as 'grpo' for this plot
    grouped_runs: Dict[Tuple[str, str], List[Dict[str, np.ndarray]]] = defaultdict(list)

    for run_dir in sorted(base_results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"): # Skip plot dirs etc.
            continue

        metrics_file = run_dir / "metrics.json"
        metrics_data = load_metrics(metrics_file)
        if metrics_data is None:
            continue

        # Parse env_id and base algo from directory name
        parts = run_dir.name.split('_')
        if len(parts) < 3: continue # Need at least env_algo_seed

        try:
            seed_str = parts[-1]
            if not seed_str.startswith("seed"): continue
            seed = int(seed_str[4:])

            # Identify algo, handling potential '_gX' for GRPO
            algo_part = parts[-2]
            if algo_part.startswith("g") and len(parts) > 3 and parts[-3] == "grpo":
                 base_algo = "grpo"
                 env_id = "_".join(parts[:-3])
            else:
                 base_algo = algo_part
                 env_id = "_".join(parts[:-2])

            # Add valid run data to the group
            group_key = (env_id, base_algo)
            grouped_runs[group_key].append(metrics_data)

        except (IndexError, ValueError):
            print(f"  Warning: Could not parse run name: {run_dir.name}")
            continue

    # Generate plot for each group
    if not grouped_runs:
         print("No valid experiment runs found to plot.")
         return

    print(f"\nFound data for {len(grouped_runs)} (environment, algorithm) combinations.")
    for (env_id, algo_name), run_data_list in grouped_runs.items():
        plot_env_algo_curves(env_id, algo_name, run_data_list, output_dir, args.total_steps)

    print("\n--- Plotting complete ---")


if __name__ == "__main__":
    main()