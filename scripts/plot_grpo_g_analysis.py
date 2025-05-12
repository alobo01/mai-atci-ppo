"""
Plots GRPO G-parameter analysis curves from experiment results.

For each environment where GRPO was run, generates plots comparing the
learning curves (mean +/- std dev across seeds) for different values
of the 'group_size' (G) hyperparameter.
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
    plt.style.use('seaborn-v0_8-darkgrid')
    # Use a color palette suitable for multiple lines
    cmap = plt.get_cmap("viridis") # Or "plasma", "tab10", "tab20"
    PLOTTING_ENABLED = True
except ImportError:
    print("Warning: Matplotlib not found or backend error. Plotting disabled.")
    PLOTTING_ENABLED = False

# --- Constants ---
INTERPOLATION_STEPS = 200 # Number of points to interpolate onto

# --- Helper Functions (load_metrics, interpolate_rewards - copied from plot_learning_curves) ---
def load_metrics(filepath: Path) -> Optional[Dict[str, np.ndarray]]:
    if not filepath.is_file(): return None
    try:
        with open(filepath, 'r') as f: data = json.load(f)
        if "steps" in data and "avg_episodic_reward" in data and \
           len(data["steps"]) > 1 and len(data["avg_episodic_reward"]) > 1:
            steps = np.array(data["steps"], dtype=np.float64)
            rewards = np.array(data["avg_episodic_reward"], dtype=np.float64)
            min_len = min(len(steps), len(rewards))
            return {"steps": steps[:min_len], "rewards": rewards[:min_len]}
        else: return None
    except Exception: return None

def interpolate_rewards(run_data: Dict[str, np.ndarray], common_step_axis: np.ndarray) -> Optional[np.ndarray]:
    steps = run_data["steps"]; rewards = run_data["rewards"]
    try:
        if np.any(np.diff(steps) <= 0):
            valid_indices = np.where(np.diff(steps, prepend=steps[0]-1) > 0)[0]
            if len(valid_indices) < 2: return None
            steps = steps[valid_indices]; rewards = rewards[valid_indices]
        interpolated = np.interp(common_step_axis, steps, rewards, left=rewards[0], right=rewards[-1])
        return interpolated
    except Exception: return None

# --- Main Plotting Function ---

def plot_grpo_g_comparison(
    env_id: str,
    g_value_runs: Dict[int, List[Dict[str, np.ndarray]]], # Keyed by G value
    output_dir: Path,
    total_steps: int
):
    """Plots GRPO learning curves for different G values on the same axes."""
    if not g_value_runs:
        print(f"No GRPO run data found for {env_id}. Skipping plot.")
        return

    print(f"Processing GRPO G-analysis for {env_id}...")
    plt.figure(figsize=(12, 7))
    common_step_axis = np.linspace(0, total_steps, INTERPOLATION_STEPS)

    # Determine colors based on number of G values
    g_values_sorted = sorted(g_value_runs.keys())
    colors = cmap(np.linspace(0, 1, len(g_values_sorted)))

    for idx, g_val in enumerate(g_values_sorted):
        run_metrics_list = g_value_runs[g_val]
        if not run_metrics_list: continue

        interpolated_rewards_list: List[np.ndarray] = []
        print(f"  Processing G={g_val} ({len(run_metrics_list)} seeds)...")
        for i, run_data in enumerate(run_metrics_list):
            interpolated = interpolate_rewards(run_data, common_step_axis)
            if interpolated is not None:
                interpolated_rewards_list.append(interpolated)
            else:
                print(f"    Skipping seed {i} for G={g_val} due to interpolation issues.")

        if len(interpolated_rewards_list) < 1:
            print(f"    No valid runs for G={g_val} after interpolation.")
            continue

        # Aggregate results for this G value
        rewards_array = np.vstack(interpolated_rewards_list)
        mean_rewards = np.mean(rewards_array, axis=0)
        num_seeds = rewards_array.shape[0]

        # Plot Mean Curve for this G
        label = f"G={g_val} ({num_seeds} seeds)"
        plt.plot(common_step_axis, mean_rewards, label=label, color=colors[idx], linewidth=2)

        # Optionally plot Std Dev Range (can get cluttered)
        if len(interpolated_rewards_list) >= 2:
             std_rewards = np.std(rewards_array, axis=0)
             plt.fill_between(common_step_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, color=colors[idx], alpha=0.15) # Lighter alpha
        # Optional: Add marker for final performance?
        # final_perf_mean = mean_rewards[-1]
        # plt.scatter(common_step_axis[-1], final_perf_mean, color=colors[idx], marker='o', s=50, zorder=5)


    plt.title(f"GRPO Group Size (G) Analysis: {env_id}")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Episodic Reward")
    plt.legend(title="Group Size (G)", fontsize="medium", loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plot_filename = output_dir / f"grpo_g_analysis_{env_id}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved G-analysis plot: {plot_filename.name}")

# --- Main Execution ---

def main():
    if not PLOTTING_ENABLED:
        return

    parser = argparse.ArgumentParser(description="Plot GRPO G-parameter analysis from RL experiment results.")
    parser.add_argument(
        "results_dir",
        type=str,
        help="Base directory containing experiment run folders (e.g., 'experiment_runs')."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None, # Default to results_dir / '_plots_grpo_g'
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
    output_dir = Path(args.output_dir) if args.output_dir else base_results_dir / "_plots_grpo_g_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not base_results_dir.is_dir():
        print(f"Error: Results directory not found: {base_results_dir}")
        return

    print(f"Scanning for GRPO results in: {base_results_dir}")
    print(f"Saving G-analysis plots to: {output_dir}")

    # Group runs by (env_id, g_value)
    grouped_grpo_runs: Dict[str, Dict[int, List[Dict[str, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))

    for run_dir in sorted(base_results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or "grpo" not in run_dir.name:
            continue # Skip non-dirs, plot dirs, and non-GRPO runs

        metrics_file = run_dir / "metrics.json"
        metrics_data = load_metrics(metrics_file)
        if metrics_data is None:
            continue

        # Parse env_id, g_value from directory name (e.g., env_grpo_gX_seedY)
        parts = run_dir.name.split('_')
        if len(parts) < 4: continue # Need at least env_grpo_gX_seedY

        try:
            seed_str = parts[-1]
            g_str = parts[-2]
            algo_str = parts[-3]

            if not seed_str.startswith("seed") or not g_str.startswith("g") or algo_str != "grpo":
                 continue

            g_value = int(g_str[1:])
            env_id = "_".join(parts[:-3])

            # Add valid run data to the nested group
            grouped_grpo_runs[env_id][g_value].append(metrics_data)

        except (IndexError, ValueError):
            print(f"  Warning: Could not parse GRPO run name: {run_dir.name}")
            continue

    # Generate plot for each environment
    if not grouped_grpo_runs:
         print("No valid GRPO experiment runs found to plot.")
         return

    print(f"\nFound GRPO data for {len(grouped_grpo_runs)} environments.")
    for env_id, g_value_data in grouped_grpo_runs.items():
        plot_grpo_g_comparison(env_id, g_value_data, output_dir, args.total_steps)

    print("\n--- GRPO G-Analysis Plotting complete ---")


if __name__ == "__main__":
    main()