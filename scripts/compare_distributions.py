import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
BASE_SWEEPS_DIR = Path("sweeps")  # Adjust if your 'sweeps' folder is elsewhere
PLOTS_OUTPUT_DIR = Path("analysis_plots_dist_comparison") # New dedicated output directory
REWARD_SMOOTHING_WINDOW = 5 # Apply a rolling mean to rewards for smoother plots

# Regex to parse folder names (same as before)
FOLDER_PATTERN = re.compile(
    r"^(?P<env_id>[a-zA-Z0-9_.-]+(?:-v\d+)?)"
    r"_(?P<algo>ppo|grpo)"
    r"_seed(?P<seed>\d+)"
    r"_ent(?P<entropy_coef>[\d.eE+-]+)"
    r"_lr(?P<lr>[\d.eE+-]+)"
    r"_(?P<distribution_type>normal|beta)"
    r"(?:_g(?P<group_size>\d+))?$"
)

# Key timing metrics to compare
TIMING_KEYS_OF_INTEREST = [
    "rollout_phase", "update_phase",
    "actor_pass", "critic_pass", # PPO (MLP or head)
    "actor_mlp", "ref_actor_mlp", # GRPO (MLP or general)
    "backward_pass", "optimizer_step", # General
    "cnn_feature_pass", "cnn_feature", "ref_cnn_feature", # CNN specifics
    "actor_head", "ref_actor_head" # CNN head specifics
]

# --- Helper Functions ---
def parse_folder_name(folder_name_str: str):
    match = FOLDER_PATTERN.match(folder_name_str)
    if match:
        params = match.groupdict()
        try:
            # Convert types, keep original strings for some params if not strictly numeric for this script
            params['seed'] = int(params['seed'])
            # For this comparison, we don't strictly need numeric LR/EC/GS,
            # but keeping conversion for consistency with potential future use.
            params['entropy_coef'] = float(params['entropy_coef'])
            params['lr'] = float(params['lr'])
            if params['group_size'] is not None:
                params['group_size'] = int(params['group_size'])
            else:
                params['group_size'] = np.nan # Important for distinguishing PPO runs
            return params
        except ValueError:
            # print(f"  Warning: Could not parse numeric param in '{folder_name_str}' for dist comparison.")
            return None # Or return params with raw strings if that's acceptable
    return None

def load_learning_curve_data(metrics_file: Path, parsed_params_base: dict):
    data_list = []
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        steps_list = metrics.get("steps", [])
        rewards_list = metrics.get("avg_episodic_reward", [])

        if not steps_list or len(steps_list) != len(rewards_list):
            # print(f"  Warning: Mismatch or missing steps/rewards in {metrics_file}")
            return pd.DataFrame()

        for step, reward in zip(steps_list, rewards_list):
            if reward is not None: # Handle potential nulls in reward list
                # Create a full entry for this step, inheriting all parsed params
                entry = parsed_params_base.copy()
                entry['step'] = step
                entry['avg_episodic_reward'] = float(reward)
                data_list.append(entry)
        
        df = pd.DataFrame(data_list)
        
        if REWARD_SMOOTHING_WINDOW > 1 and not df.empty:
            # Smooth per individual run before any aggregation
            # A unique run identifier is the combination of all params (algo, dist, seed, lr, ec, gs)
            # parsed_params_base already contains these unique identifiers for a run.
            # We need to ensure groupby uses all distinguishing keys of a single run.
            # The DataFrame `df` at this point contains data for ONE run. So, direct rolling is fine.
            df['avg_episodic_reward_smoothed'] = df['avg_episodic_reward'].rolling(
                window=REWARD_SMOOTHING_WINDOW, min_periods=1, center=True).mean()
        elif not df.empty:
            df['avg_episodic_reward_smoothed'] = df['avg_episodic_reward']


        return df
    except Exception as e:
        # print(f"  Error loading learning curve data from {metrics_file}: {e}")
        return pd.DataFrame()

def load_timings_data(timings_file: Path, parsed_params_base: dict):
    data_list = []
    try:
        with open(timings_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    step = log_entry.get("step")
                    if step is None:
                        continue
                    for key in TIMING_KEYS_OF_INTEREST:
                        if key in log_entry and isinstance(log_entry[key], dict):
                            avg_ms = log_entry[key].get("avg_ms")
                            if avg_ms is not None:
                                entry = parsed_params_base.copy()
                                entry['step'] = step
                                entry['timing_key'] = key
                                entry['avg_ms'] = float(avg_ms)
                                data_list.append(entry)
                except json.JSONDecodeError:
                    # print(f"  Warning: Skipping malformed JSON line in {timings_file}")
                    continue
        return pd.DataFrame(data_list)
    except Exception as e:
        # print(f"  Error loading timings data from {timings_file}: {e}")
        return pd.DataFrame()

def discover_and_load_all_distribution_comparison_data(base_sweeps_dir: Path):
    """
    Discovers all runs, parses parameters, and loads learning curve and timing data.
    Adds an 'algo_dist_label' for easy plotting.
    """
    all_learning_curves_list = []
    all_timings_list = []

    for sweep_group_path in sorted(base_sweeps_dir.iterdir()):
        if not sweep_group_path.is_dir():
            continue
        print(f"\nProcessing sweep group for distribution comparison: {sweep_group_path.name}")
        for run_folder in sorted(sweep_group_path.iterdir()):
            if not run_folder.is_dir():
                continue
            
            parsed_params = parse_folder_name(run_folder.name)
            if not parsed_params:
                # print(f"  Skipping folder (no param match): {run_folder.name}")
                continue

            # Create the specific label for plotting
            algo_dist_label = f"{parsed_params['algo'].upper()}_{parsed_params['distribution_type'].capitalize()}"
            parsed_params_with_label = parsed_params.copy()
            parsed_params_with_label['algo_dist_label'] = algo_dist_label
            parsed_params_with_label['env_id_short'] = parsed_params['env_id'].split('-v')[0] # For cleaner titles

            metrics_file = run_folder / "metrics.json"
            timings_file = run_folder / "timings.jsonl"

            if metrics_file.exists():
                lc_df = load_learning_curve_data(metrics_file, parsed_params_with_label)
                if not lc_df.empty:
                    all_learning_curves_list.append(lc_df)
            # else:
                # print(f"  Warning: metrics.json not found in {run_folder}")

            if timings_file.exists():
                timings_df = load_timings_data(timings_file, parsed_params_with_label)
                if not timings_df.empty:
                    all_timings_list.append(timings_df)
            # else:
                # print(f"  Note: timings.jsonl not found in {run_folder}")
    
    df_learning_curves = pd.concat(all_learning_curves_list, ignore_index=True) if all_learning_curves_list else pd.DataFrame()
    df_timings = pd.concat(all_timings_list, ignore_index=True) if all_timings_list else pd.DataFrame()
    
    return df_learning_curves, df_timings

def generate_distribution_comparison_plots(
    df_learning_curves: pd.DataFrame,
    df_timings: pd.DataFrame,
    output_dir: Path
):
    """Generates plots comparing algorithm and distribution types."""
    if df_learning_curves.empty and df_timings.empty:
        print("No data available for distribution comparison plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Assuming all runs are for the same environment for a given set of plots.
    # If multiple envs were in BASE_SWEEPS_DIR, this will plot them together or pick one.
    # For cleaner plots, ensure BASE_SWEEPS_DIR points to runs of a single env, or adapt to loop over env_id.
    env_id_str = "MultiEnv"
    if not df_learning_curves.empty and 'env_id_short' in df_learning_curves:
        unique_envs = df_learning_curves['env_id_short'].unique()
        if len(unique_envs) == 1:
            env_id_str = unique_envs[0]
        else:
            print(f"Warning: Multiple environments found: {unique_envs}. Plots will aggregate or be labeled 'MultiEnv'.")


    print(f"\n--- Generating Distribution Comparison Plots (Env: {env_id_str}) ---")

    # Reward Plot
    if not df_learning_curves.empty:
        plt.figure(figsize=(14, 8))
        sns.lineplot(
            data=df_learning_curves,
            x='step',
            y='avg_episodic_reward_smoothed', # Use the smoothed reward
            hue='algo_dist_label',
            errorbar='sd',  # Show standard deviation across all runs in each group
            palette='tab10', # A good colormap for distinct lines
            legend='full'
        )
        plt.title(f"Algorithm & Distribution Performance Comparison ({env_id_str})\n(Aggregated over all Hyperparameters & Seeds)")
        plt.xlabel("Training Steps")
        reward_y_label = "Avg Episodic Reward"
        if REWARD_SMOOTHING_WINDOW > 1:
            reward_y_label += f" (Smoothed, Window {REWARD_SMOOTHING_WINDOW})"
        plt.ylabel(reward_y_label)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / f"{env_id_str}_dist_rewards_comparison.png")
        plt.close()
        print(f"  Saved rewards distribution comparison plot for {env_id_str}.")

    # Timing Plots
    if not df_timings.empty:
        unique_timing_keys = df_timings['timing_key'].unique()
        for timing_key in unique_timing_keys:
            timing_key_df = df_timings[df_timings['timing_key'] == timing_key]
            if not timing_key_df.empty:
                plt.figure(figsize=(14, 8))
                sns.lineplot(
                    data=timing_key_df,
                    x='step',
                    y='avg_ms',
                    hue='algo_dist_label',
                    errorbar='sd',
                    palette='tab10',
                    legend='full'
                )
                plt.title(f"Timing Comparison: {timing_key} ({env_id_str})\n(Aggregated over all Hyperparameters & Seeds)")
                plt.xlabel("Training Steps")
                plt.ylabel("Average Time (ms)")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(output_dir / f"{env_id_str}_dist_timing_{timing_key}_comparison.png")
                plt.close()
        print(f"  Saved timing distribution comparison plots for {env_id_str}.")

# --- Main Execution ---
def main():
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure main output dir exists

    if not BASE_SWEEPS_DIR.exists():
        print(f"Error: Base sweeps directory '{BASE_SWEEPS_DIR}' not found.")
        return

    df_learning_curves, df_timings = discover_and_load_all_distribution_comparison_data(BASE_SWEEPS_DIR)

    if df_learning_curves.empty and df_timings.empty:
        print("No data loaded from any sweep group. Cannot generate distribution comparison plots.")
        return

    generate_distribution_comparison_plots(df_learning_curves, df_timings, PLOTS_OUTPUT_DIR)

    print(f"\nDistribution comparison plotting complete. Check the '{PLOTS_OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()