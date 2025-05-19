import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# --- Configuration ---
BASE_SWEEPS_DIR = Path("sweeps")
PLOTS_OUTPUT_DIR = Path("analysis_plots")
LATEX_OUTPUT_DIR = Path("latex_tables")

REWARD_SMOOTHING_WINDOW = 5
LAST_N_REWARDS_FOR_FINAL_PERF = 10 # For heatmap and LaTeX table values

FOLDER_PATTERN = re.compile(
    r"^(?P<env_id>[a-zA-Z0-9_.-]+(?:-v\d+)?)"
    r"_(?P<algo>ppo|grpo)"
    r"_seed(?P<seed>\d+)"
    r"_ent(?P<entropy_coef>[\d.eE+-]+)"
    r"_lr(?P<lr>[\d.eE+-]+)"
    r"_(?P<distribution_type>normal|beta)"
    r"(?:_g(?P<group_size>\d+))?$"
)

TIMING_KEYS_OF_INTEREST = [
    "rollout_phase", "update_phase", "actor_pass", "critic_pass",
    "actor_mlp", "ref_actor_mlp", "backward_pass", "optimizer_step",
    "cnn_feature_pass", "cnn_feature", "ref_cnn_feature", # Added from GRPO
    "actor_head", "ref_actor_head" # Added from GRPO
]

# --- Helper Functions ---
def parse_folder_name(folder_name_str: str):
    match = FOLDER_PATTERN.match(folder_name_str)
    if match:
        params = match.groupdict()
        try:
            params['seed'] = int(params['seed'])
            params['entropy_coef'] = float(params['entropy_coef'])
            params['lr'] = float(params['lr'])
            if params['group_size'] is not None:
                params['group_size'] = int(params['group_size'])
            else:
                # Ensure group_size is float for PPO as well for consistent dtype with GRPO later
                params['group_size'] = np.nan
            return params
        except ValueError: return None
    return None

def load_metrics_for_final_performance(metrics_file: Path, parsed_params: dict):
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        rewards = metrics.get("avg_episodic_reward", [])
        # Ensure rewards are numeric, coercing errors, then drop NaNs for calculation
        valid_rewards = pd.to_numeric(pd.Series(rewards), errors='coerce').dropna().tolist()
        if not valid_rewards: return None
        last_n = valid_rewards[-LAST_N_REWARDS_FOR_FINAL_PERF:]
        if not last_n: return None # Should not happen if valid_rewards is not empty
        final_perf = np.mean(last_n)
        entry = parsed_params.copy()
        entry['final_performance'] = float(final_perf) # Ensure it's a Python float
        return entry
    except Exception: return None

def load_learning_curve_data(metrics_file: Path, parsed_params: dict):
    data_list = []
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        steps = metrics.get("steps", [])
        rewards = metrics.get("avg_episodic_reward", [])

        if not steps or len(steps) != len(rewards):
            # print(f"Warning: Steps/rewards mismatch or empty in {metrics_file}. Skipping.")
            return pd.DataFrame()

        for step, reward in zip(steps, rewards):
            # We will handle non-numeric rewards robustly when creating the DataFrame
            entry = parsed_params.copy()
            entry['step'] = step # step should be numeric from JSON
            entry['avg_episodic_reward'] = reward # Keep as is, will be coerced later
            data_list.append(entry)

        if not data_list: # Should not happen if steps/rewards were not empty and matched length
            return pd.DataFrame()

        df = pd.DataFrame(data_list)

        # Ensure crucial columns are numeric. Coerce errors to NaN.
        df['step'] = pd.to_numeric(df['step'], errors='coerce')
        df['avg_episodic_reward'] = pd.to_numeric(df['avg_episodic_reward'], errors='coerce')

        # Drop rows where step or initial reward is NaN, as they can't be plotted meaningfully
        df.dropna(subset=['step', 'avg_episodic_reward'], inplace=True)

        if df.empty: # If all data was bad
            # print(f"Warning: All data became NaN after numeric conversion for {metrics_file}. Skipping.")
            return pd.DataFrame()

        if REWARD_SMOOTHING_WINDOW > 1:
            # Only smooth if there are non-NaN rewards to smooth
            if not df['avg_episodic_reward'].isnull().all():
                run_id_cols = [col for col in parsed_params.keys() if col not in ['step', 'avg_episodic_reward']]
                # Ensure all run_id_cols exist in df, important if parsed_params had extra keys not in data_list construction
                run_id_cols = [col for col in run_id_cols if col in df.columns]

                if not run_id_cols: # Safety: if all id cols were somehow removed
                     df['avg_episodic_reward'] = df['avg_episodic_reward'].rolling(window=REWARD_SMOOTHING_WINDOW, min_periods=1, center=True).mean()
                else:
                    df['avg_episodic_reward'] = df.groupby(run_id_cols, dropna=False)['avg_episodic_reward']\
                                                .transform(lambda x: x.rolling(window=REWARD_SMOOTHING_WINDOW, min_periods=1, center=True).mean())
            # else:
            #     print(f"Info: Skipping smoothing for {metrics_file} as all rewards are NaN after numeric conversion.")
        return df
    except Exception as e:
        # print(f"Error loading learning curve data from {metrics_file}: {e}")
        return pd.DataFrame()

def load_timings_data(timings_file: Path, parsed_params: dict):
    data_list = []
    try:
        with open(timings_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    step = log_entry.get("step")
                    if step is None: continue
                    for key in TIMING_KEYS_OF_INTEREST:
                        if key in log_entry and isinstance(log_entry[key], dict):
                            avg_ms = log_entry[key].get("avg_ms")
                            if avg_ms is not None:
                                entry = parsed_params.copy()
                                entry['step'] = float(step) # Ensure numeric
                                entry['timing_key'] = key
                                entry['avg_ms'] = float(avg_ms)
                                data_list.append(entry)
                except json.JSONDecodeError: continue
        return pd.DataFrame(data_list)
    except Exception: return pd.DataFrame()

def discover_and_load_all_data(base_sweeps_dir: Path):
    all_final_perf_list, all_learning_curves_list, all_timings_list = [], [], []
    for sweep_group_path in sorted(base_sweeps_dir.iterdir()):
        if not sweep_group_path.is_dir(): continue
        print(f"\nProcessing sweep group for all data: {sweep_group_path.name}")
        for run_folder in sorted(sweep_group_path.iterdir()):
            if not run_folder.is_dir(): continue
            parsed_params = parse_folder_name(run_folder.name)
            if not parsed_params:
                # print(f"  Skipping folder due to parsing failure: {run_folder.name}")
                continue
            parsed_params['sweep_group'] = sweep_group_path.name

            metrics_file = run_folder / "metrics.json"
            timings_file = run_folder / "timings.jsonl"
            if metrics_file.exists():
                final_perf_entry = load_metrics_for_final_performance(metrics_file, parsed_params)
                if final_perf_entry: all_final_perf_list.append(final_perf_entry)
                lc_df = load_learning_curve_data(metrics_file, parsed_params)
                if not lc_df.empty: all_learning_curves_list.append(lc_df)
            if timings_file.exists():
                timings_df = load_timings_data(timings_file, parsed_params)
                if not timings_df.empty: all_timings_list.append(timings_df)

    df_final_perf = pd.DataFrame(all_final_perf_list) if all_final_perf_list else pd.DataFrame()
    df_learning_curves = pd.concat(all_learning_curves_list, ignore_index=True) if all_learning_curves_list else pd.DataFrame()
    df_timings = pd.concat(all_timings_list, ignore_index=True) if all_timings_list else pd.DataFrame()
    return df_final_perf, df_learning_curves, df_timings

# --- Plotting Functions ---
def generate_heatmaps(df_final_perf: pd.DataFrame, output_base_path: Path):
    if df_final_perf.empty: return
    heatmap_plot_dir = output_base_path / "Hyperparameter_Heatmaps"
    heatmap_plot_dir.mkdir(parents=True, exist_ok=True)
    # Ensure types before groupby
    df_final_perf['algo'] = df_final_perf['algo'].astype(str)
    df_final_perf['distribution_type'] = df_final_perf['distribution_type'].astype(str)
    if 'group_size' in df_final_perf.columns: # Ensure group_size is float for consistent handling of NaN
         df_final_perf['group_size'] = pd.to_numeric(df_final_perf['group_size'], errors='coerce')


    grouping_keys = ['algo', 'distribution_type']
    # Only add group_size to grouping if it's a meaningful column (e.g., not all NaN)
    if 'group_size' in df_final_perf.columns and df_final_perf['group_size'].notna().any():
        grouping_keys.append('group_size')


    for name_tuple, group in df_final_perf.groupby(grouping_keys, dropna=False):
        current_algo = name_tuple[0]
        current_dist = name_tuple[1]
        current_gs = name_tuple[2] if len(name_tuple) > 2 and 'group_size' in grouping_keys else np.nan


        if current_algo == 'ppo' and pd.notna(current_gs): continue
        if current_algo == 'grpo' and pd.isna(current_gs) and 'group_size' in grouping_keys : continue


        title_parts = [f"Algo: {current_algo.upper()}", f"Dist: {current_dist.capitalize()}"]
        filename_parts = [current_algo, current_dist]
        if pd.notna(current_gs):
            title_parts.append(f"GroupSize: {int(current_gs)}")
            filename_parts.append(f"g{int(current_gs)}")

        if group.empty or group['lr'].nunique() < 2 or group['entropy_coef'].nunique() < 2: continue
        
        # Ensure lr and entropy_coef are numeric before pivot
        group['lr'] = pd.to_numeric(group['lr'], errors='coerce')
        group['entropy_coef'] = pd.to_numeric(group['entropy_coef'], errors='coerce')
        group.dropna(subset=['lr', 'entropy_coef'], inplace=True)
        if group.empty: continue

        agg_perf = group.groupby(['lr', 'entropy_coef'])['final_performance'].agg(['mean', 'std', 'count']).reset_index()
        if agg_perf.empty: continue

        try:
            mean_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='mean')
            std_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='std')
            count_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='count')
        except Exception: continue # Could be due to non-unique index/columns after coerce if not careful

        if mean_pivot.empty: continue

        annot_data = mean_pivot.copy().astype(object)
        for r_idx in mean_pivot.index:
            for c_idx in mean_pivot.columns:
                mean_val = mean_pivot.loc[r_idx, c_idx] if pd.notna(mean_pivot.loc[r_idx, c_idx]) else np.nan
                std_val = std_pivot.loc[r_idx, c_idx] if pd.notna(std_pivot.loc[r_idx, c_idx]) else np.nan
                count_val = count_pivot.loc[r_idx, c_idx] if pd.notna(count_pivot.loc[r_idx, c_idx]) else np.nan
                
                if pd.notna(mean_val) and pd.notna(count_val): # Std can be NaN for count=1
                    annot_str = f"{mean_val:.1f}"
                    if pd.notna(std_val) and count_val > 1 : # Only show std if count > 1
                        annot_str += f"\n±{std_val:.1f}"
                    annot_str += f"\n(n={int(count_val)})"
                    annot_data.loc[r_idx, c_idx] = annot_str
                else:
                    annot_data.loc[r_idx, c_idx] = ""
        plt.figure(figsize=(max(8, len(mean_pivot.columns)*2), max(6, len(mean_pivot.index)*1)))
        sns.heatmap(mean_pivot, annot=annot_data, fmt='s', cmap="viridis", linewidths=.5, cbar_kws={'label': f'Mean Final Reward (Last {LAST_N_REWARDS_FOR_FINAL_PERF} Logs)'})
        plt.title(f"Performance Heatmap: LR vs. EC\n" + ", ".join(title_parts), fontsize=14)
        plt.xlabel("Entropy Coefficient", fontsize=12); plt.ylabel("Learning Rate", fontsize=12)
        plt.tight_layout()
        plt.savefig(heatmap_plot_dir / ("_".join(filename_parts) + "_LR_vs_EC_heatmap.png")); plt.close()
        print(f"  Saved heatmap: {heatmap_plot_dir / ('_'.join(filename_parts) + '_LR_vs_EC_heatmap.png')}")

def _plot_single_parameter_effect_curves(metrics_df_subset, timings_df_subset, algo, dist_type, varying_param, fixed_values, sweep_group_name, output_base_path):
    title_suffix_parts = [f"{k}={v}" for k, v in fixed_values.items() if pd.notna(v) or k == 'group_size'] # show group_size=nan
    if 'group_size' in fixed_values and pd.isna(fixed_values['group_size']) and algo == 'ppo':
        title_suffix_parts = [p for p in title_suffix_parts if not p.startswith('group_size=')] # Don't show group_size=nan for PPO title
    title_suffix = ", ".join(title_suffix_parts)

    plot_filename_suffix_parts = []
    for k, v in fixed_values.items():
        if pd.notna(v):
            plot_filename_suffix_parts.append(f"{k.replace('_','')[0:2]}{v}")
        elif k == 'group_size' and pd.isna(v) and algo == 'grpo': # for GRPO, a fixed NaN group_size might be a specific case
             plot_filename_suffix_parts.append(f"{k.replace('_','')[0:2]}nan")
    plot_filename_suffix = "_".join(plot_filename_suffix_parts).replace(".","p")


    plot_dir = output_base_path / sweep_group_name / f"{algo}_{dist_type}" / f"Vary_{varying_param}"
    if plot_filename_suffix: plot_dir = plot_dir / plot_filename_suffix # Use modified suffix
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_df_subset.empty and varying_param in metrics_df_subset.columns and metrics_df_subset[varying_param].nunique() > 0:
        # Ensure x and y are numeric before plotting
        metrics_df_subset['step'] = pd.to_numeric(metrics_df_subset['step'], errors='coerce')
        metrics_df_subset['avg_episodic_reward'] = pd.to_numeric(metrics_df_subset['avg_episodic_reward'], errors='coerce')
        
        # Drop rows if key plotting variables are NaN AFTER coercion
        plot_data = metrics_df_subset.dropna(subset=['step', 'avg_episodic_reward', varying_param])

        if not plot_data.empty:
            plt.figure(figsize=(12, 7));
            sns.lineplot(data=plot_data, x='step', y='avg_episodic_reward', hue=varying_param, errorbar='sd', palette='viridis', legend='full')
            plot_title = f"Reward ({algo.upper()}-{dist_type.capitalize()}) Varying: {varying_param}" + (f"\nFixed: {title_suffix}" if title_suffix else "")
            plt.title(plot_title); plt.xlabel("Training Steps"); plt.ylabel(f"Avg Reward (Smooth {REWARD_SMOOTHING_WINDOW})")
            plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(plot_dir / f"rewards_vary_{varying_param}.png"); plt.close()
        # else:
            # print(f"  Skipping reward plot for {algo}-{dist_type}, Varying: {varying_param}, Fixed: {fixed_values} due to no valid data after NaN drop.")

    if not timings_df_subset.empty and varying_param in timings_df_subset.columns and timings_df_subset[varying_param].nunique() > 0:
        timings_df_subset['step'] = pd.to_numeric(timings_df_subset['step'], errors='coerce')
        timings_df_subset['avg_ms'] = pd.to_numeric(timings_df_subset['avg_ms'], errors='coerce')
        
        for timing_key_to_plot in timings_df_subset['timing_key'].unique():
            timing_key_df = timings_df_subset[timings_df_subset['timing_key'] == timing_key_to_plot]
            plot_data_timing = timing_key_df.dropna(subset=['step', 'avg_ms', varying_param])
            if not plot_data_timing.empty:
                plt.figure(figsize=(12, 7)); sns.lineplot(data=plot_data_timing, x='step', y='avg_ms', hue=varying_param, errorbar='sd', palette='viridis', legend='full')
                plot_title = f"{timing_key_to_plot} ({algo.upper()}-{dist_type.capitalize()}) Varying: {varying_param}" + (f"\nFixed: {title_suffix}" if title_suffix else "")
                plt.title(plot_title); plt.xlabel("Training Steps"); plt.ylabel("Average Time (ms)")
                plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(plot_dir / f"timing_{timing_key_to_plot}_vary_{varying_param}.png"); plt.close()


def iterate_and_plot_parameter_effects(metrics_df, timings_df, current_algo, current_dist_type, varying_param, fixed_params_options, sweep_group_name, output_base_path):
    fixed_param_names = [k for k in fixed_params_options.keys() if k != varying_param and fixed_params_options[k]]

    if not fixed_param_names: # Only varying_param, no other params to fix
        _plot_single_parameter_effect_curves(metrics_df, timings_df, current_algo, current_dist_type, varying_param, {}, sweep_group_name, output_base_path)
        return

    value_lists = [fixed_params_options[name] for name in fixed_param_names]
    for specific_fixed_values_tuple in itertools.product(*value_lists):
        fixed_values = dict(zip(fixed_param_names, specific_fixed_values_tuple))
        
        metrics_subset, timings_subset = metrics_df.copy(), timings_df.copy() # Start with the group data for this algo/dist

        valid_subset = True
        for p_name, p_val in fixed_values.items():
            if not metrics_subset.empty and p_name in metrics_subset.columns:
                if pd.isna(p_val): # Handle fixing to NaN (e.g., group_size for PPO)
                    metrics_subset = metrics_subset[metrics_subset[p_name].isna()]
                else:
                    # Ensure comparison is with correct type if p_val is from unique list
                    # metrics_subset[p_name] should already be numeric if loaded correctly
                    metrics_subset = metrics_subset[metrics_subset[p_name] == p_val]
            
            if not timings_subset.empty and p_name in timings_subset.columns:
                if pd.isna(p_val):
                    timings_subset = timings_subset[timings_subset[p_name].isna()]
                else:
                    timings_subset = timings_subset[timings_subset[p_name] == p_val]

            # Check if filtering made the subset empty unnecessarily
            # (This check was too aggressive, subset can become empty legitimately)
            # if metrics_subset.empty and (timings_subset.empty if not timings_df.empty else True) and p_name in metrics_df.columns:
            #    valid_subset = False; break
        
        # Only plot if there's actually data left after filtering for fixed values
        if not metrics_subset.empty or (not timings_subset.empty and not timings_df.empty):
            _plot_single_parameter_effect_curves(metrics_subset, timings_subset, current_algo, current_dist_type, varying_param, fixed_values, sweep_group_name, output_base_path)


def generate_parameter_comparison_plots_for_sweep_group(metrics_df_sg, timings_df_sg, sweep_group_name, output_base_path):
    if metrics_df_sg.empty and timings_df_sg.empty: return

    # Determine grouping df based on availability
    df_for_grouping = metrics_df_sg if not metrics_df_sg.empty else timings_df_sg
    if df_for_grouping.empty: return # Should not happen if first check passed

    df_for_grouping['algo'] = df_for_grouping['algo'].astype(str)
    df_for_grouping['distribution_type'] = df_for_grouping['distribution_type'].astype(str)
    if 'group_size' in df_for_grouping.columns: # Ensure float type for consistent NaN handling
        df_for_grouping['group_size'] = pd.to_numeric(df_for_grouping['group_size'], errors='coerce')


    for (algo, dist_type), group_df_for_iter in df_for_grouping.groupby(['algo', 'distribution_type']):
        print(f"  Generating param plots for: {algo.upper()}-{dist_type.capitalize()} in {sweep_group_name}")

        # Filter original DFs for the current algo/dist_type combination
        current_metrics_df = metrics_df_sg[(metrics_df_sg['algo'] == algo) & (metrics_df_sg['distribution_type'] == dist_type)] if not metrics_df_sg.empty else pd.DataFrame()
        current_timings_df = timings_df_sg[(timings_df_sg['algo'] == algo) & (timings_df_sg['distribution_type'] == dist_type)] if not timings_df_sg.empty else pd.DataFrame()

        # Use group_df_for_iter (which is already filtered for algo/dist) to get unique param values
        # Ensure params are numeric before finding uniques
        unique_lrs = sorted(pd.to_numeric(group_df_for_iter['lr'], errors='coerce').dropna().unique()) if 'lr' in group_df_for_iter else []
        unique_ecs = sorted(pd.to_numeric(group_df_for_iter['entropy_coef'], errors='coerce').dropna().unique()) if 'entropy_coef' in group_df_for_iter else []
        
        unique_gss = []
        if algo == 'grpo' and 'group_size' in group_df_for_iter:
            unique_gss = sorted(pd.to_numeric(group_df_for_iter['group_size'], errors='coerce').dropna().unique())
        
        # Parameter options for fixing
        param_options = {}
        if unique_lrs: param_options['lr'] = unique_lrs
        if unique_ecs: param_options['entropy_coef'] = unique_ecs
        if algo == 'grpo' and unique_gss:
            param_options['group_size'] = unique_gss
        elif algo == 'ppo': # For PPO, group_size is fixed to NaN
            param_options['group_size'] = [np.nan]


        if unique_lrs:
            fixed_opts_for_lr_vary = {k: v for k, v in param_options.items() if k != 'lr'}
            iterate_and_plot_parameter_effects(current_metrics_df, current_timings_df, algo, dist_type, 'lr', fixed_opts_for_lr_vary, sweep_group_name, output_base_path)
        if unique_ecs:
            fixed_opts_for_ec_vary = {k: v for k, v in param_options.items() if k != 'entropy_coef'}
            iterate_and_plot_parameter_effects(current_metrics_df, current_timings_df, algo, dist_type, 'entropy_coef', fixed_opts_for_ec_vary, sweep_group_name, output_base_path)
        if algo == 'grpo' and unique_gss:
            fixed_opts_for_gs_vary = {k: v for k, v in param_options.items() if k != 'group_size'}
            iterate_and_plot_parameter_effects(current_metrics_df, current_timings_df, algo, dist_type, 'group_size', fixed_opts_for_gs_vary, sweep_group_name, output_base_path)


def generate_overall_algorithm_comparison_plots(all_metrics_df, all_timings_df, output_base_path):
    if all_metrics_df.empty and all_timings_df.empty: return

    overall_plot_dir = output_base_path / "Overall_Algorithm_Comparison"
    overall_plot_dir.mkdir(parents=True, exist_ok=True)

    env_id = "UnknownEnv"
    # Determine env_id from whichever df is available
    source_df_for_env = all_metrics_df if not all_metrics_df.empty else all_timings_df
    if not source_df_for_env.empty and 'env_id' in source_df_for_env and source_df_for_env['env_id'].nunique() > 0 :
        env_id = source_df_for_env['env_id'].unique()[0]

    print(f"\n--- Generating Overall Algorithm Comparison Plots (Env: {env_id}) ---")

    if not all_metrics_df.empty:
        # Create combined algo_dist column for hue
        df_plot_metrics = all_metrics_df.copy()
        df_plot_metrics['algo_dist'] = df_plot_metrics['algo'].astype(str) + "_" + df_plot_metrics['distribution_type'].astype(str)
        
        # Ensure key columns are numeric
        df_plot_metrics['step'] = pd.to_numeric(df_plot_metrics['step'], errors='coerce')
        df_plot_metrics['avg_episodic_reward'] = pd.to_numeric(df_plot_metrics['avg_episodic_reward'], errors='coerce')
        df_plot_metrics.dropna(subset=['step', 'avg_episodic_reward'], inplace=True)

        if not df_plot_metrics.empty:
            plt.figure(figsize=(14, 8));
            sns.lineplot(data=df_plot_metrics, x='step', y='avg_episodic_reward', hue='algo_dist', errorbar='sd', palette='tab10', legend='full')
            plt.title(f"Overall Algorithm Performance Comparison ({env_id})\n(Smoothed Reward, Aggregated over all Hyperparameters & Seeds)")
            plt.xlabel("Training Steps"); plt.ylabel(f"Avg Episodic Reward (Smooth {REWARD_SMOOTHING_WINDOW})")
            plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(overall_plot_dir / f"{env_id}_overall_rewards.png"); plt.close()
            print(f"  Saved overall rewards plot for {env_id}.")
        # else:
        #    print(f"  Skipping overall rewards plot for {env_id} due to no valid data after NaN drop.")


    if not all_timings_df.empty:
        df_plot_timings = all_timings_df.copy()
        df_plot_timings['algo_dist'] = df_plot_timings['algo'].astype(str) + "_" + df_plot_timings['distribution_type'].astype(str)
        
        df_plot_timings['step'] = pd.to_numeric(df_plot_timings['step'], errors='coerce')
        df_plot_timings['avg_ms'] = pd.to_numeric(df_plot_timings['avg_ms'], errors='coerce')

        for timing_key in df_plot_timings['timing_key'].unique():
            timing_key_df = df_plot_timings[df_plot_timings['timing_key'] == timing_key]
            plot_data_timing = timing_key_df.dropna(subset=['step', 'avg_ms']) # Drop if step or avg_ms is NaN

            if not plot_data_timing.empty:
                plt.figure(figsize=(14, 8));
                sns.lineplot(data=plot_data_timing, x='step', y='avg_ms', hue='algo_dist', errorbar='sd', palette='tab10', legend='full')
                plt.title(f"Overall Timing: {timing_key} ({env_id})\n(Aggregated over all Hyperparameters & Seeds)")
                plt.xlabel("Training Steps"); plt.ylabel("Average Time (ms)")
                plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(overall_plot_dir / f"{env_id}_overall_timing_{timing_key}.png"); plt.close()
        print(f"  Saved overall timing plots for {env_id}.")


# --- LaTeX Table Generation ---
def generate_latex_tables(df_final_perf: pd.DataFrame, output_dir: Path):
    if df_final_perf.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)

    df_final_perf_copy = df_final_perf.copy() # Work on a copy
    df_final_perf_copy['algo'] = df_final_perf_copy['algo'].astype(str)
    df_final_perf_copy['distribution_type'] = df_final_perf_copy['distribution_type'].astype(str)
    if 'group_size' in df_final_perf_copy.columns:
        df_final_perf_copy['group_size'] = pd.to_numeric(df_final_perf_copy['group_size'], errors='coerce')

    grouping_keys = ['algo', 'distribution_type']
    if 'group_size' in df_final_perf_copy.columns and df_final_perf_copy['group_size'].notna().any():
        grouping_keys.append('group_size')


    for name_tuple, group in df_final_perf_copy.groupby(grouping_keys, dropna=False):
        current_algo = name_tuple[0]
        current_dist = name_tuple[1]
        current_gs = name_tuple[2] if len(name_tuple) > 2 and 'group_size' in grouping_keys else np.nan


        if current_algo == 'ppo' and pd.notna(current_gs): continue
        if current_algo == 'grpo' and pd.isna(current_gs) and 'group_size' in grouping_keys: continue
        
        table_caption_parts = [f"{current_algo.upper()}", f"{current_dist.capitalize()}"]
        filename_parts = [current_algo, current_dist]
        if pd.notna(current_gs):
            table_caption_parts.append(f"G={int(current_gs)}")
            filename_parts.append(f"g{int(current_gs)}")
        
        if group.empty: continue
        # Ensure lr and ec are numeric before groupby and pivot
        group['lr'] = pd.to_numeric(group['lr'], errors='coerce')
        group['entropy_coef'] = pd.to_numeric(group['entropy_coef'], errors='coerce')
        group.dropna(subset=['lr', 'entropy_coef', 'final_performance'], inplace=True) # final_performance must be valid
        
        if group.empty or group['lr'].nunique() < 1 or group['entropy_coef'].nunique() < 1: continue

        agg_perf = group.groupby(['lr', 'entropy_coef'])['final_performance'].agg(['mean', 'std', 'count']).reset_index()
        if agg_perf.empty: continue
        
        try:
            mean_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='mean')
            std_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='std')
            # count_pivot for std dev condition (already available in agg_perf)
        except Exception as e:
            # print(f"Could not pivot for LaTeX table {filename_parts}: {e}")
            continue
        if mean_pivot.empty: continue

        lrs = sorted(mean_pivot.index.unique())
        ecs = sorted(mean_pivot.columns.unique())
        
        def format_header_val(v):
            return f"{v:.0e}" if abs(v) < 1e-2 and v != 0 else f"{v:.4g}".rstrip('0').rstrip('.')

        
        col_headers = ["LR \\ EC"] + [format_header_val(ec) for ec in ecs]
        
        latex_str = "\\begin{table}[H]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{Final Performance (Mean $\\pm$ Std. Dev.) for {', '.join(table_caption_parts)}. Averaged over last {LAST_N_REWARDS_FOR_FINAL_PERF} reward logs.}}\n"
        latex_str += "\\label{tab:" + "_".join(filename_parts).lower() + "}\n"
        latex_str += "\\resizebox{\\textwidth}{!}{%\n"
        latex_str += "\\begin{tabular}{l|" + "c" * len(ecs) + "}\n"
        latex_str += "\\toprule\n"
        latex_str += " & ".join(col_headers) + " \\\\\n"
        latex_str += "\\midrule\n"
        
        for lr_val in lrs:
            row_data = [format_header_val(lr_val)]
            for ec_val in ecs:
                # Get corresponding count to decide on std dev display
                count_val_series = agg_perf[(agg_perf['lr'] == lr_val) & (agg_perf['entropy_coef'] == ec_val)]['count']
                count_val = count_val_series.iloc[0] if not count_val_series.empty else 0

                mean_val = mean_pivot.loc[lr_val, ec_val] if (lr_val, ec_val) in mean_pivot.stack().index else np.nan
                std_val = std_pivot.loc[lr_val, ec_val] if (lr_val, ec_val) in std_pivot.stack().index else np.nan
                
                if pd.notna(mean_val):
                    cell_str = f"${mean_val:.1f}$"
                    if pd.notna(std_val) and count_val > 1: # Only show std if count > 1
                        cell_str = f"${mean_val:.1f} \\pm {std_val:.1f}$"
                    row_data.append(cell_str)
                else:
                    row_data.append("-")
            latex_str += " & ".join(row_data) + " \\\\\n"
            
        latex_str += "\\bottomrule\n"
        latex_str += "\\end{tabular}}\n"
        latex_str += "\\end{table}\n"
        
        tex_filename = "_".join(filename_parts) + "_perf_table.tex"
        with open(output_dir / tex_filename, 'w') as f:
            f.write(latex_str)
        print(f"  Saved LaTeX table: {output_dir / tex_filename}")

# --- Main Execution ---
def main():
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_SWEEPS_DIR.exists():
        print(f"Error: Base sweeps directory '{BASE_SWEEPS_DIR}' not found.")
        return

    df_final_perf, df_learning_curves, df_timings = discover_and_load_all_data(BASE_SWEEPS_DIR)

    if df_final_perf.empty and df_learning_curves.empty and df_timings.empty:
        print("No data loaded from any sweep group. Exiting.")
        return

    if not df_final_perf.empty:
        print("\n--- Generating Hyperparameter Heatmaps ---")
        generate_heatmaps(df_final_perf, PLOTS_OUTPUT_DIR)
    
    if not df_final_perf.empty:
        print("\n--- Generating LaTeX Performance Tables ---")
        generate_latex_tables(df_final_perf, LATEX_OUTPUT_DIR)

    print("\n--- Generating Parameter-Specific Learning Curve Plots (per sweep group) ---")
    # Use unique sweep groups from the loaded data
    # Ensure 'sweep_group' column exists
    all_dfs = [df for df in [df_learning_curves, df_timings, df_final_perf] if not df.empty and 'sweep_group' in df.columns]
    if not all_dfs:
        print("No data with 'sweep_group' information found to generate per-sweep plots.")
    else:
        # Consolidate sweep groups from all available dataframes
        all_sweep_groups = pd.Series(dtype=str)
        if not df_learning_curves.empty and 'sweep_group' in df_learning_curves.columns:
            all_sweep_groups = pd.concat([all_sweep_groups, df_learning_curves['sweep_group']]).unique()
        elif not df_timings.empty and 'sweep_group' in df_timings.columns: # Use timings if learning curves is empty
            all_sweep_groups = pd.concat([all_sweep_groups, df_timings['sweep_group']]).unique()
        else: # Fallback to final_perf if others are empty or lack sweep_group
             all_sweep_groups = pd.concat([all_sweep_groups, df_final_perf['sweep_group']]).unique()


        for sg_name in sorted(all_sweep_groups):
            print(f" Processing sweep group for plots: {sg_name}")
            metrics_df_sg = df_learning_curves[df_learning_curves['sweep_group'] == sg_name] if not df_learning_curves.empty and 'sweep_group' in df_learning_curves.columns else pd.DataFrame()
            timings_df_sg = df_timings[df_timings['sweep_group'] == sg_name] if not df_timings.empty and 'sweep_group' in df_timings.columns else pd.DataFrame()
            
            if not metrics_df_sg.empty or not timings_df_sg.empty:
                 generate_parameter_comparison_plots_for_sweep_group(metrics_df_sg, timings_df_sg, sg_name, PLOTS_OUTPUT_DIR)
            # else:
                # print(f"  No learning curve or timing data for sweep group: {sg_name}")
    
    if not df_learning_curves.empty or not df_timings.empty:
        print("\n--- Generating Overall Algorithm Comparison Plots ---") # This print was missing before
        generate_overall_algorithm_comparison_plots(df_learning_curves, df_timings, PLOTS_OUTPUT_DIR)

    print(f"\nAll processing complete. Check '{PLOTS_OUTPUT_DIR}' and '{LATEX_OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()