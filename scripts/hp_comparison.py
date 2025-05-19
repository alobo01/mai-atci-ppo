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
                params['group_size'] = np.nan
            return params
        except ValueError: return None
    return None

def load_metrics_for_final_performance(metrics_file: Path, parsed_params: dict):
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        rewards = metrics.get("avg_episodic_reward", [])
        valid_rewards = [r for r in rewards if r is not None]
        if not valid_rewards: return None
        last_n = valid_rewards[-LAST_N_REWARDS_FOR_FINAL_PERF:]
        if not last_n: return None
        final_perf = np.mean(last_n)
        entry = parsed_params.copy()
        entry['final_performance'] = float(final_perf)
        return entry
    except Exception: return None

def load_learning_curve_data(metrics_file: Path, parsed_params: dict):
    data_list = []
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        steps = metrics.get("steps", [])
        rewards = metrics.get("avg_episodic_reward", [])
        if not steps or len(steps) != len(rewards): return pd.DataFrame()
        for step, reward in zip(steps, rewards):
            if reward is not None:
                entry = parsed_params.copy()
                entry['step'] = step
                entry['avg_episodic_reward'] = float(reward)
                data_list.append(entry)
        df = pd.DataFrame(data_list)
        if REWARD_SMOOTHING_WINDOW > 1 and not df.empty:
            run_id_cols = [col for col in parsed_params.keys() if col not in ['step', 'avg_episodic_reward']]
            if not df.empty:
                 df['avg_episodic_reward'] = df.groupby(run_id_cols)['avg_episodic_reward']\
                                               .transform(lambda x: x.rolling(window=REWARD_SMOOTHING_WINDOW, min_periods=1, center=True).mean())
        return df
    except Exception: return pd.DataFrame()

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
                                entry['step'] = step
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
            if not parsed_params: continue
            parsed_params['sweep_group'] = sweep_group_path.name # Add sweep group info

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
    df_final_perf = pd.DataFrame(all_final_perf_list)
    df_learning_curves = pd.concat(all_learning_curves_list, ignore_index=True) if all_learning_curves_list else pd.DataFrame()
    df_timings = pd.concat(all_timings_list, ignore_index=True) if all_timings_list else pd.DataFrame()
    return df_final_perf, df_learning_curves, df_timings

# --- Plotting Functions ---
def generate_heatmaps(df_final_perf: pd.DataFrame, output_base_path: Path):
    if df_final_perf.empty: return
    heatmap_plot_dir = output_base_path / "Hyperparameter_Heatmaps"
    heatmap_plot_dir.mkdir(parents=True, exist_ok=True)
    df_final_perf['algo'] = df_final_perf['algo'].astype(str)
    df_final_perf['distribution_type'] = df_final_perf['distribution_type'].astype(str)
    
    grouping_keys = ['algo', 'distribution_type']
    if 'group_size' in df_final_perf.columns:
        grouping_keys.append('group_size')

    for name_tuple, group in df_final_perf.groupby(grouping_keys, dropna=False): # dropna=False to keep PPO's NaN group_size
        current_algo, current_dist = name_tuple[0], name_tuple[1]
        current_gs = name_tuple[2] if len(name_tuple) > 2 else np.nan

        if current_algo == 'ppo' and pd.notna(current_gs): continue # PPO shouldn't have group_size
        if current_algo == 'grpo' and pd.isna(current_gs): continue # GRPO should have group_size

        title_parts = [f"Algo: {current_algo.upper()}", f"Dist: {current_dist.capitalize()}"]
        filename_parts = [current_algo, current_dist]
        if pd.notna(current_gs):
            title_parts.append(f"GroupSize: {int(current_gs)}")
            filename_parts.append(f"g{int(current_gs)}")

        if group.empty or group['lr'].nunique() < 2 or group['entropy_coef'].nunique() < 2: continue
        agg_perf = group.groupby(['lr', 'entropy_coef'])['final_performance'].agg(['mean', 'std', 'count']).reset_index()
        try:
            mean_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='mean')
            std_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='std')
            count_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='count')
        except Exception: continue
        if mean_pivot.empty: continue
        annot_data = mean_pivot.copy().astype(object)
        for r_idx in mean_pivot.index:
            for c_idx in mean_pivot.columns:
                mean_val, std_val, count_val = mean_pivot.loc[r_idx, c_idx], std_pivot.loc[r_idx, c_idx], count_pivot.loc[r_idx, c_idx]
                if pd.notna(mean_val): annot_data.loc[r_idx, c_idx] = f"{mean_val:.1f}\n±{std_val:.1f}\n(n={int(count_val)})"
                else: annot_data.loc[r_idx, c_idx] = ""
        plt.figure(figsize=(max(8, len(mean_pivot.columns)*2), max(6, len(mean_pivot.index)*1)))
        sns.heatmap(mean_pivot, annot=annot_data, fmt='s', cmap="viridis", linewidths=.5, cbar_kws={'label': f'Mean Final Reward (Last {LAST_N_REWARDS_FOR_FINAL_PERF} Logs)'})
        plt.title(f"Performance Heatmap: LR vs. EC\n" + ", ".join(title_parts), fontsize=14)
        plt.xlabel("Entropy Coefficient", fontsize=12); plt.ylabel("Learning Rate", fontsize=12)
        plt.tight_layout()
        plt.savefig(heatmap_plot_dir / ("_".join(filename_parts) + "_LR_vs_EC_heatmap.png")); plt.close()
        print(f"  Saved heatmap: {heatmap_plot_dir / ('_'.join(filename_parts) + '_LR_vs_EC_heatmap.png')}")

def _plot_single_parameter_effect_curves(metrics_df_subset, timings_df_subset, algo, dist_type, varying_param, fixed_values, sweep_group_name, output_base_path):
    title_suffix_parts = [f"{k}={v}" for k, v in fixed_values.items() if pd.notna(v)]
    title_suffix = ", ".join(title_suffix_parts)
    plot_filename_suffix = "_".join([f"{k.replace('_','')[0:2]}{v}" for k, v in fixed_values.items() if pd.notna(v)]).replace(".","p")
    plot_dir = output_base_path / sweep_group_name / f"{algo}_{dist_type}" / f"Vary_{varying_param}"
    if title_suffix_parts: plot_dir = plot_dir / plot_filename_suffix
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_df_subset.empty and varying_param in metrics_df_subset.columns and metrics_df_subset[varying_param].nunique() > 0:
        plt.figure(figsize=(12, 7)); sns.lineplot(data=metrics_df_subset, x='step', y='avg_episodic_reward', hue=varying_param, errorbar='sd', palette='viridis', legend='full')
        plot_title = f"Reward ({algo.upper()}-{dist_type.capitalize()}) Varying: {varying_param}" + (f"\nFixed: {title_suffix}" if title_suffix else "")
        plt.title(plot_title); plt.xlabel("Training Steps"); plt.ylabel(f"Avg Reward (Smooth {REWARD_SMOOTHING_WINDOW})")
        plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(plot_dir / f"rewards_vary_{varying_param}.png"); plt.close()
    if not timings_df_subset.empty and varying_param in timings_df_subset.columns and timings_df_subset[varying_param].nunique() > 0:
        for timing_key_to_plot in timings_df_subset['timing_key'].unique():
            timing_key_df = timings_df_subset[timings_df_subset['timing_key'] == timing_key_to_plot]
            if not timing_key_df.empty:
                plt.figure(figsize=(12, 7)); sns.lineplot(data=timing_key_df, x='step', y='avg_ms', hue=varying_param, errorbar='sd', palette='viridis', legend='full')
                plot_title = f"{timing_key_to_plot} ({algo.upper()}-{dist_type.capitalize()}) Varying: {varying_param}" + (f"\nFixed: {title_suffix}" if title_suffix else "")
                plt.title(plot_title); plt.xlabel("Training Steps"); plt.ylabel("Average Time (ms)")
                plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(plot_dir / f"timing_{timing_key_to_plot}_vary_{varying_param}.png"); plt.close()

def iterate_and_plot_parameter_effects(metrics_df, timings_df, current_algo, current_dist_type, varying_param, fixed_params_options, sweep_group_name, output_base_path):
    fixed_param_names = [k for k in fixed_params_options.keys() if k != varying_param and fixed_params_options[k]]
    if not fixed_param_names:
        _plot_single_parameter_effect_curves(metrics_df, timings_df, current_algo, current_dist_type, varying_param, {}, sweep_group_name, output_base_path)
        return
    value_lists = [fixed_params_options[name] for name in fixed_param_names]
    for specific_fixed_values_tuple in itertools.product(*value_lists):
        fixed_values = dict(zip(fixed_param_names, specific_fixed_values_tuple))
        metrics_subset, timings_subset = metrics_df.copy(), timings_df.copy()
        valid_subset = True
        for p_name, p_val in fixed_values.items():
            if pd.isna(p_val):
                 metrics_subset = metrics_subset[metrics_subset[p_name].isna()]
                 if not timings_subset.empty: timings_subset = timings_subset[timings_subset[p_name].isna()]
            else:
                metrics_subset = metrics_subset[metrics_subset[p_name] == p_val]
                if not timings_subset.empty: timings_subset = timings_subset[timings_subset[p_name] == p_val]
            if metrics_subset.empty and (timings_subset.empty if not timings_df.empty else True) and p_name in metrics_df.columns:
                valid_subset = False; break
        if valid_subset and (not metrics_subset.empty or (not timings_subset.empty and not timings_df.empty)):
            _plot_single_parameter_effect_curves(metrics_subset, timings_subset, current_algo, current_dist_type, varying_param, fixed_values, sweep_group_name, output_base_path)

def generate_parameter_comparison_plots_for_sweep_group(metrics_df_sg, timings_df_sg, sweep_group_name, output_base_path):
    if metrics_df_sg.empty and timings_df_sg.empty: return
    df_for_grouping = (metrics_df_sg if not metrics_df_sg.empty else timings_df_sg).copy()
    df_for_grouping['algo'] = df_for_grouping['algo'].astype(str)
    df_for_grouping['distribution_type'] = df_for_grouping['distribution_type'].astype(str)
    for (algo, dist_type), _ in df_for_grouping.groupby(['algo', 'distribution_type']):
        print(f"  Generating param plots for: {algo.upper()}-{dist_type.capitalize()} in {sweep_group_name}")
        metrics_df_group = metrics_df_sg[(metrics_df_sg['algo'] == algo) & (metrics_df_sg['distribution_type'] == dist_type)] if not metrics_df_sg.empty else pd.DataFrame()
        timings_df_group = timings_df_sg[(timings_df_sg['algo'] == algo) & (timings_df_sg['distribution_type'] == dist_type)] if not timings_df_sg.empty else pd.DataFrame()
        unique_lrs = sorted(metrics_df_group['lr'].dropna().unique()) if 'lr' in metrics_df_group and not metrics_df_group.empty else []
        unique_ecs = sorted(metrics_df_group['entropy_coef'].dropna().unique()) if 'entropy_coef' in metrics_df_group and not metrics_df_group.empty else []
        unique_gss = sorted(metrics_df_group['group_size'].dropna().unique()) if algo == 'grpo' and 'group_size' in metrics_df_group and not metrics_df_group.empty else []
        if unique_lrs: iterate_and_plot_parameter_effects(metrics_df_group, timings_df_group, algo, dist_type, 'lr', {'entropy_coef': unique_ecs, 'group_size': unique_gss if algo == 'grpo' else [np.nan]}, sweep_group_name, output_base_path)
        if unique_ecs: iterate_and_plot_parameter_effects(metrics_df_group, timings_df_group, algo, dist_type, 'entropy_coef', {'lr': unique_lrs, 'group_size': unique_gss if algo == 'grpo' else [np.nan]}, sweep_group_name, output_base_path)
        if algo == 'grpo' and unique_gss: iterate_and_plot_parameter_effects(metrics_df_group, timings_df_group, algo, dist_type, 'group_size', {'lr': unique_lrs, 'entropy_coef': unique_ecs}, sweep_group_name, output_base_path)

def generate_overall_algorithm_comparison_plots(all_metrics_df, all_timings_df, output_base_path):
    if all_metrics_df.empty and all_timings_df.empty: return
    overall_plot_dir = output_base_path / "Overall_Algorithm_Comparison"
    overall_plot_dir.mkdir(parents=True, exist_ok=True)
    env_id = "UnknownEnv"
    if not all_metrics_df.empty and 'env_id' in all_metrics_df: env_id = all_metrics_df['env_id'].unique()[0]
    if not all_metrics_df.empty: all_metrics_df['algo_dist'] = all_metrics_df['algo'].astype(str) + "_" + all_metrics_df['distribution_type'].astype(str)
    if not all_timings_df.empty: all_timings_df['algo_dist'] = all_timings_df['algo'].astype(str) + "_" + all_timings_df['distribution_type'].astype(str)
    print(f"\n--- Generating Overall Algorithm Comparison Plots (Env: {env_id}) ---")
    if not all_metrics_df.empty:
        plt.figure(figsize=(14, 8)); sns.lineplot(data=all_metrics_df, x='step', y='avg_episodic_reward', hue='algo_dist', errorbar='sd', palette='tab10', legend='full')
        plt.title(f"Overall Algorithm Performance Comparison ({env_id})\n(Smoothed Reward, Aggregated over all Hyperparameters & Seeds)")
        plt.xlabel("Training Steps"); plt.ylabel(f"Avg Episodic Reward (Smooth {REWARD_SMOOTHING_WINDOW})")
        plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(overall_plot_dir / f"{env_id}_overall_rewards.png"); plt.close()
        print(f"  Saved overall rewards plot for {env_id}.")
    if not all_timings_df.empty:
        for timing_key in all_timings_df['timing_key'].unique():
            timing_key_df = all_timings_df[all_timings_df['timing_key'] == timing_key]
            if not timing_key_df.empty:
                plt.figure(figsize=(14, 8)); sns.lineplot(data=timing_key_df, x='step', y='avg_ms', hue='algo_dist', errorbar='sd', palette='tab10', legend='full')
                plt.title(f"Overall Timing: {timing_key} ({env_id})\n(Aggregated over all Hyperparameters & Seeds)")
                plt.xlabel("Training Steps"); plt.ylabel("Average Time (ms)")
                plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout(); plt.savefig(overall_plot_dir / f"{env_id}_overall_timing_{timing_key}.png"); plt.close()
        print(f"  Saved overall timing plots for {env_id}.")

# --- LaTeX Table Generation ---
def generate_latex_tables(df_final_perf: pd.DataFrame, output_dir: Path):
    if df_final_perf.empty: return
    output_dir.mkdir(parents=True, exist_ok=True)
    df_final_perf['algo'] = df_final_perf['algo'].astype(str)
    df_final_perf['distribution_type'] = df_final_perf['distribution_type'].astype(str)
    
    grouping_keys = ['algo', 'distribution_type']
    if 'group_size' in df_final_perf.columns: grouping_keys.append('group_size')

    for name_tuple, group in df_final_perf.groupby(grouping_keys, dropna=False):
        current_algo, current_dist = name_tuple[0], name_tuple[1]
        current_gs = name_tuple[2] if len(name_tuple) > 2 else np.nan
        if current_algo == 'ppo' and pd.notna(current_gs): continue
        if current_algo == 'grpo' and pd.isna(current_gs): continue
        
        table_caption_parts = [f"{current_algo.upper()}", f"{current_dist.capitalize()}"]
        filename_parts = [current_algo, current_dist]
        if pd.notna(current_gs):
            table_caption_parts.append(f"G={int(current_gs)}")
            filename_parts.append(f"g{int(current_gs)}")
        
        if group.empty or group['lr'].nunique() < 1 or group['entropy_coef'].nunique() < 1: continue
        agg_perf = group.groupby(['lr', 'entropy_coef'])['final_performance'].agg(['mean', 'std']).reset_index()
        try:
            mean_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='mean')
            std_pivot = agg_perf.pivot(index='lr', columns='entropy_coef', values='std')
        except Exception: continue
        if mean_pivot.empty: continue

        lrs = sorted(mean_pivot.index.unique())
        ecs = sorted(mean_pivot.columns.unique())
        
        # Format LRs and ECs for LaTeX headers (scientific notation for small values)
        def format_header_val(v):
            return f"{v:.0e}" if abs(v) < 1e-2 and v != 0 else f"{v:.4f}".rstrip('0').rstrip('.')
        
        col_headers = ["LR \\ EC"] + [format_header_val(ec) for ec in ecs]
        
        latex_str = "\\begin{table}[H]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{Final Performance (Mean $\\pm$ Std. Dev.) for {', '.join(table_caption_parts)}. Averaged over last {LAST_N_REWARDS_FOR_FINAL_PERF} reward logs.}}\n"
        latex_str += "\\label{tab:" + "_".join(filename_parts).lower() + "}\n"
        latex_str += "\\resizebox{\\textwidth}{!}{%\n" # To make table fit page width
        latex_str += "\\begin{tabular}{l|" + "c" * len(ecs) + "}\n"
        latex_str += "\\toprule\n"
        latex_str += " & ".join(col_headers) + " \\\\\n"
        latex_str += "\\midrule\n"
        
        for lr in lrs:
            row_data = [format_header_val(lr)]
            for ec in ecs:
                mean_val = mean_pivot.loc[lr, ec] if (lr, ec) in mean_pivot.stack().index else np.nan
                std_val = std_pivot.loc[lr, ec] if (lr, ec) in std_pivot.stack().index else np.nan
                if pd.notna(mean_val) and pd.notna(std_val):
                    row_data.append(f"${mean_val:.1f} \\pm {std_val:.1f}$")
                else:
                    row_data.append("-") # Or empty: ""
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

    # 1. Generate Heatmaps
    if not df_final_perf.empty:
        print("\n--- Generating Hyperparameter Heatmaps ---")
        generate_heatmaps(df_final_perf, PLOTS_OUTPUT_DIR)
    
    # 2. Generate LaTeX Tables
    if not df_final_perf.empty:
        print("\n--- Generating LaTeX Performance Tables ---")
        generate_latex_tables(df_final_perf, LATEX_OUTPUT_DIR)

    # 3. Generate Parameter-Specific Learning Curve Plots (Per Sweep Group)
    print("\n--- Generating Parameter-Specific Learning Curve Plots (per sweep group) ---")
    for sweep_group_dir in sorted(BASE_SWEEPS_DIR.iterdir()):
        if not sweep_group_dir.is_dir(): continue
        sg_name = sweep_group_dir.name
        # Filter the global DFs for the current sweep group
        metrics_df_sg = df_learning_curves[df_learning_curves['sweep_group'] == sg_name] if not df_learning_curves.empty else pd.DataFrame()
        timings_df_sg = df_timings[df_timings['sweep_group'] == sg_name] if not df_timings.empty else pd.DataFrame()
        if not metrics_df_sg.empty or not timings_df_sg.empty:
             generate_parameter_comparison_plots_for_sweep_group(metrics_df_sg, timings_df_sg, sg_name, PLOTS_OUTPUT_DIR)
    
    # 4. Generate Overall Algorithm Comparison Plots
    if not df_learning_curves.empty or not df_timings.empty:
        print("\n--- Generating Overall Algorithm Comparison Plots ---")
        generate_overall_algorithm_comparison_plots(df_learning_curves, df_timings, PLOTS_OUTPUT_DIR)

    print(f"\nAll processing complete. Check '{PLOTS_OUTPUT_DIR}' and '{LATEX_OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()