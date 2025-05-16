# parse_hpo_results.py
import json
from pathlib import Path
import pandas as pd

HPO_LOG_DIR = Path("experiment_runs_hpo")
METRICS_FILENAME = "metrics.json"
CONFIG_FILENAME_PATTERN = "config_hpo_*.json" # In the configs_hpo/temp_run_XX dir

def get_final_evaluation_reward(run_log_dir: Path) -> float:
    """
    Parses the agent's .log file to find the final 'Evaluation Complete: Avg Reward'.
    This is a bit fragile as it relies on exact log string format.
    """
    agent_log_file = next(run_log_dir.glob("logs/*.log"), None) # Get the first .log file
    if not agent_log_file:
        return float('-inf')
    
    best_eval_reward = float('-inf')
    try:
        with open(agent_log_file, 'r') as f:
            for line in f:
                if "Evaluation Complete: Avg Reward =" in line:
                    # Example: "INFO | Evaluation Complete: Avg Reward = -128.28 (over 1 eps)"
                    try:
                        reward_str = line.split("Avg Reward =")[1].split("(over")[0].strip()
                        eval_reward = float(reward_str)
                        # We might have multiple evaluations if video_interval is hit
                        # Let's take the one associated with the final step or just the last one found
                        best_eval_reward = eval_reward # Overwrite with later ones
                    except (IndexError, ValueError) as e:
                        print(f"Could not parse eval reward from line in {agent_log_file.name}: {line.strip()} - Error: {e}")
                        continue # Skip this line
        return best_eval_reward
    except Exception as e:
        print(f"Error reading log file {agent_log_file}: {e}")
        return float('-inf')


def analyze_hpo_runs(base_hpo_dir: Path):
    results = []
    if not base_hpo_dir.is_dir():
        print(f"Error: HPO directory {base_hpo_dir} not found.")
        return

    for run_dir in base_hpo_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_plots"):
            continue

        metrics_path = run_dir / METRICS_FILENAME
        # The actual HPO config was in `configs_hpo/temp_run_XX/config_hpo_XX.json`
        # The run_dir name itself contains the HPO params.
        # Let's try to parse params from dir name or load the copied config if it's inside run_dir.
        
        # Attempt to find the specific config.json copied by your HPO script into the run's log dir
        # This assumes your HPO script copies the config_hpo_XX.json into the run's root.
        # If not, you'll need to parse from the run_dir.name.
        
        # Let's assume the config file for the run is stored inside the run directory
        # (e.g., your main script could save its full config object there)
        # If not, we parse from directory name.
        
        run_name = run_dir.name
        params = {"run_name": run_name}
        # Basic parsing from name (can be made more robust)
        name_parts = run_name.split('_')
        for part in name_parts:
            if part.startswith("lr"): params["lr"] = part[2:].replace('p', '.')
            elif part.startswith("ent"): params["entropy_coef"] = part[3:].replace('p', '.')
            elif part.startswith("kl"): params["kl_coef"] = part[2:].replace('p', '.') # kl_ not kl
            elif part.startswith("gro"): params["group_size"] = part[3:]
        
        # Extract performance metrics
        avg_training_reward_last_n = None
        max_training_reward = None
        final_eval_reward = get_final_evaluation_reward(run_dir) # Get from .log file

        if metrics_path.is_file():
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                
                rewards = [r for r in metrics_data.get("avg_episodic_reward", []) if r is not None]
                if rewards:
                    avg_training_reward_last_n = sum(rewards[-5:]) / len(rewards[-5:]) if len(rewards) >= 5 else sum(rewards) / len(rewards)
                    max_training_reward = max(rewards)
            except json.JSONDecodeError:
                print(f"Could not decode metrics.json for {run_name}")
            except Exception as e:
                print(f"Error processing metrics for {run_name}: {e}")

        results.append({
            **params,
            "avg_train_reward_last5": avg_training_reward_last_n,
            "max_train_reward": max_training_reward,
            "final_eval_reward": final_eval_reward
        })

    if not results:
        print("No HPO results found to analyze.")
        return

    df = pd.DataFrame(results)
    
    # Sort by the most important metric, e.g., final_eval_reward
    df_sorted = df.sort_values(by="final_eval_reward", ascending=False)
    
    print("\n--- HPO Results Summary (Sorted by Final Eval Reward) ---")
    print(df_sorted.to_string())

    # Save to CSV
    csv_path = base_hpo_dir / "_hpo_summary.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")

    if not df_sorted.empty:
        best_run = df_sorted.iloc[0]
        print("\n--- Best Performing Run ---")
        print(best_run)
        # You can then find the corresponding config_hpo_XX.json or reconstruct from params.
        # Finding the original config_hpo_XX.json would require the hyperparam_search.py
        # to also log which HPO run_id (e.g., hpo91) corresponds to which config file.
        # For now, the run_name in the summary contains the hyperparams.


if __name__ == "__main__":
    analyze_hpo_runs(HPO_LOG_DIR)