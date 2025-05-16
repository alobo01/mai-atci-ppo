# hyperparam_search.py
import subprocess
import json
import os
import itertools
from pathlib import Path
import shutil

# --- Configuration for the Search ---
BASE_CONFIG_FILE = "HalfCheetah-v4_grpo_base.json" # Your best current config
EXPERIMENT_SCRIPT = "run_experiment.py" # Your main script
BASE_LOG_DIR = "experiment_runs_hpo"
CONFIG_DIR_HPO = "configs_hpo" # Directory to store generated configs

# Parameters to search over
# Define ranges or lists of values for key hyperparameters
# Example:
search_space = {
    "grpo_config.lr": [0.0003, 0.0001, 0.00005],
    "grpo_config.entropy_coef": [0.001, 0.0001, 0.00001, 0.0],
    "grpo_config.kl_coef": [0.001, 0.0001, 0.00005, 0.0],
    "grpo_config.group_size": [32, 64],
    # Add other parameters like ref_update_interval if desired
}

# --- Helper Function to Modify Config ---
def get_nested_value(cfg_dict, key_path):
    keys = key_path.split('.')
    val = cfg_dict
    for key in keys:
        val = val[key]
    return val

def set_nested_value(cfg_dict, key_path, value):
    keys = key_path.split('.')
    d = cfg_dict
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

# --- Main Search Loop ---
def run_search():
    base_config_path = Path(BASE_CONFIG_FILE)
    if not base_config_path.exists():
        print(f"Error: Base config file {BASE_CONFIG_FILE} not found.")
        return

    with open(base_config_path, 'r') as f:
        base_config_data = json.load(f)

    hpo_config_path = Path(CONFIG_DIR_HPO)
    if hpo_config_path.exists():
        shutil.rmtree(hpo_config_path) # Clean up old generated configs
    hpo_config_path.mkdir(parents=True, exist_ok=True)

    # Generate all combinations of parameters
    param_names = list(search_space.keys())
    param_value_combinations = list(itertools.product(*(search_space[k] for k in param_names)))

    print(f"Starting hyperparameter search with {len(param_value_combinations)} combinations.")
    
    run_id_counter = 0
    for i, param_values in enumerate(param_value_combinations):
        current_config_data = base_config_data.copy() # Start fresh from base
        run_params_str_parts = []

        print(f"\n--- Combination {i+1}/{len(param_value_combinations)} ---")
        for key_path, value in zip(param_names, param_values):
            print(f"  Setting {key_path} = {value}")
            set_nested_value(current_config_data, key_path, value)
            # Create a short string for this param-value for the run name
            key_short = key_path.split('.')[-1][:3] # e.g., "lr", "ent", "kl_", "gro"
            run_params_str_parts.append(f"{key_short}{value}")

        # Create a unique run name and config file name
        run_id_counter +=1
        params_suffix = "_".join(run_params_str_parts).replace('.', 'p') # Make it filename-friendly
        
        # Construct a more descriptive run_name for the config itself
        # The main script will further append seed etc.
        current_config_data["run_name"] = f"{base_config_data.get('env_id', 'env')}_{base_config_data.get('algo', 'algo')}_hpo{run_id_counter}_{params_suffix}"
        
        new_config_filename = hpo_config_path / f"config_hpo_{run_id_counter}.json"
        with open(new_config_filename, 'w') as f:
            json.dump(current_config_data, f, indent=4)

        # Run the experiment
        # Pass the directory of the generated configs to your main script
        cmd = [
            "python", EXPERIMENT_SCRIPT,
            "--config-dir", str(hpo_config_path), # Point to the dir with THIS ONE config
            # If your script runs all configs in a dir, you'd need to make it run a specific one
            # Or, save each config to its own temp dir.
            # For simplicity, let's assume your script can be modified or it just runs the one.
            # A better way: modify run_experiment.py to take a single --config-file argument.
            # For now, we'll make a temp dir for each config.

            # "--single-config-file", str(new_config_filename), # Ideal if your script supports this
            "--base-log-dir", BASE_LOG_DIR,
            # Add other fixed args like --workers, --device if needed
        ]
        
        # To run a single config, we'll create a temporary directory for it
        temp_single_config_dir = hpo_config_path / f"temp_run_{run_id_counter}"
        temp_single_config_dir.mkdir(exist_ok=True)
        shutil.copy(new_config_filename, temp_single_config_dir / new_config_filename.name)

        cmd_single_run = [
            "python", EXPERIMENT_SCRIPT,
            "--config-dir", str(temp_single_config_dir),
            "--base-log-dir", BASE_LOG_DIR,
            "--workers", "1", # Run one at a time for HPO unless you have many resources
        ]

        print(f"Running: {' '.join(cmd_single_run)}")
        try:
            process = subprocess.Popen(cmd_single_run)
            process.wait() # Wait for this run to finish before starting the next
            if process.returncode != 0:
                print(f"Warning: Experiment run {run_id_counter} exited with code {process.returncode}")
        except Exception as e:
            print(f"Error running experiment {run_id_counter}: {e}")
        finally:
            shutil.rmtree(temp_single_config_dir) # Clean up temp dir

    print("\n--- Hyperparameter search finished ---")

if __name__ == "__main__":
    # Create a dummy base config if it doesn't exist for testing the search script
    if not Path(BASE_CONFIG_FILE).exists():
        print(f"Creating dummy base config: {BASE_CONFIG_FILE}")
        dummy_cfg = {
            "env_id": "HalfCheetah-v4", "algo": "grpo", "seed": 0, "total_steps": 10000,
            "grpo_config": { "lr": 0.0003, "entropy_coef": 0.001, "kl_coef": 0.001, "group_size": 32, "ref_update_interval":100000}
        }
        with open(BASE_CONFIG_FILE, 'w') as f:
            json.dump(dummy_cfg, f, indent=4)
    
    # Create a dummy experiment script if it doesn't exist
    if not Path(EXPERIMENT_SCRIPT).exists():
        print(f"Creating dummy experiment script: {EXPERIMENT_SCRIPT}")
        with open(EXPERIMENT_SCRIPT, 'w') as f:
            f.write("import time, argparse, pathlib, json\n")
            f.write("parser = argparse.ArgumentParser()\n")
            f.write("parser.add_argument('--config-dir', type=str)\n")
            f.write("parser.add_argument('--base-log-dir', type=str)\n")
            f.write("parser.add_argument('--workers', type=str)\n")
            f.write("args = parser.parse_args()\n")
            f.write("config_files = list(pathlib.Path(args.config_dir).glob('*.json'))\n")
            f.write("cfg_data = json.load(open(config_files[0]))\n")
            f.write("run_name = cfg_data.get('run_name', 'dummy_run')\n")
            f.write("print(f'Dummy run_experiment.py: Simulating run for {run_name} from {args.config_dir} to {args.base_log_dir}')\n")
            f.write("time.sleep(1)\n")
            f.write("print('Dummy run finished.')\n")

    run_search()