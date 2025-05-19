import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Directory containing experiment runs
root_dir = 'experiment_runs'

# Prepare a list to collect all data
records = []

# Walk through the directory to find metrics.json files
for dirpath, dirnames, filenames in os.walk(root_dir):
    if 'metrics.json' in filenames:
        folder_name = os.path.basename(dirpath)
        # Parse env_id, config, seed from folder name: env_config_seedX
        match = re.match(r'(.+?)_(.+?)_seed(\d+)', folder_name)
        if not match:
            continue
        env_id, config, seed = match.groups()
        seed = int(seed)
        
        # Load metrics
        metrics_path = os.path.join(dirpath, 'metrics.json')
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        # If metrics.json is a dict of lists
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            # If it's a list of records
            df = pd.DataFrame(data)
        
        # Expect columns 'step' and 'reward'; adjust if your files differ
        if 'step' not in df.columns:
            df = df.rename(columns={'timesteps': 'step'})
        if 'reward' not in df.columns:
            possible = [c for c in df.columns if 'reward' in c.lower()]
            if possible:
                df = df.rename(columns={possible[0]: 'reward'})
        
        # Attach metadata
        df['env_id'] = env_id
        df['config'] = config
        df['seed'] = seed
        
        records.append(df[['env_id', 'config', 'seed', 'steps', 'reward']])

# Concatenate all records
all_df = pd.concat(records, ignore_index=True)

# Aggregate: compute mean and std across seeds for each env_id, config, and step
agg = all_df.groupby(['env_id', 'config', 'steps']).agg(
    reward_mean=('reward', 'mean'),
    reward_std=('reward', 'std')
).reset_index()

# Plotting setup
envs = agg['env_id'].unique()
n_env = len(envs)
n_cols = 3
n_rows = int(np.ceil(n_env / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)

# Colors can be automatic
for idx, env in enumerate(envs):
    ax = axes[idx // n_cols][idx % n_cols]
    env_data = agg[agg['env_id'] == env]
    
    for config, cfg_data in env_data.groupby('config'):
        steps = cfg_data['steps']
        mean = cfg_data['reward_mean']
        std = cfg_data['reward_std'].fillna(0)
        
        ax.plot(steps, mean, label=config)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    
    ax.set_title(env)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True)

# Hide unused subplots
for idx in range(n_env, n_rows * n_cols):
    fig.delaxes(axes[idx // n_cols][idx % n_cols])

plt.tight_layout()
plt.show()
