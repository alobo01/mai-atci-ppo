import os
import re
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ------------------------------------------------------------

LOG_DIR = "finalLogs"
OUTPUT_DIR = "plots_envs"
ENVIRONMENTS = [
    "Hopper-v4",
    "Walker2d-v4",
    "Swimmer-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "CarRacing-v3"
]

# Possible configurations to check
MODELS = ["ppo", "grpo"]
DISTS = ["normal", "beta"]
G_VALUES = ["4", "16", "64"]

# regex to extract:
#  env, model (ppo|grpo), seed, ent, lr, dist (normal|beta), optional G
FOLDER_RE = re.compile(
    r"^"
    r"(?P<env>.+)"                             # e.g. Hopper-v4, InvertedDoublePendulum-v4
    r"_(?P<model>ppo|grpo)"
    r"_seed(?P<seed>\d+)"
    r"_ent(?P<ent>[\deE\.-]+)"                 # now allows scientific notation
    r"_lr(?P<lr>[\deE\.-]+)"                   # likewise for learning‐rate if you ever use sci-notation
    r"_(?P<dist>normal|beta)"
    r"(?:_g(?P<G>\d+))?"                       # optional _g<N> for GRPO
    r"$"
)

MIN_STEPS = 90000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLLECT DATA ------------------------------------------------------

data = defaultdict(lambda: defaultdict(dict))
skipped_min_steps = []
seen_cfgs = set()

for fname in os.listdir(LOG_DIR):
    full = os.path.join(LOG_DIR, fname)
    if not os.path.isdir(full):
        continue

    m = FOLDER_RE.match(fname)
    if not m:
        continue

    env = m.group("env")
    model = m.group("model")
    seed = m.group("seed")
    dist = m.group("dist")
    G = m.group("G") or None

    # ignore invalid combos: GRPO must have G
    if model == "grpo" and G is None:
        continue
    # PPO should have no G
    if model == "ppo" and G is not None:
        continue

    if env not in ENVIRONMENTS:
        continue

    seen_cfgs.add((env, model, dist, G))

    metrics_path = os.path.join(full, "metrics.json")
    if not os.path.isfile(metrics_path):
        continue

    with open(metrics_path, "r") as f:
        js = json.load(f)

    steps = np.array(js.get("steps", []), dtype=float)
    rews  = np.array(js.get("avg_episodic_reward", []), dtype=float)

    max_steps = steps.max() if steps.size > 0 else 0
    if max_steps < MIN_STEPS:
        skipped_min_steps.append(((env, model.upper(), dist.upper(), f"G{G}" if G else None), int(max_steps)))
        continue

    data[(env, model, dist, G)][seed] = {"steps": steps, "rews": rews}

# --- REPORT MISSING OR INSUFFICIENT -----------------------------------

if skipped_min_steps:
    print("The following configurations did not satisfy the minimum steps (900,000):")
    for cfg, steps in skipped_min_steps:
        print(f"  • {cfg}: only {steps:,d} steps")

print("\nConfigurations with zero seeds:")
for env in ENVIRONMENTS:
    for model in MODELS:
        for dist in DISTS:
            G_options = [None] if model == "ppo" else G_VALUES
            for G in G_options:
                cfg = (env, model, dist, G)
                if cfg not in seen_cfgs:
                    label = f"{model.upper()}_{dist.upper()}" + (f"_G{G}" if G else "")
                    print(f"  • {env}, {label}")

# --- PLOTTING & SAVING ------------------------------------------------

for env in ENVIRONMENTS:
    plt.figure(figsize=(8, 5))
    any_curve = False

    for (e, model, dist, G), runs in data.items():
        if e != env or not runs:
            continue

        if len(runs) == 1:
            label_single = f"{model.upper()}_{dist.upper()}" + (f"_G{G}" if G else "")
            print(f"Warning: only one seed found for {label_single} in {env}")

        all_steps = [runs[s]["steps"] for s in runs]
        union_steps = np.array(sorted(set().union(*all_steps)))

        mat = [np.interp(union_steps, runs[s]["steps"], runs[s]["rews"]) for s in runs]
        mat = np.vstack(mat)

        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        label = f"{model.upper()}_{dist.upper()}" + (f"_G{G}" if G else "")
        plt.plot(union_steps, mean, label=label)
        plt.fill_between(union_steps, mean - std, mean + std, alpha=0.2)
        any_curve = True

    if not any_curve:
        plt.close()
        continue

    plt.title(f"{env} – Avg Episodic Reward")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Episodic Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"{env}.png")
    plt.savefig(out_path)
    plt.close()
