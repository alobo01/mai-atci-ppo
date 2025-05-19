import os
import re
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ------------------------------------------------------------

LOG_DIR      = "finalLogs"
OUTPUT_DIR   = "plots_envs"
ENVIRONMENTS = [
    "Hopper-v4",
    "Walker2d-v4",
    "Swimmer-v4",
    "InvertedPendulum-v4",
    "InvertedDoublePendulum-v4",
    "CarRacing-v3"
]

MODELS   = ["ppo", "grpo"]
DISTS    = ["normal", "beta"]
G_VALUES = ["4", "16", "64"]

FOLDER_RE = re.compile(
    r"^"
    r"(?P<env>.+)"
    r"_(?P<model>ppo|grpo)"
    r"_seed(?P<seed>\d+)"
    r"_ent(?P<ent>[\deE\.-]+)"
    r"_lr(?P<lr>[\deE\.-]+)"
    r"_(?P<dist>normal|beta)"
    r"(?:_g(?P<G>\d+))?"
    r"$"
)

# per-environment minimum steps
MIN_STEPS_DEFAULT = 900_000
MIN_STEPS_PER_ENV = {
    "CarRacing-v3": 90_000,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLLECT DATA ------------------------------------------------------

data              = defaultdict(lambda: defaultdict(dict))
skipped_min_steps = []
seen_cfgs         = set()

for fname in os.listdir(LOG_DIR):
    full = os.path.join(LOG_DIR, fname)
    if not os.path.isdir(full):
        continue

    m = FOLDER_RE.match(fname)
    if not m:
        continue

    env   = m.group("env")
    model = m.group("model")
    seed  = m.group("seed")
    dist  = m.group("dist")
    G     = m.group("G") or None

    # ignore invalid combos
    if model == "grpo" and G is None:
        continue
    if model == "ppo"  and G is not None:
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

    # select the right threshold for this env
    min_required = MIN_STEPS_PER_ENV.get(env, MIN_STEPS_DEFAULT)
    if max_steps < min_required:
        skipped_min_steps.append(
            ((env, model.upper(), dist.upper(), f"G{G}" if G else None),
             int(max_steps),
             min_required)
        )
        continue

    data[(env, model, dist, G)][seed] = {"steps": steps, "rews": rews}

# --- REPORT MISSING OR INSUFFICIENT -----------------------------------

if skipped_min_steps:
    print("The following configs did not meet their minimum steps:")
    for (cfg, got, want) in skipped_min_steps:
        print(f"  • {cfg}: only {got:,d} < required {want:,d}")

print("\nConfigs with zero seeds:")
for env in ENVIRONMENTS:
    for model in MODELS:
        for dist in DISTS:
            G_opts = [None] if model == "ppo" else G_VALUES
            for G in G_opts:
                cfg = (env, model, dist, G)
                if cfg not in seen_cfgs:
                    label = f"{model.upper()}_{dist.upper()}" + (f"_G{G}" if G else "")
                    print(f"  • {env}, {label}")

# --- PLOTTING & SAVING ------------------------------------------------

END_STEP      = 1_000_000    # fixed x‐axis limit
GRID_POINTS   = 500          # how many x‐samples between 0 and END_STEP
SMOOTH_WINDOW = 5            # moving‐average window size

for env in ENVIRONMENTS:
    plt.figure(figsize=(8, 5))
    any_curve = False

    # build our common x‐axis
    grid = np.linspace(0, END_STEP, GRID_POINTS)

    for (e, model, dist, G), runs in data.items():
        if e != env or not runs:
            continue
        any_curve = True
        label = f"{model.upper()}_{dist.upper()}" + (f"_G{G}" if G else "")

        mat = []
        for s in runs:
            steps = runs[s]["steps"]
            rews  = runs[s]["rews"]

            # 1) linear interp + carry‐forward last value
            y = np.interp(grid, steps, rews)

            # 2) causal smoothing: average over the last SMOOTH_WINDOW points
            #    pad the front with the first value so we can 'valid'-convolve
            pad = np.full(SMOOTH_WINDOW-1, y[0])
            y_padded = np.concatenate((pad, y))
            kernel   = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
            y_smooth = np.convolve(y_padded, kernel, mode="valid")

            mat.append(y_smooth)

        mat  = np.vstack(mat)
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)

        plt.plot(grid, mean, label=label)
        plt.fill_between(grid,
                         mean - std,
                         mean + std,
                         alpha=0.2)

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
    plt.savefig(out_path, dpi=150)
    plt.close()
