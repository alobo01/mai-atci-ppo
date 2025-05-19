import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CONFIG ------------------------------------------------------------
LOG_DIR = "finalLogs"
OUTPUT_DIR = "plots_envs"
# include MLP-specific phases
PHASES = [
    "cnn_feature",
    "actor_head",
    "rollout_phase",
    "ref_cnn_feature",
    "ref_actor_head",
    "backward_pass",
    "optimizer_step",
    "update_phase",
    # MLP phases
    "actor_mlp",
    "ref_actor_mlp"
]
DISTS = ["normal"]
G_VALUES = ["4", "16", "64"]
TARGET_ENV = "CarRacing-v3"

# possible labels and mapping to config tuples
LABELS = [
    "PPO_NORMAL",
    "GRPO_NORMAL_G4",
    "GRPO_NORMAL_G16",
    "GRPO_NORMAL_G64",
]
LABEL_TO_CFG = {
    "PPO_NORMAL": ("PPO", "NORMAL", None),
    **{f"GRPO_{d.upper()}_G{g}": ("GRPO", d.upper(), f"G{g}") for d in DISTS for g in G_VALUES}
}

# regex to extract folder info
FOLDER_RE = re.compile(
    r"^(?P<env>.+)_(?P<model>ppo|grpo)_seed(?P<seed>\d+)_ent(?P<ent>[\d\.]+)"
    r"_lr(?P<lr>[\deE\.-]+)_(?P<dist>normal|beta)(?:_g(?P<G>\d+))?$"
)

# storage for timings: timings[(model, dist, G)][phase] = []
timings = defaultdict(lambda: defaultdict(list))

# parse timing files (only CarRacing)
for fname in os.listdir(LOG_DIR):
    m = FOLDER_RE.match(fname)
    if not m:
        continue
    env = m.group("env")
    if env != TARGET_ENV:
        continue
    model = m.group("model")
    dist = m.group("dist")
    G = m.group("G")
    # skip invalid combos
    if model == "grpo" and G is None:
        continue
    if model == "ppo" and G is not None:
        continue
    cfg = (model.upper(), dist.upper(), f"G{G}" if G else None)
    tfile = os.path.join(LOG_DIR, fname, "timings.jsonl")
    if not os.path.isfile(tfile):
        continue
    # collect per-phase avg_ms lists
    with open(tfile) as f:
        for line in f:
            data = json.loads(line)
            for phase in PHASES:
                if phase in data:
                    timings[cfg][phase].append(data[phase]["avg_ms"])

# choose a qualitative colormap
cmap = plt.get_cmap('tab10')

# generate vertical boxplots for each phase (CarRacing only)
for phase in PHASES:
    # gather data for each label
    data_to_plot = []
    for lbl in LABELS:
        cfg = LABEL_TO_CFG[lbl]
        vals = timings[cfg].get(phase, [])
        # filter only positive timings
        filtered = [v for v in vals if v > 0]
        data_to_plot.append(filtered if filtered else [0])

    plt.figure(figsize=(12, 6))
    # create boxplot with colored boxes and no outliers
    bp = plt.boxplot(data_to_plot, tick_labels=LABELS, notch=True,
                     showfliers=False, patch_artist=True)
    # style each box
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=cmap(i % cmap.N), linewidth=1.5)
    # style medians
    for median in bp['medians']:
        median.set(color='black', linewidth=1.5)
    # style whiskers and caps
    for whisker in bp['whiskers'] + bp['caps']:
        whisker.set(color='gray', linewidth=1)

    plt.suptitle(f"Timing Distribution on {TARGET_ENV}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("ms per call")
    plt.title(f"Phase: {phase}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"box_{TARGET_ENV}_{phase}.png"))
    plt.close()
