from typing import Any, Dict, List, Optional, Tuple, Union
import os
import json
import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator

# --- Pydantic Config Schemas ---
class NetworkConfig(BaseModel):
    network_type: str = Field("mlp", pattern="^(mlp|cnn)$")
    mlp_hidden_dims: Tuple[int, ...] = (64, 64)
    cnn_output_features: PositiveInt = 256

class PPOConfig(BaseModel):
    distribution_type: str = Field("normal", pattern="^(normal|beta)$")
    lam: PositiveFloat = 0.95  # GAE lambda
    clip_eps: PositiveFloat = 0.2
    ppo_epochs: PositiveInt = 10
    num_minibatches: PositiveInt = 32
    entropy_coef: float = 0.01
    value_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    target_kl: Optional[PositiveFloat] = None
    rollout_steps: PositiveInt = 2048
    lr: PositiveFloat = 3e-4

class GRPOConfig(BaseModel):
    distribution_type: str = Field("normal", pattern="^(normal|beta)$")
    group_size: PositiveInt = 64
    update_epochs: PositiveInt = 10
    max_grad_norm: PositiveFloat = 0.5
    entropy_coef: float = 0.001
    kl_coef: float = 0.01
    ref_update_interval: PositiveInt = 10_000
    minibatch_size: PositiveInt = 256
    lr: PositiveFloat = 1e-4
    rollout_steps_per_trajectory: PositiveInt = 1000

class ExperimentConfig(BaseModel):
    env_id: str
    algo: str = Field(pattern="^(ppo|grpo)$")
    seed: int = 0
    gamma: PositiveFloat = Field(0.99, ge=0.0, le=1.0)
    total_steps: PositiveInt = 1_000_000
    log_interval: PositiveInt = 5000
    checkpoint_interval: PositiveInt = 50000
    video_interval: PositiveInt = 100_000
    run_name: Optional[str] = None
    base_log_dir: str = "experiment_runs"
    verbose: bool = False
    max_episode_steps: Optional[PositiveInt] = None

    network_config: NetworkConfig = Field(default_factory=NetworkConfig)
    ppo_config: Optional[PPOConfig] = None
    grpo_config: Optional[GRPOConfig] = None

    @field_validator('algo')
    def algo_name_check(cls, v: str) -> str:
        if v.lower() not in ["ppo", "grpo"]:
            raise ValueError("Algorithm must be 'ppo' or 'grpo'.")
        return v.lower()

    @field_validator('ppo_config', 'grpo_config')
    def ensure_correct_algo_config(cls, v: Optional[Union[PPOConfig, GRPOConfig]], values: Any) -> Optional[Union[PPOConfig, GRPOConfig]]:
        data = values.data
        algo = data.get('algo')
        if algo == 'ppo' and v is None and data.get('ppo_config') is None:
            return PPOConfig()
        if algo == 'grpo' and v is None and data.get('grpo_config') is None:
            return GRPOConfig()
        if algo == 'ppo' and data.get('grpo_config') is not None:
            raise ValueError("grpo_config should not be provided when algo is 'ppo'")
        if algo == 'grpo' and data.get('ppo_config') is not None:
            raise ValueError("ppo_config should not be provided when algo is 'grpo'")
        return v

    def get_algo_specific_config(self) -> Union[PPOConfig, GRPOConfig]:
        if self.algo == "ppo":
            if self.ppo_config is None:
                self.ppo_config = PPOConfig()
            return self.ppo_config
        if self.algo == "grpo":
            if self.grpo_config is None:
                self.grpo_config = GRPOConfig()
            return self.grpo_config
        raise ValueError(f"Unknown algorithm type for specific config: {self.algo}")

class RolloutBufferSamples(BaseModel):
    observations: torch.Tensor
    actions: torch.Tensor
    actions_canonical_unclipped: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

# --- Streamlit Dashboard ---
# Workaround: ensure an asyncio event loop and avoid Streamlit watcher error on torch._classes
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
# Monkey-patch torch._classes path to prevent __getattr__ errors
if 'torch' in globals():
    try:
        import torch as _torch
        if hasattr(_torch, '_classes') and hasattr(_torch._classes, '__path__'):
            _torch._classes.__path__ = []
    except Exception:
        pass

st.set_page_config(page_title="RL Experiment Dashboard", layout="wide")
st.title("📊 Reinforcement Learning Experiment Dashboard")
st.markdown(
    "Welcome! Use this dashboard to explore your PPO and GRPO experiment runs across different environments."
)

# --- Configuration ---
EXP_DIR = st.sidebar.text_input("Experiment Logs Directory", "finalLogs")

# Sidebar: experiment filtering & selection
st.sidebar.header("Experiment Selection")
valid_runs = []
import json
from pydantic import ValidationError
try:
    for d in os.listdir(EXP_DIR):
        run_path = os.path.join(EXP_DIR, d)
        if os.path.isdir(run_path):
            cfg_path = os.path.join(run_path, "config.json")
            met_path = os.path.join(run_path, "metrics.json")
            tim_path = os.path.join(run_path, "timings.jsonl")
            # Require config, metrics, and timing files
            if os.path.exists(cfg_path) and os.path.exists(met_path) and os.path.exists(tim_path):
                # Validate config via Pydantic
                try:
                    with open(cfg_path) as cf:
                        cfg_data = json.load(cf)
                    # Will raise if invalid
                    ExperimentConfig(**cfg_data)
                    valid_runs.append(d)
                except (json.JSONDecodeError, ValidationError):
                    # Skip runs with invalid or unparsable config
                    continue
except FileNotFoundError:
    st.error(f"Directory '{EXP_DIR}' not found. Please check the path.")
    st.stop()

if not valid_runs:
    st.warning("No valid runs found (missing config, metrics, or timing files).")
    st.stop()

# Environment & run selection
envs = sorted({run.split("_")[0] for run in valid_runs})
selected_env = st.sidebar.selectbox("Environment", ["All"] + envs)
# Optional filter: only show runs with videos
enable_video_filter = st.sidebar.checkbox("Only runs with videos", value=False)
# Precompute video availability
def has_videos(run_dir: str) -> bool:
    rp = os.path.join(EXP_DIR, run_dir)
    for root, _, files in os.walk(rp):
        for f in files:
            if f.endswith((".mp4", ".webm", ".gif")):
                return True
    return False

# Filter runs by selected environment and video availability
runs = [r for r in valid_runs if (selected_env == "All" or r.startswith(selected_env))]
if enable_video_filter:
    runs = [r for r in runs if has_videos(r)]
# Select run
selected_run = st.sidebar.selectbox("Run Directory", runs)
run_path = os.path.join(EXP_DIR, selected_run)

# Load experiment config
config_file = os.path.join(run_path, "config.json")
if os.path.exists(config_file):
    with open(config_file) as f:
        raw = json.load(f)
    exp_cfg = ExperimentConfig(**raw)
else:
    exp_cfg = None

st.sidebar.subheader("Experiment Configuration")
if exp_cfg:
    st.sidebar.json(exp_cfg.model_dump(), expanded=False)
else:
    st.sidebar.info("No config.json found for this run.")

# Quick metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Environment", exp_cfg.env_id if exp_cfg else selected_env)
algo = exp_cfg.algo.upper() if exp_cfg else selected_run.split("_")[1].upper()
algoconf = exp_cfg.get_algo_specific_config() if exp_cfg else None
dist = algoconf.distribution_type.upper() if algoconf and hasattr(algoconf, 'distribution_type') else "-"
gsize = getattr(algoconf, 'group_size', "-")
col2.metric("Algorithm", algo)
col3.metric("Distribution", dist)
col4.metric("Group Size", gsize)

# Tabs for interactive views
tab1, tab2, tab3, tab4 = st.tabs(["Learning Curve", "Timing Analysis", "Videos", "Logs"])

# --- Learning Curve Tab ---
with tab1:
    st.subheader("Learning Curve 📈")
    metrics_file = os.path.join(run_path, "metrics.json")
    if os.path.exists(metrics_file):
        df = pd.read_json(metrics_file)
        # Sort by steps and interpolate missing reward values
        if "steps" in df.columns:
            df = df.sort_values("steps")
        if "avg_episodic_reward" in df.columns:
            df["avg_episodic_reward"] = df["avg_episodic_reward"].interpolate().fillna(method="bfill").fillna(method="ffill")
        if "steps" in df.columns and "avg_episodic_reward" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df["steps"], df["avg_episodic_reward"], label="Reward")
            ax.set_xlabel("Training Steps")
            ax.set_ylabel("Average Episodic Reward")
            ax.set_title("Reward vs. Training Steps")
            ax.grid(True)
            st.pyplot(fig)
            # Show final performance summary
            final_score = df["avg_episodic_reward"].dropna().tail(10).mean()
            st.metric("Final Performance (mean last 10)", f"{final_score:.2f}")
        else:
            st.error("Invalid format in metrics.json.")
    else:
        st.info("metrics.json not found in this run.")

# --- Timing Analysis Tab ---
with tab2:
    st.subheader("Timing Profile ⏱️")
    timing_file = os.path.join(run_path, "timings.jsonl")
    if os.path.exists(timing_file):
        records = []
        for line in open(timing_file):
            entry = json.loads(line)
            for phase, info in entry.items():
                if isinstance(info, dict) and "avg_ms" in info:
                    records.append({"phase": phase, "avg_ms": info["avg_ms"]})
        df_t = pd.DataFrame(records)
        if not df_t.empty:
            phases = sorted(df_t["phase"].unique())
            for phase in phases:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                data = df_t[df_t["phase"] == phase]["avg_ms"]
                ax2.boxplot(data, notch=True, showfliers=False)
                ax2.set_title(f"{phase} (ms per call)")
                ax2.set_ylabel("Avg ms")
                st.pyplot(fig2)
        else:
            st.info("No timing records available.")
    else:
        st.info("timings.jsonl not found in this run.")

# --- Videos Tab ---
with tab3:
    st.subheader("Experiment Videos 🎥")
    video_files = []
    for root, dirs, files in os.walk(run_path):
        for f in files:
            if f.endswith((".mp4", ".webm", ".gif")):
                video_files.append(os.path.join(root, f))
    if video_files:
        for vid in video_files:
            st.video(vid)
    else:
        st.info("No videos found for this run.")

# --- Logs Tab ---
with tab4:
    st.subheader("Run Logs 📝")
    log_entries = [f for f in os.listdir(run_path) if f.endswith(".log")]
    if log_entries:
        log_path = os.path.join(run_path, log_entries[0])
        lines = open(log_path).read().splitlines()
        st.text_area("Last 50 lines of log", "".join(lines[-50:]), height=300)
    else:
        st.info("No log file found in this run.")
