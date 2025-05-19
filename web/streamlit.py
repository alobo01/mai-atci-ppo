import os
import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
EXP_DIR = st.sidebar.text_input("Experiment Logs Directory", "finalLogs")

# Page setup
st.set_page_config(page_title="RL Experiment Dashboard", layout="wide")
st.title("📊 Reinforcement Learning Experiment Dashboard")
st.markdown(
    "Welcome! Use this dashboard to explore your PPO and GRPO experiment runs across different environments."
)

# Sidebar: experiment filtering & selection
st.sidebar.header("Experiment Selection")
# List all subdirectories in EXP_DIR
try:
    all_runs = [d for d in os.listdir(EXP_DIR) if os.path.isdir(os.path.join(EXP_DIR, d))]
except FileNotFoundError:
    st.error(f"Directory '{EXP_DIR}' not found. Please check the path.")
    st.stop()
# Extract unique environments from folder names
envs = sorted({run.split("_")[0] for run in all_runs})
selected_env = st.sidebar.selectbox("Environment", ["All"] + envs)
# Filter runs by selected environment
runs = [r for r in all_runs if (selected_env == "All" or r.startswith(selected_env))]
selected_run = st.sidebar.selectbox("Run Directory", runs)

# If no run selected, prompt and exit
if not selected_run:
    st.warning("No runs found in the specified directory.")
    st.stop()

# Paths for the selected run
run_path = os.path.join(EXP_DIR, selected_run)

# --- Display configuration ---
config_file = os.path.join(run_path, "config.json")
config = {}
if os.path.exists(config_file):
    with open(config_file) as f:
        config = json.load(f)
st.sidebar.subheader("Experiment Configuration")
if config:
    st.sidebar.json(config, expanded=False)
else:
    st.sidebar.info("No config.json found for this run.")

# Quick metrics display
col1, col2, col3, col4 = st.columns(4)
col1.metric("Environment", config.get("env_id", selected_env))
algo = config.get("algo", selected_run.split("_")[1].upper())
dist = config.get("grpo_config", {}).get("distribution_type", config.get("ppo_config", {}).get("distribution_type", "normal")).upper()
gsize = config.get("grpo_config", {}).get("group_size", "-")
col2.metric("Algorithm", algo)
col3.metric("Distribution", dist)
col4.metric("Group Size", gsize)

# --- Quick RL Algorithm Overview ---
with st.expander("🔍 What are PPO and GRPO?"):
    st.markdown(
        """
        - **PPO (Proximal Policy Optimization):**
          An on-policy RL algorithm that stabilizes training by clipping policy updates.
        - **GRPO (Grouped PPO):**
          Builds on PPO by maintaining a reference policy and grouping updates for lower variance and improved exploration.
        """
    )

# Tabs for interactive views
tab1, tab2, tab3, tab4 = st.tabs(["Learning Curve", "Timing Analysis", "Videos", "Logs"])

# --- Learning Curve Tab ---
with tab1:
    st.subheader("Learning Curve 📈")
    metrics_file = os.path.join(run_path, "metrics.json")
    if os.path.exists(metrics_file):
        df = pd.read_json(metrics_file)
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
        if records:
            df_t = pd.DataFrame(records)
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
    # Search for .log files
    log_entries = []
    for f in os.listdir(run_path):
        if f.endswith(".log"):
            log_entries.append(os.path.join(run_path, f))
    if log_entries:
        log_path = log_entries[0]
        lines = open(log_path).read().splitlines()
        st.text_area("Last 50 lines of log", "\n".join(lines[-50:]), height=300)
    else:
        st.info("No log file found in this run.")
