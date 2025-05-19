#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import imageio
import numpy as np

from algorithms.base_agent import BaseAgent
from algorithms.ppo import PPO
from algorithms.grpo import GRPO_NoCritic
from utils import env_utils, torch_utils
from utils.pydantic_models import ExperimentConfig

# Algo registry must match what you used during training:
ALGO_REGISTRY = {
    "ppo": PPO,
    "grpo": GRPO_NoCritic,
}

def load_experiment_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    cfg['_config_file_path'] = str(config_path.resolve())
    return ExperimentConfig(**cfg)

def rollout_episode(env, agent, max_steps: int):
    """Returns list of RGB frames for one episode rollout."""
    obs = env.reset()
    frames = []
    for _ in range(max_steps):
        # render before action so you see initial state
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        with torch.no_grad():
            action = agent.get_action(obs)  # or agent.act(obs), depending on your API
        obs, reward, done, _ = env.step(action)
        if done:
            # capture final frame
            frames.append(env.render(mode='rgb_array'))
            break
    return frames

def main():
    p = argparse.ArgumentParser("Load saved actors and generate one‐episode videos")
    p.add_argument("--base-log-dir", "-ld", type=Path, default=Path("finalLogs"),
                   help="Top‐level folder containing all your runs")
    p.add_argument("--device", "-d", type=str, default=None,
                   help="torch device (cpu or cuda). Auto‐detect if not set.")
    args = p.parse_args()

    device = torch_utils.get_device(args.device)
    base = args.base_log_dir.expanduser()
    if not base.is_dir():
        print(f"[error] base‐log‐dir not found: {base}")
        return

    videos_dir = base / "videos"
    videos_dir.mkdir(exist_ok=True)

    # find every subfolder that has checkpoints/actor.pt
    for run_dir in base.iterdir():
        ckpt = run_dir / "checkpoints" / "actor.pt"
        cfg_file = run_dir / "config.json"
        if not ckpt.exists() or not cfg_file.exists():
            continue

        print(f"→ Generating video for run {run_dir.name}")
        # 1) load config
        config = load_experiment_config(cfg_file)

        # 2) build env
        env = env_utils.make_env(
            env_id=config.env_id,
            seed=config.seed,
            max_episode_steps=config.max_episode_steps
        )

        # 3) instantiate agent and load weights
        AlgoClass = ALGO_REGISTRY[config.algo.lower()]
        agent = AlgoClass(env=env, config=config, device_str=device)
        # assume your agent has a load() or similar:
        agent.load_checkpoint(str(ckpt))  

        # 4) rollout
        frames = rollout_episode(env, agent, config.max_episode_steps)

        # 5) save .mp4
        out_path = videos_dir / f"{run_dir.name}.mp4"
        imageio.mimsave(str(out_path), frames, fps=30)
        print(f"  → saved to {out_path}")

        env.close()

if __name__ == "__main__":
    main()
