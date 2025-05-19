#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import imageio
import numpy as np
from pydantic import ValidationError

import gymnasium as gym

from algorithms.ppo import PPO
from algorithms.grpo import GRPO_NoCritic
from utils import torch_utils
from utils.pydantic_models import ExperimentConfig

ALGO_REGISTRY = {
    "ppo": PPO,
    "grpo": GRPO_NoCritic,
}

def load_experiment_config(config_path: Path) -> ExperimentConfig:
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    algo = cfg.get("algo", "").lower()
    if algo == "ppo" and "grpo_config" in cfg:
        cfg.pop("grpo_config", None)
    elif algo == "grpo" and "ppo_config" in cfg:
        cfg.pop("ppo_config", None)

    try:
        return ExperimentConfig(**cfg)
    except ValidationError as e:
        print(f"[warning] config validation failed: {e}. Falling back to un‐validated construct().")
        return ExperimentConfig.construct(**cfg)

def rollout_episode(env, agent, max_steps: int):
    """
    Roll out one episode, collecting frames.
    Stops when env signals done or when max_steps is reached.
    """
    # Gymnasium reset returns (obs, info)
    obs, _ = env.reset()

    frames = []
    for _ in range(max_steps):
        # always returns an RGB array because of render_mode
        frame = env.render()
        frames.append(frame)

        # agent.get_action may return (action, extras) or action directly
        out = agent.get_action(obs)
        action = out[0] if isinstance(out, tuple) else out

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            frames.append(env.render())
            break

    return frames

def main():
    parser = argparse.ArgumentParser("Generate one-episode videos from saved actors")
    parser.add_argument(
        "--base-log-dir", "-ld",
        type=Path,
        default=Path("finalLogs"),
        help="Top‐level folder containing individual run subfolders"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Torch device (cpu or cuda). Auto‐detect if not set."
    )
    args = parser.parse_args()

    device = torch_utils.get_device(args.device)
    base = args.base_log_dir.expanduser()
    if not base.is_dir():
        print(f"[error] base-log-dir not found: {base}")
        return

    videos_dir = base / "videos"
    videos_dir.mkdir(exist_ok=True)

    for run_dir in sorted(base.iterdir()):
        ckpt = run_dir / "checkpoints" / "actor.pt"
        cfg_file = run_dir / "config.json"
        if not ckpt.exists() or not cfg_file.exists():
            continue

        print(f"\n→ Generating video for run: {run_dir.name}")

        # 1) load config
        config = load_experiment_config(cfg_file)

        # 2) make env without TimeLimit, but with rgb_array rendering
        env = gym.make(
            config.env_id,
            render_mode="rgb_array"
        )
        if config.seed is not None:
            env.reset(seed=config.seed)

        # 3) instantiate agent
        AlgoClass = ALGO_REGISTRY[config.algo]
        agent = AlgoClass(env=env, config=config, device_str=device)

        # 4) load actor weights
        state_dict = torch.load(ckpt, map_location=device)
        agent.actor.load_state_dict(state_dict)
        agent.actor.eval()

        # 5) rollout up to max_episode_steps (or default 10k)
        max_steps = int(config.max_episode_steps or 10_000)
        frames = rollout_episode(env, agent, max_steps)

        # 6) save mp4
        out_path = videos_dir / f"{run_dir.name}.mp4"
        imageio.mimsave(str(out_path), frames, fps=30)
        print(f"  ↳ saved video to: {out_path}")

        env.close()

if __name__ == "__main__":
    main()
