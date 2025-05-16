"""
Abstract Base Class for RL Agents.

Author: Antonio Lobo
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

import utils
from utils import _GLFW_LOCK, NpArray, Tensor, Device, Loggable, Timing


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        device: Optional[Device] = None,
    ) -> None:
        """
        Initializes the base agent.

        Args:
            env: The Gymnasium environment instance.
            config: Dictionary of hyperparameters and settings.
            device: The torch device to use (e.g., 'cpu', 'cuda'). Auto-detects if None.
        """
        self.env = env
        self.config = config
        self.device = utils.get_device(device)

        # Extract common config parameters with defaults
        self.seed: int = config.get("seed", 0)
        self.gamma: float = config.get("gamma", 0.99)
        self.total_steps: int = config.get("total_steps", 1_000_000)
        self.log_interval: int = config.get("log_interval", 5000) # Steps between logging
        self.checkpoint_interval: int = config.get("checkpoint_interval", 50000) # Steps between saving
        self.video_interval: int = config.get("video_interval", 100_000) # Steps between eval videos

        # Setup directories (relative to script execution or configurable root)
        run_name = config.get("run_name", f"{self.env.spec.id}_{config['algo']}_seed{self.seed}")
        base_dir = Path(config.get("base_log_dir", "experiment_runs")) / run_name
        self.log_dir = base_dir / "logs"
        self.ckpt_dir = base_dir / "checkpoints"
        self.vid_dir = base_dir / "videos"
        self.results_file = base_dir / "metrics.json"
        self.timings_file = base_dir / "timings.jsonl"

        # Ensure directories exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vid_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.verbose: bool = config.get("verbose", False)
        log_level = logging.DEBUG if self.verbose else logging.INFO
        self.logger = utils.get_logger(run_name, self.log_dir, level=log_level, enabled=True)
        self.logger.info(f"Initializing agent for {run_name} on device {self.device}")
        self.logger.info(f"Config: {config}")

        # Initialize timing utility
        self.timer = Timing()

        # Placeholders for networks and optimizers (to be defined in subclasses)
        self.actor: Optional[nn.Module] = None
        self.critic: Optional[nn.Module] = None
        self.actor_optimizer: Optional[Optimizer] = None
        self.critic_optimizer: Optional[Optimizer] = None

        # Observation and action space info
        self.obs_shape = env.observation_space.shape
        self.action_space = env.action_space
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim = int(np.prod(env.action_space.shape))
            self.is_continuous = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = env.action_space.n
            self.is_continuous = False
        else:
            raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")


    @abstractmethod
    def _setup_networks_and_optimizers(self) -> None:
        """Initialize actor/critic networks and their optimizers."""
        pass

    @abstractmethod
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Any]:
        """
        Selects an action based on the observation.

        Args:
            obs: Current environment observation.
            deterministic: If True, select action deterministically (e.g., mean).

        Returns:
            Tuple of (action, action_info), where action_info might contain
            log_prob, value estimate, etc., depending on the algorithm.
        """
        pass

    @abstractmethod
    def _rollout(self) -> Tuple[Dict[str, Any], Dict[str, Loggable]]:
        """
        Collects experience (rollout) from the environment.

        Returns:
            Tuple of (batch_data, rollout_info).
            - batch_data: Dictionary containing collected transitions (obs, actions, etc.).
            - rollout_info: Dictionary with summary stats for the rollout
                             (e.g., avg_reward, avg_length).
        """
        pass

    @abstractmethod
    def _update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Performs a learning update using the collected batch data.

        Args:
            batch: Data collected during the rollout.

        Returns:
            Dictionary containing loss values and other update metrics.
        """
        pass


    def train(self) -> None:
        """Main training loop."""
        self._setup_networks_and_optimizers() # Ensure networks are ready

        # Load checkpoint if exists
        start_step = utils.load_checkpoint(self, self.ckpt_dir)
        metrics = utils.load_metrics(self.results_file)
        global_step = start_step
        last_log_step = start_step
        last_ckpt_step = start_step
        last_vid_step = start_step

        self.logger.info(f"Starting training from step {global_step}...")

        while global_step < self.total_steps:
            # --- Rollout Phase ---
            with self.timer("rollout"):
                batch, rollout_info = self._rollout()
            steps_collected = batch.get("n_steps", 0) # Assuming batch contains this info
            if steps_collected == 0:
                self.logger.warning("Rollout collected 0 steps. Check rollout logic.")
                continue
            global_step += steps_collected

            # --- Update Phase ---
            with self.timer("update"):
                update_info = self._update(batch)

            # --- Logging ---
            if global_step >= last_log_step + self.log_interval:
                avg_reward = rollout_info.get('avg_episodic_reward', np.nan)
                avg_length = rollout_info.get('avg_episode_length', np.nan)

                metrics["steps"].append(global_step)
                metrics["avg_episodic_reward"].append(float(avg_reward))
                metrics["avg_episode_length"].append(float(avg_length))

                # Log performance
                self.logger.info(
                    f"Step: {global_step}/{self.total_steps} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f}"
                )
                # Log losses and other update metrics
                loss_str = " | ".join([f"{k}: {v:.3f}" for k, v in update_info.items()])
                self.logger.info(f"Update Info: {loss_str}")

                # Log timings
                timing_summary = self.timer.summary(reset=True) # Reset timer after logging
                utils.save_timings(timing_summary, self.timings_file, global_step)
                timing_str = " | ".join([f"{k}: {v['avg_ms']:.2f}ms" for k, v in timing_summary.items()])
                self.logger.info(f"Timings (avg ms): {timing_str}")

                # Save metrics to file
                utils.save_metrics(metrics, self.results_file)
                last_log_step = global_step

            # --- Checkpointing ---
            if global_step >= last_ckpt_step + self.checkpoint_interval:
                utils.save_checkpoint(self, self.ckpt_dir, global_step)
                last_ckpt_step = global_step

            # --- Evaluation & Video Recording ---
            if global_step >= last_vid_step + self.video_interval:
                self.evaluate_and_record_video(num_episodes=3, current_step=global_step)
                last_vid_step = global_step

        self.logger.info("Training finished.")
        # Final checkpoint and metrics save
        utils.save_checkpoint(self, self.ckpt_dir, global_step)
        utils.save_metrics(metrics, self.results_file)
        # Final video
        self.evaluate_and_record_video(num_episodes=1, current_step=global_step, prefix="final")
        self.env.close() # Close the main training env


    def evaluate_and_record_video(
        self,
        num_episodes: int = 3,
        current_step: int = 0,
        deterministic: bool = True,
        prefix: str = "eval"
    ) -> None:
        """Runs evaluation episodes and saves the best one as video."""
        self.logger.info(f"Starting evaluation ({num_episodes} episodes, deterministic={deterministic})...")
        eval_env = utils.make_env(
            self.env.spec.id,
            render_mode="rgb_array",
            seed=self.seed + 1000 + current_step # Use a different seed for eval
        )
        best_reward = -float('inf')
        best_frames: List[NpArray] = []
        eval_rewards: List[float] = []

        # Ensure networks are in evaluation mode
        if self.actor: self.actor.eval()
        if self.critic: self.critic.eval()

        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            frames: List[NpArray] = []
            ep_steps = 0
            max_ep_len = getattr(eval_env, "_max_episode_steps", 1000)

            while not done and ep_steps < max_ep_len:
                with torch.no_grad(), self.timer("eval_action"):
                    action, _ = self.get_action(obs, deterministic=deterministic)

                with self.timer("eval_env_step"):
                    obs, reward, terminated, truncated, _ = eval_env.step(action)

                done = terminated or truncated
                total_reward += reward
                ep_steps += 1
                with _GLFW_LOCK:
                    try:
                        frame = eval_env.render()
                    except Exception as e:
                        self.logger.warning(f"Render error: {e}")
                        frame = None
                if frame is not None:
                    frames.append(utils.overlay_text(frame, f"Ep: {ep+1} Step: {ep_steps} R: {total_reward:.2f}"))

            eval_rewards.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                best_frames = frames
            self.logger.debug(f"Eval Ep {ep+1}: Reward={total_reward:.2f}, Steps={ep_steps}")

        eval_env.close()
        avg_eval_reward = np.mean(eval_rewards)
        self.logger.info(f"Evaluation Complete: Avg Reward = {avg_eval_reward:.2f} (over {num_episodes} episodes)")

        if best_frames:
            video_filename = self.vid_dir / f"{prefix}_step{current_step}_det{deterministic}_bestR{best_reward:.1f}.mp4"
            utils.save_video(best_frames, video_filename)
            self.logger.info(f"Saved best evaluation video: {video_filename.name}")

        # Restore training mode for networks
        if self.actor: self.actor.train()
        if self.critic: self.critic.train()