import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils import (checkpoint_utils, env_utils, logging_utils,
                   torch_utils, video_plot_utils)
from utils.pydantic_models import ExperimentConfig # Using Pydantic
from utils.timing_utils import Timing
from utils.torch_utils import NpArray, Tensor, Device
from utils.video_plot_utils import Loggable # For metrics

# Supported distributions for policy networks
SUPPORTED_DISTRIBUTIONS = ["normal", "beta", "categorical"]


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(
        self,
        env: gym.Env,
        config: ExperimentConfig, # Use Pydantic model
        device_str: Optional[Device] = None,
    ) -> None:
        """
        Initializes the base agent.

        Args:
            env: The Gymnasium environment instance.
            config: Pydantic ExperimentConfig object.
            device_str: The torch device string (e.g., 'cpu', 'cuda'). Auto-detects if None.
        """
        self.env = env
        self.config = config # This config includes the original config.run_name if it was set
        self.device = torch_utils.get_device(device_str)

        # Common config parameters
        self.seed: int = config.seed
        self.gamma: float = config.gamma
        self.total_steps: int = config.total_steps
        self.log_interval: int = config.log_interval
        self.checkpoint_interval: int = config.checkpoint_interval
        self.video_interval: int = config.video_interval
        
        algo_specific_conf = config.get_algo_specific_config()
        
        # Construct the run name ALWAYS from detailed config parameters for directory naming.
        # The original config.run_name (if provided in JSON) is ignored for directory naming
        # but is preserved in the saved config.json within the run's directory.
        run_name_parts = [
            self.env.spec.id if self.env.spec else config.env_id,
            config.algo,
            f"seed{self.seed}"
        ]
        
        # Child agent (PPO/GRPO) sets self.entropy_coef in its __init__ before super()
        # This uses the value from algo_specific_conf which might have been swept.
        # Here, we use algo_specific_conf directly for consistency in naming.
        if hasattr(algo_specific_conf, 'entropy_coef'):
            run_name_parts.append(f"ent{getattr(algo_specific_conf, 'entropy_coef')}")
        
        if hasattr(algo_specific_conf, 'lr'):
            run_name_parts.append(f"lr{getattr(algo_specific_conf, 'lr')}")
        
        if hasattr(algo_specific_conf, 'distribution_type'):
            run_name_parts.append(f"{getattr(algo_specific_conf, 'distribution_type')}")

        if config.algo == "grpo" and hasattr(algo_specific_conf, 'group_size'):
             run_name_parts.append(f"g{getattr(algo_specific_conf, 'group_size')}")

        effective_run_name = "_".join(run_name_parts)

        base_dir = Path(config.base_log_dir) / effective_run_name
        self.log_dir = base_dir / "logs"
        self.ckpt_dir = base_dir / "checkpoints"
        self.vid_dir = base_dir / "videos"
        self.results_file = base_dir / "metrics.json"
        self.timings_file = base_dir / "timings.jsonl"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.vid_dir.mkdir(parents=True, exist_ok=True)

        log_level = logging.DEBUG if config.verbose else logging.INFO
        self.logger = logging_utils.get_logger(effective_run_name, self.log_dir, level=log_level, enabled=True)
        self.logger.info(f"Initializing agent for {effective_run_name} on device {self.device}")
        self.logger.info(f"Full Config (including original run_name='{config.run_name}'): {config.model_dump_json(indent=2)}")


        self.timer = Timing()

        # Network and optimizer placeholders (defined in subclasses)
        self.actor: Optional[nn.Module] = None
        self.critic: Optional[nn.Module] = None 
        self.actor_optimizer: Optional[Optimizer] = None
        self.critic_optimizer: Optional[Optimizer] = None
        
        self.cnn_feature_extractor: Optional[nn.Module] = None
        self.actor_head: Optional[nn.Module] = None
        self.critic_head: Optional[nn.Module] = None

        self.obs_shape: Tuple[int, ...] = env.observation_space.shape
        self.action_space: gym.Space = env.action_space

        if isinstance(env.action_space, gym.spaces.Box):
            self.action_dim: int = int(np.prod(env.action_space.shape))
            self.is_continuous: bool = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim: int = env.action_space.n # type: ignore
            self.is_continuous: bool = False
        else:
            raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")
        
        self._setup_networks_and_optimizers()


    @abstractmethod
    def _setup_networks_and_optimizers(self) -> None:
        """Initialize actor/critic networks and their optimizers."""
        pass

    @abstractmethod
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Dict[str, Any]]:
        """
        Selects an action based on the observation.

        Args:
            obs: Current environment observation.
            deterministic: If True, select action deterministically.

        Returns:
            Tuple of (action_env_scale, action_info).
            action_env_scale: Action scaled to environment's bounds.
            action_info: Dictionary with log_prob, value (if PPO), etc.
                         log_prob is for the action in its canonical range (e.g., [-1,1] for Normal pre-clip, [0,1] for Beta).
        """
        pass

    @abstractmethod
    def _rollout(self) -> Tuple[Any, Dict[str, Loggable]]: # Return type depends on buffer or batch data
        """
        Collects experience (rollout) from the environment.

        Returns:
            Tuple of (batch_data_or_buffer, rollout_info).
            - batch_data_or_buffer: Data collected, can be a buffer instance or raw batch dict.
            - rollout_info: Dictionary with summary stats (avg_reward, avg_length).
        """
        pass

    @abstractmethod
    def _update(self, data_source: Any) -> Dict[str, float]: # data_source can be buffer or batch
        """
        Performs a learning update.

        Args:
            data_source: Data for update (e.g., RolloutBuffer or a batch dict).

        Returns:
            Dictionary containing loss values and other update metrics.
        """
        pass


    def train(self) -> None:
        """Main training loop."""
        start_step = checkpoint_utils.load_checkpoint(self, self.ckpt_dir)
        metrics = video_plot_utils.load_metrics(self.results_file)
        
        global_step = start_step
        last_log_step = start_step - (start_step % self.log_interval) if start_step > 0 else 0
        last_ckpt_step = start_step - (start_step % self.checkpoint_interval) if start_step > 0 else 0
        last_vid_step = start_step - (start_step % self.video_interval) if start_step > 0 else 0


        self.logger.info(f"Starting training from step {global_step}...")

        while global_step < self.total_steps:
            with self.timer("rollout_phase"): 
                rollout_data, rollout_info = self._rollout()
            
            steps_this_rollout = rollout_info.get("steps_collected_this_rollout", 0)
            if steps_this_rollout == 0 and isinstance(rollout_data, dict) and "n_steps" in rollout_data: 
                steps_this_rollout = rollout_data.get("n_steps", 0)
            elif steps_this_rollout == 0 and hasattr(rollout_data, 'size'): 
                steps_this_rollout = rollout_data.size
            
            if steps_this_rollout == 0 :
                 self.logger.warning("Rollout collected 0 steps. Check rollout logic or batch info.")
                 if global_step == start_step : 
                     self.logger.error("Failed to collect any steps on first rollout. Aborting.")
                     break
            
            global_step += steps_this_rollout

            with self.timer("update_phase"): 
                update_info = self._update(rollout_data) 

            if global_step >= last_log_step + self.log_interval:
                avg_reward = rollout_info.get('avg_episodic_reward', np.nan)
                avg_length = rollout_info.get('avg_episode_length', np.nan)

                metrics["steps"].append(global_step)
                metrics["avg_episodic_reward"].append(float(avg_reward) if not np.isnan(avg_reward) else None)
                metrics["avg_episode_length"].append(float(avg_length) if not np.isnan(avg_length) else None)


                self.logger.info(
                    f"Step: {global_step}/{self.total_steps} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f}"
                )
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in update_info.items()])
                self.logger.info(f"Update Info: {loss_str}")

                timing_summary = self.timer.summary(reset=True)
                video_plot_utils.save_timings(timing_summary, self.timings_file, global_step)
                timing_str = " | ".join([f"{k}: {v['avg_ms']:.2f}ms" for k, v in timing_summary.items()])
                self.logger.info(f"Timings (avg ms): {timing_str}")

                video_plot_utils.save_metrics(metrics, self.results_file)
                last_log_step = global_step

            if global_step >= last_ckpt_step + self.checkpoint_interval:
                checkpoint_utils.save_checkpoint(self, self.ckpt_dir, global_step)
                last_ckpt_step = global_step

            if global_step >= last_vid_step + self.video_interval:
                self.evaluate_and_record_video(num_episodes=3, current_step=global_step)
                last_vid_step = global_step
            
            if global_step >= self.total_steps:
                break

        self.logger.info(f"Training finished at step {global_step}.")
        checkpoint_utils.save_checkpoint(self, self.ckpt_dir, global_step)
        video_plot_utils.save_metrics(metrics, self.results_file)
        self.evaluate_and_record_video(num_episodes=1, current_step=global_step, prefix="final")
        self.env.close()


    def evaluate_and_record_video(
        self,
        num_episodes: int = 3,
        current_step: int = 0,
        deterministic: bool = True,
        prefix: str = "eval"
    ) -> None:
        """Runs evaluation episodes and saves the best one as a video."""
        self.logger.info(f"Starting evaluation ({num_episodes} episodes, deterministic={deterministic})...")
        
        eval_env_seed = self.seed + 1000 + current_step 
        eval_env = env_utils.make_env(
            self.env.spec.id if self.env.spec else self.config.env_id, 
            render_mode="rgb_array",
            seed=eval_env_seed,
            max_episode_steps=self.config.max_episode_steps
        )

        best_reward_eval = -float('inf')
        best_frames_eval: List[NpArray] = []
        eval_rewards_list: List[float] = []

        actor_is_training = self.actor.training if self.actor else False
        critic_is_training = self.critic.training if self.critic else False
        cnn_is_training = self.cnn_feature_extractor.training if self.cnn_feature_extractor else False
        actor_head_is_training = self.actor_head.training if self.actor_head else False
        critic_head_is_training = self.critic_head.training if self.critic_head else False


        if self.actor: self.actor.eval()
        if self.critic: self.critic.eval()
        if self.cnn_feature_extractor: self.cnn_feature_extractor.eval()
        if self.actor_head: self.actor_head.eval()
        if self.critic_head: self.critic_head.eval()

        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            terminated, truncated = False, False
            ep_total_reward = 0.0
            ep_frames: List[NpArray] = []
            ep_step_count = 0
            max_eval_ep_len = getattr(eval_env, "_max_episode_steps", self.config.total_steps)


            while not (terminated or truncated) and ep_step_count < max_eval_ep_len :
                with torch.no_grad(), self.timer("eval_action_select"): 
                    action_env, _ = self.get_action(obs, deterministic=deterministic)
                
                with self.timer("eval_env_interact"):
                    obs, reward, terminated, truncated, _ = eval_env.step(action_env)
                
                ep_total_reward += reward
                ep_step_count += 1
                
                try:
                    frame = eval_env.render()
                    if frame is not None:
                        ep_frames.append(video_plot_utils.overlay_text(
                            frame, f"Ep: {ep+1} Step: {ep_step_count} R: {ep_total_reward:.2f}"
                        ))
                except Exception as e:
                    self.logger.warning(f"Render error during evaluation: {e}")
            
            eval_rewards_list.append(ep_total_reward)
            if ep_total_reward > best_reward_eval:
                best_reward_eval = ep_total_reward
                best_frames_eval = ep_frames
            self.logger.debug(f"Eval Ep {ep+1}: Reward={ep_total_reward:.2f}, Steps={ep_step_count}")

        eval_env.close()
        avg_eval_reward_val = np.mean(eval_rewards_list) if eval_rewards_list else np.nan
        self.logger.info(f"Evaluation Complete: Avg Reward = {avg_eval_reward_val:.2f} (over {num_episodes} eps)")

        if best_frames_eval:
            video_filename = self.vid_dir / f"{prefix}_step{current_step}_det{deterministic}_bestR{best_reward_eval:.1f}.mp4"
            video_plot_utils.save_video(best_frames_eval, video_filename)
            self.logger.info(f"Saved best evaluation video: {video_filename.name}")

        if self.actor and actor_is_training: self.actor.train()
        if self.critic and critic_is_training: self.critic.train()
        if self.cnn_feature_extractor and cnn_is_training: self.cnn_feature_extractor.train()
        if self.actor_head and actor_head_is_training: self.actor_head.train()
        if self.critic_head and critic_head_is_training: self.critic_head.train()