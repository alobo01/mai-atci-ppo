import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution

from algorithms.base_agent import BaseAgent, SUPPORTED_DISTRIBUTIONS
from algorithms.buffer import RolloutBuffer
from networks.mlp import FeedForwardNN
from networks.cnn import CNNFeatureExtractor
from utils import distribution_utils, torch_utils
from utils.pydantic_models import ExperimentConfig, PPOConfig
from utils.torch_utils import NpArray, Tensor, Device
from utils.video_plot_utils import Loggable


class PPO(BaseAgent):
    """Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        env: gym.Env,
        config: ExperimentConfig,
        device_str: Optional[Device] = None,
    ) -> None:
        

        algo_config = config.get_algo_specific_config()
        if not isinstance(algo_config, PPOConfig):
            raise TypeError("PPO agent requires PPOConfig.")
        self.ppo_cfg: PPOConfig = algo_config
        
        if self.ppo_cfg.distribution_type not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution type: {self.ppo_cfg.distribution_type}. "
                             f"Supported are: {SUPPORTED_DISTRIBUTIONS}")

        # PPO specific hyperparameters
        self.lam: float = self.ppo_cfg.lam
        self.clip_eps: float = self.ppo_cfg.clip_eps
        self.ppo_epochs: int = self.ppo_cfg.ppo_epochs
        self.num_minibatches: int = self.ppo_cfg.num_minibatches
        self.entropy_coef: float = self.ppo_cfg.entropy_coef
        self.value_coef: float = self.ppo_cfg.value_coef
        self.max_grad_norm: float = self.ppo_cfg.max_grad_norm
        self.target_kl: Optional[float] = self.ppo_cfg.target_kl
        self.rollout_steps: int = self.ppo_cfg.rollout_steps

        # Network type specific setup is in _setup_networks_and_optimizers
        self.network_cfg = config.network_config

        # Batch and minibatch size calculation
        self.batch_size = self.rollout_steps # Assuming one env for now
        if self.batch_size % self.num_minibatches != 0:
            # Adjust num_minibatches to be a divisor, or adjust batch_size
            # For simplicity, let's adjust num_minibatches to be as close as possible
            # while being a divisor.
            self.num_minibatches = max(1, round(self.batch_size / (self.batch_size // self.num_minibatches)))
            self.logger.warning(
                f"rollout_steps ({self.batch_size}) not perfectly divisible by "
                f"num_minibatches ({self.ppo_cfg.num_minibatches}). "
                f"Adjusted num_minibatches to {self.num_minibatches}."
            )
        self.minibatch_size = self.batch_size // self.num_minibatches

        # After specific config setup, call the parent constructor
        super().__init__(env, config, device_str) # _setup_networks_and_optimizers is called by super

        # Initialize Rollout Buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_shape=self.obs_shape,
            action_shape=(self.action_dim,) if self.is_continuous else (), # Empty tuple for discrete scalar
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.lam,
            is_continuous=self.is_continuous
        )
        
        # Internal state for rollout
        self._current_obs: NpArray = self.env.reset(seed=self.seed)[0]
        self._current_episode_reward: float = 0.0
        self._current_episode_length: int = 0
        self._rollout_episode_rewards: List[float] = [] # Rewards from episodes *completed in this rollout*
        self._rollout_episode_lengths: List[int] = [] # Lengths from episodes *completed in this rollout*


    def _setup_networks_and_optimizers(self) -> None:
        """Initializes PPO's actor and critic networks and optimizers."""
        lr = self.ppo_cfg.lr # Use LR from PPOConfig
        
        # Actor output dim depends on distribution type
        if self.ppo_cfg.distribution_type in ["normal", "beta"]:
            actor_output_dim = self.action_dim * 2 # mean+std or alpha+beta
        elif self.ppo_cfg.distribution_type == "categorical":
            actor_output_dim = self.action_dim # logits
        else:
            raise ValueError(f"Unhandled distribution type for actor_output_dim: {self.ppo_cfg.distribution_type}")
            
        critic_output_dim = 1

        if self.network_cfg.network_type == "mlp":
            self.actor = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                actor_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)
            self.critic = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                critic_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, eps=1e-5)
            self.logger.info("Using separate MLP Actor and Critic networks.")

        elif self.network_cfg.network_type == "cnn":
            self.cnn_feature_extractor = CNNFeatureExtractor(
                obs_shape=self.obs_shape, # Assumes (C, H, W) or handled by CNN
                output_features=self.network_cfg.cnn_output_features
            ).to(self.device)
            self.actor_head = FeedForwardNN(
                input_dim=self.network_cfg.cnn_output_features,
                output_dim=actor_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)
            self.critic_head = FeedForwardNN(
                input_dim=self.network_cfg.cnn_output_features,
                output_dim=critic_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)

            # Option 1: Separate optimizers (original choice for clarity)
            actor_params = list(self.cnn_feature_extractor.parameters()) + list(self.actor_head.parameters())
            critic_params = list(self.cnn_feature_extractor.parameters()) + list(self.critic_head.parameters())
            # Note: If base is shared, its gradients will be summed from both actor and critic losses.
            # This is a common setup.
            self.actor_optimizer = Adam(actor_params, lr=lr, eps=1e-5)
            self.critic_optimizer = Adam(critic_params, lr=lr, eps=1e-5) # Critic needs its own if base is updated by it too
            self.logger.info("Using CNN Feature Extractor with separate MLP heads and optimizers (shared base gradients will sum).")

        else:
             raise ValueError(f"Unknown network type: {self.network_cfg.network_type}")

    def _get_features(self, obs: Tensor) -> Tensor:
        """Extracts features from observations."""
        if self.network_cfg.network_type == "cnn":
            if self.cnn_feature_extractor is None:
                raise RuntimeError("CNN feature extractor not initialized.")
            with self.timer("cnn_feature_pass"):
                return self.cnn_feature_extractor(obs)
        return obs # For MLP, obs are features

    def _get_distribution_from_features(self, features: Tensor) -> Tuple[Distribution, distribution_utils.ActionPostprocessor]:
        """Gets action distribution from features."""
        actor_module = self.actor_head if self.network_cfg.network_type == "cnn" else self.actor
        if actor_module is None:
            raise RuntimeError("Actor module (or head) not initialized.")
        
        with self.timer("actor_pass"): # Renamed timer key
            actor_output = actor_module(features)
        
        return distribution_utils.create_distribution_from_actor_output(
            actor_output, self.ppo_cfg.distribution_type, self.action_space, self.action_dim
        )

    def _get_value_from_features(self, features: Tensor) -> Tensor:
        """Gets value estimate from features."""
        critic_module = self.critic_head if self.network_cfg.network_type == "cnn" else self.critic
        if critic_module is None:
            raise RuntimeError("Critic module (or head) not initialized.")
        
        with self.timer("critic_pass"): # Renamed timer key
            value = critic_module(features)
        return value.squeeze(-1) # Ensure shape (batch_size,)

    @torch.no_grad()
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Dict[str, Any]]:
        """
        Selects an action, computes value estimate and log_prob.
        log_prob is for the action in its canonical range.
        """
        obs_tensor = torch_utils.to_tensor(obs, self.device).unsqueeze(0)
        features = self._get_features(obs_tensor)
        
        dist, postprocessor = self._get_distribution_from_features(features)
        value = self._get_value_from_features(features)

        if deterministic and False:
            if self.ppo_cfg.distribution_type == "normal":
                # For Normal, mean is the deterministic action before any postprocessing (like clipping)
                # The distribution `dist` is Independent(Normal(...))
                action_canonical = dist.base_dist.mean # type: ignore
            elif self.ppo_cfg.distribution_type == "beta":
                # For Beta, mean is a good deterministic choice (in [0,1])
                action_canonical = dist.mean # type: ignore
            elif self.ppo_cfg.distribution_type == "categorical":
                action_canonical = torch.argmax(dist.logits, dim=-1, keepdim=not self.is_continuous) # type: ignore
            else:
                self.logger.warning(f"Deterministic action not fully defined for {self.ppo_cfg.distribution_type}, sampling.")
                action_canonical = dist.sample()
        else:
            action_canonical = dist.sample()

        log_prob = dist.log_prob(action_canonical) # Log_prob of the canonical action
        action_env_scale = postprocessor(action_canonical).squeeze(0).cpu().numpy()

        action_info = {
            "value": value.squeeze(0),      # Scalar tensor
            "log_prob": log_prob.squeeze(0), # Scalar tensor (log_prob of canonical action)
            "action_canonical": action_canonical.squeeze(0) # Store canonical action for _update
        }
        return action_env_scale, action_info

    def _rollout(self) -> Tuple[RolloutBuffer, Dict[str, Loggable]]:
        """Collects PPO rollout data and stores it in the RolloutBuffer."""
        self.buffer.reset()
        self._rollout_episode_rewards.clear()
        self._rollout_episode_lengths.clear()
        
        rollout_start_time = time.perf_counter()

        for step in range(self.rollout_steps):
            action_env_scale, info = self.get_action(self._current_obs, deterministic=False)
            value = info["value"].item()         # From action_info
            log_prob = info["log_prob"].item()   # From action_info

            next_obs, reward, terminated, truncated, _ = self.env.step(action_env_scale)
            done = terminated or truncated

            # Add to buffer (episode_start is True if current step is start of new ep)
            # This is tricky: we know an episode ended on the previous step if `done` is True now.
            # So, the current `self._current_obs` is the start of a new episode if the previous step resulted in `done`.
            # Let's track `ep_start_flag` explicitly.
            
            self.buffer.add(
                self._current_obs,
                action_env_scale, # Store action in env scale, will be unscaled for Beta in _update
                reward,
                self._current_episode_length == 0, # True if this is the first step of current episode
                value,
                log_prob,
                info["action_canonical"]
            )

            self._current_obs = next_obs
            self._current_episode_reward += reward
            self._current_episode_length += 1

            if done:
                self._rollout_episode_rewards.append(self._current_episode_reward)
                self._rollout_episode_lengths.append(self._current_episode_length)
                if self.config.verbose:
                    self.logger.debug(f"Rollout: Episode finished. Reward={self._current_episode_reward:.2f}, Length={self._current_episode_length}")
                
                self._current_obs, _ = self.env.reset()
                self._current_episode_reward = 0.0
                self._current_episode_length = 0
                # For GAE: if an episode ends, the "last_value" for that trajectory segment is 0
                # The buffer's finish_path will be called with last_value=0 if the env was 'done'
                # OR with the current critic's estimate of next_obs if rollout ends mid-episode.
                # We call finish_path only once after the full rollout_steps.

        # Compute GAE for the completed rollout
        with torch.no_grad():
            last_obs_tensor = torch_utils.to_tensor(self._current_obs, self.device).unsqueeze(0)
            features_last = self._get_features(last_obs_tensor)
            last_value = self._get_value_from_features(features_last).item()
        
        self.buffer.finish_path(last_value=last_value if not done else 0.0) # If last step was 'done', last_value is 0

        rollout_duration = time.perf_counter() - rollout_start_time
        rollout_info: Dict[str, Loggable] = {
            "rollout_duration_s": rollout_duration,
            "steps_per_second": self.rollout_steps / rollout_duration if rollout_duration > 0 else 0,
            "steps_collected_this_rollout": self.rollout_steps,
        }
        if self._rollout_episode_rewards:
            rollout_info["avg_episodic_reward"] = np.mean(self._rollout_episode_rewards)
            rollout_info["avg_episode_length"] = np.mean(self._rollout_episode_lengths)
        else: # No episode completed in this rollout
            rollout_info["avg_episodic_reward"] = np.nan 
            rollout_info["avg_episode_length"] = np.nan
        
        return self.buffer, rollout_info


    def _update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Performs PPO update steps using data from the RolloutBuffer."""
        all_losses = {"policy_loss": [], "value_loss": [], "entropy_loss": [], "approx_kl": [], "clip_fraction": []}

        # Ensure networks are in training mode
        if self.actor: self.actor.train()
        if self.critic: self.critic.train()
        if self.cnn_feature_extractor: self.cnn_feature_extractor.train()
        if self.actor_head: self.actor_head.train()
        if self.critic_head: self.critic_head.train()

        # Prepare action space bounds for Beta unscaling
        action_space_low_t = torch_utils.to_tensor(self.action_space.low, self.device, dtype=torch.float32)
        action_space_high_t = torch_utils.to_tensor(self.action_space.high, self.device, dtype=torch.float32)

        for epoch in range(self.ppo_epochs):
            num_samples_processed_this_epoch = 0
            for batch in buffer.get(batch_size=self.minibatch_size):
                mb_obs = batch.observations
                mb_actions_env_scale = batch.actions
                mb_actions_canonical_unclipped = batch.actions_canonical_unclipped # Use this for log_prob
                mb_old_log_probs = batch.log_probs # These are log_probs of canonical actions
                mb_advantages = batch.advantages
                mb_returns = batch.returns
                
                # Compute new log_probs, values, entropy
                features = self._get_features(mb_obs)
                dist, _ = self._get_distribution_from_features(features)
                current_values = self._get_value_from_features(features)

                # For log_prob calculation, actions need to be in the distribution's canonical space
                if self.ppo_cfg.distribution_type == "beta":
                    # Beta dist expects actions in [0,1]. Unscale from env_scale.
                    actions_for_log_prob = distribution_utils.unscale_action_for_beta_log_prob(
                        mb_actions_env_scale, action_space_low_t, action_space_high_t
                    )
                elif self.ppo_cfg.distribution_type == "normal":
                    # actions_canonical = distribution_utils.unscale_action_for_beta_log_prob(
                    #     mb_actions_env_scale, action_space_low_t, action_space_high_t
                    # )
                    actions_for_log_prob = mb_actions_canonical_unclipped
                    actions_canonical = mb_actions_env_scale # Use env scale actions if Normal is not squashed
                elif self.ppo_cfg.distribution_type == "categorical":
                    actions_for_log_prob = mb_actions_env_scale # Discrete actions are already canonical
                else:
                    raise NotImplementedError
                
                current_log_probs = dist.log_prob(actions_for_log_prob)
                entropy = dist.entropy().mean()

                # Policy Loss
                log_ratio = current_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value Loss
                value_loss = 0.5 * F.mse_loss(current_values, mb_returns)
                total_loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

                # Optimization
                if self.actor_optimizer: self.actor_optimizer.zero_grad()
                if self.critic_optimizer: self.critic_optimizer.zero_grad() # Separate critic opt
                
                with self.timer("backward_pass"): total_loss.backward()
                
                # Grad clipping
                params_to_clip = []
                if self.network_cfg.network_type == "cnn":
                    # If optimizers are separate for actor/critic paths sharing base:
                    if self.cnn_feature_extractor: params_to_clip.extend(self.cnn_feature_extractor.parameters())
                    if self.actor_head: params_to_clip.extend(self.actor_head.parameters())
                    if self.critic_head: params_to_clip.extend(self.critic_head.parameters())
                else: # MLP
                    if self.actor: params_to_clip.extend(self.actor.parameters())
                    if self.critic: params_to_clip.extend(self.critic.parameters())
                
                trainable_params = [p for p in params_to_clip if p.requires_grad]
                if trainable_params:
                    nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

                with self.timer("optimizer_step"):
                    if self.actor_optimizer: self.actor_optimizer.step()
                    if self.critic_optimizer: self.critic_optimizer.step() # Step critic opt

                # Track metrics
                with torch.no_grad():
                    approx_kl = (log_ratio).mean().item()
                    clipped = ratio.gt(1 + self.clip_eps) | ratio.lt(1 - self.clip_eps)
                    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                
                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["value_loss"].append(value_loss.item())
                all_losses["entropy_loss"].append(entropy.item())
                all_losses["approx_kl"].append(approx_kl)
                all_losses["clip_fraction"].append(clip_fraction)
                num_samples_processed_this_epoch += mb_obs.shape[0]

            # KL Early Stopping
            if self.target_kl is not None:
                # Ensure there are KL values for this epoch before averaging
                kls_this_epoch = all_losses["approx_kl"][- (num_samples_processed_this_epoch // self.minibatch_size) :]
                if kls_this_epoch: # Check if list is not empty
                    avg_kl_epoch = np.mean(kls_this_epoch)
                    if avg_kl_epoch > 1.5 * self.target_kl:
                        self.logger.warning(
                            f"Early stopping PPO epoch {epoch+1} due to high KL: {avg_kl_epoch:.4f} > {1.5 * self.target_kl:.4f}"
                        )
                        break
        
        avg_losses = {key: np.mean(val_list) if val_list else 0.0 for key, val_list in all_losses.items()}
        return avg_losses