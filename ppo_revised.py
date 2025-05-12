"""
Proximal Policy Optimization (PPO) Agent.

Inherits from BaseAgent and implements PPO-specific logic using
Gaussian or Beta policies.

Author: Antonio Lobo
"""
from __future__ import annotations

import time # For rollout loop timing if needed
from typing import Any, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Distribution, Normal, Independent, Beta, Categorical

import utils
from base_agent import BaseAgent
from networks import NETWORK_REGISTRY, FeedForwardNN, CNNFeatureExtractor
from utils import NpArray, Tensor, Device, Loggable


class PPO(BaseAgent):
    """Proximal Policy Optimization (PPO) Agent."""

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        device: Optional[Device] = None,
    ) -> None:
        """
        Initializes the PPO agent.

        Args:
            env: The Gymnasium environment instance.
            config: Dictionary of hyperparameters and settings. Must include
                    PPO specific params like 'clip_eps', 'ppo_epochs', etc.
            device: The torch device to use.
        """
        # Set algo name for logging/dirs before calling super
        config['algo'] = config.get('algo', 'ppo_gauss') # Default to Gaussian
        super().__init__(env, config, device)

        # PPO specific hyperparameters
        self.lam: float = config.get("lam", 0.95) # GAE lambda
        self.clip_eps: float = config.get("clip_eps", 0.2) # PPO clip range
        self.ppo_epochs: int = config.get("ppo_epochs", 10) # Update epochs per rollout
        self.num_minibatches: int = config.get("num_minibatches", 32) # Num minibatches
        self.entropy_coef: float = config.get("entropy_coef", 0.01)
        self.value_coef: float = config.get("value_coef", 0.5)
        self.max_grad_norm: float = config.get("max_grad_norm", 0.5)
        self.target_kl: Optional[float] = config.get("target_kl", None) # Optional KL early stopping
        self.rollout_steps: int = config.get("rollout_steps", 2048) # Steps per rollout

        # Network type selection
        self.network_type: str = config.get("network_type", "mlp") # 'mlp' or 'cnn'
        self.cnn_output_features: int = config.get("cnn_output_features", 256) # Configurable CNN output size
        self.mlp_hidden_dims: Tuple[int, ...] = tuple(config.get("mlp_hidden_dims", [64, 64])) # Configurable MLP hidden dims

        self.cnn_feature_extractor: Optional[CNNFeatureExtractor] = None
        self.actor_head: Optional[FeedForwardNN] = None
        self.critic_head: Optional[FeedForwardNN] = None

        # Batch and minibatch size calculation
        self.batch_size = self.rollout_steps # Assuming one env for now
        if self.batch_size % self.num_minibatches != 0:
            raise ValueError("rollout_steps must be divisible by num_minibatches")
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Internal state for rollout
        self._current_obs: NpArray = self.env.reset(seed=self.seed)[0]
        self._current_episode_reward: float = 0.0
        self._current_episode_length: int = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

    def _setup_networks_and_optimizers(self) -> None:
        """Initializes PPO's actor and critic networks and optimizers."""
        lr = self.config.get("lr", 3e-4)
        actor_output_dim = self.action_dim * 2 # Assume Gaussian/Beta needs 2 params per dim
        critic_output_dim = 1

        if self.network_type == "mlp":
            self.actor = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                actor_output_dim,
                hidden_dims=self.mlp_hidden_dims
            ).to(self.device)
            self.critic = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                critic_output_dim,
                hidden_dims=self.mlp_hidden_dims
            ).to(self.device)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=lr, eps=1e-5)
            self.logger.info("Using separate MLP Actor and Critic networks.")

        elif self.network_type == "cnn":
            self.cnn_feature_extractor = CNNFeatureExtractor(
                obs_shape=self.obs_shape,
                output_features=self.cnn_output_features
            ).to(self.device)

            # Actor Head takes features from CNN
            self.actor_head = FeedForwardNN(
                input_dim=self.cnn_output_features,
                output_dim=actor_output_dim,
                hidden_dims=self.mlp_hidden_dims # Can use same or different hidden dims for heads
            ).to(self.device)

            # Critic Head takes features from CNN
            self.critic_head = FeedForwardNN(
                input_dim=self.cnn_output_features,
                output_dim=critic_output_dim,
                hidden_dims=self.mlp_hidden_dims
            ).to(self.device)

            # --- Optimizer Strategy ---
            # Option 1: Single optimizer for all (base + heads) - Simpler
            # all_params = list(self.cnn_feature_extractor.parameters()) + \
            #              list(self.actor_head.parameters()) + \
            #              list(self.critic_head.parameters())
            # self.actor_optimizer = Adam(all_params, lr=lr, eps=1e-5)
            # self.critic_optimizer = None # Only one optimizer needed
            # self.logger.info("Using CNN Feature Extractor with separate MLP heads (Single Optimizer).")

            # Option 2: Separate optimizers (requires careful param grouping)
            self.actor_optimizer = Adam(list(self.cnn_feature_extractor.parameters()) + list(self.actor_head.parameters()), lr=lr, eps=1e-5)
            self.critic_optimizer = Adam(list(self.cnn_feature_extractor.parameters()) + list(self.critic_head.parameters()), lr=lr, eps=1e-5)
            self.logger.info("Using CNN Feature Extractor with separate MLP heads (Separate Optimizers - Warning: Shared Base).")

        else:
             raise ValueError(f"Unknown network type: {self.network_type}")
        

    # --- Action Selection ---

    def _get_features(self, obs: Tensor) -> Tensor:
        """Extracts features from observations (using CNN or identity for MLP)."""
        if self.network_type == "cnn":
            if self.cnn_feature_extractor is None:
                raise RuntimeError("CNN feature extractor not initialized.")
            with self.timer("cnn_feature_pass"):
                return self.cnn_feature_extractor(obs)
        else: # MLP case, features are the observations themselves
            return obs

    def _get_distribution(self, obs: Tensor) -> Tuple[Distribution, utils.ActionPostprocessor]:
        """Gets the action distribution (handles MLP and CNN cases)."""
        features = self._get_features(obs)
        actor_network = self.actor_head if self.network_type == "cnn" else self.actor
        if actor_network is None:
             raise RuntimeError("Actor network not initialized.")

        with self.timer("actor_head_pass"): # Time only the head pass
            actor_output = actor_network(features)
        return utils.get_action_distribution(actor_output, self.action_space)

    def _get_value(self, obs: Tensor) -> Tensor:
        """Gets the value estimate (handles MLP and CNN cases)."""
        features = self._get_features(obs)
        critic_network = self.critic_head if self.network_type == "cnn" else self.critic
        if critic_network is None:
            raise RuntimeError("Critic network not initialized.")

        with self.timer("critic_head_pass"): # Time only the head pass
            value = critic_network(features)
        return value.squeeze(-1) # Ensure shape (batch_size,)

    @torch.no_grad()
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Dict[str, Tensor]]:
        """Select action, optionally compute value and log_prob."""
        obs_tensor = utils.to_tensor(obs, self.device).unsqueeze(0) # Add batch dim

        dist, postprocessor = self._get_distribution(obs_tensor)
        value = self._get_value(obs_tensor) # Shape (1,)

        if deterministic:
            # For Gaussian, mean is deterministic; for Beta, mode or mean.
            if isinstance(dist, Independent) and isinstance(dist.base_dist, Normal):
                 raw_action = dist.base_dist.mean
            elif isinstance(dist, Beta):
                 # Mean = alpha / (alpha + beta)
                 # Mode = (alpha - 1) / (alpha + beta - 2) if alpha, beta > 1
                 raw_action = dist.mean # Use mean for Beta deterministic action
            elif isinstance(dist, Categorical):
                 raw_action = torch.argmax(dist.logits, dim=-1)
            else:
                 # Fallback to sampling if deterministic logic unclear
                 self.logger.warning(f"Deterministic action not implemented for {type(dist)}, sampling instead.")
                 raw_action = dist.sample()
        else:
            raw_action = dist.sample()

        log_prob = dist.log_prob(raw_action)
        action = postprocessor(raw_action).squeeze(0).cpu().numpy() # Remove batch dim

        # Ensure log_prob and value are scalars tensors before returning
        action_info = {
            "value": value.squeeze(0),     # Shape ()
            "log_prob": log_prob.squeeze(0) # Shape ()
        }
        return action, action_info


    # --- Rollout Phase ---

    def _rollout(self) -> Tuple[Dict[str, Any], Dict[str, Loggable]]:
        """Collect PPO rollout data (obs, actions, rewards, dones, values, log_probs)."""
        # Buffers for rollout data
        obs_buf = np.zeros((self.rollout_steps, *self.obs_shape), dtype=np.float32)
        act_buf = np.zeros((self.rollout_steps, self.action_dim), dtype=np.float32) if self.is_continuous else np.zeros((self.rollout_steps,), dtype=np.int64)
        rew_buf = np.zeros((self.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((self.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((self.rollout_steps,), dtype=np.float32)
        logp_buf = np.zeros((self.rollout_steps,), dtype=np.float32)

        start_time = time.perf_counter()

        for step in range(self.rollout_steps):
            # --- Get action and value from policy ---
            action, info = self.get_action(self._current_obs, deterministic=False)
            value = info["value"].item()
            log_prob = info["log_prob"].item()

            # --- Store current step data ---
            obs_buf[step] = self._current_obs
            act_buf[step] = action
            val_buf[step] = value
            logp_buf[step] = log_prob

            # --- Environment step ---
            with self.timer("env_step"):
                 next_obs, reward, terminated, truncated, env_info = self.env.step(action)
            done = terminated or truncated

            rew_buf[step] = reward
            done_buf[step] = float(done) # Store as float (0.0 or 1.0)

            # --- Update current state and episode trackers ---
            self._current_obs = next_obs
            self._current_episode_reward += reward
            self._current_episode_length += 1

            if done:
                # Record episode stats
                self._episode_rewards.append(self._current_episode_reward)
                self._episode_lengths.append(self._current_episode_length)
                # Log episode end if verbose
                if self.verbose:
                     self.logger.debug(f"Episode finished: Reward={self._current_episode_reward:.2f}, Length={self._current_episode_length}")
                # Reset environment and trackers
                self._current_obs, _ = self.env.reset()
                self._current_episode_reward = 0.0
                self._current_episode_length = 0


        # --- Compute GAE and Returns ---
        with torch.no_grad(), self.timer("gae_computation"):
             # Get value of the *last* observation for bootstrapping
             last_obs_tensor = utils.to_tensor(self._current_obs, self.device).unsqueeze(0)
             last_value = self._get_value(last_obs_tensor).item()

             adv_buf = np.zeros_like(rew_buf)
             gae_lambda = 0.0
             for t in reversed(range(self.rollout_steps)):
                 if t == self.rollout_steps - 1:
                     next_non_terminal = 1.0 - float(done) # Use final done state
                     next_value = last_value
                 else:
                     next_non_terminal = 1.0 - done_buf[t + 1]
                     next_value = val_buf[t + 1]

                 delta = rew_buf[t] + self.gamma * next_value * next_non_terminal - val_buf[t]
                 gae_lambda = delta + self.gamma * self.lam * next_non_terminal * gae_lambda
                 adv_buf[t] = gae_lambda

             ret_buf = adv_buf + val_buf # Returns are advantages + values

        # --- Prepare batch dictionary ---
        batch = {
            "obs": obs_buf,
            "actions": act_buf,
            "values": val_buf,
            "log_probs": logp_buf,
            "advantages": adv_buf,
            "returns": ret_buf,
            "n_steps": self.rollout_steps,
        }
        # Convert numpy arrays to tensors on the correct device
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                batch[key] = utils.to_tensor(val, self.device)

        # Normalize advantages (important!)
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)

        # --- Prepare rollout info dictionary ---
        rollout_duration = time.perf_counter() - start_time
        steps_per_second = self.rollout_steps / rollout_duration if rollout_duration > 0 else 0
        rollout_info = {
            "rollout_duration_s": rollout_duration,
            "steps_per_second": steps_per_second,
        }
        if self._episode_rewards: # If any episodes finished during rollout
             rollout_info["avg_episodic_reward"] = np.mean(self._episode_rewards)
             rollout_info["avg_episode_length"] = np.mean(self._episode_lengths)
             # Clear buffers for next rollout period
             self._episode_rewards.clear()
             self._episode_lengths.clear()
        else: # No episodes finished, report NaN or previous value?
             # Reporting NaN might be clearer if no data available for this specific rollout
             rollout_info["avg_episodic_reward"] = np.nan
             rollout_info["avg_episode_length"] = np.nan


        return batch, rollout_info


    # --- Update Phase ---

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Performs PPO update steps over the collected batch."""
        all_losses = {"policy_loss": [], "value_loss": [], "entropy_loss": [], "approx_kl": [], "clip_fraction": []}

        # Ensure networks are in training mode
        if self.actor: self.actor.train()
        if self.critic: self.critic.train()

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(self.batch_size) # Shuffle indices

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_obs = batch["obs"][mb_indices]
                mb_actions = batch["actions"][mb_indices]
                mb_old_values = batch["values"][mb_indices]
                mb_old_log_probs = batch["log_probs"][mb_indices]
                mb_advantages = batch["advantages"][mb_indices]
                mb_returns = batch["returns"][mb_indices]

                # --- Calculate current policy distribution and values ---
                dist, _ = self._get_distribution(mb_obs)
                current_values = self._get_value(mb_obs)
                current_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ensure log_probs have the correct shape (minibatch_size,)
                # For Independent(Normal), log_prob sums over action dim automatically.
                # For Beta, need to sum manually if action_dim > 1. Check _get_distribution.
                # For Categorical, log_prob is already correct shape.
                if current_log_probs.ndim > 1 and self.is_continuous:
                    # This might happen if Independent wasn't used correctly
                    current_log_probs = current_log_probs.sum(dim=-1)


                # --- Policy Loss (Clipped Surrogate Objective) ---
                log_ratio = current_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # --- Value Loss (Clipped version optional, usually MSE) ---
                # Clip value loss: (v_pred - v_target)^2 vs (v_clip - v_target)^2
                # values_clipped = mb_old_values + torch.clamp(current_values - mb_old_values, -self.clip_eps, self.clip_eps)
                # vf_loss1 = F.mse_loss(current_values, mb_returns, reduction='none')
                # vf_loss2 = F.mse_loss(values_clipped, mb_returns, reduction='none')
                # value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                # Standard MSE value loss:
                value_loss = 0.5 * F.mse_loss(current_values, mb_returns)


                # --- Total Loss ---
                loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

                # --- Optimization Step ---
                if self.network_type == "cnn":
                    if self.actor_optimizer: # The single optimizer
                         self.actor_optimizer.zero_grad()
                    else: raise RuntimeError("Optimizer not initialized for CNN")
                else: # MLP case
                    if self.actor_optimizer: self.actor_optimizer.zero_grad()
                    if self.critic_optimizer: self.critic_optimizer.zero_grad()

                with self.timer("backward_pass"):
                    loss.backward()

                # Gradient Clipping
                params_to_clip = []
                if self.network_type == "cnn":
                    if self.cnn_feature_extractor: params_to_clip.extend(self.cnn_feature_extractor.parameters())
                    if self.actor_head: params_to_clip.extend(self.actor_head.parameters())
                    if self.critic_head: params_to_clip.extend(self.critic_head.parameters())
                else: # MLP
                    if self.actor: params_to_clip.extend(self.actor.parameters())
                    if self.critic: params_to_clip.extend(self.critic.parameters())

                if params_to_clip:
                    nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)


                with self.timer("optimizer_step"):
                    if self.network_type == "cnn":
                        if self.actor_optimizer: self.actor_optimizer.step()
                    else: # MLP
                        if self.actor_optimizer: self.actor_optimizer.step()
                        if self.critic_optimizer: self.critic_optimizer.step()

                # --- Track Metrics ---
                with torch.no_grad():
                    approx_kl = (log_ratio).mean().item() # Simplified KL approximation
                    clipped = ratio.gt(1 + self.clip_eps) | ratio.lt(1 - self.clip_eps)
                    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["value_loss"].append(value_loss.item())
                all_losses["entropy_loss"].append(entropy.item())
                all_losses["approx_kl"].append(approx_kl)
                all_losses["clip_fraction"].append(clip_fraction)

            # --- KL Early Stopping ---
            avg_kl_epoch = np.mean(all_losses["approx_kl"][-self.num_minibatches:]) # Avg KL over last epoch
            if self.target_kl is not None and avg_kl_epoch > 1.5 * self.target_kl:
                self.logger.warning(f"Early stopping at epoch {epoch+1} due to high KL: {avg_kl_epoch:.4f} > {1.5 * self.target_kl:.4f}")
                break


        # Return average losses over all epochs and minibatches
        avg_losses = {key: np.mean(val) for key, val in all_losses.items()}
        return avg_losses
