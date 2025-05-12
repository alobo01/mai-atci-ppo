"""
PPO Agent using a Beta distribution for continuous actions.

Inherits from PPO and overrides distribution-specific methods,
compatible with both MLP and separate CNN+Head architectures.

Author: Antonio Lobo
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta, Distribution, Independent # Added Independent

import utils
from ppo_revised import PPO # Import the updated base PPO
from utils import NpArray, Tensor, Device


class PPO_Beta(PPO):
    """
    PPO using a Beta distribution for continuous actions.

    Handles rescaling actions between the Beta distribution's native [0, 1]
    range and the environment's action space bounds [low, high].
    Works with both MLP networks and CNNFeatureExtractor + MLP Head architectures.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        device: Optional[Device] = None,
    ) -> None:
        """Initializes the PPO_Beta agent."""
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("PPO_Beta requires a Box action space.")
        if not np.all(np.isfinite(env.action_space.low)) or \
           not np.all(np.isfinite(env.action_space.high)):
            raise ValueError("PPO_Beta requires finite action space bounds.")

        # Ensure algo name is set correctly before calling super
        config['algo'] = 'ppo_beta'
        super().__init__(env, config, device)

        # Cache action bounds and range as tensors on the correct device
        # Ensure dtype matches policy network outputs (float32 usually)
        self._low = utils.to_tensor(self.action_space.low, self.device, dtype=torch.float32)
        self._high = utils.to_tensor(self.action_space.high, self.device, dtype=torch.float32)
        self._range = self._high - self._low
        # Add small epsilon to range to prevent division by zero if low==high
        self._range = torch.max(self._range, torch.tensor(1e-8, device=self.device))


        self.logger.name = f"PPO_Beta_{env.spec.id}_seed{self.seed}" # More specific name
        self.logger.info(f"Initialized PPO_Beta agent for {self.logger.name}.")
        # Network setup is handled by the parent PPO's _setup_networks_and_optimizers


    def _raw_to_alpha_beta(self, raw_actor_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Converts raw network output to Beta distribution parameters alpha and beta."""
        if raw_actor_output.shape[-1] != self.action_dim * 2:
             raise ValueError(f"Expected actor output dim {self.action_dim * 2}, got {raw_actor_output.shape[-1]}")
        raw_alpha, raw_beta = torch.chunk(raw_actor_output, 2, dim=-1)
        # Ensure alpha and beta are >= 1 for unimodal distribution
        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta) + 1.0
        return alpha, beta

    # --- Override distribution method ---

    def _get_distribution(self, obs: Tensor) -> Tuple[Distribution, utils.ActionPostprocessor]:
        """
        Gets the Beta action distribution based on observation features.
        Handles both MLP and CNN network types defined in the parent PPO.
        """
        # 1. Get features using the parent's method (handles CNN/MLP)
        features = self._get_features(obs)

        # 2. Get raw output from the appropriate actor network (head or full MLP)
        actor_network = self.actor_head if self.network_type == "cnn" else self.actor
        if actor_network is None:
            raise RuntimeError("Actor network (or head) not initialized.")

        with self.timer("actor_head_pass"): # Time the final actor network pass
            raw_actor_output = actor_network(features)

        # 3. Convert raw output to alpha, beta parameters
        alpha, beta = self._raw_to_alpha_beta(raw_actor_output)

        # 4. Create the Beta distribution
        # Use Independent to treat batch dim and action dim correctly
        # Beta(alpha, beta) creates dist with shape (batch, action_dim)
        # Independent sums log_prob over the last dim (action_dim)
        base_dist = Beta(alpha, beta)
        dist = Independent(base_dist, 1) # Sum log_prob across action dim

        # 5. Define the postprocessor to scale [0, 1] samples to [low, high]
        def postprocessor(raw_action_01: Tensor) -> Tensor:
            # raw_action_01 is sampled from Beta, should be in [0, 1]
            # Clamp for numerical stability just in case sampling gives exactly 0 or 1
            raw_action_01 = torch.clamp(raw_action_01, 1e-6, 1.0 - 1e-6)
            env_action = self._low + self._range * raw_action_01
            # Final clamp to ensure bounds are strictly met
            return torch.clamp(env_action, self._low, self._high)

        return dist, postprocessor

    # get_action is inherited from PPO and works correctly with the overridden _get_distribution.
    # _rollout is inherited from PPO and works correctly.

    # --- Override update method for correct log_prob calculation ---

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Performs PPO update steps, ensuring log_probs are calculated correctly
        for the Beta distribution using unscaled actions.
        """
        # 1. Unscale actions from environment range [low, high] back to [0, 1]
        actions_env_scale = batch["actions"]
        # Ensure tensors are on the same device
        low = self._low.to(actions_env_scale.device)
        range_ = self._range.to(actions_env_scale.device)

        actions_01_scale = (actions_env_scale - low) / range_
        # Clamp to be strictly within (0, 1) for Beta log_prob stability
        actions_01_scale = torch.clamp(actions_01_scale, 1e-6, 1.0 - 1e-6)
        # Store for use in the loop, replacing the original actions if needed,
        # or just use this variable directly. Let's use it directly.

        # --- Standard PPO Update Loop ---
        all_losses = {"policy_loss": [], "value_loss": [], "entropy_loss": [], "approx_kl": [], "clip_fraction": []}

        # Ensure networks are in training mode (handled by parent PPO potentially, but safe to do here)
        if self.actor: self.actor.train()
        if self.critic: self.critic.train()
        if self.cnn_feature_extractor: self.cnn_feature_extractor.train()
        if self.actor_head: self.actor_head.train()
        if self.critic_head: self.critic_head.train()

        # Calculate batch/minibatch sizes (needed if PPO doesn't store them)
        current_batch_size = batch["obs"].shape[0]
        if current_batch_size % self.num_minibatches != 0:
             # This case should be handled by PPO init, but double-check
             self.logger.warning("Batch size not divisible by num_minibatches during update.")
        actual_minibatch_size = current_batch_size // self.num_minibatches


        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(current_batch_size, device=self.device) # Generate indices on target device

            for start in range(0, current_batch_size, actual_minibatch_size):
                end = start + actual_minibatch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_obs = batch["obs"][mb_indices]
                mb_actions_01 = actions_01_scale[mb_indices] # Use UNSCALED actions
                mb_old_log_probs = batch["log_probs"][mb_indices] # From rollout (was already on 0-1 scale)
                mb_advantages = batch["advantages"][mb_indices]
                mb_returns = batch["returns"][mb_indices]

                # --- Calculate current policy distribution and values ---
                # _get_distribution handles Beta creation and CNN/MLP cases
                dist, _ = self._get_distribution(mb_obs) # dist is Independent(Beta)
                # _get_value handles value estimation and CNN/MLP cases
                current_values = self._get_value(mb_obs)

                # --- Calculate log_prob using the UNSCALED actions [0, 1] ---
                # The log_prob from Independent(Beta) already sums over action dim.
                current_log_probs = dist.log_prob(mb_actions_01) # Shape (minibatch_size,)

                # --- Calculate entropy ---
                # Entropy from Independent(Beta) also sums over action dim.
                entropy = dist.entropy().mean() # Mean entropy over the minibatch

                # --- Policy Loss (Clipped Surrogate Objective) ---
                log_ratio = current_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                policy_loss_1 = mb_advantages * ratio
                policy_loss_2 = mb_advantages * torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # --- Value Loss (Standard MSE) ---
                value_loss = 0.5 * F.mse_loss(current_values, mb_returns)

                # --- Total Loss ---
                loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss

                # --- Optimization Step (using parent PPO's optimizer logic) ---
                # Determine which optimizers to use based on network type
                using_single_optimizer = (self.network_type == "cnn" and self.critic_optimizer is None)

                if using_single_optimizer:
                    if self.actor_optimizer: self.actor_optimizer.zero_grad()
                    else: raise RuntimeError("Single optimizer not initialized for CNN.")
                else: # MLP case or separate optimizers for CNN
                    if self.actor_optimizer: self.actor_optimizer.zero_grad()
                    if self.critic_optimizer: self.critic_optimizer.zero_grad()

                with self.timer("backward_pass"):
                    loss.backward()

                # Gradient Clipping (apply to all trainable parameters)
                params_to_clip = []
                if self.network_type == "cnn":
                    if self.cnn_feature_extractor: params_to_clip.extend(self.cnn_feature_extractor.parameters())
                    if self.actor_head: params_to_clip.extend(self.actor_head.parameters())
                    if self.critic_head: params_to_clip.extend(self.critic_head.parameters())
                else: # MLP
                    if self.actor: params_to_clip.extend(self.actor.parameters())
                    if self.critic: params_to_clip.extend(self.critic.parameters())
                # Filter out non-trainable params if necessary (shouldn't be needed with Adam)
                trainable_params_to_clip = [p for p in params_to_clip if p.requires_grad]
                if trainable_params_to_clip:
                    nn.utils.clip_grad_norm_(trainable_params_to_clip, self.max_grad_norm)

                with self.timer("optimizer_step"):
                    if using_single_optimizer:
                        if self.actor_optimizer: self.actor_optimizer.step()
                    else: # Separate optimizers
                        if self.actor_optimizer: self.actor_optimizer.step()
                        if self.critic_optimizer: self.critic_optimizer.step()

                # --- Track Metrics ---
                with torch.no_grad():
                    approx_kl = (log_ratio).mean().item() # Mean over minibatch
                    clipped = ratio.gt(1 + self.clip_eps) | ratio.lt(1 - self.clip_eps)
                    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["value_loss"].append(value_loss.item())
                all_losses["entropy_loss"].append(entropy.item())
                all_losses["approx_kl"].append(approx_kl)
                all_losses["clip_fraction"].append(clip_fraction)

            # --- KL Early Stopping (Optional, inherited logic from PPO) ---
            if self.target_kl is not None:
                 # Calculate avg KL for the epoch
                 kls_epoch = all_losses["approx_kl"][-self.num_minibatches:]
                 avg_kl_epoch = np.mean(kls_epoch) if kls_epoch else 0.0
                 if avg_kl_epoch > 1.5 * self.target_kl:
                     self.logger.warning(f"Early stopping at epoch {epoch+1}/{self.ppo_epochs} due to high KL: {avg_kl_epoch:.4f} > {1.5 * self.target_kl:.4f}")
                     break

        # Return average losses over all update steps performed
        avg_losses = {key: np.mean(val) if val else 0.0 for key, val in all_losses.items()}
        return avg_losses