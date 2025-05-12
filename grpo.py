"""
Generalised REINFORCE with a Reference Policy (GRPO - No Critic).

Inherits from BaseAgent and implements GRPO logic.

Author: Antonio Lobo
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, kl_divergence, Independent, Normal
from torch.optim import Adam

import utils
from base_agent import BaseAgent
from networks import NETWORK_REGISTRY, FeedForwardNN, CNNFeatureExtractor
from utils import NpArray, Tensor, Device, Loggable


class GRPO_NoCritic(BaseAgent):
    """GRPO Agent without a critic/value baseline."""

    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        device: Optional[Device] = None,
    ) -> None:
        """Initializes the GRPO agent."""
        config['algo'] = 'grpo' # Set algo name
        super().__init__(env, config, device)

        # GRPO specific hyperparameters
        self.G: int = config.get("group_size", 64) # Trajectories per group/prompt
        self.update_epochs: int = config.get("update_epochs", 10)
        self.max_grad_norm: float = config.get("max_grad_norm", 0.5)
        self.entropy_coef: float = config.get("entropy_coef", 0.001)
        self.kl_coef: float = config.get("kl_coef", 0.01) # Weight for KL penalty vs reference
        self.ref_update_interval: int = config.get("ref_update_interval", 10_000) # Steps

        # Minibatch setup for update phase
        self.minibatch_size: int = config.get("minibatch_size", 256) # Per-step minibatch

        # Internal state
        self._steps_since_ref_update: int = 0
        self.actor_ref: Optional[nn.Module] = None # Frozen reference policy network

        # Network type selection
        self.network_type: str = config.get("network_type", "mlp")
        self.cnn_output_features: int = config.get("cnn_output_features", 256)
        self.mlp_hidden_dims: Tuple[int, ...] = tuple(config.get("mlp_hidden_dims", [64, 64]))
        self.cnn_feature_extractor: Optional[CNNFeatureExtractor] = None
        self.actor_head: Optional[FeedForwardNN] = None

    def _setup_networks_and_optimizers(self) -> None:
        lr = self.config.get("lr", 1e-4)
        actor_output_dim = self.action_dim * 2 # Assume Gaussian/Beta

        self.critic = None # No critic in this GRPO version
        self.critic_optimizer = None

        if self.network_type == "mlp":
            self.actor = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                actor_output_dim,
                hidden_dims=self.mlp_hidden_dims
            ).to(self.device)
            # Reference Actor
            self.actor_ref = copy.deepcopy(self.actor).eval()
            for param in self.actor_ref.parameters(): param.requires_grad = False
            # Optimizer
            self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
            self.logger.info("Initialized GRPO MLP actor and reference.")

        elif self.network_type == "cnn":
            self.cnn_feature_extractor = CNNFeatureExtractor(
                obs_shape=self.obs_shape,
                output_features=self.cnn_output_features
            ).to(self.device)
            self.actor_head = FeedForwardNN(
                input_dim=self.cnn_output_features,
                output_dim=actor_output_dim,
                hidden_dims=self.mlp_hidden_dims
            ).to(self.device)

            # Combine actor parts for easier handling
            self.actor = nn.Sequential(self.cnn_feature_extractor, self.actor_head) # Define self.actor this way

            # Reference Actor (needs same structure)
            self.actor_ref = copy.deepcopy(self.actor).eval()
            for param in self.actor_ref.parameters(): param.requires_grad = False

            # Optimizer for the main actor (base + head)
            self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
            self.logger.info("Initialized GRPO CNN feature extractor + MLP head actor and reference.")
        else:
            raise ValueError(f"Unknown network type: {self.network_type}")

    # --- Action Selection & Distribution ---


    def _get_features(self, obs: Tensor, use_reference: bool = False) -> Tensor:
        """Extracts features (only relevant for CNN)."""
        if self.network_type == "cnn":
            network = self.actor_ref if use_reference else self.actor
            if network is None or not isinstance(network, nn.Sequential):
                 raise RuntimeError("GRPO CNN actor not initialized correctly.")
            cnn_base = network[0] # Access the CNNFeatureExtractor part
            timer_key = "ref_cnn_feature_pass" if use_reference else "cnn_feature_pass"
            with torch.no_grad() if use_reference else torch.enable_grad(), self.timer(timer_key):
                 features = cnn_base(obs)
            return features
        else:
            return obs # No feature extraction for MLP


    def _get_distribution(
        self,
        obs: Tensor,
        use_reference: bool = False
    ) -> Tuple[Distribution, utils.ActionPostprocessor]:
        """Gets the action distribution (handles MLP and CNN)."""
        features = self._get_features(obs, use_reference) # Get features first

        if self.network_type == "cnn":
            network = self.actor_ref if use_reference else self.actor
            if network is None or not isinstance(network, nn.Sequential):
                 raise RuntimeError("GRPO CNN actor not initialized correctly.")
            actor_h = network[1] # Access the actor head MLP part
        else: # MLP
            actor_h = self.actor_ref if use_reference else self.actor
            if actor_h is None: raise RuntimeError("GRPO MLP actor not initialized.")

        timer_key = "ref_actor_head_pass" if use_reference else "actor_head_pass"
        with torch.no_grad() if use_reference else torch.enable_grad(), self.timer(timer_key):
            actor_output = actor_h(features) # Pass features to the head

        # Assuming Gaussian/Beta dist based on output dim
        return utils.get_action_distribution(actor_output, self.action_space)



    @torch.no_grad()
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Dict[str, Tensor]]:
        """Select action using the *current* actor network."""
        obs_tensor = utils.to_tensor(obs, self.device).unsqueeze(0)
        dist, postprocessor = self._get_distribution(obs_tensor, use_reference=False)

        if deterministic:
            # Same deterministic logic as PPO
            if isinstance(dist, Independent) and isinstance(dist.base_dist, Normal):
                 raw_action = dist.base_dist.mean
            # Add Beta/Categorical cases if GRPO supports them
            else:
                 self.logger.warning(f"Deterministic action not implemented for {type(dist)}, sampling instead.")
                 raw_action = dist.sample()
        else:
            raw_action = dist.sample()

        log_prob = dist.log_prob(raw_action)
        action = postprocessor(raw_action).squeeze(0).cpu().numpy()

        # GRPO doesn't use value estimate in action selection info
        action_info = {"log_prob": log_prob.squeeze(0)} # Shape ()
        return action, action_info


    # --- Rollout Phase (Group Rollout) ---

    def _rollout(self) -> Tuple[Dict[str, Any], Dict[str, Loggable]]:
        """Collects a group of G trajectories."""
        obs_list, act_list, logp_list, rew_list, len_list = ([] for _ in range(5))
        group_total_rewards: List[float] = []
        total_steps_collected = 0
        start_time = time.perf_counter()

        for i in range(self.G): # Collect G trajectories
            traj_obs, traj_act, traj_logp, traj_rew = ([] for _ in range(4))
            # Use unique seed per trajectory within the group if needed, or rely on env auto-reset seed
            obs, _ = self.env.reset() # Seed handled by base class env or make_env
            done = False
            traj_len = 0
            traj_total_reward = 0.0
            max_ep_len = getattr(self.env, "_max_episode_steps", 1000)

            while not done and traj_len < max_ep_len:
                # Get action using the *current* policy
                action, info = self.get_action(obs, deterministic=False)
                log_prob = info["log_prob"].item()

                # Store step data for this trajectory
                traj_obs.append(obs)
                traj_act.append(action)
                traj_logp.append(log_prob)

                # Environment step
                with self.timer("env_step"):
                     next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                traj_rew.append(reward)
                traj_total_reward += reward
                traj_len += 1
                obs = next_obs

            # Trajectory finished
            group_total_rewards.append(traj_total_reward)
            len_list.append(traj_len)
            total_steps_collected += traj_len

            # Extend the main buffers
            obs_list.extend(traj_obs)
            act_list.extend(traj_act)
            logp_list.extend(traj_logp)
            rew_list.extend(traj_rew) # We don't directly use per-step rewards in GRPO update

            if self.verbose:
                self.logger.debug(f"Group {i+1}/{self.G}: Reward={traj_total_reward:.2f}, Length={traj_len}")


        # --- Calculate per-trajectory advantages (normalized returns) ---
        group_returns_np = np.array(group_total_rewards, dtype=np.float32)
        # Normalize returns across the group to get advantages
        adv_per_traj = (group_returns_np - group_returns_np.mean()) / (group_returns_np.std() + 1e-8)

        # --- Assign advantage to each step within a trajectory ---
        # Simple version: Uniformly distribute trajectory advantage over its steps
        # More complex weighting (e.g., discounted) could be used.
        adv_list: List[float] = []
        for traj_idx, length in enumerate(len_list):
            step_advantage = adv_per_traj[traj_idx] # Could divide by length: / length
            adv_list.extend([step_advantage] * length)

        if len(adv_list) != total_steps_collected:
             self.logger.error("Mismatch between collected steps and advantages!")
             # Handle error appropriately

        # --- Prepare batch dictionary ---
        batch = {
            "obs": np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.float32) if self.is_continuous else np.array(act_list, dtype=np.int64),
            "log_probs": np.array(logp_list, dtype=np.float32), # Log probs from the *acting* policy
            "advantages": np.array(adv_list, dtype=np.float32),
            "n_steps": total_steps_collected,
        }
        # Convert to tensors
        for key, val in batch.items():
             if isinstance(val, np.ndarray):
                 batch[key] = utils.to_tensor(val, self.device)

        # --- Prepare rollout info dictionary ---
        rollout_duration = time.perf_counter() - start_time
        steps_per_second = total_steps_collected / rollout_duration if rollout_duration > 0 else 0
        rollout_info = {
            "rollout_duration_s": rollout_duration,
            "steps_per_second": steps_per_second,
            "avg_episodic_reward": group_returns_np.mean(), # Avg return over the G trajectories
            "avg_episode_length": np.mean(len_list),
            "group_size": self.G,
        }

        # Update steps counter for reference policy update
        self._steps_since_ref_update += total_steps_collected

        return batch, rollout_info


    # --- Update Phase ---

    def _update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Performs GRPO update steps."""
        all_losses = {"policy_loss": [], "kl_loss": [], "entropy_loss": [], "total_loss": []}
        num_steps_in_batch = batch["n_steps"]
        current_batch_size = num_steps_in_batch # Effective batch size is all steps in group

        if current_batch_size == 0:
             self.logger.warning("GRPO update called with empty batch.")
             return {k: 0.0 for k in all_losses}

        # Determine number of minibatches based on total steps
        if current_batch_size < self.minibatch_size:
             actual_minibatch_size = current_batch_size
             num_minibatches = 1
        elif current_batch_size % self.minibatch_size != 0:
             # Adjust minibatch size slightly if not perfectly divisible
             num_minibatches = (current_batch_size // self.minibatch_size) + 1
             actual_minibatch_size = (current_batch_size + num_minibatches -1) // num_minibatches # Ceiling division
             self.logger.warning(f"Batch size {current_batch_size} not divisible by minibatch_size {self.minibatch_size}. Using {num_minibatches} minibatches of size ~{actual_minibatch_size}.")
        else:
             num_minibatches = current_batch_size // self.minibatch_size
             actual_minibatch_size = self.minibatch_size


        if self.actor: self.actor.train() # Ensure actor is in training mode

        for epoch in range(self.update_epochs):
            indices = torch.randperm(current_batch_size)

            for i in range(num_minibatches):
                start = i * actual_minibatch_size
                end = min(start + actual_minibatch_size, current_batch_size)
                if start >= end: continue # Skip if start index is out of bounds
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_obs = batch["obs"][mb_indices]
                mb_actions = batch["actions"][mb_indices]
                mb_old_log_probs = batch["log_probs"][mb_indices] # LogProbs from policy *at rollout time*
                mb_advantages = batch["advantages"][mb_indices]

                # --- Get current policy distribution ---
                dist, _ = self._get_distribution(mb_obs, use_reference=False)
                current_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Sum log_probs if needed (like in PPO Beta)
                if current_log_probs.ndim > 1 and self.is_continuous:
                     current_log_probs = current_log_probs.sum(dim=-1)

                # --- Policy Gradient Loss (REINFORCE style with importance sampling) ---
                # Ratio uses log_probs from policy *now* vs policy *at rollout*
                log_ratio = current_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                # Note: GRPO paper might use slightly different objective.
                # This is Importance-Weighted PG.
                policy_loss = -(ratio * mb_advantages).mean()

                # --- KL Divergence Penalty vs Reference Policy ---
                with torch.no_grad():
                     dist_ref, _ = self._get_distribution(mb_obs, use_reference=True)

                # Ensure KL calculation handles potential distribution type differences
                # Requires distributions to be of compatible types.
                kl_div = kl_divergence(dist, dist_ref).mean() # Mean KL over the batch

                # --- Total Loss ---
                loss = policy_loss + self.kl_coef * kl_div - self.entropy_coef * entropy

                # # --- Optimization Step ---
                self.actor_optimizer.zero_grad()
                with self.timer("backward_pass"):
                    loss.backward()
                # Clip gradients for the *entire* actor network (base + head if CNN)
                if self.actor: # self.actor is the Sequential model in CNN case
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                with self.timer("optimizer_step"):
                    self.actor_optimizer.step()

                # --- Track Losses ---
                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["kl_loss"].append(kl_div.item())
                all_losses["entropy_loss"].append(entropy.item())
                all_losses["total_loss"].append(loss.item())

        # --- Reference Policy Update ---
        if self._steps_since_ref_update >= self.ref_update_interval:
            self.logger.info(f"Updating reference policy after {self._steps_since_ref_update} steps.")
            self.actor_ref.load_state_dict(self.actor.state_dict())
            self._steps_since_ref_update = 0 # Reset counter

        # Return average losses
        avg_losses = {key: np.mean(val) for key, val in all_losses.items()}
        return avg_losses