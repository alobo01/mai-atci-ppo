import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, kl_divergence
from torch.optim import Adam

from algorithms.base_agent import BaseAgent, SUPPORTED_DISTRIBUTIONS
from algorithms.buffer import RolloutBuffer # GRPO can use parts of this buffer
from networks.mlp import FeedForwardNN
from networks.cnn import CNNFeatureExtractor
from utils import distribution_utils, torch_utils
from utils.pydantic_models import ExperimentConfig, GRPOConfig
from utils.torch_utils import NpArray, Tensor, Device
from utils.video_plot_utils import Loggable


class GRPO_NoCritic(BaseAgent):
    """GRPO Agent without a critic/value baseline."""

    def __init__(
        self,
        env: gym.Env,
        config: ExperimentConfig,
        device_str: Optional[Device] = None,
    ) -> None:
        
        algo_config = config.get_algo_specific_config()
        if not isinstance(algo_config, GRPOConfig):
            raise TypeError("GRPO agent requires GRPOConfig.")
        self.grpo_cfg: GRPOConfig = algo_config

        if self.grpo_cfg.distribution_type not in SUPPORTED_DISTRIBUTIONS:
            raise ValueError(f"Unsupported distribution type: {self.grpo_cfg.distribution_type}. "
                             f"Supported are: {SUPPORTED_DISTRIBUTIONS}")
        

        

        # GRPO specific hyperparameters
        self.G: int = self.grpo_cfg.group_size
        self.update_epochs: int = self.grpo_cfg.update_epochs
        self.max_grad_norm: float = self.grpo_cfg.max_grad_norm
        self.entropy_coef: float = self.grpo_cfg.entropy_coef
        self.kl_coef: float = self.grpo_cfg.kl_coef
        self.ref_update_interval: int = self.grpo_cfg.ref_update_interval
        self.rollout_steps_per_trajectory: int = self.grpo_cfg.rollout_steps_per_trajectory

        # Minibatch setup for GRPO update phase
        self.minibatch_size: int = self.grpo_cfg.minibatch_size

        self.network_cfg = config.network_config
        self._steps_since_ref_update: int = 0
        self.actor_ref: Optional[nn.Module] = None # Frozen reference policy network
        
        # After config set we can call super init
        super().__init__(env, config, device_str)

        # Buffer for GRPO: it will store obs, actions, log_probs from acting policy,
        # and then advantages will be assigned after group normalization.
        # Total buffer size needs to accommodate G * rollout_steps_per_trajectory
        self.buffer_size_grpo = self.G * self.rollout_steps_per_trajectory
        self.buffer = RolloutBuffer(
            buffer_size=self.buffer_size_grpo,
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape if self.is_continuous else (),
            device=self.device,
            # Gamma/GAE lambda are not used by GRPO's core logic from this buffer
            gamma=0.0, gae_lambda=0.0, # Placeholder
            is_continuous=self.is_continuous
        )
        self._current_obs_grpo: NpArray = self.env.reset(seed=self.seed)[0] # For group rollouts

       

    def _setup_networks_and_optimizers(self) -> None:
        lr = self.grpo_cfg.lr
        if self.grpo_cfg.distribution_type in ["normal", "beta"]:
            actor_output_dim = self.action_dim * 2
        elif self.grpo_cfg.distribution_type == "categorical":
            actor_output_dim = self.action_dim
        else:
            raise ValueError(f"Unhandled dist type for GRPO actor: {self.grpo_cfg.distribution_type}")

        self.critic = None # No critic in this GRPO version
        self.critic_optimizer = None

        if self.network_cfg.network_type == "mlp":
            self.actor = FeedForwardNN(
                int(np.prod(self.obs_shape)),
                actor_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)
        elif self.network_cfg.network_type == "cnn":
            self.cnn_feature_extractor = CNNFeatureExtractor(
                obs_shape=self.obs_shape,
                output_features=self.network_cfg.cnn_output_features
            ).to(self.device)
            self.actor_head = FeedForwardNN(
                input_dim=self.network_cfg.cnn_output_features,
                output_dim=actor_output_dim,
                hidden_dims=self.network_cfg.mlp_hidden_dims
            ).to(self.device)
            # For GRPO, self.actor is the Sequential model (base + head)
            self.actor = nn.Sequential(self.cnn_feature_extractor, self.actor_head) # type: ignore
        else:
            raise ValueError(f"Unknown network type for GRPO: {self.network_cfg.network_type}")

        self.actor_ref = copy.deepcopy(self.actor).eval()
        for param in self.actor_ref.parameters(): param.requires_grad = False
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.logger.info(f"Initialized GRPO actor (type: {self.network_cfg.network_type}, dist: {self.grpo_cfg.distribution_type}) and reference.")

    def _get_features(self, obs: Tensor, use_reference: bool = False) -> Tensor:
        """Extracts features from observations."""
        if self.network_cfg.network_type == "cnn":
            extractor_to_use = None
            if use_reference:
                if isinstance(self.actor_ref, nn.Sequential): extractor_to_use = self.actor_ref[0] # type: ignore
            else:
                if isinstance(self.actor, nn.Sequential): extractor_to_use = self.actor[0] # type: ignore
            
            if extractor_to_use is None or not isinstance(extractor_to_use, CNNFeatureExtractor):
                 raise RuntimeError("GRPO CNN feature extractor not correctly assigned.")
            
            timer_key = "ref_cnn_feature" if use_reference else "cnn_feature"
            with torch.no_grad() if use_reference else torch.enable_grad(), self.timer(timer_key):
                return extractor_to_use(obs)
        return obs

    def _get_distribution_from_features(
        self, features: Tensor, use_reference: bool = False
    ) -> Tuple[Distribution, distribution_utils.ActionPostprocessor]:
        """Gets action distribution from features."""
        actor_module_to_use = None
        if self.network_cfg.network_type == "cnn":
            if use_reference:
                if isinstance(self.actor_ref, nn.Sequential): actor_module_to_use = self.actor_ref[1] # type: ignore
            else:
                if isinstance(self.actor, nn.Sequential): actor_module_to_use = self.actor[1] # type: ignore
        else: # MLP
            actor_module_to_use = self.actor_ref if use_reference else self.actor
        
        if actor_module_to_use is None or not isinstance(actor_module_to_use, FeedForwardNN):
            raise RuntimeError("GRPO Actor module (or head) not correctly assigned.")

        timer_key = "ref_actor_head" if use_reference and self.network_cfg.network_type == "cnn" else \
                    ("ref_actor_mlp" if use_reference else \
                    ("actor_head" if self.network_cfg.network_type == "cnn" else "actor_mlp"))

        with torch.no_grad() if use_reference else torch.enable_grad(), self.timer(timer_key):
            actor_output = actor_module_to_use(features)
        
        return distribution_utils.create_distribution_from_actor_output(
            actor_output, self.grpo_cfg.distribution_type, self.action_space, self.action_dim
        )

    @torch.no_grad()
    def get_action(self, obs: NpArray, deterministic: bool = False) -> Tuple[NpArray, Dict[str, Any]]:
        obs_tensor = torch_utils.to_tensor(obs, self.device).unsqueeze(0)
        features = self._get_features(obs_tensor, use_reference=False) # Use current actor
        dist, postprocessor = self._get_distribution_from_features(features, use_reference=False)

        if deterministic:
            if self.grpo_cfg.distribution_type == "normal": action_canonical = dist.base_dist.mean # type: ignore
            elif self.grpo_cfg.distribution_type == "beta": action_canonical = dist.mean # type: ignore
            elif self.grpo_cfg.distribution_type == "categorical": action_canonical = torch.argmax(dist.logits, dim=-1, keepdim=not self.is_continuous) # type: ignore
            else: action_canonical = dist.sample()
        else:
            action_canonical = dist.sample()
        
        log_prob = dist.log_prob(action_canonical)
        action_env_scale = postprocessor(action_canonical).squeeze(0).cpu().numpy()
        
        action_info = { 
            "log_prob": log_prob.squeeze(0),
            "action_canonical": action_canonical.squeeze(0)
        } # GRPO doesn't use 'value' from here
        return action_env_scale, action_info

    def _rollout(self) -> Tuple[RolloutBuffer, Dict[str, Loggable]]:
        self.buffer.reset()
        group_trajectory_rewards: List[float] = []
        group_trajectory_lengths: List[int] = []
        total_steps_in_group = 0
        
        rollout_start_time = time.perf_counter()

        for i_traj in range(self.G):
            obs_traj, _ = self.env.reset() # Use self._current_obs_grpo for consistency if stepping multiple envs
            
            traj_reward_sum = 0.0
            traj_len = 0
            
            temp_traj_obs: List[NpArray] = []
            temp_traj_actions: List[NpArray] = []
            temp_traj_actions_canonical: List[NpArray] = []
            temp_traj_log_probs: List[float] = []

            for _ in range(self.rollout_steps_per_trajectory):
                action_env, info = self.get_action(obs_traj, deterministic=False) # From current policy
                log_p = info["log_prob"].item()
                action_canonical_np = info["action_canonical"].cpu().numpy()

                temp_traj_obs.append(obs_traj)
                temp_traj_actions.append(action_env)
                temp_traj_actions_canonical.append(action_canonical_np)
                temp_traj_log_probs.append(log_p)

                next_obs_traj, reward, terminated, truncated, _ = self.env.step(action_env)
                done_traj = terminated or truncated
                
                traj_reward_sum += reward
                traj_len += 1
                obs_traj = next_obs_traj

                if done_traj:
                    break
            
            # Store trajectory data in buffer
            for k in range(traj_len):
                self.buffer.add_grpo_step(
                    temp_traj_obs[k],
                    temp_traj_actions[k], # This is action_env_scale
                    temp_traj_log_probs[k],
                    temp_traj_actions_canonical[k] # Pass the unclipped action
                )
            
            group_trajectory_rewards.append(traj_reward_sum)
            group_trajectory_lengths.append(traj_len)
            total_steps_in_group += traj_len

        # Calculate group-normalized returns as advantages
        group_returns_np = np.array(group_trajectory_rewards, dtype=np.float32)
        adv_per_traj = (group_returns_np - group_returns_np.mean()) / (np.std(group_returns_np) + 1e-8)

        # Assign advantages to each step in the buffer
        # Simplest: uniform distribution of trajectory advantage over its steps
        all_step_advantages = np.zeros(total_steps_in_group, dtype=np.float32)
        current_pos = 0
        for i_traj in range(self.G):
            num_steps_this_traj = group_trajectory_lengths[i_traj]
            # step_adv = adv_per_traj[i_traj] / num_steps_this_traj if num_steps_this_traj > 0 else 0.0 # Spread over steps
            step_adv = adv_per_traj[i_traj] # Or assign full trajectory advantage to each step
            all_step_advantages[current_pos : current_pos + num_steps_this_traj] = step_adv
            current_pos += num_steps_this_traj
        
        self.buffer.assign_grpo_advantages(all_step_advantages)
        self._steps_since_ref_update += total_steps_in_group

        rollout_duration = time.perf_counter() - rollout_start_time
        rollout_info: Dict[str, Loggable] = {
            "rollout_duration_s": rollout_duration,
            "steps_per_second": total_steps_in_group / rollout_duration if rollout_duration > 0 else 0,
            "steps_collected_this_rollout": total_steps_in_group,
            "avg_episodic_reward": np.mean(group_trajectory_rewards) if group_trajectory_rewards else np.nan,
            "avg_episode_length": np.mean(group_trajectory_lengths) if group_trajectory_lengths else np.nan,
        }
        return self.buffer, rollout_info

    def _update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        all_losses = {"policy_loss": [], "kl_loss": [], "entropy_loss": [], "total_loss": []}
        if buffer.size == 0:
            self.logger.warning("GRPO update called with empty buffer.")
            return {k: 0.0 for k in all_losses}

        if self.actor: self.actor.train()
        
        action_space_low_t = torch_utils.to_tensor(self.action_space.low, self.device, dtype=torch.float32)
        action_space_high_t = torch_utils.to_tensor(self.action_space.high, self.device, dtype=torch.float32)

        for epoch in range(self.update_epochs):
            for batch in buffer.get_grpo_batches(batch_size=self.minibatch_size):
                mb_obs = batch.observations
                mb_actions_env_scale = batch.actions
                mb_old_log_probs = batch.log_probs # LogProbs from acting policy at rollout
                mb_advantages = batch.advantages   # Group-normalized returns
                mb_actions_canonical_unclipped = batch.actions_canonical_unclipped

                features = self._get_features(mb_obs, use_reference=False)
                dist, _ = self._get_distribution_from_features(features, use_reference=False)
                
                # Unscale actions for log_prob if Beta
                if self.grpo_cfg.distribution_type == "beta":
                    actions_canonical = distribution_utils.unscale_action_for_beta_log_prob(
                        mb_actions_env_scale, action_space_low_t, action_space_high_t
                    )
                else: # Normal or Categorical
                    actions_canonical = mb_actions_canonical_unclipped

                current_log_probs = dist.log_prob(actions_canonical)
                entropy = dist.entropy().mean()

                # Policy Gradient Loss (Importance Sampling)
                log_ratio = current_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)
                clip_eps = 0.2  # PPO default
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()

                # KL Divergence vs Reference Policy
                with torch.no_grad():
                    features_ref = self._get_features(mb_obs, use_reference=True)
                    dist_ref, _ = self._get_distribution_from_features(features_ref, use_reference=True)
                
                # Ensure dist and dist_ref are compatible for kl_divergence
                # For Independent(BaseDist), kl_divergence should work if BaseDists are same type
                kl_div = kl_divergence(dist, dist_ref).mean()

                total_loss = policy_loss + self.kl_coef * kl_div - self.entropy_coef * entropy

                if self.actor_optimizer: self.actor_optimizer.zero_grad()
                with self.timer("backward_pass"): total_loss.backward()
                
                if self.actor: # self.actor is Sequential for CNN, or MLP direct
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                
                with self.timer("optimizer_step"):
                    if self.actor_optimizer: self.actor_optimizer.step()
                
                all_losses["policy_loss"].append(policy_loss.item())
                all_losses["kl_loss"].append(kl_div.item())
                all_losses["entropy_loss"].append(entropy.item())
                all_losses["total_loss"].append(total_loss.item())

        if self._steps_since_ref_update >= self.ref_update_interval:
            self.logger.info(f"Updating GRPO reference policy after {self._steps_since_ref_update} steps.")
            if self.actor and self.actor_ref:
                 self.actor_ref.load_state_dict(self.actor.state_dict())
            self._steps_since_ref_update = 0

        return {key: np.mean(val_list) if val_list else 0.0 for key, val_list in all_losses.items()}