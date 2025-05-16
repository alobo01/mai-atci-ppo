from typing import Dict, Generator, Optional
import numpy as np
import torch

from utils.pydantic_models import RolloutBufferSamples
from utils.torch_utils import Tensor, to_tensor # Assuming it's in utils/


class RolloutBuffer:
    """
    A buffer for storing trajectories experienced by an agent interacting with
    the environment. Used for PPO and similar on-policy algorithms.
    """
    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_shape: tuple, # For continuous, this is action_dim. For discrete, usually (1,) or empty.
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        is_continuous: bool = True
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape if isinstance(action_shape, tuple) else (action_shape,) # Ensure tuple
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.is_continuous = is_continuous

        self.observations: np.ndarray = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        
        action_dtype = np.float32 if is_continuous else np.int64
        # For discrete, action_shape might be () if env.action_space.shape is (), adjust if needed
        # For PPO, act_buf usually stores the action itself.
        actual_action_shape = self.action_shape if self.action_shape and self.action_shape[0] > 0 else ()

        self.actions: np.ndarray = np.zeros((buffer_size, *actual_action_shape), dtype=action_dtype)
        self.rewards: np.ndarray = np.zeros((buffer_size,), dtype=np.float32)
        self.advantages: np.ndarray = np.zeros((buffer_size,), dtype=np.float32)
        self.returns: np.ndarray = np.zeros((buffer_size,), dtype=np.float32)
        self.episode_starts: np.ndarray = np.zeros((buffer_size,), dtype=np.float32) # For GAE calculation
        self.values: np.ndarray = np.zeros((buffer_size,), dtype=np.float32) # For PPO
        self.log_probs: np.ndarray = np.zeros((buffer_size,), dtype=np.float32) # For PPO
        self.actions_canonical_unclipped: np.ndarray = np.zeros((buffer_size, *actual_action_shape), dtype=np.float32)
        self.ptr: int = 0
        self.path_start_idx: int = 0 # For GAE
        self.full: bool = False


    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        episode_start: bool, # True if this is the first step of an episode
        value: float,        # Value estimate for PPO
        log_prob: float,     # Log probability of the action for PPO
        action_canonical: np.ndarray # The unclipped action
    ) -> None:
        """
        Adds a new transition to the buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            episode_start: Whether this step is the start of a new episode.
            value: Value estimate from the critic (for PPO).
            log_prob: Log probability of the action (for PPO).
        """
        if self.ptr >= self.buffer_size:
            # This case should ideally be handled by resetting the buffer
            # or indicating it's full before calling add.
            # For now, let's overwrite (circular buffer behavior).
            self.ptr = 0
            self.full = True # Mark as full if we wrap around

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.episode_starts[self.ptr] = float(episode_start)
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.actions_canonical_unclipped[self.ptr] = action_canonical
        
        self.ptr += 1

    def finish_path(self, last_value: float = 0.0) -> None:
        """
        Finalizes the GAE computation for the current path (trajectory segment).
        Should be called at the end of a rollout or when an episode terminates.

        Args:
            last_value: Value estimate of the state after the last recorded step.
                        0.0 if the episode terminated.
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards_path = self.rewards[path_slice]
        values_path = self.values[path_slice]
        episode_starts_path = self.episode_starts[path_slice]

        gae_lambda_val = 0.0
        for t in reversed(range(len(rewards_path))):
            # Non-terminal is 1 if not episode_start at t+1, or if t is last step in segment
            if t == len(rewards_path) - 1: # Last step in the current segment
                next_non_terminal = 1.0 - self.episode_starts[self.ptr] if self.ptr < self.buffer_size and not self.episode_starts[self.ptr] else (1.0 if not self.episode_starts[self.path_start_idx] else 0.0) # Check next step in buffer if available
                next_value = last_value
            else:
                next_non_terminal = 1.0 - episode_starts_path[t + 1]
                next_value = values_path[t + 1]
            
            delta = rewards_path[t] + self.gamma * next_value * next_non_terminal - values_path[t]
            gae_lambda_val = delta + self.gamma * self.gae_lambda * next_non_terminal * gae_lambda_val
            self.advantages[self.path_start_idx + t] = gae_lambda_val
        
        self.returns[path_slice] = self.advantages[path_slice] + values_path
        self.path_start_idx = self.ptr

        if self.ptr >= self.buffer_size:
            self.full = True


    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """
        Returns a generator of minibatches from the buffer.
        Data is shuffled before creating minibatches.
        Advantages are normalized.

        Args:
            batch_size: The size of each minibatch. If None, returns the whole buffer
                        as one batch (after shuffling and converting to tensors).

        Yields:
            RolloutBufferSamples objects for each minibatch.
        """
        if not self.full and self.ptr < self.buffer_size:
            # If buffer is not full, only use data up to self.ptr
            actual_size = self.ptr
        else:
            actual_size = self.buffer_size

        if actual_size == 0:
            # print("Warning: Attempting to get batches from an empty buffer.")
            return # Yield nothing

        indices = np.random.permutation(actual_size)

        # Normalize advantages
        advantages_np = self.advantages[:actual_size]
        normalized_advantages = (advantages_np - np.mean(advantages_np)) / (np.std(advantages_np) + 1e-8)

        # Convert all relevant numpy arrays to tensors
        observations_tensor = to_tensor(self.observations[:actual_size], self.device)
        actions_tensor = to_tensor(self.actions[:actual_size], self.device) # Dtype handled by to_tensor
        log_probs_tensor = to_tensor(self.log_probs[:actual_size], self.device)
        advantages_tensor = to_tensor(normalized_advantages, self.device)
        returns_tensor = to_tensor(self.returns[:actual_size], self.device)
        values_tensor = to_tensor(self.values[:actual_size], self.device)
        actions_canonical_tensor = to_tensor(self.actions_canonical_unclipped[:actual_size], self.device)

        if batch_size is None:
            batch_size = actual_size

        start_idx = 0
        while start_idx < actual_size:
            end_idx = min(start_idx + batch_size, actual_size)
            mb_indices = indices[start_idx:end_idx]
            
            yield RolloutBufferSamples(
                observations=observations_tensor[mb_indices],
                actions_canonical_unclipped=actions_canonical_tensor[mb_indices],
                actions=actions_tensor[mb_indices],
                log_probs=log_probs_tensor[mb_indices],
                advantages=advantages_tensor[mb_indices],
                returns=returns_tensor[mb_indices],
                values=values_tensor[mb_indices]
            )
            start_idx += batch_size
            
    def reset(self) -> None:
        """Resets the buffer pointer and path start index."""
        self.ptr = 0
        self.path_start_idx = 0
        self.full = False

    @property
    def size(self) -> int:
        """Current number of elements in the buffer."""
        return self.ptr if not self.full else self.buffer_size

    # GRPO Specific Buffer Usage (Alternative to GAE advantages)
    # GRPO calculates advantages differently (normalized group returns).
    # It might populate 'advantages' directly after a group rollout,
    # or use a separate buffer/mechanism.
    # For now, this buffer is primarily PPO-oriented with GAE.
    # GRPO can adapt by:
    # 1. Not using `finish_path` and `returns` from this buffer.
    # 2. Populating `self.advantages` with its group-normalized returns directly.
    #    `self.log_probs` would still be from the acting policy during group rollout.
    
    def add_grpo_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        action_canonical: np.ndarray
        # For GRPO, reward/value are not stored per step in the same way
        # Instead, advantage is calculated per trajectory after group rollout
    ) -> None:
        """Simplified add for GRPO, primarily storing obs, actions, log_probs."""
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.full = True

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.actions_canonical_unclipped[self.ptr] = action_canonical
        # 'rewards', 'values', 'episode_starts' are not strictly needed for GRPO's update
        # if advantages are computed externally and assigned.
        self.ptr += 1

    def assign_grpo_advantages(self, advantages_per_step: np.ndarray) -> None:
        """
        Assigns pre-computed advantages to the steps in the buffer.
        Used by GRPO after calculating group-normalized returns.
        Assumes advantages_per_step has length self.ptr or self.buffer_size.
        """
        current_fill = self.ptr if not self.full else self.buffer_size
        if len(advantages_per_step) != current_fill:
            raise ValueError(f"Length of GRPO advantages ({len(advantages_per_step)}) "
                             f"does not match buffer fill ({current_fill}).")
        self.advantages[:current_fill] = advantages_per_step
        # GRPO does not use 'returns' or 'values' in the same way as PPO for its main objective.

    def get_grpo_batches(self, batch_size: int) -> Generator[RolloutBufferSamples, None, None]:
        """
        Returns a generator of minibatches for GRPO.
        Assumes `assign_grpo_advantages` has been called.
        Does not include `returns` or `values` as they are not used by GRPO's core objective.
        """
        actual_size = self.ptr if not self.full else self.buffer_size
        if actual_size == 0:
            return

        indices = np.random.permutation(actual_size)

        # Advantages are assumed to be already normalized by GRPO logic before assignment
        observations_tensor = to_tensor(self.observations[:actual_size], self.device)
        actions_tensor = to_tensor(self.actions[:actual_size], self.device)
        log_probs_tensor = to_tensor(self.log_probs[:actual_size], self.device)
        advantages_tensor = to_tensor(self.advantages[:actual_size], self.device)
        actions_canonical_tensor = to_tensor(self.actions_canonical_unclipped[:actual_size], self.device)


        start_idx = 0
        while start_idx < actual_size:
            end_idx = min(start_idx + batch_size, actual_size)
            mb_indices = indices[start_idx:end_idx]
            
            yield RolloutBufferSamples(
                observations=observations_tensor[mb_indices],
                actions=actions_tensor[mb_indices],
                actions_canonical_unclipped=actions_canonical_tensor[mb_indices],
                log_probs=log_probs_tensor[mb_indices],
                advantages=advantages_tensor[mb_indices],
                # returns and values are None for GRPO samples from this method
            )
            start_idx += batch_size