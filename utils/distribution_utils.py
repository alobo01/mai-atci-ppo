from typing import Tuple, Callable
import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Normal, Beta, Categorical, Independent
import gymnasium as gym

from .torch_utils import Tensor # Assuming torch_utils is in the same directory

ActionPostprocessor = Callable[[Tensor], Tensor]

def create_distribution_from_actor_output(
    actor_output: Tensor,
    distribution_type: str,
    action_space: gym.Space,
    action_dim: int, # Explicitly pass action_dim
) -> Tuple[Distribution, ActionPostprocessor]:
    """
    Creates an action distribution from the actor network's output.
    The Tanh squashing transformation for Normal distribution has been removed.
    Actions sampled from Normal distribution will be directly used and should be
    clipped by the environment or a wrapper if they exceed action_space bounds.

    Args:
        actor_output: Raw tensor output from the actor network.
        distribution_type: String specifying 'normal' or 'beta'.
        action_space: The environment's action space.
        action_dim: The dimensionality of the action space.

    Returns:
        A tuple containing:
            - dist: The torch.distributions.Distribution object.
            - postprocessor: A function to map raw samples to environment actions.
    """
    env_low = torch.as_tensor(action_space.low, device=actor_output.device, dtype=torch.float32)
    env_high = torch.as_tensor(action_space.high, device=actor_output.device, dtype=torch.float32)

    if distribution_type == "normal":
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("Normal distribution requires Box action space.")
        
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        std = torch.exp(log_std.clamp(min=-20, max=2)) # Clamping for stability
        
        base_dist = Normal(mean, std)
        # If action_dim > 1, Independent sums log_probs over action dimensions.
        # If action_dim == 1, Independent is effectively an identity wrapper here.
        dist = Independent(base_dist, 1) # Reinterpret batch_of_independent as multivariate

        def postprocessor_normal(raw_action: Tensor) -> Tensor:
            # Raw action is sampled directly from Normal(mean, std)
            # It's the model's responsibility to learn mean/std s.t. samples are mostly within bounds.
            # We must clip to ensure actions are valid for the environment.
            return torch.clamp(raw_action, env_low, env_high)

        return dist, postprocessor_normal

    elif distribution_type == "beta":
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError("Beta distribution requires Box action space.")
        
        raw_alpha, raw_beta = torch.chunk(actor_output, 2, dim=-1)
        # Ensure alpha, beta >= 1 for a unimodal distribution (can be <1 for U-shaped)
        # Using softplus + 1 ensures > 1.
        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta) + 1.0
        
        base_dist = Beta(alpha, beta) # Samples are in [0, 1]
        dist = Independent(base_dist, 1)

        def postprocessor_beta(raw_action_01: Tensor) -> Tensor:
            # raw_action_01 is sampled from Beta, in [0, 1]
            # Clamp for numerical stability at boundaries, though Beta should handle this.
            raw_action_01_clamped = torch.clamp(raw_action_01, 1e-6, 1.0 - 1e-6)
            env_action = env_low + (env_high - env_low) * raw_action_01_clamped
            return torch.clamp(env_action, env_low, env_high) # Final safety clamp

        return dist, postprocessor_beta

    elif distribution_type == "categorical": # For discrete action spaces
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("Categorical distribution requires Discrete action space.")
        # actor_output here are logits
        dist = Categorical(logits=actor_output)
        
        def postprocessor_categorical(raw_action: Tensor) -> Tensor:
            # Raw action is the discrete action index itself
            return raw_action
        
        return dist, postprocessor_categorical

    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

def unscale_action_for_beta_log_prob(
    action_env_scale: Tensor,
    action_space_low: Tensor,
    action_space_high: Tensor
) -> Tensor:
    """
    Unscales an action from the environment's [low, high] range back to [0, 1]
    for Beta distribution log_prob calculation. Clamps the result to (eps, 1-eps).

    Args:
        action_env_scale: Action tensor in the environment's scale.
        action_space_low: Tensor of lower bounds of the action space.
        action_space_high: Tensor of upper bounds of the action space.

    Returns:
        Action tensor scaled to (eps, 1-eps).
    """
    action_range = action_space_high - action_space_low
    # Add small epsilon to range to prevent division by zero if low==high
    action_range = torch.max(action_range, torch.tensor(1e-8, device=action_range.device))

    action_01_scale = (action_env_scale - action_space_low) / action_range
    # Clamp to be strictly within (0, 1) for Beta log_prob stability
    eps = 1e-6
    return torch.clamp(action_01_scale, eps, 1.0 - eps)