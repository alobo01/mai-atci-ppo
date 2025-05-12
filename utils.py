"""
Shared helpers for RL experiments.

Author: Antonio Lobo
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from functools import wraps
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional,
                    Protocol, Tuple, Type, Union)

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from torch.distributions import (Beta, Categorical, Distribution,
                                 Independent, Normal, TransformedDistribution)
from torch.distributions.transforms import TanhTransform
import torch.nn.functional as F

if TYPE_CHECKING:
    from .base_agent import BaseAgent # Avoid circular import

# Type Aliases
NpArray = np.ndarray
Tensor = torch.Tensor
Device = Union[torch.device, str]
Loggable = Union[int, float, str, bool]


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    enabled: bool = True, # Enable by default for experiment runs
) -> logging.Logger:
    """
    Returns a configured python `logging.Logger`.

    Args:
        name: Logger name (e.g., PPO_CarRacing_seed0).
        log_dir: Optional directory to save log file (<name>.log).
        level: Logging level (DEBUG, INFO, ...).
        enabled: Master switch. If False, uses NullHandler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Prevent duplicate handlers
        return logger

    if not enabled:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.WARNING) # Still allow critical messages
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional File Handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Prevent propagation to root logger
    logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """Seed python, numpy, torch (+ cuda) & set Gym deterministic flags."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
        # Potentially slower, but ensures reproducibility
        torch.backends.cudnn.deterministic = deterministic_torch
        torch.backends.cudnn.benchmark = not deterministic_torch # Faster if False
        if deterministic_torch:
             # Ensure PyTorch >= 1.7 for this
            try:
                 torch.use_deterministic_algorithms(True)
            except AttributeError:
                 print("Warning: torch.use_deterministic_algorithms not available. Requires PyTorch 1.7+.")
            # Set CUBLAS workspace config for deterministic matmuls if needed
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"


# ─────────────────────────────────────────────────────────────────────────────
# Environment Handling
# ─────────────────────────────────────────────────────────────────────────────

def make_env(
    env_id: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    # Add other potential wrapper args here if needed
) -> gym.Env:
    """
    Creates a Gymnasium environment, seeds it, and applies TimeLimit.

    Args:
        env_id: Standard Gym environment ID string.
        seed: Optional random seed for the environment.
        render_mode: Gym render mode ('human', 'rgb_array', etc.).
        max_episode_steps: Override default max steps. If None, uses env.spec.

    Returns:
        Initialized Gymnasium environment.
    """
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as e:
        print(f"Error creating env '{env_id}': {e}")
        # Attempt without render_mode if it was the cause
        if render_mode:
            try:
                print(f"Retrying without render_mode...")
                env = gym.make(env_id)
            except Exception as e2:
                 raise RuntimeError(f"Failed to create env '{env_id}' even without render_mode.") from e2
        else:
            raise e


    # Apply TimeLimit wrapper unless already applied
    if not isinstance(env, gym.wrappers.TimeLimit):
        if max_episode_steps is None:
            spec_max = getattr(env.spec, "max_episode_steps", None)
            # Use reasonable default if spec is missing max steps
            max_episode_steps = spec_max if spec_max is not None else 1000
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    # Seed the environment
    if seed is not None:
        # Use reset(seed=...) for modern Gym versions
        env.reset(seed=seed)
        # Also seed the action space for reproducibility if possible
        env.action_space.seed(seed)

    return env


# ─────────────────────────────────────────────────────────────────────────────
# Torch & Tensor Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device(device: Optional[Device] = None) -> torch.device:
    """Gets the torch device, defaulting to CUDA if available, else CPU."""
    if isinstance(device, torch.device):
        return device
    if device == "cuda" or (device is None and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(
    arr: Union[NpArray, List[Any], Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Converts input to a Torch Tensor on the specified device."""
    if isinstance(arr, Tensor):
        return arr.to(device=device, dtype=dtype)
    # Handle lists potentially containing non-numeric types carefully if needed
    # For standard RL usage, converting to numpy first is robust.
    np_arr = np.asarray(arr)
    return torch.as_tensor(np_arr, dtype=dtype, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Action Distribution Handling
# ─────────────────────────────────────────────────────────────────────────────

class ActionPostprocessor(Protocol):
    def __call__(self, raw_action: Tensor) -> Tensor: ...

def get_action_distribution(
    actor_output: Tensor, # Raw output from the actor network
    action_space: gym.Space,
) -> Tuple[Distribution, ActionPostprocessor]:
    """
    Creates the appropriate action distribution based on the action space.

    Args:
        actor_output: The raw tensor output from the actor network.
        action_space: The environment's action space instance.

    Returns:
        A tuple containing:
            - dist: The torch.distributions.Distribution object.
            - postprocessor: A function to map raw samples to env actions.
    """
    if isinstance(action_space, gym.spaces.Box):
        # Assumes actor_output contains mean and log_std concatenated
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        std = torch.exp(log_std.clamp(min=-20, max=2)) # Clamp for stability

        # Use Normal for potentially multi-dimensional actions
        base_dist = Normal(mean, std)
        # Use Independent to handle correlations if needed, sums log_prob across dims
        # For diagonal covariance assumed here, Independent sums log probs.
        dist = Independent(base_dist, 1) # Sum over the last dim (action dim)

        # Tanh squashing for bounded actions [-1, 1] -> [low, high]
        # Note: Some implementations skip squashing for PPO, relying on clipping.
        #       Let's include it as it's common, especially with Beta dist.
        #       If using Beta dist, it handles bounds differently.
        # This logic assumes Gaussian PPO needs squashing.
        # If using Beta, the PPO_Beta class will override _dist.

        # low/high tensors on the correct device
        low = torch.as_tensor(action_space.low, device=mean.device, dtype=mean.dtype)
        high = torch.as_tensor(action_space.high, device=mean.device, dtype=mean.dtype)
        action_range = high - low

        # Use TanhTransform if action bounds are finite and symmetric around 0 often
        # For general bounds, rescale manually after sampling
        # Let's provide a manual rescale postprocessor
        def postprocessor(raw_action: Tensor) -> Tensor:
            # Assume raw_action comes from Normal dist (unbounded)
            # Apply tanh to squash to (-1, 1)
            squashed_action = torch.tanh(raw_action)
            # Rescale to [low, high]
            env_action = low + action_range * (squashed_action + 1.0) / 2.0
            # Clamp for safety, although tanh should prevent exceeding bounds much
            return torch.clamp(env_action, low, high)

        # If we want a Tanh-squashed *distribution* itself (alters log_prob):
        # squash_transform = TanhTransform(cache_size=1)
        # dist = TransformedDistribution(base_dist, squash_transform)
        # dist = Independent(dist, 1) # Apply independent after transform
        # def postprocessor_squashed(y: Tensor) -> Tensor: # y is already in [-1, 1]
        #     return low + action_range * (y + 1.0) / 2.0
        # return dist, postprocessor_squashed

        # Returning the Normal dist + postprocessor applying tanh and rescale
        return dist, postprocessor

    elif isinstance(action_space, gym.spaces.Discrete):
        # Assumes actor_output contains logits
        dist = Categorical(logits=actor_output)
        # Postprocessor is identity for discrete actions
        def identity_postprocessor(raw_action: Tensor) -> Tensor:
            return raw_action
        return dist, identity_postprocessor

    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported.")


def get_beta_distribution(
    actor_output: Tensor, # Raw output (alpha_raw, beta_raw)
    action_space: gym.spaces.Box, # Must be Box space
) -> Tuple[Distribution, ActionPostprocessor]:
    """Creates a Beta distribution for continuous actions in [0, 1]."""
    if not isinstance(action_space, gym.spaces.Box):
         raise ValueError("Beta distribution only supports Box action spaces.")

    raw_alpha, raw_beta = torch.chunk(actor_output, 2, dim=-1)
    # Softplus + 1 ensures alpha, beta > 1 (unimodal distribution)
    alpha = F.softplus(raw_alpha) + 1.0
    beta = F.softplus(raw_beta) + 1.0
    dist = Beta(alpha, beta)

    # Postprocessor to scale [0, 1] samples to env's [low, high]
    low = torch.as_tensor(action_space.low, device=alpha.device, dtype=alpha.dtype)
    high = torch.as_tensor(action_space.high, device=alpha.device, dtype=alpha.dtype)
    action_range = high - low

    def postprocessor(raw_action_01: Tensor) -> Tensor:
        # raw_action_01 is sampled from Beta, already in [0, 1]
        # Clamp for numerical stability at boundaries
        raw_action_01 = torch.clamp(raw_action_01, 1e-6, 1.0 - 1e-6)
        env_action = low + action_range * raw_action_01
        return torch.clamp(env_action, low, high) # Final clamp for safety

    # For Beta, we often need the inverse transform for log_prob calculation
    # The PPO_Beta class handles this internally.

    return dist, postprocessor

# ─────────────────────────────────────────────────────────────────────────────
# Timing Utilities
# ─────────────────────────────────────────────────────────────────────────────

class Timing:
    """Simple context manager and storage for timing code blocks."""
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._start_times: Dict[str, float] = {}

    def __enter__(self) -> Timing:
        # Allows timing multiple things simultaneously if needed
        # but primarily used with 'with timer("key"):'
        return self

    def __call__(self, key: str) -> Timing:
        """Starts timing for a specific key."""
        self._start_times[key] = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stops timing for the most recently started key."""
        # This basic version assumes only one timer active via __call__ at a time
        # More robust implementation might track a stack of keys
        if self._start_times:
            key = list(self._start_times.keys())[-1] # Get last started key
            elapsed = time.perf_counter() - self._start_times.pop(key)
            self.totals[key] = self.totals.get(key, 0.0) + elapsed
            self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self, reset: bool = True) -> Dict[str, Dict[str, float]]:
        """Returns average times per key and optionally resets."""
        avg_times = {}
        for key, total_time in self.totals.items():
            count = self.counts.get(key, 1) # Avoid division by zero
            avg_times[key] = {
                "total_ms": total_time * 1000.0,
                "count": count,
                "avg_ms": (total_time / count) * 1000.0 if count > 0 else 0.0,
            }
        if reset:
            self.totals.clear()
            self.counts.clear()
            self._start_times.clear()
        return avg_times

# Decorator version (simpler for functions)
def timed(key: str, timer_instance: Timing) -> Callable[[Callable], Callable]:
    """Decorator to time a function using a Timing instance."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with timer_instance(key):
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(agent: 'BaseAgent', dir_path: Path, step: int) -> None:
    """Saves agent state (networks, optimizers) to directory."""
    dir_path.mkdir(parents=True, exist_ok=True)
    save_data = {"step": step}

    # Actor network and optimizer
    if hasattr(agent, "actor") and agent.actor is not None:
        torch.save(agent.actor.state_dict(), dir_path / "actor.pt")
    if hasattr(agent, "actor_optimizer") and agent.actor_optimizer is not None:
        torch.save(agent.actor_optimizer.state_dict(), dir_path / "opt_actor.pt")

    # Critic network and optimizer (optional)
    if hasattr(agent, "critic") and agent.critic is not None:
        torch.save(agent.critic.state_dict(), dir_path / "critic.pt")
    if hasattr(agent, "critic_optimizer") and agent.critic_optimizer is not None:
        torch.save(agent.critic_optimizer.state_dict(), dir_path / "opt_critic.pt")

    # Save step count
    with open(dir_path / "checkpoint_info.json", "w") as f:
        json.dump(save_data, f)

    agent.logger.debug(f"Checkpoint saved at step {step} to {dir_path}")


def load_checkpoint(agent: 'BaseAgent', dir_path: Path) -> int:
    """Loads agent state from directory. Returns starting step."""
    start_step = 0
    info_path = dir_path / "checkpoint_info.json"
    actor_path = dir_path / "actor.pt"
    opt_actor_path = dir_path / "opt_actor.pt"
    critic_path = dir_path / "critic.pt"
    opt_critic_path = dir_path / "opt_critic.pt"

    if not actor_path.is_file():
        agent.logger.info("No checkpoint found, starting fresh.")
        return start_step # Fresh run

    # Load step count
    if info_path.is_file():
        with open(info_path, "r") as f:
            try:
                info = json.load(f)
                start_step = info.get("step", 0)
            except json.JSONDecodeError:
                agent.logger.warning(f"Could not read {info_path}, step count lost.")

    agent.logger.info(f"Loading checkpoint from {dir_path}, resuming at step {start_step}")

    # Load actor
    if hasattr(agent, "actor") and agent.actor is not None:
        agent.actor.load_state_dict(torch.load(actor_path, map_location=agent.device))
    if hasattr(agent, "actor_optimizer") and agent.actor_optimizer is not None and opt_actor_path.is_file():
        agent.actor_optimizer.load_state_dict(torch.load(opt_actor_path, map_location=agent.device))

    # Load critic (optional)
    if hasattr(agent, "critic") and agent.critic is not None and critic_path.is_file():
        agent.critic.load_state_dict(torch.load(critic_path, map_location=agent.device))
    if hasattr(agent, "critic_optimizer") and agent.critic_optimizer is not None and opt_critic_path.is_file():
        agent.critic_optimizer.load_state_dict(torch.load(opt_critic_path, map_location=agent.device))

    # Ensure networks are in eval/train mode appropriately after loading
    if hasattr(agent, "actor"): agent.actor.train()
    if hasattr(agent, "critic"): agent.critic.train()

    return start_step

# ─────────────────────────────────────────────────────────────────────────────
# Video / Plotting Helpers
# ─────────────────────────────────────────────────────────────────────────────

def overlay_text(frame: NpArray, text: str, color: Tuple[int,int,int]=(255,255,255)) -> NpArray:
    """Draws text with a semi-transparent background box on the frame."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = 5
    # Top-left corner box
    top_left = (margin, margin)
    bottom_right = (margin + w + 2*margin, margin + h + baseline + 2*margin)
    # Draw black rectangle background with some transparency
    overlay = out.copy()
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1) # Black filled
    alpha = 0.6 # Transparency factor
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    # Put white text on top
    text_pos = (margin + margin, margin + h + margin)
    cv2.putText(out, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return out

def save_video(frames: List[NpArray], filename: Union[str, Path], fps: int = 30) -> None:
    """Saves a list of frames as an MP4 video."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(filename, frames, fps=fps, quality=8) # quality (0-10)

def save_metrics(metrics: Dict[str, List[Loggable]], filename: Union[str, Path]) -> None:
    """Saves metrics dictionary to a JSON file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

def load_metrics(filename: Union[str, Path]) -> Dict[str, List[Loggable]]:
    """Loads metrics dictionary from a JSON file."""
    if not Path(filename).is_file():
        return {"steps": [], "avg_episodic_reward": [], "avg_episode_length": []} # Default structure
    with open(filename, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}. Returning empty metrics.")
            return {"steps": [], "avg_episodic_reward": [], "avg_episode_length": []}

def save_timings(timing_summary: Dict[str, Dict[str, float]], filename: Union[str, Path], step: int) -> None:
    """Appends timing summary for the current step to a JSON Lines file."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    log_entry = {"step": step, **timing_summary}
    with open(filename, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

