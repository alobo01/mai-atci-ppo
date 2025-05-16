import threading
from typing import Optional

import gymnasium as gym

# A single lock that covers any glfwInit()/RegisterClassEx
# to prevent issues with parallel environment creation (e.g. in Atari).
_GLFW_LOCK = threading.Lock()

def make_env(
    env_id: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
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
    env: gym.Env
    # Using a lock for gym.make can prevent some threading issues with GUI libraries
    # like glfw used by some environments (e.g. MuJoCo rendering)
    with _GLFW_LOCK:
        try:
            env = gym.make(env_id, render_mode=render_mode)
        except Exception as e:
            print(f"Error creating env '{env_id}' with render_mode='{render_mode}': {e}")
            if render_mode:
                try:
                    print("Retrying environment creation without render_mode...")
                    env = gym.make(env_id)
                except Exception as e2:
                    raise RuntimeError(f"Failed to create env '{env_id}' even without render_mode.") from e2
            else:
                raise e

    # Apply TimeLimit wrapper unless already applied
    # Some envs (like from `gymnasium.make(..., max_episode_steps=X)`) might come pre-wrapped
    is_time_limit_wrapped = False
    temp_env = env
    while hasattr(temp_env, 'env'):
        if isinstance(temp_env, gym.wrappers.TimeLimit):
            is_time_limit_wrapped = True
            break
        temp_env = temp_env.env # Check inner envs

    if not is_time_limit_wrapped:
        if max_episode_steps is None:
            spec_max_steps = getattr(env.spec, "max_episode_steps", None)
            max_episode_steps = spec_max_steps if spec_max_steps is not None else 1000 # Default if not in spec
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if seed is not None:
        # Modern way to seed env and action_space
        obs, info = env.reset(seed=seed) # Returns obs and info
        env.action_space.seed(seed)

    return env