from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, field_validator

class NetworkConfig(BaseModel):
    network_type: str = Field("mlp", pattern="^(mlp|cnn)$")
    mlp_hidden_dims: Tuple[int, ...] = (64, 64)
    cnn_output_features: PositiveInt = 256

class PPOConfig(BaseModel):
    distribution_type: str = Field("normal", pattern="^(normal|beta)$")
    lam: PositiveFloat = 0.95  # GAE lambda
    clip_eps: PositiveFloat = 0.2
    ppo_epochs: PositiveInt = 10
    num_minibatches: PositiveInt = 32
    entropy_coef: float = 0.01
    value_coef: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    target_kl: Optional[PositiveFloat] = None
    rollout_steps: PositiveInt = 2048
    lr: PositiveFloat = 3e-4

class GRPOConfig(BaseModel):
    distribution_type: str = Field("normal", pattern="^(normal|beta)$") # Added for GRPO
    group_size: PositiveInt = 64
    update_epochs: PositiveInt = 10
    max_grad_norm: PositiveFloat = 0.5
    entropy_coef: float = 0.001 # Can be negative for Beta if desired
    kl_coef: float = 0.01
    ref_update_interval: PositiveInt = 10_000
    minibatch_size: PositiveInt = 256 # For GRPO update phase
    lr: PositiveFloat = 1e-4
    rollout_steps_per_trajectory: PositiveInt = 1000 # Approx max steps per trajectory in a group


class ExperimentConfig(BaseModel):
    env_id: str
    algo: str = Field(pattern="^(ppo|grpo)$") # Simplified, specific type (normal/beta) in PPOConfig
    seed: int = 0
    gamma: PositiveFloat = Field(0.99, ge=0.0, le=1.0)
    total_steps: PositiveInt = 1_000_000
    log_interval: PositiveInt = 5000
    checkpoint_interval: PositiveInt = 50000
    video_interval: PositiveInt = 100_000
    run_name: Optional[str] = None
    base_log_dir: str = "experiment_runs"
    verbose: bool = False
    max_episode_steps: Optional[PositiveInt] = None # For TimeLimit wrapper

    network_config: NetworkConfig = Field(default_factory=NetworkConfig)
    
    # Algorithm-specific configs
    ppo_config: Optional[PPOConfig] = None
    grpo_config: Optional[GRPOConfig] = None

    @field_validator('algo')
    def algo_name_check(cls, v: str) -> str:
        if v.lower() not in ["ppo", "grpo"]: # PPO covers both ppo_gauss and ppo_beta now
            raise ValueError("Algorithm must be 'ppo' or 'grpo'.")
        return v.lower()

    @field_validator('ppo_config', 'grpo_config')
    def ensure_correct_algo_config(cls, v: Optional[Union[PPOConfig, GRPOConfig]], values: Any) -> Optional[Union[PPOConfig, GRPOConfig]]:
        # Pydantic v2 way to access other fields is via values.data
        data = values.data 
        algo = data.get('algo')
        if algo == 'ppo' and v is None and data.get('ppo_config') is None:
            # If algo is ppo and ppo_config is not provided, create default
            # This happens if user only specifies "algo": "ppo"
            return PPOConfig() 
        if algo == 'grpo' and v is None and data.get('grpo_config') is None:
            return GRPOConfig()
        if algo == 'ppo' and data.get('grpo_config') is not None:
            raise ValueError("grpo_config should not be provided when algo is 'ppo'")
        if algo == 'grpo' and data.get('ppo_config') is not None:
            raise ValueError("ppo_config should not be provided when algo is 'grpo'")
        return v
    
    # Helper to get the specific config
    def get_algo_specific_config(self) -> Union[PPOConfig, GRPOConfig]:
        if self.algo == "ppo":
            if self.ppo_config is None: # Should be created by validator if not present
                 self.ppo_config = PPOConfig()
            return self.ppo_config
        elif self.algo == "grpo":
            if self.grpo_config is None:
                self.grpo_config = GRPOConfig()
            return self.grpo_config
        raise ValueError(f"Unknown algorithm type for specific config: {self.algo}")

class RolloutBufferSamples(BaseModel):
    observations: torch.Tensor
    actions: torch.Tensor
    actions_canonical_unclipped: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: Optional[torch.Tensor] = None # For PPO
    values: Optional[torch.Tensor] = None  # For PPO

    class Config:
        arbitrary_types_allowed = True