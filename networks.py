"""
Neural network architectures for RL agents.

Author: Antonio Lobo
"""
from __future__ import annotations

from typing import Dict, Tuple, Type, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

class FeedForwardNN(nn.Module):
    """
    A simple feed-forward neural network (MLP) for Actor or Critic.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Args:
            input_dim: Dimension of the input observation.
            output_dim: Dimension of the output (action dim * 2 for Gaussian Actor,
                       action dim * 2 for Beta Actor, 1 for Critic).
            hidden_dims: Tuple of hidden layer sizes.
            activation: Activation function class (e.g., nn.Tanh, nn.ReLU).
        """
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            in_dim = h_dim
        # Output layer (linear activation)
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass."""
        # Ensure input is float
        if obs.dtype != torch.float32:
            obs = obs.float()
        return self.network(obs)

class CNNFeatureExtractor(nn.Module): # Renamed from CNNNetwork for clarity
    """
    A CNN for extracting features from image-based observations.
    Outputs a feature vector.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...], # e.g., (C, H, W) = (3, 96, 96)
        output_features: int = 256, # Size of the feature vector output
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Args:
            obs_shape: Shape of the input observation (Channels, Height, Width).
            output_features: Number of features output by the CNN base.
            activation: Activation function class.
        """
        super().__init__()
        c, h, w = obs_shape
        # Example CNN architecture (adjust based on input size and complexity)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), # (N, 32, 23, 23) if input 96x96
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (N, 64, 10, 10)
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (N, 64, 8, 8)
            activation(),
            nn.Flatten(),
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            # Create dummy input on CPU to avoid potential CUDA initialization here
            dummy_input = torch.zeros(1, *obs_shape, device='cpu')
            conv_out_size = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential( # Changed fc to Sequential
            nn.Linear(conv_out_size, output_features),
            activation() # Apply activation after the linear layer
        )
        self.output_features = output_features


    def forward(self, obs: Tensor) -> Tensor:
        """Extract features from image observations."""
        # Preprocess: scale observations (e.g., to [0, 1] or [-1, 1])
        if obs.dtype != torch.float32:
            obs = obs.float()
        # Ensure normalization happens correctly (move maybe to wrapper or agent?)
        if obs.max() > 1.1: # Allow for slight overshoot post-normalization
             obs = obs / 255.0

        # Handle potential missing batch dimension
        if obs.ndim == 3:
            obs = obs.unsqueeze(0) # Add batch dim: (C, H, W) -> (1, C, H, W)

        # Permute if needed (e.g., from HWC to CHW if env provides HWC)
        # Assuming input `obs` is already in CHW format (common for PyTorch CNNs)
        if obs.shape[-1] in [3, 1]: # Check if last dim is channels
            obs = obs.permute(0, 3, 1, 2)
        features = self.conv(obs)
        features = self.fc(features) # Pass through final linear + activation
        return features



# Registry - simplify, just point to basic building blocks
NETWORK_REGISTRY: Dict[str, Type[nn.Module]] = {
    "mlp": FeedForwardNN,
    "cnn": CNNFeatureExtractor, # Use this for feature extraction
}