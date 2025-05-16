from typing import Tuple, Type
import torch
import torch.nn as nn

Tensor = torch.Tensor

class CNNFeatureExtractor(nn.Module):
    """A CNN for extracting features from image-based observations."""
    def __init__(
        self,
        obs_shape: Tuple[int, ...], # Expected: (C, H, W) e.g., (3 or 4, 84, 84)
        output_features: int = 256,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Args:
            obs_shape: Shape of the input observation (Channels, Height, Width).
            output_features: Number of features output by the CNN base.
            activation: Activation function class for hidden layers.
        """
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"Expected obs_shape in (C, H, W) format, got {obs_shape}")
        
        c, h, w = obs_shape # Channels, Height, Width

        # Standard CNN architecture (Nature DQN style, adjust as needed)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), # Output: (H-8)/4+1
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: (H'-4)/2+1
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # Output: (H''-3)/1+1
            activation(),
            nn.Flatten(), # Flatten all dimensions except batch
        )

        # Calculate the size of the flattened features after conv layers
        # by doing a dummy forward pass.
        with torch.no_grad():
            # Create a dummy input on CPU to avoid potential CUDA init issues here
            dummy_input = torch.zeros(1, *obs_shape, device='cpu')
            conv_out_size = self.conv_layers(dummy_input).shape[1]

        # Fully connected layer to get the desired number of output features
        self.fc_layer = nn.Sequential(
            nn.Linear(conv_out_size, output_features),
            activation() # Activation after the FC layer
        )
        self.output_features = output_features

    def forward(self, obs: Tensor) -> Tensor:
        """
        Extracts features from image observations.

        Args:
            obs: Batch of observations, expected shape (B, C, H, W) or (B, H, W, C).
                 Input is normalized to [0, 1] if max value > 1.1.
                 Permuted to (B, C, H, W) if necessary.

        Returns:
            A tensor of features of shape (B, output_features).
        """
        if obs.dtype != torch.float32:
            obs = obs.float()
        
        # Normalize pixel values if they seem to be in [0, 255] range
        if obs.max() > 1.1: # Small tolerance for already normalized data
            obs = obs / 255.0

        # Ensure obs is 4D (Batch, Channels, Height, Width)
        if obs.ndim == 3: # (C, H, W) -> (1, C, H, W)
            obs = obs.unsqueeze(0)
        
        # Permute if channels are last: (B, H, W, C) -> (B, C, H, W)
        if obs.shape[1] != self.conv_layers[0].in_channels and obs.shape[-1] == self.conv_layers[0].in_channels:
            obs = obs.permute(0, 3, 1, 2)
        elif obs.shape[1] != self.conv_layers[0].in_channels:
             raise ValueError(f"Input observation channel dimension ({obs.shape[1]}) "
                              f"does not match CNN input channels ({self.conv_layers[0].in_channels}). "
                              f"Obs shape: {obs.shape}")


        features = self.conv_layers(obs)
        features = self.fc_layer(features)
        return features