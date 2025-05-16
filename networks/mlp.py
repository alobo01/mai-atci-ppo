from typing import Tuple, Type, List
import torch
import torch.nn as nn

Tensor = torch.Tensor

class FeedForwardNN(nn.Module):
    """A simple feed-forward neural network (MLP)."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: Type[nn.Module] = nn.Tanh,
    ) -> None:
        """
        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Tuple of hidden layer sizes.
            activation: Activation function class (e.g., nn.Tanh, nn.ReLU).
        """
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, output_dim)) # Output layer is linear
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        if x.dtype != torch.float32: # Ensure input is float32
            x = x.float()
        return self.network(x)