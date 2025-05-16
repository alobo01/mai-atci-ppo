from typing import Any, List, Optional, Union
import numpy as np
import torch

NpArray = np.ndarray
Tensor = torch.Tensor
Device = Union[torch.device, str]

def get_device(device_str: Optional[Device] = None) -> torch.device:
    """
    Gets the torch device. Defaults to CUDA if available, otherwise CPU.

    Args:
        device_str: Optional device string ('cpu', 'cuda') or torch.device object.

    Returns:
        The selected torch.device.
    """
    if isinstance(device_str, torch.device):
        return device_str
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    if device_str == "cpu":
        return torch.device("cpu")
    # Auto-detect
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(
    arr: Union[NpArray, List[Any], int, float, Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Converts input (numpy array, list, scalar, or existing tensor)
    to a Torch Tensor on the specified device and dtype.

    Args:
        arr: The input data.
        device: The target torch device.
        dtype: The target torch dtype.

    Returns:
        A torch.Tensor.
    """
    if isinstance(arr, Tensor):
        return arr.to(device=device, dtype=dtype)
    if isinstance(arr, (list, np.ndarray)):
        np_arr = np.asarray(arr) # Handles lists of numbers
        return torch.as_tensor(np_arr, dtype=dtype, device=device)
    if isinstance(arr, (int, float)): # Handle scalar inputs
        return torch.tensor(arr, dtype=dtype, device=device)
    raise TypeError(f"Unsupported type for to_tensor: {type(arr)}")