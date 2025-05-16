import os
import random
import numpy as np
import torch

def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """
    Seeds python, numpy, torch (+ cuda) for reproducibility.

    Args:
        seed: The random seed to use.
        deterministic_torch: If True, sets PyTorch to use deterministic algorithms,
                             which can be slower.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if multi-GPU
        torch.backends.cudnn.deterministic = deterministic_torch
        torch.backends.cudnn.benchmark = not deterministic_torch
        if deterministic_torch:
            # For PyTorch versions that support it (>=1.7)
            if hasattr(torch, 'use_deterministic_algorithms'):
                try:
                    torch.use_deterministic_algorithms(True)
                except RuntimeError as e:
                    print(f"Warning: Could not enforce deterministic algorithms: {e}. Try `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `:16:8`.")
            if hasattr(torch, 'backends') and hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                 torch.backends.cuda.matmul.allow_tf32 = False # Disable TF32 for determinism
            if hasattr(torch, 'backends') and hasattr(torch.backends.cudnn, 'allow_tf32'):
                 torch.backends.cudnn.allow_tf32 = False
            # Set CUBLAS workspace config for deterministic matmuls if needed
            # Removed here for efficiency, but can be set in the environment
            # This might be necessary if torch.use_deterministic_algorithms is not enough
            # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # Or ":16:8"