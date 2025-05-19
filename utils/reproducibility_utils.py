import os
import random
import numpy as np
import torch

def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    """
    Seeds python, numpy, torch (+ cuda) for reproducibility.
    (No performance-hitting deterministic settings here.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)