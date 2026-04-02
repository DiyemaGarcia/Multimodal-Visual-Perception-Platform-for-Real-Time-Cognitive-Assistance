import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed:          Integer seed value.
        deterministic: If True, enforce deterministic CUDA operations
                       (may slightly reduce performance).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass


def get_rng_state() -> dict:
    """Capture the current RNG state of all backends."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    """Restore a previously captured RNG state."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])