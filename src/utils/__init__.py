"""
src.utils
---------
Shared utilities across the project.

Modules:
    config_loader — YAML configuration loading with dot-key access
    logger        — Structured logger with file and console handlers
    seed          — Full reproducibility across Python, NumPy, PyTorch
    cuda_utils    — Device management, mixed precision, parameter counting
"""

from src.utils.config_loader import ConfigLoader, load_config
from src.utils.logger import get_logger, EpochLogger
from src.utils.seed import set_seed, get_rng_state, restore_rng_state
from src.utils.cuda_utils import (
    get_device,
    print_gpu_info,
    move_to_device,
    autocast_context,
    count_parameters,
    clip_gradients,
)

__all__ = [
    "ConfigLoader",
    "load_config",
    "get_logger",
    "EpochLogger",
    "set_seed",
    "get_rng_state",
    "restore_rng_state",
    "get_device",
    "print_gpu_info",
    "move_to_device",
    "autocast_context",
    "count_parameters",
    "clip_gradients",
]