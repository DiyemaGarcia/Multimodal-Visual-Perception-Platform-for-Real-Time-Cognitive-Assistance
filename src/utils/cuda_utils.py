import torch
import torch.nn as nn
from typing import Optional, Tuple
from contextlib import contextmanager


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Return the best available device.

    Args:
        prefer_cuda: If True and CUDA is available, returns a CUDA device.

    Returns:
        torch.device instance.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_gpu_info() -> None:
    """Print available CUDA device information."""
    if not torch.cuda.is_available():
        print("No CUDA device available. Running on CPU.")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / (1024 ** 3)
        print(
            f"GPU {i}: {props.name} | "
            f"Compute {props.major}.{props.minor} | "
            f"Memory: {mem_total:.2f} GB"
        )


def move_to_device(
    batch: dict,
    device: torch.device,
) -> dict:
    """
    Move all tensors in a dict-based batch to the target device.

    Args:
        batch:  Dictionary of field_name -> tensor.
        device: Target device.

    Returns:
        Same dictionary with all tensors moved.
    """
    return {
        key: val.to(device) if isinstance(val, torch.Tensor) else val
        for key, val in batch.items()
    }


@contextmanager
def autocast_context(enabled: bool = True, dtype: torch.dtype = torch.float16):
    """
    Context manager for mixed-precision training via torch.cuda.amp.

    Args:
        enabled: Whether to enable autocast.
        dtype:   Data type to use (float16 or bfloat16).
    """
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            yield
    else:
        yield


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters of a model.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clip gradients by global norm and return the norm before clipping.

    Args:
        model:    The neural network.
        max_norm: Maximum gradient norm.

    Returns:
        Gradient norm before clipping.
    """
    return nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()