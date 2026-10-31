"""Stacking encoded observations into batches, compact storage for rollouts, and moving batches to the GPU.

Stored compactly: vocabulary indices and target positions as int16, features as float16, masks as uint8.
That roughly halves rollout memory; `to_device` restores the network's dtypes.
"""

import numpy as np
import torch


def compact(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    compacted = {}
    for name, array in arrays.items():
        if name.endswith("_indices") or name.endswith("_targets"):
            compacted[name] = array.astype(np.int16)
        elif name.endswith("_mask"):
            compacted[name] = array.astype(np.uint8)
        else:
            compacted[name] = array.astype(np.float16)
    return compacted


def stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([observation[name] for observation in observations]) for name in observations[0]}


def to_device(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return network_dtypes(upload(arrays, device))


def upload(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Copy to the device as stored (compact dtypes stay compact)."""
    return {name: torch.from_numpy(np.ascontiguousarray(array)).to(device, non_blocking=True) for name, array in arrays.items()}


def network_dtypes(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Indices and targets as int64, everything else as float32."""
    return {name: tensor.long() if name.endswith("_indices") or name.endswith("_targets") else tensor.float()
            for name, tensor in tensors.items()}
