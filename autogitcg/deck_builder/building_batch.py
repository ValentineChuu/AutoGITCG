"""Pads deck-building observations of different lengths into one batch (masks mark the real tokens).

Stored observations keep their natural lengths, which is far smaller than padding every step to the largest
collection. `observation_batch.to_device` then moves the batch to the GPU with the right dtypes.
"""

import numpy as np


def pad_and_stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    batch = {"global_features": np.stack([observation["global_features"] for observation in observations])}
    for name in observations[0]:
        if name == "global_features":
            continue
        arrays = [observation[name] for observation in observations]
        longest = max(1, max(array.shape[0] for array in arrays))  # an empty group still needs one (masked) column
        padded = np.zeros((len(arrays), longest, *arrays[0].shape[1:]), dtype=arrays[0].dtype)
        for row, array in enumerate(arrays):
            padded[row, :array.shape[0]] = array
        batch[name] = padded
    return batch
