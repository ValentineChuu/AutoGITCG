"""Splits observations into batches of similar length with at most `token_budget` padded tokens each.

Character picks show the whole collection (~700 tokens) while card picks show ~400; mixing them would pad
every row to the longest and run out of GPU memory, so rows are grouped by length first.
"""

import numpy as np


def observation_length(observation: dict[str, np.ndarray]) -> int:
    return 1 + sum(len(array) for name, array in observation.items() if name.endswith("_mask"))


def token_budget_batches(lengths: list[int], token_budget: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Row indices per batch. With rng: random order within equal lengths and random batch order (for training)."""
    order = rng.permutation(len(lengths)) if rng is not None else np.arange(len(lengths))
    order = order[np.argsort(np.asarray(lengths)[order], kind="stable")]
    batches, start = [], 0
    while start < len(order):
        end = start + 1
        while end < len(order) and (end + 1 - start) * lengths[order[end]] <= token_budget:
            end += 1
        batches.append(order[start:end])
        start = end
    if rng is not None:
        batches = [batches[index] for index in rng.permutation(len(batches))]
    return batches
