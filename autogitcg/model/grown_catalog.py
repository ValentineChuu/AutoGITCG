"""Catalog facts appended after a network was saved (e.g. entity effects): the network is built for the
current, wider catalog table, and the saved catalog projection gets zero input weights for the new columns, so it
behaves exactly as when it was saved. Shared by the combat and deck-builder networks."""

import numpy as np
import torch


def checked_catalog_width(saved_width: int, catalog_table: np.ndarray) -> int:
    """The width to build the catalog projection with: the current table's (it may only have grown)."""
    if catalog_table.shape[1] < saved_width:
        raise ValueError(f"catalog table has {catalog_table.shape[1]} columns, the network was saved with {saved_width}")
    return catalog_table.shape[1]


def padded_to(saved: torch.Tensor, in_features: int) -> torch.Tensor:
    """A projection weight (out, in) widened with zero columns to `in_features`."""
    if saved.shape[1] >= in_features:
        return saved
    return torch.cat([saved, saved.new_zeros(saved.shape[0], in_features - saved.shape[1])], dim=1)


def pad_catalog_projection(module, state_dict: dict, prefix: str, *arguments) -> None:
    """load_state_dict pre-hook for a module with `catalog_projection`."""
    name = f"{prefix}catalog_projection.weight"
    if name in state_dict:
        state_dict[name] = padded_to(state_dict[name], module.catalog_projection.in_features)
