"""Drops padding columns that no row in the batch uses, so the transformer only runs over real tokens.

Only groups whose positions carry no fixed meaning are trimmed (entities, knowledge, history, actions);
characters, hand (mine vs the opponent's halves) and dice keep their layout. Action logits are padded
back to full width by the network so choices stay comparable with the stored ones.
"""

import torch

TRIMMED_GROUPS = ("entity", "knowledge", "history", "action")


def trim_padding(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # One GPU-to-CPU read for all groups: each read makes the CPU wait for the GPU, which then idles while the
    # CPU queues the next work (with a separate read per group, that wait happened 8 times per forward pass).
    lengths = torch.stack([_used_length(batch[f"{group}_mask"]) for group in TRIMMED_GROUPS]).tolist()
    trimmed = dict(batch)
    for group, length in zip(TRIMMED_GROUPS, lengths):
        for name, tensor in batch.items():
            if name.startswith(f"{group}_"):
                trimmed[name] = tensor[:, :length]
    return trimmed


def _used_length(mask: torch.Tensor) -> torch.Tensor:
    """1 + the last column any row uses (at least 1), as a 0-d tensor on the mask's device."""
    used = (mask > 0).any(dim=0)
    positions = torch.arange(1, used.shape[0] + 1, device=mask.device)
    return (positions * used).max().clamp(min=1)
