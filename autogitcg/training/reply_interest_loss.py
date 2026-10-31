"""Training the reply-interest output (policies/reply_selection.py) on the fully checked decisions: it sees each
decision as the selector does, before any replies (reply columns zeroed, every action marked "not checked"), and
learns each action's target, how much its probability changed once its replies were known."""

import torch

from autogitcg.observation.encode_actions import REPLY_START, UNCHECKED_COLUMN


def first_look(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The batch as the selector saw it: no replies or follow-ups, every real action token marked "not checked"."""
    features = batch["action_features"].clone()
    features[..., REPLY_START:] = 0
    features[..., UNCHECKED_COLUMN] = batch["action_mask"].to(features.dtype)
    return {**batch, "action_features": features}


def reply_interest_loss(network, batch: dict[str, torch.Tensor], targets: torch.Tensor, has_target: torch.Tensor,
                        checks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared error over the real action tokens of fully checked rows, and (for reporting) the share of the total
    probability change the `checks` highest-interest actions cover."""
    rows = has_target.nonzero().squeeze(1)
    if len(rows) == 0:
        zero = torch.zeros((), device=targets.device)
        return zero, zero
    looked = first_look({name: tensor[rows] for name, tensor in batch.items()})
    with torch.autocast(targets.device.type, dtype=torch.bfloat16, enabled=targets.device.type == "cuda"):
        interest = network(looked).reply_interest.float()
    mask = looked["action_mask"].float()
    wanted = targets[rows]
    loss = ((interest - wanted).pow(2) * mask).sum() / mask.sum().clamp(min=1)
    with torch.no_grad():
        ranked = (interest.masked_fill(mask == 0, -1e9)).topk(min(checks, interest.shape[1]), dim=1).indices
        coverage = wanted.gather(1, ranked).sum() / wanted.sum().clamp(min=1e-9)
    return loss, coverage
