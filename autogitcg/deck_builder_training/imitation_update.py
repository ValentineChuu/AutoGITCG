"""Imitation loss over teacher-forced steps: -log(probability of picking any still-missing pick of the example).

Returns the mean loss and how often the single most likely pick belongs to the example (top-1 accuracy).
"""

import numpy as np
import torch

from autogitcg.deck_builder.builder_network import BuilderNetwork
from autogitcg.deck_builder.building_batch import pad_and_stack
from autogitcg.deck_builder.token_budget_batches import observation_length, token_budget_batches
from autogitcg.model.observation_batch import to_device


def imitation_pass(network: BuilderNetwork, observations: list[dict[str, np.ndarray]], token_budget: int, device: torch.device,
                   optimizer: torch.optim.Optimizer | None = None, rng: np.random.Generator | None = None,
                   loss_weight: float = 1.0) -> dict[str, float]:
    """Trains when an optimizer is given; otherwise only measures. loss_weight scales the gradient step (an imitation
    anchor inside reinforcement learning uses less than 1)."""
    network.train(optimizer is not None)
    lengths = [observation_length(observation) for observation in observations]
    total_loss = total_correct = 0.0
    for rows in token_budget_batches(lengths, token_budget, rng):
        batch = to_device(pad_and_stack([observations[row] for row in rows]), device)
        targets = batch.pop("candidate_target") > 0
        with torch.set_grad_enabled(optimizer is not None), \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = network(batch).candidate_logits
        loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~targets, float("-inf")), dim=1)).mean()
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            (loss * loss_weight).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(rows)
        total_correct += targets.gather(1, logits.argmax(dim=1, keepdim=True)).float().sum().item()
    return {"loss": total_loss / len(observations), "top1_in_deck": total_correct / len(observations)}
