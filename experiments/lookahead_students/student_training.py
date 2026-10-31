"""Supervised training of a student network on recorded look-ahead labels, with a validation pass after each
epoch. Labels stay in memory on the CPU; each minibatch is copied to the device."""

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from experiments.lookahead_students.label_dataset import LabelSet
from experiments.lookahead_students.student_loss import student_losses
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.observation_batch import network_dtypes, upload


@dataclass
class StudentOptions:
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 5e-5
    anchor_weight: float = 1.0
    # Off by default: with a few thousand games the value output overfits (epsilon 1: validation 0.60 -> 0.94),
    # and the student becomes the next searcher, which relies on it.
    value_weight: float = 0.0
    win_weight: float = 0.0
    value_anchor_weight: float = 1.0
    search_win_weight: float = 0.0  # win-chance output toward search's win chance (label rule "evaluator")
    move_value_weight: float = 0.0  # move-value output toward each candidate's search score (label rule "evaluator")
    centered_move_value_weight: float = 0.0  # the same, differences between the candidates only
    search_value_weight: float = 0.0  # value output toward the played move's search score (lower value_anchor_weight with it)
    max_gradient_norm: float = 0.5
    seed: int = 0


def _batch(labels: LabelSet, indices: np.ndarray, start_network: CombatNetwork, device: torch.device):
    """(observations, targets), the targets including the starting network's values (for the value drift)."""
    observations, targets = labels.rows(indices)
    batch, targets = network_dtypes(upload(observations, device)), upload(targets, device)
    with torch.no_grad():
        targets["start_values"] = _forward(start_network, batch, device).value.float()
    return batch, targets


def _forward(network: CombatNetwork, batch: dict[str, torch.Tensor], device: torch.device):
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return network(batch)


def _total_loss(losses: dict[str, torch.Tensor], options: StudentOptions) -> torch.Tensor:
    return (losses["imitation"] + options.anchor_weight * losses["anchor"] + options.value_weight * losses["value"]
            + options.win_weight * losses["win"] + options.value_anchor_weight * losses["value_drift"]
            + options.search_win_weight * losses["search_win"] + options.move_value_weight * losses["move_value"]
            + options.centered_move_value_weight * losses["centered_move_value"] + options.search_value_weight * losses["search_value"])


@torch.no_grad()
def evaluate_student(network: CombatNetwork, start_network: CombatNetwork, labels: LabelSet, options: StudentOptions,
                     device: torch.device) -> dict[str, float]:
    """Label-weighted means of every loss and accuracy over the set."""
    network.eval()
    sums: dict[str, float] = {}
    for start in range(0, len(labels), options.minibatch_size):
        indices = np.arange(start, min(start + options.minibatch_size, len(labels)))
        batch, targets = _batch(labels, indices, start_network, device)
        losses = student_losses(_forward(network, batch, device), batch, targets)
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value) * len(indices)
    return {name: total / len(labels) for name, total in sums.items()}


def train_student(network: CombatNetwork, training: LabelSet, validation: LabelSet, options: StudentOptions,
                  device: torch.device, on_epoch: Callable[[int, dict], None] = lambda epoch, metrics: None) -> None:
    """on_epoch(epoch, metrics) after each epoch (epoch 0 = before training), e.g. to save the network."""
    optimizer = torch.optim.AdamW(network.parameters(), lr=options.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(options.seed)
    start_network = copy.deepcopy(network).eval()
    on_epoch(0, {"validation": evaluate_student(network, start_network, validation, options, device)})
    for epoch in range(1, options.epochs + 1):
        network.train()
        order = rng.permutation(len(training))
        sums, batches = {}, 0
        for start in range(0, len(order) - options.minibatch_size + 1, options.minibatch_size):
            batch, targets = _batch(training, order[start:start + options.minibatch_size], start_network, device)
            losses = student_losses(_forward(network, batch, device), batch, targets)
            optimizer.zero_grad(set_to_none=True)
            _total_loss(losses, options).backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), options.max_gradient_norm)
            optimizer.step()
            for name, value in losses.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            batches += 1
        training_metrics = {name: total / max(batches, 1) for name, total in sums.items()}
        on_epoch(epoch, {"training": training_metrics,
                         "validation": evaluate_student(network, start_network, validation, options, device)})
