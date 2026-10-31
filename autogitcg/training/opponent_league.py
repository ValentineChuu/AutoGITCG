"""Who sits across from the learner: itself, a past snapshot, the frozen reference, the heuristic bot or random play.

Self-play games train on both seats. Snapshot games keep the learner from forgetting how to beat older
strategies; the scripted opponents are a small safety net (alpha 2 beat them ~95-100%, so they teach little).
Reference games (frozen alpha 2.40) come in mirror pairs and measure skill per matchup category
(training/adaptive_difficulty.py); each sample of REFERENCE starts two games.
"""

import random

import torch

SELF = "self"
SNAPSHOT = "snapshot"
HEURISTIC = "heuristic"
RANDOM = "random"
REFERENCE = "reference"
# alpha 1 and 2 used {self 0.55, snapshot 0.2, heuristic 0.2, random 0.05}; alpha 3 {self 0.6, snapshot 0.33, heuristic 0.05, random 0.02}.
OPPONENT_WEIGHTS = {SELF: 0.55, SNAPSHOT: 0.3, REFERENCE: 0.08, HEURISTIC: 0.05, RANDOM: 0.02}
MAX_SNAPSHOTS = 20


class OpponentLeague:
    def __init__(self, rng: random.Random, weights: dict[str, float] | None = None, has_reference: bool = False):
        self.rng = rng
        self.weights = dict(weights or OPPONENT_WEIGHTS)
        if not has_reference:
            self.weights[SELF] = self.weights.get(SELF, 0.0) + self.weights.pop(REFERENCE, 0.0)
        self.snapshots: list[tuple[int, dict[str, torch.Tensor]]] = []  # (update, CPU state dict), oldest first

    def add_snapshot(self, network: torch.nn.Module, update: int) -> None:
        self.snapshots.append((update, {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}))
        del self.snapshots[:-MAX_SNAPSHOTS]

    def random_snapshot(self) -> tuple[int, dict[str, torch.Tensor]] | None:
        return self.rng.choice(self.snapshots) if self.snapshots else None

    def sample_opponent(self) -> str:
        weights = dict(self.weights)
        if not self.snapshots:
            weights[SELF] = weights.get(SELF, 0.0) + weights.pop(SNAPSHOT, 0.0)
        return self.rng.choices(list(weights), weights=list(weights.values()))[0]
