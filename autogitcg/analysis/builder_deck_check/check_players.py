"""The players that play both seats of a check game: a frozen combat network, the scripted heuristic, or random."""

import random
from pathlib import Path

import torch

from autogitcg.bridge.messages import Decision
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.heuristic_policy import HeuristicPolicy
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.random_policy import random_response
from autogitcg.training.checkpoints import load_checkpoint, network_config_from


class NetworkPlayer:
    needs_encoding = True

    def __init__(self, checkpoint: Path, encoder: ObservationEncoder, weights: Path | None = None):
        """weights: a training weights file (runs/<name>/weights/update_*.pt) to load instead of the checkpoint's."""
        checkpoint_data = load_checkpoint(checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = CombatNetwork(network_config_from(checkpoint_data), encoder.catalog_features.table).to(self.device)
        state = torch.load(weights, map_location="cpu", weights_only=True) if weights else checkpoint_data["network"]
        missing, unexpected = network.load_state_dict(state, strict=False)
        if any(not name.startswith("win_head.") for name in missing) or unexpected:
            raise RuntimeError(f"incompatible weights: missing={missing}, unexpected={unexpected}")
        self.policy = NetworkPolicy(network, self.device)

    def respond(self, pending: list[tuple[Decision, object]]) -> list[dict]:
        return [choice.response for choice in self.policy.choose(pending)]


class ScriptedPlayer:
    needs_encoding = False

    def __init__(self, heuristic: bool, seed: int):
        self.heuristic = HeuristicPolicy() if heuristic else None
        self.rng = random.Random(seed)

    def respond(self, pending: list[tuple[Decision, object]]) -> list[dict]:
        if self.heuristic:
            return [self.heuristic.respond(decision, self.rng) for decision, _ in pending]
        return [random_response(decision, self.rng) for decision, _ in pending]
