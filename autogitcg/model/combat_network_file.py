"""Loads a combat network from a checkpoint or frozen model file."""

from pathlib import Path

import torch

from autogitcg.model.combat_network import CombatNetwork
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.training.checkpoints import load_checkpoint, network_config_from


# Outputs added after some networks were trained: a network saved before them loads with these untrained. Only the
# separate win-chance output so far (alpha 1 and early alpha 2); it doesn't affect play.
LATER_OUTPUTS = ("win_head.",)


def load_combat_network(path: Path, encoder: ObservationEncoder, device: torch.device) -> CombatNetwork:
    checkpoint = load_checkpoint(path)
    network = CombatNetwork(network_config_from(checkpoint), encoder.catalog_features.table).to(device)
    missing, unexpected = network.load_state_dict(checkpoint["network"], strict=False)
    if unexpected or any(not name.startswith(LATER_OUTPUTS) for name in missing):
        raise RuntimeError(f"{path}: weights don't fit the network (missing {missing}, unexpected {unexpected})")
    return network
