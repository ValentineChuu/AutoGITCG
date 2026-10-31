"""Load the final combat network used by the diagnostic."""

from pathlib import Path

import torch

from autogitcg.model.combat_network import CombatNetwork
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.training.checkpoints import load_checkpoint, network_config_from


def load_policy(checkpoint_path: Path, greedy: bool = False) -> tuple[ObservationEncoder, NetworkPolicy]:
    encoder = ObservationEncoder()
    checkpoint = load_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = CombatNetwork(network_config_from(checkpoint), encoder.catalog_features.table).to(device)
    network.load_state_dict(checkpoint["network"])
    return encoder, NetworkPolicy(network, device, greedy=greedy)
