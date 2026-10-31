"""Saving and restoring training state: network, optimizer, league snapshots and progress counters."""

from dataclasses import asdict
from pathlib import Path

import torch

from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import NetworkConfig, grown_feature_widths
from autogitcg.training.opponent_league import OpponentLeague


def save_checkpoint(path: Path, network: CombatNetwork, optimizer: torch.optim.Optimizer, league: OpponentLeague,
                    progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "network_config": asdict(network.config),
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "snapshots": league.snapshots,
        "progress": progress,
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def network_config_from(checkpoint: dict) -> NetworkConfig:
    """The saved config, with feature widths grown to the current encoder's (network_config.grown_feature_widths)."""
    return grown_feature_widths(NetworkConfig(**checkpoint["network_config"]))
