"""Creating, saving and loading the deck builder network with its encoder."""

from dataclasses import asdict
from pathlib import Path

import torch

from autogitcg.deck_builder.builder_network import BuilderConfig, BuilderNetwork
from autogitcg.deck_builder.building_encoder import BuildingEncoder
from autogitcg.observation.catalog_features import CatalogFeatures
from autogitcg.observation.vocabulary import build_vocabulary


class BuilderSetup:
    """The vocabulary, catalog facts and encoder shared by every builder network (same as the combat network's)."""

    def __init__(self):
        self.vocabulary = build_vocabulary()
        self.catalog_features = CatalogFeatures(self.vocabulary)
        self.encoder = BuildingEncoder(self.vocabulary)

    def new_network(self, **hyperparameters) -> BuilderNetwork:
        config = BuilderConfig(len(self.vocabulary), self.catalog_features.width, **hyperparameters)
        return BuilderNetwork(config, self.catalog_features.table)

    def network_from(self, checkpoint: dict) -> BuilderNetwork:
        network = BuilderNetwork(BuilderConfig(**checkpoint["builder_config"]), self.catalog_features.table)
        network.load_state_dict(checkpoint["builder"])
        return network


def save_builder_checkpoint(path: Path, network: BuilderNetwork, optimizer: torch.optim.Optimizer | None, progress: dict,
                            extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"builder_config": asdict(network.config), "builder": network.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None, "progress": progress, **(extra or {})}, temporary)
    temporary.replace(path)


def load_builder_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
