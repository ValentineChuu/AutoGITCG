"""The deck builder network: a transformer over the building tokens; scores every legal pick and estimates the
deck's eventual reward.

Built like the combat network: token = projected features + group embedding + definition vector (learned
embedding + projected catalog facts). With the same width, the definition embedding and catalog projection
can be copied from a combat checkpoint (copy_card_knowledge), so the builder starts out knowing the cards.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from autogitcg.deck_builder.building_encoder import FEATURE_WIDTHS, GLOBAL_WIDTH, TOKEN_GROUPS
from autogitcg.model.grown_catalog import checked_catalog_width, pad_catalog_projection, padded_to

IDENTITY_EMBEDDING_STD = 0.02
COPIED_FROM_COMBAT = ("definition_embedding.weight", "catalog_projection.weight", "catalog_projection.bias")


@dataclass
class BuilderConfig:
    vocabulary_size: int
    catalog_width: int
    width: int = 128
    layers: int = 3
    attention_heads: int = 4


@dataclass
class BuilderOutput:
    candidate_logits: torch.Tensor  # (batch, candidates); -inf on padding
    value: torch.Tensor  # (batch,): expected game reward of the finished deck


def _score_head(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))


class BuilderNetwork(nn.Module):
    def __init__(self, config: BuilderConfig, catalog_table: np.ndarray):
        super().__init__()
        config = replace(config, catalog_width=checked_catalog_width(config.catalog_width, catalog_table))
        self.config = config
        width = config.width
        self.register_buffer("catalog_table", torch.as_tensor(catalog_table, dtype=torch.float32), persistent=False)
        self.definition_embedding = nn.Embedding(config.vocabulary_size, width, padding_idx=0)
        nn.init.normal_(self.definition_embedding.weight, std=IDENTITY_EMBEDDING_STD)
        with torch.no_grad():
            self.definition_embedding.weight[0].zero_()
        self.catalog_projection = nn.Linear(config.catalog_width, width)
        self.global_projection = nn.Linear(GLOBAL_WIDTH, width)
        self.feature_projections = nn.ModuleDict({group: nn.Linear(FEATURE_WIDTHS[group], width) for group in TOKEN_GROUPS})
        self.group_embedding = nn.Embedding(len(TOKEN_GROUPS) + 1, width)  # +1 for the global token
        layer = nn.TransformerEncoderLayer(width, config.attention_heads, 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(width)
        self.candidate_head = _score_head(width)
        self.value_head = _score_head(width)
        self._register_load_state_dict_pre_hook(pad_catalog_projection, with_module=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> BuilderOutput:
        definition_vectors = self.definition_embedding.weight + self.catalog_projection(self.catalog_table)
        tokens = [(self.global_projection(batch["global_features"]) + self.group_embedding.weight[0]).unsqueeze(1)]
        masks = [torch.ones_like(batch["global_features"][:, :1], dtype=torch.bool)]
        for group_number, group in enumerate(TOKEN_GROUPS, start=1):
            tokens.append(self.feature_projections[group](batch[f"{group}_features"]) + self.group_embedding.weight[group_number]
                          + functional.embedding(batch[f"{group}_indices"], definition_vectors, padding_idx=0))
            masks.append(batch[f"{group}_mask"] > 0)
        real = torch.cat(masks, dim=1)
        encoded = self.final_norm(self.transformer(torch.cat(tokens, dim=1), src_key_padding_mask=~real))
        candidates = encoded[:, -tokens[-1].shape[1]:]
        logits = self.candidate_head(candidates).squeeze(-1).float()
        return BuilderOutput(
            candidate_logits=logits.masked_fill(~masks[-1], float("-inf")),
            value=torch.tanh(self.value_head(encoded[:, 0]).squeeze(-1)).float(),
        )


def copy_card_knowledge(builder: BuilderNetwork, combat_state: dict[str, torch.Tensor]) -> None:
    """Start from what a trained combat network learned about each card, character and skill."""
    own_state = builder.state_dict()
    for name in COPIED_FROM_COMBAT:
        if name == "catalog_projection.weight":  # a combat network saved before catalog facts were appended
            combat_state = {**combat_state, name: padded_to(combat_state[name], own_state[name].shape[1])}
        if combat_state[name].shape != own_state[name].shape:
            raise ValueError(f"{name}: combat {tuple(combat_state[name].shape)} vs builder {tuple(own_state[name].shape)}")
        own_state[name] = combat_state[name].clone()
    builder.load_state_dict(own_state)
