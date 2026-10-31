"""The combat network: every token group goes through one transformer; heads score choices and estimate the value.

Token = live features (projected) + token-group embedding + for identified things the definition vector,
which is a learned embedding plus the projected catalog facts of that definition. Action tokens also get
their targets' board tokens added, so "attack this character" and "attack that one" look different.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from autogitcg.model.grown_catalog import checked_catalog_width, pad_catalog_projection
from autogitcg.model.network_config import NetworkConfig
from autogitcg.model.trim_padding import trim_padding
from autogitcg.observation.encode_board import CHARACTER_COUNT
from autogitcg.observation.encode_hand import MAX_HAND_PER_SIDE
from autogitcg.observation.encode_actions import PREVIEW_START
from autogitcg.training.future_targets import FUTURE_WIDTH

# Sequence order; groups with "_indices" arrays carry a definition identity.
TOKEN_GROUPS = ["character", "entity", "hand", "dice", "knowledge", "history", "action"]
IDENTIFIED_GROUPS = {"character", "entity", "hand", "knowledge", "history", "action"}
IDENTITY_EMBEDDING_STD = 0.02


@dataclass
class NetworkOutput:
    action_logits: torch.Tensor  # (batch, MAX_ACTIONS): action / chooseActive / selectCard
    dice_logits: torch.Tensor  # (batch, MAX_DICE): reroll this die
    hand_logits: torch.Tensor  # (batch, MAX_HAND_PER_SIDE): swap this opening-hand card
    value: torch.Tensor  # (batch,): expected (speed-adjusted) reward in [-1, 1] for the deciding player
    win_logit: torch.Tensor  # (batch,): log-odds that the deciding player wins (plain win chance, for calibration/concede)
    move_values: torch.Tensor | None = None  # (batch, MAX_ACTIONS): expected final reward after playing each action token
    futures: torch.Tensor | None = None  # (batch, MAX_ACTIONS, FUTURE_WIDTH): how the board changes a few own decisions after each action
    reply_interest: torch.Tensor | None = None  # (batch, MAX_ACTIONS): how much checking each action's replies would change its probability


def _score_head(width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))


class CombatNetwork(nn.Module):
    def __init__(self, config: NetworkConfig, catalog_table: np.ndarray):
        super().__init__()
        # Catalog facts appended since the network was saved get zero input weights (model/grown_catalog.py).
        config = replace(config, catalog_width=checked_catalog_width(config.catalog_width, catalog_table))
        self.config = config
        width = config.width
        self.register_buffer("catalog_table", torch.as_tensor(catalog_table, dtype=torch.float32), persistent=False)
        self.definition_embedding = nn.Embedding(config.vocabulary_size, width, padding_idx=0)
        # Small start: the default N(0, 1) (norm ~11) drowned what training learned (~2.4 after 600 updates in alpha 1).
        nn.init.normal_(self.definition_embedding.weight, std=IDENTITY_EMBEDDING_STD)
        with torch.no_grad():
            self.definition_embedding.weight[0].zero_()
        self.catalog_projection = nn.Linear(config.catalog_width, width)
        self.feature_projections = nn.ModuleDict({
            group: nn.Linear(feature_width, width) for group, feature_width in config.feature_widths.items()
        })
        self.group_embedding = nn.Embedding(len(TOKEN_GROUPS) + 1, width)  # +1 for the global token
        self.target_projection = nn.Linear(width, width)
        layer = nn.TransformerEncoderLayer(width, config.attention_heads, 4 * width, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(width)
        self.action_head = _score_head(width)
        self.dice_head = _score_head(width)
        self.hand_head = _score_head(width)
        self.value_head = _score_head(width)
        self.win_head = _score_head(width)
        self.move_value_head = _score_head(width)  # Monte-Carlo move values (ppo_update: move_value_coefficient)
        # How the board changes a few own decisions after each move (future_targets.py). The last layer starts at zero,
        # so the prediction starts at "nothing changes" (predicting whole boards, main eta 2, never beat that).
        self.future_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, FUTURE_WIDTH))
        nn.init.zeros_(self.future_head[2].weight)
        nn.init.zeros_(self.future_head[2].bias)
        # Learned reply selector (docs/architecture.md): scored on the observation without replies.
        self.reply_interest_head = _score_head(width)
        self._register_load_state_dict_pre_hook(_pad_grown_feature_projections, with_module=True)
        self._register_load_state_dict_pre_hook(pad_catalog_projection, with_module=True)
        self._register_load_state_dict_pre_hook(_keep_new_heads, with_module=True)

    def forward(self, batch: dict[str, torch.Tensor]) -> NetworkOutput:
        action_width = batch["action_mask"].shape[1]
        batch = trim_padding(batch)
        definition_vectors = self.definition_embedding.weight + self.catalog_projection(self.catalog_table)
        tokens = [self._project("global", batch["global_features"]).unsqueeze(1) + self.group_embedding.weight[0]]
        masks = [torch.ones_like(batch["global_features"][:, :1], dtype=torch.bool)]
        for group_number, group in enumerate(TOKEN_GROUPS, start=1):
            group_tokens = self._project(group, batch[f"{group}_features"]) + self.group_embedding.weight[group_number]
            if group in IDENTIFIED_GROUPS:
                group_tokens = group_tokens + functional.embedding(batch[f"{group}_indices"].long(), definition_vectors, padding_idx=0)
            if group == "action":
                group_tokens = group_tokens + self._target_vectors(batch["action_targets"].long(), tokens[1], tokens[2])
            tokens.append(group_tokens)
            masks.append(batch[f"{group}_mask"] > 0)

        sequence = torch.cat(tokens, dim=1)
        real = torch.cat(masks, dim=1)
        encoded = self.final_norm(self.transformer(sequence, src_key_padding_mask=~real))

        sizes = [1] + [token_group.shape[1] for token_group in tokens[1:]]
        by_group = dict(zip(["global"] + TOKEN_GROUPS, torch.split(encoded, sizes, dim=1)))
        action_padding = (0, action_width - by_group["action"].shape[1])
        return NetworkOutput(
            action_logits=functional.pad(self.action_head(by_group["action"]).squeeze(-1), action_padding),
            dice_logits=self.dice_head(by_group["dice"]).squeeze(-1),
            hand_logits=self.hand_head(by_group["hand"][:, :MAX_HAND_PER_SIDE]).squeeze(-1),
            value=torch.tanh(self.value_head(by_group["global"][:, 0]).squeeze(-1)),
            win_logit=self.win_head(by_group["global"][:, 0]).squeeze(-1),
            move_values=functional.pad(torch.tanh(self.move_value_head(by_group["action"]).squeeze(-1)), action_padding),
            futures=functional.pad(self.future_head(by_group["action"]), (0, 0) + action_padding),
            reply_interest=functional.pad(self.reply_interest_head(by_group["action"]).squeeze(-1), action_padding),
        )

    def _project(self, group: str, features: torch.Tensor) -> torch.Tensor:
        """The group's feature projection; features stored before a feature was appended get zeros for it."""
        projection = self.feature_projections[group]
        missing = projection.in_features - features.shape[-1]
        features = functional.pad(features, (0, missing)) if missing > 0 else features
        if group == "action" and self.config.preview_scale != 1.0:  # previews and replies are the last action columns
            scale = torch.ones(projection.in_features, device=features.device)
            scale[PREVIEW_START:] = self.config.preview_scale
            features = features * scale
        return projection(features)

    def _target_vectors(self, targets: torch.Tensor, characters: torch.Tensor, entities: torch.Tensor) -> torch.Tensor:
        """Sum of the (pre-transformer) board tokens each action points at; targets are board positions or -1."""
        board = torch.cat([characters, entities], dim=1)  # positions: characters first, then entities
        assert board.shape[1] > CHARACTER_COUNT
        batch_size, action_count, target_count = targets.shape
        positions = targets.clamp(min=0).reshape(batch_size, action_count * target_count, 1).expand(-1, -1, board.shape[-1])
        gathered = torch.gather(board, 1, positions).reshape(batch_size, action_count, target_count, -1)
        valid = (targets >= 0).unsqueeze(-1)
        return self.target_projection((gathered * valid).sum(dim=2))


def _pad_grown_feature_projections(module: "CombatNetwork", state_dict: dict, prefix: str, *arguments) -> None:
    """Saved networks from before a feature was appended to a group: zero input weights for the new columns."""
    for group, projection in module.feature_projections.items():
        name = f"{prefix}feature_projections.{group}.weight"
        saved = state_dict.get(name)
        if saved is not None and saved.shape[1] < projection.in_features:
            padding = saved.new_zeros(saved.shape[0], projection.in_features - saved.shape[1])
            state_dict[name] = torch.cat([saved, padding], dim=1)


def _keep_new_heads(module: "CombatNetwork", state_dict: dict, prefix: str, *arguments) -> None:
    """Saved networks from before the move-value, future or reply-interest head: the head keeps its fresh initialization."""
    for head in ("move_value_head", "future_head", "reply_interest_head"):
        if not any(name.startswith(f"{prefix}{head}.") for name in state_dict):
            for name, tensor in getattr(module, head).state_dict().items():
                state_dict[f"{prefix}{head}.{name}"] = tensor.clone()
