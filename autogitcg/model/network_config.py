"""Sizes of the combat network: its own hyperparameters plus the observation widths it reads."""

from dataclasses import dataclass, field, replace

from autogitcg.observation.encode_actions import ACTION_WIDTH
from autogitcg.observation.encode_board import CHARACTER_WIDTH, ENTITY_WIDTH
from autogitcg.observation.encode_dice import DICE_WIDTH
from autogitcg.observation.encode_global import GLOBAL_WIDTH
from autogitcg.observation.encode_hand import hand_width
from autogitcg.observation.encode_memory import HISTORY_WIDTH, KNOWLEDGE_WIDTH
from autogitcg.observation.observation_encoder import ObservationEncoder


@dataclass
class NetworkConfig:
    vocabulary_size: int
    catalog_width: int
    feature_widths: dict[str, int] = field(default_factory=dict)  # token group -> live feature width
    width: int = 128
    layers: int = 4
    attention_heads: int = 4
    reads_previews: bool = False  # trained with action previews: games it plays need EnginePool(previews=True)
    reads_replies: bool = False  # also reads the opponent's replies to each action: games need EnginePool(replies=True)
    # Action preview and reply inputs are multiplied by this before the projection. Their weights start at zero, and the
    # optimizer moves each weight about one learning-rate step per update, so a scale of k lets the network learn
    # to use previews about k times faster without changing how it plays at the start.
    preview_scale: float = 1.0


STATIC_FEATURE_WIDTHS = {"global": GLOBAL_WIDTH, "character": CHARACTER_WIDTH, "entity": ENTITY_WIDTH, "dice": DICE_WIDTH,
                         "knowledge": KNOWLEDGE_WIDTH, "history": HISTORY_WIDTH, "action": ACTION_WIDTH}


def grown_feature_widths(config: NetworkConfig) -> NetworkConfig:
    """A saved network's config with every group's width raised to the current encoder's: features added since are
    appended at the end of their group, and CombatNetwork gives their input weights zeros, so the network behaves
    exactly as when it was saved (and can learn to use them)."""
    widths = {group: max(width, STATIC_FEATURE_WIDTHS.get(group, width)) for group, width in config.feature_widths.items()}
    return replace(config, feature_widths=widths)


def network_config_for(encoder: ObservationEncoder, **hyperparameters) -> NetworkConfig:
    feature_widths = {
        "global": GLOBAL_WIDTH,
        "character": CHARACTER_WIDTH,
        "entity": ENTITY_WIDTH,
        "hand": hand_width(len(encoder.attachment_ids)),
        "dice": DICE_WIDTH,
        "knowledge": KNOWLEDGE_WIDTH,
        "history": HISTORY_WIDTH,
        "action": ACTION_WIDTH,
    }
    return NetworkConfig(len(encoder.vocabulary), encoder.catalog_features.width, feature_widths, **hyperparameters)
