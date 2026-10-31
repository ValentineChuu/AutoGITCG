"""What the board looks like a few of the same player's decisions later: the targets of the network's future
prediction (docs/training.md). Read from the stored observations' character features, so they share
the deciding player's orientation (own characters first)."""

import numpy as np

from autogitcg.observation.encode_board import CHARACTER_COUNT, CHARACTERS_PER_SIDE
from autogitcg.observation.enums import ELEMENTS

FUTURE_HORIZONS = (3, 10)  # own decisions ahead
# Columns of encode_board._character_features: is-mine, slot one-hot, then active, defeated, health/10,
# max health/10, energy/3, max energy/3, energy full, special energy, then the aura elements.
_ACTIVE = 1 + CHARACTERS_PER_SIDE
_COLUMNS = [_ACTIVE, _ACTIVE + 1, _ACTIVE + 2, _ACTIVE + 4] + [_ACTIVE + 8 + element for element in range(len(ELEMENTS))]
FEATURES_PER_CHARACTER = len(_COLUMNS)
FUTURE_WIDTH = len(FUTURE_HORIZONS) * CHARACTER_COUNT * FEATURES_PER_CHARACTER


def board_summary(observation: dict[str, np.ndarray]) -> np.ndarray:
    """(CHARACTER_COUNT * FEATURES_PER_CHARACTER,): active, defeated, HP, energy and aura of every character slot."""
    features = observation["character_features"][:, _COLUMNS].astype(np.float32)
    return (features * observation["character_mask"][:, None]).reshape(-1)


def future_targets(observations: list[dict[str, np.ndarray]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One player's observations in game order -> per step (targets (FUTURE_WIDTH,), mask (len(FUTURE_HORIZONS),)).
    A horizon past the player's last decision is masked."""
    summaries = [board_summary(observation) for observation in observations]
    empty = np.zeros_like(summaries[0]) if summaries else None
    targets, masks = [], []
    for index in range(len(observations)):
        ahead = [index + horizon for horizon in FUTURE_HORIZONS]
        targets.append(np.concatenate([summaries[step] if step < len(summaries) else empty for step in ahead]).astype(np.float16))
        masks.append(np.array([step < len(summaries) for step in ahead], dtype=np.float32))
    return targets, masks
