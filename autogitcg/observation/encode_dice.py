"""Dice tokens: one per die I hold, so reroll decisions can be made per die."""

import numpy as np

from autogitcg.observation.enums import DICE_TYPE_COUNT, ELEMENTS, OMNI_DICE

MAX_DICE = 16
DICE_WIDTH = DICE_TYPE_COUNT + 2


def character_element_dice(state: dict, who: int, catalog_tags_by_character: dict[int, list[str]]) -> tuple[int | None, set[int]]:
    """Dice type of my active character's element, and of all my alive characters' elements."""
    player = state["player"][who]
    active_element, alive_elements = None, set()
    for character in player.get("character", []):
        tags = catalog_tags_by_character.get(character.get("definitionId"), [])
        element = next((ELEMENTS.index(tag) + 1 for tag in tags if tag in ELEMENTS), None)
        if element is None or character.get("defeated", False):
            continue
        alive_elements.add(element)
        if character["id"] == player.get("activeCharacterId"):
            active_element = element
    return active_element, alive_elements


def encode_dice(dice: list[int], active_element: int | None, alive_elements: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features (MAX_DICE, DICE_WIDTH), mask (MAX_DICE,))."""
    features = np.zeros((MAX_DICE, DICE_WIDTH), dtype=np.float32)
    mask = np.zeros(MAX_DICE, dtype=np.float32)
    for position, die in enumerate(dice[:MAX_DICE]):
        features[position, die] = 1.0
        features[position, DICE_TYPE_COUNT] = float(die == active_element or die == OMNI_DICE)
        features[position, DICE_TYPE_COUNT + 1] = float(die in alive_elements or die == OMNI_DICE)
        mask[position] = 1.0
    return features, mask
