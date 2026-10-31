"""Hand tokens: my cards (identity, current cost, playability) and the opponent's cards (identity hidden).

The exposed state only carries a card's printed cost, so the current (discounted) cost
and playability are read from the action list. Markers attached to hand cards
(Electric Shock, cost changes, Untunable, ...) are visible on both sides.
"""

import math

import numpy as np

from autogitcg.observation.enums import COST_TYPE_COUNT, VALID
from autogitcg.observation.vocabulary import Vocabulary

MAX_HAND_PER_SIDE = 10
MAX_HAND = 2 * MAX_HAND_PER_SIDE  # mine first, then the opponent's
VALIDITY_COUNT = 7


def cost_vector(requirements: list[dict]) -> list[float]:
    vector = [0.0] * COST_TYPE_COUNT
    for requirement in requirements:
        vector[requirement.get("type", 0)] += requirement.get("count", 0) / 4
    return vector


def hand_width(attachment_count: int) -> int:
    return 1 + COST_TYPE_COUNT + 3 + VALIDITY_COUNT + attachment_count + 1


def _card_actions(actions: list[dict]) -> tuple[dict[int, dict], set[int]]:
    """Best (valid-first) playCard action per hand card id, and card ids that can be tuned now."""
    play_by_card: dict[int, dict] = {}
    tunable: set[int] = set()
    for action in actions:
        kind, value = action["action"]["$case"], action["action"]["value"]
        if kind == "playCard":
            current = play_by_card.get(value["cardId"])
            if current is None or (current.get("validity", VALID) != VALID and action.get("validity", VALID) == VALID):
                play_by_card[value["cardId"]] = action
        elif kind == "elementalTuning" and action.get("validity", VALID) == VALID:
            tunable.add(value["removedCardId"])
    return play_by_card, tunable


def _attachment_features(card: dict, attachment_ids: list[int]) -> list[float]:
    present = [0.0] * len(attachment_ids)
    stacks = 0
    for attachment in card.get("attachment", []):
        if attachment.get("definitionId") in attachment_ids:
            present[attachment_ids.index(attachment["definitionId"])] = 1.0
        stacks += attachment.get("variableValue", 0)
    return present + [math.log1p(stacks)]


def encode_hand(state: dict, who: int, actions: list[dict], vocabulary: Vocabulary,
                attachment_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (indices (MAX_HAND,), features (MAX_HAND, width), mask (MAX_HAND,))."""
    play_by_card, tunable = _card_actions(actions)
    width = hand_width(len(attachment_ids))
    indices = np.zeros(MAX_HAND, dtype=np.int64)
    features = np.zeros((MAX_HAND, width), dtype=np.float32)
    mask = np.zeros(MAX_HAND, dtype=np.float32)
    for side_offset, player_index in enumerate((who, 1 - who)):
        is_mine = side_offset == 0
        for slot, card in enumerate(state["player"][player_index].get("handCard", [])[:MAX_HAND_PER_SIDE]):
            position = side_offset * MAX_HAND_PER_SIDE + slot
            play = play_by_card.get(card["id"]) if is_mine else None
            validity = play.get("validity", VALID) if play else None
            if play:
                current_cost = cost_vector(play.get("requiredCost", []))
            elif is_mine:
                current_cost = cost_vector(card.get("definitionCost", []))
            else:
                current_cost = [0.0] * COST_TYPE_COUNT
            indices[position] = vocabulary.index("entity", card.get("definitionId"))
            features[position] = (
                [float(is_mine)] + current_cost
                + [float(play is not None), float(validity == VALID), float(card["id"] in tunable)]
                + [1.0 if validity == code else 0.0 for code in range(VALIDITY_COUNT)]
                + _attachment_features(card, attachment_ids)
            )
            mask[position] = 1.0
    return indices, features, mask
