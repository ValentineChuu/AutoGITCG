"""Tokens for what the player remembers: deck knowledge and recent public actions (from a GameTracker)."""

import numpy as np

from autogitcg.observation.enums import ACTION_KINDS
from autogitcg.observation.game_tracker import GameTracker
from autogitcg.observation.vocabulary import Vocabulary

KNOWLEDGE_CATEGORIES = ["my_remaining_deck", "opponent_deck_list", "opponent_played"]
MAX_KNOWLEDGE = 96
KNOWLEDGE_WIDTH = len(KNOWLEDGE_CATEGORIES) + 1
MAX_HISTORY = 16
HISTORY_WIDTH = 1 + len(ACTION_KINDS) + 3
ROUND_AGE_COLUMN = 1 + len(ACTION_KINDS)
HIDDEN_DEFINITION_COLUMN = ROUND_AGE_COLUMN + 1
ORDER_COLUMN = ROUND_AGE_COLUMN + 2  # position, newest first: the transformer has no positions, so order needs a feature


def encode_knowledge(tracker: GameTracker, vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One token per (category, card definition) with a count. Returns (indices, features, mask)."""
    groups = [tracker.my_remaining_deck, tracker.opponent_deck or {}, tracker.opponent_played_cards]
    rows = [
        (vocabulary.index("entity", definition_id), category, count)
        for category, counts in enumerate(groups)
        for definition_id, count in counts.items() if count > 0
    ]
    indices = np.zeros(MAX_KNOWLEDGE, dtype=np.int64)
    features = np.zeros((MAX_KNOWLEDGE, KNOWLEDGE_WIDTH), dtype=np.float32)
    mask = np.zeros(MAX_KNOWLEDGE, dtype=np.float32)
    for position, (index, category, count) in enumerate(rows[:MAX_KNOWLEDGE]):
        indices[position] = index
        features[position, category] = 1.0
        features[position, -1] = count / 2
        mask[position] = 1.0
    return indices, features, mask


def encode_history(tracker: GameTracker, round_number: int, vocabulary: Vocabulary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The most recent public actions, newest first. Returns (indices, features, mask)."""
    indices = np.zeros(MAX_HISTORY, dtype=np.int64)
    features = np.zeros((MAX_HISTORY, HISTORY_WIDTH), dtype=np.float32)
    mask = np.zeros(MAX_HISTORY, dtype=np.float32)
    for position, event in enumerate(reversed(tracker.events[-MAX_HISTORY:])):
        indices[position] = vocabulary.index(event.vocabulary_kind, event.definition_id) if event.vocabulary_kind else 0
        features[position, 0] = float(event.actor_is_me)
        features[position, 1 + ACTION_KINDS.index(event.action_kind)] = 1.0
        features[position, ROUND_AGE_COLUMN] = (round_number - event.round_number) / 15
        features[position, HIDDEN_DEFINITION_COLUMN] = float(event.vocabulary_kind is not None and event.definition_id is None)
        features[position, ORDER_COLUMN] = position / MAX_HISTORY
        mask[position] = 1.0
    return indices, features, mask
