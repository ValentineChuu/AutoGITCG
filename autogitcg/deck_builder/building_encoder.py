"""Turns a deck under construction into network input: one global token plus four token groups.

- picked: characters and cards chosen so far (one token per distinct card, with its copy count);
- opponent: the stage's characters, cards and pre-equipped cards, when the opponent is known (stage mode);
- owned: the collection's action cards, shown only while picking characters (so the lineup can fit the cards);
- candidate: every legal next pick. The network scores these tokens; the chosen one is the pick.

Identities use the combat network's vocabulary, so card embeddings can be copied from a combat checkpoint.
Arrays have natural lengths here; building_batch.pad_and_stack pads a batch.
"""

import math
from collections import Counter

import numpy as np

from autogitcg.deck_builder.building_state import CHARACTER, BuildingState
from autogitcg.deck_builder.building_task import PVP_MODE
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, DECK_CHARACTERS, has_lineup_requirement, max_copies, required_character
from autogitcg.observation.vocabulary import Vocabulary

TOKEN_GROUPS = ["picked", "opponent", "owned", "candidate"]
GLOBAL_WIDTH = 8
FEATURE_WIDTHS = {"picked": 4, "opponent": 5, "owned": 3, "candidate": 8}


class BuildingEncoder:
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

    def encode(self, state: BuildingState, candidates: list[tuple[str, int]]) -> dict[str, np.ndarray]:
        groups = {
            "picked": self._picked(state),
            "opponent": self._opponent(state),
            "owned": self._owned(state) if state.picking_characters else [],
            "candidate": self._candidates(state, candidates),
        }
        arrays = {"global_features": self._global(state)}
        for group, tokens in groups.items():
            arrays[f"{group}_indices"] = np.array([index for index, _ in tokens], dtype=np.int64).reshape(-1)
            arrays[f"{group}_features"] = np.array([features for _, features in tokens], dtype=np.float32).reshape(-1, FEATURE_WIDTHS[group])
            arrays[f"{group}_mask"] = np.ones(len(tokens), dtype=np.float32)
        return arrays

    def _global(self, state: BuildingState) -> np.ndarray:
        collection = state.task.collection
        return np.array([
            float(state.picking_characters),
            len(state.characters) / DECK_CHARACTERS,
            sum(state.cards.values()) / DECK_ACTION_CARDS,
            float(state.task.stage is not None),
            float(state.task.mode == PVP_MODE),
            math.log1p(collection.matches_played) / 8,
            len(collection.characters) / 50,
            sum(collection.action_cards.values()) / 300,
        ], dtype=np.float32)

    def _character(self, character: int) -> int:
        return self.vocabulary.index("character", character)

    def _card(self, card: int) -> int:
        return self.vocabulary.index("entity", card)

    def _picked(self, state: BuildingState) -> list:
        tokens = [(self._character(character), [1.0, 0.0, 0.0, 0.0]) for character in state.characters]
        tokens += [(self._card(card), [0.0, 1.0, copies / 2, _is_talent(card)]) for card, copies in sorted(state.cards.items())]
        return tokens

    def _opponent(self, state: BuildingState) -> list:
        stage = state.task.stage
        if stage is None:
            return []
        tokens = [(self._character(character), [1.0, 0.0, 0.0, 0.0, 0.0]) for character in stage.enemy_deck.characters]
        tokens += [(self._card(card), [0.0, 1.0, 0.0, copies / 2, _is_talent(card)])
                   for card, copies in sorted(Counter(stage.enemy_deck.cards).items())]
        tokens += [(self._card(item.definition_id), [0.0, 0.0, 1.0, 0.0, _is_talent(item.definition_id)])
                   for item in stage.enemy_equipment]
        return tokens

    def _owned(self, state: BuildingState) -> list:
        return [(self._card(card), [min(owned, max_copies(card)) / 2, _is_talent(card), float(has_lineup_requirement(card))])
                for card, owned in sorted(state.task.collection.action_cards.items()) if owned > 0]

    def _candidates(self, state: BuildingState, candidates: list[tuple[str, int]]) -> list:
        tokens = []
        if state.picking_characters:
            talent_copies = Counter()
            for card, owned in state.task.collection.action_cards.items():
                if (owner := required_character(card)) is not None:
                    talent_copies[owner] += min(owned, max_copies(card))
        for kind, identifier in candidates:
            if kind == CHARACTER:
                tokens.append((self._character(identifier), [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, talent_copies[identifier] / 2, 0.0]))
            else:
                owned = state.task.collection.action_cards[identifier]
                picked = state.cards[identifier]
                remaining = state.card_limits()[identifier] - picked
                tokens.append((self._card(identifier), [0.0, 1.0, min(owned, 2) / 2, picked / 2, remaining / 2,
                                                        _is_talent(identifier), 0.0, float(has_lineup_requirement(identifier))]))
        return tokens


def _is_talent(card: int) -> float:
    return float(required_character(card) is not None)
