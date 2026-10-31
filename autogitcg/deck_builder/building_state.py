"""A deck under construction: the picks so far and the legal next picks.

First 3 characters, then 30 action-card copies, one per step. Only legal picks are ever offered, so every
finished deck passes deck_validation.deck_problems.
"""

from collections import Counter
from dataclasses import dataclass, field

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.building_task import BuildingTask
from autogitcg.deck_builder.lineup_feasibility import feasible_next_characters, legal_copies
from autogitcg.deck_builder.usable_weapons import without_unusable_weapons
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, DECK_CHARACTERS

CHARACTER = "character"
CARD = "card"
BUILDING_STEPS = DECK_CHARACTERS + DECK_ACTION_CARDS


@dataclass
class BuildingState:
    task: BuildingTask
    characters: list[int] = field(default_factory=list)
    cards: Counter = field(default_factory=Counter)
    _card_limits: dict[int, int] | None = None  # legal copies for the finished lineup

    @property
    def picking_characters(self) -> bool:
        return len(self.characters) < DECK_CHARACTERS

    @property
    def finished(self) -> bool:
        return not self.picking_characters and sum(self.cards.values()) == DECK_ACTION_CARDS

    def card_limits(self) -> dict[int, int]:
        if self._card_limits is None:
            self._card_limits = legal_copies(self.characters, self.task.collection.action_cards)
            if self.task.skip_unusable_weapons:
                self._card_limits = without_unusable_weapons(self._card_limits, self.characters)
        return self._card_limits

    def candidates(self) -> list[tuple[str, int]]:
        """Legal next picks as (CHARACTER or CARD, id), in a stable order."""
        collection = self.task.collection
        if self.picking_characters:
            characters = feasible_next_characters(self.characters, collection.characters, collection.action_cards)
            return [(CHARACTER, character) for character in sorted(characters)]
        return [(CARD, card) for card, limit in sorted(self.card_limits().items()) if self.cards[card] < limit]

    def apply(self, candidate: tuple[str, int]) -> None:
        kind, identifier = candidate
        if kind == CHARACTER:
            self.characters.append(identifier)
        else:
            self.cards[identifier] += 1

    def deck(self) -> Deck:
        return Deck(characters=list(self.characters), cards=sorted(self.cards.elements()))
