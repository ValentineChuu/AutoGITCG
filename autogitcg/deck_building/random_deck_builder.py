"""A random legal deck from a collection: the placeholder until the learned deck builder exists."""

import random
from collections import Counter

from autogitcg.data.deck import Deck
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, DECK_CHARACTERS, card_allowed, max_copies


LINEUP_ATTEMPTS = 50


def random_deck(characters: list[int], action_cards: Counter, rng: random.Random) -> Deck:
    """Tries random lineups until one leaves at least 30 legal action cards in the collection."""
    for _ in range(LINEUP_ATTEMPTS):
        lineup = rng.sample(characters, DECK_CHARACTERS)
        pool = [
            card
            for card, owned in action_cards.items() if card_allowed(card, lineup)
            for _ in range(min(owned, max_copies(card)))
        ]
        if len(pool) >= DECK_ACTION_CARDS:
            return Deck(characters=lineup, cards=sorted(rng.sample(pool, DECK_ACTION_CARDS)))
    raise ValueError("no lineup leaves 30 legal action cards in this collection")
