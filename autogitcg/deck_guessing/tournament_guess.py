"""Guesses from real tournament decks: the decks people brought with the most similar lineups, weighted by how
well they fit the cards the opponent has shown. Tournament players copy the meta, so for a known lineup the
decks others played with it are the most human-like guess.

Weight of a deck: exp(LINEUP_WEIGHT x shared characters + REVEALED_WEIGHT x share of revealed cards it holds),
over the TOP_DECKS best. Pass `excluded` to leave the opponent's own list out (offline tests), and never guess test
matchups from the locked test lineups' decks in game tests.
"""

import math
from collections import Counter

from autogitcg.data.deck import Deck

LINEUP_WEIGHT = 4.0  # tuned on training-split decks: 61% -> 66% of hidden cards covered
REVEALED_WEIGHT = 24.0
TOP_DECKS = 10


class TournamentGuesser:
    def __init__(self, decks: list[Deck]):
        self.decks = [(frozenset(deck.characters), Counter(deck.cards), tuple(sorted(deck.cards))) for deck in decks]

    def guesses(self, lineup: tuple[int, ...], revealed: Counter, excluded: tuple[int, ...] | None = None) -> list[tuple[float, Counter]]:
        """(weight, card counts) of the best-fitting decks; excluded: a card list (sorted) to leave out."""
        characters = frozenset(lineup)
        revealed_total = sum(revealed.values())
        scored = []
        for deck_characters, cards, card_list in self.decks:
            if card_list == excluded:
                continue
            shared = len(characters & deck_characters)
            if shared == 0:
                continue
            held = sum(min(cards[card], count) for card, count in revealed.items()) / revealed_total if revealed_total else 1.0
            scored.append((LINEUP_WEIGHT * shared + REVEALED_WEIGHT * held, cards))
        scored.sort(key=lambda entry: -entry[0])
        best = scored[:TOP_DECKS]
        if not best:
            return []
        top = best[0][0]
        return [(math.exp(score - top), cards) for score, cards in best]
