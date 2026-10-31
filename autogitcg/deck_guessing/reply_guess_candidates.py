"""Candidate decks for guessed card replies (engine/reference/bridge/reply_guess.mjs): the tournament decks closest to a
blind opponent's lineup, sent once at game creation with their lineup score. The engine re-weights them by the cards
the opponent plays, as tournament_guess.py does, and deals a guessed hand from one of them. The opponent's own list is
left out, as in the game tests (a person's exact list is unknown; similar published decks are what a player knows)."""

from autogitcg.data.deck import Deck
from autogitcg.deck_guessing.tournament_guess import LINEUP_WEIGHT

CANDIDATES = 30  # more than tournament_guess.TOP_DECKS, so the revealed cards can still reorder them


class ReplyGuessCandidates:
    def __init__(self, decks: list[Deck], count: int = CANDIDATES):
        self.decks = [(frozenset(deck.characters), list(deck.cards), tuple(sorted(deck.cards))) for deck in decks]
        self.count = count

    def __call__(self, opponent: Deck) -> list[list]:
        """[[lineup score, card ids], ...], best first; empty if no deck shares a character."""
        characters, own_list = frozenset(opponent.characters), tuple(sorted(opponent.cards))
        scored = [(LINEUP_WEIGHT * shared, cards) for deck_characters, cards, card_list in self.decks
                  if card_list != own_list and (shared := len(characters & deck_characters))]
        scored.sort(key=lambda entry: -entry[0])
        return [[score, cards] for score, cards in scored[:self.count]]
