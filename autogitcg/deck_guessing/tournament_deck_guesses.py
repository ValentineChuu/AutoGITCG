"""Tournament-deck guesses for look-ahead copies (docs/search.md): at each searched decision, decks
drawn by weight from the tournament decks that best fit the opponent's lineup and the cards they have revealed so
far (tournament_guess.py). Covers about 66% of a human opponent's hidden cards; Chudledee's guesses about 10%.

Game tests on the tournament test matchups must pass those decks' lists as `excluded`, so no test deck is guessed
from its own list."""

import random
from collections import Counter

from autogitcg.data.deck import Deck
from autogitcg.data.tournament_decks import downloaded_versions, load_tournament_decks
from autogitcg.deck_guessing.tournament_guess import TournamentGuesser


class TournamentDeckGuesses:
    def __init__(self, decks_per_guess: int, seed: int, excluded: list[Deck] = (), versions: list[str] | None = None):
        excluded_lists = {tuple(sorted(deck.cards)) for deck in excluded}
        decks = [entry.deck for entry in load_tournament_decks(versions or downloaded_versions())]
        self.decks = [deck for deck in decks if tuple(sorted(deck.cards)) not in excluded_lists]  # also for guessed card replies
        self.guesser = TournamentGuesser(self.decks)
        self.decks_per_guess = decks_per_guess
        self.rng = random.Random(seed)

    def guesses_for(self, lineup: tuple[int, ...], revealed: Counter) -> list[Counter] | None:
        """`decks_per_guess` decks drawn by weight (with repeats), or None if no tournament deck shares a character."""
        weighted = self.guesser.guesses(tuple(lineup), revealed)
        if not weighted:
            return None
        weights, decks = zip(*weighted)
        return self.rng.choices(decks, weights=weights, k=self.decks_per_guess)


def opponent_guesses(deck_guesses, lineup: tuple[int, ...], revealed: Counter) -> list[Counter] | None:
    """Guessed opponent decks for a look-ahead copy: tournament guesses use the revealed cards; Chudledee guesses
    (a dict-like keyed by lineup, search/deck_guess.py) are fixed per lineup."""
    if deck_guesses is None:
        return None
    if isinstance(deck_guesses, TournamentDeckGuesses):
        return deck_guesses.guesses_for(lineup, revealed)
    return deck_guesses.get(lineup)
