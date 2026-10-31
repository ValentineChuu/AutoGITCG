"""Guessing a blind opponent's hidden cards (autogitcg/deck_guessing/)."""

from collections import Counter

from autogitcg.data.deck import Deck
from autogitcg.deck_guessing.guess_quality import overlap_share
from autogitcg.deck_guessing.remaining_cards import expected_remaining
from autogitcg.deck_guessing.tournament_deck_guesses import opponent_guesses
from autogitcg.deck_guessing.tournament_guess import TournamentGuesser

LINEUP = (1, 2, 3)
HOLDS_REVEALED = Deck(characters=list(LINEUP), cards=[10] * 2 + [20] * 28)
OTHER = Deck(characters=list(LINEUP), cards=[30] * 30)
UNRELATED = Deck(characters=[7, 8, 9], cards=[40] * 30)


def test_decks_holding_the_revealed_cards_weigh_more():
    guesses = TournamentGuesser([HOLDS_REVEALED, OTHER, UNRELATED]).guesses(LINEUP, Counter({10: 1}))
    weights = {tuple(sorted(cards)): weight for weight, cards in guesses}
    assert tuple(sorted(Counter(UNRELATED.cards))) not in weights  # no shared character
    assert weights[tuple(sorted(Counter(HOLDS_REVEALED.cards)))] > weights[tuple(sorted(Counter(OTHER.cards)))]


def test_the_opponents_own_list_can_be_left_out():
    guesses = TournamentGuesser([HOLDS_REVEALED, OTHER]).guesses(LINEUP, Counter(), excluded=tuple(sorted(OTHER.cards)))
    assert [cards for _, cards in guesses] == [Counter(HOLDS_REVEALED.cards)]


def test_expected_remaining_removes_revealed_cards_and_keeps_the_hidden_count():
    expected = expected_remaining([(1.0, Counter(HOLDS_REVEALED.cards))], Counter({10: 2}), 28)
    assert abs(sum(expected.values()) - 28) < 1e-9 and 10 not in expected
    assert overlap_share(expected, Counter({20: 28})) == 1.0


def test_chudledee_style_guesses_are_looked_up_by_lineup():
    assert opponent_guesses({LINEUP: [Counter({5: 30})]}, LINEUP, Counter()) == [Counter({5: 30})]
    assert opponent_guesses(None, LINEUP, Counter()) is None
