"""Deck share codes decode to the decks the tournament tables name."""

from autogitcg.analysis.definition_names import definition_names
from autogitcg.data.share_code import decode_share_code
from autogitcg.deck_building.deck_validation import deck_problems

# from the 7.1 weekly tournament table: lineup "Goldflame Qucusaur Tyrant\Lyney\Yae Miko"
CODE = "IyHR9SIQEkGS95QQGmIylp4aDVIx3NweKLJB+uQgD5IxDOUSDzJBP/IUEOIR1fMeEEEB"


def test_tournament_code_decodes_to_its_lineup():
    deck = decode_share_code(CODE)
    names = {identifier: name for identifier, kind, name in definition_names() if kind == "character"}
    assert sorted(names[character] for character in deck.characters) == ["Goldflame Qucusaur Tyrant", "Lyney", "Yae Miko"]
    assert len(deck.cards) == 30 and not deck_problems(deck)
