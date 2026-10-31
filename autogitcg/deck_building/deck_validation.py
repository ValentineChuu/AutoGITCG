"""Checks a full deck against the deck-building rules and against what our engine supports."""

from collections import Counter
from functools import cache

from autogitcg.data.deck import Deck
from autogitcg.data.dimbreath_tables import load_table
from autogitcg.data.piovium_catalog import load_catalog
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, DECK_CHARACTERS, card_allowed, max_copies


@cache
def _buildable_ids() -> tuple[frozenset[int], frozenset[int]]:
    """(characters, action cards) that are obtainable, deck-buildable and implemented in the engine."""
    catalog = load_catalog()
    engine_characters = {character["id"] for character in catalog["characters"]}
    engine_entities = {entity["id"] for entity in catalog["entities"]}
    deck_ids = {row["id"] for row in load_table("GCGDeckCardExcelConfigData")}
    characters = {row["id"] for row in load_table("GCGCharExcelConfigData")
                  if row.get("isCanObtain") and row["id"] in deck_ids and row["id"] in engine_characters}
    cards = {row["id"] for row in load_table("GCGCardExcelConfigData")
             if row.get("isCanObtain") and row["id"] in deck_ids and row["id"] in engine_entities}
    return frozenset(characters), frozenset(cards)


def buildable_characters() -> frozenset[int]:
    """Characters a player can own and put in a deck (and the engine implements)."""
    return _buildable_ids()[0]


def buildable_action_cards() -> frozenset[int]:
    """Action cards a player can own and put in a deck (and the engine implements)."""
    return _buildable_ids()[1]


def deck_problems(deck: Deck) -> list[str]:
    """Every rule the deck breaks, as readable messages; empty when the deck is legal."""
    characters, cards = _buildable_ids()
    problems = []
    if len(deck.characters) != DECK_CHARACTERS or len(set(deck.characters)) != DECK_CHARACTERS:
        problems.append(f"needs {DECK_CHARACTERS} different characters, has {deck.characters}")
    if len(deck.cards) != DECK_ACTION_CARDS:
        problems.append(f"needs {DECK_ACTION_CARDS} action cards, has {len(deck.cards)}")
    problems += [f"unknown or unusable character {character}" for character in deck.characters if character not in characters]
    for card, copies in sorted(Counter(deck.cards).items()):
        if card not in cards:
            problems.append(f"unknown or unusable card {card}")
            continue
        if copies > max_copies(card):
            problems.append(f"card {card}: {copies} copies, limit {max_copies(card)}")
        if not card_allowed(card, list(deck.characters)):
            problems.append(f"card {card}: talent or lineup requirement not met")
    return problems
