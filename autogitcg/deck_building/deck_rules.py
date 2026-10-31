"""Deck-building legality: which action cards a lineup may include (from GCGDeckCardExcelConfigData)."""

from collections import Counter
from functools import cache

from autogitcg.data.dimbreath_tables import load_table

MAX_COPIES_PER_CARD = 2
MAX_COPIES_PER_SINGLE_COPY_CARD = 1
# Arcane Legends, and Elemental Transfigurations (no tournament deck of 4.2-7.1 holds two copies of one)
SINGLE_COPY_TAGS = ("GCG_TAG_LEGEND", "GCG_TAG_CARD_BLESSING")
DECK_CHARACTERS = 3
DECK_ACTION_CARDS = 30


@cache
def _requirements() -> dict[int, tuple[int | None, Counter]]:
    """Card id -> (required character for talents or None, required lineup tags with counts)."""
    requirements = {}
    for row in load_table("GCGDeckCardExcelConfigData"):
        tags = Counter(tag for tag in row.get("relatedCharacterTagList", []) if tag != "GCG_TAG_NONE")
        requirements[row["id"]] = (row.get("relatedCharacterId") or None, tags)
    return requirements


@cache
def _single_copy_cards() -> frozenset[int]:
    return frozenset(row["id"] for row in load_table("GCGCardExcelConfigData")
                     if any(tag in row.get("tagList", []) for tag in SINGLE_COPY_TAGS))


def max_copies(card: int) -> int:
    return MAX_COPIES_PER_SINGLE_COPY_CARD if card in _single_copy_cards() else MAX_COPIES_PER_CARD


@cache
def _character_tags() -> dict[int, set[str]]:
    return {row["id"]: set(row.get("tagList", [])) for row in load_table("GCGCharExcelConfigData")}


def card_allowed(card: int, lineup: list[int]) -> bool:
    """Talents need their character in the lineup; tag requirements need that many lineup characters with the tag
    (e.g. an Elemental Resonance listing CRYO twice needs 2 Cryo characters)."""
    required_character, required_tags = _requirements().get(card, (None, Counter()))
    if required_character is not None and required_character not in lineup:
        return False
    character_tags = _character_tags()
    return all(sum(tag in character_tags.get(character, set()) for character in lineup) >= count
               for tag, count in required_tags.items())


def required_character(card: int) -> int | None:
    """The character a talent card belongs to, or None for every other card."""
    return _requirements().get(card, (None, Counter()))[0]


def has_lineup_requirement(card: int) -> bool:
    """Whether the card is only legal with certain lineups (talents, resonances, nation or weapon-type cards)."""
    required, tags = _requirements().get(card, (None, Counter()))
    return required is not None or bool(tags)
