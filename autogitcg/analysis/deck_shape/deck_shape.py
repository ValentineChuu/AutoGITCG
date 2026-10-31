"""Average shape of a set of decks, to compare Chudledee's decks with people's: card cost, free cards, cards at two
copies, special card kinds per deck, and how many different lineups."""

from collections import Counter
from functools import cache

from autogitcg.data.deck import Deck
from autogitcg.data.dimbreath_tables import load_table

KIND_TAGS = {"arcane legends": "GCG_TAG_LEGEND", "transfigurations": "GCG_TAG_CARD_BLESSING",
             "resonances": "GCG_TAG_RESONANCE", "talents": "GCG_TAG_TALENT", "weapons": "GCG_TAG_WEAPON",
             "artifacts": "GCG_TAG_ARTIFACT", "food": "GCG_TAG_FOOD"}


@cache
def _card_rows() -> dict[int, dict]:
    return {row["id"]: row for row in load_table("GCGCardExcelConfigData")}


def card_cost(card: int) -> int:
    return sum(cost.get("count", 0) for cost in _card_rows().get(card, {}).get("costList", []) if cost)


def deck_shape(decks: list[Deck]) -> dict[str, float]:
    cards = [card for deck in decks for card in deck.cards]
    shape = {"decks": len(decks), "lineups": len({tuple(sorted(deck.characters)) for deck in decks}),
             "average card cost": sum(map(card_cost, cards)) / len(cards),
             "free card share": sum(card_cost(card) == 0 for card in cards) / len(cards),
             "cards at two copies": sum(sum(copies == 2 for copies in Counter(deck.cards).values()) for deck in decks) / len(decks)}
    for kind, tag in KIND_TAGS.items():
        shape[kind] = sum(tag in _card_rows().get(card, {}).get("tagList", []) for card in cards) / len(decks)
    return shape
