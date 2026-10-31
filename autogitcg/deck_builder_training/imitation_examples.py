"""Example decks to imitate: the hand-built synergy decks and the legal NPC decks (warm start before reinforcement
learning), or human-built decks (human_example_decks.py; PvP tasks).

Each example is set in a collection that contains the deck plus a random simulated collection, so the builder
learns to find the deck's cards among others. The task (PvP or a random stage) varies too; imitation ignores it.
"""

import random
from collections import Counter

from autogitcg.data.deck import Deck
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection

SYNERGY_REPEATS = 4  # synergy decks are curated, NPC decks are only themed; weigh synergy decks up


def example_decks(synergy_decks: list[Deck], npc_decks: list[Deck]) -> list[Deck]:
    """Only legal decks: a deck made before a rule change (e.g. a tightened copy limit) can't be rebuilt pick by pick."""
    legal_synergy = [deck for deck in synergy_decks if not deck_problems(deck)]
    legal_npc = [deck for deck in npc_decks if not deck_problems(deck)]
    if len(legal_synergy) < len(synergy_decks) or len(legal_npc) < len(npc_decks):
        print(f"left out illegal example decks: {len(synergy_decks) - len(legal_synergy)} synergy, {len(npc_decks) - len(legal_npc)} NPC")
    return legal_synergy * SYNERGY_REPEATS + legal_npc


def example_task(deck: Deck, collections: list[Collection], stages: list[Stage], rng: random.Random,
                 pvp_share: float = 0.5) -> BuildingTask:
    base = rng.choice(collections)
    characters = tuple(sorted(set(base.characters) | set(deck.characters)))
    cards = Counter(base.action_cards)
    for card, copies in Counter(deck.cards).items():
        cards[card] = max(cards[card], copies)
    collection = Collection(characters, cards, base.matches_played)
    if rng.random() >= pvp_share:
        return BuildingTask(collection, STAGE_MODE, rng.choice(stages), "imitation")
    return BuildingTask(collection, PVP_MODE, None, "imitation")
