"""Opponent decks for PvP games: what other players might bring.

A mix of real tournament decks (the human meta, minus the locked test lineups), hand-built synergy decks, NPC
decks, random decks from simulated collections, and decks the builder itself made recently (so it keeps facing
its own best ideas instead of only a fixed pool).
"""

import random
from collections import deque

from autogitcg.data.deck import Deck
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.training.deck_pools import DeckPools

# Chudledee v1 used {synergy 0.35, npc_deck 0.15, collection 0.2, built 0.3}, before tournament decks existed.
SOURCE_WEIGHTS = {"tournament": 0.3, "built": 0.25, "synergy": 0.2, "collection": 0.15, "npc_deck": 0.1}
RECENT_BUILT_DECKS = 1000


class PvpOpponents:
    def __init__(self, pools: DeckPools, legal_npc_deck_ids: list[int], rng: random.Random):
        self.pools = pools
        self.npc_deck_ids = legal_npc_deck_ids
        self.rng = rng
        self.recent_built: deque[Deck] = deque(maxlen=RECENT_BUILT_DECKS)

    def add_built(self, decks: list[Deck]) -> None:
        self.recent_built.extend(decks)

    def sample(self) -> tuple[Deck, str]:
        weights = dict(SOURCE_WEIGHTS)
        if not self.recent_built:
            del weights["built"]
        if not self.pools.tournament_decks:
            del weights["tournament"]
        source = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        if source == "tournament":
            index = self.rng.randrange(len(self.pools.tournament_decks))
            return self.pools.tournament_decks[index], f"tournament:{index}"
        if source == "synergy":
            index = self.rng.randrange(len(self.pools.synergy_decks))
            return self.pools.synergy_decks[index], f"synergy:{index}"
        if source == "npc_deck":
            deck_id = self.rng.choice(self.npc_deck_ids)
            return self.pools.npc_decks[deck_id], f"npc_deck:{deck_id}"
        if source == "collection":
            index = self.rng.randrange(len(self.pools.collections))
            collection = self.pools.collections[index]
            return random_deck(list(collection.characters), collection.action_cards, self.rng), f"collection:{index}"
        return self.rng.choice(self.recent_built), "built"
