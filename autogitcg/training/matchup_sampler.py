"""Picks the decks for each training game.

First a matchup category (training/matchup_categories.py), weighted toward the categories the learner currently
loses (training/adaptive_difficulty.py); then decks for it:
- stage: a player deck (a random legal deck from a simulated collection, a synergy deck, or a deck Chudledee
  built for that stage) vs a real PvE stage (lineup + pre-equipped talents). The player side usually knows
  the stage deck (NPC mode).
- tournament / chudledee: a real tournament deck or a Chudledee PvP deck vs mostly human-built or Chudledee
  decks (HUMAN_LIKE_OPPONENTS), so both sides play coherent, competitive lists.
- synergy: a hand-built coherent deck (data/decks/) vs another synergy, NPC or collection deck.
- npc_decks: two NPC 3+30 decks; collections: two random collection decks, like PvP between real players.
The first deck chosen is the "focus" side (the category is named after it). Seats are swapped
at random so either side can go first; deck knowledge is withheld at random so one network learns both NPC
mode and blind mode. Every deck carries its origin (e.g. "collection:12", "stage:2201001").
"""

import random
from dataclasses import dataclass, replace

from autogitcg.bridge.initial_equipment import InitialEquipment
from autogitcg.data.deck import Deck
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.training.adaptive_difficulty import AdaptiveDifficulty
from autogitcg.training.deck_pools import DeckPools
from autogitcg.training.matchup_categories import base_category_weights

KNOWS_STAGE_DECK = 0.8  # chance the player side is told the stage's deck
KNOWS_OPPONENT_DECK = 0.5  # every other case
HUMAN_LIKE_OPPONENTS = {"tournament": 0.5, "chudledee": 0.25, "synergy": 0.125, "collection": 0.125}  # for tournament/chudledee


@dataclass
class Matchup:
    source: str
    decks: tuple[Deck, Deck]
    equipment: tuple[InitialEquipment, ...]
    opponent_deck_known: tuple[bool, bool]
    deck_origins: tuple[str, str]  # per seat, e.g. "collection:<index>", "npc_deck:<id>", "stage:<level id>", "synergy:<index>",
    # "tournament:<index>", "chudledee:<index>", "chudledee_stage:<level id>:<index>"
    category: str = ""
    focus_seat: int = 0  # seat holding the category's focus deck


class MatchupSampler:
    def __init__(self, pools: DeckPools, rng: random.Random, adaptive: bool = True, source_weights: dict[str, float] | None = None,
                 human_deck_opponents: dict[str, float] | None = None):
        """human_deck_opponents overrides HUMAN_LIKE_OPPONENTS (opponent kinds for tournament/chudledee decks)."""
        self.pools = pools
        self.human_deck_opponents = human_deck_opponents or HUMAN_LIKE_OPPONENTS
        self.rng = rng
        self.npc_deck_ids = sorted(pools.npc_decks)
        self.stages_by_kind: dict[str, list] = {}
        for stage in pools.stages:
            self.stages_by_kind.setdefault(stage.kind, []).append(stage)
        self.synergy_indices_by_archetype: dict[str, list[int]] = {}
        for index, archetype in enumerate(pools.synergy_archetypes):
            self.synergy_indices_by_archetype.setdefault(archetype, []).append(index)
        self.difficulty = AdaptiveDifficulty(base_category_weights(pools, source_weights))
        self.adaptive = adaptive

    def record_result(self, category: str, pair_score: float) -> None:
        """The learner's mean score over a mirror pair against the reference (training/adaptive_difficulty.py)."""
        self.difficulty.record(category, pair_score)

    def sample(self) -> Matchup:
        weights = self.difficulty.weights() if self.adaptive else self.difficulty.base_weights
        category = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        source, *details = category.split("/")
        if source == "stage":
            kind, player_type = details
            stage = self.rng.choice(self.stages_by_kind[kind])
            if player_type == "chudledee" and stage.level_id in self.pools.chudledee.stage:
                player = self._chudledee_stage_deck(stage.level_id)
            else:
                player = self._synergy_deck() if player_type == "synergy" else self._collection_deck()
            matchup = Matchup(source, (player[0], stage.enemy_deck), stage.enemy_equipment,
                              (self.rng.random() < KNOWS_STAGE_DECK, self.rng.random() < KNOWS_OPPONENT_DECK),
                              (player[1], f"stage:{stage.level_id}"), category)
        else:
            if source in ("tournament", "chudledee"):
                first = self._tournament_deck() if source == "tournament" else self._chudledee_deck()
                opponent_kind = self.rng.choices(list(self.human_deck_opponents), weights=list(self.human_deck_opponents.values()))[0]
                second = {"tournament": self._tournament_deck, "chudledee": self._chudledee_deck,
                          "synergy": self._synergy_deck, "collection": self._collection_deck}[opponent_kind]()
            elif source == "synergy":
                first = self._synergy_deck(details[0])
                second = self.rng.choice([self._synergy_deck, self._npc_deck, self._collection_deck])()
            else:
                pick = self._npc_deck if source == "npc_decks" else self._collection_deck
                first, second = pick(), pick()
            matchup = Matchup(source, (first[0], second[0]), (),
                              (self.rng.random() < KNOWS_OPPONENT_DECK, self.rng.random() < KNOWS_OPPONENT_DECK),
                              (first[1], second[1]), category)
        return swapped(matchup) if self.rng.random() < 0.5 else matchup

    def _npc_deck(self) -> tuple[Deck, str]:
        deck_id = self.rng.choice(self.npc_deck_ids)
        return self.pools.npc_decks[deck_id], f"npc_deck:{deck_id}"

    def _synergy_deck(self, archetype: str | None = None) -> tuple[Deck, str]:
        if not self.pools.synergy_decks:
            return self._collection_deck()
        candidates = self.synergy_indices_by_archetype[archetype] if archetype else range(len(self.pools.synergy_decks))
        index = self.rng.choice(list(candidates))
        return self.pools.synergy_decks[index], f"synergy:{index}"

    def _tournament_deck(self) -> tuple[Deck, str]:
        if not self.pools.tournament_decks:
            return self._collection_deck()
        index = self.rng.randrange(len(self.pools.tournament_decks))
        return self.pools.tournament_decks[index], f"tournament:{index}"

    def _chudledee_deck(self) -> tuple[Deck, str]:
        if not self.pools.chudledee.pvp:
            return self._collection_deck()
        index = self.rng.randrange(len(self.pools.chudledee.pvp))
        return self.pools.chudledee.pvp[index], f"chudledee:{index}"

    def _chudledee_stage_deck(self, level_id: int) -> tuple[Deck, str]:
        index = self.rng.randrange(len(self.pools.chudledee.stage[level_id]))
        return self.pools.chudledee.stage[level_id][index], f"chudledee_stage:{level_id}:{index}"

    def _collection_deck(self) -> tuple[Deck, str]:
        index = self.rng.randrange(len(self.pools.collections))
        collection = self.pools.collections[index]
        return random_deck(list(collection.characters), collection.action_cards, self.rng), f"collection:{index}"


def swapped(matchup: Matchup) -> Matchup:
    """The same game with the two decks (and their equipment, knowledge and origins) in the other seats."""
    return Matchup(
        matchup.source,
        (matchup.decks[1], matchup.decks[0]),
        tuple(replace(item, who=1 - item.who) for item in matchup.equipment),
        (matchup.opponent_deck_known[1], matchup.opponent_deck_known[0]),
        (matchup.deck_origins[1], matchup.deck_origins[0]),
        matchup.category,
        1 - matchup.focus_seat,
    )
