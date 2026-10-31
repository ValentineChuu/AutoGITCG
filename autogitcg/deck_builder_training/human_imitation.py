"""Teacher-forced building steps from human-built decks, sampled by weight, as PvP tasks. Shared by the imitation warm
start and the imitation anchor during reinforcement learning.

Most examples are set in the full card pool, as people build from near-complete collections. In a small simulated
collection plus the deck, rare cards (Arcane Legends, Transfigurations) are almost only there because the example deck
holds them, so imitation learned "take every rare card you have" (6.5 Legends per deck vs people's 1.2).
"""

import random
from collections import Counter

import numpy as np

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.building_encoder import BuildingEncoder
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.deck_builder_training.imitation_examples import example_task
from autogitcg.deck_builder_training.imitation_steps import imitation_steps
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection

HUMAN_PVP_SHARE = 1.0  # tournament and tester decks are PvP decks
FULL_POOL_SHARE = 0.8
FULL_POOL_COPIES = 2


class HumanImitation:
    def __init__(self, weighted_decks: list[tuple[Deck, float]], collections: list[Collection], stages: list[Stage],
                 encoder: BuildingEncoder, rng: random.Random):
        self.decks = [deck for deck, _ in weighted_decks]
        self.weights = [weight for _, weight in weighted_decks]
        self.collections, self.stages, self.encoder, self.rng = collections, stages, encoder, rng
        self.full_pool = Collection(tuple(sorted(buildable_characters())),
                                    Counter({card: FULL_POOL_COPIES for card in buildable_action_cards()}), 0)

    def steps(self, deck_count: int) -> list[dict[str, np.ndarray]]:
        decks = self.rng.choices(self.decks, weights=self.weights, k=deck_count)
        return self.steps_for(decks)

    def steps_for(self, decks: list[Deck]) -> list[dict[str, np.ndarray]]:
        return [step for deck in decks for step in imitation_steps(self._task(deck), deck, self.encoder, self.rng)]

    def _task(self, deck: Deck) -> BuildingTask:
        if self.rng.random() < FULL_POOL_SHARE:
            return BuildingTask(self.full_pool, PVP_MODE, None, "imitation")
        return example_task(deck, self.collections, self.stages, self.rng, HUMAN_PVP_SHARE)
