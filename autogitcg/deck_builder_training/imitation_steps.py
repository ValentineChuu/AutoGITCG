"""Teacher-forced building steps for imitation: replay an example deck pick by pick in a random order.

A deck has no order, so each step's target is the whole set of still-missing picks (characters not yet picked,
cards with copies still missing). The loss (imitation_update.py) maximizes the probability of picking any of
them; the replay then continues with one of them, chosen at random.
"""

import random
from collections import Counter

import numpy as np

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.building_encoder import BuildingEncoder
from autogitcg.deck_builder.building_state import CHARACTER, BuildingState
from autogitcg.deck_builder.building_task import BuildingTask
from autogitcg.model.observation_batch import compact


def imitation_steps(task: BuildingTask, deck: Deck, encoder: BuildingEncoder, rng: random.Random) -> list[dict[str, np.ndarray]]:
    """Observations with an extra `candidate_target` array: 1 for candidates that belong to the example deck."""
    state = BuildingState(task)
    target_characters = set(deck.characters)
    target_cards = Counter(deck.cards)
    observations = []
    while not state.finished:
        candidates = state.candidates()
        targets = [
            (identifier in target_characters and identifier not in state.characters) if kind == CHARACTER
            else state.cards[identifier] < target_cards[identifier]
            for kind, identifier in candidates
        ]
        if not any(targets):
            raise ValueError(f"example deck {deck} cannot be built from its task's collection")
        observation = encoder.encode(state, candidates)
        observation["candidate_target"] = np.array(targets, dtype=np.float32)
        observations.append(compact(observation))
        state.apply(rng.choice([candidate for candidate, is_target in zip(candidates, targets) if is_target]))
    return observations
