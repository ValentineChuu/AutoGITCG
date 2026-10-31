"""Guesses from Chudledee, the deck builder: decks for the opponent's lineup that already hold the cards they have
shown, so the rest is filled around them (their second copies, cards that work with them and with the lineup).
With nothing revealed this is the lineup-only guess search/deck_guess.py makes at the start of a game."""

from collections import Counter
from pathlib import Path

import torch

from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_state import CARD, BuildingState
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, max_copies
from autogitcg.deck_building.deck_validation import buildable_action_cards
from autogitcg.training.collection_pool import Collection


class ChudledeeGuesser:
    def __init__(self, builder_checkpoint: Path, seed: int, device: torch.device):
        self.setup = BuilderSetup()
        network = self.setup.network_from(load_builder_checkpoint(builder_checkpoint)).to(device)
        self.policy = BuilderPolicy(network, self.setup.encoder, device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.every_card = Counter({card: max_copies(card) for card in buildable_action_cards()})

    def guesses(self, requests: list[tuple[tuple[int, ...], Counter]], per_request: int) -> list[list[tuple[float, Counter]]]:
        """For each (lineup, revealed cards): `per_request` sampled decks holding the revealed cards, equal weights.
        Revealed cards a deck of that lineup can't hold (e.g. cards created by effects) are left out of the start."""
        tasks, started = [], []
        for index, (lineup, revealed) in enumerate(requests):
            for _ in range(per_request):
                task = BuildingTask(Collection(tuple(lineup), self.every_card, 0), PVP_MODE, collection_origin=f"guess:{index}")
                state = BuildingState(task, characters=list(lineup))
                limits = state.card_limits()
                for card, count in sorted(revealed.items()):
                    for _ in range(min(count, limits.get(card, 0))):
                        if sum(state.cards.values()) < DECK_ACTION_CARDS:
                            state.apply((CARD, card))
                tasks.append(task)
                started.append(state)
        built = self.policy.build(tasks, self.generator, started)
        return [[(1.0, Counter(deck.deck.cards)) for deck in built[index * per_request:(index + 1) * per_request]]
                for index in range(len(requests))]
