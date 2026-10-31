"""Where each match's deck comes from: Chudledee building for the stage, or a random legal deck."""

import random
from collections import Counter
from pathlib import Path

import torch

from autogitcg.data.deck import Deck
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import STAGE_MODE, BuildingTask
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.progression.account import Account
from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection


class ChudledeeDecks:
    def __init__(self, builder_checkpoint: Path, skip_unusable_weapons: bool = False):
        self.skip_unusable_weapons = skip_unusable_weapons
        setup = BuilderSetup()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network = setup.network_from(load_builder_checkpoint(builder_checkpoint)).to(self.device)
        self.builder = BuilderPolicy(network, setup.encoder, self.device, greedy=True)

    def decks(self, requests: list[tuple[Account, Stage]]) -> list[Deck]:
        tasks = [BuildingTask(Collection(tuple(account.owned_characters()), Counter(account.owned_action_cards()), account.matches_played),
                              STAGE_MODE, stage, "playthrough", self.skip_unusable_weapons) for account, stage in requests]
        return [built.deck for built in self.builder.build(tasks)] if tasks else []


class RandomDecks:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def decks(self, requests: list[tuple[Account, Stage]]) -> list[Deck]:
        return [random_deck(account.owned_characters(), account.owned_action_cards(), self.rng) for account, _ in requests]
