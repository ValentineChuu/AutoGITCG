"""Blind PvP matchups of a locked-test tournament deck (human-built) against a Chudledee PvP deck, for the play-versus-tune
diagnostic on human decks. The tournament deck's origin starts with TOURNAMENT_ORIGIN."""

import collections
import random

import torch

from autogitcg.data.tournament_test_set import load_test_decks
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters, deck_problems
from autogitcg.training.collection_pool import Collection
from autogitcg.training.matchup_sampler import Matchup, swapped

TOURNAMENT_ORIGIN = "tournament_test"
DECKS_PER_SOURCE = 40


def tournament_vs_chudledee_matchups(builder_checkpoint, count: int, seed: int) -> list[Matchup]:
    tournament = [entry for entry in load_test_decks() if not deck_problems(entry.deck)][:DECKS_PER_SOURCE]
    setup = BuilderSetup()
    device = torch.device("cpu")
    builder = BuilderPolicy(setup.network_from(load_builder_checkpoint(builder_checkpoint)).to(device), setup.encoder, device)
    everything = Collection(tuple(sorted(buildable_characters())), collections.Counter({card: 2 for card in buildable_action_cards()}), 0)
    chudledee = [built.deck for built in builder.build([BuildingTask(everything, PVP_MODE, None, "everything")] * DECKS_PER_SOURCE,
                                                       generator=torch.Generator().manual_seed(seed))]
    rng = random.Random(seed)
    matchups = []
    for _ in range(count):
        human = rng.choice(tournament)
        matchup = Matchup("card_play_tournament", (human.deck, rng.choice(chudledee)), (), (False, False),
                          (f"{TOURNAMENT_ORIGIN}:{human.lineup_name}", "chudledee"))
        matchups.append(swapped(matchup) if rng.random() < 0.5 else matchup)
    return matchups
