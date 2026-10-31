"""Deck builder: every built deck is legal (also from tight starter collections), padding does not change the
network's choices, imitation replays an example deck exactly, and judged games come back through the players."""

import random
from collections import Counter

import numpy as np
import pytest
import torch

from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup
from autogitcg.deck_builder.building_batch import pad_and_stack
from autogitcg.deck_builder.building_state import BuildingState
from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.deck_builder.lineup_feasibility import lineup_is_feasible
from autogitcg.deck_builder.token_budget_batches import token_budget_batches
from autogitcg.deck_building.deck_rules import required_character
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.deck_builder_training.imitation_steps import imitation_steps
from autogitcg.model.observation_batch import to_device
from autogitcg.progression.shop import Shop
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.collection_pool import Collection, simulate_collections
from autogitcg.training.deck_pools import load_synergy_decks

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def setting():
    stages, _ = build_stage_catalog()
    starter_account = ProgressionSimulator(stages, Shop(), random.Random(0)).account
    starter = Collection(tuple(starter_account.owned_characters()), Counter(starter_account.owned_action_cards()), 0)
    collections = [starter] + simulate_collections(stages, 6, 6, seed=3)
    setup = BuilderSetup()
    torch.manual_seed(0)
    return stages, collections, setup, setup.new_network(width=32, layers=1).eval()


def test_built_decks_are_legal(setting):
    stages, collections, setup, network = setting
    rng = random.Random(0)
    tasks = [BuildingTask(collection, STAGE_MODE, rng.choice(stages)) if number % 2 else BuildingTask(collection, PVP_MODE)
             for number, collection in enumerate(collections)]
    assert sum(collections[0].action_cards.values()) == 30  # the starter collection: exactly one legal card pool
    for greedy in (False, True):
        for built in BuilderPolicy(network, setup.encoder, CPU, greedy=greedy).build(tasks):
            assert not deck_problems(built.deck), (built.deck, deck_problems(built.deck))
            assert len(built.steps) == 33


def test_infeasible_lineups_are_never_offered():
    """A collection whose only cards are one deck's 30 (with talents): every offered lineup must still reach 30 legal cards."""
    synergy, _ = load_synergy_decks()
    deck = next(deck for deck in synergy if any(required_character(card) for card in deck.cards))
    assert any(required_character(card) for card in deck.cards), "test needs a synergy deck with a talent"
    extra_characters = [character for character in (1101, 1201, 1301, 1401, 1501) if character not in deck.characters]
    collection = Collection(tuple(deck.characters) + tuple(extra_characters), Counter(deck.cards), 10)
    state = BuildingState(BuildingTask(collection, PVP_MODE))
    while state.picking_characters:
        candidates = state.candidates()
        state.apply(random.Random(len(state.characters)).choice(candidates))
    assert lineup_is_feasible(state.characters, collection.action_cards)


def test_padding_does_not_change_choices(setting):
    stages, collections, setup, network = setting
    state = BuildingState(BuildingTask(collections[0], STAGE_MODE, stages[0]))
    short = setup.encoder.encode(state, state.candidates())
    long_state = BuildingState(BuildingTask(collections[-1], PVP_MODE))
    long = setup.encoder.encode(long_state, long_state.candidates())
    with torch.no_grad():
        alone = network(to_device(pad_and_stack([short]), CPU))
        together = network(to_device(pad_and_stack([short, long]), CPU))
    real = len(short["candidate_mask"])
    assert torch.allclose(alone.candidate_logits[0, :real], together.candidate_logits[0, :real], atol=1e-4)
    assert torch.allclose(alone.value[0], together.value[0], atol=1e-4)


def test_token_budget_batches_cover_every_row_once():
    lengths = [700] * 5 + [400] * 50 + [60] * 3
    batches = token_budget_batches(lengths, 4000, np.random.default_rng(0))
    assert sorted(int(row) for batch in batches for row in batch) == list(range(len(lengths)))
    assert all(len(batch) * max(lengths[row] for row in batch) <= 4000 or len(batch) == 1 for batch in batches)


def test_imitation_replays_the_example_deck(setting):
    stages, collections, setup, _ = setting
    synergy, _ = load_synergy_decks()
    deck = synergy[0]
    base = collections[0]
    cards = Counter(base.action_cards)
    for card, copies in Counter(deck.cards).items():
        cards[card] = max(cards[card], copies)
    task = BuildingTask(Collection(tuple(set(base.characters) | set(deck.characters)), cards, 0), PVP_MODE)
    steps = imitation_steps(task, deck, setup.encoder, random.Random(1))
    assert len(steps) == 33 and all(step["candidate_target"].sum() >= 1 for step in steps)


def test_judged_games_come_back(setting, tmp_path):
    """Two game-player processes with a tiny random combat network play a few stage and PvP games."""
    from autogitcg.deck_builder_training.deck_games import deck_games
    from autogitcg.deck_builder_training.deck_rewards import deck_result
    from autogitcg.deck_builder_training.game_player_group import GamePlayerGroup
    from autogitcg.deck_builder_training.game_player_process import PlayerSettings
    from autogitcg.model.combat_network import CombatNetwork
    from autogitcg.model.network_config import network_config_for
    from autogitcg.observation.observation_encoder import ObservationEncoder
    from dataclasses import asdict
    stages, collections, setup, network = setting
    encoder = ObservationEncoder()
    combat = CombatNetwork(network_config_for(encoder, width=32, layers=1), encoder.catalog_features.table)
    path = tmp_path / "combat.pt"
    torch.save({"network_config": asdict(combat.config), "network": combat.state_dict()}, path)
    rng = random.Random(4)
    tasks = [BuildingTask(collections[-1], STAGE_MODE, rng.choice(stages)), BuildingTask(collections[-1], PVP_MODE)]
    built = BuilderPolicy(network, setup.encoder, CPU).build(tasks)
    schedules = [deck_games(deck.task, deck.deck, 2, rng, lambda: (deck.deck, "mirror")) for deck in built]
    with GamePlayerGroup(2, PlayerSettings(path, workers=1, concurrent_games=4)) as players:
        outcomes = players.play([game for schedule in schedules for game in schedule])
    assert [outcome.job for outcome in outcomes] == list(range(4))
    for number, schedule in enumerate(schedules):
        result = deck_result(schedule, outcomes[2 * number:2 * number + 2])
        assert result.games == 2 and result.failures == 0


def test_lineup_diversity_counts_only_near_complete_collections():
    from autogitcg.deck_builder_training.deck_diversity import lineup_diversity
    metrics = lineup_diversity([("pvp", 147, [1, 2, 3]), ("pvp", 147, [3, 2, 1]), ("pvp", 147, [4, 5, 6]), ("pvp", 10, [7, 8, 9])])
    assert metrics == {"near_complete_pvp_decks": 3, "near_complete_pvp_distinct_lineups": 2,
                       "near_complete_pvp_top_lineup_share": 2 / 3}
