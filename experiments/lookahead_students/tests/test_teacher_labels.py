"""Look-ahead labels (lookahead_teacher/): recorded from real look-ahead games, stacked, and trained on."""

import random

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.data.npc_decks import load_npc_decks
from experiments.lookahead_students.label_collector import LabelCollector
from experiments.lookahead_students.label_dataset import confident_target, label_set
from experiments.lookahead_students.student_training import StudentOptions, train_student
from experiments.lookahead_students.teacher_label import TeacherLabel
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import network_config_for
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.matchup_sampler import Matchup


def test_labels_from_lookahead_games_train_a_student():
    torch.manual_seed(0)
    encoder = ObservationEncoder()
    network = CombatNetwork(network_config_for(encoder, width=32, layers=2), encoder.catalog_features.table)
    device = torch.device("cpu")
    collector = LabelCollector(unsearched_share=1.0, rng=random.Random(0))
    # An untrained network is never confident, so nearly every action decision is searched.
    settings = LookaheadSettings(candidates=2, redeals=2, horizon=2)
    decks = list(load_npc_decks().values())[:2]
    labels, rows = [], []
    with EnginePool(worker_count=2) as pool:
        table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                               NetworkPolicy(network, device), settings, seed=0, on_searcher_move=collector.on_searcher_move)
        for seed in range(2):
            matchup = Matchup("npc_decks", (decks[0], decks[1]), (), (True, True), ("npc_deck:a", "npc_deck:b"))
            table.start_game(matchup, seed, searcher_seat=seed, row={})
        while table.game_count:
            for row in table.step():
                rows.append(row)
                labels.extend(collector.finish(row["game_id"], row))

    assert all("failure" not in row for row in rows), rows
    assert table.counts["searched decisions"] > 0
    assert any(label.searched for label in labels)
    for label in labels:
        assert 0 <= label.choice_index < len(label.probabilities)
        assert label.value_target != 0.0 or label.won == 0.0

    labels_as_arrays = label_set(labels)
    assert np.allclose(labels_as_arrays.probabilities.sum(axis=1), 1.0, atol=0.01)
    assert bool((labels_as_arrays.target_masks.sum(axis=1) >= 1).all())  # every target counts itself (and identical copies)
    assert len(label_set(labels, rule="confident")) == len(labels)
    epochs = []
    train_student(network, labels_as_arrays, labels_as_arrays, StudentOptions(epochs=1, minibatch_size=16), device,
                  lambda epoch, metrics: epochs.append(metrics))
    assert len(epochs) == 2
    assert all(np.isfinite(value) for value in epochs[-1]["validation"].values())


def test_confident_rule_only_takes_paired_significant_wins():
    """A candidate becomes the target only when it beats the network's own choice re-deal by re-deal, beyond noise."""
    clear = TeacherLabel({}, 0, 0, np.ones(3, dtype=np.float16) / 3, True, False, candidate_indices=[0, 2, 1],
                         candidate_redeal_scores=[[0.0, 0.1, -0.1, 0.0], [0.5, 0.6, 0.4, 0.5], [0.1, -0.9, 0.9, 0.0]])
    assert confident_target(clear) == 2  # candidate 1 (choice 2) wins every re-deal by about 0.5; candidate 2 is noise
    noisy = TeacherLabel({}, 0, 0, np.ones(3, dtype=np.float16) / 3, True, False, candidate_indices=[0, 1],
                         candidate_redeal_scores=[[0.0, 0.0, 0.0, 0.0], [0.9, -0.8, 0.7, -0.6]])
    assert confident_target(noisy) is None
