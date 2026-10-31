"""Future-board targets (training/future_targets.py) and the network's future head."""

import numpy as np
import torch

from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.encode_board import CHARACTER_COUNT, CHARACTER_WIDTH, _character_features
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.training.future_targets import FEATURES_PER_CHARACTER, FUTURE_HORIZONS, FUTURE_WIDTH, board_summary, future_targets


def _observation(health: int) -> dict:
    character = {"health": health, "maxHealth": 10, "energy": 2, "maxEnergy": 3, "aura": 2}  # aura code 2: hydro
    features = np.zeros((CHARACTER_COUNT, CHARACTER_WIDTH), dtype=np.float16)
    features[0] = _character_features(character, True, 0, True)
    mask = np.zeros(CHARACTER_COUNT, dtype=np.uint8)
    mask[0] = 1
    return {"character_features": features, "character_mask": mask}


def test_board_summary_reads_active_health_energy_and_aura():
    summary = board_summary(_observation(health=7))
    first = summary[:FEATURES_PER_CHARACTER]
    assert first[0] == 1 and first[1] == 0  # active, not defeated
    assert np.isclose(first[2], 0.7, atol=1e-3) and np.isclose(first[3], 2 / 3, atol=1e-3)
    assert first[4:].sum() == 1  # one aura element
    assert not summary[FEATURES_PER_CHARACTER:].any()  # empty slots stay zero


def test_targets_look_ahead_and_mask_past_the_end():
    observations = [_observation(health=10 - index % 10) for index in range(12)]
    targets, masks = future_targets(observations)
    width = FUTURE_WIDTH // len(FUTURE_HORIZONS)
    assert np.isclose(targets[0][2], board_summary(observations[FUTURE_HORIZONS[0]])[2], atol=1e-3)
    assert np.isclose(targets[0][width + 2], board_summary(observations[FUTURE_HORIZONS[1]])[2], atol=1e-3)
    assert masks[0].tolist() == [1, 1]
    assert masks[5].tolist() == [1, 0]  # 5 + 10 is past the last decision
    assert masks[11].tolist() == [0, 0]


def test_v2_loads_with_a_future_head_that_starts_at_nothing_changes():
    network = load_combat_network("models/beta-10.8.pt", ObservationEncoder(), torch.device("cpu"))
    last = network.future_head[-1]
    assert last.out_features == FUTURE_WIDTH and not last.weight.any() and not last.bias.any()
