"""Action previews as network input (observation/encode_previews.py): real random-play games with previews on."""

import dataclasses
import random

import numpy as np
import pytest
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.model.observation_batch import stack, to_device
from autogitcg.observation.encode_actions import PREVIEW_START
from autogitcg.observation.encode_previews import PREVIEW_WIDTH
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.random_policy import random_response

GAME_COUNT = 6
HAS_PREVIEW = PREVIEW_START + PREVIEW_WIDTH - 1
DAMAGE_DEALT = PREVIEW_START + PREVIEW_WIDTH - 3


def _play(previews: bool) -> list:
    rng = random.Random(11)
    decks = list(load_npc_decks().values())
    encoder = ObservationEncoder()
    recorders, encoded = {}, []
    with EnginePool(worker_count=2, previews=previews) as pool:
        for _ in range(GAME_COUNT):
            deck_pair = tuple(rng.sample(decks, 2))
            game_id = pool.create_game(*deck_pair, seed=rng.randrange(2**31))
            recorders[game_id] = MatchRecorder(deck_pair, opponent_deck_known=(False, False))
        while pool.has_active_games():
            for event in pool.poll():
                if not isinstance(event, Decision):
                    continue
                tracker = recorders[event.game_id].on_decision(event)
                encoded.append((event, encoder.encode(event, tracker)))
                response = random_response(event, random.Random(hash((event.game_id, event.who, len(encoded)))))
                recorders[event.game_id].on_response(event, response)
                pool.respond(event, response)
    return encoded


@pytest.fixture(scope="module")
def preview_games():
    return _play(previews=True)


def test_every_action_token_has_a_preview(preview_games):
    action_decisions = [(decision, encoded) for decision, encoded in preview_games if decision.kind == "action"]
    assert action_decisions
    for decision, encoded in action_decisions:
        tokens = encoded.arrays["action_mask"] > 0
        assert (encoded.arrays["action_features"][tokens, HAS_PREVIEW] == 1).all()


def test_skills_show_their_damage(preview_games):
    damage = [encoded.arrays["action_features"][token, DAMAGE_DEALT]
              for decision, encoded in preview_games if decision.kind == "action"
              for token, request_index in enumerate(encoded.request_indices)
              if decision.request["action"][request_index]["action"]["$case"] == "useSkill"]
    assert damage and np.mean(np.array(damage) > 0) > 0.5


def test_games_without_previews_have_zero_preview_features():
    for decision, encoded in _play(previews=False)[:400]:
        assert not encoded.arrays["action_features"][:, PREVIEW_START:].any()


def test_networks_without_preview_weights_ignore_previews(preview_games):
    """beta 10.8 (trained before previews) loads with zero weights for them, so its outputs don't change."""
    network = load_combat_network("models/beta-10.8.pt", ObservationEncoder(), torch.device("cpu"))
    arrays = [encoded.arrays for decision, encoded in preview_games if decision.kind == "action"][:64]
    blanked = [dict(array, action_features=np.concatenate([array["action_features"][:, :PREVIEW_START],
                                                            np.zeros_like(array["action_features"][:, PREVIEW_START:])], axis=1))
               for array in arrays]
    with torch.no_grad():
        with_previews = network(to_device(stack(arrays), torch.device("cpu")))
        without = network(to_device(stack(blanked), torch.device("cpu")))
    assert torch.allclose(with_previews.action_logits, without.action_logits, equal_nan=True)
    assert torch.allclose(with_previews.value, without.value)


def test_preview_settings_survive_saving_and_loading(tmp_path):
    """A preview network loads with previews on (a test that ran a preview network with them off once gave a wrong result)."""
    network = load_combat_network("models/beta-10.8.pt", ObservationEncoder(), torch.device("cpu"))
    config = dataclasses.replace(network.config, reads_previews=True, preview_scale=10.0)
    path = tmp_path / "preview_network.pt"
    torch.save({"network_config": dataclasses.asdict(config), "network": network.state_dict()}, path)
    loaded = load_combat_network(path, ObservationEncoder(), torch.device("cpu"))
    assert loaded.config.reads_previews and loaded.config.preview_scale == 10.0


def test_preview_scale_multiplies_only_preview_inputs(preview_games):
    network = load_combat_network("models/beta-10.8.pt", ObservationEncoder(), torch.device("cpu"))
    with torch.no_grad():
        network.feature_projections["action"].weight[:, PREVIEW_START:] = 0.01
    arrays = [encoded.arrays for decision, encoded in preview_games if decision.kind == "action"][:16]
    scaled = [dict(array, action_features=np.concatenate([array["action_features"][:, :PREVIEW_START],
                                                          10 * array["action_features"][:, PREVIEW_START:]], axis=1))
              for array in arrays]
    with torch.no_grad():
        expected = network(to_device(stack(scaled), torch.device("cpu"))).action_logits
        network.config = dataclasses.replace(network.config, preview_scale=10.0)
        actual = network(to_device(stack(arrays), torch.device("cpu"))).action_logits
    assert torch.allclose(actual, expected, atol=1e-5, equal_nan=True)
