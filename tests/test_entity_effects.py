"""Entity facts in the catalog (observation/entity_effects.py): what summons and statuses do, read from the
card code; networks saved with the narrower catalog load with zero weights for the new columns and play unchanged."""

import random

import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.observation_batch import network_dtypes, stack, upload
from autogitcg.observation.entity_effects import EFFECT_WIDTH, effect_features, load_entity_effects
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.random_policy import random_response
from autogitcg.training.checkpoints import load_checkpoint, network_config_from

SESSHOU_SAKURA, TENKO_THUNDERBOLTS, BURNING = 114081, 114083, 115


def test_entity_effects_are_read_from_the_card_code():
    effects = load_entity_effects()
    assert effects[SESSHOU_SAKURA] == {"element": "Electro", "damage": 1}
    assert effects[TENKO_THUNDERBOLTS] == {"element": "Electro", "damage": 3}
    assert effects[BURNING]["element"] == "Pyro"
    assert len(effect_features(SESSHOU_SAKURA)) == EFFECT_WIDTH and sum(effect_features(None)) == 0


def _decisions(count: int = 40):
    rng = random.Random(5)
    decks = tuple(rng.sample(list(load_npc_decks().values()), 2))
    encoder = ObservationEncoder()
    recorder = MatchRecorder(decks, opponent_deck_known=(True, True))
    encoded = []
    with EnginePool(worker_count=1) as pool:
        pool.create_game(*decks, seed=9)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    tracker = recorder.on_decision(event)
                    if len(encoded) < count:
                        encoded.append(encoder.encode(event, tracker).arrays)
                    response = random_response(event, rng)
                    recorder.on_response(event, response)
                    pool.respond(event, response)
    return encoder, encoded


def test_a_network_saved_before_the_entity_facts_plays_unchanged():
    encoder, encoded = _decisions()
    checkpoint = load_checkpoint("models/beta-10.8.pt")
    config = network_config_from(checkpoint)
    table = encoder.catalog_features.table
    assert config.catalog_width == table.shape[1] - EFFECT_WIDTH  # beta 10.8 predates the entity facts
    before = CombatNetwork(config, table[:, :config.catalog_width])
    after = CombatNetwork(config, table)
    for network in (before, after):
        network.load_state_dict(checkpoint["network"])
        network.eval()
    assert after.catalog_projection.in_features == table.shape[1]
    assert not after.catalog_projection.weight[:, config.catalog_width:].any()
    batch = network_dtypes(upload(stack(encoded), torch.device("cpu")))
    with torch.no_grad():
        old, new = before(batch), after(batch)
    assert torch.allclose(old.action_logits, new.action_logits, atol=1e-5)
    assert torch.allclose(old.value, new.value, atol=1e-5)


def test_chudledee_saved_before_the_entity_facts_loads_with_zero_weights_for_them():
    from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint

    setup = BuilderSetup()
    checkpoint = load_builder_checkpoint("models/chudledee-v3.4.pt")
    saved_width = checkpoint["builder_config"]["catalog_width"]
    weight = setup.network_from(checkpoint).catalog_projection.weight
    assert weight.shape[1] == setup.catalog_features.width > saved_width
    assert not weight[:, saved_width:].any()
