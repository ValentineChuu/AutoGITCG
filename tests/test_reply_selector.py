"""The learned reply selector (docs/architecture.md, policies/reply_selection.py): replies on demand from the
engine, the "not checked" input, the selector's first look and its loss."""

import random

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, RepliesReady
from autogitcg.bridge.reply_views import OPEN
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import network_config_for
from autogitcg.model.observation_batch import network_dtypes, stack, upload
from autogitcg.observation.encode_actions import MAX_ACTIONS, REPLY_START, UNCHECKED_COLUMN
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.random_policy import random_response
from autogitcg.policies.reply_selection import checked_actions, interest_target, on_demand
from autogitcg.training.reply_interest_loss import first_look, reply_interest_loss


def _play_on_demand(checks: int = 2) -> list[tuple[Decision, object, list[int]]]:
    """One game with replies on demand for seat 0: the first `checks` valid actions are checked."""
    rng = random.Random(3)
    deck_pair = tuple(rng.sample(list(load_npc_decks().values()), 2))
    encoder = ObservationEncoder()
    recorder = MatchRecorder(deck_pair, opponent_deck_known=(True, True))
    waiting, results = {}, []
    with EnginePool(worker_count=1, replies=True) as pool:
        pool.create_game(*deck_pair, seed=11, reply_views=[on_demand(OPEN), None])
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, RepliesReady):
                    decision, tracker, asked = waiting.pop(event.who)
                    decision.replies, decision.replies_checked = event.replies, event.checked
                    results.append((decision, encoder.encode(decision, tracker), asked))
                    response = random_response(decision, rng)
                    recorder.on_response(decision, response)
                    pool.respond(decision, response)
                elif isinstance(event, Decision):
                    tracker = recorder.on_decision(event)
                    if event.replies_on_demand:
                        valid = [index for index, action in enumerate(event.request["action"]) if action.get("validity", 0) == 0]
                        waiting[event.who] = (event, tracker, valid[:checks])
                        pool.request_replies(event, valid[:checks])
                        continue
                    response = random_response(event, rng)
                    recorder.on_response(event, response)
                    pool.respond(event, response)
    return results


def test_replies_on_demand_come_only_for_the_checked_actions():
    results = _play_on_demand()
    assert results and all(decision.who == 0 for decision, _, _ in results)
    for decision, encoded, asked in results:
        assert sorted(decision.replies_checked) == sorted(asked)
        assert all((reply is not None) == (index in asked) for index, reply in enumerate(decision.replies))
        features = encoded.arrays["action_features"]
        for token, request_index in enumerate(encoded.request_indices):
            assert features[token, UNCHECKED_COLUMN] == float(request_index not in asked)


def test_the_first_look_hides_replies_and_marks_every_action_unchecked():
    _, encoded, _ = _play_on_demand()[0]
    batch = network_dtypes(upload(stack([encoded.arrays]), torch.device("cpu")))
    looked = first_look(batch)
    real = looked["action_mask"][0] > 0
    assert not looked["action_features"][0, :, REPLY_START:UNCHECKED_COLUMN].any()
    assert (looked["action_features"][0, real, UNCHECKED_COLUMN] == 1).all()
    assert torch.equal(looked["action_features"][0, :, :REPLY_START], batch["action_features"][0, :, :REPLY_START])


def test_the_selector_checks_its_most_interesting_actions_or_all():
    interest = np.array([0.1, 0.9, 0.5, 0.0])
    assert checked_actions(interest, [4, 6, 7, 9], checks=2, full=False) == [6, 7]
    assert checked_actions(interest, [4, 6, 7, 9], checks=2, full=True) == [4, 6, 7, 9]
    target = interest_target(np.array([0.5, 0.5]), np.array([0.2, 0.8]))
    assert target.shape == (MAX_ACTIONS,) and np.allclose(target[:2], [0.3, 0.3]) and not target[2:].any()


def test_the_interest_loss_trains_only_on_fully_checked_rows():
    encoder = ObservationEncoder()
    network = CombatNetwork(network_config_for(encoder, width=32, layers=1), encoder.catalog_features.table)
    encoded = [encoded for _, encoded, _ in _play_on_demand()[:4]]
    batch = network_dtypes(upload(stack([item.arrays for item in encoded]), torch.device("cpu")))
    targets = torch.rand(len(encoded), MAX_ACTIONS)
    loss, coverage = reply_interest_loss(network, batch, targets, torch.tensor([True, False, True, False]), checks=2)
    loss.backward()
    assert loss > 0 and 0 <= coverage <= 1
    assert network.reply_interest_head[2].weight.grad.abs().sum() > 0
    none, _ = reply_interest_loss(network, batch, targets, torch.zeros(len(encoded), dtype=torch.bool), checks=2)
    assert none == 0


def test_checked_moves_also_get_follow_ups():
    results = _play_on_demand()
    with_follow_ups = 0
    for decision, encoded, asked in results:
        for index, reply in enumerate(decision.replies):
            if reply is None:
                continue
            follow = reply.get("followUps")
            assert follow is not None and set(follow) >= {"mostLost", "defeated", "mostDamage", "lines", "stoppedAtRandom", "capped"}
            with_follow_ups += follow["lines"] > 0
        features = encoded.arrays["action_features"]
        for token, request_index in enumerate(encoded.request_indices):
            computed = features[token, -1]
            assert computed == float(decision.replies[request_index] is not None)
    assert with_follow_ups > 0
