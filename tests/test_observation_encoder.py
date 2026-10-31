"""Encode every decision of real (random-play) games through the bridge and check the result."""

import random
import time
from collections import Counter

import numpy as np
import pytest

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.observation.vocabulary import HIDDEN_INDEX
from autogitcg.policies.random_policy import random_response

GAME_COUNT = 20


@pytest.fixture(scope="module")
def encoded_games():
    rng = random.Random(7)
    decks = list(load_npc_decks().values())
    encoder = ObservationEncoder()
    recorders: dict[int, MatchRecorder] = {}
    results = {"encoded": [], "decisions": [], "seconds": 0.0, "trackers": []}
    with EnginePool(worker_count=4) as pool:
        for game_index in range(GAME_COUNT):
            deck_pair = tuple(rng.sample(decks, 2))
            game_id = pool.create_game(*deck_pair, seed=rng.randrange(2**31))
            recorders[game_id] = MatchRecorder(deck_pair, opponent_deck_known=(game_index % 2 == 0, False))
        while pool.has_active_games():
            for event in pool.poll():
                if not isinstance(event, Decision):
                    continue
                recorder = recorders[event.game_id]
                tracker = recorder.on_decision(event)
                started = time.perf_counter()
                encoded = encoder.encode(event, tracker)
                results["seconds"] += time.perf_counter() - started
                results["encoded"].append(encoded)
                results["decisions"].append(event)
                response = random_response(event, rng)
                recorder.on_response(event, response)
                pool.respond(event, response)
    results["trackers"] = [tracker for recorder in recorders.values() for tracker in recorder.trackers]
    return results


def test_shapes_are_fixed_and_finite(encoded_games):
    reference = encoded_games["encoded"][0].arrays
    for encoded in encoded_games["encoded"]:
        for name, array in encoded.arrays.items():
            assert array.shape == reference[name].shape, name
            assert np.isfinite(array).all(), name


def test_every_valid_action_has_a_token(encoded_games):
    for decision, encoded in zip(encoded_games["decisions"], encoded_games["encoded"]):
        assert encoded.dropped_tokens == 0
        if decision.kind == "action":
            valid = [index for index, action in enumerate(decision.request["action"]) if action.get("validity", 0) == 0]
            assert encoded.request_indices == valid


def test_visible_definitions_are_in_vocabulary(encoded_games):
    hidden_hits = Counter()
    for encoded in encoded_games["encoded"]:
        arrays = encoded.arrays
        hidden_hits["character"] += int(((arrays["character_indices"] == HIDDEN_INDEX) & (arrays["character_mask"] > 0)).sum())
        hidden_hits["entity"] += int(((arrays["entity_indices"] == HIDDEN_INDEX) & (arrays["entity_mask"] > 0)).sum())
        my_hand = (arrays["hand_mask"] > 0) & (arrays["hand_features"][:, 0] == 1.0)
        hidden_hits["my hand"] += int(((arrays["hand_indices"] == HIDDEN_INDEX) & my_hand).sum())
        hidden_hits["action"] += int(((arrays["action_indices"] == HIDDEN_INDEX) & (arrays["action_mask"] > 0)).sum())
    assert sum(hidden_hits.values()) == 0, hidden_hits


def test_tracked_deck_counts_stay_valid(encoded_games):
    for tracker in encoded_games["trackers"]:
        assert all(count >= 0 for count in tracker.my_remaining_deck.values())
        assert sum(tracker.my_remaining_deck.values()) <= 30


def test_opponent_hand_identities_stay_hidden(encoded_games):
    for encoded in encoded_games["encoded"]:
        arrays = encoded.arrays
        opponent_hand = (arrays["hand_mask"] > 0) & (arrays["hand_features"][:, 0] == 0.0)
        assert (arrays["hand_indices"][opponent_hand] == HIDDEN_INDEX).all()


def test_report_encoding_speed(encoded_games):
    count = len(encoded_games["encoded"])
    print(f"\nencoded {count} decisions, {1e6 * encoded_games['seconds'] / count:.0f} µs per decision")
