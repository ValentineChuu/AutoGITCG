"""The opponent's replies as network input (engine/reference/bridge/opponent_replies.mjs, observation/encode_replies.py)."""

import random

import numpy as np
import pytest

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.observation.encode_actions import REPLY_START
from autogitcg.observation.encode_board import CHARACTERS_PER_SIDE
from autogitcg.bridge.reply_views import HIDDEN, OPEN
from autogitcg.observation.encode_replies import CARD_SOURCES, NEXT_DECIDERS, OUTCOME_WIDTH
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.random_policy import random_response

WORST_HEALTH = REPLY_START + len(NEXT_DECIDERS)
STOP_START = WORST_HEALTH + OUTCOME_WIDTH  # replies if I ended my turn after a fast action
CARD_START = STOP_START + OUTCOME_WIDTH  # the opponent's card replies
SOURCE_START = CARD_START + OUTCOME_WIDTH


def _play(replies: bool, games: int = 2, views=None) -> list:
    rng = random.Random(5)
    decks = list(load_npc_decks().values())
    encoder = ObservationEncoder()
    recorders, encoded = {}, []
    with EnginePool(worker_count=2, replies=replies) as pool:
        for _ in range(games):
            deck_pair = tuple(rng.sample(decks, 2))
            game_id = pool.create_game(*deck_pair, seed=rng.randrange(2**31), reply_views=views)
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
def reply_games():
    return _play(replies=True)


def test_every_action_token_says_who_decides_next(reply_games):
    for decision, encoded in reply_games:
        valid = sum((action.get("validity", 0) == 0) for action in decision.request.get("action", []))
        if decision.kind != "action" or valid < 2:
            continue  # single-move decisions get no replies (nothing to choose)
        tokens = encoded.arrays["action_mask"] > 0
        assert decision.replies is not None
        assert (encoded.arrays["action_features"][tokens, REPLY_START:REPLY_START + len(NEXT_DECIDERS)].sum(axis=1) == 1).all()


def test_replies_show_threats_to_my_characters(reply_games):
    worst = np.concatenate([encoded.arrays["action_features"][encoded.arrays["action_mask"] > 0, WORST_HEALTH:WORST_HEALTH + CHARACTERS_PER_SIDE]
                            for decision, encoded in reply_games if decision.kind == "action"])
    assert (worst < 0).any()  # some replies would damage my characters
    assert (worst <= 0.5).all()  # nobody is healed by much more than 5 HP


def test_games_without_replies_have_zero_reply_features():
    for decision, encoded in _play(replies=False, games=1)[:300]:
        assert decision.replies is None
        assert not encoded.arrays["action_features"][:, REPLY_START:].any()


def _reply_columns(games, start: int, width: int) -> np.ndarray:
    return np.concatenate([encoded.arrays["action_features"][encoded.arrays["action_mask"] > 0, start:start + width]
                           for decision, encoded in games if decision.kind == "action"])


def test_fast_actions_show_the_replies_after_ending_the_turn(reply_games):
    stop = _reply_columns(reply_games, STOP_START, OUTCOME_WIDTH)
    first_block = _reply_columns(reply_games, WORST_HEALTH, OUTCOME_WIDTH)
    assert (stop[:, :CHARACTERS_PER_SIDE] < 0).any()  # card plays and tuning now see threats too
    assert not ((stop != 0).any(axis=1) & (first_block != 0).any(axis=1)).any()  # one block or the other


def test_hidden_decks_give_no_card_replies(reply_games):
    assert not _reply_columns(reply_games, CARD_START, OUTCOME_WIDTH + len(CARD_SOURCES)).any()


def test_open_decks_give_card_replies():
    games = _play(replies=True, games=1, views=[OPEN, HIDDEN])
    cards = _reply_columns([(decision, encoded) for decision, encoded in games if decision.who == 0], CARD_START, OUTCOME_WIDTH)
    sources = _reply_columns([(decision, encoded) for decision, encoded in games if decision.who == 0], SOURCE_START, len(CARD_SOURCES))
    assert cards[:, -1].any()  # some actions leave the opponent card replies
    assert sources[:, CARD_SOURCES.index("open")].any() and not sources[:, CARD_SOURCES.index("guessed")].any()
    assert not _reply_columns([(decision, encoded) for decision, encoded in games if decision.who == 1], CARD_START, OUTCOME_WIDTH).any()


def test_seats_without_a_view_get_no_replies():
    for decision, encoded in _play(replies=True, games=1, views=[None, HIDDEN])[:400]:
        if decision.who == 0:
            assert decision.replies is None


def test_replies_are_the_opponents_whole_turn(reply_games):
    """Each reply holds the lines of the opponent's turn after the action (theta): my characters' HP change and
    defeat per line, and whether the line needed a card (never with hidden decks)."""
    lines = [line for decision, _ in reply_games for reply in (decision.replies or []) if reply for line in reply.get("lines", [])]
    assert lines
    for line in lines[:2000]:
        assert all(len(pair) == 2 and isinstance(pair[1], bool) for pair in line["mine"])
        assert line["usedCard"] is False
    assert any(change < 0 for line in lines for change, _ in line["mine"])
