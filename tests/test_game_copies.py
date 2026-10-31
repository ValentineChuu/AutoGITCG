"""Game copies for look-ahead search (bridge/game_session.mjs copy(), the onStepStart core patch)."""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.policies.random_policy import random_response


def test_copies_at_action_decisions_replay_and_play_to_the_end():
    rng = random.Random(0)
    decks = list(load_npc_decks().values())[:4]
    copy_ids, copy_outcomes = set(), []
    with EnginePool(worker_count=2) as pool:
        for seed in range(4):
            pool.create_game(decks[seed], decks[(seed + 1) % 4], seed)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    response = random_response(event, rng)
                    if event.game_id not in copy_ids and event.kind == "action" and len(copy_ids) < 24:
                        copy_id = pool.copy_game(event, response, seed=len(copy_ids))
                        copy_ids.add(copy_id)
                    pool.respond(event, response)
                elif event.game_id in copy_ids:
                    copy_outcomes.append(event)
    assert len(copy_ids) == 24
    failures = [outcome.message[:300] for outcome in copy_outcomes if isinstance(outcome, GameFailed)]
    assert not failures, failures
    assert all(isinstance(outcome, GameFinished) for outcome in copy_outcomes)


def test_a_game_replayed_on_another_worker_gives_identical_copies():
    """ground_truth/position_mirrors.py: copies of a mirror must play exactly like copies of the original."""
    decks = list(load_npc_decks().values())[:2]
    with EnginePool(worker_count=2) as pool:
        original = pool.create_game(decks[0], decks[1], seed=5, worker_index=0)
        rng, responses, held = random.Random(1), {0: [], 1: []}, None
        while held is None:
            for event in pool.poll():
                if isinstance(event, Decision):
                    if event.kind == "action" and len(responses[0]) + len(responses[1]) >= 8:
                        held = event
                        break
                    response = random_response(event, rng)
                    responses[event.who].append(response)
                    pool.respond(event, response)
        mirror_id = pool.create_game(decks[0], decks[1], seed=5, worker_index=1)
        remaining, mirror_held = {who: list(moves) for who, moves in responses.items()}, None
        while mirror_held is None:
            for event in pool.poll():
                if isinstance(event, Decision) and event.game_id == mirror_id:
                    if remaining[event.who]:
                        pool.respond(event, remaining[event.who].pop(0))
                    else:
                        mirror_held = event
        assert (mirror_held.who, mirror_held.request) == (held.who, held.request)
        first_move = random_response(held, random.Random(2))
        copies = {pool.copy_game(held, first_move, seed=9): random.Random(3),
                  pool.copy_game(mirror_held, first_move, seed=9): random.Random(3)}
        endings = {}
        while len(endings) < 2:
            for event in pool.poll():
                if event.game_id not in copies:
                    continue
                if isinstance(event, Decision):
                    pool.respond(event, random_response(event, copies[event.game_id]))
                else:
                    endings[event.game_id] = (type(event).__name__, getattr(event, "winner", None), getattr(event, "rounds", None))
        assert len(set(endings.values())) == 1, endings
        assert pool.worker_index_of(original) == 0 and pool.worker_index_of(mirror_id) == 1
