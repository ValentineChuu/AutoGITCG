"""Every PvE stage plays to the end once with its real opponent lineup and equipment.

This catches custom-kit bugs that single-character trials miss (e.g. two copies of a monster on one team).
Takes about a minute.
"""

import random
from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.policies.random_policy import random_response
from autogitcg.progression.stage_catalog import build_stage_catalog


def test_every_stage_plays_to_the_end():
    stages, unsupported = build_stage_catalog()
    assert not unsupported
    rng = random.Random(4)
    player_decks = list(load_npc_decks().values())
    stage_by_game, failures = {}, Counter()
    with EnginePool(worker_count=12) as pool:
        for stage in stages:
            game_id = pool.create_game(rng.choice(player_decks), stage.enemy_deck, rng.randrange(2**31), stage.enemy_equipment)
            stage_by_game[game_id] = stage
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    pool.respond(event, random_response(event, rng))
                elif isinstance(event, GameFailed):
                    stage = stage_by_game[event.game_id]
                    failures[f"{stage.kind} {stage.level_id} {stage.name}: {event.message.splitlines()[0][:100]}"] += 1
    assert not failures, failures
