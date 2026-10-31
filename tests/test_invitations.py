"""Invitation stages load completely, and Serious Showdown NPCs really start with their talent equipped."""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.invitations import FRIENDLY_FRACAS, SERIOUS_SHOWDOWN, load_invitation_stages
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.policies.random_policy import random_response


def test_every_invitation_has_stages_and_a_talent():
    stages = load_invitation_stages()
    showdowns = [stage for stage in stages if stage.difficulty == SERIOUS_SHOWDOWN]
    assert len(showdowns) == 108
    assert len([stage for stage in stages if stage.difficulty == FRIENDLY_FRACAS]) == 324
    assert all(len(stage.enemy_equipment) == 1 for stage in showdowns)


def test_serious_showdown_talent_is_equipped_and_games_finish():
    rng = random.Random(3)
    showdowns = [stage for stage in load_invitation_stages(npc_side=1) if stage.difficulty == SERIOUS_SHOWDOWN]
    player_decks = list(load_npc_decks().values())
    sample = rng.sample(showdowns, 12)
    expected = {}
    finished, failures, equipped_seen = 0, [], set()
    with EnginePool(worker_count=4) as pool:
        for stage in sample:
            game_id = pool.create_game(rng.choice(player_decks), stage.enemy_deck, rng.randrange(2**31), stage.enemy_equipment)
            expected[game_id] = stage
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    stage = expected[event.game_id]
                    item = stage.enemy_equipment[0]
                    character = event.state["player"][1]["character"][item.character_index]
                    if any(entity["definitionId"] == item.definition_id for entity in character.get("entity", [])):
                        equipped_seen.add(event.game_id)
                    pool.respond(event, random_response(event, rng))
                elif isinstance(event, GameFinished):
                    finished += 1
                elif isinstance(event, GameFailed):
                    failures.append(event.message.splitlines()[0])
    assert not failures, failures
    assert finished == len(sample)
    assert equipped_seen == set(expected)
