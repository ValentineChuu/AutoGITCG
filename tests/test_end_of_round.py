"""The end-of-round input (engine/reference/bridge/end_of_round.mjs, observation/encode_end_of_round.py): when a player really
ends the round after the opponent already has, the end phase follows at once, so its prediction should match what
actually happens to every character. Nearly always: the copy re-deals what the deciding player can't see (the
opponent's hand and dice), so an end-phase effect that depends on those can differ (1 character in 186 round ends
when measured)."""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.bridge.reply_views import OPEN
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.observation.encode_end_of_round import END_OF_ROUND_WIDTH, end_of_round_features
from autogitcg.policies.random_policy import random_response


def _health(state: dict) -> list[list[tuple[int, bool]]]:
    return [[(character["health"], character["defeated"]) for character in player["character"]] for player in state["player"]]


def test_end_of_round_matches_the_real_end_phase():
    rng = random.Random(4)
    decks = list(load_npc_decks().values())
    checked = with_input = characters = mismatched = 0
    with EnginePool(worker_count=2, replies=True) as pool:
        for game in range(6):
            pool.create_game(*rng.sample(decks, 2), seed=100 + game, reply_views=[OPEN, OPEN])
        pending = {}  # game id -> (predicted change, health before, round)
        while pool.has_active_games():
            for event in pool.poll():
                if not isinstance(event, Decision):
                    continue
                if event.game_id in pending and event.state["roundNumber"] > pending[event.game_id][2]:
                    predicted, before, _ = pending.pop(event.game_id)
                    after = _health(event.state)
                    for seat in (0, 1):
                        for index, (health, defeated) in enumerate(before[seat]):
                            expected = predicted[seat][index]
                            characters += 1
                            mismatched += (after[seat][index][0] - health != expected["health"]
                                           or (after[seat][index][1] and not defeated) != expected["defeated"])
                    checked += 1
                response = random_response(event, rng)
                if event.end_of_round:
                    with_input += 1
                    assert len(end_of_round_features(event.end_of_round, event.who)) == END_OF_ROUND_WIDTH
                    opponent = event.state["player"][1 - event.who]
                    chosen = event.request["action"][response["chosenActionIndex"]]
                    if opponent.get("declaredEnd") and chosen["action"]["$case"] == "declareEnd":
                        pending[event.game_id] = (event.end_of_round["change"], _health(event.state), event.state["roundNumber"])
                pool.respond(event, response)
    assert with_input > 50 and checked >= 5
    assert mismatched <= 0.03 * characters
