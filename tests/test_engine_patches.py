"""Behavior our core patches must keep (engine/reference/patches/apply_core_patches.mjs)."""

import random

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.initial_equipment import InitialEquipment
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.deck import Deck
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.policies.random_policy import random_response

LISA = 1409
PULSATING_WITCH = 214091  # Lisa's talent: on switching to Lisa, the opponent's active character gets Conductive
CONDUCTIVE_LISA = 114091


def test_pre_equipped_talent_reacting_to_the_first_switch_in_seat_zero():
    """Seat 0's start-of-game switch used to be handled before seat 1 had an active character."""
    rng = random.Random(0)
    lisa_deck = Deck(characters=[LISA, 1101, 1201], cards=[])
    opponent = next(iter(load_npc_decks().values()))
    outcomes, conductive_seen = [], False
    with EnginePool(worker_count=2) as pool:
        for seed in range(6):
            pool.create_game(lisa_deck, opponent, seed, (InitialEquipment(who=0, character_index=0, definition_id=PULSATING_WITCH),))
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    if event.kind == "chooseActive" and event.who == 0 and event.state["roundNumber"] == 0:
                        lisa_id = next(character["id"] for character in event.state["player"][0]["character"]
                                       if character["definitionId"] == LISA)
                        pool.respond(event, {"activeCharacterId": lisa_id})
                        continue
                    opponent_active = event.state["player"][1].get("activeCharacterId")
                    for character in event.state["player"][1].get("character", []):
                        if character["id"] == opponent_active:
                            conductive_seen |= any(entity.get("definitionId") == CONDUCTIVE_LISA for entity in character.get("entity", []))
                    pool.respond(event, random_response(event, rng))
                elif isinstance(event, GameFinished | GameFailed):
                    outcomes.append(event)
    assert all(isinstance(outcome, GameFinished) for outcome in outcomes), [getattr(outcome, "message", "")[:200] for outcome in outcomes]
    assert conductive_seen
