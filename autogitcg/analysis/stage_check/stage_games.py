"""Plays NPC stage games for given (stage, deck) pairs, as the progression playthrough does: one player answers both
sides, the deck's side knows the stage deck, seats are random. Failed games are replayed."""

import random
from collections.abc import Callable

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.deck import Deck
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.progression.stage import Stage
from autogitcg.training.matchup_sampler import Matchup, swapped

GAMES_IN_FLIGHT = 24
MAX_FAILURES_PER_GAME = 3


def play_stage_games(games: list[tuple[Stage, Deck]], player, encoder: ObservationEncoder, pool: EnginePool, seed: int,
                     on_result: Callable[[Stage, bool, int], None]) -> None:
    """on_result(stage, deck side won, rounds) once per finished game, in finishing order."""
    rng = random.Random(seed)
    queue = [(stage, deck, 0) for stage, deck in reversed(games)]
    active = {}  # game id -> (stage, deck, failures, deck side seat, recorder)
    while queue or active:
        while queue and len(active) < GAMES_IN_FLIGHT:
            stage, deck, failures = queue.pop()
            matchup = Matchup("stage", (deck, stage.enemy_deck), stage.enemy_equipment, (True, False), ("deck", f"stage:{stage.level_id}"))
            seat = rng.randrange(2)
            if seat:
                matchup = swapped(matchup)
            game_id = pool.create_game(*matchup.decks, seed=rng.randrange(2**31), equipment=matchup.equipment)
            active[game_id] = (stage, deck, failures, seat, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
        pending = []
        for event in pool.poll(timeout=1):
            stage, deck, failures, seat, recorder = active[event.game_id]
            if isinstance(event, Decision):
                pending.append((event, encoder.encode(event, recorder.on_decision(event))))
                continue
            del active[event.game_id]
            if isinstance(event, GameFinished):
                on_result(stage, event.winner == seat, event.rounds)
            elif isinstance(event, GameFailed) and failures + 1 < MAX_FAILURES_PER_GAME:
                queue.append((stage, deck, failures + 1))
        for (decision, _), response in zip(pending, player.respond(pending)):
            active[decision.game_id][4].on_response(decision, response)
            pool.respond(decision, response)
