"""Plays check games with one player on both seats.

Every matchup is played as given and with seats swapped; the result is reported for the deck in seat 0 of the
matchup as given (its focus deck).
"""

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.training.matchup_sampler import Matchup, swapped


def play_matchups(pool: EnginePool, encoder: ObservationEncoder, player, matchups: list[tuple[str, Matchup]],
                  repeats: int, seed: int, concurrent_games: int = 96) -> list[dict]:
    """matchups: (label, matchup). One row per game: label, focus_won, draw, rounds, or failure."""
    queue = [(label, swapped(matchup) if focus_seat else matchup, focus_seat, seed + number * 2 * repeats + repeat * 2 + focus_seat)
             for number, (label, matchup) in enumerate(matchups)
             for repeat in range(repeats) for focus_seat in (0, 1)]
    queue.reverse()
    active, rows = {}, []
    while queue or active:
        while queue and len(active) < concurrent_games:
            label, matchup, focus_seat, game_seed = queue.pop()
            game_id = pool.create_game(*matchup.decks, seed=game_seed, equipment=matchup.equipment)
            active[game_id] = ({"label": label, "focus_seat": focus_seat}, MatchRecorder(matchup.decks, matchup.opponent_deck_known))
        pending = []
        for event in pool.poll(timeout=1):
            row, recorder = active[event.game_id]
            if isinstance(event, Decision):
                tracker = recorder.on_decision(event)
                pending.append((event, encoder.encode(event, tracker) if player.needs_encoding else None))
            elif isinstance(event, (GameFinished, GameFailed)):
                if isinstance(event, GameFinished):
                    row.update(focus_won=event.winner == row["focus_seat"], draw=event.winner is None, rounds=event.rounds)
                else:
                    row.update(failure=event.message.splitlines()[0][:200])
                rows.append(row)
                del active[event.game_id]
        for (decision, _), response in zip(pending, player.respond(pending)):
            active[decision.game_id][1].on_response(decision, response)
            pool.respond(decision, response)
    return rows
