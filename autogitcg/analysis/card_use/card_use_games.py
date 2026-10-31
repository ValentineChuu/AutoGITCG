"""Plays blind PvP games between labelled decks, counting card use per label. Each label has its own player (the same
one for every label, or different networks to compare them)."""

import random
from collections import Counter, defaultdict

from autogitcg.analysis.card_use.card_use_stats import record_action, record_game
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFinished
from autogitcg.data.deck import Deck
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder

GAMES_IN_FLIGHT = 24


def play_card_use_games(pairings: list[tuple[tuple[Deck, str], tuple[Deck, str]]], players: dict, encoder: ObservationEncoder,
                        pool: EnginePool, seed: int, on_finished=lambda: None) -> dict[str, Counter]:
    """pairings: ((deck, label), (deck, label)) per game, seats in that order. Failed games are dropped."""
    rng = random.Random(seed)
    stats: dict[str, Counter] = defaultdict(Counter)
    queue = list(reversed(pairings))
    active = {}  # game id -> (labels by seat, recorder)
    while queue or active:
        while queue and len(active) < GAMES_IN_FLIGHT:
            (first, first_label), (second, second_label) = queue.pop()
            game_id = pool.create_game(first, second, seed=rng.randrange(2**31))
            active[game_id] = ((first_label, second_label), MatchRecorder((first, second), (False, False)))
        pending = []
        for event in pool.poll(timeout=1):
            if isinstance(event, Decision):
                pending.append((event, encoder.encode(event, active[event.game_id][1].on_decision(event))))
                continue
            labels, _ = active.pop(event.game_id)
            if isinstance(event, GameFinished):
                for seat, label in enumerate(labels):
                    record_game(stats[label], event.winner == seat, event.rounds)
            on_finished()
        by_label = defaultdict(list)
        for item in pending:
            by_label[active[item[0].game_id][0][item[0].who]].append(item)
        for label, items in by_label.items():
            for (decision, _), response in zip(items, players[label].respond(items)):
                labels, recorder = active[decision.game_id]
                recorder.on_response(decision, response)
                if decision.kind == "action":
                    record_action(stats[label], decision.request["action"], response["chosenActionIndex"],
                                  decision.state["player"][decision.who])
                pool.respond(decision, response)
    return stats
