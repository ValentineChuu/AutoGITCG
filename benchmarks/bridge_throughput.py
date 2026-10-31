"""Measure end-to-end throughput of Python driving Node engine workers with random decisions.

Usage: python -m benchmarks.bridge_throughput --workers 12 --games-per-worker 8 --total-games 600
"""

import argparse
import random
import time
from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.policies.random_policy import random_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--games-per-worker", type=int, default=8)
    parser.add_argument("--total-games", type=int, default=600)
    parser.add_argument("--previews", action="store_true", help="games carry action previews (network input)")
    parser.add_argument("--replies", action="store_true", help="games also carry the opponent's replies to each action (network input)")
    arguments = parser.parse_args()

    rng = random.Random(0)
    decks = list(load_npc_decks().values())
    outcomes, failures = Counter(), Counter()
    decisions = rounds = started = 0

    with EnginePool(arguments.workers, previews=arguments.previews, replies=arguments.replies) as pool:
        def start_game() -> None:
            nonlocal started
            deck0, deck1 = rng.sample(decks, 2)
            pool.create_game(deck0, deck1, seed=rng.randrange(2**31))
            started += 1

        for _ in range(min(arguments.total_games, arguments.workers * arguments.games_per_worker)):
            start_game()
        started_at = time.perf_counter()
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    decisions += 1
                    pool.respond(event, random_response(event, rng))
                    continue
                if isinstance(event, GameFinished):
                    outcomes[event.winner] += 1
                    rounds += event.rounds
                elif isinstance(event, GameFailed):
                    failures[event.message.splitlines()[0][:100]] += 1
                if started < arguments.total_games:
                    start_game()
        seconds = time.perf_counter() - started_at

    finished = sum(outcomes.values())
    print(f"workers={arguments.workers} concurrent/worker={arguments.games_per_worker}")
    print(f"finished {finished}, failed {sum(failures.values())} in {seconds:.1f}s")
    print(f"games/s {finished / seconds:.1f}  decisions/s {decisions / seconds:.0f}  avg rounds {rounds / max(finished, 1):.2f}")
    print(f"winners: {dict(outcomes)}")
    for message, count in failures.most_common(5):
        print(f"  failure x{count}: {message}")


if __name__ == "__main__":
    main()
