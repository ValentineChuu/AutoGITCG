"""Run simulated accounts through the progression, playing every planned match for real in the engine.

Uses random decks and random play for now, so it checks the plumbing (stages, decks, equipment, objectives,
rewards) rather than progress speed. Usage:
    python -m benchmarks.progression_playthrough --accounts 24 --matches 1500 --workers 12
"""

import argparse
import random
import time
from collections import Counter

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.deck_building.random_deck_builder import random_deck
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.policies.random_policy import random_response
from autogitcg.progression.match_result import match_result_from
from autogitcg.progression.shop import Shop
from autogitcg.progression.simulator import ProgressionSimulator
from autogitcg.progression.stage_catalog import build_stage_catalog

PLAYER, NPC = 0, 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accounts", type=int, default=24)
    parser.add_argument("--matches", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=12)
    arguments = parser.parse_args()

    stages, unsupported = build_stage_catalog()
    print(f"{len(stages)} stages supported, {len(unsupported)} left out")
    shop, rng = Shop(), random.Random(0)
    simulators = [ProgressionSimulator(stages, shop, random.Random(seed)) for seed in range(arguments.accounts)]
    running = {}  # game id -> (simulator, planned match, recorder)
    played, failures, kinds = 0, Counter(), Counter()
    started_at = time.perf_counter()

    with EnginePool(arguments.workers) as pool:
        def start_next(simulator: ProgressionSimulator) -> None:
            planned = simulator.next_match()
            if planned is None:
                return
            account = simulator.account
            deck = random_deck(account.owned_characters(), account.owned_action_cards(), rng)
            stage = planned.stage
            recorder = MatchRecorder((deck, stage.enemy_deck), opponent_deck_known=(True, True))
            game_id = pool.create_game(deck, stage.enemy_deck, rng.randrange(2**31), stage.enemy_equipment)
            running[game_id] = (simulator, planned, recorder)

        for simulator in simulators:
            start_next(simulator)
        while pool.has_active_games():
            for event in pool.poll():
                simulator, planned, recorder = running[event.game_id]
                if isinstance(event, Decision):
                    recorder.on_decision(event)
                    response = random_response(event, rng)
                    recorder.on_response(event, response)
                    pool.respond(event, response)
                    continue
                del running[event.game_id]
                if isinstance(event, GameFinished):
                    simulator.record(planned, match_result_from(event, PLAYER, recorder.trackers[PLAYER]))
                    kinds[planned.stage.kind] += 1
                    played += 1
                elif isinstance(event, GameFailed):
                    failures[f"{planned.stage.kind} {planned.stage.level_id}: {event.message.splitlines()[0][:100]}"] += 1
                if played < arguments.matches:
                    start_next(simulator)

    seconds = time.perf_counter() - started_at
    print(f"played {played} matches in {seconds:.0f}s ({played / seconds:.1f}/s), {sum(failures.values())} failed")
    print("by stage kind:", dict(kinds))
    for message, count in failures.most_common(10):
        print(f"  failure x{count}: {message}")
    levels = Counter(simulator.account.level for simulator in simulators)
    characters = sorted(len(simulator.account.owned_characters()) for simulator in simulators)
    print(f"account levels: {dict(sorted(levels.items()))}; characters owned: {characters}")


if __name__ == "__main__":
    main()
