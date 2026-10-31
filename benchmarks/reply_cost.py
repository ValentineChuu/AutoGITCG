"""Engine cost of the selector's extra inputs per decision (theta step 6): blind games on test tournament decks with
replies on demand, 4 random checked moves per decision and every move on 15% of decisions (the training setting), no
network (random answers), so only the engine work is measured. --worker points at another bridge folder to compare
versions (e.g. a copy of the bridge from before follow-ups and whole-turn replies).

    python -m benchmarks.reply_cost --workers 8 --games 24
"""

import argparse
import random
import time
from pathlib import Path

import autogitcg.bridge.engine_worker as engine_worker
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, RepliesReady
from autogitcg.bridge.reply_views import reply_views
from autogitcg.deck_guessing.reply_guess_candidates import ReplyGuessCandidates
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.policies.random_policy import random_response
from autogitcg.policies.reply_selection import on_demand
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--checks", type=int, default=4)
    parser.add_argument("--full-share", type=float, default=0.15)
    parser.add_argument("--worker", type=Path, help="another bridge's worker.mjs")
    arguments = parser.parse_args()
    if arguments.worker:
        engine_worker.WORKER_SCRIPT = arguments.worker.resolve()
    rng = random.Random(7)
    every = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)
    matchups = [every[number % len(every)] for number in range(arguments.games)]  # cycled, each with its own seed
    guesses = ReplyGuessCandidates(TournamentDeckGuesses(4, 7, excluded=[deck for matchup in matchups for deck in matchup.decks]).decks)
    waiting, decisions, checked_moves = {}, 0, 0
    started = time.perf_counter()
    with EnginePool(arguments.workers, replies=True) as pool:
        for number, matchup in enumerate(matchups):
            views = [on_demand(view) for view in reply_views((True, True), (False, False), matchup.decks, guesses)]
            pool.create_game(*matchup.decks, seed=1000 + number, reply_views=views)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, RepliesReady):
                    decision = waiting.pop((event.game_id, event.who))
                    pool.respond(decision, random_response(decision, rng))
                elif isinstance(event, Decision):
                    decisions += 1
                    if event.replies_on_demand:
                        valid = [index for index, action in enumerate(event.request["action"]) if action.get("validity", 0) == 0]
                        picked = valid if rng.random() < arguments.full_share else rng.sample(valid, min(arguments.checks, len(valid)))
                        checked_moves += len(picked)
                        waiting[(event.game_id, event.who)] = event
                        pool.request_replies(event, picked)
                        continue
                    pool.respond(event, random_response(event, rng))
    seconds = time.perf_counter() - started
    print(f"{decisions} decisions in {seconds:.0f} s on {arguments.workers} workers: {decisions / seconds:.1f} decisions/s, "
          f"{decisions / seconds / arguments.workers:.2f} per worker; {checked_moves} moves checked")


if __name__ == "__main__":
    main()
