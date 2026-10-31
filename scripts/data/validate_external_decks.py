"""Validate a JSON file of externally built decks: deck rules first, then real engine games.

Each legal deck plays heuristic-vs-heuristic games against random NPC decks, in both seats. Engine failures
are reported, along with each deck's win rate there as a rough strength signal.

Input: a list of {"name", "archetype", "characters": [3 ids], "cards": [30 ids], "plan"}.

Usage:
    python scripts/data/validate_external_decks.py data/decks/synergy_decks.json --games 10
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.data.deck import Deck
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.policies.heuristic_policy import HeuristicPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("decks", type=Path)
    parser.add_argument("--games", type=int, default=10, help="games per legal deck (seats alternate)")
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    entries = json.loads(arguments.decks.read_text())
    legal = []
    for number, entry in enumerate(entries):
        problems = deck_problems(Deck(characters=entry["characters"], cards=entry["cards"]))
        if problems:
            print(f"ILLEGAL #{number} {entry['name']} ({entry['archetype']}):")
            for problem in problems:
                print(f"    {problem}")
        else:
            legal.append((number, entry))
    print(f"{len(legal)} of {len(entries)} decks are legal")

    rng = random.Random(0)
    heuristic = HeuristicPolicy()
    opponents = list(load_npc_decks().values())
    game_owner: dict[int, tuple[int, int]] = {}  # game id -> (deck number, seat of that deck)
    wins, played, failures = Counter(), Counter(), {}
    with EnginePool(arguments.workers) as pool:
        for number, entry in legal:
            deck = Deck(characters=entry["characters"], cards=entry["cards"])
            for game in range(arguments.games):
                seat = game % 2
                decks = (deck, rng.choice(opponents)) if seat == 0 else (rng.choice(opponents), deck)
                game_owner[pool.create_game(*decks, seed=rng.randrange(2**31))] = (number, seat)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    pool.respond(event, heuristic.respond(event, rng))
                elif isinstance(event, GameFinished):
                    number, seat = game_owner[event.game_id]
                    played[number] += 1
                    wins[number] += event.winner == seat
                elif isinstance(event, GameFailed):
                    number, _ = game_owner[event.game_id]
                    failures.setdefault(number, event.message.splitlines()[0][:150])

    for number, entry in legal:
        status = f"ENGINE FAILURE: {failures[number]}" if number in failures else ""
        rate = wins[number] / played[number] if played[number] else float("nan")
        print(f"#{number:2} {entry['archetype']:20} {entry['name'][:40]:40} win {rate:.2f} ({played[number]} games) {status}")
    print(f"engine failures in {len(failures)} decks")


if __name__ == "__main__":
    main()
