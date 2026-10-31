"""Write the locked tournament test set and the tournament benchmark built from it. Run once; refuses to overwrite.

The benchmark: the most played list of the 40 most played held-out lineups, paired into 100 blind PvP matchups
(data/benchmarks/tournament_benchmark_v1.json). Training plays it against frozen alpha 2.40 (scripts/training/train.py).

Usage:
    python scripts/data/fetch_tournament_decks.py --versions ...   # all versions first
    python scripts/data/lock_tournament_test_set.py
"""

import random

from autogitcg.data.tournament_decks import downloaded_versions, load_tournament_decks, most_played_per_lineup
from autogitcg.data.tournament_test_set import TEST_SET_FILE, load_test_decks, save_test_set
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, save_benchmark_matchups
from autogitcg.training.matchup_sampler import Matchup

BENCHMARK_LINEUPS = 40
BENCHMARK_MATCHUPS = 100
BENCHMARK_SEED = 777


def main() -> None:
    for path in (TEST_SET_FILE, TOURNAMENT_BENCHMARK_FILE):
        if path.exists():
            raise SystemExit(f"{path} exists; the test set is locked (make a v2 file instead of overwriting)")
    versions = downloaded_versions()
    decks = load_tournament_decks(versions)
    save_test_set(decks)
    held_out = load_test_decks()
    lineups = most_played_per_lineup(held_out)
    print(f"versions {versions[0]}..{versions[-1]}: {len(decks)} decks; held out {len(held_out)} decks in {len(lineups)} lineups")

    rng = random.Random(BENCHMARK_SEED)
    chosen = lineups[:BENCHMARK_LINEUPS]
    matchups = []
    for _ in range(BENCHMARK_MATCHUPS):
        first, second = rng.sample(range(len(chosen)), 2)
        matchups.append(Matchup("tournament_test", (chosen[first].deck, chosen[second].deck), (), (False, False),
                                (f"tournament_test:{first}", f"tournament_test:{second}")))
    save_benchmark_matchups(matchups, TOURNAMENT_BENCHMARK_FILE)
    for index, deck in enumerate(chosen[:10]):
        print(f"  {index}: {deck.lineup_name} ({deck.games} games of this list)")
    print(f"wrote {TEST_SET_FILE} and {TOURNAMENT_BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
