"""Write the meta test: blind matchups between the tournament decks a PvP run trains on: a PvP bot plays
today's tournament decks, so it is tested on them (the locked tournament test is 87% decks from before 6.4, which PvP
training leaves out). Run once; refuses to overwrite.

The most played list of each of the 40 most played lineups from FIRST_VERSION on (test lineups left out, as in
training), paired into 100 blind matchups: data/benchmarks/meta_benchmark_v1.json. These decks are training decks,
so this measures play in the current meta, not generalization to unseen decks (that is the tournament test).

    python scripts/data/lock_meta_test.py
"""

import random

from autogitcg.data.tournament_decks import downloaded_versions, load_tournament_decks, most_played_per_lineup
from autogitcg.data.tournament_test_set import training_decks
from autogitcg.training.benchmark_matchups import META_BENCHMARK_FILE, save_benchmark_matchups
from autogitcg.training.deck_pools import version_key
from autogitcg.training.matchup_sampler import Matchup

FIRST_VERSION = "6-4"  # theta's --tournament-from
BENCHMARK_LINEUPS = 40
BENCHMARK_MATCHUPS = 100
BENCHMARK_SEED = 1028


def main() -> None:
    if META_BENCHMARK_FILE.exists():
        raise SystemExit(f"{META_BENCHMARK_FILE} exists; the test is locked (make a v2 file instead of overwriting)")
    versions = [version for version in downloaded_versions() if version_key(version) >= version_key(FIRST_VERSION)]
    lineups = most_played_per_lineup(training_decks(load_tournament_decks(versions)))
    chosen = lineups[:BENCHMARK_LINEUPS]
    print(f"versions {versions[0]}..{versions[-1]}: {len(lineups)} training lineups; the {len(chosen)} most played:")
    for index, deck in enumerate(chosen):
        print(f"  {index:2} {deck.games:3} games  {deck.lineup_name}")
    rng = random.Random(BENCHMARK_SEED)
    matchups = []
    for _ in range(BENCHMARK_MATCHUPS):
        first, second = rng.sample(range(len(chosen)), 2)
        matchups.append(Matchup("meta", (chosen[first].deck, chosen[second].deck), (), (False, False),
                                (f"meta:{first}", f"meta:{second}")))
    save_benchmark_matchups(matchups, META_BENCHMARK_FILE)
    print(f"wrote {len(matchups)} matchups to {META_BENCHMARK_FILE}")


if __name__ == "__main__":
    main()
