"""Download the community weekly tournament match tables into data/raw/tournaments/ and check every deck decodes
and is legal in our engine.

Usage:
    python scripts/data/fetch_tournament_decks.py --versions 7-0 7-1
    python scripts/data/fetch_tournament_decks.py --versions 4-2 4-3 4-5 4-6 4-7 4-8 5-0 5-1 5-2 5-3 5-4 5-5 5-6 5-8 6-0 6-1 6-2 6-3 6-4 6-5 6-6 6-7 7-0 7-1   # all (4-4, 5-7 have no table)
"""

import argparse
import urllib.error
import urllib.request

from autogitcg.data.tournament_decks import TOURNAMENT_FOLDER, load_tournament_decks, most_played_per_lineup, weekly_table_path, weekly_table_url
from autogitcg.deck_building.deck_validation import deck_problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", nargs="+", default=["7-0", "7-1"])
    arguments = parser.parse_args()
    TOURNAMENT_FOLDER.mkdir(parents=True, exist_ok=True)
    fetched = []
    for version in arguments.versions:
        request = urllib.request.Request(weekly_table_url(version), headers={"User-Agent": "AutoGITCG"})
        try:
            weekly_table_path(version).write_bytes(urllib.request.urlopen(request, timeout=60).read())
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            print(f"version {version}: no tournament table on the site, skipped")
            continue
        fetched.append(version)
    undecodable: list[str] = []
    decks = load_tournament_decks(fetched, undecodable=undecodable)
    print(f"{len(undecodable)} share codes name cards our catalog lacks, left out")
    illegal = [deck for deck in decks if deck_problems(deck.deck)]
    print(f"{len(decks)} distinct decks, {sum(deck.games for deck in decks)} deck-games, {len(illegal)} illegal in our engine")
    for deck in illegal:
        print(f"  illegal: {deck.lineup_name}: {deck_problems(deck.deck)}")
    for deck in most_played_per_lineup(decks)[:15]:
        print(f"  {deck.games:3d} games of this list, {deck.wins / deck.games:.0%} won: {deck.lineup_name}")


if __name__ == "__main__":
    main()
