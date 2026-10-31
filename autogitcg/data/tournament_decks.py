"""Real tournament decks from the GITCG community weekly tournaments (gitcg-community.vercel.app).

Each game version has one match table (downloaded by scripts/data/fetch_tournament_decks.py into data/raw/tournaments/):
one row per game with both players' lineups, deck share codes and scores. Decks are grouped by share code with
how often they were played and won.
"""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from autogitcg.data.deck import Deck
from autogitcg.data.share_code import decode_share_code

TOURNAMENT_FOLDER = Path(__file__).resolve().parents[2] / "data" / "raw" / "tournaments"
SITE = "https://gitcg-community.vercel.app"


@dataclass(frozen=True)
class TournamentDeck:
    deck: Deck
    lineup_name: str  # e.g. "Goldflame Qucusaur Tyrant / Lyney / Yae Miko"
    games: int
    wins: int


def weekly_table_url(version: str) -> str:
    return f"{SITE}/weekly/{version}/matchdata.csv"


def weekly_table_path(version: str, folder: Path = TOURNAMENT_FOLDER) -> Path:
    return folder / f"weekly_{version}.csv"


def downloaded_versions(folder: Path = TOURNAMENT_FOLDER) -> list[str]:
    """Versions whose match table is in the folder, oldest first (e.g. ["4-2", ..., "7-1"])."""
    versions = [path.stem.removeprefix("weekly_") for path in folder.glob("weekly_*.csv")]
    return sorted(versions, key=lambda version: tuple(int(part) for part in version.split("-")))


def load_tournament_decks(versions: list[str], folder: Path = TOURNAMENT_FOLDER, undecodable: list[str] | None = None) -> list[TournamentDeck]:
    """Every deck played in counted games of these versions, most played first. Share codes naming cards our
    catalog lacks (older versions) are left out, and listed in `undecodable` when given."""
    games, wins, lineup_names = Counter(), Counter(), {}
    for version in versions:
        with weekly_table_path(version, folder).open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["isIncluded"] != "TRUE" or row["isBye"] == "TRUE":
                    continue
                for side in "12":
                    code = row[f"deckcode{side}"]
                    if not code:
                        continue
                    games[code] += 1
                    wins[code] += row[f"score{side}"] == "1"
                    lineup_names[code] = row[f"characters{side}"].replace("\\\\", " / ")
    decks = []
    for code, count in games.most_common():
        try:
            deck = decode_share_code(code)
        except KeyError:
            if undecodable is not None:
                undecodable.append(code)
            continue
        decks.append(TournamentDeck(deck, lineup_names[code], count, wins[code]))
    return decks


def most_played_per_lineup(decks: list[TournamentDeck]) -> list[TournamentDeck]:
    """One deck per lineup (its most played list), lineups ordered by total games."""
    lineup_games = Counter()
    best: dict[tuple, TournamentDeck] = {}
    for deck in decks:
        lineup = tuple(sorted(deck.deck.characters))
        lineup_games[lineup] += deck.games
        if lineup not in best or deck.games > best[lineup].games:
            best[lineup] = deck
    return [best[lineup] for lineup, _ in lineup_games.most_common()]
