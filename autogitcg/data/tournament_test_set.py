"""The locked test set of real tournament decks: lineups that training never sees.

About a fifth of all lineups (chosen by a fixed hash of their character ids, so the choice doesn't depend on
which versions are downloaded) are held out. scripts/data/lock_tournament_test_set.py writes them once to
data/benchmarks/tournament_test_set_v1.json; from then on that file decides, so later downloads can add
training decks without ever moving a lineup between training and test.
"""

import hashlib
import json
from pathlib import Path

from autogitcg.data.deck import Deck
from autogitcg.data.tournament_decks import TournamentDeck

TEST_SET_FILE = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "tournament_test_set_v1.json"
TEST_SHARE_PERCENT = 20

Lineup = tuple[int, int, int]


def lineup_of(deck: Deck) -> Lineup:
    return tuple(sorted(deck.characters))


def hashed_into_test(lineup: Lineup) -> bool:
    digest = hashlib.sha256("-".join(map(str, lineup)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100 < TEST_SHARE_PERCENT


def save_test_set(decks: list[TournamentDeck], path: Path = TEST_SET_FILE) -> None:
    """Every deck of every held-out lineup, most played first."""
    held_out = [deck for deck in decks if hashed_into_test(lineup_of(deck.deck))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"lineup_name": deck.lineup_name, "games": deck.games, "wins": deck.wins,
                                 "characters": list(deck.deck.characters), "cards": list(deck.deck.cards)}
                                for deck in held_out], indent=1))


def load_test_decks(path: Path = TEST_SET_FILE) -> list[TournamentDeck]:
    return [TournamentDeck(Deck(characters=entry["characters"], cards=entry["cards"]), entry["lineup_name"], entry["games"], entry["wins"])
            for entry in json.loads(path.read_text())]


def test_lineups(path: Path = TEST_SET_FILE) -> set[Lineup]:
    """Lineups in the locked file."""
    return {lineup_of(deck.deck) for deck in load_test_decks(path)}


def training_decks(decks: list[TournamentDeck], path: Path = TEST_SET_FILE) -> list[TournamentDeck]:
    """Decks training may use: not a locked test lineup, and not a lineup the hash holds out."""
    locked = test_lineups(path)
    return [deck for deck in decks if lineup_of(deck.deck) not in locked and not hashed_into_test(lineup_of(deck.deck))]
