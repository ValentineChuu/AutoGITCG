"""Human-built decks for Chudledee to imitate: real weekly-tournament decks (the training split; newer versions and
more-played decks weigh more), plus, when a playtest log is present (runs/playtest/games.jsonl), the decks people built
themselves to play against the bot.

Weights are sampling weights, not counts: a deck played in G games weighs sqrt(G). Decks from before CURRENT_RULES_SINCE
weigh OLD_RULES_WEIGHT times that: 78-99% of them hold a card changed since (Piovium's old_versions history; from 6.4
on 0-46%, measured), so they teach deck structure more than exact card choices. Tester decks weigh
TESTER_WEIGHT each.
"""

import json
import math
from pathlib import Path

from autogitcg.data.deck import Deck
from autogitcg.data.tournament_decks import TOURNAMENT_FOLDER, load_tournament_decks
from autogitcg.data.tournament_test_set import load_test_decks, training_decks
from autogitcg.deck_building.deck_validation import deck_problems

CURRENT_RULES_SINCE = (6, 4)
OLD_RULES_WEIGHT = 1 / 6
TESTER_WEIGHT = 3.0
PLAYTEST_GAMES = Path("runs/playtest/games.jsonl")


def tournament_versions(folder: Path = TOURNAMENT_FOLDER) -> list[str]:
    """Downloaded versions, oldest first ("4-2" < "4-10" sorts numerically)."""
    versions = [path.stem.removeprefix("weekly_") for path in folder.glob("weekly_*.csv")]
    return sorted(versions, key=_version_key)


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("-"))


def human_example_decks(include_testers: bool = True) -> list[tuple[Deck, float]]:
    """(deck, sampling weight) for every legal training-split tournament deck, plus legal playtester decks."""
    versions = tournament_versions()
    weighted = {}
    for version in versions:
        version_weight = 1.0 if _version_key(version) >= CURRENT_RULES_SINCE else OLD_RULES_WEIGHT
        for entry in training_decks(load_tournament_decks([version])):
            if deck_problems(entry.deck):
                continue
            key = (tuple(entry.deck.characters), tuple(sorted(entry.deck.cards)))
            weight = math.sqrt(entry.games) * version_weight
            weighted[key] = max(weighted.get(key, 0.0), weight)  # the same list in several versions counts once, at its best
    if include_testers and PLAYTEST_GAMES.exists():
        for row in map(json.loads, PLAYTEST_GAMES.open()):
            deck = Deck(row["deck"]["characters"], row["deck"]["cards"])
            if row.get("deck_source") == "player" and not deck_problems(deck):
                weighted[(tuple(deck.characters), tuple(sorted(deck.cards)))] = TESTER_WEIGHT
    return [(Deck(list(characters), list(cards)), weight) for (characters, cards), weight in weighted.items()]


def held_out_human_decks() -> list[Deck]:
    """Locked-test tournament decks: never imitated, used to check that imitation generalizes."""
    return [entry.deck for entry in load_test_decks() if not deck_problems(entry.deck)]
