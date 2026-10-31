"""Decks Chudledee built for Chudledum's training (scripts/deck_builder/build_chudledee_training_decks.py).

PvP decks are built for simulated collections of every size; stage decks are built for one stage each (the
builder sees the stage's deck), so stage games can pit the stage against a deck made to beat it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from autogitcg.data.deck import Deck

# The decks the published networks trained with are kept in the repository; rebuilding writes the same file.
CHUDLEDEE_DECKS_FILE = Path(__file__).resolve().parents[2] / "data" / "training" / "chudledee_training_decks.json"


@dataclass
class ChudledeeTrainingDecks:
    pvp: list[Deck] = field(default_factory=list)
    stage: dict[int, list[Deck]] = field(default_factory=dict)  # by stage level id


def save_chudledee_training_decks(decks: ChudledeeTrainingDecks, builder: str, path: Path = CHUDLEDEE_DECKS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"builder": builder, "pvp": [_deck_json(deck) for deck in decks.pvp],
                                "stage": {str(level_id): [_deck_json(deck) for deck in stage_decks] for level_id, stage_decks in decks.stage.items()}}))


def load_chudledee_training_decks(path: Path = CHUDLEDEE_DECKS_FILE) -> ChudledeeTrainingDecks:
    """Empty when the file hasn't been built; training then leaves Chudledee's decks out."""
    if not path.exists():
        return ChudledeeTrainingDecks()
    data = json.loads(path.read_text())
    return ChudledeeTrainingDecks([_deck_from_json(entry) for entry in data["pvp"]],
                                  {int(level_id): [_deck_from_json(entry) for entry in entries] for level_id, entries in data["stage"].items()})


def _deck_json(deck: Deck) -> dict:
    return {"characters": list(deck.characters), "cards": list(deck.cards)}


def _deck_from_json(entry: dict) -> Deck:
    return Deck(characters=entry["characters"], cards=entry["cards"])
