"""Every source of decks for training games, loaded the same way by the trainer and each actor process."""

import json
from dataclasses import dataclass
from pathlib import Path

from autogitcg.data.deck import Deck
from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.data.tournament_decks import downloaded_versions, load_tournament_decks
from autogitcg.data.tournament_test_set import training_decks
from autogitcg.progression.stage import Stage
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.chudledee_training_decks import ChudledeeTrainingDecks, load_chudledee_training_decks
from autogitcg.training.collection_pool import Collection

SYNERGY_DECK_FOLDER = Path(__file__).resolve().parents[2] / "data" / "decks"


@dataclass
class DeckPools:
    stages: list[Stage]
    npc_decks: dict[int, Deck]  # NPC 3+30 decks by deck id
    collections: list[Collection]  # simulated player collections
    synergy_decks: list[Deck]  # hand-built coherent decks (data/decks/*.json), validated by scripts/data/validate_external_decks.py
    synergy_archetypes: list[str]  # per synergy deck, e.g. "vaporize"
    tournament_decks: list[Deck]  # real weekly tournament decks, minus the locked test lineups (data/tournament_test_set.py)
    chudledee: ChudledeeTrainingDecks  # decks Chudledee built (scripts/deck_builder/build_chudledee_training_decks.py)


def load_synergy_decks(folder: Path = SYNERGY_DECK_FOLDER) -> tuple[list[Deck], list[str]]:
    """(decks, archetype per deck)."""
    decks, archetypes = [], []
    for path in sorted(folder.glob("*.json")):
        for entry in json.loads(path.read_text()):
            decks.append(Deck(characters=entry["characters"], cards=entry["cards"]))
            archetypes.append(entry.get("archetype", "unlabelled"))
    return decks, archetypes


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("-"))


def load_training_tournament_decks(first_version: str | None = None) -> list[Deck]:
    """Every distinct list from the downloaded tournament tables, test lineups left out; with `first_version` (e.g.
    "6-4"), only versions from that one on (older decks mostly hold cards changed since)."""
    versions = [version for version in downloaded_versions() if first_version is None or version_key(version) >= version_key(first_version)]
    return [deck.deck for deck in training_decks(load_tournament_decks(versions))]


def load_deck_pools(collections: list[Collection], tournament_from: str | None = None) -> DeckPools:
    stages, _ = build_stage_catalog()
    return DeckPools(stages, load_npc_decks(), collections, *load_synergy_decks(), load_training_tournament_decks(tournament_from),
                     load_chudledee_training_decks())
