"""Load NPC decks from Dimbreath's GCGDeckExcelConfigData."""

import json
from pathlib import Path

from autogitcg.data.deck import Deck

DECK_TABLE = Path(__file__).resolve().parents[2] / "data" / "raw" / "GCGDeckExcelConfigData.json"


def load_npc_decks(path: Path = DECK_TABLE) -> dict[int, Deck]:
    """Return standard-shaped decks (3 characters, 30 cards) keyed by deck id."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["id"]: Deck(characters=entry["characterList"], cards=entry["cardList"])
        for entry in entries
        if len(entry.get("characterList", [])) == 3 and len(entry.get("cardList", [])) == 30  # fields may be omitted when empty
    }
