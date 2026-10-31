"""What an entity does, as catalog facts: its main damage element and amount and its healing, read
from the card code by scripts/data/export_entity_effects.py into data/derived/entity_effects.json. With them the network
sees what a summon or status on the board is worth, and the reaction danger of its element, instead of learning it
only from experience with that one entity. Entries without a file or an effect get zeros."""

import json
from functools import lru_cache
from pathlib import Path

EFFECTS_FILE = Path(__file__).resolve().parents[2] / "data" / "derived" / "entity_effects.json"
DAMAGE_ELEMENTS = ["Physical", "Cryo", "Hydro", "Pyro", "Electro", "Anemo", "Geo", "Dendro", "Piercing"]
EFFECT_WIDTH = len(DAMAGE_ELEMENTS) + 3


@lru_cache(maxsize=1)
def load_entity_effects(path: Path = EFFECTS_FILE) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {int(definition_id): effect for definition_id, effect in json.loads(path.read_text()).items()}


def effect_features(definition_id: int | None) -> list[float]:
    """Damage element (one-hot), damage / 5, "damage computed during play", healing / 5."""
    effect = load_entity_effects().get(definition_id) if definition_id is not None else None
    if not effect:
        return [0.0] * EFFECT_WIDTH
    element = effect.get("element")
    damage, heal = effect.get("damage"), effect.get("heal")
    return ([float(element == name) for name in DAMAGE_ELEMENTS]
            + [(damage or 0) / 5, float(element is not None and damage is None), (heal or 0) / 5])
