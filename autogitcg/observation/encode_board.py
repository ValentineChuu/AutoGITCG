"""Board tokens: up to 4 characters per side (mine first, in slot order) followed by every entity on the field.

Most lineups have 3 characters; some PvE opponents have 1, 2 or 4. Unused character slots are masked.

Also returns a map from engine object id to board token position, so actions can point at targets.
"""

import math
from dataclasses import dataclass

import numpy as np

from autogitcg.observation.catalog_features import DESCRIPTIVE_VARIABLES
from autogitcg.observation.enums import (
    AURA_ELEMENT_CODES, CHARACTER_TAG_BITS, ELEMENTS, ENTITY_TAG_BITS, ENTITY_TYPE_COMBAT_STATUS,
    ENTITY_TYPE_EQUIPMENT, ENTITY_TYPE_STATUS, ENTITY_TYPE_SUMMON, ENTITY_TYPE_SUPPORT, EQUIPMENT_TYPE_COUNT,
)
from autogitcg.observation.vocabulary import Vocabulary

CHARACTERS_PER_SIDE = 4
CHARACTER_COUNT = 2 * CHARACTERS_PER_SIDE
MAX_ENTITIES = 64
ENTITY_KIND_CODES = [ENTITY_TYPE_STATUS, ENTITY_TYPE_COMBAT_STATUS, ENTITY_TYPE_SUMMON, ENTITY_TYPE_SUPPORT, ENTITY_TYPE_EQUIPMENT]
HINT_ICON_COUNT = 10  # DamageType 0-9: the element a summon deals, shown on its card



@dataclass
class BoardTokens:
    character_indices: np.ndarray  # (CHARACTER_COUNT,) vocabulary indices
    character_features: np.ndarray  # (CHARACTER_COUNT, CHARACTER_WIDTH)
    character_mask: np.ndarray  # (CHARACTER_COUNT,) 1 for real characters
    entity_indices: np.ndarray  # (MAX_ENTITIES,)
    entity_features: np.ndarray  # (MAX_ENTITIES, ENTITY_WIDTH)
    entity_mask: np.ndarray  # (MAX_ENTITIES,) 1 for real tokens
    position_by_object_id: dict[int, int]  # characters 0..CHARACTER_COUNT-1, entities after
    dropped_entities: int  # entities beyond MAX_ENTITIES (should stay 0)


def _aura_elements(aura: int) -> list[float]:
    elements = [0.0] * len(ELEMENTS)
    for code in (aura & 0xF, aura >> 4):
        if code in AURA_ELEMENT_CODES:
            elements[AURA_ELEMENT_CODES[code]] = 1.0
    return elements


def _character_features(character: dict, is_mine: bool, slot: int, is_active: bool) -> list[float]:
    slot_one_hot = [1.0 if slot == index else 0.0 for index in range(CHARACTERS_PER_SIDE)]
    max_energy = character.get("maxEnergy", 0)
    tags = character.get("tags", 0)
    return (
        [float(is_mine)] + slot_one_hot
        + [float(is_active), float(character.get("defeated", False))]
        + [character.get("health", 0) / 10, character.get("maxHealth", 0) / 10,
           character.get("energy", 0) / 3, max_energy / 3,
           float(max_energy > 0 and character.get("energy", 0) >= max_energy),
           float(bool(character.get("specialEnergyName")))]
        + _aura_elements(character.get("aura", 0))
        + [float(bool(tags & bit)) for bit in CHARACTER_TAG_BITS]
    )


def _entity_features(entity: dict, is_mine: bool, attached_slot: int | None, attached_to_active: bool, order: int,
                     counters: dict[str, int]) -> list[float]:
    """order: position within its zone (effects resolve in order of entry); counters: every counter the client displays."""
    kind = [1.0 if entity.get("type") == code else 0.0 for code in ENTITY_KIND_CODES]
    slot = [1.0 if attached_slot == index else 0.0 for index in range(CHARACTERS_PER_SIDE)]
    has_variable = "variableValue" in entity
    variable_name = entity.get("variableName")
    variable_one_hot = [1.0 if variable_name == name else 0.0 for name in DESCRIPTIVE_VARIABLES]
    variable_one_hot.append(float(has_variable and variable_name not in DESCRIPTIVE_VARIABLES))
    equipment = [1.0 if entity.get("equipment") == code else 0.0 for code in range(EQUIPMENT_TYPE_COUNT)]
    hint_icon = entity.get("hintIcon")
    hint = [1.0 if hint_icon == code else 0.0 for code in range(HINT_ICON_COUNT)] + [float(hint_icon is not None)]
    tags = entity.get("tags", 0)
    return (
        [float(is_mine)] + kind + slot + [float(attached_to_active), order / 8]
        + [float(has_variable), math.log1p(max(entity.get("variableValue", 0), 0))]
        + variable_one_hot
        + [float(entity.get("hasUsagePerRound", False))]
        + [math.log1p(max(counters.get(name, 0), 0)) for name in DESCRIPTIVE_VARIABLES]
        + [float(name in counters) for name in DESCRIPTIVE_VARIABLES]
        + equipment
        + [float(bool(tags & bit)) for bit in ENTITY_TAG_BITS]
        + hint
    )


CHARACTER_WIDTH = len(_character_features({}, False, 0, False))
ENTITY_WIDTH = len(_entity_features({}, False, None, False, 0, {}))


def encode_board(state: dict, who: int, vocabulary: Vocabulary, counters_by_id: dict[int, dict[str, int]]) -> BoardTokens:
    character_indices = np.zeros(CHARACTER_COUNT, dtype=np.int64)
    character_features = np.zeros((CHARACTER_COUNT, CHARACTER_WIDTH), dtype=np.float32)
    character_mask = np.zeros(CHARACTER_COUNT, dtype=np.float32)
    entity_rows: list[tuple[int, list[float], int]] = []  # (vocabulary index, features, object id)
    position_by_object_id: dict[int, int] = {}

    for side_offset, player_index in enumerate((who, 1 - who)):
        player = state["player"][player_index]
        is_mine = side_offset == 0
        active_id = player.get("activeCharacterId")
        for slot, character in enumerate(player.get("character", [])[:CHARACTERS_PER_SIDE]):
            position = side_offset * CHARACTERS_PER_SIDE + slot
            character_mask[position] = 1.0
            character_indices[position] = vocabulary.index("character", character.get("definitionId"))
            character_features[position] = _character_features(character, is_mine, slot, character["id"] == active_id)
            position_by_object_id[character["id"]] = position
            for order, entity in enumerate(character.get("entity", [])):
                entity_rows.append((vocabulary.index("entity", entity.get("definitionId")),
                                    _entity_features(entity, is_mine, slot, character["id"] == active_id, order, counters_by_id.get(entity["id"], {})), entity["id"]))
        for area in ("combatStatus", "summon", "support"):
            for order, entity in enumerate(player.get(area, [])):
                entity_rows.append((vocabulary.index("entity", entity.get("definitionId")),
                                    _entity_features(entity, is_mine, None, False, order, counters_by_id.get(entity["id"], {})), entity["id"]))

    entity_indices = np.zeros(MAX_ENTITIES, dtype=np.int64)
    entity_features = np.zeros((MAX_ENTITIES, ENTITY_WIDTH), dtype=np.float32)
    entity_mask = np.zeros(MAX_ENTITIES, dtype=np.float32)
    for position, (index, features, object_id) in enumerate(entity_rows[:MAX_ENTITIES]):
        entity_indices[position] = index
        entity_features[position] = features
        entity_mask[position] = 1.0
        position_by_object_id[object_id] = CHARACTER_COUNT + position

    return BoardTokens(character_indices, character_features, character_mask, entity_indices, entity_features, entity_mask,
                       position_by_object_id, max(0, len(entity_rows) - MAX_ENTITIES))
