"""Static facts about every vocabulary entry, as one fixed-width feature row per index.

The network combines each row with that index's learned embedding, so rarely seen
cards start from a meaningful representation instead of a blank one.
"""

import numpy as np

from autogitcg.observation.entity_effects import effect_features
from autogitcg.observation.enums import COST_TYPE_COUNT
from autogitcg.observation.vocabulary import Vocabulary

DEFINITION_KINDS = ["character", "entity", "skill"]
ENTITY_KINDS = ["status", "combatStatus", "summon", "eventCard", "equipment", "support"]
SKILL_KINDS = ["normal", "elemental", "burst", "technique"]
# Initial values of these variables describe what an entity does (e.g. a 2-point shield, 3 usages).
DESCRIPTIVE_VARIABLES = ["usage", "usagePerRound", "duration", "shield", "progress", "layer", "nightsoul", "count", "barrierUsage"]


def _cost_vector(cost: dict | None) -> list[float]:
    vector = [0.0] * COST_TYPE_COUNT
    for requirement_type, count in (cost or {}).items():
        vector[int(requirement_type)] = count / 4
    return vector


def _multi_hot(values: list[str], names: list[str]) -> list[float]:
    present = set(values)
    return [1.0 if name in present else 0.0 for name in names]


def _one_hot(value: str, names: list[str]) -> list[float]:
    return [1.0 if value == name else 0.0 for name in names]


def _skill_costs_by_kind(skills: list[dict]) -> list[float]:
    """Cost of the first normal, elemental and burst skill, in that order."""
    features = []
    for kind in ["normal", "elemental", "burst"]:
        skill = next((candidate for candidate in skills if candidate["skillType"] == kind), None)
        features += _cost_vector(skill["cost"] if skill else None)
    return features


class CatalogFeatures:
    def __init__(self, vocabulary: Vocabulary):
        definitions = [entry for entry in vocabulary.definitions if entry is not None]
        self.tag_names = sorted({tag for _, definition in definitions for tag in definition.get("tags", definition.get("ownerTags", []))})
        self.trigger_names = sorted({
            trigger for _, definition in definitions
            for trigger in definition.get("triggers", []) + definition.get("passiveTriggers", [])
        })
        rows = [self._row(entry) for entry in vocabulary.definitions]
        self.table = np.asarray(rows, dtype=np.float32)

    @property
    def width(self) -> int:
        return self.table.shape[1]

    def _row(self, entry: tuple[str, dict] | None) -> list[float]:
        kind, definition = entry if entry else ("", {})
        character = definition if kind == "character" else {}
        entity = definition if kind == "entity" else {}
        skill = definition if kind == "skill" else {}
        variables = character.get("variables", {})
        entity_variables = entity.get("variables", {})
        return (
            _one_hot(kind, DEFINITION_KINDS)
            + [1.0 if entry is None else 0.0]  # padding / hidden
            + _multi_hot(definition.get("tags", definition.get("ownerTags", [])), self.tag_names)
            + _multi_hot(definition.get("triggers", []) + definition.get("passiveTriggers", []), self.trigger_names)
            # Characters
            + [variables.get("maxHealth", 0) / 10, variables.get("maxEnergy", 0) / 3,
               (character.get("specialEnergy") or {}).get("slotSize", 0) / 7]
            + (_skill_costs_by_kind(character.get("skills", [])) if character else [0.0] * (3 * COST_TYPE_COUNT))
            # Entities and cards
            + _one_hot(entity.get("type", ""), ENTITY_KINDS)
            + _cost_vector(entity.get("playCost"))
            + [1.0 if entity.get("disableTuning") else 0.0, 1.0 if entity.get("isDeckCard") else 0.0]
            + [np.log1p(max(entity_variables.get(name, 0), 0)) for name in DESCRIPTIVE_VARIABLES]
            # Skills
            + _one_hot(skill.get("skillType", ""), SKILL_KINDS)
            + _cost_vector(skill.get("cost"))
            + [1.0 if skill.get("gainEnergy") else 0.0, 1.0 if skill.get("shouldFast") else 0.0]
            # Appended last, so networks saved before load with zero weights for them: what an entity does
            + effect_features(entity.get("id") if entity else None)
        )
