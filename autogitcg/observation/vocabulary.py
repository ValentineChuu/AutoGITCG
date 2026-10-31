"""One shared index space for every definition the bot can see: characters, entities/cards and skills.

Index 0 is padding, 1 is "hidden or unknown". The network keeps one embedding row per index.
"""

from dataclasses import dataclass, field

from autogitcg.data.piovium_catalog import load_catalog

PADDING_INDEX = 0
HIDDEN_INDEX = 1


@dataclass
class Vocabulary:
    index_by_key: dict[tuple[str, float], int] = field(default_factory=dict)
    definitions: list[tuple[str, dict] | None] = field(default_factory=lambda: [None, None])

    def add(self, kind: str, identifier: float, definition: dict) -> None:
        key = (kind, identifier)
        if key not in self.index_by_key:
            self.index_by_key[key] = len(self.definitions)
            self.definitions.append((kind, definition))

    def index(self, kind: str, identifier: float | None) -> int:
        """kind is "character", "entity" or "skill". Definition id 0 means hidden."""
        if not identifier:
            return HIDDEN_INDEX
        return self.index_by_key.get((kind, identifier), HIDDEN_INDEX)

    def __len__(self) -> int:
        return len(self.definitions)


def build_vocabulary() -> Vocabulary:
    catalog = load_catalog()
    vocabulary = Vocabulary()
    for character in catalog["characters"]:
        vocabulary.add("character", character["id"], character)
    for entity in catalog["entities"]:
        vocabulary.add("entity", entity["id"], entity)
    for character in catalog["characters"]:
        for skill in character["skills"]:
            vocabulary.add("skill", skill["id"], {**skill, "ownerTags": character["tags"]})
    for entity in catalog["entities"]:
        for skill in entity["skills"]:
            vocabulary.add("skill", skill["id"], {**skill, "ownerTags": entity["tags"]})
    return vocabulary
