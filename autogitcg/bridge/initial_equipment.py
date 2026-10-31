"""A card equipped on a character before the game starts (e.g. a Serious Showdown NPC's talent)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InitialEquipment:
    who: int
    character_index: int  # position in that player's deck character list
    definition_id: int

    def to_message(self) -> dict:
        return {"who": self.who, "characterIndex": self.character_index, "definitionId": self.definition_id}
