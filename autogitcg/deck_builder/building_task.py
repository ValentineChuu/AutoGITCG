"""What the deck builder is asked to do: build a deck from this collection, for this kind of opponent.

Stage mode is an NPC fight: the stage's deck is known before building, so the builder can counter-pick.
PvP mode: the opponent is unknown while building.
"""

from dataclasses import dataclass

from autogitcg.progression.stage import Stage
from autogitcg.training.collection_pool import Collection

STAGE_MODE = "stage"
PVP_MODE = "pvp"


@dataclass(frozen=True)
class BuildingTask:
    collection: Collection
    mode: str  # STAGE_MODE or PVP_MODE
    stage: Stage | None = None  # the opponent in stage mode; its deck and equipment are shown to the builder
    collection_origin: str = ""  # e.g. "collection:12", for logs
    skip_unusable_weapons: bool = False  # never offer weapons the lineup can't equip (usable_weapons.py; experiment)

    @property
    def origin(self) -> str:
        opponent = f"stage:{self.stage.level_id}" if self.stage else self.mode
        return f"{self.collection_origin}|{opponent}"
