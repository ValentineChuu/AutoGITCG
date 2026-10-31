"""A playable PvE stage and its objectives, as the progression simulator sees it."""

from dataclasses import dataclass

from autogitcg.bridge.initial_equipment import InitialEquipment
from autogitcg.data.deck import Deck

OPEN_WORLD = "open_world"
INVITATION_FRIENDLY = "invitation_friendly"
INVITATION_SERIOUS = "invitation_serious"
TAVERN_FRIENDLY = "tavern_friendly"
TAVERN_SERIOUS = "tavern_serious"
WEEKLY_GUEST = "weekly_guest"
ASCENSION = "ascension"


@dataclass(frozen=True)
class Objective:
    challenge_type: str  # GCG_CHALLENGE_* from GCGChallengeExcelConfigData
    progress: int  # the challenge's target number (rounds, defeated characters, times, ...)
    parameters: tuple[str, ...]  # positional, "" when unused; e.g. (card tag, card type) for PLAY_CARD_WITH_TAG_AND_TYPE_TIMES
    rewards: tuple[tuple[int, int], ...]  # (item id, count)


@dataclass(frozen=True)
class Stage:
    level_id: int  # also the game id
    name: str
    kind: str
    required_player_level: int
    enemy_deck: Deck
    enemy_equipment: tuple[InitialEquipment, ...]  # applied to the NPC, who plays as player 1
    objectives: tuple[Objective, ...]
    letter_cost: int = 0  # Match Invitation Letters spent to start (once per invitation)
    group: int = 0  # stages sharing a group are versions of one opponent (invitation character, guest, ...)
    raises_level_cap_to: int | None = None  # Ascension Challenges unlock the next player level
