"""Build every PvE stage the progression simulator can schedule, from Dimbreath tables.

Stages whose opponent the engine cannot run are left out (`build_stage_catalog` reports them).
Starter characters' invitations need no letter; every other invitation costs one letter, paid on
the first attempt at its Friendly Fracas.
"""

from autogitcg.data.deck import Deck
from autogitcg.data.dimbreath_tables import english, load_table
from autogitcg.data.invitations import SERIOUS_SHOWDOWN, load_invitation_stages
from autogitcg.data.piovium_catalog import characters_by_id, entities_by_id
from autogitcg.progression.stage import (
    ASCENSION, INVITATION_FRIENDLY, INVITATION_SERIOUS, OPEN_WORLD, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST,
    Objective, Stage,
)

FREE_INVITATIONS = {1301, 1103, 1501, 1401}  # Diluc, Kaeya, Sucrose, Fischl
# Each Ascension Challenge opens every level up to the next gate (gates sit before levels 3, 5, 7 and 10).
ASCENSION_LEVEL_CAPS = {20001: 4, 20002: 6, 20003: 9, 20004: 10}
TAVERN_SERIOUS_PLAYER_LEVEL = 7
PRACTICE_STAGE_NAMES = {"Atra", "Belén", "Gary", "Makri", "Nanaya Yuumi"}


class _Tables:
    def __init__(self):
        self.games = {row["id"]: row for row in load_table("GCGGameExcelConfigData")}
        self.decks = {row["id"]: row for row in load_table("GCGDeckExcelConfigData")}
        self.reward_rows = {row["levelId"]: row for row in load_table("GCGGameRewardExcelConfigData")}
        self.rewards = {row["rewardId"]: row for row in load_table("RewardExcelConfigData")}
        self.challenges = {row["id"]: row for row in load_table("GCGChallengeExcelConfigData")}

    def enemy_deck(self, level_id: int) -> Deck:
        row = self.decks.get(self.games.get(level_id, {}).get("enemyCardGroupId"), {})
        return Deck(characters=row.get("characterList", []), cards=row.get("cardList", []))

    def name(self, level_id: int) -> str:
        return english(self.reward_rows.get(level_id, {}).get("levelNameTextMapHash")) or str(level_id)

    def required_level(self, level_id: int) -> int:
        for condition in self.reward_rows.get(level_id, {}).get("condList", []):
            if condition.get("type") == "GCG_LEVEL":
                return condition["paramList"][0]
        return 0

    def objectives(self, level_id: int) -> tuple[Objective, ...]:
        objectives = []
        for entry in self.reward_rows.get(level_id, {}).get("challengeRewardList", []):
            challenge = self.challenges.get(entry.get("challengeId"))
            if challenge is None:
                continue
            items = tuple(
                (item["itemId"], item.get("itemCount", 1))
                for item in self.rewards.get(entry.get("rewardId"), {}).get("rewardItemList", []) if item.get("itemId")
            )
            objectives.append(Objective(challenge["type"], challenge.get("progress", 0),
                                        tuple(challenge.get("paramList", [])), items))  # positional; may contain ""
        return tuple(objectives)


def _supported(deck: Deck) -> bool:
    characters, entities = characters_by_id(), entities_by_id()
    return bool(deck.characters) and all(c in characters for c in deck.characters) and all(c in entities for c in deck.cards)


def _invitation_stages(tables: _Tables) -> list[Stage]:
    stages = []
    for invitation in load_invitation_stages(npc_side=1):
        serious = invitation.difficulty == SERIOUS_SHOWDOWN
        free = invitation.invited_character_id in FREE_INVITATIONS
        stages.append(Stage(
            level_id=invitation.game_id, name=tables.name(invitation.game_id),
            kind=INVITATION_SERIOUS if serious else INVITATION_FRIENDLY,
            required_player_level=invitation.required_player_level,
            enemy_deck=invitation.enemy_deck, enemy_equipment=invitation.enemy_equipment,
            objectives=tables.objectives(invitation.game_id),
            letter_cost=0 if (serious or free) else 1, group=invitation.invited_character_id,
        ))
    return stages


def _tavern_stages(tables: _Tables) -> list[Stage]:
    stages = []
    for boss in load_table("GCGBossLevelExcelConfigData"):
        for level_key, kind in (("normalLevelId", TAVERN_FRIENDLY), ("hardLevelId", TAVERN_SERIOUS)):
            level_id = boss.get(level_key)
            if not level_id:
                continue
            required = boss.get("unlockParam", 0) if kind == TAVERN_FRIENDLY else max(TAVERN_SERIOUS_PLAYER_LEVEL, boss.get("unlockParam", 0))
            stages.append(Stage(level_id, tables.name(level_id), kind, required, tables.enemy_deck(level_id), (),
                                tables.objectives(level_id), group=boss["id"]))
    return stages


def _weekly_stages(tables: _Tables) -> list[Stage]:
    """One stage per guest deck variant; `required_player_level` is the level at which that variant applies."""
    stages = []
    for guest in load_table("GCGWeekLevelExcelConfigData"):
        for variant in guest.get("levelCondList", []):
            level_id = variant.get("levelId")
            if level_id:
                stages.append(Stage(level_id, tables.name(level_id), WEEKLY_GUEST, variant.get("gcgLevel", 0),
                                    tables.enemy_deck(level_id), (), (), group=guest["id"]))
    return stages


def _open_world_stages(tables: _Tables) -> list[Stage]:
    stages = []
    for world_level in load_table("GCGWorldLevelExcelConfigData"):
        level_id = world_level["levelId"]
        name = tables.name(level_id)
        objectives = tables.objectives(level_id)
        if name in PRACTICE_STAGE_NAMES or not any(objective.rewards for objective in objectives):
            continue
        stages.append(Stage(level_id, name, OPEN_WORLD, tables.required_level(level_id), tables.enemy_deck(level_id), (),
                            objectives, group=world_level.get("npcId", 0)))
    return stages


def _ascension_stages(tables: _Tables) -> list[Stage]:
    locks = {row["levelId"]: row.get("unlockLevel", 0) for row in load_table("GCGLevelLockExcelConfigData")}
    return [
        Stage(level_id, tables.name(level_id), ASCENSION, locks.get(level_id, 0), tables.enemy_deck(level_id), (),
              tables.objectives(level_id), raises_level_cap_to=cap)
        for level_id, cap in ASCENSION_LEVEL_CAPS.items()
    ]


def build_stage_catalog() -> tuple[list[Stage], list[Stage]]:
    """Returns (supported stages, stages left out because the engine can't run their opponent)."""
    tables = _Tables()
    stages = (_invitation_stages(tables) + _tavern_stages(tables) + _weekly_stages(tables)
              + _open_world_stages(tables) + _ascension_stages(tables))
    supported = [stage for stage in stages if _supported(stage.enemy_deck)]
    unsupported = [stage for stage in stages if not _supported(stage.enemy_deck)]
    return supported, unsupported
