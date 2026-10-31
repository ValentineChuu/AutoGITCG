"""Character Invitation stages (Friendly Fracas and Serious Showdown) from Dimbreath data.

The invited character is the deck character whose card says "Reward for inviting [the] <name>", where
<name> comes from the invitation's avatar; starter characters (whose cards say "initial deck") are matched
by card name instead. In a Serious Showdown that character starts with its
talent equipped; when a character has two talent cards, the original (lowest id) is assumed.
"""

from dataclasses import dataclass

from autogitcg.bridge.initial_equipment import InitialEquipment
from autogitcg.data.deck import Deck
from autogitcg.data.dimbreath_tables import english, load_table

FRIENDLY_FRACAS = "friendly_fracas"
SERIOUS_SHOWDOWN = "serious_showdown"
SERIOUS_SHOWDOWN_PLAYER_LEVEL = 7


@dataclass(frozen=True)
class InvitationStage:
    invited_character_id: int  # the character card this invitation awards
    difficulty: str
    required_player_level: int
    game_id: int
    enemy_deck: Deck
    enemy_equipment: tuple[InitialEquipment, ...]  # for the NPC playing as `npc_side`


def _invited_character(deck: Deck, avatar_name: str, access_text_by_card: dict[int, str],
                       name_by_card: dict[int, str]) -> int | None:
    markers = (f"inviting {avatar_name} to", f"inviting the {avatar_name} to")
    by_text = [character for character in deck.characters
               if any(marker in access_text_by_card.get(character, "") for marker in markers)]
    if by_text:
        return by_text[0]
    return next((character for character in deck.characters if name_by_card.get(character) == avatar_name), None)


def load_invitation_stages(npc_side: int = 1) -> list[InvitationStage]:
    games = {row["id"]: row for row in load_table("GCGGameExcelConfigData")}
    decks = {row["id"]: row for row in load_table("GCGDeckExcelConfigData")}
    avatar_names = {row["id"]: english(row.get("nameTextMapHash")) for row in load_table("AvatarExcelConfigData")}
    deck_cards = load_table("GCGDeckCardExcelConfigData")
    access_text_by_card = {row["id"]: english(row.get("accessDescTextMapHash")) or "" for row in deck_cards}
    name_by_card = {row["id"]: english(row.get("nameTextMapHash")) for row in load_table("GCGCharExcelConfigData")}
    talents_by_character: dict[int, list[int]] = {}
    for row in deck_cards:
        if row.get("relatedCharacterId"):
            talents_by_character.setdefault(row["relatedCharacterId"], []).append(row["id"])

    def enemy_deck(game_id: int) -> Deck:
        row = decks[games[game_id]["enemyCardGroupId"]]
        return Deck(characters=row.get("characterList", []), cards=row.get("cardList", []))

    stages = []
    for invitation in load_table("GCGCharacterLevelExcelConfigData"):
        hard_deck = enemy_deck(invitation["hardLevelId"])
        invited = _invited_character(hard_deck, avatar_names.get(invitation.get("avatarId"), ""), access_text_by_card, name_by_card)
        if invited is None:
            raise ValueError(f"invitation {invitation['id']}: invited character not found in its deck")
        for level in invitation.get("normalLevelList", []):
            if level.get("levelId"):
                stages.append(InvitationStage(invited, FRIENDLY_FRACAS, level.get("gcgLevel", 0), level["levelId"],
                                              enemy_deck(level["levelId"]), ()))
        talent = min(talents_by_character.get(invited, []), default=None)
        equipment = (InitialEquipment(npc_side, hard_deck.characters.index(invited), talent),) if talent else ()
        stages.append(InvitationStage(invited, SERIOUS_SHOWDOWN, SERIOUS_SHOWDOWN_PLAYER_LEVEL, invitation["hardLevelId"],
                                      hard_deck, equipment))
    return stages
