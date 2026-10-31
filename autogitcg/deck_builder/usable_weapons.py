"""Weapon cards the lineup can't equip: a weapon needs a character of its weapon type (a bow for Fischl, not a polearm).

Decks may legally hold any weapon, but an unusable one only ever serves as Elemental Tuning fodder. Chudledee picked
such weapons in about 80% of its early-account decks.
"""

from functools import cache

from autogitcg.data.dimbreath_tables import load_table
from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS

WEAPON_TAG = "GCG_TAG_WEAPON"
WEAPON_TYPE_PREFIX = "GCG_TAG_WEAPON_"


def _weapon_type(tags: list[str]) -> str | None:
    return next((tag for tag in tags if tag.startswith(WEAPON_TYPE_PREFIX)), None)


@cache
def _weapon_card_types() -> dict[int, str]:
    """Weapon card id -> its weapon type tag."""
    return {row["id"]: weapon_type for row in load_table("GCGCardExcelConfigData")
            if WEAPON_TAG in (tags := row.get("tagList", [])) and (weapon_type := _weapon_type(tags))}


@cache
def _character_weapon_types() -> dict[int, str | None]:
    return {row["id"]: _weapon_type(row.get("tagList", [])) for row in load_table("GCGCharExcelConfigData")}


def unusable_weapons(lineup: list[int], cards) -> set[int]:
    """The weapon cards among `cards` that no lineup character can equip."""
    usable_types = {_character_weapon_types().get(character) for character in lineup}
    weapon_types = _weapon_card_types()
    return {card for card in cards if card in weapon_types and weapon_types[card] not in usable_types}


def without_unusable_weapons(card_limits: dict[int, int], lineup: list[int]) -> dict[int, int]:
    """`card_limits` minus the unusable weapons, unless that would leave fewer than a deck's worth of cards."""
    unusable = unusable_weapons(lineup, card_limits)
    remaining = {card: limit for card, limit in card_limits.items() if card not in unusable}
    return remaining if sum(remaining.values()) >= DECK_ACTION_CARDS else card_limits
