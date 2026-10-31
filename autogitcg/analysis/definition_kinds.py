"""Static facts used to classify moves and decks: skill type, card category, nominal dice cost (from the Piovium catalog)."""

from functools import cache

from autogitcg.data.piovium_catalog import load_catalog
from autogitcg.observation.enums import ELEMENTS

SKILL_TYPES = ("normal", "elemental", "burst")
CARD_CATEGORIES = ("talent", "equipment", "support", "event")
ENERGY_COST_KEY = "9"
LEGEND_COST_KEY = "10"


def nominal_dice(cost: dict | None) -> int:
    """Dice in a cost, ignoring energy and the Legend limit (discounts are not known from the logs)."""
    return sum(amount for key, amount in (cost or {}).items() if key not in (ENERGY_COST_KEY, LEGEND_COST_KEY))


@cache
def skill_facts() -> dict[int, tuple[str, int]]:
    """Skill id -> (skill type, nominal dice cost)."""
    return {skill["id"]: (skill["skillType"], nominal_dice(skill.get("cost")))
            for character in load_catalog()["characters"] for skill in character["skills"]}


def card_category(entity: dict) -> str:
    if "talent" in entity["tags"]:
        return "talent"
    if entity["type"] == "equipment":
        return "equipment"
    if entity["type"] == "support":
        return "support"
    return "event"


@cache
def card_facts() -> dict[int, tuple[str, int, tuple[str, ...]]]:
    """Deck card id -> (category, nominal dice cost, tags)."""
    return {entity["id"]: (card_category(entity), nominal_dice(entity.get("playCost")), tuple(entity["tags"]))
            for entity in load_catalog()["entities"] if entity.get("isDeckCard")}


@cache
def character_element() -> dict[int, str]:
    """Character id -> element tag ("" when it has none)."""
    return {character["id"]: next((tag for tag in character["tags"] if tag in ELEMENTS), "")
            for character in load_catalog()["characters"]}
