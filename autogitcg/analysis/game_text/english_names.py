"""English name lookup for character, card/entity and skill ids, with a readable fallback for unnamed ids."""

from functools import cache

from autogitcg.analysis.definition_names import definition_names


@cache
def _names_by_kind_and_id() -> dict[tuple[str, int], str]:
    return {(kind, identifier): name for identifier, kind, name in definition_names()}



def character_name(definition_id: int) -> str:
    return _names_by_kind_and_id().get(("character", definition_id), f"character {definition_id}")


def card_name(definition_id: int) -> str:
    """Cards, and the statuses/summons/supports that share the card id space."""
    return _names_by_kind_and_id().get(("card", definition_id), f"#{definition_id}")


def skill_name(definition_id: int) -> str:
    return _names_by_kind_and_id().get(("skill", definition_id), f"skill {definition_id}")
