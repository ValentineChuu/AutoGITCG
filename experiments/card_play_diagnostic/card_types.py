"""Card categories used by the play-versus-tune diagnostic."""

from functools import cache

from autogitcg.data.piovium_catalog import entities_by_id


CARD_TYPES = ("weapon", "artifact", "support", "talent", "event", "food")


@cache
def card_type_by_id() -> dict[int, str]:
    result = {}
    for definition_id, entity in entities_by_id().items():
        if not entity.get("isDeckCard"):
            continue
        tags = set(entity.get("tags", ()))
        if "talent" in tags:
            category = "talent"
        elif "weapon" in tags:
            category = "weapon"
        elif "artifact" in tags:
            category = "artifact"
        elif entity["type"] == "support":
            category = "support"
        elif "food" in tags:
            category = "food"
        elif entity["type"] == "eventCard":
            category = "event"
        else:
            continue
        result[definition_id] = category
    return result


def dice_bucket(count: int) -> str:
    if count <= 2:
        return "0-2"
    if count <= 5:
        return "3-5"
    return "6+"
