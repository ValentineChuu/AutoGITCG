"""A sign of thoughtless card plays: healing food eaten by a character already at full health.
(The engine's willBeEffectless only marks plays an opponent's effect cancels, so it can't flag pointless plays.)"""

from collections import Counter
from functools import cache

from autogitcg.data.dimbreath_tables import english, load_table
from autogitcg.data.description_text import readable_text

FOOD_TAG = "GCG_TAG_FOOD"


@cache
def healing_food() -> frozenset[int]:
    """Food cards whose text heals (not the buff, shield or revive food)."""
    return frozenset(row["id"] for row in load_table("GCGCardExcelConfigData") if FOOD_TAG in row.get("tagList", [])
                     and "heal" in readable_text(english(row.get("descTextMapHash")) or "").lower())


def record_play_quality(stats: Counter, chosen: dict, me: dict) -> None:
    """chosen: the chosen action option; me: the acting player's state."""
    if chosen["action"]["$case"] != "playCard":
        return
    play = chosen["action"]["value"]
    if play["cardDefinitionId"] not in healing_food():
        return
    stats["healing food plays"] += 1
    characters = {character["id"]: character for character in me.get("character", [])}
    targets = [characters[target] for target in play.get("targetIds", []) if target in characters]
    stats["healing food at full health"] += any(target["health"] >= target["maxHealth"] for target in targets)
