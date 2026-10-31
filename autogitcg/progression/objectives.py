"""Decide whether a stage objective (a GCGChallenge) was achieved in a match.

Objectives count only in a won match. Challenge types the match result can't measure yet
(e.g. element reactions, heal or shield totals) return None and are treated as not achieved.
"""

from functools import cache

from autogitcg.data.dimbreath_tables import load_table
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.stage import Objective


@cache
def _card_tags_and_types() -> dict[int, tuple[set[str], str]]:
    return {row["id"]: (set(row.get("tagList", [])), row.get("cardType", "")) for row in load_table("GCGCardExcelConfigData")}


def _cards_with(result: MatchResult, tag: str, card_type: str) -> int:
    tags_and_types = _card_tags_and_types()
    count = 0
    for card in result.cards_played:
        tags, type_ = tags_and_types.get(card, (set(), ""))
        if (not tag or tag in tags) and (not card_type or card_type == type_):
            count += 1
    return count


def objective_achieved(objective: Objective, result: MatchResult) -> bool | None:
    if not result.won:
        return False
    match objective.challenge_type:
        case "GCG_CHALLENGE_WIN":
            return True
        case "GCG_CHALLENGE_DEAD_CHARACTER_NUM":
            return result.my_defeated < objective.progress
        case "GCG_CHALLENGE_ROUND_NUM":
            return result.rounds <= objective.progress
        case "GCG_CHALLENGE_SKILL_TAG_Q_TIMES":
            return result.bursts_used >= objective.progress
        case "GCG_CHALLENGE_PLAY_CARD_TIMES":
            return len(result.cards_played) >= objective.progress
        case "GCG_CHALLENGE_PLAY_CARD_WITH_TAG_AND_TYPE_TIMES":
            tag = objective.parameters[0] if len(objective.parameters) > 0 else ""
            card_type = objective.parameters[1] if len(objective.parameters) > 1 else ""
            return _cards_with(result, tag, card_type) >= objective.progress
        case "GCG_CHALLENGE_ELEMENT_REACTION_TIMES":
            return None if result.reactions is None else result.reactions >= objective.progress
    return None


MEASURABLE_CHALLENGES = {
    "GCG_CHALLENGE_WIN", "GCG_CHALLENGE_DEAD_CHARACTER_NUM", "GCG_CHALLENGE_ROUND_NUM", "GCG_CHALLENGE_SKILL_TAG_Q_TIMES",
    "GCG_CHALLENGE_PLAY_CARD_TIMES", "GCG_CHALLENGE_PLAY_CARD_WITH_TAG_AND_TYPE_TIMES",
}


def objective_measurable(objective: Objective) -> bool:
    """Whether match results can ever achieve this objective (element reactions aren't measured yet)."""
    return objective.challenge_type in MEASURABLE_CHALLENGES
