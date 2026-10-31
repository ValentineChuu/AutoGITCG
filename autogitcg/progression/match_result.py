"""What the progression simulator needs to know about a finished match, from the player's point of view."""

from dataclasses import dataclass
from functools import cache

from autogitcg.bridge.messages import GameFinished
from autogitcg.data.piovium_catalog import load_catalog
from autogitcg.observation.game_tracker import GameTracker


@dataclass(frozen=True)
class MatchResult:
    won: bool
    rounds: int
    my_defeated: int  # my characters defeated
    bursts_used: int
    cards_played: tuple[int, ...]  # card definition ids, in play order
    reactions: int | None = None  # None when not measured


@cache
def _skill_types() -> dict[float, str]:
    catalog = load_catalog()
    return {
        skill["id"]: skill["skillType"]
        for owner in catalog["characters"] + catalog["entities"]
        for skill in owner.get("skills", [])
    }


def match_result_from(finished: GameFinished, who: int, tracker: GameTracker) -> MatchResult:
    my_events = [event for event in tracker.events if event.actor_is_me]
    skill_types = _skill_types()
    bursts = sum(1 for event in my_events
                 if event.action_kind == "useSkill" and skill_types.get(event.definition_id) == "burst")
    cards = tuple(int(event.definition_id) for event in my_events
                  if event.action_kind == "playCard" and event.definition_id is not None)
    return MatchResult(won=finished.winner == who, rounds=finished.rounds, my_defeated=finished.defeated[who],
                       bursts_used=bursts, cards_played=cards)
