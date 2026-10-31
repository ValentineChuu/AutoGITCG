"""Immediate reward for playing an action card (`--card-play-bonus`, plus `--card-kind-bonus` per card kind).

Unlike the potential-based shaping this changes what the best strategy is: it nudges Chudledum toward playing its
cards, as people do (19 cards a game against its 7 in PvP games against people), at a small cost in pure win rate.
Fading it out again lost the habit (delta 7-9, docs/training.md).
"""

from autogitcg.bridge.messages import Decision
from autogitcg.data.card_kinds import card_kind


def card_play_bonus(decision: Decision, response: dict, weight: float, kind_weights: dict[str, float] | None = None) -> float:
    """weight for any card played, plus kind_weights[its kind] (data/card_kinds.py) if given."""
    if (weight == 0.0 and not kind_weights) or decision.kind != "action":
        return 0.0
    chosen = decision.request["action"][response["chosenActionIndex"]]["action"]
    if chosen["$case"] != "playCard":
        return 0.0
    return weight + (kind_weights or {}).get(card_kind(chosen["value"]["cardDefinitionId"]), 0.0)
