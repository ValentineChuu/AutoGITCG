"""The single global token: phase, turn, round, dice summary, card counts and decision context."""

import numpy as np

from autogitcg.observation.encode_end_of_round import end_of_round_features
from autogitcg.observation.enums import DECISION_KINDS, DICE_TYPE_COUNT, PHASE_COUNT
from autogitcg.observation.game_tracker import GameTracker


def _global_features(state: dict, who: int, decision_kind: str, tracker: GameTracker | None, end_of_round: dict | None = None) -> list[float]:
    me, opponent = state["player"][who], state["player"][1 - who]
    my_dice = me.get("dice", [])
    dice_counts = [0.0] * DICE_TYPE_COUNT
    for die in my_dice:
        dice_counts[die] += 1 / 8
    round_number = state.get("roundNumber", 0)
    declared_first = tracker.declared_end_first_this_round(round_number) if tracker else None

    def alive(player: dict) -> float:
        return sum(1 for character in player.get("character", []) if not character.get("defeated", False)) / 3

    return (
        [1.0 if state.get("phase", 0) == phase else 0.0 for phase in range(PHASE_COUNT)]
        + [1.0 if decision_kind == kind else 0.0 for kind in DECISION_KINDS]
        + [round_number / 15, float(state.get("currentTurn", 0) == who)]
        + [float(me.get("declaredEnd", False)), float(opponent.get("declaredEnd", False))]
        + [float(declared_first is True), float(declared_first is False)]
        + [float(me.get("legendUsed", False)), float(opponent.get("legendUsed", False))]
        + dice_counts
        + [len(my_dice) / 16, len(opponent.get("dice", [])) / 16, float(len(my_dice) % 2 == 0)]
        + [len(me.get("handCard", [])) / 10, len(opponent.get("handCard", [])) / 10]
        + [len(me.get("pileCard", [])) / 30, len(opponent.get("pileCard", [])) / 30]
        + [alive(me), alive(opponent)]
        + [len(me.get("summon", [])) / 4, len(opponent.get("summon", [])) / 4]
        + [len(me.get("support", [])) / 4, len(opponent.get("support", [])) / 4]
        + [float(tracker is not None and tracker.opponent_deck is not None)]
        + [float(bool(state.get("canPlunging", [False, False])[who]))]  # appended: older networks load (network_config.py)
        + end_of_round_features(end_of_round, who)  # appended last: older networks load with zero weights for it
    )


GLOBAL_WIDTH = len(_global_features({"player": [{}, {}]}, 0, "action", None))


def encode_global(state: dict, who: int, decision_kind: str, tracker: GameTracker, end_of_round: dict | None = None) -> np.ndarray:
    return np.asarray(_global_features(state, who, decision_kind, tracker, end_of_round), dtype=np.float32)
