"""Which of a stored position's compared moves a move (by its request index) is: the same action, or a tuning of a
card with the same definition (the ground truth kept only one of those). None if it wasn't compared."""

from autogitcg.bridge.messages import Decision
from experiments.ground_truth.candidate_moves import same_move_key
from experiments.tree_search.move_options import MoveOption, move_key


def compared_move_index(line: dict, decision: Decision, request_index: int) -> int | None:
    action = decision.request["action"][request_index]
    wanted = same_move_key(decision, MoveOption(move_key(action), {"chosenActionIndex": request_index}, 0.0))
    for index, move in enumerate(line["moves"]):
        if same_move_key(decision, MoveOption(move["key"], move["response"], move["probability"])) == wanted:
            return index
    return None
