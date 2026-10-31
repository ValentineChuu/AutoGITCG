"""Which moves of a position the ground truth compares: the network's most likely ones, then its most likely move
of every kind not yet included (so an unlikely setup card is still tested). Tuning away two copies of the same card
is the same move, so only one of them is kept."""

from autogitcg.bridge.messages import Decision
from experiments.tree_search.move_options import MoveOption, move_options
from autogitcg.observation.observation_encoder import EncodedDecision


def move_kind(option: MoveOption) -> str:
    return option.key.split('"kind": "')[1].split('"')[0]


def same_move_key(decision: Decision, option: MoveOption) -> str:
    """Tunings of cards with the same definition count as one move."""
    action = decision.request["action"][option.response["chosenActionIndex"]]["action"]
    if action["$case"] != "elementalTuning":
        return option.key
    hand = {card["id"]: card["definitionId"] for card in decision.state["player"][decision.who].get("handCard", [])}
    return f"tune {hand.get(action['value']['removedCardId'])}"


def candidate_moves(decision: Decision, encoded: EncodedDecision, probabilities, likely_moves: int) -> list[MoveOption]:
    ranked = sorted(move_options(decision, encoded, probabilities).values(), key=lambda option: -option.probability)
    chosen, seen = [], set()
    for option in ranked:
        if len(chosen) < likely_moves and same_move_key(decision, option) not in seen:
            chosen.append(option)
            seen.add(same_move_key(decision, option))
    kinds = {move_kind(option) for option in chosen}
    for option in ranked:
        if move_kind(option) not in kinds:
            chosen.append(option)
            kinds.add(move_kind(option))
    return chosen
