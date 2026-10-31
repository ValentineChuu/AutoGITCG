"""Translates a network choice (token index or yes/no bits) into the engine's response for that decision."""

from autogitcg.bridge.messages import Decision
from autogitcg.observation.encode_dice import MAX_DICE
from autogitcg.observation.encode_hand import MAX_HAND_PER_SIDE
from autogitcg.observation.observation_encoder import EncodedDecision


def response_from_choice(decision: Decision, encoded: EncodedDecision, choice_index: int, choice_bits: list[float]) -> dict:
    me = decision.state["player"][decision.who]
    match decision.kind:
        case "action":
            request_index = encoded.request_indices[choice_index]
            action = decision.request["action"][request_index]
            return {"chosenActionIndex": request_index, "usedDice": action["autoSelectedDice"]}
        case "chooseActive":
            return {"activeCharacterId": encoded.request_indices[choice_index]}
        case "selectCard":
            return {"selectedDefinitionId": encoded.request_indices[choice_index]}
        case "rerollDice":
            dice = me.get("dice", [])[:MAX_DICE]
            return {"diceToReroll": [die for die, bit in zip(dice, choice_bits) if bit > 0]}
        case "switchHands":
            hand = me.get("handCard", [])[:MAX_HAND_PER_SIDE]
            return {"removedHandIds": [card["id"] for card, bit in zip(hand, choice_bits) if bit > 0]}
    raise ValueError(f"unknown decision kind: {decision.kind}")
