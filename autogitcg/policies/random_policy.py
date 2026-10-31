"""Uniformly random choices for bridge decisions; a baseline opponent and a bridge smoke test."""

import random

from autogitcg.bridge.messages import Decision

VALID = 0  # ActionValidity.VALID


def random_response(decision: Decision, rng: random.Random) -> dict:
    request = decision.request
    match decision.kind:
        case "action":
            valid = [(index, action) for index, action in enumerate(request["action"]) if action.get("validity", VALID) == VALID]
            index, action = rng.choice(valid)
            return {"chosenActionIndex": index, "usedDice": action["autoSelectedDice"]}
        case "chooseActive":
            return {"activeCharacterId": rng.choice(request["candidateIds"])}
        case "selectCard":
            return {"selectedDefinitionId": rng.choice(request["candidateDefinitionIds"])}
        case "rerollDice":
            return {"diceToReroll": []}
        case "switchHands":
            return {"removedHandIds": []}
    raise ValueError(f"unknown decision kind: {decision.kind}")
