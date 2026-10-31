"""A compact, analysis-friendly description of one move: what kind of move, and which skill/card/character it used."""

from dataclasses import dataclass

from autogitcg.bridge.messages import Decision


@dataclass
class MoveSummary:
    round: int
    decision_kind: str  # action, chooseActive, rerollDice, switchHands, selectCard
    action_kind: str  # useSkill, playCard, switchActive, elementalTuning, declareEnd; "" for other decision kinds
    definition_id: int  # skill / card / character definition involved; for rerolls and mulligans the number changed
    is_fast: bool = False
    dice_paid: int = 0


def summarize_move(decision: Decision, response: dict) -> MoveSummary:
    state, round_number = decision.state, decision.state.get("roundNumber", 0)
    me = state["player"][decision.who]
    match decision.kind:
        case "action":
            action = decision.request["action"][response["chosenActionIndex"]]
            kind, value = action["action"]["$case"], action["action"]["value"]
            definition_id = 0
            match kind:
                case "useSkill":
                    definition_id = value["skillDefinitionId"]
                case "playCard":
                    definition_id = value["cardDefinitionId"]
                case "switchActive":
                    definition_id = value["characterDefinitionId"]
                case "elementalTuning":
                    definition_id = next((card["definitionId"] for card in me.get("handCard", [])
                                          if card["id"] == value["removedCardId"]), 0)
            return MoveSummary(round_number, decision.kind, kind, definition_id, action.get("isFast", False),
                               len(response.get("usedDice", [])))
        case "chooseActive":
            definition_by_id = {character["id"]: character["definitionId"] for character in me.get("character", [])}
            return MoveSummary(round_number, decision.kind, "", definition_by_id.get(response["activeCharacterId"], 0))
        case "selectCard":
            return MoveSummary(round_number, decision.kind, "", response["selectedDefinitionId"])
        case "rerollDice":
            return MoveSummary(round_number, decision.kind, "", len(response["diceToReroll"]))
        case "switchHands":
            return MoveSummary(round_number, decision.kind, "", len(response["removedHandIds"]))
    raise ValueError(f"unknown decision kind: {decision.kind}")
