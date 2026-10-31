"""Identify card instances that can legally be played or tuned in one action request."""

from collections import defaultdict
import hashlib
import json

from experiments.card_play_diagnostic.card_types import card_type_by_id, dice_bucket
from autogitcg.bridge.messages import Decision
from autogitcg.observation.observation_encoder import EncodedDecision


def decision_fingerprint(decision: Decision) -> str:
    payload = {"who": decision.who, "kind": decision.kind, "request": decision.request,
               "state": decision.state, "counters": decision.counters}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def comparable_cards(decision: Decision, encoded: EncodedDecision, probabilities) -> list[dict]:
    if decision.kind != "action":
        return []
    hand_definitions = {
        card["id"]: card["definitionId"] for card in decision.state["player"][decision.who].get("handCard", [])
    }
    play_positions = defaultdict(list)
    tune_positions = defaultdict(list)
    for position, request_index in enumerate(encoded.request_indices):
        action = decision.request["action"][request_index]["action"]
        value = action["value"]
        if action["$case"] == "playCard":
            play_positions[value["cardDefinitionId"]].append(position)
        elif action["$case"] == "elementalTuning":
            definition_id = hand_definitions.get(value["removedCardId"])
            tune_positions[definition_id].append(position)
    me = decision.state["player"][decision.who]
    total_dice = len(me.get("dice", []))
    result = []
    for definition_id in sorted(play_positions.keys() & tune_positions.keys()):
        category = card_type_by_id().get(definition_id)
        if category is None:
            continue
        plays, tunes = play_positions[definition_id], tune_positions[definition_id]
        result.append({
            "card_id": definition_id,
            "card_type": category,
            "round": decision.state["roundNumber"],
            "dice": total_dice,
            "dice_bucket": dice_bucket(total_dice),
            "omni_dice": me.get("dice", []).count(8),
            "play_probability": float(sum(probabilities[position] for position in plays)),
            "tune_probability": float(sum(probabilities[position] for position in tunes)),
            "play_option_count": len(plays),
        })
    return result


def forced_option_indices(decision: Decision, encoded: EncodedDecision, definition_id: int, branch: str) -> list[int]:
    hand_definitions = {
        card["id"]: card["definitionId"] for card in decision.state["player"][decision.who].get("handCard", [])
    }
    positions = []
    for position, request_index in enumerate(encoded.request_indices):
        action = decision.request["action"][request_index]["action"]
        value = action["value"]
        if branch == "play" and action["$case"] == "playCard" and value["cardDefinitionId"] == definition_id:
            positions.append(position)
        if branch == "tune" and action["$case"] == "elementalTuning" and hand_definitions.get(
                value["removedCardId"]) == definition_id:
            positions.append(position)
    return positions


def response_for_position(decision: Decision, encoded: EncodedDecision, position: int) -> dict:
    request_index = encoded.request_indices[position]
    action = decision.request["action"][request_index]
    return {"chosenActionIndex": request_index, "usedDice": action["autoSelectedDice"]}
