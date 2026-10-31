"""The moves of an action decision as tree edges: a key that names the same move in every game copy, the engine
response that plays it, and the network's probability for it."""

import json
from dataclasses import dataclass

from autogitcg.bridge.messages import Decision
from autogitcg.observation.observation_encoder import EncodedDecision

# willBeEffectless can differ between copies with different hidden cards; mainDamageTargetId is only sent when
# previews are on (keys stay the same with or without them, e.g. ground-truth positions recorded without previews)
COPY_DEPENDENT_FIELDS = ("willBeEffectless", "mainDamageTargetId")


@dataclass(frozen=True)
class MoveOption:
    key: str
    response: dict
    probability: float


def move_key(action: dict) -> str:
    """The same move (kind, card or skill, targets) gets the same key in every copy of the game."""
    value = {name: field for name, field in action["action"]["value"].items() if name not in COPY_DEPENDENT_FIELDS}
    return json.dumps({"kind": action["action"]["$case"], "value": value}, sort_keys=True)


def move_options(decision: Decision, encoded: EncodedDecision, probabilities) -> dict[str, MoveOption]:
    """The decision's moves the network can choose (one per action token), by key."""
    options = {}
    for choice_index, request_index in enumerate(encoded.request_indices):
        action = decision.request["action"][request_index]
        key = move_key(action)
        response = {"chosenActionIndex": request_index, "usedDice": action["autoSelectedDice"]}
        options[key] = MoveOption(key, response, float(probabilities[choice_index]))
    return options
