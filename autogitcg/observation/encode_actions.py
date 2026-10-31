"""Action tokens: one per choice the bot can make right now. The policy scores each token.

For "action" decisions only valid actions become tokens; `request_indices` maps each token
back to its index in the engine's action list. Reroll and opening-hand decisions have no
action tokens: they are answered per dice token / hand token instead.
"""

from dataclasses import dataclass

import numpy as np

from autogitcg.observation.encode_hand import cost_vector
from autogitcg.observation.encode_previews import PREVIEW_WIDTH, preview_features
from autogitcg.observation.encode_follow_ups import FOLLOW_UP_WIDTH, follow_up_features
from autogitcg.observation.encode_replies import REPLY_WIDTH, reply_features
from autogitcg.observation.enums import ACTION_KINDS, COST_TYPE_COUNT, VALID
from autogitcg.observation.vocabulary import Vocabulary

MAX_ACTIONS = 64
MAX_TARGETS = 2
TOKEN_KINDS = ACTION_KINDS + ["chooseActive", "selectCard"]
PREVIEW_START = len(TOKEN_KINDS) + COST_TYPE_COUNT + 2  # the engine's look-ahead inputs come last: the preview, then the replies
REPLY_START = PREVIEW_START + PREVIEW_WIDTH  # (encode_previews.py, encode_replies.py, encode_follow_ups.py)
UNCHECKED_COLUMN = REPLY_START + REPLY_WIDTH - 1  # the replies' "not checked" flag (learned reply selector)
ACTION_WIDTH = REPLY_START + REPLY_WIDTH + FOLLOW_UP_WIDTH  # follow-ups appended last: older networks load with zero weights for them


@dataclass
class ActionTokens:
    indices: np.ndarray  # (MAX_ACTIONS,) vocabulary index of the skill/card/character involved
    features: np.ndarray  # (MAX_ACTIONS, ACTION_WIDTH)
    targets: np.ndarray  # (MAX_ACTIONS, MAX_TARGETS) board token positions, -1 for none
    mask: np.ndarray  # (MAX_ACTIONS,)
    request_indices: list  # per token: the engine action index, character id or card definition id to answer with
    dropped: int


def _token(kind: str, cost: list[float], is_fast: bool, effectless: bool, preview: list[float] | None = None,
           reply: list[float] | None = None, follow_up: list[float] | None = None) -> list[float]:
    return ([1.0 if kind == name else 0.0 for name in TOKEN_KINDS] + cost + [float(is_fast), float(effectless)]
            + (preview if preview is not None else [0.0] * PREVIEW_WIDTH) + (reply if reply is not None else [0.0] * REPLY_WIDTH)
            + (follow_up if follow_up is not None else [0.0] * FOLLOW_UP_WIDTH))


def _engine_action_rows(request: dict, state: dict, who: int, previews: bool, replies: list | None, checked: set[int] | None,
                        hand_definitions: dict[int, int], position_by_object_id: dict[int, int], vocabulary: Vocabulary):
    for request_index, action in enumerate(request["action"]):
        if action.get("validity", VALID) != VALID:
            continue
        kind, value = action["action"]["$case"], action["action"]["value"]
        target_ids = []
        match kind:
            case "useSkill":
                index = vocabulary.index("skill", value["skillDefinitionId"])
                target_ids = value.get("targetIds", [])
            case "playCard":
                index = vocabulary.index("entity", value["cardDefinitionId"])
                target_ids = value.get("targetIds", [])
            case "switchActive":
                index = vocabulary.index("character", value["characterDefinitionId"])
                target_ids = [value["characterId"]]
            case "elementalTuning":
                index = vocabulary.index("entity", hand_definitions.get(value["removedCardId"]))
            case _:
                index = 0
        targets = [position_by_object_id.get(target_id, -1) for target_id in target_ids]
        features = _token(kind, cost_vector(action.get("requiredCost", [])), action.get("isFast", False), value.get("willBeEffectless", False),
                          preview_features(action.get("preview") if previews else None, state, who),
                          reply_features(replies[request_index] if replies else None, state, who,
                                         unchecked=checked is not None and request_index not in checked),
                          follow_up_features((replies[request_index] or {}).get("followUps") if replies else None))
        yield index, features, targets, request_index


def encode_actions(decision_kind: str, request: dict, state: dict, who: int,
                   position_by_object_id: dict[int, int], vocabulary: Vocabulary, previews: bool = False,
                   replies: list | None = None, checked: set[int] | None = None) -> ActionTokens:
    """previews: the actions carry real engine previews (Decision.previews); replies: Decision.replies; checked: with
    replies on demand, the actions whose replies were computed (the others are marked "not checked")."""
    if decision_kind == "action":
        hand_definitions = {card["id"]: card["definitionId"] for card in state["player"][who].get("handCard", [])}
        rows = list(_engine_action_rows(request, state, who, previews, replies, checked, hand_definitions, position_by_object_id, vocabulary))
    elif decision_kind == "chooseActive":
        definition_by_id = {character["id"]: character["definitionId"] for character in state["player"][who].get("character", [])}
        rows = [(vocabulary.index("character", definition_by_id.get(character_id)), _token("chooseActive", [0.0] * COST_TYPE_COUNT, False, False),
                 [position_by_object_id.get(character_id, -1)], character_id) for character_id in request["candidateIds"]]
    elif decision_kind == "selectCard":
        rows = [(vocabulary.index("entity", definition_id), _token("selectCard", [0.0] * COST_TYPE_COUNT, False, False), [], definition_id)
                for definition_id in request["candidateDefinitionIds"]]
    else:
        rows = []

    indices = np.zeros(MAX_ACTIONS, dtype=np.int64)
    features = np.zeros((MAX_ACTIONS, ACTION_WIDTH), dtype=np.float32)
    targets = np.full((MAX_ACTIONS, MAX_TARGETS), -1, dtype=np.int64)
    mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
    request_indices = []
    for position, (index, row_features, row_targets, request_index) in enumerate(rows[:MAX_ACTIONS]):
        indices[position] = index
        features[position] = row_features
        targets[position, :len(row_targets[:MAX_TARGETS])] = row_targets[:MAX_TARGETS]
        mask[position] = 1.0
        request_indices.append(request_index)
    return ActionTokens(indices, features, targets, mask, request_indices, max(0, len(rows) - MAX_ACTIONS))
