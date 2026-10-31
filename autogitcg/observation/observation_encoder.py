"""Turns one Decision (plus the player's GameTracker) into fixed-shape arrays for the network.

Every array has a fixed shape so observations can be stacked into batches; variable-length
token groups come with a mask (1 = real token, 0 = padding).
"""

from dataclasses import dataclass

import numpy as np

from autogitcg.bridge.messages import Decision
from autogitcg.data.piovium_catalog import attachment_ids
from autogitcg.observation.catalog_features import CatalogFeatures
from autogitcg.observation.encode_actions import encode_actions
from autogitcg.observation.encode_board import encode_board
from autogitcg.observation.encode_dice import character_element_dice, encode_dice
from autogitcg.observation.encode_global import encode_global
from autogitcg.observation.encode_hand import encode_hand
from autogitcg.observation.encode_memory import encode_history, encode_knowledge
from autogitcg.observation.game_tracker import GameTracker
from autogitcg.observation.vocabulary import build_vocabulary


@dataclass
class EncodedDecision:
    arrays: dict[str, np.ndarray]
    request_indices: list  # how to translate a chosen action token back into an engine response
    dropped_tokens: int  # tokens cut off by size limits (should stay 0; tracked to tune limits)


class ObservationEncoder:
    def __init__(self):
        self.vocabulary = build_vocabulary()
        self.catalog_features = CatalogFeatures(self.vocabulary)
        self.attachment_ids = attachment_ids()
        self.tags_by_character = {
            definition["id"]: definition["tags"]
            for entry in self.vocabulary.definitions if entry and entry[0] == "character"
            for definition in [entry[1]]
        }

    def encode(self, decision: Decision, tracker: GameTracker) -> EncodedDecision:
        state, who = decision.state, decision.who
        actions = decision.request.get("action", []) if decision.kind == "action" else []
        board = encode_board(state, who, self.vocabulary, decision.counters)
        hand_indices, hand_features, hand_mask = encode_hand(state, who, actions, self.vocabulary, self.attachment_ids)
        active_element, alive_elements = character_element_dice(state, who, self.tags_by_character)
        dice_features, dice_mask = encode_dice(state["player"][who].get("dice", []), active_element, alive_elements)
        knowledge_indices, knowledge_features, knowledge_mask = encode_knowledge(tracker, self.vocabulary)
        history_indices, history_features, history_mask = encode_history(tracker, state.get("roundNumber", 0), self.vocabulary)
        action_tokens = encode_actions(decision.kind, decision.request, state, who, board.position_by_object_id, self.vocabulary,
                                       decision.previews, decision.replies,
                                       set(decision.replies_checked or ()) if decision.replies_on_demand else None)

        arrays = {
            "global_features": encode_global(state, who, decision.kind, tracker, decision.end_of_round),
            "character_indices": board.character_indices,
            "character_features": board.character_features,
            "character_mask": board.character_mask,
            "entity_indices": board.entity_indices,
            "entity_features": board.entity_features,
            "entity_mask": board.entity_mask,
            "hand_indices": hand_indices,
            "hand_features": hand_features,
            "hand_mask": hand_mask,
            "dice_features": dice_features,
            "dice_mask": dice_mask,
            "knowledge_indices": knowledge_indices,
            "knowledge_features": knowledge_features,
            "knowledge_mask": knowledge_mask,
            "history_indices": history_indices,
            "history_features": history_features,
            "history_mask": history_mask,
            "action_indices": action_tokens.indices,
            "action_features": action_tokens.features,
            "action_targets": action_tokens.targets,
            "action_mask": action_tokens.mask,
        }
        return EncodedDecision(arrays, action_tokens.request_indices, board.dropped_entities + action_tokens.dropped)
