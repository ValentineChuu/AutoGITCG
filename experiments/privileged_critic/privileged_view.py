"""A training-only view of a decision with the opponent's hidden information revealed (their hand's cards and their
dice faces), for a critic that judges positions better than the player can see. Needs games created with
`privileged=True` (bridge/engine_pool.py). The policy never sees this view.

The encoding is the normal one with two changes: the opponent's hand cards carry their identities, and the global
token gets the opponent's dice counts per type appended (so a critic grown from a normal network starts with zero
weights for them, network_config.grown_feature_widths).
"""

import copy

import numpy as np

from autogitcg.bridge.messages import Decision
from autogitcg.observation.enums import DICE_TYPE_COUNT
from autogitcg.observation.game_tracker import GameTracker
from autogitcg.observation.observation_encoder import EncodedDecision, ObservationEncoder

PRIVILEGED_GLOBAL_EXTRA = DICE_TYPE_COUNT  # opponent dice per type, /8 like the player's own


def revealed_state(state: dict, who: int) -> dict:
    """The decision's state with the opponent's hand definitions and dice filled in from opponentPrivate."""
    private = state["opponentPrivate"]
    revealed = copy.deepcopy(state)
    opponent = revealed["player"][1 - who]
    for card, definition_id in zip(opponent.get("handCard", []), private["hand"]):
        card["definitionId"] = definition_id
    opponent["dice"] = list(private["dice"])
    return revealed


def encode_privileged(encoder: ObservationEncoder, decision: Decision, tracker: GameTracker) -> EncodedDecision:
    state = revealed_state(decision.state, decision.who)
    encoded = encoder.encode(Decision(decision.game_id, decision.who, decision.kind, decision.request, state, decision.counters,
                                      decision.previews, decision.replies), tracker)
    opponent_dice = np.zeros(PRIVILEGED_GLOBAL_EXTRA, dtype=np.float32)
    for die in state["player"][1 - decision.who]["dice"]:
        opponent_dice[die] += 1 / 8
    encoded.arrays["global_features"] = np.concatenate([encoded.arrays["global_features"], opponent_dice])
    return encoded
