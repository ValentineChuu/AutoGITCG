"""Sample a network response from a decision-local seed, independent of asynchronous batch order."""

import random

from autogitcg.bridge.messages import Decision
from autogitcg.observation.observation_encoder import EncodedDecision
from autogitcg.policies.network_responses import response_from_choice


def seeded_response(decision: Decision, encoded: EncodedDecision, probabilities, seed: int) -> dict:
    random_generator = random.Random(seed)
    if decision.kind in ("action", "chooseActive", "selectCard"):
        choice_index = random_generator.choices(range(len(encoded.request_indices)),
                                                weights=probabilities, k=1)[0]
        bits = [0.0] * 8
    else:
        choice_index = 0
        bits = [float(random_generator.random() < probability) for probability in probabilities]
        bits += [0.0] * (8 - len(bits))
    return response_from_choice(decision, encoded, choice_index, bits)
