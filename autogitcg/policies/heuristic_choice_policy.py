"""The scripted heuristic (heuristic_policy.py) behind the network-policy interface, so a SplitPolicy can hand it some
decision kinds (e.g. beta 10.8 playing everything but rerolls). Its choices carry no network outputs."""

import random

import numpy as np

from autogitcg.bridge.messages import Decision
from autogitcg.observation.enums import DECISION_KINDS
from autogitcg.observation.observation_encoder import EncodedDecision
from autogitcg.policies.heuristic_policy import HeuristicPolicy
from autogitcg.policies.network_policy import NetworkChoice
from autogitcg.model.policy_distribution import CHOICE_BITS


class HeuristicChoicePolicy:
    def __init__(self, seed: int = 0):
        self.heuristic = HeuristicPolicy()
        self.rng = random.Random(seed)

    def choose(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        return [NetworkChoice(self.heuristic.respond(decision, self.rng), DECISION_KINDS.index(decision.kind), 0,
                              np.zeros(CHOICE_BITS, dtype=np.float32), 0.0, 0.0, 0.0, 0.5, 0.0, np.zeros(0, dtype=np.float32))
                for decision, _ in pending]
