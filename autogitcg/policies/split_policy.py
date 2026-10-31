"""Two policies sharing one seat: one answers the listed decision kinds, the other everything else. Used to find which
part of a network's play makes it better or worse than another's (e.g. a student's moves with beta 10.8's dice rerolls)."""

from autogitcg.bridge.messages import Decision
from autogitcg.observation.observation_encoder import EncodedDecision
from autogitcg.policies.network_policy import NetworkChoice


class SplitPolicy:
    def __init__(self, main, fallback, main_kinds: set[str]):
        self.main = main
        self.fallback = fallback
        self.main_kinds = main_kinds

    def choose(self, pending: list[tuple[Decision, EncodedDecision]]) -> list[NetworkChoice]:
        main_rows = [row for row, (decision, _) in enumerate(pending) if decision.kind in self.main_kinds]
        fallback_rows = [row for row, (decision, _) in enumerate(pending) if decision.kind not in self.main_kinds]
        choices = [None] * len(pending)
        for policy, rows in ((self.main, main_rows), (self.fallback, fallback_rows)):
            for row, choice in zip(rows, policy.choose([pending[row] for row in rows])):
                choices[row] = choice
        return choices
