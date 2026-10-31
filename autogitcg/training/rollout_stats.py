"""Counters from rollouts: games and learner wins per opponent, round-limit games, failures, decisions."""

from collections import Counter
from dataclasses import dataclass, field

from autogitcg.training.opponent_league import SELF


@dataclass
class RolloutStats:
    games: Counter = field(default_factory=Counter)  # finished games per opponent
    learner_wins: Counter = field(default_factory=Counter)  # per opponent (self-play excluded)
    round_limit: int = 0
    failures: Counter = field(default_factory=Counter)
    failure_examples: dict[str, str] = field(default_factory=dict)  # first full report per failure summary
    decisions: int = 0  # all seats

    def merge(self, other: "RolloutStats") -> None:
        self.games.update(other.games)
        self.learner_wins.update(other.learner_wins)
        self.round_limit += other.round_limit
        self.failures.update(other.failures)
        for summary, report in other.failure_examples.items():
            self.failure_examples.setdefault(summary, report)
        self.decisions += other.decisions

    def win_rates(self) -> dict[str, float]:
        return {opponent: self.learner_wins[opponent] / count for opponent, count in self.games.items() if opponent != SELF and count}
