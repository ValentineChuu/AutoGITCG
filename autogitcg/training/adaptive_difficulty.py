"""Shifts training toward the matchup categories where the learner plays worst.

Skill is measured with mirror pairs against a frozen reference opponent (alpha 2.40): the learner plays a
matchup once with each deck, same seed, and the pair's mean score (0, 0.5 or 1) cancels out which deck is
stronger. (Until alpha 3 the focus side's plain win rate was used; in self-play that mostly measured deck
strength, so weights barely moved.) Each category keeps a moving average of its pair scores.

A category's hardness = exp((mean score over categories - its score) / TEMPERATURE), clipped to
[1 / MAXIMUM_BOOST, MAXIMUM_BOOST]: 10 points below average about doubles it. Sampling weight =
(1 - ADAPTIVE_SHARE) * base weight + ADAPTIVE_SHARE * (base weight * hardness, renormalized).
"""

import math

ADAPTIVE_SHARE = 0.5
MOVING_AVERAGE_RATE = 0.05  # per finished pair in that category
PRIOR_SCORE = 0.5  # the learner starts as the reference
TEMPERATURE = 0.15
MAXIMUM_BOOST = 4.0


class AdaptiveDifficulty:
    def __init__(self, base_weights: dict[str, float]):
        self.base_weights = base_weights
        self.scores = {category: PRIOR_SCORE for category in base_weights}

    def record(self, category: str, pair_score: float) -> None:
        if category in self.scores:
            self.scores[category] += MOVING_AVERAGE_RATE * (pair_score - self.scores[category])

    def weights(self) -> dict[str, float]:
        average = sum(self.base_weights[category] * score for category, score in self.scores.items())
        hardness = {category: self.base_weights[category] * _clipped_boost((average - self.scores[category]) / TEMPERATURE)
                    for category in self.base_weights}
        hardness_total = sum(hardness.values())
        return {category: (1 - ADAPTIVE_SHARE) * self.base_weights[category] + ADAPTIVE_SHARE * hardness[category] / hardness_total
                for category in self.base_weights}


def _clipped_boost(exponent: float) -> float:
    return min(max(math.exp(exponent), 1 / MAXIMUM_BOOST), MAXIMUM_BOOST)
