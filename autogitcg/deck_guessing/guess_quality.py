"""How good a guess of the hidden cards is, against the opponent's real remaining cards."""

import math
from collections import Counter

SMOOTHING = 0.01  # probability mass per card kind no guess named, so a miss costs a finite amount


def overlap_share(expected: dict[int, float], truth: Counter) -> float:
    """Share of the real hidden cards the guess covers: sum over cards of min(expected, real) / real total."""
    total = sum(truth.values())
    return sum(min(expected.get(card, 0.0), count) for card, count in truth.items()) / total if total else 0.0


def mean_log_likelihood(expected: dict[int, float], truth: Counter, card_kinds: int) -> float:
    """Average log-probability of each real hidden card when one is drawn from the guess (smoothed over every
    card kind), the score of "which card do they hold next"."""
    size = sum(expected.values())
    total = sum(truth.values())
    if not total:
        return 0.0
    def probability(card: int) -> float:
        share = expected.get(card, 0.0) / size if size else 0.0
        return (1 - SMOOTHING) * share + SMOOTHING / card_kinds
    return sum(count * math.log(probability(card)) for card, count in truth.items()) / total
