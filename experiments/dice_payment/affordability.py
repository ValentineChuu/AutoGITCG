"""Paying a cost with dice (GITCG rules): void = any dice, an element = that element or omni, aligned = all of one
element (omni fills in). Energy is checked against the character; legend points are ignored."""

from collections import Counter
from itertools import combinations

from autogitcg.observation.enums import OMNI_DICE

VOID, ALIGNED, ENERGY, LEGEND = 0, 8, 9, 10


def can_pay(cost: list[dict], dice: list[int], energy: int) -> bool:
    if any(entry["type"] == ENERGY and entry["count"] > energy for entry in cost):
        return False
    return any(_fits(cost, payment) for payment in payments(cost, dice))


def payments(cost: list[dict], dice: list[int]) -> list[tuple[int, ...]]:
    """Every distinct multiset of dice that pays the cost exactly."""
    size = sum(entry["count"] for entry in cost if entry["type"] not in (ENERGY, LEGEND))
    if size > len(dice):
        return []
    return [payment for payment in set(tuple(sorted(choice)) for choice in combinations(dice, size)) if _fits(cost, payment)]


def remaining(dice: list[int], paid: list[int] | tuple[int, ...]) -> list[int]:
    left = Counter(dice)
    left.subtract(paid)
    return sorted(left.elements())


def _fits(cost: list[dict], payment: tuple[int, ...]) -> bool:
    left = Counter(payment)
    for entry in cost:  # specific elements first, from their own element, then omni
        if 1 <= entry["type"] <= 7:
            own = min(left[entry["type"]], entry["count"])
            left[entry["type"]] -= own
            if left[OMNI_DICE] < entry["count"] - own:
                return False
            left[OMNI_DICE] -= entry["count"] - own
    for entry in cost:
        if entry["type"] == ALIGNED and entry["count"]:
            element = max(range(1, 8), key=lambda each: left[each])
            own = min(left[element], entry["count"])
            if left[OMNI_DICE] < entry["count"] - own:
                return False
            left[element] -= own
            left[OMNI_DICE] -= entry["count"] - own
    void = sum(entry["count"] for entry in cost if entry["type"] == VOID)
    return sum(left.values()) == void
