"""Paired results of a position's compared moves: every guessed deal is played once with each move, so comparing two
moves deal by deal cancels the luck the deal brings (common random numbers)."""

import math
from dataclasses import dataclass, field


@dataclass
class MoveComparison:
    move_count: int
    results: list[dict[int, float]] = field(default_factory=list)  # per move: deal -> +1 win / -1 loss

    def __post_init__(self):
        self.results = [{} for _ in range(self.move_count)]

    def add(self, move: int, deal: int, result: float) -> None:
        self.results[move][deal] = result

    def means(self) -> list[float | None]:
        return [sum(results.values()) / len(results) if results else None for results in self.results]

    def best(self) -> int:
        means = self.means()
        return max(range(self.move_count), key=lambda move: means[move] if means[move] is not None else -math.inf)

    def paired_difference(self, move: int, other: int) -> tuple[float, float, int]:
        """(mean of move - other over deals both finished, its standard error, deals)."""
        deals = self.results[move].keys() & self.results[other].keys()
        differences = [self.results[move][deal] - self.results[other][deal] for deal in deals]
        if len(differences) < 2:
            return 0.0, math.inf, len(differences)
        mean = sum(differences) / len(differences)
        variance = sum((difference - mean) ** 2 for difference in differences) / (len(differences) - 1)
        return mean, math.sqrt(variance / len(differences)), len(differences)

    def separated(self, separation: float) -> bool:
        """The best move leads every other by `separation` standard errors (deal by deal)."""
        best = self.best()
        for move in range(self.move_count):
            if move == best:
                continue
            mean, error, _ = self.paired_difference(best, move)
            if mean <= separation * error:
                return False
        return True

    def all_identical(self) -> bool:
        """Every move had the same result on every deal so far (e.g. a position lost whatever is played)."""
        deals = set.intersection(*(set(results) for results in self.results)) if self.results else set()
        return bool(deals) and all(len({results[deal] for results in self.results}) == 1 for deal in deals)

    def summary(self) -> list[dict]:
        """Per move: mean result, deals, and its paired difference to the best move (with standard error)."""
        best = self.best()
        rows = []
        for move, mean in enumerate(self.means()):
            difference, error, deals = self.paired_difference(move, best) if move != best else (0.0, 0.0, len(self.results[move]))
            rows.append({"mean": mean, "deals": len(self.results[move]), "difference_to_best": difference,
                         "difference_error": error if math.isfinite(error) else None, "paired_deals": deals})
        return rows
