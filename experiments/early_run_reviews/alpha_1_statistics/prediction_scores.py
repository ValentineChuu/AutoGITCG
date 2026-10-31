"""Scores for probabilistic win predictions: log loss, Brier score, accuracy, AUC, expected calibration error."""

import numpy as np

PROBABILITY_FLOOR = 1e-4
CALIBRATION_BINS = 10


def log_loss(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    predicted = np.clip(predicted, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    return float(-np.mean(outcomes * np.log(predicted) + (1 - outcomes) * np.log(1 - predicted)))


def brier_score(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of the predicted chance; 0.25 = always saying 50%, 0 = perfect."""
    return float(np.mean((predicted - outcomes) ** 2))


def accuracy(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((predicted > 0.5) == (outcomes == 1)))


def area_under_curve(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    """Chance that a random win got a higher prediction than a random loss (0.5 = no skill), ties counted half."""
    order = np.argsort(predicted, kind="mergesort")
    ranks = np.empty(len(predicted))
    sorted_values = predicted[order]
    _, first_index, counts = np.unique(sorted_values, return_index=True, return_counts=True)
    average_ranks = first_index + (counts + 1) / 2
    ranks[order] = np.repeat(average_ranks, counts)
    positives = outcomes == 1
    positive_count, negative_count = positives.sum(), (~positives).sum()
    return float((ranks[positives].sum() - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count))


def expected_calibration_error(predicted: np.ndarray, outcomes: np.ndarray) -> float:
    """Average |predicted - actual| over equal-width bins, weighted by bin size."""
    bins = np.clip((predicted * CALIBRATION_BINS).astype(int), 0, CALIBRATION_BINS - 1)
    error = 0.0
    for number in range(CALIBRATION_BINS):
        in_bin = bins == number
        if in_bin.any():
            error += in_bin.mean() * abs(predicted[in_bin].mean() - outcomes[in_bin].mean())
    return float(error)


def all_scores(predicted: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {"log loss": log_loss(predicted, outcomes), "Brier": brier_score(predicted, outcomes),
            "accuracy": accuracy(predicted, outcomes), "AUC": area_under_curve(predicted, outcomes),
            "calibration error": expected_calibration_error(predicted, outcomes)}
