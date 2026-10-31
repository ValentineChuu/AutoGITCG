"""Confidence intervals for rates and means (normal approximations; samples here are thousands of games)."""

import math

import numpy as np

Z_95 = 1.96


def wilson_interval(successes: float, total: float) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if total == 0:
        return math.nan, math.nan
    rate = successes / total
    denominator = 1 + Z_95 ** 2 / total
    center = (rate + Z_95 ** 2 / (2 * total)) / denominator
    half_width = Z_95 * math.sqrt(rate * (1 - rate) / total + Z_95 ** 2 / (4 * total ** 2)) / denominator
    return center - half_width, center + half_width


def rate_with_interval(outcomes: np.ndarray) -> tuple[float, float, float, int]:
    """(rate, low, high, count) for a 0/1 array."""
    count = len(outcomes)
    if count == 0:
        return math.nan, math.nan, math.nan, 0
    low, high = wilson_interval(float(outcomes.sum()), count)
    return float(outcomes.mean()), low, high, count


def mean_with_interval(values: np.ndarray) -> tuple[float, float]:
    """(mean, 95% half-width)."""
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return math.nan, math.nan
    return float(values.mean()), Z_95 * float(values.std(ddof=1)) / math.sqrt(len(values))


def difference_of_means(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """(mean(first) - mean(second), 95% half-width), unpaired."""
    first, second = first[~np.isnan(first)], second[~np.isnan(second)]
    if len(first) < 2 or len(second) < 2:
        return math.nan, math.nan
    standard_error = math.sqrt(first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second))
    return float(first.mean() - second.mean()), Z_95 * standard_error


def format_rate(rate: float, low: float, high: float) -> str:
    return f"{rate:.3f} [{low:.3f}, {high:.3f}]"
