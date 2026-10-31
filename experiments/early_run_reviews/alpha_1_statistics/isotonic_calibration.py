"""Isotonic recalibration of the value head: a monotone map from raw win chance to observed win rate.

Fitted with pool-adjacent-violators on predictions pre-grouped into fine bins (fast on millions of points).
The shared calibrator is fitted on round-start win chances of learner seats, updates 400-600, even game indexes;
odd game indexes are left for evaluation.
"""

from dataclasses import dataclass

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.seat_selection import trained_learner_rows, win_chance

FINE_BINS = 400
FIRST_ROUND = 1  # round 0 is the mulligan, before any dice are rolled


@dataclass
class IsotonicCalibrator:
    knots: np.ndarray  # raw win chance
    calibrated: np.ndarray  # observed win rate at each knot (non-decreasing)

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.knots, self.calibrated)


def fit_isotonic(raw: np.ndarray, outcomes: np.ndarray) -> IsotonicCalibrator:
    bins = np.clip((raw * FINE_BINS).astype(int), 0, FINE_BINS - 1)
    weights = np.bincount(bins, minlength=FINE_BINS).astype(np.float64)
    sums = np.bincount(bins, weights=outcomes, minlength=FINE_BINS)
    raw_sums = np.bincount(bins, weights=raw, minlength=FINE_BINS)
    used = weights > 0
    blocks = [[sums[index], weights[index], raw_sums[index]] for index in np.flatnonzero(used)]
    merged: list[list[float]] = []
    for block in blocks:  # pool adjacent violators
        merged.append(block)
        while len(merged) > 1 and merged[-2][0] / merged[-2][1] > merged[-1][0] / merged[-1][1]:
            last = merged.pop()
            merged[-1] = [merged[-1][0] + last[0], merged[-1][1] + last[1], merged[-1][2] + last[2]]
    knots = np.array([raw_sum / weight for _, weight, raw_sum in merged])
    calibrated = np.array([total / weight for total, weight, _ in merged])
    return IsotonicCalibrator(knots, calibrated)


def round_start_points(table: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(raw win chance, outcome, row, round) for every round start (round >= 1) of the given rows."""
    values = table["round_values"][rows, FIRST_ROUND:]
    known = ~np.isnan(values)
    row_of_point = np.repeat(rows, known.sum(axis=1))
    round_of_point = (np.nonzero(known)[1] + FIRST_ROUND).astype(np.int16)
    return win_chance(values[known]), table["won"][row_of_point].astype(np.float64), row_of_point, round_of_point


_shared: dict[int, IsotonicCalibrator] = {}


def shared_calibrator(table: dict) -> IsotonicCalibrator:
    if id(table) not in _shared:
        rows = np.flatnonzero(trained_learner_rows(table) & (table["game"] % 2 == 0))
        raw, outcomes, _, _ = round_start_points(table, rows)
        _shared[id(table)] = fit_isotonic(raw, outcomes)
    return _shared[id(table)]
