"""Means and rates per training-update bucket, with 95% intervals, for over-training figures."""

import numpy as np

from autogitcg.analysis.confidence import Z_95, wilson_interval

MINIMUM_ROWS = 30


def bucket_means(values: np.ndarray, buckets: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bucket starts, means, 95% half-widths) over the selected, non-NaN rows."""
    selected = selected & ~np.isnan(values)
    starts, means, half_widths = [], [], []
    for bucket in np.unique(buckets[selected]):
        in_bucket = values[selected & (buckets == bucket)]
        if len(in_bucket) >= MINIMUM_ROWS:
            starts.append(bucket)
            means.append(in_bucket.mean())
            half_widths.append(Z_95 * in_bucket.std(ddof=1) / np.sqrt(len(in_bucket)))
    return np.array(starts), np.array(means), np.array(half_widths)


def bucket_rates(outcomes: np.ndarray, buckets: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(bucket starts, rates, lows, highs) of a 0/1 outcome, with Wilson intervals."""
    starts, rates, lows, highs = [], [], [], []
    for bucket in np.unique(buckets[selected]):
        in_bucket = outcomes[selected & (buckets == bucket)]
        if len(in_bucket) >= MINIMUM_ROWS:
            low, high = wilson_interval(float(in_bucket.sum()), len(in_bucket))
            starts.append(bucket)
            rates.append(in_bucket.mean())
            lows.append(low)
            highs.append(high)
    return np.array(starts), np.array(rates), np.array(lows), np.array(highs)
