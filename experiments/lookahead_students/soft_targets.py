"""Soft move targets that trust the search only as far as its noise allows (label rule "soft").

Each candidate's advantage over the network's own choice is the paired mean difference across re-deals, shrunk
toward 0 by its share of signal, mean² / (mean² + standard error²): a near-tie (the usual case; median gap 0.57
standard errors on teacher_deep_01) barely moves the network's probabilities, a clear win moves them strongly.
The network's probabilities are then multiplied by exp(advantage / temperature) and renormalized, so the student
stays anchored to its own play everywhere and is never taught to avoid its own move (try 3's collapse).
"""

import numpy as np

from experiments.lookahead_students.teacher_label import TeacherLabel

MINIMUM_PAIRED_REDEALS = 3


def shrunk_advantage(differences: list[float]) -> float:
    """The paired mean difference, shrunk by its share of signal over noise."""
    mean = float(np.mean(differences))
    squared_error = float(np.var(differences, ddof=1)) / len(differences)
    return mean * mean * mean / (mean * mean + squared_error) if mean * mean + squared_error > 0 else 0.0


def soft_probabilities(label: TeacherLabel, temperature: float) -> np.ndarray:
    """The network's own probabilities, each paired candidate reweighted by its shrunk advantage."""
    probabilities = label.probabilities.astype(np.float64)
    paired = getattr(label, "candidate_redeal_scores", [])
    for candidate in range(1, len(paired)):
        differences = [score - own for score, own in zip(paired[candidate], paired[0]) if score is not None and own is not None]
        if len(differences) < MINIMUM_PAIRED_REDEALS:
            continue
        advantage = shrunk_advantage(differences)
        probabilities[label.candidate_indices[candidate]] *= np.exp(np.clip(advantage / temperature, -20, 20))
    return (probabilities / probabilities.sum()).astype(np.float32)
