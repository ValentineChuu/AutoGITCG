"""All recorded labels of one or more runs, stacked into arrays, split into training and validation by shard
(a shard's labels come from the same games, so splitting by label would leak games into validation).

The move to learn at a searched decision depends on the label rule:
- "best": the candidate with the highest mean search score (the played move when scores are missing);
- "played": the move actually played (the network's own sampled choice unless another won by the margin);
- "overrides": only where look-ahead overrode the network by the margin (the played move); every other searched
  decision is anchored like an unsearched one (epsilon 2 learned noise from the other two rules).
- "confident": only where a candidate beats the network's own choice re-deal by re-deal with the paired mean
  difference still above 0 after subtracting two standard errors (and at least the override margin); everything
  else is anchored. Built for the fair search, where most "best" picks were near-ties (median gap 0.001).
- "evaluator": no move is learned anywhere (all anchored); used to train the win-chance output on search scores
  (StudentOptions.search_win_weight, like a chess engine's evaluation trained on search results), or the move-value
  output on each candidate's search score (StudentOptions.move_value_weight).
- "improved": no hard move; at searched decisions the anchor distribution itself is improved: each candidate's
  probability is multiplied by exp((its score - the network's own choice's score) / temperature) and renormalized,
  so better-scoring moves gain in proportion (MuZero-style policy improvement) without becoming certain.
- "soft": like "improved", but each advantage is first shrunk by how sure search is of it (soft_targets.py), so
  near-ties leave the network's probabilities alone and clear wins move them strongly.
With `minimum_probability`, a move to learn that the searching network found less likely than that is anchored
instead: a quarter of teacher_01's overrides were moves beta 10.8 gave under 0.4%, and learning them wrecked students
(30-37% vs beta 10.8, from about 51%).
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.lookahead_students.label_shards import LABEL_FOLDER, read_shard
from experiments.lookahead_students.soft_targets import soft_probabilities
from experiments.lookahead_students.teacher_label import TeacherLabel
from autogitcg.model.observation_batch import stack
from autogitcg.observation.encode_actions import MAX_ACTIONS


@dataclass
class LabelSet:
    observations: dict[str, np.ndarray]  # compacted, stacked
    kinds: np.ndarray  # int64
    choice_indices: np.ndarray  # int64: the move to learn (see the label rule)
    probabilities: np.ndarray  # float32 (rows, MAX_ACTIONS): the searching network's own probabilities
    searched: np.ndarray  # float32 0/1
    imitated: np.ndarray  # float32 0/1: the student learns the move here (else it is anchored to the probabilities)
    overridden: np.ndarray  # float32 0/1: the move to learn isn't the network's own choice
    value_targets: np.ndarray  # float32
    won: np.ndarray  # float32
    target_masks: np.ndarray  # uint8 (rows, MAX_ACTIONS): the move to learn and every action token identical to it
    search_win_targets: np.ndarray  # float32: win chance by search for the played move ((mean score + 1) / 2), -1 if none
    move_value_targets: np.ndarray  # float32 (rows, MAX_ACTIONS): each scored candidate's mean search score, NaN elsewhere
    search_value_targets: np.ndarray  # float32: the played move's mean search score (value scale with score_by value), NaN if none

    def __len__(self) -> int:
        return len(self.kinds)

    def rows(self, indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """(observations, targets) for the given rows."""
        observations = {name: array[indices] for name, array in self.observations.items()}
        targets = {"kinds": self.kinds[indices], "choice_indices": self.choice_indices[indices],
                   "probabilities": self.probabilities[indices], "searched": self.searched[indices], "imitated": self.imitated[indices],
                   "target_masks": self.target_masks[indices], "search_win_targets": self.search_win_targets[indices],
                   "move_value_targets": self.move_value_targets[indices], "search_value_targets": self.search_value_targets[indices],
                   "overridden": self.overridden[indices], "value_targets": self.value_targets[indices], "won": self.won[indices]}
        return observations, targets


LABEL_RULES = ("best", "played", "overrides", "improved", "soft", "confident", "evaluator")
CONFIDENT_MARGIN = 0.05
MINIMUM_PAIRED_REDEALS = 3


def confident_target(label: TeacherLabel) -> int | None:
    """The candidate (choice index) that beats the network's own choice with confidence (see "confident"), if any."""
    paired = getattr(label, "candidate_redeal_scores", [])
    if not paired:
        return None
    best_index, best_lower_bound = None, 0.0
    for candidate in range(1, len(paired)):
        differences = [score - own for score, own in zip(paired[candidate], paired[0]) if score is not None and own is not None]
        if len(differences) < MINIMUM_PAIRED_REDEALS:
            continue
        mean = float(np.mean(differences))
        lower_bound = mean - 2 * float(np.std(differences, ddof=1)) / np.sqrt(len(differences))
        if mean >= CONFIDENT_MARGIN and lower_bound > best_lower_bound:
            best_index, best_lower_bound = label.candidate_indices[candidate], lower_bound
    return best_index


def search_win_target(label: TeacherLabel) -> float:
    """The win chance search found for the move actually played: mean of its copies' scores (2p - 1, or ±1 for
    finished copies) mapped back to a probability. -1 when the decision wasn't searched or its copies all failed."""
    paired = getattr(label, "candidate_redeal_scores", [])
    if not paired or label.choice_index not in label.candidate_indices:
        return -1.0
    scores = [score for score in paired[label.candidate_indices.index(label.choice_index)] if score is not None]
    return (float(np.mean(scores)) + 1) / 2 if scores else -1.0


def search_value_target(label: TeacherLabel) -> float:
    """The played move's mean search score from the deciding player's side, NaN when it has none."""
    scores = getattr(label, "candidate_scores", [])
    if not scores or label.choice_index not in label.candidate_indices:
        return float("nan")
    score = scores[label.candidate_indices.index(label.choice_index)]
    return float("nan") if score is None else float(score)


def move_value_targets(label: TeacherLabel) -> np.ndarray:
    """Each scored candidate's mean search score (2p - 1 from the deciding player's side) at its action token."""
    targets = np.full(MAX_ACTIONS, np.nan, dtype=np.float32)
    for index, score in zip(label.candidate_indices, getattr(label, "candidate_scores", [])):
        if score is not None:
            targets[index] = score
    return targets


def equivalent_action_mask(observation: dict[str, np.ndarray], choice_index: int) -> np.ndarray:
    """1 for the chosen action token and every token encoded identically (e.g. two copies of the same card)."""
    mask = np.zeros(MAX_ACTIONS, dtype=np.uint8)
    present = observation["action_mask"] > 0
    for index in np.flatnonzero(present):
        if all(np.array_equal(observation[name][index], observation[name][choice_index])
               for name in ("action_indices", "action_features", "action_targets")):
            mask[index] = 1
    return mask


def improved_probabilities(label: TeacherLabel, temperature: float) -> np.ndarray:
    """The network's own probabilities, with each scored candidate reweighted by its search score (see "improved")."""
    probabilities = label.probabilities.astype(np.float64)
    scores = getattr(label, "candidate_scores", [])
    if not scores or scores[0] is None:
        return probabilities.astype(np.float32)
    for index, score in zip(label.candidate_indices, scores):
        if score is not None:
            probabilities[index] *= np.exp(np.clip((score - scores[0]) / temperature, -20, 20))
    return (probabilities / probabilities.sum()).astype(np.float32)


def target_and_overridden(label: TeacherLabel, rule: str) -> tuple[int, bool]:
    """(the move to learn, whether it differs from the network's own choice)."""
    scores = getattr(label, "candidate_scores", [])  # labels recorded before scores were kept have none
    if rule == "confident":
        target = confident_target(label)
        return (target, True) if target is not None else (label.choice_index, False)
    if rule in ("played", "overrides") or not scores or scores[0] is None:
        return label.choice_index, label.overridden
    best = max(range(len(scores)), key=lambda index: scores[index] if scores[index] is not None else -np.inf)
    return label.candidate_indices[best], best != 0


def label_set(labels: list[TeacherLabel], rule: str = "best", minimum_probability: float = 0.0,
              temperature: float = 0.1) -> LabelSet:
    probabilities = np.zeros((len(labels), MAX_ACTIONS), dtype=np.float32)
    for row, label in enumerate(labels):
        own = (improved_probabilities(label, temperature) if rule == "improved"
               else soft_probabilities(label, temperature) if rule == "soft" else label.probabilities)
        probabilities[row, :len(own)] = own
    targets = [target_and_overridden(label, rule) for label in labels]
    searched = np.array([label.searched for label in labels], dtype=np.float32)
    plausible = np.array([float(label.probabilities[target]) >= minimum_probability for label, (target, _) in zip(labels, targets)],
                         dtype=np.float32)
    overridden = np.array([overridden for _, overridden in targets], dtype=np.float32)
    return LabelSet(
        stack([label.observation for label in labels]),
        np.array([label.kind for label in labels], dtype=np.int64),
        np.array([target for target, _ in targets], dtype=np.int64),
        probabilities,
        searched,
        np.zeros_like(searched) if rule in ("improved", "soft", "evaluator") else (overridden if rule in ("overrides", "confident") else searched) * plausible,
        overridden,
        np.array([label.value_target for label in labels], dtype=np.float32),
        np.array([label.won for label in labels], dtype=np.float32),
        np.stack([equivalent_action_mask(label.observation, target) for label, (target, _) in zip(labels, targets)]),
        np.array([search_win_target(label) for label in labels], dtype=np.float32),
        np.stack([move_value_targets(label) for label in labels]),
        np.array([search_value_target(label) for label in labels], dtype=np.float32),
    )


def load_label_sets(runs: list[Path], validation_share: float, seed: int, rule: str = "best",
                    minimum_probability: float = 0.0, temperature: float = 0.1) -> tuple[LabelSet, LabelSet]:
    """(training, validation). At least one shard goes to validation when there are two or more."""
    shards = sorted(path for run in runs for path in (run / LABEL_FOLDER).glob("*.pkl.gz"))
    if len(shards) < 2:
        raise ValueError(f"need at least 2 label shards, found {len(shards)} in {runs}")
    random.Random(seed).shuffle(shards)
    validation_count = max(1, round(len(shards) * validation_share))
    validation = [label for path in shards[:validation_count] for label in read_shard(path)]
    training = [label for path in shards[validation_count:] for label in read_shard(path)]
    return (label_set(training, rule, minimum_probability, temperature),
            label_set(validation, rule, minimum_probability, temperature))
