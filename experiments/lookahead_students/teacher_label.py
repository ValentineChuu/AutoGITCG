"""One training label from a look-ahead game: an observation of the searching player and the move to learn there.

At searched decisions it keeps every candidate's search score, so training can learn the best-scoring move (not
only the one played, which keeps the network's sampled choice unless another wins by the margin). The unsearched
(confident) decisions only anchor the student where look-ahead has nothing to add (experiments/lookahead_students/notes.md).
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeacherLabel:
    observation: dict[str, np.ndarray]  # compacted encoder arrays (model/observation_batch.compact)
    kind: int  # index in DECISION_KINDS (always "action" for now)
    choice_index: int  # the move played
    probabilities: np.ndarray  # float16: the searching network's own probabilities over the options
    searched: bool
    overridden: bool  # look-ahead picked a different move than the network's own choice
    value_target: float = 0.0  # the game's final reward from this player's side
    won: float = 0.0
    candidate_indices: list[int] = field(default_factory=list)  # searched: the choice indices look-ahead tried
    candidate_scores: list[float | None] = field(default_factory=list)  # searched: each one's mean score
    candidate_redeal_scores: list[list[float | None]] = field(default_factory=list)  # searched: per candidate, per re-deal
