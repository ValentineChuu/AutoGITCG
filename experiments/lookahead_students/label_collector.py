"""Turns the searcher's final moves in a LookaheadTable into TeacherLabels, completed when each game ends.

Every searched decision becomes a label; of the unsearched (confident) action decisions, only `unsearched_share`
are kept, since they teach nothing new and only anchor the student. Labels of failed games are dropped.
"""

import random

import numpy as np

from autogitcg.bridge.messages import Decision
from experiments.lookahead_students.teacher_label import TeacherLabel
from autogitcg.model.observation_batch import compact
from autogitcg.observation.observation_encoder import EncodedDecision
from autogitcg.policies.network_policy import NetworkChoice
from autogitcg.search.lookahead_table import SearchResult


class LabelCollector:
    def __init__(self, unsearched_share: float, rng: random.Random):
        self.unsearched_share = unsearched_share
        self.rng = rng
        self.open: dict[int, list[TeacherLabel]] = {}  # game id -> labels so far

    def on_searcher_move(self, decision: Decision, encoded: EncodedDecision, own_choice: NetworkChoice,
                         chosen_index: int, search: SearchResult | None) -> None:
        searched = search is not None
        if decision.kind != "action" or (not searched and self.rng.random() >= self.unsearched_share):
            return
        self.open.setdefault(decision.game_id, []).append(TeacherLabel(
            compact(encoded.arrays), own_choice.kind, chosen_index, own_choice.probabilities.astype(np.float16),
            searched, chosen_index != own_choice.choice_index,
            candidate_indices=search.candidate_indices if searched else [],
            candidate_scores=search.mean_scores if searched else [],
            candidate_redeal_scores=search.paired_scores if searched else []))

    def finish(self, game_id: int, row: dict) -> list[TeacherLabel]:
        """The game's labels with their outcome (`row` from LookaheadTable.step); none if the game failed."""
        labels = self.open.pop(game_id, [])
        if "failure" in row:
            return []
        for label in labels:
            label.value_target = row["searcher_reward"]
            label.won = float(row["searcher_won"])
        return labels
