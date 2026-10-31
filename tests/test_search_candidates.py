"""Which moves the look-ahead tries (search/decision_search.candidate_choices)."""

import numpy as np

from autogitcg.search.decision_search import candidate_choices
from autogitcg.search.lookahead_settings import LookaheadSettings

PROBABILITIES = np.array([0.50, 0.30, 0.15, 0.04, 0.01])
KINDS = ["useSkill", "useSkill", "switchActive", "playCard", "declareEnd"]


def test_without_every_move_kind_only_the_most_likely_moves_are_tried():
    assert candidate_choices(0, PROBABILITIES, LookaheadSettings(candidates=2), KINDS) == [0, 1]


def test_every_move_kind_adds_the_most_likely_move_of_each_untried_kind():
    settings = LookaheadSettings(candidates=2, every_move_kind=True)
    assert candidate_choices(0, PROBABILITIES, settings, KINDS) == [0, 1, 2, 3, 4]
    assert candidate_choices(1, PROBABILITIES, settings, KINDS) == [1, 0, 2, 3, 4]


def test_a_confident_network_still_skips_the_search():
    settings = LookaheadSettings(every_move_kind=True)
    assert candidate_choices(0, np.array([0.95, 0.03, 0.02]), settings, ["useSkill", "playCard", "declareEnd"]) == []
