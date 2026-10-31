"""Per-seat behaviour measures (one float per row), shared by the evolution, win/loss and heuristic sections.

Most are per round, because games got shorter over training (8-10 rounds early, about 6 late) and per-game
counts would mix "plays differently" with "plays fewer rounds".
"""

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.seat_selection import cards_played, skills_used


def _per_round(table: dict, counts: np.ndarray) -> np.ndarray:
    return counts / np.maximum(table["rounds"], 1)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator.astype(np.float64), denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


BEHAVIOUR_MEASURES = {
    "normal attacks / round": lambda table: _per_round(table, table["skill_normal"]),
    "elemental skills / round": lambda table: _per_round(table, table["skill_elemental"]),
    "bursts / round": lambda table: _per_round(table, table["skill_burst"]),
    "burst share of skills": lambda table: _ratio(table["skill_burst"], skills_used(table)),
    "cards played / round": lambda table: _per_round(table, cards_played(table)),
    "talents played / game": lambda table: table["card_talent"].astype(np.float64),
    "tunings / round": lambda table: _per_round(table, table["tunings"]),
    "voluntary switches / round": lambda table: _per_round(table, table["switches"]),
    "actions / round": lambda table: _per_round(table, table["actions"]),
    "share of rounds it ended first": lambda table: _ratio(table["end_first_rounds"], table["rounds"]),
    "nominal dice spent / round": lambda table: _per_round(table, table["dice_spent"]),
    "dice rerolled / reroll": lambda table: _ratio(table["dice_rerolled"], table["reroll_decisions"]),
    "mulligan cards changed": lambda table: table["mulligan_changed"].astype(np.float64),
}

OUTCOME_MEASURES = {
    **BEHAVIOUR_MEASURES,
    "bursts / game": lambda table: table["skill_burst"].astype(np.float64),
    "cards played / game": lambda table: cards_played(table).astype(np.float64),
    "nominal dice spent / game": lambda table: table["dice_spent"].astype(np.float64),
    "own characters defeated": lambda table: table["own_defeated"].astype(np.float64),
    "deck: action cards": lambda table: table["action_cards"].astype(np.float64),
    "deck: distinct cards": lambda table: table["distinct_cards"].astype(np.float64),
    "deck: talent copies": lambda table: table["deck_talents"].astype(np.float64),
    "deck: largest same-element group": lambda table: table["largest_element_group"].astype(np.float64),
    "deck: mean card cost": lambda table: table["deck_mean_cost"].astype(np.float64),
}
