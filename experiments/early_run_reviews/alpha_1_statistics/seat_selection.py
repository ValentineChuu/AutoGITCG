"""Common row selections and derived columns on the per-seat table."""

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.game_rows import CONTROLLERS, OPPONENTS, SOURCES

TRAINED_FROM_UPDATE = 400  # "trained" behaviour: updates 400-600, where the benchmark has plateaued
BUCKET_UPDATES = 25


def learner_rows(table: dict) -> np.ndarray:
    return table["controller"] == CONTROLLERS.index("learner")


def trained_learner_rows(table: dict) -> np.ndarray:
    return learner_rows(table) & (table["update"] >= TRAINED_FROM_UPDATE)


def opponent_is(table: dict, opponent: str) -> np.ndarray:
    return table["opponent"] == OPPONENTS.index(opponent)


def controller_is(table: dict, controller: str) -> np.ndarray:
    return table["controller"] == CONTROLLERS.index(controller)


def source_is(table: dict, source: str) -> np.ndarray:
    return table["source"] == SOURCES.index(source)


def win_chance(values: np.ndarray) -> np.ndarray:
    """Value head output in [-1, 1] -> win chance in [0, 1]."""
    return (values.astype(np.float64) + 1) / 2


def cards_played(table: dict) -> np.ndarray:
    return table["card_talent"] + table["card_equipment"] + table["card_support"] + table["card_event"]


def skills_used(table: dict) -> np.ndarray:
    return table["skill_normal"] + table["skill_elemental"] + table["skill_burst"] + table["skill_other"]


def opponent_row(rows: np.ndarray) -> np.ndarray:
    """Row index of the other seat in the same game."""
    return rows ^ 1


def update_buckets(table: dict) -> np.ndarray:
    """Bucket start per row; a final partial bucket (e.g. the lone update 600) joins the previous one."""
    last_full_bucket = (int(table["update"].max()) + 1) // BUCKET_UPDATES * BUCKET_UPDATES - BUCKET_UPDATES
    return np.minimum(table["update"] // BUCKET_UPDATES * BUCKET_UPDATES, last_full_bucket)


def matchup_label(table: dict, rows: np.ndarray) -> np.ndarray:
    """'source / opponent' text per row, for grouping."""
    return np.array([f"{SOURCES[source]} / {OPPONENTS[opponent]}"
                     for source, opponent in zip(table["source"][rows], table["opponent"][rows])])
