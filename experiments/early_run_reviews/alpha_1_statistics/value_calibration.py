"""How trustworthy the value head's win estimate is, and what that means for a concede rule.

Points are round starts (the seat's first decision of each round >= 1), learner seats, updates 400-600.
The isotonic calibrator is fitted on even game indexes and every score here is on odd game indexes.
"""

from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.figure_style import NEUTRAL_COLOR, TEXT_COLOR, plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.game_rows import OPPONENTS, SOURCES
from experiments.early_run_reviews.alpha_1_statistics.isotonic_calibration import round_start_points, shared_calibrator
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.prediction_scores import all_scores, brier_score, expected_calibration_error
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import opponent_is, source_is, trained_learner_rows

RELIABILITY_BINS = 20
CONCEDE_THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20)
ROUND_GROUPS = ((1, 2), (3, 4), (5, 6), (7, 15))


def report_value_calibration(table: dict, folder: Path, tables: TableCollector) -> None:
    calibrator = shared_calibrator(table)
    held_out = trained_learner_rows(table) & (table["game"] % 2 == 1)
    raw, outcomes, rows, rounds = round_start_points(table, np.flatnonzero(held_out))
    calibrated = calibrator(raw)
    _score_table(table, raw, calibrated, outcomes, rows, rounds, tables)
    _calibrator_table(calibrator, tables)
    _plot(table, raw, calibrated, outcomes, rows, rounds, calibrator, folder)
    _concede_table(table, calibrator, held_out, tables)


def _score_table(table, raw, calibrated, outcomes, rows, rounds, tables) -> None:
    groups = {"all": np.ones(len(raw), dtype=bool)}
    for opponent in OPPONENTS:
        groups[f"vs {opponent}"] = table["opponent"][rows] == OPPONENTS.index(opponent)
    for source in SOURCES:
        groups[f"source {source}"] = table["source"][rows] == SOURCES.index(source)
    groups["stage, learner holds the collection deck"] = (source_is(table, "stage") & (table["stage_side"] == 0))[rows]
    groups["NPC mode (knows opponent deck)"] = table["knows_opponent_deck"][rows] == 1
    groups["blind mode"] = table["knows_opponent_deck"][rows] == 0
    for first, last in ROUND_GROUPS:
        groups[f"rounds {first}-{last}"] = (rounds >= first) & (rounds <= last)
    result = []
    for label, selected in groups.items():
        raw_scores = all_scores(raw[selected], outcomes[selected])
        result.append([label, f"{int(selected.sum()):,}", f"{raw[selected].mean():.3f}", f"{outcomes[selected].mean():.3f}",
                       f"{raw_scores['Brier']:.3f}", f"{brier_score(calibrated[selected], outcomes[selected]):.3f}",
                       f"{raw_scores['calibration error']:.3f}",
                       f"{expected_calibration_error(calibrated[selected], outcomes[selected]):.3f}",
                       f"{raw_scores['AUC']:.3f}"])
    tables.add("Value head calibration (held-out round starts, updates 400-600)",
               ["points", "round starts", "mean predicted", "actual win rate", "Brier raw", "Brier calibrated",
                "calibration error raw", "calibration error calibrated", "AUC"], result,
               note="One point per learner seat per round start. The calibrator is one isotonic fit over all "
                    "opponents (fitted on other games); a per-opponent fit would remove the remaining bias.")


def _calibrator_table(calibrator, tables) -> None:
    points = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98)
    tables.add("Isotonic map: raw value-head win chance -> observed win rate",
               ["raw"] + [f"{point:.2f}" for point in points],
               [["observed"] + [f"{value:.3f}" for value in calibrator(np.array(points))]])


def _reliability(predicted: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.clip((predicted * RELIABILITY_BINS).astype(int), 0, RELIABILITY_BINS - 1)
    centers, rates = [], []
    for number in range(RELIABILITY_BINS):
        in_bin = bins == number
        if in_bin.sum() >= 200:
            centers.append(predicted[in_bin].mean())
            rates.append(outcomes[in_bin].mean())
    return np.array(centers), np.array(rates)


def _plot(table, raw, calibrated, outcomes, rows, rounds, calibrator, folder) -> None:
    figure, (opponent_axis, round_axis, histogram_axis) = plt.subplots(1, 3, figsize=(16, 5))
    for axis in (opponent_axis, round_axis):
        axis.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="perfect calibration")
        axis.set_xlabel("predicted win chance (raw value head)")
        axis.set_ylabel("actual win rate")
    for position, opponent in enumerate(OPPONENTS):
        selected = table["opponent"][rows] == OPPONENTS.index(opponent)
        opponent_axis.plot(*_reliability(raw[selected], outcomes[selected]), marker="o", color=series_color(position),
                           label=f"vs {opponent}")
    opponent_axis.plot(calibrator.knots, calibrator.calibrated, color=TEXT_COLOR, linewidth=1,
                       label="isotonic map (all opponents)")
    opponent_axis.set_title("Calibration by opponent")
    for position, (first, last) in enumerate(ROUND_GROUPS):
        selected = (rounds >= first) & (rounds <= last)
        round_axis.plot(*_reliability(raw[selected], outcomes[selected]), marker="o", color=series_color(position + 4),
                        label=f"rounds {first}-{last}")
    round_axis.set_title("Calibration by round")
    histogram_axis.hist([raw[outcomes == 1], raw[outcomes == 0]], bins=40, stacked=True,
                        color=[series_color(2), series_color(7)], label=["eventually won", "eventually lost"])
    histogram_axis.set_xlabel("predicted win chance (raw)")
    histogram_axis.set_ylabel("round starts")
    histogram_axis.set_title("How confident the value head is")
    for axis in (opponent_axis, round_axis, histogram_axis):
        axis.legend()
    save_figure(figure, folder, "value_calibration.png")



def _concede_table(table: dict, calibrator, held_out: np.ndarray, tables: TableCollector) -> None:
    populations = {
        "vs heuristic": held_out & opponent_is(table, "heuristic"),
        "stage, learner holds the collection deck (not vs random)":
            held_out & source_is(table, "stage") & (table["stage_side"] == 0) & ~opponent_is(table, "random"),
        "self-play": held_out & opponent_is(table, "self"),
    }
    result = []
    for label, selected in populations.items():
        rows = np.flatnonzero(selected)
        chances = calibrator(np.nan_to_num((table["round_values"][rows, 1:].astype(np.float64) + 1) / 2, nan=1.0))
        won = table["won"][rows] == 1
        final_rounds = table["rounds"][rows]
        for threshold in CONCEDE_THRESHOLDS:
            below = chances < threshold
            conceded = below.any(axis=1)
            concede_round = np.where(conceded, below.argmax(axis=1) + 1, 0)
            thrown_wins = (conceded & won).sum()
            saved = (final_rounds - concede_round + 1)[conceded]
            result.append([label, f"{threshold:.2f}", f"{len(rows):,}", f"{conceded.mean():.3f}",
                           f"{(won[conceded].mean() if conceded.any() else 0):.3f}",
                           f"{thrown_wins / max(won.sum(), 1):.3f}", f"{saved.mean() if conceded.any() else 0:.2f}"])
    tables.add("What a 'concede below p' rule would have done (held-out games, updates 400-600)",
               ["games", "concede when calibrated chance < p", "games", "share conceded", "win rate of conceded games",
                "share of all wins thrown away", "mean rounds saved per concession"], result,
               note="Checked at each round start from round 1 with the isotonic-calibrated chance. 'Rounds saved' "
                    "counts the concession round itself.")
