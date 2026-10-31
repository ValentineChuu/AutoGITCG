"""What distinguishes the learner's wins from its losses (trained updates only).

Two designs:
- self-play, paired: winner vs loser of the same game. Both seats run the same network, so differences are
  deck, seat, luck and how the game unfolded, never "which policy".
- vs the heuristic, unpaired: the learner's own wins vs its own losses.
Both are correlations. Many measures are partly consequences of winning (a winner's characters survive, so
it has more turns to burst), not causes.
"""

from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.behaviour_measures import OUTCOME_MEASURES
from autogitcg.analysis.confidence import Z_95, difference_of_means
from experiments.early_run_reviews.alpha_1_statistics.figure_style import plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import opponent_is, trained_learner_rows

NOT_IN_FIGURE = ("own characters defeated",)  # trivially tied to the result


def report_outcome_factors(table: dict, folder: Path, tables: TableCollector) -> None:
    trained = trained_learner_rows(table)
    decided_self_play = trained & opponent_is(table, "self") & (table["seat"] == 0) & (table["round_limit"] == 0)
    first_rows = np.flatnonzero(decided_self_play)
    winner_rows = first_rows + (1 - table["won"][first_rows])
    loser_rows = winner_rows ^ 1
    against_heuristic = np.flatnonzero(trained & opponent_is(table, "heuristic"))
    learner_wins = against_heuristic[table["won"][against_heuristic] == 1]
    learner_losses = against_heuristic[table["won"][against_heuristic] == 0]

    paired, unpaired = {}, {}
    rows = []
    for name, measure in OUTCOME_MEASURES.items():
        values = measure(table)
        spread = np.nanstd(values[trained])
        differences = values[winner_rows] - values[loser_rows]
        differences = differences[~np.isnan(differences)]
        paired_mean = differences.mean()
        paired_half = Z_95 * differences.std(ddof=1) / np.sqrt(len(differences))
        paired[name] = (paired_mean / spread, paired_half / spread)
        heuristic_difference, heuristic_half = difference_of_means(values[learner_wins], values[learner_losses])
        unpaired[name] = (heuristic_difference / spread, heuristic_half / spread)
        rows.append([name, f"{np.nanmean(values[winner_rows]):.2f}", f"{np.nanmean(values[loser_rows]):.2f}",
                     f"{paired_mean:+.2f} ± {paired_half:.2f}",
                     f"{np.nanmean(values[learner_wins]):.2f}", f"{np.nanmean(values[learner_losses]):.2f}",
                     f"{heuristic_difference:+.2f} ± {heuristic_half:.2f}"])
    tables.add("Wins vs losses (updates 400-600)",
               ["measure", "self-play winner", "self-play loser", "paired difference",
                "learner wins vs heuristic", "learner losses vs heuristic", "difference"], rows,
               note=f"Self-play: {len(first_rows):,} decided games (paired: winner minus loser of the same game). "
                    f"Vs heuristic: {len(learner_wins):,} learner wins and {len(learner_losses):,} losses "
                    "(round-limit games count as losses). ± is a 95% interval.")
    _plot(paired, unpaired, folder)


def _plot(paired: dict, unpaired: dict, folder: Path) -> None:
    names = [name for name in OUTCOME_MEASURES if name not in NOT_IN_FIGURE]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 0.38 * len(names) + 1.2))
    for offset, (label, effects, color) in enumerate((
            ("self-play: winner minus loser (same game)", paired, series_color(0)),
            ("vs heuristic: learner wins minus learner losses", unpaired, series_color(1)))):
        means = np.array([effects[name][0] for name in names])
        half_widths = np.array([effects[name][1] for name in names])
        axis.errorbar(means, positions + (offset - 0.5) * 0.3, xerr=half_widths, fmt="o", color=color, label=label,
                      markersize=6, capsize=0)
    axis.axvline(0, color="#8a8984", linewidth=1)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlabel("difference in standard deviations of the measure (95% interval)")
    axis.set_title("What winners do differently (updates 400-600)")
    axis.legend(loc="lower right")
    save_figure(figure, folder, "wins_vs_losses.png")
