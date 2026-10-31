"""When games are decided, from the learner's own (calibrated) win estimate at each round start.

- decided round: the first round from which the estimate stays on the side of the final result
  (>= 0.8 for a win, <= 0.2 for a loss) until the end; games that never lock in count as decided at the end.
- comeback: won after the estimate was <= 0.2 at some round start; collapse: lost after it was >= 0.8.
- swing: mean absolute change of the estimate from one round start to the next.
"""

import warnings
from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.figure_style import plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.isotonic_calibration import FIRST_ROUND, shared_calibrator
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import matchup_label, opponent_is, trained_learner_rows

CLEAR_WIN = 0.8
CLEAR_LOSS = 0.2
PLOTTED_ROUNDS = 10


def report_game_flow(table: dict, folder: Path, tables: TableCollector) -> None:
    calibrator = shared_calibrator(table)
    rows = np.flatnonzero(trained_learner_rows(table) & (table["game"] % 2 == 1))
    raw = table["round_values"][rows, FIRST_ROUND:].astype(np.float64)
    chances = np.where(np.isnan(raw), np.nan, calibrator(np.nan_to_num((raw + 1) / 2)))
    won = table["won"][rows] == 1
    rounds = table["rounds"][rows].astype(np.int64)
    decided = _decided_round(chances, won, rounds)
    _flow_table(table, rows, chances, won, rounds, decided, tables)
    _plot(table, rows, chances, won, rounds, decided, folder)


def _decided_round(chances: np.ndarray, won: np.ndarray, rounds: np.ndarray) -> np.ndarray:
    """First round (1-based) from which every later round start is clearly on the winning/losing side."""
    on_final_side = np.where(won[:, None], chances >= CLEAR_WIN, chances <= CLEAR_LOSS) | np.isnan(chances)
    decided = rounds + 1  # never clear before the end
    for round_index in range(chances.shape[1] - 1, -1, -1):
        stays = on_final_side[:, round_index:].all(axis=1) & (round_index + 1 <= rounds)
        decided = np.where(stays, round_index + 1, decided)
    return decided


def _flow_table(table, rows, chances, won, rounds, decided, tables) -> None:
    labels = matchup_label(table, rows)
    stage_collection = (table["source"][rows] == 0) & (table["stage_side"][rows] == 0)
    labels = np.where(stage_collection, np.char.add(labels, " (collection side)"),
                      np.where(table["source"][rows] == 0, np.char.add(labels, " (stage side)"), labels))
    with warnings.catch_warnings():  # one-round games have no round-to-round change
        warnings.simplefilter("ignore", RuntimeWarning)
        minimum = np.nanmin(chances, axis=1)
        maximum = np.nanmax(chances, axis=1)
        swing = np.nanmean(np.abs(np.diff(chances, axis=1)), axis=1)
    crossings = np.nansum(np.abs(np.diff(np.sign(chances - 0.5), axis=1)) == 2, axis=1)
    result = []
    for label in sorted(set(labels)):
        selected = labels == label
        if selected.sum() < 300:
            continue
        behind, ahead = selected & (minimum <= CLEAR_LOSS), selected & (maximum >= CLEAR_WIN)
        clear_early = (rounds[selected] - decided[selected] + 1)
        result.append([label, f"{int(selected.sum()):,}", f"{won[selected].mean():.3f}", f"{rounds[selected].mean():.2f}",
                       f"{decided[selected].mean():.2f}", f"{np.mean(clear_early >= 3):.3f}",
                       f"{np.mean(decided[selected] > rounds[selected]):.3f}",
                       f"{won[behind].mean():.3f} (n={int(behind.sum()):,})",
                       f"{1 - won[ahead].mean():.3f} (n={int(ahead.sum()):,})",
                       f"{np.nanmean(swing[selected]):.3f}", f"{crossings[selected].mean():.2f}"])
    tables.add("When games are decided (learner seats, held-out games, updates 400-600)",
               ["matchup", "learner seats", "win rate", "rounds", "decided round", "clear >= 3 rounds before the end",
                "never clear before the end", f"comeback rate (was <= {CLEAR_LOSS})",
                f"collapse rate (was >= {CLEAR_WIN})", "swing per round", "times crossing 50%"], result,
               note="Chances are isotonic-calibrated (one map for all opponents; it under-rates the learner vs "
                    "heuristic and random). 'Decided round' = first round start from which the estimate stays "
                    f">= {CLEAR_WIN} (wins) or <= {CLEAR_LOSS} (losses) to the end; 'clear >= 3 rounds before the "
                    "end' counts the deciding round itself.")


def _plot(table, rows, chances, won, rounds, decided, folder) -> None:
    figure, (trajectory_axis, decided_axis) = plt.subplots(1, 2, figsize=(14, 5))
    for position, opponent in enumerate(("self", "heuristic")):
        selected = opponent_is(table, opponent)[rows]
        for outcome, style in ((True, "-"), (False, "--")):
            in_group = selected & (won == outcome)
            means = [np.nanmean(chances[in_group, index]) for index in range(PLOTTED_ROUNDS)]
            trajectory_axis.plot(range(1, PLOTTED_ROUNDS + 1), means, linestyle=style, marker="o",
                                 color=series_color(position),
                                 label=f"vs {opponent}, {'won' if outcome else 'lost'} (n={int(in_group.sum()):,})")
    trajectory_axis.set_xlabel("round")
    trajectory_axis.set_ylabel("mean calibrated win chance at round start")
    trajectory_axis.set_title("Average win-estimate trajectory (games still running)")
    trajectory_axis.legend()
    rounds_before_end = rounds - decided + 1
    for position, opponent in enumerate(("self", "heuristic", "snapshot")):
        selected = opponent_is(table, opponent)[rows]
        counts = np.bincount(np.clip(rounds_before_end[selected], 0, 8), minlength=9) / selected.sum()
        decided_axis.bar(np.arange(9) + (position - 1) * 0.27, counts, width=0.25, color=series_color(position),
                         label=f"vs {opponent}")
    decided_axis.set_xticks(range(9), ["never"] + [str(number) for number in range(1, 8)] + ["8+"])
    decided_axis.set_xlabel("rounds (incl. the deciding one) the result was already clear before the game ended")
    decided_axis.set_ylabel("share of learner seats")
    decided_axis.set_title("How early the outcome becomes clear")
    decided_axis.legend()
    save_figure(figure, folder, "game_flow.png")
