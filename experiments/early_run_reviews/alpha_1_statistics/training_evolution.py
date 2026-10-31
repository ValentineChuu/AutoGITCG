"""How results, behaviour and policy confidence changed over training."""

from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.behaviour_measures import BEHAVIOUR_MEASURES
from experiments.early_run_reviews.alpha_1_statistics.bucket_series import bucket_means, bucket_rates
from experiments.early_run_reviews.alpha_1_statistics.figure_style import NEUTRAL_COLOR, plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import (BUCKET_UPDATES, controller_is, learner_rows, opponent_is, source_is,
                                               update_buckets)

OPPONENT_ORDER = ("heuristic", "snapshot", "random")
SOURCE_ORDER = ("stage", "npc_decks", "collections")
CONFIDENCE_KINDS = ("action", "chooseActive", "rerollDice", "switchHands")
SNAPSHOT_BUCKETS = (0, 100, 300, 575)


def report_training_evolution(table: dict, folder: Path, tables: TableCollector) -> None:
    buckets = update_buckets(table)
    learner = learner_rows(table)
    _plot_win_rates(table, buckets, learner, folder)
    _plot_behaviour(table, buckets, learner, folder)
    _plot_confidence(table, buckets, learner, folder)
    _behaviour_table(table, buckets, learner, tables)
    _confidence_table(table, buckets, learner, tables)


def _plot_win_rates(table: dict, buckets: np.ndarray, learner: np.ndarray, folder: Path) -> None:
    won = table["won"].astype(np.float64)
    figure, (opponent_axis, source_axis, length_axis) = plt.subplots(1, 3, figsize=(15, 4))
    for position, opponent in enumerate(OPPONENT_ORDER):
        starts, rates, lows, highs = bucket_rates(won, buckets, learner & opponent_is(table, opponent))
        opponent_axis.plot(starts, rates, color=series_color(position), label=f"vs {opponent}")
        opponent_axis.fill_between(starts, lows, highs, color=series_color(position), alpha=0.2, linewidth=0)
    opponent_axis.set_title("Learner win rate in training games")
    opponent_axis.set_ylabel("win rate")
    against_heuristic = learner & opponent_is(table, "heuristic")
    for position, source in enumerate(SOURCE_ORDER):
        starts, rates, lows, highs = bucket_rates(won, buckets, against_heuristic & source_is(table, source))
        source_axis.plot(starts, rates, color=series_color(position + 3), label=source)
        source_axis.fill_between(starts, lows, highs, color=series_color(position + 3), alpha=0.2, linewidth=0)
    source_axis.set_title("Win rate vs heuristic, by matchup source")
    rounds = table["rounds"].astype(np.float64)
    for position, opponent in enumerate(OPPONENT_ORDER):
        starts, means, _ = bucket_means(rounds, buckets, learner & opponent_is(table, opponent))
        length_axis.plot(starts, means, color=series_color(position), label=f"vs {opponent}")
    one_row_per_self_play_game = learner & opponent_is(table, "self") & (table["seat"] == 0)
    starts, means, _ = bucket_means(rounds, buckets, one_row_per_self_play_game)
    length_axis.plot(starts, means, color=series_color(6), label="self-play")
    length_axis.set_title("Rounds per game")
    for axis in (opponent_axis, source_axis, length_axis):
        axis.set_xlabel(f"training update (buckets of {BUCKET_UPDATES})")
        axis.legend()
    save_figure(figure, folder, "win_rate_over_training.png")


def _plot_behaviour(table: dict, buckets: np.ndarray, learner: np.ndarray, folder: Path) -> None:
    heuristic = controller_is(table, "heuristic")
    names = list(BEHAVIOUR_MEASURES)
    columns = 4
    figure, axes = plt.subplots((len(names) + columns - 1) // columns, columns, figsize=(16, 3.2 * ((len(names) + columns - 1) // columns)))
    for axis, name in zip(axes.flat, names):
        values = BEHAVIOUR_MEASURES[name](table)
        starts, means, half_widths = bucket_means(values, buckets, learner)
        axis.plot(starts, means, color=series_color(0), label="learner")
        axis.fill_between(starts, means - half_widths, means + half_widths, color=series_color(0), alpha=0.2, linewidth=0)
        starts, means, _ = bucket_means(values, buckets, heuristic)
        axis.plot(starts, means, color=NEUTRAL_COLOR, linestyle="--", label="heuristic bot")
        axis.set_title(name)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    axes.flat[0].legend()
    figure.suptitle("Behaviour over training (all learner seats; dashed: the heuristic bot in the same training games)")
    save_figure(figure, folder, "behaviour_over_training.png")


def _confidence_series(table: dict, kind: str, quantity: str) -> np.ndarray:
    decisions = table[f"{kind}_decisions"]
    return np.divide(table[f"{kind}_{quantity}"], decisions, out=np.full(len(decisions), np.nan), where=decisions > 0)


def _plot_confidence(table: dict, buckets: np.ndarray, learner: np.ndarray, folder: Path) -> None:
    figure, (probability_axis, entropy_axis, normalized_axis) = plt.subplots(1, 3, figsize=(15, 4))
    for position, kind in enumerate(CONFIDENCE_KINDS):
        for axis, quantity in ((probability_axis, "probability"), (entropy_axis, "entropy")):
            starts, means, _ = bucket_means(_confidence_series(table, kind, quantity), buckets, learner)
            axis.plot(starts, means, color=series_color(position), label=kind)
    normalized = _normalized_action_entropy(table)
    starts, means, _ = bucket_means(normalized, buckets, learner)
    normalized_axis.plot(starts, means, color=series_color(0), label="action")
    probability_axis.set_title("Mean probability of the chosen option")
    entropy_axis.set_title("Mean policy entropy (nats; reroll/mulligan summed over dice/cards)")
    normalized_axis.set_title("Action entropy / log(number of options)")
    for axis in (probability_axis, entropy_axis, normalized_axis):
        axis.set_xlabel(f"training update (buckets of {BUCKET_UPDATES})")
        axis.legend()
    save_figure(figure, folder, "policy_confidence_over_training.png")


def _normalized_action_entropy(table: dict) -> np.ndarray:
    return np.divide(table["action_entropy"], table["action_log_options"], out=np.full(len(table["seat"]), np.nan),
                     where=table["action_log_options"] > 0)


def _behaviour_table(table: dict, buckets: np.ndarray, learner: np.ndarray, tables: TableCollector) -> None:
    heuristic_late = controller_is(table, "heuristic") & (buckets == SNAPSHOT_BUCKETS[-1])
    rows = []
    for name, measure in BEHAVIOUR_MEASURES.items():
        values = measure(table)
        row = [name]
        for bucket in SNAPSHOT_BUCKETS:
            selected = learner & (buckets == bucket) & ~np.isnan(values)
            row.append(f"{values[selected].mean():.2f}")
        row.append(f"{np.nanmean(values[heuristic_late]):.2f}")
        rows.append(row)
    counts = [int((learner & (buckets == bucket)).sum()) for bucket in SNAPSHOT_BUCKETS]
    header = ["measure"] + [f"learner, updates {bucket}-{bucket + BUCKET_UPDATES - 1} (n={count:,})"
                            for bucket, count in zip(SNAPSHOT_BUCKETS, counts)]
    header.append(f"heuristic, same late games (n={int(heuristic_late.sum()):,})")
    tables.add("Behaviour over training (means per learner seat)", header, rows,
               note="n = learner seats (a self-play game contributes two). 95% intervals are under ±0.02 for every "
                    "per-round mean at these sample sizes.")


def _confidence_table(table: dict, buckets: np.ndarray, learner: np.ndarray, tables: TableCollector) -> None:
    rows = []
    for kind in CONFIDENCE_KINDS:
        for quantity in ("probability", "entropy"):
            values = _confidence_series(table, kind, quantity)
            rows.append([f"{kind}: mean {quantity}"] + [f"{np.nanmean(values[learner & (buckets == bucket)]):.3f}"
                                                         for bucket in SNAPSHOT_BUCKETS])
    normalized = _normalized_action_entropy(table)
    rows.append(["action: entropy / log(options)"] + [f"{np.nanmean(normalized[learner & (buckets == bucket)]):.3f}"
                                                       for bucket in SNAPSHOT_BUCKETS])
    tables.add("Policy confidence over training", ["quantity"] + [f"updates {bucket}-{bucket + BUCKET_UPDATES - 1}"
                                                                   for bucket in SNAPSHOT_BUCKETS], rows,
               note="Per-seat means of per-decision values. Reroll and mulligan probabilities are of the whole "
                    "keep/change pattern, and their entropies are summed over dice or cards.")
