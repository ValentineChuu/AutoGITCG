"""Seat effect (going first) and NPC mode vs blind mode (knowing the opponent's deck).

Seats and deck knowledge are both assigned at random by the matchup sampler, so these comparisons are
randomized experiments: differences are causal, up to sampling noise.
"""

from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.bucket_series import bucket_rates
from autogitcg.analysis.confidence import format_rate, rate_with_interval
from experiments.early_run_reviews.alpha_1_statistics.figure_style import NEUTRAL_COLOR, plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.game_rows import SOURCES
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import (BUCKET_UPDATES, learner_rows, opponent_is, source_is,
                                               trained_learner_rows, update_buckets)


def report_seat_and_knowledge(table: dict, folder: Path, tables: TableCollector) -> None:
    trained = trained_learner_rows(table)
    self_play_seat0 = learner_rows(table) & opponent_is(table, "self") & (table["seat"] == 0)
    _seat_table(table, trained, tables)
    knowledge_rates = _knowledge_table(table, trained, tables)
    _plot(table, self_play_seat0, knowledge_rates, folder)


def _cell(outcomes: np.ndarray) -> str:
    rate, low, high, count = rate_with_interval(outcomes)
    return f"{format_rate(rate, low, high)} (n={count:,})"


def _seat_table(table: dict, trained: np.ndarray, tables: TableCollector) -> None:
    rows = []
    for source in (None, *SOURCES):
        in_source = np.ones(len(trained), dtype=bool) if source is None else source_is(table, source)
        self_play = trained & opponent_is(table, "self") & (table["seat"] == 0) & in_source
        heuristic = trained & opponent_is(table, "heuristic") & in_source
        rows.append([source or "all",
                     _cell(table["won"][self_play]),
                     _cell(table["won"][heuristic & (table["seat"] == 0)]),
                     _cell(table["won"][heuristic & (table["seat"] == 1)])])
    tables.add("Seat effect (updates 400-600)",
               ["source", "self-play: seat 0 win rate", "learner vs heuristic, learner in seat 0",
                "learner vs heuristic, learner in seat 1"], rows,
               note="Seat 0 acts first in round 1. Round-limit games count as a loss for both seats.")


def _knowledge_table(table: dict, trained: np.ndarray, tables: TableCollector) -> dict:
    knows = table["knows_opponent_deck"] == 1
    opponent_knows = table["knows_opponent_deck"][np.arange(len(knows)) ^ 1] == 1
    self_play, heuristic = trained & opponent_is(table, "self"), trained & opponent_is(table, "heuristic")
    collection_side = source_is(table, "stage") & (table["stage_side"] == 0)
    stage_side = source_is(table, "stage") & (table["stage_side"] == 1)
    groups = {
        "self-play npc_decks, only this seat knows": (self_play & source_is(table, "npc_decks") & knows & ~opponent_knows, None),
        "self-play collections, only this seat knows": (self_play & source_is(table, "collections") & knows & ~opponent_knows, None),
        "self-play stage, collection side": (self_play & collection_side, knows),
        "self-play stage, stage side": (self_play & stage_side, knows),
        "vs heuristic, npc_decks": (heuristic & source_is(table, "npc_decks"), knows),
        "vs heuristic, collections": (heuristic & source_is(table, "collections"), knows),
        "vs heuristic, stage collection side": (heuristic & collection_side, knows),
        "vs heuristic, stage side": (heuristic & stage_side, knows),
    }
    rows, figure_rates = [], {}
    for label, (selected, knowledge) in groups.items():
        if knowledge is None:
            outcomes = table["won"][selected]
            rows.append([label, _cell(outcomes), "(other seat blind)", "-"])
            figure_rates[label] = (rate_with_interval(outcomes), None)
            continue
        known, blind = table["won"][selected & knowledge], table["won"][selected & ~knowledge]
        difference = known.mean() - blind.mean()
        half_width = 1.96 * np.sqrt(known.var() / len(known) + blind.var() / len(blind))
        rows.append([label, _cell(known), _cell(blind), f"{difference:+.3f} ± {half_width:.3f}"])
        figure_rates[label] = (rate_with_interval(known), rate_with_interval(blind))
    tables.add("NPC mode (knows the opponent's deck) vs blind mode (updates 400-600)",
               ["learner seats", "win rate, knows the deck", "win rate, blind", "difference (95%)"], rows,
               note="Compared within one source and side, because knowledge odds differ by side in stage games "
                    "(collection side 80%, stage side 50%) and pooling would mix in the side's own win rate. In "
                    "symmetric self-play the clean test is games where exactly one seat knows: a win rate above 0.5 is "
                    "the value of the information.")
    return figure_rates


def _plot(table: dict, self_play_seat0: np.ndarray, knowledge_rates: dict, folder: Path) -> None:
    figure, (seat_axis, knowledge_axis) = plt.subplots(1, 2, figsize=(14, 4.5))
    starts, rates, lows, highs = bucket_rates(table["won"].astype(np.float64), update_buckets(table), self_play_seat0)
    seat_axis.plot(starts, rates, color=series_color(0), label="seat 0 win rate in self-play")
    seat_axis.fill_between(starts, lows, highs, color=series_color(0), alpha=0.2, linewidth=0)
    seat_axis.axhline(0.5, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
    seat_axis.set_xlabel(f"training update (buckets of {BUCKET_UPDATES})")
    seat_axis.set_title("Seat effect over training")
    seat_axis.legend()
    labels = list(knowledge_rates)
    positions = np.arange(len(labels))
    for offset, (index, label_text) in enumerate(((0, "knows the opponent's deck"), (1, "blind"))):
        rates, errors, kept = [], [], []
        for position, label in enumerate(labels):
            entry = knowledge_rates[label][index]
            if entry is not None:
                rate, low, high, _ = entry
                rates.append(rate)
                errors.append([rate - low, high - rate])
                kept.append(position)
        knowledge_axis.errorbar(rates, np.array(kept) + (offset - 0.5) * 0.3, xerr=np.array(errors).T, fmt="o",
                                markersize=7, color=series_color(offset), label=label_text)
    knowledge_axis.set_yticks(positions, labels)
    knowledge_axis.invert_yaxis()
    knowledge_axis.set_xlim(0.4, 1.0)
    knowledge_axis.set_xlabel("learner win rate (95% interval)")
    knowledge_axis.set_title("NPC mode vs blind mode (updates 400-600)")
    knowledge_axis.legend(loc="upper right")
    save_figure(figure, folder, "seat_and_knowledge.png")
