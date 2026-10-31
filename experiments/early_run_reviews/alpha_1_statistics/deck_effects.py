"""How much the deck decides the game: deck features, PvE stage groups, a deck-only win model, hardest lineups."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from autogitcg.analysis.confidence import format_rate, rate_with_interval
from experiments.early_run_reviews.alpha_1_statistics.deck_rows import STAGE_GROUPS
from experiments.early_run_reviews.alpha_1_statistics.deck_strength_model import self_play_deck_model
from experiments.early_run_reviews.alpha_1_statistics.game_rows import CONTROLLERS
from experiments.early_run_reviews.alpha_1_statistics.figure_style import NEUTRAL_COLOR, plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.name_lookup import NameLookup
from experiments.early_run_reviews.alpha_1_statistics.prediction_scores import all_scores
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import opponent_is, source_is, trained_learner_rows, win_chance

STAGE_OPPONENTS = ("heuristic", "self", "snapshot")
LISTED_DEFINITIONS = 12
LINEUP_MINIMUM_GAMES = 150
LISTED_LINEUPS = 15
EDGE_BINS = 10


def report_deck_effects(table: dict, names: NameLookup, folder: Path, tables: TableCollector) -> None:
    trained = trained_learner_rows(table)
    _deck_feature_rates(table, trained, tables)
    _stage_groups(table, trained, folder, tables)
    _deck_model(table, names, tables)
    _deck_edge_against_heuristic(table, folder, tables)
    _hardest_stage_lineups(table, trained, names, tables)


def _deck_feature_rates(table: dict, trained: np.ndarray, tables: TableCollector) -> None:
    groups = {
        "no action cards (PvE stage deck)": table["action_cards"] == 0,
        "30 action cards": table["action_cards"] == 30,
        "talent in deck": table["deck_talents"] > 0,
        "no talent in deck": (table["deck_talents"] == 0) & (table["action_cards"] == 30),
        "resonance card in deck": table["deck_resonance"] > 0,
        "3 characters, 3 different elements": (table["character_count"] == 3) & (table["distinct_elements"] == 3),
        "3 characters, 2 share an element": (table["character_count"] == 3) & (table["largest_element_group"] == 2),
        "3 characters, all one element": (table["character_count"] == 3) & (table["largest_element_group"] == 3),
        "1 or 2 characters (PvE)": table["character_count"] < 3,
        "4 characters (PvE)": table["character_count"] == 4,
    }
    rows = []
    for label, in_group in groups.items():
        row = [label]
        for opponent in ("heuristic", "self"):
            row.append(_rate_cell(table["won"][trained & opponent_is(table, opponent) & in_group]))
        rows.append(row)
    tables.add("Learner win rate by deck feature (updates 400-600)", ["learner's deck", "vs heuristic", "self-play"], rows,
               note="Rate [95% interval] (n). In self-play the average is 0.5 by construction, so a group's rate "
                    "measures its edge over the decks it meets. Groups overlap and are confounded with source "
                    "(PvE decks are stage decks).")


def _rate_cell(outcomes: np.ndarray) -> str:
    rate, low, high, count = rate_with_interval(outcomes)
    return f"{format_rate(rate, low, high)} (n={count:,})" if count else "-"


def _stage_groups(table: dict, trained: np.ndarray, folder: Path, tables: TableCollector) -> None:
    stage = trained & source_is(table, "stage")
    stage_group_of_game = np.maximum(table["stage_group"], table["stage_group"][np.arange(len(table["seat"])) ^ 1])
    collection_side = stage & (table["stage_side"] == 0)
    stage_side = stage & (table["stage_side"] == 1)
    rows, figure_rates = [], defaultdict(list)
    for group_index, group in enumerate(STAGE_GROUPS):
        in_group = stage_group_of_game == group_index
        row = [group]
        for opponent in STAGE_OPPONENTS:
            outcomes = table["won"][collection_side & in_group & opponent_is(table, opponent)]
            row.append(_rate_cell(outcomes))
            figure_rates[opponent].append(rate_with_interval(outcomes))
        row.append(_rate_cell(table["won"][stage_side & in_group & opponent_is(table, "heuristic")]))
        rows.append(row)
    tables.add("PvE stages by group (updates 400-600)",
               ["stage group", "learner (collection deck) vs heuristic playing the stage deck",
                "learner vs learner playing the stage deck (self-play)", "learner vs snapshot playing the stage deck",
                "learner playing the stage deck vs heuristic"], rows,
               note="Groups are inferred from the stage's enemy deck (see deck_rows.py); many weekly-guest stages "
                    "reuse invitation decks and are counted under 'invitation'.")
    figure, axis = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(STAGE_GROUPS))
    for offset, opponent in enumerate(STAGE_OPPONENTS):
        rates = np.array([rate for rate, _, _, _ in figure_rates[opponent]])
        errors = np.array([[rate - low, high - rate] for rate, low, high, _ in figure_rates[opponent]]).T
        axis.bar(positions + (offset - 1) * 0.27, rates, width=0.25, yerr=errors, color=series_color(offset),
                 label=f"stage deck played by {opponent if opponent != 'self' else 'the learner'}", error_kw={"linewidth": 1})
    axis.axhline(0.5, color=NEUTRAL_COLOR, linewidth=1, linestyle="--")
    axis.set_xticks(positions, STAGE_GROUPS)
    axis.set_ylabel("learner win rate (collection deck)")
    axis.set_title("Learner with a random collection deck vs PvE stage decks (updates 400-600)")
    axis.legend()
    save_figure(figure, folder, "stage_groups.png")


def _deck_model(table: dict, names: NameLookup, tables: TableCollector) -> None:
    model, training_rows, held_out_rows = self_play_deck_model(table)
    outcomes = table["won"][held_out_rows].astype(np.float64)
    seat_only = np.full(len(held_out_rows), outcomes.mean())
    deck_prediction = 1 / (1 + np.exp(-model.logits(table, held_out_rows)))
    rows = [["constant (always the average)", *_score_cells(seat_only, outcomes)],
            ["deck-only model (characters + cards + seat)", *_score_cells(deck_prediction, outcomes)]]
    for round_number in (0, 1, 3, 5):
        values = table["round_values"][held_out_rows, round_number]
        known = ~np.isnan(values)
        rows.append([f"value head at start of round {round_number} (n={int(known.sum()):,})",
                     *_score_cells(win_chance(values[known]), outcomes[known])])
    tables.add("How predictable self-play results are from the decks alone",
               ["predictor (held-out self-play games, updates 300-600)", "log loss", "Brier", "accuracy", "AUC",
                "calibration error"], rows,
               note=f"Trained on {len(training_rows):,} games, scored on {len(held_out_rows):,} other games. "
                    "Lower log loss/Brier is better; AUC 0.5 = no skill, 1 = perfect. The value head at round 0 "
                    "sees both lineups (and the opponent's cards in NPC mode) before any play.")
    _list_definitions(model, names, "character", "Character strength in self-play (deck-only model)", tables)
    _list_definitions(model, names, "card", "Card strength in self-play (deck-only model, per copy)", tables)


def _score_cells(predicted: np.ndarray, outcomes: np.ndarray) -> list[str]:
    return [f"{value:.3f}" for value in all_scores(predicted, outcomes).values()]


def _list_definitions(model, names: NameLookup, kind: str, title: str, tables: TableCollector) -> None:
    entries = [(weight, feature[1], appearances) for feature, weight, appearances
               in zip(model.feature_ids, model.weights, model.appearances) if feature[0] == kind]
    entries.sort(reverse=True)
    name_of = names.character if kind == "character" else names.card
    rows = [[name_of(definition), f"{weight:+.2f}", f"{appearances:,}"]
            for weight, definition, appearances in entries[:LISTED_DEFINITIONS]]
    rows.append(["...", "", ""])
    rows += [[name_of(definition), f"{weight:+.2f}", f"{appearances:,}"]
             for weight, definition, appearances in entries[-LISTED_DEFINITIONS:]]
    tables.add(title, [kind, "log-odds effect", "decks containing it (training games)"], rows,
               note=f"Strongest and weakest {LISTED_DEFINITIONS} of {len(entries)} {kind}s seen at least 200 times. "
                    "+0.4 log-odds turns a 50% game into about 60%. NPC characters always come with their fixed NPC "
                    "deck, so their effect includes that deck. Correlated features share credit; read as a ranking, "
                    "not exact values.")


def _deck_edge_against_heuristic(table: dict, folder: Path, tables: TableCollector) -> None:
    model, _, held_out_rows = self_play_deck_model(table)
    against_heuristic = np.flatnonzero(trained_learner_rows(table) & opponent_is(table, "heuristic"))
    learner_edge = 1 / (1 + np.exp(-model.logits(table, against_heuristic)))
    self_play_edge = 1 / (1 + np.exp(-model.logits(table, held_out_rows)))
    edges = np.quantile(np.concatenate([learner_edge, self_play_edge]), np.linspace(0, 1, EDGE_BINS + 1))
    figure, axis = plt.subplots(figsize=(7, 5))
    rows = []
    for position, (label, predicted, outcomes) in enumerate((
            ("self-play (held-out)", self_play_edge, table["won"][held_out_rows]),
            ("learner vs heuristic", learner_edge, table["won"][against_heuristic]))):
        bins = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, EDGE_BINS - 1)
        centers, rates, lows, highs = [], [], [], []
        for number in range(EDGE_BINS):
            rate, low, high, count = rate_with_interval(outcomes[bins == number])
            if count >= 30:
                centers.append(predicted[bins == number].mean())
                rates.append(rate)
                lows.append(low)
                highs.append(high)
                if label == "learner vs heuristic":
                    rows.append([f"{edges[number]:.2f}-{edges[number + 1]:.2f}", f"{count:,}", format_rate(rate, low, high)])
        axis.plot(centers, rates, marker="o", markersize=6, color=series_color(position), label=label)
        axis.fill_between(centers, lows, highs, color=series_color(position), alpha=0.2, linewidth=0)
    axis.plot([0, 1], [0, 1], color=NEUTRAL_COLOR, linestyle="--", linewidth=1, label="deck decides everything")
    axis.set_xlabel("deck-only win chance for the learner's deck (model fitted on self-play)")
    axis.set_ylabel("actual learner win rate")
    axis.set_title("Deck edge vs result (updates 400-600)")
    axis.legend()
    save_figure(figure, folder, "deck_edge.png")
    tables.add("Learner vs heuristic by deck edge (updates 400-600)",
               ["deck-only win chance of the learner's deck", "games", "learner win rate"], rows,
               note="The deck-only model says how often the learner's deck beats the heuristic's deck when both are "
                    "played by the same network. The gap between the two curves is what better play is worth.")


def _hardest_stage_lineups(table: dict, trained: np.ndarray, names: NameLookup, tables: TableCollector) -> None:
    collection_side = np.flatnonzero(trained & source_is(table, "stage") & (table["stage_side"] == 0)
                                     & ~opponent_is(table, "random"))
    outcomes_by_lineup = defaultdict(list)
    for row in collection_side:
        lineup = tuple(int(character) for character in table["characters"][row ^ 1] if character)
        outcomes_by_lineup[lineup].append((table["won"][row], table["opponent_controller"][row] == CONTROLLERS.index("heuristic")))
    rows = []
    for lineup, results in outcomes_by_lineup.items():
        if len(results) >= LINEUP_MINIMUM_GAMES:
            results = np.array(results)
            rows.append((results[:, 0].mean(), lineup, results))
    rows.sort()
    overall = table["won"][collection_side].mean()
    tables.add("Hardest stage lineups for the learner (collection deck, updates 400-600)",
               ["stage lineup", "games", "learner win rate", "win rate when the heuristic plays the stage"],
               [[names.lineup(lineup), f"{len(results):,}", format_rate(*rate_with_interval(results[:, 0])[:3]),
                 f"{results[results[:, 1] == 1, 0].mean():.2f} (n={int(results[:, 1].sum())})"]
                for _, lineup, results in rows[:LISTED_LINEUPS]],
               note=f"Stage lineups with at least {LINEUP_MINIMUM_GAMES} games where the learner holds the collection "
                    f"deck (opponents: self-play, snapshots, heuristic; not random). Overall rate {overall:.3f} over "
                    f"{len(collection_side):,} games and {len(rows)} lineups that qualify.")
