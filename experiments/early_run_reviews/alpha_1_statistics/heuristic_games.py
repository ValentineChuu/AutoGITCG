"""Games against the scripted heuristic (updates 400-600): what the heuristic does when it still wins.

Uses the per-seat table plus one indexed query over these games' card moves (played vs tuned away, per card).
"""

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.behaviour_measures import OUTCOME_MEASURES
from autogitcg.analysis.confidence import format_rate, rate_with_interval
from experiments.early_run_reviews.alpha_1_statistics.deck_strength_model import self_play_deck_model
from autogitcg.analysis.definition_kinds import CARD_CATEGORIES, card_facts
from experiments.early_run_reviews.alpha_1_statistics.figure_style import plt, save_figure, series_color
from experiments.early_run_reviews.alpha_1_statistics.isotonic_calibration import shared_calibrator
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.name_lookup import NameLookup
from experiments.early_run_reviews.alpha_1_statistics.seat_selection import TRAINED_FROM_UPDATE, controller_is, opponent_is, win_chance

MEASURES_SHOWN = ("bursts / game", "bursts / round", "elemental skills / round", "normal attacks / round",
                  "cards played / round", "talents played / game", "tunings / round", "voluntary switches / round",
                  "share of rounds it ended first", "nominal dice spent / round", "own characters defeated")
LISTED_CARDS = 15
MINIMUM_CARD_EVENTS = 150


def report_heuristic_games(table: dict, names: NameLookup, folder: Path, tables: TableCollector, database: Path) -> None:
    trained = table["update"] >= TRAINED_FROM_UPDATE
    heuristic_rows = np.flatnonzero(trained & controller_is(table, "heuristic") & opponent_is(table, "heuristic"))
    learner_rows = heuristic_rows ^ 1
    heuristic_won = table["won"][heuristic_rows] == 1
    learner_won = table["won"][learner_rows] == 1
    _behaviour_table(table, heuristic_rows, learner_rows, heuristic_won, learner_won, tables)
    _deck_edge_table(table, learner_rows, heuristic_won, tables)
    _surprise_table(table, learner_rows, heuristic_won, tables)
    card_counts = _card_move_counts(database)
    _card_use_table(card_counts, tables)
    _plot_card_use(card_counts, folder)
    _tuned_cards_table(card_counts, names, tables)


def _behaviour_table(table, heuristic_rows, learner_rows, heuristic_won, learner_won, tables) -> None:
    rows = []
    for name in MEASURES_SHOWN:
        values = OUTCOME_MEASURES[name](table)
        rows.append([name, f"{np.nanmean(values[heuristic_rows[heuristic_won]]):.2f}",
                     f"{np.nanmean(values[heuristic_rows[~heuristic_won]]):.2f}",
                     f"{np.nanmean(values[learner_rows[heuristic_won]]):.2f}",
                     f"{np.nanmean(values[learner_rows[learner_won]]):.2f}"])
    tables.add("Heuristic games (updates 400-600): the two sides in heuristic wins and losses",
               ["measure", f"heuristic, its wins (n={int(heuristic_won.sum()):,})",
                f"heuristic, its losses (n={int((~heuristic_won).sum()):,})",
                "learner, in games the heuristic won", "learner, in games it won"], rows)


def _deck_edge_table(table, learner_rows, heuristic_won, tables) -> None:
    model, _, _ = self_play_deck_model(table)
    learner_edge = 1 / (1 + np.exp(-model.logits(table, learner_rows)))
    rows = []
    for threshold in (0.5, 0.3, 0.2):
        worse_deck = learner_edge < threshold
        rows.append([f"learner's deck-only win chance < {threshold}", f"{worse_deck.mean():.3f}",
                     f"{worse_deck[heuristic_won].mean():.3f}",
                     format_rate(*rate_with_interval(heuristic_won[worse_deck].astype(float))[:3]),
                     format_rate(*rate_with_interval(heuristic_won[~worse_deck].astype(float))[:3])])
    tables.add("Do heuristic wins come from the deck? (updates 400-600)",
               ["condition", "share of all heuristic games", "share of heuristic wins", "heuristic win rate when true",
                "heuristic win rate otherwise"], rows,
               note=f"{len(learner_rows):,} games. Deck-only win chance from the model fitted on self-play "
                    "(same network on both sides), so it measures the matchup, not the play.")


def _surprise_table(table, learner_rows, heuristic_won, tables) -> None:
    calibrator = shared_calibrator(table)
    rows = []
    for round_number in (1, 2, 3, 4):
        values = table["round_values"][learner_rows, round_number]
        known = ~np.isnan(values)
        chances = calibrator(win_chance(values[known]))
        lost, won = heuristic_won[known], ~heuristic_won[known]
        rows.append([str(round_number), f"{chances[won].mean():.3f}", f"{chances[lost].mean():.3f}",
                     f"{(chances[lost] >= 0.5).mean():.3f} (n={int(lost.sum()):,})"])
    tables.add("Learner's own (calibrated) win estimate in heuristic games",
               ["round start", "mean estimate, learner won", "mean estimate, learner lost",
                "share of losses where it still thought it was ahead"], rows)


def _card_move_counts(database: Path) -> dict:
    """(controller, action_kind, card id) -> count, and games per controller, over trained heuristic games."""
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    query = """
        SELECT CASE WHEN m.seat = 0 THEN g.seat0_controller ELSE g.seat1_controller END, m.action_kind, m.definition_id,
               COUNT(*)
        FROM games g JOIN moves m ON m.game_key = g.game_key
        WHERE g.opponent = 'heuristic' AND g.update_number >= ? AND m.action_kind IN ('playCard', 'elementalTuning')
        GROUP BY 1, 2, 3
    """
    counts = {(controller, kind, card): count for controller, kind, card, count in connection.execute(query, (TRAINED_FROM_UPDATE,))}
    games = connection.execute("SELECT COUNT(*) FROM games WHERE opponent = 'heuristic' AND update_number >= ?",
                               (TRAINED_FROM_UPDATE,)).fetchone()[0]
    connection.close()
    return {"counts": counts, "games": games}


def _by_category(card_counts: dict) -> dict:
    facts = card_facts()
    totals = defaultdict(int)
    for (controller, kind, card), count in card_counts["counts"].items():
        totals[(controller, kind, facts.get(card, ("event", 0, ()))[0])] += count
    return totals


def _card_use_table(card_counts: dict, tables: TableCollector) -> None:
    totals, games = _by_category(card_counts), card_counts["games"]
    rows = []
    for category in CARD_CATEGORIES:
        row = [category]
        for controller in ("learner", "heuristic"):
            played, tuned = totals[(controller, "playCard", category)], totals[(controller, "elementalTuning", category)]
            row += [f"{played / games:.2f}", f"{tuned / games:.2f}", f"{played / max(played + tuned, 1):.2f}"]
        rows.append(row)
    tables.add(f"Cards played vs tuned away, per game (vs heuristic, updates 400-600, {games:,} games)",
               ["card category", "learner played", "learner tuned", "learner share played",
                "heuristic played", "heuristic tuned", "heuristic share played"], rows,
               note="Elemental tuning discards a card to change one die to the active character's element.")


def _plot_card_use(card_counts: dict, folder: Path) -> None:
    totals, games = _by_category(card_counts), card_counts["games"]
    figure, axis = plt.subplots(figsize=(9, 4))
    positions = np.arange(len(CARD_CATEGORIES))
    for offset, controller in enumerate(("learner", "heuristic")):
        played = np.array([totals[(controller, "playCard", category)] for category in CARD_CATEGORIES]) / games
        tuned = np.array([totals[(controller, "elementalTuning", category)] for category in CARD_CATEGORIES]) / games
        x = positions + (offset - 0.5) * 0.4
        axis.bar(x, played, width=0.38, color=series_color(offset), label=f"{controller}: played")
        axis.bar(x, tuned, width=0.38, bottom=played, color=series_color(offset), alpha=0.35, label=f"{controller}: tuned away")
    axis.set_xticks(positions, CARD_CATEGORIES)
    axis.set_ylabel("cards per game")
    axis.set_title("What happens to cards: played or tuned into dice (games vs heuristic, updates 400-600)")
    axis.legend()
    save_figure(figure, folder, "cards_played_vs_tuned.png")


def _tuned_cards_table(card_counts: dict, names: NameLookup, tables: TableCollector) -> None:
    counts = card_counts["counts"]
    learner_cards = {card for controller, _, card in counts if controller == "learner"}
    entries = []
    for card in learner_cards:
        played, tuned = counts.get(("learner", "playCard", card), 0), counts.get(("learner", "elementalTuning", card), 0)
        heuristic_played = counts.get(("heuristic", "playCard", card), 0)
        heuristic_tuned = counts.get(("heuristic", "elementalTuning", card), 0)
        if played + tuned >= MINIMUM_CARD_EVENTS and heuristic_played + heuristic_tuned >= MINIMUM_CARD_EVENTS:
            entries.append((played / (played + tuned), card, played + tuned, heuristic_played / (heuristic_played + heuristic_tuned)))
    entries.sort()
    category = card_facts()
    rows = [[names.card(card), category.get(card, ("?",))[0], f"{events:,}", f"{share:.2f}", f"{heuristic_share:.2f}"]
            for share, card, events, heuristic_share in entries[:LISTED_CARDS]]
    rows.append(["...", "", "", "", ""])
    rows += [[names.card(card), category.get(card, ("?",))[0], f"{events:,}", f"{share:.2f}", f"{heuristic_share:.2f}"]
             for share, card, events, heuristic_share in entries[-LISTED_CARDS:]]
    tables.add("Cards the learner tunes away most / plays most (vs heuristic, updates 400-600)",
               ["card", "category", "learner plays + tunes", "learner share played", "heuristic share played"], rows,
               note=f"Cards with at least {MINIMUM_CARD_EVENTS} uses on each side. The heuristic only tunes when no "
                    "skill is affordable, so its share is a 'play it when possible' baseline.")
