"""Database and benchmark statistics used by the alpha 2 review."""

import gzip
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from autogitcg.analysis.confidence import wilson_interval
from autogitcg.analysis.definition_kinds import CARD_CATEGORIES, card_facts, skill_facts
from autogitcg.progression.stage_catalog import build_stage_catalog

WINDOWS = {"early": (1, 200), "middle": (801, 1000), "late": (1801, 2000)}


def _rate(successes: int, count: int) -> dict:
    low, high = wilson_interval(successes, count)
    return {"successes": successes, "count": count, "rate": successes / count if count else math.nan,
            "low": low, "high": high}


def _mean(total: float, squared_total: float, count: int) -> dict:
    value = total / count if count else math.nan
    variance = max((squared_total - total * total / count) / (count - 1), 0) if count > 1 else math.nan
    half_width = 1.96 * math.sqrt(variance / count) if count > 1 else math.nan
    return {"mean": value, "half_width": half_width, "count": count}


def _seat_rows() -> str:
    return """
        SELECT game_key, update_number, source, opponent, 0 AS seat, seat0_controller AS controller,
               seat0_deck_origin AS origin, seat1_deck_origin AS opponent_origin,
               seat0_knows_opponent_deck AS knows_deck,
               winner = 0 AS won, rounds
        FROM games
        UNION ALL
        SELECT game_key, update_number, source, opponent, 1, seat1_controller,
               seat1_deck_origin, seat0_deck_origin, seat1_knows_opponent_deck, winner = 1, rounds
        FROM games
    """


def reaction_statistics(connection: sqlite3.Connection, decks_file: Path) -> dict:
    rows = connection.execute(f"""
        WITH seats AS ({_seat_rows()}), totals AS (
          SELECT game_key, seat, SUM(count) AS reactions FROM game_reactions GROUP BY game_key, seat)
        SELECT seats.update_number, seats.opponent, seats.controller, seats.origin, seats.won,
               COALESCE(totals.reactions, 0)
        FROM seats LEFT JOIN totals USING (game_key, seat)
    """).fetchall()
    result = {"windows": {}, "synergy_archetypes": {}}
    for window, (low, high) in WINDOWS.items():
        result["windows"][window] = {}
        for label, predicate in {
            "learner_all": lambda row: row[2] == "learner",
            "learner_vs_heuristic": lambda row: row[2] == "learner" and row[1] == "heuristic",
            "heuristic_same_games": lambda row: row[2] == "heuristic" and row[1] == "heuristic",
        }.items():
            values = np.array([row[5] for row in rows if low <= row[0] <= high and predicate(row)], dtype=float)
            result["windows"][window][label] = _mean(float(values.sum()), float((values * values).sum()), len(values))

    decks = json.loads(decks_file.read_text())
    archetype = {index: deck["archetype"] for index, deck in enumerate(decks)}
    reaction_rows = connection.execute(f"""
        WITH seats AS ({_seat_rows()})
        SELECT seats.game_key, seats.update_number, seats.controller, seats.origin, reaction.reaction, reaction.count
        FROM seats LEFT JOIN game_reactions reaction USING (game_key, seat)
        WHERE seats.origin LIKE 'synergy:%'
    """).fetchall()
    grouped: dict[tuple[str, str, str], dict] = {}
    game_counts = connection.execute(f"""
        WITH seats AS ({_seat_rows()})
        SELECT CASE WHEN update_number <= 200 THEN 'early' WHEN update_number BETWEEN 1801 AND 2000 THEN 'late' END,
               controller, origin, COUNT(*)
        FROM seats WHERE origin LIKE 'synergy:%' AND (update_number <= 200 OR update_number >= 1801)
        GROUP BY 1, 2, 3
    """).fetchall()
    for window, controller, origin, count in game_counts:
        kind = archetype[int(origin.split(":")[1])]
        entry = grouped.setdefault((window, kind, controller), {"games": 0, "reactions": defaultdict(int)})
        entry["games"] += count
    per_game_reactions = defaultdict(lambda: defaultdict(int))
    for game_key, update, controller, origin, reaction, count in reaction_rows:
        window = "early" if update <= 200 else "late" if update >= 1801 else None
        if window is None or reaction is None:
            continue
        kind = archetype[int(origin.split(":")[1])]
        grouped[(window, kind, controller)]["reactions"][reaction] += count
        per_game_reactions[(window, kind, controller, game_key)][reaction] += count
    intended = {"vaporize": ("vaporize",), "melt": ("melt",), "swirl": ("swirl_cryo", "swirl_hydro", "swirl_pyro", "swirl_electro")}
    squared = defaultdict(lambda: [0, 0])
    for (window, kind, controller, _), game_reactions in per_game_reactions.items():
        names = intended.get(kind, (kind,))
        intended_count = sum(game_reactions.get(name, 0) for name in names)
        all_count = sum(game_reactions.values())
        squared[(window, kind, controller)][0] += intended_count * intended_count
        squared[(window, kind, controller)][1] += all_count * all_count
    for (window, kind, controller), entry in grouped.items():
        reactions = dict(entry["reactions"])
        entry["reactions"] = reactions
        names = intended.get(kind, (kind,))
        intended_total = sum(reactions.get(name, 0) for name in names)
        all_total = sum(reactions.values())
        intended_squared, all_squared = squared[(window, kind, controller)]
        intended_mean = _mean(intended_total, intended_squared, entry["games"])
        all_mean = _mean(all_total, all_squared, entry["games"])
        entry["intended_per_game"] = intended_mean["mean"]
        entry["intended_half_width"] = intended_mean["half_width"]
        entry["all_per_game"] = all_mean["mean"]
        entry["all_half_width"] = all_mean["half_width"]
        result["synergy_archetypes"].setdefault(kind, {}).setdefault(window, {})[controller] = entry
    return result


def card_statistics(connection: sqlite3.Connection) -> dict:
    facts = card_facts()
    result = {}
    for window, (low, high) in WINDOWS.items():
        games = connection.execute("SELECT COUNT(*) FROM games WHERE opponent = 'heuristic' AND update_number BETWEEN ? AND ?",
                                   (low, high)).fetchone()[0]
        rows = connection.execute("""
            SELECT CASE WHEN m.seat = 0 THEN g.seat0_controller ELSE g.seat1_controller END,
                   m.action_kind, m.definition_id, COUNT(*)
            FROM moves m JOIN games g USING (game_key)
            WHERE g.opponent = 'heuristic' AND g.update_number BETWEEN ? AND ?
              AND m.action_kind IN ('playCard', 'elementalTuning')
            GROUP BY 1, 2, 3
        """, (low, high)).fetchall()
        counts = defaultdict(int)
        for controller, action, card, count in rows:
            category = facts.get(card, ("event", 0, ()))[0]
            counts[(controller, action, category)] += count
        window_result = {"games": games}
        for controller in ("learner", "heuristic"):
            categories = {}
            for category in CARD_CATEGORIES:
                played = counts[(controller, "playCard", category)]
                tuned = counts[(controller, "elementalTuning", category)]
                categories[category] = {"played_per_game": played / games, "tuned_per_game": tuned / games,
                                        "share_played": played / max(played + tuned, 1), "events": played + tuned}
            window_result[controller] = categories
            window_result[f"{controller}_cards_per_game"] = sum(value["played_per_game"] for value in categories.values())
            window_result[f"{controller}_tunings_per_game"] = sum(value["tuned_per_game"] for value in categories.values())
        result[window] = window_result
    return result


def calibration_statistics(connection: sqlite3.Connection) -> dict:
    grouped = connection.execute("""
        WITH first_move AS (
          SELECT m.game_key, m.seat, m.round, MIN(m.move_number) AS move_number
          FROM moves m WHERE m.win_probability IS NOT NULL AND m.decision_kind = 'action'
          GROUP BY m.game_key, m.seat, m.round), samples AS (
        SELECT g.update_number, g.opponent, m.win_probability AS probability, g.winner = m.seat AS outcome
        FROM first_move f JOIN moves m
          ON m.game_key = f.game_key AND m.seat = f.seat AND m.move_number = f.move_number
        JOIN games g USING (game_key))
        SELECT CASE WHEN update_number >= 1500 THEN 'late' ELSE 'earlier' END, opponent,
               MIN(CAST(probability * 10 AS INTEGER), 9), COUNT(*), SUM(probability), SUM(outcome),
               SUM((probability - outcome) * (probability - outcome))
        FROM samples GROUP BY 1, 2, 3
    """).fetchall()
    def summarize(rows: list[tuple]) -> dict:
        combined = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
        for _, _, bin_number, count, probability_sum, outcome_sum, squared_error in rows:
            combined[bin_number][0] += count
            combined[bin_number][1] += probability_sum
            combined[bin_number][2] += outcome_sum
            combined[bin_number][3] += squared_error
        bins = [{"predicted": values[1] / values[0], "actual": values[2] / values[0], "count": values[0]}
                for _, values in sorted(combined.items())]
        count = sum(values[0] for values in combined.values())
        outcomes = sum(row[5] for row in rows)
        low, high = wilson_interval(outcomes, count)
        return {"count": count, "mean_predicted": sum(row[4] for row in rows) / count,
                "actual": outcomes / count, "actual_low": low, "actual_high": high,
                "brier": sum(row[6] for row in rows) / count,
                "calibration_error": sum(abs(row["actual"] - row["predicted"]) * row["count"] for row in bins) / count,
                "bins": bins}

    result = {"all": summarize(grouped), "late_all": summarize([row for row in grouped if row[0] == "late"])}
    for opponent in ("self", "snapshot", "heuristic", "random"):
        result[f"late_{opponent}"] = summarize([row for row in grouped if row[0] == "late" and row[1] == opponent])
    return result


def loss_statistics(connection: sqlite3.Connection, decks_file: Path) -> dict:
    decks = json.loads(decks_file.read_text())
    archetypes = {index: deck["archetype"] for index, deck in enumerate(decks)}
    rows = connection.execute(f"""
        WITH seats AS ({_seat_rows()})
        SELECT update_number, source, opponent, origin, opponent_origin, knows_deck, won, rounds
        FROM seats WHERE controller = 'learner' AND update_number >= 1500
    """).fetchall()

    def grouped_rate(key):
        groups = defaultdict(list)
        for row in rows:
            groups[key(row)].append(bool(row[6]))
        return {str(name): _rate(sum(values), len(values)) for name, values in groups.items() if name is not None}

    result = {
        "by_origin_type": grouped_rate(lambda row: row[3].split(":")[0] if row[3] else None),
        "by_source": grouped_rate(lambda row: row[1]),
        "by_opponent": grouped_rate(lambda row: row[2]),
        "by_knowledge": grouped_rate(lambda row: "known" if row[5] else "blind"),
        "synergy_archetype": grouped_rate(lambda row: archetypes[int(row[3].split(":")[1])]
                                            if row[3] and row[3].startswith("synergy:") else None),
    }
    stage_kind = {stage.level_id: stage.kind for stage in build_stage_catalog()[0]}
    result["stage_kind_player_side"] = grouped_rate(
        lambda row: stage_kind.get(int(row[4].split(":")[1]))
        if row[1] == "stage" and row[4] and row[4].startswith("stage:") else None)
    result["stage_kind_player_side_by_knowledge"] = grouped_rate(
        lambda row: f"{stage_kind.get(int(row[4].split(':')[1]))}|{'known' if row[5] else 'blind'}"
        if row[1] == "stage" and row[4] and row[4].startswith("stage:") else None)
    result["stage_kind_player_origin_opponent"] = grouped_rate(
        lambda row: (f"{stage_kind.get(int(row[4].split(':')[1]))}|{row[3].split(':')[0]}|{row[2]}")
        if row[1] == "stage" and row[4] and row[4].startswith("stage:") else None)
    return result


def heuristic_behaviour(connection: sqlite3.Connection) -> dict:
    skills, cards = skill_facts(), card_facts()
    rows = connection.execute("""
        SELECT g.winner = CASE WHEN g.seat0_controller = 'heuristic' THEN 0 ELSE 1 END AS heuristic_won,
               m.action_kind, m.definition_id, COUNT(*), SUM(g.rounds)
        FROM games g JOIN moves m USING (game_key)
        WHERE g.opponent = 'heuristic' AND g.update_number >= 1500
          AND m.seat = CASE WHEN g.seat0_controller = 'heuristic' THEN 0 ELSE 1 END
        GROUP BY 1, 2, 3
    """).fetchall()
    game_rows = connection.execute("""
        SELECT winner = CASE WHEN seat0_controller = 'heuristic' THEN 0 ELSE 1 END, COUNT(*), SUM(rounds)
        FROM games WHERE opponent = 'heuristic' AND update_number >= 1500 GROUP BY 1
    """).fetchall()
    game_counts = {won: count for won, count, _ in game_rows}
    round_counts = {won: rounds for won, _, rounds in game_rows}
    totals = defaultdict(int)
    for won, action, definition, count, _ in rows:
        if action == "useSkill":
            category = f"skill_{skills.get(definition, ('other', 0))[0]}"
        elif action == "playCard":
            category = f"card_{cards.get(definition, ('event', 0, ()))[0]}"
        else:
            category = action
        totals[(bool(won), category)] += count
    result = {}
    for won in (False, True):
        result["wins" if won else "losses"] = {
            category: {"per_game": count / game_counts[int(won)], "per_round": count / round_counts[int(won)]}
            for (outcome, category), count in totals.items() if outcome == won and category}
    result["games"] = {"wins": game_counts.get(1, 0), "losses": game_counts.get(0, 0)}
    result["rounds"] = {"wins": round_counts.get(1, 0), "losses": round_counts.get(0, 0)}
    return result


def benchmark_statistics(folder: Path) -> dict:
    by_update = {}
    late_cells = defaultdict(list)
    for path in sorted(folder.glob("update_*.jsonl.gz")):
        update = int(path.name.split("_")[1].split(".")[0])
        with gzip.open(path, "rt", encoding="utf-8") as source:
            rows = [json.loads(line) for line in source]
        heuristic = [row for row in rows if row["opponent"] == "heuristic" and "failure" not in row]
        by_update[update] = _rate(sum(row["learner_won"] for row in heuristic), len(heuristic))
        if update >= 1800:
            for row in heuristic:
                late_cells[(row["matchup"], row["deck_order"], row["learner_seat"])].append(row["learner_won"])
    final_update = max(by_update)
    with gzip.open(folder / f"update_{final_update:05d}.jsonl.gz", "rt", encoding="utf-8") as source:
        final_rows = [json.loads(line) for line in source]
    final_losses = [row for row in final_rows if row["opponent"] == "heuristic" and not row.get("learner_won", False)]
    persistence = [sum(late_cells[(row["matchup"], row["deck_order"], row["learner_seat"])]) /
                   len(late_cells[(row["matchup"], row["deck_order"], row["learner_seat"])]) for row in final_losses]
    return {"updates": by_update, "final_losses": len(final_losses),
            "final_loss_cell_prior_late_win_rate_mean": float(np.mean(persistence)),
            "final_losses_in_cells_lost_at_least_half_late": sum(rate <= 0.5 for rate in persistence)}


def write_figures(result: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    updates = sorted(result["benchmark"]["updates"])
    axes[0, 0].plot(updates, [result["benchmark"]["updates"][update]["rate"] for update in updates])
    axes[0, 0].set(title="Mirror benchmark vs heuristic", xlabel="update", ylabel="win rate", ylim=(0, 1))
    for label in ("learner_vs_heuristic", "heuristic_same_games"):
        axes[0, 1].plot(list(WINDOWS), [result["reactions"]["windows"][window][label]["mean"] for window in WINDOWS],
                        marker="o", label=label.replace("_", " "))
    axes[0, 1].set(title="Elemental reactions per game", ylabel="mean reactions")
    axes[0, 1].legend(fontsize=8)
    for controller in ("learner", "heuristic"):
        axes[1, 0].plot(list(WINDOWS), [result["cards"][window][f"{controller}_cards_per_game"] for window in WINDOWS],
                        marker="o", label=controller)
    axes[1, 0].set(title="Cards played vs heuristic", ylabel="cards per game")
    axes[1, 0].legend()
    for label, color in (("late_self", "C0"), ("late_snapshot", "C1"), ("late_heuristic", "C2"), ("late_random", "C3")):
        bins = result["calibration"][label]["bins"]
        axes[1, 1].plot([row["predicted"] for row in bins], [row["actual"] for row in bins], marker="o",
                        label=label.removeprefix("late_"), color=color)
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 1].set(title="Round-start win-head calibration (updates 1500-2000)", xlabel="predicted", ylabel="actual")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(folder / "training_behaviour_overview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    categories = list(CARD_CATEGORIES)
    positions = np.arange(len(categories))
    for offset, controller in enumerate(("learner", "heuristic")):
        category_rows = result["cards"]["late"][controller]
        played = np.array([category_rows[category]["played_per_game"] for category in categories])
        tuned = np.array([category_rows[category]["tuned_per_game"] for category in categories])
        x = positions + (offset - 0.5) * 0.38
        axes[0].bar(x, played, width=0.36, label=f"{controller}: played")
        axes[0].bar(x, tuned, width=0.36, bottom=played, alpha=0.35, label=f"{controller}: tuned")
    axes[0].set_xticks(positions, categories)
    axes[0].set(title="Late card use vs heuristic", ylabel="cards per game")
    axes[0].legend(fontsize=7)
    archetypes = ("vaporize", "melt", "swirl")
    x = np.arange(len(archetypes))
    for offset, window in enumerate(("early", "late")):
        values = [result["reactions"]["synergy_archetypes"][kind][window]["learner"]["intended_per_game"]
                  for kind in archetypes]
        axes[1].bar(x + (offset - 0.5) * 0.36, values, width=0.34, label=window)
    axes[1].set_xticks(x, archetypes)
    axes[1].set(title="Intended reactions in synergy decks (learner)", ylabel="intended reactions per game")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(folder / "cards_and_synergy_reactions.png", dpi=160)
    plt.close(figure)

    origins = result["losses"]["by_origin_type"]
    labels = sorted(origins)
    rates = [origins[label]["rate"] for label in labels]
    errors = [[origins[label]["rate"] - origins[label]["low"] for label in labels],
              [origins[label]["high"] - origins[label]["rate"] for label in labels]]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    axis.bar(labels, rates, yerr=errors, capsize=3)
    axis.set(title="Late learner win rate by its deck origin", ylabel="win rate", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(folder / "late_win_rate_by_deck_origin.png", dpi=160)
    plt.close(figure)


def write_run_statistics(database: Path, benchmarks: Path, decks: Path, folder: Path, output: Path) -> dict:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    result = {
        "counts": dict(connection.execute("SELECT 'games', COUNT(*) FROM games UNION ALL SELECT 'moves', COUNT(*) FROM moves")),
        "reactions": reaction_statistics(connection, decks),
        "cards": card_statistics(connection),
        "calibration": calibration_statistics(connection),
        "losses": loss_statistics(connection, decks),
        "heuristic_behaviour": heuristic_behaviour(connection),
        "benchmark": benchmark_statistics(benchmarks),
    }
    connection.close()
    write_figures(result, folder)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
