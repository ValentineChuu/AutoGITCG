"""A compact, self-describing CSV bundle of a run's games for analysis in outside tools (e.g. R, pandas, a spreadsheet).

The full SQLite database is too large to upload, so the per-move table is summarized per game and seat,
and full move-by-move data is kept only for a random sample of games. Output folder (zipped too):

    README.md                 what every file and column means
    games.csv                 one row per game
    game_characters.csv       lineup per seat
    game_cards.csv            deck contents per seat
    game_equipment.csv        pre-equipped cards (Serious Showdown talents)
    seat_summary.csv          per game and seat: move counts by kind, skill types, cards played vs tuned, ...
    moves_sample.csv          every move of a random sample of games
    definitions.csv           English names for ids
    training_metrics.csv      one row per training update

Usage:
    python scripts/training/export_for_external_analysis.py --run runs/alpha_01 [--sample-games 5000]
"""

import argparse
import csv
import json
import shutil
import sqlite3
from pathlib import Path

from autogitcg.data.piovium_catalog import load_catalog

README = Path(__file__).resolve().parent / "external_analysis_readme.md"
DIRECT_TABLES = ("games", "game_characters", "game_cards", "game_equipment", "definitions")
METRIC_COLUMNS = ["update", "learner_steps", "learning_rate", "entropy", "approx_kl", "clip_fraction", "value_loss",
                  "explained_variance", "round_limit_rate", "policy_lag", "epochs_completed"]

SEAT_SUMMARY = """
SELECT m.game_key, m.seat,
    COUNT(*) AS moves,
    SUM(m.decision_kind = 'action') AS action_decisions,
    SUM(m.action_kind = 'useSkill') AS skills_used,
    SUM(s.skill_type = 'normal') AS normal_attacks,
    SUM(s.skill_type = 'elemental') AS elemental_skills,
    SUM(s.skill_type = 'burst') AS bursts,
    SUM(m.action_kind = 'playCard') AS cards_played,
    SUM(m.action_kind = 'elementalTuning') AS cards_tuned,
    SUM(m.action_kind = 'switchActive') AS switches,
    SUM(m.decision_kind = 'chooseActive') AS forced_active_choices,
    SUM(CASE WHEN m.decision_kind = 'rerollDice' THEN m.definition_id ELSE 0 END) AS dice_rerolled,
    SUM(CASE WHEN m.decision_kind = 'switchHands' THEN m.definition_id ELSE 0 END) AS opening_cards_swapped,
    ROUND(AVG(m.chosen_probability), 4) AS mean_chosen_probability,
    ROUND(AVG(m.entropy), 4) AS mean_entropy,
    MIN(m.round) AS first_round, MAX(m.round) AS last_round
FROM moves m LEFT JOIN skill_types s ON s.skill_id = m.definition_id AND m.action_kind = 'useSkill'
GROUP BY m.game_key, m.seat
"""
FIRST_AND_LAST_VALUE = """
SELECT game_key, seat, value FROM moves WHERE value IS NOT NULL AND (game_key, seat, move_number) IN (
    SELECT game_key, seat, {pick}(move_number) FROM moves WHERE value IS NOT NULL GROUP BY game_key, seat)
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sample-games", type=int, default=5000)
    parser.add_argument("--output", type=Path, help="default: <run>/external_analysis")
    arguments = parser.parse_args()
    output = arguments.output or arguments.run / "external_analysis"
    output.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{arguments.run / 'games.sqlite'}?mode=ro", uri=True)
    connection.execute("CREATE TEMP TABLE skill_types (skill_id INTEGER PRIMARY KEY, skill_type TEXT)")
    connection.executemany("INSERT OR IGNORE INTO temp.skill_types VALUES (?, ?)", [
        (skill["id"], skill["skillType"]) for character in load_catalog()["characters"] for skill in character["skills"]])

    for table in DIRECT_TABLES:
        _write_query(connection, f"SELECT * FROM {table}", output / f"{table}.csv")
    _write_seat_summary(connection, output / "seat_summary.csv")
    _write_query(connection, f"""
        SELECT m.* FROM moves m JOIN (SELECT game_key FROM games ORDER BY random() LIMIT {int(arguments.sample_games)}) s
        ON s.game_key = m.game_key ORDER BY m.game_key, m.move_number""", output / "moves_sample.csv")
    _write_metrics(arguments.run / "metrics.jsonl", output / "training_metrics.csv")
    shutil.copy(README, output / "README.md")
    archive = shutil.make_archive(str(output), "zip", output)
    print(f"wrote {output} and {archive}")


def _write_query(connection: sqlite3.Connection, query: str, path: Path) -> None:
    cursor = connection.execute(query)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column[0] for column in cursor.description)
        writer.writerows(cursor)


def _write_seat_summary(connection: sqlite3.Connection, path: Path) -> None:
    first = {(key, seat): value for key, seat, value in connection.execute(FIRST_AND_LAST_VALUE.format(pick="MIN"))}
    last = {(key, seat): value for key, seat, value in connection.execute(FIRST_AND_LAST_VALUE.format(pick="MAX"))}
    cursor = connection.execute(SEAT_SUMMARY)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([column[0] for column in cursor.description] + ["first_value", "last_value"])
        for row in cursor:
            writer.writerow(list(row) + [first.get((row[0], row[1])), last.get((row[0], row[1]))])


def _write_metrics(metrics_path: Path, path: Path) -> None:
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(METRIC_COLUMNS + ["win_rate_vs_heuristic", "win_rate_vs_snapshot", "win_rate_vs_random",
                                          "benchmark_vs_heuristic", "benchmark_vs_random"])
        for record in records:
            win_rates = record.get("win_rates", {})
            writer.writerow([record.get(name) for name in METRIC_COLUMNS] + [
                win_rates.get("heuristic"), win_rates.get("snapshot"), win_rates.get("random"),
                record.get("evaluation_heuristic", {}).get("win_rate"), record.get("evaluation_random", {}).get("win_rate")])


if __name__ == "__main__":
    main()
