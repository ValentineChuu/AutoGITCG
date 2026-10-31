"""Export a run's raw game logs into SQLite (and optionally CSV per table) for statistics tools.

Incremental: files already loaded are skipped, so it can be rerun while training continues.

Usage:
    python scripts/training/export_games.py --run runs/main                 # -> runs/main/games.sqlite
    python scripts/training/export_games.py --run runs/main --csv runs/main/csv
"""

import argparse
import csv
import sqlite3
from pathlib import Path

from autogitcg.analysis.definition_names import definition_names
from autogitcg.analysis.game_database import create_indexes, export_games_file, load_definition_names, open_database

TABLES = ("games", "game_characters", "game_cards", "game_equipment", "game_reactions", "game_reaction_causes", "moves", "definitions")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database", type=Path, help="default: <run>/games.sqlite")
    parser.add_argument("--csv", type=Path, help="also write one CSV per table into this folder")
    arguments = parser.parse_args()

    database_path = arguments.database or arguments.run / "games.sqlite"
    connection = open_database(database_path)
    load_definition_names(connection, definition_names())
    added = 0
    for path in sorted((arguments.run / "games").glob("update_*.jsonl.gz")):
        added += export_games_file(connection, path)
        connection.commit()
    create_indexes(connection)
    connection.commit()
    total = connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"added {added} games; {total} in {database_path}")
    if arguments.csv:
        _write_csv(connection, arguments.csv)


def _write_csv(connection: sqlite3.Connection, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        cursor = connection.execute(f"SELECT * FROM {table}")
        with (folder / f"{table}.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(column[0] for column in cursor.description)
            writer.writerows(cursor)
    print(f"CSV files in {folder}")


if __name__ == "__main__":
    main()
