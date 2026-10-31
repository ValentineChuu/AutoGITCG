"""Figures and tables for the alpha 1 statistics review (reports/alpha_1_statistics.md).

Reads the run's game database once into a per-seat cache (data/derived/statistics_<run>.npz, about 1 minute for
alpha 1), then writes PNG figures and every generated table (tables.md) into the figures folder. Never writes
to the run folder. The sections are fixed to alpha 1: most look at updates 400-600 and its games against the
scripted heuristic, so later runs (no heuristic opponent, other update ranges) leave them empty.

Usage:
    python experiments/early_run_reviews/review_alpha_1_statistics.py --run runs/alpha_01
    python experiments/early_run_reviews/review_alpha_1_statistics.py --run runs/alpha_01 --sections calibration flow --rebuild-cache
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from experiments.early_run_reviews.alpha_1_statistics.deck_effects import report_deck_effects
from experiments.early_run_reviews.alpha_1_statistics.game_flow import report_game_flow
from experiments.early_run_reviews.alpha_1_statistics.heuristic_games import report_heuristic_games
from experiments.early_run_reviews.alpha_1_statistics.markdown_tables import TableCollector
from experiments.early_run_reviews.alpha_1_statistics.name_lookup import NameLookup
from experiments.early_run_reviews.alpha_1_statistics.outcome_factors import report_outcome_factors
from experiments.early_run_reviews.alpha_1_statistics.seat_and_knowledge import report_seat_and_knowledge
from experiments.early_run_reviews.alpha_1_statistics.seat_table import load_seat_table, open_read_only
from experiments.early_run_reviews.alpha_1_statistics.training_evolution import report_training_evolution
from experiments.early_run_reviews.alpha_1_statistics.value_calibration import report_value_calibration

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass
class ReportInputs:
    table: dict
    names: NameLookup
    database: Path
    folder: Path
    tables: TableCollector


SECTIONS = {
    "evolution": lambda inputs: report_training_evolution(inputs.table, inputs.folder, inputs.tables),
    "outcomes": lambda inputs: report_outcome_factors(inputs.table, inputs.folder, inputs.tables),
    "decks": lambda inputs: report_deck_effects(inputs.table, inputs.names, inputs.folder, inputs.tables),
    "flow": lambda inputs: report_game_flow(inputs.table, inputs.folder, inputs.tables),
    "calibration": lambda inputs: report_value_calibration(inputs.table, inputs.folder, inputs.tables),
    "seats": lambda inputs: report_seat_and_knowledge(inputs.table, inputs.folder, inputs.tables),
    "heuristic": lambda inputs: report_heuristic_games(inputs.table, inputs.names, inputs.folder, inputs.tables,
                                                       inputs.database),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database", type=Path, help="default: <run>/games.sqlite")
    parser.add_argument("--cache", type=Path, help="default: data/derived/statistics_<run name>.npz")
    parser.add_argument("--figures", type=Path, help="default: <run>/report/figures")
    parser.add_argument("--sections", nargs="+", choices=list(SECTIONS), default=list(SECTIONS))
    parser.add_argument("--rebuild-cache", action="store_true")
    arguments = parser.parse_args()

    run_name = arguments.run.resolve().name
    database = arguments.database or arguments.run / "games.sqlite"
    cache = arguments.cache or REPOSITORY / "data" / "derived" / f"statistics_{run_name}.npz"
    folder = arguments.figures or arguments.run / "report" / "figures"
    folder.mkdir(parents=True, exist_ok=True)

    table = load_seat_table(database, cache, rebuild=arguments.rebuild_cache)
    connection = open_read_only(database)
    names = NameLookup(connection)
    connection.close()
    inputs = ReportInputs(table, names, database, folder, TableCollector())
    for section in arguments.sections:
        print(f"\n== {section} ==", flush=True)
        SECTIONS[section](inputs)
    every_section = arguments.sections == list(SECTIONS)
    tables_file = folder / ("tables.md" if every_section else f"tables_{'_'.join(arguments.sections)}.md")
    inputs.tables.write(tables_file, f"Generated tables for {run_name}")
    print(f"\nwrote figures and tables to {folder}")


if __name__ == "__main__":
    main()
