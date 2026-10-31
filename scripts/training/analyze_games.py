"""Review plots from a run's game database (build it first with scripts/training/export_games.py).

Writes into <run>/plots/:
    value_calibration.png  predicted win chance (value head) vs actual win rate, early / middle / late in training
    seat_and_source.png    learner win rate by seat and by matchup source, over training
    game_length.png        rounds per game over training, per opponent

Usage:
    python scripts/training/analyze_games.py --run runs/alpha_01
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be chosen first)
import numpy as np  # noqa: E402

CALIBRATION_BINS = 10
UPDATE_BUCKET = 50


def _learner_outcomes(connection: sqlite3.Connection):
    """(update, seat, won, source, opponent, rounds) for every learner seat in every game."""
    return connection.execute("""
        SELECT update_number, 0, COALESCE(winner, -1) = 0, source, opponent, rounds FROM games WHERE seat0_controller = 'learner'
        UNION ALL
        SELECT update_number, 1, COALESCE(winner, -1) = 1, source, opponent, rounds FROM games WHERE seat1_controller = 'learner'
    """).fetchall()


def plot_value_calibration(connection: sqlite3.Connection, folder: Path) -> None:
    rows = np.array(connection.execute("""
        SELECT g.update_number, m.value, CASE WHEN g.winner = m.seat THEN 1.0 ELSE 0.0 END
        FROM moves m JOIN games g ON g.game_key = m.game_key
        WHERE m.value IS NOT NULL AND m.decision_kind = 'action'
    """).fetchall(), dtype=np.float64)
    updates, predicted, won = rows[:, 0], (rows[:, 1] + 1) / 2, rows[:, 2]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect calibration")
    thirds = np.quantile(updates, [0, 1 / 3, 2 / 3, 1])
    for label, low, high in zip(("early", "middle", "late"), thirds[:-1], thirds[1:]):
        selected = (updates >= low) & (updates <= high)
        bins = np.clip((predicted[selected] * CALIBRATION_BINS).astype(int), 0, CALIBRATION_BINS - 1)
        centers, rates = [], []
        for number in range(CALIBRATION_BINS):
            in_bin = bins == number
            if in_bin.sum() >= 200:
                centers.append(predicted[selected][in_bin].mean())
                rates.append(won[selected][in_bin].mean())
        axis.plot(centers, rates, marker="o", label=f"{label} (updates {int(low)}-{int(high)})")
    axis.set_xlabel("predicted win chance (value head)")
    axis.set_ylabel("actual win rate")
    axis.set_title("Value calibration (learner action decisions)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(folder / "value_calibration.png", dpi=120)


def _rate_by_bucket(rows, key_index: int):
    series: dict[str, dict[int, list[int]]] = {}
    for row in rows:
        bucket = row[0] // UPDATE_BUCKET * UPDATE_BUCKET
        counts = series.setdefault(str(row[key_index]), {}).setdefault(bucket, [0, 0])
        counts[0] += row[2]
        counts[1] += 1
    return {key: sorted((bucket, wins / total) for bucket, (wins, total) in buckets.items() if total >= 30)
            for key, buckets in series.items()}


def plot_seat_and_source(connection: sqlite3.Connection, folder: Path) -> None:
    outcomes = _learner_outcomes(connection)
    self_play = [row for row in outcomes if row[4] == "self"]
    against_heuristic = [row for row in outcomes if row[4] == "heuristic"]
    figure, (seat_axis, source_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for seat, points in sorted(_rate_by_bucket(self_play, 1).items()):
        seat_axis.plot(*zip(*points), marker="o", label=f"seat {seat}")
    seat_axis.set_title("Self-play win rate by seat")
    for source, points in sorted(_rate_by_bucket(against_heuristic, 3).items()):
        source_axis.plot(*zip(*points), marker="o", label=source)
    source_axis.set_title("Win rate vs heuristic by matchup source")
    for axis in (seat_axis, source_axis):
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    source_axis.set_xlabel(f"update (buckets of {UPDATE_BUCKET})")
    figure.tight_layout()
    figure.savefig(folder / "seat_and_source.png", dpi=120)


def plot_game_length(connection: sqlite3.Connection, folder: Path) -> None:
    rows = connection.execute("""
        SELECT update_number / ? * ?, opponent, AVG(rounds), COUNT(*) FROM games GROUP BY 1, 2 HAVING COUNT(*) >= 30
    """, (UPDATE_BUCKET, UPDATE_BUCKET)).fetchall()
    figure, axis = plt.subplots(figsize=(10, 4))
    for opponent in sorted({row[1] for row in rows}):
        points = [(row[0], row[2]) for row in rows if row[1] == opponent]
        axis.plot(*zip(*points), marker="o", label=opponent)
    axis.set_title("Average rounds per game")
    axis.set_xlabel(f"update (buckets of {UPDATE_BUCKET})")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(folder / "game_length.png", dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--database", type=Path, help="default: <run>/games.sqlite")
    arguments = parser.parse_args()
    connection = sqlite3.connect(arguments.database or arguments.run / "games.sqlite")
    folder = arguments.run / "plots"
    folder.mkdir(exist_ok=True)
    plot_value_calibration(connection, folder)
    plot_seat_and_source(connection, folder)
    plot_game_length(connection, folder)
    print(f"wrote review plots to {folder}")


if __name__ == "__main__":
    main()
