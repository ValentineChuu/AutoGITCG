"""Per-seat rows from the `games` table: one row per (game, seat), row index = 2 * game index + seat.

Codes are small integers so every analysis can work on plain numpy arrays (no pandas in the venv).
"""

import sqlite3

import numpy as np

SOURCES = ("stage", "npc_decks", "collections", "tournament", "synergy", "chudledee")
OPPONENTS = ("self", "snapshot", "heuristic", "random", "reference")
CONTROLLERS = ("learner", "snapshot", "heuristic", "random", "reference")  # reference: the frozen network in mirror pairs


def load_game_rows(connection: sqlite3.Connection) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Columns per seat row, plus game_key -> game index."""
    games = connection.execute("""
        SELECT game_key, update_number, source, opponent, seat0_controller, seat1_controller,
               seat0_knows_opponent_deck, seat1_knows_opponent_deck, winner, rounds, seat0_defeated, seat1_defeated
        FROM games ORDER BY rowid
    """).fetchall()
    game_index = {row[0]: index for index, row in enumerate(games)}
    game_count = len(games)
    seat = np.tile(np.array([0, 1], dtype=np.int8), game_count)
    game = np.repeat(np.arange(game_count, dtype=np.int32), 2)

    def per_game(position: int, dtype) -> np.ndarray:
        return np.array([row[position] for row in games], dtype=dtype)

    def per_seat(position_seat0: int, position_seat1: int, dtype) -> np.ndarray:
        values = np.empty(2 * game_count, dtype=dtype)
        values[0::2] = per_game(position_seat0, dtype)
        values[1::2] = per_game(position_seat1, dtype)
        return values

    def coded(values: list[str], names: tuple[str, ...]) -> np.ndarray:
        return np.array([names.index(value) for value in values], dtype=np.int8)

    controller = np.empty(2 * game_count, dtype=np.int8)
    controller[0::2] = coded([row[4] for row in games], CONTROLLERS)
    controller[1::2] = coded([row[5] for row in games], CONTROLLERS)
    winner = np.array([-1 if row[8] is None else row[8] for row in games], dtype=np.int8)
    columns = {
        "game": game,
        "seat": seat,
        "update": np.repeat(per_game(1, np.int32), 2),
        "source": np.repeat(coded([row[2] for row in games], SOURCES), 2),
        "opponent": np.repeat(coded([row[3] for row in games], OPPONENTS), 2),
        "controller": controller,
        "opponent_controller": controller.reshape(-1, 2)[:, ::-1].reshape(-1),
        "knows_opponent_deck": per_seat(6, 7, np.int8),
        "won": (np.repeat(winner, 2) == seat).astype(np.int8),
        "round_limit": np.repeat(winner == -1, 2).astype(np.int8),
        "rounds": np.repeat(per_game(9, np.int16), 2),
        "own_defeated": per_seat(10, 11, np.int8),
        "opponent_defeated": per_seat(11, 10, np.int8),
    }
    return columns, game_index
