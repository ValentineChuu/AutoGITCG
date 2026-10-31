"""The per-seat analysis table: games + move aggregates + deck facts, cached as one .npz file.

Building it reads the whole game database once (a few minutes for alpha 1); every report section then works
on these numpy columns. Rows are (game, seat) with row index = 2 * game index + seat, except the card_* columns,
which are a sparse list (card_row, card_id, card_copies).
"""

import sqlite3
from pathlib import Path

import numpy as np

from experiments.early_run_reviews.alpha_1_statistics.deck_rows import load_deck_rows
from experiments.early_run_reviews.alpha_1_statistics.game_rows import load_game_rows
from experiments.early_run_reviews.alpha_1_statistics.move_aggregates import aggregate_moves

CACHE_VERSION = 2


def open_read_only(database: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)


def build_seat_table(database: Path) -> dict[str, np.ndarray]:
    connection = open_read_only(database)
    print("reading games ...", flush=True)
    columns, game_index = load_game_rows(connection)
    row_count = len(columns["seat"])
    print("reading decks ...", flush=True)
    columns.update(load_deck_rows(connection, game_index, row_count, columns["source"]))
    print("reading moves (one pass) ...", flush=True)
    columns.update(aggregate_moves(connection, game_index, row_count))
    connection.close()
    return columns


def load_seat_table(database: Path, cache: Path, rebuild: bool = False) -> dict[str, np.ndarray]:
    if cache.exists() and not rebuild:
        with np.load(cache) as stored:
            if int(stored["cache_version"]) == CACHE_VERSION:
                return {name: stored[name] for name in stored.files if name != "cache_version"}
    columns = build_seat_table(database)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, cache_version=np.array(CACHE_VERSION), **columns)
    return columns
