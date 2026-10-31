"""SQLite database built from a run's raw game logs (games/*.jsonl.gz), for statistics.

Tables:
    games            one row per game (matchup source, controllers per seat, deck origins, winner, rounds, damage, ...)
    game_characters  lineup per seat, in slot order
    game_cards       deck contents per seat (copies per card)
    game_equipment   pre-equipped cards (Serious Showdown talents)
    game_reactions   elemental reactions triggered per seat (runs from alpha 2 on)
    game_reaction_causes  reactions by cause: caller type (character/summon/status...) and skill type (alpha 3 on)
    moves            every move in order; network moves include chosen probability, entropy, value, win chance, option count
    definitions      English names for character / card / skill ids
    exported_files   which raw files are already loaded (export is incremental)

`game_key` is "<update>:<line>" and joins everything. Seats are 0 and 1; `winner` is NULL for round-limit games.
"""

import gzip
import json
import sqlite3
from collections import Counter
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_key TEXT PRIMARY KEY, update_number INTEGER, policy_version INTEGER, source TEXT, opponent TEXT,
    seat0_controller TEXT, seat1_controller TEXT, seat0_knows_opponent_deck INTEGER, seat1_knows_opponent_deck INTEGER,
    seed INTEGER, winner INTEGER, rounds INTEGER, seat0_defeated INTEGER, seat1_defeated INTEGER, move_count INTEGER,
    seat0_deck_origin TEXT, seat1_deck_origin TEXT, snapshot_version INTEGER, actor_index INTEGER,
    seat0_damage_dealt INTEGER, seat1_damage_dealt INTEGER
);
CREATE TABLE IF NOT EXISTS game_reactions (game_key TEXT, seat INTEGER, reaction TEXT, count INTEGER);
CREATE TABLE IF NOT EXISTS game_reaction_causes (game_key TEXT, seat INTEGER, reaction TEXT, caller_type TEXT,
    skill_type TEXT, count INTEGER);
CREATE TABLE IF NOT EXISTS game_characters (game_key TEXT, seat INTEGER, slot INTEGER, character_id INTEGER);
CREATE TABLE IF NOT EXISTS game_cards (game_key TEXT, seat INTEGER, card_id INTEGER, copies INTEGER);
CREATE TABLE IF NOT EXISTS game_equipment (game_key TEXT, seat INTEGER, character_slot INTEGER, definition_id INTEGER);
CREATE TABLE IF NOT EXISTS moves (
    game_key TEXT, move_number INTEGER, seat INTEGER, round INTEGER, decision_kind TEXT, action_kind TEXT,
    definition_id INTEGER, is_fast INTEGER, dice_paid INTEGER, chosen_probability REAL, entropy REAL, value REAL,
    win_probability REAL, option_count INTEGER
);
CREATE TABLE IF NOT EXISTS definitions (id INTEGER, kind TEXT, name TEXT, PRIMARY KEY (id, kind));
CREATE TABLE IF NOT EXISTS exported_files (file TEXT PRIMARY KEY);
"""
INDEXES = """
CREATE INDEX IF NOT EXISTS moves_by_game ON moves (game_key);
CREATE INDEX IF NOT EXISTS characters_by_id ON game_characters (character_id);
CREATE INDEX IF NOT EXISTS cards_by_id ON game_cards (card_id);
CREATE INDEX IF NOT EXISTS games_by_update ON games (update_number);
CREATE INDEX IF NOT EXISTS reactions_by_game ON game_reactions (game_key);
"""
MOVE_COLUMNS = ["seat", "round", "decision_kind", "action_kind", "definition_id", "is_fast", "dice_paid",
                "chosen_probability", "entropy", "value", "win_probability", "option_count"]
# alpha 1 records had no move_columns: 6 base columns, then 4 network columns for network seats.
LEGACY_MOVE_COLUMNS = ["seat", "round", "decision_kind", "action_kind", "definition_id", "is_fast",
                       "chosen_probability", "entropy", "value", "option_count"]


def open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def load_definition_names(connection: sqlite3.Connection, rows: list[tuple[int, str, str]]) -> None:
    connection.executemany("INSERT OR REPLACE INTO definitions VALUES (?, ?, ?)", rows)


def export_games_file(connection: sqlite3.Connection, path: Path) -> int:
    """Loads one games/update_NNNNN.jsonl.gz file unless already loaded; returns the number of games added."""
    if connection.execute("SELECT 1 FROM exported_files WHERE file = ?", (path.name,)).fetchone():
        return 0
    update_number = int(path.name.split("_")[1].split(".")[0])
    with gzip.open(path, "rt", encoding="utf-8") as source:
        records = [json.loads(line) for line in source]
    for line_number, record in enumerate(records):
        _insert_game(connection, f"{update_number}:{line_number}", update_number, record)
    connection.execute("INSERT INTO exported_files VALUES (?)", (path.name,))
    return len(records)


def create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(INDEXES)


def _insert_game(connection: sqlite3.Connection, game_key: str, update_number: int, record: dict) -> None:
    seats, knows, defeated = record["seats"], record["opponent_deck_known"], record["defeated"]
    origins = record.get("deck_origins", [None, None])
    damage = record.get("damage_dealt", [None, None])
    connection.execute("INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        game_key, update_number, record["policy_version"], record["source"], record["opponent"], seats[0], seats[1],
        int(knows[0]), int(knows[1]), record["seed"], record["winner"], record["rounds"], defeated[0], defeated[1],
        len(record["moves"]), origins[0], origins[1], record.get("snapshot_version"), record.get("actor_index"),
        damage[0], damage[1]))
    connection.executemany("INSERT INTO game_reactions VALUES (?, ?, ?, ?)", [
        (game_key, seat, reaction, count) for seat, reactions in enumerate(record.get("reactions", []))
        for reaction, count in reactions.items()])
    for seat, deck in enumerate(record["decks"]):
        connection.executemany("INSERT INTO game_characters VALUES (?, ?, ?, ?)",
                               [(game_key, seat, slot, character) for slot, character in enumerate(deck["characters"])])
        connection.executemany("INSERT INTO game_cards VALUES (?, ?, ?, ?)",
                               [(game_key, seat, card, copies) for card, copies in sorted(Counter(deck["cards"]).items())])
    connection.executemany("INSERT INTO game_equipment VALUES (?, ?, ?, ?)", [
        (game_key, item["who"], item["character_index"], item["definition_id"]) for item in record["equipment"]])
    connection.executemany("INSERT INTO game_reaction_causes VALUES (?, ?, ?, ?, ?, ?)", [
        (game_key, seat, *cause.split("|"), count) for seat, causes in enumerate(record.get("reaction_causes", []))
        for cause, count in causes.items()])
    columns = record.get("move_columns", LEGACY_MOVE_COLUMNS)
    connection.executemany("INSERT INTO moves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (game_key, move_number, *(dict(zip(columns, move)).get(column) for column in MOVE_COLUMNS))
        for move_number, move in enumerate(record["moves"])])
