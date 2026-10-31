"""Per-seat deck facts: lineup, card list, simple coherence features, and which seat holds a PvE stage deck.

The game logs don't say which seat of a `stage` game holds the stage deck, so it is recovered by matching
each deck exactly against the stage catalog's enemy decks (a random collection deck practically never
matches one exactly).
"""

import sqlite3
from collections import Counter

import numpy as np

from autogitcg.analysis.definition_kinds import card_facts, character_element
from experiments.early_run_reviews.alpha_1_statistics.game_rows import SOURCES
from autogitcg.progression.stage_catalog import build_stage_catalog

MAX_CHARACTERS = 4
# Stage groups. Many stages share one enemy deck across kinds (e.g. a weekly guest reuses an invitation deck), so a
# deck is labelled by precedence: pre-equipped talent > tavern > ascension > invitation > weekly guest > open world.
STAGE_GROUPS = ("serious_showdown", "tavern", "ascension", "invitation", "weekly_guest", "open_world")
_GROUP_OF_KIND = {"tavern_friendly": "tavern", "tavern_serious": "tavern", "ascension": "ascension",
                  "invitation_friendly": "invitation", "invitation_serious": "invitation",
                  "weekly_guest": "weekly_guest", "open_world": "open_world"}
FETCH_SIZE = 200_000


def load_deck_rows(connection: sqlite3.Connection, game_index: dict[str, int], row_count: int,
                   source: np.ndarray) -> dict[str, np.ndarray]:
    characters = _lineups(connection, game_index, row_count)
    card_row, card_id, card_copies = _cards(connection, game_index)
    columns = {"characters": characters, "card_row": card_row, "card_id": card_id, "card_copies": card_copies}
    columns.update(_card_features(row_count, characters, card_row, card_id, card_copies))
    columns["pre_equipped"] = _pre_equipped(connection, game_index, row_count)
    columns.update(_stage_side(row_count, source, characters, card_row, card_id, card_copies, columns["pre_equipped"]))
    return columns


def _lineups(connection: sqlite3.Connection, game_index: dict[str, int], row_count: int) -> np.ndarray:
    characters = np.zeros((row_count, MAX_CHARACTERS), dtype=np.int32)
    for game_key, seat, slot, character_id in connection.execute(
            "SELECT game_key, seat, slot, character_id FROM game_characters ORDER BY rowid"):
        if slot < MAX_CHARACTERS:
            characters[2 * game_index[game_key] + seat, slot] = character_id
    return characters


def _cards(connection: sqlite3.Connection, game_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, ids, copies = [], [], []
    cursor = connection.execute("SELECT game_key, seat, card_id, copies FROM game_cards ORDER BY rowid")
    while chunk := cursor.fetchmany(FETCH_SIZE):
        for game_key, seat, card, count in chunk:
            rows.append(2 * game_index[game_key] + seat)
            ids.append(card)
            copies.append(count)
    return np.array(rows, dtype=np.int32), np.array(ids, dtype=np.int32), np.array(copies, dtype=np.int8)


def _card_features(row_count: int, characters: np.ndarray, card_row: np.ndarray, card_id: np.ndarray,
                   card_copies: np.ndarray) -> dict[str, np.ndarray]:
    facts = card_facts()
    category = np.array([facts.get(card, ("event", 0, ()))[0] for card in card_id])
    cost = np.array([facts.get(card, ("event", 0, ()))[1] for card in card_id], dtype=np.float64)
    resonance = np.array(["resonance" in facts.get(card, ("", 0, ()))[2] for card in card_id])
    copies = card_copies.astype(np.float64)

    def per_row(weights: np.ndarray) -> np.ndarray:
        return np.bincount(card_row, weights=weights, minlength=row_count)

    action_cards = per_row(copies)
    features = {
        "action_cards": action_cards.astype(np.int16),
        "distinct_cards": np.bincount(card_row, minlength=row_count).astype(np.int16),
        "deck_talents": per_row(copies * (category == "talent")).astype(np.int16),
        "deck_equipment": per_row(copies * (category == "equipment")).astype(np.int16),
        "deck_supports": per_row(copies * (category == "support")).astype(np.int16),
        "deck_events": per_row(copies * (category == "event")).astype(np.int16),
        "deck_resonance": per_row(copies * resonance).astype(np.int16),
        "deck_mean_cost": np.divide(per_row(copies * cost), action_cards, out=np.full(row_count, np.nan),
                                    where=action_cards > 0).astype(np.float32),
    }
    element_of = character_element()
    elements = [[element_of.get(int(character), "") for character in lineup if character] for lineup in characters]
    features["character_count"] = (characters > 0).sum(axis=1).astype(np.int8)
    features["distinct_elements"] = np.array([len(set(lineup)) for lineup in elements], dtype=np.int8)
    features["largest_element_group"] = np.array(
        [max(Counter(lineup).values()) if lineup else 0 for lineup in elements], dtype=np.int8)
    return features


def _pre_equipped(connection: sqlite3.Connection, game_index: dict[str, int], row_count: int) -> np.ndarray:
    equipped = np.zeros(row_count, dtype=np.int8)
    for game_key, seat in connection.execute("SELECT DISTINCT game_key, seat FROM game_equipment"):
        equipped[2 * game_index[game_key] + seat] = 1
    return equipped


def _stage_side(row_count: int, source: np.ndarray, characters: np.ndarray, card_row: np.ndarray,
                card_id: np.ndarray, card_copies: np.ndarray, pre_equipped: np.ndarray) -> dict[str, np.ndarray]:
    """stage_side = 1 for the seat holding the stage deck; stage_group = index into STAGE_GROUPS (-1 elsewhere)."""
    groups_by_signature: dict[tuple, set[str]] = {}
    stages, _ = build_stage_catalog()
    for stage in stages:
        signature = (tuple(stage.enemy_deck.characters), tuple(sorted(Counter(stage.enemy_deck.cards).items())))
        groups_by_signature.setdefault(signature, set()).add(_GROUP_OF_KIND[stage.kind])
    stage_rows = np.flatnonzero(source == SOURCES.index("stage"))
    cards_by_row: dict[int, list[tuple[int, int]]] = {}
    in_stage = np.isin(card_row, stage_rows)
    for row, card, copies in zip(card_row[in_stage], card_id[in_stage], card_copies[in_stage]):
        cards_by_row.setdefault(int(row), []).append((int(card), int(copies)))
    stage_side = np.zeros(row_count, dtype=np.int8)
    stage_group = np.full(row_count, -1, dtype=np.int8)
    for row in stage_rows:
        signature = (tuple(int(character) for character in characters[row] if character),
                     tuple(sorted(cards_by_row.get(int(row), []))))
        groups = groups_by_signature.get(signature)
        if groups:
            stage_side[row] = 1
            groups = groups | {"serious_showdown"} if pre_equipped[row] else groups
            stage_group[row] = min(STAGE_GROUPS.index(group) for group in groups)
    return {"stage_side": stage_side, "stage_group": stage_group}
