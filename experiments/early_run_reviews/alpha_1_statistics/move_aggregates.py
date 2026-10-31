"""One streaming pass over `moves`: per-seat move counts, policy confidence sums and the value trajectory.

Moves are read in rowid order, which is game order with moves in sequence (the export inserts game by game),
so no sort over the 25.7M-row table is needed.
"""

import math
import sqlite3

import numpy as np

from autogitcg.analysis.definition_kinds import card_facts, skill_facts

COUNT_COLUMNS = (
    "skill_normal", "skill_elemental", "skill_burst", "skill_other",
    "card_talent", "card_equipment", "card_support", "card_event",
    "tunings", "switches", "declare_ends", "actions", "fast_actions", "end_first_rounds",
    "choose_active", "reroll_decisions", "dice_rerolled", "mulligan_changed", "later_hand_switches", "select_card",
    "dice_spent",
)
NETWORK_KINDS = ("action", "chooseActive", "rerollDice", "switchHands", "selectCard")
NETWORK_SUMS = ("decisions", "probability", "entropy", "log_options")
MAX_ROUND = 15  # round values are kept for rounds 0 (mulligan) .. MAX_ROUND
FETCH_SIZE = 50_000

_COLUMN = {name: index for index, name in enumerate(COUNT_COLUMNS)}
_KIND = {name: index for index, name in enumerate(NETWORK_KINDS)}


def aggregate_moves(connection: sqlite3.Connection, game_index: dict[str, int], row_count: int,
                    progress_every: int = 2_000_000) -> dict[str, np.ndarray]:
    counts = np.zeros((row_count, len(COUNT_COLUMNS)), dtype=np.int32)
    network = np.zeros((row_count, len(NETWORK_KINDS), len(NETWORK_SUMS)), dtype=np.float64)
    round_values = np.full((row_count, MAX_ROUND + 1), np.nan, dtype=np.float32)
    last_value = np.full(row_count, np.nan, dtype=np.float32)
    game = _GameAccumulator(skill_facts(), card_facts())
    cursor = connection.execute("""
        SELECT game_key, seat, round, decision_kind, action_kind, definition_id, is_fast,
               chosen_probability, entropy, value, option_count
        FROM moves ORDER BY rowid
    """)
    current_key, processed = None, 0
    while rows := cursor.fetchmany(FETCH_SIZE):
        for row in rows:
            if row[0] != current_key:
                if current_key is not None:
                    game.flush(2 * game_index[current_key], counts, network, round_values, last_value)
                current_key = row[0]
                game.reset()
            game.add(row)
        processed += len(rows)
        if progress_every and processed % progress_every < FETCH_SIZE:
            print(f"  moves read: {processed:,}", flush=True)
    if current_key is not None:
        game.flush(2 * game_index[current_key], counts, network, round_values, last_value)
    columns = {name: counts[:, index] for index, name in enumerate(COUNT_COLUMNS)}
    for kind_index, kind in enumerate(NETWORK_KINDS):
        for sum_index, name in enumerate(NETWORK_SUMS):
            columns[f"{kind}_{name}"] = network[:, kind_index, sum_index]
    columns["round_values"] = round_values
    columns["last_value"] = last_value
    return columns


class _GameAccumulator:
    """Counts for the two seats of the game being read."""

    def __init__(self, skills: dict[int, tuple[str, int]], cards: dict[int, tuple[str, int, tuple[str, ...]]]):
        self.skills = skills
        self.cards = cards

    def reset(self) -> None:
        self.counts = [[0] * len(COUNT_COLUMNS) for _ in range(2)]
        self.network = [[[0.0] * len(NETWORK_SUMS) for _ in NETWORK_KINDS] for _ in range(2)]
        self.round_values: list[dict[int, float]] = [{}, {}]
        self.last_value: list[float] = [math.nan, math.nan]
        self.rounds_ended: set[int] = set()

    def add(self, row: tuple) -> None:
        _, seat, round_number, decision_kind, action_kind, definition_id, is_fast, probability, entropy, value, options = row
        counts = self.counts[seat]
        if decision_kind == "action":
            counts[_COLUMN["actions"]] += 1
            counts[_COLUMN["fast_actions"]] += is_fast
            self._add_action(counts, round_number, seat, action_kind, definition_id)
        elif decision_kind == "chooseActive":
            counts[_COLUMN["choose_active"]] += 1
        elif decision_kind == "rerollDice":
            counts[_COLUMN["reroll_decisions"]] += 1
            counts[_COLUMN["dice_rerolled"]] += definition_id
        elif decision_kind == "switchHands":
            counts[_COLUMN["mulligan_changed" if round_number == 0 else "later_hand_switches"]] += definition_id
        elif decision_kind == "selectCard":
            counts[_COLUMN["select_card"]] += 1
        if value is not None:
            sums = self.network[seat][_KIND[decision_kind]]
            sums[0] += 1
            sums[1] += probability
            sums[2] += entropy
            sums[3] += math.log(max(options, 1))
            self.round_values[seat].setdefault(round_number, value)
            self.last_value[seat] = value

    def _add_action(self, counts: list[int], round_number: int, seat: int, action_kind: str, definition_id: int) -> None:
        if action_kind == "useSkill":
            skill_type, cost = self.skills.get(definition_id, ("other", 0))
            counts[_COLUMN.get(f"skill_{skill_type}", _COLUMN["skill_other"])] += 1
            counts[_COLUMN["dice_spent"]] += cost
        elif action_kind == "playCard":
            category, cost, _ = self.cards.get(definition_id, ("event", 0, ()))
            counts[_COLUMN[f"card_{category}"]] += 1
            counts[_COLUMN["dice_spent"]] += cost
        elif action_kind == "elementalTuning":
            counts[_COLUMN["tunings"]] += 1  # converts a die, spends none
        elif action_kind == "switchActive":
            counts[_COLUMN["switches"]] += 1
            counts[_COLUMN["dice_spent"]] += 1  # nominal switch cost
        elif action_kind == "declareEnd":
            counts[_COLUMN["declare_ends"]] += 1
            if round_number not in self.rounds_ended:
                self.rounds_ended.add(round_number)
                counts[_COLUMN["end_first_rounds"]] += 1

    def flush(self, first_row: int, counts: np.ndarray, network: np.ndarray, round_values: np.ndarray,
              last_value: np.ndarray) -> None:
        for seat in range(2):
            row = first_row + seat
            counts[row] = self.counts[seat]
            network[row] = self.network[seat]
            for round_number, value in self.round_values[seat].items():
                if round_number <= MAX_ROUND:
                    round_values[row, round_number] = value
            last_value[row] = self.last_value[seat]
