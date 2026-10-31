"""A deck-only win model: L2-regularized logistic regression on (own deck minus opponent deck) + seat.

Fitted on self-play games, where both seats run the same network, so whatever it explains is the deck (and seat),
not who played better. Features: one per character (+1 own lineup, -1 opponent lineup) and one per action card
(+copies own deck, -copies opponent deck); a constant for "I am seat 0". Characters and cards seen fewer than
MINIMUM_APPEARANCES times are dropped. Uses torch only as a sparse L-BFGS solver.
"""

from dataclasses import dataclass

import numpy as np
import torch

from experiments.early_run_reviews.alpha_1_statistics.game_rows import OPPONENTS

MINIMUM_APPEARANCES = 200
L2_STRENGTH = 1e-3
SOLVER_THREADS = 4


@dataclass
class DeckStrengthModel:
    feature_ids: list[tuple[str, int]]  # ("seat", 0), ("character", id) or ("card", id)
    weights: np.ndarray
    appearances: np.ndarray

    def logits(self, table: dict, rows: np.ndarray) -> np.ndarray:
        """Predicted log-odds that the seat in each row beats the other seat, from the decks and seat only."""
        matrix = deck_difference_matrix(table, rows, {feature: index for index, feature in enumerate(self.feature_ids)})
        return matrix @ self.weights


def deck_difference_matrix(table: dict, rows: np.ndarray, feature_index: dict) -> "_SparseMatrix":
    """Sparse (rows x features) design matrix: own deck minus the other seat's deck, plus the seat sign."""
    row_positions, columns, values = _entries(table, rows, feature_index)
    return _SparseMatrix(row_positions, columns, values, len(rows), len(feature_index))


def fit_deck_strength(table: dict, rows: np.ndarray) -> DeckStrengthModel:
    """rows: the seat-0 rows of decided self-play games; target = seat 0 won."""
    counts: dict[tuple[str, int], int] = {}
    for character in table["characters"][rows].ravel():
        if character:
            counts[("character", int(character))] = counts.get(("character", int(character)), 0) + 1
    in_rows = np.isin(table["card_row"], rows)
    for card in table["card_id"][in_rows]:
        counts[("card", int(card))] = counts.get(("card", int(card)), 0) + 1
    feature_ids = [("seat", 0)] + sorted(feature for feature, count in counts.items() if count >= MINIMUM_APPEARANCES)
    feature_index = {feature: index for index, feature in enumerate(feature_ids)}
    matrix = deck_difference_matrix(table, rows, feature_index)
    targets = torch.tensor(table["won"][rows], dtype=torch.float64)
    weights = _solve(matrix.as_torch(), targets)
    appearances = np.array([counts.get(feature, len(rows)) for feature in feature_ids])
    return DeckStrengthModel(feature_ids, weights, appearances)


def _solve(matrix: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    torch.set_num_threads(SOLVER_THREADS)
    weights = torch.zeros(matrix.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def loss_function():
        optimizer.zero_grad()
        logits = torch.sparse.mm(matrix, weights.unsqueeze(1)).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) + L2_STRENGTH * (weights[1:] ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(loss_function)
    return weights.detach().numpy()


def _entries(table: dict, rows: np.ndarray, feature_index: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position_of_row = {int(row): position for position, row in enumerate(rows)}
    row_positions, columns, values = [], [], []
    seat_column = feature_index[("seat", 0)]
    for position, row in enumerate(rows):
        row_positions.append(position)
        columns.append(seat_column)
        values.append(1.0 if table["seat"][row] == 0 else -1.0)
        for sign, deck_row in ((1.0, row), (-1.0, row ^ 1)):
            for character in table["characters"][deck_row]:
                column = feature_index.get(("character", int(character)))
                if character and column is not None:
                    row_positions.append(position)
                    columns.append(column)
                    values.append(sign)
    for sign, deck_rows in ((1.0, rows), (-1.0, rows ^ 1)):
        positions_by_deck_row = {int(deck_row): position_of_row[int(row)] for row, deck_row in zip(rows, deck_rows)}
        selected = np.isin(table["card_row"], deck_rows)
        for deck_row, card, copies in zip(table["card_row"][selected], table["card_id"][selected], table["card_copies"][selected]):
            column = feature_index.get(("card", int(card)))
            if column is not None:
                row_positions.append(positions_by_deck_row[int(deck_row)])
                columns.append(column)
                values.append(sign * float(copies))
    return np.array(row_positions, dtype=np.int64), np.array(columns, dtype=np.int64), np.array(values, dtype=np.float64)


class _SparseMatrix:
    """Minimal sparse matrix: `matrix @ vector` in numpy, `as_torch()` for the solver."""

    def __init__(self, row_positions: np.ndarray, columns: np.ndarray, values: np.ndarray, row_count: int, column_count: int):
        self.row_positions, self.columns, self.values = row_positions, columns, values
        self.shape = (row_count, column_count)

    def __matmul__(self, vector: np.ndarray) -> np.ndarray:
        return np.bincount(self.row_positions, weights=self.values * vector[self.columns], minlength=self.shape[0])

    def as_torch(self) -> torch.Tensor:
        indices = torch.tensor(np.stack([self.row_positions, self.columns]), dtype=torch.int64)
        return torch.sparse_coo_tensor(indices, torch.tensor(self.values, dtype=torch.float64), self.shape,
                                       check_invariants=True).coalesce()


SELF_PLAY_FROM_UPDATE = 300
_fitted: dict[int, tuple[DeckStrengthModel, np.ndarray, np.ndarray]] = {}


def self_play_deck_model(table: dict) -> tuple[DeckStrengthModel, np.ndarray, np.ndarray]:
    """(model, training rows, held-out rows): seat-0 rows of decided self-play games from update 300 on, split by
    game index parity (even = training). Fitted once per table."""
    if id(table) not in _fitted:
        selected = ((table["opponent"] == OPPONENTS.index("self")) & (table["seat"] == 0) & (table["round_limit"] == 0)
                    & (table["update"] >= SELF_PLAY_FROM_UPDATE))
        rows = np.flatnonzero(selected)
        training_rows, held_out_rows = rows[table["game"][rows] % 2 == 0], rows[table["game"][rows] % 2 == 1]
        _fitted[id(table)] = (fit_deck_strength(table, training_rows), training_rows, held_out_rows)
    return _fitted[id(table)]
