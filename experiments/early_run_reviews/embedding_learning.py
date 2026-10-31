"""Identity-embedding drift and card-behaviour neighbourhoods for saved runs."""

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from autogitcg.analysis.definition_kinds import card_facts
from autogitcg.observation.observation_encoder import ObservationEncoder


def load_weights(run: Path, update: int) -> dict[str, torch.Tensor]:
    return torch.load(run / "weights" / f"update_{update:05d}.pt", map_location="cpu", weights_only=True)


def embedding_summary(run: Path, updates: list[int]) -> list[dict]:
    initial = load_weights(run, 0)["definition_embedding.weight"].float().numpy()
    rows = []
    for update in updates:
        current = load_weights(run, update)["definition_embedding.weight"].float().numpy()
        row_norm = np.linalg.norm(current[1:], axis=1)
        drift = np.linalg.norm(current[1:] - initial[1:], axis=1)
        rows.append({
            "update": update,
            "mean_embedding_norm": float(row_norm.mean()),
            "mean_drift_norm": float(drift.mean()),
            "median_drift_norm": float(np.median(drift)),
            "drift_to_initial_norm": float(drift.mean() / np.linalg.norm(initial[1:], axis=1).mean()),
        })
    return rows


def card_play_rates(database: Path, minimum_events: int = 100) -> dict[int, tuple[float, int]]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    rows = connection.execute("""
        SELECT m.definition_id,
               SUM(m.action_kind = 'playCard'), SUM(m.action_kind = 'elementalTuning')
        FROM moves m JOIN games g ON g.game_key = m.game_key
        WHERE g.update_number >= 1500
          AND (CASE WHEN m.seat = 0 THEN g.seat0_controller ELSE g.seat1_controller END) = 'learner'
          AND m.action_kind IN ('playCard', 'elementalTuning')
        GROUP BY m.definition_id
    """).fetchall()
    connection.close()
    return {card: (played / (played + tuned), played + tuned)
            for card, played, tuned in rows if played + tuned >= minimum_events}


def behaviour_neighbour_test(run: Path, update: int, rates: dict[int, tuple[float, int]], neighbours: int = 5) -> dict:
    encoder = ObservationEncoder()
    facts = card_facts()
    usable = [(card, encoder.vocabulary.index("entity", card), rate, count, facts.get(card, ("unknown",))[0])
              for card, (rate, count) in rates.items()
              if encoder.vocabulary.index("entity", card) > 1]
    weights = load_weights(run, update)["definition_embedding.weight"].float().numpy()
    vectors = np.array([weights[index] for _, index, _, _, _ in usable])
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    distances = 1 - vectors @ vectors.T
    np.fill_diagonal(distances, np.inf)
    play_rates = np.array([row[2] for row in usable])
    nearest = np.argpartition(distances, neighbours, axis=1)[:, :neighbours]
    nearest_difference = np.abs(play_rates[:, None] - play_rates[nearest]).mean()
    rng = np.random.default_rng(20261014)
    random_difference = np.mean([np.abs(play_rates - play_rates[rng.permutation(len(play_rates))]).mean()
                                 for _ in range(1000)])
    categories = np.array([row[4] for row in usable])
    category_matched = []
    for _ in range(1000):
        permuted = np.arange(len(play_rates))
        for category in np.unique(categories):
            indices = np.flatnonzero(categories == category)
            permuted[indices] = rng.permutation(indices)
        category_matched.append(np.abs(play_rates - play_rates[permuted]).mean())
    category_matched_difference = float(np.mean(category_matched))
    return {
        "update": update,
        "cards": len(usable),
        "neighbours": neighbours,
        "nearest_mean_absolute_play_rate_difference": float(nearest_difference),
        "random_pair_mean_absolute_play_rate_difference": float(random_difference),
        "category_matched_random_pair_difference": category_matched_difference,
        "relative_improvement": float(1 - nearest_difference / random_difference),
        "relative_improvement_vs_category_matched": float(1 - nearest_difference / category_matched_difference),
    }


def write_embedding_review(alpha_1: Path, alpha_2: Path, database: Path, folder: Path, output: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    updates_01 = [0, 100, 200, 300, 400, 500, 600]
    updates_02 = list(range(0, 2001, 200))
    rates = card_play_rates(database)
    result = {
        "alpha 1": embedding_summary(alpha_1, updates_01),
        "alpha 2": embedding_summary(alpha_2, updates_02),
        "card_behaviour_neighbours": [behaviour_neighbour_test(alpha_2, update, rates)
                                        for update in (0, 200, 600, 1000, 1400, 1800, 2000)],
    }
    figure, (norm_axis, behaviour_axis) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, rows in (("alpha 1", result["alpha 1"]), ("alpha 2", result["alpha 2"])):
        norm_axis.plot([row["update"] for row in rows], [row["mean_drift_norm"] for row in rows], marker="o",
                       label=f"{name}: learned drift")
        norm_axis.axhline(rows[0]["mean_embedding_norm"], linestyle="--", alpha=0.6,
                          label=f"{name}: initial norm")
    norm_axis.set(title="Identity embedding: learned movement vs start", xlabel="update", ylabel="mean row norm")
    norm_axis.set_yscale("symlog", linthresh=0.1)
    norm_axis.legend(fontsize=8)
    norm_axis.grid(alpha=0.25)
    rows = result["card_behaviour_neighbours"]
    behaviour_axis.plot([row["update"] for row in rows],
                        [row["nearest_mean_absolute_play_rate_difference"] for row in rows], marker="o",
                        label="5 nearest identity vectors")
    behaviour_axis.axhline(rows[0]["random_pair_mean_absolute_play_rate_difference"], linestyle="--", color="gray",
                           label="random card pairs")
    behaviour_axis.axhline(rows[0]["category_matched_random_pair_difference"], linestyle=":", color="gray",
                           label="random pairs, same card type")
    behaviour_axis.set(title="Do nearby cards get played alike?", xlabel="update",
                       ylabel="mean absolute play-share difference")
    behaviour_axis.legend(fontsize=8)
    behaviour_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(folder / "embedding_learning.png", dpi=160)
    plt.close(figure)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
