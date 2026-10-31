"""Does the move-value output rank moves like the look-ahead does? An offline check on recorded look-ahead labels.

For each searched decision whose candidates the look-ahead clearly told apart (best minus worst mean score at least
--spread), compares the look-ahead's ranking of its candidates with two rankings from each network: its policy
(action scores) and its move-value output. Reports how often each picks the look-ahead's best candidate (a random
pick would match 1 / candidates) and the mean rank correlation within decisions.

Usage:
    python experiments/lookahead_students/compare_move_values.py --labels runs/teacher_02 --networks models/beta-10.8.pt runs/epsilon_12/check_update20.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.lookahead_students.label_shards import LABEL_FOLDER, read_shard
from autogitcg.model.observation_batch import network_dtypes, stack, upload
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.model.combat_network_file import load_combat_network

BATCH = 512


def separated_labels(run: Path, spread: float, limit: int) -> list:
    chosen = []
    for shard in sorted((run / LABEL_FOLDER).glob("*.pkl.gz")):
        for label in read_shard(shard):
            scores = label.candidate_scores
            if (label.searched and len(scores) >= 2 and None not in scores and max(scores) - min(scores) >= spread):
                chosen.append(label)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


@torch.no_grad()
def network_scores(network, labels: list, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(action scores, move values) per label, each (labels, MAX_ACTIONS)."""
    network.eval()
    policy_parts, value_parts = [], []
    for start in range(0, len(labels), BATCH):
        batch = network_dtypes(upload(stack([label.observation for label in labels[start:start + BATCH]]), device))
        output = network(batch)
        policy_parts.append(output.action_logits.float().cpu().numpy())
        value_parts.append(output.move_values.float().cpu().numpy())
    return np.concatenate(policy_parts), np.concatenate(value_parts)


def agreement(labels: list, scores: np.ndarray) -> dict[str, float]:
    best_matches, correlations = [], []
    for row, label in enumerate(labels):
        candidate_scores = scores[row, label.candidate_indices]
        best_matches.append(int(np.argmax(candidate_scores)) == int(np.argmax(label.candidate_scores)))
        if len(label.candidate_indices) >= 3:
            correlation = np.corrcoef(_ranks(candidate_scores), _ranks(np.array(label.candidate_scores)))[0, 1]
            if np.isfinite(correlation):
                correlations.append(correlation)
    return {"picks_search_best": float(np.mean(best_matches)), "rank_correlation": float(np.mean(correlations))}


def _ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("runs/teacher_02"))
    parser.add_argument("--networks", type=Path, nargs="+", required=True)
    parser.add_argument("--spread", type=float, default=0.2, help="minimum best-minus-worst mean candidate score")
    parser.add_argument("--limit", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ObservationEncoder()
    labels = separated_labels(arguments.labels, arguments.spread, arguments.limit)
    chance = float(np.mean([1 / len(label.candidate_indices) for label in labels]))
    report = {"decisions": len(labels), "random_picks_search_best": chance, "networks": {}}
    for path in arguments.networks:
        policy, move_values = network_scores(load_combat_network(path, encoder, device), labels, device)
        report["networks"][str(path)] = {"policy": agreement(labels, policy), "move_value": agreement(labels, move_values)}
    print(json.dumps(report, indent=2))
    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
