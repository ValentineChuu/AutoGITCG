"""Compares methods scored on the ground-truth set (experiments/ground_truth/score_methods_on_ground_truth.py) with beta 10.8's own top move
and a random compared move: how often each picks the true best move, its held-out regret (win-chance points lost
against the best, ground_truth/held_out_regret.py), and for methods that score every move, how often its scores order
two moves the right way when the ground truth clearly separates them (pairwise accuracy; 50% = guessing).

Only positions with a choice count (not every move with identical results).

Usage:
    python experiments/ground_truth/compare_methods_on_ground_truth.py --ground-truth runs/ground_truth/shard*.jsonl ... \
        --method "one-move judge=runs/ground_truth_scores/judge_shard*.jsonl" --method ... [--output runs/ground_truth_comparison.json]
"""

import argparse
import glob
import json
import statistics
from pathlib import Path

from experiments.ground_truth.held_out_regret import held_out_regret
from experiments.ground_truth.stored_positions import load_positions


def clear_pairs(line: dict, errors: float = 2.0) -> list[tuple[int, int]]:
    """Move pairs (better, worse) whose paired difference is beyond `errors` standard errors."""
    results = line["results"]
    pairs = []
    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            differences = [a - b for a, b in zip(results[first], results[second]) if a != 0 and b != 0]
            if len(differences) < 16:
                continue
            mean, error = statistics.mean(differences), statistics.stdev(differences) / len(differences) ** 0.5
            if error > 0 and abs(mean) > errors * error:
                pairs.append((first, second) if mean > 0 else (second, first))
    return pairs


def describe(name: str, lines: dict[str, dict], picks: dict[str, int | None], scores: dict[str, list] | None) -> dict:
    """picks: position name -> chosen compared move (None: a move that wasn't compared)."""
    scored = [position for position in lines if position in picks]
    compared = [position for position in scored if picks[position] is not None]
    regrets = [value for position in compared if (value := held_out_regret(lines[position], picks[position])) is not None]
    separated = [position for position in compared if lines[position]["separated"]]
    separated_regrets = [value for position in separated if (value := held_out_regret(lines[position], picks[position])) is not None]
    row = {
        "positions": len(scored),
        "picked a move outside the compared ones": len(scored) - len(compared),
        "picks the best": f"{100 * sum(picks[position] == lines[position]['best'] for position in compared) / max(len(compared), 1):.1f}%",
        "differs from beta 10.8's top move": f"{100 * sum(picks[position] != 0 for position in compared) / max(len(compared), 1):.1f}%",
        "held-out regret": round(statistics.mean(regrets), 2) if regrets else None,
        "held-out regret, separated positions": round(statistics.mean(separated_regrets), 2) if separated_regrets else None,
    }
    if scores is not None:
        right = total = 0
        for position in scored:
            for better, worse in clear_pairs(lines[position]):
                values = scores[position]
                if values[better] is not None and values[worse] is not None:
                    total += 1
                    right += values[better] > values[worse]
        row["pairwise accuracy on clearly different moves"] = f"{100 * right / max(total, 1):.1f}% of {total}"
    return {name: row}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, nargs="+", required=True)
    parser.add_argument("--method", action="append", default=[], help='"name=glob of score files"')
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    lines = {position.name: position.line for position in load_positions(arguments.ground_truth)
             if len({row["mean"] for row in position.line["summary"]}) > 1}
    report = {}
    report |= describe("beta 10.8's top move", lines, {position: 0 for position in lines}, None)
    network_scores = {position: [move["probability"] for move in line["moves"]] for position, line in lines.items()}
    report |= describe("beta 10.8's probabilities as scores", lines, {position: 0 for position in lines}, network_scores)
    random_regret = [statistics.mean(values) for line in lines.values()
                     if None not in (values := [held_out_regret(line, move) for move in range(len(line["moves"]))])]
    report["a random compared move"] = {"held-out regret": round(statistics.mean(random_regret), 2)}
    for entry in arguments.method:
        name, pattern = entry.split("=", 1)
        picks, scores = {}, {}
        for path in sorted(glob.glob(pattern)):
            for text in Path(path).read_text().splitlines():
                row = json.loads(text)
                if row["position"] in lines and "lost" not in row:
                    picks[row["position"]] = row["chosen_move"]
                    scores[row["position"]] = row["scores"]
        report |= describe(name, lines, picks, scores)
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)


if __name__ == "__main__":
    main()
