"""Scores the network's own choices on the ground-truth set (experiments/ground_truth/build_ground_truth.py): how often its most likely
move is the true best, how much win chance its picks lose against the best (regret), compared with a random compared
move, and where the losses are (by the kind of the best move vs the picked one, by round).

Win chance is the -1..+1 result mean / 2 + 0.5, so a regret of 0.10 on the result scale is 5 points of win chance.

The best move by measured mean is partly the luckiest one, so regret against it is too high (winner's curse). The
held-out regret avoids that: the best move is chosen on the even deals and the regret measured on the odd deals.

Usage:
    python experiments/ground_truth/analyze_ground_truth.py runs/ground_truth/*.jsonl [--output experiments/ground_truth/results/ground_truth_v2.json]
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from experiments.ground_truth.held_out_regret import held_out_regret


def move_kind(move: dict) -> str:
    return json.loads(move["key"])["kind"]


def regret_points(position: dict, move: int) -> float:
    """Win-chance points the move loses against the position's best move (paired difference)."""
    return -position["summary"][move]["difference_to_best"] / 2 * 100


def clearly_worse(position: dict, move: int, errors: float = 2.0) -> bool:
    """The move is worse than the best by more than `errors` paired standard errors."""
    row = position["summary"][move]
    return move != position["best"] and row["difference_error"] is not None and \
        -row["difference_to_best"] > errors * row["difference_error"]


def load(paths: list[Path]) -> list[dict]:
    positions = []
    for path in paths:
        for line in path.read_text().splitlines():
            position = json.loads(line)
            if all(row["mean"] is not None for row in position["summary"]):
                positions.append(position)
    return positions


def describe(positions: list[dict]) -> dict:
    identical = [position for position in positions if len({row["mean"] for row in position["summary"]}) == 1]
    decided = [position for position in positions if position not in identical]
    top_regret = [regret_points(position, 0) for position in decided]
    sampled = [position for position in decided if position["network_choice"] is not None]
    sampled_regret = [regret_points(position, position["network_choice"]) for position in sampled]
    held_out_top = [value for position in decided if (value := held_out_regret(position, 0)) is not None]
    held_out_sampled = [value for position in sampled if (value := held_out_regret(position, position["network_choice"])) is not None]
    held_out_random = [statistics.mean(values) for position in decided
                       if None not in (values := [held_out_regret(position, move) for move in range(len(position["moves"]))])]
    random_regret = [statistics.mean(regret_points(position, move) for move in range(len(position["moves"]))) for position in decided]
    best_move_regret = [position for position in decided if position["best"] != 0]
    mistakes = Counter()  # (beta 10.8's top kind -> best kind) where beta 10.8's top move is clearly worse
    mistake_points = defaultdict(list)
    for position in decided:
        if clearly_worse(position, 0):
            pair = f"{move_kind(position['moves'][0])} -> {move_kind(position['moves'][position['best']])}"
            mistakes[pair] += 1
            mistake_points[pair].append(regret_points(position, 0))
    by_round = defaultdict(list)
    for position in decided:
        by_round[position["round"]].append(regret_points(position, 0))
    best_rank = Counter(position["best"] for position in decided)
    return {
        "positions": len(positions),
        "every move had identical results (nothing to choose)": len(identical),
        "positions with a choice": len(decided),
        "separated (best clearly ahead of every other move)": sum(position["separated"] for position in decided),
        "deals: median / mean": (statistics.median(position["deals"] for position in positions), round(statistics.mean(position["deals"] for position in positions))),
        "beta 10.8's top move is the best": f"{100 * sum(position['best'] == 0 for position in decided) / len(decided):.1f}%",
        "beta 10.8's top move clearly worse than the best (2 standard errors)": f"{100 * sum(clearly_worse(position, 0) for position in decided) / len(decided):.1f}%",
        "HELD-OUT regret, win-chance points: beta 10.8's top move / beta 10.8 as it plays (sampled) / a random compared move": (
            round(statistics.mean(held_out_top), 2), round(statistics.mean(held_out_sampled), 2), round(statistics.mean(held_out_random), 2)),
        "regret of beta 10.8's top move vs the measured best (inflated by luck): mean / median": (round(statistics.mean(top_regret), 2), round(statistics.median(top_regret), 2)),
        "regret of beta 10.8's sampled move (as it plays)": round(statistics.mean(sampled_regret), 2),
        "regret of a random compared move": round(statistics.mean(random_regret), 2),
        "positions where beta 10.8's top move isn't the measured best": len(best_move_regret),
        "rank (by beta 10.8's probability) of the true best move": dict(sorted(best_rank.items())),
        "beta 10.8's clear mistakes, its kind -> best kind (count, mean points lost)": {
            pair: (count, round(statistics.mean(mistake_points[pair]), 1)) for pair, count in mistakes.most_common()},
        "mean regret of beta 10.8's top move by round": {round_number: (len(values), round(statistics.mean(values), 2))
                                                  for round_number, values in sorted(by_round.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = describe(load(arguments.files))
    text = json.dumps(report, indent=1)
    print(text)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)


if __name__ == "__main__":
    main()
