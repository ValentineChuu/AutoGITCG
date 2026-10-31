"""Cards played and tuned per game in a run's training games, per block of updates (human-deck games, learner sides),
with the fair benchmark vs the reference when a block ends on an evaluation update.
    python experiments/card_play_diagnostic/card_play_trend.py --run runs/delta_04 [--block 20] [--first 1] [--last 150]
"""

import argparse
import gzip
import json
from pathlib import Path

HUMAN_DECK_SOURCES = ("tournament", "chudledee")


def block_counts(run: Path, first: int, last: int) -> tuple[int, float, float] | None:
    sides = played = tuned = 0
    for update in range(first, last + 1):
        path = run / "games" / f"update_{update:05d}.jsonl.gz"
        if not path.exists():
            return None
        for line in gzip.open(path, "rt"):
            game = json.loads(line)
            if game["source"] not in HUMAN_DECK_SOURCES:
                continue
            for seat, controller in enumerate(game["seats"]):
                if controller != "learner":
                    continue
                sides += 1
                for move in game["moves"]:
                    played += move[0] == seat and move[3] == "playCard"
                    tuned += move[0] == seat and move[3] == "elementalTuning"
    return sides, played / max(sides, 1), tuned / max(sides, 1)


def reference_scores(run: Path) -> dict[int, float]:
    scores = {}
    for line in (run / "metrics.jsonl").open() if (run / "metrics.jsonl").exists() else []:
        row = json.loads(line)
        if isinstance(row.get("evaluation_reference"), dict):
            scores[row["update"]] = row["evaluation_reference"]["win_rate"]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=10_000)
    arguments = parser.parse_args()
    scores = reference_scores(arguments.run)
    for start in range(arguments.first, arguments.last + 1, arguments.block):
        counts = block_counts(arguments.run, start, start + arguments.block - 1)
        if counts is None:
            break
        sides, played, tuned = counts
        end = start + arguments.block - 1
        score = f" | vs reference {scores[end]:.3f}" if end in scores else ""
        print(f"updates {start}-{end}: {sides} sides, played {played:.2f}, tuned {tuned:.2f} per game{score}")


if __name__ == "__main__":
    main()
