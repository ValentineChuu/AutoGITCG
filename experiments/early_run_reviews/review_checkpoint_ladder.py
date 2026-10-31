"""Run the alpha 1 comparison, final head-to-head, and alpha 2 checkpoint ladder."""

import argparse
from pathlib import Path

from experiments.early_run_reviews.checkpoint_ladder import run_evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-1", type=Path, default=Path("runs/alpha_01"))
    parser.add_argument("--alpha-2", type=Path, default=Path("runs/alpha_02"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--ladder-matchups", type=int, default=12)
    parser.add_argument("--cache", type=Path, default=Path("data/derived/alpha_2_ladder_games.json"))
    parser.add_argument("--figures", type=Path, default=Path("experiments/early_run_reviews/reports/figures/alpha_2"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/alpha_2_checkpoint_review.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    run_evaluations(arguments.alpha_1, arguments.alpha_2, arguments.workers, arguments.ladder_matchups,
                    arguments.cache, arguments.figures, arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
