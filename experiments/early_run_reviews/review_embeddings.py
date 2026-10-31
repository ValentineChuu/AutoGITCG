"""Measure identity embedding learning for alpha 2 and compare it with alpha 1."""

import argparse
from pathlib import Path

from experiments.early_run_reviews.embedding_learning import write_embedding_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-1", type=Path, default=Path("runs/alpha_01"))
    parser.add_argument("--alpha-2", type=Path, default=Path("runs/alpha_02"))
    parser.add_argument("--figures", type=Path, default=Path("experiments/early_run_reviews/reports/figures/alpha_2"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/alpha_2_embedding_review.json"))
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    write_embedding_review(arguments.alpha_1, arguments.alpha_2, arguments.alpha_2 / "games.sqlite",
                           arguments.figures, arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
