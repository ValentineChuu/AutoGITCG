"""Generate database-backed statistics and figures for the alpha 2 review."""

import argparse
from pathlib import Path

from experiments.early_run_reviews.run_statistics import write_run_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/alpha_02"))
    parser.add_argument("--decks", type=Path, default=Path("data/decks/synergy_decks.json"))
    parser.add_argument("--figures", type=Path, default=Path("experiments/early_run_reviews/reports/figures/alpha_2"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/alpha_2_review_statistics.json"))
    arguments = parser.parse_args()
    write_run_statistics(arguments.run / "games.sqlite", arguments.run / "benchmarks", arguments.decks,
                         arguments.figures, arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
