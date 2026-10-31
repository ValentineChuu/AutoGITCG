"""Measure conservative evidence of deliberate reaction play in benchmark replays."""

import argparse
from pathlib import Path

from experiments.early_run_reviews.replay_intent import write_replay_intent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/alpha_02"))
    parser.add_argument("--figures", type=Path, default=Path("experiments/early_run_reviews/reports/figures/alpha_2"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/main_02_reaction_intent.json"))
    arguments = parser.parse_args()
    write_replay_intent(arguments.run, arguments.figures, arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
