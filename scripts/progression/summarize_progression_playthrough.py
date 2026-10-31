"""Compare progression playthroughs side by side, as markdown tables printed to the terminal.

Each argument is a folder written by scripts/progression/run_progression_playthrough.py, where simulated accounts
played the whole progression from the starter deck with one way of choosing decks (an "arm": Chudledee, random decks,
...). One column per arm:

- progress: accounts, matches, win rate, how many accounts collected every character, the median (and 90th
  percentile) number of matches and weeks to half and to all of the characters, and the median character count and
  player level at fixed match numbers;
- per stage kind (friendly and serious invitations, friendly and serious tavern duels, weekly guests, ascension
  challenges): win rate and first-attempt win rate;
- per arm, the hardest stages: losses before the first win, and how many accounts never beat them.

Usage:
    python scripts/progression/summarize_progression_playthrough.py runs/playthrough_chudledee runs/playthrough_random_decks runs/playthrough_estimated
"""

import argparse
from pathlib import Path

from autogitcg.progression_playthrough.playthrough_summary import hardest_stages, load_matches, progress_measures, stage_kind_measures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    arguments = parser.parse_args()
    arms = {run.name.removeprefix("playthrough_"): load_matches(run) for run in arguments.runs}

    progress = {arm: progress_measures(accounts) for arm, accounts in arms.items()}
    print("| measure | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for measure in progress[next(iter(arms))]:
        print(f"| {measure} | " + " | ".join(_format(progress[arm].get(measure)) for arm in arms) + " |")

    kinds = {arm: stage_kind_measures(accounts) for arm, accounts in arms.items()}
    print("\n| stage kind (win rate / first-attempt win rate) | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for kind in sorted({kind for measures in kinds.values() for kind in measures}):
        cells = [f"{kinds[arm][kind]['win rate']:.0%} / {kinds[arm][kind]['first-attempt win rate']:.0%} ({kinds[arm][kind]['matches']})"
                 if kind in kinds[arm] else "" for arm in arms]
        print(f"| {kind} | " + " | ".join(cells) + " |")

    for arm, accounts in arms.items():
        print(f"\nhardest stages, {arm}:")
        for stage in hardest_stages(accounts):
            print(f"  {stage['stage']} ({stage['kind']}): {stage['losses before first win']} losses before the first win, "
                  f"{stage['accounts never beating it']} accounts never beat it")


def _format(value) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.1%}" if value < 1 else f"{value:.0f}"
    return str(value)


if __name__ == "__main__":
    main()
