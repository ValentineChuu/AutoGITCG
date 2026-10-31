"""A short text report on a training run, running or finished, printed to the terminal.

1. One line per benchmark after --since: the update, win rate against the scripted heuristic, rounds per win,
   reactions per game, win rate against random play, and engine failures.
2. The training games of the last --recent-updates updates: how they are spread over matchup categories (stage,
   tournament, NPC decks, ...) and how often the learner wins each, then what triggers the learner's elemental
   reactions (its skills by type, summons, statuses).

Usage:
    python scripts/training/run_progress.py --run runs/theta_01 --since 100 --recent-updates 20
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def benchmark_lines(run: Path, since: int) -> list[str]:
    lines = []
    for line in (run / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["update"] <= since or "evaluation_heuristic" not in record:
            continue
        heuristic, random_play = record["evaluation_heuristic"], record["evaluation_random"]
        lines.append(f"{record['update']:5d}  vs heuristic {heuristic['win_rate']:.3f}  rounds/win {heuristic['mean_rounds_of_wins']:.2f}  "
                     f"reactions/game {heuristic['learner_reactions_per_game']:.2f}  vs random {random_play['win_rate']:.3f}  "
                     f"failures {heuristic['failures'] + random_play['failures']}")
    return lines


def recent_game_summary(run: Path, recent_updates: int) -> list[str]:
    files = sorted((run / "games").glob("update_*.jsonl.gz"))[-recent_updates:]
    games, focus_played, focus_won = Counter(), Counter(), Counter()
    causes = Counter()
    for path in files:
        for line in gzip.open(path, "rt"):
            record = json.loads(line)
            group = "/".join(record["category"].split("/")[:2])
            games[group] += 1
            focus = record["focus_seat"]
            if record["seats"][focus] == "learner":
                focus_played[group] += 1
                focus_won[group] += record["winner"] == focus
            for seat, seat_causes in enumerate(record["reaction_causes"]):
                if record["seats"][seat] == "learner":
                    for cause, count in seat_causes.items():
                        caller_type, skill_type = cause.split("|")[1:]
                        causes[f"{caller_type}:{skill_type}" if caller_type == "character" else caller_type] += count
    total_games = sum(games.values())
    lines = [f"last {len(files)} updates: {total_games} games"]
    for group, count in games.most_common():
        rate = focus_won[group] / focus_played[group] if focus_played[group] else float("nan")
        lines.append(f"  {group:38} share {count / total_games:5.1%}  focus-side win {rate:.2f} ({focus_played[group]} games)")
    total_reactions = sum(causes.values())
    lines.append("learner reaction causes: " + ", ".join(f"{cause} {count / total_reactions:.0%}" for cause, count in causes.most_common()))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--since", type=int, default=0, help="show benchmarks after this update")
    parser.add_argument("--recent-updates", type=int, default=20)
    arguments = parser.parse_args()
    print("\n".join(benchmark_lines(arguments.run, arguments.since)))
    print("\n".join(recent_game_summary(arguments.run, arguments.recent_updates)))


if __name__ == "__main__":
    main()
