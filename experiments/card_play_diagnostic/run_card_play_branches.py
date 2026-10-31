#!/usr/bin/env python3
"""Run paired counterfactual rollouts from collected play-versus-tune decisions."""

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.card_play_diagnostic.branch import run_branches
from experiments.card_play_diagnostic.situation_selection import select_situations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/alpha-2.40.pt"))
    parser.add_argument("--situations", type=Path, default=Path("data/derived/card_play_situations.json"))
    parser.add_argument("--per-type", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=40)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--concurrent-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output", type=Path, default=Path("data/derived/card_play_branches.json"))
    arguments = parser.parse_args()
    source = json.loads(arguments.situations.read_text())
    selected = select_situations(source["situations"], arguments.per_type, arguments.seed)
    finished = 0

    def on_game_done() -> None:
        nonlocal finished
        finished += 1

    results, failures = run_branches(arguments.checkpoint, selected, arguments.rollouts, arguments.workers,
                                     arguments.concurrent_games, arguments.seed, on_game_done)
    payload = {
        "metadata": {"checkpoint": str(arguments.checkpoint), "rollouts_per_branch": arguments.rollouts,
                     "selected_by_type": Counter(item["card_type"] for item in selected), "seed": arguments.seed,
                     "method": "deterministic prefix replay; forced branch; stochastic final network on both seats"},
        "selected_situation_ids": [item["situation_id"] for item in selected],
        "results": results, "failures": failures,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({**payload["metadata"], "results": len(results), "failures": len(failures),
                      "output": str(arguments.output)}, indent=2))


if __name__ == "__main__":
    main()
