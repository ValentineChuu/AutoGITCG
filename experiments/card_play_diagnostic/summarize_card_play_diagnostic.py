#!/usr/bin/env python3
"""Write machine-readable summary tables for the card-play diagnostic."""

import argparse
import json
from pathlib import Path

from experiments.card_play_diagnostic.diagnostic_statistics import (
    branch_summaries, branch_summaries_by_dice, card_summaries, situation_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--situations", type=Path, default=Path("data/derived/card_play_situations.json"))
    parser.add_argument("--branches", type=Path, default=Path("data/derived/card_play_branches.json"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/card_play_summary.json"))
    arguments = parser.parse_args()
    situations = json.loads(arguments.situations.read_text())
    branches = json.loads(arguments.branches.read_text())
    situation_by_id = {item["situation_id"]: item for item in situations["situations"]}
    for result in branches["results"]:
        result.setdefault("source_game", situation_by_id[result["situation_id"]]["game_number"])
    payload = {
        "collection_metadata": situations["metadata"], "branch_metadata": branches["metadata"],
        "failures": branches["failures"],
        "network_by_type_and_dice": situation_summaries(situations["situations"]),
        "counterfactual_by_type": branch_summaries(branches["results"]),
        "counterfactual_by_type_and_dice": branch_summaries_by_dice(branches["results"]),
        "counterfactual_by_card": card_summaries(branches["results"]),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"output": str(arguments.output), "failures": len(payload["failures"]),
                      "categories": len(payload["counterfactual_by_type"])}, indent=2))


if __name__ == "__main__":
    main()
