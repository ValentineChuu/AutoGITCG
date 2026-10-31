#!/usr/bin/env python3
"""Collect final-network play-versus-tune decisions from fixed benchmark decks, or from human tournament decks."""

import argparse
import json
from pathlib import Path

from experiments.card_play_diagnostic.collect import collect_situations
from experiments.card_play_diagnostic.tournament_matchups import TOURNAMENT_ORIGIN, tournament_vs_chudledee_matchups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/alpha-2.40.pt"))
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--concurrent-games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--output", type=Path, default=Path("data/derived/card_play_situations.json"))
    parser.add_argument("--tournament-decks", action="store_true",
                        help="locked-test tournament decks vs Chudledee decks, tournament side only (default: benchmark decks)")
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v3.4.pt"))
    arguments = parser.parse_args()
    matchups, origin_prefix = None, ""
    if arguments.tournament_decks:
        matchups = tournament_vs_chudledee_matchups(arguments.builder, arguments.games, arguments.seed)
        origin_prefix = TOURNAMENT_ORIGIN
    finished = 0

    def on_game_done() -> None:
        nonlocal finished
        finished += 1

    situations, metadata = collect_situations(arguments.checkpoint, arguments.games, arguments.workers, arguments.seed,
                                               concurrent_games=arguments.concurrent_games, matchups=matchups,
                                               origin_prefix=origin_prefix, on_game_done=on_game_done)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"metadata": metadata, "situations": situations}, separators=(",", ":")))
    print(json.dumps({**metadata, "output": str(arguments.output)}, indent=2))


if __name__ == "__main__":
    main()
