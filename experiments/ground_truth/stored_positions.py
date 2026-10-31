"""Ground-truth positions as experiments/ground_truth/build_ground_truth.py stored them (one JSON line each), and the matchup each was
taken in. A position is named "<file>:<line number>" so results of other methods can be joined to it."""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups
from autogitcg.training.matchup_sampler import Matchup, swapped


@dataclass(frozen=True)
class StoredPosition:
    name: str
    line: dict


def load_positions(paths: list[Path]) -> list[StoredPosition]:
    """Every stored position whose moves all have results (the analysis skips the rest too)."""
    positions = []
    for path in paths:
        for number, text in enumerate(path.read_text().splitlines()):
            line = json.loads(text)
            if all(row["mean"] is not None for row in line["summary"]):
                positions.append(StoredPosition(f"{path}:{number}", line))
    return positions


def position_matchup(line: dict, blind: bool = True) -> Matchup:
    """The tournament test matchup of the position's game, decks in the stored order."""
    matchup = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[line["matchup"]]
    matchup = swapped(matchup) if line["deck_order"] else matchup
    return dataclasses.replace(matchup, opponent_deck_known=(not blind, not blind))
