"""Does tree search help? Chudledum with tree search against plain Chudledum (default: beta 10.8 both) on the locked tournament
test matchups, like scripts/evaluation/test_lookahead.py (experiments/tree_search/notes.md).

Usage:
    python experiments/tree_search/play_tree_search.py --matchups 100 --blind --guess-builder models/chudledee-v3.4.pt \
        --output runs/tree_search_test/own_turn.json                  # step 1: the tree branches over our own turn
    python experiments/tree_search/play_tree_search.py ... --branch-opponent ...      # step 2: the opponent's replies branch too
"""

import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.deck_guess import build_guesses
from experiments.tree_search.tree_search_match import play_tree_search_matches
from experiments.tree_search.tree_search_settings import TreeSearchSettings
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"), help="the searching player's network")
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--matchups", type=int, default=100, help="test matchups (4 games each)")
    parser.add_argument("--skip-matchups", type=int, default=0, help="start at this matchup (split one test over machines)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=TreeSearchSettings.iterations)
    parser.add_argument("--parallel", type=int, default=TreeSearchSettings.parallel)
    parser.add_argument("--exploration", type=float, default=TreeSearchSettings.exploration)
    parser.add_argument("--branch-opponent", action="store_true", help="the opponent's moves branch too")
    parser.add_argument("--confident-probability", type=float, default=TreeSearchSettings.confident_probability)
    parser.add_argument("--leaf-horizon-rounds", type=int, default=0, help="score left-tree copies by win chance this many rounds later; 0 = play to the end")
    parser.add_argument("--blind", action="store_true", help="deck lists unknown to both players (blind PvP)")
    parser.add_argument("--guess-builder", type=Path, help="blind: Chudledee checkpoint that guesses the opponent's deck")
    parser.add_argument("--guesses", type=int, default=4, help="guessed decks per opponent lineup")
    parser.add_argument("--device", default="", help='e.g. "cuda:1"; default: the first GPU, or the CPU')
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    settings = TreeSearchSettings(arguments.iterations, arguments.parallel, arguments.exploration, arguments.branch_opponent,
                                  arguments.confident_probability, arguments.leaf_horizon_rounds)
    selected = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[arguments.skip_matchups:arguments.skip_matchups + arguments.matchups]
    matchups = [dataclasses.replace(matchup, opponent_deck_known=(not arguments.blind, not arguments.blind)) for matchup in selected]
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    searcher, rollout = NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True)
    opponent = NetworkPolicy(load_combat_network(arguments.opponent, encoder, device), device)

    deck_guesses = None
    if arguments.guess_builder:
        lineups = [tuple(deck.characters) for matchup in matchups for deck in matchup.decks]
        deck_guesses = build_guesses(lineups, arguments.guess_builder, arguments.guesses, arguments.seed)
    total = 4 * len(matchups)
    finished = []
    progress_file = arguments.output.with_suffix(".progress.json")  # for watching a cloud run from the desktop
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    def on_game_finished(row: dict) -> None:
        finished.append(row)
        played = [game for game in finished if "failure" not in game]
        wins = sum(game["searcher_won"] for game in played)
        progress_file.write_text(json.dumps({"finished": len(finished), "total": total, "wins": wins, "played": len(played)}))
        print(f"{len(finished)}/{total} games: tree search wins {wins / max(len(played), 1):.1%} of {len(played)}", flush=True)

    moves_by_kind = Counter()  # played moves of searched decisions, by kind, and how often search changed the kind

    def on_searcher_move(decision, encoded, own_choice, played_response, search) -> None:
        if search is None:
            return
        actions = decision.request["action"]
        played_kind = actions[played_response["chosenActionIndex"]]["action"]["$case"]
        own_kind = actions[own_choice.response["chosenActionIndex"]]["action"]["$case"]
        moves_by_kind[f"played {played_kind}"] += 1
        if played_kind != own_kind:
            moves_by_kind[f"{own_kind} -> {played_kind}"] += 1

    started = time.perf_counter()
    with EnginePool(arguments.workers) as pool:
        torch.manual_seed(arguments.seed)
        rows, table = play_tree_search_matches(pool, encoder, searcher, rollout, opponent, matchups, arguments.seed + arguments.skip_matchups,
                                               settings, on_game_finished, deck_guesses, on_searcher_move)
    played = [row for row in rows if "failure" not in row]
    summary = {
        "settings": dataclasses.asdict(settings), "blind": arguments.blind, "seed": arguments.seed,
        "skip_matchups": arguments.skip_matchups, "matchups": len(matchups),
        "guess_builder": str(arguments.guess_builder) if arguments.guess_builder else None,
        "network": str(arguments.network), "opponent": str(arguments.opponent),
        "games": len(rows), "failures": len(rows) - len(played),
        "wins": sum(row["searcher_won"] for row in played),
        "win_rate": sum(row["searcher_won"] for row in played) / max(len(played), 1),
        "seconds": time.perf_counter() - started, "counts": dict(table.counts), "moves_by_kind": dict(moves_by_kind),
        "search_seconds_median": float(np.median(table.search_seconds)) if table.search_seconds else None,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
