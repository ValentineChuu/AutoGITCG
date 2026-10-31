"""How long look-ahead takes per decision in a real game: plays test matchups one game at a time (searcher vs plain
network, deck lists known) on the CPU, and reports the time of every searched decision and the search count per game.

Usage (CPU only; the desktop GPU is off limits):
    CUDA_VISIBLE_DEVICES="" python scripts/evaluation/measure_lookahead_latency.py --games 8 --workers 4 --torch-threads 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4, help="engine processes for the game copies")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--redeals", type=int, default=LookaheadSettings.redeals)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--margin", type=float, default=LookaheadSettings.margin)
    parser.add_argument("--confident-probability", type=float, default=LookaheadSettings.confident_probability)
    parser.add_argument("--every-move-kind", action="store_true")
    parser.add_argument("--score-by", choices=("win", "value"), default=LookaheadSettings.score_by)
    parser.add_argument("--own-view-only", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    torch.set_num_threads(arguments.torch_threads)
    settings = LookaheadSettings(arguments.candidates, arguments.redeals, arguments.horizon,
                                 confident_probability=arguments.confident_probability, margin=arguments.margin,
                                 every_move_kind=arguments.every_move_kind, score_by=arguments.score_by,
                                 own_view_only=arguments.own_view_only)
    device = torch.device("cpu")
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    matchups = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)
    searches_per_game, game_seconds, seconds = [], [], []
    with EnginePool(arguments.workers, previews=network.config.reads_previews,
                        replies=network.config.reads_replies) as pool:
        for game in range(arguments.games):
            table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                                   NetworkPolicy(network, device), settings, seed=game)
            table.start_game(matchups[game], seed=game, searcher_seat=game % 2, row={})
            started = time.perf_counter()
            while table.game_count:
                table.step()
            game_seconds.append(time.perf_counter() - started)
            searches_per_game.append(len(table.search_seconds))
            seconds.extend(table.search_seconds)
            print(f"game {game}: {len(table.search_seconds)} searches, {game_seconds[-1]:.0f}s, "
                  f"median search {np.median(table.search_seconds or [0]):.2f}s", flush=True)
    summary = {"settings": vars(arguments) | {"network": str(arguments.network), "output": str(arguments.output)},
               "searches_per_game": float(np.mean(searches_per_game)), "game_seconds": float(np.mean(game_seconds)),
               "search_seconds": {f"p{q}": float(np.percentile(seconds, q)) for q in (10, 50, 90, 99)} if seconds else {}}
    print(json.dumps(summary, indent=2))
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
