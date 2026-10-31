"""Ground-truth set: at sampled unsure decisions of plain network games, the true best move by playing every compared
move out on the same guessed deals (experiments/ground_truth/, experiments/tree_search/notes.md). Blind PvP on the locked
tournament test matchups by default. Run several shards side by side (--skip-matchups): one process drives about
6 engine workers.

Output: <output>.jsonl (one position per line: game seed and every response before it, the compared moves with the
network's probabilities and paired results) and <output>.pickle (per position: the Decision and the sampled seat's
MatchRecorder, to encode it for any network; one pickle.dump per position, appended as each finishes: read with
pickle.load in a loop until EOFError).

Usage:
    python experiments/ground_truth/build_ground_truth.py --matchups 20 --skip-matchups 0 --blind \
        --guess-builder models/chudledee-v3.4.pt --output runs/ground_truth/shard0
"""

import argparse
import dataclasses
import json
import pickle
import time
from pathlib import Path

import torch

from autogitcg.bridge.engine_pool import EnginePool
from experiments.ground_truth.ground_truth_settings import GroundTruthSettings
from experiments.ground_truth.ground_truth_table import GroundTruthTable, Position
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.search.deck_guess import build_guesses
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups
from autogitcg.training.matchup_sampler import swapped


def position_line(row: dict, position: Position, matchup_number: int, deck_order: int, separation: float) -> dict:
    comparison = position.comparison
    own_index = position.choice.response["chosenActionIndex"]
    return {
        "matchup": matchup_number, "deck_order": deck_order, "seat": row["sampled_seat"], "seed": row["seed"],
        "responses_before": row["responses_before"], "round": position.decision.state.get("roundNumber"),
        "moves": [{"key": move.key, "response": move.response, "probability": move.probability} for move in position.moves],
        "network_choice": next((index for index, move in enumerate(position.moves) if move.response["chosenActionIndex"] == own_index), None),
        "network_top": 0,  # moves are ranked by the network's probability first
        "best": comparison.best(), "separated": comparison.separated(separation), "deals": position.deals_completed,
        "summary": comparison.summary(),
        "results": [[comparison.results[move].get(deal, 0) for deal in range(position.deals_started)] for move in range(len(position.moves))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--matchups", type=int, default=20)
    parser.add_argument("--skip-matchups", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-deals", type=int, default=GroundTruthSettings.max_deals)
    parser.add_argument("--deals-in-flight", type=int, default=GroundTruthSettings.deals_in_flight)
    parser.add_argument("--minimum-deals", type=int, default=GroundTruthSettings.minimum_deals)
    parser.add_argument("--max-open-copies", type=int, default=GroundTruthSettings.max_open_copies)
    parser.add_argument("--sample-probability", type=float, default=GroundTruthSettings.sample_probability)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--guess-builder", type=Path)
    parser.add_argument("--guesses", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    arguments = parser.parse_args()

    settings = dataclasses.replace(GroundTruthSettings(), max_deals=arguments.max_deals, deals_in_flight=arguments.deals_in_flight,
                                   minimum_deals=arguments.minimum_deals, max_open_copies=arguments.max_open_copies,
                                   sample_probability=arguments.sample_probability)
    selected = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[arguments.skip_matchups:arguments.skip_matchups + arguments.matchups]
    matchups = [dataclasses.replace(matchup, opponent_deck_known=(not arguments.blind, not arguments.blind)) for matchup in selected]
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    player, rollout = NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True)
    deck_guesses = None
    if arguments.guess_builder:
        lineups = [tuple(deck.characters) for matchup in matchups for deck in matchup.decks]
        deck_guesses = build_guesses(lineups, arguments.guess_builder, arguments.guesses, arguments.seed)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    lines_file = arguments.output.with_suffix(".jsonl").open("w")
    pickle_file = arguments.output.with_suffix(".pickle").open("wb")
    games = 4 * len(matchups)
    progress_file = arguments.output.with_suffix(".progress.json")
    started = time.perf_counter()
    done = {"games": 0, "positions": 0}
    origin = {}  # game id -> (matchup number, deck order)

    def report(table: GroundTruthTable | None = None) -> None:
        details = {} if table is None else {
            "minutes": round((time.perf_counter() - started) / 60, 1), **table.counts,
            "open copies": len(table.copies), "open positions": len(table.positions),
            "deals completed in open positions": sum(position.deals_completed for position in table.positions)}
        progress_file.write_text(json.dumps({"finished": done["games"], "total": games, "positions": done["positions"], **details}))

    def on_position(row: dict, position: Position) -> None:
        matchup_number, deck_order = origin[row["game_id"]]
        lines_file.write(json.dumps(position_line(row, position, matchup_number, deck_order, settings.separation)) + "\n")
        lines_file.flush()
        pickle.dump({"decision": position.decision, "recorder": position.recorder}, pickle_file)
        pickle_file.flush()
        done["positions"] += 1

    with EnginePool(arguments.workers) as pool:
        torch.manual_seed(arguments.seed)
        table = GroundTruthTable(pool, encoder, player, rollout, settings, arguments.seed, deck_guesses, on_position)
        for number, original in enumerate(matchups):
            for deck_order, matchup in enumerate((original, swapped(original))):
                for seat in (0, 1):
                    row = {}
                    table.start_game(matchup, arguments.seed + arguments.skip_matchups + number, seat, row)
                    origin[row["game_id"]] = (arguments.skip_matchups + number, deck_order)
        open_games, reported = len(table.games), time.perf_counter()
        while table.busy:
            table.step(timeout=5)
            if len(table.games) != open_games or time.perf_counter() - reported > 30:
                done["games"] += open_games - len(table.games)
                open_games, reported = len(table.games), time.perf_counter()
                report(table)
    lines_file.close()
    pickle_file.close()
    summary = {"positions": done["positions"], "games": games, "seconds": time.perf_counter() - started,
               "counts": dict(table.counts), "settings": dataclasses.asdict(settings)}
    arguments.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
