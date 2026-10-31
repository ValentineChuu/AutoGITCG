"""Two networks against each other in blind games, each searching with its own judge (autogitcg/search/judges.py;
"none" = no search), on test matchups (the tournament test or the meta test). Each matchup is played 4 times: both deck orders, each network in each
seat. Writes per game a readable move list (games/<n>.txt) and a replayable log
(games/<n>.jsonl.gz, analysis/duel/game_log.py), plus summary.json. The matchups are the benchmark's in a fixed shuffled order (--order-seed);
--skip-matchups and --matchups pick a slice, so shards on several machines split one test.

In each side's search, the game copies are played by its own network without reply inputs (copies never compute
replies); its real moves read them, with the learned selector.

    python scripts/evaluation/judge_duel.py --device cuda --first models/theta-1.13.pt --first-judge quick \
        --second models/eta-7.3.pt --second-judge quick --benchmark meta --matchups 100 --output runs/duel/theta_1.13_vs_eta_7.3
"""

import argparse
import dataclasses
import json
import random
from pathlib import Path

import torch

from autogitcg.analysis.duel.readable_game import ReadableGame
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.analysis.duel.game_log import GameLog
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.search.judges import JUDGES
from autogitcg.training.benchmark_matchups import META_BENCHMARK_FILE, TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups
from autogitcg.training.matchup_sampler import Matchup

def swapped(matchup: Matchup) -> Matchup:
    return dataclasses.replace(matchup, decks=matchup.decks[::-1], deck_origins=matchup.deck_origins[::-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--first-name", default="")
    parser.add_argument("--second-name", default="")
    parser.add_argument("--first-judge", choices=list(JUDGES), default="deep")
    parser.add_argument("--second-judge", choices=list(JUDGES), default="deep")
    parser.add_argument("--benchmark", choices=("tournament", "meta"), default="tournament",
                        help="tournament: the locked held-out test matchups; meta: the most played 6.4+ training decks")
    parser.add_argument("--matchups", type=int, default=5, help="test matchups (4 games each)")
    parser.add_argument("--skip-matchups", type=int, default=0, help="start after this many (shards)")
    parser.add_argument("--order-seed", type=int, default=20261027, help="shuffles the matchup order (same for every shard)")
    parser.add_argument("--reply-checks", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--device", default="cpu", help='the networks\' device, e.g. "cuda"')
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    judges = (JUDGES[arguments.first_judge], JUDGES[arguments.second_judge])
    benchmark = load_benchmark_matchups(META_BENCHMARK_FILE if arguments.benchmark == "meta" else TOURNAMENT_BENCHMARK_FILE)
    random.Random(arguments.order_seed).shuffle(benchmark)
    matchups = [dataclasses.replace(matchup, opponent_deck_known=(False, False))
                for matchup in benchmark[arguments.skip_matchups:arguments.skip_matchups + arguments.matchups]]
    device = torch.device(arguments.device)
    encoder = ObservationEncoder()
    players = {}  # name -> (searcher, rollout)
    for path, name in ((arguments.first, arguments.first_name), (arguments.second, arguments.second_name)):
        network = load_combat_network(path, encoder, device)
        players[name or f"{path.parent.name} {path.stem}"] = (NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True))
    names = list(players)
    # Held-out test decks are left out of the guesses; the meta test's decks are public lists the guesses keep.
    excluded = [deck for matchup in matchups for deck in matchup.decks] if arguments.benchmark == "tournament" else []
    guesses = TournamentDeckGuesses(max(judge.redeals for judge in judges), arguments.seed, excluded=excluded)
    games_directory = arguments.output / "games"
    games_directory.mkdir(parents=True, exist_ok=True)
    readable, logs, rows = {}, {}, []

    def on_real_move(row, decision, response_value, search) -> None:
        readable[row["game_id"]].on_move(decision, response_value, search)
        logs[row["game_id"]].on_move(decision, response_value, search, human=False)

    def on_game_end(row, event, recorder) -> None:
        number = row["number"]
        (games_directory / f"{number:02d}.txt").write_text(readable.pop(row["game_id"]).finish(row))
        logs.pop(row["game_id"]).end({"winner": row.get("winner"), "rounds": row.get("rounds"), "failure": row.get("failure")})

    with EnginePool(arguments.workers, replies=True) as pool:
        first_searcher = players[names[0]][0]
        table = LookaheadTable(pool, encoder, first_searcher, players[names[0]][1], first_searcher, judges[0], arguments.seed,
                               guesses, on_game_end=on_game_end, on_real_move=on_real_move,
                               reply_selection=ReplySelection(arguments.reply_checks))
        for number, original in enumerate(matchups):
            for deck_order, matchup in enumerate((original, swapped(original))):
                for first_seat in (0, 1):
                    seat_names = {first_seat: names[0], 1 - first_seat: names[1]}
                    game_number = len(rows) + 1
                    row = {"number": game_number, "matchup": number, "deck_order": deck_order,
                           "names": [seat_names[0], seat_names[1]], "decks": list(matchup.deck_origins)}
                    seed = arguments.seed + arguments.skip_matchups + number
                    game_id = table.start_game(matchup, seed, None, row, searchers={seat: players[name] for seat, name in seat_names.items()},
                                               judges={first_seat: judges[0], 1 - first_seat: judges[1]})
                    lineups = " vs ".join(", ".join(str(character) for character in deck.characters) for deck in matchup.decks)
                    readable[game_id] = ReadableGame(f"Game {game_number}: {seat_names[0]} (seat 0, {matchup.deck_origins[0]}) vs "
                                                     f"{seat_names[1]} (seat 1, {matchup.deck_origins[1]}); characters {lineups}", seat_names)
                    logs[game_id] = GameLog(games_directory / f"{game_number:02d}.jsonl.gz", {
                        "seed": seed, "human_seat": None, "searcher_seat": None, "names": row["names"],
                        "decks": [dataclasses.asdict(deck) for deck in matchup.decks],
                        "equipment": [dataclasses.asdict(item) for item in matchup.equipment],
                        "deck_known": [False, False], "lookahead": dataclasses.asdict(judges[0]),
                        "judges": {seat_names[first_seat]: arguments.first_judge, seat_names[1 - first_seat]: arguments.second_judge}})
                    rows.append(row)
        finished = 0
        while table.game_count:
            for row in table.step():
                finished += 1
                wins = {name: sum(done.get("winner") is not None and done["names"][done["winner"]] == name for done in rows) for name in names}
                print(f"{finished}/{len(rows)} games: " + " / ".join(f"{name} {count}" for name, count in wins.items()), flush=True)

    played = [row for row in rows if "failure" not in row]
    summary = {"names": names, "judges": [arguments.first_judge, arguments.second_judge], "benchmark": arguments.benchmark,
               "matchups": [arguments.skip_matchups, arguments.skip_matchups + len(matchups)], "games": len(rows), "failures": len(rows) - len(played),
               "wins": {name: sum(row.get("winner") is not None and row["names"][row["winner"]] == name for row in played) for name in names},
               "search": dict(table.counts), "rows": rows}
    (arguments.output / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=1))


if __name__ == "__main__":
    main()
