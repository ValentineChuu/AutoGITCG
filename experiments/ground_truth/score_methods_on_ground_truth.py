"""Scores a search method on the ground-truth set (experiments/tree_search/notes.md, step 3). Every stored position is
replayed to its decision (ground_truth/position_replay_pool.py) and handed to a look-ahead or tree-search table whose
own move is the network's most likely one (greedy). Per position, the method's chosen move and its score for every
compared move it tried are written as compared-move indices, for experiments/ground_truth/compare_methods_on_ground_truth.py.

Methods:
    --method lookahead --horizon 1 --candidates 8 --every-move-kind --redeals 16 --margin 0
        the network's win chance right after each move, as a one-move judge (more --horizon: deeper; --score-by value
        and --own-view-only: see LookaheadSettings)
    --method lookahead --candidates 4 --horizon 24 --redeals 4 --margin 0.05
        the playtest site's look-ahead (autogitcg/playtest/bot_table.LOOKAHEAD)
    --method tree --iterations 64
        the tree search of the 2026-10-20 tests

Usage:
    python experiments/ground_truth/score_methods_on_ground_truth.py runs/ground_truth/shard*.jsonl ... --method lookahead --horizon 1 \
        --shard 0 --shards 5 --guess-builder models/chudledee-v3.4.pt --output runs/ground_truth_scores/judge_shard0
"""

import argparse
import dataclasses
import json
import time
from collections import deque
from pathlib import Path

import torch

from autogitcg.bridge.engine_pool import EnginePool
from experiments.ground_truth.compared_move_index import compared_move_index
from autogitcg.deck_guessing.reply_guess_candidates import ReplyGuessCandidates
from experiments.ground_truth.position_replay_pool import PositionReplayPool
from experiments.ground_truth.stored_positions import StoredPosition, load_positions, position_matchup
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.search.deck_guess import build_guesses
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from experiments.tree_search.tree_search_settings import TreeSearchSettings
from experiments.tree_search.tree_search_table import TreeSearchTable


def lookahead_scores(line: dict, decision, encoded, search) -> dict:
    """Per compared move: the mean score of its copies (None: not tried)."""
    scores = [None] * len(line["moves"])
    if search is not None:
        for candidate, mean in zip(search.candidate_indices, search.mean_scores):
            index = compared_move_index(line, decision, encoded.request_indices[candidate])
            if index is not None and scores[index] is None:
                scores[index] = mean
    return {"scores": scores}


def tree_scores(line: dict, decision, search) -> dict:
    """Per compared move: the root's visits and mean result (None: not visited)."""
    visits, scores = [None] * len(line["moves"]), [None] * len(line["moves"])
    if search is not None:
        for key, (count, mean) in search.root_visits().items():
            index = compared_move_index(line, decision, search.options[key].response["chosenActionIndex"])
            if index is not None and visits[index] is None:
                visits[index], scores[index] = count, mean
    return {"scores": scores, "visits": visits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--method", choices=("lookahead", "tree"), required=True)
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--open-positions", type=int, default=12, help="positions replayed or searched at once")
    parser.add_argument("--candidates", type=int, default=LookaheadSettings.candidates)
    parser.add_argument("--redeals", type=int, default=LookaheadSettings.redeals)
    parser.add_argument("--horizon", type=int, default=LookaheadSettings.horizon)
    parser.add_argument("--horizon-rounds", type=int, default=0)
    parser.add_argument("--margin", type=float, default=LookaheadSettings.margin)
    parser.add_argument("--every-move-kind", action="store_true")
    parser.add_argument("--score-by", choices=("win", "value"), default=LookaheadSettings.score_by)
    parser.add_argument("--own-view-only", action="store_true", help="score copies only at the searcher's own decisions")
    parser.add_argument("--iterations", type=int, default=TreeSearchSettings.iterations)
    parser.add_argument("--branch-opponent", action="store_true")
    parser.add_argument("--move-value-weight", type=float, default=0.0, help="the searcher's move choice adds weight x the move-value output")
    parser.add_argument("--move-value-candidates", type=int, default=0, help="only the policy's this many top moves get that bonus (0 = all)")
    parser.add_argument("--guess-builder", type=Path)
    parser.add_argument("--guesses", type=int, default=4)
    parser.add_argument("--guess-source", choices=("chudledee", "tournament"), default="tournament",
                        help="blind: guess the opponent's deck from similar tournament decks (opponents are modelled from public "
                             "data: about 66%% of hidden cards right, the matchups' own decks left out), or, for comparison only, "
                             "Chudledee (--guess-builder; about 10%%)")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True, help="path without extension")
    arguments = parser.parse_args()

    positions = load_positions(arguments.files)[arguments.shard::arguments.shards]
    device = torch.device(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    greedy = NetworkPolicy(network, device, greedy=True)
    searcher = NetworkPolicy(network, device, greedy=True, move_value_weight=arguments.move_value_weight,
                             move_value_candidates=arguments.move_value_candidates)
    deck_guesses = None
    if arguments.guess_source == "tournament":
        deck_guesses = TournamentDeckGuesses(arguments.redeals, arguments.seed,
                                             excluded=[deck for position in positions for deck in position_matchup(position.line).decks])
    elif arguments.guess_builder:
        lineups = [tuple(deck.characters) for position in positions for deck in position_matchup(position.line).decks]
        deck_guesses = build_guesses(lineups, arguments.guess_builder, arguments.guesses, arguments.seed)
    if arguments.method == "lookahead":
        settings = LookaheadSettings(candidates=arguments.candidates, redeals=arguments.redeals, horizon=arguments.horizon,
                                     margin=arguments.margin, every_move_kind=arguments.every_move_kind,
                                     horizon_rounds=arguments.horizon_rounds, score_by=arguments.score_by,
                                     own_view_only=arguments.own_view_only)
    else:
        settings = TreeSearchSettings(iterations=arguments.iterations, branch_opponent=arguments.branch_opponent)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    lines_file = arguments.output.with_suffix(".jsonl").open("w")
    progress_file = arguments.output.with_suffix(".progress.json")
    waiting = deque(positions)
    by_game: dict[int, StoredPosition] = {}  # adopted game id -> its position
    answered: list[int] = []  # adopted games whose move is final, stopped after the table's step
    done = {"positions": 0, "lost": 0}
    started = time.perf_counter()

    def write(row: dict) -> None:
        lines_file.write(json.dumps(row) + "\n")
        lines_file.flush()
        done["positions"] += 1
        done["lost"] += "lost" in row
        progress_file.write_text(json.dumps({"finished": done["positions"], "total": len(positions), **done,
                                             "minutes": round((time.perf_counter() - started) / 60, 1)}))

    def on_reached(position: StoredPosition, decision, recorder) -> None:
        line = position.line
        opponent_deck = position_matchup(line).decks[1 - line["seat"]]
        row = {"searcher_seat": line["seat"], "human_seat": None, "opponent_lineup": tuple(sorted(opponent_deck.characters))}
        table.adopt_game(decision.game_id, row, recorder)
        by_game[decision.game_id] = position

    def finish(decision, encoded, own_request_index: int, chosen_request_index: int, scores: dict) -> None:
        position = by_game.pop(decision.game_id)
        write({"position": position.name, "searched": scores["scores"] != [None] * len(position.line["moves"]),
               "own_move": compared_move_index(position.line, decision, own_request_index),
               "chosen_move": compared_move_index(position.line, decision, chosen_request_index), **scores})
        answered.append(decision.game_id)

    def on_lookahead_move(decision, encoded, own_choice, chosen_index, search) -> None:
        finish(decision, encoded, own_choice.response["chosenActionIndex"], encoded.request_indices[chosen_index],
               lookahead_scores(by_game[decision.game_id].line, decision, encoded, search))

    def on_tree_move(decision, encoded, own_choice, played_response, search) -> None:
        finish(decision, encoded, own_choice.response["chosenActionIndex"], played_response["chosenActionIndex"],
               tree_scores(by_game[decision.game_id].line, decision, search))

    with EnginePool(arguments.workers, previews=network.config.reads_previews,
                    replies=network.config.reads_replies) as engine_pool:
        reply_guesses = ReplyGuessCandidates(deck_guesses.decks) if isinstance(deck_guesses, TournamentDeckGuesses) else None
        pool = PositionReplayPool(engine_pool, on_reached, lambda position: write({"position": position.name, "lost": "replay"}),
                                  reply_guesses)
        torch.manual_seed(arguments.seed)
        if arguments.method == "lookahead":
            table = LookaheadTable(pool, encoder, searcher, greedy, greedy, settings, arguments.seed, deck_guesses,
                                   on_searcher_move=on_lookahead_move)
        else:
            table = TreeSearchTable(pool, encoder, greedy, greedy, greedy, settings, arguments.seed, deck_guesses,
                                    on_searcher_move=on_tree_move)
        while done["positions"] < len(positions):
            while waiting and len(pool.replays) + table.game_count < arguments.open_positions:
                pool.replay(waiting.popleft())
            for row in table.step(timeout=5):
                if (position := by_game.pop(row["game_id"], None)) is not None:
                    write({"position": position.name, "lost": "game ended during search"})
            for game_id in answered:
                table.stop_game(game_id)
            answered.clear()
    lines_file.close()
    summary = {"method": arguments.method, "settings": dataclasses.asdict(settings), "positions": len(positions), **done,
               "seconds": time.perf_counter() - started, "replay counts": dict(pool.counts), "table counts": dict(table.counts)}
    arguments.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
