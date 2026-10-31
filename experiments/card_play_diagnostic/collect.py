"""Collect real final-network decisions where one card can be played or tuned."""

import copy
import random
from collections import defaultdict
from pathlib import Path

import torch

from experiments.card_play_diagnostic.network import load_policy
from experiments.card_play_diagnostic.options import comparable_cards, decision_fingerprint
from experiments.card_play_diagnostic.sampling import seeded_response
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.training.benchmark_matchups import load_benchmark_matchups, matchup_to_json


def collect_situations(checkpoint_path: Path, game_count: int, workers: int, seed: int,
                       per_game_type_limit: int = 2, concurrent_games: int = 1, matchups: list | None = None,
                       origin_prefix: str = "", on_game_done=lambda: None) -> tuple[list[dict], dict]:
    """matchups: played in turn (default: the fixed benchmark matchups). origin_prefix: keep only decisions of the
    seat whose deck origin starts with it (e.g. the tournament deck's side)."""
    torch.manual_seed(seed)
    encoder, policy = load_policy(checkpoint_path)
    matchups = matchups or load_benchmark_matchups()
    random_generator = random.Random(seed)
    games = {}
    situations = []
    failures = []
    with EnginePool(workers) as pool:
        next_game_number = 0

        def start_one(number: int) -> None:
            matchup = matchups[number % len(matchups)]
            game_seed = seed + number
            game_id = pool.create_game(*matchup.decks, seed=game_seed, equipment=matchup.equipment)
            games[game_id] = {
                "number": number,
                "seed": game_seed,
                "matchup": matchup,
                "recorder": MatchRecorder(matchup.decks, matchup.opponent_deck_known),
                "history": [[], []],
                "seen": defaultdict(int),
                "reservoirs": defaultdict(list),
            }

        while next_game_number < game_count and len(games) < concurrent_games:
            start_one(next_game_number)
            next_game_number += 1
        while pool.has_active_games():
            pending = []
            for event in pool.poll():
                game = games[event.game_id]
                if isinstance(event, Decision):
                    tracker = game["recorder"].on_decision(event)
                    pending.append((event, encoder.encode(event, tracker)))
                elif isinstance(event, GameFailed):
                    on_game_done()
                    failures.append({"game": game["number"], "message": event.message})
                    del games[event.game_id]
                    if next_game_number < game_count:
                        start_one(next_game_number)
                        next_game_number += 1
                elif isinstance(event, GameFinished):
                    on_game_done()
                    for reservoir in game["reservoirs"].values():
                        situations.extend(reservoir)
                    del games[event.game_id]
                    if next_game_number < game_count:
                        start_one(next_game_number)
                        next_game_number += 1
            for (decision, encoded), choice in zip(pending, policy.choose(pending)):
                game = games[decision.game_id]
                kept_seat = game["matchup"].deck_origins[decision.who].startswith(origin_prefix)
                for facts in comparable_cards(decision, encoded, choice.probabilities) if kept_seat else ():
                    category = facts["card_type"]
                    game["seen"][category] += 1
                    situation = {
                        **facts,
                        "value": choice.value,
                        "win_probability": choice.win_probability,
                        "seat": decision.who,
                        "decision_fingerprint": decision_fingerprint(decision),
                        "game_number": game["number"],
                        "seed": game["seed"],
                        "matchup": matchup_to_json(game["matchup"]),
                        "history": copy.deepcopy(game["history"]),
                    }
                    reservoir = game["reservoirs"][category]
                    if len(reservoir) < per_game_type_limit:
                        reservoir.append(situation)
                    else:
                        replacement = random_generator.randrange(game["seen"][category])
                        if replacement < per_game_type_limit:
                            reservoir[replacement] = situation
                decision_number = len(game["history"][decision.who])
                response = seeded_response(decision, encoded, choice.probabilities,
                                           seed + game["number"] * 1_000_003 + decision.who * 100_003 + decision_number)
                game["recorder"].on_response(decision, response)
                game["history"][decision.who].append({"kind": decision.kind, "response": response})
                pool.respond(decision, response)
    situations.sort(key=lambda item: (item["game_number"], item["card_type"], item["round"], item["card_id"],
                                      item["decision_fingerprint"]))
    for number, situation in enumerate(situations):
        situation["situation_id"] = number
    return situations, {"games": game_count, "situations": len(situations), "failures": failures,
                        "checkpoint": str(checkpoint_path), "seed": seed,
                        "concurrent_games": concurrent_games,
                        "policy": "stochastic network on both seats"}
