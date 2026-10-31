"""Replay saved prefixes, force play or tune, and roll both branches out with the final network."""

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.card_play_diagnostic.network import load_policy
from experiments.card_play_diagnostic.options import decision_fingerprint, forced_option_indices, response_for_position
from experiments.card_play_diagnostic.sampling import seeded_response
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFailed, GameFinished
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.training.benchmark_matchups import matchup_from_json


@dataclass
class BranchGame:
    situation: dict
    branch: str
    rollout: int
    recorder: MatchRecorder
    history_positions: list[int]
    forced: bool = False
    invalid: bool = False
    decisions_after_branch: int = 0


def run_branches(checkpoint_path: Path, situations: list[dict], rollouts: int, workers: int,
                 concurrent_games: int, seed: int, on_game_done=lambda: None) -> tuple[list[dict], list[dict]]:
    """on_game_done() after each finished or failed game (e.g. to report progress)."""
    torch.manual_seed(seed)
    encoder, policy = load_policy(checkpoint_path)
    tasks = [(situation, branch, rollout) for situation in situations for rollout in range(rollouts)
             for branch in ("play", "tune")]
    random.Random(seed).shuffle(tasks)
    games = {}
    results, failures = [], []
    with EnginePool(workers) as pool:
        def start_one(task) -> None:
            situation, branch, rollout = task
            matchup = matchup_from_json(situation["matchup"])
            game_id = pool.create_game(*matchup.decks, seed=situation["seed"], equipment=matchup.equipment)
            games[game_id] = BranchGame(situation, branch, rollout,
                                        MatchRecorder(matchup.decks, matchup.opponent_deck_known), [0, 0])

        while tasks and len(games) < concurrent_games:
            start_one(tasks.pop())
        while pool.has_active_games():
            pending = []
            for event in pool.poll():
                game = games[event.game_id]
                if isinstance(event, Decision):
                    tracker = game.recorder.on_decision(event)
                    history = game.situation["history"][event.who]
                    position = game.history_positions[event.who]
                    if position < len(history):
                        saved = history[position]
                        if saved["kind"] != event.kind:
                            failures.append(_failure(game, f"prefix kind {event.kind} != {saved['kind']}"))
                            game.invalid = True
                            response = _safe_response(event)
                        else:
                            response = saved["response"]
                        game.history_positions[event.who] += 1
                        game.recorder.on_response(event, response)
                        pool.respond(event, response)
                    else:
                        pending.append((event, encoder.encode(event, tracker), game))
                elif isinstance(event, GameFinished):
                    on_game_done()
                    if game.forced and not game.invalid:
                        results.append({
                        "situation_id": game.situation["situation_id"], "card_id": game.situation["card_id"],
                        "card_type": game.situation["card_type"], "dice_bucket": game.situation["dice_bucket"],
                        "source_game": game.situation["game_number"],
                        "branch": game.branch, "rollout": game.rollout,
                        "won": event.winner == game.situation["seat"], "draw": event.winner is None,
                        "rounds": event.rounds, "remaining_rounds": event.rounds - game.situation["round"],
                        "decisions_after_branch": game.decisions_after_branch,
                        })
                    del games[event.game_id]
                    if tasks:
                        start_one(tasks.pop())
                elif isinstance(event, GameFailed):
                    on_game_done()
                    failures.append(_failure(game, event.message))
                    del games[event.game_id]
                    if tasks:
                        start_one(tasks.pop())
            if pending:
                choices = policy.choose([(decision, encoded) for decision, encoded, _ in pending])
                for (decision, encoded, game), choice in zip(pending, choices):
                    if not game.forced:
                        if decision.who != game.situation["seat"] or decision.kind != "action":
                            failures.append(_failure(game, "did not reach the saved action decision"))
                            game.invalid = True
                            response = choice.response
                        elif decision_fingerprint(decision) != game.situation["decision_fingerprint"]:
                            failures.append(_failure(game, "replayed decision fingerprint differs from saved decision"))
                            game.invalid = True
                            response = choice.response
                        else:
                            positions = forced_option_indices(
                                decision, encoded, game.situation["card_id"], game.branch)
                            if not positions:
                                failures.append(_failure(game, f"no legal {game.branch} option at branch"))
                                game.invalid = True
                                response = choice.response
                            else:
                                weights = [float(choice.probabilities[position]) for position in positions]
                                branch_seed = seed + game.situation["situation_id"] * 100_003 + game.rollout * 2
                                position = random.Random(branch_seed).choices(positions, weights=weights, k=1)[0]
                                response = response_for_position(decision, encoded, position)
                        game.forced = True
                    else:
                        branch_number = 0 if game.branch == "play" else 1
                        response_seed = (seed + game.situation["situation_id"] * 10_000_019
                                         + game.rollout * 100_003 + branch_number * 10_007
                                         + game.decisions_after_branch * 101 + decision.who)
                        response = seeded_response(decision, encoded, choice.probabilities, response_seed)
                        game.decisions_after_branch += 1
                    game.recorder.on_response(decision, response)
                    pool.respond(decision, response)
    results.sort(key=lambda item: (item["situation_id"], item["branch"], item["rollout"]))
    failures.sort(key=lambda item: (item["situation_id"], item["branch"], item["rollout"], item["message"]))
    return results, failures


def _failure(game: BranchGame, message: str) -> dict:
    return {"situation_id": game.situation["situation_id"], "branch": game.branch,
            "rollout": game.rollout, "message": message.splitlines()[0][:300]}


def _safe_response(decision: Decision) -> dict:
    """Only used to let a mismatched replay terminate and produce a useful failure record."""
    if decision.kind == "action":
        valid = next((index for index, action in enumerate(decision.request["action"])
                      if action.get("validity", 0) == 0), 0)
        return {"chosenActionIndex": valid, "usedDice": decision.request["action"][valid]["autoSelectedDice"]}
    if decision.kind == "chooseActive":
        return {"activeCharacterId": decision.request["candidateIds"][0]}
    if decision.kind == "rerollDice":
        return {"diceToReroll": []}
    if decision.kind == "switchHands":
        return {"removedHandIds": []}
    return {"selectedDefinitionId": decision.request["candidateDefinitionIds"][0]}
