"""How many checked moves does the reply selector need? (would 2 do instead of 4?) A replies network plays
blind games against plain beta 10.8 on the locked test matchups; at each of its decisions every move's replies are computed,
and the share of the total probability change (what the selector is trained on) that its top 1, 2, 3, 4 and 6 picks by
reply interest would have covered is recorded.

    python scripts/evaluation/measure_reply_coverage.py --network models/eta-7.3.pt --matchups 20 --output runs/reply_coverage/u150.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision, GameFinished, GameFailed, RepliesReady
from autogitcg.bridge.reply_views import reply_views
from autogitcg.deck_guessing.reply_guess_candidates import ReplyGuessCandidates
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.match_recorder import MatchRecorder
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.reply_selection import interest_target, on_demand
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups

PICKS = (1, 2, 3, 4, 6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--matchups", type=int, default=20)
    parser.add_argument("--skip-matchups", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    matchups = load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[arguments.skip_matchups:arguments.skip_matchups + arguments.matchups]
    device = torch.device("cpu")
    encoder = ObservationEncoder()
    network = NetworkPolicy(load_combat_network(arguments.network, encoder, device), device)
    opponent = NetworkPolicy(load_combat_network(arguments.opponent, encoder, device), device)
    excluded = [deck for matchup in matchups for deck in matchup.decks]
    guesses = ReplyGuessCandidates(TournamentDeckGuesses(4, arguments.seed, excluded=excluded).decks)
    torch.manual_seed(arguments.seed)
    games, waiting = {}, {}  # game id -> (recorder, network seat); game id -> (decision, tracker, first look)
    covered = {picks: 0.0 for picks in PICKS}
    total, decisions, per_decision = 0.0, 0, {picks: [] for picks in PICKS}

    def respond(game_id, decision, choice) -> None:
        recorder = games[game_id][0]
        recorder.on_response(decision, choice.response)
        pool.respond(decision, choice.response)

    with EnginePool(arguments.workers, replies=True) as pool:
        for number, matchup in enumerate(matchups):
            for seat in (0, 1):
                views = reply_views((seat == 0, seat == 1), (False, False), matchup.decks, guesses)
                views = [on_demand(view) for view in views]
                game_id = pool.create_game(*matchup.decks, seed=arguments.seed + number, reply_views=views)
                games[game_id] = (MatchRecorder(matchup.decks, opponent_deck_known=(False, False)), seat)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, (GameFinished, GameFailed)):
                    games.pop(event.game_id, None)
                elif isinstance(event, RepliesReady):
                    decision, tracker, first = waiting.pop(event.game_id)
                    decision.replies, decision.replies_checked = event.replies, event.checked
                    encoded = encoder.encode(decision, tracker)
                    second = network.choose([(decision, encoded)])[0]
                    target = interest_target(first.probabilities, second.probabilities)[:len(encoded.request_indices)]
                    interest = first.reply_interest[:len(encoded.request_indices)]
                    order = np.argsort(-interest)
                    decisions += 1
                    total += float(target.sum())
                    for picks in PICKS:
                        part = float(target[order[:picks]].sum())
                        covered[picks] += part
                        if target.sum() > 1e-6:
                            per_decision[picks].append(part / float(target.sum()))
                    respond(event.game_id, decision, second)
                elif isinstance(event, Decision):
                    recorder, seat = games[event.game_id]
                    tracker = recorder.on_decision(event)
                    encoded = encoder.encode(event, tracker)
                    if event.who == seat and event.replies_on_demand:
                        first = network.choose([(event, encoded)])[0]
                        waiting[event.game_id] = (event, tracker, first)
                        pool.request_replies(event, list(encoded.request_indices))
                        continue
                    policy = network if event.who == seat else opponent
                    respond(event.game_id, event, policy.choose([(event, encoded)])[0])
            print(f"{decisions} decisions checked", flush=True)
    summary = {"decisions": decisions,
               "covered share of the total change": {picks: covered[picks] / max(total, 1e-9) for picks in PICKS},
               "median per-decision coverage": {picks: float(np.median(values)) if values else None for picks, values in per_decision.items()}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
