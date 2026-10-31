"""Why does a network hold its bursts? (The replies network eta 7.3 used 53% of its ready bursts, beta 10.8 68%.)
Blind games vs plain beta 10.8 on the locked test matchups, no search. For the network's decisions with a burst ready: the
probability it gave the burst, what it did instead, the round, and its active character's HP. For its characters that
died with full energy: whether they were active (able to burst) at their last decision before dying.

    python experiments/held_bursts/explain_held_bursts.py --network models/eta-7.3.pt --reply-checks 4 --matchups 25 --output runs/burst_use/u150_why.json
"""

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.held_bursts.burst_option import is_burst
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.deck_guessing.tournament_deck_guesses import TournamentDeckGuesses
from autogitcg.model.combat_network_file import load_combat_network
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.policies.network_policy import NetworkPolicy
from autogitcg.policies.reply_selection import ReplySelection
from autogitcg.search.lookahead_settings import LookaheadSettings
from autogitcg.search.lookahead_table import LookaheadTable
from autogitcg.training.benchmark_matchups import TOURNAMENT_BENCHMARK_FILE, load_benchmark_matchups
from autogitcg.training.matchup_sampler import swapped

NO_SEARCH = LookaheadSettings(confident_probability=0.0)
PROBABILITY_BANDS = [(0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--reply-checks", type=int, default=0)
    parser.add_argument("--matchups", type=int, default=25)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    matchups = [dataclasses.replace(matchup, opponent_deck_known=(False, False))
                for matchup in load_benchmark_matchups(TOURNAMENT_BENCHMARK_FILE)[:arguments.matchups]]
    device = torch.device("cpu")
    encoder = ObservationEncoder()
    network = load_combat_network(arguments.network, encoder, device)
    opponent_network = load_combat_network(arguments.opponent, encoder, device)
    deck_guesses = TournamentDeckGuesses(NO_SEARCH.redeals, arguments.seed, excluded=[deck for matchup in matchups for deck in matchup.decks])
    ready, instead, by_round, by_band, deaths = Counter(), Counter(), Counter(), Counter(), Counter()
    held_by_character, used_by_character, charged_deaths_by_character = Counter(), Counter(), Counter()
    burst_probabilities = []
    last_seen = defaultdict(dict)  # game id -> character id -> (full energy, was active) at its owner's last decision

    def on_searcher_move(decision, encoded, own_choice, chosen_index, search) -> None:
        if decision.kind != "action":
            return
        actions = decision.request["action"]
        bursts = [token for token, index in enumerate(encoded.request_indices)
                  if is_burst(actions[index]) and actions[index].get("validity", 0) == 0]
        if not bursts:
            return
        probability = float(np.sum(own_choice.probabilities[bursts]))
        chosen = actions[encoded.request_indices[chosen_index]]
        used = is_burst(chosen)
        player = decision.state["player"][decision.who]
        active = next(character for character in player["character"] if character["id"] == player["activeCharacterId"])
        ready["decisions"] += 1
        ready["used"] += used
        burst_probabilities.append(probability)
        band = next(band for band in PROBABILITY_BANDS if band[0] <= probability < band[1])
        by_band[f"{band} decisions"] += 1
        by_band[f"{band} used"] += used
        round_number = min(decision.state["roundNumber"], 6)
        by_round[f"round {round_number} decisions"] += 1
        by_round[f"round {round_number} used"] += used
        if used:
            used_by_character[active["definitionId"]] += 1
        elif not chosen.get("isFast"):
            held_by_character[active["definitionId"]] += 1
        if not used:
            kind = chosen["action"]["$case"]
            instead[kind + (" (fast)" if chosen.get("isFast") else "")] += 1
            instead[f"active HP {'<=3' if active['health'] <= 3 else '4-6' if active['health'] <= 6 else '7+'}"] += 1

    def on_real_move(row, decision, response_value, search) -> None:
        for seat, player in enumerate(decision.state["player"]):
            if seat != row["searcher_seat"]:
                continue
            for character in player["character"]:
                seen = last_seen[decision.game_id]
                if character["defeated"] or character["health"] == 0:
                    if character["id"] in seen:
                        full, was_active = seen.pop(character["id"])
                        deaths["defeated"] += 1
                        if full:
                            deaths["died charged"] += 1
                            charged_deaths_by_character[character["definitionId"]] += 1
                            deaths["died charged while active" if was_active else "died charged on the bench"] += 1
                elif decision.who == seat:
                    seen[character["id"]] = (character["maxEnergy"] > 0 and character["energy"] >= character["maxEnergy"],
                                             character["id"] == player["activeCharacterId"])

    with EnginePool(arguments.workers, previews=network.config.reads_previews, replies=network.config.reads_replies) as pool:
        torch.manual_seed(arguments.seed)
        table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                               NetworkPolicy(opponent_network, device), NO_SEARCH, arguments.seed, deck_guesses,
                               on_searcher_move=on_searcher_move, on_real_move=on_real_move,
                               reply_selection=ReplySelection(arguments.reply_checks) if arguments.reply_checks else None)
        games = 0
        for number, original in enumerate(matchups):
            for matchup in (original, swapped(original)):
                for seat in (0, 1):
                    table.start_game(matchup, arguments.seed + number, seat, {"matchup": number})
                    games += 1
        finished = 0
        while table.game_count:
            finished += len(table.step())
            print(f"{finished}/{games} games", flush=True)
    summary = {"ready": dict(ready), "used share": ready["used"] / max(ready["decisions"], 1),
               "mean burst probability": float(np.mean(burst_probabilities)) if burst_probabilities else None,
               "by burst probability": dict(by_band), "by round": dict(by_round), "instead": dict(instead.most_common()),
               "deaths": dict(deaths),
               "by character (active when ready)": {str(definition): {"held": held_by_character[definition], "used": used_by_character[definition],
                                                                     "died charged": charged_deaths_by_character[definition]}
                                                    for definition in sorted(set(held_by_character) | set(used_by_character) | set(charged_deaths_by_character),
                                                                             key=lambda each: -held_by_character[each] - charged_deaths_by_character[each])}}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"network": str(arguments.network), "games": games, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
