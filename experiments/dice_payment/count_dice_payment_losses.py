"""Does the engine's automatic dice payment cost Chudledum skills? (The bot has always paid with the
engine's `autoSelectedDice`.) Blind games of a network against plain beta 10.8 on the locked tournament test matchups, no
search. At every card play and tuning: would another legal payment have kept more of the active character's skills
affordable (skills_lost.py)? Also checks the affordability rule against the engine's validity.

    python experiments/dice_payment/count_dice_payment_losses.py --network models/beta-10.8.pt --matchups 50 --output runs/dice_payment/beta 10.8.json
"""

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.dice_payment.affordability import can_pay
from experiments.dice_payment.skills_lost import skills_lost
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


def active_energy(player: dict) -> int:
    return next(character["energy"] for character in player["character"] if character["id"] == player["activeCharacterId"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--opponent", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--reply-checks", type=int, default=0)
    parser.add_argument("--matchups", type=int, default=50)
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
    counts, lost_by_kind, examples = Counter(), Counter(), []

    def on_real_move(row, decision, response_value, search) -> None:
        if decision.kind != "action":
            return
        player = decision.state["player"][decision.who]
        dice, energy, actions = player["dice"], active_energy(player), decision.request["action"]
        for action in actions:  # the affordability rule against the engine
            if action["action"]["$case"] == "useSkill":
                counts["skills checked"] += 1
                counts["engine valid, rule says unaffordable"] += action.get("validity", 0) == 0 and not can_pay(action["requiredCost"], dice, energy)
        chosen = actions[response_value["chosenActionIndex"]]
        case = chosen["action"]["$case"]
        if case not in ("playCard", "elementalTuning") or not response_value.get("usedDice"):
            return
        side = "network" if decision.who == row["searcher_seat"] else "opponent"
        counts[f"{side} {case} paying dice"] += 1
        lost = skills_lost(actions, chosen, response_value["usedDice"], dice, energy)
        if lost:
            counts[f"{side} {case} lost a skill"] += 1
            lost_by_kind[case] += 1
            if len(examples) < 20:
                examples.append({"case": case, "dice": dice, "paid": response_value["usedDice"], "cost": chosen["requiredCost"],
                                 "lost skills": sorted(lost)})

    with EnginePool(arguments.workers, previews=network.config.reads_previews, replies=network.config.reads_replies) as pool:
        torch.manual_seed(arguments.seed)
        table = LookaheadTable(pool, encoder, NetworkPolicy(network, device), NetworkPolicy(network, device, greedy=True),
                               NetworkPolicy(opponent_network, device), NO_SEARCH, arguments.seed, deck_guesses,
                               on_real_move=on_real_move,
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
    summary = dict(counts)
    for side in ("network", "opponent"):
        for case in ("playCard", "elementalTuning"):
            paid = counts[f"{side} {case} paying dice"]
            summary[f"{side} {case} share losing a skill"] = counts[f"{side} {case} lost a skill"] / max(paid, 1)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps({"network": str(arguments.network), "games": games, "summary": summary,
                                            "examples": examples}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
