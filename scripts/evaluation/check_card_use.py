"""How Chudledum uses action cards with human-built decks vs Chudledee's decks: it plays both sides of blind PvP games,
a locked-test tournament deck (never seen in training) against a Chudledee PvP deck. Humans in tournament videos play
about 4.4 cards per round.
    python scripts/evaluation/check_card_use.py --combat models/beta-10.8.pt --games 300 --output runs/card_use/beta-10.8.json
"""

import argparse
import collections
import json
import random
from pathlib import Path

import torch

from autogitcg.analysis.builder_deck_check.check_players import NetworkPlayer
from autogitcg.analysis.card_use.card_use_games import play_card_use_games
from autogitcg.analysis.card_use.card_use_stats import summary
from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.data.tournament_test_set import load_test_decks
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters, deck_problems
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.training.collection_pool import Collection

DECKS_PER_SOURCE = 40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combat", type=Path, default=Path("models/beta-10.8.pt"))
    parser.add_argument("--chudledee-side-combat", type=Path,
                        help="a different network for the Chudledee decks' side (e.g. beta 10.8), to test the --combat network's own play")
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v3.4.pt"))
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    tournament = [entry.deck for entry in load_test_decks() if not deck_problems(entry.deck)][:DECKS_PER_SOURCE]
    setup = BuilderSetup()
    device = torch.device("cpu")
    builder = BuilderPolicy(setup.network_from(load_builder_checkpoint(arguments.builder)).to(device), setup.encoder, device)
    everything = Collection(tuple(sorted(buildable_characters())), collections.Counter({card: 2 for card in buildable_action_cards()}), 0)
    chudledee = [built.deck for built in builder.build([BuildingTask(everything, PVP_MODE, None, "everything")] * DECKS_PER_SOURCE,
                                                       generator=torch.Generator().manual_seed(arguments.seed))]
    rng = random.Random(arguments.seed)
    pairings = []
    for _ in range(arguments.games):
        pairing = [(rng.choice(tournament), "tournament"), (rng.choice(chudledee), "chudledee")]
        pairings.append(tuple(pairing if rng.random() < 0.5 else reversed(pairing)))

    encoder = ObservationEncoder()
    finished = 0

    def on_finished() -> None:
        nonlocal finished
        finished += 1

    with EnginePool(arguments.workers) as pool:
        player = NetworkPlayer(arguments.combat, encoder)
        chudledee_player = NetworkPlayer(arguments.chudledee_side_combat, encoder) if arguments.chudledee_side_combat else player
        stats = play_card_use_games(pairings, {"tournament": player, "chudledee": chudledee_player}, encoder, pool, arguments.seed, on_finished)
    report = {label: summary(counts) for label, counts in stats.items()}
    for label, numbers in report.items():
        print(label, {name: round(value, 3) for name, value in numbers.items()})
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps({"combat": str(arguments.combat), "chudledee_side_combat": str(arguments.chudledee_side_combat or arguments.combat), "games": arguments.games, "seed": arguments.seed,
                                                "by_deck_source": report}, indent=1))


if __name__ == "__main__":
    main()
