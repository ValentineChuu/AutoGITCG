"""Compare what Chudledee's decks look like with what people build, as a table printed to the terminal.

Each given deck builder builds --decks PvP decks with every card available; people's decks are the tournament decks
from game version 6.4 on. Per column, averaged over its decks: how many different lineups, average card cost, share of
free cards, cards included at two copies, and how many arcane legends, transfigurations, resonances, talents, weapons,
artifacts and food cards a deck has (autogitcg/analysis/deck_shape/deck_shape.py). This showed, for example, that
Chudledee v3 puts more cards at two copies than v2.

    python scripts/deck_builder/describe_builder_decks.py models/chudledee-v2.4.pt models/chudledee-v3.4.pt
"""

import argparse
import collections
from pathlib import Path

import torch

from autogitcg.analysis.deck_shape.deck_shape import deck_shape
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, BuildingTask
from autogitcg.deck_builder_training.human_example_decks import human_example_decks
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters
from autogitcg.training.collection_pool import Collection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("builders", type=Path, nargs="+")
    parser.add_argument("--decks", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()
    everything = Collection(tuple(sorted(buildable_characters())), collections.Counter({card: 2 for card in buildable_action_cards()}), 0)
    setup = BuilderSetup()
    device = torch.device("cpu")
    shapes = {"people (6.4+)": deck_shape([deck for deck, weight in human_example_decks(include_testers=False) if weight >= 1])}
    for path in arguments.builders:
        policy = BuilderPolicy(setup.network_from(load_builder_checkpoint(path)).to(device), setup.encoder, device)
        built = policy.build([BuildingTask(everything, PVP_MODE, None, "everything")] * arguments.decks,
                             generator=torch.Generator().manual_seed(arguments.seed))
        shapes[str(path)] = deck_shape([deck.deck for deck in built])
    names = list(next(iter(shapes.values())))
    print(f"{'':28s}" + "".join(f"{label[-26:]:>28s}" for label in shapes))
    for name in names:
        print(f"{name:28s}" + "".join(f"{shape[name]:>28.2f}" for shape in shapes.values()))


if __name__ == "__main__":
    main()
