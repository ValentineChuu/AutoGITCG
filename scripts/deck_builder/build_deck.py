"""Show the decks a builder checkpoint makes, with names: for simulated collections, against a stage or for PvP.

Usage:
    python scripts/deck_builder/build_deck.py --builder models/chudledee-v1.6.pt --collections 3
    python scripts/deck_builder/build_deck.py --builder runs/chudledee_imitation/builder.pt --stage 2201001 --sample
"""

import argparse
from collections import Counter
from pathlib import Path

import torch

from autogitcg.analysis.definition_names import definition_names
from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.collection_pool import simulate_collections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", type=Path, required=True, help="builder checkpoint")
    parser.add_argument("--stage", type=int, help="stage level id (default: PvP)")
    parser.add_argument("--collections", type=int, default=3, help="simulated collections to build for")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sample", action="store_true", help="sample picks instead of taking the most likely")
    arguments = parser.parse_args()

    setup = BuilderSetup()
    network = setup.network_from(load_builder_checkpoint(arguments.builder))
    stages, _ = build_stage_catalog()
    stage = next(stage for stage in stages if stage.level_id == arguments.stage) if arguments.stage else None
    collections = simulate_collections(stages, arguments.collections, 1, arguments.seed)
    tasks = [BuildingTask(collection, STAGE_MODE if stage else PVP_MODE, stage, f"collection:{index}")
             for index, collection in enumerate(collections)]
    names = {(kind, identifier): name for identifier, kind, name in definition_names()}
    if stage:
        print(f"opponent: {stage.name} ({stage.kind}): "
              + ", ".join(names.get(("character", character), str(character)) for character in stage.enemy_deck.characters))
    for built in BuilderPolicy(network, setup.encoder, torch.device("cpu"), greedy=not arguments.sample).build(tasks):
        collection = built.task.collection
        print(f"\n{built.task.collection_origin}: {len(collection.characters)} characters, "
              f"{sum(collection.action_cards.values())} cards owned, {collection.matches_played} matches played")
        print("  characters: " + ", ".join(names.get(("character", character), str(character)) for character in built.deck.characters))
        for card, copies in sorted(Counter(built.deck.cards).items()):
            print(f"  {copies} x {names.get(('card', card), str(card))}")


if __name__ == "__main__":
    main()
