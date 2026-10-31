"""Build the pool of Chudledee decks used in Chudledum's training (data/training/chudledee_training_decks.json).

Picks are sampled, not the most likely, so the pool shows the variety of decks Chudledee considers good
instead of repeating its single favourite lineup.

Usage:
    python scripts/deck_builder/build_chudledee_training_decks.py --builder models/chudledee-v1.6.pt
    # PvP decks from the full card pool (as people build), stage decks kept from the current file:
    python scripts/deck_builder/build_chudledee_training_decks.py --builder runs/chudledee_human_imitation/builder.pt --full-pool-decks 3000 --keep-stage-decks
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from autogitcg.deck_builder.builder_policy import BuilderPolicy
from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint
from autogitcg.deck_builder.building_task import PVP_MODE, STAGE_MODE, BuildingTask
from autogitcg.deck_building.deck_validation import buildable_action_cards, buildable_characters, deck_problems
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.chudledee_training_decks import (CHUDLEDEE_DECKS_FILE, ChudledeeTrainingDecks, load_chudledee_training_decks,
                                                         save_chudledee_training_decks)
from autogitcg.training.collection_pool import Collection, simulate_collections

TASKS_PER_CHUNK = 256  # every pick is kept in memory until its chunk is built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder", type=Path, default=Path("models/chudledee-v1.6.pt"))
    parser.add_argument("--accounts", type=int, default=150)
    parser.add_argument("--snapshots-per-account", type=int, default=4)
    parser.add_argument("--decks-per-stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--full-pool-decks", type=int, default=0, help="build this many PvP decks from every card (2 copies) instead of simulated collections")
    parser.add_argument("--keep-stage-decks", action="store_true", help="reuse the current file's stage decks instead of building new ones")
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    setup = BuilderSetup()
    builder = BuilderPolicy(setup.network_from(load_builder_checkpoint(arguments.builder)).to(device), setup.encoder, device)
    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, arguments.accounts, arguments.snapshots_per_account, arguments.seed)
    if arguments.full_pool_decks:
        full_pool = Collection(tuple(sorted(buildable_characters())), Counter({card: 2 for card in buildable_action_cards()}), 0)
        tasks = [BuildingTask(full_pool, PVP_MODE, None, "full_pool") for _ in range(arguments.full_pool_decks)]
    else:
        tasks = [BuildingTask(collection, PVP_MODE, None, f"collection:{index}") for index, collection in enumerate(collections)]
    kept = load_chudledee_training_decks() if arguments.keep_stage_decks else None
    if not kept:
        tasks += [BuildingTask(rng.choice(collections), STAGE_MODE, stage, "stage") for stage in stages for _ in range(arguments.decks_per_stage)]
    print(f"{len(collections)} collections, {len(stages)} stages: building {len(tasks)} decks", flush=True)

    decks = ChudledeeTrainingDecks(stage=kept.stage if kept else {})
    for start in range(0, len(tasks), TASKS_PER_CHUNK):
        for built in builder.build(tasks[start:start + TASKS_PER_CHUNK], generator):
            assert not deck_problems(built.deck), deck_problems(built.deck)
            if built.task.mode == PVP_MODE:
                decks.pvp.append(built.deck)
            else:
                decks.stage.setdefault(built.task.stage.level_id, []).append(built.deck)
        print(f"  {min(start + TASKS_PER_CHUNK, len(tasks))}/{len(tasks)}", flush=True)
    builder_name = str(arguments.builder)
    if kept:
        builder_name = f"pvp: {arguments.builder}; stage decks kept from: {json.loads(CHUDLEDEE_DECKS_FILE.read_text())['builder']}"
    save_chudledee_training_decks(decks, builder_name)
    lineups = {tuple(sorted(deck.characters)) for deck in decks.pvp}
    print(f"{len(decks.pvp)} PvP decks ({len(lineups)} distinct lineups), stage decks for {len(decks.stage)} stages")


if __name__ == "__main__":
    main()
