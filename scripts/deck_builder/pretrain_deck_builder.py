"""Warm-start the deck builder by imitating the synergy decks and the legal NPC decks, then save a checkpoint
that scripts/deck_builder/train_deck_builder.py can start from (--initialize-from).

A few synergy and NPC decks are held out to check that imitation generalizes instead of memorizing.

Usage:
    python scripts/deck_builder/pretrain_deck_builder.py --output runs/chudledee_imitation --epochs 20
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch loads

import numpy as np
import torch

from autogitcg.data.npc_decks import load_npc_decks
from autogitcg.deck_builder.builder_network import copy_card_knowledge
from autogitcg.deck_builder.builder_setup import BuilderSetup, save_builder_checkpoint
from autogitcg.deck_building.deck_validation import deck_problems
from autogitcg.deck_builder_training.builder_training_config import COMBAT_V1
from autogitcg.deck_builder_training.imitation_examples import example_decks, example_task
from autogitcg.deck_builder_training.imitation_steps import imitation_steps
from autogitcg.deck_builder_training.imitation_update import imitation_pass
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.checkpoints import load_checkpoint
from autogitcg.training.collection_pool import simulate_collections
from autogitcg.training.deck_pools import load_synergy_decks

HELD_OUT_SYNERGY = 6
HELD_OUT_NPC = 60
TOKEN_BUDGET = 65536


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="directory for builder.pt and metrics.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--combat-checkpoint", type=Path, default=COMBAT_V1)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, 16, 8, arguments.seed)
    synergy, _ = load_synergy_decks()
    npc = [deck for _, deck in sorted(load_npc_decks().items()) if not deck_problems(deck)]
    rng.shuffle(synergy)
    rng.shuffle(npc)
    training_decks = example_decks(synergy[HELD_OUT_SYNERGY:], npc[HELD_OUT_NPC:])
    held_out_decks = synergy[:HELD_OUT_SYNERGY] + npc[:HELD_OUT_NPC]

    setup = BuilderSetup()
    network = setup.new_network()
    copy_card_knowledge(network, load_checkpoint(arguments.combat_checkpoint)["network"])
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=arguments.learning_rate)
    update_rng = np.random.default_rng(arguments.seed)

    def steps_for(decks):
        return [step for deck in decks for step in imitation_steps(example_task(deck, collections, stages, rng), deck, setup.encoder, rng)]

    held_out_steps = steps_for(held_out_decks)
    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "metrics.jsonl").open("w") as metrics_file:
        for epoch in range(1, arguments.epochs + 1):
            training = imitation_pass(network, steps_for(training_decks), TOKEN_BUDGET, device, optimizer, update_rng)
            held_out = imitation_pass(network, held_out_steps, TOKEN_BUDGET, device)
            record = {"epoch": epoch, "training": training, "held_out": held_out}
            metrics_file.write(json.dumps(record) + "\n")
            print(f"epoch {epoch:3d} | training loss {training['loss']:.3f} top-1 in deck {training['top1_in_deck']:.3f} "
                  f"| held-out loss {held_out['loss']:.3f} top-1 in deck {held_out['top1_in_deck']:.3f}", flush=True)
    save_builder_checkpoint(arguments.output / "builder.pt", network, None, {"update": 0, "decks_built": 0},
                            {"imitation_epochs": arguments.epochs})


if __name__ == "__main__":
    main()
