"""Teach Chudledee to build like people: imitation of real tournament decks (the training split; decks from before
6.4's rules weigh 1/6) and playtesters' decks, starting from an existing builder so it keeps its card knowledge.
Held-out check: the locked-test tournament decks, never imitated.

Usage:
    python scripts/deck_builder/pretrain_builder_on_human_decks.py --initialize-from models/chudledee-v3.4.pt \
        --output runs/chudledee_human_imitation --epochs 8
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch loads

import numpy as np
import torch

from autogitcg.deck_builder.builder_setup import BuilderSetup, load_builder_checkpoint, save_builder_checkpoint
from autogitcg.deck_builder_training.human_example_decks import held_out_human_decks, human_example_decks
from autogitcg.deck_builder_training.human_imitation import HumanImitation
from autogitcg.deck_builder_training.imitation_update import imitation_pass
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.collection_pool import simulate_collections

TOKEN_BUDGET = 65536
HELD_OUT_DECKS = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="directory for builder.pt and metrics.jsonl")
    parser.add_argument("--initialize-from", type=Path, required=True, help="builder checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--decks-per-epoch", type=int, default=3000, help="sampled by weight, with repeats")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, 16, 8, arguments.seed)
    setup = BuilderSetup()
    network = setup.network_from(load_builder_checkpoint(arguments.initialize_from)).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=arguments.learning_rate)
    update_rng = np.random.default_rng(arguments.seed)
    imitation = HumanImitation(human_example_decks(), collections, stages, setup.encoder, rng)
    held_out_steps = imitation.steps_for(rng.sample(held_out_human_decks(), HELD_OUT_DECKS))

    arguments.output.mkdir(parents=True, exist_ok=True)
    with (arguments.output / "metrics.jsonl").open("w") as metrics_file:
        before = imitation_pass(network, held_out_steps, TOKEN_BUDGET, device)
        metrics_file.write(json.dumps({"epoch": 0, "held_out": before}) + "\n")
        print(f"start     | held-out loss {before['loss']:.3f} top-1 in deck {before['top1_in_deck']:.3f}", flush=True)
        for epoch in range(1, arguments.epochs + 1):
            training = imitation_pass(network, imitation.steps(arguments.decks_per_epoch), TOKEN_BUDGET, device, optimizer, update_rng)
            held_out = imitation_pass(network, held_out_steps, TOKEN_BUDGET, device)
            metrics_file.write(json.dumps({"epoch": epoch, "training": training, "held_out": held_out}) + "\n")
            print(f"epoch {epoch:3d} | training loss {training['loss']:.3f} top-1 in deck {training['top1_in_deck']:.3f} "
                  f"| held-out loss {held_out['loss']:.3f} top-1 in deck {held_out['top1_in_deck']:.3f}", flush=True)
            save_builder_checkpoint(arguments.output / "builder.pt", network, None, {"update": 0, "decks_built": 0},
                                    {"human_imitation_epochs": epoch, "initialized_from": str(arguments.initialize_from)})


if __name__ == "__main__":
    main()
