"""Save one network from your own training run as a small model file: the network and its config, without the
optimizer state and past-opponent snapshots a checkpoint carries. Use it on a run you trained yourself; the published
models were made the same way, for every milestone at once, by export_model_release.py.

The file loads like a checkpoint (load_checkpoint + network_config_from), so every script that takes a model file
takes it.

Usage:
    python scripts/training/freeze_combat_network.py --checkpoint runs/my_run/checkpoint_latest.pt --output models/my-run.pt
    python scripts/training/freeze_combat_network.py --checkpoint runs/my_run/checkpoint_latest.pt \
        --weights runs/my_run/weights/update_00180.pt --output models/my-run-update-180.pt   # weights saved between checkpoints
"""

import argparse
from pathlib import Path

import torch

from autogitcg.training.checkpoints import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, help="a run's weights/update_NNNNN.pt to use instead of the checkpoint's network")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    checkpoint = load_checkpoint(arguments.checkpoint)
    if arguments.weights:
        checkpoint["network"] = torch.load(arguments.weights, map_location="cpu")
        checkpoint["progress"] = {**checkpoint["progress"], "update": int(arguments.weights.stem.split("_")[-1])}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"network_config": checkpoint["network_config"], "network": checkpoint["network"],
                "progress": checkpoint["progress"], "frozen_from": str(arguments.weights or arguments.checkpoint)}, arguments.output)
    print(f"{arguments.output}: update {checkpoint['progress']['update']} of {arguments.checkpoint}")


if __name__ == "__main__":
    main()
