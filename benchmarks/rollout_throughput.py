"""Rollout speed of the actor processes alone (untrained network, no learner updates).

Usage:
    python -m benchmarks.rollout_throughput --actors 4 --workers-per-actor 3 --games-per-actor 64 --seconds 60
"""

import argparse
import random
import subprocess
import time

import torch

from autogitcg.model.combat_network import CombatNetwork
from autogitcg.model.network_config import network_config_for
from autogitcg.observation.observation_encoder import ObservationEncoder
from autogitcg.progression.stage_catalog import build_stage_catalog
from autogitcg.training.actor_group import ActorGroup
from autogitcg.training.actor_process import ActorSettings
from autogitcg.training.collection_pool import simulate_collections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--workers-per-actor", type=int, default=3)
    parser.add_argument("--games-per-actor", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=60)
    arguments = parser.parse_args()

    encoder = ObservationEncoder()
    stages, _ = build_stage_catalog()
    collections = simulate_collections(stages, 8, 8, seed=0)
    config = network_config_for(encoder)
    network = CombatNetwork(config, encoder.catalog_features.table)
    template = ActorSettings(0, arguments.workers_per_actor, arguments.games_per_actor, 512, 0.95, config, seed=random.randrange(1000))
    with ActorGroup(arguments.actors, template, collections, steps_per_update=8192) as actors:
        actors.broadcast(0, network.state_dict(), lambda: None)
        actors.gather(1)  # wait until every actor is up and playing
        started = time.perf_counter()
        steps = decisions = 0
        while time.perf_counter() - started < arguments.seconds:
            chunk, stats, _ = actors.gather(1)
            steps += len(chunk)
            decisions += stats.decisions
        seconds = time.perf_counter() - started
        memory = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip().replace("\n", "; ")
    print(f"{arguments.actors} actors x {arguments.workers_per_actor} workers, {arguments.games_per_actor} games each: "
          f"{decisions / seconds:.0f} decisions/s, {steps / seconds:.0f} learner steps/s")
    print(f"GPU memory per process: {memory}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
