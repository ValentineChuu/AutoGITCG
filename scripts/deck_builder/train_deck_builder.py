"""Train the deck builder against the frozen combat policy. Resumes if the run directory has a checkpoint.

Usage:
    python scripts/deck_builder/train_deck_builder.py --run runs/chudledee_v1 --updates 300
    python scripts/deck_builder/train_deck_builder.py --run runs/chudledee_v1 --initialize-from runs/chudledee_imitation/builder.pt
"""

import argparse
import os
import signal
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch loads; game players inherit it

from autogitcg.deck_builder_training.builder_trainer import train_builder
from autogitcg.deck_builder_training.builder_training_config import BuilderTrainingConfig


def main() -> None:
    defaults = BuilderTrainingConfig(Path("."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=defaults.updates)
    parser.add_argument("--combat-checkpoint", type=Path, default=defaults.combat_checkpoint)
    parser.add_argument("--initialize-from", type=Path, help="builder checkpoint to start a fresh run from")
    parser.add_argument("--decks-per-update", type=int, default=defaults.decks_per_update)
    parser.add_argument("--games-per-deck", type=int, default=defaults.games_per_deck)
    parser.add_argument("--game-processes", type=int, default=defaults.game_processes)
    parser.add_argument("--workers-per-process", type=int, default=defaults.workers_per_process)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--final-learning-rate", type=float, default=defaults.final_learning_rate)
    parser.add_argument("--entropy-coefficient", type=float, default=defaults.entropy_coefficient)
    parser.add_argument("--evaluate-every", type=int, default=defaults.evaluate_every)
    parser.add_argument("--evaluation-tasks-per-mode", type=int, default=defaults.evaluation_tasks_per_mode)
    parser.add_argument("--accounts", type=int, default=defaults.accounts)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--stage-share", type=float, default=defaults.stage_share, help="share of tasks that are NPC stages")
    parser.add_argument("--imitation-weight", type=float, default=defaults.imitation_weight,
                        help="imitation anchor on human decks after each update (e.g. 0.5); 0 = off")
    parser.add_argument("--imitation-decks-per-update", type=int, default=defaults.imitation_decks_per_update)
    arguments = parser.parse_args()
    # Plain `kill` stops like Ctrl+C; exit code 143 marks a deliberate stop (not restarted as a crash).
    signal.signal(signal.SIGTERM, lambda signal_number, frame: _stop_deliberately(signal_number))
    config = BuilderTrainingConfig(
        run_directory=arguments.run, updates=arguments.updates, combat_checkpoint=arguments.combat_checkpoint,
        initialize_from=arguments.initialize_from, decks_per_update=arguments.decks_per_update,
        games_per_deck=arguments.games_per_deck, game_processes=arguments.game_processes,
        workers_per_process=arguments.workers_per_process, learning_rate=arguments.learning_rate,
        final_learning_rate=arguments.final_learning_rate, entropy_coefficient=arguments.entropy_coefficient,
        evaluate_every=arguments.evaluate_every,
        evaluation_tasks_per_mode=arguments.evaluation_tasks_per_mode, accounts=arguments.accounts, seed=arguments.seed,
        stage_share=arguments.stage_share, imitation_weight=arguments.imitation_weight,
        imitation_decks_per_update=arguments.imitation_decks_per_update)
    train_builder(config)


def _stop_deliberately(signal_number: int) -> None:
    print(f"stopped by signal {signal_number}", file=sys.stderr, flush=True)
    raise SystemExit(128 + signal_number)


if __name__ == "__main__":
    main()
