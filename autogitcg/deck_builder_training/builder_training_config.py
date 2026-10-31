"""Deck builder training settings. Defaults are reasoned starting points, not tuned values."""

from dataclasses import dataclass
from pathlib import Path

COMBAT_V1 = Path("models/alpha-2.40.pt")  # frozen combat policy: alpha 2.40 (scripts/training/freeze_combat_network.py)


@dataclass
class BuilderTrainingConfig:
    run_directory: Path
    updates: int = 300
    combat_checkpoint: Path = COMBAT_V1
    initialize_from: Path | None = None  # a builder checkpoint (e.g. from imitation pretraining) to start from
    # Tasks and games
    decks_per_update: int = 256
    games_per_deck: int = 4  # half in each seat
    stage_share: float = 0.5  # of tasks; the rest are PvP
    accounts: int = 64
    snapshots_per_account: int = 8
    game_processes: int = 4
    workers_per_process: int = 3
    concurrent_games_per_process: int = 64
    # PPO
    epochs: int = 4
    training_token_budget: int = 65536  # padded tokens per minibatch (~100-160 picks; see token_budget_batches.py)
    # After imitation the policy is sharp: at 1e-4 one epoch moved it by KL ~0.12 (target 0.03); 3e-5 gave ~0.014.
    learning_rate: float = 3e-5
    final_learning_rate: float = 1e-5
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03
    # Imitation anchor: after each PPO update, one imitation step on human decks (human_example_decks.py) so winning
    # doesn't pull the decks away from how people build. 0 = off.
    imitation_weight: float = 0.0
    imitation_decks_per_update: int = 32
    # Network (width must match the combat network to copy its card knowledge)
    network_width: int = 128
    network_layers: int = 3
    # Benchmark and storage
    evaluate_every: int = 10
    evaluation_tasks_per_mode: int = 48
    evaluation_games_per_deck: int = 8
    keep_checkpoint_every: int = 50
    seed: int = 0
