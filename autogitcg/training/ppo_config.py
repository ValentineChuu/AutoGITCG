"""PPO hyperparameters. Defaults are common starting points, not tuned values."""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    steps_per_update: int = 32768  # learner decisions collected before each update
    epochs: int = 3
    minibatch_size: int = 512  # 1024 peaks at ~3.7 GB, too much next to the actors on an 8 GB card
    learning_rate: float = 1e-4  # alpha 1: 3e-4 let KL climb past 0.03 by update 60; at 2e-4 the KL stop cut half the updates
    final_learning_rate: float = 1e-4  # linear decay to this by the last update; equal to learning_rate = constant
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    win_coefficient: float = 0.25  # weight of the win-chance output's cross-entropy loss
    move_value_coefficient: float = 0.0  # weight of the move-value output's squared error to the final reward (played moves)
    future_coefficient: float = 0.0  # weight of the future-board prediction's squared error (training/future_targets.py)
    reply_interest_coefficient: float = 0.0  # weight of the reply selector's squared error (training/reply_interest_loss.py)
    reply_checks: int = 4  # actions the selector checks per decision (for the coverage metric; policies/reply_selection.py)
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03  # remaining epochs of an update are skipped once an epoch's mean KL exceeds this
    gae_lambda: float = 0.95  # discount is 1: games are short and only the final result counts
