"""One player's decisions in one game, and their advantages once the result is known.

Reward is only the final result (training/game_result.py: a speed-adjusted win score, or -1), so with
discount 1 the return of every step is the final reward, and GAE blends that with the value estimates of
later steps. Each step also stores whether the game was won, the target of the separate win-chance output.
"""

from dataclasses import dataclass, field

import numpy as np

from autogitcg.training.future_targets import future_targets


@dataclass
class Step:
    observation: dict[str, np.ndarray]  # compacted encoder arrays
    kind: int
    choice_index: int
    choice_bits: np.ndarray
    log_prob: float
    value: float
    policy_version: int = 0  # learner weights version that made this decision
    advantage: float = 0.0
    value_target: float = 0.0
    won: float = 0.0
    policy_log_prob: float | None = None  # the policy's own; set when the behaviour (log_prob) mixed in exploration
    potential: float = 0.0  # position score at this decision (training/position_potential.py), for reward shaping
    result: float = 0.0  # the game's final reward for this player: the Monte-Carlo target of the move value
    bonus: float = 0.0  # immediate reward for this move (training/card_play_bonus.py)
    future: np.ndarray | None = None  # the board a few own decisions later (training/future_targets.py)
    future_mask: np.ndarray | None = None  # per horizon: 1 if the game lasted that long
    reply_target: np.ndarray | None = None  # fully checked decisions: each action's reply interest target (policies/reply_selection.py)


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)

    def finish(self, final_reward: float, won: bool, gae_lambda: float, shaping_weight: float = 0.0) -> list[Step]:
        """Fills advantage, value_target and won for every step; returns the steps.
        shaping_weight: each step also gets weight x (next step's potential - its own), with the potential counted as
        0 after the game ends, so the shaping sums to minus the first step's potential (policy-invariant)."""
        if self.steps and "character_features" in self.steps[0].observation:  # real observations (tests use stand-ins)
            for step, target, mask in zip(self.steps, *future_targets([step.observation for step in self.steps])):
                step.future, step.future_mask = target, mask
        advantage = 0.0
        next_value = 0.0
        next_potential = 0.0
        for index in reversed(range(len(self.steps))):
            step = self.steps[index]
            reward = final_reward if index == len(self.steps) - 1 else 0.0
            reward += shaping_weight * (next_potential - step.potential) + step.bonus
            next_potential = step.potential
            delta = reward + next_value - step.value
            advantage = delta + gae_lambda * advantage
            step.advantage = advantage
            step.value_target = advantage + step.value
            step.won = float(won)
            step.result = final_reward
            next_value = step.value
        return self.steps
