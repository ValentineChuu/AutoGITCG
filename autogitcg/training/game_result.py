"""Final reward per player: a win is worth more the faster it came; every loss is -1.

Win: 1 - SPEED_PENALTY_PER_ROUND * rounds (about 0.84-0.94 for typical games), so one round faster is worth
about one percentage point of win chance. Loss, including reaching the round limit (a loss for both): -1.
Slow losses are deliberately not punished extra, so the bot never learns to lose quickly once behind;
giving up is the job of a separate concede rule.
"""

from autogitcg.bridge.messages import GameFinished

SPEED_PENALTY_PER_ROUND = 0.02


def won(finished: GameFinished, who: int) -> bool:
    return finished.winner == who


def final_reward(finished: GameFinished, who: int, speed_penalty: float = SPEED_PENALTY_PER_ROUND) -> float:
    """speed_penalty 0: a plain win/loss reward (training option; search and tests count wins only)."""
    return 1.0 - speed_penalty * finished.rounds if won(finished, who) else -1.0
