"""A deck's reward: the mean game reward from its own seat over its judged games (failed games are left out).

Same per-game reward as combat training (training/game_result.py): a win is worth 1 - 0.02 x rounds, a loss -1.
"""

from dataclasses import dataclass

from autogitcg.bridge.messages import GameFinished
from autogitcg.deck_builder_training.deck_games import ScheduledGame
from autogitcg.deck_builder_training.game_outcome import GameOutcome
from autogitcg.training.game_result import final_reward


@dataclass
class DeckResult:
    reward: float | None  # None when every game failed
    wins: int
    games: int  # finished games
    failures: int
    mean_rounds: float


def deck_result(games: list[ScheduledGame], outcomes: list[GameOutcome]) -> DeckResult:
    rewards, wins, rounds, failures = [], 0, 0, 0
    for game, outcome in zip(games, outcomes):
        if outcome.failure is not None:
            failures += 1
            continue
        rewards.append(final_reward(GameFinished(0, outcome.winner, outcome.rounds), game.deck_seat))
        wins += outcome.winner == game.deck_seat
        rounds += outcome.rounds
    finished = len(rewards)
    return DeckResult(sum(rewards) / finished if finished else None, wins, finished, failures, rounds / max(finished, 1))
