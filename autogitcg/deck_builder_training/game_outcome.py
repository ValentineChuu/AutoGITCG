"""The result of one judged game, as sent back from a game-player process."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GameOutcome:
    job: int  # position in the list of games handed to MatchPlayerGroup.play
    winner: int | None = None  # None: round limit (a loss for both) or failure
    rounds: int = 0
    failure: str | None = None  # first line of the engine error, when the game failed
