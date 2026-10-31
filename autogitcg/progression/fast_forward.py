"""A rough stand-in for playing a match, to fast-forward accounts without the engine or a trained bot.

Only for generating realistic collection states (e.g. to bootstrap training); once the bot plays the
progression for real, its actual results replace this.
"""

import random

from autogitcg.progression.account import Account
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.stage import Stage

BASE_WIN_RATE = 0.6
WIN_RATE_PER_LEVEL = 0.06  # per player level above the stage's requirement
WIN_RATE_PER_CHARACTER = 0.005  # a bigger collection gives better decks


def estimated_result(account: Account, stage: Stage, rng: random.Random) -> MatchResult:
    win_rate = (BASE_WIN_RATE + WIN_RATE_PER_LEVEL * (account.level - stage.required_player_level)
                + WIN_RATE_PER_CHARACTER * len(account.owned_characters()))
    won = rng.random() < min(0.95, max(0.15, win_rate))
    return MatchResult(won=won, rounds=rng.randint(4, 10), my_defeated=rng.choice([0, 0, 1, 1, 2, 3 if not won else 2]),
                       bursts_used=rng.randint(0, 4), cards_played=())
