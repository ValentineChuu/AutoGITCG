"""When an account's playthrough ends: it owns every buildable character, or it hit the match cap.

The simulator itself never ends (weekly guests return every week), so a stopping rule is needed.
"""

from autogitcg.deck_building.deck_validation import buildable_characters
from autogitcg.progression.account import Account


def playthrough_over(account: Account, max_matches: int) -> bool:
    return account.matches_played >= max_matches or len(account.owned_characters()) >= len(buildable_characters())
