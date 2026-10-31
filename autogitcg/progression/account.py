"""The state of one simulated player account."""

from collections import Counter
from dataclasses import dataclass, field

from autogitcg.progression.economy import FIRST_LEVEL_CAP, level_for_exp


@dataclass
class Account:
    cards: Counter = field(default_factory=Counter)  # card id -> copies owned (character cards: 1)
    lucky_coins: int = 0
    lucky_coins_earned: int = 0  # lifetime total, used by Player's Manual tasks
    exp: int = 0
    level_cap: int = FIRST_LEVEL_CAP
    letters: int = 0
    letters_bought: Counter = field(default_factory=Counter)  # letter shop good id -> bought
    shop_bought: Counter = field(default_factory=Counter)  # card id -> copies bought from the shop
    completed_objectives: set[tuple[int, int]] = field(default_factory=set)  # (level id, objective index)
    invited: set[int] = field(default_factory=set)  # invitation groups (characters) a letter was spent on
    cleared_stages: Counter = field(default_factory=Counter)  # stage kind -> stages won at least once
    won_levels: set[int] = field(default_factory=set)
    week: int = 0
    weekly_guests: list[int] = field(default_factory=list)  # guest groups visiting this week
    weekly_guests_beaten: set[int] = field(default_factory=set)
    weekly_wins: int = 0
    weekly_reward_claimed: bool = False
    manual_tasks_done: set[str] = field(default_factory=set)
    matches_played: int = 0
    matches_this_week: int = 0

    @property
    def level(self) -> int:
        return min(level_for_exp(self.exp), self.level_cap)

    def owned_characters(self) -> list[int]:
        return [card for card, copies in self.cards.items() if copies > 0 and card < 10000]

    def owned_action_cards(self) -> Counter:
        return Counter({card: copies for card, copies in self.cards.items() if copies > 0 and card >= 10000})
