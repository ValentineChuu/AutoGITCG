"""Weekly Guest Challenges and the Sennight Gauntlet reward."""

import random

from autogitcg.progression.account import Account
from autogitcg.progression.economy import (
    GUESTS_PER_WEEK, LUCKY_COIN_CAP, MAX_LEVEL, WEEKLY_EXP, WEEKLY_UNLOCK_LEVEL, WEEKLY_WINS_REQUIRED, weekly_lucky_coins,
)
from autogitcg.progression.rewards import add_coins, add_exp
from autogitcg.progression.stage import Stage


def start_week(account: Account, guest_groups: list[int], rng: random.Random) -> None:
    """Advance to the next week: new guests arrive and the Sennight Gauntlet resets."""
    account.week += 1
    account.matches_this_week = 0
    account.weekly_guests = rng.sample(guest_groups, GUESTS_PER_WEEK) if account.level >= WEEKLY_UNLOCK_LEVEL else []
    account.weekly_guests_beaten = set()
    account.weekly_wins = 0
    account.weekly_reward_claimed = False


def guest_stage(variants: list[Stage], player_level: int) -> Stage:
    """Guests change decks with the player's level: use the highest variant the player has reached."""
    reached = [stage for stage in variants if stage.required_player_level <= player_level]
    return max(reached or variants, key=lambda stage: stage.required_player_level)


def record_guest_win(account: Account, group: int) -> None:
    if group in account.weekly_guests_beaten:
        return
    account.weekly_guests_beaten.add(group)
    account.weekly_wins += 1
    # Like objective rewards, the weekly reward can't be claimed while at the Lucky Coin cap.
    if account.weekly_wins >= WEEKLY_WINS_REQUIRED and not account.weekly_reward_claimed and account.lucky_coins < LUCKY_COIN_CAP:
        account.weekly_reward_claimed = True
        if account.level < MAX_LEVEL:
            add_exp(account, WEEKLY_EXP)
        add_coins(account, weekly_lucky_coins(account.level))
