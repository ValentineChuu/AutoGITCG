"""Applying rewards to an account: items, EXP with level-up rewards, and stage objectives."""

from autogitcg.progression.account import Account
from autogitcg.progression.economy import LETTER_REWARD_LEVELS, LUCKY_COIN_CAP
from autogitcg.progression.items import INVITATION_LETTER, LUCKY_COIN, PLAYER_EXP, card_id_by_item
from autogitcg.progression.match_result import MatchResult
from autogitcg.progression.objectives import objective_achieved
from autogitcg.progression.stage import Stage


def add_coins(account: Account, amount: int) -> None:
    account.lucky_coins += amount
    account.lucky_coins_earned += amount


def add_exp(account: Account, amount: int) -> None:
    before = account.level
    account.exp += amount
    grant_level_rewards(account, before, account.level)


def grant_level_rewards(account: Account, old_level: int, new_level: int) -> None:
    for level in range(old_level + 1, new_level + 1):
        if level in LETTER_REWARD_LEVELS:
            account.letters += 1


def add_items(account: Account, items: tuple[tuple[int, int], ...]) -> None:
    cards = card_id_by_item()
    for item_id, count in items:
        if item_id == LUCKY_COIN:
            add_coins(account, count)
        elif item_id == PLAYER_EXP:
            add_exp(account, count)
        elif item_id == INVITATION_LETTER:
            account.letters += count
        elif item_id in cards:
            account.cards[cards[item_id]] += count
        # Other items (Primogems, cosmetics) don't affect card progression.


def claim_objectives(account: Account, stage: Stage, result: MatchResult) -> list[int]:
    """Claim every achieved, unclaimed objective; returns the claimed objective indices.

    While the account is at the Lucky Coin cap, objectives that pay coins can't be claimed (they stay
    open). Below the cap, a reward that goes over it is kept in full."""
    claimed = []
    for index, objective in enumerate(stage.objectives):
        key = (stage.level_id, index)
        if key in account.completed_objectives or not objective_achieved(objective, result):
            continue
        pays_coins = any(item_id == LUCKY_COIN for item_id, _ in objective.rewards)
        if pays_coins and account.lucky_coins >= LUCKY_COIN_CAP:
            continue
        account.completed_objectives.add(key)
        add_items(account, objective.rewards)
        claimed.append(index)
    return claimed
