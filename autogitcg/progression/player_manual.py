"""TCG Player's Manual tasks (wiki: TCG Player's Manual), the only source of a few non-shop cards.

Chapters unlock in order. Tasks the simulator can't observe (Dynamic Skins, character proficiency)
are approximated by matches played.
"""

from collections.abc import Callable
from dataclasses import dataclass

from autogitcg.progression.account import Account
from autogitcg.progression.card_names import card_id
from autogitcg.progression.rewards import add_coins
from autogitcg.progression.stage import INVITATION_SERIOUS, TAVERN_FRIENDLY, TAVERN_SERIOUS, WEEKLY_GUEST


@dataclass(frozen=True)
class ManualTask:
    key: str
    condition: Callable[[Account], bool]
    coins: int = 0
    cards: tuple[tuple[str, int], ...] = ()


def _total_cards(account: Account) -> int:
    return sum(account.cards.values())


def _characters(account: Account) -> int:
    return len(account.owned_characters())


def _objectives(account: Account) -> int:
    return len(account.completed_objectives)


CHAPTERS: list[tuple[str, list[ManualTask], int]] = [
    ("chapter 1", [
        ManualTask("buy a letter", lambda a: sum(a.letters_bought.values()) >= 1, cards=(("The Bestest Travel Companion!", 1),)),
        ManualTask("20 objectives", lambda a: _objectives(a) >= 20, coins=500),
        ManualTask("1 guest challenge", lambda a: a.cleared_stages[WEEKLY_GUEST] >= 1, coins=500),
        ManualTask("2 valid decks", lambda a: True, coins=500),
        ManualTask("buy 5 cards", lambda a: sum(a.shop_bought.values()) >= 5, cards=(("Mondstadt Hash Brown", 1),)),
        ManualTask("3000 coins earned", lambda a: a.lucky_coins_earned >= 3000, cards=(("Mondstadt Hash Brown", 1),)),
    ], 500),
    ("chapter 2", [
        ManualTask("70 cards", lambda a: _total_cards(a) >= 70, cards=(("Paimon", 1),)),
        ManualTask("10 characters", lambda a: _characters(a) >= 10, cards=(("Toss-Up", 1),)),
        ManualTask("40 objectives", lambda a: _objectives(a) >= 40, coins=700),
        ManualTask("3 guest challenges", lambda a: a.cleared_stages[WEEKLY_GUEST] >= 3, coins=700),
        ManualTask("10000 coins earned", lambda a: a.lucky_coins_earned >= 10_000, cards=(("Toss-Up", 1),)),
        ManualTask("player level 3", lambda a: a.level >= 3, coins=700),
    ], 1000),
    ("chapter 3", [
        ManualTask("140 cards", lambda a: _total_cards(a) >= 140, coins=1000),
        ManualTask("15 characters", lambda a: _characters(a) >= 15, coins=1000),
        ManualTask("60 objectives", lambda a: _objectives(a) >= 60, coins=1000),
        ManualTask("dynamic skin (approximated)", lambda a: a.matches_played >= 40, coins=1000),
        ManualTask("proficiency (approximated)", lambda a: a.matches_played >= 60, coins=1000),
        ManualTask("3 tavern friendly fracas", lambda a: a.cleared_stages[TAVERN_FRIENDLY] >= 3, coins=1000),
        ManualTask("25000 coins earned", lambda a: a.lucky_coins_earned >= 25_000, coins=1000),
    ], 1500),
    ("chapter 4", [
        ManualTask("220 cards", lambda a: _total_cards(a) >= 220, coins=1000),
        ManualTask("25 characters", lambda a: _characters(a) >= 25, coins=1000),
        ManualTask("100 objectives", lambda a: _objectives(a) >= 100, coins=1000),
        ManualTask("3 dynamic skins (approximated)", lambda a: a.matches_played >= 100, coins=1000),
        ManualTask("proficiency on 10 (approximated)", lambda a: a.matches_played >= 120, coins=1000),
        ManualTask("15 serious showdowns", lambda a: a.cleared_stages[INVITATION_SERIOUS] >= 15, coins=1000),
        ManualTask("5 tavern serious showdowns", lambda a: a.cleared_stages[TAVERN_SERIOUS] >= 5, coins=1000),
        ManualTask("50000 coins earned", lambda a: a.lucky_coins_earned >= 50_000, coins=1000),
    ], 2000),
]


def claim_manual_rewards(account: Account) -> None:
    """Claim every finished task of the current chapter, and the chapter reward when all are done."""
    for chapter, tasks, completion_coins in CHAPTERS:
        for task in tasks:
            key = f"{chapter}: {task.key}"
            if key not in account.manual_tasks_done and task.condition(account):
                account.manual_tasks_done.add(key)
                add_coins(account, task.coins)
                for name, count in task.cards:
                    account.cards[card_id(name)] += count
        completion_key = f"{chapter}: complete"
        if completion_key in account.manual_tasks_done:
            continue
        if all(f"{chapter}: {task.key}" in account.manual_tasks_done for task in tasks):
            account.manual_tasks_done.add(completion_key)
            add_coins(account, completion_coins)
        else:
            return  # later chapters unlock after this one is complete
