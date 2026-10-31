"""Progression rules and numbers from the Genshin Impact fandom wiki (read 2026-10-13, game version 7.1).

Pages used (https://genshin-impact.fandom.com/wiki/...): Player_EXP, Lucky_Coin, Genius_Invokation_TCG/Card_Shop,
Genius_Invokation_TCG/Invitational, Match_Invitation_Letter, Sennight_Gauntlet, Ascension_Challenge,
Genius_Invokation_TCG/Stage (and its 1,045 stage pages, for rewards and requirements). Numbers the wiki doesn't
state are marked as inferred where they are set.
"""

# Cumulative Player EXP needed for levels 1..10.
LEVEL_EXP = [0, 300, 1050, 2000, 3300, 4750, 6350, 8100, 10000, 12450]
MAX_LEVEL = 10
FIRST_LEVEL_CAP = 2  # Ascension Challenges raise the cap to 4, 6, 9 and 10

# Level rewards we model: 1 Match Invitation Letter at each of levels 1-9, 500 Lucky Coins at level 1.
LETTER_REWARD_LEVELS = range(1, 10)
LEVEL_ONE_COINS = 500

LUCKY_COIN_CAP = 20_000  # objective rewards can't be claimed while at the cap; overflow below it is kept
MAX_LETTERS_HELD = 10
MAX_COPIES_PER_CARD = 2  # deck rule: 3 characters + 30 action cards, at most 2 copies of a card
DECK_ACTION_CARDS = 30
DECK_CHARACTERS = 3

# Weekly Guest Challenges / Sennight Gauntlet.
WEEKLY_UNLOCK_LEVEL = 2
GUESTS_PER_WEEK = 4
WEEKLY_WINS_REQUIRED = 4  # not stated on the wiki; inferred from 800 EXP = 4 x the old 200 EXP per guest
WEEKLY_EXP = 800  # levels 2-9; 0 at level 10


def weekly_lucky_coins(level: int) -> int:
    if level >= 10:
        return 6000
    if level >= 7:
        return 4800
    if level >= 5:
        return 3600
    return 2400


def level_for_exp(exp: int) -> int:
    return max(level for level, needed in enumerate(LEVEL_EXP, start=1) if exp >= needed)
