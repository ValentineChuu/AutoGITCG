"""The account a player has right after the tutorial (wiki: Starting Deck, Come Try Genius Invokation TCG!)."""

from autogitcg.data.deck import Deck
from autogitcg.data.dimbreath_tables import english, load_table
from autogitcg.progression.account import Account
from autogitcg.progression.card_names import card_id
from autogitcg.progression.economy import LEVEL_ONE_COINS
from autogitcg.progression.rewards import add_coins, grant_level_rewards

STARTER_LINEUP = ["Diluc", "Kaeya", "Sucrose"]
TUTORIAL_EXTRA_CHARACTERS = ["Fischl"]
# The 30-card starting deck: the 28 cards Dimbreath marks "Obtained from the initial deck", plus these two.
STARTER_EXTRA_CARDS = ["Paimon", "The Bestest Travel Companion!"]


def starter_deck() -> Deck:
    initial = [
        row["id"] for row in load_table("GCGDeckCardExcelConfigData")
        if row["id"] >= 10000 and "initial deck" in (english(row.get("accessDescTextMapHash")) or "")
    ]
    cards = sorted(initial + [card_id(name) for name in STARTER_EXTRA_CARDS])
    return Deck(characters=[card_id(name) for name in STARTER_LINEUP], cards=cards)


def starter_account() -> Account:
    account = Account()
    deck = starter_deck()
    for card in deck.characters + [card_id(name) for name in TUTORIAL_EXTRA_CHARACTERS] + deck.cards:
        account.cards[card] += 1
    add_coins(account, LEVEL_ONE_COINS)
    grant_level_rewards(account, 0, 1)
    return account
