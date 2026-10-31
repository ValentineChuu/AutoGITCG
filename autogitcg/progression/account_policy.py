"""How a simulated account spends its coins and picks its next stage (scripted, randomized by Personality)."""

import random
from functools import cache

from autogitcg.data.dimbreath_tables import load_table
from autogitcg.progression.account import Account
from autogitcg.progression.economy import LUCKY_COIN_CAP, MAX_COPIES_PER_CARD
from autogitcg.progression.personality import Personality
from autogitcg.progression.shop import Shop
from autogitcg.progression.stage import ASCENSION, INVITATION_FRIENDLY, Stage

COIN_PRESSURE = LUCKY_COIN_CAP - 5000  # above this, shop down so objective rewards stay claimable
TARGET_BONUS = 6.0
CAPPED_ASCENSION_BONUS = 20.0


@cache
def _card_types() -> dict[int, str]:
    return {row["id"]: row.get("cardType", "") for row in load_table("GCGCardExcelConfigData")}


class AccountPolicy:
    def __init__(self, personality: Personality, shop: Shop, rng: random.Random):
        self.personality = personality
        self.shop = shop
        self.rng = rng

    def go_shopping(self, account: Account, wants_letter: bool) -> None:
        if wants_letter:
            self.shop.buy_letter(account)
        budget_floor = min(self.personality.coins_kept, COIN_PRESSURE)
        card_types = _card_types()
        while account.lucky_coins > budget_floor:
            candidates = [
                card for card, offer in self.shop.cards.items()
                if account.cards[card] < MAX_COPIES_PER_CARD and self.shop.card_available(account, card)
            ]
            if not candidates:
                return
            weights = [self.personality.card_type_weights.get(card_types.get(card, ""), 1.0) for card in candidates]
            card = self.rng.choices(candidates, weights)[0]
            if account.lucky_coins - self.shop.cards[card].price < budget_floor and account.lucky_coins < COIN_PRESSURE:
                return  # keep the reserve unless coins are close to the cap
            self.shop.buy_card(account, card)

    def wants_letter(self, account: Account, letter_invitations: list[Stage]) -> bool:
        """Buy a letter when out of letters and some invitation the account could play is still unopened."""
        return account.letters == 0 and any(
            stage.group not in account.invited and stage.required_player_level <= account.level for stage in letter_invitations
        )

    def choose_stage(self, account: Account, available: list[Stage]) -> Stage:
        exp_capped = account.level == account.level_cap and account.level < 10
        weights = []
        for stage in available:
            weight = self.personality.kind_weights.get(stage.kind, 1.0)
            if stage.kind == INVITATION_FRIENDLY and stage.group in self.personality.target_characters:
                weight *= TARGET_BONUS
            if stage.kind == ASCENSION and exp_capped:
                weight *= CAPPED_ASCENSION_BONUS
            weights.append(weight)
        return self.rng.choices(available, weights)[0]
