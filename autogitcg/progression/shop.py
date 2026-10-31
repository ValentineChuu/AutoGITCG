"""The Card Shop: action cards with a price and lifetime stock, and Match Invitation Letters by level."""

from dataclasses import dataclass

from autogitcg.data.dimbreath_tables import load_table
from autogitcg.progression.account import Account
from autogitcg.progression.economy import MAX_LETTERS_HELD
from autogitcg.progression.items import INVITATION_LETTER, LUCKY_COIN

UNLIMITED = 10**9


@dataclass(frozen=True)
class CardOffer:
    card_id: int
    price: int
    stock: int  # lifetime purchase limit


@dataclass(frozen=True)
class LetterOffer:
    goods_id: int
    price: int
    stock: int
    required_level: int


def _price(goods: dict) -> int:
    return next((cost.get("count", 0) for cost in goods.get("costItems", []) if cost.get("id") == LUCKY_COIN), 0)


class Shop:
    def __init__(self):
        goods_by_id = {row["goodsId"]: row for row in load_table("ShopGoodsExcelConfigData") if "goodsId" in row}
        self.cards = {}
        for row in load_table("GCGDeckCardExcelConfigData"):
            goods = goods_by_id.get(row.get("shopGoodId"))
            if goods and _price(goods):
                self.cards[row["id"]] = CardOffer(row["id"], _price(goods), goods.get("buyLimit") or UNLIMITED)
        self.letters = sorted(
            (LetterOffer(goods["goodsId"], _price(goods), goods.get("buyLimit") or UNLIMITED,
                         int(goods.get("preconditionParamList", ["0"])[0] or 0))
             for goods in goods_by_id.values() if goods.get("itemId") == INVITATION_LETTER and _price(goods)),
            key=lambda offer: offer.required_level,
        )

    def card_available(self, account: Account, card_id: int) -> bool:
        offer = self.cards.get(card_id)
        return offer is not None and account.shop_bought[card_id] < offer.stock and account.lucky_coins >= offer.price

    def buy_card(self, account: Account, card_id: int) -> bool:
        if not self.card_available(account, card_id):
            return False
        account.lucky_coins -= self.cards[card_id].price
        account.shop_bought[card_id] += 1
        account.cards[card_id] += 1
        return True

    def buy_letter(self, account: Account) -> bool:
        if account.letters >= MAX_LETTERS_HELD:
            return False
        for offer in self.letters:
            if (offer.required_level <= account.level and account.letters_bought[offer.goods_id] < offer.stock
                    and account.lucky_coins >= offer.price):
                account.lucky_coins -= offer.price
                account.letters_bought[offer.goods_id] += 1
                account.letters += 1
                return True
        return False
