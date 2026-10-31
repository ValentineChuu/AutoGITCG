"""Item ids used by GITCG rewards and the shop, and the mapping from card items to card ids."""

from functools import cache

from autogitcg.data.dimbreath_tables import load_table

PLAYER_EXP = 146
LUCKY_COIN = 107021
INVITATION_LETTER = 107020


@cache
def card_id_by_item() -> dict[int, int]:
    """Reward/shop item id -> character or action card id."""
    return {row["itemID"]: row["id"] for row in load_table("GCGDeckCardExcelConfigData") if row.get("itemID")}
