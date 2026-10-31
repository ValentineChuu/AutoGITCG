"""Look up card ids by their English name (for the handful of cards the wiki names directly)."""

from functools import cache

from autogitcg.data.dimbreath_tables import english, load_table


@cache
def card_ids_by_name() -> dict[str, int]:
    ids = {}
    for table in ("GCGCharExcelConfigData", "GCGCardExcelConfigData"):
        for row in load_table(table):
            name = english(row.get("nameTextMapHash"))
            if name and row.get("isCanObtain") and name not in ids:
                ids[name] = row["id"]
    return ids


def card_id(name: str) -> int:
    return card_ids_by_name()[name]
