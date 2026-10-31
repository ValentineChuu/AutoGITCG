"""Cross-check Dimbreath game data against the Piovium engine catalog.

Writes data/derived/data_audit.md. Requires scripts/data/fetch_dimbreath.py and engine/reference/tools/export_catalog.mjs first.
"""

from collections import Counter
from pathlib import Path

from autogitcg.data.dimbreath_tables import english, load_table
from autogitcg.data.piovium_catalog import characters_by_id, entities_by_id

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "derived" / "data_audit.md"
DECK_CARD_TYPES = {"GCG_CARD_EVENT", "GCG_CARD_MODIFY", "GCG_CARD_ASSIST"}
# Piovium's DiceRequirementType numbering, keyed to Dimbreath cost type names.
COST_TYPE_TO_PIOVIUM = {
    "GCG_COST_DICE_VOID": "0", "GCG_COST_DICE_CRYO": "1", "GCG_COST_DICE_HYDRO": "2",
    "GCG_COST_DICE_PYRO": "3", "GCG_COST_DICE_ELECTRO": "4", "GCG_COST_DICE_ANEMO": "5",
    "GCG_COST_DICE_GEO": "6", "GCG_COST_DICE_DENDRO": "7", "GCG_COST_DICE_SAME": "8",
    "GCG_COST_ENERGY": "9", "GCG_COST_LEGEND": "10",
}


def dimbreath_cost(cost_list: list[dict]) -> dict[str, int]:
    cost = {}
    for entry in cost_list:
        # Empty slots omit both costType and count.
        if entry.get("count", 0) > 0:
            key = COST_TYPE_TO_PIOVIUM.get(entry["costType"], entry["costType"])
            cost[key] = cost.get(key, 0) + entry["count"]
    return cost


def piovium_cost(cost: dict | None) -> dict[str, int]:
    return {key: value for key, value in (cost or {}).items() if value > 0}


# Newer Dimbreath exports omit fields whose value is false/0/empty, so named fields are read with .get().
def label(identifier: int, name_hash: int | None) -> str:
    return f"{identifier} {english(name_hash) or '?'}" if name_hash else str(identifier)


def audit_characters(lines: list[str]) -> None:
    dimbreath = {row["id"]: row for row in load_table("GCGCharExcelConfigData")}
    obtainable = {identifier for identifier, row in dimbreath.items() if row.get("isCanObtain", False)}
    piovium = characters_by_id()
    skill_rows = {row["id"]: row for row in load_table("GCGSkillExcelConfigData")}

    lines.append("## Characters\n")
    lines.append(f"- Dimbreath: {len(dimbreath)} rows, {len(obtainable)} obtainable. Piovium: {len(piovium)}.")
    missing = sorted(obtainable - piovium.keys())
    lines.append(f"- Obtainable in Dimbreath but missing from Piovium ({len(missing)}): "
                 + ", ".join(label(i, dimbreath[i].get("nameTextMapHash")) for i in missing))
    extra = sorted(piovium.keys() - obtainable)
    lines.append(f"- In Piovium but not obtainable in Dimbreath ({len(extra)}): "
                 + ", ".join(label(i, dimbreath[i].get("nameTextMapHash")) if i in dimbreath else f"{i} (no Dimbreath row)" for i in extra))

    stat_mismatches, skill_mismatches = [], []
    for identifier in sorted(obtainable & piovium.keys()):
        row, character = dimbreath[identifier], piovium[identifier]
        if (row.get("hp", 0), row.get("maxEnergy", 0)) != (character["variables"]["maxHealth"], character["variables"]["maxEnergy"]):
            stat_mismatches.append(f"{label(identifier, row.get('nameTextMapHash'))}: Dimbreath hp/energy {row.get('hp', 0)}/{row.get('maxEnergy', 0)}, "
                                   f"Piovium {character['variables']['maxHealth']}/{character['variables']['maxEnergy']}")
        for skill in character["skills"]:
            skill_row = skill_rows.get(skill["id"])
            if skill_row is None:
                skill_mismatches.append(f"{label(identifier, row.get('nameTextMapHash'))}: skill {skill['id']} has no Dimbreath row")
            elif dimbreath_cost(skill_row.get("costList", [])) != piovium_cost(skill["cost"]):
                skill_mismatches.append(f"{label(identifier, row.get('nameTextMapHash'))}: skill {skill['id']} cost "
                                        f"Dimbreath {dimbreath_cost(skill_row.get('costList', []))} vs Piovium {piovium_cost(skill['cost'])}")
    lines.append(f"- HP/energy mismatches ({len(stat_mismatches)}):")
    lines.extend(f"  - {entry}" for entry in stat_mismatches)
    lines.append(f"- Skill cost mismatches ({len(skill_mismatches)}):")
    lines.extend(f"  - {entry}" for entry in skill_mismatches)
    lines.append("")


def audit_action_cards(lines: list[str]) -> None:
    dimbreath = {row["id"]: row for row in load_table("GCGCardExcelConfigData")}
    obtainable = {i for i, row in dimbreath.items() if row.get("isCanObtain", False) and row.get("cardType") in DECK_CARD_TYPES}
    piovium = entities_by_id()
    piovium_deck_cards = {i for i, entity in piovium.items() if entity["isDeckCard"] and entity["obtainable"]}

    lines.append("## Action cards\n")
    lines.append(f"- Dimbreath obtainable deck cards: {len(obtainable)}. Piovium obtainable deck cards: {len(piovium_deck_cards)}.")
    missing = sorted(obtainable - piovium.keys())
    lines.append(f"- Obtainable in Dimbreath but missing from Piovium ({len(missing)}): "
                 + ", ".join(label(i, dimbreath[i].get("nameTextMapHash")) for i in missing))
    extra = sorted(piovium_deck_cards - obtainable)
    lines.append(f"- Obtainable in Piovium but not in Dimbreath ({len(extra)}): "
                 + ", ".join(label(i, dimbreath[i].get("nameTextMapHash")) if i in dimbreath else f"{i} (no Dimbreath row)" for i in extra))

    cost_mismatches = []
    for identifier in sorted(obtainable & piovium.keys()):
        expected, actual = dimbreath_cost(dimbreath[identifier].get("costList", [])), piovium_cost(piovium[identifier]["playCost"])
        if expected != actual:
            cost_mismatches.append(f"{label(identifier, dimbreath[identifier].get('nameTextMapHash'))}: Dimbreath {expected} vs Piovium {actual}")
    lines.append(f"- Play cost mismatches ({len(cost_mismatches)}):")
    lines.extend(f"  - {entry}" for entry in cost_mismatches)
    lines.append("")


def audit_npc_decks(lines: list[str]) -> None:
    piovium_characters, piovium_entities = characters_by_id(), entities_by_id()
    character_rows = {row["id"]: row for row in load_table("GCGCharExcelConfigData")}
    card_rows = {row["id"]: row for row in load_table("GCGCardExcelConfigData")}
    decks = load_table("GCGDeckExcelConfigData")
    shape = Counter((len(deck.get("characterList", [])), len(deck.get("cardList", []))) for deck in decks)

    unsupported_decks, unknown_ids = 0, Counter()
    for deck in decks:
        unknown = [i for i in deck.get("characterList", []) if i not in piovium_characters]
        unknown += [i for i in deck.get("cardList", []) if i not in piovium_entities]
        if unknown:
            unsupported_decks += 1
            unknown_ids.update(set(unknown))

    lines.append("## NPC decks (GCGDeckExcelConfigData)\n")
    lines.append(f"- {len(decks)} decks. Shapes (characters, cards): {dict(shape.most_common(8))}")
    lines.append(f"- Decks referencing ids Piovium lacks: {unsupported_decks}")
    lines.append(f"- Most common unknown ids (id, name, decks affected):")
    for identifier, count in unknown_ids.most_common(30):
        row = character_rows.get(identifier) or card_rows.get(identifier)
        lines.append(f"  - {label(identifier, row.get('nameTextMapHash') if row else None)}: {count}")
    lines.append("")


SOURCE_PATTERNS = [
    ("starter deck", "initial deck"),
    ("character invitation", "inviting"),
    ("tavern challenge", "Tavern Challenge"),
    ("player manual", "Player Manual"),
]


def acquisition_source(row: dict) -> str:
    access_text = english(row.get("accessDescTextMapHash")) or ""
    for source, pattern in SOURCE_PATTERNS:
        if pattern in access_text:
            return source
    if row.get("shopGoodId"):
        return "shop"
    return f"other: {access_text[:80]}" if access_text else "unknown"


def audit_collection(lines: list[str]) -> None:
    piovium_characters, piovium_entities = characters_by_id(), entities_by_id()
    rows = load_table("GCGDeckCardExcelConfigData")
    unknown = [row["id"] for row in rows if row["id"] not in piovium_characters and row["id"] not in piovium_entities]
    goods = {good["goodsId"]: good for good in load_table("ShopGoodsExcelConfigData") if "goodsId" in good}
    prices = Counter(
        (good["costItems"][0].get("count"), good.get("buyLimit"))
        for row in rows if (good := goods.get(row.get("shopGoodId")))
    )
    sources = Counter(("character" if row["id"] < 10000 else "action card", acquisition_source(row)) for row in rows)

    lines.append("## Collection and acquisition (GCGDeckCardExcelConfigData)\n")
    lines.append(f"- {len(rows)} collectible cards. Not in Piovium: {len(unknown)} {unknown[:20]}")
    lines.append("- Acquisition sources:")
    lines.extend(f"  - {kind}, {source}: {count}" for (kind, source), count in sorted(sources.items()))
    lines.append(f"- Shop prices in Lucky Coins as (price, buy limit): {dict(prices.most_common())}")
    lines.append("")


def main() -> None:
    lines = ["# Data audit: Dimbreath vs Piovium\n"]
    audit_characters(lines)
    audit_action_cards(lines)
    audit_npc_decks(lines)
    audit_collection(lines)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
