"""English names for every character, card and skill id that can appear in game records (from Dimbreath)."""

from autogitcg.data.dimbreath_tables import english, load_table

# Text-map placeholders the game fills in at runtime (the Wanderer's name is chosen by the player).
PLACEHOLDER_NAMES = {"#{REALNAME[ID(1)|DELAYHANDLE(true)]}": "Wanderer"}
NAME_TABLES = {"character": "GCGCharExcelConfigData", "card": "GCGCardExcelConfigData", "skill": "GCGSkillExcelConfigData"}


def definition_names() -> list[tuple[int, str, str]]:
    """(id, kind, name) rows; ids without an English name are left out."""
    rows = []
    for kind, table in NAME_TABLES.items():
        for row in load_table(table):
            name = english(row.get("nameTextMapHash"))
            if name:
                rows.append((row["id"], kind, PLACEHOLDER_NAMES.get(name, name)))
    return rows
