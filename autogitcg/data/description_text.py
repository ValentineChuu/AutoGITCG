"""Readable card text: Dimbreath's descriptions with their placeholders filled in and colour tags removed.

Placeholders: $[A<id>] character, $[C<id>] card, $[S<id>] skill, $[K<id>] keyword (Pyro DMG, Combat Action, ...).
"""

import re
from functools import cache

from autogitcg.data.dimbreath_tables import english, load_table

PLACEHOLDER = re.compile(r"\$\[([ACSK])(\d+)\]")
TAG = re.compile(r"</?color[^>]*>")
ICON = re.compile(r"\{SPRITE_PRESET#\d+\}")  # inline element icons; the element's name follows them
TABLES = {"A": ("GCGCharExcelConfigData", "nameTextMapHash"), "C": ("GCGCardExcelConfigData", "nameTextMapHash"),
          "S": ("GCGSkillExcelConfigData", "nameTextMapHash"), "K": ("GCGKeywordExcelConfigData", "titleTextMapHash")}
MAX_NESTING = 3  # keyword titles can contain placeholders themselves


@cache
def _names(kind: str) -> dict[int, str]:
    table, field = TABLES[kind]
    return {row["id"]: english(row.get(field)) or "" for row in load_table(table)}


def _name(match: re.Match) -> str:
    return _names(match.group(1)).get(int(match.group(2))) or match.group(0)


def readable_text(text: str) -> str:
    for _ in range(MAX_NESTING):
        if not PLACEHOLDER.search(text):
            break
        text = PLACEHOLDER.sub(_name, text)
    return ICON.sub("", TAG.sub("", text)).replace("\\n", "\n")
