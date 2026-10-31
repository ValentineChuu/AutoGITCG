"""Print what the game data says about a character: a reference sheet for adding a character to the engine by hand.

Some NPC opponents (Serious Showdown bosses, tavern and weekly guests) use characters that Piovium's card data
doesn't have. Those are written by hand in GTS in the engine (engine/reference/custom_data/), and this script prints
what that needs, from Dimbreath's game tables: stats, tags, each skill (id, name, cost, description) and, recursively,
every status, summon and card the descriptions mention, with their placeholders resolved to names. It also points at a
same-name character Piovium already has, whose source is the best template.

Usage: python scripts/data/describe_character.py <character id> [<character id> ...]
Numbers written as $[D__KEY__...] are not in the text; look them up on the wiki or in Piovium's twin.
"""

import re
import sys
from pathlib import Path

from autogitcg.data.dimbreath_tables import english, load_table
from autogitcg.data.piovium_catalog import characters_by_id, entities_by_id

ROOT = Path(__file__).resolve().parents[2]
PIOVIUM_DATA_SOURCE = ROOT / "engine" / "reference" / "piovium" / "packages" / "data" / "src"
COST_NAMES = {"GCG_COST_DICE_VOID": "Unaligned", "GCG_COST_DICE_SAME": "Matching", "GCG_COST_ENERGY": "Energy",
              "GCG_COST_LEGEND": "Legend"}


def clean(text: str | None) -> str:
    return re.sub(r"<[^>]*>|\{SPRITE_PRESET#\d+\}", "", text or "").replace("\\n", "\n").strip()


class GameText:
    def __init__(self):
        self.characters = {row["id"]: row for row in load_table("GCGCharExcelConfigData")}
        self.cards = {row["id"]: row for row in load_table("GCGCardExcelConfigData")}
        self.skills = {row["id"]: row for row in load_table("GCGSkillExcelConfigData")}
        self.keywords = {row["id"]: clean(english(row.get("titleTextMapHash"))) for row in load_table("GCGKeywordExcelConfigData")}

    def name(self, table: dict, identifier: int) -> str:
        return clean(english(table.get(identifier, {}).get("nameTextMapHash"))) or "?"

    def resolve(self, text: str) -> str:
        """Replace reference placeholders with readable names, keeping the id for cards and statuses."""
        text = re.sub(r"\$\[C(\d+)\]", lambda match: f"[{self.name(self.cards, int(match[1]))} #{match[1]}]", text)
        text = re.sub(r"\$\[A(\d+)\]", lambda match: f"[{self.name(self.characters, int(match[1]))} #{match[1]}]", text)
        text = re.sub(r"\$\[S(\d+)\]", lambda match: f"[skill {self.name(self.skills, int(match[1]))} #{match[1]}]", text)
        return re.sub(r"\$\[K(\d+)\]", lambda match: self.keywords.get(int(match[1]), f"keyword {match[1]}"), text)


def cost_text(cost_list: list[dict]) -> str:
    parts = []
    for entry in cost_list:
        if entry.get("count", 0) > 0:
            cost_type = entry["costType"]
            parts.append(f"{entry['count']} {COST_NAMES.get(cost_type, cost_type.removeprefix('GCG_COST_DICE_').title())}")
    return ", ".join(parts) or "free"


def twin_source(name: str, text: GameText) -> str | None:
    implemented = characters_by_id()
    for character_id in implemented:
        if text.name(text.characters, character_id) == name:
            # Prefer current definitions over old_versions/ (balance-change history).
            for path in sorted(PIOVIUM_DATA_SOURCE.rglob("*.gts"), key=lambda path: "old_versions" in path.parts):
                if path.relative_to(PIOVIUM_DATA_SOURCE).parts[0] != "autogitcg" and f"id {character_id} as " in path.read_text(encoding="utf-8"):
                    return f"{path.relative_to(ROOT)} (implemented character {character_id})"
    return None


def describe(character_id: int, text: GameText) -> None:
    row = text.characters[character_id]
    name = text.name(text.characters, character_id)
    tags = [tag.removeprefix("GCG_TAG_").lower() for tag in row.get("tagList", []) if tag != "GCG_TAG_NONE"]
    print(f"\n### {name} ({character_id})")
    print(f"HP {row.get('hp', 0)}, max energy {row.get('maxEnergy', 0)}, tags: {', '.join(tags)}")
    print(f"Implemented same-name character to use as a template: {twin_source(name, text) or 'none'}")

    known_entities = entities_by_id()
    referenced: list[int] = []
    for skill_id in row.get("skillList", []):
        if not skill_id:
            continue
        skill = text.skills.get(skill_id, {})
        skill_tags = [tag.removeprefix("GCG_TAG_").lower() for tag in skill.get("skillTagList", []) if tag != "GCG_TAG_NONE"]
        description = clean(english(skill.get("descTextMapHash")))
        print(f"\n- skill {skill_id} {text.name(text.skills, skill_id)} [{', '.join(skill_tags)}] cost: {cost_text(skill.get('costList', []))}")
        print("  " + (text.resolve(description) or "(no description: hidden/internal skill)").replace("\n", "\n  "))
        referenced += [int(found) for found in re.findall(r"\$\[C(\d+)\]", description)]

    seen = set()
    while referenced:
        card_id = referenced.pop(0)
        if card_id in seen or card_id not in text.cards:
            continue
        seen.add(card_id)
        card = text.cards[card_id]
        status = "already in Piovium" if card_id in known_entities else "NEW, must be defined"
        description = clean(english(card.get("descTextMapHash")))
        print(f"\n- {card.get('cardType', '').removeprefix('GCG_CARD_').lower()} {card_id} {text.name(text.cards, card_id)} ({status})")
        print("  " + (text.resolve(description) or "(no description)").replace("\n", "\n  "))
        referenced += [int(found) for found in re.findall(r"\$\[C(\d+)\]", description)]


if __name__ == "__main__":
    game_text = GameText()
    for argument in sys.argv[1:]:
        describe(int(argument), game_text)
