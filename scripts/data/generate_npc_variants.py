"""Generate GTS definitions for NPC-only characters that are stat variants of an implemented character.

A variant has the same skill ids as a "twin" that is already implemented (by Piovium or by one of our
hand-written kits) and differs only in HP or energy. Its definition reuses the twin's skills and tags.
Writes engine/reference/custom_data/variants/ (fully regenerated). Requires engine/reference/piovium (run
`npm run build:data` in engine/ once) and data/raw.
"""

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from autogitcg.data.dimbreath_tables import english, load_table

ROOT = Path(__file__).resolve().parents[2]
PIOVIUM_DATA_SOURCE = ROOT / "engine" / "reference" / "piovium" / "packages" / "data" / "src"
CUSTOM_DATA = ROOT / "engine" / "reference" / "custom_data"
OUTPUT_DIRECTORY = CUSTOM_DATA / "variants"
OUTPUT_VIRTUAL_DIRECTORY = "autogitcg/variants"  # where variant files end up inside the data package's src/
CHARACTER_BLOCK = re.compile(r"define character \{(?P<body>.*?)\n\};", re.S)
IMPORT_STATEMENT = re.compile(r'import\s*\{(?P<names>[^}]*)\}\s*from\s*"(?P<module>[^"]+)"')
DEFINED_NAME = re.compile(r"\bid \d+ as (\w+);")


@dataclass
class SourceFile:
    virtual_path: str  # path inside the data package's src/, e.g. "characters/pyro/x.gts" or "autogitcg/tavern_a/y.gts"
    text: str

    def defined_names(self) -> set[str]:
        return set(DEFINED_NAME.findall(self.text))

    def imported_modules(self) -> dict[str, str]:
        """Imported name -> virtual path of the module it comes from (relative imports only)."""
        modules = {}
        for match in IMPORT_STATEMENT.finditer(self.text):
            if not match["module"].startswith("."):
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(self.virtual_path), match["module"]))
            for name in match["names"].split(","):
                name = name.replace("type ", "").strip()
                if name:
                    modules[name] = target
        return modules


def stage_character_ids() -> set[int]:
    """Characters used by Tavern Challenge, Weekly Guest, open-world and Ascension Challenge opponents."""
    games = {row["id"]: row for row in load_table("GCGGameExcelConfigData")}
    decks = {row["id"]: row for row in load_table("GCGDeckExcelConfigData")}
    level_ids = [row.get(key) for row in load_table("GCGBossLevelExcelConfigData") for key in ("normalLevelId", "hardLevelId")]
    level_ids += [level.get("levelId") for row in load_table("GCGWeekLevelExcelConfigData") for level in row.get("levelCondList", [])]
    level_ids += [row["levelId"] for row in load_table("GCGWorldLevelExcelConfigData")]
    level_ids += [row["levelId"] for row in load_table("GCGQuestLevelExcelConfigData")]
    return {
        character
        for level_id in level_ids if level_id in games
        for character in decks.get(games[level_id].get("enemyCardGroupId"), {}).get("characterList", [])
    }


def source_files() -> list[SourceFile]:
    """Piovium's own definitions plus our hand-written kits (generated variants are excluded)."""
    files = [
        SourceFile(path.relative_to(PIOVIUM_DATA_SOURCE).as_posix(), path.read_text(encoding="utf-8"))
        for path in PIOVIUM_DATA_SOURCE.rglob("*.gts")
        if path.relative_to(PIOVIUM_DATA_SOURCE).parts[0] != "autogitcg"  # our own synced copies
    ]
    files += [
        SourceFile("autogitcg/" + path.relative_to(CUSTOM_DATA).as_posix(), path.read_text(encoding="utf-8"))
        for path in CUSTOM_DATA.rglob("*.gts")
        if path.relative_to(CUSTOM_DATA).parts[0] != OUTPUT_DIRECTORY.name
    ]
    return files


def character_blocks(files: list[SourceFile]) -> dict[int, tuple[SourceFile, str]]:
    """Character id -> (file defining it, body of its `define character` block)."""
    blocks = {}
    for source in files:
        for match in CHARACTER_BLOCK.finditer(source.text):
            id_match = re.search(r"\bid (\d+) as ", match["body"])
            if id_match:
                blocks[int(id_match[1])] = (source, match["body"])
    return blocks


def block_field(body: str, name: str) -> str:
    """A field of a `define character` block; fields may span several lines."""
    return " ".join(re.search(rf"^\s*{name}\s+(.*?);", body, re.M | re.S)[1].split())


def variant_source(variant: dict, twin_file: SourceFile, twin_body: str, name: str) -> str:
    tags, skills = block_field(twin_body, "tags"), block_field(twin_body, "skills")
    defined_here, imported = twin_file.defined_names(), twin_file.imported_modules()
    names_by_module: dict[str, list[str]] = {}
    for skill in dict.fromkeys(skill.strip() for skill in skills.split(",")):
        module = twin_file.virtual_path if skill in defined_here else imported.get(skill)
        if module is None:
            raise ValueError(f"{name}: cannot find where skill {skill} is defined")
        names_by_module.setdefault(module, []).append(skill)
    imports = "".join(
        f'import {{ {", ".join(names)} }} from "{posixpath.relpath(module, OUTPUT_VIRTUAL_DIRECTORY)}";\n'
        for module, names in names_by_module.items()
    )
    return (
        f"// Generated by scripts/data/generate_npc_variants.py. Do not edit by hand.\n"
        f"// {name} ({variant['id']}): stat variant of the implemented character in {twin_file.virtual_path}.\n"
        f"{imports}\n"
        f"define character {{\n"
        f"  id {variant['id']} as Npc{variant['id']};\n"
        f"  tags {tags};\n"
        f"  health {variant.get('hp', 0)};\n"
        f"  energy {variant.get('maxEnergy', 0)};\n"
        f"  skills {skills};\n"
        f"}};\n"
    )


def main() -> None:
    rows = {row["id"]: row for row in load_table("GCGCharExcelConfigData")}
    blocks = character_blocks(source_files())
    twins_by_skills = {
        tuple(sorted(skill for skill in rows[character].get("skillList", []) if skill)): character
        for character in blocks if character in rows
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for old_file in OUTPUT_DIRECTORY.glob("*.gts"):
        old_file.unlink()
    generated = 0
    for character in sorted(stage_character_ids() - blocks.keys()):
        row = rows.get(character)
        twin = twins_by_skills.get(tuple(sorted(skill for skill in (row or {}).get("skillList", []) if skill)))
        if row is None or twin is None:
            continue
        twin_file, twin_body = blocks[twin]
        name = english(row.get("nameTextMapHash")) or str(character)
        (OUTPUT_DIRECTORY / f"npc_{character}.gts").write_text(variant_source(row, twin_file, twin_body, name), encoding="utf-8")
        generated += 1
    print(f"generated {generated} variant characters into {OUTPUT_DIRECTORY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
