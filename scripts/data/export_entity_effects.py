"""What each entity does, read from the card code (the entity facts in the catalog): for every summon, status, combat status
and card defined in Piovium's GTS sources and engine/reference/custom_data, its main damage element and amount (the summon's
displayed `hint` when it has one, else the first `:damage(...)`; "computed" when the amount isn't a plain number) and
its healing. The network gets these as catalog facts (observation/entity_effects.py), so it can judge what an entity on
the board is worth, and the reaction danger of its element.

    python scripts/data/export_entity_effects.py   # -> data/derived/entity_effects.json
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "engine" / "reference" / "piovium" / "packages" / "data" / "src", ROOT / "engine" / "reference" / "custom_data"]
OUTPUT = ROOT / "data" / "derived" / "entity_effects.json"
SKIPPED_KINDS = {"skill", "character"}
BLOCK = re.compile(r"define (\w+) \{(.*?)\n\};", re.S)
HINT = re.compile(r"hint DamageType\.(\w+),\s*([^;]+);")
DAMAGE = re.compile(r":damage\(\s*DamageType\.(\w+),\s*([^,)]+)")
HEAL = re.compile(r":heal\(\s*([^,)]+)")
VARIABLE = re.compile(r"\bvariable (\w+),\s*(\d+)")


def amount(text: str, variables: dict[str, int]) -> int | None:
    """A plain number, or an entity variable's starting value (e.g. Sesshou Sakura's `atk`); None if computed."""
    text = text.strip()
    if text.isdigit():
        return int(text)
    named = [value for name, value in variables.items() if re.search(rf"\b{name}\b", text)]
    return named[0] if len(named) == 1 else None


def effects_of(body: str) -> dict:
    hint, damage, heal = HINT.search(body), DAMAGE.search(body), HEAL.search(body)
    variables = {name: int(value) for name, value in VARIABLE.findall(body)}
    effect = {}
    if hint and hint.group(1) == "Heal":
        effect["heal"] = amount(hint.group(2), variables)
    elif hint:
        effect["element"], effect["damage"] = hint.group(1), amount(hint.group(2), variables)
    elif damage:
        effect["element"], effect["damage"] = damage.group(1), amount(damage.group(2), variables)
    if heal and "heal" not in effect:
        effect["heal"] = amount(heal.group(1), variables)
    return effect


def main() -> None:
    effects = {}
    for source in SOURCES:
        for path in sorted(source.rglob("*.gts")):
            if "old_versions" in path.parts:
                continue
            for kind, body in BLOCK.findall(path.read_text(encoding="utf-8")):
                found = re.search(r"\bid (\d+)", body)
                if kind in SKIPPED_KINDS or not found:
                    continue
                effect = effects_of(body)
                if effect:
                    effects[found.group(1)] = effect
    OUTPUT.write_text(json.dumps(effects, indent=1, sort_keys=True))
    damaging = [effect for effect in effects.values() if "element" in effect]
    print(f"{len(effects)} entities with an effect: {len(damaging)} deal damage "
          f"({sum(effect['damage'] is None for effect in damaging)} computed), "
          f"{sum('heal' in effect for effect in effects.values())} heal -> {OUTPUT}")


if __name__ == "__main__":
    main()
