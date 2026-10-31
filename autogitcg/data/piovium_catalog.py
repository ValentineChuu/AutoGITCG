"""Read the Piovium catalog exported by engine/reference/tools/export_catalog.mjs."""

import json
from functools import cache
from pathlib import Path

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "derived" / "piovium_catalog.json"


@cache
def load_catalog() -> dict[str, list[dict]]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def characters_by_id() -> dict[int, dict]:
    return {character["id"]: character for character in load_catalog()["characters"]}


def entities_by_id() -> dict[int, dict]:
    return {entity["id"]: entity for entity in load_catalog()["entities"]}


def attachment_ids() -> list[int]:
    return sorted(attachment["id"] for attachment in load_catalog()["attachments"])
