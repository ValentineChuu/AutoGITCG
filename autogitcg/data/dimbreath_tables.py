"""Read Dimbreath AnimeGameData tables and English text from data/raw/."""

import json
from functools import cache
from pathlib import Path

RAW_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "raw"


@cache
def load_table(name: str) -> list[dict]:
    """Load an ExcelBinOutput table, e.g. load_table("GCGCardExcelConfigData")."""
    return json.loads((RAW_DIRECTORY / f"{name}.json").read_text(encoding="utf-8"))


@cache
def _english_text() -> dict[str, str]:
    text = json.loads((RAW_DIRECTORY / "TextMap_MediumEN.json").read_text(encoding="utf-8"))
    text.update(json.loads((RAW_DIRECTORY / "TextMapEN.json").read_text(encoding="utf-8")))
    return text


def english(text_hash: int) -> str | None:
    return _english_text().get(str(text_hash))
