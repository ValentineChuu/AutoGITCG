"""Decode the game's deck share codes (the text players paste to share a deck).

A code is 51 base64 bytes: 50 obfuscated bytes plus a last byte added to each of them. Undoing that gives 33
twelve-bit share ids (3 characters, then 30 action cards), which Piovium's share_id.json maps to definition ids
(copied to data/derived by engine/reference/tools/export_catalog.mjs, so machines without the Piovium sources have it).
Ported from Piovium's assets-manager (sharing.ts, decodeRaw).
"""

import base64
import json
from functools import cache
from pathlib import Path

from autogitcg.data.deck import Deck

SHARE_ID_TABLE = Path(__file__).resolve().parents[2] / "data" / "derived" / "piovium_share_ids.json"
CODE_BYTES = 51
SHARE_IDS = 33


@cache
def _definition_ids() -> dict[int, int]:
    return {int(share_id): definition_id for share_id, definition_id in json.loads(SHARE_ID_TABLE.read_text()).items()}


def decode_share_code(code: str) -> Deck:
    data = list(base64.b64decode(code))
    if len(data) != CODE_BYTES:
        raise ValueError(f"a share code has {CODE_BYTES} bytes, this one has {len(data)}")
    offset = data.pop()
    half = len(data) // 2
    reordered = [(data[2 * index] - offset) & 0xFF for index in range(half)]
    reordered += [(data[2 * index + 1] - offset) & 0xFF for index in range(half)] + [0]
    share_ids = []
    for group in range(len(reordered) // 3):
        first, second, third = reordered[3 * group:3 * group + 3]
        share_ids += [(first << 4) + (second >> 4), ((second & 0xF) << 8) + third]
    definition_ids = [_definition_ids()[share_id] for share_id in share_ids[:SHARE_IDS]]
    return Deck(characters=definition_ids[:3], cards=sorted(definition_ids[3:]))
