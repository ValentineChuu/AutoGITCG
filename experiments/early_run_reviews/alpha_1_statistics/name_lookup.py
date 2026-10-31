"""English names for character and card ids, from the game database's `definitions` table."""

import sqlite3


class NameLookup:
    def __init__(self, connection: sqlite3.Connection):
        self.names: dict[tuple[str, int], str] = {
            (kind, definition_id): name for definition_id, kind, name in connection.execute("SELECT id, kind, name FROM definitions")}

    def character(self, character_id: int) -> str:
        return self.names.get(("character", int(character_id)), str(character_id))

    def card(self, card_id: int) -> str:
        return self.names.get(("card", int(card_id)), str(card_id))

    def lineup(self, characters) -> str:
        return ", ".join(self.character(character) for character in characters if character)
