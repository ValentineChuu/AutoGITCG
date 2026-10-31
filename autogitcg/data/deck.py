"""A deck as the engine sees it: 3 character ids and 30 action card ids."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Deck:
    characters: list[int]
    cards: list[int]
