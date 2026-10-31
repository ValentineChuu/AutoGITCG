"""Which characters can still lead to a legal deck: the lineup must leave at least 30 legal action-card copies.

Cards without lineup requirements are legal with any lineup; when the collection owns 30 copies of those, every
lineup works. Otherwise (early collections) the remaining picks are searched until one completion works.
"""

from collections import Counter
from itertools import combinations

from autogitcg.deck_building.deck_rules import DECK_ACTION_CARDS, DECK_CHARACTERS, card_allowed, has_lineup_requirement, max_copies


def legal_copies(lineup: list[int], action_cards: Counter) -> dict[int, int]:
    """Card id -> how many copies this lineup may put in its deck from this collection."""
    return {card: min(owned, max_copies(card)) for card, owned in action_cards.items()
            if owned > 0 and card_allowed(card, lineup)}


def lineup_is_feasible(lineup: list[int], action_cards: Counter) -> bool:
    return sum(legal_copies(lineup, action_cards).values()) >= DECK_ACTION_CARDS


def feasible_next_characters(picked: list[int], owned_characters: tuple[int, ...], action_cards: Counter) -> list[int]:
    """Owned, not yet picked characters that can be completed into a lineup with enough legal cards."""
    remaining = [character for character in owned_characters if character not in picked]
    unrestricted = sum(min(owned, max_copies(card)) for card, owned in action_cards.items()
                       if owned > 0 and not has_lineup_requirement(card))
    if unrestricted >= DECK_ACTION_CARDS:
        return remaining if len(remaining) >= DECK_CHARACTERS - len(picked) else []
    still_needed = DECK_CHARACTERS - len(picked) - 1
    return [character for character in remaining
            if any(lineup_is_feasible([*picked, character, *completion], action_cards)
                   for completion in combinations([other for other in remaining if other != character], still_needed))]
