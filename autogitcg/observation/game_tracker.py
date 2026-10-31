"""What one player remembers about a game beyond the current board, using public information only."""

from collections import Counter
from dataclasses import dataclass

from autogitcg.data.deck import Deck


@dataclass
class PublicEvent:
    actor_is_me: bool
    action_kind: str  # an ACTION_KINDS entry
    vocabulary_kind: str | None  # "entity" (card), "skill", "character" or None
    definition_id: float | None  # None when the definition is not visible to this player
    round_number: int


class GameTracker:
    def __init__(self, my_deck: Deck, opponent_deck: Deck | None, reveal_opponent_tuned_cards: bool = True):
        self.my_remaining_deck = Counter(my_deck.cards)
        self.opponent_deck = Counter(opponent_deck.cards) if opponent_deck is not None else None
        self.opponent_played_cards = Counter()  # cards the opponent played or tuned (both are revealed)
        self.events: list[PublicEvent] = []
        self.reveal_opponent_tuned_cards = reveal_opponent_tuned_cards
        self._seen_hand_cards: dict[int, int] = {}  # hand card instance id -> definition id

    def observe_state(self, state: dict, who: int) -> None:
        """Count newly seen hand cards as drawn from my deck. Cards created by effects
        can't be told apart from draws, so counts are clamped at zero."""
        for card in state["player"][who].get("handCard", []):
            if card["id"] not in self._seen_hand_cards:
                self._seen_hand_cards[card["id"]] = card["definitionId"]
                if self.my_remaining_deck[card["definitionId"]] > 0:
                    self.my_remaining_deck[card["definitionId"]] -= 1

    def reset_own_cards(self, remaining_deck: Counter, hand_cards_by_id: dict[int, int]) -> None:
        """For a look-ahead copy whose hidden cards were dealt again: my deck and hand as dealt there."""
        self.my_remaining_deck = Counter(remaining_deck)
        self._seen_hand_cards = dict(hand_cards_by_id)

    def return_cards_to_deck(self, card_ids: list[int]) -> None:
        """Cards swapped back during the opening hand go back into my deck."""
        for card_id in card_ids:
            definition_id = self._seen_hand_cards.pop(card_id, None)
            if definition_id is not None:
                self.my_remaining_deck[definition_id] += 1

    def record_action(self, actor_is_me: bool, action: dict, round_number: int, tuned_card_definition: int | None = None) -> None:
        """tuned_card_definition: for elementalTuning, the tuned card's definition id as seen by the actor."""
        kind, value = action["$case"], action["value"]
        vocabulary_kind, definition_id = None, None
        match kind:
            case "useSkill":
                vocabulary_kind, definition_id = "skill", value["skillDefinitionId"]
            case "playCard":
                vocabulary_kind, definition_id = "entity", value["cardDefinitionId"]
                if not actor_is_me:
                    self.opponent_played_cards[definition_id] += 1
            case "switchActive":
                vocabulary_kind, definition_id = "character", value["characterDefinitionId"]
            case "elementalTuning":
                if actor_is_me or self.reveal_opponent_tuned_cards:
                    vocabulary_kind, definition_id = "entity", tuned_card_definition
                    if not actor_is_me and tuned_card_definition is not None:
                        self.opponent_played_cards[tuned_card_definition] += 1
        self.events.append(PublicEvent(actor_is_me, kind, vocabulary_kind, definition_id, round_number))

    def declared_end_first_this_round(self, round_number: int) -> bool | None:
        """True if I declared end first this round, False if the opponent did, None if nobody has."""
        for event in self.events:
            if event.round_number == round_number and event.action_kind == "declareEnd":
                return event.actor_is_me
        return None
