"""Feeds both players' GameTrackers from the decisions and responses flowing through the bridge."""

import copy
from collections import Counter

from autogitcg.bridge.messages import Decision
from autogitcg.data.deck import Deck
from autogitcg.observation.game_tracker import GameTracker


class MatchRecorder:
    def __init__(self, decks: tuple[Deck, Deck], opponent_deck_known: tuple[bool, bool]):
        """opponent_deck_known[who]: whether that player is told the opponent's deck list (NPC mode)."""
        self.trackers = [
            GameTracker(decks[who], decks[1 - who] if opponent_deck_known[who] else None)
            for who in (0, 1)
        ]

    def copy_for_redeal(self, decision: Decision, opponent_cards: list[int] | None = None,
                        opponent_hand_ids: list[int] = ()) -> "MatchRecorder":
        """A copy for a game copy made at `decision` (search). opponent_cards: the opponent's hand then deck as dealt
        in that copy (the engine puts them into the existing slots in that order, keeping entity ids). Their tracker is
        set to match: remaining deck = the dealt deck, and their hand card ids map to the dealt hand, so the simulated
        opponent knows its own (copy's) cards and doesn't count its hand twice."""
        recorder = copy.deepcopy(self)
        if opponent_cards is not None:
            recorder.trackers[1 - decision.who].reset_own_cards(Counter(opponent_cards[len(opponent_hand_ids):]),
                                                                dict(zip(opponent_hand_ids, opponent_cards)))
        return recorder

    def on_decision(self, decision: Decision) -> GameTracker:
        """Call before encoding; returns the deciding player's tracker."""
        tracker = self.trackers[decision.who]
        tracker.observe_state(decision.state, decision.who)
        return tracker

    def on_response(self, decision: Decision, response_value: dict) -> None:
        if decision.kind == "switchHands":
            self.trackers[decision.who].return_cards_to_deck(response_value.get("removedHandIds", []))
        elif decision.kind == "action":
            action = decision.request["action"][response_value["chosenActionIndex"]]["action"]
            tuned_card_definition = None
            if action["$case"] == "elementalTuning":
                hand = decision.state["player"][decision.who].get("handCard", [])
                tuned_card_definition = next(
                    (card["definitionId"] for card in hand if card["id"] == action["value"]["removedCardId"]), None)
            for who, tracker in enumerate(self.trackers):
                tracker.record_action(who == decision.who, action, decision.state["roundNumber"], tuned_card_definition)
