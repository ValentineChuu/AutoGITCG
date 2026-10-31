"""Mirrors of a held position on other engine workers. A game copy can only be made on the worker holding the game,
so with few positions open most workers sat idle (2026-10-20: 17 of 30). A mirror is the same game created on
another worker with the same decks and seed and fed the same responses; games are deterministic, so it reaches the
same decision and its copies are as good as the original's."""

from collections import deque
from dataclasses import dataclass

from autogitcg.bridge.messages import Decision


@dataclass
class MirrorReplay:
    position: object  # ground_truth_table.Position
    remaining: dict[int, deque]  # per seat: logged responses not replayed yet

    @staticmethod
    def of(position, responses_before: dict[int, list]) -> "MirrorReplay":
        return MirrorReplay(position, {who: deque(responses) for who, responses in responses_before.items()})

    def next_response(self, decision: Decision) -> dict | None:
        """The logged response to replay, or None once the log is used up (the mirror is at the position)."""
        logged = self.remaining[decision.who]
        return logged.popleft() if logged else None

    def reached(self, decision: Decision) -> bool:
        """The mirror stands at the position's decision: logs used up and the same request for the same player."""
        original = self.position.decision
        return (not any(self.remaining.values()) and decision.who == original.who and decision.kind == original.kind
                and decision.request == original.request)
