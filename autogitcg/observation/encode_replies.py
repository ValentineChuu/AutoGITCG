"""What the opponent could do right after each action (engine/reference/bridge/opponent_replies.mjs): the back-and-forth a
player works out ("if I do this, can they kill my active?"). Appended to every action token after its preview.

A reply is the opponent's whole turn after my action: every distinct line of their moves
until their turn ends (turn_search.mjs). Per action, in three blocks of the same outcome summary (over a set of lines:
the worst HP change and whether any line defeats each of my characters, counted from now so my own action's effect
is included, the most damage any line deals to my side, and how many lines there are):
- who decides next (the opponent, me again after a fast action, another kind of decision, or the game ended), then
  the summary of their lines that need no card when they decide next;
- after a fast action: the same if I ended my turn there;
- the summary of their lines that need one of their cards, from their real remaining cards when their deck list is
  open or from a guessed hand when blind, and which of the two it was (none when blind without a guess);
- "not checked": with the learned reply selector (docs/architecture.md), an action whose replies weren't
  computed, so the network can tell "checked, no danger" from "unknown".
Actions without replies (games created without them, single-move decisions, seats without a view) get zeros.
"""

from autogitcg.observation.encode_board import CHARACTERS_PER_SIDE
from autogitcg.observation.encode_previews import PER_CHARACTER, preview_features

NEXT_DECIDERS = ("opponent", "self", "other", "finished")
CARD_SOURCES = ("open", "guessed")
OUTCOME_WIDTH = 2 * CHARACTERS_PER_SIDE + 2
REPLY_WIDTH = len(NEXT_DECIDERS) + 3 * OUTCOME_WIDTH + len(CARD_SOURCES) + 1


def _outcome_summary(previews: list, state: dict, who: int) -> list[float]:
    outcomes = [preview_features(preview, state, who) for preview in previews]
    worst_health = [min((outcome[position * len(PER_CHARACTER)] for outcome in outcomes), default=0.0)
                    for position in range(CHARACTERS_PER_SIDE)]
    defeated = [max((outcome[position * len(PER_CHARACTER) + 1] for outcome in outcomes), default=0.0)
                for position in range(CHARACTERS_PER_SIDE)]
    most_damage = max((-sum(min(0.0, outcome[position * len(PER_CHARACTER)]) for position in range(CHARACTERS_PER_SIDE))
                       for outcome in outcomes), default=0.0)
    return worst_health + defeated + [most_damage, min(len(outcomes), 8) / 4]


def _lines_summary(lines: list[dict]) -> list[float]:
    """The outcome summary over whole-turn lines (each: my characters' [HP change, defeated] in board order)."""
    changes = [[(change / 10, float(defeated)) for change, defeated in line["mine"][:CHARACTERS_PER_SIDE]] for line in lines]
    for line in changes:
        line += [(0.0, 0.0)] * (CHARACTERS_PER_SIDE - len(line))
    worst_health = [min((line[position][0] for line in changes), default=0.0) for position in range(CHARACTERS_PER_SIDE)]
    defeated = [max((line[position][1] for line in changes), default=0.0) for position in range(CHARACTERS_PER_SIDE)]
    most_damage = max((-sum(min(0.0, health) for health, _ in line) for line in changes), default=0.0)
    return worst_health + defeated + [most_damage, min(len(changes), 8) / 4]


def reply_features(reply: dict | None, state: dict, who: int, unchecked: bool = False) -> list[float]:
    if reply is None:
        return [0.0] * (REPLY_WIDTH - 1) + [float(unchecked)]
    stopped = reply.get("stopped", False)
    if "lines" in reply:  # whole-turn replies (theta)
        board_replies = _lines_summary([line for line in reply["lines"] if not line["usedCard"]])
        card_replies = _lines_summary([line for line in reply["lines"] if line["usedCard"]])
    else:  # single-action previews (eta 4-05)
        board_replies = _outcome_summary(reply.get("previews", []), state, who)
        card_replies = _outcome_summary(reply.get("cardPreviews", []), state, who)
    empty = [0.0] * OUTCOME_WIDTH
    return ([float(reply.get("next") == decider) for decider in NEXT_DECIDERS]
            + (empty if stopped else board_replies) + (board_replies if stopped else empty)
            + card_replies
            + [float(reply.get("cards") == source) for source in CARD_SOURCES] + [0.0])
