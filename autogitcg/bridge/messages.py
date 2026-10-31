"""Events coming back from engine workers (see engine/reference/bridge/worker.mjs for the wire protocol)."""

from dataclasses import dataclass


@dataclass
class Decision:
    """A player must choose. `kind` is the request's $case: action, chooseActive, rerollDice, switchHands, selectCard."""

    game_id: int
    who: int
    kind: str
    request: dict
    state: dict
    counters: dict[int, dict[str, int]]  # every displayed counter per board entity id
    previews: bool = False  # the request's actions carry real previews (games created with previews)
    replies: list | None = None  # per action: what the opponent could do right after it (games created with replies)
    replies_on_demand: bool = False  # replies come only for the actions asked for (EnginePool.request_replies)
    replies_checked: list[int] | None = None  # with replies on demand, once they arrived: the action indices checked
    end_of_round: dict | None = None  # seats with replies: what the end phase would do if both ended the round now


@dataclass
class GameFinished:
    game_id: int
    winner: int | None
    rounds: int
    defeated: tuple[int, int] = (0, 0)  # characters defeated, per player
    reactions: tuple[dict[int, int], dict[int, int]] = ({}, {})  # per player: reaction code -> count (see observation/enums.py)
    damage_dealt: tuple[int, int] = (0, 0)  # total damage dealt by each player (heals excluded)
    reaction_causes: tuple[dict[str, int], dict[str, int]] = ({}, {})  # per player: "code|caller type|skill type" -> count


@dataclass
class GameFailed:
    game_id: int
    message: str


@dataclass
class Notification:
    """An engine notification for the seat a person plays (games with human_seat only), passed on to their web board."""

    game_id: int
    who: int
    notification: dict


@dataclass
class RepliesReady:
    """The replies asked for with EnginePool.request_replies (null for actions not checked)."""

    game_id: int
    who: int
    replies: list
    checked: list[int]


Event = Decision | GameFinished | GameFailed | Notification | RepliesReady


def parse_event(message: dict) -> Event:
    match message["type"]:
        case "decision":
            request = message["request"]
            counters = {int(entity_id): values for entity_id, values in message.get("counters", {}).items()}
            return Decision(message["game"], message["who"], request["$case"], request["value"], message["state"], counters,
                            message.get("previews", False), message.get("replies"), message.get("repliesOnDemand", False),
                            end_of_round=message.get("endOfRound"))
        case "finished":
            reactions = tuple({int(code): count for code, count in side.items()} for side in message.get("reactions", ({}, {})))
            return GameFinished(message["game"], message["winner"], message["rounds"], tuple(message.get("defeated", (0, 0))),
                                reactions, tuple(message.get("damageDealt", (0, 0))),
                                tuple(message.get("reactionCauses", ({}, {}))))
        case "failed":
            return GameFailed(message["game"], message["message"])
        case "notify":
            return Notification(message["game"], message["who"], message["notification"])
        case "replies":
            return RepliesReady(message["game"], message["who"], message["replies"], message["checked"])
    raise ValueError(f"unknown engine message type: {message['type']}")
