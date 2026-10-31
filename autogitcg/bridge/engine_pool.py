"""A pool of Node engine workers running many games concurrently.

Typical loop:
    with EnginePool(worker_count=12) as pool:
        pool.create_game(deck0, deck1, seed)
        while pool.has_active_games():
            for event in pool.poll():
                if isinstance(event, Decision):
                    pool.respond(event, choose(event))
"""

import queue
from dataclasses import asdict
from itertools import count

from autogitcg.bridge.engine_worker import EngineWorker
from autogitcg.bridge.initial_equipment import InitialEquipment
from autogitcg.bridge.messages import Decision, Event, GameFailed, GameFinished
from autogitcg.data.deck import Deck


class EnginePool:
    def __init__(self, worker_count: int, previews: bool = False, replies: bool = False):
        """previews: every game (and look-ahead copy) carries the engine's preview of each action, which networks
        trained with previews read (observation/encode_previews.py); the engine runs about 2x slower.
        replies: every action also carries the opponent's possible replies to it (engine/reference/bridge/opponent_replies.mjs,
        observation/encode_replies.py); implies previews."""
        self.previews = previews or replies
        self.replies = replies
        self.events: queue.Queue[Event] = queue.Queue()
        self.workers = [EngineWorker(self.events) for _ in range(worker_count)]
        self.worker_by_game: dict[int, EngineWorker] = {}
        self.next_game_id = count()

    def create_game(self, deck0: Deck, deck1: Deck, seed: int, equipment: list[InitialEquipment] = (), privileged: bool = False,
                    human_seat: int | None = None, worker_index: int | None = None, reply_views: list[dict | None] | None = None) -> int:
        """privileged: every decision's state also carries "opponentPrivate" (the opponent's hand and dice), for training
        analysis only; observation encoding never reads it (observation/privileged_view.py does).
        human_seat: a seat played by a person through a web board: its notifications arrive as Notification events.
        reply_views: with replies on, per seat its view of the opponent's cards or None for no replies (bridge/reply_views.py);
        default: both seats get skill and switch replies only.
        worker_index: run it on this worker (default: the least busy). Copies of a game run on its worker, so search
        that needs many copies of one position replays the game on other workers (ground_truth/position_mirrors.py)."""
        game_id = next(self.next_game_id)
        worker = self.workers[worker_index] if worker_index is not None else \
            min(self.workers, key=lambda candidate: len(candidate.active_game_ids))
        worker.active_game_ids.add(game_id)
        self.worker_by_game[game_id] = worker
        worker.send({"type": "create", "game": game_id, "decks": [asdict(deck0), asdict(deck1)], "seed": seed,
                     "equipment": [item.to_message() for item in equipment], **({"privileged": True} if privileged else {}),
                     **({"humanSeat": human_seat} if human_seat is not None else {}),
                     **({"previews": True} if self.previews else {}), **({"replies": True} if self.replies else {}),
                     **({"replyViews": reply_views} if self.replies and reply_views is not None else {})})
        return game_id

    def copy_game(self, decision: Decision, response_value: dict, seed: int, opponent_cards: list[int] | None = None) -> int:
        """Look-ahead: a copy of the game at this pending action decision (hidden cards re-dealt with `seed`) that plays
        `response_value` first, then asks for every later decision. A copy whose first decision doesn't match fails.
        opponent_cards: a guess (card ids) for the opponent's whole hand and deck, instead of their real cards re-dealt."""
        copy_id = next(self.next_game_id)
        worker = self.worker_by_game[decision.game_id]
        worker.active_game_ids.add(copy_id)
        self.worker_by_game[copy_id] = worker
        worker.send({"type": "copy", "game": decision.game_id, "copy": copy_id, "who": decision.who, "seed": seed,
                     "response": {"$case": decision.kind, "value": response_value},
                     **({"opponentCards": opponent_cards} if opponent_cards is not None else {})})
        return copy_id

    def stop_game(self, game_id: int) -> None:
        """End a game silently. Events it sent before stopping may still arrive; callers should ignore them."""
        worker = self.worker_by_game.pop(game_id, None)
        if worker is not None:
            worker.active_game_ids.discard(game_id)
            worker.send({"type": "stop", "game": game_id})

    def request_replies(self, decision: Decision, request_indices: list[int]) -> None:
        """For a decision with replies_on_demand: compute the replies of these actions only; a RepliesReady event follows."""
        self.worker_by_game[decision.game_id].send({"type": "replies", "game": decision.game_id, "who": decision.who,
                                                     "indices": list(request_indices)})

    def respond(self, decision: Decision, response_value: dict) -> None:
        """Answer a decision; `response_value` is the body for that request kind, e.g. {"chosenActionIndex": 3, "usedDice": [...]}."""
        self.worker_by_game[decision.game_id].send({
            "type": "respond",
            "game": decision.game_id,
            "who": decision.who,
            "response": {"$case": decision.kind, "value": response_value},
        })

    def poll(self, timeout: float | None = None) -> list[Event]:
        """Block until at least one event arrives (or timeout), then drain everything already queued."""
        events = []
        try:
            events.append(self.events.get(timeout=timeout))
            while True:
                events.append(self.events.get_nowait())
        except queue.Empty:
            pass
        for event in events:
            if isinstance(event, GameFinished | GameFailed):
                worker = self.worker_by_game.pop(event.game_id, None)
                if worker is not None:
                    worker.active_game_ids.discard(event.game_id)
        return events

    def worker_index_of(self, game_id: int) -> int | None:
        worker = self.worker_by_game.get(game_id)
        return self.workers.index(worker) if worker is not None else None

    def has_active_games(self) -> bool:
        return bool(self.worker_by_game)

    def close(self) -> None:
        for worker in self.workers:
            worker.close()

    def __enter__(self) -> "EnginePool":
        return self

    def __exit__(self, *exception_info) -> None:
        self.close()
