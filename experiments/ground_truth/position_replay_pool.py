"""An engine pool that first brings stored ground-truth positions back: each position's game is created again from its
seed and fed its logged responses (games are deterministic, as the mirrors rely on). Once a game stands at the
position's decision, `on_reached(position, decision, recorder)` is called (the caller hands the game to a search
table) and the decision is passed on like any other event. Every other event passes through unchanged, so a table
built on this pool doesn't notice the replays."""

from collections import Counter, deque
from dataclasses import dataclass

from autogitcg.bridge.engine_pool import EnginePool
from autogitcg.bridge.messages import Decision
from autogitcg.bridge.reply_views import reply_views
from experiments.ground_truth.stored_positions import StoredPosition, position_matchup
from autogitcg.observation.match_recorder import MatchRecorder
from experiments.tree_search.move_options import move_key


@dataclass
class Replay:
    position: StoredPosition
    remaining: dict[int, deque]  # per seat: logged responses not replayed yet
    recorder: MatchRecorder  # sees every replayed decision and response, like the original game's


def stands_at(line: dict, decision: Decision) -> bool:
    """The decision is the stored one: same player, an action decision, and every compared move is the same move."""
    if decision.who != line["seat"] or decision.kind != "action":
        return False
    actions = decision.request["action"]
    return all(move["response"]["chosenActionIndex"] < len(actions)
               and move_key(actions[move["response"]["chosenActionIndex"]]) == move["key"] for move in line["moves"])


class PositionReplayPool:
    def __init__(self, pool: EnginePool, on_reached, on_lost=lambda position: None, reply_guesses=None):
        """on_lost(position): its replay failed or didn't reach the stored decision. With replies on, only the
        position's seat gets them; blind positions' card replies are guessed with reply_guesses (bridge/reply_views.py)."""
        self.pool = pool
        self.reply_guesses = reply_guesses
        self.on_reached = on_reached
        self.on_lost = on_lost
        self.replays: dict[int, Replay] = {}
        self.counts = Counter()
        self.next_worker = 0

    def replay(self, position: StoredPosition) -> None:
        matchup = position_matchup(position.line)
        seat = position.line["seat"]
        views = reply_views((seat == 0, seat == 1), matchup.opponent_deck_known, matchup.decks, self.reply_guesses) \
            if self.pool.replies else None
        game_id = self.pool.create_game(*matchup.decks, seed=position.line["seed"], equipment=matchup.equipment, reply_views=views,
                                        worker_index=self.next_worker)
        self.next_worker = (self.next_worker + 1) % len(self.pool.workers)
        remaining = {int(who): deque(responses) for who, responses in position.line["responses_before"].items()}
        self.replays[game_id] = Replay(position, remaining, MatchRecorder(matchup.decks, matchup.opponent_deck_known))

    def poll(self, timeout: float | None = None) -> list:
        passed = []
        for event in self.pool.poll(timeout):
            replay = self.replays.get(event.game_id)
            if replay is None:
                passed.append(event)
            elif not isinstance(event, Decision):
                self._lose(event.game_id, replay, "replays failed")
            elif replay.remaining[event.who]:
                response = replay.remaining[event.who].popleft()
                replay.recorder.on_decision(event)
                replay.recorder.on_response(event, response)
                self.pool.respond(event, response)
            elif any(replay.remaining.values()) or not stands_at(replay.position.line, event):
                self.pool.stop_game(event.game_id)
                self._lose(event.game_id, replay, "replays diverged")
            else:
                del self.replays[event.game_id]
                self.counts["positions reached"] += 1
                self.on_reached(replay.position, event, replay.recorder)
                passed.append(event)
        return passed

    def _lose(self, game_id: int, replay: Replay, reason: str) -> None:
        del self.replays[game_id]
        self.counts[reason] += 1
        self.on_lost(replay.position)

    def __getattr__(self, name):
        return getattr(self.pool, name)
