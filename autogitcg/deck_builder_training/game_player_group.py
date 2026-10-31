"""Starts the game-player processes and plays batches of judged games on them.

All processes pull from one shared job queue, so a slow process never holds up the batch's other games.
"""

import multiprocessing
import queue

from autogitcg.deck_builder_training.deck_games import ScheduledGame
from autogitcg.deck_builder_training.game_outcome import GameOutcome
from autogitcg.deck_builder_training.game_player_process import PlayerFailed, PlayerSettings, run_game_player

RESULT_TIMEOUT_SECONDS = 120


class GamePlayerGroup:
    def __init__(self, process_count: int, settings: PlayerSettings):
        context = multiprocessing.get_context("spawn")  # CUDA and V8 state must not be forked
        self.jobs = context.Queue()
        self.results = context.Queue()
        self.stop_event = context.Event()
        self.processes = [context.Process(target=run_game_player, daemon=True,
                                          args=(settings, self.jobs, self.results, self.stop_event))
                          for _ in range(process_count)]
        for process in self.processes:
            process.start()

    def play(self, games: list[ScheduledGame]) -> list[GameOutcome]:
        """One outcome per game, in the order given."""
        for job, game in enumerate(games):
            self.jobs.put((job, game))
        outcomes: list[GameOutcome | None] = [None] * len(games)
        for _ in games:
            try:
                message = self.results.get(timeout=RESULT_TIMEOUT_SECONDS)
            except queue.Empty:
                dead = [index for index, process in enumerate(self.processes) if not process.is_alive()]
                raise RuntimeError(f"no game finished for {RESULT_TIMEOUT_SECONDS}s; dead game players: {dead}")
            if isinstance(message, PlayerFailed):
                raise RuntimeError(f"a game player failed:\n{message.message}")
            outcomes[message.job] = message
        return outcomes

    def close(self) -> None:
        self.stop_event.set()
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()

    def __enter__(self) -> "GamePlayerGroup":
        return self

    def __exit__(self, *exception_info) -> None:
        self.close()
